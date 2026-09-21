"""Qwen-Image-2.1 single-stream DiT, ported to Apple MLX.

Faithful port of diffusers `QwenImage21Transformer2DModel`
(transformer_qwenimage21.py). No KV cache — always runs the full
block-causal prefill each step (correctness first; cache is only a speed
optimization). Block-causal attention is expressed as an additive mask
fed to `mx.fast.scaled_dot_product_attention`.

Config (Qwen/Qwen-Image-2.1):
  in_channels=64, num_layers=32, num_attention_heads=32,
  attention_head_dim=128 (inner_dim=4096), context_in_dim=4096,
  mlp_ratio=3, axes_dims_rope=(16,56,56), eps=1e-6, causal_condition=True.
"""

import math

import mlx.core as mx
import mlx.nn as nn

_IMG_TOKENS_PER_SLOT = 4


# --------------------------------------------------------------------------
# RoPE (3-axis frame/height/width, complex freqs)
# --------------------------------------------------------------------------
class QwenImage21Rope:
    """Precompute complex rotary freqs for the 3 axes.

    Mirrors diffusers QwenImage21Rope. Freqs stored as (cos, sin) real pairs
    so we can apply the complex rotation without an mlx complex dtype.
    axes_dim entries are the *complex* dim per axis; the real rope dim is 2x.
    """

    def __init__(self, theta: int = 10000, axes_dim=(16, 56, 56)):
        self.theta = theta
        self.axes_dim = list(axes_dim)
        pos_index = mx.arange(8192)
        neg_index = mx.arange(1024)[::-1] * -1 - 1
        index = mx.concatenate([pos_index, neg_index]).astype(mx.float32)  # (9216,)
        self._cos = []
        self._sin = []
        for dim in self.axes_dim:
            # freqs: outer(index, 1/theta^(arange(0,dim,2)/dim)) -> (N, dim/2)
            inv = 1.0 / mx.power(theta, mx.arange(0, dim, 2).astype(mx.float32) / dim)
            freqs = mx.outer(index, inv)  # (N, dim/2)
            self._cos.append(mx.cos(freqs))
            self._sin.append(mx.sin(freqs))

    def __call__(self, img_shapes, image_pad_mask):
        """Return (cos, sin) each (seq_len, sum(axes_dim)//2) for the joint seq.

        img_shapes: list of (frame,h,w) latent-token shapes, condition images
            first, target last.
        image_pad_mask: (seq_len,) bool python-list-able mx.array, True at image tokens.
        """
        is_image_token = image_pad_mask.tolist()
        total_len = len(is_image_token)

        frame_index, image_height_index, image_width_index = [], [], []
        cursor, position = 0, 0
        for _, height, width in img_shapes:
            # find next True at/after cursor
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend(
                [h for h in range(-(height - height // 2), height // 2) for _ in range(width)]
            )
            image_width_index.extend(
                [w for _ in range(height) for w in range(-(width - width // 2), width // 2)]
            )
        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index = mx.array(frame_index, dtype=mx.int32)
        height_index = mx.array(frame_index)
        width_index = mx.array(frame_index)
        img_pos = [i for i, v in enumerate(is_image_token) if v]
        if img_pos:
            img_pos = mx.array(img_pos, dtype=mx.int32)
            height_index[img_pos] = mx.array(image_height_index, dtype=mx.int32)
            width_index[img_pos] = mx.array(image_width_index, dtype=mx.int32)

        cos = mx.concatenate(
            [self._cos[0][frame_index], self._cos[1][height_index], self._cos[2][width_index]],
            axis=-1,
        )  # (seq, D/2)
        sin = mx.concatenate(
            [self._sin[0][frame_index], self._sin[1][height_index], self._sin[2][width_index]],
            axis=-1,
        )
        return cos, sin


def apply_rope(x, cos, sin):
    """Apply complex rotary to x.

    x:  (B, H, S, D)   D even
    cos,sin: (S, D/2)
    Equivalent to diffusers apply_rotary_emb_qwen(use_real=False):
    view x as complex pairs (x0,x1), multiply by e^{i*theta}:
      out0 = x0*cos - x1*sin
      out1 = x0*sin + x1*cos
    then interleave back.
    """
    B, H, S, D = x.shape
    x = x.reshape(B, H, S, D // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    c = cos.reshape(1, 1, S, D // 2)
    s = sin.reshape(1, 1, S, D // 2)
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    out = mx.stack([o0, o1], axis=-1).reshape(B, H, S, D)
    return out.astype(x.dtype)


# --------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------
class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm with effective scale = weight+1, computed in fp32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        dt = x.dtype
        x = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rrms * (self.weight.astype(mx.float32) + 1)).astype(dt)


class RMSNormHead(nn.Module):
    """Plain RMSNorm over head_dim (norm_q / norm_k), scale = weight."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        dt = x.dtype
        x = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rrms * self.weight.astype(mx.float32)).astype(dt)


def layernorm_noaffine(x, eps=1e-6):
    dt = x.dtype
    x = x.astype(mx.float32)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    out = (x - mean) * mx.rsqrt(var + eps)
    return out.astype(dt)


# --------------------------------------------------------------------------
# Sub-blocks
# --------------------------------------------------------------------------
class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, timestep_dim=256, max_period=10000, time_factor=1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        self.freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half).astype(mx.float32) / half
        )
        # TimestepEmbedding: linear_1 (256->emb, no bias) SiLU linear_2 (emb->emb, no bias)
        self.linear_1 = nn.Linear(timestep_dim, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def time_proj(self, timestep):
        t = self.time_factor * timestep.astype(mx.float32)
        args = t[:, None] * self.freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.timestep_dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, timestep, dtype):
        proj = self.time_proj(timestep).astype(dtype)
        h = nn.silu(self.linear_1(proj))
        return self.linear_2(h)


class TextProjection(nn.Module):
    def __init__(self, context_in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x):
        x = self.text_norm(x)
        x = self.in_layer(x)
        x = nn.gelu_approx(x)  # tanh approximation
        return self.out_layer(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)

    def __call__(self, x):
        return self.out(nn.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        # to_out is a ModuleList[Linear, Dropout] in diffusers -> to_out.0.weight
        self.to_out = [nn.Linear(inner, dim, bias=False)]
        self.norm_q = RMSNormHead(dim_head, eps=eps)
        self.norm_k = RMSNormHead(dim_head, eps=eps)
        self.scale = dim_head ** -0.5

    def __call__(self, x, cos, sin, attn_mask):
        B, S, _ = x.shape
        H, Dh = self.heads, self.dim_head
        q = self.to_q(x).reshape(B, S, H, Dh)
        k = self.to_k(x).reshape(B, S, H, Dh)
        v = self.to_v(x).reshape(B, S, H, Dh)
        # RMSNorm over head dim (last axis)
        q = self.norm_q(q)
        k = self.norm_k(k)
        # to (B,H,S,Dh)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attn_mask
        )  # (B,H,S,Dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)
        return self.to_out[0](out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.attn = Attention(dim, num_heads, head_dim, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def _modulate(self, x, mod_params, target_mask):
        scale, gate = mx.split(mod_params, 2, axis=-1)
        scale = _select_mod_rows(scale, target_mask)
        gate = _select_mod_rows(gate, target_mask)
        return x * (1 + scale), gate

    def __call__(self, x, modulation, cos, sin, attn_mask, target_mask):
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        xm, gate1 = self._modulate(layernorm_noaffine(x, self.eps), mod1, target_mask)
        attn_out = self.attn(xm, cos, sin, attn_mask)
        x = x + mx.tanh(gate1) * attn_out
        xm2, gate2 = self._modulate(layernorm_noaffine(x, self.eps), mod2, target_mask)
        x = x + mx.tanh(gate2) * self.img_mlp(xm2)
        return x


class AdaLayerNormContinuous(nn.Module):
    """Final norm, scale-only."""

    def __init__(self, embedding_dim, cond_dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(cond_dim, embedding_dim, bias=False)
        self.eps = eps

    def __call__(self, x, cond, target_mask):
        scale = self.linear(nn.silu(cond).astype(x.dtype))
        scale = _select_mod_rows(scale, target_mask)
        return layernorm_noaffine(x, self.eps) * (1 + scale)


def _select_mod_rows(params, target_mask):
    """Broadcast per-sample modulation over tokens.

    params: (B,dim) without causal_condition, else (B+1,dim) (last row = t=0).
    target_mask: (seq,) bool. None -> every token uses its own row.
    """
    if target_mask is None:
        return params[:, None, :]
    real = params[:-1][:, None, :]        # (B,1,dim)
    zero = params[-1:][None, :, :]        # (1,1,dim)
    m = target_mask.reshape(1, -1, 1)
    return mx.where(m, real, zero)


# --------------------------------------------------------------------------
# Top-level model
# --------------------------------------------------------------------------
class QwenImage21Transformer(nn.Module):
    def __init__(
        self,
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        causal_condition=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_condition = causal_condition
        self.eps = eps

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=axes_dims_rope)
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)
        # modulation: Sequential(SiLU, Linear) -> weight key modulation.1.weight
        self.modulation = [nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False)]

        self.transformer_blocks = [
            TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
            for _ in range(num_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

    @staticmethod
    def build_token_metadata(image_pad_mask, img_shapes):
        """image_ids (-1 text, unique id per image block) + target_token_mask."""
        pad = image_pad_mask.tolist()
        img_pos = [i for i, v in enumerate(pad) if v]
        block_lengths = [f * h * w for (f, h, w) in img_shapes]
        if sum(block_lengths) != len(img_pos):
            raise ValueError(
                f"img_shapes -> {sum(block_lengths)} image tokens but mask marks {len(img_pos)}"
            )
        seq = len(pad)
        image_ids = [-1] * seq
        block_ids = []
        for bid, ln in enumerate(block_lengths):
            block_ids.extend([bid] * ln)
        for p, bid in zip(img_pos, block_ids):
            image_ids[p] = bid
        target_mask = [False] * seq
        for p in img_pos[-block_lengths[-1]:]:
            target_mask[p] = True
        return image_ids, mx.array(target_mask)

    @staticmethod
    def build_block_causal_mask(image_ids, key_valid, dtype):
        """Additive attention mask (S,S): 0 allowed, -inf blocked.

        allowed = (q>=kv or same_image_block) and key_valid[kv].
        image_ids: python list length S. key_valid: (S,) bool mx.array or None.
        """
        S = len(image_ids)
        ids = mx.array(image_ids, dtype=mx.int32)
        q = ids.reshape(S, 1)
        kv = ids.reshape(1, S)
        qi = mx.arange(S).reshape(S, 1)
        kvi = mx.arange(S).reshape(1, S)
        same_block = (q == kv) & (q >= 0)
        allowed = ((qi >= kvi) | same_block)
        if key_valid is not None:
            allowed = allowed & key_valid.reshape(1, S)
        # Additive mask in the compute dtype. A fp32 mask on a bf16 model kicks
        # SDPA off its fused kernel onto a ~9x slower path, so build it in `dtype`.
        neg = mx.array(-mx.inf, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        add = mx.where(allowed, zero, neg)
        return add.reshape(1, 1, S, S)

    def __call__(
        self,
        hidden_states,          # (B, img_seq, in_channels)
        encoder_hidden_states,  # (B, txt_seq, context_in_dim)
        timestep,               # (B,) scaled /1000
        img_shapes,             # list of (f,h,w), condition first, target last
        img_mask,               # (B, vlm_seq) bool, True at VLM image slots
        encoder_hidden_states_mask=None,  # (B, txt_seq) bool
    ):
        B = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # expand each VLM image slot to 2x2 latent tokens
        repeats = mx.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]  # (vlm_seq,)
        repeats_list = repeats.tolist()
        img_mask_row = img_mask[0].tolist()
        image_pad_list = []
        for v, r in zip(img_mask_row, repeats_list):
            image_pad_list.extend([bool(v)] * r)
        image_pad_mask = mx.array(image_pad_list)  # (seq,)

        target_tokens = 1
        for d in img_shapes[-1]:
            target_tokens *= d
        zeros_tail = mx.zeros(
            (B, target_tokens // 4, encoder_hidden_states.shape[2]), dtype=encoder_hidden_states.dtype
        )
        joint = mx.concatenate([encoder_hidden_states, zeros_tail], axis=1)
        # repeat_interleave along seq by repeats
        joint = _repeat_interleave_axis1(joint, repeats_list)
        # scatter image latents into image positions
        joint = _scatter_image(joint, image_pad_list, hidden_states)

        cos, sin = self.pos_embed(img_shapes, image_pad_mask)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes)

        timestep = timestep.astype(hidden_states.dtype)
        if self.causal_condition:
            timestep = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)], axis=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states.dtype)
        modulation = self.modulation[0](nn.silu(temb))

        # joint key-valid mask for right-padded prompts
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            seq = len(image_pad_list)
            kv = [True] * seq
            text_positions = [i for i, v in enumerate(image_pad_list) if not v]
            ehm = encoder_hidden_states_mask[0].tolist()
            vlm_text = [not b for b in img_mask_row[: len(ehm)]]
            ehm_text = [ehm[i] for i, keep in enumerate(vlm_text) if keep]
            for pos, val in zip(text_positions, ehm_text):
                kv[pos] = bool(val)
            joint_key_valid = mx.array(kv)

        attn_mask = self.build_block_causal_mask(image_ids, joint_key_valid, hidden_states.dtype)

        x = joint
        for block in self.transformer_blocks:
            x = block(x, modulation, cos, sin, attn_mask, modulation_mask)

        x = self.norm_out(x, temb, modulation_mask)
        out = self.proj_out(x)
        return out


def _repeat_interleave_axis1(t, repeats_list):
    """t: (B,S,D). repeats_list: python list len S. Returns (B, sum, D)."""
    idx = []
    for i, r in enumerate(repeats_list):
        idx.extend([i] * r)
    idx = mx.array(idx, dtype=mx.int32)
    return t[:, idx, :]


def _scatter_image(joint, image_pad_list, image_latents):
    """Place image_latents (B,n_img,D) into joint (B,seq,D) at True positions."""
    pos = [i for i, v in enumerate(image_pad_list) if v]
    pos = mx.array(pos, dtype=mx.int32)
    # build via concatenation-free scatter: joint[:, pos, :] = image_latents
    joint[:, pos, :] = image_latents
    return joint
