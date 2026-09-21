"""Qwen3-VL language model (text-only) for Qwen-Image-2.1, self-contained MLX.

Internalized from mflux common_models/qwen3_vl + flux2 text encoder so the
package has no mflux runtime dependency. Text-only path (no vision tower);
returns the last decoder-layer output BEFORE the final RMSNorm — that is what
the Qwen-Image transformer was trained on.
"""
import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x):
        dt = x.dtype
        x = x.astype(mx.float32)
        var = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        return (self.weight.astype(mx.float32) * x).astype(dt)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_e = (q * cos) + (_rotate_half(q) * sin)
    k_e = (k * cos) + (_rotate_half(k) * sin)
    return q_e, k_e


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 attention_bias=False, rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    @staticmethod
    def _repeat_kv(x, n_rep):
        b, kvh, s, hd = x.shape
        x = mx.expand_dims(x, axis=2)
        x = mx.broadcast_to(x, (b, kvh, n_rep, s, hd))
        return x.reshape(b, kvh * n_rep, s, hd)

    def __call__(self, x, attention_mask, position_embeddings):
        b, q_len, _ = x.shape
        q = self.q_proj(x).reshape(b, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, q_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        cos, sin = position_embeddings
        q, k = _apply_rope(q, k, cos, sin)
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)
        out = scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=self.scaling, mask=attention_mask,
        ).astype(q.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(b, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, attention_bias=False, rms_norm_eps=1e-6):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim,
                                   attention_bias, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def __call__(self, x, attention_mask, position_embeddings):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_embeddings)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, position_ids):
        if position_ids.ndim == 1:
            position_ids = mx.expand_dims(position_ids, axis=0)
        inv = mx.expand_dims(mx.expand_dims(self.inv_freq, 0), 0)
        pos = mx.expand_dims(position_ids.astype(mx.float32), -1)
        freqs = pos * inv
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


class Qwen3TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 num_key_value_heads, intermediate_size, rope_theta=5000000.0,
                 rms_norm_eps=1e-6, head_dim=128, attention_bias=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            DecoderLayer(hidden_size, num_attention_heads, num_key_value_heads,
                         head_dim, intermediate_size, attention_bias, rms_norm_eps)
            for _ in range(num_hidden_layers)
        ]
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_theta)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        b, s = input_ids.shape
        h = self.embed_tokens(input_ids)
        dt = h.dtype
        if attention_mask is None:
            attention_mask = mx.ones((b, s), dtype=mx.int32)
        pad = mx.where(attention_mask == 1,
                       mx.zeros(attention_mask.shape, dtype=dt),
                       mx.full(attention_mask.shape, -float("inf"), dtype=dt))
        pad = mx.expand_dims(mx.expand_dims(pad, 1), 1)
        if s == 1:
            causal = mx.zeros((b, 1, 1, 1), dtype=dt)
        else:
            idx = mx.arange(s, dtype=mx.int32)
            tri = mx.expand_dims(idx, 0) > mx.expand_dims(idx, 1)
            causal = mx.where(tri, mx.full((s, s), -float("inf"), dtype=dt),
                              mx.zeros((s, s), dtype=dt))
            causal = mx.broadcast_to(
                mx.expand_dims(mx.expand_dims(causal, 0), 0), (b, 1, s, s))
        mask4d = causal + pad
        pos = mx.broadcast_to(mx.expand_dims(mx.arange(s, dtype=mx.int32), 0), (b, s))
        pe = self.rotary_emb(h, pos)
        hs_list = [h] if output_hidden_states else None
        for layer in self.layers:
            h = layer(h, mask4d, pe)
            if output_hidden_states:
                hs_list.append(h)
        return self.norm(h), hs_list
