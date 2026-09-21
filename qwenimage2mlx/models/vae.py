"""AutoencoderKLQwenImage21 decoder, ported to MLX (single-frame image path).

Faithful port of diffusers autoencoder_kl_qwenimage21.py, is_residual=True.
For single-frame decode every CausalConv3d collapses to a 2D same-padding
conv, temporal ops are no-ops, and DupUp3D keeps T=1. So this is a pure 2D
NHWC pipeline.

RMS_norm here is diffusers `QwenImage21RMS_norm`: L2-normalize over the
channel axis (F.normalize, eps 1e-12), then * sqrt(dim) * gamma.
"""

import mlx.core as mx
import mlx.nn as nn

Z_DIM = 64
DIM = 144
DIM_MULT = [1, 2, 4, 8, 8]
NUM_RES = 2
# temperal_upsample = temperal_downsample[::-1] = [True,True,True,False]
TEMPORAL_UP = [True, True, True, False]
# dims = [dim*u for u in [dim_mult[-1]] + dim_mult[::-1]]
DIMS = [DIM * u for u in ([DIM_MULT[-1]] + DIM_MULT[::-1])]  # [1152,1152,1152,576,288,144]
OUT_CH = 4


def _rms_norm(x, gamma, dim):
    """x: (...,C) NHWC. L2 normalize over last axis * sqrt(dim) * gamma."""
    dt = x.dtype
    x = x.astype(mx.float32)
    denom = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)
    out = x / denom * (dim ** 0.5) * gamma.astype(mx.float32)
    return out.astype(dt)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        return _rms_norm(x, self.gamma, self.dim)


class Conv2dSame(nn.Module):
    """CausalConv3d for single frame = Conv2d with same padding = k//2."""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k // 2)

    def __call__(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm1 = RMSNorm(in_dim)
        self.conv1 = Conv2dSame(in_dim, out_dim, 3)
        self.norm2 = RMSNorm(out_dim)
        self.conv2 = Conv2dSame(out_dim, out_dim, 3)
        self.has_shortcut = in_dim != out_dim
        if self.has_shortcut:
            self.conv_shortcut = Conv2dSame(in_dim, out_dim, 1)

    def __call__(self, x):
        h = self.conv_shortcut(x) if self.has_shortcut else x
        x = nn.silu(self.norm1(x))
        x = self.conv1(x)
        x = nn.silu(self.norm2(x))
        x = self.conv2(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        # to_qkv / proj are 1x1 Conv2d in diffusers -> use Linear over channels
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x):
        # x: (B,H,W,C) NHWC
        identity = x
        B, H, W, C = x.shape
        xn = self.norm(x)
        qkv = self.to_qkv(xn)  # (B,H,W,3C)
        qkv = qkv.reshape(B, H * W, 3, C)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        # single head attention over H*W tokens
        scale = C ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)
        out = attn @ v  # (B,HW,C)
        out = out.reshape(B, H, W, C)
        out = self.proj(out)
        return out + identity


class MidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = [ResidualBlock(dim, dim), ResidualBlock(dim, dim)]
        self.attentions = [AttentionBlock(dim)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


def _upsample_nearest2x(x):
    # x: (B,H,W,C) -> (B,2H,2W,C) nearest-exact, computed fp32
    dt = x.dtype
    x = x.astype(mx.float32)
    B, H, W, C = x.shape
    x = mx.repeat(x, 2, axis=1)
    x = mx.repeat(x, 2, axis=2)
    return x.astype(dt)


class Upsampler(nn.Module):
    """QwenImage21Resample upsample (2d or 3d). Single-frame: spatial only.

    upsample3d additionally has a time_conv (no-op for lone first frame) whose
    weights still live in the checkpoint.
    """

    def __init__(self, dim, is_3d):
        super().__init__()
        self.is_3d = is_3d
        # resample = Sequential(Upsample(noparam), Conv2d(dim,dim,3,pad1)) -> resample.1
        self.resample = [None, Conv2dSame(dim, dim, 3)]
        if is_3d:
            self.time_conv = Conv2dSame(dim, dim * 2, 1)  # unused for single frame

    def __call__(self, x):
        x = _upsample_nearest2x(x)
        x = self.resample[1](x)
        return x


class ResidualUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, temporal_up, up_flag):
        super().__init__()
        self.up_flag = up_flag
        self.resnets = [ResidualBlock(in_dim, out_dim)] + [
            ResidualBlock(out_dim, out_dim) for _ in range(NUM_RES)
        ]  # num_res_blocks + 1 = 3
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temporal_up = temporal_up
        if up_flag:
            self.upsampler = Upsampler(out_dim, is_3d=temporal_up)
            self.factor_t = 2 if temporal_up else 1
            self.factor_s = 2

    def _dup_shortcut(self, x_in):
        """DupUp3D single-frame: channel repeat_interleave then spatial expand.

        factor = factor_t*factor_s*factor_s; repeats = out_dim*factor // in_dim.
        For single frame with first_chunk=True, temporal expansion trimmed to T=1,
        so net = spatial 2x + channel selection to out_dim.
        repeats along channel: out*factor//in. Then reshape spatial by factor_s.
        """
        B, H, W, C = x_in.shape
        factor = self.factor_t * self.factor_s * self.factor_s
        repeats = self.out_dim * factor // self.in_dim
        x = mx.repeat(x_in, repeats, axis=-1)  # (B,H,W,C*repeats)
        # now channels = C*repeats = out_dim*factor. Fold factor_s*factor_s into space.
        # torch DupUp3D reshapes (..., out_dim, factor_t, factor_s, factor_s) then
        # permutes to expand T,H,W. Single-frame -> take factor_t slice = last.
        Cn = x.shape[-1]
        x = x.reshape(B, H, W, self.out_dim, self.factor_t, self.factor_s, self.factor_s)
        # first_chunk: keep last temporal frame (drop factor_t-1 leading)
        x = x[:, :, :, :, -1, :, :]  # (B,H,W,out_dim,fs,fs)
        # expand spatial: (B,H,fs,W,fs,out_dim) -> (B,H*fs,W*fs,out_dim)
        x = x.transpose(0, 1, 4, 2, 5, 3)  # (B,H,fs,W,fs,out_dim)
        x = x.reshape(B, H * self.factor_s, W * self.factor_s, self.out_dim)
        return x

    def __call__(self, x):
        x_copy = x
        for r in self.resnets:
            x = r(x)
        if self.up_flag:
            x = self.upsampler(x)
            x = x + self._dup_shortcut(x_copy)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = Conv2dSame(Z_DIM, DIMS[0], 3)  # 64->1152
        self.mid_block = MidBlock(DIMS[0])
        # up_blocks: zip(dims[:-1], dims[1:]), i=0..4
        pairs = list(zip(DIMS[:-1], DIMS[1:]))
        self.up_blocks = []
        for i, (ind, outd) in enumerate(pairs):
            up_flag = i != (len(DIM_MULT) - 1)  # i != 4
            tup = TEMPORAL_UP[i] if up_flag else False
            self.up_blocks.append(ResidualUpBlock(ind, outd, tup, up_flag))
        self.norm_out = RMSNorm(DIMS[-1])  # 144
        self.conv_out = Conv2dSame(DIMS[-1], OUT_CH, 3)

    def __call__(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for ub in self.up_blocks:
            x = ub(x)
        x = nn.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x


class VAE21(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = Conv2dSame(Z_DIM, Z_DIM, 1)
        self.decoder = Decoder()

    def decode(self, z):
        """z: (B,H,W,64) NHWC latents in VAE space. Returns (B,H,W,4) in [-1,1]."""
        z = self.post_quant_conv(z)
        out = self.decoder(z)
        return mx.clip(out, -1.0, 1.0)
