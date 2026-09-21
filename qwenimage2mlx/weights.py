"""Build the three MLX models and load Qwen-Image-2.1 safetensors into them."""
import glob
import json
import os

import mlx.core as mx

from .models.transformer import QwenImage21Transformer
from .models.vae import VAE21
from .models.text_encoder import Qwen3TextEncoder


# ---------------- transformer ----------------
def _map_transformer_key(k):
    if k.startswith("modulation.1."):
        return "modulation.0." + k[len("modulation.1."):]
    return k.replace("time_text_embed.timestep_embedder.", "time_text_embed.")


def build_transformer(model_dir, dtype):
    tdir = os.path.join(model_dir, "transformer")
    cfg = json.load(open(os.path.join(tdir, "config.json")))
    model = QwenImage21Transformer(
        patch_size=cfg["patch_size"], in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"], num_layers=cfg["num_layers"],
        attention_head_dim=cfg["attention_head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        context_in_dim=cfg["context_in_dim"], mlp_ratio=cfg["mlp_ratio"],
        eps=cfg["eps"], causal_condition=cfg["causal_condition"],
    )
    weights = {}
    for f in sorted(glob.glob(os.path.join(tdir, "*.safetensors"))):
        for k, v in mx.load(f).items():
            weights[_map_transformer_key(k)] = v.astype(dtype)
    model.load_weights(list(weights.items()), strict=False)
    return model, cfg


# ---------------- vae ----------------
def _remap_vae(k, v):
    if not (k.startswith("post_quant_conv") or k.startswith("decoder.")):
        return None, None
    if k.endswith(".gamma"):
        return k, v.reshape(-1)
    if k.endswith(".weight") and v.ndim == 4:
        v = v.transpose(0, 2, 3, 1)
        base = k[: -len(".weight")]
        if base.endswith("to_qkv") or base.endswith("proj"):
            return k, v
        return base + ".conv.weight", v
    if k.endswith(".bias"):
        base = k[: -len(".bias")]
        if base.endswith("to_qkv") or base.endswith("proj"):
            return k, v
        return base + ".conv.bias", v
    return k, v


def build_vae(model_dir, dtype):
    vdir = os.path.join(model_dir, "vae")
    m = VAE21()
    weights = {}
    for f in sorted(glob.glob(os.path.join(vdir, "*.safetensors"))):
        for k, v in mx.load(f).items():
            nk, nv = _remap_vae(k, v)
            if nk is not None:
                weights[nk] = nv.astype(dtype)
    m.load_weights(list(weights.items()), strict=False)
    vcfg = json.load(open(os.path.join(vdir, "config.json")))
    return m, vcfg


# ---------------- text encoder ----------------
def _map_te_key(k):
    p = "model.language_model."
    return k[len(p):] if k.startswith(p) else None


def build_text_encoder(model_dir, dtype):
    tdir = os.path.join(model_dir, "text_encoder")
    cfg = json.load(open(os.path.join(tdir, "config.json")))
    tc = cfg.get("text_config", cfg)
    enc = Qwen3TextEncoder(
        vocab_size=tc["vocab_size"], hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        intermediate_size=tc["intermediate_size"],
        rope_theta=tc["rope_theta"], rms_norm_eps=tc["rms_norm_eps"],
        head_dim=tc["head_dim"], attention_bias=False,
    )
    weights = {}
    for f in sorted(glob.glob(os.path.join(tdir, "*.safetensors"))):
        for k, v in mx.load(f).items():
            nk = _map_te_key(k)
            if nk is not None:
                weights[nk] = v.astype(dtype)
    enc.load_weights(list(weights.items()), strict=False)
    return enc, cfg
