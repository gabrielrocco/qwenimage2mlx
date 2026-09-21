"""Real-time q8/q4 quantization with on-disk caching.

Quantizes the transformer and text encoder (the heavy Linear-bound modules).
The VAE is left in fp16 — quantizing it visibly degrades the decode.

Layout of a saved quant (e.g. ~/.cache/qwenimage2mlx/q8/):
    transformer.safetensors      # quantized weights (packed) + scales/biases
    text_encoder.safetensors
    meta.json                    # {bits, group_size}
The VAE is always loaded fresh from the original in fp16.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn

GROUP_SIZE = 64


def _flatten_params(module):
    from mlx.utils import tree_flatten
    return dict(tree_flatten(module.parameters()))


def quantize_module(module, bits):
    """In-place quantize all eligible Linear/Embedding layers."""
    nn.quantize(module, group_size=GROUP_SIZE, bits=bits)
    return module


def save_quant(bits, transformer, text_encoder, dest):
    os.makedirs(dest, exist_ok=True)
    mx.save_safetensors(os.path.join(dest, "transformer.safetensors"),
                        _flatten_params(transformer))
    mx.save_safetensors(os.path.join(dest, "text_encoder.safetensors"),
                        _flatten_params(text_encoder))
    json.dump({"bits": bits, "group_size": GROUP_SIZE},
              open(os.path.join(dest, "meta.json"), "w"))


def has_saved_quant(dest):
    return (
        os.path.exists(os.path.join(dest, "transformer.safetensors"))
        and os.path.exists(os.path.join(dest, "text_encoder.safetensors"))
        and os.path.exists(os.path.join(dest, "meta.json"))
    )


def load_quant_into(transformer, text_encoder, dest):
    """Apply quant structure then load packed weights from `dest`."""
    meta = json.load(open(os.path.join(dest, "meta.json")))
    bits = meta["bits"]
    quantize_module(transformer, bits)
    quantize_module(text_encoder, bits)
    transformer.load_weights(os.path.join(dest, "transformer.safetensors"), strict=False)
    text_encoder.load_weights(os.path.join(dest, "text_encoder.safetensors"), strict=False)
    return transformer, text_encoder
