<div align="center">

# qwenimage2mlx

**Run [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) text-to-image on Apple Silicon, natively in [MLX](https://github.com/ml-explore/mlx).**

Like [mflux](https://github.com/filipstrand/mflux) — but for the **2.1** architecture that mflux does not support.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MLX](https://img.shields.io/badge/backend-MLX-black)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M5-lightgrey)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

</div>

---

Qwen-Image-2.1 uses a **block-causal single-stream DiT** (4096-dim, `causal_condition`) with a **z_dim=64 VAE** — a different architecture from the original Qwen-Image and from Qwen-Image-2512. This package is a from-scratch MLX reimplementation of the transformer, VAE, and Qwen3-VL text encoder, each validated numerically against the reference `diffusers` implementation (relative error ~1e-5).

- 🎨 **Text-to-image** with the full 2.1 quality (complex text rendering intact)
- ⚡ **Real-time q8 / q4 quantization**, cached to disk and reused
- 📦 **Fault-tolerant download** from the Hugging Face Hub — resumable, self-verifying, with a clean progress bar
- 🖥️ **CLI + Python API**
- 🍎 Runs on the Apple GPU via MLX — no PyTorch/CUDA at inference

---

## Install

```bash
pip install git+https://github.com/gabrielrocco/qwenimage2mlx.git
```

Requires Python 3.10+ on Apple Silicon (M1–M5). Weights download automatically on first run (~31 GB, once) to `~/.cache/qwenimage2mlx/`.

## Quick start

### CLI

```bash
qwen21-generate "a cat holding a sign that says 'MLX WORKS', photorealistic" --open
```

```bash
# 8-bit quantized, custom size, open when done
qwen21-generate "a red fox in snow, cinematic" -q q8 -W 1024 -H 1024 -s 20 --open
```

| flag | default | description |
|------|---------|-------------|
| `prompt` | — | text prompt (positional) |
| `-n, --negative-prompt` | `" "` | negative prompt |
| `-q, --quantize` | none (bf16) | `q8` or `q4` — cached and reused after first run |
| `-W, --width` | `1024` | image width |
| `-H, --height` | `1024` | image height |
| `-s, --steps` | `20` | inference steps |
| `--cfg` | `4.0` | true CFG scale |
| `--seed` | `42` | random seed |
| `-o, --output` | `~/qwen21_out.png` | output PNG path |
| `--open` | off | open the image after generating |
| `--no-save` | — | don't persist the quantization to disk |

### Python

```python
from qwenimage2mlx import QwenImage21

pipe = QwenImage21()                      # bf16 (default, best quality)
# pipe = QwenImage21(quantize="q8")       # 8-bit — quantizes once, then reuses the cache

img = pipe.generate(
    'a corgi wearing sunglasses on a beach, photorealistic',
    width=1024, height=1024,
    steps=20, true_cfg_scale=4.0, seed=42,
    open=False,   # open the image after generating
    save=True,    # write the PNG to disk
)                 # -> PIL.Image
img.save("out.png")
```

## Precision modes

| mode | call | transformer RAM | quality | notes |
|------|------|-----------------|---------|-------|
| **bf16** | `QwenImage21()` | ~13 GB | best | recommended when RAM allows |
| **q8** | `QwenImage21(quantize="q8")` | ~7.5 GB | ~identical to bf16 | good RAM/quality trade-off |
| **q4** | `QwenImage21(quantize="q4")` | ~4 GB | reduced | for memory-constrained Macs |

The VAE always runs in bf16 (quantizing it degrades the decode); quantization applies to the transformer and text encoder. First use of a quant level quantizes and **saves** it under `~/.cache/qwenimage2mlx/q8|q4/`; later runs load it back in seconds.

## How it works

```
prompt ──► Qwen3-VL text encoder (MLX) ──┐
                                         ├──► block-causal DiT, 20 steps (MLX) ──► VAE decode (MLX) ──► PNG
noise latents ───────────────────────────┘
```

Three components, each a faithful MLX port validated against `diffusers`:

| component | file | validated |
|-----------|------|-----------|
| single-stream DiT (block-causal, `causal_condition`) | `models/transformer.py` | ✅ ~8e-6 |
| AutoencoderKLQwenImage21 decoder (z_dim=64) | `models/vae.py` | ✅ ~7e-6 |
| Qwen3-VL text encoder (4096-dim, 36 layers) | `models/text_encoder.py` | ✅ ~5e-6 |

## Storage

```
~/.cache/qwenimage2mlx/
├── original/     # HF weights (bf16 source, downloaded once)
├── q8/           # cached 8-bit quantization
└── q4/           # cached 4-bit quantization
```

Override the location with the `QWENIMAGE2MLX_CACHE` environment variable.

## Dependencies

`mlx` · `transformers` (tokenizer) · `diffusers` (scheduler) · `torch` (scheduler step) · `huggingface_hub` · `safetensors` · `numpy` · `pillow` · `tqdm`

Installed automatically by `pip`.

## Performance notes

- Best quality is **20 steps + CFG 4.0**. Fewer steps (6–8) are fine for quick previews but look under-cooked.
- On Apple Silicon the runtime is dominated by dense matmuls in the ~7B DiT, which are bandwidth/compute-bound on the GPU. MLX's advantage here is **memory efficiency and quantization**, letting the model run on Macs where a full PyTorch load would not fit — raw matmul throughput is bounded by the same hardware either way.
- No official few-step (Lightning) LoRA exists for the 2.1 architecture yet; when one ships it can be merged into the transformer's linear layers.

## License

Apache-2.0. Model weights are © their original authors ([Qwen](https://huggingface.co/Qwen/Qwen-Image-2.1)) under their own license.
