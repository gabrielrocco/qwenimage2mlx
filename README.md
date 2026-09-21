# qwenimage2mlx

Run **Qwen-Image-2.1** text-to-image on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) — like [mflux](https://github.com/filipstrand/mflux), but for the 2.1 architecture (block-causal single-stream DiT, `causal_condition`, 4096-dim, z_dim=64 VAE) that mflux does not support.

Downloads the original weights from the Hugging Face Hub, optionally quantizes to **q8/q4 in real time**, caches the quantization, and generates images on the GPU.

## Install

```bash
pip install qwenimage2mlx
```

## Python

```python
from qwenimage2mlx import QwenImage21

pipe = QwenImage21(quantize="q8")          # None (fp16) | "q8" | "q4"
img = pipe.generate(
    'a cat holding a sign that says "MLX WORKS", photorealistic',
    width=1024, height=1024, steps=20, true_cfg_scale=4.0,
    open=False,     # open the image after generating
    save=True,      # write the PNG to disk
)
```

- First run with `quantize="q8"` quantizes and **saves** the result to `~/.cache/qwenimage2mlx/q8/`; later runs load it back instantly.
- The original weights download once to `~/.cache/qwenimage2mlx/original/`.

## CLI

```bash
qwen21-generate "a red fox in snow, cinematic" -q q8 --steps 20 --open
```

```
-q/--quantize   q8 | q4          (cached + reused after first run)
-W/-H           width / height   (default 1024)
-s/--steps      inference steps  (default 20)
--cfg           true CFG scale   (default 4.0)
--seed          seed             (default 42)
-o/--output     PNG path
--open          open image after generating
--no-save       don't persist the quantization
```

## Storage

```
~/.cache/qwenimage2mlx/
├── original/     # HF weights (fp16 source)
├── q8/           # cached 8-bit quantization
└── q4/           # cached 4-bit quantization
```

Override with `QWENIMAGE2MLX_CACHE`.

## Notes

- The VAE always runs in fp16 (quantizing it degrades the decode); q8/q4 apply to the transformer and text encoder.
- Downloads are resumable and fault-tolerant: interrupted files continue, complete files are skipped.
