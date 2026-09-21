"""CLI: qwen21-generate 'a cat holding a sign' -q q8 --open"""
import argparse

import mlx.core as mx

from .pipeline import QwenImage21


def main():
    p = argparse.ArgumentParser(prog="qwen21-generate",
                                description="Generate images with Qwen-Image-2.1 on MLX.")
    p.add_argument("prompt", help="text prompt")
    p.add_argument("-n", "--negative-prompt", default=" ")
    p.add_argument("-q", "--quantize", choices=["q8", "q4"], default=None,
                   help="quantize to q8/q4 (cached and reused after first run)")
    p.add_argument("-W", "--width", type=int, default=1024)
    p.add_argument("-H", "--height", type=int, default=1024)
    p.add_argument("-s", "--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--open", action="store_true", help="open the image after generating")
    p.add_argument("--no-save", dest="save", action="store_false",
                   help="do not persist the quantization to disk")
    args = p.parse_args()

    pipe = QwenImage21(quantize=args.quantize, dtype=mx.bfloat16, save=args.save)
    pipe.generate(
        args.prompt, negative_prompt=args.negative_prompt,
        width=args.width, height=args.height, steps=args.steps,
        true_cfg_scale=args.cfg, seed=args.seed,
        output=args.output, open=args.open, save=True,
    )


if __name__ == "__main__":
    main()
