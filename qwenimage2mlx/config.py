"""Paths and constants for qwenimage2mlx."""
import os

HF_REPO = "Qwen/Qwen-Image-2.1"

# Only these subfolders are needed for txt2img.
NEEDED_DIRS = ["transformer", "vae", "text_encoder", "processor", "scheduler"]
NEEDED_FILES = ["model_index.json"]

CACHE_DIR = os.environ.get(
    "QWENIMAGE2MLX_CACHE",
    os.path.expanduser("~/.cache/qwenimage2mlx"),
)
ORIGINAL_DIR = os.path.join(CACHE_DIR, "original")


def quant_dir(bits):
    """Directory for a saved quantization ('q8'->8, 'q4'->4)."""
    return os.path.join(CACHE_DIR, f"q{bits}")


SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
