"""qwenimage2mlx — run Qwen-Image-2.1 on Apple Silicon via MLX, like mflux."""
from .pipeline import QwenImage21
from .download import ensure_original

__version__ = "0.1.0"
__all__ = ["QwenImage21", "ensure_original"]
