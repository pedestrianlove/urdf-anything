"""Training: DiT trainer and distributed setup."""
from .distributed import setup_distributed, cleanup_distributed
from .trainer import DiTTrainer

__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "DiTTrainer",
]
