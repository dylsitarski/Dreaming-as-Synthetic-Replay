"""Classifier architectures."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Two conv blocks followed by an MLP head.

    `width` scales capacity (channels and hidden units). `dropout` defaults to 0 so that the
    plain model overfits. The dropout baseline sets it above 0.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        width: int = 32,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 2 * width * (image_size // 4) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: dict, in_channels: int, num_classes: int, image_size: int) -> nn.Module:
    cfg = dict(cfg)
    arch = cfg.pop("arch", "small_cnn")
    if arch == "small_cnn":
        return SmallCNN(in_channels, num_classes, image_size, **cfg)
    raise ValueError(f"Unknown arch {arch!r}")
