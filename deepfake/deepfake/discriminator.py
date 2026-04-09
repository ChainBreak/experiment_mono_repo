"""Spatial discriminator on encoder latent: per-location identity logits (PixelGAN-style on features)."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class Discriminator(nn.Module):
    """Same-resolution conv stack: ``[B, C, H, W]`` logits -> ``[B, num_classes, H, W]``."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        in_channels = int(config.in_channels)
        num_classes = int(config.num_classes)
        hidden_raw = getattr(config, "hidden_channels", None)
        hidden = int(hidden_raw) if hidden_raw is not None else min(in_channels, 128)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
