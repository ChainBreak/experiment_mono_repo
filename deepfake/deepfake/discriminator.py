"""Spatial discriminator on encoder latent: per-location identity logits (PixelGAN-style on features)."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class Discriminator(nn.Module):
    """Same-resolution conv stack: ``[B, C, H, W]`` logits -> ``[B, num_classes, H, W]``."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        in_channels = int(config.in_channels)
        hidden_channels = int(config.hidden_channels)
        num_classes = int(config.num_classes)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
