"""Convolutional encoder: stem, staged BasicBlock and DownBlock at stage boundaries."""

from collections.abc import Callable, Generator

import torch
import torch.nn as nn
from omegaconf import DictConfig


def group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class Encoder(nn.Module):
    """Encoder built from ``blocks_per_stage`` and ``channels_per_stage``.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.blocks_per_stage = list(config.blocks)
        self.channels_per_stage = list(config.channels)
        self.in_channels = int(config.in_channels)

        self.layers = nn.Sequential(
            *list(self.yield_layers())
        )

    def yield_layers(self) -> Generator[nn.Module, None, None]:
        in_channels = self.in_channels
        blocks_and_channels = zip(self.blocks_per_stage, self.channels_per_stage) 

        for i, (blocks, out_channels) in enumerate(blocks_and_channels):
            if i > 0:
                yield DownBlock(in_channels, out_channels)
                in_channels = out_channels

            for _ in range(blocks):
                yield BasicBlock(in_channels, out_channels)
                in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with group norm, SiLU, and a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        norm: Callable[[int], nn.Module] = group_norm,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = norm(out_channels)
        self.activation = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = norm(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm(out_channels),
            )
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = out + shortcut
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Residual block whose first convolution uses stride 2 (spatial halving)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: Callable[[int], nn.Module] = group_norm,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.block = BasicBlock(
            in_channels,
            out_channels,
            stride=2,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
