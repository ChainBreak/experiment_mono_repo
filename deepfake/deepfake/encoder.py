"""Convolutional encoder: stem, staged BasicBlock and DownBlock at stage boundaries."""

from collections.abc import Generator, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, ReLU, and a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 3,
        norm: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
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
        self.downsample: nn.Module | None
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
        else:
            self.downsample = None

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
        *,
        kernel_size: int = 3,
        norm: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
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


class Encoder(nn.Module):
    """Encoder built from ``blocks`` and ``channels`` per stage.

    Stage ``i`` applies ``blocks[i]`` blocks at width ``channels[i]``.
    The first block of each stage after the first uses :class:`DownBlock` (stride-2)
    to halve spatial size; other blocks use :class:`BasicBlock`.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        blocks = list(config.blocks)
        channels = list(config.channels)
        if len(blocks) != len(channels):
            raise ValueError("blocks and channels must have the same length")
        if len(blocks) < 1:
            raise ValueError("at least one stage is required")
        self.in_channels = int(config.in_channels)
        self.blocks_per_stage = tuple(blocks)
        self.channels_per_stage = tuple(channels)
        self.layers = nn.Sequential(
            *list(self.yield_layers(self.blocks_per_stage, self.channels_per_stage))
        )

    def yield_layers(
        self,
        blocks: Sequence[int],
        channels: Sequence[int],
    ) -> Generator[nn.Module, None, None]:
        kernel_size = 3
        padding = kernel_size // 2
        yield nn.Conv2d(
            self.in_channels,
            channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        yield nn.BatchNorm2d(channels[0])
        yield nn.ReLU(inplace=True)

        for stage_index in range(len(blocks)):
            for block_index in range(blocks[stage_index]):
                if stage_index == 0:
                    in_ch = channels[0]
                    out_ch = channels[0]
                elif block_index == 0:
                    in_ch = channels[stage_index - 1]
                    out_ch = channels[stage_index]
                else:
                    in_ch = channels[stage_index]
                    out_ch = channels[stage_index]
                if stage_index > 0 and block_index == 0:
                    yield DownBlock(in_ch, out_ch)
                else:
                    yield BasicBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
