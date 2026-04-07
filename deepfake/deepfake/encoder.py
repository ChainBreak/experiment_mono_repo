"""Convolutional encoder: stem, staged residual blocks, max-pooling between stages."""

from collections.abc import Generator, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

import deepfake.blocks as blocks_module


class Encoder(nn.Module):
    """Encoder built from ``blocks`` and ``channels`` per stage.

    Stage ``i`` applies ``blocks[i]`` residual blocks at width ``channels[i]``.
    Between stages (except after the last), a 2x2 max-pool halves spatial size.
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
                yield blocks_module.ResidualBlock(in_ch, out_ch)

            if stage_index < len(blocks) - 1:
                yield nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
