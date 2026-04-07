"""Convolutional decoder: staged residual blocks, upsampling between stages, RGB head."""

from collections.abc import Generator, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

import deepfake.blocks as blocks_module


class Decoder(nn.Module):
    """Decoder with ``channels`` ordered bottleneck (coarse) to full resolution.

    Stage ``i`` applies ``blocks[i]`` residual blocks at width ``channels[i]``.
    Between stages (except after the last), spatial size doubles via nearest upsampling.
    The final 1x1 convolution maps ``channels[-1]`` to ``out_channels``.
    Output activation is identity (no tanh); match your data range in training/loss.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        blocks = list(config.blocks)
        channels = list(config.channels)
        if len(blocks) != len(channels):
            raise ValueError("blocks and channels must have the same length")
        if len(blocks) < 1:
            raise ValueError("at least one stage is required")
        self.out_channels = int(config.out_channels)
        self.blocks_per_stage = tuple(blocks)
        self.channels_per_stage = tuple(channels)
        self.upsample_mode = str(config.get("upsample_mode", "nearest"))
        self.layers = nn.Sequential(
            *list(self.yield_layers(self.blocks_per_stage, self.channels_per_stage))
        )

    def yield_layers(
        self,
        blocks: Sequence[int],
        channels: Sequence[int],
    ) -> Generator[nn.Module, None, None]:
        for stage_index in range(len(blocks)):
            if stage_index > 0:
                yield nn.Upsample(scale_factor=2.0, mode=self.upsample_mode)

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

        yield nn.Conv2d(channels[-1], self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
