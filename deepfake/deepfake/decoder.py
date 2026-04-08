"""Convolutional decoder: staged IdentityBlock, upsampling between stages, RGB head."""

from collections.abc import Generator, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig


class IdentityBlock(nn.Module):
    """Residual block with channel-wise modulation from a conditioning vector."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_dim: int,
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
        self.identity_linear = nn.Linear(identity_dim, out_channels)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
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
        scale = self.identity_linear(identity)
        out = out * scale.unsqueeze(-1).unsqueeze(-1)
        return out


class UpBlock(nn.Module):
    """Upsample then apply an :class:`IdentityBlock`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_dim: int,
        *,
        scale_factor: float = 2.0,
        mode: str = "nearest",
        kernel_size: int = 3,
        norm: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.block = IdentityBlock(
            in_channels,
            out_channels,
            identity_dim,
            stride=1,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.block(x, identity)


class Decoder(nn.Module):
    """Decoder with ``channels`` ordered bottleneck (coarse) to full resolution.

    Stage ``i`` applies ``blocks[i]`` identity blocks at width ``channels[i]``.
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
        self.identity_dim = int(config.get("identity_dim", 128))
        self.layers = nn.ModuleList(
            list(self.yield_layers(self.blocks_per_stage, self.channels_per_stage))
        )

    def yield_layers(
        self,
        blocks: Sequence[int],
        channels: Sequence[int],
    ) -> Generator[nn.Module, None, None]:
        identity_dim = self.identity_dim
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
                yield IdentityBlock(in_ch, out_ch, identity_dim)

        yield nn.Conv2d(channels[-1], self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            if isinstance(module, IdentityBlock):
                x = module(x, identity)
            else:
                x = module(x)
        return x
