"""Convolutional decoder: staged IdentityBlock, learned upsampling between stages, RGB head."""

from collections.abc import Callable, Generator

import torch
import torch.nn as nn
from omegaconf import DictConfig


def group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class UpBlock(nn.Module):
    """2× spatial upsample: 1×1 conv expands channels for ``PixelShuffle(2)``, then shuffle."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # PixelShuffle(r) maps (N, C·r², H, W) → (N, C, H·r, W·r); r=2 ⇒ 4× channel expansion.
        # self.conv = nn.Conv2d(
        #     in_channels,
        #     in_channels * 4,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        # )
        # self.shuffle = nn.PixelShuffle(2)

        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class Decoder(nn.Module):
    """Decoder: 1×1 from ``latent_dim`` to the first stage width, then staged blocks (coarse to fine)."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.blocks_per_stage = list(config.blocks)
        self.channels_per_stage = list(config.channels)
        self.out_channels = int(config.out_channels)
        self.identity_dim = int(config.identity_dim)
        self.latent_dim = int(config.latent_dim)

        self.layers = nn.ModuleList(list(self.yield_layers()))

    def yield_layers(self) -> Generator[nn.Module, None, None]:
       
        in_channels = int(self.channels_per_stage[0])
        yield nn.Conv2d(
            self.latent_dim,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        blocks_and_channels = zip(self.blocks_per_stage, self.channels_per_stage)

        for i, (blocks, out_channels) in enumerate(blocks_and_channels):
            if i > 0:
                yield UpBlock(in_channels)

            for _ in range(blocks):
                yield IdentityBlock(
                    in_channels,
                    out_channels,
                    self.identity_dim,
                )
                in_channels = out_channels

        yield nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            if isinstance(module, IdentityBlock):
                x = module(x, identity)
            else:
                x = module(x)
        return x


class IdentityBlock(nn.Module):
    """Residual block with channel-wise modulation from a conditioning vector."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_dim: int,
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
            stride=1,
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
 
        self.identity_linear1 = nn.Linear(identity_dim, out_channels)
        self.identity_linear2 = nn.Linear(identity_dim, out_channels)

        self.downsample: nn.Module | None = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm(out_channels),
            )

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.conv1(x)
        out = out * self.identity_linear1(identity).unsqueeze(-1).unsqueeze(-1)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = out * self.identity_linear2(identity).unsqueeze(-1).unsqueeze(-1)
        out = self.norm2(out)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        out = out + shortcut
        out = self.activation(out)
        return out
