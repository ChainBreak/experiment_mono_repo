"""ResNet-style residual blocks and optional up/down variants."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
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
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.activation(out)
        return out


class ResidualBlockDown(nn.Module):
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
        self.block = ResidualBlock(
            in_channels,
            out_channels,
            stride=2,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlockUp(nn.Module):
    """Upsample by ``scale_factor`` then apply a residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        scale_factor: int = 2,
        mode: str = "nearest",
        kernel_size: int = 3,
        norm: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.block = ResidualBlock(
            in_channels,
            out_channels,
            stride=1,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.block(x)
