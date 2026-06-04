import torch
import torch.nn as nn
from ghostconfig import GhostConfig


class CifarCnn(nn.Module):
    def __init__(self, config: GhostConfig):
        super().__init__()
        conv_channels = config.get("conv_channels", [32, 64, 128])
        hidden_size = config.get("hidden_size", 256)
        num_classes = config.get("num_classes", 10)

        layers = []
        in_channels = 3
        for out_channels in conv_channels:
            layers.append(ConditionBlock(in_channels, out_channels, num_classes))
            in_channels = out_channels

        spatial_size = 32 // (2 ** len(conv_channels))
        flattened_size = conv_channels[-1] * spatial_size * spatial_size

        layers += [
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, image: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        x = image
        for layer in self.layers:
            if isinstance(layer, ConditionBlock):
                x = layer(x, probabilities)
            else:
                x = layer(x)
        return x


class ConditionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.condition_projection = nn.Linear(condition_dim, out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.relu(self.conv(x)))
        channel_weights = self.condition_projection(condition).unsqueeze(-1).unsqueeze(-1)
        return features * channel_weights
