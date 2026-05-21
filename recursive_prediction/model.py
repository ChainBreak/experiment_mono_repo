import torch
import torch.nn as nn
from ghostconfig import GhostConfig


class CifarCnn(nn.Module):
    def __init__(self, config: GhostConfig):
        super().__init__()
        conv_channels = config.get("conv_channels", [32, 64, 128])
        hidden_size = config.get("hidden_size", 256)

        # Stack conv blocks: Conv2d -> ReLU -> MaxPool2d(2)
        # Input: (B, 3, 32, 32)
        # After 3 pools: (B, channels[-1], 4, 4)
        conv_layers = []
        in_channels = 3
        for out_channels in conv_channels:
            conv_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_layers)

        spatial_size = 32 // (2 ** len(conv_channels))
        flattened_size = conv_channels[-1] * spatial_size * spatial_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(image)
        return self.classifier(features)
