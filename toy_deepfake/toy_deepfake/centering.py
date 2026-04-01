from __future__ import annotations

import torch
from torch import nn


class Centering(nn.Module):
    """Transforms encoder output using cluster identity; placeholder returns the input unchanged."""

    def forward(self, point: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return point
