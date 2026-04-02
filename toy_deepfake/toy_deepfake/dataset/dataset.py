from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class EllipseClusterDataset(Dataset):
    """Synthetic 2D points on three Archimedean spirals (one per identity), with fixed center offsets."""

    def __init__(
        self,
        length: int,
        point_dim: int = 2,
        spiral_scale: float = 0.5,
        radius_noise_std: float = 0.1,
    ) -> None:
        if point_dim != 2:
            raise ValueError("point_dim must be 2 for this dataset")
        self._length = length
        self._point_dim = point_dim
        self._spiral_scale = spiral_scale
        self._radius_noise_std = radius_noise_std

        self._centers = torch.tensor(
            [
                [-4.0, -3.0],
                [6.0, -1.0],
                [0.0, 6.0],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        i = torch.randint(0, 3, (), dtype=torch.long)
        angle = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
        radius = angle * self._spiral_scale + torch.randn((), dtype=torch.float32) * self._radius_noise_std
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        point = torch.stack([radius * cos_a, radius * sin_a]) + self._centers[i]
        return {"point": point, "identity": i}
