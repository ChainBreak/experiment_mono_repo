from __future__ import annotations

import torch
from torch.utils.data import Dataset


class EllipseClusterDataset(Dataset):
    """Synthetic 2D points from three Gaussian clusters (fixed centers, shared covariance)."""

    def __init__(
        self,
        length: int,
        point_dim: int = 2,
    ) -> None:
        if point_dim != 2:
            raise ValueError("point_dim must be 2 for this dataset")
        self._length = length
        self._point_dim = point_dim

        self._centers = torch.tensor(
            [
                [-4.0, -3.0],
                [6.0, -1.0],
                [0.0, 6.0],
            ],
            dtype=torch.float32,
        )

        # Hard-coded 2x2 covariance; sampling uses Cholesky L with L @ L^T = Σ.
        self._covariance = torch.tensor(
            [
                [1.0, 0.35],
                [0.35, 0.64],
            ],
            dtype=torch.float32,
        )
        self._chol = torch.linalg.cholesky(self._covariance)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        i = torch.randint(0, 3, (), dtype=torch.long)
        z = torch.randn(self._point_dim, dtype=torch.float32)
        point = self._chol @ z + self._centers[i]
        return {"point": point, "identity": i}
