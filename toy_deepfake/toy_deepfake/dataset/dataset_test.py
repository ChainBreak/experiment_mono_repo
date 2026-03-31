from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from toy_deepfake.dataset.dataset import EllipseClusterDataset

# Project root: .../toy_deepfake/ (contains pyproject.toml, test_outputs/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_OUTPUTS = _PROJECT_ROOT / "test_outputs"


@pytest.fixture(autouse=True)
def _ensure_test_outputs_dir() -> None:
    TEST_OUTPUTS.mkdir(parents=True, exist_ok=True)


def test_ellipse_cluster_dataset_scatter_plot() -> None:
    torch.manual_seed(0)
    n = 3000
    ds = EllipseClusterDataset(length=n, point_dim=2)
    assert len(ds) == n

    xs: list[float] = []
    ys: list[float] = []
    ids: list[int] = []
    for i in range(n):
        batch = ds[i]
        p = batch["point"]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        ids.append(int(batch["identity"].item()))

    assert set(ids) <= {0, 1, 2}
    assert len(set(ids)) == 3

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for k in (0, 1, 2):
        mask = [j for j, id_ in enumerate(ids) if id_ == k]
        ax.scatter(
            [xs[j] for j in mask],
            [ys[j] for j in mask],
            s=4,
            alpha=0.5,
            c=colors[k],
            label=f"cluster {k}",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2)
    ax.set_title("EllipseClusterDataset samples")
    fig.tight_layout()

    out_path = TEST_OUTPUTS / "ellipse_clusters.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0
