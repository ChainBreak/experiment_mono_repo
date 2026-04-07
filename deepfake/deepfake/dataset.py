"""Identity-indexed image dataset with Albumentations."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import transforms as albumentations_pytorch_transforms
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def build_augmentation_pipeline(height: int, width: int) -> A.Compose:
    """Affine, color jitter, random resize crop, then float 0–1 and CHW tensor."""
    return A.Compose(
        [
            A.Affine(
                scale=(0.92, 1.08),
                rotate=(-12.0, 12.0),
                shear=(-8.0, 8.0),
                fit_output=False,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=(0.85, 1.15),
                contrast=(0.85, 1.15),
                saturation=(0.85, 1.15),
                hue=(-0.05, 0.05),
                p=1.0,
            ),
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.ToFloat(max_value=255.0),
            albumentations_pytorch_transforms.ToTensorV2(),
        ]
    )


def _collect_image_paths(identity_folders: list[str | Path]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for identity_index, folder in enumerate(identity_folders):
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Identity folder is not a directory: {root}")
        paths: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(path)
        if not paths:
            raise ValueError(f"No images found under identity folder: {root}")
        for path in sorted(paths):
            samples.append((path, identity_index))
    if not samples:
        raise ValueError("No images indexed: identity_folders is empty or produced no paths.")
    return samples


def _load_image_rgb_numpy(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


class IdentityImageDataset(Dataset):
    """One sample per image file; same augmented tensor for input and target."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        identity_folders: list[str | Path],
    ) -> None:
        self._samples = _collect_image_paths(list(identity_folders))
        self._transform = build_augmentation_pipeline(height=height, width=width)

    @classmethod
    def from_config(cls, config: DictConfig) -> IdentityImageDataset:
        dataset_config = config.dataset
        return cls(
            width=int(dataset_config.width),
            height=int(dataset_config.height),
            identity_folders=list(dataset_config.identity_folders),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _paths_for_sample(self, identity_index: int, image_path: Path) -> tuple[Path, Path]:
        """Future: return two paths when input/target may differ; same path for now."""
        return (image_path, image_path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, identity_index = self._samples[index]
        _input_path, _target_path = self._paths_for_sample(identity_index, image_path)
        image_numpy = _load_image_rgb_numpy(_input_path)
        augmented = self._transform(image=image_numpy)["image"]
        identity_tensor = torch.tensor([identity_index], dtype=torch.long)
        return {
            "input_image": augmented,
            "target_image": augmented,
            "identity": identity_tensor,
        }
