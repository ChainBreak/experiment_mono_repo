"""Tests for identity image dataset."""

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

import deepfake.dataset as dataset_module


def test_sample_keys_shapes_ranges_and_input_equals_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        identity_directory = Path(tmp) / "id0"
        identity_directory.mkdir()
        image_path = identity_directory / "a.png"
        Image.new("RGB", (32, 48), color=(128, 64, 32)).save(image_path)

        config = OmegaConf.create(
            {
                "dataset": {
                    "width": 64,
                    "height": 64,
                    "identity_folders": [str(identity_directory)],
                }
            }
        )
        dataset = dataset_module.IdentityImageDataset(config.dataset)
        assert len(dataset) == 1
        sample = dataset[0]
        assert set(sample.keys()) == {"input_image", "target_image", "identity"}
        assert sample["input_image"].shape == (3, 64, 64)
        assert sample["target_image"].shape == (3, 64, 64)
        assert sample["identity"].shape == (1,)
        assert sample["identity"].dtype == torch.long
        assert int(sample["identity"][0].item()) == 0
        assert torch.equal(sample["input_image"], sample["target_image"])
        assert torch.all(sample["input_image"] >= 0.0)
        assert torch.all(sample["input_image"] <= 1.0)
