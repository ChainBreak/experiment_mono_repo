"""Shape checks for encoder/decoder on a fixed spatial size."""

import torch
from omegaconf import OmegaConf

import deepfake.decoder as decoder_module
import deepfake.encoder as encoder_module


def test_64_roundtrip_shapes() -> None:
    batch_size = 1
    height = 64
    width = 64
    encoder = encoder_module.Encoder(
        OmegaConf.create(
            {
                "in_channels": 3,
                "blocks": [2, 2, 4, 4],
                "channels": [32, 64, 128, 128],
            }
        )
    )
    decoder = decoder_module.Decoder(
        OmegaConf.create(
            {
                "out_channels": 3,
                "blocks": [4, 4, 2, 2],
                "channels": [128, 128, 64, 32],
            }
        )
    )
    x = torch.randn(batch_size, 3, height, width)
    latent = encoder(x)
    assert latent.shape == (batch_size, 128, 8, 8)
    reconstruction = decoder(latent)
    assert reconstruction.shape == (batch_size, 3, height, width)
