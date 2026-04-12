"""Shape checks for encoder/decoder on a fixed spatial size."""

import torch
from omegaconf import OmegaConf

import deepfake.decoder as decoder_module
import deepfake.encoder as encoder_module


def test_64_roundtrip_shapes() -> None:
    batch_size = 1
    height = 64
    width = 64
    latent_dim = 8
    encoder = encoder_module.Encoder(
        OmegaConf.create(
            {
                "latent_dim": latent_dim,
                "in_channels": 3,
                "blocks": [2, 2, 4, 4],
                "channels": [32, 64, 128, 256],
            }
        )
    )
    identity_dim = 128
    decoder = decoder_module.Decoder(
        OmegaConf.create(
            {
                "latent_dim": latent_dim,
                "out_channels": 3,
                "blocks": [4, 4, 2, 2],
                "channels": [256, 128, 64, 32],
                "identity_dim": identity_dim,
                "upsample_mode": "bilinear",
            }
        )
    )
    x = torch.randn(batch_size, 3, height, width)
    latent = encoder(x)
    assert latent.shape == (batch_size, latent_dim, 8, 8)
    identity = torch.randn(batch_size, identity_dim)
    reconstruction = decoder(latent, identity)
    assert reconstruction.shape == (batch_size, 3, height, width)
