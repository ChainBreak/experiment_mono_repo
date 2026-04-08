"""Lightning module: encoder–decoder reconstruction with identity conditioning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import lightning as L

import deepfake.dataset as dataset_module
import deepfake.decoder as decoder_module
import deepfake.encoder as encoder_module


class LitModule(L.LightningModule):
    """Encoder–decoder autoencoder with learned identity embeddings."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder_module.Encoder(config.encoder)
        self.decoder = decoder_module.Decoder(config.decoder)
        self.identity_embedding = nn.Embedding(
            len(config.dataset.identity_folders),
            int(config.decoder.identity_dim),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        latent = self.encoder(batch["input_image"])
        identity_vector = self.identity_embedding(batch["identity"].squeeze(-1).long())
        reconstruction = self.decoder(latent, identity_vector)
        loss = F.mse_loss(reconstruction, batch["target_image"])
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_image_as_grid(batch["input_image"].detach(), "train/input_image")
            self.log_image_as_grid(reconstruction.detach(), "train/reconstruction")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.config.training.learning_rate),
        )

    def train_dataloader(self) -> DataLoader:
        dataset = dataset_module.IdentityImageDataset(self.config.dataset)
        return DataLoader(
            dataset,
            batch_size=int(self.config.training.batch_size),
            num_workers=int(self.config.training.workers),
        )

    def log_image_as_grid(self, images: torch.Tensor, tag: str) -> None:
        """Log a batch of images (N, C, H, W) as a grid via ``torchvision.utils.make_grid``."""
        if self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return
        n = int(images.shape[0])
        if n == 0:
            raise ValueError("images must contain at least one image")
        nrow = int(math.ceil(math.sqrt(n)))
        grid = make_grid(images.detach(), nrow=nrow, padding=0)
        experiment.add_image(
            tag,
            grid.cpu(),
            self.global_step,
            dataformats="CHW",
        )
