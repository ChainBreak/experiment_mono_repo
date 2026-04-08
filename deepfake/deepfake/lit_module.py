"""Lightning module: encoder–decoder reconstruction with identity conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
            num_workers=0,
        )
