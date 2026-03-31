from __future__ import annotations

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from toy_deepfake.dataset.dataset import EllipseClusterDataset

class ToyAutoencoderLitModule(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self._config = config

        self.encoder = self.create_encoder()
        self.intermediate = self.create_intermediate()
        self.decoder = self.create_decoder()

    def create_encoder(self) -> nn.Module:
    
        return nn.Sequential(
            nn.Linear(
                in_features=self._config.model.input_dim,
                out_features=self._config.model.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self._config.model.hidden_dim, 
                out_features=self._config.model.latent_dim),
        )

    def create_intermediate(self) -> nn.Module:
        return nn.Identity()

    def create_decoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                in_features=self._config.model.latent_dim,
                out_features=self._config.model.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self._config.model.hidden_dim, 
                out_features=self._config.model.input_dim),
        )


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["point"]
        identity = batch["identity"]
        z = self.encoder(x)
        y = self.decoder(z)
        loss = nn.functional.mse_loss(y, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), 
            lr=self._config.training.learning_rate,
        )

    def train_dataloader(self) -> DataLoader:

        dataset = EllipseClusterDataset(
            length=self._config.dataset.num_samples,
            point_dim=self._config.dataset.point_dim,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self._config.training.batch_size,
            num_workers=self._config.training.num_workers,
            shuffle=True,
        )
