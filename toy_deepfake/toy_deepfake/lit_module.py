from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from toy_deepfake.dataset.dataset import EllipseClusterDataset


class ToyAutoencoderLitModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        ds_cfg = config["dataset"]
        self._dataset = EllipseClusterDataset(
            length=ds_cfg["num_samples"],
            point_dim=ds_cfg.get("point_dim", 2),
        )
        self.encoder = self.create_encoder()
        self.intermediate = self.create_intermediate()
        self.decoder = self.create_decoder()

    def create_encoder(self) -> nn.Module:
        m = self._config["model"]
        inp, hid, lat = m["input_dim"], m["hidden_dim"], m["latent_dim"]
        return nn.Sequential(
            nn.Linear(inp, hid),
            nn.SiLU(),
            nn.Linear(hid, lat),
        )

    def create_intermediate(self) -> nn.Module:
        return nn.Identity()

    def create_decoder(self) -> nn.Module:
        m = self._config["model"]
        inp, hid, lat = m["input_dim"], m["hidden_dim"], m["latent_dim"]
        return nn.Sequential(
            nn.Linear(lat, hid),
            nn.SiLU(),
            nn.Linear(hid, inp),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.intermediate(z)
        return self.decoder(z)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["point"]
        pred = self(x)
        loss = nn.functional.mse_loss(pred, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self._config["training"]["learning_rate"]
        return torch.optim.Adam(self.parameters(), lr=lr)

    def train_dataloader(self) -> DataLoader:
        t = self._config["training"]
        nw = t.get("num_workers", 0)
        return DataLoader(
            self._dataset,
            batch_size=t["batch_size"],
            shuffle=True,
            num_workers=nw,
            persistent_workers=nw > 0,
        )
