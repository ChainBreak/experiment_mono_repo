from __future__ import annotations

from collections import defaultdict

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from toy_deepfake.centering import Centering
from toy_deepfake.dataset.dataset import EllipseClusterDataset

class ToyAutoencoderLitModule(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self._config = config

        self.encoder = self.create_encoder()
        self.centering = Centering(
            num_identities=self._config.model.num_identities,
            shape=(self._config.model.latent_dim,),
        )
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
        z_offset, loss_center = self.centering(z, identity)
        y = self.decoder(z_offset)
        loss = nn.functional.mse_loss(y, x)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_center", loss_center, prog_bar=True)
        return loss + loss_center

    def on_validation_start(self) -> None:
        self._val_points: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["point"]
        identity = batch["identity"]
        z = self.encoder(x)
        z_offset, loss_center = self.centering(z, identity)
        y = self.decoder(z_offset)
        self._val_points["x"].append(x.detach().cpu())
        self._val_points["z"].append(z.detach().cpu())
        self._val_points["z_offset"].append(z_offset.detach().cpu())
        self._val_points["y"].append(y.detach().cpu())
        self._val_points["identity"].append(identity.detach().cpu())

    def on_validation_epoch_end(self) -> None:

        ident = torch.cat(self._val_points["identity"], dim=0).squeeze(-1).numpy()
        centers = self.centering.identity_centers.detach().cpu().numpy()

        for name in ["x", "z", "z_offset", "y"]:
            points = torch.cat(self._val_points[name], dim=0).numpy()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(points[:, 0], points[:, 1], c=ident, cmap="tab10", s=4, alpha=0.7)
            # ax.scatter(centers[:, 0], centers[:, 1], color="black", s=4, marker="x")
            ax.set_title(name)
            ax.set_aspect("equal", adjustable="box")
            self._log_figure(name, fig)

    def _log_figure(self, name: str, fig: plt.Figure) -> None:
        logger = self.logger
        if logger is None:
            plt.close(fig)
            return
        loggers = getattr(logger, "loggers", [logger])
        for log in loggers:
            if isinstance(log, TensorBoardLogger):
                log.experiment.add_figure(name, fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), 
            lr=self._config.training.learning_rate,
        )

    def train_dataloader(self) -> DataLoader:

        dataset = EllipseClusterDataset(
            length=self._config.dataset_train.num_samples,
            point_dim=self._config.dataset_train.point_dim,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self._config.training.batch_size,
            num_workers=self._config.training.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = EllipseClusterDataset(
            length=self._config.dataset_val.num_samples,
            point_dim=self._config.dataset_val.point_dim,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self._config.training.batch_size,
            num_workers=self._config.training.num_workers,
            shuffle=False,
        )
