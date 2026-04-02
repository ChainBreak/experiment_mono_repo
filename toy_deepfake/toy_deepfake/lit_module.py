from __future__ import annotations

from collections import defaultdict
from itertools import chain

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
        self.discriminator = self.create_discriminator()
        self.identity_offsets = nn.Parameter(torch.zeros(self._config.model.num_identities, self._config.model.input_dim),requires_grad=True)
        self.decoder = self.create_decoder()

        self.automatic_optimization = False

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
                out_features=self._config.model.input_dim,
                ),
        )
    
    def create_discriminator(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                in_features=self._config.model.latent_dim,
                out_features=self._config.model.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self._config.model.hidden_dim, 
                out_features=self._config.model.num_identities),
        )


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        main_optimizer, discriminator_optimizer = self.optimizers()
        x = batch["point"]
        identity = batch["identity"]

        # Main update

        main_optimizer.zero_grad()
        
        z = self.encoder(x)
        z_offset = z + self.identity_offsets[identity]
        y = self.decoder(z_offset)
        center_loss = self.descriminator_center_loss(z)
        reconstruction_loss = nn.functional.mse_loss(y, x)
        main_loss =  reconstruction_loss + center_loss

        main_loss.backward()
        main_optimizer.step()

        # Discriminator update
        discriminator_optimizer.zero_grad()
        discriminator_loss = self.discriminator_update_loss(z.detach(), identity)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        self.log("main_loss", main_loss, prog_bar=True)
        self.log("reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("center_loss", center_loss, prog_bar=True)
        self.log("discriminator_loss", discriminator_loss, prog_bar=True)


    def descriminator_center_loss(self, z: torch.Tensor) -> torch.Tensor:
        pred_logits = self.discriminator(z)
        # Uniform softmax iff class logits are equal; minimize per-sample variance across classes.
        return pred_logits.var(dim=-1).mean()

    def discriminator_update_loss(self, z: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        pred_logits = self.discriminator(z)
        loss = nn.functional.cross_entropy(pred_logits,identity)
        return loss

    def on_validation_start(self) -> None:
        self._val_points: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["point"]
        identity = batch["identity"]
        z = self.encoder(x)
        z_offset = z + self.identity_offsets[identity]
        y = self.decoder(z_offset)
        self._val_points["x"].append(x.detach().cpu())
        self._val_points["z"].append(z.detach().cpu())
        self._val_points["z_offset"].append(z_offset.detach().cpu())
        self._val_points["y"].append(y.detach().cpu())
        self._val_points["identity"].append(identity.detach().cpu())

    def on_validation_epoch_end(self) -> None:

        ident = torch.cat(self._val_points["identity"], dim=0).squeeze(-1).numpy()
        # centers = self.centering.identity_centers.detach().cpu().numpy()

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
        main_params = chain(
            self.encoder.parameters(),
            [self.identity_offsets],
            self.decoder.parameters(),
        )
        main_optimizer = torch.optim.Adam(
            params=main_params, 
            lr=self._config.training.learning_rate,
        )

        discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self._config.training.learning_rate,
        )

        return [main_optimizer, discriminator_optimizer]

    def train_dataloader(self) -> DataLoader:

        dataset = EllipseClusterDataset(
            length=self._config.dataset_train.num_samples,
            point_dim=self._config.dataset_train.point_dim,
            spiral_scale=self._config.dataset_train.spiral_scale,
            radius_noise_std=self._config.dataset_train.radius_noise_std,
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
            spiral_scale=self._config.dataset_val.spiral_scale,
            radius_noise_std=self._config.dataset_val.radius_noise_std,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self._config.training.batch_size,
            num_workers=self._config.training.num_workers,
            shuffle=False,
        )
