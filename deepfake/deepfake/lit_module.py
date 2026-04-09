"""Lightning module: encoder–decoder reconstruction with identity conditioning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import lightning as L

import deepfake.dataset as dataset_module
import deepfake.decoder as decoder_module
import deepfake.discriminator as discriminator_module
import deepfake.encoder as encoder_module


class LitModule(L.LightningModule):
    """Encoder–decoder autoencoder with learned identity embeddings and latent-space discriminator."""

    automatic_optimization = False

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        OmegaConf.resolve(config)
        self.config = config
        self.encoder = encoder_module.Encoder(config.encoder)
        self.decoder = decoder_module.Decoder(config.decoder)
        num_identities = len(config.dataset.identity_folders)
        self.identity_embedding = nn.Embedding(
            num_identities,
            int(config.decoder.identity_dim),
        )

        d_cfg = config.discriminator
        num_classes = int(d_cfg.num_classes)
        assert num_classes == num_identities, (
            f"discriminator.num_classes ({num_classes}) must match "
            f"len(dataset.identity_folders) ({num_identities})"
        )
        assert int(config.latent_dim) == int(d_cfg.in_channels)
        self.discriminator = discriminator_module.Discriminator(d_cfg)
        self._lambda_inv = float(d_cfg.lambda_inv)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        optimizers = self.optimizers()
        opt_ae = optimizers[0]
        opt_d = optimizers[1]

        x = batch["input_image"]
        target_img = batch["target_image"]
        identity_idx = batch["identity"].squeeze(-1).long()

        latent = self.encoder(x)
        identity_vector = self.identity_embedding(identity_idx)
        reconstruction = self.decoder(latent, identity_vector)
        loss_recon = F.mse_loss(reconstruction, target_img)

        was_training_d = self.discriminator.training
        self.discriminator.eval()
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        logits_inv = self.discriminator(latent)
        loss_inv = logits_inv.var().mean()
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        self.discriminator.train(was_training_d)

        loss_ae = loss_recon + self._lambda_inv * loss_inv

        opt_ae.zero_grad()
        self.manual_backward(loss_ae)
        opt_ae.step()

        logits_d = self.discriminator(latent.detach())
        b, _, h, w = logits_d.shape
        ce_target = identity_idx[:, None, None].expand(b, h, w)
        loss_d = F.cross_entropy(logits_d, ce_target)

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        self.log("train_loss", loss_ae, prog_bar=True)
        self.log("train_loss_recon", loss_recon)
        self.log("train_loss_inv", loss_inv)
        self.log("train_loss_d", loss_d)

        if batch_idx == 0:
            self.log_image_as_grid(batch["input_image"].detach(), "train/input_image")
            self.log_image_as_grid(reconstruction.detach(), "train/reconstruction")

    def configure_optimizers(self):
        lr = float(self.config.training.learning_rate)
        d_cfg = self.config.discriminator
        d_lr = float(getattr(d_cfg, "learning_rate", lr))
        params_ae = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.identity_embedding.parameters())
        )
        opt_ae = torch.optim.Adam(params_ae, lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr)
        return [opt_ae, opt_d]

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
