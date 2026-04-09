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

        self.discriminator = discriminator_module.Discriminator(config.discriminator)
   
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        optimizer_auto_encoder, optimizer_discriminator = self.optimizers()

        input_img = batch["input_image"]
        target_img = batch["target_image"]
        identity_idx = batch["identity"].squeeze(-1).long()

        # Update the autoencoder
        latent = self.encoder(input_img)
        identity_vector = self.identity_embedding(identity_idx)
        reconstruction = self.decoder(latent, identity_vector)

        discriminator_logits = self.discriminator(latent)
        loss_discriminator_applied = discriminator_logits.var(dim=1).mean()
        loss_reconstruction = F.mse_loss(reconstruction, target_img)

        loss_auto_encoder = loss_reconstruction + loss_discriminator_applied

        optimizer_auto_encoder.zero_grad()
        self.manual_backward(loss_auto_encoder)
        optimizer_auto_encoder.step()


        # Update the discriminator
        discriminator_logits = self.discriminator(latent.detach())
        b, _, h, w = discriminator_logits.shape
        ce_target = identity_idx[:, None, None].expand(b, h, w)
        loss_discriminator_update = F.cross_entropy(discriminator_logits, ce_target)

        optimizer_discriminator.zero_grad()
        self.manual_backward(loss_discriminator_update)
        optimizer_discriminator.step()

        self.log("train_loss", loss_auto_encoder, prog_bar=True)
        self.log("train_loss_reconstruction", loss_reconstruction)
        self.log("train_loss_discriminator_applied", loss_discriminator_applied)
        self.log("train_loss_discriminator_update", loss_discriminator_update)

        if batch_idx == 0:
            self.log_image_as_grid(input_img, "train/input_image")
            self.log_image_as_grid(reconstruction, "train/reconstruction")

    def configure_optimizers(self):
        lr = float(self.config.training.learning_rate)
        d_cfg = self.config.discriminator
        d_lr = float(getattr(d_cfg, "learning_rate", lr))
        params_ae = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.identity_embedding.parameters())
        )
        optimizer_auto_encoder = torch.optim.Adam(params_ae, lr=lr)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr)
        return [optimizer_auto_encoder, optimizer_discriminator]

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

        images = images.detach()

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
