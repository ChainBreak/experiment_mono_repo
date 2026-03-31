from __future__ import annotations

from pathlib import Path

import click
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from toy_deepfake.lit_module import ToyAutoencoderLitModule


@click.group()
def cli() -> None:
    """Toy deepfake CLI."""



@cli.command("train")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def train(config_path: Path) -> None:
    """Train the toy autoencoder from a YAML config."""
    config: DictConfig = OmegaConf.load(config_path)

    training = config.training
    seed = OmegaConf.select(config, "training.seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    model = ToyAutoencoderLitModule(config)

    loggers: list = []
    log_dir = training.log_dir
    name = training.logger_name
    if training.get("use_tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir=log_dir, name=name))
    if training.get("use_csv", False):
        loggers.append(CSVLogger(save_dir=log_dir, name=name))
    if not loggers:
        loggers.append(CSVLogger(save_dir=log_dir, name=name))

    trainer = Trainer(
        max_epochs=training.max_epochs,
        logger=loggers,
        default_root_dir=log_dir,
        accelerator=training.get("accelerator", "auto"),
        devices=training.get("devices", 1),
    )
    trainer.fit(model)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
