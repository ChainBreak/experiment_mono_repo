from pathlib import Path

import albumentations
from typing import Any, cast
import click
import lightning as L
import omegaconf
import torch
from lightning.pytorch.loggers import TensorBoardLogger

import deepfake.lit_module as lit_module_module
import deepfake.render as render_module


@click.group()
@click.version_option(package_name="deepfake")
def main() -> None:
    """Deepfake experiments CLI."""


@main.command("hello")
def hello() -> None:
    """Smoke test that the environment is wired."""
    click.echo("deepfake: ok")


@main.command("check")
def check() -> None:
    """Import core deps (useful after install)."""
    click.echo(f"torch {torch.__version__}")
    click.echo(f"lightning {L.__version__}")
    click.echo(f"albumentations {albumentations.__version__}")


@main.command("train")
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Resume training from this Lightning checkpoint (weights, optimizer, step).",
)
def train(config_path: Path, checkpoint: Path | None) -> None:
    """Train the encoder–decoder from a YAML config file."""
    loaded_config = omegaconf.OmegaConf.load(config_path)
    omegaconf.OmegaConf.resolve(loaded_config)
    config_dict = omegaconf.OmegaConf.to_container(loaded_config, resolve=True)
    config_dict = cast(dict[str, Any], config_dict)
    model = lit_module_module.LitModule(config_dict)
    trainer = L.Trainer(
        max_epochs=int(loaded_config.training.max_epochs),
        limit_train_batches=int(loaded_config.training.steps_per_epoch),
        logger=TensorBoardLogger(save_dir="lightning_logs"),
    )
    # ckpt_path = str(checkpoint) if checkpoint is not None else None
    trainer.fit(model, ckpt_path=checkpoint)


main.add_command(render_module.render_cmd)


if __name__ == "__main__":
    main()
