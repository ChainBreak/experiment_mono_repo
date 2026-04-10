from pathlib import Path

import albumentations
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
def train(config_path: Path) -> None:
    """Train the encoder–decoder from a YAML config file."""
    loaded_config = omegaconf.OmegaConf.load(config_path)
    omegaconf.OmegaConf.resolve(loaded_config)
    model = lit_module_module.LitModule(loaded_config)
    trainer = L.Trainer(
        max_epochs=int(loaded_config.training.max_epochs),
        limit_train_batches=int(loaded_config.training.steps_per_epoch),
        logger=TensorBoardLogger(save_dir="lightning_logs"),
    )
    trainer.fit(model)


main.add_command(render_module.render_cmd)


if __name__ == "__main__":
    main()
