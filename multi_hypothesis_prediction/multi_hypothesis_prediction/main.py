"""CLI entrypoint for multi-hypothesis prediction."""

import click
import pathlib
import datetime
from multi_hypothesis_prediction.lit_module import LitModule
import lightning as L
from multi_hypothesis_prediction.dataloader import DataLoader
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from ghostconfig import GhostConfig

@click.group()
@click.version_option()
def cli() -> None:
    """Multi-hypothesis prediction CLI."""
    pass


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to the config yaml file.")
def train(config_path: str) -> None:
    """Train the model."""
    config = GhostConfig.create(config_path)

    lit_module = LitModule(config["model"])
    trainer = L.Trainer(
        max_steps=config["training"].get("max_steps", 30000),
        logger=TensorBoardLogger("lightning_logs"),
        )
    trainer.fit(lit_module,
        train_dataloaders=DataLoader(config["dataset"]),
    )
    config.check()
    plot(lit_module, config)

def plot(model: LitModule, config: GhostConfig) -> None:
    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    with torch.no_grad():

        batch = next(iter(DataLoader(config["dataset"])))
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x).sample()
        plt.scatter(x.numpy(), y.numpy(), label="True", alpha=0.1)
        plt.scatter(x.numpy(), y_pred.numpy(), label="Predicted", alpha=0.1)
        plt.legend()
        plt.savefig(output_path)



if __name__ == "__main__":
    cli()
