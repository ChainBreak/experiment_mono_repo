"""CLI entrypoint for multi-hypothesis prediction."""

import click
import pathlib
import datetime
from multi_hypothesis_prediction.lit_module import LitModule
import lightning as L
from multi_hypothesis_prediction.dataloader import DataLoader
import matplotlib.pyplot as plt
import torch

@click.group()
@click.version_option()
def cli() -> None:
    """Multi-hypothesis prediction CLI."""
    pass


@cli.command()
def train() -> None:
    """Train the model."""
    lit_module = LitModule()
    trainer = L.Trainer(max_steps=1000)
    trainer.fit(lit_module,
        train_dataloaders=DataLoader(batch_size=256),
    )
    plot(lit_module)

def plot(model: LitModule) -> None:
    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    with torch.no_grad():

        batch_size = 10_000
        batch = next(iter(DataLoader(batch_size=batch_size)))
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x)
        plt.scatter(x.numpy(), y.numpy(), label="True")
        plt.scatter(x.numpy(), y_pred.numpy(), label="Predicted")
        plt.legend()
        plt.savefig(output_path)



if __name__ == "__main__":
    cli()
