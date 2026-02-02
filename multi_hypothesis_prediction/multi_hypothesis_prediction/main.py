"""CLI entrypoint for multi-hypothesis prediction."""

import click
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
    trainer = L.Trainer(max_steps=100)
    trainer.fit(lit_module,
        train_dataloaders=DataLoader(batch_size=10),
    )
    plot(lit_module)

def plot(model: LitModule) -> None:
    with torch.no_grad():

        batch_size = 10_000
        batch = next(iter(DataLoader(batch_size=batch_size)))
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x)
        plt.scatter(x.numpy(), y.numpy(), label="True")
        plt.scatter(x.numpy(), y_pred.numpy(), label="Predicted")
        plt.legend()
        plt.savefig("prediction.png")



if __name__ == "__main__":
    cli()
