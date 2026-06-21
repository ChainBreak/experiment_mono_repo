import click
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from ghostconfig import GhostConfig

from lit_module import LitModule


@click.command()
@click.option("--config-path", default="config.yaml", help="Path to the config yaml file.")
def train(config_path: str) -> None:
    config = GhostConfig.create(config_path)

    module = LitModule(config)
    trainer = L.Trainer(
        max_epochs=config["training"].get("max_epochs", 50),
        limit_train_batches=config["training"].get("steps_per_epoch", 200),
        logger=TensorBoardLogger("lightning_logs"),
        check_val_every_n_epoch=1,
    )
    trainer.fit(module)
    config.check()


if __name__ == "__main__":
    train()
