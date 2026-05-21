import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from ghostconfig import GhostConfig

from lit_module import LitModule


def train(config_path: str = "config.yaml") -> None:
    config = GhostConfig.create(config_path)

    lit_module = LitModule(config)
    trainer = L.Trainer(
        max_epochs=config["training"].get("max_epochs", 20),
        logger=TensorBoardLogger("lightning_logs"),
    )
    trainer.fit(lit_module)
    config.check()


if __name__ == "__main__":
    train()
