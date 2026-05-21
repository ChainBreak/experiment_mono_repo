import pathlib

import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from ghostconfig import GhostConfig

from lit_module import LitModule


def train(config_path: str = "config.yaml") -> None:
    config = GhostConfig.create(config_path)

    lit_module = LitModule(config)
    trainer = L.Trainer(
        max_epochs=config["training"].get("max_epochs", 20),
        # Skip the sanity validation check so history lists stay aligned by epoch
        num_sanity_val_steps=0,
        logger=TensorBoardLogger("lightning_logs"),
    )
    trainer.fit(lit_module)
    config.check()

    plot_metrics(lit_module)


def plot_metrics(lit_module: LitModule) -> None:
    pathlib.Path("outputs").mkdir(exist_ok=True)

    epochs = list(range(1, len(lit_module.train_loss_history) + 1))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle("CIFAR-10 Training Metrics", fontsize=14)

    axes[0, 0].plot(epochs, lit_module.train_loss_history)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, lit_module.val_loss_history)
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(epochs, lit_module.train_accuracy_history)
    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 1].plot(epochs, lit_module.val_accuracy_history)
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    output_path = "outputs/metrics.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved metrics plot to {output_path}")


if __name__ == "__main__":
    train()
