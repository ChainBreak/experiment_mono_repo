import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from ghostconfig import GhostConfig
from typing import Any
from model import CifarCnn

# CIFAR-10 channel means and stds (precomputed over the training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class LitModule(L.LightningModule):
    def __init__(self, config: GhostConfig | dict[str, Any]):
        super().__init__()
        config = GhostConfig.create(config)
        self.save_hyperparameters({"config": config.to_dict()})
        self.automatic_optimization = False

        self.model = CifarCnn(config["model"])
        self.learning_rate = config["training"].get("learning_rate", 1e-3)
        self.batch_size = config["training"].get("batch_size", 128)
        self.num_iterations = config["training"].get("num_iterations", 3)
        self.num_classes = config["model"].get("num_classes", 10)
        self.data_dir = config["dataset"].get("data_dir", "./data")

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        optimizer = self.optimizers()
        optimizer.zero_grad()

        probabilities = torch.ones(images.shape[0], self.num_classes, device=images.device) / self.num_classes

        for i in range(self.num_iterations):
            logits = self.model(images, probabilities)
            probabilities = torch.softmax(logits, dim=-1).detach()
            loss = nn.functional.cross_entropy(logits, labels)
            self.manual_backward(loss)

            accuracy = (logits.argmax(dim=1) == labels).float().mean()
            self.log(f"loss_train/step_{i}", loss, on_step=True, on_epoch=False)
            self.log(f"accuracy_train/step_{i}", accuracy, on_step=True, on_epoch=False)

        optimizer.step()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch

        probabilities = torch.ones(images.shape[0], self.num_classes, device=images.device) / self.num_classes

        for i in range(self.num_iterations):
            logits = self.model(images, probabilities)
            probabilities = torch.softmax(logits, dim=-1).detach()
            loss = nn.functional.cross_entropy(logits, labels)

            accuracy = (logits.argmax(dim=1) == labels).float().mean()
            self.log(f"loss_val/step_{i}", loss, on_step=False, on_epoch=True)
            self.log(f"accuracy_val/step_{i}", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def setup(self, stage: str) -> None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        # Use the official CIFAR-10 train split and divide 80/20 for train/val
        full_train = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=train_transform)
        full_val = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=val_transform)

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size
        split_generator = torch.Generator().manual_seed(42)

        train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size], generator=split_generator)

        self.train_dataset = torch.utils.data.Subset(full_train, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_val, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)
