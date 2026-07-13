import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from ghostconfig import GhostConfig

from goal_conditioned_rl import dataset as dataset_module


class PolicyLitModule(L.LightningModule):
    def __init__(
        self,
        config: GhostConfig,
        observation_dimension: int,
        action_dimension: int,
        transition_dataset: dataset_module.TransitionDataset,
    ):
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})

        self.action_dimension = action_dimension
        self.learning_rate = config["training"].get("learning_rate", 1e-3)
        self.batch_size = config["training"].get("batch_size", 256)
        self.transition_dataset = transition_dataset

        hidden_size = config["policy"].get("hidden_size", 128)
        self.network = nn.Sequential(
            nn.Linear(observation_dimension, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dimension),
        )

    def select_action(self, observation: torch.Tensor) -> int:
        # TODO: replace with policy.network(observation).argmax() once an RL objective is implemented
        return torch.randint(0, self.action_dimension, (1,)).item()

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        observations, actions, _ = batch
        # Placeholder: supervised cross-entropy toward the recorded (random) action.
        # This is not a real RL objective — replace with policy gradient or Q-learning loss.
        logits = self.network(observations)
        loss = F.cross_entropy(logits, actions)
        self.log("loss_train", loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.transition_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
