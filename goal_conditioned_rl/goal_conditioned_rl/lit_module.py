import numpy as np
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
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 256)
        self.num_distance_classes = config.get("num_distance_classes", 100)
        self.num_action_classes = config.get("num_action_classes", 11)
        self.distance_prob_threshold = config.get("distance_prob_threshold", 0.05)
        self.num_dataloader_workers = config.get("num_dataloader_workers", 4)
        self.transition_dataset = transition_dataset

        hidden_size = config.get("hidden_size", 128)

        self.distance_model = nn.Sequential(
            nn.Linear(observation_dimension*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_distance_classes),
        )

        self.action_model = nn.Sequential(
            nn.Linear(observation_dimension*2 + self.num_distance_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_action_classes),
        )


    def select_action(self, current_observation: np.ndarray, goal_observation: np.ndarray) -> np.ndarray:
        current = torch.tensor(current_observation, dtype=torch.float32).unsqueeze(0)
        goal = torch.tensor(goal_observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            distance_logits = self.forward_distance_model(current, goal)
            distance_probs = F.softmax(distance_logits, dim=1)

            # Actual distances are only considered if they are above a threshold
            distances_above_threshold = distance_probs > self.distance_prob_threshold

            # Find the index of the first value above the threshold (first True in the mask)
            # This is the estimated smallest distance
            smallest_distance_estimate = torch.argmax(distances_above_threshold.int(), dim=1)

            # Predict actions that will get from current to goal in the desired number of steps
            action_logits = self.forward_action_model(current, goal, smallest_distance_estimate)
            action_probs = F.softmax(action_logits, dim=1)
            action_bin = torch.multinomial(action_probs, num_samples=1).item()

        # Convert discrete bin index back to continuous action value in [-2, 2]
        continuous_action = action_bin / (self.num_action_classes - 1) * 4.0 - 2.0
        return np.array([continuous_action], dtype=np.float32)

    def forward_distance_model(self, current_observation: torch.Tensor, goal_observation: torch.Tensor) -> torch.Tensor:
        x = torch.cat([current_observation, goal_observation], dim=1)
        x = self.distance_model(x)
        return x

    def forward_action_model(self, current_observation: torch.Tensor, goal_observation: torch.Tensor, num_of_steps: torch.Tensor) -> torch.Tensor:
        """Predict actions that will get from current to goal in the desired number of steps"""
        steps_one_hot = F.one_hot(num_of_steps, num_classes=self.num_distance_classes)
        x = torch.cat([current_observation, goal_observation, steps_one_hot], dim=1)
        x = self.action_model(x)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        current_observation = batch["current_observation"]
        goal_observation = batch["goal_observation"]
        num_steps = batch["num_steps"]
        action = batch["action"]

        # Distance model: predict how many steps separate current from goal
        distance_logits = self.forward_distance_model(current_observation, goal_observation)
        clamped_steps = num_steps.clamp(0, self.num_distance_classes - 1)
        distance_loss = F.cross_entropy(distance_logits, clamped_steps)

        # Action model: predict which action to take; discretize the continuous
        # recorded action into num_action_classes evenly-spaced bins over [-2, 2]
        action_bins = (
            ((action + 2.0) / 4.0 * (self.num_action_classes - 1))
            .round()
            .long()
            .clamp(0, self.num_action_classes - 1)
            .squeeze(-1)
        )
        action_logits = self.forward_action_model(current_observation, goal_observation, clamped_steps)
        action_loss = F.cross_entropy(action_logits, action_bins)

        loss = distance_loss + action_loss
        self.log("distance_loss_train", distance_loss, on_step=True, on_epoch=False)
        self.log("action_loss_train", action_loss, on_step=True, on_epoch=False)
        self.log("loss_train", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self) -> DataLoader:
        print(f"Training on {len(self.transition_dataset)} episodes")
        return DataLoader(
            self.transition_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
        )
