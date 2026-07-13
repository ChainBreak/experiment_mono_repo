import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# Each __getitem__ call samples one episode then picks two frames within it: a
# "current" frame and a later "goal" frame. The frame filenames encode their
# position within the episode, so num_steps = goal_frame_number -
# current_frame_number is exact. This makes the dataset suitable for
# goal-conditioned learning: given (current_observation, goal_observation,
# num_steps), the recorded action is what the policy did at the current frame
# on its way toward that future state.
class TransitionDataset(Dataset):
    def __init__(self, data_directory: str):
        self.data_directory = pathlib.Path(data_directory)
        # Each element is a sorted list of frame paths for one episode
        self._episodes: list[list[pathlib.Path]] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the data directory to pick up newly recorded episodes."""
        episode_directories = sorted(
            path for path in self.data_directory.glob("episode_*") if path.is_dir()
        )
        episodes = []
        for episode_directory in episode_directories:
            frames = sorted(episode_directory.glob("frame_*.npz"))
            # Need at least two frames to form a (current, goal) pair
            if len(frames) >= 2:
                episodes.append(frames)
        self._episodes = episodes

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> dict:
        episode_frames = self._episodes[index]

        current_idx = random.randint(0, len(episode_frames) - 2)
        goal_idx = random.randint(current_idx + 1, len(episode_frames) - 1)

        current_frame = np.load(episode_frames[current_idx])
        goal_frame = np.load(episode_frames[goal_idx])

        # Frame numbers are encoded in the filename, so subtraction gives exact step distance
        current_frame_number = _frame_number(episode_frames[current_idx])
        goal_frame_number = _frame_number(episode_frames[goal_idx])
        num_steps = goal_frame_number - current_frame_number

        return {
            "current_observation": torch.tensor(current_frame["observation"], dtype=torch.float32),
            "goal_observation": torch.tensor(goal_frame["observation"], dtype=torch.float32),
            "num_steps": torch.tensor(num_steps, dtype=torch.long),
            "action": torch.tensor(current_frame["action"], dtype=torch.float32),
            "reward": torch.tensor(float(current_frame["reward"]), dtype=torch.float32),
        }


def _frame_number(frame_path: pathlib.Path) -> int:
    return int(frame_path.stem.split("_")[1])
