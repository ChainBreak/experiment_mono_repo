import pathlib
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, data_directory: str):
        self.data_directory = pathlib.Path(data_directory)
        self._frame_paths: list[pathlib.Path] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the data directory to pick up newly recorded episodes."""
        episode_directories = sorted(
            path for path in self.data_directory.glob("episode_*") if path.is_dir()
        )
        frame_paths = []
        for episode_directory in episode_directories:
            episode_frames = sorted(episode_directory.glob("frame_*.npz"))
            frame_paths.extend(episode_frames)
        self._frame_paths = frame_paths

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = np.load(self._frame_paths[index])
        observation = torch.tensor(frame["observation"], dtype=torch.float32)
        action = torch.tensor(int(frame["action"]), dtype=torch.long)
        reward = torch.tensor(float(frame["reward"]), dtype=torch.float32)
        return observation, action, reward

    def episode_index(self, index: int) -> int:
        """Return the episode number for a given flat frame index."""
        match = re.search(r"episode_(\d+)", self._frame_paths[index].parent.name)
        return int(match.group(1))

    def frame_index(self, index: int) -> int:
        """Return the within-episode frame number for a given flat frame index."""
        match = re.search(r"frame_(\d+)", self._frame_paths[index].stem)
        return int(match.group(1))
