# Goal-Conditioned RL: Acrobot Offline RL Scaffold

An iterated offline RL loop applied to the Acrobot-v1 gymnasium environment.

## How it works

Each iteration of the loop:
1. **Record** — the current policy interacts with the environment. Each episode is written to its own numbered folder on disk (`recorded_episodes/episode_00000/`, `episode_00001/`, …). Each step within an episode is saved as `frame_00000.npz`, `frame_00001.npz`, … containing the observation, action, and reward. Because both indices are monotonically increasing across all runs, you can load any two frames from disk and immediately know their episode and how many steps apart they are.
2. **Train** — the policy is trained on all recorded frames (using Lightning with early stopping as the plateau condition).
3. **Render** — periodically, a video of one full episode is written to the `videos/` directory.

> **Note:** The policy currently samples random actions — the RL algorithm is left as a TODO. The training objective is a placeholder to keep the loop runnable end-to-end.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py --config-path config.yaml
```

## Config

All hyperparameters live in `config.yaml`. The top-level `experiments` list allows multiple experiment configs to be run sequentially via `run_all_experiments`.
