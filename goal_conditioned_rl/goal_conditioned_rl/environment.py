import pathlib

import gymnasium
import imageio
import numpy as np

from goal_conditioned_rl import lit_module as lit_module_module


def record_episodes(
    policy: lit_module_module.PolicyLitModule,
    episode_count: int,
    environment_id: str,
    data_directory: str,
) -> None:
    data_path = pathlib.Path(data_directory)
    data_path.mkdir(parents=True, exist_ok=True)

    # Continue numbering from the highest existing episode folder
    next_episode_index = _next_episode_index(data_path)

    environment = gymnasium.make(environment_id)
    for i in range(episode_count):
        episode_path = data_path / f"episode_{next_episode_index + i:05d}"
        episode_path.mkdir()
        # Sample a random goal from the observation space for this episode
        goal_observation = environment.observation_space.sample()
        _run_episode(policy, environment, episode_path, goal_observation)

    environment.close()


def render_video(
    policy: lit_module_module.PolicyLitModule,
    environment_id: str,
    output_path: str,
) -> None:
    environment = gymnasium.make(environment_id, render_mode="rgb_array")
    observation, _ = environment.reset()
    goal_observation = environment.observation_space.sample()
    frames = [environment.render()]

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = policy.select_action(observation, goal_observation)
        observation, _, terminated, truncated, _ = environment.step(action)
        frames.append(environment.render())

    environment.close()

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_file), frames, fps=30)


def _run_episode(
    policy: lit_module_module.PolicyLitModule,
    environment: gymnasium.Env,
    episode_path: pathlib.Path,
    goal_observation: np.ndarray,
) -> None:
    observation, _ = environment.reset()
    frame_index = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = policy.select_action(observation, goal_observation)
        next_observation, reward, terminated, truncated, _ = environment.step(action)

        frame_path = episode_path / f"frame_{frame_index:05d}.npz"
        np.savez(frame_path, observation=observation, action=action, reward=reward)

        observation = next_observation
        frame_index += 1


def _next_episode_index(data_path: pathlib.Path) -> int:
    existing = [
        path for path in data_path.glob("episode_*") if path.is_dir()
    ]
    if not existing:
        return 0
    highest = max(int(path.name.split("_")[1]) for path in existing)
    return highest + 1
