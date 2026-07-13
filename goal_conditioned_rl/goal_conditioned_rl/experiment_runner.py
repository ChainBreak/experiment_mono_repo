import time

import gymnasium
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from ghostconfig import GhostConfig

from goal_conditioned_rl import dataset as dataset_module
from goal_conditioned_rl import environment as environment_module
from goal_conditioned_rl import lit_module as lit_module_module


def run_all_experiments(config: GhostConfig) -> None:
    video_directory = config.get("video_directory", "videos")
    data_directory = config.get("data_directory", "recorded_episodes")
    for experiment_config in config["experiments"]:
        ExperimentRunner(experiment_config, video_directory, data_directory).run()


class ExperimentRunner:
    def __init__(self, config: GhostConfig, video_directory: str, data_directory: str):
        self.config = config

        environment_id = config["environment"].get("id", "Pendulum-v1")
        probe_environment = gymnasium.make(environment_id)
        observation_dimension = probe_environment.observation_space.shape[0]
        action_dimension = probe_environment.action_space.shape[0]
        probe_environment.close()

        self.environment_id = environment_id

        # Each experiment writes into its own subfolder so outputs never collide
        experiment_name = config.get("name", "experiment")
        self.data_directory = f"{data_directory}/{experiment_name}"
        self.video_directory = f"{video_directory}/{experiment_name}"
        self.max_iterations = config["runner"].get("max_iterations", 10)
        self.episodes_per_iteration = config["runner"].get("episodes_per_iteration", 20)
        self.render_interval_seconds = config["runner"].get("render_interval_seconds", 60)

        self.transition_dataset = dataset_module.TransitionDataset(self.data_directory)
        self.policy = lit_module_module.PolicyLitModule(
            config["policy"],
            observation_dimension,
            action_dimension,
            self.transition_dataset,
        )

        # Wall-clock time of the last video render; start at 0 so the first iteration renders
        self._last_render_time = 0.0

    def run(self) -> None:
        experiment_name = self.config.get("name", "experiment")
        for i in range(self.max_iterations):
            print(f"[{experiment_name}] Iteration {i}")
            self.record_environment_interactions()
            self.train_model_until_plateau()
            if self.render_time_elapsed():
                self.render_environment_video(iteration=i)

    def record_environment_interactions(self) -> None:
        environment_module.record_episodes(
            self.policy,
            self.episodes_per_iteration,
            self.environment_id,
            self.data_directory,
        )
        self.transition_dataset.refresh()

    def train_model_until_plateau(self) -> None:
        # EarlyStopping on training loss acts as the plateau condition
        early_stopping = EarlyStopping(
            monitor="loss_train",
            patience=3,
            mode="min",
            check_on_train_epoch_end=True,
        )
        logger = TensorBoardLogger("lightning_logs", name=self.config.get("name", "experiment"))
        
        trainer = L.Trainer(
            max_epochs=self.config["training"].get("max_epochs", 50),
            callbacks=[early_stopping],
            logger=logger,
            enable_progress_bar=False,
        )
        trainer.fit(self.policy)

    def render_time_elapsed(self) -> bool:
        return (time.monotonic() - self._last_render_time) >= self.render_interval_seconds

    def render_environment_video(self, iteration: int) -> None:
        output_path = f"{self.video_directory}/iteration_{iteration:05d}.gif"
        environment_module.render_video(self.policy, self.environment_id, output_path)
        self._last_render_time = time.monotonic()
        print(f"  Video saved to {output_path}")
