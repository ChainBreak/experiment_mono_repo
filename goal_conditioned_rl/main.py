import click
from ghostconfig import GhostConfig

from goal_conditioned_rl import experiment_runner as experiment_runner_module


@click.command()
@click.option("--config-path", default="config.yaml", help="Path to the config yaml file.")
def train(config_path: str) -> None:
    config = GhostConfig.create(config_path)
    experiment_runner_module.run_all_experiments(config)
    config.check()


if __name__ == "__main__":
    train()
