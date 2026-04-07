import click


@click.group()
@click.version_option(package_name="deepfake")
def main() -> None:
    """Deepfake experiments CLI."""


@main.command("hello")
def hello() -> None:
    """Smoke test that the environment is wired."""
    click.echo("deepfake: ok")


@main.command("check")
def check() -> None:
    """Import core deps (useful after install)."""
    import albumentations  # noqa: F401
    import lightning as L  # noqa: F401
    import torch

    click.echo(f"torch {torch.__version__}")
    click.echo(f"lightning {L.__version__}")
    click.echo(f"albumentations {albumentations.__version__}")


if __name__ == "__main__":
    main()
