import click


@click.command()
def main() -> None:
    """Hello world CLI entry point."""
    click.echo("Hello, world!")


if __name__ == "__main__":
    main()
