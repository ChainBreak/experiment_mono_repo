# Multi-Hypothesis Prediction

A Python project for multi-hypothesis prediction.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Create virtual environment and install dependencies
uv sync
```

## Usage

After installation, you can use the CLI:

```bash
# Show help
uv run multi-hypothesis-prediction --help

```

## Development

```bash
# Install with dev dependencies
uv sync

# Run the CLI directly
uv run python -m multi_hypothesis_prediction.main
```

## Tricks that made it work.
 - Using soft min between hypothesis and data.
 - Decaying the soft min temperature.
 - Cosine learning rate scheduler.
 - Scaling the cross entropy loss down by 1000 to match the mse loss.

