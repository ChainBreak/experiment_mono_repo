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

# Run example command
uv run multi-hypothesis-prediction hello --name "User"

# Run prediction
uv run multi-hypothesis-prediction predict input.txt --output results.txt
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run the CLI directly
uv run python -m multi_hypothesis_prediction.main
```
