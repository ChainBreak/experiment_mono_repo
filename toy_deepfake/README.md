# toy-deepfake

Toy deepfake experiments.

## Setup

Create a virtual environment and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

## CLI

Train the toy point autoencoder with a YAML config:

```bash
toy-deepfake train path/to/config.yaml
```

A default template ships with the package at `toy_deepfake/config.yaml` (next to the Python package in this repo). Copy it or point `train` at that file after cloning.

Other commands:

```bash
toy-deepfake hello
```

Or:

```bash
python -m toy_deepfake.main train toy_deepfake/config.yaml
```
