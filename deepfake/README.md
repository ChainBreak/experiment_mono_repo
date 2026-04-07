# deepfake

Experiments for deepfake-related modeling (detection or synthesis) using PyTorch and Lightning.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

From this directory:

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

By default, `uv sync` installs CPU builds of PyTorch from PyPI. For NVIDIA GPUs, pick a wheel from the [PyTorch install selector](https://pytorch.org/get-started/locally/) and point uv at the extra index, for example:

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 uv sync
```

Adjust the CUDA tag (`cu124`, etc.) to match your driver and PyTorch’s published wheels.

## CLI

```bash
uv run deepfake --help
uv run deepfake hello
uv run deepfake check
```

Or:

```bash
python -m deepfake.cli --help
```

## Stack

- PyTorch
- PyTorch Lightning
- Albumentations
- Click
