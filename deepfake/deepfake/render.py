"""Batch render: encode input images and decode under selected identity embeddings."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

import deepfake.lit_module as lit_module_module
from deepfake.dataset import _IMAGE_SUFFIXES


def _list_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(path)
    return sorted(paths)


def _parse_identities(identities: str) -> list[int]:
    parts = [p.strip() for p in identities.split(",") if p.strip()]
    if not parts:
        raise click.BadParameter("at least one identity index is required", param_hint="identities")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise click.BadParameter(f"not an integer: {p!r}", param_hint="identities") from e
    return out


def _assert_checkpoint_has_config(checkpoint: Path) -> None:
    load_kw: dict = {"map_location": "cpu"}
    try:
        ckpt = torch.load(checkpoint, **load_kw, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, **load_kw)
    hp = ckpt.get("hyper_parameters")
    ok = False
    if hp is not None:
        if isinstance(hp, dict):
            ok = "config" in hp
        else:
            ok = hasattr(hp, "config")
    if not ok:
        raise click.UsageError(
            f"Checkpoint {checkpoint} has no hyper_parameters.config. "
            "Train or save a checkpoint after LitModule saves hyperparameters."
        )


@click.command("render")
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Lightning checkpoint path.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory of input images (searched recursively).",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for output PNGs.",
)
@click.option(
    "--identities",
    type=str,
    required=True,
    help="Comma-separated identity indices, e.g. 0,1,2.",
)
def render_cmd(
    checkpoint: Path,
    input_dir: Path,
    output_dir: Path,
    identities: str,
) -> None:
    """Encode each image, decode once per identity, concat input + decodes horizontally."""
    identity_ids = _parse_identities(identities)
    _assert_checkpoint_has_config(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lit_module_module.LitModule.load_from_checkpoint(
        checkpoint,
        map_location=device,
    )
    model.eval()
    model.to(device)

    height = int(model.config.dataset.height)
    width = int(model.config.dataset.width)
    num_emb = model.identity_embedding.num_embeddings
    for i in identity_ids:
        if i < 0 or i >= num_emb:
            raise click.UsageError(
                f"Identity index {i} out of range [0, {num_emb - 1}]."
            )

    image_paths = _list_images(input_dir.expanduser().resolve())
    if not image_paths:
        raise click.UsageError(f"No images found under {input_dir}.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = T.Compose(
        [
            T.Resize((height, width)),
            T.ToTensor(),
        ]
    )

    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            batch = to_tensor(rgb).unsqueeze(0).to(device)
            latent = model.encoder(batch)
            panels: list[torch.Tensor] = [batch]
            for idx in identity_ids:
                id_t = torch.tensor([idx], device=device, dtype=torch.long)
                vec = model.identity_embedding(id_t)
                panels.append(model.decoder(latent, vec))
            strip = torch.cat(panels, dim=-1)
            strip = strip.clamp(0.0, 1.0)
            out_path = output_dir / f"{path.stem}.png"
            save_image(strip, out_path, nrow=1)

    click.echo(f"Wrote {len(image_paths)} image(s) to {output_dir}")
