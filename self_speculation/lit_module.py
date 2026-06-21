import pathlib

import torch
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ghostconfig import GhostConfig

import dataset as dataset_module
import model as model_module


class LitModule(L.LightningModule):
    def __init__(self, config: GhostConfig):
        super().__init__()
        self.save_hyperparameters({"config": config.to_dict()})

        self.num_bits = config["dataset"].get("num_bits", 12)
        self.num_tokens = config["dataset"].get("num_tokens", 2)
        self.learning_rate = config["training"].get("learning_rate", 1e-3)
        self.batch_size = config["training"].get("batch_size", 256)
        self.num_validation_samples = config["validation"].get("num_samples", 1000)
        self.max_refinement_steps = config["validation"].get("max_refinement_steps", 50)
        self.histogram_dir = pathlib.Path(config["validation"].get("histogram_dir", "histograms"))
        self.mixture_config = config["dataset"]["mixture"]

        self.denoiser = model_module.BitDenoiser(config["model"], self.num_bits, self.num_tokens)
        self.dataset_config = config["dataset"]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        clean = batch  # [batch_size, num_bits]
        batch_size = clean.shape[0]

        noise_level = torch.rand(batch_size, device=self.device)
        corruption_mask = torch.rand(batch_size, self.num_bits, device=self.device) < noise_level[:, None]

        random_tokens = torch.randint(0, self.num_tokens, (batch_size, self.num_bits), device=self.device)
        corrupted = torch.where(corruption_mask, random_tokens, clean)

        one_hot_corrupted = F.one_hot(corrupted, num_classes=self.num_tokens).float()
        logits = self.denoiser(one_hot_corrupted)

        loss = F.cross_entropy(logits.reshape(-1, self.num_tokens), clean.reshape(-1))
        self.log("loss_train", loss, on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        generated_integers = self._generate_samples(self.num_validation_samples)
        real_integers = dataset_module.sample_integers(
            self.mixture_config, count=self.num_validation_samples, num_bits=self.num_bits
        )

        figure = _plot_histogram(
            generated_integers.cpu().numpy(),
            real_integers.cpu().numpy(),
            num_bits=self.num_bits,
            epoch=self.current_epoch,
        )

        self.histogram_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.histogram_dir / f"epoch_{self.current_epoch:04d}.png", dpi=100)

        if self.logger:
            self.logger.experiment.add_figure("generated_vs_real", figure, global_step=self.current_epoch)

        plt.close(figure)

    def configure_optimizers(self):
        return torch.optim.Adam(self.denoiser.parameters(), lr=self.learning_rate)

    def train_dataloader(self) -> DataLoader:
        bit_dataset = dataset_module.BitSequenceDataset(self.dataset_config)
        return DataLoader(bit_dataset, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        # A single dummy batch; the real validation work happens in on_validation_epoch_end.
        dummy = TensorDataset(torch.zeros(1))
        return DataLoader(dummy, batch_size=1)

    def validation_step(self, batch, batch_idx: int) -> None:
        pass

    def _generate_samples(self, count: int) -> torch.Tensor:
        self.denoiser.eval()

        current_tokens = torch.randint(
            0, self.num_tokens, (count, self.num_bits), device=self.device
        )
        old_distribution = torch.full(
            (count, self.num_bits, self.num_tokens),
            fill_value=1.0 / self.num_tokens,
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(self.max_refinement_steps):
                one_hot_current = F.one_hot(current_tokens, num_classes=self.num_tokens).float()
                new_distribution = torch.softmax(self.denoiser(one_hot_current), dim=-1)

                old_probability = old_distribution.gather(
                    dim=-1, index=current_tokens.unsqueeze(-1)
                ).squeeze(-1)
                new_probability = new_distribution.gather(
                    dim=-1, index=current_tokens.unsqueeze(-1)
                ).squeeze(-1)

                accept_probability = (new_probability / old_probability.clamp(min=1e-8)).clamp(max=1.0)
                reject_mask = torch.rand_like(accept_probability) > accept_probability

                if not reject_mask.any():
                    break

                current_tokens = _resample_rejected_tokens(
                    current_tokens, reject_mask, new_distribution, old_probability
                )
                old_distribution = new_distribution

        self.denoiser.train()
        return dataset_module.bits_to_integer(current_tokens)


def _resample_rejected_tokens(
    current_tokens: torch.Tensor,
    reject_mask: torch.Tensor,
    new_distribution: torch.Tensor,
    old_probability: torch.Tensor,
) -> torch.Tensor:
    """For each rejected position, sample a new token from the residual distribution.

    The residual distribution removes the probability mass of the old token
    (old_probability) then renormalises, preventing re-selecting the same token.
    """
    token_one_hot = F.one_hot(current_tokens, num_classes=new_distribution.shape[-1]).float()
    residual = new_distribution - old_probability.unsqueeze(-1) * token_one_hot
    residual = residual.clamp(min=0.0)

    residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalised_residual = residual / residual_sum

    flat_residual = normalised_residual.reshape(-1, new_distribution.shape[-1])
    resampled_flat = torch.multinomial(flat_residual, num_samples=1).squeeze(-1)
    resampled_tokens = resampled_flat.reshape(current_tokens.shape)

    return torch.where(reject_mask, resampled_tokens, current_tokens)


def _plot_histogram(
    generated: "np.ndarray",
    real: "np.ndarray",
    num_bits: int,
    epoch: int,
) -> "plt.Figure":
    max_value = 2 ** num_bits - 1
    bins = min(100, max_value + 1)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(real, bins=bins, range=(0, max_value), alpha=0.5, label="Real (Gaussian mixture)", density=True)
    axis.hist(generated, bins=bins, range=(0, max_value), alpha=0.5, label="Generated (model)", density=True)
    axis.set_xlabel("Integer value")
    axis.set_ylabel("Density")
    axis.set_title(f"Generated vs. real distribution — epoch {epoch}")
    axis.legend()
    figure.tight_layout()
    return figure
