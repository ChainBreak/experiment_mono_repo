import torch
from torch.utils.data import IterableDataset
from ghostconfig import GhostConfig


class BitSequenceDataset(IterableDataset):
    def __init__(self, config: GhostConfig):
        self.num_bits = config.get("num_bits", 12)
        self.mixture_config = config["mixture"]

    def __iter__(self):
        while True:
            integer = sample_integers(self.mixture_config, count=1, num_bits=self.num_bits)[0]
            yield integer_to_bits(integer, self.num_bits)


def sample_integers(mixture_config: GhostConfig, count: int, num_bits: int) -> torch.Tensor:
    means = torch.tensor(mixture_config.get("means", [0.0]), dtype=torch.float32)
    variances = torch.tensor(mixture_config.get("variances", [1.0]), dtype=torch.float32)
    probabilities = torch.tensor(mixture_config.get("probabilities", [1.0]), dtype=torch.float32)

    component_indices = torch.multinomial(probabilities, count, replacement=True)
    selected_means = means[component_indices]
    selected_stds = variances[component_indices].sqrt()

    samples = torch.normal(selected_means, selected_stds)
    max_value = 2 ** num_bits - 1
    return samples.round().long().clamp(0, max_value)


def integer_to_bits(integer: int | torch.Tensor, num_bits: int) -> torch.Tensor:
    if isinstance(integer, torch.Tensor):
        integer = integer.item()
    integer = int(integer)
    bits = [(integer >> (num_bits - 1 - position)) & 1 for position in range(num_bits)]
    return torch.tensor(bits, dtype=torch.long)


def bits_to_integer(bits: torch.Tensor) -> torch.Tensor:
    """Convert a batch of MSB-first bit sequences back to integers.

    Args:
        bits: shape [..., num_bits], values in {0, 1}

    Returns:
        shape [...], integer values
    """
    num_bits = bits.shape[-1]
    powers = torch.tensor(
        [2 ** (num_bits - 1 - position) for position in range(num_bits)],
        dtype=torch.long,
        device=bits.device,
    )
    return (bits * powers).sum(dim=-1)
