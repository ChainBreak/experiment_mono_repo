import torch
import torch.nn as nn
from ghostconfig import GhostConfig


class BitDenoiser(nn.Module):
    def __init__(self, config: GhostConfig, num_bits: int, num_tokens: int):
        super().__init__()
        self.num_bits = num_bits
        self.num_tokens = num_tokens

        hidden_size = config.get("hidden_size", 256)
        num_layers = config.get("num_layers", 3)

        input_size = num_bits * num_tokens
        output_size = num_bits * num_tokens

        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, one_hot_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            one_hot_sequence: [batch, num_bits, num_tokens]

        Returns:
            logits: [batch, num_bits, num_tokens]
        """
        batch_size = one_hot_sequence.shape[0]
        flat = one_hot_sequence.reshape(batch_size, -1).float()
        output = self.network(flat)
        return output.reshape(batch_size, self.num_bits, self.num_tokens)
