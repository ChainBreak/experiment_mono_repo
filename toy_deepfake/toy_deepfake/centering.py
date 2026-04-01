from __future__ import annotations

import torch
from torch import nn


class Centering(nn.Module):
    """Transforms encoder output using cluster identity; placeholder returns the input unchanged."""
    def __init__(self, num_identities: int, shape: tuple[int, ...], ema_decay: float = 0.99) -> None:
        super().__init__()
        self.shape = shape
        self.num_identities = num_identities
        self.ema_decay = ema_decay
        self.identity_centers = nn.Parameter(torch.zeros(num_identities, *shape),requires_grad=False)
        self.identity_offsets = nn.Parameter(torch.zeros(num_identities, *shape),requires_grad=True)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if self.training:
            self.update_identity_centers(x, identity)
        
        x_target = x - self.identity_centers[identity]

        loss_center = nn.functional.mse_loss(x,x_target.detach())

        x_offset = x + self.identity_offsets[identity]

        return x_offset, loss_center

        

    def update_identity_centers(self, x: torch.Tensor, identity: torch.Tensor) -> None:
        with torch.no_grad():
            for i in range(x.shape[0]):
                ident = int(identity[i].item())
                self.identity_centers[ident] = self.identity_centers[ident] * self.ema_decay + x[i] * (1 - self.ema_decay)

   