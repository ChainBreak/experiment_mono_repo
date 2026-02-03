from lightning import LightningModule
import torch
import torch.nn as nn

class LitModule(LightningModule):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, y = batch["x"], batch["y"]

        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)