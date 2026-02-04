from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

class LitModule(LightningModule):
    def __init__(self, 
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_predictions: int = 64,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_predictions*output_dim),
            Reshape(num_predictions, output_dim),
        )

        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_predictions),
        )

    def forward(self, x: torch.Tensor) -> "MultiHypothesisPrediction":
        x = self.model(x)
        return MultiHypothesisPrediction(
            predictions=self.pred_head(x),
            prob_logits=self.prob_head(x),
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, y = batch["x"], batch["y"]

        multi_hypothesis_prediction = self(x)
        loss = multi_hypothesis_prediction.loss(y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.005)

@dataclasses.dataclass
class MultiHypothesisPrediction():
    predictions: torch.Tensor
    prob_logits: torch.Tensor

    def sample(self) -> torch.Tensor:
        probs = torch.softmax(self.prob_logits, dim=1)
        prediction_index = torch.multinomial(probs, 1).unsqueeze(2)
        return self.predictions.gather(1, prediction_index)

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        y = y.unsqueeze(1)

        loss_per_prediction = (self.predictions - y).pow(2).mean(dim=2)

        prediction_index = torch.argmin(loss_per_prediction, dim=1, keepdim=True)

        pred_loss = loss_per_prediction.gather(1, prediction_index).mean()
        prob_loss = F.cross_entropy(self.prob_logits, prediction_index.squeeze(1), reduction="none").mean()
        return pred_loss + 0.01*prob_loss


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.shape)