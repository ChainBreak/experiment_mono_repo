from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

class LitModule(LightningModule):
    def __init__(self, 
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_predictions: int = 6,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.warm_up_ratio = nn.Parameter(torch.tensor(1.0))
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


        self.warm_up_ratio.data *= 0.995
        loss = multi_hypothesis_prediction.loss(y, warm_up_ratio=self.warm_up_ratio.data)
        self.log("train_loss", loss, prog_bar=True)
        self.log("warm", self.warm_up_ratio.data, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

@dataclasses.dataclass
class MultiHypothesisPrediction():
    predictions: torch.Tensor
    prob_logits: torch.Tensor

    def sample(self) -> torch.Tensor:
        probs = torch.softmax(self.prob_logits, dim=1)
        prediction_index = torch.multinomial(probs, 1).unsqueeze(2)
        return self.predictions.gather(1, prediction_index)

    def loss(self, y: torch.Tensor, warm_up_ratio: float = 0.5) -> torch.Tensor:
        y = y.unsqueeze(1)
        device = y.device
        num_predictions = self.prob_logits.shape[-1]
        loss_per_prediction = (self.predictions - y).pow(2).mean(dim=2)

        min_index = torch.argmin(loss_per_prediction, dim=1, )

        one_hot_weight = F.one_hot(
            min_index, 
            num_classes=num_predictions,
        ).to(device).float()

        avg_weight = torch.ones_like(one_hot_weight)
   
        weight = one_hot_weight * (1 - warm_up_ratio) + avg_weight * warm_up_ratio

        pred_loss = (loss_per_prediction * weight).mean()
        prob_loss = F.cross_entropy(self.prob_logits, min_index)
        return pred_loss + 0.01*prob_loss


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.shape)