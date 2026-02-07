from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import numpy as np

class LitModule(LightningModule):
    def __init__(self, 
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_predictions: int = 32,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.pred_head = nn.Sequential(
            # nn.Linear(1, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_predictions*output_dim),
            Reshape(num_predictions, output_dim),
        )

        self.prob_head = nn.Sequential(
            # nn.Linear(1, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
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


        self.temperature.data *= 0.995
        loss = multi_hypothesis_prediction.loss(y, temperature=self.temperature.data, lit_module=self)
        self.log("train_loss", loss, prog_bar=True)
        self.log("temperature", self.temperature.data, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=30000,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

@dataclasses.dataclass
class MultiHypothesisPrediction():
    predictions: torch.Tensor
    prob_logits: torch.Tensor

    def sample(self) -> torch.Tensor:
        probs = torch.softmax(self.prob_logits, dim=1)
        prediction_index = torch.multinomial(probs, 1).unsqueeze(2)
        return self.predictions.gather(1, prediction_index)

    def loss(self, y: torch.Tensor, temperature: float = 1000, lit_module: LitModule | None = None) -> torch.Tensor:
        y = y.unsqueeze(1)

        loss_per_prediction = (self.predictions - y).pow(2).mean(dim=2)
        temperature = max(temperature, 1e-10)
        soft_min =torch.softmax(-loss_per_prediction/temperature, dim=1).detach()

        # probs = torch.softmax(self.prob_logits, dim=1)
        # weight = probs * soft_min
        # weight = weight / weight.sum(dim=1, keepdim=True)

        # weight = weight.detach()
        weight = soft_min
       
        pred_loss = (loss_per_prediction * weight).mean()

        # prob_loss = F.kl_div(
        #     input=F.log_softmax(self.prob_logits, dim=1), 
        #     target=soft_min.clamp(min=1e-10),
        #     log_target=False,
        #     reduction='batchmean')

        min_index = torch.argmax(weight, dim=1)
        prob_loss = F.cross_entropy(self.prob_logits, min_index)

        if lit_module is not None:
            lit_module.log("prob_loss", prob_loss, prog_bar=False)
            lit_module.log("pred_loss", pred_loss, prog_bar=False)

        return pred_loss + 0.0001*prob_loss


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.shape)