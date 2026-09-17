from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import numpy as np
from ghostconfig import GhostConfig

class LitModule(LightningModule):
    def __init__(self, config: GhostConfig):
        super().__init__()

        self.save_hyperparameters({"config": config.to_dict()})

        
        hidden_dim = config.get("hidden_dim", 32)
        output_dim = config.get("output_dim", 1)
        num_hypotheses = config.get("num_hypotheses", 32)
        self.learning_rate = config.get("learning_rate", 0.001)

        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )

        self.multi_hypothesis_prediction = MultiHypothesisPrediction(
            config=config,
            classifier=HypothesesClassifier(config=config["classifier"], input_dim=hidden_dim, output_dim=output_dim, num_hypotheses=num_hypotheses),
            predictor=HypothesesPredictor(config=config["predictor"], input_dim=hidden_dim, num_hypotheses=num_hypotheses),
            generator=OutputGenerator(config=config["generator"], input_dim=hidden_dim, num_hypotheses=num_hypotheses, output_dim=output_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(100.0))

        # Example input for model summary
        self.example_input_array = torch.randn(1, 1)

        config.check()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.multi_hypothesis_prediction.sample(x)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, y = batch["x"], batch["y"]

        x = self.model(x)
        predictor_loss, reconstruction_loss = self.multi_hypothesis_prediction.loss(x, y, float(self.temperature.data), self.log)
        loss = 0.01*predictor_loss +reconstruction_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_predictor_loss", predictor_loss, prog_bar=True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("temperature", self.temperature.data, prog_bar=False)

        self.temperature.data *= 0.999
        self.temperature.data = torch.clamp(self.temperature.data, min=1.0)
    
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10000,
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

class MultiHypothesisPrediction(nn.Module):
    
    def __init__(self,
            config: GhostConfig,
            classifier: nn.Module,
            predictor: nn.Module,
            generator: nn.Module,
        ):
        super().__init__()
        self.classifier = classifier
        self.predictor = predictor
        self.generator = generator

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        hypotheses_logits = self.predictor(x)
        hypotheses_one_hot = F.gumbel_softmax(hypotheses_logits, tau=1.0, hard=True, dim=1)
        output = self.generator(x, hypotheses_one_hot)
        return output


    def loss(self, x: torch.Tensor, y: torch.Tensor, temperature: float, log: callable) -> tuple[torch.Tensor, torch.Tensor]:        

        classifier_logits = self.classifier(x.detach(), y)
        classifier_probs = torch.softmax(classifier_logits, dim=1)

        # Hard in the forward pass, soft Gumbel probabilities in the backward pass
        pass_through_one_hot = F.gumbel_softmax(classifier_logits, tau=1.0, hard=True, dim=1)
        classifier_sampled_index = pass_through_one_hot.argmax(dim=1)
        # classifier_target_index = torch.argmax(classifier_probs, dim=1).detach()

        log("max_classifier_probs", torch.max(classifier_probs, dim=1).values.mean())
        log("num_unique_indexes", torch.unique(classifier_sampled_index).shape[0])

        output_pred = self.generator(x, pass_through_one_hot)
        reconstruction_loss = F.mse_loss(output_pred, y)

        predictor_logits = self.predictor(x)
        predictor_loss = F.cross_entropy(predictor_logits, classifier_sampled_index)
        
        return predictor_loss, reconstruction_loss


class HypothesesClassifier(nn.Module):
    def __init__(self, config: GhostConfig, input_dim: int, output_dim: int, num_hypotheses: int):
        super().__init__()
        hidden_dims = config.get("hidden_dims", [32])
        self.classifier = FullyConnected([input_dim + output_dim] + hidden_dims)
        self.output_linear = nn.Linear(hidden_dims[-1], num_hypotheses)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.classifier(torch.cat([x, y], dim=1))
        x = self.output_linear(x)
        return x


class HypothesesPredictor(nn.Module):
    def __init__(self, config: GhostConfig, input_dim: int, num_hypotheses: int):
        super().__init__()
        hidden_dims = config.get("hidden_dims", [32])
        self.predictor = FullyConnected([input_dim] + hidden_dims)
        self.output_linear = nn.Linear(hidden_dims[-1], num_hypotheses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predictor(x)
        x = self.output_linear(x)
        return x

class OutputGenerator(nn.Module):
    def __init__(self, config: GhostConfig, input_dim: int, num_hypotheses: int, output_dim: int):
        super().__init__()

        self.input_model = nn.Sequential(
            FullyConnected([input_dim] + config.get("input_hidden_dims", [32])),
            nn.Linear(config.get("input_hidden_dims", [32])[-1], output_dim),
            )
        self.on_hot_model = nn.Sequential(
            FullyConnected([num_hypotheses] + config.get("on_hot_hidden_dims", [32])),
            nn.Linear(config.get("on_hot_hidden_dims", [32])[-1], output_dim),
            )


    def forward(self, x: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:

        x = self.input_model(x) + self.on_hot_model(one_hot)

        return x


class FullyConnected(nn.Module):
    def __init__(self, dim_list: list[int]):
        super().__init__()

        input_list = dim_list[:-1]
        output_list = dim_list[1:]

        self.layers = nn.Sequential()
        for input_dim, output_dim in zip(input_list, output_list):
            self.layers.append( nn.Linear(input_dim, output_dim))
            self.layers.append( nn.SiLU())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)