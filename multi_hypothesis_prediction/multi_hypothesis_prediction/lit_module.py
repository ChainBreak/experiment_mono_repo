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
        config.check()

        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim)
        )

        self.multi_hypothesis_prediction = MultiHypothesisPrediction(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_hypotheses=num_hypotheses,
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Example input for model summary
        self.example_input_array = torch.randn(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.multi_hypothesis_prediction.sample(x)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, y = batch["x"], batch["y"]

        x = self.model(x)
        predictor_loss, reconstruction_loss = self.multi_hypothesis_prediction.loss(x, y,self.temperature.data,self.log)
        loss = 0.0001*predictor_loss +reconstruction_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_predictor_loss", predictor_loss, prog_bar=True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("temperature", self.temperature.data, prog_bar=False)

        self.temperature.data *= 0.999
    
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
        input_dim: int,
        output_dim: int,
        num_hypotheses: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.hypotheses_classifier = FullyConnected([input_dim + output_dim, hidden_dim, num_hypotheses])
        self.hypotheses_predictor = FullyConnected([input_dim, hidden_dim, num_hypotheses])
        self.output_generator = FullyConnected([input_dim + num_hypotheses, hidden_dim,hidden_dim, output_dim])
        self.num_hypotheses = num_hypotheses

    def sample(self, x: torch.Tensor) -> torch.Tensor:

        # Get a probability distribution over hypotheses
        hypotheses_logits = self.hypotheses_predictor(x)
        hypotheses_probs = torch.softmax(hypotheses_logits, dim=1)

        # Sample a single hypothesis
        hypotheses_index = torch.multinomial(hypotheses_probs, 1).squeeze(1)

        # Convert the hypothesis index to a one-hot vector
        hypotheses_one_hot = F.one_hot(hypotheses_index, self.num_hypotheses)

        # Concatenate the hypothesis one-hot vector to the input
        hypothesis_conditioned_x = torch.cat([x, hypotheses_one_hot], dim=1)

        # Generate the output conditioned on the chosen hypothesis
        output = self.output_generator(hypothesis_conditioned_x)
        return output


    def loss(self, x: torch.Tensor, y: torch.Tensor, temperature: torch.Tensor,log: callable) -> tuple[torch.Tensor, torch.Tensor]:        

        classifier_logits = self.hypotheses_classifier(torch.cat([x, y], dim=1))
        classifier_probs = torch.softmax(classifier_logits, dim=1)
        classifier_sampled_index = torch.multinomial(classifier_probs, 1).squeeze(1).detach()
        classifier_target_index = torch.argmax(classifier_probs, dim=1).detach()
        classifier_one_hot = F.one_hot(classifier_sampled_index, self.num_hypotheses).detach()

        pass_through_one_hot =  classifier_probs - classifier_probs.detach() + classifier_one_hot


        log("max_classifier_probs", torch.max(classifier_probs))
        log("num_unique_indexes", torch.unique(classifier_sampled_index).shape[0])

        output_pred = self.output_generator(torch.cat([x, pass_through_one_hot], dim=1))
        reconstruction_loss = F.mse_loss(output_pred, y)

        predictor_logits = self.hypotheses_predictor(x)
        predictor_loss = F.cross_entropy(predictor_logits, classifier_sampled_index)
        
        return predictor_loss, reconstruction_loss



class FullyConnected(nn.Module):
    def __init__(self, dim_list: list[int]):
        super().__init__()

        input_list = dim_list[:-1]
        output_list = dim_list[1:]

        self.layers = nn.Sequential()
        for input_dim, output_dim in zip(input_list, output_list):
            self.layers.append( nn.SiLU())
            self.layers.append( nn.Linear(input_dim, output_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)