
from collections.abc import Iterator
import torch

class DataLoader():

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            batch = self._generate_batch()

            yield batch


    def _generate_batch(self) -> dict[str, torch.Tensor]:
        x = torch.rand(self.batch_size,1)
        y = x**2
        return {"x": x, "y": y}
