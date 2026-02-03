
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

        f1 = x
        f2 = x**2

        f = torch.cat([f1, f2], dim=1)

        index = torch.randint(0, f.shape[1], (self.batch_size,1))
        y = torch.gather(f, dim=1, index=index)
        # y = y.repeat(1,5)
        return {"x": x, "y": y}
