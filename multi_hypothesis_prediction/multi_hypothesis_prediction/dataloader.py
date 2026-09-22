
from collections.abc import Iterator
import torch
from ghostconfig import GhostConfig

class DataLoader():

    def __init__(self, config: GhostConfig):
        self.batch_size = config.get("batch_size", 256)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            batch = self._generate_batch()
            yield batch


    def _generate_batch(self) -> dict[str, torch.Tensor]:
        x = torch.rand(self.batch_size,1)

        f1 = 0*x + 3 -1.5
        f2 = 0*x + 2 -1.5
        f3 = 0*x + 1 -1.5
        f4 = 0*x + 0 -1.5

        b1 = torch.sigmoid((x-0.25)*30)
        b2 = torch.sigmoid((x-0.5)*30)
        b3 = torch.sigmoid((x-0.75)*30)

        f12 = (f1+f2)/2    
        f1234 = (f1+f2+f3+f4)/4
        f34 = (f3+f4)/2

        f12_ = b1*f12 + (1-b1)*f1234  
        f34_ = b1*f34 + (1-b1)*f1234    


        f1 = b2*f1 + (1-b2)*f12_
        f2 = b2*f2 + (1-b2)*f12_
        f3 = b3*f3 + (1-b3)*f34_
        f4 = b3*f4 + (1-b3)*f34_


        f = torch.cat([f1, f2, f3, f4], dim=1)
        weights = torch.tensor([60,25,10,5]).float()

        index = torch.multinomial(weights, self.batch_size, replacement=True).unsqueeze(1)
        y = torch.gather(f, dim=1, index=index)
        
        return {"x": x, "y": y}
