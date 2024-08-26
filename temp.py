import torch
from kornia.geometry.boxes import Boxes


# either B, N, 4, 2 or N, 4, 2
a = torch.tensor([2.0, 0, 0.0, 0, 2.0, 2.0, 0, 2.0], dtype=torch.float32)
b = a.reshape(1, 1, 4, 2)
b = b.repeat(10, 2, 1, 1)
boxes = Boxes(b)
# succeeds
print(boxes.compute_area())


a = a.reshape(1, 4, 2)
boxes = Boxes(a)
# succeeds
print(boxes.compute_area().shape)
