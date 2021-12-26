import torch
import model

a = torch.tensor([[1, 2, 3, 4, 5]])
b = torch.tensor([[[2, 2, 2, 2, 2]]])
c = a.max(dim=1).values
d = torch.tensor([1, 2, 3])
print(a.argmax(dim=1))
