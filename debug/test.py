import torch
from pprint import pprint

xs = []
for l in range(3):
    t = torch.randn(2, 4)
    xs.append(t)

pprint(xs)
x = torch.stack(xs, dim=-1).max(dim=-1)[0]
print(x)