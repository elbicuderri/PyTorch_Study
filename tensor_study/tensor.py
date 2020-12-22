import torch
import torch.nn as nn

a = torch.randn(1,2,3,4)

print(a.size())

b = a.unsqueeze(0)

print(b.size())

c = a.squeeze()

print(c.size())

d = torch.rand(3, 1, 2, 1, 4)

d = d.unsqueeze(0)

print(d.size())

t = torch.rand(16, 3, 32, 32)

t = t.view(16, -1)

print(t.size())

x = torch.arange(10)

print(x.clamp(min=3))

print(x[(x < 2) & (x > 8)])

print(x[x.remainder(2) == 0])

batch = 32

i = torch.rand(batch, 3, 228, 228)

print(i.view(batch, -1).size())

print(i.permute(0, 2, 3, 1).size())

x = torch.arange(10) # [10]

print(x.unsqueeze(0).unsqueeze(-1).size())

print(x.unsqueeze(0).unsqueeze(-1).squeeze(2).size())