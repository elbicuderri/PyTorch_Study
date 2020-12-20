import torch

a = torch.randn(3, 2, 2)

b = a[0, : ,:]

print(b.size())

c = torch.ones(1, 2, 2)

print(a)

print(c)

a[0, :, :].unsqueeze_(0) += c

print(a)