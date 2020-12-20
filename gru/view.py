import torch

# aa = torch.randn(3,4)

# print(aa.size())

# print(aa.size(0))

# print(aa.size(-1))

# aa = aa.view(aa.size(0), 4)

# print(aa.size())

# print(aa.size(0))

# print(aa.size(-1))

# bb = torch.randn(1, 1, 3)

# print(bb.size(), bb)

# bb = bb.squeeze()

# print(bb.size(), bb)


cc = torch.rand(2, 2, 2)

print(cc)

cc[:, 1, :] += 3

print(cc)

