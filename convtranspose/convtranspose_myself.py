import torch
import torch.nn as nn

cifar = torch.randn((1, 3, 32, 32))

"""
be careful when stride > 1
"""
out = nn.ConvTranspose2d(in_channels=3,
                         out_channels=6,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         )(cifar) ## torch.Size([1, 6, 63, 63])

print(out.size())

out2 = nn.ConvTranspose2d(in_channels=3,
                         out_channels=6,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         output_padding=1,
                         )(cifar) ## torch.Size([1, 6, 64, 64])

print(out2.size())


