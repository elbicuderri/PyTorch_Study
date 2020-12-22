import torch
import torch.nn as nn

#Fully Connected Layer 
a = torch.randn(32, 3, 228, 228)

b = torch.randn(32, 228, 228, 3)

w1 = torch.randn(10, 3 * 228 * 228)

w2 = torch.randn(228 * 228 * 3, 10)

y1 = torch.einsum("nchw, kchw-> nk", a, w1.reshape(10, 3, 228, 228)) #PyTorch

y2 = torch.einsum("nhwc, hwck-> nk", b, w2.reshape(228, 228, 3, 10)) #TensorFlow

print(y1.size())

print(y2.size())