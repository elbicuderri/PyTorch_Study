import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantize(v, s, p, isActivation=False):
    ## v : Input Tensor
    ## s : step size, a learnable parameter specific
    ## p : Quantization bits of precision

    if isActivation == True:
        Qn = 0
        Qp = 2**(p) - 1

        num_of_activations = v.numel()
        gradScaleFactor = 1 / sqrt(num_of_activations * Qp)

    else:
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1

        num_of_weights = v.numel()
        gradScaleFactor = 1 / sqrt(num_of_weights * Qp)


    s = gradscale(s, gradScaleFactor)
    v = torch.div(v, s)
    v = v.clamp(min=Qn, max=Qp)
    vbar = roundpass(v)
    vhat = vbar * s
    return vhat

## forward
N = 1
C = 2
H = 3
W = 3
kH = 2
kW = 2
K = 3
bits = 3

x = torch.randn((N, C, H, W), dtype=torch.float32, requires_grad=True) # input_tensor

x_s = torch.randn((N, C, H, W), dtype=torch.float32, requires_grad=True)

x_s = torch.randn((N, C, H, W), dtype=torch.float32, requires_grad=True)

w = torch.randn((K, C, kH, kW), dtype=torch.float32, requires_grad=True) # weight_tensor 

w_s = torch.randn((K, C, kH, kW), dtype=torch.float32, requires_grad=True)

x_quantized = quantize(x, x_s, bits, True)

w_quantized = quantize(w, w_s, bits, False)

y = F.conv2d(x, w, padding=1)

loss_gradient = torch.randn(y.size())

print(y.size())

y.backward(gradient=loss_gradient)

print(w.grad)

y_s = F.conv2d(x_quantized, w_quantized, padding=1)

print(y_s)

quantized_loss_gradient = torch.randn(size=y_s.size())

print(quantized_loss_gradient)

y_s.backward(gradient=quantized_loss_gradient)

print(w_quantized)

print(w_quantized.dtype)

