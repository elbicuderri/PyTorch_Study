import numpy as np
import torch
from math import sqrt

def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    yDetach = (yOut - yGrad).detach()
    y = yDetach + yGrad
    return y

def roundpass(x):
    yOut = torch.round(x)
    yGrad = x
    yDetach = (yOut - yGrad).detach()
    y = yDetach + yGrad
    return y



def quantize(v, s, p, isActivation=False):
    ## v : Input Tensor
    ## s : step size, a learnable parameter specific
    ## p : Quantization bits of precision

    if isActivation == True:
        Qn = 0
        Qp = 2**(p) - 1

        num_of_activations = 1
        for d in v.shape:
            num_of_activations *= d

        # num_of_activations = v.shape[0]*v.shape[1]*v.shape[2]*v.shape[3]
        gradScaleFactor = 1 / sqrt(num_of_activations * Qp)

    else:
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1

        num_of_weights = 1
        for d in v.shape:
            num_of_weights *= d

        # num_of_weights = v.shape[0]*v.shape[1]*v.shape[2]*v.shape[3]
        gradScaleFactor = 1 / sqrt(num_of_weights * Qp)


    s = gradscale(s, gradScaleFactor)
    v = torch.div(v, s)
    v = v.clamp(min=Qn, max=Qp)
    vbar = roundpass(v)
    vhat = vbar * s
    return vhat

## forward
x = torch.randn((1, 3, 32, 32), dtype=torch.float32, requires_grad=True) # input_tensor

w = torch.randn((6, 3, 3, 3), dtype=torch.float32, requires_grad=True) # weight_tensor 

x_quantized = quantize(x, 600, 8, True)

w_quantized = quantize(w, 600, 8, False)

print(x)

print(x_quantized)

print(w)

print(w_quantized)