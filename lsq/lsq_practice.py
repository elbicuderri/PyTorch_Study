from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

## Learned Step Size Quantization

## forward
x = torch.randn((1, 3, 32, 32), requires_grad=True) # input_tensor

print(x)

x_detached = x.detach()

print(x_detached)

# print(x.shape)

w = torch.randn(6, 3, 3, 3) # weight_tensor 

print(w[0,0,:,:])

ww = F.hardtanh(w, min_val=0, max_val=2**(4) - 1)

print(ww[0,0,:,:])

# y = np.einsum("nchw, kcij-> nkpq")

y = torch.randn(1, 6, 30, 30)

##===============================================================
## Qn & Qp

bit = 4

Qn_weight = 2**(bit-1)     ## weight : signed
Qp_weight = 2**(bit-1) - 1

Qn_activation = 0          ## weight : unsigned
Qp_activation = 2**(bit) - 1

##===============================================================

num_of_weights = w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]  ## number of weights

num_of_activations = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3] ## number of activations

w_step_size_g = 1 / sqrt(num_of_weights*Qp_weight)

x_step_size_g = 1 / sqrt(num_of_activations&Qp_activation)

x_step_size = torch.div(x, w_step_size_g)

w_step_size = torch.div(w, x_step_size_g)

print(w_step_size)

print(w_step_size.shape)

# print(x_step_size)

# print(x_step_size.size())

x_ = x_step_size.clamp(min=(-Qn_activation), max=Qp_activation) # input clip

w_ = w_step_size.clamp(min=(-Qn_weight), max=Qp_weight) # weight clip

print(w_)

print(w_.shape)

x_round = torch.round(x_)

w_round = torch.round(w_)

x_quantized = x_round * x_step_size_g

w_quantized = w_round * w_step_size_g

## internal_flag
internal_flag = ((x_step_size > (-Qn_activation)) - (x_step_size >= Qp_activation)).float()

print(internal_flag)

## backward

# if (x_step_size > (-Qn_activation) and x_step_size < Qp_activation):
#     gradient = 





# x_clip = F.hardtanh(x_step_size, min_val=-Qn_activation, max_val=Qp_activation) # ???

# print(x_clip)





# x_round = torch.round(x_clip)

# print(x_round)

# sign = x_round.sign()

# print(sign)

# x_round_2 = torch.floor(torch.abs(x_round)/2**(bit))*(2**(bit))

# print(x_round_2)

# x_round_2 = torch.mul(x_round_2, sign)

# print(x_round_2)

# x_restore = torch.mul(x_round, scale)

# print(x_restore)


#=====================================================================================================
