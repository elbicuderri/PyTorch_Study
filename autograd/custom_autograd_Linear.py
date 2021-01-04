import torch
import torch.nn as nn
import torch.nn.functional as F

class custom_Linear(nn.Linear):
    def forward(self, _input):
        return Custom_Linear_AGfn_getAround.apply(_input, self.weight, self.bias)

class Custom_Linear_AGfn_getAround(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, _weight, _bias):
        print('Custom forward')
        with torch.enable_grad():
            detached_input = _input.detach()
            detached_input.requires_grad_(True)
            detached_weight = _weight.detach()
            detached_weight.requires_grad_(True)
            detached_bias = _bias.detach()
            detached_bias.requires_grad_(True)
            _tmp = F.linear(detached_input, detached_weight, detached_bias)
        ctx.saved_input = detached_input
        ctx.saved_param = detached_weight, detached_bias
        ctx.save_for_backward(_tmp)
        _output = _tmp.detach()
        return _output

    @staticmethod
    def backward(ctx, grad_out):
        print('Custom backward')
        _tmp, = ctx.saved_tensors
        _weight, _bias = ctx.saved_param
        detached_input = ctx.saved_input
        with torch.enable_grad():
            _tmp.backward(grad_out)
        return detached_input.grad, _weight.grad, _bias.grad