import torch
from torch import nn
from torch.autograd import Function

class DummyFunction(Function):
    @staticmethod
    def forward(ctx, input, constant):
        ctx.save_for_backward(input, constant)
        output = input * constant
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print("DummyFunction.backward() invoked")
        input, constant = ctx.saved_tensors
        grad_input = grad_constant = None

        if ctx.needs_input_grad[0]:
            grad_input = constant.expand_as(input)
        if ctx.needs_input_grad[1]:
            grad_constant = input.sum()

        return grad_input, grad_constant

class mean_mod(nn.Module):
    def __init__(self):
        super(mean_mod, self).__init__()

    def forward(self, input):
        mean = torch.mean(input)
        return input - mean

input = torch.tensor([1., 2, 3, 4, 50], requires_grad=True)
dummy = DummyFunction()

print("create mean_mod")
mod = mean_mod()
print("trace mean_mod")
from torch.jit import fuser
with fuser('fuser1'):
    traced = torch.jit.script(mod)
print("run traced mean_mod")
o = traced(input)

# print("run dummy")
# c = torch.tensor([5.,])
# o = dummy.apply(o, c)

print("run o.backward()")
o.backward(torch.tensor([1., 0, 0, 0, 0]))
