import torch
from torch import nn
from torch.autograd import Function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod')
parser.add_argument('-fuse_cpu', action='store_true')
parser.add_argument('-profiling', action='store_true')
args = parser.parse_args()

torch._C._jit_set_profiling_executor(args.profiling)
torch._C._jit_set_profiling_mode(args.profiling)
torch._C._jit_override_can_fuse_on_cpu(args.fuse_cpu)
# torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(args.fuse_cpu)
# torch._C._jit_set_nvfuser_enabled(False)

class Fused(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        mean = torch.mean(input)
        return input - mean


class FuseMe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, stride=2)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


input = torch.randn(12, 1, 16, 16, dtype=torch.float, requires_grad=True)
print(input.shape)

if args.mod == "Fused":
    mod = Fused()
elif args.mod == "FuseMe":
    mod = FuseMe()

torch.set_grad_enabled(True)
print("** scripted = torch.jit.script(mod) ******************************")
scripted = torch.jit.script(mod)

print("** out = scripted(input) ******************************")
out = scripted(input)

print("** out.backward(torch.randn_like(out)) ******************************")
out.backward(torch.randn_like(out))
