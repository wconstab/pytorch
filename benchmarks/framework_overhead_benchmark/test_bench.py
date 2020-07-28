import pytest
import torch

class SimpleAddModule(torch.nn.Module):
    def __init__(self, num_iter):
        super().__init__()
        self.num_iter = num_iter

    def forward(self, x, y):
        for _ in range(self.num_iter):
            x = torch.add(x, y)
        return x

@pytest.mark.parametrize('num_iter', [1, 10, 100, 1000], ids=['single', 'loop-10', 'loop-100', 'loop-1k'])
@pytest.mark.parametrize('jit', [True, False], ids=['jit', 'no-jit'])
def test_add(benchmark, jit, num_iter):
    mod = SimpleAddModule(num_iter)
    if jit:
        mod = torch.jit.script(mod)
    inputs = [torch.randn(1) for _ in range(2)]
    benchmark(mod, *inputs)
