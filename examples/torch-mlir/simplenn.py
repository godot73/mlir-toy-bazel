"""Simple example of torch_mlir.

Mostly copied from:
- Youtube clip: https://youtu.be/ZpwlVxsD9_U?si=_Q-xPXVdcYTITwWB
- https://github.com/llvm/torch-mlir/blob/main/projects/pt1/examples/torchscript_resnet18_all_output_types.py
"""

import torch
import torch_mlir


class SimpleNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10, bias=False)
        self.linear.weight = torch.nn.Parameter(torch.ones(10, 16))
        self.relu = torch.nn.ReLU()
        self.train(False)

    def forward(self, input):
        return self.relu(self.linear(input))


if __name__ == '__main__':
    input = torch.randn(2, 16)
    snn = SimpleNN()
    out = snn(input)

    module = torch_mlir.compile(snn, torch.ones(2, 16), output_type='torch')
    print("TORCH OutputType\n",
          module.operation.get_asm(large_elements_limit=10))
    module = torch_mlir.compile(snn,
                                torch.ones(2, 16),
                                output_type='linalg-on-tensors')
    print("LINALG_ON_TENSORS OutputType\n",
          module.operation.get_asm(large_elements_limit=10))
    module = torch_mlir.compile(snn, torch.ones(2, 16), output_type='tosa')
    print("TOSA OutputType\n",
          module.operation.get_asm(large_elements_limit=10))
