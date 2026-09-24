"""RMSNorm: normalize each token vector, then apply learned coordinate scales."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        mean_square = x.square().mean(dim=-1, keepdim=True)
        normalized = x / torch.sqrt(mean_square + self.eps)
        return normalized * self.weight


if __name__ == "__main__":
    normalization = RMSNorm(d_model=4)
    x = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]])
    with torch.no_grad():
        output = normalization(x)
    print("Input vector:", x[0, 0])
    print("Normalized vector:", output[0, 0])
    print("Shape is preserved:", tuple(output.shape))
