"""Turn each final token representation into scores over the vocabulary."""

import torch
from torch import nn

from reimplementation.normalization import RMSNorm


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        logits = self.w_out(h)
        return logits
