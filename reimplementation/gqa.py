"""Causal grouped-query attention: independent Q, one shared K/V per group."""

import torch
from torch import nn

from reimplementation.rope import RotaryPositionEmbedding


class GroupedQueryCausalAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, num_kv_heads: int,
        *, rope_theta: float | None = None,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by a positive num_heads")
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("num_kv_heads must be positive and divide num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.rope = (
            RotaryPositionEmbedding(self.head_dim, rope_theta)
            if rope_theta is not None else None
        )
        self.w_q = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.w_k = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_kv_heads)]
        )
        self.w_v = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_kv_heads)]
        )
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, token positions, d_model]. Compute shared K/V only once.
        keys = [projection(x) for projection in self.w_k]
        values = [projection(x) for projection in self.w_v]
        length = x.shape[1]
        positions = torch.arange(length, device=x.device)
        if self.rope is not None:
            keys = [self.rope(k, positions) for k in keys]
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()

        head_outputs = []
        head_weights = []
        for head_id, projection in enumerate(self.w_q):
            q = projection(x)
            if self.rope is not None:
                q = self.rope(q, positions)
            group_id = head_id // self.heads_per_group
            k = keys[group_id]
            v = values[group_id]
            scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
            weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            head_outputs.append(weights @ v)
            head_weights.append(weights)

        # Shared K/V accumulate gradients from all query heads in their group.
        output = self.w_out(torch.cat(head_outputs, dim=-1))
        # Output: [batch, positions, d_model]; weights: [batch, heads, positions, positions].
        return output, torch.stack(head_weights, dim=1)
