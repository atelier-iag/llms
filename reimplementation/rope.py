"""Rotary positions: rotate adjacent feature pairs without learned parameters."""

import math

import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE requires a positive, even head_dim")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [..., token positions, head_dim]; positions: [token positions].
        if x.ndim < 2 or x.shape[-1] != self.head_dim or not x.is_floating_point():
            raise ValueError("RoPE expects floating-point vectors ending in head_dim")
        if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
            raise ValueError("RoPE needs one position per token")

        # Pair 0 turns by 1 radian per position; later pairs turn more slowly.
        # Use at least float32 for angles, even if vectors use a lower precision.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        pair_indices = torch.arange(0, self.head_dim, 2, device=x.device, dtype=dtype)
        frequencies = self.theta ** (-pair_indices / self.head_dim)
        angles = positions.to(device=x.device, dtype=dtype)[:, None] * frequencies
        cos, sin = angles.cos(), angles.sin()

        # Rotate each pair [a, b] to [a*cos - b*sin, a*sin + b*cos].
        a, b = x[..., 0::2], x[..., 1::2]
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        return torch.stack((rotated_a, rotated_b), dim=-1).flatten(-2).to(x.dtype)
