"""Packed head projections and PyTorch's optimized causal attention."""

import torch
import torch.nn.functional as F


def validate_attention_backend(backend: str) -> None:
    if backend not in ("manual", "sdpa"):
        raise ValueError("attention_backend must be 'manual' or 'sdpa'")


def project_heads(x, projections):
    # Concatenation preserves the original Parameter objects/state-dict keys.
    # Autograd distributes the packed projection gradient to every head.
    weight = torch.cat([projection.weight for projection in projections], dim=0)
    projected = F.linear(x, weight)
    return projected.reshape(*x.shape[:-1], len(projections), -1).transpose(-3, -2)


def causal_sdpa(q, k, v, rope=None):
    if rope is not None:
        positions = torch.arange(q.shape[-2], device=q.device)
        q, k = rope(q, positions), rope(k, positions)
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True,
        enable_gqa=q.shape[-3] != k.shape[-3],
    )
