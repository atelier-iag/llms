"""Causal attention heads and their combination, with inspectable weights."""

import torch
from torch import nn


class CausalAttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.w_q = nn.Linear(d_model, head_dim, bias=False)
        self.w_k = nn.Linear(d_model, head_dim, bias=False)
        self.w_v = nn.Linear(d_model, head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, token positions, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Each query is compared with every key in the same sequence.
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5

        # A position may see itself and earlier positions, never later ones.
        length = x.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, weights


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by a positive num_heads")
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            [CausalAttentionHead(d_model, head_dim) for _ in range(num_heads)]
        )
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_outputs = []
        head_weights = []
        for head in self.heads:
            output, weights = head(x)
            head_outputs.append(output)
            head_weights.append(weights)

        # Join the vectors at each position, then learn how to mix their features.
        combined = torch.cat(head_outputs, dim=-1)
        output = self.w_out(combined)
        # Output: [batch, positions, d_model]; weights: [batch, heads, positions, positions].
        return output, torch.stack(head_weights, dim=1)


if __name__ == "__main__":
    from reimplementation.embeddings import TokenEmbedding
    from reimplementation.normalization import RMSNorm

    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    attention = CausalAttentionHead(d_model=4, head_dim=2)
    token_ids = torch.tensor([[2, 5, 2]])

    with torch.no_grad():
        x = embedding(token_ids)
        output, weights = attention(x)

    print("Embedding shape:", tuple(x.shape))
    print("Attention weights (rows = querying positions):\n", weights[0])
    print("Output shape:", tuple(output.shape))
    print("Output vectors:\n", output[0])

    multi_head_attention = MultiHeadCausalAttention(d_model=4, num_heads=2)
    with torch.no_grad():
        multi_output, multi_weights = multi_head_attention(x)
    print("Multi-head output shape:", tuple(multi_output.shape))
    print("Multi-head weights shape:", tuple(multi_weights.shape))
    print("Multi-head output vectors:\n", multi_output[0])

    # Match OLMo's order: attention -> normalization -> residual addition.
    attention_norm = RMSNorm(d_model=4)
    with torch.no_grad():
        normalized_update = attention_norm(multi_output)
        residual_output = x + normalized_update
    print("Residual output shape:", tuple(residual_output.shape))
    print("Position 2 original vector:", x[0, 2])
    print("Position 2 attention update:", multi_output[0, 2])
    print("Position 2 normalized update:", normalized_update[0, 2])
    print("Position 2 after addition:", residual_output[0, 2])
