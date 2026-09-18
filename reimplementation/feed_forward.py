"""OLMo-style SwiGLU feed-forward network, applied separately at each position."""

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_size, bias=False)
        self.w3 = nn.Linear(d_model, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both learned projections receive the same vector at each position.
        a = self.w1(x)
        b = self.w3(x)
        gated = F.silu(a) * b
        return self.w2(gated)


if __name__ == "__main__":
    from reimplementation.attention import MultiHeadCausalAttention
    from reimplementation.embeddings import TokenEmbedding
    from reimplementation.normalization import RMSNorm

    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    attention = MultiHeadCausalAttention(d_model=4, num_heads=2)
    attention_norm = RMSNorm(d_model=4)
    feed_forward = FeedForward(d_model=4, hidden_size=8)
    token_ids = torch.tensor([[2, 5, 2]])

    with torch.no_grad():
        x = embedding(token_ids)
        attention_update, _ = attention(x)
        h = x + attention_norm(attention_update)
        a = feed_forward.w1(h)
        b = feed_forward.w3(h)
        output = feed_forward(h)

    print("Input after the attention residual:", tuple(h.shape))
    print("w1 projection shape:", tuple(a.shape))
    print("w3 projection shape:", tuple(b.shape))
    print("Feed-forward output shape:", tuple(output.shape))
    print("Position 2 input vector:", h[0, 2])
    print("Position 2 output vector:", output[0, 2])
