"""A Transformer block using OLMo's normalization and residual order.

Attention uses our causal MHA or GQA implementation, with optional RoPE.
"""

import torch
from torch import nn

from reimplementation.attention import MultiHeadCausalAttention
from reimplementation.feed_forward import FeedForward
from reimplementation.gqa import GroupedQueryCausalAttention
from reimplementation.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, hidden_size: int, eps: float = 1e-6,
        *, rope_theta: float | None = None, num_kv_heads: int | None = None,
    ):
        super().__init__()
        if num_kv_heads is None:
            # Keep the original modules/state-dict keys for existing checkpoints.
            self.attention = MultiHeadCausalAttention(d_model, num_heads, rope_theta=rope_theta)
        else:
            self.attention = GroupedQueryCausalAttention(
                d_model, num_heads, num_kv_heads, rope_theta=rope_theta
            )
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.feed_forward = FeedForward(d_model, hidden_size)
        self.feed_forward_norm = RMSNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_update, _ = self.attention(x)
        h = x + self.attention_norm(attention_update)
        feed_forward_update = self.feed_forward(h)
        return h + self.feed_forward_norm(feed_forward_update)


if __name__ == "__main__":
    from reimplementation.embeddings import TokenEmbedding

    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    block = TransformerBlock(d_model=4, num_heads=2, hidden_size=8)
    token_ids = torch.tensor([[2, 5, 2]])

    with torch.no_grad():
        x = embedding(token_ids)
        output = block(x)

    print("Token IDs:", token_ids)
    print("Block input shape:", tuple(x.shape))
    print("Block output shape:", tuple(output.shape))
    print("Input vectors:\n", x[0])
    print("Output vectors:\n", output[0])
