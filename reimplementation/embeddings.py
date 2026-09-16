"""The first model component: a trainable vector for each token ID."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


if __name__ == "__main__":
    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    token_ids = torch.tensor([[2, 5, 2]])
    vectors = embedding(token_ids)
    print("Token IDs:", token_ids)
    print("Embedding table shape:", tuple(embedding.weight.shape))
    print("Output shape:", tuple(vectors.shape))
    print("Output vectors:\n", vectors.detach())
    print("Repeated token, equal vectors:", torch.equal(vectors[0, 0], vectors[0, 2]))
