"""A sequence of independently learned causal Transformer blocks."""

import torch
from torch import nn

from reimplementation.block import TransformerBlock


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        eps: float = 1e-6,
        *,
        rope_theta: float | None = None,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, hidden_size, eps=eps,
                    rope_theta=rope_theta, num_kv_heads=num_kv_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return h


if __name__ == "__main__":
    from reimplementation.embeddings import TokenEmbedding

    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    decoder = TransformerDecoder(
        d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    token_ids = torch.tensor([[2, 5, 2]])

    with torch.no_grad():
        x = embedding(token_ids)
        output = decoder(x)

    print("Number of blocks:", len(decoder.blocks))
    print("Decoder input shape:", tuple(x.shape))
    print("Decoder output shape:", tuple(output.shape))
    print("Input vectors:\n", x[0])
    print("Output vectors after both blocks:\n", output[0])
