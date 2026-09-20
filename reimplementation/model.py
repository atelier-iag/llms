"""A basic causal language model, from token IDs to next-token scores.

RoPE is optional so the original baseline and its checkpoints remain usable.
The modules use PyTorch directly and do not import OLMo-core.
"""

import torch
from torch import nn

from reimplementation.decoder import TransformerDecoder
from reimplementation.embeddings import TokenEmbedding
from reimplementation.lm_head import LMHead


class CausalLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        eps: float = 1e-6,
        *,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.embeddings = TokenEmbedding(vocab_size, d_model)
        self.decoder = TransformerDecoder(
            d_model, num_heads, hidden_size, num_layers, eps=eps, rope_theta=rope_theta
        )
        self.lm_head = LMHead(d_model, vocab_size, eps=eps)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(token_ids)
        h = self.decoder(h)
        return self.lm_head(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    model.eval()
    token_ids = torch.tensor([[2, 5, 2]])

    with torch.no_grad():
        logits = model(token_ids)

    print("Input token IDs:", token_ids)
    print("Logits shape [batch, positions, vocabulary]:", tuple(logits.shape))
    print("Untrained scores at position 2, for candidate token IDs 0 to 5:")
    print(logits[0, 2])
