import torch
from torch import nn

from reimplementation.embeddings import TokenEmbedding


def test_lookup_and_gradients_match_pytorch_embedding():
    """Match the baseline's primitive, including accumulation for repeated IDs."""
    embedding = TokenEmbedding(vocab_size=6, d_model=4)
    reference = nn.Embedding(6, 4)
    with torch.no_grad():
        reference.weight.copy_(embedding.weight)

    token_ids = torch.tensor([[2, 5, 2], [0, 2, 5]])
    actual = embedding(token_ids)
    expected = reference(token_ids)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 4)
    torch.testing.assert_close(actual[0, 0], actual[0, 2])

    # Distinct contributions must accumulate into the shared row for token 2.
    contributions = torch.arange(actual.numel(), dtype=actual.dtype).view_as(actual)
    (actual * contributions).sum().backward()
    (expected * contributions).sum().backward()
    torch.testing.assert_close(embedding.weight.grad, reference.weight.grad)
    torch.testing.assert_close(
        embedding.weight.grad[2], contributions[token_ids == 2].sum(dim=0)
    )
    assert torch.count_nonzero(embedding.weight.grad[[1, 3, 4]]) == 0
