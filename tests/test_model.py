import torch

from reimplementation.loss import cross_entropy_loss, make_next_token_batch
from reimplementation.model import CausalLanguageModel


def test_logits_depend_only_on_the_visible_prefix():
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    tokens = torch.tensor([[2, 5, 2], [1, 4, 3]])
    changed_tokens = tokens.clone()
    changed_tokens[:, 2] = (changed_tokens[:, 2] + 1) % 6
    with torch.no_grad():
        logits = model(tokens)
        changed_logits = model(changed_tokens)
        prefix_logits = model(tokens[:, :2])

    assert logits.shape == (2, 3, 6)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
    torch.testing.assert_close(logits[:, :2], prefix_logits)


def test_next_token_loss_backpropagates_through_the_complete_model():
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    inputs, targets = make_next_token_batch(tokens)
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
