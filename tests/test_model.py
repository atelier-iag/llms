import pytest
import torch

from reimplementation.loss import cross_entropy_loss, make_next_token_batch
from reimplementation.model import CausalLanguageModel


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_logits_depend_only_on_the_visible_prefix(rope_theta):
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2, rope_theta=rope_theta
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


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_next_token_loss_backpropagates_through_the_complete_model(rope_theta):
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2, rope_theta=rope_theta
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


def test_rope_adds_no_parameters_and_does_not_change_initial_weights():
    config = dict(vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        baseline = CausalLanguageModel(**config)
        torch.manual_seed(7)
        rotated = CausalLanguageModel(**config, rope_theta=10000.0)
    assert baseline.state_dict().keys() == rotated.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, rotated.state_dict()[name], atol=0, rtol=0)
    for block in rotated.decoder.blocks:
        assert all(head.rope is not None for head in block.attention.heads)
