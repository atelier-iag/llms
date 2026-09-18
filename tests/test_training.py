import copy

import pytest
import torch

from reimplementation.loss import cross_entropy_loss, make_next_token_batch
from reimplementation.model import CausalLanguageModel
from reimplementation.train import train_step


@pytest.fixture
def tiny_model():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield CausalLanguageModel(
                vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2
            )
    finally:
        torch.set_num_threads(previous_threads)


def test_train_step_discards_old_gradients(tiny_model):
    reference = copy.deepcopy(tiny_model)
    tiny_model.eval()
    for parameter in tiny_model.parameters():
        parameter.grad = torch.full_like(parameter, 1000.0)
    optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    tokens = torch.tensor([[2, 5, 2, 4]])

    loss = train_step(tiny_model, optimizer, tokens)
    reference_loss = train_step(reference, reference_optimizer, tokens)
    assert tiny_model.training
    assert loss == pytest.approx(reference_loss)
    for actual, expected in zip(tiny_model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_repeated_updates_learn_the_tiny_batch(tiny_model):
    tokens = torch.tensor([[2, 5, 2, 4]])
    inputs, targets = make_next_token_batch(tokens)
    optimizer = torch.optim.AdamW(
        tiny_model.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.0
    )
    with torch.no_grad():
        initial_loss = cross_entropy_loss(tiny_model(inputs), targets).item()

    for _ in range(200):
        train_step(tiny_model, optimizer, tokens)

    tiny_model.eval()
    with torch.no_grad():
        logits = tiny_model(inputs)
        final_loss = cross_entropy_loss(logits, targets).item()
    assert final_loss < initial_loss * 0.1
    torch.testing.assert_close(logits.argmax(dim=-1), targets)
