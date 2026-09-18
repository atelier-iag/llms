import math

import torch
import torch.nn.functional as F

from reimplementation.loss import cross_entropy_loss, make_next_token_batch


def test_each_input_position_targets_the_following_token():
    tokens = torch.tensor([[2, 5, 2, 4], [0, 1, 2, 3]])
    inputs, targets = make_next_token_batch(tokens)
    torch.testing.assert_close(inputs, torch.tensor([[2, 5, 2], [0, 1, 2]]))
    torch.testing.assert_close(targets, torch.tensor([[5, 2, 4], [1, 2, 3]]))


def test_known_target_probabilities_give_the_expected_mean_loss():
    probabilities = torch.tensor(
        [[
            [0.05, 0.05, 0.10, 0.10, 0.20, 0.50],
            [0.05, 0.05, 0.25, 0.15, 0.25, 0.25],
            [0.10, 0.10, 0.10, 0.10, 0.50, 0.10],
        ]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[5, 2, 4]])
    loss = cross_entropy_loss(probabilities.log(), targets)
    # Correct-token probabilities are 1/2, 1/4 and 1/2.
    expected = torch.tensor(4 * math.log(2) / 3, dtype=torch.float64)
    torch.testing.assert_close(loss, expected)


def test_loss_and_gradients_match_pytorch_cross_entropy():
    logits = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[5, 2, 4], [1, 2, 3]])
    actual = cross_entropy_loss(logits, targets)
    expected = F.cross_entropy(logits.reshape(-1, 6), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
