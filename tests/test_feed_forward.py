import torch
from olmo_core.nn.feed_forward import FeedForward as OLMoFeedForward

from reimplementation.feed_forward import FeedForward


def test_outputs_and_gradients_match_olmo_feed_forward():
    feed_forward = FeedForward(d_model=4, hidden_size=8).double()
    reference = OLMoFeedForward(
        d_model=4, hidden_size=8, bias=False, dtype=torch.float64
    )
    reference.load_state_dict(feed_forward.state_dict())

    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = feed_forward(x)
    expected = reference(reference_x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in feed_forward.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)


def test_a_changed_position_cannot_change_other_positions():
    feed_forward = FeedForward(d_model=4, hidden_size=8)
    x = torch.randn(2, 3, 4)
    original = feed_forward(x)
    changed_x = x.clone()
    changed_x[0, 1] += 100
    changed = feed_forward(changed_x)

    torch.testing.assert_close(original[0, [0, 2]], changed[0, [0, 2]])
    torch.testing.assert_close(original[1], changed[1])
