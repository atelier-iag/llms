import torch

from reimplementation.block import TransformerBlock


def test_complete_block_is_causal_and_all_parameters_receive_gradients():
    block = TransformerBlock(d_model=4, num_heads=2, hidden_size=8)
    x = torch.randn(2, 3, 4, requires_grad=True)
    original = block(x)
    changed_x = x.detach().clone()
    changed_x[:, 2] += 100
    with torch.no_grad():
        changed = block(changed_x)

    assert original.shape == x.shape
    torch.testing.assert_close(original[:, :2], changed[:, :2])
    prefix_gradient = torch.autograd.grad(
        original[:, :2].sum(), x, retain_graph=True
    )[0]
    assert torch.count_nonzero(prefix_gradient[:, 2]) == 0

    original.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_zero_updates_preserve_input_and_identity_gradient():
    block = TransformerBlock(d_model=4, num_heads=2, hidden_size=8)
    with torch.no_grad():
        block.attention.w_out.weight.zero_()
        block.feed_forward.w2.weight.zero_()

    x = torch.randn(2, 3, 4, requires_grad=True)
    output = block(x)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
