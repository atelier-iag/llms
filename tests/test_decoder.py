import torch

from reimplementation.decoder import TransformerDecoder


def test_blocks_have_independent_parameters_and_all_receive_gradients():
    decoder = TransformerDecoder(
        d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    first_parameters = {p.data_ptr() for p in decoder.blocks[0].parameters()}
    second_parameters = {p.data_ptr() for p in decoder.blocks[1].parameters()}
    assert first_parameters.isdisjoint(second_parameters)

    x = torch.randn(2, 3, 4, requires_grad=True)
    output = decoder(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for block in decoder.blocks:
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_future_tokens_remain_hidden_through_the_stack():
    decoder = TransformerDecoder(
        d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    x = torch.randn(2, 4, 4, requires_grad=True)
    original = decoder(x)
    changed_x = x.detach().clone()
    changed_x[:, 2:] += 100
    with torch.no_grad():
        changed = decoder(changed_x)
    torch.testing.assert_close(original[:, :2], changed[:, :2])

    prefix_gradient = torch.autograd.grad(original[:, :2].sum(), x)[0]
    assert torch.count_nonzero(prefix_gradient[:, 2:]) == 0
