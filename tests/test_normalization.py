import torch

from reimplementation.normalization import RMSNorm


def test_outputs_and_gradients_match_pytorch_rmsnorm():
    normalization = RMSNorm(d_model=4, eps=1e-6).double()
    reference = torch.nn.RMSNorm(4, eps=1e-6).double()
    with torch.no_grad():
        normalization.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        reference.weight.copy_(normalization.weight)

    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = normalization(x)
    expected = reference(reference_x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(normalization.weight.grad, reference.weight.grad)


def test_zero_vectors_have_finite_outputs_and_gradients():
    normalization = RMSNorm(d_model=4)
    x = torch.zeros(2, 3, 4, requires_grad=True)
    output = normalization(x)
    torch.testing.assert_close(output, torch.zeros_like(x))
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(normalization.weight.grad).all()
