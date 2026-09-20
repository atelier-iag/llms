import pytest
import torch
import torch.nn.functional as F

from reimplementation.attention import CausalAttentionHead, MultiHeadCausalAttention


def test_output_and_gradients_match_pytorch_causal_attention():
    attention = CausalAttentionHead(d_model=4, head_dim=2).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    actual, _ = attention(x)
    expected = F.scaled_dot_product_attention(
        attention.w_q(x).unsqueeze(1),
        attention.w_k(x).unsqueeze(1),
        attention.w_v(x).unsqueeze(1),
        is_causal=True,
    ).squeeze(1)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 2)
    parameters = (x, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_future_tokens_cannot_affect_earlier_outputs(rope_theta):
    attention = CausalAttentionHead(d_model=4, head_dim=2, rope_theta=rope_theta)
    x = torch.randn(2, 3, 4)
    original, weights = attention(x)
    changed_x = x.clone()
    changed_x[:, 2] += 100
    changed, _ = attention(changed_x)

    torch.testing.assert_close(original[:, :2], changed[:, :2])
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))
    torch.testing.assert_close(weights[:, 0, 0], torch.ones(2))


def test_multi_head_outputs_weights_and_gradients_match_pytorch():
    attention = MultiHeadCausalAttention(d_model=4, num_heads=2).double()
    reference = torch.nn.MultiheadAttention(
        embed_dim=4, num_heads=2, bias=False, batch_first=True
    ).double()
    projection_names = ("w_q", "w_k", "w_v")
    with torch.no_grad():
        # PyTorch packs the heads into larger Q, K and V matrices.
        reference.in_proj_weight.copy_(
            torch.cat(
                [
                    torch.cat([getattr(head, name).weight for head in attention.heads])
                    for name in projection_names
                ]
            )
        )
        reference.out_proj.weight.copy_(attention.w_out.weight)

    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, actual_weights = attention(x)
    future_positions = torch.ones(3, 3, dtype=torch.bool).triu(diagonal=1)
    expected, expected_weights = reference(
        reference_x,
        reference_x,
        reference_x,
        attn_mask=future_positions,
        average_attn_weights=False,
    )
    assert actual.shape == (2, 3, 4)
    assert actual_weights.shape == (2, 2, 3, 3)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_weights, expected_weights)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(attention.w_out.weight.grad, reference.out_proj.weight.grad)
    for name, reference_gradient in zip(
        projection_names, reference.in_proj_weight.grad.chunk(3)
    ):
        actual_gradient = torch.cat(
            [getattr(head, name).weight.grad for head in attention.heads]
        )
        torch.testing.assert_close(actual_gradient, reference_gradient)
