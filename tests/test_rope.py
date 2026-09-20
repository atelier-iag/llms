import math

import pytest
import torch
import torch.nn.functional as F

from reimplementation.attention import CausalAttentionHead
from reimplementation.rope import RotaryPositionEmbedding


def complex_reference(x, positions):
    """Independent rotation formula: multiply each complex pair by exp(i * angle)."""
    d = x.shape[-1]
    frequencies = x.new_tensor([10_000 ** (-2 * i / d) for i in range(d // 2)])
    angles = positions[:, None] * frequencies
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2))
    rotated = pairs * torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(rotated).flatten(-2)


def test_known_vectors_show_each_pairs_angular_speed():
    x = torch.tensor([[[1., 0., 1., 0.], [1., 0., 1., 0.]]], dtype=torch.float64)
    actual = RotaryPositionEmbedding(4)(x, torch.tensor([0, 1]))
    expected = x.new_tensor([[[1, 0, 1, 0],
                             [math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)]]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("head_dim", [2, 4, 48])
def test_rotation_matches_complex_reference_preserves_norm_and_gradients(head_dim):
    x = torch.randn(2, 5, head_dim, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(5)
    original = x.detach().clone()
    actual = RotaryPositionEmbedding(head_dim)(x, positions)
    expected = complex_reference(x, positions)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(actual[:, 0], x[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    probe = torch.randn_like(x)
    actual_gradient, = torch.autograd.grad((actual * probe).sum(), x)
    expected_gradient, = torch.autograd.grad((expected * probe).sum(), x)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_common_position_shift_preserves_all_query_key_scores():
    rope = RotaryPositionEmbedding(48)
    q = torch.randn(2, 5, 48, dtype=torch.float64)
    k = torch.randn_like(q)
    positions = torch.arange(5)
    original = rope(q, positions) @ rope(k, positions).transpose(-2, -1)
    shifted = rope(q, positions + 17) @ rope(k, positions + 17).transpose(-2, -1)
    torch.testing.assert_close(original, shifted, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rotation_preserves_vector_dtype(dtype):
    x = torch.randn(2, 5, 4, dtype=dtype, requires_grad=True)
    output = RotaryPositionEmbedding(4)(x, torch.arange(5))
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("head_dim,theta", [(0, 10000), (3, 10000), (4, 0),
                                           (4, -1), (4, float("nan")), (4, float("inf"))])
def test_invalid_rope_configuration_is_rejected(head_dim, theta):
    with pytest.raises(ValueError, match="RoPE"):
        CausalAttentionHead(d_model=4, head_dim=head_dim, rope_theta=theta)


def test_rotation_rejects_mismatched_positions_or_features():
    rope = RotaryPositionEmbedding(4)
    with pytest.raises(ValueError, match="one position per token"):
        rope(torch.randn(2, 3, 4), torch.arange(2))
    with pytest.raises(ValueError, match="head_dim"):
        rope(torch.randn(2, 3, 2), torch.arange(3))


def test_two_token_example_rotates_queries_and_keys_but_not_values():
    attention = CausalAttentionHead(2, 2, rope_theta=10000).double()
    with torch.no_grad():
        attention.w_q.weight.copy_(torch.tensor([[1, 1], [0, 0]]))
        attention.w_k.weight.copy_(attention.w_q.weight)
        attention.w_v.weight.copy_(10 * torch.eye(2))
    x = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    output, weights = attention(x)
    # Position 1 turns by one radian. Its score with position 0 is cos(1).
    previous = math.exp(math.cos(1) / math.sqrt(2))
    current = math.exp(1 / math.sqrt(2))
    expected_weights = x.new_tensor([[[1, 0], [previous, current]]])
    expected_weights[:, 1] /= previous + current
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, 10 * expected_weights)


def test_rope_attention_outputs_and_gradients_match_complex_rotation_and_sdpa():
    attention = CausalAttentionHead(8, 4, rope_theta=10000).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(5)
    actual, _ = attention(x)
    expected = F.scaled_dot_product_attention(
        complex_reference(attention.w_q(x), positions).unsqueeze(1),
        complex_reference(attention.w_k(x), positions).unsqueeze(1),
        attention.w_v(x).unsqueeze(1),
        is_causal=True,
    ).squeeze(1)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
