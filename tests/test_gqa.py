import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reimplementation.attention import MultiHeadCausalAttention
from reimplementation.gqa import GroupedQueryCausalAttention
from reimplementation.model import CausalLanguageModel


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_outputs_and_shared_parameter_gradients_match_sdpa(num_kv_heads, rope_theta):
    attention = GroupedQueryCausalAttention(8, 4, num_kv_heads, rope_theta=rope_theta).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    actual, weights = attention(x)

    # Reference: materialize K/V per query head and let PyTorch compute attention.
    # repeat_interleave's backward sums all contributions to the shared projections.
    q = torch.stack([layer(x) for layer in attention.w_q], dim=1)
    k = torch.stack([layer(x) for layer in attention.w_k], dim=1)
    v = torch.stack([layer(x) for layer in attention.w_v], dim=1)
    if attention.rope is not None:
        positions = torch.arange(x.shape[1])
        q = attention.rope(q, positions)
        k = attention.rope(k, positions)
    k = k.repeat_interleave(4 // num_kv_heads, dim=1)
    v = v.repeat_interleave(4 // num_kv_heads, dim=1)
    expected_heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = attention.w_out(expected_heads.transpose(1, 2).flatten(-2))

    assert actual.shape == (2, 5, 8)
    assert weights.shape == (2, 4, 5, 5)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *attention.parameters())
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_one_kv_group_per_query_head_matches_existing_mha(rope_theta):
    mha = MultiHeadCausalAttention(8, 4, rope_theta=rope_theta).double()
    gqa = GroupedQueryCausalAttention(8, 4, 4, rope_theta=rope_theta).double()
    with torch.no_grad():
        for index, head in enumerate(mha.heads):
            for name in ("w_q", "w_k", "w_v"):
                getattr(gqa, name)[index].weight.copy_(getattr(head, name).weight)
        gqa.w_out.weight.copy_(mha.w_out.weight)
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    actual, actual_weights = gqa(x)
    expected, expected_weights = mha(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
def test_gqa_is_causal_in_values_and_gradients(rope_theta):
    attention = GroupedQueryCausalAttention(8, 4, 2, rope_theta=rope_theta).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    original, weights = attention(x)
    changed_x = x.detach().clone()
    changed_x[:, 3:] += 100
    changed, _ = attention(changed_x)
    torch.testing.assert_close(original[:, :3], changed[:, :3])
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
    gradient, = torch.autograd.grad(original[:, :3].sum(), x)
    assert torch.count_nonzero(gradient[:, 3:]) == 0


def test_each_shared_key_and_value_projection_runs_once():
    attention = GroupedQueryCausalAttention(16, 8, 2, rope_theta=10000)
    counts = {}

    def count_call(module, inputs, output):
        counts[module] = counts.get(module, 0) + 1

    layers = [*attention.w_k, *attention.w_v]
    hooks = [layer.register_forward_hook(count_call) for layer in layers]
    try:
        attention(torch.randn(2, 5, 16))
    finally:
        for hook in hooks:
            hook.remove()
    assert counts == {layer: 1 for layer in layers}


@pytest.mark.parametrize("d_model,num_heads,num_kv_heads", [
    (8, 0, 1), (8, 3, 1), (8, 4, 0), (8, 4, -1), (8, 4, 3), (8, 4, 8),
])
def test_invalid_head_counts_are_rejected(d_model, num_heads, num_kv_heads):
    with pytest.raises(ValueError):
        GroupedQueryCausalAttention(d_model, num_heads, num_kv_heads)


def test_full_size_gqa_configuration_and_parameter_count():
    root = Path(__file__).resolve().parents[1]
    rope = json.loads((root / "experiments/rope_config.json").read_text())
    gqa = json.loads((root / "experiments/gqa_config.json").read_text())
    with torch.device("meta"):
        model = CausalLanguageModel(**gqa["model"])
    assert sum(p.numel() for p in model.parameters()) == 94_124_928
    for block in model.decoder.blocks:
        attention = block.attention
        assert len(attention.w_q) == 8
        assert len(attention.w_k) == len(attention.w_v) == 2
        assert attention.head_dim == 48
        assert attention.heads_per_group == 4
        assert attention.rope.theta == 10000
    assert gqa.pop("name") == "simple-384-rope-gqa2"
    rope.pop("name")
    assert gqa["model"].pop("num_kv_heads") == 2
    assert gqa == rope
