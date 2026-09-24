import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from reimplementation.attention import MultiHeadCausalAttention
from reimplementation.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.gqa import GroupedQueryCausalAttention
from reimplementation.model import CausalLanguageModel
from reimplementation.precision import training_autocast


def attention_pair(kv_heads, rope, dtype=torch.float64, device="cpu"):
    torch.manual_seed(3)
    if kv_heads is None:
        manual = MultiHeadCausalAttention(32, 4, rope_theta=rope)
    else:
        manual = GroupedQueryCausalAttention(32, 4, kv_heads, rope_theta=rope)
    manual = manual.to(device=device, dtype=dtype)
    optimized = copy.deepcopy(manual)
    optimized.attention_backend = "sdpa"
    return manual, optimized


@pytest.mark.parametrize("kv_heads", [None, 1, 2, 4])
@pytest.mark.parametrize("rope", [None, 10000.0])
def test_sdpa_matches_manual_outputs_and_every_gradient(kv_heads, rope):
    manual, optimized = attention_pair(kv_heads, rope)
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected, weights = manual(reference_x)
    actual, absent_weights = optimized(x)
    assert weights is not None and absent_weights is None
    assert manual.state_dict().keys() == optimized.state_dict().keys()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    probe = torch.randn_like(actual)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-9, atol=1e-10)
    for a, b in zip(optimized.parameters(), manual.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("kv_heads", [None, 1, 2, 4])
def test_sdpa_is_causal_and_works_on_single_token(kv_heads):
    _, model = attention_pair(kv_heads, 10000.0)
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    original, _ = model(x)
    changed = x.detach().clone()
    changed[:, 3:] += 100
    modified, _ = model(changed)
    torch.testing.assert_close(original[:, :3], modified[:, :3], rtol=0, atol=0)
    gradient, = torch.autograd.grad(original[:, :3].sum(), x)
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    single, _ = model(x[:, :1])
    torch.testing.assert_close(single, original[:, :1])


def test_sdpa_checkpoint_records_backend_and_preserves_legacy_parameters(tmp_path):
    config = dict(vocab_size=16, d_model=16, num_heads=4, num_kv_heads=2,
                  hidden_size=32, num_layers=2, rope_theta=10000.0)
    legacy = CausalLanguageModel(**config).eval()
    config["attention_backend"] = "sdpa"
    model = CausalLanguageModel(**config).eval()
    model.load_state_dict(legacy.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), legacy(tokens), rtol=1e-5, atol=1e-6)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config, step=1, context_length=4, tokenizer={})
    loaded, metadata = load_checkpoint(path)
    assert metadata["model_config"]["attention_backend"] == "sdpa"
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens), model(tokens), rtol=0, atol=0)


def test_flash_gqa_cuda_bf16_outputs_and_gradients_are_close_to_manual():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("CUDA BF16 unavailable")
    manual, optimized = attention_pair(2, 10000.0, dtype=torch.float32, device="cuda")
    x = torch.randn(2, 32, 32, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with training_autocast("cuda", "bf16"):
        expected, _ = manual(reference_x)
        # This test must execute FlashAttention, with no silent math fallback.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual, _ = optimized(x)
    probe = torch.randn_like(actual)
    (expected.float() * probe.float()).sum().backward()
    (actual.float() * probe.float()).sum().backward()
    for a, b in [(actual, expected), (x.grad, reference_x.grad),
                 *[(a.grad, b.grad) for a, b in zip(optimized.parameters(), manual.parameters())]]:
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        relative_error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert relative_error < 0.025


def test_sdpa_configuration_changes_only_backend_and_name():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "experiments/bf16_config.json").read_text())
    optimized = json.loads((root / "experiments/sdpa_config.json").read_text())
    baseline.pop("name"); optimized.pop("name")
    assert optimized["model"].pop("attention_backend") == "sdpa"
    assert baseline == optimized


def test_unknown_attention_backend_is_rejected():
    with pytest.raises(ValueError, match="attention_backend"):
        GroupedQueryCausalAttention(8, 4, 2, attention_backend="typo")
