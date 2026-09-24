import copy
import json
import math
from pathlib import Path

import pytest
import torch

from reimplementation.loss import cross_entropy_loss
from reimplementation.model import CausalLanguageModel
from reimplementation.normalization import RMSNorm
from reimplementation.precision import training_autocast, validate_precision
from reimplementation.train import train_step
from reimplementation.train_baseline import run


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_computes_low_precision_logits_and_updates_fp32_weights(device):
    if device == "cuda" and (not torch.cuda.is_available()
                            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        pytest.skip("native CUDA BF16 unavailable")
    validate_precision("bf16", device)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            torch.manual_seed(0)
            model = CausalLanguageModel(vocab_size=16, d_model=16, num_heads=4,
                                       num_kv_heads=2, hidden_size=32, num_layers=2,
                                       rope_theta=10000.0).to(device)
            reference = copy.deepcopy(model)
            tokens = torch.tensor([[2, 5, 2, 4]], device=device)
            before = model.decoder.blocks[0].attention.w_q[0].weight.detach().clone()
            with training_autocast(device, "bf16"):
                logits = model(tokens[:, :-1])
                assert logits.dtype == torch.bfloat16
                assert cross_entropy_loss(logits, tokens[:, 1:]).dtype == torch.float32
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss = train_step(model, optimizer,
                              tokens, max_grad_norm=1.0, precision="bf16")
            reference_loss = train_step(reference, torch.optim.AdamW(reference.parameters(), lr=0.001),
                                        tokens, max_grad_norm=1.0)
            assert math.isfinite(loss)
            assert loss == pytest.approx(reference_loss, abs=0.02)
            assert not torch.equal(before, model.decoder.blocks[0].attention.w_q[0].weight)
            for parameter in model.parameters():
                assert parameter.dtype == torch.float32
                assert parameter.grad is not None and parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                assert torch.isfinite(parameter).all()
                assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
                assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    finally:
        torch.set_num_threads(previous_threads)


def test_bf16_reductions_are_fp32_and_match_explicit_fp32_reference():
    logits = torch.tensor([[[30.0, -20.0, 0.0]]], dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([[2]])
    actual = cross_entropy_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 3), targets.flatten())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    x = torch.tensor([[[1024.0, 0.5, -8.0, 1.0]]], dtype=torch.bfloat16)
    norm = RMSNorm(4)
    torch.testing.assert_close(norm(x), norm(x.float()), rtol=0, atol=0)
    assert norm(x).dtype == torch.float32


def test_invalid_precision_is_rejected_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        run({"precision": "fp16"}, tmp_path, device="cpu", output_root=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_bf16_config_only_changes_precision_and_experiment_name():
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "experiments/gqa_config.json").read_text())
    actual = json.loads((root / "experiments/bf16_config.json").read_text())
    reference.pop("name")
    actual.pop("name")
    assert actual.pop("precision") == "bf16"
    assert actual == reference
