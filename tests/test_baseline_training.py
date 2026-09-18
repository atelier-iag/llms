import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reimplementation import train_baseline
from reimplementation.checkpoint import load_checkpoint
from reimplementation.data import TokenFile
from reimplementation.model import CausalLanguageModel
from reimplementation.train import train_step
from reimplementation.train_baseline import learning_rate


@pytest.fixture(autouse=True)
def isolated_cpu_state():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("length,batch_size", [(12, 2), (13, 2), (14, 2), (17, 3)])
def test_epoch_scores_each_target_once_including_tail(tmp_path, length, batch_size):
    path = tmp_path / "tokens.npy"
    np.arange(length, dtype=np.uint32).tofile(path)
    original = path.read_bytes()
    data = TokenFile(path, sequence_length=3, vocab_size=length)
    batches = list(data.epoch_batches(batch_size, np.random.default_rng(5)))
    assert len(batches) == data.epoch_batch_count(batch_size)
    targets = torch.cat([batch[:, 1:].flatten() for batch in batches])
    assert sorted(targets.tolist()) == list(range(1, length))
    for batch in batches:
        torch.testing.assert_close(batch[:, 1:], batch[:, :-1] + 1)
        assert batch.shape[1] <= 4
    repeated = list(data.epoch_batches(batch_size, np.random.default_rng(5)))
    for a, b in zip(batches, repeated):
        torch.testing.assert_close(a, b)
    ordered = torch.cat([batch[:, 1:].flatten() for batch in data.epoch_batches(batch_size)])
    assert ordered.tolist() == list(range(1, length))
    assert path.read_bytes() == original


def test_warmup_and_cosine_schedule_endpoints():
    rates = [learning_rate(step, 10, peak=0.001, warmup=2, floor=0.1) for step in range(1, 11)]
    assert rates[:2] == pytest.approx([0.0005, 0.001])
    assert rates[-1] == pytest.approx(0.0001)
    assert all(a >= b for a, b in zip(rates[1:], rates[2:]))
    with pytest.raises(ValueError):
        learning_rate(1, 2, peak=0.001, warmup=2, floor=0.1)


def test_optional_gradient_clipping_bounds_sgd_update():
    model = CausalLanguageModel(vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    train_step(model, optimizer, torch.tensor([[2, 5, 2, 4]]), max_grad_norm=0.01)
    after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    assert (after - before).norm().item() == pytest.approx(0.01, abs=1e-6)


def test_full_baseline_run_records_coverage_curve_and_reload(tmp_path, monkeypatch):
    class TestTokenizer:
        def get_vocab_size(self):
            return 6

        def encode(self, text, add_special_tokens):
            return SimpleNamespace(ids=[2, 5, 2])

        def token_to_id(self, text):
            return 0

        def decode(self, ids, skip_special_tokens):
            return " ".join(map(str, ids))

    for split, repeats in [("train", 20), ("val", 11)]:
        np.tile(np.array([2, 5, 2, 4], dtype=np.uint32), repeats).tofile(tmp_path / f"{split}.npy")
    config = {
        "name": "test", "seed": 0,
        "model": dict(vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2),
        "sequence_length": 3, "batch_size": 2, "epochs": 1,
        "optimizer": dict(lr=0.01, betas=[0.9, 0.95], weight_decay=0.1),
        "warmup_updates": 2, "min_lr_ratio": 0.1, "max_grad_norm": 1.0,
        "evaluation_windows": 4, "evaluate_every": 5, "log_every": 3,
        "prompts": ["A prompt"], "max_new_tokens": 4,
    }
    monkeypatch.setattr(train_baseline, "load_tokenizer", lambda _: TestTokenizer())
    run_dir = train_baseline.run(config, tmp_path, device="cpu", output_root=tmp_path / "runs")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["training_target_tokens"] == 79
    assert metrics["updates"] == 14
    assert metrics["initial_full_validation"]["scored_tokens"] == 43
    assert metrics["final_full_validation"]["scored_tokens"] == 43
    assert metrics["final_full_validation"]["loss"] < metrics["initial_full_validation"]["loss"]
    assert [record["step"] for record in metrics["evaluation_history"]] == [0, 5, 10, 14]
    assert sum(record["window_target_tokens"] for record in metrics["training_history"]) == 79
    assert metrics["checkpoint_reload_max_logit_error"] == 0
    assert metrics["data_unchanged"]
    model, metadata = load_checkpoint(Path(metrics["checkpoint"]))
    assert metadata["step"] == 14
    assert model.embeddings.weight.shape == (6, 4)
    recovery = torch.load(run_dir / "recovery.pt", weights_only=True)
    assert recovery["training_target_tokens"] == 79
    assert recovery["step"] == 14
    assert len(recovery["optimizer_state"]["state"]) > 0
    assert not (run_dir / "recovery.tmp").exists()
