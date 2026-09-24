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


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
@pytest.mark.parametrize("num_kv_heads", [None, 1])
def test_full_baseline_run_records_coverage_curve_and_reload(
    tmp_path, monkeypatch, rope_theta, num_kv_heads,
):
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
    if rope_theta is not None:
        config["model"]["rope_theta"] = rope_theta
    if num_kv_heads is not None:
        config["model"]["num_kv_heads"] = num_kv_heads
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
    assert metadata["model_config"] == config["model"]
    assert model.embeddings.weight.shape == (6, 4)
    recovery = torch.load(run_dir / "recovery.pt", weights_only=True)
    assert recovery["training_target_tokens"] == 79
    assert recovery["step"] == 14
    assert len(recovery["optimizer_state"]["state"]) > 0
    assert not (run_dir / "recovery.tmp").exists()


def test_rope_comparison_changes_only_position_encoding_and_experiment_name():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "reimplementation/baseline_config.json").read_text())
    rope = json.loads((root / "experiments/rope_config.json").read_text())
    assert rope.pop("name") == "simple-384-rope"
    baseline.pop("name")
    assert rope["model"].pop("rope_theta") == 10000.0
    assert rope == baseline


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    tokenizer = SimpleNamespace(
        get_vocab_size=lambda: 6,
        encode=lambda text, add_special_tokens: SimpleNamespace(ids=[2, 5, 2]),
        token_to_id=lambda text: 0,
        decode=lambda ids, skip_special_tokens: " ".join(map(str, ids)),
    )
    monkeypatch.setattr(train_baseline, "load_tokenizer", lambda _: tokenizer)
    rng = np.random.default_rng(4)
    for split, length in [("train", 80), ("val", 44)]:
        rng.integers(6, size=length, dtype=np.uint32).tofile(tmp_path / f"{split}.npy")
    return {
        "name": "resume-test", "seed": 0,
        "model": dict(vocab_size=6, d_model=4, num_heads=2, num_kv_heads=1,
                      hidden_size=8, num_layers=2, rope_theta=10000.0),
        "sequence_length": 3, "batch_size": 2, "epochs": 2,
        "optimizer": dict(lr=0.01, betas=[0.9, 0.95], weight_decay=0.1),
        "warmup_updates": 2, "min_lr_ratio": 0.1, "max_grad_norm": 1.0,
        "evaluation_windows": 4, "evaluate_every": 5, "log_every": 3,
        "prompts": ["A prompt"], "max_new_tokens": 4,
    }


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("checkpoint_step,legacy", [(5, False), (15, False), (15, True), (28, False)])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, resume_case, checkpoint_step, legacy):
    batches = []
    stop_after = None

    def stochastic_step(model, optimizer, tokens, **kwargs):
        if stop_after is not None and len(batches) == stop_after:
            raise KeyboardInterrupt
        batches.append(tokens.clone())
        # Exercise RNG restoration as well as deterministic corpus permutations.
        augmented = tokens.clone()
        augmented[0, 0] = torch.randint(6, ())
        return train_step(model, optimizer, augmented, **kwargs)

    monkeypatch.setattr(train_baseline, "train_step", stochastic_step)
    reference_dir = train_baseline.run(resume_case, tmp_path, device="cpu", output_root=tmp_path / "full")
    reference_batches = batches.copy()
    reference = torch.load(reference_dir / "recovery.pt", weights_only=True)
    reference_metrics = json.loads((reference_dir / "metrics.json").read_text())

    if checkpoint_step < 28:
        batches.clear()
        stop_after = checkpoint_step + 2  # Discard unsaved updates after the checkpoint.
        with pytest.raises(KeyboardInterrupt):
            train_baseline.run(resume_case, tmp_path, device="cpu", output_root=tmp_path / "interrupted")
        source_dir = next((tmp_path / "interrupted").iterdir())
    else:
        source_dir = reference_dir  # Finish evaluation/generation after a final recovery save.
    source = source_dir / "recovery.pt"
    saved = torch.load(source, weights_only=True)
    assert saved["step"] == checkpoint_step
    if legacy:
        del saved["run_state"]
        torch.save(saved, source)
        with (source_dir / "progress.jsonl").open("a") as stream:
            stream.write('{"event": "train",')  # A log line torn by the crash.
    before = source.read_bytes()
    batches.clear()
    stop_after = None
    resumed_dir = train_baseline.run(resume_case, tmp_path, device="cpu",
                                      output_root=tmp_path / "resumed", resume=source)
    actual = torch.load(resumed_dir / "recovery.pt", weights_only=True) if checkpoint_step < 28 else saved
    actual_model, _ = load_checkpoint(resumed_dir / "model.pt")
    assert_nested_equal(actual_model.state_dict(), reference["model_state"])
    assert_nested_equal(actual["optimizer_state"], reference["optimizer_state"])
    assert_nested_equal(actual["torch_rng_state"], reference["torch_rng_state"])
    assert_nested_equal(batches, reference_batches[checkpoint_step:])
    metrics = json.loads((resumed_dir / "metrics.json").read_text())
    assert metrics["training_target_tokens"] == 158
    assert metrics["updates"] == 28
    assert metrics["session"]["updates"] == 28 - checkpoint_step
    assert metrics["session"]["training_target_tokens"] == 158 - saved["training_target_tokens"]
    for field in ["initial_full_validation", "final_full_validation", "evaluation_history", "generations"]:
        assert metrics[field] == reference_metrics[field]
    fields = ["step", "target_tokens", "lr", "mean_training_loss", "window_target_tokens"]
    assert [[row[key] for key in fields] for row in metrics["training_history"]] == [
        [row[key] for key in fields] for row in reference_metrics["training_history"]
    ]
    assert metrics["training_seconds"] is None  # No misleading full-run cost after resumption.
    assert metrics["wall_seconds"] is None
    assert metrics["peak_cuda_allocated_bytes"] is None
    assert metrics["checkpoint_reload_max_logit_error"] == 0
    assert source.read_bytes() == before
    events = [json.loads(line)["event"] for line in (resumed_dir / "progress.jsonl").read_text().splitlines()]
    assert "resume" in events
    assert "initial" not in events


@pytest.mark.parametrize("mismatch", ["config", "data", "count", "inference"])
def test_resume_rejects_incompatible_checkpoint_before_creating_run(tmp_path, resume_case, mismatch):
    source_dir = train_baseline.run(resume_case, tmp_path, device="cpu", output_root=tmp_path / "original")
    source = source_dir / "recovery.pt"
    state = torch.load(source, weights_only=True)
    if mismatch == "config":
        resume_case["seed"] += 1
        message = "configuration/tokenizer"
    elif mismatch == "data":
        data_path = tmp_path / "train.npy"
        tokens = np.fromfile(data_path, dtype=np.uint32)
        tokens[0] = (tokens[0] + 1) % 6
        tokens.tofile(data_path)
        message = "data hashes"
    elif mismatch == "count":
        state["training_target_tokens"] -= 1
        torch.save(state, source)
        message = "token count"
    else:
        source = source_dir / "model.pt"
        message = "optimizer state"
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=message):
        train_baseline.run(resume_case, tmp_path, device="cpu", output_root=output_root, resume=source)
    assert not output_root.exists()


def test_resume_cli_uses_saved_configuration(tmp_path, monkeypatch, resume_case):
    source_dir = train_baseline.run(resume_case, tmp_path, device="cpu", output_root=tmp_path / "original")
    source = source_dir / "recovery.pt"
    monkeypatch.setattr(train_baseline, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["train_baseline", "--device", "cpu", "--data-dir", str(tmp_path),
                                    "--resume", str(source)])
    train_baseline.main()
    resumed_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((resumed_dir / "config.json").read_text()) == resume_case
