import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.language_model import evaluate
from reimplementation.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.data import TokenFile, open_splits
from reimplementation.generate import generate
from reimplementation.model import CausalLanguageModel
from reimplementation import train_corpus


CONFIG = dict(vocab_size=6, d_model=4, num_heads=2, hidden_size=16, num_layers=2)


@pytest.fixture(autouse=True)
def isolate_torch_state():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(previous_threads)


def test_raw_windows_preserve_source_and_include_last_target(tmp_path):
    path = tmp_path / "train.npy"
    np.arange(12, dtype=np.uint32).tofile(path)
    original = path.read_bytes()
    data = TokenFile(path, sequence_length=3, vocab_size=12)
    batch = data.batch(np.array([0, 8]))
    assert batch.dtype == torch.long
    assert batch.tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert data.evaluation_starts(3).tolist() == [0, 4, 8]
    assert not data.tokens.flags.writeable
    batch.fill_(0)
    assert path.read_bytes() == original
    torch.testing.assert_close(
        data.sample(np.random.default_rng(5), 10),
        data.sample(np.random.default_rng(5), 10),
    )
    with pytest.raises(ValueError, match="offset"):
        data.batch(np.array([9]))


@pytest.mark.parametrize("contents", [b"", b"12345", np.array([0, 1], dtype=np.uint32).tobytes()])
def test_rejects_incomplete_raw_files(tmp_path, contents):
    path = tmp_path / "bad.npy"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="raw uint32"):
        TokenFile(path, sequence_length=2, vocab_size=6)


def test_rejects_npy_containers_and_invalid_vocabulary(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array([0, 1, 2], dtype=np.uint32))
    with pytest.raises(ValueError, match="container"):
        TokenFile(path, sequence_length=2, vocab_size=6)
    np.array([0, 1, 6], dtype=np.uint32).tofile(path)
    with pytest.raises(ValueError, match="vocabulary"):
        TokenFile(path, sequence_length=2, vocab_size=6)


def test_validation_cannot_alias_training_file(tmp_path):
    path = tmp_path / "train.npy"
    np.array([0, 1, 2], dtype=np.uint32).tofile(path)
    (tmp_path / "val.npy").symlink_to(path)
    with pytest.raises(ValueError, match="separate files"):
        open_splits(tmp_path, sequence_length=2, vocab_size=6)


def test_evaluation_weights_tokens_and_preserves_weights_gradients_and_mode():
    model = CausalLanguageModel(**CONFIG).train()
    batches = [torch.tensor([[2, 5, 2, 4]]), torch.tensor([[0, 1, 2], [3, 4, 5]])]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    with torch.no_grad():
        logits = torch.cat([model(batch[:, :-1]).reshape(-1, 6) for batch in batches])
        targets = torch.cat([batch[:, 1:].reshape(-1) for batch in batches])
        expected = F.cross_entropy(logits, targets).item()
    metrics = evaluate(model, iter(batches))
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["perplexity"] == pytest.approx(np.exp(expected))
    assert metrics["scored_tokens"] == 7
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
    model.eval()
    with pytest.raises(ValueError, match="at least one"):
        evaluate(model, [])
    assert not model.training


def test_checkpoint_reconstructs_architecture_and_identical_logits(tmp_path):
    model = CausalLanguageModel(**CONFIG).eval()
    path = tmp_path / "model.pt"
    tokenizer = {"repo_id": "test-tokenizer", "revision": "test-revision"}
    save_checkpoint(path, model, CONFIG, step=7, context_length=3, tokenizer=tokenizer)
    restored, metadata = load_checkpoint(path)
    inputs = torch.tensor([[2, 5, 2]])
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    assert metadata["model_config"] == CONFIG
    assert metadata["step"] == 7
    assert metadata["context_length"] == 3
    assert metadata["tokenizer"] == tokenizer
    assert not restored.training
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, CONFIG, step=8, context_length=3, tokenizer=tokenizer)


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, tokens):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.seen.append(tokens.tolist())
        return F.one_hot((tokens + 1) % 6, num_classes=6).float() + self.anchor


@pytest.mark.parametrize("eos, expected", [(None, [0, 1, 2, 3, 4, 5]), (4, [0, 1, 2, 3, 4])])
def test_generation_uses_latest_token_crops_context_and_stops_at_eos(eos, expected):
    model = CountingModel().train()
    prompt = torch.tensor([[0, 1, 2]])
    output = generate(model, prompt, max_new_tokens=3, context_length=2, eos_token_id=eos)
    assert output.tolist() == [expected]
    assert model.seen == [[[1, 2]], [[2, 3]], [[3, 4]]][:len(expected) - 3]
    assert prompt.tolist() == [[0, 1, 2]]
    assert model.training
    assert model.anchor.grad is None
    unchanged = generate(model, prompt, max_new_tokens=0, context_length=2)
    torch.testing.assert_close(unchanged, prompt)


def test_corpus_cli_trains_evaluates_saves_reloads_and_generates(tmp_path, monkeypatch):
    class TestTokenizer:
        def get_vocab_size(self):
            return 6

        def encode(self, text, add_special_tokens):
            return type("Encoding", (), {"ids": [2, 5, 2]})()

        def token_to_id(self, text):
            return 0

        def decode(self, ids, skip_special_tokens):
            return " ".join(map(str, ids))

    for split in ("train", "val"):
        np.tile(np.array([2, 5, 2, 4], dtype=np.uint32), 20).tofile(tmp_path / f"{split}.npy")
    monkeypatch.setattr(train_corpus, "ROOT", tmp_path)
    monkeypatch.setattr(train_corpus, "load_tokenizer", lambda spec: TestTokenizer())
    monkeypatch.setattr(sys, "argv", [
        "train_corpus", "--data-dir", str(tmp_path), "--device", "cpu", "--steps", "4",
        "--sequence-length", "3", "--d-model", "4", "--num-heads", "2", "--eval-batches", "2",
    ])
    train_corpus.main()
    metrics_path, = (tmp_path / "runs").glob("*/metrics.json")
    metrics = json.loads(metrics_path.read_text())
    assert metrics["training_target_tokens"] == 24
    assert metrics["before"]["validation"]["scored_tokens"] == 12
    assert metrics["after"]["validation"]["loss"] < metrics["before"]["validation"]["loss"]
    assert metrics["data_unchanged"]
    assert metrics["checkpoint_reload_max_logit_error"] == 0
    assert metrics["generation"]["token_ids"][:3] == [2, 5, 2]
    assert len(metrics["generation"]["token_ids"]) > 3
    restored, metadata = load_checkpoint(metrics_path.parent / "model.pt")
    assert metadata["step"] == 4
    assert restored.embeddings.weight.shape == (6, 4)
