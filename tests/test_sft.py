import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.sft import evaluate_responses, normalized_answer, token_f1
from reimplementation import train_sft
from reimplementation.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.model import CausalLanguageModel
from reimplementation.prepare_corpus import file_hash
from reimplementation.prepare_sft import CATEGORIES, group_examples, prepare, split_group
from reimplementation.sft_data import FORMAT, IGNORE_INDEX, batches, collate, load_sft
from reimplementation.sft_loss import completion_loss, evaluate_sft, sft_step
from reimplementation.tokenizer import DOLMA_TOKENIZER


@pytest.fixture(autouse=True)
def isolated_cpu_state():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(previous)


def test_response_shift_masks_prompt_and_padding_but_not_eos():
    examples = [{"ids": [1, 2, 3, 4, 0], "prompt_tokens": 2},
                {"ids": [2, 5, 0], "prompt_tokens": 1}]
    inputs, labels = collate(examples, pad_id=0)
    assert inputs.tolist() == [[1, 2, 3, 4], [2, 5, 0, 0]]
    assert labels.tolist() == [[-100, 3, 4, 0], [5, 0, -100, -100]]
    logits = torch.randn(2, 4, 6, requires_grad=True)
    loss = completion_loss(logits, labels)
    expected = torch.stack([-torch.log_softmax(logits[b, t], -1)[labels[b, t]]
                            for b in range(2) for t in range(4) if labels[b, t] != -100]).mean()
    torch.testing.assert_close(loss, expected)
    altered = logits.detach().clone()
    altered[labels == -100] = torch.randn_like(altered[labels == -100]) * 100
    torch.testing.assert_close(completion_loss(altered, labels), loss)
    loss.backward()
    assert not logits.grad[labels == -100].any()
    assert logits.grad[labels != -100].abs().sum() > 0
    with pytest.raises(ValueError, match="requires supervised"):
        completion_loss(logits, torch.full_like(labels, -100))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_right_padding_cannot_affect_real_tokens_and_metrics_are_token_weighted(backend):
    model = CausalLanguageModel(vocab_size=8, d_model=8, num_heads=2, num_layers=2,
                                hidden_size=16, attention_backend=backend, rope_theta=10000)
    rows = [{"ids": [1, 2, 3, 0], "prompt_tokens": 2},
            {"ids": [4, 5, 6, 2, 3, 1, 0], "prompt_tokens": 3}]
    inputs, _ = collate(rows, 0)
    torch.testing.assert_close(model(inputs)[0, :3], model(torch.tensor([[1, 2, 3]]))[0], rtol=1e-5, atol=1e-6)
    first = evaluate_sft(model, batches(rows, 1, 0))
    second = evaluate_sft(model, batches(rows, 2, 0))
    assert first["scored_tokens"] == second["scored_tokens"] == 6
    assert first["loss"] == pytest.approx(second["loss"], abs=1e-6)
    assert first["teacher_forced_token_accuracy"] == second["teacher_forced_token_accuracy"]


def test_grouping_keeps_shared_context_instruction_and_near_variants_together():
    common = " ".join(f"word{i}" for i in range(40))
    rows = [dict(instruction="Find a date", context="Shared source paragraph", fingerprint="a"),
            dict(instruction="Find a name", context="SHARED source paragraph", fingerprint="b"),
            dict(instruction="Find a name", context="Different source", fingerprint="c"),
            dict(instruction=common, context="", fingerprint="d"),
            dict(instruction=common + " please", context="", fingerprint="e"),
            dict(instruction="Independent example", context="", fingerprint="f")]
    groups = group_examples(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] == groups[4]
    assert len(set(groups)) == 3
    assert [split_group(group, 0) for group in groups[:3]] == [split_group(groups[0], 0)] * 3


class TinyTokenizer:
    def get_vocab_size(self):
        return 8

    def token_to_id(self, text):
        return 0

    def encode(self, text, add_special_tokens=False):
        if text.startswith("### Instruction:"):
            return SimpleNamespace(ids=[1, 2, 3])
        return SimpleNamespace(ids=[4, 5] if text != "LONG" else [4] * 20)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, (i for i in ids if i != 0)))


def test_preparation_reproducible_group_disjoint_untruncated_and_requires_valid_identity(tmp_path):
    rows = [dict(instruction=f"question-{i}", context="", response=f"answer-{i}", category=CATEGORIES[i % 4])
            for i in range(200)]
    rows += [rows[0].copy(), {**rows[1], "response": "LONG"}, {**rows[2], "response": ""}]
    kwargs = dict(tokenizer=TinyTokenizer(), budgets=dict(train=20, val=8, holdout=8),
                  max_tokens=16, max_answer_tokens=8, excluded_ids=[2])
    manifest = prepare(rows, tmp_path / "one", **kwargs)
    assert manifest == prepare(rows, tmp_path / "two", **kwargs)
    assert manifest["statistics"]["duplicates"] == 1
    assert manifest["statistics"]["length_filtered"] == 1
    assert manifest["statistics"]["quality_excluded"] == 1
    groups = {}
    for split in ("train", "val", "holdout"):
        records = [json.loads(line) for line in (tmp_path / "one" / f"{split}.jsonl").read_text().splitlines()]
        for record in records:
            assert record["source_id"] != 2
            assert record["ids"] == [1, 2, 3, 4, 5, 0]
            assert groups.setdefault(record["group"], split) == split
    # The training loader does not open the reserved holdout.
    (tmp_path / "one/holdout.jsonl").unlink()
    data, identity = load_sft(tmp_path / "one", tokenizer=DOLMA_TOKENIZER, vocab_size=8, max_tokens=16)
    assert len(data["train"]) == 20
    assert identity["sha256"] == file_hash(tmp_path / "one/manifest.json")
    with pytest.raises(FileExistsError):
        prepare(rows, tmp_path / "one", **kwargs)
    with (tmp_path / "one/train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        load_sft(tmp_path / "one", tokenizer=DOLMA_TOKENIZER, vocab_size=8, max_tokens=16)
    with pytest.raises(ValueError, match="not enough"):
        prepare(rows[:1], tmp_path / "incomplete", **kwargs)
    assert json.loads((tmp_path / "incomplete/manifest.json").read_text())["status"] == "incomplete"


def test_free_generation_receives_no_reference_answer_and_stops_at_eos():
    class PredictAnswer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.received = []

        def forward(self, inputs):
            self.received.append(inputs.clone())
            logits = torch.zeros(inputs.shape[0], inputs.shape[1], 8)
            logits[:, -1, 4 if inputs.shape[1] == 2 else 0] = 10
            return logits

    model = PredictAnswer()
    row = dict(ids=[1, 2, 4, 0], prompt_tokens=2, source_id=1, category="classification")
    scores, details = evaluate_responses(model, [row], TinyTokenizer(), context_length=16, max_new_tokens=8, eos_id=0)
    assert model.received[0].tolist() == [[1, 2]]
    assert scores["exact_matches"] == 1 and scores["eos_rate"] == 1
    assert details[0]["generated_ids"] == [4, 0]
    assert token_f1("red blue", "red green") == 0.5
    assert normalized_answer("  BLUE. ") == "blue"


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(train_sft, "load_tokenizer", lambda _: tokenizer)
    config = {"name": "sft-test", "seed": 0, "precision": "fp32",
              "model": dict(vocab_size=8, d_model=8, num_heads=2, num_kv_heads=1,
                            hidden_size=16, num_layers=2, rope_theta=10000.0, attention_backend="sdpa"),
              "sequence_length": 16, "batch_size": 2, "epochs": 2,
              "optimizer": dict(lr=0.01, betas=[0.9, 0.95], weight_decay=0.01),
              "warmup_updates": 1, "min_lr_ratio": 0.1, "max_grad_norm": 1.0,
              "evaluation_windows": 2, "evaluate_every": 2, "log_every": 3,
              "max_new_tokens": 4, "probe_max_new_tokens": 4, "generation_examples": 2,
              "train_evaluation_examples": 2, "corpus": dict(train_examples=6, val_examples=4),
              "general_evaluation": dict(require_manifest=False, validation_tokens=40)}
    manifest = dict(format_version=1, kind="sft", status="complete", prompt_format=FORMAT,
                    max_tokens=16, vocab_size=8, eos_token_id=0, provenance=dict(tokenizer=DOLMA_TOKENIZER), files={})
    for split, count in (("train", 6), ("val", 4)):
        rows = [dict(source_id=i, category="classification", group=f"{split}-{i}", fingerprint=f"{split}-{i}",
                     ids=[1, 2, 3, 4, 5, 0] + ([] if i % 2 else [0]), prompt_tokens=3) for i in range(count)]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest["files"][split] = dict(path=path.name, examples=count, sha256=file_hash(path),
                                        sequence_tokens=sum(len(row["ids"]) for row in rows),
                                        response_tokens=sum(len(row["ids"]) - 3 for row in rows))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    general = tmp_path / "general"
    general.mkdir()
    np.tile(np.array([1, 2, 6, 4], dtype=np.uint32), 10).tofile(general / "val.npy")
    probes = tmp_path / "probes.json"
    probes.write_text(json.dumps([dict(id="probe", category="extraction", instruction="Question", context="", response="Answer")]))
    source = tmp_path / "source.pt"
    model = CausalLanguageModel(**config["model"])
    save_checkpoint(source, model, config["model"], step=50, context_length=16, tokenizer=DOLMA_TOKENIZER)
    return config, dict(data_dir=tmp_path, general_data_dir=general, probes_path=probes, device="cpu"), source


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            assert_state_equal(aa, bb)
    else:
        assert a == b


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_sft_full_run_and_interrupted_resume_are_identical(tmp_path, monkeypatch, training_case, precision):
    config, kwargs, source = training_case
    config["precision"] = precision
    original_hash = file_hash(source)
    source_model, _ = load_checkpoint(source)
    calls = []
    stop_after = None

    def checked_step(model, optimizer, batch, **settings):
        if stop_after is not None and len(calls) == stop_after:
            raise KeyboardInterrupt
        if not calls:
            assert not optimizer.state
            assert_state_equal(model.state_dict(), source_model.state_dict())
            assert optimizer.param_groups[0]["lr"] == 0.01
        inputs, labels = batch
        assert not (inputs == 6).any()  # Token 6 belongs only to general dev.
        calls.append(inputs.clone())
        augmented = inputs.clone()
        augmented[0, 0] = torch.randint(1, 6, ())  # Exercise RNG restoration.
        return sft_step(model, optimizer, (augmented, labels), **settings)

    monkeypatch.setattr(train_sft, "sft_step", checked_step)
    full = train_sft.run(config, output_root=tmp_path / "full", init_from=source, **kwargs)
    expected = torch.load(full / "recovery.pt", weights_only=True)
    metrics = json.loads((full / "metrics.json").read_text())
    assert metrics["updates"] == 6 and metrics["seen_examples"] == 12
    assert metrics["response_tokens"] == 42
    assert sum(row["window_response_tokens"] for row in metrics["training_history"]) == 42
    assert metrics["initial"]["dev_response"]["scored_tokens"] == 14
    assert metrics["final"]["general_validation_full"]["scored_tokens"] == 39
    assert metrics["checkpoint_reload_max_logit_error"] == 0
    assert "dev_generation_details" not in metrics["initial"]
    assert metrics["initialization"]["sha256"] == original_hash
    calls.clear()
    stop_after = 3
    with pytest.raises(KeyboardInterrupt):
        train_sft.run(config, output_root=tmp_path / "interrupted", init_from=source, **kwargs)
    checkpoint = next((tmp_path / "interrupted").iterdir()) / "recovery.pt"
    assert torch.load(checkpoint, weights_only=True)["step"] == 2
    before = file_hash(checkpoint)
    # Keep the same stochastic operation, but skip the fresh-optimizer assertion.
    calls[:] = [None]
    stop_after = None
    resumed = train_sft.run(config, output_root=tmp_path / "resumed", resume=checkpoint, **kwargs)
    actual = torch.load(resumed / "recovery.pt", weights_only=True)
    for key in ("model_state", "optimizer_state", "torch_rng_state", "response_tokens", "seen_examples"):
        assert_state_equal(actual[key], expected[key])
    resumed_metrics = json.loads((resumed / "metrics.json").read_text())
    for key in ("initial", "final", "evaluation_history", "initialization"):
        assert resumed_metrics[key] == metrics[key]
    assert resumed_metrics["training_seconds"] is None
    assert file_hash(checkpoint) == before and file_hash(source) == original_hash


@pytest.mark.parametrize("changed", ["general", "manifest", "probes", "config", "count"])
def test_sft_resume_rejects_changed_identity_before_creating_run(tmp_path, training_case, changed):
    config, kwargs, source = training_case
    original = train_sft.run(config, output_root=tmp_path / "original", init_from=source, **kwargs)
    checkpoint = original / "recovery.pt"
    if changed == "general":
        np.zeros(40, dtype=np.uint32).tofile(kwargs["general_data_dir"] / "val.npy")
    elif changed == "manifest":
        path = tmp_path / "manifest.json"
        path.write_text(path.read_text() + "\n")
    elif changed == "probes":
        path = kwargs["probes_path"]
        path.write_text(path.read_text() + "\n")
    elif changed == "config":
        config["seed"] += 1
    else:
        state = torch.load(checkpoint, weights_only=True)
        state["response_tokens"] -= 1
        torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="identity|configuration|coverage"):
        train_sft.run(config, output_root=tmp_path / "bad", resume=checkpoint, **kwargs)
    assert not (tmp_path / "bad").exists()
