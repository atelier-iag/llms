from collections import Counter
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from experiments.sft_diagnostic_data import build_examples, prepare, solve, validate_examples
from reimplementation.sft_data import format_prompt, load_sft
from reimplementation.tokenizer import DOLMA_TOKENIZER
from reimplementation.train_sft import select_evaluation_examples


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_is_reproducible_disjoint_balanced_and_query_sensitive():
    saved = json.loads((ROOT / "experiments/sft_diagnostic_examples.json").read_text())
    assert saved == build_examples()
    assert {split: len(rows) for split, rows in saved.items()} == {"train": 32, "val": 64}
    prompts = [{format_prompt(row["instruction"], row["context"]) for row in saved[split]}
               for split in ("train", "val")]
    assert len(prompts[0]) == 32 and len(prompts[1]) == 64
    assert not prompts[0] & prompts[1]
    old = json.loads((ROOT / "experiments/sft_probes.json").read_text())
    assert not (prompts[0] | prompts[1]) & {format_prompt(row["instruction"], row["context"]) for row in old}
    for category in ("copy", "color", "name"):
        seen = Counter(row["response"] for row in saved["train"] if row["category"] == category)
        new = Counter(row["response"] for row in saved["val"] if row["category"] == category)
        assert new == {answer: 2 * count for answer, count in seen.items()}
        train_contexts = {row["context"] for row in saved["train"] if row["category"] == category}
        assert not train_contexts & {row["context"] for row in saved["val"] if row["category"] == category}
    # Identical contexts with opposite questions must demand different answers.
    for split in ("train", "val"):
        for row in saved[split]:
            if row["category"] in ("color", "name"):
                pairs = [other for other in saved[split] if other["context"] == row["context"]]
                assert len(pairs) == 2 and len({solve(other) for other in pairs}) == 2


@pytest.mark.parametrize("mutation", ["label", "duplicate", "ambiguous", "count"])
def test_diagnostic_rejects_corrupt_examples(mutation):
    rows = build_examples()
    if mutation == "label":
        rows["train"][0]["response"] = "wrong"
    elif mutation == "duplicate":
        rows["val"][0] = copy.deepcopy(rows["train"][0])
    elif mutation == "ambiguous":
        rows["train"][0]["context"] += "\nTarget word: duplicate"
    else:
        rows["train"].pop()
    with pytest.raises(ValueError):
        validate_examples(rows)


def test_prepared_diagnostic_reuses_sft_loader_and_train_only_memorization_probes(tmp_path):
    class Tokenizer:
        def get_vocab_size(self):
            return 8

        def token_to_id(self, text):
            return 0

        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1, 2, 3] if text.startswith("### Instruction:") else [4])

    source = ROOT / "experiments/sft_diagnostic_examples.json"
    manifest = prepare(source, tmp_path / "data", tokenizer=Tokenizer())
    splits, identity = load_sft(tmp_path / "data", tokenizer=DOLMA_TOKENIZER, vocab_size=8, max_tokens=256)
    assert identity["contents"] == manifest
    assert set(manifest["files"]) == {"train", "val"}
    probes = json.loads((tmp_path / "data/train-probes.json").read_text())
    assert probes == build_examples()["train"]
    assert {row["id"] for row in probes} == {row["source_id"] for row in splits["train"]}
    assert not {row["id"] for row in probes} & {row["source_id"] for row in splits["val"]}
    # All custom task categories survive evaluation, in the original file order.
    assert select_evaluation_examples(splits["val"], 64, 0) == splits["val"]
    assert select_evaluation_examples(splits["train"], 32, 0) == splits["train"]
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "data", tokenizer=Tokenizer())


def test_preparation_rejects_length_overflow_before_creating_data(tmp_path):
    class Tokenizer:
        def get_vocab_size(self):
            return 8

        def token_to_id(self, text):
            return 0

        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1] * 256)

    with pytest.raises(ValueError, match="budget"):
        prepare(ROOT / "experiments/sft_diagnostic_examples.json", tmp_path / "data", tokenizer=Tokenizer())
    assert not (tmp_path / "data").exists()


def test_diversity_changes_partners_preserving_dev_and_answer_distributions():
    original = build_examples()
    diverse = build_examples((1, 4))
    assert diverse == json.loads((ROOT / "experiments/sft_diversity_examples.json").read_text())
    assert diverse["train"][:32] == original["train"]
    assert diverse["val"] == original["val"]
    assert len(diverse["train"]) == 64
    for category in ("copy", "color", "name"):
        train = [row for row in diverse["train"] if row["category"] == category]
        val = [row for row in diverse["val"] if row["category"] == category]
        assert Counter(row["response"] for row in train) == Counter(row["response"] for row in val)
        assert not {row["context"] for row in train} & {row["context"] for row in val}
        assert all(solve(row) == row["response"] for row in train + val)
    partners = {}
    for row in diverse["train"]:
        if row["category"] == "name":
            first, second = re.findall(r"(\w+) has the", row["context"])
            partners.setdefault(first, set()).add(second)
    assert len(partners) == 6 and all(len(values) == 2 for values in partners.values())
    for bad in ((1, 2), (1, 3), (1, 1), (4,)):
        with pytest.raises(ValueError, match="offsets"):
            build_examples(bad)


def test_diversity_preparation_keeps_dev_bytes_and_training_budget(tmp_path):
    class Tokenizer:
        def get_vocab_size(self):
            return 8

        def token_to_id(self, text):
            return 0

        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1, 2, 3] if text.startswith("### Instruction:") else [4])

    original = prepare(ROOT / "experiments/sft_diagnostic_examples.json", tmp_path / "original", tokenizer=Tokenizer())
    diverse = prepare(ROOT / "experiments/sft_diversity_examples.json", tmp_path / "diverse",
                      tokenizer=Tokenizer(), train_offsets=(1, 4))
    assert (tmp_path / "original/val.jsonl").read_bytes() == (tmp_path / "diverse/val.jsonl").read_bytes()
    rows, _ = load_sft(tmp_path / "diverse", tokenizer=DOLMA_TOKENIZER, vocab_size=8, max_tokens=256)
    assert len(rows["train"]) == len(rows["val"]) == 64
    old_config = json.loads((ROOT / "experiments/sft_diagnostic_config.json").read_text())
    new_config = json.loads((ROOT / "experiments/sft_diversity_config.json").read_text())
    permitted = {"name", "epochs", "corpus", "train_evaluation_examples", "diagnostic"}
    assert {k: v for k, v in old_config.items() if k not in permitted} == {k: v for k, v in new_config.items() if k not in permitted}
    assert old_config["corpus"]["val_examples"] == new_config["corpus"]["val_examples"] == 64
    assert old_config["epochs"] * 32 == new_config["epochs"] * 64 == 3200
    assert old_config["batch_size"] == new_config["batch_size"] == 8
    for key in ("response_tokens", "sequence_tokens"):
        assert original["files"]["train"][key] * old_config["epochs"] == diverse["files"]["train"][key] * new_config["epochs"]
