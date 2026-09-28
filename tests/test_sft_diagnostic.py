from collections import Counter
import copy
import json
from pathlib import Path
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
