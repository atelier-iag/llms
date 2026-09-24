import json

import numpy as np
import pytest

from reimplementation.corpus_manifest import validate_manifest
from reimplementation.prepare_corpus import document_split, fingerprint, prepare_documents


TOKENIZER = {"repo_id": "test", "revision": "fixed"}


def documents():
    # Find distinct documents for every hash bucket; do not mock the split policy.
    found = {split: [] for split in ("train", "val", "holdout")}
    index = 0
    while any(len(rows) < 3 for rows in found.values()):
        text = f"Document {index}. " + "A scientific text with enough alphabetic content. " * 4
        split = document_split(fingerprint(text))
        if len(found[split]) < 3:
            found[split].append({"text": text, "locator": str(index)})
        index += 1
    return found


def build(output):
    found = documents()
    first = found["train"][0]
    duplicate = {**first, "text": first["text"].replace(" ", " \n ")}
    return prepare_documents(
        {"one": [first, duplicate, {"text": "short", "locator": "bad"}]
                 + [rows[0] for rows in found.values()],
         "two": [first] + [rows[2] for rows in found.values()]},
        output, encode=lambda text: [1, 2, 3, 4], eos_id=0, vocab_size=6,
        budgets={"train": 8, "val": 8, "holdout": 8},
        provenance={"tokenizer": TOKENIZER},
    )


def test_exact_budgets_deduplication_disjointness_and_reproducibility(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    manifest = build(one)
    assert build(two) == manifest
    assert manifest["status"] == "complete"
    assert manifest["sources"]["one"]["duplicates"] >= 2
    assert manifest["sources"]["one"]["filtered"] == 1
    rows = [json.loads(line) for line in (one / "documents.jsonl").read_text().splitlines()]
    assert len({row["fingerprint"] for row in rows}) == len(rows)
    for split in ("train", "val", "holdout"):
        data = np.fromfile(one / f"{split}.npy", dtype="<u4")
        assert len(data) == 8
        assert (one / f"{split}.npy").read_bytes() == (two / f"{split}.npy").read_bytes()
        offsets = [row["start_token"] for row in rows if row["split"] == split]
        assert offsets == [0, 4]
        assert data.tolist() == [1, 2, 3, 0] * 2
    before = (one / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build(one)
    assert (one / "manifest.json").read_bytes() == before


def test_exhausted_source_leaves_unusable_manifest(tmp_path):
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_documents({"empty": []}, output, encode=lambda _: [], eos_id=0,
                          vocab_size=6, budgets={"train": 4, "val": 4, "holdout": 4})
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
    with pytest.raises(ValueError, match="complete"):
        validate_manifest(output, tokenizer=TOKENIZER, vocab_size=6)


def test_manifest_verifies_training_data_without_reading_holdout(tmp_path):
    build(tmp_path / "data")
    data = tmp_path / "data"
    (data / "holdout.npy").unlink()
    assert validate_manifest(data, tokenizer=TOKENIZER, vocab_size=6)["sha256"]
    with pytest.raises(ValueError, match="tokenizer/vocabulary"):
        validate_manifest(data, tokenizer={"repo_id": "wrong"}, vocab_size=6)
    tokens = np.fromfile(data / "train.npy", dtype="<u4")
    tokens[0] = 5
    tokens.tofile(data / "train.npy")
    with pytest.raises(ValueError, match="hash differs"):
        validate_manifest(data, tokenizer=TOKENIZER, vocab_size=6)


def test_required_manifest_cannot_fall_back_to_legacy_data(tmp_path):
    assert validate_manifest(tmp_path, tokenizer=TOKENIZER, vocab_size=6) is None
    with pytest.raises(ValueError, match="requires a corpus manifest"):
        validate_manifest(tmp_path, tokenizer=TOKENIZER, vocab_size=6, required=True)
