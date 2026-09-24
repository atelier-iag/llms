import json
from pathlib import Path

import numpy as np
import pytest

from reimplementation.corpus_manifest import validate_manifest
from reimplementation.prepare_corpus import file_hash, prepare_documents, write_manifest
from reimplementation.subset_corpus import prepare_subset


@pytest.fixture
def parent(tmp_path):
    def documents(source):
        for i in range(10_000):
            yield {"text": f"Domain {source}, document {i}. " + "An alphabetic scientific document. " * 5,
                   "locator": f"{source}:{i}"}

    output = tmp_path / "parent"
    prepare_documents({s: documents(s) for s in ("a", "b")}, output,
                      encode=lambda _: [1, 2, 3, 4], eos_id=0, vocab_size=6,
                      budgets={"train": 20, "val": 10, "holdout": 10},
                      provenance={"tokenizer": {"repo_id": "test", "revision": "pinned"}})
    return output


def test_balanced_subset_preserves_reserved_data_and_parent_without_retokenizing(parent, tmp_path):
    before = {file.name: file.read_bytes() for file in parent.iterdir()}
    out = tmp_path / "subset"
    manifest = prepare_subset(parent, out, train_tokens=16)
    assert manifest["status"] == "complete"
    assert prepare_subset(parent, tmp_path / "repeat", train_tokens=16) == manifest
    assert np.fromfile(out / "train.npy", dtype="<u4").tolist() == [1, 2, 3, 4, 0, 1, 2, 0] * 2
    for split in ("val", "holdout"):
        assert (out / f"{split}.npy").read_bytes() == before[f"{split}.npy"]
    for stats in manifest["sources"].values():
        assert stats["tokens"]["train"] == 8
        assert stats["documents"]["train"] == 2
        assert stats["newly_truncated_train_documents"] == 1
    parent_rows = [json.loads(line) for line in before["documents.jsonl"].decode().splitlines()]
    rows = [json.loads(line) for line in (out / "documents.jsonl").read_text().splitlines()]
    assert [r for r in rows if r["split"] != "train"] == [r for r in parent_rows if r["split"] != "train"]
    by_fingerprint = {r["fingerprint"]: r for r in parent_rows}
    assert len({r["fingerprint"] for r in rows}) == len(rows)
    for row in rows:
        if row["split"] == "train":
            original = by_fingerprint[row["fingerprint"]]
            assert row["parent_start_token"] == original["start_token"]
            assert row["parent_tokens"] == original["tokens"]
    assert {file.name: file.read_bytes() for file in parent.iterdir()} == before
    assert validate_manifest(out, tokenizer=manifest["provenance"]["tokenizer"], vocab_size=6, required=True)
    with pytest.raises(FileExistsError):
        prepare_subset(parent, out, train_tokens=16)


@pytest.mark.parametrize("file", ["train.npy", "holdout.npy", "documents.jsonl"])
def test_changed_parent_is_rejected_before_creating_subset(parent, tmp_path, file):
    with (parent / file).open("ab") as stream:
        stream.write(b"bad")
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        prepare_subset(parent, out, train_tokens=16)
    assert not out.exists()


def test_index_must_cover_actual_parent_file_even_with_valid_hash(parent, tmp_path):
    path = parent / "documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["start_token"] += 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = json.loads((parent / "manifest.json").read_text())
    manifest["document_index_sha256"] = file_hash(path)
    write_manifest(parent, manifest)
    with pytest.raises(ValueError, match="index coverage"):
        prepare_subset(parent, tmp_path / "bad", train_tokens=16)


@pytest.mark.parametrize("budget", [0, 15, 20, 24])
def test_invalid_subset_budget_is_rejected(parent, tmp_path, budget):
    with pytest.raises(ValueError):
        prepare_subset(parent, tmp_path / "bad", train_tokens=budget)
    assert not (tmp_path / "bad").exists()


def test_scaling_configuration_preserves_model_recipe_and_dev():
    root = Path(__file__).resolve().parents[1]
    full = json.loads((root / "experiments/pretraining_50m_config.json").read_text())
    subset = json.loads((root / "experiments/pretraining_25m_config.json").read_text())
    full.pop("name"); subset.pop("name")
    assert full["corpus"].pop("train_tokens") == 50_000_000
    assert subset["corpus"].pop("train_tokens") == 25_000_000
    assert full.pop("warmup_updates") == 250
    assert subset.pop("warmup_updates") == 125
    assert full == subset
