import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.final_holdout import load_lm_holdout, load_sft_holdout, verify_control
from reimplementation.prepare_corpus import file_hash
from reimplementation.sft_data import FORMAT, batches, IGNORE_INDEX
from reimplementation.tokenizer import DOLMA_TOKENIZER


def write_manifest(directory, manifest, kind):
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return {"directory": str(directory), "kind": kind, "manifest_sha256": file_hash(path),
            "holdout": copy.deepcopy(manifest["files"]["holdout"])}


def lm_fixture(directory):
    files = {}
    for split, values in (("train", [1, 2, 3, 4, 5]), ("val", [2, 3, 4, 5, 6]),
                          ("holdout", [1, 3, 2, 5, 4, 6, 3, 7, 2, 4, 1])):
        path = directory / (split + ".npy")
        np.array(values, dtype=np.uint32).tofile(path)
        files[split] = {"path": path.name, "tokens": len(values), "sha256": file_hash(path)}
    manifest = {"format_version": 1, "status": "complete", "vocab_size": 8,
                "provenance": {"tokenizer": DOLMA_TOKENIZER}, "files": files}
    return manifest, write_manifest(directory, manifest, "lm")


def sft_fixture(directory):
    files = {}
    for index, split in enumerate(("train", "val", "holdout")):
        row = {"ids": [1, 2 + index, 0], "prompt_tokens": 1, "source_id": index,
               "category": "classification", "group": split, "fingerprint": split + "-fingerprint"}
        path = directory / (split + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        files[split] = {"path": path.name, "sha256": file_hash(path), "examples": 1, "groups": 1,
                        "categories": {"classification": 1}, "sequence_tokens": 3,
                        "response_tokens": 2, "max_sequence_tokens": 3}
    manifest = {"format_version": 1, "status": "complete", "kind": "sft", "prompt_format": FORMAT,
                "vocab_size": 8, "eos_token_id": 0, "max_tokens": 8, "max_answer_tokens": 2,
                "provenance": {"tokenizer": DOLMA_TOKENIZER}, "files": files}
    return manifest, write_manifest(directory, manifest, "sft")


def test_lm_holdout_covers_every_target_once_including_tail(tmp_path):
    _, spec = lm_fixture(tmp_path)
    corpus = load_lm_holdout(spec, context_length=4, vocab_size=8)
    targets = [token for batch in corpus.epoch_batches(2) for row in batch.tolist() for token in row[1:]]
    assert targets == [3, 2, 5, 4, 6, 3, 7, 2, 4, 1]
    assert file_hash(tmp_path / "holdout.npy") == spec["holdout"]["sha256"]


@pytest.mark.parametrize("mutation", ["hash", "count", "split", "alias"])
def test_lm_holdout_rejects_identity_size_and_split_errors(tmp_path, mutation):
    manifest, spec = lm_fixture(tmp_path)
    if mutation == "hash":
        (tmp_path / "holdout.npy").write_bytes(b"changed")
    elif mutation == "count":
        manifest["files"]["holdout"]["tokens"] += 1
        spec = write_manifest(tmp_path, manifest, "lm")
    elif mutation == "split":
        manifest["files"]["holdout"]["path"] = "val.npy"
        spec = write_manifest(tmp_path, manifest, "lm")
    else:
        (tmp_path / "holdout.npy").unlink()
        (tmp_path / "holdout.npy").hardlink_to(tmp_path / "train.npy")
    with pytest.raises(ValueError):
        load_lm_holdout(spec, context_length=4, vocab_size=8)


def test_sft_holdout_supervises_answer_and_eos_only(tmp_path):
    _, spec = sft_fixture(tmp_path)
    rows, eos = load_sft_holdout(spec, context_length=8, vocab_size=8)
    assert rows[0]["source_id"] == 2 and eos == 0
    _, labels = next(batches(rows, 8, eos))
    assert labels[labels != IGNORE_INDEX].tolist() == [4, 0]


@pytest.mark.parametrize("mutation", ["group_leak", "id_leak", "eos", "token", "count"])
def test_sft_holdout_rejects_leakage_bad_tokens_and_counts(tmp_path, mutation):
    manifest, _ = sft_fixture(tmp_path)
    path = tmp_path / "holdout.jsonl"
    row = json.loads(path.read_text())
    if mutation == "group_leak":
        row["group"] = "train"
    elif mutation == "id_leak":
        row["source_id"] = 0
    elif mutation == "eos":
        row["ids"][-1] = 7
    elif mutation == "token":
        row["ids"][1] = 8
    else:
        manifest["files"]["holdout"]["response_tokens"] += 1
    path.write_text(json.dumps(row) + "\n")
    manifest["files"]["holdout"]["sha256"] = file_hash(path)
    spec = write_manifest(tmp_path, manifest, "sft")
    with pytest.raises(ValueError):
        load_sft_holdout(spec, context_length=8, vocab_size=8)


def test_dev_preflight_rejects_changed_score_coverage_or_nonfinite_loss():
    expected = {"loss": 5.2, "scored_tokens": 32768}
    verify_control(dict(expected), expected, 1e-6)
    verify_control({**expected, "loss": 5.2 + 1e-7}, expected, 1e-6)
    for measured in ({**expected, "loss": 5.3}, {**expected, "scored_tokens": 42},
                     {**expected, "loss": float("nan")}):
        with pytest.raises(ValueError):
            verify_control(measured, expected, 1e-6)
