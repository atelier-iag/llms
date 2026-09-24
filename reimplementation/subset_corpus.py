"""Derive a smaller balanced training corpus while freezing dev and holdout."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from reimplementation.corpus_manifest import validate_manifest
from reimplementation.prepare_corpus import SPLITS, file_hash, write_manifest


def prepare_subset(parent_dir, output_dir, *, train_tokens):
    parent_dir, output = Path(parent_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    parent_hash = file_hash(parent_dir / "manifest.json")
    parent = json.loads((parent_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(parent_dir, tokenizer=parent["provenance"]["tokenizer"],
                      vocab_size=parent["vocab_size"], required=True)
    sources = list(parent["sources"])
    if (not sources or not isinstance(train_tokens, int) or train_tokens < len(sources)
            or train_tokens % len(sources)):
        raise ValueError("training budget must be positive and divisible by the source count")
    quota = train_tokens // len(sources)
    if train_tokens >= parent["files"]["train"]["tokens"] or any(
        quota > parent["sources"][source]["tokens"]["train"] for source in sources
    ):
        raise ValueError("subset must be smaller and fit every source's training budget")
    for split in SPLITS:
        entry = parent["files"][split]
        path = parent_dir / f"{split}.npy"
        if (entry["path"] != path.name or path.stat().st_size != 4 * entry["tokens"]
                or file_hash(path) != entry["sha256"]):
            raise ValueError(f"parent {split} file differs from its manifest")
    index_path = parent_dir / "documents.jsonl"
    if file_hash(index_path) != parent["document_index_sha256"]:
        raise ValueError("parent document index hash differs")
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    # Check that the index covers each file once and agrees with domain quotas.
    offsets = dict.fromkeys(SPLITS, 0)
    counts = {source: dict.fromkeys(SPLITS, 0) for source in sources}
    seen = set()
    for row in rows:
        split, source = row["split"], row["source"]
        if (split not in SPLITS or source not in sources or row["tokens"] < 1
                or row["start_token"] != offsets[split] or row["fingerprint"] in seen):
            raise ValueError("invalid parent document index coverage or duplicate fingerprint")
        seen.add(row["fingerprint"])
        offsets[split] += row["tokens"]
        counts[source][split] += row["tokens"]
    if (any(offsets[split] != parent["files"][split]["tokens"] for split in SPLITS)
            or any(counts[source] != parent["sources"][source]["tokens"] for source in sources)):
        raise ValueError("parent index token counts differ from the manifest")
    train = np.memmap(parent_dir / "train.npy", mode="r", dtype="<u4")
    if int(train.max()) >= parent["vocab_size"]:
        raise ValueError("parent training token outside vocabulary")
    if any(train[row["start_token"] + row["tokens"] - 1] != parent["eos_token_id"]
           for row in rows if row["split"] == "train"):
        raise ValueError("parent training document must end in EOS")

    manifest = copy.deepcopy(parent)
    manifest.update(status="building", files={}, document_index_sha256=None)
    manifest["token_budget"]["train"] = train_tokens
    manifest["provenance"]["parent_corpus"] = {
        "manifest_sha256": parent_hash,
        "document_index_sha256": parent["document_index_sha256"],
        "files": parent["files"],
    }
    manifest["subset_policy"] = (
        "First training documents in parent order, equal token quota per source; "
        "truncate the final document if needed and retain its EOS. "
        "Dev and holdout copied byte-for-byte; no retokenization or new split assignment."
    )
    manifest["sources"] = {
        source: {"tokens": dict.fromkeys(SPLITS, 0), "documents": dict.fromkeys(SPLITS, 0),
                 "newly_truncated_train_documents": 0} for source in sources
    }
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output, manifest)
    try:
        for split in ("val", "holdout"):
            with (parent_dir / f"{split}.npy").open("rb") as src, (output / f"{split}.npy").open("xb") as dst:
                shutil.copyfileobj(src, dst)
        position = 0
        with (output / "train.npy").open("xb") as dst, (output / "documents.jsonl").open("x", encoding="utf-8") as index:
            for original in rows:
                row = dict(original)
                split, source = row["split"], row["source"]
                stats = manifest["sources"][source]
                if split == "train":
                    remaining = quota - stats["tokens"]["train"]
                    if remaining == 0:
                        continue
                    count = min(row["tokens"], remaining)
                    start = row["start_token"]
                    tokens = np.array(train[start:start + count], dtype="<u4")
                    # Reuse the parent document's existing EOS at a new cut point.
                    tokens[-1] = train[start + row["tokens"] - 1]
                    tokens.tofile(dst)
                    newly_truncated = count < row["tokens"]
                    row.update(parent_start_token=start, parent_tokens=row["tokens"],
                               start_token=position, tokens=count,
                               truncated=row["truncated"] or newly_truncated)
                    position += count
                    stats["newly_truncated_train_documents"] += newly_truncated
                stats["tokens"][split] += row["tokens"]
                stats["documents"][split] += 1
                index.write(json.dumps(row) + "\n")
        if position != train_tokens or any(
            stats["tokens"]["train"] != quota for stats in manifest["sources"].values()
        ):
            raise RuntimeError("subset did not meet its domain quotas")
        manifest["files"] = {
            split: {"path": f"{split}.npy",
                    "tokens": sum(s["tokens"][split] for s in manifest["sources"].values()),
                    "documents": sum(s["documents"][split] for s in manifest["sources"].values()),
                    "sha256": file_hash(output / f"{split}.npy")}
            for split in SPLITS
        }
        for split in ("val", "holdout"):
            if manifest["files"][split] != parent["files"][split]:
                raise RuntimeError(f"reserved {split} split changed")
        if (file_hash(parent_dir / "manifest.json") != parent_hash
                or file_hash(index_path) != parent["document_index_sha256"]
                or any(file_hash(parent_dir / f"{split}.npy") != parent["files"][split]["sha256"]
                       for split in SPLITS)):
            raise RuntimeError("parent corpus changed during extraction")
        manifest["document_index_sha256"] = file_hash(output / "documents.jsonl")
        manifest["status"] = "complete"
        write_manifest(output, manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="incomplete", error_type=type(error).__name__)
        write_manifest(output, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=25_000_000)
    args = parser.parse_args()
    manifest = prepare_subset(args.parent_dir, args.output_dir, train_tokens=args.train_tokens)
    print(json.dumps({"complete": str(args.output_dir), "files": manifest["files"]}), flush=True)


if __name__ == "__main__":
    main()
