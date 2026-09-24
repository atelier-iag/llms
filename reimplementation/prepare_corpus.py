"""Prepare an immutable, document-disjoint train/dev/holdout corpus."""

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import unicodedata

import numpy as np
import requests
import zstandard
from huggingface_hub import HfApi, hf_hub_url

from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


REPO = "allenai/dolma3_pool"
REVISION = "6462556697df1a8f5c953727e9c686629ad98b68"
SOURCES = (
    "olmocr_science_pdfs-science_math_and_technology-part1",
    "olmocr_science_pdfs-education_and_jobs",
    "olmocr_science_pdfs-history_and_geography",
    "olmocr_science_pdfs-software_development",
)
SPLITS = ("train", "val", "holdout")  # val is the development split.


def fingerprint(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def document_split(digest, seed=0):
    # Fixed probabilities, independent of requested corpus size or source domain.
    value = hashlib.sha256(f"atelier-iag-v1:{seed}:{digest}".encode()).digest()
    bucket = int.from_bytes(value[:8], "big") % 10_000
    return "train" if bucket < 9600 else "val" if bucket < 9800 else "holdout"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(output, manifest):
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")


def prepare_documents(sources, output, *, encode, eos_id, vocab_size, budgets,
                      seed=0, provenance=None):
    """Consume ordered source iterables; finish only if every domain quota is met.

    Each document is a dict containing text and a stable locator. Encoders return
    IDs without special tokens. Output is a new directory, never an existing one.
    """
    if not sources or set(budgets) != set(SPLITS):
        raise ValueError("require sources and train/val/holdout budgets")
    if any(not isinstance(n, int) or n < len(sources) or n % len(sources) for n in budgets.values()):
        raise ValueError("positive token budgets must divide evenly across sources")
    if not 0 <= eos_id < vocab_size:
        raise ValueError("EOS must be in the vocabulary")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    quota = {split: count // len(sources) for split, count in budgets.items()}
    manifest = {
        "format_version": 1, "status": "building", "provenance": provenance or {},
        "seed": seed, "token_budget": budgets, "vocab_size": vocab_size, "eos_token_id": eos_id,
        "format": "raw little-endian uint32; .npy is a legacy filename, no NumPy header",
        "split_policy": "SHA-256 of normalized document text, salted with seed; 96/2/2 buckets",
        "deduplication": "NFKC + collapsed whitespace; exact normalized text, globally across sources",
        "filters": {"min_characters": 100, "max_characters": 2_000_000,
                    "min_alphabetic_fraction_of_nonspace": 0.2},
        "sources": {}, "files": {},
    }
    write_manifest(output, manifest)
    seen = set()
    totals = dict.fromkeys(SPLITS, 0)
    written = {split: set() for split in SPLITS}
    try:
        with ExitStack() as stack:
            streams = {split: stack.enter_context((output / f"{split}.npy").open("xb"))
                       for split in SPLITS}
            index = stack.enter_context((output / "documents.jsonl").open("x", encoding="utf-8"))
            for source, documents in sources.items():
                close = getattr(documents, "close", None)
                if close is not None:
                    stack.callback(close)
                stats = {"visited": 0, "filtered": 0, "duplicates": 0, "truncated": 0,
                         "tokens": dict.fromkeys(SPLITS, 0), "documents": dict.fromkeys(SPLITS, 0)}
                manifest["sources"][source] = stats
                for document in documents:
                    stats["visited"] += 1
                    if stats["visited"] % 1000 == 0:
                        print(json.dumps({"source": source, "visited": stats["visited"],
                                          "tokens": stats["tokens"]}), flush=True)
                    text = document.get("text")
                    if not isinstance(text, str) or not 100 <= len(text) <= 2_000_000:
                        stats["filtered"] += 1
                        continue
                    nonspace = sum(not c.isspace() for c in text)
                    if not nonspace or sum(c.isalpha() for c in text) / nonspace < 0.2:
                        stats["filtered"] += 1
                        continue
                    digest = fingerprint(text)
                    if digest in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(digest)
                    split = document_split(digest, seed)
                    remaining = quota[split] - stats["tokens"][split]
                    if not remaining:
                        continue
                    ids = list(encode(text)) + [eos_id]
                    if min(ids) < 0 or max(ids) >= vocab_size:
                        raise ValueError("encoder produced a token outside the vocabulary")
                    original_count = len(ids)
                    ids = ids[:remaining]
                    ids[-1] = eos_id  # Even a quota-truncated final document ends with EOS.
                    np.asarray(ids, dtype="<u4").tofile(streams[split])
                    index.write(json.dumps({"source": source, "locator": document["locator"],
                                            "fingerprint": digest, "split": split,
                                            "start_token": totals[split], "tokens": len(ids),
                                            "original_tokens": original_count,
                                            "truncated": len(ids) < original_count}) + "\n")
                    written[split].add(digest)
                    totals[split] += len(ids)
                    stats["tokens"][split] += len(ids)
                    stats["documents"][split] += 1
                    stats["truncated"] += len(ids) < original_count
                    if stats["tokens"] == quota:
                        break
                close = getattr(documents, "close", None)
                if close is not None:
                    close()
                if stats["tokens"] != quota:
                    raise ValueError(f"source exhausted before all quotas were filled: {source}")
                write_manifest(output, manifest)
                print(json.dumps({"source_complete": source, "tokens": stats["tokens"],
                                  "visited": stats["visited"], "duplicates": stats["duplicates"]}), flush=True)
        if any(written[a] & written[b] for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout"))):
            raise RuntimeError("document leakage between splits")
        if totals != budgets:
            raise RuntimeError("token budgets do not match")
        manifest["files"] = {split: {"path": f"{split}.npy", "tokens": totals[split],
                                      "documents": len(written[split]),
                                      "sha256": file_hash(output / f"{split}.npy")}
                             for split in SPLITS}
        manifest["document_index_sha256"] = file_hash(output / "documents.jsonl")
        manifest["normalized_document_overlap"] = 0
        manifest["status"] = "complete"
        write_manifest(output, manifest)
        return manifest
    except BaseException as error:
        manifest["status"] = "incomplete"
        manifest["error_type"] = type(error).__name__
        write_manifest(output, manifest)
        raise


def stream_source(source):
    api = HfApi()
    shards = sorted(entry.path for entry in api.list_repo_tree(
        repo_id=REPO, repo_type="dataset", revision=REVISION,
        path_in_repo=f"data/{source}", recursive=False)
        if entry.path.endswith(".jsonl.zst"))
    for shard in shards:
        print(json.dumps({"streaming_shard": shard, "revision": REVISION}), flush=True)
        url = hf_hub_url(repo_id=REPO, repo_type="dataset", revision=REVISION, filename=shard)
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with zstandard.ZstdDecompressor().stream_reader(response.raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as lines:
                    for number, line in enumerate(lines, 1):
                        try:
                            document = json.loads(line)
                        except json.JSONDecodeError:
                            document = {}
                        yield {"text": document.get("text") if isinstance(document, dict) else None,
                               "locator": f"{shard}:{number}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--dev-tokens", type=int, default=1_000_000)
    parser.add_argument("--holdout-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must not already exist")
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        parser.error("tokenizer has no expected EOS token")
    manifest = prepare_documents(
        {source: stream_source(source) for source in SOURCES}, args.output_dir,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        eos_id=eos, vocab_size=tokenizer.get_vocab_size(),
        budgets={"train": args.train_tokens, "val": args.dev_tokens, "holdout": args.holdout_tokens},
        seed=args.seed, provenance={"dataset": {"repo_id": REPO, "revision": REVISION},
                                    "tokenizer": DOLMA_TOKENIZER},
    )
    print(json.dumps({"complete": str(args.output_dir), "files": manifest["files"]}), flush=True)


if __name__ == "__main__":
    main()
