"""Pin, filter, group and split short Dolly instruction/response examples."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from huggingface_hub import hf_hub_download

from reimplementation.prepare_corpus import file_hash, write_manifest
from reimplementation.sft_data import FORMAT, encode_example
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


REPO = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_FILE = "databricks-dolly-15k.jsonl"
CATEGORIES = ("classification", "closed_qa", "information_extraction", "summarization")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_examples(rows, threshold=0.8):
    """Connected components: identical instruction/context or similar prompt.

    Similarity is exact Jaccard of word-trigram sets, using an inverted index
    to count shared shingles. It catches lexical variants, not semantic copies.
    """
    parents = list(range(len(rows)))

    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    exact = {}
    index = defaultdict(list)
    sizes = []
    for i, row in enumerate(rows):
        for kind in ("instruction", "context"):
            value = normalize(row[kind])
            if value:
                key = (kind, value)
                if key in exact:
                    union(i, exact[key])
                exact[key] = i
        words = re.findall(r"\w+", normalize(row["instruction"] + " " + row["context"]))
        shingles = {tuple(words[j:j + 3]) for j in range(len(words) - 2)}
        sizes.append(len(shingles))
        overlaps = Counter(j for shingle in shingles for j in index[shingle])
        for j, shared in overlaps.items():
            if shared / (len(shingles) + sizes[j] - shared) >= threshold:
                union(i, j)
        for shingle in shingles:
            index[shingle].append(i)
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row["fingerprint"])
    keys = {root: digest("\n".join(sorted(values))) for root, values in members.items()}
    return [keys[find(i)] for i in range(len(rows))]


def split_group(group, seed):
    bucket = int(digest(f"sft-dolly-v1:{seed}:{group}")[:16], 16) % 10_000
    return "train" if bucket < 8000 else "val" if bucket < 9000 else "holdout"


def balanced_select(rows, count, seed):
    queues = {category: sorted((row for row in rows if row["category"] == category),
                              key=lambda row: digest(f"{seed}:{row['fingerprint']}"))
              for category in CATEGORIES}
    positions = dict.fromkeys(CATEGORIES, 0)
    selected = []
    while len(selected) < count:
        before = len(selected)
        for category in CATEGORIES:
            if positions[category] < len(queues[category]) and len(selected) < count:
                selected.append(queues[category][positions[category]])
                positions[category] += 1
        if len(selected) == before:
            raise ValueError(f"not enough filtered examples: need {count}, found {len(selected)}")
    return selected


def prepare(rows, output, *, tokenizer, budgets, seed=0, max_tokens=256,
            max_answer_tokens=80, excluded_ids=(), provenance=None):
    if set(budgets) != {"train", "val", "holdout"} or any(type(n) is not int or n < 1 for n in budgets.values()):
        raise ValueError("require positive train/dev/holdout example counts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    eos = tokenizer.token_to_id("<|endoftext|>")
    manifest = {"format_version": 1, "kind": "sft", "status": "building", "seed": seed,
                "prompt_format": FORMAT, "max_tokens": max_tokens, "max_answer_tokens": max_answer_tokens,
                "vocab_size": tokenizer.get_vocab_size(), "eos_token_id": eos,
                "provenance": {"tokenizer": DOLMA_TOKENIZER, **(provenance or {})},
                "license": "CC-BY-SA-3.0; attribution: Databricks and contributing Wikipedia authors",
                "grouping": "same normalized instruction/context or prompt word-trigram Jaccard >= 0.8; transitive components",
                "split_policy": "group hash salted by seed; 80/10/10 buckets; category-balanced deterministic selection",
                "excluded_source_ids": sorted(excluded_ids), "files": {}, "statistics": {}}
    write_manifest(output, manifest)
    stats = Counter()
    seen = set()
    eligible = []
    try:
        for number, raw in enumerate(rows, 1):
            stats["visited"] += 1
            if number in excluded_ids:
                stats["quality_excluded"] += 1
                continue
            if (raw.get("category") not in CATEGORIES
                    or any(not isinstance(raw.get(key), str) for key in ("instruction", "context", "response"))
                    or not raw["instruction"].strip() or not raw["response"].strip()):
                stats["category_or_empty_filtered"] += 1
                continue
            row = {key: raw[key].strip() for key in ("instruction", "context", "response", "category")}
            if any("<|endoftext|>" in row[key] for key in ("instruction", "context", "response")):
                stats["special_token_filtered"] += 1
                continue
            encoded = encode_example(tokenizer, row["instruction"], row["context"], row["response"], eos)
            answer_tokens = len(encoded["ids"]) - encoded["prompt_tokens"] - 1
            if not 1 <= answer_tokens <= max_answer_tokens or len(encoded["ids"]) > max_tokens:
                stats["length_filtered"] += 1
                continue
            key = digest(json.dumps([normalize(row[k]) for k in ("instruction", "context", "response")]))
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            eligible.append({**row, **encoded, "fingerprint": key, "source_id": number})
        groups = group_examples(eligible)
        pools = defaultdict(list)
        for row, group in zip(eligible, groups):
            row["group"] = group
            pools[split_group(group, seed)].append(row)
        stats["eligible"] = len(eligible)
        stats["groups"] = len(set(groups))
        manifest["statistics"] = dict(stats)
        manifest["available_examples"] = {split: len(pools[split]) for split in budgets}
        manifest["available_categories"] = {split: dict(Counter(row["category"] for row in pools[split])) for split in budgets}
        for split, count in budgets.items():
            chosen = balanced_select(pools[split], count, seed)
            path = output / f"{split}.jsonl"
            # Raw texts remain in the pinned source cache; token files stay local.
            with path.open("x", encoding="utf-8") as stream:
                for row in chosen:
                    saved = {key: row[key] for key in ("source_id", "category", "fingerprint", "group", "ids", "prompt_tokens")}
                    stream.write(json.dumps(saved) + "\n")
            manifest["files"][split] = {"path": path.name, "examples": count, "sha256": file_hash(path),
                                        "groups": len({row["group"] for row in chosen}),
                                        "categories": dict(Counter(row["category"] for row in chosen)),
                                        "sequence_tokens": sum(len(row["ids"]) for row in chosen),
                                        "response_tokens": sum(len(row["ids"]) - row["prompt_tokens"] for row in chosen),
                                        "max_sequence_tokens": max(len(row["ids"]) for row in chosen)}
        manifest["status"] = "complete"
        write_manifest(output, manifest)
        return manifest
    except BaseException as error:
        manifest["status"] = "incomplete"
        manifest["error_type"] = type(error).__name__
        write_manifest(output, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-review", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=3000)
    parser.add_argument("--dev-examples", type=int, default=300)
    parser.add_argument("--holdout-examples", type=int, default=300)
    args = parser.parse_args()
    source = Path(hf_hub_download(REPO, SOURCE_FILE, repo_type="dataset", revision=REVISION))
    review = json.loads(args.quality_review.read_text(encoding="utf-8"))
    if review["source_sha256"] != file_hash(source):
        parser.error("quality review must refer to this pinned source file")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    manifest = prepare(rows, args.output_dir, tokenizer=load_tokenizer(DOLMA_TOKENIZER),
                       budgets={"train": args.train_examples, "val": args.dev_examples, "holdout": args.holdout_examples},
                       excluded_ids=review["excluded_source_ids"],
                       provenance={"dataset": {"repo_id": REPO, "revision": REVISION, "file": SOURCE_FILE,
                                                "sha256": file_hash(source)},
                                   "quality_review_sha256": file_hash(args.quality_review)})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
