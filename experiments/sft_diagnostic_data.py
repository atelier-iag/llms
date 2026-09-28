"""Build 32 training and 64 novel combinations for a controlled SFT diagnostic."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from reimplementation.prepare_corpus import file_hash, write_manifest
from reimplementation.prepare_sft import digest, normalize
from reimplementation.sft_data import FORMAT, encode_example, format_prompt
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


WORDS = ("apple", "river", "stone", "cloud", "bread", "chair", "music", "paper")
COLORS = ("red", "blue", "green", "yellow", "black", "white")
NAMES = ("Alice", "Bruno", "Clara", "David", "Emma", "Felix")
COUNTS = {"train": {"copy": 8, "color": 12, "name": 12},
          "val": {"copy": 16, "color": 24, "name": 24}}


def build_examples():
    splits = {}
    for split, offsets in (("train", (1,)), ("val", (2, 3))):
        rows = []

        def add(category, instruction, context, response):
            rows.append({"id": f"{split}-{category}-{len(rows) + 1:02d}", "category": category,
                         "instruction": instruction, "context": context, "response": response})

        for offset in offsets:
            for i, word in enumerate(WORDS):
                lines = [f"Target word: {word}", f"Other word: {WORDS[(i + offset) % len(WORDS)]}"]
                if i % 2:
                    lines.reverse()
                add("copy", "Copy the target word. Answer with one word.", "\n".join(lines), word)
            for i, color in enumerate(COLORS):
                other = COLORS[(i + offset) % len(COLORS)]
                context = f"The lantern is {color}. The basket is {other}."
                for obj, answer in (("lantern", color), ("basket", other)):
                    add("color", f"Which color is the {obj}? Reply with one color.", context, answer)
            for i, name in enumerate(NAMES):
                other = NAMES[(i + offset) % len(NAMES)]
                context = f"{name} has the key. {other} has the map."
                for obj, answer in (("key", name), ("map", other)):
                    add("name", f"Who has the {obj}? Reply with one name.", context, answer)
        splits[split] = rows
    validate_examples(splits)
    return splits


def solve(row):
    """Independent text-based answer oracle; it does not read the supplied label."""
    instruction, context = row["instruction"], row["context"]
    if row["category"] == "copy":
        if instruction != "Copy the target word. Answer with one word.":
            raise ValueError("unexpected copy instruction")
        pairs = re.findall(r"^(Target|Other) word: (\w+)$", context, re.MULTILINE)
        if len(pairs) != 2 or {k for k, _ in pairs} != {"Target", "Other"}:
            raise ValueError("ambiguous copy context")
        return dict(pairs)["Target"]
    if row["category"] == "color":
        match = re.fullmatch(r"Which color is the (lantern|basket)\? Reply with one color\.", instruction)
        pairs = re.findall(r"The (lantern|basket) is (\w+)\.", context)
    elif row["category"] == "name":
        match = re.fullmatch(r"Who has the (key|map)\? Reply with one name\.", instruction)
        pairs = [(obj, name) for name, obj in re.findall(r"(\w+) has the (key|map)\.", context)]
    else:
        raise ValueError("unknown diagnostic task")
    if match is None or len(pairs) != 2 or len(dict(pairs)) != 2 or pairs[0][1] == pairs[1][1]:
        raise ValueError("ambiguous extraction example")
    return dict(pairs)[match[1]]


def validate_examples(splits):
    if set(splits) != set(COUNTS):
        raise ValueError("expected train and val only")
    prompts, ids = set(), set()
    for split, counts in COUNTS.items():
        if Counter(row["category"] for row in splits[split]) != counts:
            raise ValueError("unexpected diagnostic task counts")
        for row in splits[split]:
            if row["response"] != solve(row):
                raise ValueError("diagnostic label differs from the text oracle")
            prompt = normalize(format_prompt(row["instruction"], row["context"]))
            if prompt in prompts or row["id"] in ids:
                raise ValueError("duplicate diagnostic prompt or ID")
            prompts.add(prompt)
            ids.add(row["id"])
    for category in COUNTS["train"]:
        train = Counter(row["response"] for row in splits["train"] if row["category"] == category)
        val = Counter(row["response"] for row in splits["val"] if row["category"] == category)
        if val != {answer: 2 * n for answer, n in train.items()}:
            raise ValueError("answer distributions must match across splits")


def prepare(source, output, *, tokenizer, max_tokens=256):
    source, output = Path(source), Path(output)
    splits = json.loads(source.read_text(encoding="utf-8"))
    validate_examples(splits)
    eos = tokenizer.token_to_id("<|endoftext|>")
    encoded = {}
    for split, rows in splits.items():
        encoded[split] = []
        for row in rows:
            example = encode_example(tokenizer, row["instruction"], row["context"], row["response"], eos)
            if len(example["ids"]) > max_tokens or len(example["ids"]) - example["prompt_tokens"] > 8:
                raise ValueError("diagnostic sequence or answer exceeds the fixed budget")
            prompt = normalize(format_prompt(row["instruction"], row["context"]))
            encoded[split].append({**example, "source_id": row["id"], "category": row["category"],
                                   "group": digest(prompt), "fingerprint": digest(prompt + "\n" + row["response"])})
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format_version": 1, "kind": "sft", "status": "building", "prompt_format": FORMAT,
                "max_tokens": max_tokens, "vocab_size": tokenizer.get_vocab_size(), "eos_token_id": eos,
                "provenance": {"tokenizer": DOLMA_TOKENIZER, "source_sha256": file_hash(source),
                               "source": "Self-authored deterministic copy/color/name combinations; no external dataset"},
                "grouping": "Exact full prompts are disjoint; templates and answer vocabulary deliberately shared",
                "split_policy": "Pair offset 1 for train, offsets 2 and 3 for val; no random split",
                "purpose": "Memorization and generalization to novel combinations of known values; not an AGI benchmark or reserved holdout",
                "files": {}}
    write_manifest(output, manifest)
    for split, rows in encoded.items():
        path = output / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        manifest["files"][split] = {"path": path.name, "sha256": file_hash(path), "examples": len(rows),
                                    "categories": dict(Counter(row["category"] for row in rows)),
                                    "groups": len({row["group"] for row in rows}),
                                    "sequence_tokens": sum(len(row["ids"]) for row in rows),
                                    "response_tokens": sum(len(row["ids"]) - row["prompt_tokens"] for row in rows),
                                    "max_sequence_tokens": max(len(row["ids"]) for row in rows)}
    # train_sft's probes slot measures memorization here, deliberately on TRAIN prompts.
    probe_path = output / "train-probes.json"
    probe_path.write_text(json.dumps(splits["train"], indent=2) + "\n", encoding="utf-8")
    manifest["train_probes"] = {"path": probe_path.name, "sha256": file_hash(probe_path), "role": "training-set memorization"}
    manifest["status"] = "complete"
    write_manifest(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("sft_diagnostic_examples.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir, tokenizer=load_tokenizer(DOLMA_TOKENIZER))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
