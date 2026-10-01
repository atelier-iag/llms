"""Paired transfer probes: change instructions, sentence order, or SFT values."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from experiments.sft_diagnostic_data import build_examples as original_examples
from reimplementation.prepare_corpus import file_hash
from reimplementation.sft_data import encode_example, format_prompt
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


CONDITIONS = ("control", "rephrase", "reorder", "new_values")
COUNTS = {"copy": 16, "color": 24, "name": 24}
VALUE_MAPS = {
    "copy": dict(zip(
        ("apple", "river", "stone", "cloud", "bread", "chair", "music", "paper"),
        ("table", "window", "garden", "forest", "water", "train", "flower", "planet"))),
    "color": dict(zip(
        ("red", "blue", "green", "yellow", "black", "white"),
        ("purple", "orange", "pink", "brown", "gray", "cyan"))),
    "name": dict(zip(
        ("Alice", "Bruno", "Clara", "David", "Emma", "Felix"),
        ("Grace", "Henry", "Irene", "Jack", "Karen", "Louis"))),
}


def rephrase(row):
    instruction = row["instruction"]
    if row["category"] == "copy":
        return "Return the target word. Answer with one word."
    if row["category"] == "color":
        return instruction.replace("Which color is the ", "What is the color of the ")
    return instruction.replace("Who has the ", "Which person has the ")


def reverse_context(row):
    if row["category"] == "copy":
        return "\n".join(reversed(row["context"].split("\n")))
    return ". ".join(reversed(row["context"][:-1].split(". "))) + "."


def replace_values(text, mapping):
    return re.sub(r"\b\w+\b", lambda match: mapping.get(match[0], match[0]), text)


def build_examples(source):
    if source != original_examples((1, 4)):
        raise ValueError("expected the frozen 64/64 SFT source")
    groups = {}
    for condition in CONDITIONS:
        rows = []
        for original in source["val"]:
            row = {**original, "id": f"{condition}-{original['id']}",
                   "base_id": original["id"], "condition": condition}
            if condition == "rephrase":
                row["instruction"] = rephrase(original)
            elif condition == "reorder":
                row["context"] = reverse_context(original)
            elif condition == "new_values":
                mapping = VALUE_MAPS[row["category"]]
                row["context"] = replace_values(original["context"], mapping)
                row["response"] = mapping[original["response"]]
            rows.append(row)
        groups[condition] = rows
    return groups


def solve(row):
    """Read the answer from the text; do not use response or base_id."""
    instruction, context = row["instruction"], row["context"]
    if row["category"] == "copy":
        if not re.fullmatch(r"(?:Copy|Return) the target word\. Answer with one word\.", instruction):
            raise ValueError("unsupported copy instruction")
        match = re.fullmatch(r"(Target|Other) word: (\w+)\n(Target|Other) word: (\w+)", context)
        if match is None or match[1] == match[3] or match[2] == match[4]:
            raise ValueError("ambiguous copy context")
        return {match[1]: match[2], match[3]: match[4]}["Target"]
    if row["category"] == "color":
        query = re.fullmatch(r"(?:Which color is the |What is the color of the )(lantern|basket)\? Reply with one color\.", instruction)
        match = re.fullmatch(r"The (lantern|basket) is (\w+)\. The (lantern|basket) is (\w+)\.", context)
        pairs = [(match[1], match[2]), (match[3], match[4])] if match else []
    elif row["category"] == "name":
        query = re.fullmatch(r"(?:Who has|Which person has) the (key|map)\? Reply with one name\.", instruction)
        match = re.fullmatch(r"(\w+) has the (key|map)\. (\w+) has the (key|map)\.", context)
        pairs = [(match[2], match[1]), (match[4], match[3])] if match else []
    else:
        raise ValueError("unknown task")
    if (query is None or len(pairs) != 2 or len(dict(pairs)) != 2
            or pairs[0][1] == pairs[1][1]):
        raise ValueError("ambiguous extraction example")
    return dict(pairs)[query[1]]


def validate_examples(groups, source):
    if groups != build_examples(source):
        raise ValueError("examples differ from the fixed one-factor transformations")
    train_prompts = {format_prompt(r["instruction"], r["context"]) for r in source["train"]}
    prompts, ids = set(), set()
    for condition in CONDITIONS:
        rows = groups[condition]
        if Counter(r["category"] for r in rows) != COUNTS:
            raise ValueError("unexpected task counts")
        for row in rows:
            prompt = format_prompt(row["instruction"], row["context"])
            if row["response"] != solve(row):
                raise ValueError("response differs from the independent text oracle")
            if prompt in prompts or prompt in train_prompts or row["id"] in ids:
                raise ValueError("duplicate prompt/ID or overlap with SFT training")
            prompts.add(prompt)
            ids.add(row["id"])
        for category in ("color", "name"):
            contexts = {}
            for row in rows:
                if row["category"] == category:
                    contexts.setdefault(row["context"], []).append(row)
            if len(contexts) != 12 or any(len(pair) != 2 or len({solve(r) for r in pair}) != 2
                                          for pair in contexts.values()):
                raise ValueError("expected two opposite queries per extraction context")
    for category, mapping in VALUE_MAPS.items():
        seen = {r["response"] for r in source["train"] if r["category"] == category}
        if set(mapping) != seen or set(mapping.values()) & seen or len(set(mapping.values())) != len(mapping):
            raise ValueError("new SFT values must be disjoint and mapped one-to-one")


def encode_groups(groups, tokenizer, *, context_length=256, max_new_tokens=8):
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("missing EOS token")
    encoded, statistics = {}, {}
    for condition, rows in groups.items():
        encoded[condition] = []
        statistics[condition] = {}
        for row in rows:
            example = encode_example(tokenizer, row["instruction"], row["context"], row["response"], eos)
            boundary = example["prompt_tokens"]
            if (boundary < 1 or boundary + max_new_tokens > context_length
                    or len(example["ids"]) - boundary > max_new_tokens):
                raise ValueError("prompt or reference exceeds the unchanged generation budget")
            if tokenizer.decode(example["ids"][boundary:-1], skip_special_tokens=True) != row["response"]:
                raise ValueError("answer tokenization does not round-trip")
            encoded[condition].append({**example, "source_id": row["id"], "category": row["category"]})
        for category in COUNTS:
            examples = [r for r in encoded[condition] if r["category"] == category]
            statistics[condition][category] = {
                "examples": len(examples),
                "prompt_tokens_min": min(r["prompt_tokens"] for r in examples),
                "prompt_tokens_max": max(r["prompt_tokens"] for r in examples),
                "answer_tokens_without_eos": dict(sorted(Counter(
                    len(r["ids"]) - r["prompt_tokens"] - 1 for r in examples).items())),
            }
    return encoded, statistics


def prepare(source_path, examples_path, manifest_path, tokenizer):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    groups = build_examples(source)
    validate_examples(groups, source)
    _, statistics = encode_groups(groups, tokenizer)
    examples_path, manifest_path = Path(examples_path), Path(manifest_path)
    if examples_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite the frozen probe files")
    examples_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "kind": "sft-transfer-diagnostic-v1",
        "source": "Self-authored paired transformations of the existing dev; no external data",
        "source_sha256": file_hash(source_path),
        "examples_sha256": file_hash(examples_path),
        "conditions": list(CONDITIONS),
        "examples_per_condition": 64,
        "categories_per_condition": COUNTS,
        "context_length": 256, "max_new_tokens": 8,
        "tokenizer": DOLMA_TOKENIZER,
        "value_maps": VALUE_MAPS,
        "tokenization": statistics,
        "checks": {"text_oracle_matches_all_labels": True, "no_training_prompt_overlap": True,
                   "one_factor_changed_per_variant": True, "new_values_absent_from_sft_answers": True,
                   "reserved_holdouts_read": False},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/sft_diversity_examples.json"))
    parser.add_argument("--examples", type=Path, default=Path("experiments/sft_transfer_examples.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/sft_transfer_manifest.json"))
    args = parser.parse_args()
    manifest = prepare(args.source, args.examples, args.manifest, load_tokenizer(DOLMA_TOKENIZER))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
