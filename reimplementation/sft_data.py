"""Independent, right-padded instruction/answer sequences with answer-only targets."""

import json
from pathlib import Path

import numpy as np
import torch

from reimplementation.prepare_corpus import file_hash


IGNORE_INDEX = -100
FORMAT = "instruction-context-response-v1"


def format_prompt(instruction, context=""):
    return (f"### Instruction:\n{instruction.strip()}\n\n"
            + (f"### Context:\n{context.strip()}\n\n" if context.strip() else "")
            + "### Response:\n")


def encode_example(tokenizer, instruction, context, response, eos_id):
    # Encode the two segments separately: the answer boundary is exact, and
    # the prompt IDs are identical at training and free-generation time.
    prompt = tokenizer.encode(format_prompt(instruction, context), add_special_tokens=False).ids
    answer = tokenizer.encode(response.strip(), add_special_tokens=False).ids
    return {"ids": prompt + answer + [eos_id], "prompt_tokens": len(prompt)}


def collate(examples, pad_id):
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(len(row["ids"]) for row in examples) - 1
    inputs = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, IGNORE_INDEX)
    for index, row in enumerate(examples):
        ids, boundary = row["ids"], row["prompt_tokens"]
        if not 1 <= boundary < len(ids):
            raise ValueError("example must have a nonempty prompt and completion")
        length = len(ids) - 1
        inputs[index, :length] = torch.tensor(ids[:-1])
        # The LAST prompt position predicts the FIRST answer token.
        labels[index, boundary - 1:length] = torch.tensor(ids[boundary:])
    return inputs, labels


def batches(examples, batch_size, pad_id, rng=None):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    order = np.arange(len(examples))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield collate([examples[int(i)] for i in order[start:start + batch_size]], pad_id)


def load_sft(data_dir, *, tokenizer, vocab_size, max_tokens):
    """Verify train/dev identities without opening the reserved holdout."""
    data_dir = Path(data_dir)
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("kind") != "sft" or manifest.get("prompt_format") != FORMAT
            or manifest.get("provenance", {}).get("tokenizer") != tokenizer
            or manifest.get("vocab_size") != vocab_size or manifest.get("max_tokens") != max_tokens):
        raise ValueError("incompatible SFT manifest")
    eos = manifest["eos_token_id"]
    if type(eos) is not int or not 0 <= eos < vocab_size:
        raise ValueError("invalid SFT EOS")
    splits = {}
    groups = set()
    fingerprints = set()
    for split in ("train", "val"):
        entry = manifest["files"][split]
        if entry["path"] != f"{split}.jsonl":
            raise ValueError("unexpected SFT filename")
        file = data_dir / entry["path"]
        if file_hash(file) != entry["sha256"]:
            raise ValueError("SFT file hash differs from manifest")
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        if not rows or len(rows) != entry["examples"]:
            raise ValueError("SFT example count differs from manifest")
        current_groups = set()
        for row in rows:
            ids, boundary = row["ids"], row["prompt_tokens"]
            if (not 2 <= len(ids) <= max_tokens or type(boundary) is not int
                    or not 1 <= boundary < len(ids) - 1 or ids[-1] != eos
                    or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)):
                raise ValueError("invalid SFT token sequence or response boundary")
            if row["fingerprint"] in fingerprints or row["group"] in groups:
                raise ValueError("duplicate SFT example or group leakage")
            fingerprints.add(row["fingerprint"])
            current_groups.add(row["group"])
        groups.update(current_groups)
        if (sum(len(row["ids"]) for row in rows) != entry["sequence_tokens"]
                or sum(len(row["ids"]) - row["prompt_tokens"] for row in rows) != entry["response_tokens"]):
            raise ValueError("SFT token counts differ from manifest")
        splits[split] = rows
    return splits, {"path": str(path.resolve()), "sha256": file_hash(path), "contents": manifest}
