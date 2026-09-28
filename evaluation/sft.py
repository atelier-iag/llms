"""Fixed free-generation diagnostics, distinct from teacher-forced SFT dev loss."""

from collections import Counter
import re
import unicodedata

import torch

from reimplementation.generate import generate


def normalized_answer(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!?")


def token_f1(prediction, reference):
    predicted = re.findall(r"\w+", normalized_answer(prediction))
    expected = re.findall(r"\w+", normalized_answer(reference))
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def evaluate_responses(model, examples, tokenizer, *, context_length, max_new_tokens, eos_id):
    rows = []
    private = []
    for example in examples:
        boundary = example["prompt_tokens"]
        prompt_ids = example["ids"][:boundary]
        budget = min(max_new_tokens, context_length - len(prompt_ids))
        if budget < 1:
            raise ValueError("generation prompt leaves no room for a response")
        output = generate(model, torch.tensor([prompt_ids]), context_length=context_length,
                          max_new_tokens=budget, eos_token_id=eos_id)[0].tolist()[boundary:]
        prediction = tokenizer.decode(output, skip_special_tokens=True)
        reference = tokenizer.decode(example["ids"][boundary:-1], skip_special_tokens=True)
        without_eos = output[:-1] if output and output[-1] == eos_id else output
        ngrams = [tuple(without_eos[i:i + 4]) for i in range(len(without_eos) - 3)]
        row = {"source_id": example["source_id"], "category": example["category"],
               "exact_match": normalized_answer(prediction) == normalized_answer(reference),
               "token_f1": token_f1(prediction, reference),
               "stopped_at_eos": bool(output and output[-1] == eos_id),
               "generated_tokens": len(output), "generation_budget": budget,
               "repeated_4gram_fraction": 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0}
        rows.append(row)
        private.append({**row, "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                        "reference": reference, "prediction": prediction, "generated_ids": output})
    if not rows:
        raise ValueError("generation evaluation requires examples")
    metrics = {"examples": len(rows), "records": rows, "method": "greedy",
               "exact_matches": sum(row["exact_match"] for row in rows),
               "exact_match_rate": sum(row["exact_match"] for row in rows) / len(rows),
               "mean_token_f1": sum(row["token_f1"] for row in rows) / len(rows),
               "eos_rate": sum(row["stopped_at_eos"] for row in rows) / len(rows),
               "mean_repeated_4gram_fraction": sum(row["repeated_4gram_fraction"] for row in rows) / len(rows)}
    return metrics, private
