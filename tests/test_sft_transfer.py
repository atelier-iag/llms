import copy
import json
from pathlib import Path

import pytest

from evaluation.sft_transfer import summarize, verify_control
from experiments.sft_diagnostic_data import build_examples as original_examples
from experiments.sft_transfer_data import (
    CONDITIONS, VALUE_MAPS, build_examples, encode_groups, solve, validate_examples,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_transfer_probes_change_only_the_intended_factor():
    source = original_examples((1, 4))
    groups = build_examples(source)
    assert groups == json.loads((ROOT / "experiments/sft_transfer_examples.json").read_text())
    validate_examples(groups, source)
    for i, base in enumerate(source["val"]):
        control = groups["control"][i]
        assert all(control[key] == base[key] for key in ("instruction", "context", "response", "category"))
        for condition in CONDITIONS[1:]:
            row = groups[condition][i]
            assert row["base_id"] == base["id"]
            changed = {key for key in ("instruction", "context", "response") if row[key] != base[key]}
            assert changed == ({"instruction"} if condition == "rephrase" else
                               {"context"} if condition == "reorder" else {"context", "response"})
            assert solve(row) == row["response"]
        inverse = {v: k for k, v in VALUE_MAPS[base["category"]].items()}
        assert inverse[groups["new_values"][i]["response"]] == base["response"]


@pytest.mark.parametrize("category,instruction,context,expected", [
    ("copy", "Return the target word. Answer with one word.",
     "Other word: forest\nTarget word: garden", "garden"),
    ("color", "What is the color of the lantern? Reply with one color.",
     "The basket is cyan. The lantern is purple.", "purple"),
    ("color", "Which color is the basket? Reply with one color.",
     "The basket is cyan. The lantern is purple.", "cyan"),
    ("name", "Which person has the map? Reply with one name.",
     "Irene has the map. Grace has the key.", "Irene"),
    ("name", "Who has the key? Reply with one name.",
     "Irene has the map. Grace has the key.", "Grace"),
])
def test_oracle_uses_requested_relation_despite_order_and_wrong_label(category, instruction, context, expected):
    assert solve({"category": category, "instruction": instruction, "context": context,
                  "response": "intentionally wrong"}) == expected


@pytest.mark.parametrize("mutation", ["label", "two_factors", "duplicate", "omission"])
def test_transfer_validation_rejects_invalid_or_uncontrolled_examples(mutation):
    source = original_examples((1, 4))
    groups = build_examples(source)
    if mutation == "label":
        groups["new_values"][0]["response"] = "wrong"
    elif mutation == "two_factors":
        groups["rephrase"][0]["context"] = groups["reorder"][0]["context"]
    elif mutation == "duplicate":
        groups["new_values"][1] = copy.deepcopy(groups["new_values"][0])
    else:
        groups["reorder"].pop()
    with pytest.raises(ValueError):
        validate_examples(groups, source)


def test_oracle_rejects_duplicate_relation_and_trailing_facts():
    row = {"category": "name", "instruction": "Which person has the map? Reply with one name.",
           "context": "Grace has the map. Irene has the map."}
    with pytest.raises(ValueError):
        solve(row)
    row["context"] = "Grace has the key. Irene has the map. Jack has the map."
    with pytest.raises(ValueError):
        solve(row)


def test_paired_analysis_counts_regressions_gains_and_joint_context_success():
    groups = build_examples(original_examples((1, 4)))
    rows = groups["rephrase"]
    details = [{"source_id": row["id"], "category": row["category"],
                "exact_match": i != 0, "stopped_at_eos": True} for i, row in enumerate(rows)]
    control = {row["base_id"]: {"exact_match": i != 1} for i, row in enumerate(rows)}
    summary = summarize(details, rows, control)
    assert summary["scores"]["total"] == {"examples": 64, "exact_matches": 63, "stopped_at_eos": 64}
    assert summary["paired_vs_control"]["copy"] == {
        "correct_in_both": 14, "lost_correct_ids": [rows[0]["base_id"]],
        "gained_correct_ids": [rows[1]["base_id"]]}
    assert summary["paired_contexts"]["name"] == {"contexts": 12, "both_correct": 12}
    with pytest.raises(ValueError, match="coverage"):
        summarize(list(reversed(details)), rows, control)
    with pytest.raises(ValueError, match="IDs"):
        summarize(details, rows, {})


def test_control_reproduction_checks_tokens_even_if_exact_score_matches():
    old = [{"source_id": "val-copy-01", "generated_ids": [3, 0], "exact_match": False}]
    current = [{**old[0], "source_id": "control-val-copy-01"}]
    verify_control(current, old)
    current[0]["generated_ids"] = [4, 0]
    with pytest.raises(ValueError, match="generation"):
        verify_control(current, old)


def test_reference_must_fit_full_budget_without_silent_truncation():
    class Tokenizer:
        def token_to_id(self, text):
            return 0

        def encode(self, text, add_special_tokens=False):
            class Encoded:
                ids = [1] * (250 if text.startswith("###") else 1)
            return Encoded()

    with pytest.raises(ValueError, match="budget"):
        encode_groups(build_examples(original_examples((1, 4))), Tokenizer())
