"""Evaluate frozen paired SFT transfer probes without constructing an optimizer."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch

from evaluation.sft import evaluate_responses
from experiments.sft_transfer_data import CONDITIONS, COUNTS, encode_groups, validate_examples
from reimplementation.checkpoint import load_checkpoint
from reimplementation.prepare_corpus import file_hash
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


def summarize(details, raw_rows, control=None):
    if len(details) != len(raw_rows) or [r["source_id"] for r in details] != [r["id"] for r in raw_rows]:
        raise ValueError("generation coverage or order differs from the probes")
    counts = {}
    for category in (*COUNTS, "total"):
        selected = [r for r in details if category == "total" or r["category"] == category]
        counts[category] = {
            "examples": len(selected),
            "exact_matches": sum(r["exact_match"] for r in selected),
            "stopped_at_eos": sum(r["stopped_at_eos"] for r in selected),
        }
    by_base = {raw["base_id"]: row for raw, row in zip(raw_rows, details)}
    paired = {}
    if control is not None:
        if set(control) != set(by_base):
            raise ValueError("paired control IDs differ")
        for category in (*COUNTS, "total"):
            ids = [key for key, r in by_base.items() if category == "total" or r["category"] == category]
            paired[category] = {
                "correct_in_both": sum(control[key]["exact_match"] and by_base[key]["exact_match"] for key in ids),
                "lost_correct_ids": [key for key in ids if control[key]["exact_match"] and not by_base[key]["exact_match"]],
                "gained_correct_ids": [key for key in ids if not control[key]["exact_match"] and by_base[key]["exact_match"]],
            }
    contexts = {}
    for category in ("color", "name"):
        pairs = {}
        for raw, row in zip(raw_rows, details):
            if raw["category"] == category:
                pairs.setdefault(raw["context"], []).append(row)
        if any(len(pair) != 2 for pair in pairs.values()):
            raise ValueError("extraction context must have two queries")
        contexts[category] = {"contexts": len(pairs), "both_correct": sum(
            all(r["exact_match"] for r in pair) for pair in pairs.values())}
    return {"scores": counts, "paired_vs_control": paired, "paired_contexts": contexts}


def verify_control(details, archived):
    if len(details) != len(archived):
        raise ValueError("historical control count differs")
    for new, old in zip(details, archived):
        if new["source_id"] != "control-" + old["source_id"]:
            raise ValueError("historical control ID differs")
        if {k: v for k, v in new.items() if k != "source_id"} != {k: v for k, v in old.items() if k != "source_id"}:
            raise ValueError(f"historical control generation differs: {old['source_id']}")


def run(config_path, *, device, output_root):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["conditions"] != list(CONDITIONS) or config["context_length"] != 256 or config["max_new_tokens"] != 8:
        raise ValueError("unexpected frozen evaluation controls")
    source = json.loads(Path(config["source"]).read_text(encoding="utf-8"))
    groups = json.loads(Path(config["examples"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    validate_examples(groups, source)
    if (manifest["examples_sha256"] != file_hash(config["examples"])
            or manifest["source_sha256"] != file_hash(config["source"])
            or manifest["tokenizer"] != DOLMA_TOKENIZER):
        raise ValueError("probe manifest identity differs")
    archived = json.loads(Path(config["historical_generations"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(config["historical_execution"]).read_text(encoding="utf-8"))
    expected_hashes = {"initial": provenance["initial_checkpoint_sha256"], "sft": provenance["final_checkpoint_sha256"]}
    if set(config["models"]) != set(expected_hashes):
        raise ValueError("expected initial and best SFT checkpoints")
    for label, spec in config["models"].items():
        if spec["sha256"] != expected_hashes[label] or file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError(f"checkpoint identity differs: {label}")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit the protocol, data, and code before evaluation")
    tracked = subprocess.check_output(
        ["git", "ls-files", "reimplementation", "evaluation/sft.py", "evaluation/sft_transfer.py",
         "experiments/sft_transfer_data.py"], text=True).splitlines()
    paths = set(tracked) | {str(config_path), *(config[key] for key in (
        "source", "examples", "manifest", "protocol", "historical_generations", "historical_execution"))}
    input_hashes = {path: file_hash(path) for path in sorted(paths)}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    encoded, statistics = encode_groups(groups, tokenizer, context_length=config["context_length"],
                                         max_new_tokens=config["max_new_tokens"])
    if json.loads(json.dumps(statistics)) != manifest["tokenization"]:
        raise ValueError("tokenization statistics differ from the frozen manifest")
    eos = tokenizer.token_to_id("<|endoftext|>")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="sft-transfer-", dir=output_root))
    start = time.perf_counter()
    execution = {
        "kind": "sft-transfer-evaluation-v1", "status": "running", "run_directory": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "config": config, "input_sha256": input_hashes,
        "torch_version": torch.__version__, "precision": "fp32", "training_updates": 0,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "holdouts_read": False,
    }

    def write(name, value):
        (run_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def event(value):
        line = json.dumps(value, allow_nan=False)
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    write("execution.json", execution)
    event({"event": "start", "run_directory": str(run_dir), "generations": 512})
    all_metrics, generations, analysis = {}, {}, {}
    common_metadata = None
    try:
        for label, spec in config["models"].items():
            model_start = time.perf_counter()
            model, metadata = load_checkpoint(Path(spec["path"]), device=device)
            architecture = {key: metadata[key] for key in ("model_config", "tokenizer", "context_length")}
            if (metadata["tokenizer"] != DOLMA_TOKENIZER or metadata["context_length"] != config["context_length"]
                    or metadata["model_config"]["vocab_size"] != tokenizer.get_vocab_size()
                    or any(p.dtype != torch.float32 for p in model.parameters())):
                raise ValueError("checkpoint tokenizer, context, or precision differs")
            if common_metadata is not None and architecture != common_metadata:
                raise ValueError("model architectures differ")
            common_metadata = architecture
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            all_metrics[label] = {}
            generations[label] = {}
            analysis[label] = {}
            control = None
            for condition in CONDITIONS:
                scores, details = evaluate_responses(
                    model, encoded[condition], tokenizer, context_length=config["context_length"],
                    max_new_tokens=config["max_new_tokens"], eos_id=eos)
                if condition == "control":
                    verify_control(details, archived["initial" if label == "initial" else "final"]["dev"])
                    control = {raw["base_id"]: row for raw, row in zip(groups[condition], details)}
                all_metrics[label][condition] = scores
                generations[label][condition] = details
                analysis[label][condition] = summarize(details, groups[condition], control)
                write(f"{label}-generations.json", generations[label])
                write(f"{label}-metrics.json", all_metrics[label])
                event({"event": "condition_complete", "model": label, "condition": condition,
                       "exact_matches": scores["exact_matches"], "examples": scores["examples"],
                       "eos_rate": scores["eos_rate"]})
            if device == "cuda":
                torch.cuda.synchronize()
            execution.setdefault("models", {})[label] = {
                "metadata": metadata, "parameter_count": sum(p.numel() for p in model.parameters()),
                "seconds": time.perf_counter() - model_start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "historical_control_reproduced_exactly": True,
            }
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        if any(file_hash(path) != digest for path, digest in input_hashes.items()):
            raise ValueError("input/code files changed during evaluation")
        if any(file_hash(spec["path"]) != spec["sha256"] for spec in config["models"].values()):
            raise ValueError("checkpoint changed during evaluation")
        execution.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                         seconds=time.perf_counter() - start, inputs_unchanged=True, checkpoints_unchanged=True)
        write("metrics.json", all_metrics)
        write("generations.json", generations)
        write("analysis.json", analysis)
        write("execution.json", execution)
        event({"event": "complete", "run_directory": str(run_dir), "seconds": execution["seconds"]})
    except Exception as exc:
        execution.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write("execution.json", execution)
        event({"event": "failed", "error": execution["error"]})
        raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/sft_transfer_config.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run(args.config, device=args.device, output_root=args.output_root)


if __name__ == "__main__":
    main()
