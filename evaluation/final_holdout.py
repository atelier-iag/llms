"""One final evaluation of frozen checkpoints on the existing reserved holdouts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import torch

from evaluation.language_model import evaluate
from evaluation.sft import evaluate_responses
from reimplementation.checkpoint import load_checkpoint
from reimplementation.data import TokenFile
from reimplementation.prepare_corpus import file_hash
from reimplementation.prepare_sft import balanced_select
from reimplementation.sft_data import FORMAT, batches, load_sft
from reimplementation.sft_loss import evaluate_sft
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer


def load_manifest(spec, vocab_size):
    directory = Path(spec["directory"])
    path = directory / "manifest.json"
    if file_hash(path) != spec["manifest_sha256"]:
        raise ValueError("holdout manifest hash differs from the frozen plan")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("format_version") != 1
            or manifest.get("provenance", {}).get("tokenizer") != DOLMA_TOKENIZER
            or manifest.get("vocab_size") != vocab_size
            or manifest.get("files", {}).get("holdout") != spec["holdout"]):
        raise ValueError("incompatible holdout manifest")
    entry = manifest["files"]["holdout"]
    expected_name = "holdout.jsonl" if spec["kind"] == "sft" else "holdout.npy"
    if entry["path"] != expected_name:
        raise ValueError("expected an explicitly named holdout file")
    path = directory / expected_name
    for split in ("train", "val"):
        other = directory / manifest["files"][split]["path"]
        if path.samefile(other):
            raise ValueError("holdout aliases a development or training file")
    if file_hash(path) != entry["sha256"]:
        raise ValueError("holdout file hash differs from its frozen identity")
    return path, manifest


def load_lm_holdout(spec, *, context_length, vocab_size):
    path, manifest = load_manifest(spec, vocab_size)
    entry = manifest["files"]["holdout"]
    if path.stat().st_size != entry["tokens"] * 4:
        raise ValueError("holdout token count differs from manifest")
    corpus = TokenFile(path, context_length, vocab_size)
    return corpus


def load_sft_holdout(spec, *, context_length, vocab_size):
    path, manifest = load_manifest(spec, vocab_size)
    if (manifest.get("kind") != "sft" or manifest.get("prompt_format") != FORMAT
            or manifest.get("max_tokens") != context_length):
        raise ValueError("incompatible SFT holdout format")
    existing, _ = load_sft(spec["directory"], tokenizer=DOLMA_TOKENIZER,
                           vocab_size=vocab_size, max_tokens=context_length)
    previous_groups = {r["group"] for rows in existing.values() for r in rows}
    previous_fingerprints = {r["fingerprint"] for rows in existing.values() for r in rows}
    previous_ids = {r["source_id"] for rows in existing.values() for r in rows}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    entry, eos = manifest["files"]["holdout"], manifest["eos_token_id"]
    if not rows or len(rows) != entry["examples"] or type(eos) is not int or not 0 <= eos < vocab_size:
        raise ValueError("invalid SFT holdout count or EOS")
    ids_seen, fingerprints = set(), set()
    for row in rows:
        ids, boundary = row["ids"], row["prompt_tokens"]
        if (not 2 <= len(ids) <= context_length or type(boundary) is not int
                or not 1 <= boundary < len(ids) - 1 or ids[-1] != eos
                or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
                or len(ids) - boundary > manifest["max_answer_tokens"] + 1):
            raise ValueError("invalid SFT holdout token sequence")
        if (row["group"] in previous_groups or row["fingerprint"] in previous_fingerprints
                or row["source_id"] in previous_ids):
            raise ValueError("SFT holdout overlaps train/dev")
        if row["source_id"] in ids_seen or row["fingerprint"] in fingerprints:
            raise ValueError("duplicate SFT holdout example")
        ids_seen.add(row["source_id"])
        fingerprints.add(row["fingerprint"])
    if (len({r["group"] for r in rows}) != entry["groups"]
            or dict(Counter(r["category"] for r in rows)) != entry["categories"]
            or sum(len(r["ids"]) for r in rows) != entry["sequence_tokens"]
            or sum(len(r["ids"]) - r["prompt_tokens"] for r in rows) != entry["response_tokens"]
            or max(len(r["ids"]) for r in rows) != entry["max_sequence_tokens"]):
        raise ValueError("SFT holdout counts differ from manifest")
    return rows, eos


def verify_control(measured, expected, tolerance):
    if (measured["scored_tokens"] != expected["scored_tokens"]
            or not math.isfinite(measured["loss"])
            or abs(measured["loss"] - expected["loss"]) > tolerance):
        raise ValueError("historical dev control differs; do not score holdout")


def category_scores(records):
    result = {}
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        result[category] = {
            "examples": len(rows), "exact_matches": sum(r["exact_match"] for r in rows),
            "mean_token_f1": sum(r["token_f1"] for r in rows) / len(rows),
            "eos_rate": sum(r["stopped_at_eos"] for r in rows) / len(rows),
        }
    return result


def run(config_path, *, device, output_root):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["precision"] != "fp32" or config["context_length"] != 256 or config["batch_size"] != 8:
        raise ValueError("unexpected frozen numerical settings")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit the final protocol and code before reading holdouts")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output([
        "git", "ls-files", "reimplementation", "evaluation/final_holdout.py",
        "evaluation/language_model.py", "evaluation/sft.py"], text=True).splitlines()
    paths = set(tracked) | {str(config_path), config["protocol"]}
    for spec in config["models"].values():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("checkpoint differs from the frozen plan")
        if file_hash(spec["historical_metrics"]) != spec["historical_metrics_sha256"]:
            raise ValueError("historical metrics identity differs")
        paths.add(spec["historical_metrics"])
    for spec in config["datasets"].values():
        manifest_path = str(Path(spec["directory"]) / "manifest.json")
        if file_hash(manifest_path) != spec["manifest_sha256"]:
            raise ValueError("dataset manifest differs from the frozen plan")
        paths.add(manifest_path)
    dev_spec = config["dev_control"]
    if file_hash(dev_spec["path"]) != dev_spec["sha256"]:
        raise ValueError("dev control corpus differs")
    paths.add(dev_spec["path"])
    input_hashes = {p: file_hash(p) for p in sorted(paths)}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    vocab_size = tokenizer.get_vocab_size()
    context, batch_size = config["context_length"], config["batch_size"]
    dev = TokenFile(dev_spec["path"], context, vocab_size)
    if len(dev.tokens) != dev_spec["tokens"]:
        raise ValueError("dev control token count differs")
    offsets = dev.evaluation_starts(dev_spec["windows"])
    Path(output_root).mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="final-holdout-", dir=output_root))
    start = time.perf_counter()
    execution = {
        "kind": "final-holdout-v1", "status": "dev_preflight", "git_commit": commit,
        "run_directory": str(run_dir), "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "input_sha256": input_hashes, "training_updates": 0,
        "torch_version": torch.__version__, "precision": "fp32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }

    def write(name, value):
        path = run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def event(value):
        line = json.dumps(value, allow_nan=False)
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    def load_model(spec):
        model, metadata = load_checkpoint(Path(spec["path"]), device=device)
        if (metadata["model_config"] != spec["model_config"] or metadata["step"] != spec["step"]
                or metadata["context_length"] != context or metadata["tokenizer"] != DOLMA_TOKENIZER
                or sum(p.numel() for p in model.parameters()) != spec["parameter_count"]
                or any(p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError("checkpoint architecture, step, tokenizer, or dtype differs")
        return model

    def release_cuda():
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    metrics = {"dev_controls": {}, "models": {}}
    write("execution.json", execution)
    event({"event": "start", "run_directory": str(run_dir)})
    try:
        # Complete every historical dev check before opening any holdout.
        for label, spec in config["models"].items():
            model = load_model(spec)
            measured = evaluate(model, (dev.batch(offsets[i:i + batch_size])
                                        for i in range(0, len(offsets), batch_size)))
            verify_control(measured, spec["dev_control_expected"], dev_spec["loss_absolute_tolerance"])
            metrics["dev_controls"][label] = measured
            write("metrics.json", metrics)
            event({"event": "dev_control_passed", "model": label, "loss": measured["loss"]})
            del model
            release_cuda()
        execution["first_holdout_opened_at"] = datetime.now(timezone.utc).isoformat()
        execution["status"] = "holdout_evaluation"
        write("execution.json", execution)
        corpora, sft_rows, eos = {}, None, None
        for name, spec in config["datasets"].items():
            if spec["kind"] == "lm":
                corpora[name] = load_lm_holdout(spec, context_length=context, vocab_size=vocab_size)
            else:
                sft_rows, eos = load_sft_holdout(spec, context_length=context, vocab_size=vocab_size)
        generation_config = config["dolly_generation"]
        selected = balanced_select(sft_rows, generation_config["examples"], generation_config["seed"])
        if dict(Counter(r["category"] for r in selected)) != {
                k: 6 for k in ("classification", "closed_qa", "information_extraction", "summarization")}:
            raise ValueError("expected the fixed six-per-category generation sample")
        execution["dolly_generation_selection"] = [
            {"source_id": r["source_id"], "category": r["category"],
             "prompt_tokens": r["prompt_tokens"], "generation_budget": min(
                 generation_config["max_new_tokens"], context - r["prompt_tokens"])} for r in selected]
        execution["holdout_sha256"] = {name: spec["holdout"]["sha256"] for name, spec in config["datasets"].items()}
        write("execution.json", execution)
        for label, spec in config["models"].items():
            model = load_model(spec)
            model_start = time.perf_counter()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            metrics["models"][label] = {}
            for dataset in spec["holdouts"]:
                event({"event": "holdout_start", "model": label, "dataset": dataset})
                if dataset in corpora:
                    score = evaluate(model, corpora[dataset].epoch_batches(batch_size))
                    if score["scored_tokens"] != config["datasets"][dataset]["holdout"]["tokens"] - 1:
                        raise ValueError("holdout target coverage differs")
                else:
                    response = evaluate_sft(model, batches(sft_rows, batch_size, eos))
                    if response["scored_tokens"] != config["datasets"][dataset]["holdout"]["response_tokens"]:
                        raise ValueError("SFT holdout response coverage differs")
                    generated, private = evaluate_responses(
                        model, selected, tokenizer, context_length=context,
                        max_new_tokens=generation_config["max_new_tokens"], eos_id=eos)
                    write(label + "-dolly-generations-private.json", private)
                    score = {"response": response, "generation": generated,
                             "generation_by_category": category_scores(generated["records"])}
                metrics["models"][label][dataset] = score
                write("metrics.json", metrics)
                event({"event": "holdout_complete", "model": label, "dataset": dataset,
                       "perplexity": score.get("perplexity", score.get("response", {}).get("perplexity")),
                       "exact_matches": score.get("generation", {}).get("exact_matches")})
            if device == "cuda":
                torch.cuda.synchronize()
            execution.setdefault("models", {})[label] = {
                "seconds": time.perf_counter() - model_start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
            write("execution.json", execution)
            del model
            release_cuda()
        if any(file_hash(path) != sha for path, sha in input_hashes.items()):
            raise ValueError("code, protocol, or historical inputs changed")
        if any(file_hash(s["path"]) != s["sha256"] for s in config["models"].values()):
            raise ValueError("checkpoint changed")
        for spec in config["datasets"].values():
            path = Path(spec["directory"]) / spec["holdout"]["path"]
            if file_hash(path) != spec["holdout"]["sha256"]:
                raise ValueError("holdout data changed")
        execution.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                         seconds=time.perf_counter() - start, inputs_unchanged=True,
                         checkpoints_unchanged=True, holdouts_unchanged=True)
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
    parser.add_argument("--config", type=Path, default=Path("experiments/final_holdout_config.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run(args.config, device=args.device, output_root=args.output_root)


if __name__ == "__main__":
    main()
