"""Measure a simple model over complete corpus passes before changing its architecture."""

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.language_model import evaluate
from reimplementation.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.corpus_manifest import validate_manifest
from reimplementation.data import open_splits
from reimplementation.generate import generate
from reimplementation.model import CausalLanguageModel
from reimplementation.precision import validate_precision
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer
from reimplementation.train import train_step
from reimplementation.train_corpus import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def learning_rate(step: int, total_steps: int, peak: float, warmup: int, floor: float) -> float:
    """One-based update number: linear warmup, then cosine decay to floor * peak."""
    if not 1 <= step <= total_steps or not 0 <= warmup < total_steps:
        raise ValueError("invalid step count or warmup duration")
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return peak * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


def recovery_history(checkpoint: dict, path: Path) -> dict:
    """Read histories saved in new checkpoints, or beside a legacy checkpoint."""
    if "run_state" in checkpoint:
        return checkpoint["run_state"]
    records = []
    lines = path.with_name("progress.jsonl").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A crash can leave the final log line incomplete.
    step = checkpoint["step"]
    initial = next(record for record in records if record["event"] == "initial")
    history = [record for record in records
               if record["event"] == "train" and record["step"] <= step]
    evaluations = [{"step": 0, "target_tokens": 0, **initial["samples"]}]
    evaluations.extend({key: value for key, value in record.items() if key != "event"}
                       for record in records
                       if record["event"] == "evaluation" and record["step"] <= step)
    if not history or history[-1]["step"] != step or evaluations[-1]["step"] != step:
        raise ValueError("legacy recovery requires a checkpoint at a completed logging boundary")
    return {"initial_full_validation": initial["full_validation"],
            "evaluation_history": evaluations, "training_history": history,
            "window_nll": 0.0, "window_tokens": 0}


def run(config: dict, data_dir: Path, *, device: str, output_root: Path,
        resume: Path | None = None) -> Path:
    precision = config.get("precision", "fp32")
    validate_precision(precision, device)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    corpus_config = config.get("corpus", {})
    corpus_manifest = validate_manifest(
        data_dir, tokenizer=DOLMA_TOKENIZER, vocab_size=config["model"]["vocab_size"],
        required=corpus_config.get("require_manifest", False),
    )
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    if tokenizer.get_vocab_size() != config["model"]["vocab_size"]:
        raise ValueError("model vocabulary must match the tokenizer")
    prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in config["prompts"]]
    if any(not ids for ids in prompts):
        raise ValueError("generation prompts must be nonempty")
    batch_size = config["batch_size"]
    train, val = open_splits(data_dir, config["sequence_length"], config["model"]["vocab_size"])
    for name, split in (("train", train), ("validation", val)):
        expected = corpus_config.get(f"{name}_tokens")
        if expected is not None and len(split.tokens) != expected:
            raise ValueError(f"unexpected {name} corpus token budget")
    total_steps = config["epochs"] * train.epoch_batch_count(batch_size)
    if config["epochs"] < 1 or not 0 <= config["warmup_updates"] < total_steps:
        raise ValueError("epochs must be positive and warmup shorter than the training run")
    for key in ("evaluation_windows", "evaluate_every", "log_every", "max_new_tokens"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= config["min_lr_ratio"] <= 1 or config["max_grad_norm"] <= 0:
        raise ValueError("invalid learning-rate floor or gradient clipping limit")
    splits = {"train": train, "validation": val}
    offsets = {name: split.evaluation_starts(config["evaluation_windows"])
               for name, split in splits.items()}
    hashes = {name: file_sha256(split.path) for name, split in splits.items()}
    recovery = restored_history = None
    resume_step = resume_seen = 0
    if resume is not None:
        resume = Path(resume).resolve()
        recovery = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
        if recovery.get("format_version") != 1 or "optimizer_state" not in recovery:
            raise ValueError("--resume requires a recovery checkpoint with optimizer state")
        if (recovery["config"] != config or recovery["model_config"] != config["model"]
                or recovery["context_length"] != config["sequence_length"]
                or recovery["tokenizer"] != DOLMA_TOKENIZER):
            raise ValueError("resume configuration/tokenizer differs from the checkpoint")
        if recovery["data_sha256"] != hashes:
            raise ValueError("resume data hashes differ from the checkpoint")
        manifest_hash = corpus_manifest["sha256"] if corpus_manifest else None
        if recovery.get("corpus_manifest_sha256") != manifest_hash:
            raise ValueError("resume corpus manifest differs from the checkpoint")
        resume_step, resume_seen = recovery["step"], recovery["training_target_tokens"]
        if not isinstance(resume_step, int) or not 0 < resume_step <= total_steps:
            raise ValueError("invalid recovery update count")
        epochs_done, batches_done = divmod(resume_step, train.epoch_batch_count(batch_size))
        full_windows = (len(train.tokens) - 1) // config["sequence_length"]
        expected_seen = (epochs_done * (len(train.tokens) - 1)
                         + min(batches_done * batch_size, full_windows) * config["sequence_length"])
        if resume_seen != expected_seen:
            raise ValueError("recovery token count does not match the data order")
        cuda_states = recovery["cuda_rng_states"]
        if bool(cuda_states) != (device == "cuda") or (
            device == "cuda" and len(cuda_states) != torch.cuda.device_count()
        ):
            raise ValueError("resume requires the same CPU/CUDA device configuration")
        restored_history = recovery_history(recovery, resume)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="simple-baseline-", dir=output_root))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    model = CausalLanguageModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    if recovery is not None:
        model.load_state_dict(recovery.pop("model_state"), strict=True)
        optimizer.load_state_dict(recovery.pop("optimizer_state"))
        # Model construction consumes RNG state; restore it only after loading.
        torch.set_rng_state(recovery["torch_rng_state"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(recovery["cuda_rng_states"])
        del recovery

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def measure_samples():
        measurements = {}
        for name, split in splits.items():
            starts = offsets[name]
            measurements[name] = evaluate(model, (
                split.batch(starts[index:index + batch_size])
                for index in range(0, len(starts), batch_size)
            ))
        return measurements

    def event(record):
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def save_recovery(step, seen):
        # A rolling checkpoint in this new run directory avoids losing a long run.
        # Keep optimizer and RNG states as well as the inference model metadata.
        state = {
            "format_version": 1, "model_config": config["model"],
            "model_state": model.state_dict(), "step": step,
            "context_length": config["sequence_length"], "tokenizer": DOLMA_TOKENIZER,
            "optimizer_state": optimizer.state_dict(), "config": config,
            "data_sha256": hashes, "training_target_tokens": seen,
            "corpus_manifest_sha256": corpus_manifest["sha256"] if corpus_manifest else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "run_state": {
                "initial_full_validation": initial_full_validation,
                "evaluation_history": evaluations, "training_history": history,
                "window_nll": window_nll, "window_tokens": window_tokens,
            },
        }
        temporary = run_dir / "recovery.tmp"
        torch.save(state, temporary)
        temporary.replace(run_dir / "recovery.pt")

    parameter_count = sum(p.numel() for p in model.parameters())
    event({"event": "start", "run_dir": str(run_dir), "parameters": parameter_count,
           "total_updates": total_steps, "target_tokens": config["epochs"] * (len(train.tokens) - 1)})
    wall_start = time.perf_counter()
    window_nll = 0.0
    window_tokens = 0
    if restored_history is None:
        initial_samples = measure_samples()
        event({"event": "validation_full_start"})
        initial_full_validation = evaluate(model, val.epoch_batches(batch_size))
        event({"event": "initial", "samples": initial_samples, "full_validation": initial_full_validation})
        evaluations = [{"step": 0, "target_tokens": 0, **initial_samples}]
        history = []
    else:
        initial_full_validation = restored_history["initial_full_validation"]
        evaluations = restored_history["evaluation_history"]
        history = restored_history["training_history"]
        window_nll = restored_history["window_nll"]
        window_tokens = restored_history["window_tokens"]
        event({"event": "resume", "checkpoint": str(resume), "step": resume_step,
               "target_tokens": resume_seen, "remaining_updates": total_steps - resume_step})
    seen, step = resume_seen, 0
    train_seconds = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    segment_start = time.perf_counter()
    for epoch in range(config["epochs"]):
        rng = np.random.default_rng(config["seed"] + epoch)
        for tokens in train.epoch_batches(batch_size, rng):
            step += 1
            if step <= resume_step:
                # Regenerate the same epoch permutation, without replaying updates.
                continue
            lr = learning_rate(step, total_steps, config["optimizer"]["lr"],
                               config["warmup_updates"], config["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = train_step(model, optimizer, tokens.to(device),
                              max_grad_norm=config["max_grad_norm"], precision=precision)
            count = tokens.shape[0] * (tokens.shape[1] - 1)
            seen += count
            window_tokens += count
            window_nll += loss * count
            if step % config["log_every"] == 0 or step == total_steps:
                record = {"event": "train", "step": step, "target_tokens": seen,
                          "lr": lr, "mean_training_loss": window_nll / window_tokens,
                          "window_target_tokens": window_tokens,
                          "session_start_step": resume_step,
                          "elapsed_seconds": time.perf_counter() - wall_start}
                history.append(record)
                event(record)
                window_nll = 0.0
                window_tokens = 0
            if step % config["evaluate_every"] == 0 or step == total_steps:
                synchronize()
                train_seconds += time.perf_counter() - segment_start
                metrics = measure_samples()
                record = {"step": step, "target_tokens": seen, **metrics}
                evaluations.append(record)
                event({"event": "evaluation", **record})
                save_recovery(step, seen)
                synchronize()
                segment_start = time.perf_counter()

    expected_targets = config["epochs"] * (len(train.tokens) - 1)
    if seen != expected_targets or step != total_steps:
        raise RuntimeError("training did not cover the planned corpus passes")
    peak_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else None
    event({"event": "validation_full_end"})
    final_full_validation = evaluate(model, val.epoch_batches(batch_size))
    checkpoint = run_dir / "model.pt"
    save_checkpoint(checkpoint, model, config["model"], step=step,
                    context_length=config["sequence_length"], tokenizer=DOLMA_TOKENIZER)
    probe = val.batch(offsets["validation"][:1])[:, :-1].to(device)
    with torch.no_grad():
        model.eval()
        expected_logits = model(probe).cpu()
    del optimizer, model
    model, _ = load_checkpoint(checkpoint, device=device)
    with torch.no_grad():
        actual_logits = model(probe).cpu()
    reload_error = (actual_logits - expected_logits).abs().max().item()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    generations = []
    for text, ids in zip(config["prompts"], prompts):
        output = generate(model, torch.tensor([ids]), max_new_tokens=config["max_new_tokens"],
                          context_length=config["sequence_length"],
                          eos_token_id=tokenizer.token_to_id("<|endoftext|>"))
        generations.append({"prompt": text, "token_ids": output[0].tolist(),
                            "text": tokenizer.decode(output[0].tolist(), skip_special_tokens=True)})
    if hashes != {name: file_sha256(split.path) for name, split in splits.items()}:
        raise RuntimeError("source data changed during training")
    if corpus_manifest and file_sha256(Path(corpus_manifest["path"])) != corpus_manifest["sha256"]:
        raise RuntimeError("corpus manifest changed during training")
    session = {
        "start_step": resume_step, "updates": step - resume_step,
        "training_target_tokens": seen - resume_seen,
        "training_seconds": train_seconds, "wall_seconds": time.perf_counter() - wall_start,
        "training_target_tokens_per_second": ((seen - resume_seen) / train_seconds
                                              if train_seconds > 0 else None),
        "peak_cuda_allocated_bytes": peak_bytes,
    }
    result = {
        "experiment": "simple-model-baseline", "config": config, "parameter_count": parameter_count,
        "embedding_parameters": model.embeddings.weight.numel(),
        "transformer_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "dtype": "float32",
        "training_precision": precision, "evaluation_precision": "fp32",
        "matmul_precision": torch.get_float32_matmul_precision(), "tokenizer": DOLMA_TOKENIZER,
        "data": {name: {"path": str(split.path), "tokens": len(split.tokens), "sha256": hashes[name],
                        "evaluation_offsets": offsets[name].tolist()} for name, split in splits.items()},
        "data_unchanged": True, "updates": step, "training_target_tokens": seen,
        "corpus_manifest": corpus_manifest,
        # Legacy recovery files do not contain complete timing/memory measurements.
        # Never present a resumed session's cost as the cost of the whole run.
        **{key: session[key] if resume is None else None for key in (
            "training_seconds", "wall_seconds", "training_target_tokens_per_second",
            "peak_cuda_allocated_bytes")},
        "resume_from": str(resume) if resume is not None else None, "session": session,
        "evaluation_history": evaluations,
        "training_history": history, "initial_full_validation": initial_full_validation,
        "final_full_validation": final_full_validation,
        "checkpoint": str(checkpoint), "checkpoint_reload_max_logit_error": reload_error,
        "generation_method": "greedy", "generations": generations,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    event({"event": "complete", "metrics": str(run_dir / "metrics.json"),
           "full_validation": final_full_validation, "session_training_seconds": train_seconds})
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, help="recovery.pt to continue in a new run directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "baseline/olmo3/mini/data")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU run")
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.resume is not None:
        recovery = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        config = recovery["config"]
        del recovery
    else:
        config = json.loads(Path(__file__).with_name("baseline_config.json").read_text(encoding="utf-8"))
    run(config, args.data_dir, device=args.device, output_root=ROOT / "runs", resume=args.resume)


if __name__ == "__main__":
    main()
