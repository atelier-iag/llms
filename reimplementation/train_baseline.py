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
from reimplementation.data import open_splits
from reimplementation.generate import generate
from reimplementation.model import CausalLanguageModel
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


def run(config: dict, data_dir: Path, *, device: str, output_root: Path) -> Path:
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    if tokenizer.get_vocab_size() != config["model"]["vocab_size"]:
        raise ValueError("model vocabulary must match the tokenizer")
    prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in config["prompts"]]
    if any(not ids for ids in prompts):
        raise ValueError("generation prompts must be nonempty")
    batch_size = config["batch_size"]
    train, val = open_splits(data_dir, config["sequence_length"], config["model"]["vocab_size"])
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
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="simple-baseline-", dir=output_root))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    model = CausalLanguageModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])

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
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device == "cuda" else [],
        }
        temporary = run_dir / "recovery.tmp"
        torch.save(state, temporary)
        temporary.replace(run_dir / "recovery.pt")

    parameter_count = sum(p.numel() for p in model.parameters())
    event({"event": "start", "run_dir": str(run_dir), "parameters": parameter_count,
           "total_updates": total_steps, "target_tokens": config["epochs"] * (len(train.tokens) - 1)})
    wall_start = time.perf_counter()
    initial_samples = measure_samples()
    event({"event": "validation_full_start"})
    initial_full_validation = evaluate(model, val.epoch_batches(batch_size))
    event({"event": "initial", "samples": initial_samples, "full_validation": initial_full_validation})
    evaluations = [{"step": 0, "target_tokens": 0, **initial_samples}]
    history = []
    seen = step = 0
    train_seconds = 0.0
    window_nll = 0.0
    window_tokens = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    segment_start = time.perf_counter()
    for epoch in range(config["epochs"]):
        rng = np.random.default_rng(config["seed"] + epoch)
        for tokens in train.epoch_batches(batch_size, rng):
            step += 1
            lr = learning_rate(step, total_steps, config["optimizer"]["lr"],
                               config["warmup_updates"], config["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = train_step(model, optimizer, tokens.to(device), max_grad_norm=config["max_grad_norm"])
            count = tokens.shape[0] * (tokens.shape[1] - 1)
            seen += count
            window_tokens += count
            window_nll += loss * count
            if step % config["log_every"] == 0 or step == total_steps:
                record = {"event": "train", "step": step, "target_tokens": seen,
                          "lr": lr, "mean_training_loss": window_nll / window_tokens,
                          "window_target_tokens": window_tokens,
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
    result = {
        "experiment": "simple-model-baseline", "config": config, "parameter_count": parameter_count,
        "embedding_parameters": model.embeddings.weight.numel(),
        "transformer_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "dtype": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(), "tokenizer": DOLMA_TOKENIZER,
        "data": {name: {"path": str(split.path), "tokens": len(split.tokens), "sha256": hashes[name],
                        "evaluation_offsets": offsets[name].tolist()} for name, split in splits.items()},
        "data_unchanged": True, "updates": step, "training_target_tokens": seen,
        "training_seconds": train_seconds, "wall_seconds": time.perf_counter() - wall_start,
        "training_target_tokens_per_second": seen / train_seconds,
        "peak_cuda_allocated_bytes": peak_bytes, "evaluation_history": evaluations,
        "training_history": history, "initial_full_validation": initial_full_validation,
        "final_full_validation": final_full_validation,
        "checkpoint": str(checkpoint), "checkpoint_reload_max_logit_error": reload_error,
        "generation_method": "greedy", "generations": generations,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    event({"event": "complete", "metrics": str(run_dir / "metrics.json"),
           "full_validation": final_full_validation, "training_seconds": train_seconds})
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("baseline_config.json"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "baseline/olmo3/mini/data")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU run")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args.data_dir, device=args.device, output_root=ROOT / "runs")


if __name__ == "__main__":
    main()
