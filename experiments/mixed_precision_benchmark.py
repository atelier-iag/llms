"""Short FP32/BF16 training benchmark on the real GQA model and corpus."""

import argparse
import gc
import hashlib
import itertools
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.language_model import evaluate
from reimplementation.data import open_splits
from reimplementation.model import CausalLanguageModel
from reimplementation.precision import validate_precision
from reimplementation.train import train_step
from reimplementation.train_baseline import learning_rate
from reimplementation.train_corpus import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def trial(config, batches, validation, total_steps, precision, warmup):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config["seed"])
    model = CausalLanguageModel(**config["model"])
    initial_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(tensor.numpy().tobytes())
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    initial_validation = evaluate(model, validation)
    losses = []
    for index, tokens in enumerate(batches):
        if index == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        lr = learning_rate(index + 1, total_steps, config["optimizer"]["lr"],
                           config["warmup_updates"], config["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses.append(train_step(model, optimizer, tokens.cuda(),
                                 max_grad_norm=config["max_grad_norm"], precision=precision))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated()
    scored = sum(batch.shape[0] * (batch.shape[1] - 1) for batch in batches[warmup:])
    finite = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                 and torch.isfinite(p).all().item() for p in model.parameters())
    if not finite:
        raise FloatingPointError("nonfinite parameters or missing/nonfinite gradients")
    result = {
        "precision": precision, "initial_state_sha256": initial_hash.hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "warmup_updates": warmup, "measured_updates": len(batches) - warmup,
        "measured_target_tokens": scored, "seconds": seconds,
        "target_tokens_per_second": scored / seconds, "peak_cuda_allocated_bytes": peak_bytes,
        "losses_before_updates": losses, "finite_weights_and_gradients": finite,
        "initial_validation_fp32": initial_validation,
        "final_validation_fp32": evaluate(model, validation),
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("warmup, steps and repeats must be positive")
    validate_precision("bf16", "cuda")
    torch.set_num_threads(1)
    config = json.loads((ROOT / "experiments/gqa_config.json").read_text())
    train, val = open_splits(ROOT / "baseline/olmo3/mini/data", config["sequence_length"],
                             config["model"]["vocab_size"])
    total_steps = config["epochs"] * train.epoch_batch_count(config["batch_size"])
    count = args.warmup + args.steps
    batches = list(itertools.islice(train.epoch_batches(config["batch_size"],
                                    np.random.default_rng(config["seed"])), count))
    if len(batches) != count:
        parser.error("benchmark exceeds available first-epoch batches")
    validation = [val.batch(val.evaluation_starts(4))]
    data_hashes = {name: file_sha256(split.path) for name, split in (("train", train), ("val", val))}
    output_root = ROOT / "runs"
    output_root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="precision-benchmark-", dir=output_root))
    trials = []
    for repeat in range(args.repeats):
        order = ("fp32", "bf16") if repeat % 2 == 0 else ("bf16", "fp32")
        for precision in order:
            record = trial(config, batches, validation, total_steps, precision, args.warmup)
            trials.append(record)
            print(json.dumps({key: record[key] for key in
                              ("precision", "seconds", "target_tokens_per_second",
                               "peak_cuda_allocated_bytes")}), flush=True)
    assert len({record["initial_state_sha256"] for record in trials}) == 1
    assert data_hashes == {name: file_sha256(split.path) for name, split in (("train", train), ("val", val))}
    summary = {precision: {
        "median_target_tokens_per_second": statistics.median(
            record["target_tokens_per_second"] for record in trials if record["precision"] == precision),
        "max_peak_cuda_allocated_bytes": max(
            record["peak_cuda_allocated_bytes"] for record in trials if record["precision"] == precision),
    } for precision in ("fp32", "bf16")}
    sources = list((ROOT / "reimplementation").glob("*.py")) + [Path(__file__), ROOT / "evaluation/language_model.py"]
    result = {"experiment": "short-fp32-bf16-comparison", "config": config,
              "torch_version": str(torch.__version__), "hardware": torch.cuda.get_device_name(),
              "matmul_precision": torch.get_float32_matmul_precision(), "data_sha256": data_hashes,
              "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in sources},
              "validation_targets": 4 * config["sequence_length"], "trials": trials, "summary": summary,
              "scope": "Short training throughput/stability check, not a full-corpus quality comparison. "
                       "Timed steps include CPU-to-GPU transfer; CPU batches are prepared beforehand. "
                       "Initialization, warmup and FP32 validation are outside the timer."}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Metrics:", run_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
