"""Compare manual and packed SDPA attention at fixed BF16 training precision."""

import argparse
import copy
import itertools
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.mixed_precision_benchmark import trial
from reimplementation.data import open_splits
from reimplementation.loss import cross_entropy_loss
from reimplementation.model import CausalLanguageModel
from reimplementation.precision import training_autocast, validate_precision
from reimplementation.train_corpus import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    validate_precision("bf16", "cuda")
    torch.set_num_threads(1)
    baseline = json.loads((ROOT / "experiments/bf16_config.json").read_text())
    optimized = json.loads((ROOT / "experiments/sdpa_config.json").read_text())
    train, val = open_splits(ROOT / "baseline/olmo3/mini/data", baseline["sequence_length"],
                             baseline["model"]["vocab_size"])
    count = args.steps + args.warmup
    batches = list(itertools.islice(train.epoch_batches(baseline["batch_size"],
                                    np.random.default_rng(baseline["seed"])), count))
    if len(batches) != count:
        parser.error("benchmark exceeds the first epoch")
    validation = [val.batch(val.evaluation_starts(4))]
    total_steps = baseline["epochs"] * train.epoch_batch_count(baseline["batch_size"])
    hashes = {name: file_sha256(split.path) for name, split in (("train", train), ("val", val))}
    trials = []
    for backend in ("manual", "sdpa", "sdpa", "manual"):
        config = copy.deepcopy(optimized if backend == "sdpa" else baseline)
        record = trial(config, batches, validation, total_steps, "bf16", args.warmup)
        record["attention_backend"] = backend
        trials.append(record)
        print(json.dumps({key: record[key] for key in ("attention_backend", "seconds",
                          "target_tokens_per_second", "peak_cuda_allocated_bytes")}), flush=True)
    assert len({record["initial_state_sha256"] for record in trials}) == 1
    assert hashes == {name: file_sha256(split.path) for name, split in (("train", train), ("val", val))}

    # Identify the automatically selected kernel on the actual model and batch.
    model = CausalLanguageModel(**optimized["model"]).cuda()
    tokens = batches[0].cuda()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        with training_autocast("cuda", "bf16"):
            loss = cross_entropy_loss(model(tokens[:, :-1]), tokens[:, 1:])
        loss.backward()
        torch.cuda.synchronize()
    kernels = sorted({event.key for event in profile.key_averages()
                      if "scaled_dot_product" in event.key or "flash" in event.key.lower()})
    summary = {backend: {
        "median_target_tokens_per_second": statistics.median(
            r["target_tokens_per_second"] for r in trials if r["attention_backend"] == backend),
        "max_peak_cuda_allocated_bytes": max(
            r["peak_cuda_allocated_bytes"] for r in trials if r["attention_backend"] == backend),
    } for backend in ("manual", "sdpa")}
    run_dir = Path(tempfile.mkdtemp(prefix="sdpa-benchmark-", dir=ROOT / "runs"))
    sources = (list((ROOT / "reimplementation").glob("*.py"))
               + [Path(__file__), ROOT / "experiments/mixed_precision_benchmark.py",
                  ROOT / "evaluation/language_model.py"])
    result = {"experiment": "short-manual-sdpa-comparison", "reference_config": baseline,
              "sdpa_config": optimized, "hardware": torch.cuda.get_device_name(),
              "torch_version": str(torch.__version__), "matmul_precision": torch.get_float32_matmul_precision(),
              "data_sha256": hashes, "source_sha256": {
                  str(path.relative_to(ROOT)): file_sha256(path) for path in sources},
              "validation_targets": 4 * baseline["sequence_length"],
              "trials": trials, "summary": summary, "profiled_attention_operations": kernels,
              "scope": "BF16 short training benchmark; FP32 validation on 1024 targets. "
                       "Same initial weights and batches. CPU-to-GPU transfers are timed; "
                       "initialization, CPU batch preparation, warmup, validation and profiling are not."}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Attention operations:", kernels, flush=True)
    print("Metrics:", run_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
