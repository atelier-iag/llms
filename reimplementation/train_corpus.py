"""Short single-device training run on the baseline's existing token files."""

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "baseline/olmo3/mini/data")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prompt", default="The purpose of science is")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "sequence_length", "d_model", "num_heads",
                 "num_layers", "eval_batches"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads:
        parser.error("--d-model must be divisible by --num-heads")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU run")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        parser.error("--prompt must encode to at least one token")
    config = dict(vocab_size=tokenizer.get_vocab_size(), d_model=args.d_model,
                  num_heads=args.num_heads, hidden_size=4 * args.d_model,
                  num_layers=args.num_layers)
    train, val = open_splits(args.data_dir, args.sequence_length, config["vocab_size"])
    splits = {"train": train, "validation": val}
    starts = {name: split.evaluation_starts(args.eval_batches * args.batch_size)
              for name, split in splits.items()}
    data_info = {
        name: {"path": str(split.path), "tokens": len(split.tokens),
               "sha256": file_sha256(split.path), "evaluation_starts": starts[name].tolist()}
        for name, split in splits.items()
    }
    model = CausalLanguageModel(**config).to(args.device)
    optimizer_config = dict(lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)

    def measure():
        metrics = {}
        for name, split in splits.items():
            offsets = starts[name]
            batches = (split.batch(offsets[i:i + args.batch_size])
                       for i in range(0, len(offsets), args.batch_size))
            metrics[name] = evaluate(model, batches)
        return metrics

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="reimplementation-corpus-", dir=runs))
    print(f"Run: {run_dir}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    before = measure()
    print(f"Initial validation: {before['validation']}", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses_before_update = []
    for step in range(1, args.steps + 1):
        tokens = train.sample(rng, args.batch_size).to(args.device)
        loss = train_step(model, optimizer, tokens)
        losses_before_update.append(loss)
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"Update {step}: loss before update = {loss:.6f}", flush=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    after = measure()
    checkpoint_path = run_dir / "model.pt"
    save_checkpoint(checkpoint_path, model, config, step=args.steps,
                    context_length=args.sequence_length, tokenizer=DOLMA_TOKENIZER)

    # Verify reconstruction from the saved file, rather than generating from memory only.
    probe = val.batch(starts["validation"][:1])[:, :-1].to(args.device)
    with torch.no_grad():
        model.eval()
        expected_logits = model(probe).cpu()
    del optimizer, model
    model, metadata = load_checkpoint(checkpoint_path, device=args.device)
    with torch.no_grad():
        actual_logits = model(probe).cpu()
    reload_max_error = (actual_logits - expected_logits).abs().max().item()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    generated = generate(model, torch.tensor([prompt_ids]), max_new_tokens=32,
                         context_length=metadata["context_length"],
                         eos_token_id=tokenizer.token_to_id("<|endoftext|>"))
    for name, split in splits.items():
        if file_sha256(split.path) != data_info[name]["sha256"]:
            raise RuntimeError(f"Source token file changed during the run: {split.path}")
    result = {
        "experiment": "short-corpus-pipeline-check", "seed": args.seed,
        "device": args.device,
        "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "dtype": "float32", "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "tokenizer": DOLMA_TOKENIZER,
        "model_config": config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": {"name": "AdamW", **optimizer_config},
        "updates": args.steps, "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "training_target_tokens": args.steps * args.batch_size * args.sequence_length,
        "training_seconds": training_seconds, "peak_cuda_allocated_bytes": peak_bytes,
        "data": data_info, "data_unchanged": True,
        "before": before, "after": after, "losses_before_update": losses_before_update,
        "checkpoint": str(checkpoint_path), "checkpoint_reload_max_logit_error": reload_max_error,
        "generation": {"method": "greedy", "prompt": args.prompt,
                       "max_new_tokens": 32, "token_ids": generated[0].tolist(),
                       "text": tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Final validation: {after['validation']}", flush=True)
    print(f"Checkpoint reload max logit error: {reload_max_error}")
    print(f"Generated text: {result['generation']['text']}")
    print(f"Training seconds: {training_seconds:.2f}")
    print(f"Metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
