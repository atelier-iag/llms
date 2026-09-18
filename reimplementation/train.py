"""Train the tiny model on one fixed batch to check that it can learn."""

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from reimplementation.loss import cross_entropy_loss, make_next_token_batch
from reimplementation.model import CausalLanguageModel


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, tokens: torch.Tensor,
    *, max_grad_norm: float | None = None,
) -> float:
    """Apply one weight update and return the loss measured before that update."""
    model.train()
    inputs, targets = make_next_token_batch(tokens)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss is not finite")
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")

    torch.manual_seed(args.seed)
    # A single CPU thread is sufficient for this tiny, reproducible exercise.
    torch.set_num_threads(1)
    config = dict(vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2)
    model = CausalLanguageModel(**config)
    tokens = torch.tensor([[2, 5, 2, 4]])
    inputs, targets = make_next_token_batch(tokens)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
    )

    losses_before_update = []
    for step in range(1, args.steps + 1):
        loss = train_step(model, optimizer, tokens)
        losses_before_update.append(loss)
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"Update {step:3d}: loss before update = {loss:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        final_loss = cross_entropy_loss(logits, targets).item()
        predictions = logits.argmax(dim=-1)
        target_probabilities = torch.softmax(logits, dim=-1).gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

    result = {
        "experiment": "fixed-tiny-batch-memorization",
        "device": "cpu",
        "dtype": "float32",
        "torch_version": str(torch.__version__),
        "seed": args.seed,
        "model_config": config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": {
            "name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0
        },
        "updates": args.steps,
        "tokens": tokens.tolist(),
        "inputs": inputs.tolist(),
        "targets": targets.tolist(),
        "initial_loss": losses_before_update[0],
        "final_loss": final_loss,
        "losses_before_update": losses_before_update,
        "final_predictions": predictions.tolist(),
        "final_target_probabilities": target_probabilities.tolist(),
    }
    runs = Path(__file__).resolve().parents[1] / "runs"
    runs.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="reimplementation-tiny-batch-", dir=runs))
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Initial loss: {result['initial_loss']:.6f}")
    print(f"Loss after {args.steps} updates: {final_loss:.6f}")
    print("Expected next tokens:", targets.tolist())
    print("Predicted next tokens:", predictions.tolist())
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
