"""Load a checkpoint and continue a prompt one token at a time."""

import argparse
from pathlib import Path

import torch
from torch import nn

from reimplementation.checkpoint import load_checkpoint
from reimplementation.tokenizer import load_tokenizer


@torch.no_grad()
def generate(model: nn.Module, token_ids: torch.Tensor, *, max_new_tokens: int,
             context_length: int, eos_token_id: int | None = None) -> torch.Tensor:
    """Greedy decoding of one nonempty prompt; recompute its context each time."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
        raise ValueError("generation expects one nonempty prompt [1, positions]")
    if max_new_tokens < 0 or context_length < 1:
        raise ValueError("max_new_tokens must be nonnegative and context_length positive")
    was_training = model.training
    model.eval()
    tokens = token_ids.to(next(model.parameters()).device)
    try:
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -context_length:])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="The purpose of science is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative")
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(args.checkpoint, device=args.device)
    tokenizer = load_tokenizer(metadata["tokenizer"])
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not ids:
        parser.error("--prompt must encode to at least one token")
    tokens = generate(
        model, torch.tensor([ids]), max_new_tokens=args.max_new_tokens,
        context_length=metadata["context_length"],
        eos_token_id=tokenizer.token_to_id("<|endoftext|>"),
    )
    print(tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
