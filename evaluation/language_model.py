"""Token-weighted next-token loss on an explicitly supplied validation set."""

import math
from collections.abc import Iterable

import torch
from torch import nn

from reimplementation.loss import cross_entropy_loss, make_next_token_batch


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[torch.Tensor]) -> dict:
    was_training = model.training
    device = next(model.parameters()).device
    total_nll = 0.0
    token_count = 0
    model.eval()
    try:
        for tokens in batches:
            inputs, targets = make_next_token_batch(tokens.to(device))
            loss = cross_entropy_loss(model(inputs), targets).item()
            if not math.isfinite(loss):
                raise FloatingPointError("Validation loss is not finite")
            total_nll += loss * targets.numel()
            token_count += targets.numel()
    finally:
        model.train(was_training)
    if not token_count:
        raise ValueError("evaluation requires at least one target token")
    loss = total_nll / token_count
    return {"loss": loss, "perplexity": math.exp(loss), "scored_tokens": token_count}
