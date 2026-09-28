"""Answer-only next-token loss, including EOS and excluding prompt/right padding."""

import math

import torch
import torch.nn.functional as F

from reimplementation.precision import training_autocast
from reimplementation.sft_data import IGNORE_INDEX


def completion_loss(logits, labels):
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError("expected aligned logits and labels")
    mask = labels != IGNORE_INDEX
    if not mask.any():
        raise ValueError("completion loss requires supervised response tokens")
    selected = logits[mask]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    return F.cross_entropy(selected, labels[mask])


def sft_step(model, optimizer, batch, *, precision="fp32", max_grad_norm=1.0):
    inputs, labels = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with training_autocast(inputs.device, precision):
        loss = completion_loss(model(inputs), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("SFT loss is not finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item()


@torch.no_grad()
def evaluate_sft(model, batches):
    was_training = model.training
    device = next(model.parameters()).device
    total_nll = tokens = correct = 0
    model.eval()
    try:
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            mask = labels != IGNORE_INDEX
            count = int(mask.sum())
            loss = completion_loss(logits, labels).item()
            if not math.isfinite(loss):
                raise FloatingPointError("SFT evaluation loss is not finite")
            total_nll += loss * count
            tokens += count
            correct += int((logits[mask].argmax(dim=-1) == labels[mask]).sum())
            del logits  # Do not keep a full vocabulary tensor during the next forward.
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("SFT evaluation requires response tokens")
    loss = total_nll / tokens
    return {"loss": loss, "perplexity": math.exp(loss), "scored_tokens": tokens,
            "teacher_forced_token_accuracy": correct / tokens}
