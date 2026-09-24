"""Shift an unpadded token batch and measure next-token prediction loss."""

import torch
import torch.nn.functional as F


def make_next_token_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn [batch, positions + 1] tokens into aligned inputs and targets."""
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("tokens must be a batch of sequences with at least two tokens")
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return inputs, targets


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average the negative log probability of the correct token at each position.

    Targets must already be aligned with logits and contain valid vocabulary IDs.
    This first implementation expects unpadded batches, with no ignored targets.
    """
    if logits.ndim != 3 or logits.shape[:2] != targets.shape or targets.numel() == 0:
        raise ValueError("expected nonempty logits [batch, positions, vocab] and aligned targets")
    # Keep log probabilities and the final reduction in FP32 under BF16 autocast.
    # Leave FP32 and FP64 callers unchanged.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    log_probabilities = F.log_softmax(logits, dim=-1)
    correct_log_probabilities = log_probabilities.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return -correct_log_probabilities.mean()


if __name__ == "__main__":
    from reimplementation.model import CausalLanguageModel

    torch.manual_seed(0)
    model = CausalLanguageModel(
        vocab_size=6, d_model=4, num_heads=2, hidden_size=8, num_layers=2
    )
    model.eval()
    tokens = torch.tensor([[2, 5, 2, 4]])
    inputs, targets = make_next_token_batch(tokens)

    with torch.no_grad():
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        # Display ordinary probabilities to explain the loss; the loss itself
        # uses log_softmax directly for numerical stability.
        probabilities = torch.softmax(logits, dim=-1)
        correct_probabilities = probabilities.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

    print("Training excerpt:", tokens)
    print("Inputs:", inputs)
    print("Targets:", targets)
    print("Logits shape:", tuple(logits.shape))
    print("Probability assigned to each correct target:", correct_probabilities)
    print("Mean cross-entropy loss:", loss.item())
