"""Save and reconstruct an inference model, including its architecture."""

from pathlib import Path

import torch

from reimplementation.model import CausalLanguageModel


def save_checkpoint(path: Path, model: CausalLanguageModel, config: dict,
                    *, step: int, context_length: int, tokenizer: dict) -> None:
    checkpoint = {
        "format_version": 1,
        "model_config": config,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": step,
        "context_length": context_length,
        "tokenizer": tokenizer,
    }
    # Refuse to overwrite an existing checkpoint. Each training run gets a new directory.
    with Path(path).open("xb") as stream:
        torch.save(checkpoint, stream)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[CausalLanguageModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("unsupported checkpoint version")
    model = CausalLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint.pop("model_state"), strict=True)
    model.to(device).eval()
    return model, checkpoint
