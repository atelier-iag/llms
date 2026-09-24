"""Training precision policy: FP32 parameters, optional BF16 autocast."""

import torch


def validate_precision(precision: str, device: str | torch.device) -> None:
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'")
    device = torch.device(device)
    if precision == "bf16":
        if device.type not in ("cpu", "cuda"):
            raise ValueError("BF16 training supports CPU and CUDA only")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("BF16 CUDA training requires CUDA")
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("this CUDA device has no native BF16 support")


def training_autocast(device: str | torch.device, precision: str):
    # Capability is checked once by the runner; avoid querying the GPU per step.
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'")
    return torch.autocast(device_type=torch.device(device).type,
                          dtype=torch.bfloat16, enabled=precision == "bf16")
