"""Small runtime helpers for PULSE experiments."""
from __future__ import annotations

import gc

import torch


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rebuild_weight(payload: dict, layer: int, key: str = "pulse_weight") -> torch.Tensor:
    """Rebuild a serialized PULSE weight tensor from a payload JSON object."""
    weight_tensors = payload["per_layer"][str(layer)]["weight_tensors"]
    return torch.tensor(weight_tensors[key], dtype=torch.float32)
