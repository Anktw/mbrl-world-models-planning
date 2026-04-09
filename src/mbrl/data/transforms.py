"""Basic data transforms used by training and evaluation."""

from __future__ import annotations

import torch


def normalize_observation(obs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize observations per batch for stable optimization."""

    mean = obs.mean(dim=0, keepdim=True)
    std = obs.std(dim=0, keepdim=True)
    return (obs - mean) / (std + eps)
