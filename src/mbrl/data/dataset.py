"""Torch dataset helpers for transition data."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class TransitionTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper over a sampled replay batch."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        sizes = {tensor.shape[0] for tensor in batch.values()}
        if len(sizes) != 1:
            raise ValueError("All tensors in batch must share first dimension")
        self.batch = batch
        self.size = next(iter(sizes))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {name: tensor[index] for name, tensor in self.batch.items()}
