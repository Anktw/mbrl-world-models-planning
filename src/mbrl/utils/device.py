"""Device selection helpers."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Resolve requested device name to a torch.device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
