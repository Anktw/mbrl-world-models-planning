"""Planner interfaces and result contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanningResult:
    """Result returned by a planner in latent space."""

    action: torch.Tensor
    predicted_return: float


class Planner(ABC):
    """Abstract planner API for latent-space planning."""

    @abstractmethod
    def plan(self, initial_latent: torch.Tensor) -> PlanningResult:
        """Produce one action from the current latent state."""
