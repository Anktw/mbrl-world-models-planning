"""Lightweight reward predictor over latent vectors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RewardPredictor(nn.Module):
    """Small MLP that maps latent z to scalar reward."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.network(latents)


def reward_prediction_loss(
    predicted_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error loss for reward regression."""

    return F.mse_loss(predicted_rewards, target_rewards, reduction="mean")
