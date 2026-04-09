"""Cross-Entropy Method planner in latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM, RSSMState
from mbrl.planners.base import Planner, PlanningResult


@dataclass(frozen=True)
class CEMConfig:
    """Hyperparameters controlling CEM planning behavior."""

    horizon: int = 8
    num_samples: int = 128
    num_iterations: int = 5
    elite_fraction: float = 0.1
    min_std: float = 0.05
    discount: float = 0.99
    action_low: float = -2.0
    action_high: float = 2.0


class CEMPlanner(Planner):
    """Plan by iteratively fitting a Gaussian over elite action sequences."""

    def __init__(
        self,
        rssm: RSSM,
        reward_predictor: RewardPredictor,
        action_dim: int,
        device: torch.device,
        config: CEMConfig | None = None,
    ) -> None:
        self.rssm = rssm.to(device)
        self.reward_predictor = reward_predictor.to(device)
        self.action_dim = action_dim
        self.device = device
        self.config = config or CEMConfig()

    @torch.no_grad()
    def plan(self, initial_latent: torch.Tensor) -> PlanningResult:
        self.rssm.eval()
        self.reward_predictor.eval()

        latent = initial_latent.to(self.device)
        if latent.ndim != 2 or latent.shape[0] != 1:
            raise ValueError("initial_latent must have shape (1, latent_dim)")

        mean = torch.zeros(self.config.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.config.horizon, self.action_dim, device=self.device)

        elite_count = max(1, int(self.config.num_samples * self.config.elite_fraction))
        best_sequence = mean.clone()
        best_return = float("-inf")

        for _ in range(self.config.num_iterations):
            noise = torch.randn(
                self.config.num_samples,
                self.config.horizon,
                self.action_dim,
                device=self.device,
            )
            action_sequences = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            action_sequences = action_sequences.clamp(
                self.config.action_low,
                self.config.action_high,
            )

            returns = self._evaluate_action_sequences(latent, action_sequences)
            top_values, top_indices = torch.topk(returns, k=elite_count, dim=0)
            elite = action_sequences[top_indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(self.config.min_std)

            if float(top_values[0].item()) > best_return:
                best_return = float(top_values[0].item())
                best_sequence = action_sequences[top_indices[0]]

        return PlanningResult(action=best_sequence[0].detach().cpu(), predicted_return=best_return)

    def _evaluate_action_sequences(
        self,
        initial_latent: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action_sequences.shape[0]
        repeated_latent = initial_latent.repeat(batch_size, 1)
        state = self.rssm.initial_state(batch_size=batch_size, device=self.device)
        state = RSSMState(
            deterministic=state.deterministic,
            stochastic=repeated_latent,
            prior_mean=state.prior_mean,
            prior_logvar=state.prior_logvar,
        )

        returns = torch.zeros(batch_size, device=self.device)
        discount = 1.0
        for step in range(self.config.horizon):
            actions = action_sequences[:, step, :]
            state = self.rssm.imagine_step(state, actions)
            rewards = self.reward_predictor(state.stochastic).squeeze(-1)
            returns = returns + discount * rewards
            discount *= self.config.discount
        return returns
