"""Recurrent state space model for latent dynamics prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RSSMState:
    """Compact recurrent state containing deterministic and stochastic parts."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_mean: torch.Tensor
    prior_logvar: torch.Tensor


class RSSM(nn.Module):
    """GRU-based RSSM that predicts the next latent state."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        deterministic_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim
        self.hidden_dim = hidden_dim

        self.recurrent = nn.GRUCell(latent_dim + action_dim, deterministic_dim)
        self.prior_network = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.posterior_network = nn.Sequential(
            nn.Linear(deterministic_dim + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Return a zero-initialized recurrent state."""

        zeros = torch.zeros(batch_size, self.deterministic_dim, device=device)
        stochastic = torch.zeros(batch_size, self.latent_dim, device=device)
        prior_mean = torch.zeros(batch_size, self.latent_dim, device=device)
        prior_logvar = torch.zeros(batch_size, self.latent_dim, device=device)
        return RSSMState(
            deterministic=zeros,
            stochastic=stochastic,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a stochastic latent with the reparameterization trick."""

        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mean + noise * standard_deviation

    def prior(self, deterministic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next stochastic latent from the deterministic state."""

        parameters = self.prior_network(deterministic)
        mean, logvar = torch.chunk(parameters, 2, dim=-1)
        return mean, logvar

    def posterior(
        self,
        deterministic: torch.Tensor,
        observation_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Condition the stochastic latent on the deterministic state and encoded observation."""

        inputs = torch.cat([deterministic, observation_latent], dim=-1)
        parameters = self.posterior_network(inputs)
        mean, logvar = torch.chunk(parameters, 2, dim=-1)
        return mean, logvar

    def imagine_step(self, state: RSSMState, action: torch.Tensor) -> RSSMState:
        """Advance the model one step using its own stochastic state."""

        inputs = torch.cat([state.stochastic, action], dim=-1)
        deterministic = self.recurrent(inputs, state.deterministic)
        prior_mean, prior_logvar = self.prior(deterministic)
        stochastic = self.reparameterize(prior_mean, prior_logvar)
        return RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )

    def observe_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        observation_latent: torch.Tensor,
    ) -> RSSMState:
        """Condition the model on an encoded observation while preserving the prior."""

        inputs = torch.cat([state.stochastic, action], dim=-1)
        deterministic = self.recurrent(inputs, state.deterministic)
        prior_mean, prior_logvar = self.prior(deterministic)
        posterior_mean, posterior_logvar = self.posterior(deterministic, observation_latent)
        stochastic = self.reparameterize(posterior_mean, posterior_logvar)
        return RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )

    def rollout_sequence(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, list[RSSMState]]:
        """Roll out a latent sequence and return predicted means for each step."""

        batch_size = initial_latent.shape[0]
        state = RSSMState(
            deterministic=torch.zeros(
                batch_size,
                self.deterministic_dim,
                device=initial_latent.device,
            ),
            stochastic=initial_latent,
            prior_mean=torch.zeros(batch_size, self.latent_dim, device=initial_latent.device),
            prior_logvar=torch.zeros(batch_size, self.latent_dim, device=initial_latent.device),
        )

        predicted_latents: list[torch.Tensor] = []
        states: list[RSSMState] = []
        for action in actions.unbind(dim=1):
            state = self.imagine_step(state, action)
            predicted_latents.append(state.prior_mean)
            states.append(state)

        return torch.stack(predicted_latents, dim=1), states
