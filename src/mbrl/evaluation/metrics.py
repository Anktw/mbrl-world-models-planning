"""Evaluation metric helpers for planner and model assessment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionErrorMetrics:
    """Prediction error metrics for learned model components."""

    rssm_latent_mse: float
    reward_mse: float


@dataclass(frozen=True)
class SampleEfficiencyMetrics:
    """Sample efficiency summary built from cycle logs."""

    collected_steps: list[int]
    cumulative_collected_steps: list[int]
    executed_rewards: list[float]
    cumulative_executed_rewards: list[float]
    reward_per_step_curve: list[float]


def cumulative_reward(episode_rewards: list[float]) -> float:
    """Compute cumulative reward over episodes."""

    if not episode_rewards:
        return 0.0
    return float(np.sum(np.asarray(episode_rewards, dtype=np.float64)))


def sample_efficiency(
    collected_steps: list[int],
    executed_rewards: list[float],
) -> SampleEfficiencyMetrics:
    """Compute reward-per-step progression from cycle-level metrics."""

    if len(collected_steps) != len(executed_rewards):
        raise ValueError("collected_steps and executed_rewards must have equal length")

    cumulative_steps = np.cumsum(np.asarray(collected_steps, dtype=np.int64))
    cumulative_rewards = np.cumsum(np.asarray(executed_rewards, dtype=np.float64))
    reward_per_step = cumulative_rewards / np.maximum(cumulative_steps.astype(np.float64), 1.0)

    return SampleEfficiencyMetrics(
        collected_steps=[int(value) for value in collected_steps],
        cumulative_collected_steps=[int(value) for value in cumulative_steps.tolist()],
        executed_rewards=[float(value) for value in executed_rewards],
        cumulative_executed_rewards=[float(value) for value in cumulative_rewards.tolist()],
        reward_per_step_curve=[float(value) for value in reward_per_step.tolist()],
    )
