"""Plotting helpers for the evaluation report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_planner_rewards(
    cem_rewards: list[float],
    mcts_rewards: list[float],
    output_path: str | Path,
) -> None:
    """Plot planner returns per episode and average comparison."""

    figure = plt.figure(figsize=(8, 4))
    axis = figure.add_subplot(1, 1, 1)
    episodes = np.arange(1, len(cem_rewards) + 1)
    axis.plot(episodes, cem_rewards, marker="o", label="CEM")
    axis.plot(episodes, mcts_rewards, marker="o", label="MCTS")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Cumulative reward")
    axis.set_title("Planner reward comparison")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_sample_efficiency(
    cumulative_steps: list[int],
    reward_per_step_curve: list[float],
    output_path: str | Path,
) -> None:
    """Plot reward-per-sample progression."""

    figure = plt.figure(figsize=(8, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(cumulative_steps, reward_per_step_curve, marker="o")
    axis.set_xlabel("Cumulative collected steps")
    axis.set_ylabel("Cumulative executed reward / step")
    axis.set_title("Sample efficiency")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_prediction_errors(rssm_mse: float, reward_mse: float, output_path: str | Path) -> None:
    """Plot key model prediction errors."""

    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(1, 1, 1)
    labels = ["RSSM latent MSE", "Reward MSE"]
    values = [rssm_mse, reward_mse]
    axis.bar(labels, values)
    axis.set_ylabel("Error")
    axis.set_title("Prediction errors")
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_latent_space(
    latents_2d: np.ndarray,
    rewards: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot 2D latent embedding colored by reward."""

    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(1, 1, 1)
    scatter = axis.scatter(latents_2d[:, 0], latents_2d[:, 1], c=rewards, cmap="viridis", s=18)
    axis.set_xlabel("Latent component 1")
    axis.set_ylabel("Latent component 2")
    axis.set_title("Latent space visualization")
    figure.colorbar(scatter, ax=axis, label="Reward")
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
