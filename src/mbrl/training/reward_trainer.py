"""Training utilities for latent reward prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.reward import RewardPredictor, reward_prediction_loss
from mbrl.models.vae import VAE


@dataclass(frozen=True)
class RewardTrainingResult:
    """Summary returned by reward predictor training."""

    epoch_losses: list[float]
    checkpoint_path: Path
    comparison_path: Path


class RewardTrainer:
    """Supervised trainer that learns rewards from frozen VAE latents."""

    def __init__(
        self,
        predictor: RewardPredictor,
        vae: VAE,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.predictor = predictor.to(device)
        self.vae = vae.to(device)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.optimizer = Adam(self.predictor.parameters(), lr=learning_rate)
        self.device = device

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.predictor.train()
        observations = batch["obs"].to(self.device)
        rewards = batch["rewards"].to(self.device)

        with torch.no_grad():
            latents, _ = self.vae.encode(observations)

        predicted_rewards = self.predictor(latents)
        loss = reward_prediction_loss(predicted_rewards, rewards)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"total": float(loss.item())}

    def fit(
        self,
        replay_buffer: ReplayBuffer,
        epochs: int,
        batch_size: int,
        steps_per_epoch: int,
        checkpoint_path: str | Path,
        comparison_path: str | Path,
        seed: int,
    ) -> RewardTrainingResult:
        """Train the predictor and save checkpoint and validation visualization."""

        losses: list[float] = []
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for step in range(steps_per_epoch):
                batch = replay_buffer.sample(
                    batch_size=batch_size,
                    seed=seed + epoch * 100 + step,
                )
                step_losses = self.train_step(batch)
                epoch_losses.append(step_losses["total"])
            mean_epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
            losses.append(mean_epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} reward_loss={mean_epoch_loss:.6f}")

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.predictor.state_dict()}, checkpoint_file)

        validation_batch = replay_buffer.sample(
            batch_size=min(batch_size, len(replay_buffer)),
            seed=seed,
        )
        self._save_reward_comparison(validation_batch, comparison_path)

        return RewardTrainingResult(
            epoch_losses=losses,
            checkpoint_path=checkpoint_file,
            comparison_path=Path(comparison_path),
        )

    @torch.no_grad()
    def _save_reward_comparison(
        self,
        batch: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> None:
        self.predictor.eval()
        observations = batch["obs"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        latents, _ = self.vae.encode(observations)
        predicted = self.predictor(latents)

        actual_values = rewards.squeeze(-1).cpu().numpy()
        predicted_values = predicted.squeeze(-1).cpu().numpy()

        figure = plt.figure(figsize=(6, 5))
        axis = figure.add_subplot(1, 1, 1)
        axis.scatter(actual_values, predicted_values, alpha=0.7)
        min_value = float(min(actual_values.min(), predicted_values.min()))
        max_value = float(max(actual_values.max(), predicted_values.max()))
        axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
        axis.set_xlabel("Actual reward")
        axis.set_ylabel("Predicted reward")
        axis.set_title("Reward predictor validation")
        figure.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)
        plt.close(figure)
