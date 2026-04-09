"""Training utilities for the RSSM latent dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.optim import Adam

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE


@dataclass(frozen=True)
class RSSMTrainingResult:
    """Summary returned by the RSSM trainer."""

    epoch_losses: list[float]
    checkpoint_path: Path
    comparison_path: Path


class RSSMTrainer:
    """Sequence trainer that predicts future latents from replay buffer rollouts."""

    def __init__(
        self,
        rssm: RSSM,
        vae: VAE,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.rssm = rssm.to(device)
        self.vae = vae.to(device)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.optimizer = Adam(self.rssm.parameters(), lr=learning_rate)
        self.device = device

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.rssm.train()
        obs_sequences = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)

        batch_size, sequence_length_plus_one = obs_sequences.shape[:2]
        flat_obs = obs_sequences.reshape(-1, *obs_sequences.shape[2:])
        with torch.no_grad():
            encoded_mean, _ = self.vae.encode(flat_obs)
        latents = encoded_mean.reshape(batch_size, sequence_length_plus_one, -1)

        predicted_latents, _ = self.rssm.rollout_sequence(
            initial_latent=latents[:, 0],
            actions=actions,
        )
        targets = latents[:, 1:]
        loss = F.mse_loss(predicted_latents, targets, reduction="mean")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"total": float(loss.item())}

    def fit(
        self,
        replay_buffer: ReplayBuffer,
        epochs: int,
        batch_size: int,
        sequence_length: int,
        steps_per_epoch: int,
        checkpoint_path: str | Path,
        comparison_path: str | Path,
        seed: int,
    ) -> RSSMTrainingResult:
        """Train the RSSM and save a checkpoint plus latent comparison figure."""

        losses: list[float] = []
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for step in range(steps_per_epoch):
                batch = replay_buffer.sample_sequences(
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    seed=seed + epoch * 100 + step,
                )
                step_losses = self.train_step(batch)
                epoch_losses.append(step_losses["total"])
            mean_epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
            losses.append(mean_epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} rssm_loss={mean_epoch_loss:.6f}")

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.rssm.state_dict()}, checkpoint_file)

        validation_batch = replay_buffer.sample_sequences(
            sequence_length=sequence_length,
            batch_size=min(batch_size, len(replay_buffer)),
            seed=seed,
        )
        self._save_latent_comparison(validation_batch, comparison_path)

        return RSSMTrainingResult(
            epoch_losses=losses,
            checkpoint_path=checkpoint_file,
            comparison_path=Path(comparison_path),
        )

    @torch.no_grad()
    def _save_latent_comparison(
        self,
        batch: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> None:
        self.rssm.eval()
        obs_sequences = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        batch_size, sequence_length_plus_one = obs_sequences.shape[:2]
        flat_obs = obs_sequences.reshape(-1, *obs_sequences.shape[2:])
        encoded_mean, _ = self.vae.encode(flat_obs)
        latents = encoded_mean.reshape(batch_size, sequence_length_plus_one, -1)
        predicted_latents, _ = self.rssm.rollout_sequence(
            initial_latent=latents[:, 0],
            actions=actions,
        )

        actual = latents[0, 1:, 0].cpu().numpy()
        predicted = predicted_latents[0, :, 0].cpu().numpy()
        figure = plt.figure(figsize=(8, 4))
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(actual, label="actual latent[0]")
        axis.plot(predicted, label="predicted latent[0]", linestyle="--")
        axis.set_title("RSSM latent rollout comparison")
        axis.set_xlabel("Step")
        axis.set_ylabel("Latent value")
        axis.legend()
        figure.tight_layout()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)
        plt.close(figure)
