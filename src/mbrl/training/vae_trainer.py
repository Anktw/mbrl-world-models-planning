"""Training utilities for the convolutional VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.vae import VAE, vae_loss


@dataclass(frozen=True)
class VAETrainingResult:
    """Summary returned by the VAE training loop."""

    epoch_losses: list[float]
    checkpoint_path: Path
    reconstruction_path: Path


class VAETrainer:
    """Minimal trainer that samples directly from the replay buffer."""

    def __init__(
        self,
        model: VAE,
        learning_rate: float,
        kl_beta: float,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.kl_beta = kl_beta
        self.device = device

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()
        inputs = batch.to(self.device)
        outputs = self.model(inputs)
        losses = vae_loss(
            reconstruction=outputs["reconstruction"],
            inputs=inputs,
            mean=outputs["mean"],
            logvar=outputs["logvar"],
            kl_beta=self.kl_beta,
        )

        self.optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        self.optimizer.step()

        return {
            "total": float(losses.total.item()),
            "reconstruction": float(losses.reconstruction.item()),
            "kl": float(losses.kl.item()),
        }

    def fit(
        self,
        replay_buffer: ReplayBuffer,
        epochs: int,
        batch_size: int,
        steps_per_epoch: int,
        checkpoint_path: str | Path,
        reconstruction_path: str | Path,
        seed: int,
    ) -> VAETrainingResult:
        """Train on replay buffer samples and save artifacts."""

        losses: list[float] = []
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for step in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size=batch_size, seed=seed + epoch * 100 + step)
                step_losses = self.train_step(batch["obs"])
                epoch_losses.append(step_losses["total"])
            mean_epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
            losses.append(mean_epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} loss={mean_epoch_loss:.6f}")

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, checkpoint_file)

        sample_batch = replay_buffer.sample(
            batch_size=min(batch_size, len(replay_buffer)),
            seed=seed,
        )
        self._save_reconstructions(sample_batch["obs"], reconstruction_path)

        return VAETrainingResult(
            epoch_losses=losses,
            checkpoint_path=checkpoint_file,
            reconstruction_path=Path(reconstruction_path),
        )

    @torch.no_grad()
    def _save_reconstructions(self, inputs: torch.Tensor, output_path: str | Path) -> None:
        self.model.eval()
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        reconstructions = outputs["reconstruction"].cpu()
        originals = inputs.cpu()

        count = min(6, originals.shape[0])
        figure, axes = plt.subplots(2, count, figsize=(count * 2, 4))
        if count == 1:
            axes = [[axes[0]], [axes[1]]]

        for index in range(count):
            original = originals[index].permute(1, 2, 0).numpy()
            reconstruction = reconstructions[index].permute(1, 2, 0).numpy()
            axes[0][index].imshow(original)
            axes[0][index].axis("off")
            axes[0][index].set_title("Original")
            axes[1][index].imshow(reconstruction)
            axes[1][index].axis("off")
            axes[1][index].set_title("Reconstruction")

        figure.tight_layout()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)
        plt.close(figure)
