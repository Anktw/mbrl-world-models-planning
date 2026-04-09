from pathlib import Path

import numpy as np
import torch

from mbrl.data.buffer import ReplayBuffer, Transition
from mbrl.models.vae import VAE
from mbrl.training.vae_trainer import VAETrainer


def _image(value: float) -> np.ndarray:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[:, 16:48, 16:48] = value
    return image


def _populate_buffer(buffer: ReplayBuffer, count: int) -> None:
    for index in range(count):
        image = _image(float(index % 2))
        buffer.add(
            Transition(
                observation=image,
                action=np.array([0.0], dtype=np.float32),
                reward=0.0,
                next_observation=image,
                done=False,
            )
        )


def test_vae_trainer_reduces_loss_and_saves_artifacts(tmp_path: Path) -> None:
    buffer = ReplayBuffer(capacity=32, obs_shape=(3, 64, 64), action_dim=1)
    _populate_buffer(buffer, 32)

    model = VAE(image_channels=3, hidden_dim=64, latent_dim=8)
    trainer = VAETrainer(model=model, learning_rate=1e-3, kl_beta=0.1, device=torch.device("cpu"))

    result = trainer.fit(
        replay_buffer=buffer,
        epochs=3,
        batch_size=8,
        steps_per_epoch=4,
        checkpoint_path=tmp_path / "vae.pt",
        reconstruction_path=tmp_path / "vae.png",
        seed=7,
    )

    assert len(result.epoch_losses) == 3
    assert result.epoch_losses[-1] <= result.epoch_losses[0]
    assert result.checkpoint_path.exists()
    assert result.reconstruction_path.exists()
