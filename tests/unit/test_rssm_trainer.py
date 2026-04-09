from pathlib import Path

import numpy as np
import torch

from mbrl.data.buffer import ReplayBuffer, Transition
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.training.rssm_trainer import RSSMTrainer


def _image(value: float) -> np.ndarray:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[:, 20:44, 20:44] = value
    return image


def _populate_buffer(buffer: ReplayBuffer, count: int) -> None:
    image = _image(0.5)
    for _ in range(count):
        buffer.add(
            Transition(
                observation=image,
                action=np.array([0.0], dtype=np.float32),
                reward=0.0,
                next_observation=image,
                done=False,
            )
        )


def test_rssm_trainer_reduces_loss_and_saves_artifacts(tmp_path: Path) -> None:
    buffer = ReplayBuffer(capacity=32, obs_shape=(3, 64, 64), action_dim=1)
    _populate_buffer(buffer, 32)

    vae = VAE(image_channels=3, hidden_dim=64, latent_dim=8)
    rssm = RSSM(latent_dim=8, action_dim=1, deterministic_dim=32, hidden_dim=64)
    trainer = RSSMTrainer(rssm=rssm, vae=vae, learning_rate=1e-3, device=torch.device("cpu"))

    result = trainer.fit(
        replay_buffer=buffer,
        epochs=3,
        batch_size=4,
        sequence_length=4,
        steps_per_epoch=3,
        checkpoint_path=tmp_path / "rssm.pt",
        comparison_path=tmp_path / "rssm.png",
        seed=7,
    )

    assert len(result.epoch_losses) == 3
    assert result.epoch_losses[-1] <= result.epoch_losses[0]
    assert result.checkpoint_path.exists()
    assert result.comparison_path.exists()
