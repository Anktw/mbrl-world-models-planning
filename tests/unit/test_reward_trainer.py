from pathlib import Path

import numpy as np
import torch

from mbrl.data.buffer import ReplayBuffer, Transition
from mbrl.models.reward import RewardPredictor
from mbrl.models.vae import VAE
from mbrl.training.reward_trainer import RewardTrainer


def _image(value: float) -> np.ndarray:
    image = np.zeros((3, 64, 64), dtype=np.float32)
    image[:, 12:52, 12:52] = value
    return image


def _populate_buffer(buffer: ReplayBuffer, count: int) -> None:
    for index in range(count):
        reward = 1.5 if index % 2 == 0 else -1.5
        image_value = 0.8 if index % 2 == 0 else 0.2
        image = _image(image_value)
        buffer.add(
            Transition(
                observation=image,
                action=np.array([0.0], dtype=np.float32),
                reward=reward,
                next_observation=image,
                done=False,
            )
        )


def test_reward_trainer_reduces_loss_and_saves_artifacts(tmp_path: Path) -> None:
    buffer = ReplayBuffer(capacity=48, obs_shape=(3, 64, 64), action_dim=1)
    _populate_buffer(buffer, 48)

    vae = VAE(image_channels=3, hidden_dim=64, latent_dim=8)
    predictor = RewardPredictor(latent_dim=8, hidden_dim=16)
    trainer = RewardTrainer(
        predictor=predictor,
        vae=vae,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    )

    result = trainer.fit(
        replay_buffer=buffer,
        epochs=3,
        batch_size=8,
        steps_per_epoch=4,
        checkpoint_path=tmp_path / "reward.pt",
        comparison_path=tmp_path / "reward.png",
        seed=7,
    )

    assert len(result.epoch_losses) == 3
    assert result.epoch_losses[-1] <= result.epoch_losses[0]
    assert result.checkpoint_path.exists()
    assert result.comparison_path.exists()
