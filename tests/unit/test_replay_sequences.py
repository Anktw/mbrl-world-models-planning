import numpy as np

from mbrl.data.buffer import ReplayBuffer, Transition


def _transition(index: int, done: bool = False) -> Transition:
    obs = np.full((3,), fill_value=float(index), dtype=np.float32)
    next_obs = np.full((3,), fill_value=float(index + 1), dtype=np.float32)
    return Transition(
        observation=obs,
        action=np.array([0.0], dtype=np.float32),
        reward=float(index),
        next_observation=next_obs,
        done=done,
    )


def test_sample_sequences_shapes() -> None:
    buffer = ReplayBuffer(capacity=20, obs_shape=(3,), action_dim=1)
    for index in range(12):
        buffer.add(_transition(index, done=False))

    batch = buffer.sample_sequences(sequence_length=4, batch_size=3, seed=7)

    assert tuple(batch["obs"].shape) == (3, 5, 3)
    assert tuple(batch["actions"].shape) == (3, 4, 1)
    assert tuple(batch["rewards"].shape) == (3, 4, 1)
    assert tuple(batch["next_obs"].shape) == (3, 4, 3)
    assert tuple(batch["dones"].shape) == (3, 4, 1)


def test_sample_sequences_avoids_terminal_crossing() -> None:
    buffer = ReplayBuffer(capacity=20, obs_shape=(3,), action_dim=1)
    for index in range(10):
        buffer.add(_transition(index, done=(index == 4)))

    batch = buffer.sample_sequences(sequence_length=3, batch_size=5, seed=11)
    assert float(batch["dones"][:, :-1].sum().item()) == 0.0
