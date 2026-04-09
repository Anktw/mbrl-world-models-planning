import numpy as np

from mbrl.data.buffer import ReplayBuffer, Transition


def _transition(value: float) -> Transition:
    obs = np.array([value, value + 1.0, value + 2.0], dtype=np.float32)
    action = np.array([0.25], dtype=np.float32)
    next_obs = obs + 0.5
    return Transition(
        observation=obs,
        action=action,
        reward=float(value),
        next_observation=next_obs,
        done=False,
    )


def test_replay_buffer_add_and_length() -> None:
    buffer = ReplayBuffer(capacity=3, obs_shape=(3,), action_dim=1)

    buffer.add(_transition(1.0))
    buffer.add(_transition(2.0))

    assert len(buffer) == 2


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=5, obs_shape=(3,), action_dim=1)
    for i in range(5):
        buffer.add(_transition(float(i)))

    batch = buffer.sample(batch_size=4, seed=123)

    assert tuple(batch["obs"].shape) == (4, 3)
    assert tuple(batch["actions"].shape) == (4, 1)
    assert tuple(batch["rewards"].shape) == (4, 1)
    assert tuple(batch["next_obs"].shape) == (4, 3)
    assert tuple(batch["dones"].shape) == (4, 1)
