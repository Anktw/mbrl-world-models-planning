from pathlib import Path

import numpy as np

from mbrl.data.buffer import ReplayBuffer
from mbrl.data.collection import collect_random_data
from mbrl.envs.gymnasium_env import GymnasiumEnvWrapper
from mbrl.policies.random_policy import RandomPolicy


def test_collect_random_data_and_persist(tmp_path: Path) -> None:
    env = GymnasiumEnvWrapper("Pendulum-v1", seed=123)
    buffer = ReplayBuffer(capacity=64, obs_shape=env.obs_shape, action_dim=env.action_dim)
    policy = RandomPolicy()

    result = collect_random_data(env, policy, buffer, num_transitions=16, seed=123)

    assert result.transitions_collected == 16
    assert len(buffer) == 16

    batch = buffer.sample(batch_size=4, seed=123)
    assert tuple(batch["obs"].shape) == (4, 3)
    assert tuple(batch["actions"].shape) == (4, 1)
    assert tuple(batch["rewards"].shape) == (4, 1)
    assert tuple(batch["next_obs"].shape) == (4, 3)
    assert tuple(batch["dones"].shape) == (4, 1)

    saved_path = tmp_path / "buffer.npz"
    buffer.save(saved_path)
    loaded = ReplayBuffer.load(saved_path)

    assert len(loaded) == 16
    reloaded_batch = loaded.sample(batch_size=4, seed=123)
    np.testing.assert_equal(batch["obs"].shape, reloaded_batch["obs"].shape)

    env.close()
