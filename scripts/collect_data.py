"""Phase 2 data collection entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mbrl.data.buffer import ReplayBuffer
from mbrl.data.collection import collect_random_data
from mbrl.envs.gymnasium_env import GymnasiumEnvWrapper
from mbrl.policies.random_policy import RandomPolicy
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def _print_sample_transitions(buffer: ReplayBuffer, count: int) -> None:
    sample = buffer.sample(min(count, len(buffer)), seed=7)
    print(f"obs shape: {tuple(sample['obs'].shape)}")
    print(f"action shape: {tuple(sample['actions'].shape)}")
    print(f"reward shape: {tuple(sample['rewards'].shape)}")
    print("sample transitions:")
    for index in range(min(count, len(buffer))):
        print(
            {
                "obs": sample["obs"][index].tolist(),
                "action": sample["actions"][index].tolist(),
                "reward": float(sample["rewards"][index].item()),
                "next_obs": sample["next_obs"][index].tolist(),
                "done": bool(sample["dones"][index].item()),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random data into a replay buffer")
    parser.add_argument("--env", default="Pendulum-v1", help="Gymnasium environment name")
    parser.add_argument("--steps", type=int, default=128, help="Number of transitions to collect")
    parser.add_argument("--capacity", type=int, default=1000, help="Replay buffer capacity")
    parser.add_argument("--output", default="artifacts/replay_buffer.npz", help="Output path")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    set_global_seed(args.seed)

    env = GymnasiumEnvWrapper(args.env, seed=args.seed)
    buffer = ReplayBuffer(
        capacity=args.capacity,
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )
    policy = RandomPolicy()

    result = collect_random_data(
        env=env,
        policy=policy,
        buffer=buffer,
        num_transitions=args.steps,
        seed=args.seed,
    )

    _print_sample_transitions(buffer, count=3)
    print(f"collected transitions: {result.transitions_collected}")
    print(f"episodes completed: {result.episodes_completed}")
    output_path = Path(args.output)
    buffer.save(output_path)
    print(f"saved replay buffer to: {output_path}")
    env.close()
