"""Environment data collection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from mbrl.data.buffer import ReplayBuffer, Transition
from mbrl.envs.gymnasium_env import GymnasiumEnvWrapper
from mbrl.policies.random_policy import RandomPolicy


@dataclass(frozen=True)
class CollectionResult:
    """Summary of a random rollout collection run."""

    transitions_collected: int
    episodes_completed: int


def collect_random_data(
    env: GymnasiumEnvWrapper,
    policy: RandomPolicy,
    buffer: ReplayBuffer,
    num_transitions: int,
    seed: int | None = None,
) -> CollectionResult:
    """Collect random interaction data into the replay buffer."""

    observation, _ = env.reset(seed=seed)
    transitions_collected = 0
    episodes_completed = 0

    while transitions_collected < num_transitions:
        action = policy.act(env.env.action_space)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(
            Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        )

        observation = next_observation
        transitions_collected += 1

        if done:
            observation, _ = env.reset()
            episodes_completed += 1

    return CollectionResult(
        transitions_collected=transitions_collected,
        episodes_completed=episodes_completed,
    )
