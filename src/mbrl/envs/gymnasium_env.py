"""Gymnasium environment adapter with a small stable interface."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class EnvStep:
    """Typed transition output from the environment wrapper."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class GymnasiumEnvWrapper:
    """Thin adapter around a Gymnasium environment."""

    def __init__(self, env_name: str, seed: int | None = None) -> None:
        self.env = gym.make(env_name)
        self.seed = seed

        if self.env.observation_space.shape is None:
            raise ValueError("Observation space must define a shape")
        if self.env.action_space.shape is None:
            raise ValueError("Action space must define a shape")

        self.obs_shape = tuple(int(value) for value in self.env.observation_space.shape)
        self.action_shape = tuple(int(value) for value in self.env.action_space.shape)
        self.action_dim = int(np.prod(self.action_shape))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation."""

        actual_seed = self.seed if seed is None else seed
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(actual_seed)
        observation, info = self.env.reset(seed=actual_seed)
        return np.asarray(observation, dtype=np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one step."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        """Close the underlying environment."""

        self.env.close()
