"""Gymnasium wrapper that exposes RGB observations for VAE training."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class PixelStep:
    """Typed pixel observation step result."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class PixelObservationEnvWrapper:
    """Collect RGB observations from a Gymnasium environment."""

    def __init__(self, env_name: str, image_size: int = 64, seed: int | None = None) -> None:
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.seed = seed
        self.image_size = image_size

        if self.env.action_space.shape is None:
            raise ValueError("Action space must define a shape")

        self.action_shape = tuple(int(value) for value in self.env.action_space.shape)
        self.action_dim = int(np.prod(self.action_shape))
        self.obs_shape = (3, image_size, image_size)

    def _render_observation(self) -> np.ndarray:
        render_output: object = self.env.render()
        if render_output is None:
            raise RuntimeError("Environment render returned None")
        frame_array = np.asarray(render_output, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(frame_array).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).contiguous().numpy()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        actual_seed = self.seed if seed is None else seed
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(actual_seed)
        self.env.reset(seed=actual_seed)
        return self._render_observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._render_observation(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
