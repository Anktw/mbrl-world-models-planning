"""Replay buffer primitives for transition data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    """One environment step used by world-model training."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size FIFO replay buffer with typed sampling output."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        if transition.observation.shape != self.obs_shape:
            raise ValueError("Observation shape mismatch")
        if transition.next_observation.shape != self.obs_shape:
            raise ValueError("Next observation shape mismatch")
        if transition.action.shape != (self.action_dim,):
            raise ValueError("Action shape mismatch")

        idx = self._cursor
        self._obs[idx] = transition.observation
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._next_obs[idx] = transition.next_observation
        self._dones[idx] = float(transition.done)

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        rng = np.random.default_rng(seed)
        indices = rng.integers(low=0, high=self._size, size=batch_size)

        return {
            "obs": torch.from_numpy(self._obs[indices]),
            "actions": torch.from_numpy(self._actions[indices]),
            "rewards": torch.from_numpy(self._rewards[indices]),
            "next_obs": torch.from_numpy(self._next_obs[indices]),
            "dones": torch.from_numpy(self._dones[indices]),
        }

    def sample_sequences(
        self,
        sequence_length: int,
        batch_size: int,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample contiguous transition windows for sequence models."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._size < sequence_length + 1:
            raise ValueError("Not enough transitions for the requested sequence length")

        ordered = self._ordered_arrays()
        valid_starts = self._valid_sequence_starts(ordered["dones"], sequence_length)
        if not valid_starts:
            raise ValueError("No valid contiguous sequences available in the buffer")

        rng = np.random.default_rng(seed)
        start_indices = rng.choice(valid_starts, size=batch_size, replace=True)

        obs_sequences = []
        next_obs_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []

        for start in start_indices:
            end = start + sequence_length
            obs_sequences.append(ordered["obs"][start : end + 1])
            next_obs_sequences.append(ordered["next_obs"][start:end])
            action_sequences.append(ordered["actions"][start:end])
            reward_sequences.append(ordered["rewards"][start:end])
            done_sequences.append(ordered["dones"][start:end])

        return {
            "obs": torch.from_numpy(np.stack(obs_sequences, axis=0)),
            "next_obs": torch.from_numpy(np.stack(next_obs_sequences, axis=0)),
            "actions": torch.from_numpy(np.stack(action_sequences, axis=0)),
            "rewards": torch.from_numpy(np.stack(reward_sequences, axis=0)),
            "dones": torch.from_numpy(np.stack(done_sequences, axis=0)),
        }

    def _ordered_arrays(self) -> dict[str, np.ndarray]:
        if self._size < self.capacity:
            return {
                "obs": self._obs[: self._size].copy(),
                "actions": self._actions[: self._size].copy(),
                "rewards": self._rewards[: self._size].copy(),
                "next_obs": self._next_obs[: self._size].copy(),
                "dones": self._dones[: self._size].copy(),
            }

        head = slice(self._cursor, self.capacity)
        tail = slice(0, self._cursor)
        return {
            "obs": np.concatenate([self._obs[head], self._obs[tail]], axis=0),
            "actions": np.concatenate([self._actions[head], self._actions[tail]], axis=0),
            "rewards": np.concatenate([self._rewards[head], self._rewards[tail]], axis=0),
            "next_obs": np.concatenate([self._next_obs[head], self._next_obs[tail]], axis=0),
            "dones": np.concatenate([self._dones[head], self._dones[tail]], axis=0),
        }

    @staticmethod
    def _valid_sequence_starts(dones: np.ndarray, sequence_length: int) -> list[int]:
        valid_starts: list[int] = []
        last_start = dones.shape[0] - sequence_length - 1
        for start in range(last_start + 1):
            if np.any(dones[start : start + sequence_length - 1]):
                continue
            valid_starts.append(start)
        return valid_starts

    def save(self, path: str | Path) -> None:
        """Persist the buffer contents and metadata to disk."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            capacity=np.array(self.capacity, dtype=np.int64),
            size=np.array(self._size, dtype=np.int64),
            cursor=np.array(self._cursor, dtype=np.int64),
            obs_shape=np.array(self.obs_shape, dtype=np.int64),
            action_dim=np.array(self.action_dim, dtype=np.int64),
            obs=self._obs,
            actions=self._actions,
            rewards=self._rewards,
            next_obs=self._next_obs,
            dones=self._dones,
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        """Restore a replay buffer from disk."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            capacity = int(payload["capacity"].item())
            size = int(payload["size"].item())
            cursor = int(payload["cursor"].item())
            obs_shape = tuple(int(value) for value in payload["obs_shape"].tolist())
            action_dim = int(payload["action_dim"].item())

            buffer = cls(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)
            buffer._size = size
            buffer._cursor = cursor
            buffer._obs[:] = payload["obs"]
            buffer._actions[:] = payload["actions"]
            buffer._rewards[:] = payload["rewards"]
            buffer._next_obs[:] = payload["next_obs"]
            buffer._dones[:] = payload["dones"]
            return buffer
