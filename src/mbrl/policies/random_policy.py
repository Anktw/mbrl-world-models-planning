"""Random policy used to bootstrap the replay buffer."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Action sampler that delegates to the environment action space."""

    def act(self, action_space: object) -> np.ndarray:
        if not hasattr(action_space, "sample"):
            raise TypeError("action_space must expose sample()")
        action = action_space.sample()
        return np.asarray(action, dtype=np.float32)
