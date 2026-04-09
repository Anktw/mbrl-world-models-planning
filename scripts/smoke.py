"""Phase 1 smoke test for config and data wiring."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from mbrl.config import load_config
from mbrl.data import ReplayBuffer, Transition
from mbrl.utils.device import select_device
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def run_smoke(config_path: str) -> None:
    """Run a minimal end-to-end wiring check for Phase 1."""

    configure_logging()
    logger = logging.getLogger("mbrl.smoke")

    config = load_config(config_path)
    set_global_seed(config.training.seed)
    device = select_device(config.training.device)

    logger.info("Loaded config for env=%s on device=%s", config.env.name, device)

    buffer = ReplayBuffer(
        capacity=config.data.capacity,
        obs_shape=config.env.obs_shape,
        action_dim=config.env.action_dim,
    )

    for i in range(max(config.data.min_size, config.training.batch_size)):
        obs = np.full(config.env.obs_shape, fill_value=i, dtype=np.float32)
        nxt = obs + 1.0
        action = np.zeros((config.env.action_dim,), dtype=np.float32)
        reward = float(i % 2)
        done = bool(i % 7 == 0)
        buffer.add(
            Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=nxt,
                done=done,
            )
        )

    batch = buffer.sample(config.training.batch_size, seed=config.training.seed)
    logger.info(
        "Sampled batch shapes: obs=%s actions=%s rewards=%s",
        tuple(batch["obs"].shape),
        tuple(batch["actions"].shape),
        tuple(batch["rewards"].shape),
    )
    logger.info("PHASE_1_SMOKE_SUCCESS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 smoke validation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args.config)
