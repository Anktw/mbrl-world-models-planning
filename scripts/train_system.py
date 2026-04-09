"""Run the full config-driven MBRL training loop."""

from __future__ import annotations

import argparse

from mbrl.config import load_config
from mbrl.training.system_trainer import FullSystemTrainer
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MBRL training loop")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--cycles", type=int, default=None, help="Override pipeline cycles")
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=None,
        help="Override collect steps per cycle",
    )
    parser.add_argument(
        "--bootstrap-steps",
        type=int,
        default=None,
        help="Override initial bootstrap steps",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override trainer steps per epoch",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/full_loop",
        help="Artifacts output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()

    config = load_config(args.config)
    if args.cycles is not None:
        config.training.pipeline_cycles = args.cycles
    if args.collect_steps is not None:
        config.training.collect_steps_per_cycle = args.collect_steps
    if args.bootstrap_steps is not None:
        config.training.bootstrap_steps = args.bootstrap_steps
    if args.steps_per_epoch is not None:
        config.training.steps_per_epoch = args.steps_per_epoch

    set_global_seed(config.training.seed)

    trainer = FullSystemTrainer(config=config, artifacts_dir=args.artifacts_dir)
    metrics = trainer.run()

    print("final cycle summary:")
    final = metrics[-1]
    print(
        "  cycle="
        f"{final.cycle} "
        f"avg_reward={final.average_collection_reward:.6f} "
        f"vae_loss={final.vae_loss:.6f} "
        f"rssm_loss={final.rssm_loss:.6f} "
        f"reward_loss={final.reward_loss:.6f} "
        f"cem_return={final.cem_predicted_return:.6f} "
        f"mcts_return={final.mcts_predicted_return:.6f} "
        f"executed_reward={final.executed_reward:.6f}"
    )
