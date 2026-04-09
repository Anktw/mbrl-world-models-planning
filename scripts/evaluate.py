"""Run full evaluation and generate report artifacts."""

from __future__ import annotations

import argparse

from mbrl.config import load_config
from mbrl.evaluation.evaluator import SystemEvaluator
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MBRL system")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--buffer", default="artifacts/vae_buffer.npz", help="Replay buffer path")
    parser.add_argument("--vae-checkpoint", default="artifacts/vae_model.pt", help="VAE checkpoint")
    parser.add_argument(
        "--rssm-checkpoint",
        default="artifacts/rssm_model.pt",
        help="RSSM checkpoint",
    )
    parser.add_argument(
        "--reward-checkpoint",
        default="artifacts/reward_model.pt",
        help="Reward checkpoint",
    )
    parser.add_argument(
        "--metrics-csv",
        default="artifacts/full_loop/system_metrics.csv",
        help="Cycle metrics CSV from full training loop",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluation",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--episode-horizon",
        type=int,
        default=200,
        help="Max steps per eval episode",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Append optional PPO/SAC note",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()

    config = load_config(args.config)
    set_global_seed(config.training.seed)

    evaluator = SystemEvaluator(
        config=config,
        buffer_path=args.buffer,
        vae_checkpoint=args.vae_checkpoint,
        rssm_checkpoint=args.rssm_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        metrics_csv_path=args.metrics_csv,
        artifacts_dir=args.artifacts_dir,
        baseline_compare=args.compare_baselines,
    )
    result = evaluator.evaluate(episode_horizon=args.episode_horizon)

    print("evaluation summary:")
    print(
        f"  CEM cumulative reward={result.planner_comparison.cem_cumulative_reward:.6f} "
        f"MCTS cumulative reward={result.planner_comparison.mcts_cumulative_reward:.6f}"
    )
    print(
        f"  RSSM latent MSE={result.prediction_error_metrics.rssm_latent_mse:.6f} "
        f"Reward MSE={result.prediction_error_metrics.reward_mse:.6f}"
    )
    print(
        "  Sample efficiency final reward/step="
        f"{result.sample_efficiency_metrics.reward_per_step_curve[-1]:.6f}"
    )
    print(f"  Report saved to: {result.artifacts.report_path}")
