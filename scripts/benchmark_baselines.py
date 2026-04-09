"""Run Phase 9 reproducible PPO/SAC benchmark against CEM/MCTS."""

from __future__ import annotations

import argparse

from mbrl.config import load_config
from mbrl.evaluation.baselines import BaselineBenchmarker, parse_seed_list
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CEM/MCTS vs PPO/SAC")
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
        help="Reward predictor checkpoint",
    )
    parser.add_argument(
        "--metrics-csv",
        default="artifacts/full_loop/system_metrics.csv",
        help="Cycle metrics CSV from full-loop training",
    )
    parser.add_argument("--seeds", default="7,13,23", help="Comma-separated random seeds")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Evaluation episodes per seed")
    parser.add_argument("--episode-horizon", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--baseline-timesteps",
        type=int,
        default=5000,
        help="PPO/SAC train timesteps",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/baseline_benchmark",
        help="Directory for benchmark outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()

    config = load_config(args.config)
    set_global_seed(config.training.seed)

    benchmarker = BaselineBenchmarker(
        config=config,
        buffer_path=args.buffer,
        vae_checkpoint=args.vae_checkpoint,
        rssm_checkpoint=args.rssm_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        metrics_csv_path=args.metrics_csv,
        artifacts_dir=args.artifacts_dir,
        seeds=parse_seed_list(args.seeds),
        eval_episodes=args.eval_episodes,
        episode_horizon=args.episode_horizon,
        baseline_timesteps=args.baseline_timesteps,
    )
    result = benchmarker.run()

    print("phase 9 benchmark summary:")
    for summary in result.summaries:
        print(
            f"  {summary.method}: mean_reward={summary.mean_reward:.6f} "
            f"std={summary.std_reward:.6f} "
            f"mean_reward_per_step={summary.mean_reward_per_step:.8f}"
        )
    print(f"  Runs CSV: {result.artifacts.csv_path}")
    print(f"  Plot: {result.artifacts.plot_path}")
    print(f"  Report: {result.artifacts.report_path}")
