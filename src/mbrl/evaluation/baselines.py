"""Phase 9 benchmark runner for CEM/MCTS vs PPO/SAC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mbrl.config import AppConfig
from mbrl.data.buffer import ReplayBuffer
from mbrl.envs.pixel_env import PixelObservationEnvWrapper
from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.planners.base import Planner
from mbrl.planners.cem import CEMConfig, CEMPlanner
from mbrl.planners.mcts import MCTSConfig, MCTSPlanner
from mbrl.utils.device import select_device


@dataclass(frozen=True)
class MethodRun:
    """One benchmark run for a method and seed."""

    method: str
    seed: int
    cumulative_reward: float
    mean_reward: float
    reward_per_step: float


@dataclass(frozen=True)
class MethodSummary:
    """Aggregated benchmark stats across seeds."""

    method: str
    mean_reward: float
    std_reward: float
    mean_reward_per_step: float


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Benchmark output artifact paths."""

    plot_path: Path
    report_path: Path
    csv_path: Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Final benchmark output container."""

    runs: list[MethodRun]
    summaries: list[MethodSummary]
    artifacts: BenchmarkArtifacts


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated seed values."""

    values = [segment.strip() for segment in seed_text.split(",") if segment.strip()]
    if not values:
        raise ValueError("At least one seed must be provided")
    return [int(value) for value in values]


class BaselineBenchmarker:
    """Run reproducible planner and baseline RL comparisons."""

    def __init__(
        self,
        config: AppConfig,
        buffer_path: str | Path,
        vae_checkpoint: str | Path,
        rssm_checkpoint: str | Path,
        reward_checkpoint: str | Path,
        metrics_csv_path: str | Path,
        artifacts_dir: str | Path,
        seeds: list[int],
        eval_episodes: int,
        episode_horizon: int,
        baseline_timesteps: int,
    ) -> None:
        self.config = config
        self.buffer_path = Path(buffer_path)
        self.vae_checkpoint = Path(vae_checkpoint)
        self.rssm_checkpoint = Path(rssm_checkpoint)
        self.reward_checkpoint = Path(reward_checkpoint)
        self.metrics_csv_path = Path(metrics_csv_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.seeds = seeds
        self.eval_episodes = eval_episodes
        self.episode_horizon = episode_horizon
        self.baseline_timesteps = baseline_timesteps

        self.device = select_device(config.training.device)
        self.buffer = ReplayBuffer.load(self.buffer_path)

        vae_state = torch.load(self.vae_checkpoint, map_location=self.device)["model_state_dict"]
        rssm_state = torch.load(self.rssm_checkpoint, map_location=self.device)["model_state_dict"]
        reward_state = torch.load(
            self.reward_checkpoint,
            map_location=self.device,
        )["model_state_dict"]

        image_channels, vae_hidden_dim, latent_dim = self._infer_vae_dims(vae_state)
        rssm_action_dim, deterministic_dim, rssm_hidden_dim = self._infer_rssm_dims(
            rssm_state,
            latent_dim,
        )
        reward_hidden_dim = int(reward_state["network.0.weight"].shape[0])

        self.vae = VAE(
            image_channels=image_channels,
            hidden_dim=vae_hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)
        self.rssm = RSSM(
            latent_dim=latent_dim,
            action_dim=rssm_action_dim,
            deterministic_dim=deterministic_dim,
            hidden_dim=rssm_hidden_dim,
        ).to(self.device)
        self.reward_predictor = RewardPredictor(
            latent_dim=latent_dim,
            hidden_dim=reward_hidden_dim,
        ).to(self.device)

        self.vae.load_state_dict(vae_state)
        self.rssm.load_state_dict(rssm_state)
        self.reward_predictor.load_state_dict(reward_state)
        self.vae.eval()
        self.rssm.eval()
        self.reward_predictor.eval()

    def run(self) -> BenchmarkResult:
        """Execute all benchmark methods and write outputs."""

        runs: list[MethodRun] = []
        world_model_steps = self._collected_steps_from_metrics()

        for seed in self.seeds:
            cem_return = self._evaluate_planner(method="cem", seed=seed)
            mcts_return = self._evaluate_planner(method="mcts", seed=seed)
            ppo_return = self._evaluate_baseline(method="ppo", seed=seed)
            sac_return = self._evaluate_baseline(method="sac", seed=seed)

            runs.extend(
                [
                    MethodRun(
                        method="CEM",
                        seed=seed,
                        cumulative_reward=cem_return,
                        mean_reward=cem_return / self.eval_episodes,
                        reward_per_step=cem_return / max(float(world_model_steps), 1.0),
                    ),
                    MethodRun(
                        method="MCTS",
                        seed=seed,
                        cumulative_reward=mcts_return,
                        mean_reward=mcts_return / self.eval_episodes,
                        reward_per_step=mcts_return / max(float(world_model_steps), 1.0),
                    ),
                    MethodRun(
                        method="PPO",
                        seed=seed,
                        cumulative_reward=ppo_return,
                        mean_reward=ppo_return / self.eval_episodes,
                        reward_per_step=ppo_return / max(float(self.baseline_timesteps), 1.0),
                    ),
                    MethodRun(
                        method="SAC",
                        seed=seed,
                        cumulative_reward=sac_return,
                        mean_reward=sac_return / self.eval_episodes,
                        reward_per_step=sac_return / max(float(self.baseline_timesteps), 1.0),
                    ),
                ]
            )

        summaries = self._summaries(runs)
        plot_path = self.artifacts_dir / "baseline_benchmark.png"
        report_path = self.artifacts_dir / "baseline_benchmark_report.md"
        csv_path = self.artifacts_dir / "baseline_benchmark_runs.csv"

        self._write_csv(runs, csv_path)
        self._plot_summary(summaries, plot_path)
        self._write_report(summaries, report_path, world_model_steps)

        return BenchmarkResult(
            runs=runs,
            summaries=summaries,
            artifacts=BenchmarkArtifacts(
                plot_path=plot_path,
                report_path=report_path,
                csv_path=csv_path,
            ),
        )

    def _evaluate_planner(self, method: str, seed: int) -> float:
        env = PixelObservationEnvWrapper(
            env_name=self.config.env.name,
            image_size=self.config.model.vae.image_size,
            seed=seed,
        )
        planner: Planner
        if method == "cem":
            planner = CEMPlanner(
                rssm=self.rssm,
                reward_predictor=self.reward_predictor,
                action_dim=self.buffer.action_dim,
                device=self.device,
                config=CEMConfig(
                    horizon=self.config.planning.cem.horizon,
                    num_samples=self.config.planning.cem.num_samples,
                    num_iterations=self.config.planning.cem.num_iterations,
                    elite_fraction=self.config.planning.cem.elite_fraction,
                    min_std=self.config.planning.cem.min_std,
                    discount=self.config.planning.cem.discount,
                    action_low=self.config.planning.cem.action_low,
                    action_high=self.config.planning.cem.action_high,
                ),
            )
        else:
            planner = MCTSPlanner(
                rssm=self.rssm,
                reward_predictor=self.reward_predictor,
                action_dim=self.buffer.action_dim,
                device=self.device,
                config=MCTSConfig(
                    horizon=self.config.planning.mcts.horizon,
                    num_simulations=self.config.planning.mcts.num_simulations,
                    max_children=self.config.planning.mcts.max_children,
                    exploration_constant=self.config.planning.mcts.exploration_constant,
                    discount=self.config.planning.mcts.discount,
                    action_low=self.config.planning.mcts.action_low,
                    action_high=self.config.planning.mcts.action_high,
                ),
            )

        cumulative_reward = 0.0
        for episode in range(self.eval_episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            for _ in range(self.episode_horizon):
                obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    latent, _ = self.vae.encode(obs_tensor)
                result = planner.plan(latent)
                action = result.action.detach().cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            cumulative_reward += episode_reward

        env.close()
        return cumulative_reward

    def _evaluate_baseline(self, method: str, seed: int) -> float:
        try:
            from stable_baselines3 import PPO, SAC
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required for Phase 9 benchmarks. "
                "Install with: pip install -e .[baseline]"
            ) from exc

        env = gym.make(self.config.env.name)
        model: Any
        if method == "ppo":
            model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        else:
            model = SAC("MlpPolicy", env, seed=seed, verbose=0)

        model.learn(total_timesteps=self.baseline_timesteps, progress_bar=False)

        cumulative_reward = 0.0
        for episode in range(self.eval_episodes):
            observation, _ = env.reset(seed=seed + 1000 + episode)
            episode_reward = 0.0
            for _ in range(self.episode_horizon):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            cumulative_reward += episode_reward

        env.close()
        return cumulative_reward

    @staticmethod
    def _infer_vae_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        image_channels = int(state_dict["encoder.features.0.weight"].shape[1])
        latent_dim = int(state_dict["encoder.mean_head.weight"].shape[0])
        hidden_dim = int(state_dict["encoder.mean_head.weight"].shape[1])
        return image_channels, hidden_dim, latent_dim

    @staticmethod
    def _infer_rssm_dims(
        state_dict: dict[str, torch.Tensor],
        latent_dim: int,
    ) -> tuple[int, int, int]:
        deterministic_dim = int(state_dict["recurrent.weight_hh"].shape[1])
        action_dim = int(state_dict["recurrent.weight_ih"].shape[1] - latent_dim)
        hidden_dim = int(state_dict["prior_network.0.weight"].shape[0])
        return action_dim, deterministic_dim, hidden_dim

    def _collected_steps_from_metrics(self) -> int:
        if not self.metrics_csv_path.exists():
            return len(self.buffer)

        total_steps = 0
        with self.metrics_csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                total_steps += int(float(row["collected_steps"]))
        return max(total_steps, 1)

    @staticmethod
    def _summaries(runs: list[MethodRun]) -> list[MethodSummary]:
        methods = sorted({run.method for run in runs})
        summaries: list[MethodSummary] = []
        for method in methods:
            method_runs = [run for run in runs if run.method == method]
            rewards = np.asarray([run.cumulative_reward for run in method_runs], dtype=np.float64)
            reward_per_step = np.asarray(
                [run.reward_per_step for run in method_runs],
                dtype=np.float64,
            )
            summaries.append(
                MethodSummary(
                    method=method,
                    mean_reward=float(rewards.mean()),
                    std_reward=float(rewards.std(ddof=0)),
                    mean_reward_per_step=float(reward_per_step.mean()),
                )
            )
        return summaries

    @staticmethod
    def _write_csv(runs: list[MethodRun], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "method",
                    "seed",
                    "cumulative_reward",
                    "mean_reward",
                    "reward_per_step",
                ]
            )
            for run in runs:
                writer.writerow(
                    [
                        run.method,
                        run.seed,
                        f"{run.cumulative_reward:.8f}",
                        f"{run.mean_reward:.8f}",
                        f"{run.reward_per_step:.8f}",
                    ]
                )

    @staticmethod
    def _plot_summary(summaries: list[MethodSummary], path: Path) -> None:
        methods = [summary.method for summary in summaries]
        means = [summary.mean_reward for summary in summaries]
        stds = [summary.std_reward for summary in summaries]

        figure = plt.figure(figsize=(8, 4))
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(methods, means, yerr=stds, capsize=4)
        axis.set_ylabel("Cumulative reward")
        axis.set_title("Phase 9 benchmark: CEM/MCTS vs PPO/SAC")
        figure.tight_layout()

        figure.savefig(path, dpi=150)
        plt.close(figure)

    @staticmethod
    def _write_report(
        summaries: list[MethodSummary],
        path: Path,
        world_model_steps: int,
    ) -> None:
        lines = [
            "# Phase 9 Benchmark Report",
            "",
            "## Protocol",
            f"- World-model data budget (steps): {world_model_steps}",
            "- Baselines trained with fixed timesteps per seed",
            "- Metrics aggregated across seeds",
            "",
            "## Results",
        ]

        for summary in summaries:
            lines.append(
                f"- {summary.method}: mean cumulative reward={summary.mean_reward:.6f}, "
                f"std={summary.std_reward:.6f}, "
                f"mean reward/step={summary.mean_reward_per_step:.8f}"
            )

        lines.extend(
            [
                "",
                "## Notes",
                "- Higher cumulative reward is better.",
                "- Reward/step provides a sample-efficiency style comparison.",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
