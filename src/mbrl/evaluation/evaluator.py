"""End-to-end evaluator for planner comparison and model quality metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mbrl.config import AppConfig
from mbrl.data.buffer import ReplayBuffer
from mbrl.envs.pixel_env import PixelObservationEnvWrapper
from mbrl.evaluation.metrics import (
    PredictionErrorMetrics,
    SampleEfficiencyMetrics,
    cumulative_reward,
    sample_efficiency,
)
from mbrl.evaluation.visualization import (
    plot_latent_space,
    plot_planner_rewards,
    plot_prediction_errors,
    plot_sample_efficiency,
)
from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.planners.base import Planner
from mbrl.planners.cem import CEMConfig, CEMPlanner
from mbrl.planners.mcts import MCTSConfig, MCTSPlanner
from mbrl.utils.device import select_device


@dataclass(frozen=True)
class PlannerComparison:
    """Planner episode rewards and aggregate statistics."""

    cem_episode_rewards: list[float]
    mcts_episode_rewards: list[float]
    cem_cumulative_reward: float
    mcts_cumulative_reward: float


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths of generated report artifacts."""

    planner_plot: Path
    sample_efficiency_plot: Path
    prediction_error_plot: Path
    latent_space_plot: Path
    report_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    """Full evaluation output bundle."""

    planner_comparison: PlannerComparison
    sample_efficiency_metrics: SampleEfficiencyMetrics
    prediction_error_metrics: PredictionErrorMetrics
    artifacts: EvaluationArtifacts


class SystemEvaluator:
    """Evaluate planners and model prediction quality from saved checkpoints."""

    def __init__(
        self,
        config: AppConfig,
        buffer_path: str | Path,
        vae_checkpoint: str | Path,
        rssm_checkpoint: str | Path,
        reward_checkpoint: str | Path,
        metrics_csv_path: str | Path,
        artifacts_dir: str | Path,
        baseline_compare: bool = False,
    ) -> None:
        self.config = config
        self.buffer_path = Path(buffer_path)
        self.vae_checkpoint = Path(vae_checkpoint)
        self.rssm_checkpoint = Path(rssm_checkpoint)
        self.reward_checkpoint = Path(reward_checkpoint)
        self.metrics_csv_path = Path(metrics_csv_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_compare = baseline_compare

        self.device = select_device(config.training.device)
        self.buffer = ReplayBuffer.load(self.buffer_path)

        vae_state = torch.load(self.vae_checkpoint, map_location=self.device)["model_state_dict"]
        reward_state = torch.load(
            self.reward_checkpoint,
            map_location=self.device,
        )["model_state_dict"]
        rssm_state = torch.load(self.rssm_checkpoint, map_location=self.device)["model_state_dict"]

        vae_image_channels, vae_hidden_dim, vae_latent_dim = self._infer_vae_dims(vae_state)
        reward_hidden_dim = self._infer_reward_hidden_dim(reward_state)
        rssm_action_dim, rssm_deterministic_dim, rssm_hidden_dim = self._infer_rssm_dims(
            rssm_state,
            vae_latent_dim,
        )

        self.vae = VAE(
            image_channels=vae_image_channels,
            hidden_dim=vae_hidden_dim,
            latent_dim=vae_latent_dim,
        ).to(self.device)
        self.rssm = RSSM(
            latent_dim=vae_latent_dim,
            action_dim=rssm_action_dim,
            deterministic_dim=rssm_deterministic_dim,
            hidden_dim=rssm_hidden_dim,
        ).to(self.device)
        self.reward_predictor = RewardPredictor(
            latent_dim=vae_latent_dim,
            hidden_dim=reward_hidden_dim,
        ).to(self.device)

        self.vae.load_state_dict(vae_state)
        self.rssm.load_state_dict(rssm_state)
        self.reward_predictor.load_state_dict(reward_state)

        self.vae.eval()
        self.rssm.eval()
        self.reward_predictor.eval()

        self.env = PixelObservationEnvWrapper(
            env_name=config.env.name,
            image_size=config.model.vae.image_size,
            seed=config.training.seed,
        )

    def evaluate(self, episode_horizon: int = 200) -> EvaluationResult:
        """Run all evaluation routines and emit plots/report."""

        planner_comparison = self._compare_planners(episode_horizon=episode_horizon)
        prediction_error_metrics = self._prediction_errors()
        sample_efficiency_metrics = self._sample_efficiency_from_csv()

        planner_plot = self.artifacts_dir / "planner_comparison.png"
        sample_efficiency_plot = self.artifacts_dir / "sample_efficiency.png"
        prediction_error_plot = self.artifacts_dir / "prediction_errors.png"
        latent_space_plot = self.artifacts_dir / "latent_space.png"
        report_path = self.artifacts_dir / "evaluation_report.md"

        plot_planner_rewards(
            cem_rewards=planner_comparison.cem_episode_rewards,
            mcts_rewards=planner_comparison.mcts_episode_rewards,
            output_path=planner_plot,
        )
        plot_sample_efficiency(
            cumulative_steps=sample_efficiency_metrics.cumulative_collected_steps,
            reward_per_step_curve=sample_efficiency_metrics.reward_per_step_curve,
            output_path=sample_efficiency_plot,
        )
        plot_prediction_errors(
            rssm_mse=prediction_error_metrics.rssm_latent_mse,
            reward_mse=prediction_error_metrics.reward_mse,
            output_path=prediction_error_plot,
        )

        latents_2d, rewards = self._latent_embedding_points(sample_count=min(256, len(self.buffer)))
        plot_latent_space(latents_2d=latents_2d, rewards=rewards, output_path=latent_space_plot)

        artifacts = EvaluationArtifacts(
            planner_plot=planner_plot,
            sample_efficiency_plot=sample_efficiency_plot,
            prediction_error_plot=prediction_error_plot,
            latent_space_plot=latent_space_plot,
            report_path=report_path,
        )
        result = EvaluationResult(
            planner_comparison=planner_comparison,
            sample_efficiency_metrics=sample_efficiency_metrics,
            prediction_error_metrics=prediction_error_metrics,
            artifacts=artifacts,
        )
        self._write_report(result)

        if self.baseline_compare:
            self._optional_baseline_note(report_path)

        self.env.close()
        return result

    def _compare_planners(self, episode_horizon: int) -> PlannerComparison:
        cem_planner = CEMPlanner(
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
        mcts_planner = MCTSPlanner(
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

        cem_rewards = self._run_planner_episodes(cem_planner, episode_horizon)
        mcts_rewards = self._run_planner_episodes(mcts_planner, episode_horizon)
        return PlannerComparison(
            cem_episode_rewards=cem_rewards,
            mcts_episode_rewards=mcts_rewards,
            cem_cumulative_reward=cumulative_reward(cem_rewards),
            mcts_cumulative_reward=cumulative_reward(mcts_rewards),
        )

    def _run_planner_episodes(self, planner: Planner, episode_horizon: int) -> list[float]:
        rewards: list[float] = []
        for episode in range(self.config.training.planner_eval_episodes):
            observation, _ = self.env.reset(seed=self.config.training.seed + episode)
            episode_reward = 0.0
            for _ in range(episode_horizon):
                obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    latent, _ = self.vae.encode(obs_tensor)
                result = planner.plan(latent)
                action = result.action.detach().cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
        return rewards

    def _prediction_errors(self) -> PredictionErrorMetrics:
        sequence_batch = self.buffer.sample_sequences(
            sequence_length=self.config.training.sequence_length,
            batch_size=min(self.config.training.batch_size, max(1, len(self.buffer) // 2)),
            seed=self.config.training.seed,
        )
        obs_sequences = sequence_batch["obs"].to(self.device)
        actions = sequence_batch["actions"].to(self.device)

        batch_size, sequence_length_plus_one = obs_sequences.shape[:2]
        flat_obs = obs_sequences.reshape(-1, *obs_sequences.shape[2:])
        with torch.no_grad():
            encoded_mean, _ = self.vae.encode(flat_obs)
        latents = encoded_mean.reshape(batch_size, sequence_length_plus_one, -1)
        predicted_latents, _ = self.rssm.rollout_sequence(latents[:, 0], actions)
        rssm_mse = float(F.mse_loss(predicted_latents, latents[:, 1:]).item())

        reward_batch = self.buffer.sample(
            batch_size=min(self.config.training.batch_size, len(self.buffer)),
            seed=self.config.training.seed,
        )
        with torch.no_grad():
            reward_latents, _ = self.vae.encode(reward_batch["obs"].to(self.device))
            reward_pred = self.reward_predictor(reward_latents)
        reward_mse = float(F.mse_loss(reward_pred, reward_batch["rewards"].to(self.device)).item())

        return PredictionErrorMetrics(rssm_latent_mse=rssm_mse, reward_mse=reward_mse)

    def _sample_efficiency_from_csv(self) -> SampleEfficiencyMetrics:
        collected_steps: list[int] = []
        executed_rewards: list[float] = []

        with self.metrics_csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                collected_steps.append(int(float(row["collected_steps"])))
                executed_rewards.append(float(row["executed_reward"]))

        return sample_efficiency(collected_steps, executed_rewards)

    def _latent_embedding_points(self, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self.buffer.sample(batch_size=sample_count, seed=self.config.training.seed)
        with torch.no_grad():
            latents, _ = self.vae.encode(batch["obs"].to(self.device))

        centered = latents - latents.mean(dim=0, keepdim=True)
        _, _, basis = torch.pca_lowrank(centered, q=2)
        projected = centered @ basis[:, :2]
        rewards = batch["rewards"].squeeze(-1).numpy()
        return projected.cpu().numpy(), rewards

    @staticmethod
    def _infer_vae_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        image_channels = int(state_dict["encoder.features.0.weight"].shape[1])
        latent_dim = int(state_dict["encoder.mean_head.weight"].shape[0])
        hidden_dim = int(state_dict["encoder.mean_head.weight"].shape[1])
        return image_channels, hidden_dim, latent_dim

    @staticmethod
    def _infer_reward_hidden_dim(state_dict: dict[str, torch.Tensor]) -> int:
        return int(state_dict["network.0.weight"].shape[0])

    @staticmethod
    def _infer_rssm_dims(
        state_dict: dict[str, torch.Tensor],
        latent_dim: int,
    ) -> tuple[int, int, int]:
        deterministic_dim = int(state_dict["recurrent.weight_hh"].shape[1])
        action_dim = int(state_dict["recurrent.weight_ih"].shape[1] - latent_dim)
        hidden_dim = int(state_dict["prior_network.0.weight"].shape[0])
        return action_dim, deterministic_dim, hidden_dim

    def _write_report(self, result: EvaluationResult) -> None:
        lines = [
            "# Evaluation Report",
            "",
            "## Planner Comparison",
            (
                f"- CEM cumulative reward: "
                f"{result.planner_comparison.cem_cumulative_reward:.6f}"
            ),
            (
                f"- MCTS cumulative reward: "
                f"{result.planner_comparison.mcts_cumulative_reward:.6f}"
            ),
            "",
            "## Sample Efficiency",
            (
                f"- Final reward per step: "
                f"{result.sample_efficiency_metrics.reward_per_step_curve[-1]:.6f}"
            ),
            (
                f"- Total collected steps: "
                f"{result.sample_efficiency_metrics.cumulative_collected_steps[-1]}"
            ),
            "",
            "## Prediction Errors",
            f"- RSSM latent MSE: {result.prediction_error_metrics.rssm_latent_mse:.6f}",
            f"- Reward MSE: {result.prediction_error_metrics.reward_mse:.6f}",
            "",
            "## Generated Artifacts",
            f"- Planner comparison plot: {result.artifacts.planner_plot}",
            f"- Sample efficiency plot: {result.artifacts.sample_efficiency_plot}",
            f"- Prediction error plot: {result.artifacts.prediction_error_plot}",
            f"- Latent space plot: {result.artifacts.latent_space_plot}",
        ]
        result.artifacts.report_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _optional_baseline_note(report_path: Path) -> None:
        note = [
            "",
            "## Optional PPO/SAC Baselines",
            "",
            "PPO/SAC comparison is optional and depends on extra baseline tooling.",
            "Install Stable-Baselines3 and add baseline rollout scripts to append this section.",
        ]
        with report_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(note))
