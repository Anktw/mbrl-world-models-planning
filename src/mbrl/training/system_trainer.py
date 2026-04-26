"""End-to-end training loop orchestration for the full MBRL system."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from mbrl.config.base import AppConfig
from mbrl.data.buffer import ReplayBuffer, Transition
from mbrl.data.image_collection import collect_random_image_data
from mbrl.envs.pixel_env import PixelObservationEnvWrapper
from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.planners.cem import CEMConfig, CEMPlanner
from mbrl.planners.mcts import MCTSConfig, MCTSPlanner
from mbrl.policies.random_policy import RandomPolicy
from mbrl.training.reward_trainer import RewardTrainer
from mbrl.training.rssm_trainer import RSSMTrainer
from mbrl.training.vae_trainer import VAETrainer
from mbrl.utils.device import select_device


@dataclass(frozen=True)
class CycleMetrics:
    """Logged values per full training cycle."""

    cycle: int
    collected_steps: int
    average_collection_reward: float
    vae_loss: float
    rssm_loss: float
    reward_loss: float
    cem_predicted_return: float
    mcts_predicted_return: float
    executed_reward: float


class FullSystemTrainer:
    """Coordinates data collection, model updates, and planning evaluation."""

    def __init__(
        self,
        config: AppConfig,
        artifacts_dir: str | Path = "artifacts/full_loop",
    ) -> None:
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.device = select_device(config.training.device)
        self.env = PixelObservationEnvWrapper(
            env_name=config.env.name,
            image_size=config.model.vae.image_size,
            seed=config.training.seed,
        )
        self.policy = RandomPolicy()

        self.buffer = ReplayBuffer(
            capacity=config.data.capacity,
            obs_shape=self.env.obs_shape,
            action_dim=self.env.action_dim,
        )

        self.vae = VAE(
            image_channels=config.model.vae.image_channels,
            hidden_dim=config.model.vae.hidden_dim,
            latent_dim=config.model.vae.latent_dim,
        )
        self.rssm = RSSM(
            latent_dim=config.model.vae.latent_dim,
            action_dim=self.env.action_dim,
            deterministic_dim=config.model.rssm.deterministic_dim,
            hidden_dim=config.model.rssm.hidden_dim,
        )
        self.reward_predictor = RewardPredictor(
            latent_dim=config.model.vae.latent_dim,
            hidden_dim=config.model.reward.hidden_dim,
        )

        self.vae_trainer = VAETrainer(
            model=self.vae,
            learning_rate=config.training.learning_rate,
            kl_beta=config.model.vae.kl_beta,
            foreground_weight=config.model.vae.foreground_weight,
            foreground_threshold=config.model.vae.foreground_threshold,
            kl_warmup_epochs=10,
            device=self.device,
        )
        self.rssm_trainer = RSSMTrainer(
            rssm=self.rssm,
            vae=self.vae,
            learning_rate=config.training.learning_rate,
            device=self.device,
        )
        self.reward_trainer = RewardTrainer(
            predictor=self.reward_predictor,
            vae=self.vae,
            learning_rate=config.training.learning_rate,
            device=self.device,
        )

    def run(self) -> list[CycleMetrics]:
        """Run the full config-driven training loop."""

        metrics: list[CycleMetrics] = []
        self._bootstrap_buffer()
        metrics_file = self.artifacts_dir / "system_metrics.csv"
        self._init_metrics_file(metrics_file)

        for cycle in range(self.config.training.pipeline_cycles):
            if self._should_refresh_buffer(cycle):
                self._refresh_buffer()

            collection_result = collect_random_image_data(
                env=self.env,
                policy=self.policy,
                buffer=self.buffer,
                num_transitions=self.config.training.collect_steps_per_cycle,
                seed=self.config.training.seed + cycle,
            )

            batch = self.buffer.sample(
                batch_size=self.config.training.collect_steps_per_cycle,
                seed=self.config.training.seed + cycle,
            )
            average_collection_reward = float(batch["rewards"].mean().item())

            for parameter in self.vae.parameters():
                parameter.requires_grad_(True)

            vae_result = self.vae_trainer.fit(
                replay_buffer=self.buffer,
                epochs=self.config.training.vae_epochs_per_cycle,
                batch_size=self.config.training.batch_size,
                steps_per_epoch=self.config.training.steps_per_epoch,
                checkpoint_path=self.artifacts_dir / f"vae_cycle_{cycle + 1}.pt",
                reconstruction_path=self.artifacts_dir / f"vae_recon_cycle_{cycle + 1}.png",
                seed=self.config.training.seed + cycle,
            )

            for parameter in self.vae.parameters():
                parameter.requires_grad_(False)

            rssm_result = self.rssm_trainer.fit(
                replay_buffer=self.buffer,
                epochs=self.config.training.rssm_epochs_per_cycle,
                batch_size=self.config.training.batch_size,
                sequence_length=self.config.training.sequence_length,
                steps_per_epoch=self.config.training.steps_per_epoch,
                checkpoint_path=self.artifacts_dir / f"rssm_cycle_{cycle + 1}.pt",
                comparison_path=self.artifacts_dir / f"rssm_compare_cycle_{cycle + 1}.png",
                seed=self.config.training.seed + cycle,
            )

            reward_result = self.reward_trainer.fit(
                replay_buffer=self.buffer,
                epochs=self.config.training.reward_epochs_per_cycle,
                batch_size=self.config.training.batch_size,
                steps_per_epoch=self.config.training.steps_per_epoch,
                checkpoint_path=self.artifacts_dir / f"reward_cycle_{cycle + 1}.pt",
                comparison_path=self.artifacts_dir / f"reward_compare_cycle_{cycle + 1}.png",
                seed=self.config.training.seed + cycle,
            )

            cem_return, mcts_return, executed_reward = self._plan_and_execute(cycle)

            cycle_metrics = CycleMetrics(
                cycle=cycle + 1,
                collected_steps=collection_result.transitions_collected,
                average_collection_reward=average_collection_reward,
                vae_loss=vae_result.epoch_losses[-1],
                rssm_loss=rssm_result.epoch_losses[-1],
                reward_loss=reward_result.epoch_losses[-1],
                cem_predicted_return=cem_return,
                mcts_predicted_return=mcts_return,
                executed_reward=executed_reward,
            )
            metrics.append(cycle_metrics)
            self._append_metrics(metrics_file, cycle_metrics)
            print(
                "cycle="
                f"{cycle_metrics.cycle} "
                f"collected={cycle_metrics.collected_steps} "
                f"avg_reward={cycle_metrics.average_collection_reward:.6f} "
                f"vae_loss={cycle_metrics.vae_loss:.6f} "
                f"rssm_loss={cycle_metrics.rssm_loss:.6f} "
                f"reward_loss={cycle_metrics.reward_loss:.6f} "
                f"cem_return={cycle_metrics.cem_predicted_return:.6f} "
                f"mcts_return={cycle_metrics.mcts_predicted_return:.6f} "
                f"executed_reward={cycle_metrics.executed_reward:.6f}"
            )

        self.env.close()
        return metrics

    def _bootstrap_buffer(self) -> None:
        collect_random_image_data(
            env=self.env,
            policy=self.policy,
            buffer=self.buffer,
            num_transitions=self.config.training.bootstrap_steps,
            seed=self.config.training.seed,
        )

    def _should_refresh_buffer(self, cycle: int) -> bool:
        interval = self.config.training.buffer_refresh_interval
        return interval > 0 and cycle > 0 and cycle % interval == 0

    def _refresh_buffer(self) -> None:
        self.buffer = ReplayBuffer(
            capacity=self.config.data.capacity,
            obs_shape=self.env.obs_shape,
            action_dim=self.env.action_dim,
        )
        self._bootstrap_buffer()

    def _plan_and_execute(self, cycle: int) -> tuple[float, float, float]:
        cem_planner = CEMPlanner(
            rssm=self.rssm,
            reward_predictor=self.reward_predictor,
            action_dim=self.env.action_dim,
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
            action_dim=self.env.action_dim,
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

        cem_returns: list[float] = []
        mcts_returns: list[float] = []
        executed_rewards: list[float] = []

        for episode in range(self.config.training.planner_eval_episodes):
            observation, _ = self.env.reset(seed=self.config.training.seed + cycle * 100 + episode)
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                latent, _ = self.vae.encode(obs_tensor)

            cem_result = cem_planner.plan(latent)
            mcts_result = mcts_planner.plan(latent)
            cem_returns.append(cem_result.predicted_return)
            mcts_returns.append(mcts_result.predicted_return)

            chosen_action = (
                cem_result.action
                if cem_result.predicted_return >= mcts_result.predicted_return
                else mcts_result.action
            )
            action_np = chosen_action.detach().cpu().numpy().astype("float32")
            next_observation, reward, terminated, truncated, _ = self.env.step(action_np)
            self.buffer.add(
                Transition(
                    observation=observation,
                    action=action_np,
                    reward=reward,
                    next_observation=next_observation,
                    done=bool(terminated or truncated),
                )
            )
            executed_rewards.append(float(reward))

        cem_mean = float(sum(cem_returns) / len(cem_returns))
        mcts_mean = float(sum(mcts_returns) / len(mcts_returns))
        executed_mean = float(sum(executed_rewards) / len(executed_rewards))
        return cem_mean, mcts_mean, executed_mean

    @staticmethod
    def _init_metrics_file(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "cycle",
                    "collected_steps",
                    "average_collection_reward",
                    "vae_loss",
                    "rssm_loss",
                    "reward_loss",
                    "cem_predicted_return",
                    "mcts_predicted_return",
                    "executed_reward",
                ]
            )

    @staticmethod
    def _append_metrics(path: Path, metrics: CycleMetrics) -> None:
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    metrics.cycle,
                    metrics.collected_steps,
                    f"{metrics.average_collection_reward:.8f}",
                    f"{metrics.vae_loss:.8f}",
                    f"{metrics.rssm_loss:.8f}",
                    f"{metrics.reward_loss:.8f}",
                    f"{metrics.cem_predicted_return:.8f}",
                    f"{metrics.mcts_predicted_return:.8f}",
                    f"{metrics.executed_reward:.8f}",
                ]
            )
