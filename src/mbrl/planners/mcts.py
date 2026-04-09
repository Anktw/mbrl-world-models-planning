"""Monte Carlo Tree Search planner in latent space."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM, RSSMState
from mbrl.planners.base import Planner, PlanningResult


@dataclass
class MCTSNode:
    """Single node in the latent-space planning tree."""

    state: RSSMState
    depth: int
    action_from_parent: torch.Tensor | None = None
    reward_from_parent: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters controlling MCTS planning behavior."""

    horizon: int = 8
    num_simulations: int = 100
    max_children: int = 8
    exploration_constant: float = 1.2
    discount: float = 0.99
    action_low: float = -2.0
    action_high: float = 2.0


class MCTSPlanner(Planner):
    """MCTS planner with random progressive widening for continuous actions."""

    def __init__(
        self,
        rssm: RSSM,
        reward_predictor: RewardPredictor,
        action_dim: int,
        device: torch.device,
        config: MCTSConfig | None = None,
    ) -> None:
        self.rssm = rssm.to(device)
        self.reward_predictor = reward_predictor.to(device)
        self.action_dim = action_dim
        self.device = device
        self.config = config or MCTSConfig()

    @torch.no_grad()
    def plan(self, initial_latent: torch.Tensor) -> PlanningResult:
        self.rssm.eval()
        self.reward_predictor.eval()

        latent = initial_latent.to(self.device)
        if latent.ndim != 2 or latent.shape[0] != 1:
            raise ValueError("initial_latent must have shape (1, latent_dim)")

        root_state = self.rssm.initial_state(batch_size=1, device=self.device)
        root_state = RSSMState(
            deterministic=root_state.deterministic,
            stochastic=latent,
            prior_mean=root_state.prior_mean,
            prior_logvar=root_state.prior_logvar,
        )
        root = MCTSNode(state=root_state, depth=0)

        for _ in range(self.config.num_simulations):
            path, rewards = self._select_and_expand(root)
            leaf = path[-1]
            rollout_return = self._rollout(leaf)
            self._backup(path, rewards, rollout_return)

        if not root.children:
            fallback = self._sample_action()
            return PlanningResult(action=fallback.cpu(), predicted_return=0.0)

        best_child = max(root.children, key=lambda child: child.visits)
        if best_child.action_from_parent is None:
            fallback = self._sample_action()
            return PlanningResult(action=fallback.cpu(), predicted_return=best_child.value)
        return PlanningResult(
            action=best_child.action_from_parent.detach().cpu(),
            predicted_return=best_child.value,
        )

    def _select_and_expand(self, root: MCTSNode) -> tuple[list[MCTSNode], list[float]]:
        node = root
        path = [node]
        rewards: list[float] = []

        while node.depth < self.config.horizon:
            if len(node.children) < self.config.max_children:
                child = self._expand(node)
                path.append(child)
                rewards.append(child.reward_from_parent)
                return path, rewards

            node = self._select_child(node)
            path.append(node)
            rewards.append(node.reward_from_parent)

        return path, rewards

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action = self._sample_action()
        next_state = self.rssm.imagine_step(node.state, action.unsqueeze(0))
        reward = float(self.reward_predictor(next_state.stochastic).item())
        child = MCTSNode(
            state=next_state,
            depth=node.depth + 1,
            action_from_parent=action,
            reward_from_parent=reward,
        )
        node.children.append(child)
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        assert node.children
        log_term = math.log(node.visits + 1.0)

        def score(child: MCTSNode) -> float:
            exploration = self.config.exploration_constant * math.sqrt(
                log_term / (child.visits + 1e-6)
            )
            return child.value + exploration

        return max(node.children, key=score)

    def _rollout(self, node: MCTSNode) -> float:
        state = node.state
        total_return = 0.0
        discount = 1.0

        for _ in range(node.depth, self.config.horizon):
            action = self._sample_action().unsqueeze(0)
            state = self.rssm.imagine_step(state, action)
            reward = float(self.reward_predictor(state.stochastic).item())
            total_return += discount * reward
            discount *= self.config.discount

        return total_return

    def _backup(self, path: list[MCTSNode], rewards: list[float], rollout_return: float) -> None:
        returns_from_node = rollout_return

        for index in range(len(path) - 1, 0, -1):
            reward = rewards[index - 1]
            returns_from_node = reward + self.config.discount * returns_from_node
            node = path[index]
            node.visits += 1
            node.value_sum += returns_from_node

        root_return = returns_from_node
        path[0].visits += 1
        path[0].value_sum += root_return

    def _sample_action(self) -> torch.Tensor:
        action = torch.empty(self.action_dim, device=self.device).uniform_(
            self.config.action_low,
            self.config.action_high,
        )
        return action
