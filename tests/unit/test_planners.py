import torch

from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM
from mbrl.planners.cem import CEMConfig, CEMPlanner
from mbrl.planners.mcts import MCTSConfig, MCTSPlanner


def test_cem_planner_returns_action_and_value() -> None:
    rssm = RSSM(latent_dim=8, action_dim=1, deterministic_dim=16, hidden_dim=32)
    reward_predictor = RewardPredictor(latent_dim=8, hidden_dim=16)
    planner = CEMPlanner(
        rssm=rssm,
        reward_predictor=reward_predictor,
        action_dim=1,
        device=torch.device("cpu"),
        config=CEMConfig(horizon=4, num_samples=16, num_iterations=2),
    )

    initial_latent = torch.zeros(1, 8)
    result = planner.plan(initial_latent)

    assert tuple(result.action.shape) == (1,)
    assert isinstance(result.predicted_return, float)


def test_mcts_planner_returns_action_and_value() -> None:
    rssm = RSSM(latent_dim=8, action_dim=1, deterministic_dim=16, hidden_dim=32)
    reward_predictor = RewardPredictor(latent_dim=8, hidden_dim=16)
    planner = MCTSPlanner(
        rssm=rssm,
        reward_predictor=reward_predictor,
        action_dim=1,
        device=torch.device("cpu"),
        config=MCTSConfig(horizon=4, num_simulations=20, max_children=4),
    )

    initial_latent = torch.zeros(1, 8)
    result = planner.plan(initial_latent)

    assert tuple(result.action.shape) == (1,)
    assert isinstance(result.predicted_return, float)
