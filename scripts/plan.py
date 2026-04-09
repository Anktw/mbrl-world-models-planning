"""Compare CEM and MCTS planners in latent space."""

from __future__ import annotations

import argparse

import torch

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.reward import RewardPredictor
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.planners.cem import CEMConfig, CEMPlanner
from mbrl.planners.mcts import MCTSConfig, MCTSPlanner
from mbrl.utils.device import select_device
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan actions in latent space using CEM and MCTS")
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
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent size")
    parser.add_argument("--vae-hidden-dim", type=int, default=128, help="VAE hidden size")
    parser.add_argument("--rssm-hidden-dim", type=int, default=128, help="RSSM hidden size")
    parser.add_argument(
        "--deterministic-dim",
        type=int,
        default=128,
        help="RSSM deterministic state size",
    )
    parser.add_argument(
        "--reward-hidden-dim",
        type=int,
        default=64,
        help="Reward predictor hidden size",
    )
    parser.add_argument("--horizon", type=int, default=8, help="Planning horizon")
    parser.add_argument("--cem-samples", type=int, default=128, help="CEM action sequence samples")
    parser.add_argument("--cem-iterations", type=int, default=5, help="CEM update iterations")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    set_global_seed(args.seed)

    device = select_device(args.device)
    replay_buffer = ReplayBuffer.load(args.buffer)

    vae = VAE(
        image_channels=int(replay_buffer.obs_shape[0]),
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    vae_state = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae.eval()

    rssm = RSSM(
        latent_dim=args.latent_dim,
        action_dim=replay_buffer.action_dim,
        deterministic_dim=args.deterministic_dim,
        hidden_dim=args.rssm_hidden_dim,
    ).to(device)
    rssm_state = torch.load(args.rssm_checkpoint, map_location=device)
    rssm.load_state_dict(rssm_state["model_state_dict"])
    rssm.eval()

    reward_predictor = RewardPredictor(
        latent_dim=args.latent_dim,
        hidden_dim=args.reward_hidden_dim,
    ).to(device)
    reward_state = torch.load(args.reward_checkpoint, map_location=device)
    reward_predictor.load_state_dict(reward_state["model_state_dict"])
    reward_predictor.eval()

    batch = replay_buffer.sample(batch_size=1, seed=args.seed)
    with torch.no_grad():
        initial_latent, _ = vae.encode(batch["obs"].to(device))

    cem_planner = CEMPlanner(
        rssm=rssm,
        reward_predictor=reward_predictor,
        action_dim=replay_buffer.action_dim,
        device=device,
        config=CEMConfig(
            horizon=args.horizon,
            num_samples=args.cem_samples,
            num_iterations=args.cem_iterations,
        ),
    )
    mcts_planner = MCTSPlanner(
        rssm=rssm,
        reward_predictor=reward_predictor,
        action_dim=replay_buffer.action_dim,
        device=device,
        config=MCTSConfig(
            horizon=args.horizon,
            num_simulations=args.mcts_simulations,
        ),
    )

    cem_result = cem_planner.plan(initial_latent)
    mcts_result = mcts_planner.plan(initial_latent)

    print("chosen actions:")
    print(
        "  CEM action: "
        f"{cem_result.action.tolist()} "
        f"predicted_return={cem_result.predicted_return:.6f}"
    )
    print(
        "  MCTS action: "
        f"{mcts_result.action.tolist()} "
        f"predicted_return={mcts_result.predicted_return:.6f}"
    )

    if cem_result.predicted_return >= mcts_result.predicted_return:
        print("planner comparison winner: CEM")
    else:
        print("planner comparison winner: MCTS")
