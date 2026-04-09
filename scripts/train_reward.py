"""Train latent reward predictor from replay buffer data."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.reward import RewardPredictor
from mbrl.models.vae import VAE
from mbrl.training.reward_trainer import RewardTrainer
from mbrl.utils.device import select_device
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reward predictor from VAE latents")
    parser.add_argument("--buffer", default="artifacts/vae_buffer.npz", help="Replay buffer path")
    parser.add_argument(
        "--vae-checkpoint",
        default="artifacts/vae_model.pt",
        help="Trained VAE checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=8, help="Gradient steps per epoch")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent size")
    parser.add_argument("--vae-hidden-dim", type=int, default=128, help="VAE hidden size")
    parser.add_argument("--reward-hidden-dim", type=int, default=64, help="Reward MLP hidden size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--checkpoint", default="artifacts/reward_model.pt", help="Checkpoint path")
    parser.add_argument(
        "--comparison",
        default="artifacts/reward_comparison.png",
        help="Prediction comparison plot path",
    )
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    set_global_seed(args.seed)

    replay_buffer = ReplayBuffer.load(args.buffer)
    device = select_device(args.device)

    vae = VAE(
        image_channels=int(replay_buffer.obs_shape[0]),
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
    )
    vae_state = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])

    reward_predictor = RewardPredictor(
        latent_dim=args.latent_dim,
        hidden_dim=args.reward_hidden_dim,
    )
    trainer = RewardTrainer(
        predictor=reward_predictor,
        vae=vae,
        learning_rate=args.learning_rate,
        device=device,
    )

    result = trainer.fit(
        replay_buffer=replay_buffer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_path=args.checkpoint,
        comparison_path=args.comparison,
        seed=args.seed,
    )

    print(f"loss history: {result.epoch_losses}")
    print(f"checkpoint saved to: {result.checkpoint_path}")
    print(f"comparison plot saved to: {result.comparison_path}")

    validation = replay_buffer.sample(batch_size=5, seed=args.seed)
    with torch.no_grad():
        latents, _ = vae.encode(validation["obs"].to(device))
        predictions = reward_predictor(latents)
    actual_rewards = validation["rewards"].to(device)
    mse = F.mse_loss(predictions, actual_rewards).item()

    print("predicted vs actual rewards:")
    for index in range(predictions.shape[0]):
        predicted = float(predictions[index].item())
        actual = float(actual_rewards[index].item())
        print(f"  sample {index}: predicted={predicted:.6f} actual={actual:.6f}")
    print(f"reward mse: {mse:.6f}")
