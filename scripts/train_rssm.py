"""Train the RSSM on latent sequences extracted from replay buffer data."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.rssm import RSSM
from mbrl.models.vae import VAE
from mbrl.training.rssm_trainer import RSSMTrainer
from mbrl.utils.device import select_device
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RSSM on replay-buffer sequences")
    parser.add_argument("--buffer", default="artifacts/vae_buffer.npz", help="Replay buffer path")
    parser.add_argument(
        "--vae-checkpoint",
        default="artifacts/vae_model.pt",
        help="Trained VAE checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Sequence length in transitions",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=4, help="Gradient steps per epoch")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent size")
    parser.add_argument("--vae-hidden-dim", type=int, default=128, help="VAE hidden size")
    parser.add_argument("--rssm-hidden-dim", type=int, default=128, help="RSSM hidden size")
    parser.add_argument(
        "--deterministic-dim",
        type=int,
        default=128,
        help="RSSM deterministic size",
    )
    parser.add_argument("--stochastic-dim", type=int, default=32, help="RSSM stochastic size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--checkpoint", default="artifacts/rssm_model.pt", help="Checkpoint path")
    parser.add_argument(
        "--comparison",
        default="artifacts/rssm_latent_comparison.png",
        help="Comparison plot path",
    )
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    set_global_seed(args.seed)

    buffer = ReplayBuffer.load(args.buffer)
    device = select_device(args.device)

    vae = VAE(
        image_channels=int(buffer.obs_shape[0]),
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
    )
    vae_state = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])

    rssm = RSSM(
        latent_dim=args.latent_dim,
        action_dim=buffer.action_dim,
        deterministic_dim=args.deterministic_dim,
        hidden_dim=args.rssm_hidden_dim,
    )
    trainer = RSSMTrainer(rssm=rssm, vae=vae, learning_rate=args.learning_rate, device=device)

    result = trainer.fit(
        replay_buffer=buffer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_path=args.checkpoint,
        comparison_path=args.comparison,
        seed=args.seed,
    )

    print(f"loss history: {result.epoch_losses}")
    print(f"checkpoint saved to: {result.checkpoint_path}")
    print(f"latent comparison saved to: {result.comparison_path}")

    validation_batch = buffer.sample_sequences(
        sequence_length=args.sequence_length,
        batch_size=1,
        seed=args.seed,
    )
    with torch.no_grad():
        obs_sequences = validation_batch["obs"].to(device)
        actions = validation_batch["actions"].to(device)
        flat_obs = obs_sequences.reshape(-1, *obs_sequences.shape[2:])
        encoded_mean, _ = vae.encode(flat_obs)
        latents = encoded_mean.reshape(1, args.sequence_length + 1, -1)
        predicted_latents, _ = rssm.rollout_sequence(latents[:, 0], actions)
    print(f"predicted latent shape: {tuple(predicted_latents.shape)}")
    print(f"actual latent shape: {tuple(latents[:, 1:].shape)}")
    print(f"latent mse: {F.mse_loss(predicted_latents, latents[:, 1:]).item():.6f}")
