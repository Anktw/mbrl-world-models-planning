"""Train the convolutional VAE from replay buffer data."""

from __future__ import annotations

import argparse

import torch

from mbrl.data.buffer import ReplayBuffer
from mbrl.models.vae import VAE
from mbrl.training.vae_trainer import VAETrainer
from mbrl.utils.device import select_device
from mbrl.utils.logging import configure_logging
from mbrl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE from replay buffer data")
    parser.add_argument("--buffer", default="artifacts/vae_buffer.npz", help="Replay buffer path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=10, help="Gradient steps per epoch")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--kl-beta", type=float, default=0.01, help="KL weight")
    parser.add_argument(
        "--foreground-weight",
        type=float,
        default=15.0,
        help="Weight multiplier for non-background reconstruction pixels",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.05,
        help="Threshold used to classify foreground pixels",
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=10,
        help="Linearly ramp KL weight over this many epochs",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--checkpoint", default="artifacts/vae_model.pt", help="Checkpoint path")
    parser.add_argument(
        "--reconstructions",
        default="artifacts/vae_reconstructions.png",
        help="Reconstruction plot path",
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
    channels = int(buffer.obs_shape[0])
    model = VAE(image_channels=channels, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    trainer = VAETrainer(
        model=model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        foreground_weight=args.foreground_weight,
        foreground_threshold=args.foreground_threshold,
        kl_warmup_epochs=args.kl_warmup_epochs,
        device=device,
    )

    result = trainer.fit(
        replay_buffer=buffer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_path=args.checkpoint,
        reconstruction_path=args.reconstructions,
        seed=args.seed,
    )

    print(f"loss history: {result.epoch_losses}")
    print(f"checkpoint saved to: {result.checkpoint_path}")
    print(f"reconstruction image saved to: {result.reconstruction_path}")
    sample = buffer.sample(batch_size=1, seed=args.seed)
    with torch.no_grad():
        outputs = trainer.model(sample["obs"].to(device))
    print(f"latent vector z shape: {tuple(outputs['latents'].shape)}")
