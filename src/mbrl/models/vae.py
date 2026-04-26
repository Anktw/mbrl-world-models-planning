"""Convolutional variational autoencoder for image observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAELoss:
    """Container for VAE loss components."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class ConvEncoder(nn.Module):
    """CNN encoder that produces latent mean and log-variance."""

    def __init__(self, image_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.projection(self.features(inputs))
        return self.mean_head(hidden), self.logvar_head(hidden)


class ConvDecoder(nn.Module):
    """CNN decoder that reconstructs images from latent codes."""

    def __init__(self, image_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(latents)
        hidden = hidden.view(latents.shape[0], 256, 4, 4)
        return self.decoder(hidden)


class VAE(nn.Module):
    """Convolutional variational autoencoder with explicit sampling helper."""

    def __init__(self, image_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(image_channels, hidden_dim, latent_dim)
        self.decoder = ConvDecoder(image_channels, hidden_dim, latent_dim)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent code using the reparameterization trick."""

        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mean + noise * standard_deviation

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, logvar = self.encode(inputs)
        latents = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latents)
        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "logvar": logvar,
            "latents": latents,
        }


def vae_loss(
    reconstruction: torch.Tensor,
    inputs: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float,
    foreground_weight: float = 1.0,
    foreground_threshold: float = 0.0,
) -> VAELoss:
    """Compute reconstruction and KL losses for the VAE."""

    squared_error = (reconstruction - inputs) ** 2
    if foreground_weight > 1.0:
        # Upweight sparse foreground pixels so the decoder cannot minimize
        # loss by predicting only the dominant background.
        foreground_mask = (inputs > foreground_threshold).to(inputs.dtype)
        weights = 1.0 + (foreground_weight - 1.0) * foreground_mask
        reconstruction_loss = torch.mean(weights * squared_error)
    else:
        reconstruction_loss = torch.mean(squared_error)

    kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    total = reconstruction_loss + kl_beta * kl_divergence
    return VAELoss(total=total, reconstruction=reconstruction_loss, kl=kl_divergence)
