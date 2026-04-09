import torch

from mbrl.models.vae import VAE, vae_loss


def test_vae_forward_shapes() -> None:
    model = VAE(image_channels=3, hidden_dim=128, latent_dim=16)
    inputs = torch.rand(2, 3, 64, 64)

    outputs = model(inputs)

    assert tuple(outputs["reconstruction"].shape) == (2, 3, 64, 64)
    assert tuple(outputs["mean"].shape) == (2, 16)
    assert tuple(outputs["logvar"].shape) == (2, 16)
    assert tuple(outputs["latents"].shape) == (2, 16)


def test_vae_loss_returns_components() -> None:
    model = VAE(image_channels=3, hidden_dim=128, latent_dim=16)
    inputs = torch.rand(2, 3, 64, 64)
    outputs = model(inputs)

    losses = vae_loss(
        reconstruction=outputs["reconstruction"],
        inputs=inputs,
        mean=outputs["mean"],
        logvar=outputs["logvar"],
        kl_beta=0.1,
    )

    assert losses.total.item() >= 0.0
    assert losses.reconstruction.item() >= 0.0
    assert losses.kl.item() >= 0.0
