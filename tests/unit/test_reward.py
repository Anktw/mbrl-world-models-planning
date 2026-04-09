import torch

from mbrl.models.reward import RewardPredictor, reward_prediction_loss


def test_reward_predictor_forward_shape() -> None:
    model = RewardPredictor(latent_dim=16, hidden_dim=32)
    latents = torch.randn(4, 16)

    predicted_rewards = model(latents)

    assert tuple(predicted_rewards.shape) == (4, 1)


def test_reward_loss_is_non_negative() -> None:
    model = RewardPredictor(latent_dim=8, hidden_dim=16)
    latents = torch.randn(3, 8)
    targets = torch.randn(3, 1)

    loss = reward_prediction_loss(model(latents), targets)

    assert loss.item() >= 0.0
