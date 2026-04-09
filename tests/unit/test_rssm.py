import torch

from mbrl.models.rssm import RSSM


def test_rssm_rollout_shapes() -> None:
    model = RSSM(latent_dim=32, action_dim=1, deterministic_dim=64, hidden_dim=128)
    latents = torch.randn(4, 6, 32)
    actions = torch.randn(4, 5, 1)

    predicted_latents, states = model.rollout_sequence(latents[:, 0], actions)

    assert tuple(predicted_latents.shape) == (4, 5, 32)
    assert len(states) == 5
    assert tuple(states[0].deterministic.shape) == (4, 64)
    assert tuple(states[0].stochastic.shape) == (4, 32)


def test_rssm_imagine_step_updates_state() -> None:
    model = RSSM(latent_dim=16, action_dim=2, deterministic_dim=32, hidden_dim=64)
    state = model.initial_state(batch_size=3, device=torch.device("cpu"))
    action = torch.randn(3, 2)

    next_state = model.imagine_step(state, action)

    assert tuple(next_state.deterministic.shape) == (3, 32)
    assert tuple(next_state.stochastic.shape) == (3, 16)
    assert tuple(next_state.prior_mean.shape) == (3, 16)
