"""Model hyperparameter configuration schemas."""

from pydantic import BaseModel, Field


class VAEConfig(BaseModel):
    """Configuration for latent encoder-decoder module."""

    latent_dim: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=128, ge=1)
    image_channels: int = Field(default=3, ge=1)
    image_size: int = Field(default=64, ge=8)
    kl_beta: float = Field(default=0.1, ge=0.0)


class RSSMConfig(BaseModel):
    """Configuration for recurrent state-space dynamics model."""

    deterministic_dim: int = Field(default=128, ge=1)
    hidden_dim: int = Field(default=128, ge=1)
    stochastic_dim: int = Field(default=32, ge=1)


class RewardConfig(BaseModel):
    """Configuration for reward prediction head."""

    hidden_dim: int = Field(default=128, ge=1)


class ModelConfig(BaseModel):
    """Grouped model configuration container."""

    vae: VAEConfig = Field(default_factory=VAEConfig)
    rssm: RSSMConfig = Field(default_factory=RSSMConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
