"""Training configuration schemas."""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Core optimization and reproducibility settings."""

    seed: int = Field(default=7)
    batch_size: int = Field(default=32, ge=1)
    sequence_length: int = Field(default=8, ge=1)
    pipeline_cycles: int = Field(default=3, ge=1)
    bootstrap_steps: int = Field(default=128, ge=1)
    collect_steps_per_cycle: int = Field(default=64, ge=1)
    vae_epochs_per_cycle: int = Field(default=1, ge=1)
    rssm_epochs_per_cycle: int = Field(default=1, ge=1)
    reward_epochs_per_cycle: int = Field(default=1, ge=1)
    steps_per_epoch: int = Field(default=8, ge=1)
    planner_eval_episodes: int = Field(default=3, ge=1)
    buffer_refresh_interval: int = Field(default=0, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    max_epochs: int = Field(default=10, ge=1)
    device: str = Field(default="auto")
