"""Environment configuration schemas."""

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """Runtime settings for the control environment."""

    name: str = Field(default="Pendulum-v1")
    obs_shape: tuple[int, ...] = Field(default=(3,))
    action_dim: int = Field(default=1)
    continuous: bool = Field(default=True)
