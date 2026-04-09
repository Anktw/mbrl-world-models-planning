"""Root configuration loading and composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mbrl.config.env import EnvConfig
from mbrl.config.model import ModelConfig
from mbrl.config.planning import PlanningConfig
from mbrl.config.training import TrainingConfig


class DataConfig(BaseModel):
    """Data and replay buffer settings."""

    capacity: int = Field(default=10000, ge=1)
    min_size: int = Field(default=64, ge=1)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    env: EnvConfig = Field(default_factory=EnvConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config content must be a mapping at the root level")
    return loaded


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate an app configuration from YAML."""

    path = Path(config_path)
    payload = _read_yaml(path)
    return AppConfig.model_validate(payload)
