"""Planning configuration schemas used by the end-to-end loop."""

from pydantic import BaseModel, Field


class CEMPlannerConfig(BaseModel):
    """CEM planner hyperparameters."""

    horizon: int = Field(default=8, ge=1)
    num_samples: int = Field(default=128, ge=1)
    num_iterations: int = Field(default=5, ge=1)
    elite_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    min_std: float = Field(default=0.05, gt=0.0)
    discount: float = Field(default=0.99, gt=0.0, le=1.0)
    action_low: float = Field(default=-2.0)
    action_high: float = Field(default=2.0)


class MCTSPlannerConfig(BaseModel):
    """MCTS planner hyperparameters."""

    horizon: int = Field(default=8, ge=1)
    num_simulations: int = Field(default=100, ge=1)
    max_children: int = Field(default=8, ge=1)
    exploration_constant: float = Field(default=1.2, gt=0.0)
    discount: float = Field(default=0.99, gt=0.0, le=1.0)
    action_low: float = Field(default=-2.0)
    action_high: float = Field(default=2.0)


class PlanningConfig(BaseModel):
    """Top-level planning config."""

    cem: CEMPlannerConfig = Field(default_factory=CEMPlannerConfig)
    mcts: MCTSPlannerConfig = Field(default_factory=MCTSPlannerConfig)
