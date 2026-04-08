"""Plan and planned-action schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from carmel.schemas.approval import ActionKind, ApprovalRequirement


class PlannedAction(BaseModel):
    """A single action proposed by the planner."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    kind: ActionKind
    description: str
    estimated_cpu_hours: float = Field(ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    rationale: str
    approval_requirement: ApprovalRequirement
    parameters: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A deterministic plan composed of one or more actions."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    created_at: datetime
    actions: list[PlannedAction]
    rationale: str
    total_estimated_cpu_hours: float = Field(ge=0)
    requires_approval: bool
