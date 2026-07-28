# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Plan and planned-action schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from carmel.schemas.approval import ActionKind, ApprovalRequirement

#: Current plan schema version. Version 2 adds per-action ``blocking`` and
#: ``estimated_spend_usd``; every added field has a default so a Phase-1
#: (version 1) ``plan.json`` still validates unchanged.
PLAN_SCHEMA_VERSION = 2


class PlannedAction(BaseModel):
    """A single action proposed by the planner."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    kind: ActionKind
    description: str
    estimated_cpu_hours: float = Field(ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    estimated_spend_usd: float = Field(default=0.0, ge=0)
    blocking: bool = True
    rationale: str
    approval_requirement: ApprovalRequirement
    parameters: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A deterministic plan composed of one or more ordered actions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = PLAN_SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    created_at: datetime
    actions: list[PlannedAction]
    rationale: str
    total_estimated_cpu_hours: float = Field(ge=0)
    requires_approval: bool
