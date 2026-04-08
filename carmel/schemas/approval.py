"""Approval policy and decision schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionKind(StrEnum):
    """Categories of actions that may require approval."""

    T3_RUN = "t3_run"
    ARC_RUN = "arc_run"  # reserved for future
    EXPERIMENT = "experiment"  # reserved for future
    LITERATURE_SEARCH = "literature_search"  # reserved for future


class ApprovalRequirement(StrEnum):
    """Whether an action is auto-approvable or requires human approval."""

    AUTO_APPROVED = "auto_approved"
    REQUIRES_APPROVAL = "requires_approval"


class ApprovalStatus(StrEnum):
    """Final status of an approval decision."""

    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalPolicy(BaseModel):
    """Thresholds and rules for auto-approval vs human approval.

    Phase 1 only enforces compute-side T3 thresholds. Other action kinds
    are scaffolded for future expansion.
    """

    model_config = ConfigDict(extra="forbid")

    auto_approve_t3_under_cpu_hours: float = Field(default=10.0, ge=0)
    auto_approve_arc_under_cpu_hours: float = Field(default=5.0, ge=0)
    require_approval_for_experiments: bool = True
    require_approval_for_literature: bool = False


class ApprovalDecision(BaseModel):
    """A recorded approval decision for an action."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    status: ApprovalStatus
    decided_at: datetime
    decided_by: str = Field(min_length=1)
    rationale: str | None = None
