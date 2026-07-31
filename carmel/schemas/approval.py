# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Approval policy and decision schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionKind(StrEnum):
    """Categories of actions that may require approval."""

    T3_RUN = "t3_run"
    ARC_RUN = "arc_run"
    EXPERIMENT = "experiment"  # reserved for future
    LITERATURE_SEARCH = "literature_search"
    LITERATURE_CORPUS_PASS = "literature_corpus_pass"
    """Re-read the papers the workspace already holds and ground findings in them.

    A distinct kind rather than a flag on ``LITERATURE_SEARCH`` because the two differ
    in what they are permitted to do -- one may reach the network and spend on
    fetches, the other may not -- and that is exactly the sort of thing the approval
    and authorization layers key off. A flag would make the permitted behaviour
    invisible to every reader that switches on kind.
    """


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

    The compute-side T3 and ARC thresholds are both enforced (see
    ``carmel.services.approvals.evaluate_action``). Experiment and
    literature actions are scaffolded for future expansion.
    """

    model_config = ConfigDict(extra="forbid")

    auto_approve_t3_under_cpu_hours: float = Field(default=10.0, ge=0)
    auto_approve_arc_under_cpu_hours: float = Field(default=5.0, ge=0)
    require_approval_for_experiments: bool = True
    require_approval_for_literature: bool = False
    auto_approve_literature_under_usd: float = Field(default=2.0, ge=0)


class ApprovalDecision(BaseModel):
    """A recorded approval decision for an action."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    status: ApprovalStatus
    decided_at: datetime
    decided_by: str = Field(min_length=1)
    rationale: str | None = None
