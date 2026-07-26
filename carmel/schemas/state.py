# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Campaign lifecycle state schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CampaignStateValue(StrEnum):
    """Discrete states in the campaign lifecycle."""

    DRAFT = "draft"
    VALIDATED = "validated"
    READY_FOR_PLANNING = "ready_for_planning"
    PLAN_PENDING_APPROVAL = "plan_pending_approval"
    APPROVED_FOR_EXECUTION = "approved_for_execution"
    RUNNING_T3 = "running_t3"
    RUNNING_ARC = "running_arc"
    DIAGNOSTICS_READY = "diagnostics_ready"
    RESULTS_READY = "results_ready"
    COMPLETED_PHASE1 = "completed_phase1"
    BLOCKED = "blocked"
    FAILED = "failed"


class CampaignState(BaseModel):
    """Persisted state of a campaign."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1)
    state: CampaignStateValue
    updated_at: datetime
    notes: str | None = None
    failed_from: CampaignStateValue | None = None
    """The state the campaign was in immediately before it transitioned to
    ``FAILED``. ``None`` when the campaign has never failed, and cleared
    (set back to ``None``) as soon as the campaign leaves ``FAILED``.
    Used to gate the ``FAILED`` → ``APPROVED_FOR_EXECUTION`` retry edge so
    that only a genuine tool-execution failure of an already-approved plan
    (failed from ``RUNNING_T3``) can retry — a failure during validation
    or planning, before any human approved anything, must not bypass the
    HITL approval gate.
    """
