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
    RUNNING_LITERATURE = "running_literature"
    LITERATURE_READY = "literature_ready"
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
    Gates the direct recovery edges out of ``FAILED`` (see
    ``RECOVERY_TARGETS`` in :mod:`carmel.services.state_machine`): retrying
    an already-approved plan via ``APPROVED_FOR_EXECUTION`` is allowed only
    when the campaign failed from ``RUNNING_T3``, ``RUNNING_ARC``, or
    ``APPROVED_FOR_EXECUTION`` itself, and adopting results already on disk
    via ``DIAGNOSTICS_READY``/``RESULTS_READY`` only when it failed from
    that same state. A failure during validation or planning, before any
    human approved anything, must not bypass the HITL approval gate.
    """
