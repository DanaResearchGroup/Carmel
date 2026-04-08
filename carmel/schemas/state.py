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
    DIAGNOSTICS_READY = "diagnostics_ready"
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
