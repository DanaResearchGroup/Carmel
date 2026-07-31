"""Per-action execution state and plan progress — the multi-action backbone.

``PlanProgress`` is the persisted source of truth for how far a plan has
run: one :class:`ActionState` per planned action plus a ``cursor`` naming
the next action to consider. The campaign-level state becomes a
*projection* of this structure (see
:func:`carmel.services.plan_progress.aggregate_state`), never the other
way around.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from carmel.schemas.approval import ActionKind, ApprovalStatus


class ActionExecutionStatus(StrEnum):
    """Execution lifecycle of a single planned action."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionOutcome(StrEnum):
    """Typed outcome of a finished (or skipped) action."""

    NONE = "none"
    SUCCEEDED = "succeeded"
    FAILED_BLOCKING = "failed_blocking"
    FAILED_NONBLOCKING = "failed_nonblocking"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_GROUNDED_FINDINGS = "no_grounded_findings"
    REJECTED = "rejected"


#: Execution statuses from which an action never runs again.
TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        ActionExecutionStatus.SUCCEEDED,
        ActionExecutionStatus.FAILED,
        ActionExecutionStatus.SKIPPED,
    }
)


class ActionState(BaseModel):
    """Persisted per-action state within a plan."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    kind: ActionKind
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    execution_status: ActionExecutionStatus = ActionExecutionStatus.PENDING
    outcome: ActionOutcome = ActionOutcome.NONE
    attempt_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    blocking: bool = True
    updated_at: datetime
    notes: str | None = None

    def is_terminal(self) -> bool:
        """Whether this action can never run again."""
        return self.execution_status in TERMINAL_EXECUTION_STATUSES

    def is_executable(self) -> bool:
        """Whether this action could still run (now or after approval)."""
        return (
            self.execution_status in (ActionExecutionStatus.PENDING, ActionExecutionStatus.RUNNING)
            and self.approval_status != ApprovalStatus.REJECTED
        )


class PlanProgress(BaseModel):
    """Persisted progress through an ordered multi-action plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    plan_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    actions: list[ActionState]
    cursor: int = Field(default=0, ge=0)  # index of the next action to consider
    updated_at: datetime

    def next_action_id(self) -> str | None:
        """The action id at the cursor, or None when the plan is complete."""
        if self.cursor >= len(self.actions):
            return None
        return self.actions[self.cursor].action_id

    def is_complete(self) -> bool:
        """Whether the cursor has moved past the last action."""
        return self.cursor >= len(self.actions)

    def has_executable_remaining(self) -> bool:
        """Whether any action could still run (now or after approval)."""
        return any(a.is_executable() for a in self.actions)
