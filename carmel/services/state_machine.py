"""Campaign lifecycle state machine."""

from datetime import UTC, datetime
from pathlib import Path

from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.artifacts import read_json, write_json


class InvalidTransitionError(ValueError):
    """Raised when a state transition is not allowed."""


VALID_TRANSITIONS: dict[CampaignStateValue, frozenset[CampaignStateValue]] = {
    CampaignStateValue.DRAFT: frozenset(
        {
            CampaignStateValue.VALIDATED,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.VALIDATED: frozenset(
        {
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.READY_FOR_PLANNING: frozenset(
        {
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.PLAN_PENDING_APPROVAL: frozenset(
        {
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.BLOCKED,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.APPROVED_FOR_EXECUTION: frozenset(
        {
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.RUNNING_T3: frozenset(
        {
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.DIAGNOSTICS_READY: frozenset(
        {
            CampaignStateValue.COMPLETED_PHASE1,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.COMPLETED_PHASE1: frozenset(),
    CampaignStateValue.BLOCKED: frozenset(
        {
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.FAILED: frozenset(),
}


STATE_FILE_NAME = "campaign_state.json"


def can_transition(current: CampaignStateValue, target: CampaignStateValue) -> bool:
    """Check whether a state transition is allowed.

    Args:
        current: The current state.
        target: The proposed next state.

    Returns:
        True if the transition is allowed.
    """
    return target in VALID_TRANSITIONS.get(current, frozenset())


def assert_transition(current: CampaignStateValue, target: CampaignStateValue) -> None:
    """Raise InvalidTransitionError if the transition is not allowed."""
    if not can_transition(current, target):
        allowed = sorted(s.value for s in VALID_TRANSITIONS.get(current, frozenset()))
        raise InvalidTransitionError(f"Cannot transition from {current.value} to {target.value}. Allowed: {allowed}")


def load_state(workspace_root: Path) -> CampaignState:
    """Load the persisted campaign state."""
    return CampaignState.model_validate(read_json(workspace_root / STATE_FILE_NAME))


def save_state(workspace_root: Path, state: CampaignState) -> None:
    """Persist a campaign state."""
    write_json(workspace_root / STATE_FILE_NAME, state)


def update_state(
    workspace_root: Path,
    target: CampaignStateValue,
    notes: str | None = None,
) -> CampaignState:
    """Validate and persist a state transition.

    Args:
        workspace_root: The campaign workspace root.
        target: The desired next state.
        notes: Optional human-readable notes about the transition.

    Returns:
        The new persisted state.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
    """
    current_state = load_state(workspace_root)
    assert_transition(current_state.state, target)
    new_state = CampaignState(
        campaign_id=current_state.campaign_id,
        state=target,
        updated_at=datetime.now(UTC),
        notes=notes,
    )
    save_state(workspace_root, new_state)
    return new_state
