# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Campaign lifecycle state machine."""

import contextlib
import fcntl
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.artifacts import read_json, write_json

WORKSPACE_LOCK_FILE_NAME = ".carmel.lock"


@contextlib.contextmanager
def workspace_lock(workspace_root: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on a campaign workspace.

    Serializes any read-check-write or append cycle against a
    ``<workspace_root>/.carmel.lock`` file so that concurrent callers (e.g.
    two overlapping ``/run`` requests) cannot interleave their operations
    into a lost-update race. The lock is process- and thread-safe: it is
    acquired via ``fcntl.flock``, which blocks other processes and other
    threads within this process alike until released.

    This uses ``fcntl``, which is POSIX-only. That is acceptable here:
    Carmel targets Linux CI and Linux/macOS local use, never Windows.

    Args:
        workspace_root: The campaign workspace root to lock.

    Yields:
        None. The lock is held for the duration of the ``with`` block.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    lock_path = workspace_root / WORKSPACE_LOCK_FILE_NAME
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
    # NOTE (spar round 3, P1-10): there is deliberately NO
    # APPROVED_FOR_EXECUTION -> COMPLETED_PHASE1 edge — at least one action
    # must actually run, so terminal success is only reachable through a
    # RUNNING_* state.
    CampaignStateValue.APPROVED_FOR_EXECUTION: frozenset(
        {
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.RUNNING_ARC,
            CampaignStateValue.RUNNING_LITERATURE,
            CampaignStateValue.BLOCKED,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.RUNNING_LITERATURE: frozenset(
        {
            CampaignStateValue.LITERATURE_READY,
            CampaignStateValue.FAILED,
            CampaignStateValue.BLOCKED,
        }
    ),
    # NOTE (spar round 3, P1-9): deliberately NO
    # LITERATURE_READY -> APPROVED_FOR_EXECUTION edge — that cycle would
    # re-dispatch literature forever. The only cycle the literature edges
    # create is DIAGNOSTICS_READY -> RUNNING_LITERATURE -> LITERATURE_READY
    # -> RUNNING_T3 (a second T3 after literature), which increment 1
    # rejects in `validate_plan_shape`, not here: the edges stay general for
    # a future Revision Agent.
    #
    # LITERATURE_READY -> RUNNING_LITERATURE, by contrast, is REQUIRED and was
    # the omission that made `carmel corpus-pass` inert (found by live run
    # 2026.08.01, not by the suite). A corpus pass is defined as a second pass
    # "over the papers this campaign already holds", and a campaign that holds
    # papers is in LITERATURE_READY and nothing else — so without this edge the
    # feature's only real use case raised InvalidTransitionError from BOTH the
    # CLI and the UI, and appending one WEDGED the campaign: the cursor parked
    # on an undispatchable action and the T3 run behind it became unreachable.
    #
    # This does NOT reintroduce the P1-9 cycle. That concern is about re-entering
    # APPROVED_FOR_EXECUTION, which re-dispatches the WHOLE plan from the top.
    # This edge re-enters only the running state for one already-appended action
    # that the cursor consumes exactly once, under a token budget the operator
    # named for it specifically. Appending the action is the authorisation; the
    # loop it could form is bounded by the plan, which `validate_plan_shape`
    # owns — consistent with the note above that these edges stay general.
    CampaignStateValue.LITERATURE_READY: frozenset(
        {
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.RUNNING_LITERATURE,
            CampaignStateValue.COMPLETED_PHASE1,
            CampaignStateValue.BLOCKED,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.RUNNING_T3: frozenset(
        {
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.RUNNING_ARC: frozenset(
        {
            CampaignStateValue.RESULTS_READY,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.DIAGNOSTICS_READY: frozenset(
        {
            CampaignStateValue.COMPLETED_PHASE1,
            CampaignStateValue.RUNNING_LITERATURE,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.RESULTS_READY: frozenset(
        {
            CampaignStateValue.COMPLETED_PHASE1,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.COMPLETED_PHASE1: frozenset(),
    CampaignStateValue.BLOCKED: frozenset(
        {
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.FAILED,
        }
    ),
    CampaignStateValue.FAILED: frozenset(
        {
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.RESULTS_READY,
        }
    ),
}


RECOVERY_TARGETS: dict[CampaignStateValue, CampaignStateValue] = {
    CampaignStateValue.RUNNING_T3: CampaignStateValue.APPROVED_FOR_EXECUTION,
    CampaignStateValue.RUNNING_ARC: CampaignStateValue.APPROVED_FOR_EXECUTION,
    # Finding P1-8: RUNNING_LITERATURE was added to CampaignStateValue and
    # VALID_TRANSITIONS but omitted here -- the only RUNNING_* member missing
    # from this allowlist. Without it a campaign that failed during
    # literature could never take the guarded FAILED -> APPROVED_FOR_EXECUTION
    # retry edge; `/retry` would 409 forever and only `/replan` could recover
    # it, discarding whatever approvals it already held.
    CampaignStateValue.RUNNING_LITERATURE: CampaignStateValue.APPROVED_FOR_EXECUTION,
    CampaignStateValue.APPROVED_FOR_EXECUTION: CampaignStateValue.APPROVED_FOR_EXECUTION,
    CampaignStateValue.DIAGNOSTICS_READY: CampaignStateValue.DIAGNOSTICS_READY,
    CampaignStateValue.RESULTS_READY: CampaignStateValue.RESULTS_READY,
}
"""Where a campaign that failed from a given state may resume *directly*.

``APPROVED_FOR_EXECUTION`` maps to itself because a plan can fail between
being approved and being launched — the adapter refuses to start, the
workspace is unwritable — and such a campaign has an approved plan and no
run. Sending it back through planning would discard an approval it
already holds, for a run that never happened.

Every other origin — and an unrecorded one — recovers only through
``READY_FOR_PLANNING``, which is always available (see
:func:`can_transition`).
"""


STATE_FILE_NAME = "campaign_state.json"


def can_transition(
    current: CampaignStateValue,
    target: CampaignStateValue,
    failed_from: CampaignStateValue | None = None,
) -> bool:
    """Check whether a state transition is allowed.

    Recovery out of ``FAILED`` obeys one rule: an exit may only return the
    campaign to a state it has demonstrably already reached, never to a
    later one. ``READY_FOR_PLANNING`` is therefore always available — it
    re-runs planning and the HITL approval gate from the start, so it
    cannot bypass anything — while the two direct resumes in
    :data:`RECOVERY_TARGETS` are each gated on the campaign having
    actually been in that state when it failed.

    Args:
        current: The current state.
        target: The proposed next state.
        failed_from: The state the campaign failed from, if ``current`` is
            ``FAILED``. Required to permit the direct resumes:
            ``APPROVED_FOR_EXECUTION`` (retry a tool run of an
            already-approved plan) is allowed only when the campaign failed
            from ``RUNNING_T3`` or ``RUNNING_ARC``, ``DIAGNOSTICS_READY``
            (adopt diagnostics already on disk) only when it failed from
            ``DIAGNOSTICS_READY``, and ``RESULTS_READY`` (adopt ARC results
            already on disk) only when it failed from ``RESULTS_READY``. A
            campaign that failed before planning or approval — from
            ``DRAFT``, ``VALIDATED``, ``READY_FOR_PLANNING``,
            ``PLAN_PENDING_APPROVAL``, or ``BLOCKED`` — must go back
            through the approval gate, and does so via
            ``READY_FOR_PLANNING``.

    Returns:
        True if the transition is allowed.
    """
    if target not in VALID_TRANSITIONS.get(current, frozenset()):
        return False
    if current == CampaignStateValue.FAILED and target != CampaignStateValue.READY_FOR_PLANNING:
        return failed_from is not None and RECOVERY_TARGETS.get(failed_from) == target
    return True


def assert_transition(
    current: CampaignStateValue,
    target: CampaignStateValue,
    failed_from: CampaignStateValue | None = None,
) -> None:
    """Raise InvalidTransitionError if the transition is not allowed."""
    if not can_transition(current, target, failed_from):
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
    with workspace_lock(workspace_root):
        current_state = load_state(workspace_root)
        assert_transition(current_state.state, target, current_state.failed_from)
        new_state = CampaignState(
            campaign_id=current_state.campaign_id,
            state=target,
            updated_at=datetime.now(UTC),
            notes=notes,
            failed_from=current_state.state if target == CampaignStateValue.FAILED else None,
        )
        save_state(workspace_root, new_state)
        return new_state
