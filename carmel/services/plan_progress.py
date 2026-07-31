"""Persistence and crash recovery for per-action plan progress.

``plan_progress.json`` is the source of truth for how far a plan has run.
The campaign-level state is a *projection* of it (:func:`aggregate_state`)
and :func:`reconcile` repairs the two after a crash — the dispatcher calls
it first, before ever reading the cursor.
"""

from __future__ import annotations

import json
import os
import socket
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carmel.logger import get_logger
from carmel.schemas.action_state import (
    TERMINAL_EXECUTION_STATUSES,
    ActionExecutionStatus,
    ActionOutcome,
    ActionState,
    PlanProgress,
)
from carmel.schemas.approval import (
    LITERATURE_ACTION_KINDS,
    ActionKind,
    ApprovalRequirement,
    ApprovalStatus,
)
from carmel.schemas.plan import Plan, PlannedAction
from carmel.schemas.state import CampaignStateValue
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_typed_event
from carmel.services.recovery import probe_run_liveness
from carmel.services.state_machine import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    can_transition,
    load_state,
    update_state,
    workspace_lock,
)

PLAN_PROGRESS_NAME = "plan_progress.json"

#: Default staleness horizon for an attempt lease (2 x the default literature
#: ``max_wall_clock_s`` of 1800 s). A ``RUNNING`` action younger than this with
#: no other liveness signal is treated as in flight, never silently re-run.
DEFAULT_STALE_AFTER_S = 3600.0

#: Grace period for a lock whose ``info.json`` is missing or malformed: within
#: this window of the lock dir's mtime a peer may be mid-publication (between
#: ``mkdir`` and its ``info.json`` write), so the lock is treated as LIVE —
#: fail closed. Only past the grace period may unreadable metadata be stale.
DEFAULT_LOCK_GRACE_S = 30.0

_log = get_logger("services.plan_progress")


class ActionInFlightError(RuntimeError):
    """A live attempt already exists for this workspace/action."""


def _now() -> datetime:
    return datetime.now(UTC)


def _approval_from_requirement(requirement: ApprovalRequirement) -> ApprovalStatus:
    """Seed an action's approval status from its planner-evaluated requirement.

    CRITICAL (spar round 3, P0-2): ``AUTO_APPROVED -> AUTO_APPROVED`` and
    ``REQUIRES_APPROVAL -> PENDING``. Defaulting everything to ``PENDING``
    would deadlock every auto-approved plan, because the dispatcher refuses
    to run an unapproved action and nothing else would ever approve it.
    """
    if requirement == ApprovalRequirement.AUTO_APPROVED:
        return ApprovalStatus.AUTO_APPROVED
    return ApprovalStatus.PENDING


def init_progress(workspace_root: Path, plan: Plan) -> PlanProgress:
    """Create and persist fresh progress for a plan.

    Each :class:`ActionState`'s ``approval_status`` is seeded FROM the
    action's ``approval_requirement`` (see :func:`_approval_from_requirement`;
    spar round 3, P0-2). :func:`load_or_init_progress` applies the same
    mapping when migrating a Phase-1 plan.
    """
    now = _now()
    progress = PlanProgress(
        plan_id=plan.plan_id,
        campaign_id=plan.campaign_id,
        actions=[
            ActionState(
                action_id=action.action_id,
                kind=action.kind,
                approval_status=_approval_from_requirement(action.approval_requirement),
                blocking=action.blocking,
                updated_at=now,
            )
            for action in plan.actions
        ],
        cursor=0,
        updated_at=now,
    )
    save_progress(workspace_root, progress)
    return progress


def load_progress(workspace_root: Path) -> PlanProgress:
    """Load the persisted plan progress."""
    return PlanProgress.model_validate(read_json(workspace_root / PLAN_PROGRESS_NAME))


def save_progress(workspace_root: Path, progress: PlanProgress) -> None:
    """Persist plan progress."""
    write_json(workspace_root / PLAN_PROGRESS_NAME, progress)


def load_or_init_progress(workspace_root: Path, plan: Plan) -> PlanProgress:
    """Load progress, migrating a pre-progress (Phase-1) plan by initialising it.

    A persisted progress belonging to a DIFFERENT plan (the plan was
    regenerated) is also re-initialised — action ids would not line up.
    """
    path = workspace_root / PLAN_PROGRESS_NAME
    if not path.exists():
        return init_progress(workspace_root, plan)
    progress = load_progress(workspace_root)
    if progress.plan_id != plan.plan_id:
        _log.warning(
            "plan_progress.json belongs to plan %s but current plan is %s; re-initialising",
            progress.plan_id,
            plan.plan_id,
        )
        return init_progress(workspace_root, plan)
    return progress


def append_action_to_progress(workspace_root: Path, action: PlannedAction, *, index: int | None = None) -> PlanProgress:
    """Add state for an action inserted into an already-initialised plan.

    Used when an operator appends work to a live campaign. The state is inserted at
    the SAME position the action occupies in the plan, so progress and plan stay
    index-aligned; ``index=None`` appends at the end.

    Refuses to insert BEHIND the cursor. The cursor only ever moves forward (see
    :func:`advance_cursor`, which never rewinds), so an action placed behind it would
    be silently skipped forever -- a new action that never runs is worse than a
    refused command, because the operator has no way to tell the two apart from the
    plan alone.

    Raises:
        ValueError: If the plan has already progressed past the insertion point.
    """
    with workspace_lock(workspace_root):
        return append_action_to_progress_locked(workspace_root, action, index=index)


def append_action_to_progress_locked(
    workspace_root: Path, action: PlannedAction, *, index: int | None = None
) -> PlanProgress:
    """:func:`append_action_to_progress` with the workspace lock ALREADY held.

    Exists so a caller that must update the plan and progress together can hold one
    lock across both writes. ``workspace_lock`` is an ``fcntl.flock``, which conflicts
    with itself even within a single process, so such a caller cannot simply wrap the
    locking version -- it would deadlock against itself.

    Call this ONLY from inside ``with workspace_lock(workspace_root):``.
    """
    progress = load_progress(workspace_root)
    at = len(progress.actions) if index is None else index
    if at < progress.cursor:
        raise ValueError(
            f"cannot insert an action at position {at}: the plan has already progressed "
            f"past it (cursor is at {progress.cursor}), so it would never run"
        )
    progress.actions.insert(
        at,
        ActionState(
            action_id=action.action_id,
            kind=action.kind,
            approval_status=_approval_from_requirement(action.approval_requirement),
            blocking=action.blocking,
            updated_at=_now(),
        ),
    )
    progress.updated_at = _now()
    save_progress(workspace_root, progress)
    return progress


def _require_action(progress: PlanProgress, action_id: str) -> int:
    for i, action in enumerate(progress.actions):
        if action.action_id == action_id:
            return i
    raise KeyError(f"action {action_id!r} not found in plan progress")


def set_approval(workspace_root: Path, action_id: str, status: ApprovalStatus) -> PlanProgress:
    """Record an approval decision for one action.

    Approving a previously-rejected action UNDOES the skip (spar round 3,
    P0-5): when ``status`` is APPROVED/AUTO_APPROVED and the action's
    execution_status is SKIPPED, it is reset to PENDING, its outcome
    cleared, and the cursor rewound to the earliest action that is now
    executable — otherwise a rejected-then-approved action would be
    stranded behind the cursor forever. NOTE this un-skip is action-level
    only: a campaign that entered BLOCKED stays BLOCKED (main removed the
    ``BLOCKED -> APPROVED_FOR_EXECUTION`` edge — a rejected plan is
    re-planned, not un-rejected).

    The read-modify-write runs under the workspace lock so a concurrent
    dispatcher bookkeeping write cannot be lost.
    """
    with workspace_lock(workspace_root):
        return _set_approval_locked(workspace_root, action_id, status)


def _set_approval_locked(workspace_root: Path, action_id: str, status: ApprovalStatus) -> PlanProgress:
    progress = load_progress(workspace_root)
    idx = _require_action(progress, action_id)
    action = progress.actions[idx]
    updates: dict[str, Any] = {"approval_status": status, "updated_at": _now()}
    if (
        status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        and action.execution_status == ActionExecutionStatus.SKIPPED
    ):
        updates["execution_status"] = ActionExecutionStatus.PENDING
        updates["outcome"] = ActionOutcome.NONE
    progress.actions[idx] = action.model_copy(update=updates)

    executable_indices = [
        i
        for i, a in enumerate(progress.actions)
        if a.execution_status == ActionExecutionStatus.PENDING and a.approval_status != ApprovalStatus.REJECTED
    ]
    if executable_indices and executable_indices[0] < progress.cursor:
        progress.cursor = executable_indices[0]
    progress.updated_at = _now()
    save_progress(workspace_root, progress)
    return progress


def mark_running(workspace_root: Path, action_id: str, attempt_id: str) -> PlanProgress:
    """Mark an action as RUNNING with a new attempt id.

    The read-check-write is atomic under the workspace lock: of two racing
    callers, exactly one marks the action RUNNING and the other raises
    :class:`ActionInFlightError`.

    Raises:
        ActionInFlightError: If the action is already RUNNING.
        ValueError: If the action already reached a terminal status — a
            completed action must never re-run.
    """
    with workspace_lock(workspace_root):
        progress = load_progress(workspace_root)
        idx = _require_action(progress, action_id)
        action = progress.actions[idx]
        if action.execution_status == ActionExecutionStatus.RUNNING:
            raise ActionInFlightError(f"action {action_id!r} already has a live attempt")
        if action.is_terminal():
            raise ValueError(
                f"action {action_id!r} is already terminal ({action.execution_status.value}) and must never re-run"
            )
        progress.actions[idx] = action.model_copy(
            update={
                "execution_status": ActionExecutionStatus.RUNNING,
                "attempt_ids": [*action.attempt_ids, attempt_id],
                "updated_at": _now(),
            }
        )
        progress.updated_at = _now()
        save_progress(workspace_root, progress)
        return progress


def mark_finished(
    workspace_root: Path,
    action_id: str,
    *,
    status: ActionExecutionStatus,
    outcome: ActionOutcome,
    run_id: str | None = None,
    notes: str | None = None,
) -> PlanProgress:
    """Record the terminal status and outcome of an action (atomic under the workspace lock)."""
    with workspace_lock(workspace_root):
        progress = load_progress(workspace_root)
        idx = _require_action(progress, action_id)
        action = progress.actions[idx]
        updates: dict[str, Any] = {
            "execution_status": status,
            "outcome": outcome,
            "updated_at": _now(),
            "notes": notes,
        }
        if run_id is not None:
            updates["run_id"] = run_id
        progress.actions[idx] = action.model_copy(update=updates)
        progress.updated_at = _now()
        save_progress(workspace_root, progress)
        return progress


def advance_cursor(workspace_root: Path, action_id: str | None = None) -> PlanProgress:
    """Advance the cursor past THIS action (atomic under the workspace lock).

    Identity-based and idempotent (spar review finding 8): the caller passes
    the ``action_id`` it just finished at the cursor. A blind ``cursor += 1``
    is unsound here because ``mark_finished`` (one lock window) and
    ``advance_cursor`` (a separate, LATER lock window) are not atomic
    together: a concurrent ``/approve`` for the very action at the cursor
    (which does not take the dispatch lock) can reset it back to PENDING in
    between — same index, same ``action_id`` — and a blind increment would
    skip straight past it, stranding it permanently BEHIND the cursor:
    ``reconcile`` only ever advances the cursor forward over terminal
    actions and never rewinds it to an earlier PENDING one.

    Because the action's ``action_id`` alone does not change across that
    race (it is the same index the whole time), the guard checks BOTH that
    the cursor still points at ``action_id`` AND that the action there is
    still terminal (:meth:`ActionState.is_terminal`) — i.e. that it is
    still, at the moment we hold the lock, the finished action the caller
    thinks it is. If a concurrent approval reopened it to PENDING in the
    meantime, the terminal check fails, the cursor is left where it is, and
    the next dispatch pass (or ``reconcile``) re-evaluates it instead of
    losing it behind an advanced cursor.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The action the caller believes is at the cursor and has
            just finished. When omitted, preserves the old unconditional
            "advance by one" behavior for any not-yet-migrated caller.
    """
    with workspace_lock(workspace_root):
        progress = load_progress(workspace_root)
        if progress.cursor < len(progress.actions) and (
            action_id is None
            or (
                progress.actions[progress.cursor].action_id == action_id
                and progress.actions[progress.cursor].is_terminal()
            )
        ):
            progress.cursor += 1
            progress.updated_at = _now()
            save_progress(workspace_root, progress)
        return progress


def aggregate_state(progress: PlanProgress) -> CampaignStateValue | None:
    """Project the campaign state from per-action progress.

    Projection, not source of truth. Returns None while the plan is in
    flight. Terminal projection is GATED (spar round 3, P1-10) — an
    unguarded rule would let a plan whose every action was skipped report
    success:

    - ``FAILED`` when a blocking action ended ``FAILED_BLOCKING``.
    - ``COMPLETED_PHASE1`` requires: cursor past the end AND every blocking
      action SUCCEEDED AND at least one action SUCCEEDED.
    - ``BLOCKED`` when nothing executable remains and at least one action
      was REJECTED.
    - None otherwise (including "cursor past end but nothing succeeded" —
      that is a reconcile bug, and returning None surfaces it instead of
      hiding it).
    """
    if any(a.outcome == ActionOutcome.FAILED_BLOCKING for a in progress.actions):
        return CampaignStateValue.FAILED
    blocking = [a for a in progress.actions if a.blocking]
    any_succeeded = any(a.execution_status == ActionExecutionStatus.SUCCEEDED for a in progress.actions)
    if (
        progress.is_complete()
        and any_succeeded
        and all(a.execution_status == ActionExecutionStatus.SUCCEEDED for a in blocking)
    ):
        return CampaignStateValue.COMPLETED_PHASE1
    if not progress.has_executable_remaining() and any(
        a.approval_status == ApprovalStatus.REJECTED or a.outcome == ActionOutcome.REJECTED for a in progress.actions
    ):
        return CampaignStateValue.BLOCKED
    return None


# --------------------------- attempt results ----------------------------------

#: Mirrors ``carmel.services.execution.RUNS_DIR_NAME`` (kept as a literal so
#: this light module never imports the execution stack).
RUNS_DIR_NAME = "runs"


def attempt_result_path(workspace_root: Path, attempt_id: str) -> Path:
    """Path of the marker mapping one attempt to its persisted run result."""
    return workspace_root / RUNS_DIR_NAME / f"attempt_{attempt_id}.json"


def record_attempt_result(
    workspace_root: Path,
    *,
    action_id: str,
    attempt_id: str,
    run_id: str,
    status: ActionExecutionStatus,
    outcome: ActionOutcome,
) -> Path:
    """Persist the attempt-to-run mapping for a finished handler invocation.

    Written by the handler IMMEDIATELY after it persists its RunRecord (and
    other side effects), so a crash before ``mark_finished`` leaves behind a
    marker :func:`reconcile` can ADOPT instead of discarding real completed
    work by marking the stale attempt failed (adversarial review, defect 3).
    ``RunRecord`` has no attempt field and is owned elsewhere, so the mapping
    lives in this small sibling file under ``runs/``.
    """
    path = attempt_result_path(workspace_root, attempt_id)
    write_json(
        path,
        {
            "action_id": action_id,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "status": status.value,
            "outcome": outcome.value,
            "recorded_at": _now().isoformat(),
        },
    )
    return path


def read_attempt_result(workspace_root: Path, attempt_id: str) -> dict[str, Any] | None:
    """Best-effort read of a persisted attempt result; None when absent or unusable."""
    try:
        parsed = json.loads(attempt_result_path(workspace_root, attempt_id).read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _adopt_attempt_result(workspace_root: Path, action: ActionState) -> ActionState | None:
    """The repaired ActionState when the interrupted attempt persisted a result.

    A crash after the handler's side effects (diagnostics written, RunRecord
    persisted, attempt marker recorded) but before ``mark_finished`` must not
    throw the completed work away: when a terminal result was persisted for
    the action's latest attempt, reconcile adopts it — status, outcome and
    run_id — instead of marking the attempt failed (defect 3).
    """
    if not action.attempt_ids:
        return None
    recorded = read_attempt_result(workspace_root, action.attempt_ids[-1])
    if recorded is None or recorded.get("action_id") != action.action_id:
        return None
    try:
        status = ActionExecutionStatus(recorded["status"])
        outcome = ActionOutcome(recorded["outcome"])
    except KeyError, ValueError:
        return None
    run_id = recorded.get("run_id")
    if status not in TERMINAL_EXECUTION_STATUSES or not isinstance(run_id, str):
        return None
    return action.model_copy(
        update={
            "execution_status": status,
            "outcome": outcome,
            "run_id": run_id,
            "updated_at": _now(),
            "notes": "adopted persisted result from interrupted attempt",
        }
    )


# --------------------------- liveness probes ----------------------------------

DISPATCH_LOCK_DIR_NAME = ".dispatch.lock"
LOCK_INFO_NAME = "info.json"
#: Mirrors ``evidence/literature/.run.lock`` from the literature service
#: (kept as a literal so this light module never imports the agents stack).
LITERATURE_RUN_LOCK_DIR = Path("evidence/literature/.run.lock")


def read_lock_info(lock_dir: Path) -> dict[str, Any]:
    """Best-effort read of a lock dir's ``info.json``; empty dict when unreadable."""
    try:
        parsed = json.loads((lock_dir / LOCK_INFO_NAME).read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _process_start_time(pid: int) -> int | None:
    """The process start time (clock ticks since boot) from ``/proc/<pid>/stat``.

    Field 22 of the stat line. Returns None where ``/proc`` is unavailable
    (non-Linux) or the line cannot be parsed — callers degrade gracefully to
    pid-only liveness in that case.
    """
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # comm (field 2) may itself contain spaces/parens; split after its closing paren.
    _, sep, rest = stat_line.rpartition(")")
    if not sep:
        return None
    fields = rest.split()
    if len(fields) < 20:  # starttime is field 22, i.e. fields[19] after comm
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def publish_lock_info(lock_dir: Path, *, extra: dict[str, Any] | None = None) -> None:
    """Write and fsync a lock dir's ``info.json``; call IMMEDIATELY after ``mkdir``.

    Shared publication helper for every ``mkdir``-style lock (the dispatcher
    lock here, the literature run lock in
    :mod:`carmel.services.literature`). Records ``pid``, ``hostname``,
    ``started_at`` and ``pid_start`` (the pid's /proc start time, so pid reuse
    after a reboot cannot make a dead holder look live), then fsyncs the file
    and the lock directory so the metadata survives a crash. Together with the
    grace period in :func:`lock_is_live` this closes the publication race
    where a peer that sees the bare directory before ``info.json`` lands would
    wrongly break a live lock.

    Args:
        lock_dir: The already-created lock directory.
        extra: Optional extra keys to record (e.g. ``action_id``).
    """
    pid = os.getpid()
    info: dict[str, Any] = {
        "pid": pid,
        "hostname": socket.gethostname(),
        "started_at": _now().isoformat(),
        "pid_start": _process_start_time(pid),
    }
    if extra:
        info.update(extra)
    fd = os.open(lock_dir / LOCK_INFO_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, json.dumps(info).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(lock_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _lock_age_s(lock_dir: Path) -> float:
    """Seconds since the lock directory's mtime; +inf when it vanished."""
    try:
        mtime = lock_dir.stat().st_mtime
    except OSError:
        return float("inf")
    return max(0.0, datetime.now(UTC).timestamp() - mtime)


def lock_is_live(
    lock_dir: Path,
    *,
    stale_after_s: float,
    lock_grace_s: float = DEFAULT_LOCK_GRACE_S,
) -> bool:
    """Whether a ``mkdir``-style lock is held by a live attempt.

    Liveness rules, in order:

    - No lock dir: not live.
    - Lock taken on THIS host with a recorded pid: a dead pid is not live; a
      live pid is live — including our OWN pid: two dispatch attempts in the
      same server process (two /run clicks) must not break each other's lock —
      UNLESS the recorded ``pid_start`` differs from the pid's current /proc
      start time, which means the pid was REUSED (e.g. after a reboot) and the
      original holder is dead. Where start times are unavailable the live pid
      wins (degrade gracefully).
    - Otherwise (foreign host): live while ``started_at`` is younger than
      ``stale_after_s``.
    - ``info.json`` missing or malformed: LIVE within ``lock_grace_s`` of the
      lock dir's mtime — a peer may be mid-publication between ``mkdir`` and
      its ``info.json`` write, so absent metadata fails CLOSED. Only past the
      grace period is unreadable metadata treated as stale.
    """
    if not lock_dir.exists():
        return False
    info = read_lock_info(lock_dir)
    pid = info.get("pid")
    hostname = info.get("hostname")
    if hostname == socket.gethostname() and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass  # pid exists (owned by another user): fall through to the start-time check
        recorded_start = info.get("pid_start")
        if isinstance(recorded_start, int):
            current_start = _process_start_time(pid)
            if current_start is not None and current_start != recorded_start:
                return False  # pid reused: the original lock holder is dead
        return True
    started_at_raw = info.get("started_at")
    if isinstance(started_at_raw, str):
        try:
            started_at = datetime.fromisoformat(started_at_raw)
        except ValueError:
            started_at = None
        if started_at is not None:
            return (_now() - started_at).total_seconds() <= stale_after_s
    # Missing or malformed metadata: fail CLOSED within the grace period — a
    # peer may be between mkdir() and publishing info.json (defect 2).
    return _lock_age_s(lock_dir) <= lock_grace_s


def _attempt_is_live(
    workspace_root: Path,
    action: ActionState,
    *,
    stale_after_s: float,
    in_dispatch_lock: bool,
) -> bool:
    """Whether a RUNNING action still has a live attempt behind it.

    An attempt only ever runs (a) under the workspace dispatch lock or
    (b) — for literature — under the literature run lock. So the attempt is
    live when either lock is live; when the caller already HOLDS the
    exclusive dispatch lock, only the literature lock can vouch for a live
    attempt. Without that guarantee we additionally treat a lease younger
    than ``stale_after_s`` as live: never a silent second run.

    T3 is a special case (spar review finding 17): the T3/RMG process tree
    is deliberately supervised OUTSIDE the dispatch lock and OUTLIVES the
    Carmel process that launched it (that is the entire premise of
    :mod:`carmel.services.recovery` — see its module docstring). The
    dispatch lock says nothing about whether that process tree is still
    running, so for a ``T3_RUN`` action we consult real T3 liveness via
    :func:`~carmel.services.recovery.probe_run_liveness` regardless of
    ``in_dispatch_lock``. Anything other than a clean "finished" verdict
    (``SUPERVISED``, ``ORPHANED``, or the fail-closed ``UNKNOWN``) is
    treated as live, so ``reconcile`` never marks FAILED while pgid N is
    still writing into the workspace — the exact lie the process-tree kill
    exists to prevent.
    """
    if action.kind == ActionKind.T3_RUN and not probe_run_liveness(workspace_root).is_finished:
        return True
    # EVERY literature kind, not just the search (spar round 7, P1). A corpus pass takes
    # the same literature run lock, so excluding it meant a live corpus pass could be
    # judged dead and marked FAILED while it was still writing -- and then re-run. That
    # is the silent second run this whole function exists to prevent.
    if action.kind in LITERATURE_ACTION_KINDS and lock_is_live(
        workspace_root / LITERATURE_RUN_LOCK_DIR, stale_after_s=stale_after_s
    ):
        return True
    if not in_dispatch_lock:
        if lock_is_live(workspace_root / DISPATCH_LOCK_DIR_NAME, stale_after_s=stale_after_s):
            return True
        age_s = (_now() - action.updated_at).total_seconds()
        if age_s <= stale_after_s:
            return True
    return False


def reconcile(
    workspace_root: Path,
    *,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    in_dispatch_lock: bool = False,
) -> PlanProgress:
    """Crash recovery (spar round 3, P0-4 / P0-8). Idempotent.

    ``execute_next_action`` calls this FIRST, before looking at the cursor.
    Every step of the dispatcher is individually crash-safe only because
    this repair pass exists:

    - An action left ``RUNNING`` whose attempt lease is stale (no live
      lock, older than ``stale_after_s``) first has its latest attempt
      checked for a persisted terminal result, which is ADOPTED when found
      (see :func:`_adopt_attempt_result`; defect 3). Only when no result was
      persisted is it marked FAILED with outcome FAILED_BLOCKING /
      FAILED_NONBLOCKING per its ``blocking`` flag and noted as "recovered
      from interrupted attempt". A live attempt raises
      :class:`ActionInFlightError` — never a silent second run.
    - A campaign state left at ``RUNNING_*`` by a crash between
      ``mark_finished`` and the dispatcher's post-transition has the
      missing per-action post-transition REPLAYED from the finished
      action's kind and outcome (see
      :func:`_replay_missing_post_transition`; defect 1), so the terminal
      projection then runs from a legal intermediate state instead of
      wedging on an illegal single hop.
    - A cursor still pointing at an action whose execution_status is
      terminal (SUCCEEDED / FAILED / SKIPPED) is advanced. This is what
      stops a completed action being re-run after a crash between
      ``mark_finished`` and ``advance_cursor``.
    - The persisted campaign state is repaired to match the projection when
      the two disagree (crash between the post-transition and the terminal
      projection), provided the repair is a legal transition; when it is
      not, the mismatch is recorded as a decision-log warning rather than
      forced.

    Args:
        workspace_root: The campaign workspace root.
        stale_after_s: Attempt-lease staleness horizon.
        in_dispatch_lock: True when the caller already holds the exclusive
            workspace dispatch lock (so no other dispatcher attempt can be
            live and the lease-age fallback is unnecessary).

    Additionally, reconcile is RETRY-AWARE: when the persisted campaign
    state is ``APPROVED_FOR_EXECUTION`` while the progress still records a
    blocking failure, the operator has taken main's guarded
    ``FAILED -> APPROVED_FOR_EXECUTION`` retry edge (only legal for a
    campaign that failed from ``RUNNING_T3``). The failed action is reset
    to PENDING (its attempt history is kept), the cursor rewound to it,
    and the dispatcher may run it once more. Without this reset the FAILED
    projection would immediately re-fail the freshly retried campaign.

    The progress mutation itself is a single read-modify-write under the
    workspace lock; decision-log events and state repairs are emitted
    AFTER the lock is released (``append_event``/``update_state`` acquire
    the same non-reentrant lock internally).

    Returns:
        The (possibly repaired) progress.

    Raises:
        ActionInFlightError: If a RUNNING action's attempt is still live.
    """
    log_path = workspace_root / "decision_log.jsonl"
    events: list[dict[str, Any]] = []

    with workspace_lock(workspace_root):
        progress = load_progress(workspace_root)
        changed = False
        # Only a blocking failure that PRE-dates this reconcile pass can be
        # an operator-authorized retry; one manufactured by the recovery
        # loop below must stand as FAILED.
        preexisting_blocking_failures = {
            a.action_id for a in progress.actions if a.outcome == ActionOutcome.FAILED_BLOCKING
        }

        for i, action in enumerate(progress.actions):
            if action.execution_status != ActionExecutionStatus.RUNNING:
                continue
            if _attempt_is_live(workspace_root, action, stale_after_s=stale_after_s, in_dispatch_lock=in_dispatch_lock):
                raise ActionInFlightError(
                    f"action {action.action_id!r} has a live in-flight attempt; refusing a second run"
                )
            adopted = _adopt_attempt_result(workspace_root, action)
            if adopted is not None:
                progress.actions[i] = adopted
                changed = True
                events.append(
                    {
                        "event": "dispatch.attempt_adopted",
                        "action_id": action.action_id,
                        "payload": {
                            "run_id": adopted.run_id,
                            "status": adopted.execution_status.value,
                            "outcome": adopted.outcome.value,
                        },
                    }
                )
                continue
            outcome = ActionOutcome.FAILED_BLOCKING if action.blocking else ActionOutcome.FAILED_NONBLOCKING
            progress.actions[i] = action.model_copy(
                update={
                    "execution_status": ActionExecutionStatus.FAILED,
                    "outcome": outcome,
                    "updated_at": _now(),
                    "notes": "recovered from interrupted attempt",
                }
            )
            changed = True
            events.append(
                {
                    "event": "dispatch.attempt_recovered",
                    "action_id": action.action_id,
                    "payload": {"level": "warning", "outcome": outcome.value},
                }
            )

        # Retry-awareness (see docstring): reset the blocking failure the
        # operator explicitly chose to retry.
        if load_state(workspace_root).state == CampaignStateValue.APPROVED_FOR_EXECUTION:
            for i, action in enumerate(progress.actions):
                if (
                    action.outcome != ActionOutcome.FAILED_BLOCKING
                    or action.action_id not in preexisting_blocking_failures
                ):
                    continue
                progress.actions[i] = action.model_copy(
                    update={
                        "execution_status": ActionExecutionStatus.PENDING,
                        "outcome": ActionOutcome.NONE,
                        "updated_at": _now(),
                        "notes": "reset for retry after FAILED -> APPROVED_FOR_EXECUTION",
                    }
                )
                progress.cursor = min(progress.cursor, i)
                changed = True
                events.append(
                    {
                        "event": "dispatch.action_reset_for_retry",
                        "action_id": action.action_id,
                        "payload": {"attempts_so_far": len(action.attempt_ids)},
                    }
                )

        while progress.cursor < len(progress.actions) and progress.actions[progress.cursor].is_terminal():
            progress.cursor += 1
            changed = True

        if changed:
            progress.updated_at = _now()
            save_progress(workspace_root, progress)

    for event in events:
        append_typed_event(
            log_path,
            event=event["event"],
            action_id=event.get("action_id"),
            payload=event["payload"],
        )

    _replay_missing_post_transition(workspace_root, progress)
    repair_campaign_state(workspace_root, progress)
    return progress


#: The action kinds each RUNNING_* campaign state covers. RUNNING_LITERATURE covers ALL
#: literature kinds, not just the search (spar round 7, P1): a crashed corpus pass would
#: otherwise leave the campaign wedged in RUNNING_LITERATURE forever, because the replay
#: below looked only for a finished LITERATURE_SEARCH and never found one.
_KINDS_FOR_RUNNING_STATE: dict[CampaignStateValue, frozenset[ActionKind]] = {
    CampaignStateValue.RUNNING_T3: frozenset({ActionKind.T3_RUN}),
    CampaignStateValue.RUNNING_LITERATURE: LITERATURE_ACTION_KINDS,
}


def _post_transition_target(action: ActionState) -> CampaignStateValue:
    """The per-action post-transition implied by a finished action's outcome.

    Mirrors the dispatcher's ``_apply_post_transition``: a blocking failure
    implies FAILED; a finished T3 implies DIAGNOSTICS_READY; any non-blocking
    literature outcome implies LITERATURE_READY.
    """
    if action.outcome == ActionOutcome.FAILED_BLOCKING:
        return CampaignStateValue.FAILED
    if action.kind == ActionKind.T3_RUN:
        return CampaignStateValue.DIAGNOSTICS_READY
    return CampaignStateValue.LITERATURE_READY


def _replay_missing_post_transition(workspace_root: Path, progress: PlanProgress) -> None:
    """Replay a per-action post-transition lost to a crash (defect 1).

    A crash between ``mark_finished`` and the dispatcher's post-transition
    leaves the campaign state at ``RUNNING_*`` while the action is already
    terminal. Without this replay the terminal projection (e.g.
    ``RUNNING_T3 -> COMPLETED_PHASE1``) is not a legal single transition and
    the campaign wedges forever; the next pre-transition (e.g.
    ``RUNNING_LITERATURE -> RUNNING_T3``) is illegal too. So: when the
    persisted state is ``RUNNING_*`` and no action is RUNNING any more, apply
    the post-transition derived from the finished action's kind and outcome,
    restoring a legal intermediate state.
    """
    current = load_state(workspace_root).state
    kinds = _KINDS_FOR_RUNNING_STATE.get(current)
    if kinds is None:
        return
    finished = [
        a
        for a in progress.actions
        if a.kind in kinds and a.is_terminal() and a.execution_status != ActionExecutionStatus.SKIPPED
    ]
    if not finished:
        return
    target = _post_transition_target(finished[-1])
    if target == current or not can_transition(current, target):
        return
    # The ``can_transition`` check above is only advisory: this function holds no lock
    # across it and ``update_state``, so a concurrent repairer/dispatch can legally move
    # the persisted state between our read of ``current`` and this write. Mirrors
    # ``repair_campaign_state``'s identical guard around its own ``update_state`` call.
    try:
        update_state(
            workspace_root,
            target,
            notes=f"reconcile: replayed missing post-transition for action {finished[-1].action_id}",
        )
    except InvalidTransitionError:
        # A concurrent repairer (or dispatch) moved the state between our
        # read and this write; the winner's transition stands.
        _log.warning("post-transition replay lost a race for %s; leaving the winner's state", workspace_root)


def _transition_path(
    current: CampaignStateValue,
    target: CampaignStateValue,
    failed_from: CampaignStateValue | None = None,
) -> list[CampaignStateValue] | None:
    """Shortest legal multi-step path from ``current`` to ``target``; None if unreachable.

    ``failed_from``-aware: the ``FAILED -> APPROVED_FOR_EXECUTION`` retry
    edge is guarded (main allows it only when the campaign failed from
    ``RUNNING_T3``), so each expansion consults
    :func:`~carmel.services.state_machine.can_transition` with the
    ``failed_from`` the walk would actually observe — the persisted one for
    the start state, and the predecessor state for a ``FAILED`` node
    reached mid-path (``update_state`` records exactly that). Without
    this, the BFS could emit a path that ``update_state`` then rejects
    mid-repair.
    """
    if current == target:
        return []
    # Node = (state, failed_from as update_state would have persisted it).
    start = (current, failed_from if current == CampaignStateValue.FAILED else None)
    seen = {start}
    queue: deque[tuple[tuple[CampaignStateValue, CampaignStateValue | None], list[CampaignStateValue]]] = deque(
        [(start, [])]
    )
    while queue:
        (state, state_failed_from), path = queue.popleft()
        for nxt in sorted(VALID_TRANSITIONS.get(state, frozenset()), key=lambda s: s.value):
            if not can_transition(state, nxt, state_failed_from):
                continue
            nxt_node = (nxt, state if nxt == CampaignStateValue.FAILED else None)
            if nxt_node in seen:
                continue
            step_path = [*path, nxt]
            if nxt == target:
                return step_path
            seen.add(nxt_node)
            queue.append((nxt_node, step_path))
    return None


def repair_campaign_state(workspace_root: Path, progress: PlanProgress) -> None:
    """Repair the persisted campaign state to match the terminal projection.

    Applied only when the projection is terminal and disagrees with the
    persisted state. The repair WALKS the state graph (defect 1): when the
    projected terminal state is only reachable through a legal multi-step
    path (e.g. ``RUNNING_T3 -> DIAGNOSTICS_READY -> COMPLETED_PHASE1``), each
    step is applied in order. Only when no legal path exists at all is the
    mismatch recorded as a decision-log warning
    (``dispatch.state_mismatch``) instead of being forced.
    """
    projection = aggregate_state(progress)
    if projection is None:
        return
    persisted = load_state(workspace_root)
    current = persisted.state
    if current == projection:
        return
    path = _transition_path(current, projection, persisted.failed_from)
    if path is not None:
        try:
            for step in path:
                update_state(workspace_root, step, notes="reconcile: repaired to match plan progress")
        except InvalidTransitionError:
            # A concurrent repairer (or dispatch) moved the state between our
            # read and this write; the winner's repair stands.
            _log.warning("state repair lost a race for %s; leaving the winner's state", workspace_root)
        return
    _log.warning(
        "campaign state %s disagrees with projection %s and cannot be legally repaired",
        current.value,
        projection.value,
    )
    append_typed_event(
        workspace_root / "decision_log.jsonl",
        event="dispatch.state_mismatch",
        payload={
            "level": "warning",
            "persisted_state": current.value,
            "projected_state": projection.value,
        },
    )
