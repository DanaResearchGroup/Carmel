# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Shared dispatcher for ordered multi-action plans.

The dispatcher owns per-action progress and campaign-state sequencing:
it validates the plan, repairs interrupted attempts, refuses unapproved
actions, applies the pre-transition (``RUNNING_T3`` /
``RUNNING_LITERATURE``), and then STARTS the action in the background and
returns — mirroring :func:`carmel.services.execution.start_t3_action`'s
model, where a run that takes minutes to hours must never hold a web
request open. There is ONE T3 execution path: the T3 handler drives
:func:`carmel.services.execution._finish_t3_run`, the same finish path
main's ``execute_t3_action``/``start_t3_action`` use — never a parallel
copy of it.

Division of state ownership:

- The T3 finish path (``_finish_t3_run``) owns the T3 post-transitions —
  ``DIAGNOSTICS_READY`` → ``COMPLETED_PHASE1`` on success, ``FAILED`` on
  failure. This is safe under the dispatcher because
  :func:`validate_plan_shape` guarantees at most one T3 action and that it
  is the LAST action of the plan, so "T3 finished successfully" and "the
  whole plan is complete" coincide (spar round 3, P0-7 resolved by
  construction rather than by a parallel non-transitioning core).
- The dispatcher owns the literature post-transition
  (``LITERATURE_READY`` / ``FAILED``) and the terminal projection repair
  via :func:`carmel.services.plan_progress.repair_campaign_state`.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from carmel.config import AgentConfig
from carmel.logger import get_logger
from carmel.schemas.action_state import ActionExecutionStatus, ActionOutcome, PlanProgress
from carmel.schemas.approval import LITERATURE_ACTION_KINDS, ActionKind, ApprovalStatus
from carmel.schemas.campaign import Campaign
from carmel.schemas.diagnostics import DiagnosticsV1
from carmel.schemas.literature import STOP_REASON_FOR_DIMENSION, LiteratureReport, StopReason
from carmel.schemas.plan import Plan, PlannedAction
from carmel.schemas.run import FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.schemas.state import CampaignStateValue

# The execution module is referenced as a module (not via from-imports of its
# functions) so tests that monkeypatch e.g. execution._default_adapter act on
# the same binding the dispatcher calls.
from carmel.services import execution
from carmel.services.approvals import has_effective_human_rejection
from carmel.services.authorization import BudgetExceededError, envelope_for
from carmel.services.decision_log import append_typed_event
from carmel.services.execution import T3AdapterProtocol, save_run_record
from carmel.services.plan_progress import (
    DEFAULT_STALE_AFTER_S,
    DISPATCH_LOCK_DIR_NAME,
    ActionInFlightError,
    advance_cursor,
    aggregate_state,
    load_or_init_progress,
    load_progress,
    lock_is_live,
    mark_finished,
    mark_running,
    publish_lock_info,
    read_lock_info,
    reconcile,
    record_attempt_result,
    repair_campaign_state,
)
from carmel.services.planner import load_plan
from carmel.services.recovery import LockStateUnknownError, RunSupervision, start_supervision
from carmel.services.state_machine import load_state, update_state

if TYPE_CHECKING:
    from carmel.services.literature import LiteratureDeps

_log = get_logger("services.dispatcher")

#: Action kinds THIS dispatcher can execute. EXPERIMENT is deliberately absent —
#: the planner and schema can still construct it, so the dispatcher refuses it.
SUPPORTED_ACTION_KINDS = frozenset({ActionKind.T3_RUN, ActionKind.LITERATURE_SEARCH, ActionKind.LITERATURE_CORPUS_PASS})

#: The kinds the literature handler accepts. Both run the Literature Agent through the
#: same envelope; they differ only in whether the pass may reach the network.
#: Re-exported under this module's original private name; defined in the schemas
#: package so crash recovery and the dispatcher cannot drift apart on it.
_LITERATURE_ACTION_KINDS = LITERATURE_ACTION_KINDS

#: Action kinds that are executable SOMEWHERE, which is not the same question.
#:
#: ARC_RUN runs through `carmel.services.execution.start_arc_action` on the
#: single-action path it was built with, never through this dispatcher's handler
#: registry. `validate_plan_shape` must therefore judge a plan against this set, not
#: against SUPPORTED_ACTION_KINDS: conflating "no handler HERE" with "not executable"
#: made `save_plan` reject every ARC plan outright, which is how this was found.
EXECUTABLE_ACTION_KINDS = SUPPORTED_ACTION_KINDS | {ActionKind.ARC_RUN}

#: Stop reasons meaning "a budget ceiling ended the literature run".
_BUDGET_STOP_REASONS = frozenset(STOP_REASON_FOR_DIMENSION.values())

__all__ = [
    "EXECUTABLE_ACTION_KINDS",
    "SUPPORTED_ACTION_KINDS",
    "ActionHandler",
    "ActionResult",
    "DispatchTicket",
    "UnsupportedActionKindError",
    "default_handlers",
    "execute_action",
    "execute_next_action",
    "make_literature_handler",
    "make_t3_handler",
    "recover_workspace",
    "validate_plan_shape",
]


class UnsupportedActionKindError(RuntimeError):
    """No handler is registered for this action kind."""


class ActionResult(BaseModel):
    """Typed result of executing one planned action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    kind: ActionKind
    run_record: RunRecord  # persisted for literature too
    outcome: ActionOutcome
    artifacts: list[Path] = Field(default_factory=list)
    diagnostics: DiagnosticsV1 | None = None
    literature_report: LiteratureReport | None = None


class DispatchTicket:
    """Handle for a dispatched action running on a background thread.

    Mirrors the thread returned by
    :func:`carmel.services.execution.start_t3_action`: the dispatcher's
    critical section is short — it starts the run and returns — and callers
    that need the outcome (tests, the CLI, the literature-at-creation hook)
    call :meth:`wait` instead of polling the workspace. The UI redirects
    immediately and lets the auto-refreshing dashboard observe progress.
    """

    def __init__(self, action_id: str, kind: ActionKind, attempt_id: str) -> None:
        self.action_id = action_id
        self.kind = kind
        self.attempt_id = attempt_id
        self.result: ActionResult | None = None
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background run to finish (or ``timeout`` seconds)."""
        if self.thread is not None:
            self.thread.join(timeout)

    def wait(self, timeout: float | None = None) -> ActionResult | None:
        """Join the background run and return its :class:`ActionResult`.

        Returns None when the run failed before producing a result (the
        failure is persisted in the workspace; ``self.error`` carries the
        exception).

        Raises:
            TimeoutError: If ``timeout`` is given and the background run had
                not finished when it elapsed. Without this, a timed-out wait
                and a genuine failure were indistinguishable — both returned
                None — which would have left a caller passing a timeout no
                way to tell "the run failed" (safe to inspect ``self.error``
                and the persisted workspace state) from "the run is still
                going" (unsafe to do either yet). The one production caller
                (:func:`carmel.services.campaigns.start_literature_at_creation`)
                calls ``wait()`` with no timeout, so it is unaffected.
        """
        self.join(timeout)
        if timeout is not None and self.thread is not None and self.thread.is_alive():
            raise TimeoutError(f"dispatch of action {self.action_id!r} did not finish within {timeout}s")
        return self.result


class ActionHandler(Protocol):
    """Structural type for a single-action executor.

    ``supervision`` is passed only to handlers that advertise
    ``wants_supervision`` (today: the real T3 handler), so that plain
    three-argument handlers — every literature handler and every test
    double — stay conforming. A handler that receives one owns it and
    must close it exactly once; :func:`~carmel.services.execution._finish_t3_run`
    does that for the T3 path.
    """

    def __call__(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        *,
        supervision: RunSupervision | None = None,
    ) -> ActionResult: ...


def _failure_outcome(action: PlannedAction) -> ActionOutcome:
    return ActionOutcome.FAILED_BLOCKING if action.blocking else ActionOutcome.FAILED_NONBLOCKING


def _stamp_attempt_result(workspace_root: Path, action_id: str, run_id: str, outcome: ActionOutcome) -> None:
    """Record the attempt-to-run mapping for the action's live attempt.

    Called by every handler IMMEDIATELY after it persists its RunRecord, so a
    crash before ``mark_finished`` leaves a marker that
    :func:`~carmel.services.plan_progress.reconcile` can ADOPT instead of
    discarding the completed work as FAILED (defect 3). A handler invoked
    outside the dispatcher (no RUNNING attempt in the progress) records
    nothing.
    """
    try:
        progress = load_progress(workspace_root)
    except OSError, ValueError:
        return
    for action_state in progress.actions:
        if (
            action_state.action_id == action_id
            and action_state.execution_status == ActionExecutionStatus.RUNNING
            and action_state.attempt_ids
        ):
            status = (
                ActionExecutionStatus.SUCCEEDED
                if outcome in (ActionOutcome.SUCCEEDED, ActionOutcome.NO_GROUNDED_FINDINGS)
                else ActionExecutionStatus.FAILED
            )
            record_attempt_result(
                workspace_root,
                action_id=action_id,
                attempt_id=action_state.attempt_ids[-1],
                run_id=run_id,
                status=status,
                outcome=outcome,
            )
            return


def make_t3_handler(adapter: T3AdapterProtocol | None = None) -> ActionHandler:
    """Build the T3 handler: a thin wrapper over main's T3 finish path.

    The handler assumes the campaign has ALREADY entered ``RUNNING_T3``
    (the dispatcher applies that pre-transition synchronously, exactly like
    :func:`carmel.services.execution.begin_t3_run`) and drives
    :func:`carmel.services.execution._finish_t3_run` — the single T3
    execution path shared with ``execute_t3_action``/``start_t3_action``.
    ``_finish_t3_run`` owns the T3 state transitions (DIAGNOSTICS_READY →
    COMPLETED_PHASE1 on success, FAILED on failure); see the module
    docstring for why that is safe under the dispatcher.

    The run lock is normally taken by :func:`execute_next_action` *before*
    the campaign can read as ``RUNNING_T3`` and handed here to be closed by
    ``_finish_t3_run`` — the ordering ``start_t3_action`` documents. A
    direct caller that passes no ``supervision`` gets one taken here
    instead, so calling the handler on its own still holds the lock for the
    duration of the run.
    """

    def t3_handler(
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        *,
        supervision: RunSupervision | None = None,
    ) -> ActionResult:
        if action.kind != ActionKind.T3_RUN:  # fail closed on mis-routing
            # Nothing downstream will run, so this handler is the last owner
            # able to release a lock the dispatcher already took for it.
            if supervision is not None:
                supervision.close()
            raise UnsupportedActionKindError(f"T3 handler cannot execute kind {action.kind.value!r}")
        if supervision is None:
            # P1-13, second site: pass estimated_cpu_hours here too. `start_supervision`
            # defaults it to 0.0, and `spend.compute_spend` derives the in-flight CPU
            # reservation from exactly this value -- so a run supervised without it is
            # invisible to the budget gate for its whole duration, and every launch
            # decision taken meanwhile is made against a ledger that under-reports.
            supervision = start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
        run_record, diagnostics = execution._finish_t3_run(
            workspace_root,
            campaign,
            action,
            adapter if adapter is not None else execution._default_adapter(),
            datetime.now(UTC),
            supervision,
        )
        if run_record.status == RunStatus.SUCCEEDED and diagnostics is not None:
            outcome = ActionOutcome.SUCCEEDED
        else:
            outcome = _failure_outcome(action)
        _stamp_attempt_result(workspace_root, action.action_id, run_record.run_id, outcome)
        return ActionResult(
            action_id=action.action_id,
            kind=action.kind,
            run_record=run_record,
            outcome=outcome,
            diagnostics=diagnostics,
        )

    # Opt in to being handed the dispatcher's pre-taken run lock. Test
    # doubles and the literature handler deliberately do not, so they keep
    # the plain three-argument signature.
    t3_handler.wants_supervision = True  # type: ignore[attr-defined]
    return t3_handler


def _literature_unavailable_result(workspace_root: Path, action: PlannedAction, message: str) -> ActionResult:
    """Typed non-crash result for a literature action that cannot run."""
    now = datetime.now(UTC)
    run_record = RunRecord(
        run_id=uuid.uuid4().hex,
        action_id=action.action_id,
        tool_name="literature_agent",
        status=RunStatus.FAILED,
        failure_code=FailureCode.AGENT_ERROR,
        started_at=now,
        ended_at=now,
        submission_mode=SubmissionMode.LOCAL,
        error_message=message,
    )
    save_run_record(workspace_root, run_record)
    outcome = _failure_outcome(action)
    _stamp_attempt_result(workspace_root, action.action_id, run_record.run_id, outcome)
    return ActionResult(
        action_id=action.action_id,
        kind=action.kind,
        run_record=run_record,
        outcome=outcome,
    )


def _literature_outcome(report: LiteratureReport, action: PlannedAction) -> ActionOutcome:
    """Classify the outcome of THIS pass, not of the accumulated report.

    ``report.findings`` accumulates across every pass the campaign has run (schema v2),
    so testing it would let one finding from an earlier search make every later corpus
    pass report SUCCEEDED -- including a pass that ran the whole corpus and grounded
    nothing (spar round 7, P1). That is precisely the signal an operator needs, because
    a barren pass is the one that says the corpus is exhausted or the prompt is wrong.
    """
    if report.stop_reason in _BUDGET_STOP_REASONS:
        return ActionOutcome.BUDGET_EXCEEDED
    if report.stop_reason == StopReason.ERROR:
        return _failure_outcome(action)
    if not report.findings_for(report.run_id):
        return ActionOutcome.NO_GROUNDED_FINDINGS
    return ActionOutcome.SUCCEEDED


def _apply_action_budget(deps: LiteratureDeps, action: PlannedAction) -> LiteratureDeps:
    """Bind a corpus pass to the budget the operator named when appending it.

    Without this the operator's ``--budget-usd`` would be recorded on the action and
    read only by the approval gate, while the run itself spent up to whatever the
    config file's ``max_cost_usd`` happened to be -- a control that looks like a
    ceiling in the plan and is not one at run time. A safety number that does not
    bind is worse than none, because it is believed.

    Only ``max_cost_usd`` is replaced. The session and daily ceilings are process-
    and machine-wide protections against many runs in aggregate, and one operator
    authorising one action does not authorise breaching those -- so the rebuilt
    ledger CARRIES THEM OVER (spar round 7, P1). Constructing ``BudgetLedger(budget)``
    bare leaves ``daily_ledger_path=None``, which silently switches the file-backed
    daily cap off for exactly the runs an operator has just authorised extra money
    for. That is the opposite of what the paragraph above promises, and the promise is
    the dangerous half: a ceiling believed to hold is worse than one known to be
    absent.

    Raises:
        ValueError: If the action names no usable budget. Only a corpus pass reaches
            here, and for a corpus pass the operator's ``--budget-usd`` IS the
            authorisation -- appending one without a positive budget is refused at
            :func:`~carmel.services.planner.append_corpus_pass_action`. Falling back to
            the config ceiling for an action that reached this point some other way
            (a hand-edited or tampered plan) would spend up to a limit nobody
            authorised, so fail closed instead.
    """
    budget_usd = action.estimated_spend_usd
    if budget_usd is None or budget_usd <= 0:
        raise ValueError(
            f"corpus-pass action {action.action_id} names no positive budget "
            f"(estimated_spend_usd={budget_usd!r}); the operator budget is the "
            f"authorisation for this action and there is no safe default"
        )

    import dataclasses

    from carmel.agents.budget import BudgetLedger

    budget = deps.config.budget.model_copy(update={"max_cost_usd": budget_usd})
    config = deps.config.model_copy(update={"budget": budget})
    ledger = BudgetLedger(
        budget,
        session=deps.ledger.session,
        daily_ledger_path=deps.ledger.daily_ledger_path,
    )
    return dataclasses.replace(deps, config=config, ledger=ledger)


def make_literature_handler(
    agent_config: AgentConfig | None = None,
    literature_deps: LiteratureDeps | None = None,
) -> ActionHandler:
    """Build the literature handler.

    Without an :class:`AgentConfig` (and no injected deps) the handler
    refuses to run — it returns a typed failed result instead of spending
    money, and the dispatcher maps that non-blocking failure to
    ``LITERATURE_READY`` so T3 is never stopped by it.
    """

    def literature_handler(
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        *,
        supervision: RunSupervision | None = None,
    ) -> ActionResult:
        # Accepted only to satisfy ActionHandler; a literature run holds no
        # run lock. This handler does not advertise ``wants_supervision``, so
        # execute_action never hands it one (and closes any it was given).
        del supervision
        if action.kind not in _LITERATURE_ACTION_KINDS:  # fail closed on mis-routing
            raise UnsupportedActionKindError(f"literature handler cannot execute kind {action.kind.value!r}")
        if literature_deps is None and agent_config is None:
            return _literature_unavailable_result(
                workspace_root, action, "no agent config available; literature run skipped"
            )
        # Lazy import: keeps the dispatcher importable without the agents stack.
        from carmel.services.literature import build_deps, run_corpus_pass, run_literature_research, run_record_for

        deps = literature_deps if literature_deps is not None else build_deps(agent_config)  # type: ignore[arg-type]
        if action.kind == ActionKind.LITERATURE_CORPUS_PASS:
            try:
                deps = _apply_action_budget(deps, action)
            except ValueError as exc:
                # A refused budget is an unrunnable action, not a crashed dispatch:
                # surface it the same way a missing agent config is surfaced, so the
                # operator sees why nothing ran and the plan keeps the action.
                return _literature_unavailable_result(workspace_root, action, str(exc))
        run_pass = run_corpus_pass if action.kind == ActionKind.LITERATURE_CORPUS_PASS else run_literature_research
        report = run_pass(workspace_root, campaign, action, deps, config=deps.config)
        run_record = run_record_for(report, action)
        save_run_record(workspace_root, run_record)
        outcome = _literature_outcome(report, action)
        _stamp_attempt_result(workspace_root, action.action_id, run_record.run_id, outcome)
        return ActionResult(
            action_id=action.action_id,
            kind=action.kind,
            run_record=run_record,
            outcome=outcome,
            literature_report=report,
        )

    return literature_handler


def default_handlers(
    *,
    agent_config: AgentConfig | None = None,
    literature_deps: LiteratureDeps | None = None,
    t3_adapter: T3AdapterProtocol | None = None,
) -> dict[ActionKind, ActionHandler]:
    """The production handler registry: ``{T3_RUN, LITERATURE_SEARCH}``.

    ``ARC_RUN`` and ``EXPERIMENT`` are deliberately absent — dispatching
    them raises :class:`UnsupportedActionKindError`.
    """
    return {
        ActionKind.T3_RUN: make_t3_handler(t3_adapter),
        ActionKind.LITERATURE_SEARCH: make_literature_handler(agent_config, literature_deps),
        ActionKind.LITERATURE_CORPUS_PASS: make_literature_handler(agent_config, literature_deps),
    }


def validate_plan_shape(plan: Plan) -> list[str]:
    """Return human-readable problems; an empty list means OK.

    Enforced by ``save_plan`` (reject) and again by the dispatcher (fail
    closed) — spar round 3, P1-9 / P2-28:

    - every action kind must be executable by SOME path (no EXPERIMENT in an
      executable plan). ARC_RUN passes: it runs on the single-action path in
      ``carmel.services.execution``, not through this dispatcher, so judging it
      by the dispatcher's own handler registry rejected every valid ARC plan.
    - at most one T3_RUN action
    - at most one LITERATURE_SEARCH action
    - any LITERATURE_SEARCH action must precede the T3_RUN action
    - the T3_RUN action, when present, must be LAST among the plan's
      executable actions — no action of any kind (including ARC_RUN) may
      follow it. Checking literature-before-T3 alone does not guarantee
      this: a plan could still place ARC_RUN (also in
      ``EXECUTABLE_ACTION_KINDS``) after T3_RUN without tripping either of
      the two checks above, so T3-last is enforced directly here.

    T3-last is what lets the dispatcher reuse ``_finish_t3_run`` (whose
    success path ends at ``COMPLETED_PHASE1``) verbatim; see the module
    docstring.

    - action_ids must be unique
    """
    problems: list[str] = []
    for action in plan.actions:
        if action.kind not in EXECUTABLE_ACTION_KINDS:
            problems.append(f"action {action.action_id!r} has kind {action.kind.value!r}, which nothing can execute")
    t3_indices = [i for i, a in enumerate(plan.actions) if a.kind == ActionKind.T3_RUN]
    if len(t3_indices) > 1:
        problems.append(f"plan contains {len(t3_indices)} T3_RUN actions; at most one is allowed")
    literature_indices = [i for i, a in enumerate(plan.actions) if a.kind == ActionKind.LITERATURE_SEARCH]
    if len(literature_indices) > 1:
        problems.append(f"plan contains {len(literature_indices)} LITERATURE_SEARCH actions; at most one is allowed")
    # LITERATURE_CORPUS_PASS is deliberately NOT capped. A search pass reaches the
    # network, so running two is duplicated spend on the same discovery; a corpus pass
    # re-reads what is already held, and an operator may legitimately append one after
    # each batch of papers they supply.
    if t3_indices:
        first_t3 = t3_indices[0]
        for i, action in enumerate(plan.actions):
            if i <= first_t3:
                continue
            if action.kind in _LITERATURE_ACTION_KINDS:
                problems.append(
                    f"literature action {action.action_id!r} follows the T3_RUN action; literature must precede T3"
                )
            elif action.kind in EXECUTABLE_ACTION_KINDS:
                problems.append(
                    f"action {action.action_id!r} (kind {action.kind.value!r}) follows the T3_RUN action; "
                    "T3_RUN must be the last executable action in the plan"
                )
    action_ids = [a.action_id for a in plan.actions]
    if len(set(action_ids)) != len(action_ids):
        problems.append("action_ids are not unique")
    return problems


def execute_action(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    *,
    handlers: dict[ActionKind, ActionHandler] | None = None,
    supervision: RunSupervision | None = None,
) -> ActionResult:
    """Route one action to its handler by kind.

    Each handler additionally ASSERTS its own kind and raises on a
    mismatch — fail closed; a mis-routed action must never reach the wrong
    executor.

    Args:
        supervision: A run lock already taken for this action, handed to the
            handler if it advertises ``wants_supervision``. Ownership passes
            with it: the handler closes it. Handlers without the marker are
            called with the plain three-argument signature, so an unwanted
            lock is closed here rather than leaked.

    Raises:
        UnsupportedActionKindError: If no handler is registered for the kind.
    """
    if handlers is None:
        handlers = default_handlers()
    handler = handlers.get(action.kind)
    if handler is None:
        if supervision is not None:
            supervision.close()
        raise UnsupportedActionKindError(f"no handler registered for action kind {action.kind.value!r}")
    if getattr(handler, "wants_supervision", False):
        return handler(workspace_root, campaign, action, supervision=supervision)
    if supervision is not None:
        supervision.close()
    return handler(workspace_root, campaign, action)


# --------------------------- dispatcher lock ----------------------------------


class _DispatchLease:
    """Ownership token for a held ``.dispatch.lock`` lease (Finding 7).

    Release used to be an unconditional, ownership-blind
    ``shutil.rmtree(lock_dir, ignore_errors=True)``: any frame holding the
    bare ``Path`` could delete the lock dir regardless of who currently
    published its ``info.json``. That is unsound once a stale lock can be
    STOLEN out from under a dead holder (see :func:`_acquire_dispatch_lock`):
    a frame that thinks it still owns the lease but actually lost a race
    could rmtree a lease a different process has since legitimately
    acquired. Carrying this process's published identity (pid + /proc start
    time) on the token itself — rather than passing a bare ``Path`` around —
    makes the ownership check unavoidable at the type level: nothing outside
    this class ever calls ``rmtree`` on the lock dir directly, and
    :meth:`release` refuses to remove a lock dir whose ``info.json`` no
    longer names this process.
    """

    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = lock_dir
        info = read_lock_info(lock_dir)
        self._pid = info.get("pid")
        self._pid_start = info.get("pid_start")

    def release(self) -> None:
        """Remove the lock dir iff its ``info.json`` still names THIS process.

        Compares pid + recorded ``/proc`` start time (not the full
        ``info.json``, whose ``started_at``/extra keys legitimately change
        across the two :func:`~carmel.services.plan_progress.publish_lock_info`
        calls a single dispatch makes) so a second publish by the SAME
        holder does not spuriously look like a different owner.
        """
        info = read_lock_info(self.lock_dir)
        if info.get("pid") == self._pid and info.get("pid_start") == self._pid_start:
            shutil.rmtree(self.lock_dir, ignore_errors=True)


def _acquire_dispatch_lock(workspace_root: Path, *, stale_after_s: float) -> _DispatchLease:
    """Atomically acquire ``<ws>/.dispatch.lock`` (``mkdir`` primitive), as a lease.

    This is deliberately NOT :func:`carmel.services.state_machine.workspace_lock`:
    the flock there serializes short read-check-write cycles and cannot be
    held across a background run (flock is not re-entrant in-process, and
    every nested ``update_state``/``append_event`` acquires it). The
    dispatch lock instead is a LEASE that outlives the dispatcher's short
    critical section: the background thread holds it for the whole run and
    removes it on completion, and its published ``info.json`` (pid +
    /proc start time) is what lets
    :func:`~carmel.services.plan_progress.reconcile` distinguish a live
    background run from a crashed one.

    Same pid/stale policy as the literature run lock: a stale lock (dead
    pid on this host, or older than ``stale_after_s``) is broken and the
    break recorded as a decision-log warning; a live lock raises
    :class:`ActionInFlightError` (spar round 3, P0-3). A lock whose
    ``info.json`` has not landed yet is LIVE within the publication grace
    period (defect 2) — see
    :func:`~carmel.services.plan_progress.lock_is_live` — and ``info.json``
    is published (written + fsynced, with the pid's start time) IMMEDIATELY
    after ``mkdir`` via
    :func:`~carmel.services.plan_progress.publish_lock_info`.

    Stealing a stale lock is an ATOMIC RENAME, never ``rmtree`` + ``mkdir``
    (Finding 7). The ``mkdir`` below is the only atomic acquire primitive
    this function has; if it were always preceded by this frame's OWN
    unconditional ``rmtree`` of the stale lock, the ``mkdir`` could never
    detect a peer that broke and re-acquired the SAME stale lock in the
    meantime — both racers would believe they hold the lease, and whichever
    finished first would rmtree the OTHER's still-live lock out from under
    it. Renaming the stale lock dir to a process-unique path instead means
    exactly one racer's rename of a given inode can ever succeed (rename is
    atomic); every loser's rename raises ``FileNotFoundError`` and MUST
    treat that as losing the race — it owns nothing and loops back to
    re-evaluate the lock from scratch, rather than assuming it performed the
    steal.
    """
    lock_dir = workspace_root / DISPATCH_LOCK_DIR_NAME
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if lock_is_live(lock_dir, stale_after_s=stale_after_s):
                raise ActionInFlightError(
                    f"another dispatch already holds {lock_dir}; refusing a concurrent run"
                ) from None
            stale_target = lock_dir.with_name(f"{lock_dir.name}.stale.{uuid.uuid4().hex}")
            try:
                lock_dir.rename(stale_target)
            except FileNotFoundError:
                # Lost the steal race: a peer already renamed this exact
                # stale lock away (or has since replaced it with a fresh
                # one) between our liveness check and this rename landing.
                # We broke nothing and own nothing — retry from the top
                # instead of proceeding as if we performed the steal.
                continue
            _log.warning("broke stale dispatch lock %s (moved aside to %s)", lock_dir, stale_target)
            append_typed_event(
                workspace_root / "decision_log.jsonl",
                event="dispatch.lock_broken",
                payload={"level": "warning", "lock_dir": str(lock_dir), "moved_to": str(stale_target)},
            )
            # The rename gave this frame exclusive ownership of the
            # renamed-aside inode (no other racer can have a reference to
            # it), so removing it here is race-free; done so broken stale
            # locks do not accumulate in the workspace.
            shutil.rmtree(stale_target, ignore_errors=True)
            continue
        else:
            publish_lock_info(lock_dir)
            return _DispatchLease(lock_dir)


# --------------------------- the dispatcher -----------------------------------


def _pre_transition_state(kind: ActionKind) -> CampaignStateValue:
    if kind == ActionKind.T3_RUN:
        return CampaignStateValue.RUNNING_T3
    return CampaignStateValue.RUNNING_LITERATURE


def _fail_started_action(
    workspace_root: Path,
    action: PlannedAction,
    *,
    reason: str,
) -> None:
    """Finish an action that was marked RUNNING but never actually ran.

    Shared by a handler crash (:func:`_finish_dispatch`) and a
    background-thread start failure (:func:`execute_next_action`, Finding
    3): both leave a RUNNING attempt with nothing behind it, and both need
    identical repair — mark the action FAILED, apply the failure
    post-transition, advance the cursor past it, and re-run the terminal
    projection — so the workspace is never left reporting a live attempt
    that isn't.
    """
    failure = _failure_outcome(action)
    mark_finished(
        workspace_root,
        action.action_id,
        status=ActionExecutionStatus.FAILED,
        outcome=failure,
        notes=reason,
    )
    if action.kind != ActionKind.T3_RUN:
        _apply_post_transition(workspace_root, action, failure, note=reason)
    elif load_state(workspace_root).state == CampaignStateValue.RUNNING_T3:
        # The real T3 finish path (_finish_t3_run) drives the campaign
        # FAILED itself before re-raising; only a failure that never
        # reached it can leave the state wedged at RUNNING_T3.
        update_state(workspace_root, CampaignStateValue.FAILED, notes=reason)
    progress = advance_cursor(workspace_root, action.action_id)
    repair_campaign_state(workspace_root, progress)


def execute_next_action(
    workspace_root: Path,
    campaign: Campaign,
    *,
    handlers: dict[ActionKind, ActionHandler] | None = None,
) -> DispatchTicket | None:
    """Start the next executable action of the persisted plan in the background.

    The dispatcher's critical section is SHORT — it validates, repairs,
    pre-transitions, marks the attempt RUNNING, then starts the actual tool
    run on a daemon background thread and returns a
    :class:`DispatchTicket` (adopting the background execution model of
    :func:`carmel.services.execution.start_t3_action`; a T3 run takes
    minutes to hours and must never hold the caller hostage). The
    exclusive workspace dispatch lease is held from here through the END of
    the background run; its pid metadata is what
    :func:`~carmel.services.plan_progress.reconcile` uses to tell a live
    run from a crashed one.

    Inside the critical section:

    0. :func:`~carmel.services.plan_progress.reconcile` FIRST — repair any
       interrupted previous attempt before reading the cursor.
    1. Load plan + progress (migrating a Phase-1 plan if needed).
    2. An action whose approval_status is REJECTED is marked SKIPPED and
       the cursor advances past it; continue to the next.
    3. Refuse to run an action that is not APPROVED/AUTO_APPROVED — return
       None and record why; never execute a PENDING action.
    4. Pre-transition the campaign state (RUNNING_T3 / RUNNING_LITERATURE)
       synchronously — exactly like
       :func:`carmel.services.execution.begin_t3_run`. This is NOT the gate
       that decides between two racing dispatch attempts: the exclusive
       ``.dispatch.lock`` mkdir lease (:func:`_acquire_dispatch_lock`,
       acquired before step 0) already wins that race and rejects the
       loser with ``ActionInFlightError`` before it ever reaches this
       step. The transition's own ``InvalidTransitionError`` instead
       guards against the campaign being in a state this dispatch does
       not expect to find it in for some other reason — e.g. state and
       progress having drifted out of sync — and is raised in the calling
       thread, before any background work starts.
    5. ``mark_running(attempt_id)`` (atomic under the workspace lock).

    On the background thread:

    6. Run the handler. The T3 handler drives
       :func:`carmel.services.execution._finish_t3_run`, which owns the T3
       state transitions; for literature the dispatcher applies the
       post-transition itself: EVERY non-blocking outcome — SUCCEEDED,
       FAILED_NONBLOCKING, BUDGET_EXCEEDED, NO_GROUNDED_FINDINGS — maps to
       ``LITERATURE_READY`` with a warning note (spar round 3, P1-11): a
       barren or over-budget literature run must never stop T3.
    7. ``mark_finished``, advance the cursor, then apply the terminal
       state from ``aggregate_state()`` via ``repair_campaign_state``.
    8. A blocking failure sets the campaign FAILED. FAILED is no longer
       terminal: main added a guarded ``FAILED -> APPROVED_FOR_EXECUTION``
       retry edge (allowed only when the campaign failed from
       ``RUNNING_T3``); after that explicit transition, ``reconcile``
       resets the failed action for one more attempt.

    Returns None when the plan is already complete or the next action is
    not approved; before returning None the state/projection repair is
    re-run so progress and campaign state cannot be left permanently
    disagreeing (spar round 3, P0-8).

    Raises:
        ActionInFlightError: If another dispatch (or a live literature
            attempt) is already running for this workspace.
        InvalidTransitionError: If the campaign is not eligible to enter
            the required RUNNING_* state (e.g. a concurrent dispatch won
            the race). Raised in the calling thread, before any background
            work starts.
        UnsupportedActionKindError: If the plan contains an action kind
            without a registered handler.
    """
    if handlers is None:
        handlers = default_handlers()
    log_path = workspace_root / "decision_log.jsonl"
    lease = _acquire_dispatch_lock(workspace_root, stale_after_s=DEFAULT_STALE_AFTER_S)
    started_background = False
    try:
        plan = load_plan(workspace_root)
        problems = validate_plan_shape(plan)
        if problems:
            raise UnsupportedActionKindError("plan shape is not executable: " + "; ".join(problems))

        # Step 0-1: repair, then load (migrating a Phase-1 plan if needed).
        # A workspace whose filesystem cannot ``flock`` is an environment
        # fault, and main surfaces it as LockStateUnknownError (a 503) from
        # start_supervision. The dispatcher reaches a workspace lock first —
        # reconcile takes one — so the same fault must read the same way
        # instead of escaping as a raw OSError and 500-ing.
        try:
            progress = load_or_init_progress(workspace_root, plan)
            progress = reconcile(workspace_root, in_dispatch_lock=True)
        except OSError as e:
            raise LockStateUnknownError(f"Could not take the workspace lock for {workspace_root}: {e}") from e

        actions_by_id = {a.action_id: a for a in plan.actions}

        # Step 2: skip past rejected actions.
        while not progress.is_complete():
            state = progress.actions[progress.cursor]
            if state.approval_status != ApprovalStatus.REJECTED:
                break
            mark_finished(
                workspace_root,
                state.action_id,
                status=ActionExecutionStatus.SKIPPED,
                outcome=ActionOutcome.REJECTED,
                notes="skipped: action was rejected",
            )
            append_typed_event(
                log_path,
                event="dispatch.action_skipped",
                action_id=state.action_id,
                payload={"reason": "rejected"},
            )
            progress = advance_cursor(workspace_root, state.action_id)

        if progress.is_complete():
            repair_campaign_state(workspace_root, progress)
            return None

        # A blocking failure keeps the campaign FAILED until the operator
        # explicitly retries it through main's guarded
        # FAILED -> APPROVED_FOR_EXECUTION edge (only legal when the
        # campaign failed from RUNNING_T3); `reconcile` then resets the
        # failed action and this projection is no longer FAILED.
        if aggregate_state(progress) == CampaignStateValue.FAILED:
            repair_campaign_state(workspace_root, progress)
            append_typed_event(
                log_path,
                event="dispatch.refused",
                payload={
                    "reason": (
                        "campaign failed on a blocking action; retry requires the explicit "
                        "FAILED -> APPROVED_FOR_EXECUTION transition"
                    )
                },
            )
            return None

        action_state = progress.actions[progress.cursor]

        # Step 3: never execute an unapproved action.
        if action_state.approval_status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
            append_typed_event(
                log_path,
                event="dispatch.refused",
                action_id=action_state.action_id,
                payload={"reason": f"approval_status is {action_state.approval_status.value}"},
            )
            repair_campaign_state(workspace_root, progress)
            return None

        action = actions_by_id.get(action_state.action_id)
        if action is None:
            raise UnsupportedActionKindError(
                f"progress references action {action_state.action_id!r} missing from the plan"
            )
        if action.kind not in handlers:
            raise UnsupportedActionKindError(f"no handler registered for action kind {action.kind.value!r}")

        # Step 3b: the LIVE launch gate, re-run against current spend.
        #
        # `action_state.approval_status` above is a snapshot taken at plan time; budget
        # can have been consumed since by earlier runs or a failed attempt. main's
        # single-action launch paths (`start_t3_action`/`execute_t3_action`) call this
        # before any supervision or state change, and it also vetoes on a recorded human
        # REJECTED decision -- a veto the decision log can apply even though it is not
        # allowed to GRANT authorization (Finding 18).
        #
        # Routing T3 through this dispatcher bypassed `start_t3_action` and therefore
        # silently dropped both guarantees. Placed here, before `start_supervision` and
        # the pre-transition, so a refusal changes nothing and can wedge nothing.
        #
        # Split by kind, because the two guarantees have different scopes. The full gate
        # measures CPU-hours against a per-adapter ExecutionEnvelope, and
        # `authorize_action` escalates any kind that has none -- LITERATURE_SEARCH has
        # no envelope and consumes 0.0 CPU-hours, so running it through the full gate
        # would refuse every literature action ever dispatched. The human-rejection
        # veto, by contrast, is about consent rather than resources and applies to
        # every kind without exception.
        if envelope_for(action.kind) is not None:
            execution._require_launch_authorization(workspace_root, campaign, action)
        elif has_effective_human_rejection(workspace_root, action.action_id):
            raise BudgetExceededError(f"Launch of action {action.action_id} refused: a human has rejected this action.")

        # Step 4: pre-transition — the authoritative, locked concurrency
        # gate (idempotent when a crash already applied it: reconcile has
        # just proven no attempt is live).
        #
        # For T3 the run lock is taken FIRST, in this thread, so that no
        # instant exists in which the campaign reads as RUNNING_T3 while
        # nothing holds the lock — a recovery probe landing there would see
        # no supervisor and no in-flight record and offer to abandon a run
        # that was about to start. Taking it here, rather than on the
        # dispatch thread, is also what lets a run that loses the lock
        # surface to the caller (a 409) instead of being swallowed into
        # ticket.error where nobody is waiting for it.
        running_state = _pre_transition_state(action.kind)
        # Finding P1-13: pass estimated_cpu_hours, matching every launch site
        # in execution.py (start_t3_action / execute_t3_action /
        # start_arc_action). Omitting it defaults the reservation to 0.0, and
        # spend.compute_spend derives the in-flight reservation from exactly
        # this field (`reserved = active.estimated_cpu_hours if active is not
        # None else 0.0`) -- so a dispatcher-launched run in flight would
        # read as reserving nothing, and this dispatcher's own launch gate
        # (`_require_launch_authorization` -> `compute_spend`, above) would
        # be evaluated against a ledger blind to every run it itself started.
        supervision = (
            start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
            if action.kind == ActionKind.T3_RUN
            else None
        )
        try:
            if load_state(workspace_root).state != running_state:
                update_state(workspace_root, running_state, notes=f"action={action.action_id}")

            # Step 5: mark running and publish the attempt on the lease.
            attempt_id = uuid.uuid4().hex
            mark_running(workspace_root, action.action_id, attempt_id)
            publish_lock_info(lease.lock_dir, extra={"action_id": action.action_id, "attempt_id": attempt_id})
        except BaseException:
            # Nothing will reach the handler, so this frame is still the
            # lock's owner and must release it.
            if supervision is not None:
                supervision.close()
            raise

        ticket = DispatchTicket(action.action_id, action.kind, attempt_id)
        thread = threading.Thread(
            target=_finish_dispatch,
            args=(workspace_root, campaign, action, handlers, ticket, lease, supervision),
            name=f"carmel-dispatch-{campaign.campaign_id}",
            daemon=True,
        )
        ticket.thread = thread
        try:
            thread.start()
        except BaseException as exc:
            # started_background is still False here (Finding 3): it must
            # only flip once thread.start() has actually succeeded, or a
            # start failure (thread exhaustion, an RLIMIT_NPROC ceiling)
            # leaves the `finally` below believing a background run is live
            # and skips releasing the lease — this process's own pid would
            # then read as the still-live holder of `.dispatch.lock`
            # forever, wedging every future dispatch for this workspace
            # behind a 409 until the server restarts. No dispatch thread
            # will run, so nothing downstream will ever close the run lock
            # or finish the RUNNING attempt this frame just marked either;
            # both must be repaired right here instead of leaking them.
            if supervision is not None:
                supervision.close()
            _fail_started_action(
                workspace_root,
                action,
                reason=f"could not start background dispatch thread: {exc}",
            )
            raise
        # Only now — after thread.start() has returned successfully — does
        # this frame stop owning the lease; the background thread owns it
        # from here and releases it in `_finish_dispatch`'s `finally`.
        started_background = True
        return ticket
    finally:
        if not started_background:
            lease.release()


def recover_workspace(workspace_root: Path, *, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> PlanProgress:
    """Recover a workspace wedged in a ``RUNNING_*`` state after a crash (Finding 1).

    :func:`~carmel.services.plan_progress.reconcile` is the dispatcher's crash-repair
    pass. It is called from three sites: :func:`execute_next_action` here, and
    (Finding P1-1) ``/run`` and ``/abandon`` in ``carmel.ui.app``, which call it
    directly before their own preflight looks at campaign state — so, unlike an
    earlier version of this docstring claimed, reaching ``RUNNING_*`` no longer
    requires going through ``execute_next_action`` at all.

    ``recover_workspace`` itself, however, has no caller anywhere in this codebase
    today (only tests invoke it directly) — it is a repair entry point for callers
    outside the request/response cycle that ``/run``/``/abandon`` cover: an ops
    script, a future admin action, or a campaign-creation crash. Campaign creation
    runs literature synchronously inside the POST handler, so a crash mid-run there
    — the campaign left at ``RUNNING_LITERATURE`` with a ``RUNNING`` action and
    nothing behind it — is the first thing a new user can hit, and none of
    ``/run``/``/replan``/``/retry``/``/finalize``/``/approve``/``/reject`` can reach
    it (they all 409 outright while the campaign reads ``RUNNING_*``, and
    ``/abandon`` hardcodes ``RUNNING_T3``). Until now only a hand-edit of the
    persisted JSON could recover such a campaign.

    Call this directly while the campaign sits in a ``RUNNING_*`` state, with no
    need to route through ``execute_next_action`` or the UI (which would refuse to
    do anything else until this repair has run).

    Deliberately does NOT take :func:`_acquire_dispatch_lock`'s exclusive
    ``.dispatch.lock`` mkdir lease the way ``execute_next_action`` does. That lease
    is a live-run marker a background dispatch thread holds for the WHOLE run, and
    ``reconcile`` inspects it (via ``lock_is_live``) to tell a live attempt from a
    crashed one. If this function held that same lease while telling ``reconcile``
    to trust it (``in_dispatch_lock=True``), the dispatch lock would always read
    back as live — this very call would own it — and every RUNNING action would be
    misreported as still in flight, permanently defeating recovery. So this calls
    ``reconcile(..., in_dispatch_lock=False)``: liveness is judged purely from
    persisted lock state — a genuinely live ``.dispatch.lock`` or literature run
    lock still correctly raises
    :class:`~carmel.services.plan_progress.ActionInFlightError` (recovery refuses
    to race a real concurrent run) — plus ``_attempt_is_live``'s lease-age
    fallback, which exists for exactly this no-lock-held path.

    ``reconcile`` takes the short-lived workspace flock itself for its
    read-modify-write and already re-runs both the missing-post-transition replay
    and the terminal-projection repair before returning, so no additional locking
    or follow-up repair is needed here.

    Args:
        workspace_root: The campaign workspace root.
        stale_after_s: Attempt-lease staleness horizon, forwarded to ``reconcile``.

    Returns:
        The (possibly repaired) plan progress.

    Raises:
        ActionInFlightError: If the RUNNING action's attempt is still genuinely
            live (a real concurrent dispatch or literature run) — recovery
            correctly refuses rather than racing it.
    """
    return reconcile(workspace_root, stale_after_s=stale_after_s, in_dispatch_lock=False)


def _finish_dispatch(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    handlers: dict[ActionKind, ActionHandler],
    ticket: DispatchTicket,
    lease: _DispatchLease,
    supervision: RunSupervision | None = None,
) -> None:
    """Background half of a dispatch: run the handler and finish the bookkeeping.

    Owns the dispatch lease inherited from :func:`execute_next_action` and
    releases it when done — while this thread is alive the lease's pid
    metadata proves to :func:`~carmel.services.plan_progress.reconcile`
    that the attempt is live. Release goes through
    :meth:`_DispatchLease.release`, never a bare ``rmtree``, so this thread
    can never delete a lock dir a stale-lock steal race has since handed to
    a different process (Finding 7).

    ``supervision`` is the run lock :func:`execute_next_action` took before
    the campaign could read as ``RUNNING_T3``; it is handed straight to
    :func:`execute_action`, which passes ownership to the handler.
    """
    log_path = workspace_root / "decision_log.jsonl"
    try:
        try:
            result = execute_action(workspace_root, campaign, action, handlers=handlers, supervision=supervision)
        except Exception as exc:
            # A handler crash is a failed attempt, never an orphaned RUNNING
            # lease. For T3, _finish_t3_run has already persisted the failure
            # and driven the campaign FAILED; for literature the dispatcher
            # applies the failure post-transition itself.
            ticket.error = exc
            _fail_started_action(workspace_root, action, reason=f"handler raised: {exc}")
            _log.error(
                "Background dispatch of action %s failed for campaign %s",
                action.action_id,
                campaign.campaign_id,
                exc_info=True,
            )
            append_typed_event(
                log_path,
                event="dispatch.background_failed",
                action_id=action.action_id,
                payload={"level": "error", "error": str(exc)},
            )
            return

        succeeded = result.outcome in (ActionOutcome.SUCCEEDED, ActionOutcome.NO_GROUNDED_FINDINGS)
        mark_finished(
            workspace_root,
            action.action_id,
            status=ActionExecutionStatus.SUCCEEDED if succeeded else ActionExecutionStatus.FAILED,
            outcome=result.outcome,
            run_id=result.run_record.run_id,
            notes=result.run_record.error_message,
        )

        # Step 6: post-transition. T3 transitions are owned by
        # _finish_t3_run (see module docstring); literature by the
        # dispatcher.
        if action.kind != ActionKind.T3_RUN:
            _apply_post_transition(workspace_root, action, result.outcome, note=result.run_record.error_message)

        # Step 7: advance the cursor, then apply the terminal projection.
        progress = advance_cursor(workspace_root, action.action_id)
        repair_campaign_state(workspace_root, progress)
        ticket.result = result
    finally:
        lease.release()


def _apply_post_transition(
    workspace_root: Path,
    action: PlannedAction,
    outcome: ActionOutcome,
    *,
    note: str | None = None,
) -> None:
    """Apply the literature post-transition (never the terminal projection).

    T3 has no branch here: its transitions are owned by
    :func:`carmel.services.execution._finish_t3_run` (see module docstring).
    """
    if outcome == ActionOutcome.FAILED_BLOCKING:
        update_state(workspace_root, CampaignStateValue.FAILED, notes=note)
        return
    # Literature: every non-blocking outcome maps to LITERATURE_READY
    # (spar round 3, P1-11) — a barren or over-budget run must never stop T3.
    if outcome == ActionOutcome.SUCCEEDED:
        notes = None
    else:
        notes = f"warning: literature outcome {outcome.value}" + (f" ({note})" if note else "")
    update_state(workspace_root, CampaignStateValue.LITERATURE_READY, notes=notes)
