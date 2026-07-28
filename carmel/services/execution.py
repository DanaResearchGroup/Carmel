# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""High-level orchestration of T3 execution and diagnostics persistence.

This module owns the *workflow* around a T3 run: state transitions,
provenance, decision-log entries, run-record persistence, diagnostics
persistence, and SVG artifact generation. The actual T3 invocation lives
in :mod:`carmel.adapters.t3`.

The adapter is injected as a parameter to ``execute_t3_action`` so that
unit tests can drive the orchestration deterministically with an inline
test double **without** introducing a mock-mode flag in production code.
Production callers always pass the real :class:`T3Adapter`.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from carmel.logger import get_logger
from carmel.schemas.approval import ApprovalPolicy, ApprovalRequirement
from carmel.schemas.campaign import Campaign
from carmel.schemas.diagnostics import DiagnosticsV1
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import ActiveRun, FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.approvals import has_effective_human_approval, load_policy
from carmel.services.artifacts import read_json, write_json
from carmel.services.authorization import BudgetExceededError, decide_requirement
from carmel.services.decision_log import append_event
from carmel.services.drawing import SELECTION_SVG_FILENAMES, write_selection_svgs
from carmel.services.processes import kill_process_group
from carmel.services.provenance import record
from carmel.services.recovery import (
    RunLiveness,
    RunLivenessReport,
    RunSupervision,
    clear_active_run,
    probe_run_liveness,
    start_supervision,
)
from carmel.services.spend import RUNS_DIR_NAME, compute_spend
from carmel.services.state_machine import InvalidTransitionError, can_transition, load_state, update_state

DIAGNOSTICS_FILE_NAME = "diagnostics.json"
ARC_DIAGNOSTICS_FILE_NAME = "arc_diagnostics.json"
MODELS_DIR_NAME = "models"
ARC_MODELS_SUBDIR_NAME = "arc"

_log = get_logger("services.execution")


class T3AdapterProtocol(Protocol):
    """Structural type for anything that can run a T3 action.

    The real implementation is :class:`carmel.adapters.t3.T3Adapter`.
    Tests may provide an inline double conforming to this protocol — this
    is **not** a mock adapter mode in production code; production
    callers always inject the real adapter.
    """

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple[RunRecord, DiagnosticsV1 | None]: ...


class ARCAdapterProtocol(Protocol):
    """Structural type for anything that can run an ARC action.

    Same shape as :class:`T3AdapterProtocol` (the two adapters are peers). The
    real implementation is :class:`carmel.adapters.arc.ARCAdapter`. Tests may
    provide an inline double conforming to this protocol — this is **not** a mock
    adapter mode in production code; production callers always inject the real
    adapter.
    """

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple[RunRecord, DiagnosticsV1 | None]: ...


def save_run_record(workspace_root: Path, run_record: RunRecord) -> Path:
    """Persist a RunRecord under ``runs/<run_id>.json``."""
    runs_dir = workspace_root / RUNS_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_record.run_id}.json"
    write_json(path, run_record)
    return path


def save_diagnostics(workspace_root: Path, diagnostics: DiagnosticsV1) -> Path:
    """Persist diagnostics.json at the workspace root."""
    path = workspace_root / DIAGNOSTICS_FILE_NAME
    write_json(path, diagnostics)
    return path


def load_diagnostics(workspace_root: Path) -> DiagnosticsV1 | None:
    """Load persisted T3 diagnostics, if present."""
    path = workspace_root / DIAGNOSTICS_FILE_NAME
    if not path.exists():
        return None
    return DiagnosticsV1.model_validate(read_json(path))


def clear_stale_diagnostics_artifacts(workspace_root: Path) -> None:
    """Remove any previous run's ``diagnostics.json`` and selection SVGs.

    Called before every T3 run starts so that a run which never reaches
    its success branch — whether it returns a typed failure or the
    process crashes outright — cannot leave a prior run's diagnostics and
    SVGs on disk looking like current output. Tolerant of the files not
    existing (first run, or a prior run that never produced them).
    """
    (workspace_root / DIAGNOSTICS_FILE_NAME).unlink(missing_ok=True)
    models_dir = workspace_root / MODELS_DIR_NAME
    for filename in SELECTION_SVG_FILENAMES:
        (models_dir / filename).unlink(missing_ok=True)


def clear_stale_arc_diagnostics_artifacts(workspace_root: Path) -> None:
    """Remove any previous run's ``arc_diagnostics.json`` and selection SVGs.

    ARC mirror of :func:`clear_stale_diagnostics_artifacts`: called before
    every ARC run starts so that a run which never reaches its success
    branch — whether it returns a typed failure or the process crashes
    outright — cannot leave a prior run's ARC diagnostics and SVGs on disk
    looking like current output. Tolerant of the files not existing (first
    run, or a prior run that never produced them).
    """
    (workspace_root / ARC_DIAGNOSTICS_FILE_NAME).unlink(missing_ok=True)
    models_dir = workspace_root / MODELS_DIR_NAME / ARC_MODELS_SUBDIR_NAME
    for filename in SELECTION_SVG_FILENAMES:
        (models_dir / filename).unlink(missing_ok=True)


def save_arc_diagnostics(workspace_root: Path, diagnostics: DiagnosticsV1) -> Path:
    """Persist arc_diagnostics.json at the workspace root.

    ARC diagnostics are stored under a distinct filename so a standalone ARC job
    never clobbers the T3 diagnostics (the two adapters are peers).
    """
    path = workspace_root / ARC_DIAGNOSTICS_FILE_NAME
    write_json(path, diagnostics)
    return path


def load_arc_diagnostics(workspace_root: Path) -> DiagnosticsV1 | None:
    """Load persisted ARC diagnostics, if present."""
    path = workspace_root / ARC_DIAGNOSTICS_FILE_NAME
    if not path.exists():
        return None
    return DiagnosticsV1.model_validate(read_json(path))


def _default_adapter() -> T3AdapterProtocol:
    """Return the production T3 adapter (lazy import to avoid cycles)."""
    from carmel.adapters.t3 import T3Adapter

    return T3Adapter()


def _t3_tool_name() -> str:
    """Return the canonical T3 tool name (lazy import to avoid cycles)."""
    from carmel.adapters.t3 import T3_TOOL_NAME

    return T3_TOOL_NAME


def _default_arc_adapter() -> ARCAdapterProtocol:
    """Return the production ARC adapter (lazy import to avoid cycles)."""
    from carmel.adapters.arc import ARCAdapter

    return ARCAdapter()


def _arc_tool_name() -> str:
    """Return the canonical ARC tool name (lazy import to avoid cycles)."""
    from carmel.adapters.arc import ARC_TOOL_NAME

    return ARC_TOOL_NAME


def _require_launch_authorization(workspace_root: Path, campaign: Campaign, action: PlannedAction) -> None:
    """Refuse a launch the live gate escalates and no human has approved.

    The plan-time approval requirement is a snapshot: budget can have been
    consumed since (earlier runs, a retry's failed attempt), so every
    launch path re-runs the combined gate
    (:func:`carmel.services.authorization.decide_requirement`) against the
    campaign's *current* remaining budget. This runs BEFORE
    ``start_supervision`` and any state transition — so the action is not
    yet reserved in ``active_run.json`` and is never counted against
    itself, and a refusal changes nothing and can wedge nothing.

    If the live gate still auto-approves, the launch proceeds (this is
    what keeps auto-approved runs — including retries after pre-launch
    failures, which consume nothing — uninterrupted). If it escalates, a
    recorded *effective* human approval for this action authorizes the
    launch anyway (a human who approved is an override that survives
    stale-over-budget retries); an ``AUTO_APPROVED``-only record does not,
    because an auto-approval is only valid while the gate still
    auto-approves.

    This is the service-boundary twin of the UI ``/run`` route's
    decision-log check, so direct service callers cannot bypass the
    budget. Launch only: finalize, retry re-arming, re-plan, and abandon
    never pass through here.

    Raises:
        BudgetExceededError: If the live gate requires approval and no
            effective human approval is recorded for the action. Raised
            before any supervision or state change.
    """
    try:
        policy = load_policy(workspace_root)
    except FileNotFoundError:
        # No persisted policy (a bare workspace): gate under the default,
        # conservative policy rather than refusing to gate at all.
        policy = ApprovalPolicy()
    remaining = compute_spend(workspace_root).remaining(campaign.input.budgets.cpu_hours)
    requirement, rationale = decide_requirement(
        action,
        policy=policy,
        remaining_cpu_hours=remaining,
        budgets=campaign.input.budgets,
    )
    if requirement == ApprovalRequirement.AUTO_APPROVED:
        return
    if has_effective_human_approval(workspace_root, action.action_id):
        return
    raise BudgetExceededError(
        f"Launch of action {action.action_id} refused ({rationale}); "
        f"a recorded human approval for this action is required before it may run."
    )


def execute_t3_action(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: T3AdapterProtocol | None = None,
) -> tuple[RunRecord, DiagnosticsV1 | None]:
    """Run a T3 action end-to-end and persist all artifacts.

    Transitions state from ``APPROVED_FOR_EXECUTION`` →
    ``RUNNING_T3`` → ``DIAGNOSTICS_READY`` → ``COMPLETED_PHASE1`` on
    success, or → ``FAILED`` on any failure.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned T3 action.
        adapter: Optional adapter override. Production passes ``None``
            (the default real adapter is used). Tests may pass an inline
            double conforming to :class:`T3AdapterProtocol`.

    Returns:
        Tuple of (RunRecord, DiagnosticsV1 or None on failure).

    Raises:
        RunAlreadySupervisedError: If a live process already holds this
            campaign's run lock.
        BudgetExceededError: If the live authorization re-check refuses
            the launch — see :func:`_require_launch_authorization`. Raised
            before supervision or any state change.
    """
    if adapter is None:
        adapter = _default_adapter()
    _require_launch_authorization(workspace_root, campaign, action)
    supervision = start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
    try:
        started = begin_t3_run(workspace_root, action)
    except BaseException:
        supervision.close()
        raise
    return _finish_t3_run(workspace_root, campaign, action, adapter, started, supervision)


def begin_t3_run(workspace_root: Path, action: PlannedAction) -> datetime:
    """Enter ``RUNNING_T3``, refusing the run if the campaign is not eligible.

    Kept separate from the work itself so that a caller running T3 in the
    background (see :func:`start_t3_action`) still performs this check
    *synchronously*, in its own thread. That ordering is what makes a
    double-submitted run fail loudly for the user who submitted it,
    instead of being accepted and then racing a run already in flight.

    Args:
        workspace_root: The campaign workspace root.
        action: The planned T3 action.

    Returns:
        The run's start timestamp.

    Raises:
        InvalidTransitionError: If the campaign is not eligible to enter
            ``RUNNING_T3``. The workspace is left untouched.
    """
    update_state(workspace_root, CampaignStateValue.RUNNING_T3, notes=f"action={action.action_id}")
    return datetime.now(UTC)


def start_t3_action(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: T3AdapterProtocol | None = None,
) -> threading.Thread:
    """Enter ``RUNNING_T3`` now and run the action on a background thread.

    T3 runs for minutes to hours; executed inline it holds a web request
    open far past any browser or reverse-proxy timeout, so the redirect to
    the auto-refreshing dashboard never arrives and the whole
    ``RUNNING_T3`` UX is unreachable from the tab that started the run.

    The state transition is deliberately *not* backgrounded — see
    :func:`begin_t3_run` — and neither is taking the run lock. Both happen
    here, in order, before this returns.

    The order matters more than it looks. Supervision is taken *first*, so
    that no instant exists in which the campaign reads as ``RUNNING_T3``
    while no lock is held: a recovery probe landing in such a window would
    find no supervisor and no in-flight record, conclude that nothing had
    ever started, and offer to abandon a run that was about to launch.

    The returned thread is a daemon: a run outlives neither its own
    process tree (the adapter kills that on timeout) nor an operator who
    stops the server — killing the server leaves T3 running unsupervised,
    since Carmel's supervision lives in the process being stopped.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned T3 action.
        adapter: Optional adapter override; production passes ``None``.

    Returns:
        The started thread. Callers that need the run to finish — tests,
        and any future synchronous caller — can ``join()`` it rather than
        polling the workspace.

    Raises:
        InvalidTransitionError: If the campaign is not eligible to enter
            ``RUNNING_T3``. Raised in the calling thread, before any
            background work starts.
        RunAlreadySupervisedError: If a live process already holds this
            campaign's run lock. Raised before the transition, so the
            campaign is left to the run that already owns it.
        BudgetExceededError: If the live authorization re-check refuses
            the launch — see :func:`_require_launch_authorization`. Raised
            synchronously, before supervision or any state change.
    """
    if adapter is None:
        adapter = _default_adapter()
    _require_launch_authorization(workspace_root, campaign, action)
    supervision = start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
    try:
        started = begin_t3_run(workspace_root, action)
    except BaseException:
        supervision.close()
        raise

    def _run() -> None:
        try:
            _finish_t3_run(workspace_root, campaign, action, adapter, started, supervision)
        except Exception:
            # _finish_t3_run has already driven the campaign to FAILED and
            # persisted the failure; re-raising into the thread's excepthook
            # would only print a traceback nobody is waiting on.
            _log.error("Background T3 run failed for campaign %s", campaign.campaign_id, exc_info=True)

    thread = threading.Thread(target=_run, name=f"carmel-t3-{campaign.campaign_id}", daemon=True)
    try:
        thread.start()
    except BaseException:
        # The campaign is already in RUNNING_T3, but no thread will ever
        # run _finish_t3_run, so nothing would release the lock or record
        # an outcome — the exact permanent wedge this run path exists to
        # avoid, one layer up. Release supervision and fail the run so it
        # reads as recoverable (retry re-arms it) rather than as forever
        # in flight.
        supervision.close()
        with contextlib.suppress(Exception):
            update_state(
                workspace_root,
                CampaignStateValue.FAILED,
                notes="the T3 run thread could not be started",
            )
        raise
    return thread


def _finish_t3_run(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: T3AdapterProtocol,
    started: datetime,
    supervision: RunSupervision,
) -> tuple[RunRecord, DiagnosticsV1 | None]:
    """Run the action and persist every artifact, assuming ``RUNNING_T3``.

    The campaign must already have entered ``RUNNING_T3`` (see
    :func:`begin_t3_run`). Everything here is inside the protected region:
    any failure must still be able to drive the campaign to ``FAILED``
    rather than leaving it wedged in ``RUNNING_T3``.

    The whole region runs under a :class:`~carmel.services.recovery.RunSupervision`
    taken by the caller, which holds the campaign's run lock and records
    the tool's process group. That covers the one failure this region
    cannot handle itself: the process is killed outright, no ``except``
    runs, and the campaign is left in ``RUNNING_T3``. What the supervision
    leaves behind is what lets a later Carmel tell that wreckage apart
    from a run still in progress. Closing it here, rather than where it
    was taken, is what lets the lock span the whole run while still being
    acquired before the campaign ever reads as ``RUNNING_T3``.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned T3 action.
        adapter: The adapter to run.
        started: The run's start timestamp, from :func:`begin_t3_run`.
        supervision: The held run lock, closed on the way out.

    Returns:
        Tuple of (RunRecord, DiagnosticsV1 or None on failure).
    """
    with contextlib.closing(supervision):
        run_record: RunRecord | None = None
        try:
            clear_stale_diagnostics_artifacts(workspace_root)
            append_event(
                workspace_root / "decision_log.jsonl",
                {
                    "event": "t3_run_started",
                    "action_id": action.action_id,
                    "started_at": started.isoformat(),
                },
            )

            run_record, diagnostics = adapter.run(
                workspace_root=workspace_root,
                campaign=campaign,
                action=action,
                on_process_start=supervision.record_process_group,
            )

            if run_record.status == RunStatus.SUCCEEDED and diagnostics is not None:
                # Diagnostics/SVGs must be durable BEFORE the decision log
                # claims the run succeeded — otherwise a failure here would
                # leave a "succeeded" finished event contradicted by the
                # UNKNOWN-failure event that follows it.
                save_diagnostics(workspace_root, diagnostics)
                write_selection_svgs(
                    workspace_root / MODELS_DIR_NAME,
                    diagnostics.species_to_compute,
                    diagnostics.reactions_to_compute,
                    diagnostics.pdep_networks_to_compute,
                )

            save_run_record(workspace_root, run_record)
            record(
                workspace_root,
                "t3_run",
                {
                    "run_id": run_record.run_id,
                    "action_id": action.action_id,
                    "status": run_record.status.value,
                    "failure_code": run_record.failure_code.value,
                    "level_of_theory": run_record.level_of_theory,
                },
            )
            append_event(
                workspace_root / "decision_log.jsonl",
                {
                    "event": "t3_run_finished",
                    "run_id": run_record.run_id,
                    "status": run_record.status.value,
                    "failure_code": run_record.failure_code.value,
                },
            )

            if run_record.status == RunStatus.SUCCEEDED and diagnostics is not None:
                update_state(workspace_root, CampaignStateValue.DIAGNOSTICS_READY)
                update_state(workspace_root, CampaignStateValue.COMPLETED_PHASE1)
                _log.info("T3 run %s succeeded for campaign %s", run_record.run_id, campaign.campaign_id)
            else:
                update_state(
                    workspace_root,
                    CampaignStateValue.FAILED,
                    notes=run_record.error_message,
                )
                _log.warning("T3 run %s failed: %s", run_record.run_id, run_record.failure_code.value)

            return run_record, diagnostics
        except Exception as e:
            _handle_unexpected_failure(
                workspace_root,
                campaign,
                action,
                adapter,
                started,
                run_record,
                e,
                tool_name=_t3_tool_name,
                provenance_kind="t3_run",
                finished_event="t3_run_finished",
                tool_label="T3",
            )
            raise


class RunStillLiveError(RuntimeError):
    """Raised when a run cannot be abandoned because it is still executing."""


def abandon_t3_run(workspace_root: Path, campaign: Campaign) -> tuple[CampaignState, RunLivenessReport]:
    """End a campaign stuck in ``RUNNING_T3``, after proving it is not running.

    Recovers the case no ``except`` can: the Carmel process supervising a
    run was killed, so nothing ever wrote the run's ending and the
    campaign sits in ``RUNNING_T3`` indefinitely.

    Abandoning is not a state edit. It first establishes what is actually
    executing, and then makes "this run is over" true rather than merely
    recorded:

    * A run with a living supervisor is refused. It is genuinely in
      progress and will record its own outcome.
    * An orphaned tool tree is stopped before anything else happens. Marking
      the campaign FAILED while T3 and RMG keep writing into the workspace
      would be the same lie the process-tree kill exists to prevent.
    * Only then is the failure recorded and the campaign moved to FAILED,
      from where the normal recovery edges apply.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to recover.

    Returns:
        Tuple of (the new campaign state, the liveness report that
        justified abandoning the run).

    Raises:
        RunStillLiveError: If a supervisor is alive, or if an orphaned
            process group could not be stopped. In both cases the campaign
            is left untouched.
        InvalidTransitionError: If the campaign is not in ``RUNNING_T3``.
            Checked here rather than left to the caller: several other
            states can legally be failed, so without this guard abandoning
            would double as a way to fail any campaign at all.
    """
    return _abandon_run(
        workspace_root,
        campaign,
        expected_state=CampaignStateValue.RUNNING_T3,
        tool_name=_t3_tool_name(),
        event_name="t3_run_abandoned",
    )


def abandon_arc_run(workspace_root: Path, campaign: Campaign) -> tuple[CampaignState, RunLivenessReport]:
    """End a campaign stuck in ``RUNNING_ARC``, after proving it is not running.

    ARC mirror of :func:`abandon_t3_run` — see that function for the full
    rationale. The supervision machinery it drives
    (:func:`~carmel.services.recovery.probe_run_liveness`,
    :func:`~carmel.services.processes.kill_process_group`,
    :func:`~carmel.services.recovery.clear_active_run`) is tool-agnostic;
    only the guarded state and the recorded tool name differ.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to recover.

    Returns:
        Tuple of (the new campaign state, the liveness report that
        justified abandoning the run).

    Raises:
        RunStillLiveError: If a supervisor is alive, or if an orphaned
            process group could not be stopped. In both cases the campaign
            is left untouched.
        InvalidTransitionError: If the campaign is not in ``RUNNING_ARC``.
    """
    return _abandon_run(
        workspace_root,
        campaign,
        expected_state=CampaignStateValue.RUNNING_ARC,
        tool_name=_arc_tool_name(),
        event_name="arc_run_abandoned",
    )


def _abandon_run(
    workspace_root: Path,
    campaign: Campaign,
    *,
    expected_state: CampaignStateValue,
    tool_name: str,
    event_name: str,
) -> tuple[CampaignState, RunLivenessReport]:
    """Tool-agnostic core of :func:`abandon_t3_run` / :func:`abandon_arc_run`.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to recover.
        expected_state: The only running state this campaign may be in
            (``RUNNING_T3`` or ``RUNNING_ARC``). Checked here rather than
            left to the caller: several other states can legally be
            failed, so without this guard abandoning would double as a way
            to fail any campaign at all.
        tool_name: The canonical tool name recorded on the abandoned
            RunRecord.
        event_name: The decision-log/provenance event name (e.g.
            ``"t3_run_abandoned"``).

    Returns:
        Tuple of (the new campaign state, the liveness report that
        justified abandoning the run).

    Raises:
        RunStillLiveError: If a supervisor is alive, or if an orphaned
            process group could not be stopped.
        InvalidTransitionError: If the campaign is not in *expected_state*.
    """
    current = load_state(workspace_root).state
    if current != expected_state:
        raise InvalidTransitionError(
            f"Only a campaign in {expected_state.value} can have its run abandoned; this one is in {current.value}."
        )
    report = probe_run_liveness(workspace_root)
    if report.liveness in {RunLiveness.SUPERVISED, RunLiveness.UNKNOWN}:
        # UNKNOWN is refused for the same reason SUPERVISED is. Something
        # is alive that Carmel cannot account for, and the only way to
        # abandon here would be to assume it is not this run's — precisely
        # the assumption that ends with the tool still writing into a
        # workspace Carmel has already declared finished.
        raise RunStillLiveError(report.detail)

    active = report.active_run
    if report.liveness == RunLiveness.ORPHANED and active is not None and active.process_group_id is not None:
        _log.warning(
            "Abandoning campaign %s: stopping orphaned process group %s",
            campaign.campaign_id,
            active.process_group_id,
        )
        if not kill_process_group(active.process_group_id, active.command, active.leader_starttime):
            raise RunStillLiveError(
                f"Could not stop process group {active.process_group_id}; the campaign has been left "
                f"in {expected_state.value} because its tool may still be running."
            )

    if active is not None:
        save_run_record(workspace_root, _abandoned_run_record(active, report.detail, tool_name))
    clear_active_run(workspace_root)
    append_event(
        workspace_root / "decision_log.jsonl",
        {
            "event": event_name,
            "liveness": report.liveness.value,
            "detail": report.detail,
        },
    )
    record(
        workspace_root,
        event_name,
        {"liveness": report.liveness.value, "detail": report.detail},
    )
    state = update_state(workspace_root, CampaignStateValue.FAILED, notes=report.detail)
    _log.warning("Campaign %s abandoned: %s", campaign.campaign_id, report.detail)
    return state, report


def _abandoned_run_record(active: ActiveRun, detail: str, tool_name: str) -> RunRecord:
    """Build the RunRecord for a run that was abandoned rather than finished.

    The run produced no outcome of its own — that is what abandoning
    means — so it is recorded under its own failure code rather than a
    borrowed one that would imply Carmel observed the tool fail.

    Args:
        active: The in-flight record left behind by the dead supervisor.
        detail: The liveness finding that justified abandoning the run.
        tool_name: The canonical name of the tool the run belonged to.

    Returns:
        A failed RunRecord for the abandoned run.
    """
    return RunRecord(
        run_id=str(uuid4()),
        action_id=active.action_id,
        tool_name=tool_name,
        status=RunStatus.FAILED,
        failure_code=FailureCode.ABANDONED,
        started_at=active.started_at,
        ended_at=datetime.now(UTC),
        estimated_cpu_hours=active.estimated_cpu_hours,
        submission_mode=SubmissionMode.SUBPROCESS,
        error_message=detail,
    )


def _handle_unexpected_failure(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: T3AdapterProtocol | ARCAdapterProtocol,
    started: datetime,
    run_record: RunRecord | None,
    error: Exception,
    *,
    tool_name: Callable[[], str],
    provenance_kind: str,
    finished_event: str,
    tool_label: str,
) -> None:
    """Best-effort cleanup after an unexpected exception during execution.

    Tool-agnostic: shared by ``execute_t3_action`` and ``execute_arc_action``,
    parameterized by the tool's canonical name and its provenance/decision-log
    keys. This function must never raise anything: the caller's ``raise`` must
    re-surface ``error`` unmodified, not an ``InvalidTransitionError`` or an
    exception from this function's own persistence attempts. If the
    adapter already returned a real ``run_record`` before the failure, it
    is reused (and marked failed) so the persisted run_id matches the
    actual run instead of orphaning it behind a fabricated one.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned action.
        adapter: The adapter that was invoked for this run (used to derive
            ``submission_mode`` when no real run record exists).
        started: When the run started.
        run_record: The real RunRecord the adapter returned, if any.
        error: The original exception that triggered this handler.
        tool_name: Lazily-called resolver for the canonical tool name
            (e.g. ``_t3_tool_name`` or ``_arc_tool_name``), used only to
            fabricate a run record when no real ``run_record`` exists. Passed
            as a callable rather than an already-resolved string so this
            (rarely exercised) cleanup path doesn't pay for a lookup whose
            result may not even be used.
        provenance_kind: Provenance record kind (e.g. ``"t3_run"``).
        finished_event: Decision-log finished-event name
            (e.g. ``"t3_run_finished"``).
        tool_label: Human-readable tool label for log messages
            (e.g. ``"T3"``).
    """
    ended = datetime.now(UTC)
    if run_record is not None:
        failed_record = run_record.model_copy(
            update={
                "status": RunStatus.FAILED,
                "failure_code": FailureCode.UNKNOWN,
                "ended_at": ended,
                "error_message": str(error),
            }
        )
    else:
        failed_record = RunRecord(
            run_id=str(uuid4()),
            action_id=action.action_id,
            tool_name=tool_name(),
            status=RunStatus.FAILED,
            failure_code=FailureCode.UNKNOWN,
            started_at=started,
            ended_at=ended,
            submission_mode=getattr(adapter, "submission_mode", SubmissionMode.SUBPROCESS),
            error_message=str(error),
        )

    try:
        save_run_record(workspace_root, failed_record)
        record(
            workspace_root,
            provenance_kind,
            {
                "run_id": failed_record.run_id,
                "action_id": action.action_id,
                "status": failed_record.status.value,
                "failure_code": failed_record.failure_code.value,
                "level_of_theory": failed_record.level_of_theory,
            },
        )
        append_event(
            workspace_root / "decision_log.jsonl",
            {
                "event": finished_event,
                "run_id": failed_record.run_id,
                "status": failed_record.status.value,
                "failure_code": failed_record.failure_code.value,
            },
        )
    except Exception:
        _log.error(
            "Failed to persist failure artifacts for campaign %s after run exception",
            campaign.campaign_id,
            exc_info=True,
        )

    try:
        current = load_state(workspace_root).state
        if can_transition(current, CampaignStateValue.FAILED):
            update_state(workspace_root, CampaignStateValue.FAILED, notes=failed_record.error_message)
    except Exception:
        _log.error(
            "Failed to transition campaign %s to FAILED after run exception",
            campaign.campaign_id,
            exc_info=True,
        )

    _log.error("%s run raised an unexpected exception for campaign %s: %s", tool_label, campaign.campaign_id, error)


def execute_arc_action(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: ARCAdapterProtocol | None = None,
) -> tuple[RunRecord, DiagnosticsV1 | None]:
    """Run an ARC action end-to-end and persist all artifacts.

    Peer to :func:`execute_t3_action`. Transitions state from
    ``APPROVED_FOR_EXECUTION`` -> ``RUNNING_ARC`` -> ``RESULTS_READY`` ->
    ``COMPLETED_PHASE1`` on success, or -> ``FAILED`` on any failure. Owns the
    ARC workflow: state transitions, decision-log entries, provenance, and
    run-record + diagnostics persistence. As with T3, everything after the
    ``RUNNING_ARC`` transition succeeds runs inside a protected region: any
    unexpected exception still drives the campaign to ``FAILED`` rather than
    leaving it wedged in ``RUNNING_ARC``.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned ``run_arc`` action.
        adapter: Optional adapter override. Production passes ``None`` (the
            default real adapter is used). Tests may pass an inline double
            conforming to :class:`ARCAdapterProtocol`.

    The whole protected region runs under a
    :class:`~carmel.services.recovery.RunSupervision`, exactly as T3's
    does. Supervision is taken *before* the ``RUNNING_ARC`` transition, so
    no instant exists in which the campaign reads as ``RUNNING_ARC`` while
    no run lock is held: a recovery probe landing in such a window would
    find no supervisor and no in-flight record, conclude that nothing had
    ever started, and offer to abandon a run that was about to launch.

    Returns:
        Tuple of (RunRecord, DiagnosticsV1 or None on failure).

    Raises:
        InvalidTransitionError: If the campaign is not eligible to enter
            ``RUNNING_ARC``. The workspace is left untouched.
        RunAlreadySupervisedError: If a live process already holds this
            campaign's run lock.
        BudgetExceededError: If the live authorization re-check refuses
            the launch — see :func:`_require_launch_authorization`. Raised
            before supervision or any state change.
    """
    if adapter is None:
        adapter = _default_arc_adapter()
    _require_launch_authorization(workspace_root, campaign, action)
    supervision = start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
    try:
        started = begin_arc_run(workspace_root, action)
    except BaseException:
        supervision.close()
        raise
    return _finish_arc_run(workspace_root, campaign, action, adapter, started, supervision)


def begin_arc_run(workspace_root: Path, action: PlannedAction) -> datetime:
    """Enter ``RUNNING_ARC``, refusing the run if the campaign is not eligible.

    ARC mirror of :func:`begin_t3_run`. Kept separate from the work itself
    so that a caller running ARC in the background (see
    :func:`start_arc_action`) still performs this check *synchronously*, in
    its own thread. That ordering is what makes a double-submitted run fail
    loudly for the user who submitted it, instead of being accepted and
    then racing a run already in flight.

    Args:
        workspace_root: The campaign workspace root.
        action: The planned ``run_arc`` action.

    Returns:
        The run's start timestamp.

    Raises:
        InvalidTransitionError: If the campaign is not eligible to enter
            ``RUNNING_ARC``. The workspace is left untouched.
    """
    update_state(workspace_root, CampaignStateValue.RUNNING_ARC, notes=f"action={action.action_id}")
    return datetime.now(UTC)


def start_arc_action(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: ARCAdapterProtocol | None = None,
) -> threading.Thread:
    """Enter ``RUNNING_ARC`` now and run the action on a background thread.

    ARC mirror of :func:`start_t3_action` — see that function for the full
    rationale. An ARC job runs for minutes to hours; executed inline it
    holds a web request open far past any browser timeout, so the redirect
    to the auto-refreshing dashboard never arrives.

    Ordering invariants are identical to T3's: the live launch
    authorization re-check runs first (before anything is taken, so a
    refusal wedges nothing), supervision is taken second, and the
    ``RUNNING_ARC`` transition happens third — synchronously, in the
    caller's thread — so no instant exists in which the campaign reads as
    ``RUNNING_ARC`` while no run lock is held. Only the protected region
    (:func:`_finish_arc_run`) is backgrounded.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned ``run_arc`` action.
        adapter: Optional adapter override; production passes ``None``.

    Returns:
        The started daemon thread. Callers that need the run to finish —
        tests, and any future synchronous caller — can ``join()`` it
        rather than polling the workspace.

    Raises:
        InvalidTransitionError: If the campaign is not eligible to enter
            ``RUNNING_ARC``. Raised in the calling thread, before any
            background work starts.
        RunAlreadySupervisedError: If a live process already holds this
            campaign's run lock. Raised before the transition, so the
            campaign is left to the run that already owns it.
        BudgetExceededError: If the live authorization re-check refuses
            the launch — see :func:`_require_launch_authorization`. Raised
            synchronously, before supervision or any state change.
    """
    if adapter is None:
        adapter = _default_arc_adapter()
    _require_launch_authorization(workspace_root, campaign, action)
    supervision = start_supervision(workspace_root, action.action_id, action.estimated_cpu_hours)
    try:
        started = begin_arc_run(workspace_root, action)
    except BaseException:
        supervision.close()
        raise

    def _run() -> None:
        try:
            _finish_arc_run(workspace_root, campaign, action, adapter, started, supervision)
        except Exception:
            # _finish_arc_run has already driven the campaign to FAILED and
            # persisted the failure; re-raising into the thread's excepthook
            # would only print a traceback nobody is waiting on.
            _log.error("Background ARC run failed for campaign %s", campaign.campaign_id, exc_info=True)

    thread = threading.Thread(target=_run, name=f"carmel-arc-{campaign.campaign_id}", daemon=True)
    try:
        thread.start()
    except BaseException:
        # The campaign is already in RUNNING_ARC, but no thread will ever
        # run _finish_arc_run, so nothing would release the lock or record
        # an outcome — the exact permanent wedge this run path exists to
        # avoid, one layer up. Release supervision and fail the run so it
        # reads as recoverable (retry re-arms it) rather than as forever
        # in flight.
        supervision.close()
        with contextlib.suppress(Exception):
            update_state(
                workspace_root,
                CampaignStateValue.FAILED,
                notes="the ARC run thread could not be started",
            )
        raise
    return thread


def _finish_arc_run(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: ARCAdapterProtocol,
    started: datetime,
    supervision: RunSupervision,
) -> tuple[RunRecord, DiagnosticsV1 | None]:
    """Run the ARC work, persist every artifact, and leave ``RUNNING_ARC``.

    ARC mirror of :func:`_finish_t3_run`: called with the campaign already
    in ``RUNNING_ARC`` (see :func:`begin_arc_run`), everything here runs
    inside the protected region — any failure still drives the campaign to
    ``FAILED`` rather than leaving it wedged — and the passed-in
    :class:`~carmel.services.recovery.RunSupervision` (already holding the
    campaign's run lock) is closed on every exit path.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned ``run_arc`` action.
        adapter: The ARC adapter to invoke.
        started: The run's start timestamp, from :func:`begin_arc_run`.
        supervision: The live supervision guarding this run.

    Returns:
        Tuple of (RunRecord, DiagnosticsV1 or None on failure).
    """
    with contextlib.closing(supervision):
        run_record: RunRecord | None = None
        try:
            clear_stale_arc_diagnostics_artifacts(workspace_root)
            append_event(
                workspace_root / "decision_log.jsonl",
                {
                    "event": "arc_run_started",
                    "action_id": action.action_id,
                    "started_at": started.isoformat(),
                },
            )

            run_record, diagnostics = adapter.run(
                workspace_root=workspace_root,
                campaign=campaign,
                action=action,
                on_process_start=supervision.record_process_group,
            )

            if run_record.status == RunStatus.SUCCEEDED and diagnostics is not None:
                # Diagnostics/SVGs must be durable BEFORE the decision log
                # claims the run succeeded — otherwise a failure here would
                # leave a "succeeded" finished event contradicted by the
                # UNKNOWN-failure event that follows it. Same ordering as T3.
                save_arc_diagnostics(workspace_root, diagnostics)
                write_selection_svgs(
                    workspace_root / MODELS_DIR_NAME / ARC_MODELS_SUBDIR_NAME,
                    diagnostics.species_to_compute,
                    diagnostics.reactions_to_compute,
                    diagnostics.pdep_networks_to_compute,
                )

            save_run_record(workspace_root, run_record)
            record(
                workspace_root,
                "arc_run",
                {
                    "run_id": run_record.run_id,
                    "action_id": action.action_id,
                    "status": run_record.status.value,
                    "failure_code": run_record.failure_code.value,
                    "level_of_theory": run_record.level_of_theory,
                },
            )
            append_event(
                workspace_root / "decision_log.jsonl",
                {
                    "event": "arc_run_finished",
                    "run_id": run_record.run_id,
                    "status": run_record.status.value,
                    "failure_code": run_record.failure_code.value,
                },
            )

            if run_record.status == RunStatus.SUCCEEDED and diagnostics is not None:
                update_state(workspace_root, CampaignStateValue.RESULTS_READY)
                update_state(workspace_root, CampaignStateValue.COMPLETED_PHASE1)
                _log.info("ARC run %s succeeded for campaign %s", run_record.run_id, campaign.campaign_id)
            else:
                update_state(
                    workspace_root,
                    CampaignStateValue.FAILED,
                    notes=run_record.error_message,
                )
                _log.warning("ARC run %s failed: %s", run_record.run_id, run_record.failure_code.value)

            return run_record, diagnostics
        except Exception as e:
            _handle_unexpected_failure(
                workspace_root,
                campaign,
                action,
                adapter,
                started,
                run_record,
                e,
                tool_name=_arc_tool_name,
                provenance_kind="arc_run",
                finished_event="arc_run_finished",
                tool_label="ARC",
            )
            raise
