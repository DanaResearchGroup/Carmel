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

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from carmel.logger import get_logger
from carmel.schemas.campaign import Campaign
from carmel.schemas.diagnostics import DiagnosticsV1
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.schemas.state import CampaignStateValue
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_event
from carmel.services.drawing import SELECTION_SVG_FILENAMES, write_selection_svgs
from carmel.services.provenance import record
from carmel.services.state_machine import can_transition, load_state, update_state

DIAGNOSTICS_FILE_NAME = "diagnostics.json"
RUNS_DIR_NAME = "runs"
MODELS_DIR_NAME = "models"

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
    """Load persisted diagnostics, if present."""
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


def _default_adapter() -> T3AdapterProtocol:
    """Return the production T3 adapter (lazy import to avoid cycles)."""
    from carmel.adapters.t3 import T3Adapter

    return T3Adapter()


def _t3_tool_name() -> str:
    """Return the canonical T3 tool name (lazy import to avoid cycles)."""
    from carmel.adapters.t3 import T3_TOOL_NAME

    return T3_TOOL_NAME


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
    """
    if adapter is None:
        adapter = _default_adapter()

    # Validate the transition BEFORE touching any artifacts. A campaign
    # that is not actually eligible for RUNNING_T3 (e.g. a double-submitted
    # request, a stale bookmark) must raise here with the workspace
    # untouched — clearing stale diagnostics only after this succeeds.
    update_state(workspace_root, CampaignStateValue.RUNNING_T3, notes=f"action={action.action_id}")
    started = datetime.now(UTC)

    run_record: RunRecord | None = None
    try:
        # Now that the campaign has genuinely entered RUNNING_T3, every
        # subsequent step is inside this protected region: any failure
        # from here on must still be able to drive the campaign to FAILED
        # rather than leaving it wedged in RUNNING_T3.
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
        _handle_unexpected_failure(workspace_root, campaign, action, adapter, started, run_record, e)
        raise


def _handle_unexpected_failure(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    adapter: T3AdapterProtocol,
    started: datetime,
    run_record: RunRecord | None,
    error: Exception,
) -> None:
    """Best-effort cleanup after an unexpected exception in ``execute_t3_action``.

    This function must never raise anything: the caller's ``raise`` must
    re-surface ``error`` unmodified, not an ``InvalidTransitionError`` or an
    exception from this function's own persistence attempts. If the
    adapter already returned a real ``run_record`` before the failure, it
    is reused (and marked failed) so the persisted run_id matches the
    actual run instead of orphaning it behind a fabricated one.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign being executed.
        action: The planned T3 action.
        adapter: The adapter that was invoked for this run (used to derive
            ``submission_mode`` when no real run record exists).
        started: When the run started.
        run_record: The real RunRecord the adapter returned, if any.
        error: The original exception that triggered this handler.
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
            tool_name=_t3_tool_name(),
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
            "t3_run",
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
                "event": "t3_run_finished",
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

    _log.error("T3 run raised an unexpected exception for campaign %s: %s", campaign.campaign_id, error)
