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

from carmel.logger import get_logger
from carmel.schemas.campaign import Campaign
from carmel.schemas.diagnostics import DiagnosticsV1
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import RunRecord, RunStatus
from carmel.schemas.state import CampaignStateValue
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_event
from carmel.services.drawing import write_selection_svgs
from carmel.services.provenance import record
from carmel.services.state_machine import update_state

DIAGNOSTICS_FILE_NAME = "diagnostics.json"
RUNS_DIR_NAME = "runs"

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


def _default_adapter() -> T3AdapterProtocol:
    """Return the production T3 adapter (lazy import to avoid cycles)."""
    from carmel.adapters.t3 import T3Adapter

    return T3Adapter()


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

    update_state(workspace_root, CampaignStateValue.RUNNING_T3, notes=f"action={action.action_id}")
    started = datetime.now(UTC)
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
        save_diagnostics(workspace_root, diagnostics)
        write_selection_svgs(
            workspace_root / "models",
            diagnostics.species_to_compute,
            diagnostics.reactions_to_compute,
            diagnostics.pdep_networks_to_compute,
        )
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
