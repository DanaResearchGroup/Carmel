# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Cumulative CPU-hour spend, read back from the workspace's run records.

``Budgets.cpu_hours`` is a *campaign-total* budget, but until this module
existed nothing ever aggregated what a campaign had already spent:
every authorization compared an action against the full declared budget,
so ten 4-hour actions sailed through a 20-hour budget one by one. This
module is the spend ledger's read side — there is no separate ledger to
keep consistent, because the run records under ``runs/`` and the
in-flight ``active_run.json`` reservation *are* the ledger.

The reader **fails closed**: no unreadable, corrupt, or implausible
record may ever *increase* the remaining budget. A corrupt
``actual_cpu_hours`` falls back to the record's ``estimated_cpu_hours``
(which the schema guarantees is >= 0); a wholly unparseable file is
skipped with a warning (it cannot be costed, but it never credits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from carmel.logger import get_logger
from carmel.schemas.run import FailureCode, RunRecord, RunStatus
from carmel.services.artifacts import read_json
from carmel.services.recovery import load_active_run

RUNS_DIR_NAME = "runs"

_log = get_logger("services.spend")

PRE_LAUNCH_FAILURE_CODES = frozenset({FailureCode.INPUT_BUILD_ERROR, FailureCode.TOOL_NOT_FOUND})
"""Failure codes proving the tool never launched, so nothing was spent.

Every other failed run without a usable ``actual_cpu_hours`` — including
``ABANDONED``, whose whole point is that nobody observed the ending — is
charged its estimate, because the tool plausibly ran."""


@dataclass(frozen=True)
class Spend:
    """What a campaign has already spent and what is reserved in flight."""

    consumed_cpu_hours: float
    """CPU hours charged by finished run records under ``runs/``."""

    reserved_cpu_hours: float
    """The in-flight reservation from ``active_run.json``, if any."""

    def remaining(self, budget_cpu_hours: float) -> float:
        """Return the budget left after consumed and reserved spend."""
        return budget_cpu_hours - self.consumed_cpu_hours - self.reserved_cpu_hours


def _record_cost(record: RunRecord) -> float:
    """Charge one run record, failing closed on implausible data.

    * A present, finite, non-negative ``actual_cpu_hours`` is the truth.
    * A present but negative or non-finite actual is corrupt: it must
      never credit budget, so the record's estimate (>= 0 by schema) is
      charged instead.
    * No actual at all: a failed run that provably never launched costs
      zero; anything else — succeeded, abandoned, failed mid-run — is
      charged its estimate.
    """
    actual = record.actual_cpu_hours
    if actual is not None:
        if math.isfinite(actual) and actual >= 0:
            return actual
        _log.warning(
            "Run record %s has implausible actual_cpu_hours=%r; charging its estimate (%.3f) instead",
            record.run_id,
            actual,
            record.estimated_cpu_hours,
        )
        return record.estimated_cpu_hours
    if record.status == RunStatus.FAILED and record.failure_code in PRE_LAUNCH_FAILURE_CODES:
        return 0.0
    return record.estimated_cpu_hours


def compute_spend(workspace_root: Path) -> Spend:
    """Aggregate a campaign's CPU-hour spend from its workspace.

    Reads every run record under ``runs/`` — deliberately **without**
    deduplicating by ``action_id``, because retries are legitimately
    separate attempts that each spent compute — plus the in-flight
    reservation carried by ``active_run.json``.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        The campaign's :class:`Spend`.
    """
    consumed = 0.0
    runs_dir = workspace_root / RUNS_DIR_NAME
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                record = RunRecord.model_validate(read_json(path))
            except (OSError, ValueError) as e:
                # Unreadable records cannot be costed, but skipping one never
                # *credits* budget — consumed only ever grows.
                _log.warning("Skipping unreadable run record %s while computing spend: %s", path, e)
                continue
            consumed += _record_cost(record)

    active = load_active_run(workspace_root)
    reserved = active.estimated_cpu_hours if active is not None else 0.0
    return Spend(consumed_cpu_hours=consumed, reserved_cpu_hours=reserved)
