# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Real cumulative-budget enforcement: spend reader, gate wiring, launch re-check.

Covers the M9 guarantee end to end: ``Budgets.cpu_hours`` is a campaign
*total*, so spend already consumed (run records) and reserved (in-flight
``active_run.json``) must reduce what later actions may auto-approve, and
the launch paths must re-check the live gate so a stale plan-time
approval cannot spend budget that has since run out.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from carmel.schemas.approval import ApprovalPolicy, ApprovalRequirement, ApprovalStatus
from carmel.schemas.campaign import (
    Budgets,
    CampaignInput,
    InitialMixture,
    MixtureComponent,
    ReactorSystem,
    ReactorType,
    TargetObservable,
)
from carmel.schemas.diagnostics import DiagnosticsV1
from carmel.schemas.run import ActiveRun, FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.schemas.state import CampaignStateValue
from carmel.services.approvals import record_decision
from carmel.services.artifacts import write_json
from carmel.services.authorization import BudgetExceededError
from carmel.services.campaigns import create_campaign, load_campaign
from carmel.services.execution import (
    RUNS_DIR_NAME,
    abandon_t3_run,
    execute_arc_action,
    execute_t3_action,
    save_run_record,
    start_arc_action,
    start_t3_action,
)
from carmel.services.planner import generate_arc_plan, generate_initial_plan, load_plan, save_plan
from carmel.services.recovery import active_run_path, load_active_run, supervise_run
from carmel.services.spend import _recover_estimate, compute_spend
from carmel.services.state_machine import load_state, update_state

# ----------------------- fixtures & helpers ---------------------


def _make_input(name: str = "budget", cpu_hours: float = 20.0) -> CampaignInput:
    return CampaignInput(
        workspace_name=name,
        initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0, smiles="[O][O]")]),
        target_observables=[TargetObservable(name="ignition_delay")],
        target_reactor_systems=[
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, 5.0),
            )
        ],
        budgets=Budgets(cpu_hours=cpu_hours, experiment_budget=0.0),
    )


def _run_record(
    *,
    status: RunStatus = RunStatus.SUCCEEDED,
    failure_code: FailureCode = FailureCode.NONE,
    actual: float | None = None,
    estimated: float = 1.0,
    action_id: str = "a1",
) -> RunRecord:
    return RunRecord(
        run_id=str(uuid4()),
        action_id=action_id,
        tool_name="t3",
        status=status,
        failure_code=failure_code,
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        estimated_cpu_hours=estimated,
        actual_cpu_hours=actual,
        submission_mode=SubmissionMode.SUBPROCESS,
    )


def _consume(ws: Path, cpu_hours: float) -> None:
    """Record a finished run that consumed *cpu_hours* of the budget."""
    save_run_record(ws, _run_record(actual=cpu_hours, estimated=cpu_hours))


def _diagnostics(campaign_id: str, run_id: str) -> DiagnosticsV1:
    return DiagnosticsV1(campaign_id=campaign_id, run_id=run_id, generated_at=datetime.now(UTC))


class _SuccessAdapter:
    """Minimal adapter double: succeeds instantly with a tiny actual cost."""

    submission_mode = SubmissionMode.SUBPROCESS

    def run(
        self,
        workspace_root: Path,
        campaign: Any,
        action: Any,
        on_process_start: Any = None,
    ) -> tuple[RunRecord, DiagnosticsV1]:
        record = _run_record(actual=0.01, estimated=action.estimated_cpu_hours, action_id=action.action_id)
        return record, _diagnostics(campaign.campaign_id, record.run_id)


class _LaunchedFailureAdapter:
    """Fails after launching (SUBPROCESS_ERROR, no actual): the estimate is charged."""

    submission_mode = SubmissionMode.SUBPROCESS

    def run(
        self,
        workspace_root: Path,
        campaign: Any,
        action: Any,
        on_process_start: Any = None,
    ) -> tuple[RunRecord, None]:
        record = _run_record(
            status=RunStatus.FAILED,
            failure_code=FailureCode.SUBPROCESS_ERROR,
            estimated=action.estimated_cpu_hours,
            action_id=action.action_id,
        )
        return record, None


def _approved_t3_workspace(tmp_path: Path, cpu_hours: float = 20.0) -> Path:
    """A campaign in APPROVED_FOR_EXECUTION with a saved single-action T3 plan."""
    ws = tmp_path / "ws"
    campaign = create_campaign(ws, _make_input(cpu_hours=cpu_hours))
    plan = generate_initial_plan(campaign, ApprovalPolicy(), ws)
    save_plan(ws, plan)
    for target in [
        CampaignStateValue.VALIDATED,
        CampaignStateValue.READY_FOR_PLANNING,
        CampaignStateValue.PLAN_PENDING_APPROVAL,
        CampaignStateValue.APPROVED_FOR_EXECUTION,
    ]:
        update_state(ws, target)
    return ws


def _approved_arc_workspace(tmp_path: Path, cpu_hours: float = 20.0) -> Path:
    """A campaign in APPROVED_FOR_EXECUTION with a saved single-action ARC plan."""
    ws = tmp_path / "ws"
    campaign = create_campaign(ws, _make_input(name="arc-budget", cpu_hours=cpu_hours))
    plan = generate_arc_plan(campaign, workspace_root=ws)
    save_plan(ws, plan)
    for target in [
        CampaignStateValue.VALIDATED,
        CampaignStateValue.READY_FOR_PLANNING,
        CampaignStateValue.PLAN_PENDING_APPROVAL,
        CampaignStateValue.APPROVED_FOR_EXECUTION,
    ]:
        update_state(ws, target)
    return ws


# ----------------------- compute_spend --------------------------


class TestComputeSpend:
    def test_empty_workspace_spends_nothing(self, tmp_path: Path) -> None:
        spend = compute_spend(tmp_path)
        assert spend.consumed_cpu_hours == 0.0
        assert spend.reserved_cpu_hours == 0.0
        assert spend.remaining(20.0) == 20.0

    def test_consumed_from_actual(self, tmp_path: Path) -> None:
        save_run_record(tmp_path, _run_record(actual=2.5, estimated=4.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 2.5

    def test_succeeded_without_actual_counts_estimate(self, tmp_path: Path) -> None:
        save_run_record(tmp_path, _run_record(actual=None, estimated=4.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 4.0

    @pytest.mark.parametrize("code", [FailureCode.SUBPROCESS_ERROR, FailureCode.TIMEOUT, FailureCode.ABANDONED])
    def test_launched_failure_without_actual_counts_estimate(self, tmp_path: Path, code: FailureCode) -> None:
        save_run_record(tmp_path, _run_record(status=RunStatus.FAILED, failure_code=code, estimated=3.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 3.0

    @pytest.mark.parametrize("code", [FailureCode.INPUT_BUILD_ERROR, FailureCode.TOOL_NOT_FOUND])
    def test_pre_launch_failure_counts_zero(self, tmp_path: Path, code: FailureCode) -> None:
        save_run_record(tmp_path, _run_record(status=RunStatus.FAILED, failure_code=code, estimated=3.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 0.0

    def test_negative_actual_never_credits_budget(self, tmp_path: Path) -> None:
        # A corrupt negative actual must not reduce consumed spend below the
        # record's estimate, let alone below zero.
        save_run_record(tmp_path, _run_record(actual=-50.0, estimated=2.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 2.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_actual_falls_back_to_estimate(self, tmp_path: Path, bad: float) -> None:
        save_run_record(tmp_path, _run_record(actual=bad, estimated=2.0))
        assert compute_spend(tmp_path).consumed_cpu_hours == 2.0

    def test_unparseable_record_is_skipped_but_never_credits(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / RUNS_DIR_NAME
        runs_dir.mkdir()
        (runs_dir / "corrupt.json").write_text("{not json", encoding="utf-8")
        (runs_dir / "wrong-shape.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
        save_run_record(tmp_path, _run_record(actual=1.5))
        assert compute_spend(tmp_path).consumed_cpu_hours == 1.5

    def test_malformed_actual_with_valid_estimate_charges_the_estimate(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A record whose JSON is valid but fails schema validation must still
        charge its estimate — not zero, which would restore budget really spent."""
        path = save_run_record(tmp_path, _run_record(actual=1.0, estimated=5.0))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["actual_cpu_hours"] = "oops"
        path.write_text(json.dumps(data), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            spend = compute_spend(tmp_path)
        assert spend.consumed_cpu_hours == 5.0
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_fully_corrupt_json_is_skipped_with_error_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        runs_dir = tmp_path / RUNS_DIR_NAME
        runs_dir.mkdir()
        (runs_dir / "corrupt.json").write_text("{not json", encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            spend = compute_spend(tmp_path)
        assert spend.consumed_cpu_hours == 0.0
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_wrong_shape_with_no_estimate_is_skipped_with_error_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Valid JSON, fails validation, and no recoverable estimate at all."""
        runs_dir = tmp_path / RUNS_DIR_NAME
        runs_dir.mkdir()
        (runs_dir / "wrong-shape.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            spend = compute_spend(tmp_path)
        assert spend.consumed_cpu_hours == 0.0
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_malformed_actual_with_implausible_estimate_is_skipped_with_error_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A record that fails schema validation *and* whose raw
        ``estimated_cpu_hours`` is itself implausible (negative) has nothing
        recoverable to charge, so it is skipped rather than crediting
        budget."""
        path = save_run_record(tmp_path, _run_record(actual=1.0, estimated=5.0))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["actual_cpu_hours"] = "oops"
        data["estimated_cpu_hours"] = -5.0
        path.write_text(json.dumps(data), encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            spend = compute_spend(tmp_path)
        assert spend.consumed_cpu_hours == 0.0
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_retries_are_not_deduplicated_by_action_id(self, tmp_path: Path) -> None:
        # Two attempts of the same action each spent compute; both count.
        for _ in range(2):
            save_run_record(
                tmp_path,
                _run_record(
                    status=RunStatus.FAILED, failure_code=FailureCode.SUBPROCESS_ERROR, estimated=3.0, action_id="a1"
                ),
            )
        assert compute_spend(tmp_path).consumed_cpu_hours == 6.0

    def test_in_flight_reservation_is_counted(self, tmp_path: Path) -> None:
        write_json(
            active_run_path(tmp_path),
            ActiveRun(action_id="a1", started_at=datetime.now(UTC), estimated_cpu_hours=2.5, supervisor_pid=1234),
        )
        spend = compute_spend(tmp_path)
        assert spend.reserved_cpu_hours == 2.5
        assert spend.remaining(20.0) == 17.5

    def test_legacy_active_run_without_estimate_still_loads(self, tmp_path: Path) -> None:
        # Records written before ActiveRun.estimated_cpu_hours existed must
        # keep loading (extra="forbid" cuts the other way) and reserve zero.
        active_run_path(tmp_path).write_text(
            json.dumps(
                {"action_id": "a1", "started_at": datetime.now(UTC).isoformat(), "supervisor_pid": 1234},
            ),
            encoding="utf-8",
        )
        active = load_active_run(tmp_path)
        assert active is not None
        assert compute_spend(tmp_path).reserved_cpu_hours == 0.0

    def test_supervision_reserves_the_action_estimate(self, tmp_path: Path) -> None:
        # The real writer (start_supervision via supervise_run) must thread
        # the estimate into the reservation, and closing must release it.
        with supervise_run(tmp_path, "act-1", estimated_cpu_hours=3.0):
            assert compute_spend(tmp_path).reserved_cpu_hours == 3.0
        assert compute_spend(tmp_path).reserved_cpu_hours == 0.0


class TestRecoverEstimate:
    """Direct unit tests for ``_recover_estimate``'s non-dict-input branch.

    ``compute_spend`` can never reach this branch through a real
    ``runs/*.json`` file: ``read_json`` already raises ``ValueError`` (and
    gets skipped by the outer read failure handler) for any top-level JSON
    that isn't an object, before ``_recover_estimate`` is ever called. This
    exercises the defensive check directly, matching the "small pure
    helper, tested directly" style used for ARC's ``_requested_labels`` and
    friends.
    """

    @pytest.mark.parametrize("raw", [[], "x", 1, None])
    def test_non_dict_raw_returns_none(self, raw: Any) -> None:
        assert _recover_estimate(raw) is None


# ----------------------- cumulative escalation ------------------


class TestCumulativeEscalation:
    def test_t3_action_that_crosses_the_budget_escalates(self, tmp_path: Path) -> None:
        """Auto-approvable T3 actions escalate once their sum crosses the budget."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input(cpu_hours=7.0))  # each handshake estimates 3.0

        first = generate_initial_plan(campaign, ApprovalPolicy(), ws)
        assert first.requires_approval is False
        _consume(ws, 3.0)

        second = generate_initial_plan(campaign, ApprovalPolicy(), ws)
        assert second.requires_approval is False
        _consume(ws, 3.0)

        # remaining = 7 - 6 = 1.0 < 3.0: the crossing action must escalate.
        third = generate_initial_plan(campaign, ApprovalPolicy(), ws)
        assert third.requires_approval is True
        assert third.actions[0].approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL

    def test_arc_action_that_crosses_the_budget_escalates(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input(name="arc-cum", cpu_hours=3.5))  # each ARC job estimates 1.0

        for _ in range(3):
            plan = generate_arc_plan(campaign, workspace_root=ws)
            assert plan.requires_approval is False
            _consume(ws, 1.0)

        # remaining = 3.5 - 3 = 0.5 < 1.0: the crossing action must escalate.
        plan = generate_arc_plan(campaign, workspace_root=ws)
        assert plan.requires_approval is True
        assert plan.actions[0].approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert "remaining budget" in plan.rationale

    def test_in_flight_reservation_reduces_planning_budget(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input(name="arc-res", cpu_hours=20.0))
        _consume(ws, 10.0)
        write_json(
            active_run_path(ws),
            ActiveRun(action_id="in-flight", started_at=datetime.now(UTC), estimated_cpu_hours=9.5, supervisor_pid=1),
        )
        # remaining = 20 - 10 - 9.5 = 0.5 < 1.0
        plan = generate_arc_plan(campaign, workspace_root=ws)
        assert plan.requires_approval is True

    def test_without_workspace_the_full_budget_applies(self, tmp_path: Path) -> None:
        # Pure unit construction: nothing can have been spent yet.
        campaign = create_campaign(tmp_path / "ws", _make_input(cpu_hours=7.0))
        plan = generate_initial_plan(campaign, ApprovalPolicy())
        assert plan.requires_approval is False

    def test_arc_policy_threshold_now_binds(self, tmp_path: Path) -> None:
        """auto_approve_arc_under_cpu_hours is no longer dead for ARC plans."""
        campaign = create_campaign(tmp_path / "ws", _make_input(name="arc-pol"))
        plan = generate_arc_plan(campaign, policy=ApprovalPolicy(auto_approve_arc_under_cpu_hours=0.5))
        assert plan.requires_approval is True
        assert plan.actions[0].approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL


# ----------------------- launch-time re-check -------------------


class TestLaunchRecheck:
    def test_over_budget_with_only_auto_approval_is_refused_without_state_change(self, tmp_path: Path) -> None:
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")
        _consume(ws, 19.0)  # remaining 1.0 < estimate 3.0

        with pytest.raises(BudgetExceededError):
            execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())

        # Nothing was taken: state untouched, no in-flight record, lock free.
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_active_run(ws) is None
        with supervise_run(ws, "prove-the-lock-is-free"):
            pass

    def test_over_budget_with_human_approval_launches(self, tmp_path: Path) -> None:
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.APPROVED, decided_by="alon")
        _consume(ws, 19.0)

        run_record, diagnostics = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert diagnostics is not None
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_rejection_after_approval_is_refused(self, tmp_path: Path) -> None:
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.APPROVED, decided_by="alon")
        record_decision(ws, action.action_id, ApprovalStatus.REJECTED, decided_by="alon")
        _consume(ws, 19.0)

        with pytest.raises(BudgetExceededError):
            execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())

    def test_rejected_auto_approvable_action_is_refused_at_service_boundary(self, tmp_path: Path) -> None:
        """A human REJECTED decision overrides the live gate's auto-approval.

        The action here is well within budget, so the live gate alone would
        auto-approve it; a direct service caller (bypassing the UI's own
        decision-log check) must still be refused because a human rejected
        this action, and nothing about the budget re-check should override
        that rejection.
        """
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.REJECTED, decided_by="alon")

        with pytest.raises(BudgetExceededError):
            execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())

        # Nothing was taken: state untouched, no in-flight record, lock free.
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_active_run(ws) is None
        with supervise_run(ws, "prove-the-lock-is-free"):
            pass

    def test_rejected_then_approved_still_launches(self, tmp_path: Path) -> None:
        """A later human APPROVED decision supersedes an earlier REJECTED one."""
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.REJECTED, decided_by="alon")
        record_decision(ws, action.action_id, ApprovalStatus.APPROVED, decided_by="alon")

        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_within_budget_auto_launch_needs_no_recorded_decision(self, tmp_path: Path) -> None:
        """Direct service callers with a live-auto action are not interrupted.

        This is the Mockter end-to-end shape: a small auto-approved action,
        tiny estimate, well under budget, launched without any decision-log
        ceremony.
        """
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_retry_of_human_approved_action_launches_even_over_budget(self, tmp_path: Path) -> None:
        """The failed attempt consumed the budget; the human's approval survives the retry."""
        ws = _approved_t3_workspace(tmp_path, cpu_hours=4.0)  # estimate 3.0, budget 4.0
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.APPROVED, decided_by="alon")

        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_LaunchedFailureAdapter())
        assert run_record.status == RunStatus.FAILED
        assert load_state(ws).state == CampaignStateValue.FAILED
        # The failed attempt is charged its estimate: remaining = 4 - 3 = 1 < 3.
        assert compute_spend(ws).remaining(4.0) == 1.0

        # Retry re-arms the same approved action (the UI retry edge).
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="recovered: retry")
        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_retry_with_only_stale_auto_approval_is_refused(self, tmp_path: Path) -> None:
        """An auto-approval is valid only while the live gate still auto-approves."""
        ws = _approved_t3_workspace(tmp_path, cpu_hours=4.0)
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")

        execute_t3_action(ws, load_campaign(ws), action, adapter=_LaunchedFailureAdapter())
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="recovered: retry")

        with pytest.raises(BudgetExceededError):
            execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_start_t3_action_is_refused_synchronously(self, tmp_path: Path) -> None:
        ws = _approved_t3_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        _consume(ws, 19.0)

        with pytest.raises(BudgetExceededError):
            start_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_active_run(ws) is None

    def test_start_arc_action_is_refused_synchronously(self, tmp_path: Path) -> None:
        """The backgrounded ARC entry point is guarded exactly like T3's.

        The refusal must land in the calling thread — before supervision,
        before RUNNING_ARC — so the UI can map it to a 409 and nothing is
        left wedged or reserved.
        """
        ws = _approved_arc_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        _consume(ws, 19.5)  # remaining 0.5 < estimate 1.0

        with pytest.raises(BudgetExceededError):
            start_arc_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_active_run(ws) is None
        with supervise_run(ws, "prove-the-lock-is-free"):
            pass

    def test_arc_launch_is_gated_symmetrically(self, tmp_path: Path) -> None:
        ws = _approved_arc_workspace(tmp_path)
        action = load_plan(ws).actions[0]
        _consume(ws, 19.5)  # remaining 0.5 < estimate 1.0

        with pytest.raises(BudgetExceededError):
            execute_arc_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

        record_decision(ws, action.action_id, ApprovalStatus.APPROVED, decided_by="alon")
        run_record, diagnostics = execute_arc_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert diagnostics is not None
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_missing_policy_file_gates_under_the_default_policy(self, tmp_path: Path) -> None:
        """A bare workspace without a persisted policy is still gated."""
        ws = _approved_t3_workspace(tmp_path)
        (ws / "approval_policy.yaml").unlink()

        action = load_plan(ws).actions[0]
        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED

        # And the default policy really binds: over budget is still refused.
        ws2 = _approved_t3_workspace(tmp_path / "second")
        (ws2 / "approval_policy.yaml").unlink()
        _consume(ws2, 19.0)
        with pytest.raises(BudgetExceededError):
            execute_t3_action(ws2, load_campaign(ws2), load_plan(ws2).actions[0], adapter=_SuccessAdapter())

    def test_abandon_is_never_budget_blocked(self, tmp_path: Path) -> None:
        """Recovery paths are not launches: abandoning works over budget."""
        ws = _approved_t3_workspace(tmp_path)
        _consume(ws, 30.0)  # far over the 20.0 budget
        update_state(ws, CampaignStateValue.RUNNING_T3)

        state, report = abandon_t3_run(ws, load_campaign(ws))
        assert state.state == CampaignStateValue.FAILED
        assert report.is_finished

    def test_current_action_is_not_counted_against_itself(self, tmp_path: Path) -> None:
        """The re-check runs before supervision, so exact-fit budgets launch.

        If the launch reserved the action before re-checking, remaining would
        read ``budget - estimate`` and every exact-fit action would refuse
        itself.
        """
        ws = _approved_t3_workspace(tmp_path, cpu_hours=3.0)  # estimate == budget == 3.0
        action = load_plan(ws).actions[0]
        run_record, _ = execute_t3_action(ws, load_campaign(ws), action, adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED


# ----------------------- abandoned runs stay charged -------------


class TestAbandonedRunCharging:
    def test_abandoned_record_inherits_the_reservation_estimate(self, tmp_path: Path) -> None:
        """An abandoned run keeps charging its estimate after the reservation clears."""
        ws = _approved_t3_workspace(tmp_path)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        write_json(
            active_run_path(ws),
            ActiveRun(action_id="a1", started_at=datetime.now(UTC), estimated_cpu_hours=3.0, supervisor_pid=1),
        )
        assert compute_spend(ws).reserved_cpu_hours == 3.0

        state, _report = abandon_t3_run(ws, load_campaign(ws))
        assert state.state == CampaignStateValue.FAILED
        spend = compute_spend(ws)
        assert spend.reserved_cpu_hours == 0.0
        assert spend.consumed_cpu_hours == 3.0
