"""Tests for Phase 1 service modules."""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from carmel.adapters.t3 import T3_TOOL_NAME
from carmel.schemas import (
    ActionKind,
    ApprovalPolicy,
    ApprovalRequirement,
    ApprovalStatus,
    Budgets,
    CampaignInput,
    CampaignState,
    CampaignStateValue,
    DiagnosticsV1,
    FailureCode,
    InitialMixture,
    MixtureComponent,
    PDepNetworkSelection,
    PlannedAction,
    ReactionSelection,
    ReactorSystem,
    ReactorType,
    RunRecord,
    RunStatus,
    SpeciesSelection,
    SubmissionMode,
    TargetObservable,
)
from carmel.schemas.campaign import Campaign
from carmel.services.approvals import (
    evaluate_action,
    load_policy,
    record_decision,
    save_policy,
)
from carmel.services.artifacts import (
    read_json,
    read_yaml,
    write_json,
    write_text,
    write_yaml,
)
from carmel.services.campaigns import (
    create_campaign,
    find_campaign_workspace,
    list_campaigns,
    load_campaign,
)
from carmel.services.decision_log import append_event, read_events
from carmel.services.drawing import (
    render_pdep_networks_svg,
    render_reactions_svg,
    render_species_svg,
    write_selection_svgs,
)
from carmel.services.execution import (
    ARC_DIAGNOSTICS_FILE_NAME,
    DIAGNOSTICS_FILE_NAME,
    execute_arc_action,
    execute_t3_action,
    load_arc_diagnostics,
    load_diagnostics,
    save_diagnostics,
    save_run_record,
    start_t3_action,
)
from carmel.services.intake import StubIntakeParser, write_intake_review
from carmel.services.planner import (
    estimate_t3_cpu_hours,
    generate_arc_plan,
    generate_initial_plan,
    load_plan,
    plan_and_save,
    render_plan_markdown,
    save_plan,
)
from carmel.services.provenance import record
from carmel.services.state_machine import (
    InvalidTransitionError,
    can_transition,
    load_state,
    update_state,
)


def _make_input(name: str = "test") -> CampaignInput:
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
        budgets=Budgets(cpu_hours=20.0, experiment_budget=0.0),
    )


def _make_action(cpu_hours: float = 5.0, kind: ActionKind = ActionKind.T3_RUN) -> PlannedAction:
    return PlannedAction(
        action_id="a1",
        kind=kind,
        description="test action",
        estimated_cpu_hours=cpu_hours,
        rationale="testing",
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
    )


# ----------------------- artifacts ------------------------------


class TestArtifacts:
    def test_write_yaml_atomic(self, tmp_path: Path) -> None:
        path = tmp_path / "out.yaml"
        write_yaml(path, {"a": 1, "b": "x"})
        assert path.exists()
        assert read_yaml(path) == {"a": 1, "b": "x"}

    def test_write_json_atomic(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json(path, {"a": 1})
        assert read_json(path) == {"a": 1}

    def test_write_text(self, tmp_path: Path) -> None:
        path = tmp_path / "out.md"
        write_text(path, "hello")
        assert path.read_text() == "hello"

    def test_read_missing_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_yaml(tmp_path / "missing.yaml")

    def test_read_missing_json_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "missing.json")

    def test_read_yaml_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            read_yaml(path)

    def test_read_json_non_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="object"):
            read_json(path)

    def test_write_pydantic_model(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.yaml"
        write_yaml(path, ApprovalPolicy())
        loaded = ApprovalPolicy.model_validate(read_yaml(path))
        assert loaded.auto_approve_t3_under_cpu_hours == 10.0

    def test_concurrent_write_json_never_corrupts_file(self, tmp_path: Path) -> None:
        """N threads hammering the same path must never race on a shared temp file.

        A pre-fix implementation shared one ``path + ".tmp"`` name across all
        writers, so concurrent writers could interleave into that one temp
        file and rename a half-and-half document into place, or lose a race
        with another writer's rename and raise ``FileNotFoundError``. Each
        writer here also re-reads the file after writing: any transient
        corruption or missing-file window must show up as a failure.

        50 iterations per thread (with an 8-way barrier) was chosen because
        the reviewer who found this bug observed failures in 184/200 runs of
        a similarly sized loop pre-fix — decisive against the bug without
        making the test slow or flaky in CI.
        """
        path = tmp_path / "shared.json"
        write_json(path, {"n": -1})
        n_threads = 8
        n_iterations = 50
        barrier = threading.Barrier(n_threads)
        errors: list[BaseException] = []

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(n_iterations):
                try:
                    write_json(path, {"thread": thread_id, "i": i})
                    data = json.loads(path.read_text(encoding="utf-8"))
                    assert "thread" in data and "i" in data
                except BaseException as e:  # noqa: BLE001 - captured for the assertion below
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        final = json.loads(path.read_text(encoding="utf-8"))
        assert "thread" in final and "i" in final
        assert not list(tmp_path.glob("*.tmp"))

    def test_write_json_cleans_up_temp_file_on_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the rename step fails, the orphaned temp file must be removed and the error re-raised."""
        path = tmp_path / "out.json"
        real_replace = os.replace

        def failing_replace(src: str, dst: str) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated rename failure"):
            write_json(path, {"a": 1})
        monkeypatch.setattr(os, "replace", real_replace)

        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))


# ----------------------- decision log ---------------------------


class TestDecisionLog:
    def test_append_creates_file(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_event(log, {"event": "x"})
        assert log.exists()

    def test_append_is_append_only(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_event(log, {"event": "first"})
        append_event(log, {"event": "second"})
        events = read_events(log)
        assert len(events) == 2
        assert events[0]["event"] == "first"
        assert events[1]["event"] == "second"

    def test_timestamp_added(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_event(log, {"event": "x"})
        events = read_events(log)
        assert "timestamp" in events[0]

    def test_read_missing_returns_empty(self, tmp_path: Path) -> None:
        assert read_events(tmp_path / "missing.jsonl") == []

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        log.write_text('{"event":"a"}\n\n{"event":"b"}\n')
        events = read_events(log)
        assert len(events) == 2

    def test_read_skips_malformed_line(self, tmp_path: Path) -> None:
        """A truncated/garbage line must not brick every future read of the log."""
        log = tmp_path / "decision_log.jsonl"
        append_event(log, {"event": "first"})
        with open(log, "a", encoding="utf-8") as f:
            f.write('{"event": "truncated", "notes": "oops\n')  # unterminated JSON
        append_event(log, {"event": "third"})
        events = read_events(log)
        assert [e["event"] for e in events] == ["first", "third"]

    def test_concurrent_append_produces_only_valid_json_lines(self, tmp_path: Path) -> None:
        """Concurrent appenders from multiple threads must never interleave partial lines.

        8 threads x 50 appends each (with a barrier to maximize contention)
        was chosen for the same reason as the artifacts concurrency test:
        enough iterations to be decisive against interleaving, few enough to
        stay fast and non-flaky in CI.
        """
        log = tmp_path / "decision_log.jsonl"
        n_threads = 8
        n_iterations = 50
        barrier = threading.Barrier(n_threads)
        errors: list[BaseException] = []

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(n_iterations):
                try:
                    append_event(log, {"event": f"t{thread_id}-{i}"})
                except BaseException as e:  # noqa: BLE001 - captured for the assertion below
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        lines = log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_threads * n_iterations
        for line in lines:
            json.loads(line)  # raises if any line is malformed


# ----------------------- state machine --------------------------


class TestStateMachine:
    def test_valid_transition(self) -> None:
        assert can_transition(CampaignStateValue.DRAFT, CampaignStateValue.VALIDATED)

    def test_invalid_transition(self) -> None:
        assert not can_transition(CampaignStateValue.DRAFT, CampaignStateValue.RUNNING_T3)

    def test_terminal_state_no_transitions(self) -> None:
        assert not can_transition(CampaignStateValue.COMPLETED_PHASE1, CampaignStateValue.DRAFT)
        assert not can_transition(CampaignStateValue.FAILED, CampaignStateValue.DRAFT)

    def test_update_state_persists(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        new = update_state(ws, CampaignStateValue.VALIDATED, notes="ok")
        assert new.state == CampaignStateValue.VALIDATED
        loaded = load_state(ws)
        assert loaded.state == CampaignStateValue.VALIDATED
        assert loaded.notes == "ok"

    def test_update_state_rejects_invalid(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        with pytest.raises(InvalidTransitionError):
            update_state(ws, CampaignStateValue.RUNNING_T3)

    def test_concurrent_update_state_exactly_one_thread_wins(self, tmp_path: Path) -> None:
        """Two overlapping ``/run``-style requests must not both succeed.

        Without the per-workspace advisory lock, two threads can both
        ``load_state`` before either writes, both pass ``assert_transition``
        against the same stale ``current_state``, and both persist a state
        — a lost-update race (e.g. two requests both entering RUNNING_T3).
        With the lock, the whole load-check-write cycle is serialized: the
        second thread must observe the first thread's write and raise
        ``InvalidTransitionError`` for the now-stale transition.
        """
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        update_state(ws, CampaignStateValue.VALIDATED)
        update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)

        barrier = threading.Barrier(2)
        results: list[CampaignState] = []
        errors: list[BaseException] = []

        def worker() -> None:
            barrier.wait()
            try:
                results.append(update_state(ws, CampaignStateValue.RUNNING_T3))
            except InvalidTransitionError as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], InvalidTransitionError)
        assert load_state(ws).state == CampaignStateValue.RUNNING_T3

    def test_blocked_cannot_be_unrejected_to_approved_for_execution(self) -> None:
        assert not can_transition(CampaignStateValue.BLOCKED, CampaignStateValue.APPROVED_FOR_EXECUTION)

    def test_blocked_can_only_transition_to_failed(self) -> None:
        for target in CampaignStateValue:
            expected = target == CampaignStateValue.FAILED
            assert can_transition(CampaignStateValue.BLOCKED, target) == expected

    def test_update_state_refuses_unrejecting_a_blocked_campaign(self, tmp_path: Path) -> None:
        """A rejected plan must be re-planned, never un-rejected into execution."""
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        update_state(ws, CampaignStateValue.VALIDATED)
        update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        update_state(ws, CampaignStateValue.BLOCKED, notes="user-rejected")
        with pytest.raises(InvalidTransitionError):
            update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)

    def test_failed_can_retry_to_approved_for_execution_only_from_running_t3(self) -> None:
        assert can_transition(
            CampaignStateValue.FAILED,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            failed_from=CampaignStateValue.RUNNING_T3,
        )
        assert not can_transition(CampaignStateValue.FAILED, CampaignStateValue.APPROVED_FOR_EXECUTION)

    def test_failed_cannot_retry_when_failed_before_execution_was_approved(self) -> None:
        for origin in [
            CampaignStateValue.DRAFT,
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.BLOCKED,
        ]:
            assert not can_transition(
                CampaignStateValue.FAILED,
                CampaignStateValue.APPROVED_FOR_EXECUTION,
                failed_from=origin,
            )

    def test_failed_cannot_transition_elsewhere(self) -> None:
        assert not can_transition(
            CampaignStateValue.FAILED, CampaignStateValue.RUNNING_T3, failed_from=CampaignStateValue.RUNNING_T3
        )
        assert not can_transition(
            CampaignStateValue.FAILED, CampaignStateValue.COMPLETED_PHASE1, failed_from=CampaignStateValue.RUNNING_T3
        )
        assert not can_transition(
            CampaignStateValue.FAILED, CampaignStateValue.DIAGNOSTICS_READY, failed_from=CampaignStateValue.RUNNING_T3
        )

    def test_update_state_refuses_retry_when_failed_before_approval(self, tmp_path: Path) -> None:
        """D8: a campaign that failed during validation — before any plan
        existed and before any human approved anything — must not be able
        to reach APPROVED_FOR_EXECUTION. That would bypass the planning and
        HITL approval gates entirely."""
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        update_state(ws, CampaignStateValue.VALIDATED)
        update_state(ws, CampaignStateValue.FAILED, notes="boom")
        with pytest.raises(InvalidTransitionError):
            update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")

    def test_update_state_retries_campaign_that_failed_from_running_t3(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
        ]:
            update_state(ws, target)
        update_state(ws, CampaignStateValue.FAILED, notes="boom")
        retried = update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")
        assert retried.state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert retried.failed_from is None

    def test_failed_from_cleared_once_campaign_leaves_failed(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
        ]:
            update_state(ws, target)
        failed = update_state(ws, CampaignStateValue.FAILED, notes="boom")
        assert failed.failed_from == CampaignStateValue.RUNNING_T3
        retried = update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")
        assert retried.failed_from is None

    def test_full_happy_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.COMPLETED_PHASE1,
        ]:
            update_state(ws, target)
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1


# ----------------------- approvals ------------------------------


class TestApprovals:
    def test_under_threshold_auto_approved(self) -> None:
        action = _make_action(cpu_hours=2.0)
        result = evaluate_action(action, ApprovalPolicy())
        assert result == ApprovalRequirement.AUTO_APPROVED

    def test_over_threshold_requires_approval(self) -> None:
        action = _make_action(cpu_hours=50.0)
        result = evaluate_action(action, ApprovalPolicy())
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_arc_threshold_separate(self) -> None:
        action = _make_action(cpu_hours=4.0, kind=ActionKind.ARC_RUN)
        result = evaluate_action(action, ApprovalPolicy())
        assert result == ApprovalRequirement.AUTO_APPROVED

    def test_experiment_requires_approval_by_default(self) -> None:
        action = _make_action(cpu_hours=0.0, kind=ActionKind.EXPERIMENT)
        result = evaluate_action(action, ApprovalPolicy())
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_arc_over_threshold_requires_approval(self) -> None:
        action = _make_action(cpu_hours=6.0, kind=ActionKind.ARC_RUN)
        result = evaluate_action(action, ApprovalPolicy())
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_arc_exactly_at_threshold_auto_approved(self) -> None:
        policy = ApprovalPolicy()
        action = _make_action(cpu_hours=policy.auto_approve_arc_under_cpu_hours, kind=ActionKind.ARC_RUN)
        assert evaluate_action(action, policy) == ApprovalRequirement.AUTO_APPROVED

    def test_experiment_auto_approved_when_policy_allows(self) -> None:
        action = _make_action(cpu_hours=0.0, kind=ActionKind.EXPERIMENT)
        policy = ApprovalPolicy(require_approval_for_experiments=False)
        assert evaluate_action(action, policy) == ApprovalRequirement.AUTO_APPROVED

    def test_literature_auto_approved_by_default(self) -> None:
        action = _make_action(cpu_hours=0.0, kind=ActionKind.LITERATURE_SEARCH)
        assert evaluate_action(action, ApprovalPolicy()) == ApprovalRequirement.AUTO_APPROVED

    def test_literature_requires_approval_when_policy_demands(self) -> None:
        action = _make_action(cpu_hours=0.0, kind=ActionKind.LITERATURE_SEARCH)
        policy = ApprovalPolicy(require_approval_for_literature=True)
        assert evaluate_action(action, policy) == ApprovalRequirement.REQUIRES_APPROVAL

    def test_action_over_declared_budget_requires_approval(self) -> None:
        """A T3_RUN estimate within the policy threshold but over the campaign's
        own declared cpu_hours budget must not be auto-approved — the campaign's
        Budgets.cpu_hours was previously never read anywhere."""
        budgets = Budgets(cpu_hours=1.0, experiment_budget=0.0)
        action = _make_action(cpu_hours=2.0)  # under policy threshold (10.0), over budget (1.0)
        result = evaluate_action(action, ApprovalPolicy(), budgets=budgets)
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_action_under_declared_budget_still_auto_approved(self) -> None:
        budgets = Budgets(cpu_hours=20.0, experiment_budget=0.0)
        action = _make_action(cpu_hours=2.0)
        result = evaluate_action(action, ApprovalPolicy(), budgets=budgets)
        assert result == ApprovalRequirement.AUTO_APPROVED

    def test_no_budgets_falls_back_to_policy_only(self) -> None:
        action = _make_action(cpu_hours=2.0)
        assert evaluate_action(action, ApprovalPolicy(), budgets=None) == ApprovalRequirement.AUTO_APPROVED

    def test_budget_violation_recorded_in_decision_log(self, tmp_path: Path) -> None:
        ws = tmp_path
        budgets = Budgets(cpu_hours=1.0, experiment_budget=0.0)
        action = _make_action(cpu_hours=2.0)
        result = evaluate_action(action, ApprovalPolicy(), budgets=budgets, workspace_root=ws)
        assert result == ApprovalRequirement.REQUIRES_APPROVAL
        events = read_events(ws / "decision_log.jsonl")
        assert len(events) == 1
        assert events[0]["event"] == "approval_decision"
        assert events[0]["action_id"] == "a1"
        assert events[0]["status"] == ApprovalStatus.PENDING.value
        assert "budget" in events[0]["rationale"]

    def test_arc_action_over_declared_budget_requires_approval(self) -> None:
        budgets = Budgets(cpu_hours=1.0, experiment_budget=0.0)
        action = _make_action(cpu_hours=2.0, kind=ActionKind.ARC_RUN)
        result = evaluate_action(action, ApprovalPolicy(), budgets=budgets)
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_unknown_action_kind_fails_safe(self) -> None:
        """An action kind the policy does not recognize must not be auto-approved.

        Guards the HITL gate against a future ActionKind being added without a
        corresponding policy branch.
        """
        unknown = SimpleNamespace(kind="some_future_kind", estimated_cpu_hours=0.0)
        result = evaluate_action(cast(PlannedAction, unknown), ApprovalPolicy())
        assert result == ApprovalRequirement.REQUIRES_APPROVAL

    def test_record_decision_appends_to_log(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        decision = record_decision(ws, "a1", ApprovalStatus.APPROVED, decided_by="user")
        assert decision.action_id == "a1"
        events = read_events(ws / "decision_log.jsonl")
        approval_events = [e for e in events if e.get("event") == "approval_decision"]
        assert len(approval_events) == 1

    def test_save_and_load_policy(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        save_policy(ws, ApprovalPolicy(auto_approve_t3_under_cpu_hours=99.0))
        loaded = load_policy(ws)
        assert loaded.auto_approve_t3_under_cpu_hours == 99.0


# ----------------------- campaigns ------------------------------


class TestCampaigns:
    def test_create_campaign(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input("ethanol"))
        assert (ws / "campaign.yaml").exists()
        assert (ws / "approval_policy.yaml").exists()
        assert (ws / "campaign_state.json").exists()
        assert (ws / "decision_log.jsonl").exists()
        assert campaign.input.workspace_name == "ethanol"

    def test_load_campaign_roundtrip(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        original = create_campaign(ws, _make_input("ethanol"))
        loaded = load_campaign(ws)
        assert loaded.campaign_id == original.campaign_id
        assert loaded.input.workspace_name == "ethanol"

    def test_list_campaigns(self, tmp_path: Path) -> None:
        for name in ["a", "b", "c"]:
            create_campaign(tmp_path / name, _make_input(name))
        campaigns = list_campaigns(tmp_path)
        assert len(campaigns) == 3

    def test_list_campaigns_empty(self, tmp_path: Path) -> None:
        assert list_campaigns(tmp_path) == []

    def test_list_campaigns_skips_invalid(self, tmp_path: Path) -> None:
        create_campaign(tmp_path / "good", _make_input("good"))
        (tmp_path / "junk").mkdir()
        (tmp_path / "junk" / "campaign.yaml").write_text("not: a campaign")
        campaigns = list_campaigns(tmp_path)
        assert len(campaigns) == 1
        assert campaigns[0].input.workspace_name == "good"

    def test_find_workspace(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input("test"))
        found = find_campaign_workspace(tmp_path, c.campaign_id)
        assert found is not None
        assert found.name == "ws"

    def test_list_campaigns_missing_root(self, tmp_path: Path) -> None:
        assert list_campaigns(tmp_path / "does-not-exist") == []

    def test_list_campaigns_ignores_loose_files(self, tmp_path: Path) -> None:
        create_campaign(tmp_path / "good", _make_input("good"))
        (tmp_path / "stray.txt").write_text("not a workspace")
        campaigns = list_campaigns(tmp_path)
        assert [c.input.workspace_name for c in campaigns] == ["good"]

    def test_list_campaigns_ignores_dirs_without_campaign_file(self, tmp_path: Path) -> None:
        create_campaign(tmp_path / "good", _make_input("good"))
        (tmp_path / "unrelated").mkdir()
        campaigns = list_campaigns(tmp_path)
        assert [c.input.workspace_name for c in campaigns] == ["good"]

    def test_find_workspace_missing(self, tmp_path: Path) -> None:
        assert find_campaign_workspace(tmp_path, "missing-id") is None

    def test_find_workspace_scans_past_non_matching(self, tmp_path: Path) -> None:
        create_campaign(tmp_path / "aaa", _make_input("aaa"))
        target = create_campaign(tmp_path / "zzz", _make_input("zzz"))
        found = find_campaign_workspace(tmp_path, target.campaign_id)
        assert found is not None
        assert found.name == "zzz"

    def test_find_workspace_ignores_untrusted_declared_workspace_root(self, tmp_path: Path) -> None:
        """D9: ``campaign.yaml``'s ``workspace_root`` is untrusted,
        user-editable data — and now load-bearing, because
        ``clear_stale_diagnostics_artifacts`` unlinks files under the path
        this function returns. A YAML file that lies about its own
        location must not steer callers elsewhere."""
        c = create_campaign(tmp_path / "real", _make_input("real"))
        raw = read_yaml(tmp_path / "real" / "campaign.yaml")
        raw["workspace_root"] = str(tmp_path / "elsewhere" / "bogus")
        write_yaml(tmp_path / "real" / "campaign.yaml", raw)

        found = find_campaign_workspace(tmp_path, c.campaign_id)

        assert found == tmp_path / "real"

    def test_find_workspace_missing_root_returns_none(self, tmp_path: Path) -> None:
        assert find_campaign_workspace(tmp_path / "does-not-exist", "some-id") is None

    def test_find_workspace_ignores_loose_files(self, tmp_path: Path) -> None:
        # Named to sort before "zzz_real" so the loop must genuinely walk
        # past (and `continue` on) this non-directory entry.
        (tmp_path / "aaa_stray.txt").write_text("not a workspace", encoding="utf-8")
        target = create_campaign(tmp_path / "zzz_real", _make_input("real"))
        found = find_campaign_workspace(tmp_path, target.campaign_id)
        assert found == tmp_path / "zzz_real"

    def test_find_workspace_ignores_dirs_without_campaign_file(self, tmp_path: Path) -> None:
        (tmp_path / "aaa_unrelated").mkdir()
        target = create_campaign(tmp_path / "zzz_real", _make_input("real"))
        found = find_campaign_workspace(tmp_path, target.campaign_id)
        assert found == tmp_path / "zzz_real"

    def test_find_workspace_skips_invalid_campaign_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "aaa_junk").mkdir()
        (tmp_path / "aaa_junk" / "campaign.yaml").write_text("not: a campaign", encoding="utf-8")
        target = create_campaign(tmp_path / "zzz_real", _make_input("real"))
        found = find_campaign_workspace(tmp_path, target.campaign_id)
        assert found == tmp_path / "zzz_real"


# ----------------------- planner --------------------------------


class TestPlanner:
    def test_estimate_cpu_hours(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input())
        est = estimate_t3_cpu_hours(c)
        assert est == 2 * 1 + 1  # 1 reactor, 1 observable

    def test_generate_initial_plan(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input())
        plan = generate_initial_plan(c, ApprovalPolicy())
        assert len(plan.actions) == 1
        assert plan.actions[0].kind == ActionKind.T3_RUN

    def test_plan_above_threshold_requires_approval(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input())
        plan = generate_initial_plan(c, ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.5))
        assert plan.requires_approval is True

    def test_plan_under_threshold_auto_approved(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input())
        plan = generate_initial_plan(c, ApprovalPolicy(auto_approve_t3_under_cpu_hours=100.0))
        assert plan.requires_approval is False

    def test_render_markdown(self, tmp_path: Path) -> None:
        c = create_campaign(tmp_path / "ws", _make_input())
        plan = generate_initial_plan(c, ApprovalPolicy())
        md = render_plan_markdown(plan)
        assert plan.plan_id in md
        assert "T3" in md.upper() or "t3" in md

    def test_save_and_load_plan(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        c = create_campaign(ws, _make_input())
        plan = generate_initial_plan(c, ApprovalPolicy())
        save_plan(ws, plan)
        assert (ws / "plan.json").exists()
        assert (ws / "plan.md").exists()
        loaded = load_plan(ws)
        assert loaded.plan_id == plan.plan_id

    def test_plan_and_save(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        c = create_campaign(ws, _make_input())
        plan = plan_and_save(ws, c)
        assert (ws / "plan.json").exists()
        assert plan.campaign_id == c.campaign_id

    def test_plan_and_save_enforces_the_declared_budget(self, tmp_path: Path) -> None:
        """A declared budget must gate approval through the real planning path.

        ``evaluate_action`` accepting a ``budgets`` argument proves nothing on
        its own — the defect was that no production call site ever passed one,
        so ``Budgets`` had no read site and an over-budget action was still
        auto-approved with no human in the loop. This asserts the wiring, not
        the helper.
        """
        ws = tmp_path / "ws"
        campaign_input = _make_input().model_copy(update={"budgets": Budgets(cpu_hours=0.5, experiment_budget=0.0)})
        c = create_campaign(ws, campaign_input)

        plan = plan_and_save(ws, c)

        action = plan.actions[0]
        assert action.estimated_cpu_hours > 0.5, "fixture must actually exceed the budget"
        assert action.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL
        rationales = [str(e.get("rationale", "")) for e in read_events(ws / "decision_log.jsonl")]
        assert any("budget exceeded" in r for r in rationales)

    def test_plan_and_save_auto_approves_within_budget(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        c = create_campaign(ws, _make_input())
        plan = plan_and_save(ws, c)
        assert plan.actions[0].approval_requirement == ApprovalRequirement.AUTO_APPROVED


# ----------------------- drawing --------------------------------


class TestDrawing:
    def test_render_species_empty(self) -> None:
        svg = render_species_svg([])
        assert svg.startswith("<svg")
        assert "no species" in svg

    def test_render_species_full(self) -> None:
        svg = render_species_svg([SpeciesSelection(label="OH", smiles="[OH]")])
        assert "OH" in svg
        assert "[OH]" in svg

    def test_render_reactions_empty(self) -> None:
        svg = render_reactions_svg([])
        assert "no reactions" in svg

    def test_render_reactions_full(self) -> None:
        svg = render_reactions_svg([ReactionSelection(label="r1", reactants=["A", "B"], products=["C"])])
        assert "A + B" in svg
        assert "→" in svg

    def test_render_pdep_empty(self) -> None:
        svg = render_pdep_networks_svg([])
        assert "no PDep" in svg

    def test_render_pdep_full(self) -> None:
        svg = render_pdep_networks_svg([PDepNetworkSelection(network_id="N1", species=["A", "B", "C"])])
        assert "N1" in svg

    def test_write_selection_svgs(self, tmp_path: Path) -> None:
        paths = write_selection_svgs(
            tmp_path,
            species=[SpeciesSelection(label="OH")],
            reactions=[ReactionSelection(label="r1", reactants=["A"], products=["B"])],
            networks=[PDepNetworkSelection(network_id="N1", species=["A"])],
        )
        for key in ("species", "reactions", "pdep_networks"):
            assert paths[key].exists()
            assert paths[key].read_text().startswith("<svg")

    def test_render_reactions_includes_reason(self) -> None:
        svg = render_reactions_svg(
            [ReactionSelection(label="r1", reactants=["A"], products=["B"], reason="iteration 2 · success=True")]
        )
        assert "iteration 2" in svg

    def test_render_reactions_truncates_long_reason(self) -> None:
        svg = render_reactions_svg([ReactionSelection(label="r1", reactants=["A"], products=["B"], reason="x" * 200)])
        assert "x" * 60 in svg
        assert "x" * 61 not in svg

    def test_render_reactions_escapes_reason(self) -> None:
        svg = render_reactions_svg([ReactionSelection(label="r1", reactants=["A"], products=["B"], reason="<script>")])
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg

    def test_html_escape(self) -> None:
        svg = render_species_svg([SpeciesSelection(label="<script>")])
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg


# ----------------------- provenance & intake --------------------


class TestProvenance:
    def test_record_creates_file(self, tmp_path: Path) -> None:
        path = record(tmp_path, "test_event", {"a": 1})
        assert path.exists()
        assert "test_event" in path.name

    def test_record_safe_filename(self, tmp_path: Path) -> None:
        path = record(tmp_path, "weird/name with spaces", {"a": 1})
        assert "/" not in path.name


class TestIntake:
    def test_stub_parser_returns_review(self) -> None:
        parser = StubIntakeParser()
        review = parser.parse("some context")
        assert "Intake Review" in review
        assert "some context" in review

    def test_write_intake_review(self, tmp_path: Path) -> None:
        path = write_intake_review(tmp_path, "# Review")
        assert path.exists()
        assert path.read_text() == "# Review"


# ----------------------- execution path -------------------------
#
# These tests inject an inline test double conforming to the
# T3AdapterProtocol so we can drive execute_t3_action through every
# success and failure branch deterministically. This is NOT a mock
# adapter mode in production code: production callers always pass the
# real T3Adapter (or nothing, which uses the default real adapter).


def _ready_workspace(tmp_path: Path) -> Path:
    """Create a workspace already advanced to APPROVED_FOR_EXECUTION."""
    ws = tmp_path / "ws"
    campaign = create_campaign(ws, _make_input("execpath"))
    plan = generate_initial_plan(campaign, ApprovalPolicy(auto_approve_t3_under_cpu_hours=999.0))
    from carmel.services.planner import save_plan as _save

    _save(ws, plan)
    update_state(ws, CampaignStateValue.VALIDATED)
    update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
    update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
    update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)
    return ws


def _success_diagnostics(campaign_id: str, run_id: str) -> DiagnosticsV1:
    return DiagnosticsV1(
        campaign_id=campaign_id,
        run_id=run_id,
        level_of_theory="b3lyp/6-31g(d,p)",
        generated_at=datetime.now(UTC),
        species_to_compute=[SpeciesSelection(label="OH", smiles="[OH]")],
        reactions_to_compute=[ReactionSelection(label="r1", reactants=["A"], products=["B"])],
        pdep_networks_to_compute=[PDepNetworkSelection(network_id="N1", species=["A"])],
    )


class _SuccessAdapter:
    """Inline test double — simulates a successful T3 run."""

    def run(self, workspace_root, campaign, action):
        run_id = str(uuid4())
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name="t3",
            tool_version="test",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=now,
            ended_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            actual_cpu_hours=0.001,
            submission_mode=SubmissionMode.SUBPROCESS,
            command=["python", "T3.py", "input.yml"],
            level_of_theory="b3lyp/6-31g(d,p)",
        )
        return record, _success_diagnostics(campaign.campaign_id, run_id)


class _FailureAdapter:
    """Inline test double — simulates a typed-failure T3 run."""

    def __init__(self, failure_code: FailureCode, error_message: str = "boom") -> None:
        self.failure_code = failure_code
        self.error_message = error_message

    def run(self, workspace_root, campaign, action):
        run_id = str(uuid4())
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name="t3",
            status=RunStatus.FAILED,
            failure_code=self.failure_code,
            started_at=now,
            ended_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            submission_mode=SubmissionMode.SUBPROCESS,
            error_message=self.error_message,
        )
        return record, None


class TestExecutionSaveHelpers:
    def test_save_run_record_creates_file(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        record = RunRecord(
            run_id="r1",
            action_id="a1",
            tool_name="t3",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=datetime.now(UTC),
            submission_mode=SubmissionMode.SUBPROCESS,
        )
        path = save_run_record(ws, record)
        assert path.exists()
        assert path.name == "r1.json"

    def test_save_and_load_diagnostics(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        diag = _success_diagnostics("c1", "r1")
        save_diagnostics(ws, diag)
        loaded = load_diagnostics(ws)
        assert loaded is not None
        assert loaded.campaign_id == "c1"

    def test_load_diagnostics_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert load_diagnostics(tmp_path) is None


class TestExecuteT3ActionSuccess:
    def test_success_transitions_to_completed(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        assert run_record.status == RunStatus.SUCCEEDED
        assert diagnostics is not None
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_success_persists_diagnostics_json(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        assert (ws / DIAGNOSTICS_FILE_NAME).exists()
        assert load_diagnostics(ws) is not None

    def test_success_writes_selection_svgs(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        models_dir = ws / "models"
        assert (models_dir / "species_selection.svg").exists()
        assert (models_dir / "reactions_selection.svg").exists()
        assert (models_dir / "pdep_networks_selection.svg").exists()

    def test_success_writes_run_record(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, _ = execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        run_files = list((ws / "runs").glob(f"{run_record.run_id}.json"))
        assert len(run_files) == 1

    def test_success_writes_provenance(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        prov_files = list((ws / "provenance").glob("*_t3_run.json"))
        assert len(prov_files) >= 1

    def test_success_appends_decision_log_events(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        events = read_events(ws / "decision_log.jsonl")
        kinds = [e.get("event") for e in events]
        assert "t3_run_started" in kinds
        assert "t3_run_finished" in kinds


class TestExecuteT3ActionFailure:
    @pytest.mark.parametrize(
        "failure_code",
        [
            FailureCode.SUBPROCESS_ERROR,
            FailureCode.INVALID_OUTPUT,
            FailureCode.TOOL_NOT_FOUND,
            FailureCode.INPUT_BUILD_ERROR,
            FailureCode.TIMEOUT,
        ],
    )
    def test_failure_transitions_to_failed(self, tmp_path: Path, failure_code: FailureCode) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_t3_action(
            ws,
            load_campaign(ws),
            plan.actions[0],
            adapter=_FailureAdapter(failure_code),
        )
        assert run_record.status == RunStatus.FAILED
        assert run_record.failure_code == failure_code
        assert diagnostics is None
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_failure_does_not_write_diagnostics(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(
            ws,
            load_campaign(ws),
            plan.actions[0],
            adapter=_FailureAdapter(FailureCode.INVALID_OUTPUT),
        )
        assert not (ws / DIAGNOSTICS_FILE_NAME).exists()

    def test_failure_still_writes_run_record_and_provenance(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, _ = execute_t3_action(
            ws,
            load_campaign(ws),
            plan.actions[0],
            adapter=_FailureAdapter(FailureCode.SUBPROCESS_ERROR),
        )
        assert (ws / "runs" / f"{run_record.run_id}.json").exists()
        assert list((ws / "provenance").glob("*_t3_run.json"))

    def test_failure_records_finished_event(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_t3_action(
            ws,
            load_campaign(ws),
            plan.actions[0],
            adapter=_FailureAdapter(FailureCode.SUBPROCESS_ERROR),
        )
        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "t3_run_finished"]
        assert len(finished) == 1
        assert finished[0]["failure_code"] == "subprocess_error"

    def test_failure_removes_stale_diagnostics_and_svgs_from_a_prior_run(self, tmp_path: Path) -> None:
        """Regression guard: a failed run must never leave a previous run's
        diagnostics.json / selection SVGs looking like current output."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        (ws / DIAGNOSTICS_FILE_NAME).write_text("{}", encoding="utf-8")
        models_dir = ws / "models"
        models_dir.mkdir(exist_ok=True)
        stale_svgs = [
            models_dir / "species_selection.svg",
            models_dir / "reactions_selection.svg",
            models_dir / "pdep_networks_selection.svg",
        ]
        for svg_path in stale_svgs:
            svg_path.write_text("<svg>stale</svg>", encoding="utf-8")

        execute_t3_action(
            ws,
            load_campaign(ws),
            plan.actions[0],
            adapter=_FailureAdapter(FailureCode.SUBPROCESS_ERROR),
        )

        assert not (ws / DIAGNOSTICS_FILE_NAME).exists()
        for svg_path in stale_svgs:
            assert not svg_path.exists()


class TestExecuteT3ActionUnexpectedException:
    """Regression guard for C5: an exception during execution must not wedge
    the campaign in RUNNING_T3 with no reachable transition."""

    class _RaisingAdapter:
        def run(self, workspace_root: Path, campaign: object, action: PlannedAction) -> object:
            raise RuntimeError("adapter blew up unexpectedly")

    def test_exception_propagates_to_caller(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError, match="adapter blew up unexpectedly"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())

    def test_exception_leaves_campaign_failed_not_running(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_exception_writes_failed_run_record(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        run_files = list((ws / "runs").glob("*.json"))
        assert len(run_files) == 1
        saved = RunRecord.model_validate(read_json(run_files[0]))
        assert saved.status == RunStatus.FAILED
        assert saved.failure_code == FailureCode.UNKNOWN
        assert "adapter blew up unexpectedly" in (saved.error_message or "")
        # D7: the fabricated record must use the canonical tool name/mode
        # constants rather than hardcoded literals.
        assert saved.tool_name == T3_TOOL_NAME
        assert saved.submission_mode == SubmissionMode.SUBPROCESS

    def test_exception_appends_finished_event(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "t3_run_finished"]
        assert len(finished) == 1
        assert finished[0]["failure_code"] == "unknown"

    def test_campaign_can_retry_after_unexpected_exception(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        retried = update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")
        assert retried.state == CampaignStateValue.APPROVED_FOR_EXECUTION


class _RealRecordThenCrashAdapter:
    """Adapter double that returns a real RunRecord/diagnostics pair, so a
    later failure in ``execute_t3_action`` (e.g. ``save_diagnostics``
    raising) must reuse this record's ``run_id`` rather than fabricating a
    fresh one."""

    def __init__(self) -> None:
        self.run_id = "real-run-id-from-adapter"

    def run(self, workspace_root: Path, campaign: Campaign, action: PlannedAction) -> tuple[RunRecord, DiagnosticsV1]:
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=self.run_id,
            action_id=action.action_id,
            tool_name=T3_TOOL_NAME,
            tool_version="test",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            actual_cpu_hours=0.001,
            submission_mode=SubmissionMode.SUBPROCESS,
            level_of_theory="b3lyp/6-31g(d,p)",
        )
        return record, _success_diagnostics(campaign.campaign_id, self.run_id)


class TestExecuteT3ActionDefensiveHandling:
    """Fixes for the adversarial-review defects D1-D6: ordering guarantees
    and defensive failure handling around ``execute_t3_action``."""

    def test_stale_artifacts_survive_a_failed_state_transition(self, tmp_path: Path) -> None:
        """D1: clearing stale diagnostics must happen only AFTER the
        RUNNING_T3 transition is validated. A campaign that is not
        actually eligible for RUNNING_T3 must raise with the workspace
        untouched — not silently delete a prior run's artifacts first."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input("stale-guard"))
        plan = generate_initial_plan(campaign, ApprovalPolicy(auto_approve_t3_under_cpu_hours=999.0))
        from carmel.services.planner import save_plan as _save

        _save(ws, plan)
        # Deliberately left in DRAFT: not eligible for RUNNING_T3.
        (ws / DIAGNOSTICS_FILE_NAME).write_text("{}", encoding="utf-8")
        models_dir = ws / "models"
        models_dir.mkdir(exist_ok=True)
        stale_svg = models_dir / "species_selection.svg"
        stale_svg.write_text("<svg>stale</svg>", encoding="utf-8")

        with pytest.raises(InvalidTransitionError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())

        assert (ws / DIAGNOSTICS_FILE_NAME).exists()
        assert stale_svg.exists()

    def test_campaign_never_left_running_t3_when_decision_log_append_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D2: everything after the RUNNING_T3 transition succeeds must be
        inside the protected region — even a failure as early as the
        ``t3_run_started`` decision-log append must still drive the
        campaign to FAILED, never leave it wedged in RUNNING_T3."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        def _raise_append(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(execution_module, "append_event", _raise_append)

        with pytest.raises(OSError, match="disk full"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())

        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_real_run_id_preserved_when_diagnostics_persistence_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D3/D4: if the adapter already returned a real RunRecord before a
        later step (save_diagnostics) raises, the persisted failure record
        must carry that real run_id — not a fabricated uuid4 — and
        provenance must be recorded for the failure."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        adapter = _RealRecordThenCrashAdapter()

        def _raise_save_diagnostics(*args: object, **kwargs: object) -> Path:
            raise OSError("disk full while saving diagnostics")

        monkeypatch.setattr(execution_module, "save_diagnostics", _raise_save_diagnostics)

        with pytest.raises(OSError, match="disk full while saving diagnostics"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=adapter)

        run_files = list((ws / "runs").glob("*.json"))
        assert len(run_files) == 1
        saved = RunRecord.model_validate(read_json(run_files[0]))
        assert saved.run_id == adapter.run_id
        assert saved.status == RunStatus.FAILED
        assert saved.failure_code == FailureCode.UNKNOWN

        provenance_files = list((ws / "provenance").glob("*_t3_run.json"))
        assert provenance_files
        recorded = read_json(provenance_files[0])
        assert recorded["run_id"] == adapter.run_id
        assert recorded["status"] == RunStatus.FAILED.value

    def test_original_exception_not_masked_after_terminal_transition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D5: an exception raised after the campaign has already reached a
        terminal state (COMPLETED_PHASE1, which has no outgoing
        transitions) must surface as itself — never as an
        InvalidTransitionError from an unconditional FAILED transition
        attempt. The handler must check ``can_transition`` first."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        def _raise_after_completion(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom after completion")

        monkeypatch.setattr(execution_module._log, "info", _raise_after_completion)

        with pytest.raises(RuntimeError, match="boom after completion"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())

        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_original_exception_not_masked_when_handlers_own_writes_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D5: if the handler's own best-effort persistence also fails, that
        secondary failure must be swallowed (logged) — the caller must
        still see the original exception, not one raised while trying to
        record the failure."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        class _RaisingAdapter:
            def run(
                self, workspace_root: Path, campaign: Campaign, action: PlannedAction
            ) -> tuple[RunRecord, DiagnosticsV1 | None]:
                raise RuntimeError("original adapter failure")

        real_append_event = execution_module.append_event

        def _raise_on_finished_event(log_path: Path, event: dict[str, object]) -> None:
            # Let the initial "t3_run_started" append through so the
            # RuntimeError below is genuinely raised by the adapter, not by
            # this monkeypatch pre-empting it; only the handler's own
            # "t3_run_finished" write (made while recovering from the
            # adapter failure) fails.
            if event.get("event") == "t3_run_finished":
                raise OSError("decision log unavailable")
            real_append_event(log_path, event)

        monkeypatch.setattr(execution_module, "append_event", _raise_on_finished_event)

        with pytest.raises(RuntimeError, match="original adapter failure"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_RaisingAdapter())

        # The state transition itself does not depend on append_event, so
        # the campaign still lands in FAILED despite the handler's own
        # decision-log writes failing.
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_no_succeeded_finished_event_when_diagnostics_save_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D6: diagnostics/SVGs must be durable before the decision log can
        claim the run succeeded. If save_diagnostics fails, the log must
        never contain a "succeeded" t3_run_finished event for that run."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        adapter = _RealRecordThenCrashAdapter()

        def _raise_save_diagnostics(*args: object, **kwargs: object) -> Path:
            raise OSError("disk full while saving diagnostics")

        monkeypatch.setattr(execution_module, "save_diagnostics", _raise_save_diagnostics)

        with pytest.raises(OSError):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=adapter)

        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "t3_run_finished"]
        assert len(finished) == 1
        assert finished[0]["status"] != RunStatus.SUCCEEDED.value
        assert finished[0]["run_id"] == adapter.run_id

    def test_original_exception_survives_when_final_failed_transition_itself_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handler's own attempt to transition the campaign to FAILED can
        itself raise (e.g. a concurrent writer corrupted campaign_state.json
        between the earlier RUNNING_T3 transition and now). That secondary
        failure must only be logged — the caller must still see the
        original adapter exception, not the state-machine's."""
        import carmel.services.execution as execution_module

        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        class _RaisingAdapter:
            def run(
                self, workspace_root: Path, campaign: Campaign, action: PlannedAction
            ) -> tuple[RunRecord, DiagnosticsV1 | None]:
                raise RuntimeError("original adapter failure")

        real_update_state = execution_module.update_state

        def _raise_on_failed_transition(
            workspace_root: Path,
            target: CampaignStateValue,
            notes: str | None = None,
        ) -> CampaignState:
            if target == CampaignStateValue.FAILED:
                raise OSError("campaign_state.json corrupted by a concurrent writer")
            return real_update_state(workspace_root, target, notes)

        monkeypatch.setattr(execution_module, "update_state", _raise_on_failed_transition)

        with pytest.raises(RuntimeError, match="original adapter failure"):
            execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_RaisingAdapter())

        # The campaign is left in RUNNING_T3 (the FAILED transition never
        # committed), which is itself observable evidence the secondary
        # failure was swallowed rather than masking the original error.
        state = load_state(ws)
        assert state.state == CampaignStateValue.RUNNING_T3


# ----------------------- ARC planning -------------------------


class TestGenerateArcPlan:
    def test_single_arc_action(self, tmp_path: Path) -> None:
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign)
        assert len(plan.actions) == 1
        assert plan.actions[0].kind == ActionKind.ARC_RUN

    def test_species_default_from_mixture(self, tmp_path: Path) -> None:
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign)
        assert plan.actions[0].parameters["species"] == [{"label": "O2", "smiles": "[O][O]"}]

    def test_within_envelope_auto_approves(self, tmp_path: Path) -> None:
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign)  # 1 species -> 1 cpu-h, within ARC envelope + budget
        assert not plan.requires_approval
        assert plan.actions[0].approval_requirement == ApprovalRequirement.AUTO_APPROVED

    def test_workspace_root_records_authorization(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input("arcplan"))
        plan = generate_arc_plan(campaign, workspace_root=ws)
        events = read_events(ws / "decision_log.jsonl")
        authz = [e for e in events if e["event"] == "execution_envelope_authorization"]
        assert len(authz) == 1
        assert authz[0]["adapter"] == "arc"
        assert authz[0]["action_id"] == plan.actions[0].action_id
        # The logged requirement must be the one the plan actually carries.
        assert authz[0]["requirement"] == plan.actions[0].approval_requirement.value

    def test_no_workspace_root_writes_no_authorization_event(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input("arcplan"))
        generate_arc_plan(campaign)
        events = read_events(ws / "decision_log.jsonl")
        assert [e for e in events if e["event"] == "execution_envelope_authorization"] == []

    def test_over_envelope_requires_approval(self, tmp_path: Path) -> None:
        # 3 reactions -> 1 + 2*3 = 7 cpu-h, over the ARC envelope (4) -> escalate.
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign, reactions=[{"label": f"r{i}"} for i in range(3)])
        assert plan.requires_approval
        assert plan.actions[0].approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL


# ----------------------- ARC execution path -------------------------


def _arc_ready_workspace(tmp_path: Path) -> Path:
    """Create a workspace with an approved ARC plan at APPROVED_FOR_EXECUTION."""
    ws = tmp_path / "ws"
    campaign = create_campaign(ws, _make_input("arcexec"))
    plan = generate_arc_plan(campaign)
    from carmel.services.planner import save_plan as _save

    _save(ws, plan)
    update_state(ws, CampaignStateValue.VALIDATED)
    update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
    update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
    update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)
    return ws


def _arc_success_diagnostics(campaign_id: str, run_id: str) -> DiagnosticsV1:
    return DiagnosticsV1(
        campaign_id=campaign_id,
        run_id=run_id,
        level_of_theory="wb97xd/def2tzvp",
        generated_at=datetime.now(UTC),
        species_to_compute=[SpeciesSelection(label="OH", smiles="[OH]")],
        reactions_to_compute=[],
        pdep_networks_to_compute=[],
        tool_metadata={"adapter": "arc"},
    )


class _ARCSuccessAdapter:
    """Inline test double — simulates a successful ARC run."""

    def run(self, workspace_root, campaign, action):
        run_id = str(uuid4())
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name="arc",
            tool_version="test",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=now,
            ended_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            actual_cpu_hours=0.001,
            submission_mode=SubmissionMode.SUBPROCESS,
            command=["python", "ARC.py", "input.yml"],
            level_of_theory="wb97xd/def2tzvp",
        )
        return record, _arc_success_diagnostics(campaign.campaign_id, run_id)


class _ARCFailureAdapter:
    """Inline test double — simulates a typed-failure ARC run."""

    def __init__(self, failure_code: FailureCode) -> None:
        self.failure_code = failure_code

    def run(self, workspace_root, campaign, action):
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=str(uuid4()),
            action_id=action.action_id,
            tool_name="arc",
            status=RunStatus.FAILED,
            failure_code=self.failure_code,
            started_at=now,
            ended_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            submission_mode=SubmissionMode.SUBPROCESS,
            error_message="boom",
        )
        return record, None


class TestExecuteArcActionSuccess:
    def test_success_transitions_to_completed(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_arc_action(
            ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter()
        )
        assert run_record.status == RunStatus.SUCCEEDED
        assert diagnostics is not None
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_success_persists_arc_diagnostics_json(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
        assert (ws / ARC_DIAGNOSTICS_FILE_NAME).exists()
        assert load_arc_diagnostics(ws) is not None

    def test_success_writes_svgs_under_arc_subdir(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
        assert (ws / "models" / "arc" / "species_selection.svg").exists()

    def test_success_appends_decision_log_events(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
        kinds = [e.get("event") for e in read_events(ws / "decision_log.jsonl")]
        assert "arc_run_started" in kinds
        assert "arc_run_finished" in kinds

    def test_success_writes_provenance(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
        assert list((ws / "provenance").glob("*_arc_run.json"))


class TestExecuteArcActionFailure:
    def test_failure_transitions_to_failed(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_arc_action(
            ws, load_campaign(ws), plan.actions[0], adapter=_ARCFailureAdapter(FailureCode.SUBPROCESS_ERROR)
        )
        assert run_record.status == RunStatus.FAILED
        assert diagnostics is None
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_failure_does_not_write_arc_diagnostics(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        execute_arc_action(
            ws, load_campaign(ws), plan.actions[0], adapter=_ARCFailureAdapter(FailureCode.INVALID_OUTPUT)
        )
        assert not (ws / ARC_DIAGNOSTICS_FILE_NAME).exists()


class TestExecuteT3ActionDefaultAdapter:
    """Verify that the default adapter is the real T3Adapter."""

    def test_no_adapter_uses_real_t3_adapter(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        # Force "T3 not found" so the real adapter returns FAILED quickly
        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: None)
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_t3_action(ws, load_campaign(ws), plan.actions[0])
        assert run_record.status == RunStatus.FAILED
        assert run_record.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert load_state(ws).state == CampaignStateValue.FAILED


class TestStartT3Action:
    """The background entry point the web UI uses.

    The transition must stay in the caller's thread — that is what makes a
    double-submitted run fail for the user who submitted it — while the
    work itself must not.
    """

    def test_returns_a_thread_that_completes_the_run(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        thread = start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        thread.join(timeout=60)
        assert not thread.is_alive()
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_transition_happens_before_returning(self, tmp_path: Path) -> None:
        """RUNNING_T3 must be on disk by the time the caller gets control back."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        release = threading.Event()

        class _Blocking:
            def run(self, workspace_root: Path, campaign: object, action: PlannedAction) -> object:
                release.wait(timeout=60)
                raise RuntimeError("released")

        thread = start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_Blocking())
        assert load_state(ws).state == CampaignStateValue.RUNNING_T3
        release.set()
        thread.join(timeout=60)

    def test_ineligible_campaign_raises_in_the_calling_thread(self, tmp_path: Path) -> None:
        """A caller must be able to turn a rejected run into a 409, not a 500."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        first = start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        first.join(timeout=60)
        with pytest.raises(InvalidTransitionError):
            start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())

    def test_adapter_exception_is_logged_not_swallowed_silently(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nobody is waiting on the thread, so the log is the only witness."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        class _Raising:
            def run(self, workspace_root: Path, campaign: object, action: PlannedAction) -> object:
                raise RuntimeError("adapter blew up in the background")

        # Carmel's loggers set propagate=False, so caplog's root handler
        # never sees them; attach it to the emitting logger directly.
        emitter = logging.getLogger("carmel.services.execution")
        emitter.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.ERROR):
                thread = start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_Raising())
                thread.join(timeout=60)
        finally:
            emitter.removeHandler(caplog.handler)

        assert load_state(ws).state == CampaignStateValue.FAILED
        assert any("Background T3 run failed" in record.message for record in caplog.records)
