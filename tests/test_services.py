"""Tests for Phase 1 service modules."""

import errno
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

import carmel.services.execution
import carmel.services.processes
import carmel.services.recovery
from carmel.adapters.arc import ARC_TOOL_NAME
from carmel.adapters.t3 import T3_TOOL_NAME
from carmel.schemas import (
    ActionKind,
    ActiveRun,
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
    RunStillLiveError,
    abandon_arc_run,
    abandon_t3_run,
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
from carmel.services.processes import (
    ProcessGroupStatus,
    inspect_process_group,
    kill_process_group,
    process_group_command,
    process_group_exists,
    process_group_is_running,
    process_starttime,
)
from carmel.services.provenance import record
from carmel.services.recovery import (
    LockStateUnknownError,
    ProcessGroupNotRecordedError,
    RunAlreadySupervisedError,
    RunLiveness,
    active_run_path,
    load_active_run,
    probe_run_liveness,
    record_process_group,
    start_supervision,
    supervise_run,
    supervisor_is_alive,
)
from carmel.services.state_machine import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    can_transition,
    load_state,
    update_state,
)
from tests.helpers import (
    _died_within,
    _is_running,
    _shebang_leader_tree,
    _strand_active_run,
    _tool_tree,
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

    def test_blocked_can_only_be_re_planned_or_failed(self) -> None:
        for target in CampaignStateValue:
            expected = target in {CampaignStateValue.FAILED, CampaignStateValue.READY_FOR_PLANNING}
            assert can_transition(CampaignStateValue.BLOCKED, target) == expected

    def test_blocked_campaign_can_be_re_planned(self, tmp_path: Path) -> None:
        """A rejected plan is discarded and re-planned, not un-rejected.

        The only route out of BLOCKED is back to planning, so the rejected
        plan cannot reach execution: it must be regenerated and re-judged
        against the approval policy first.
        """
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        update_state(ws, CampaignStateValue.VALIDATED)
        update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        update_state(ws, CampaignStateValue.BLOCKED, notes="user-rejected")
        assert not can_transition(CampaignStateValue.BLOCKED, CampaignStateValue.APPROVED_FOR_EXECUTION)
        recovered = update_state(ws, CampaignStateValue.READY_FOR_PLANNING, notes="re-plan")
        assert recovered.state == CampaignStateValue.READY_FOR_PLANNING

    def test_every_origin_of_failed_has_at_least_one_legal_exit(self) -> None:
        """The wedge this milestone exists to remove.

        Before recovery edges existed, 7 of the 8 states that could reach
        FAILED had zero legal exits from it: the campaign was unrecoverable
        and every UI action raised. The ARC states RUNNING_ARC and
        RESULTS_READY bring the count of FAILED-reaching origins to 10.
        """
        origins = [state for state, targets in VALID_TRANSITIONS.items() if CampaignStateValue.FAILED in targets]
        assert len(origins) == 10
        for origin in [*origins, None]:
            exits = [
                target
                for target in CampaignStateValue
                if can_transition(CampaignStateValue.FAILED, target, failed_from=origin)
            ]
            assert exits, f"campaign that failed from {origin} has no legal exit"

    def test_recovery_never_advances_a_campaign_past_a_gate_it_did_not_pass(self) -> None:
        """No exit from FAILED may reach a state later than the one it failed from.

        ``READY_FOR_PLANNING`` is exempt: it is *earlier* than every origin
        and re-runs planning and approval in full.
        """
        order = [
            CampaignStateValue.DRAFT,
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.COMPLETED_PHASE1,
        ]
        for origin in order:
            for target in CampaignStateValue:
                if not can_transition(CampaignStateValue.FAILED, target, failed_from=origin):
                    continue
                if target == CampaignStateValue.READY_FOR_PLANNING:
                    continue
                assert order.index(target) <= order.index(origin), f"recovering from {origin} to {target} skips a gate"

    def test_failed_from_diagnostics_ready_resumes_at_diagnostics_ready(self, tmp_path: Path) -> None:
        """The concretely reachable wedge: T3 succeeded, finalizing did not.

        ``_finish_t3_run`` commits DIAGNOSTICS_READY and then
        COMPLETED_PHASE1. If the second write fails, the campaign lands in
        FAILED with real diagnostics already durable on disk — re-running
        a multi-hour T3 job to recover them would be absurd.
        """
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.DIAGNOSTICS_READY,
        ]:
            update_state(ws, target)
        update_state(ws, CampaignStateValue.FAILED, notes="disk full")
        resumed = update_state(ws, CampaignStateValue.DIAGNOSTICS_READY, notes="resume")
        assert resumed.state == CampaignStateValue.DIAGNOSTICS_READY
        assert can_transition(resumed.state, CampaignStateValue.COMPLETED_PHASE1)

    def test_only_the_matching_origins_unlock_a_direct_resume(self) -> None:
        """Each direct resume is opened by a named set of origins and no other.

        Spelled out rather than derived from ``RECOVERY_TARGETS``, so that
        widening that table has to be a deliberate edit here too.
        """
        for target, unlocked_by in [
            (
                CampaignStateValue.APPROVED_FOR_EXECUTION,
                {
                    CampaignStateValue.RUNNING_T3,
                    CampaignStateValue.RUNNING_ARC,
                    CampaignStateValue.APPROVED_FOR_EXECUTION,
                },
            ),
            (CampaignStateValue.DIAGNOSTICS_READY, {CampaignStateValue.DIAGNOSTICS_READY}),
            (CampaignStateValue.RESULTS_READY, {CampaignStateValue.RESULTS_READY}),
        ]:
            for origin in CampaignStateValue:
                assert can_transition(CampaignStateValue.FAILED, target, failed_from=origin) == (origin in unlocked_by)

    def test_a_plan_approved_but_never_launched_resumes_without_re_planning(self, tmp_path: Path) -> None:
        """Failing between approval and launch must not discard the approval.

        The plan was approved and no run ever happened, so there is
        nothing for a fresh planning pass to redo.
        """
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
        ]:
            update_state(ws, target)
        update_state(ws, CampaignStateValue.FAILED, notes="adapter refused to start")
        resumed = update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")
        assert resumed.state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert can_transition(resumed.state, CampaignStateValue.RUNNING_T3)

    def test_re_planning_is_available_even_when_the_origin_was_never_recorded(self) -> None:
        """``failed_from`` is None on any state file Carmel did not write.

        Such a campaign cannot use a direct resume — nothing says which
        gates it passed — but it must still not be stranded.
        """
        assert can_transition(CampaignStateValue.FAILED, CampaignStateValue.READY_FOR_PLANNING, failed_from=None)
        assert not can_transition(
            CampaignStateValue.FAILED, CampaignStateValue.APPROVED_FOR_EXECUTION, failed_from=None
        )
        assert not can_transition(CampaignStateValue.FAILED, CampaignStateValue.DIAGNOSTICS_READY, failed_from=None)

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

    def test_failed_can_retry_to_approved_for_execution_only_from_a_running_tool(self) -> None:
        assert can_transition(
            CampaignStateValue.FAILED,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            failed_from=CampaignStateValue.RUNNING_T3,
        )
        assert can_transition(
            CampaignStateValue.FAILED,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            failed_from=CampaignStateValue.RUNNING_ARC,
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

    def test_update_state_retries_campaign_that_failed_from_running_arc(self, tmp_path: Path) -> None:
        """F4: a campaign that failed mid-ARC-execution (an already-approved
        plan) must be retryable, exactly like the T3 retry edge."""
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input())
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_ARC,
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

    def run(self, workspace_root, campaign, action, on_process_start=None):
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

    def run(self, workspace_root, campaign, action, on_process_start=None):
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
        def run(
            self,
            workspace_root: Path,
            campaign: object,
            action: PlannedAction,
            on_process_start: Callable[[int, list[str]], None] | None = None,
        ) -> object:
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

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple[RunRecord, DiagnosticsV1]:
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
                self,
                workspace_root: Path,
                campaign: Campaign,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
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
                self,
                workspace_root: Path,
                campaign: Campaign,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
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

    def test_level_of_theory_recorded_in_parameters(self, tmp_path: Path) -> None:
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign, level_of_theory="wb97xd/def2tzvp")
        assert plan.actions[0].parameters["level_of_theory"] == "wb97xd/def2tzvp"

    def test_job_types_recorded_in_parameters(self, tmp_path: Path) -> None:
        campaign = create_campaign(tmp_path / "ws", _make_input("arcplan"))
        plan = generate_arc_plan(campaign, job_types={"opt": True})
        assert plan.actions[0].parameters["job_types"] == {"opt": True}


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

    def run(self, workspace_root, campaign, action, on_process_start=None):
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

    def run(self, workspace_root, campaign, action, on_process_start=None):
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
        assert load_arc_diagnostics(ws) is None


class TestExecuteArcActionUnexpectedException:
    """F5 — ARC mirror of the T3 C5 guard: an unexpected exception during
    execution must not wedge the campaign in RUNNING_ARC with no reachable
    transition."""

    class _RaisingAdapter:
        def run(
            self,
            workspace_root: Path,
            campaign: Campaign,
            action: PlannedAction,
            on_process_start: Callable[[int, list[str]], None] | None = None,
        ) -> tuple[RunRecord, DiagnosticsV1 | None]:
            raise RuntimeError("arc adapter blew up unexpectedly")

    def test_exception_propagates_to_caller(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError, match="arc adapter blew up unexpectedly"):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())

    def test_exception_leaves_campaign_failed_not_running(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_exception_writes_failed_run_record(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        run_files = list((ws / "runs").glob("*.json"))
        assert len(run_files) == 1
        saved = RunRecord.model_validate(read_json(run_files[0]))
        assert saved.status == RunStatus.FAILED
        assert saved.failure_code == FailureCode.UNKNOWN
        assert "arc adapter blew up unexpectedly" in (saved.error_message or "")
        # The fabricated record must use the canonical ARC tool name/mode
        # constants rather than hardcoded literals (mirror of D7).
        assert saved.tool_name == ARC_TOOL_NAME
        assert saved.submission_mode == SubmissionMode.SUBPROCESS

    def test_exception_appends_finished_event_and_provenance(self, tmp_path: Path) -> None:
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "arc_run_finished"]
        assert len(finished) == 1
        assert finished[0]["failure_code"] == "unknown"
        assert list((ws / "provenance").glob("*_arc_run.json"))

    def test_campaign_can_retry_after_unexpected_exception(self, tmp_path: Path) -> None:
        """F4 + F5 together: FAILED-from-RUNNING_ARC must be retryable."""
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        with pytest.raises(RuntimeError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=self._RaisingAdapter())
        retried = update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")
        assert retried.state == CampaignStateValue.APPROVED_FOR_EXECUTION


class _ARCRealRecordThenCrashAdapter:
    """ARC double that returns a real SUCCEEDED RunRecord/diagnostics pair, so
    a later persistence failure in ``execute_arc_action`` must reuse this
    record's ``run_id`` rather than fabricating a fresh one (mirror of
    ``_RealRecordThenCrashAdapter``)."""

    def __init__(self) -> None:
        self.run_id = "real-arc-run-id-from-adapter"

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple[RunRecord, DiagnosticsV1]:
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=self.run_id,
            action_id=action.action_id,
            tool_name=ARC_TOOL_NAME,
            tool_version="test",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=now,
            estimated_cpu_hours=action.estimated_cpu_hours,
            actual_cpu_hours=0.001,
            submission_mode=SubmissionMode.SUBPROCESS,
            level_of_theory="wb97xd/def2tzvp",
        )
        return record, _arc_success_diagnostics(campaign.campaign_id, self.run_id)


class TestExecuteArcActionOrderingAndStaleArtifacts:
    """F6/F7 — ARC mirrors of the T3 D1/D3/D6 ordering and stale-artifact
    guards."""

    def test_no_succeeded_finished_event_when_diagnostics_save_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F6: ARC diagnostics/SVGs must be durable before the decision log can
        claim the run succeeded. If save_arc_diagnostics fails, the log must
        never contain a "succeeded" arc_run_finished event for that run."""
        import carmel.services.execution as execution_module

        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        adapter = _ARCRealRecordThenCrashAdapter()

        def _raise_save_arc_diagnostics(*args: object, **kwargs: object) -> Path:
            raise OSError("disk full while saving arc diagnostics")

        monkeypatch.setattr(execution_module, "save_arc_diagnostics", _raise_save_arc_diagnostics)

        with pytest.raises(OSError, match="disk full while saving arc diagnostics"):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=adapter)

        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "arc_run_finished"]
        assert len(finished) == 1
        assert finished[0]["status"] != RunStatus.SUCCEEDED.value
        assert finished[0]["run_id"] == adapter.run_id
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_real_run_id_preserved_when_diagnostics_persistence_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F5/F6: the persisted failure record must carry the adapter's real
        run_id — not a fabricated uuid4 — and provenance must be recorded."""
        import carmel.services.execution as execution_module

        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        adapter = _ARCRealRecordThenCrashAdapter()

        def _raise_save_arc_diagnostics(*args: object, **kwargs: object) -> Path:
            raise OSError("disk full while saving arc diagnostics")

        monkeypatch.setattr(execution_module, "save_arc_diagnostics", _raise_save_arc_diagnostics)

        with pytest.raises(OSError, match="disk full while saving arc diagnostics"):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=adapter)

        run_files = list((ws / "runs").glob("*.json"))
        assert len(run_files) == 1
        saved = RunRecord.model_validate(read_json(run_files[0]))
        assert saved.run_id == adapter.run_id
        assert saved.status == RunStatus.FAILED
        assert saved.failure_code == FailureCode.UNKNOWN

        provenance_files = list((ws / "provenance").glob("*_arc_run.json"))
        assert provenance_files
        recorded = read_json(provenance_files[0])
        assert recorded["run_id"] == adapter.run_id
        assert recorded["status"] == RunStatus.FAILED.value

    def test_failure_removes_stale_arc_diagnostics_and_svgs_from_a_prior_run(self, tmp_path: Path) -> None:
        """F7: a failed ARC run must never leave a previous run's
        arc_diagnostics.json / models/arc SVGs looking like current output."""
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        (ws / ARC_DIAGNOSTICS_FILE_NAME).write_text("{}", encoding="utf-8")
        arc_models_dir = ws / "models" / "arc"
        arc_models_dir.mkdir(parents=True, exist_ok=True)
        stale_svgs = [
            arc_models_dir / "species_selection.svg",
            arc_models_dir / "reactions_selection.svg",
            arc_models_dir / "pdep_networks_selection.svg",
        ]
        for svg_path in stale_svgs:
            svg_path.write_text("<svg>stale</svg>", encoding="utf-8")

        execute_arc_action(
            ws, load_campaign(ws), plan.actions[0], adapter=_ARCFailureAdapter(FailureCode.SUBPROCESS_ERROR)
        )

        assert not (ws / ARC_DIAGNOSTICS_FILE_NAME).exists()
        for svg_path in stale_svgs:
            assert not svg_path.exists()

    def test_stale_arc_artifacts_survive_a_failed_state_transition(self, tmp_path: Path) -> None:
        """F7 (D1 mirror): clearing stale ARC artifacts must happen only AFTER
        the RUNNING_ARC transition is validated. An ineligible campaign must
        raise with the workspace untouched."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input("arc-stale-guard"))
        plan = generate_arc_plan(campaign)
        from carmel.services.planner import save_plan as _save

        _save(ws, plan)
        # Deliberately left in DRAFT: not eligible for RUNNING_ARC.
        (ws / ARC_DIAGNOSTICS_FILE_NAME).write_text("{}", encoding="utf-8")
        arc_models_dir = ws / "models" / "arc"
        arc_models_dir.mkdir(parents=True)
        stale_svg = arc_models_dir / "species_selection.svg"
        stale_svg.write_text("<svg>stale</svg>", encoding="utf-8")

        with pytest.raises(InvalidTransitionError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())

        assert (ws / ARC_DIAGNOSTICS_FILE_NAME).exists()
        assert stale_svg.exists()


class TestExecuteArcActionDefaultAdapter:
    """Verify that the default adapter is the real ARCAdapter."""

    def test_no_adapter_uses_real_arc_adapter(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        # Force "ARC not found" so the real adapter returns FAILED quickly
        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: None)
        ws = _arc_ready_workspace(tmp_path)
        plan = load_plan(ws)
        run_record, diagnostics = execute_arc_action(ws, load_campaign(ws), plan.actions[0])
        assert run_record.status == RunStatus.FAILED
        assert run_record.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert load_state(ws).state == CampaignStateValue.FAILED


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
            def run(
                self,
                workspace_root: Path,
                campaign: object,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
            ) -> object:
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
            def run(
                self,
                workspace_root: Path,
                campaign: object,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
            ) -> object:
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


# ----------------------- process groups -------------------------


class TestProcessGroupInspection:
    def test_a_live_group_exists(self) -> None:
        with _tool_tree() as tree:
            assert process_group_exists(tree.pgid)
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_dead_group_does_not_exist(self) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        assert not process_group_exists(tree.pgid)

    def test_a_group_we_may_not_signal_still_counts_as_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EPERM means "exists but not signalable", never "gone".

        Reporting such a group as gone would let a caller conclude a run
        had ended when it is still executing.
        """
        monkeypatch.setattr(os, "killpg", _raise(PermissionError))
        assert process_group_exists(4242)

    def test_group_command_reads_the_leaders_argv(self) -> None:
        with _tool_tree() as tree:
            assert process_group_command(tree.pgid) == tree.command
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_group_command_is_none_when_the_leader_is_gone(self) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        assert _died_within(tree.pgid)
        assert process_group_command(tree.pgid) is None

    def test_group_command_is_none_for_a_zombie_leader(self) -> None:
        """A zombie's ``/proc`` entry survives, but its cmdline reads empty.

        The leader is this process's own unreaped child, so it lingers as a
        zombie with a live ``/proc/<pid>`` but an empty ``cmdline`` — which
        must read as no command, not as an empty argv that could spuriously
        match one.
        """
        with _tool_tree() as tree:
            os.kill(tree.leader_pid, signal.SIGKILL)
            deadline = time.monotonic() + 15
            while _is_running(tree.leader_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert Path(f"/proc/{tree.leader_pid}").exists(), "the unreaped zombie keeps its /proc entry"
            assert process_group_command(tree.pgid) is None
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_matching_group_is_recognized_as_ours(self) -> None:
        with _tool_tree() as tree:
            assert inspect_process_group(tree.pgid, tree.command, None) == ProcessGroupStatus.RUNNING
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_group_running_a_different_command_is_not_ours(self) -> None:
        """The pid-reuse case: the recorded pgid is live, but not this run's."""
        with _tool_tree() as tree:
            status = inspect_process_group(tree.pgid, ["/usr/bin/something-else"], None)
            assert status == ProcessGroupStatus.UNRECOGNIZED
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_group_with_no_recorded_command_cannot_be_confirmed(self) -> None:
        with _tool_tree() as tree:
            assert inspect_process_group(tree.pgid, None, None) == ProcessGroupStatus.UNKNOWN_LIVE
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_group_whose_leader_died_is_not_reported_as_finished(self) -> None:
        """The bug this status exists to stop: live descendants read as "over".

        Carmel launches ``conda run`` (the group leader), which launches
        T3, which launches RMG. Kill the leader and the descendants are
        reparented to init but stay in the group, still writing. The
        leader's ``/proc`` entry is gone, so nothing identifies them — and
        reporting that as UNRECOGNIZED would let recovery mark the run
        finished underneath a live RMG.
        """
        with _tool_tree() as tree:
            os.kill(tree.leader_pid, signal.SIGKILL)
            tree.proc.wait(timeout=15)
            assert not Path(f"/proc/{tree.leader_pid}").exists()
            assert _is_running(tree.grandchild_pid), "the grandchild should have outlived its parent"

            assert process_group_exists(tree.pgid), "the group is still non-empty"
            assert inspect_process_group(tree.pgid, tree.command, None) == ProcessGroupStatus.UNKNOWN_LIVE

            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_reaped_group_is_not_running(self) -> None:
        """The ordinary end state: nothing left in the group at all."""
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
            tree.proc.wait(timeout=15)
            assert _died_within(tree.grandchild_pid)
        assert not process_group_is_running(tree.pgid)

    def test_a_zombie_leader_is_not_a_running_group(self) -> None:
        """A zombie answers ``killpg`` but cannot write anything.

        Counting it as running would make a fully stopped tree look like
        one that survived SIGKILL, and every abandon would then refuse.
        """
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
            assert _died_within(tree.grandchild_pid)
            # The leader is this process's own child, so nothing has
            # reaped it: it is still in the group and still signallable.
            assert process_group_exists(tree.pgid), "the zombie keeps the group non-empty"
            assert not process_group_is_running(tree.pgid), "but nothing in it is executing"

    def test_a_group_is_reported_running_when_proc_cannot_be_enumerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without ``/proc`` nothing has been *shown* to have stopped.

        The direction matters: reporting "stopped" here would let a kill
        claim success it never verified.
        """
        monkeypatch.setattr(carmel.services.processes, "PROC_ROOT", tmp_path / "no-such-proc")
        with _tool_tree() as tree:
            assert process_group_is_running(tree.pgid)
            assert inspect_process_group(tree.pgid, tree.command, None) == ProcessGroupStatus.UNKNOWN_LIVE
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_process_that_exits_mid_scan_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enumerating ``/proc`` races every process on the machine."""
        real_read = Path.read_text

        def _vanish(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "stat":
                raise OSError("vanished between listing and reading")
            return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _vanish)
        with _tool_tree() as tree:
            assert not process_group_is_running(tree.pgid), "every entry vanished, so none can be counted"
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_dead_group_is_gone_whatever_was_recorded(self) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        assert inspect_process_group(tree.pgid, tree.command, None) == ProcessGroupStatus.GONE
        assert inspect_process_group(tree.pgid, None, None) == ProcessGroupStatus.GONE

    def test_start_time_reads_field_22_of_proc_stat(self) -> None:
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            assert starttime is not None and starttime > 0
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_start_time_is_none_for_a_gone_leader(self) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        assert _died_within(tree.pgid)
        assert process_starttime(tree.pgid) is None

    def test_a_matching_start_time_confirms_the_group_is_ours(self) -> None:
        """The reuse-proof identity: right pid *and* right start time."""
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            assert inspect_process_group(tree.pgid, tree.command, starttime) == ProcessGroupStatus.RUNNING
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_stale_start_time_reads_as_a_reused_pid(self) -> None:
        """A recycled pid keeps the argv but not the start time.

        The command line still matches — this is the same live tree — but
        a start time from before the (hypothetical) reuse must be read as
        "not this run's", the case the command-line comparison alone cannot
        catch.
        """
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            assert starttime is not None
            status = inspect_process_group(tree.pgid, tree.command, starttime - 1)
            assert status == ProcessGroupStatus.UNRECOGNIZED
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_start_time_of_a_vanished_leader_is_unknown_not_finished(self) -> None:
        """Start-time identity degrades the same safe way the argv one does.

        Kill the leader while a descendant survives: the group is alive but
        its leader's ``/proc`` entry is gone, so the recorded start time can
        no longer be read. That must read as UNKNOWN_LIVE — never as the run
        being over — because the survivors are most likely T3 and RMG.
        """
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            os.kill(tree.leader_pid, signal.SIGKILL)
            tree.proc.wait(timeout=15)
            assert not Path(f"/proc/{tree.leader_pid}").exists()
            assert _is_running(tree.grandchild_pid)
            assert inspect_process_group(tree.pgid, tree.command, starttime) == ProcessGroupStatus.UNKNOWN_LIVE
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_group_of_only_zombies_is_gone_not_running(self) -> None:
        """A group with nothing executing is over, even if a zombie matches.

        Kill the whole group: the grandchild is reaped by init, but the
        leader is this process's own unreaped child, so it lingers as a
        zombie whose ``/proc`` entry — and start time — survive. The
        identity check would match that zombie, so ``inspect`` must first
        rule out a group in which nothing is actually running, or it reports
        a stopped run as still going and refuses recovery forever.
        """
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            os.killpg(tree.pgid, signal.SIGKILL)
            deadline = time.monotonic() + 15
            while _is_running(tree.leader_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert Path(f"/proc/{tree.leader_pid}").exists(), "the unreaped zombie keeps its /proc entry"
            assert process_starttime(tree.pgid) == starttime, "the zombie's start time still matches"
            assert inspect_process_group(tree.pgid, tree.command, starttime) == ProcessGroupStatus.GONE

    def test_a_shebang_launched_leader_is_recognized_from_proc(self) -> None:
        """Finding #1: a ``conda``-shaped launch must still be identifiable.

        The kernel prepends the interpreter to a ``#!`` wrapper's argv, so
        the launched ``command`` never matches ``/proc``. Comparing the
        launched argv would misfire and read a live orphan as finished. The
        kernel-observed command line (what recovery records) does match, and
        the start time confirms it beyond doubt.
        """
        with _shebang_leader_tree() as tree:
            observed = process_group_command(tree.pgid)
            assert observed is not None
            assert observed != tree.command, "the launched argv should differ from /proc's"
            assert observed[1:] == tree.command, "the kernel prepended the interpreter"

            # The launched argv is not recognized; the kernel's is.
            assert inspect_process_group(tree.pgid, tree.command, None) == ProcessGroupStatus.UNRECOGNIZED
            assert inspect_process_group(tree.pgid, observed, None) == ProcessGroupStatus.RUNNING
            starttime = process_starttime(tree.pgid)
            assert inspect_process_group(tree.pgid, tree.command, starttime) == ProcessGroupStatus.RUNNING
            os.killpg(tree.pgid, signal.SIGKILL)


class TestKillProcessGroup:
    def test_kills_the_whole_tree_not_just_the_leader(self) -> None:
        """The grandchild assertion lives in ``_tool_tree``'s exit."""
        with _tool_tree() as tree:
            assert kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.5)

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self) -> None:
        with _tool_tree(ignore_sigterm=True) as tree:
            assert kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.5)

    def test_a_descendant_outliving_the_leader_is_not_success(self) -> None:
        """The asymmetric kill: SIGTERM takes the leader, a child ignores it.

        The group is then alive with its leader gone — running, and
        unidentifiable, at the same time. Checking identity after SIGTERM
        reads that as "no longer ours" and returns success over a live
        RMG; only asking whether anything is still *executing* escalates
        to SIGKILL and actually stops it.
        """
        with _tool_tree(only_grandchild_ignores=True) as tree:
            assert kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.5)
            assert _died_within(tree.grandchild_pid), "the stubborn grandchild was never killed"

    def test_refuses_to_signal_a_group_that_is_not_ours(self) -> None:
        """The whole reason the launched argv is recorded next to the pgid.

        Signalling on a recycled pgid would kill an unrelated process
        group — the same class of collateral damage the process-tree kill
        was introduced to avoid, aimed at a stranger instead.
        """
        with _tool_tree() as tree:
            assert not kill_process_group(tree.pgid, ["/usr/bin/not-ours"], None, grace_period_s=0.5)
            # Both processes must be genuinely executing, not zombies: a
            # kill that went ahead anyway would leave corpses that still
            # satisfy killpg(pgid, 0) and a bare /proc check.
            assert _is_running(tree.leader_pid), "an unrecognized group's leader was signalled"
            assert _is_running(tree.grandchild_pid), "an unrecognized group's child was signalled"
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_an_already_dead_group_is_reported_clear(self) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        assert kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.5)

    def test_reports_failure_when_it_may_not_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _tool_tree() as tree:
            monkeypatch.setattr(os, "killpg", _forbid_signals(tree.pgid))
            assert not kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.1)
            assert _is_running(tree.grandchild_pid)
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_reports_failure_when_the_group_survives_sigkill(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A group that outlives SIGKILL must never be reported as stopped."""
        with _tool_tree(ignore_sigterm=True) as tree:
            monkeypatch.setattr(carmel.services.processes, "_REAP_TIMEOUT_S", 0.2)
            monkeypatch.setattr(os, "killpg", _swallow_sigkill(tree.pgid))
            logger = logging.getLogger("carmel.services.processes")
            logger.addHandler(caplog.handler)
            try:
                assert not kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.2)
            finally:
                logger.removeHandler(caplog.handler)
            assert "survived SIGKILL" in caplog.text
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_reports_failure_when_sigkill_specifically_is_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIGTERM may be permitted while SIGKILL is not.

        A privilege-dropping child can accept the polite signal and refuse
        the forceful one; the group is then still running and must be
        reported as such.
        """
        with _tool_tree(ignore_sigterm=True) as tree:
            monkeypatch.setattr(os, "killpg", _forbid_signal(tree.pgid, signal.SIGKILL))
            assert not kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.2)
            assert _is_running(tree.grandchild_pid)
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_signal_reporting_no_such_group_is_not_proof_the_group_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Delivery is not the same fact as termination.

        ``ESRCH`` from ``killpg`` says the signal reached nothing, which is
        fine to treat as delivered — but the verdict has to come from
        observing the group afterwards. Here the signals all report the
        group gone while it demonstrably keeps running, and the answer must
        still be "not stopped".
        """
        with _tool_tree() as tree:
            monkeypatch.setattr(carmel.services.processes, "_REAP_TIMEOUT_S", 0.2)
            monkeypatch.setattr(os, "killpg", _vanish_on_signal(tree.pgid))
            assert not kill_process_group(tree.pgid, tree.command, None, grace_period_s=0.1)
            assert _is_running(tree.grandchild_pid)
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)


def _vanish_on_signal(pgid: int) -> Callable[..., None]:
    """Return a killpg replacement where *pgid* exists but cannot be signalled.

    ``sig=0`` (the existence probe) still succeeds, so the group reads as
    live; any real signal reports it already gone.
    """
    real = os.killpg

    def _killpg(target: int, sig: int) -> None:
        if target == pgid and sig != 0:
            raise ProcessLookupError("no such process group")
        real(target, sig)

    return _killpg


def _forbid_signal(pgid: int, forbidden: signal.Signals) -> Callable[..., None]:
    """Return a killpg replacement that denies only *forbidden* to *pgid*."""
    real = os.killpg

    def _killpg(target: int, sig: int) -> None:
        if target == pgid and sig == forbidden:
            raise PermissionError("not permitted")
        real(target, sig)

    return _killpg


def _raise(exc: type[BaseException]) -> Callable[..., None]:
    """Return a killpg replacement that always raises *exc*."""

    def _killpg(_pgid: int, _sig: int) -> None:
        raise exc()

    return _killpg


def _forbid_signals(pgid: int) -> Callable[..., None]:
    """Return a killpg replacement that denies real signals to *pgid*."""
    real = os.killpg

    def _killpg(target: int, sig: int) -> None:
        if target == pgid and sig != 0:
            raise PermissionError("not permitted")
        real(target, sig)

    return _killpg


def _swallow_sigkill(pgid: int) -> Callable[..., None]:
    """Return a killpg replacement that drops every real signal to *pgid*."""
    real = os.killpg

    def _killpg(target: int, sig: int) -> None:
        if target == pgid and sig != 0:
            return
        real(target, sig)

    return _killpg


# ----------------------- run supervision & recovery -------------


def _running_workspace(tmp_path: Path) -> Path:
    """Create a workspace sitting in RUNNING_T3 with no live supervisor."""
    ws = _ready_workspace(tmp_path)
    update_state(ws, CampaignStateValue.RUNNING_T3)
    return ws


class TestSupervisorLock:
    def test_a_fresh_workspace_has_no_supervisor(self, tmp_path: Path) -> None:
        assert not supervisor_is_alive(tmp_path)

    def test_a_run_in_progress_is_detectable_from_the_same_process(self, tmp_path: Path) -> None:
        """The web request and the run thread share a process.

        ``flock`` conflicts between two file descriptors even inside one
        process, which is what makes a dashboard request able to see the
        run its own server is executing.
        """
        with supervise_run(tmp_path, "act-1"):
            assert supervisor_is_alive(tmp_path)
        assert not supervisor_is_alive(tmp_path)

    def test_a_probe_does_not_block_another_concurrent_probe(self, tmp_path: Path) -> None:
        """Two liveness probes must not read each other as a live supervisor.

        A dead run leaves the lock free. The probe takes it *shared*, so a
        second probe running while the first still holds it takes a
        compatible shared lock and correctly reads the lock as free. Were
        the probe exclusive, that concurrent probe would get ``EWOULDBLOCK``
        and falsely report a live supervisor — re-wedging a finished run on
        the auto-refreshing dashboard. Holding a shared lock here stands in
        for that concurrent probe deterministically.
        """
        import fcntl

        from carmel.services.recovery import _open_lock_file

        other_probe = _open_lock_file(tmp_path)
        fcntl.flock(other_probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            assert supervisor_is_alive(tmp_path) is False, "a shared probe must not block on another shared probe"
        finally:
            fcntl.flock(other_probe, fcntl.LOCK_UN)
            other_probe.close()

    def test_the_in_flight_record_exists_only_for_the_duration(self, tmp_path: Path) -> None:
        with supervise_run(tmp_path, "act-1"):
            active = load_active_run(tmp_path)
            assert active is not None
            assert active.action_id == "act-1"
            assert active.supervisor_pid == os.getpid()
            assert active.process_group_id is None
        assert load_active_run(tmp_path) is None

    def test_a_second_supervisor_is_refused(self, tmp_path: Path) -> None:
        with (
            supervise_run(tmp_path, "act-1"),
            pytest.raises(RunAlreadySupervisedError),
            supervise_run(tmp_path, "act-2"),
        ):
            pass

    def test_the_record_survives_a_supervisor_that_never_finishes(self, tmp_path: Path) -> None:
        """The whole point of the record: it outlives its writer."""
        _strand_active_run(tmp_path, process_group_id=4242, command=["conda", "run"])
        assert not supervisor_is_alive(tmp_path)
        active = load_active_run(tmp_path)
        assert active is not None
        assert active.process_group_id == 4242

    def test_the_launched_process_group_is_recorded(self, tmp_path: Path) -> None:
        with supervise_run(tmp_path, "act-1") as supervision:
            supervision.record_process_group(4242, ["conda", "run", "-n", "t3_env"])
            active = load_active_run(tmp_path)
            assert active is not None
            assert active.process_group_id == 4242
            assert active.command == ["conda", "run", "-n", "t3_env"]

    def test_recording_reads_the_kernel_identity_not_the_launched_argv(self, tmp_path: Path) -> None:
        """Finding #1: a ``conda``-shaped launch is recorded as ``/proc`` sees it.

        Storing the launched argv would persist ``[<wrapper>, run, ...]``,
        which never matches the ``[<python>, <wrapper>, run, ...]`` the
        kernel reports for a ``#!`` script — so a later recovery would fail
        to recognize a live orphan and declare it finished. Recording the
        kernel's own view of the command, plus the start time, is what keeps
        the tool identifiable across the deployment path that actually ships.
        """
        with _shebang_leader_tree() as tree, supervise_run(tmp_path, "act-1") as supervision:
            supervision.record_process_group(tree.pgid, tree.command)
            active = load_active_run(tmp_path)
            assert active is not None
            assert active.command == process_group_command(tree.pgid)
            assert active.command != tree.command, "the launched argv would not have matched /proc"
            assert active.leader_starttime == process_starttime(tree.pgid)
            assert active.leader_starttime is not None
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_recording_falls_back_to_the_launched_argv_when_proc_is_unreadable(self, tmp_path: Path) -> None:
        """A pgid whose ``/proc`` cannot be read keeps the passed argv as a label."""
        with supervise_run(tmp_path, "act-1") as supervision:
            supervision.record_process_group(4242, ["conda", "run"])
            active = load_active_run(tmp_path)
            assert active is not None
            assert active.command == ["conda", "run"]
            assert active.leader_starttime is None

    def test_recording_a_group_with_no_run_in_flight_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProcessGroupNotRecordedError, match="No in-flight run record"):
            record_process_group(tmp_path, 4242, ["conda"])

    def test_the_lock_dies_with_a_supervisor_in_another_process(self, tmp_path: Path) -> None:
        """The claim the whole module rests on, tested across a real process.

        Every other lock test here takes and releases the lock inside the
        test process, which proves only that ``flock`` conflicts with
        itself. The actual premise is stronger and is the reason a lock
        was chosen over a heartbeat or a pid: the *kernel* drops it when
        the holder dies, however it dies. SIGKILL runs no cleanup code, so
        nothing but the kernel can be releasing it here.
        """
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib, sys, time; sys.path.insert(0, sys.argv[1]);"
                "from carmel.services.recovery import start_supervision;"
                # Bound to a name deliberately: the lock lives exactly as
                # long as the supervision object that owns its file does.
                "held = start_supervision(pathlib.Path(sys.argv[2]), 'act-1');"
                "print('locked', flush=True); time.sleep(300)",
                str(Path(__file__).resolve().parent.parent),
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"
            assert supervisor_is_alive(tmp_path), "another process holds the lock"

            holder.kill()
            holder.wait(timeout=15)
            assert not supervisor_is_alive(tmp_path), "the kernel must drop the lock of a killed holder"
            assert load_active_run(tmp_path) is not None, "the record outlives the supervisor"
        finally:
            if holder.poll() is None:  # pragma: no cover -- only on assertion failure
                holder.kill()
                holder.wait(timeout=15)

    def test_a_lock_that_cannot_be_taken_for_an_unrelated_reason_is_not_contention(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ENOLCK`` is the absence of an answer, not the answer "held"."""

        def _no_locks(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOLCK, "no locks available")

        monkeypatch.setattr(carmel.services.recovery.fcntl, "flock", _no_locks)
        with pytest.raises(LockStateUnknownError):
            supervisor_is_alive(tmp_path)
        with pytest.raises(LockStateUnknownError):
            start_supervision(tmp_path, "act-1")

    def test_an_unopenable_lock_file_is_not_an_answer_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unwritable workspace must not read as "a supervisor is alive"."""

        def _no_open(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(carmel.services.recovery, "_open_lock_file", _no_open)
        with pytest.raises(LockStateUnknownError, match="Could not open"):
            supervisor_is_alive(tmp_path)
        with pytest.raises(LockStateUnknownError, match="Could not open"):
            start_supervision(tmp_path, "act-1")

    def test_a_record_that_cannot_be_written_at_all_releases_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing to start must not leave the campaign locked against retry.

        The lock is taken before the record is written, so a write failure
        that kept hold of it would wedge the campaign exactly as hard as
        the bug this module exists to fix.
        """

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(carmel.services.recovery, "write_json", _explode)
        with pytest.raises(OSError, match="disk full"):
            start_supervision(tmp_path, "act-1")
        monkeypatch.undo()
        assert not supervisor_is_alive(tmp_path), "the lock must not survive a failed start"

    def test_an_unreadable_record_is_treated_as_absent(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A corrupt record must not be what stops a campaign being recovered."""
        active_run_path(tmp_path).write_text("{not json", encoding="utf-8")
        logger = logging.getLogger("carmel.services.recovery")
        logger.addHandler(caplog.handler)
        try:
            assert load_active_run(tmp_path) is None
        finally:
            logger.removeHandler(caplog.handler)
        assert "unreadable active-run record" in caplog.text

    def test_a_record_that_cannot_be_written_fails_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run nothing can track must not be allowed to proceed.

        This was best-effort once, on the reasoning that bookkeeping for a
        future recovery should never break the run itself. It inverts: a
        run with no recorded process group reads afterwards exactly like a
        run that never launched, so recovery would offer to abandon a
        campaign whose T3 is still writing into it.
        """

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        with supervise_run(tmp_path, "act-1") as supervision:
            monkeypatch.setattr(carmel.services.recovery, "write_json", _explode)
            try:
                with pytest.raises(ProcessGroupNotRecordedError, match="4242"):
                    supervision.record_process_group(4242, ["conda"])
            finally:
                monkeypatch.undo()


class TestProbeRunLiveness:
    def test_a_supervised_run_is_reported_in_progress(self, tmp_path: Path) -> None:
        with supervise_run(tmp_path, "act-1"):
            report = probe_run_liveness(tmp_path)
        assert report.liveness == RunLiveness.SUPERVISED
        assert not report.is_finished

    def test_nothing_recorded_is_reported_as_such(self, tmp_path: Path) -> None:
        report = probe_run_liveness(tmp_path)
        assert report.liveness == RunLiveness.NO_RECORD
        assert report.is_finished
        assert report.active_run is None

    def test_a_run_whose_tool_never_launched_is_finished(self, tmp_path: Path) -> None:
        _strand_active_run(tmp_path, process_group_id=None, command=None)
        report = probe_run_liveness(tmp_path)
        assert report.liveness == RunLiveness.UNSUPERVISED
        assert report.is_finished

    def test_a_dead_process_group_is_finished(self, tmp_path: Path) -> None:
        with _tool_tree() as tree:
            os.killpg(tree.pgid, signal.SIGKILL)
        _strand_active_run(tmp_path, tree.pgid, tree.command)
        report = probe_run_liveness(tmp_path)
        assert report.liveness == RunLiveness.UNSUPERVISED
        assert report.is_finished

    def test_a_live_tool_with_no_supervisor_is_orphaned(self, tmp_path: Path) -> None:
        """The case a naive "mark it failed" button would silently abandon."""
        with _tool_tree() as tree:
            _strand_active_run(tmp_path, tree.pgid, tree.command)
            report = probe_run_liveness(tmp_path)
            assert report.liveness == RunLiveness.ORPHANED
            assert not report.is_finished
            assert str(tree.pgid) in report.detail
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_recycled_process_group_is_not_mistaken_for_the_run(self, tmp_path: Path) -> None:
        """A live group running something else is not evidence of this run."""
        with _tool_tree() as tree:
            _strand_active_run(tmp_path, tree.pgid, ["/usr/bin/some-other-tool"])
            report = probe_run_liveness(tmp_path)
            assert report.liveness == RunLiveness.UNSUPERVISED
            assert report.is_finished
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_conda_launched_orphan_is_recognized_not_declared_finished(self, tmp_path: Path) -> None:
        """Finding #1, end to end: the production deployment path.

        Under ``$T3_CONDA_ENV`` the tool is a ``#!`` wrapper, so the argv
        Carmel launched never matches what ``/proc`` reports. Recorded as
        recovery actually records it — the kernel-observed command and the
        start time — a live orphan reads as ORPHANED. Recorded the pre-fix
        way (launched argv, no start time) the very same live tree reads as
        UNSUPERVISED: the run declared over while it keeps writing.
        """
        with _shebang_leader_tree() as tree:
            observed = process_group_command(tree.pgid)
            starttime = process_starttime(tree.pgid)

            _strand_active_run(tmp_path, tree.pgid, observed, leader_starttime=starttime)
            report = probe_run_liveness(tmp_path)
            assert report.liveness == RunLiveness.ORPHANED
            assert not report.is_finished

            _strand_active_run(tmp_path, tree.pgid, tree.command, leader_starttime=None)
            regressed = probe_run_liveness(tmp_path)
            assert regressed.liveness == RunLiveness.UNSUPERVISED, "the pre-fix record lost a live orphan"

            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_start_time_from_before_pid_reuse_reads_as_finished(self, tmp_path: Path) -> None:
        """A stale start time on a live pgid is positive evidence of reuse.

        The command line still matches — same live tree — but a start time
        recorded before the pid was (hypothetically) recycled proves the
        group Carmel launched is gone, so the run is over. This is the case
        the argv comparison alone could never catch.
        """
        with _tool_tree() as tree:
            starttime = process_starttime(tree.pgid)
            assert starttime is not None
            _strand_active_run(tmp_path, tree.pgid, tree.command, leader_starttime=starttime - 1)
            report = probe_run_liveness(tmp_path)
            assert report.liveness == RunLiveness.UNSUPERVISED
            assert report.is_finished
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_tool_that_outlived_its_launcher_is_never_called_finished(self, tmp_path: Path) -> None:
        """The corruption path: live RMG reported as a run that has ended.

        Carmel launches ``conda run`` as the group leader; T3 and RMG live
        beneath it. Kill only the leader and the descendants survive in
        the group, still writing into the workspace — but the leader's
        ``/proc`` entry is gone, so nothing identifies them any more.
        Classifying that as "this run's processes have ended" is what let
        a campaign be abandoned out from under a live RMG.
        """
        with _tool_tree() as tree:
            _strand_active_run(tmp_path, tree.pgid, tree.command)
            os.kill(tree.leader_pid, signal.SIGKILL)
            tree.proc.wait(timeout=15)
            assert _is_running(tree.grandchild_pid), "the grandchild should have outlived its parent"

            report = probe_run_liveness(tmp_path)
            assert report.liveness == RunLiveness.UNKNOWN
            assert not report.is_finished, "a live descendant must never read as finished"
            assert str(tree.pgid) in report.detail

            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_lock_that_cannot_be_interrogated_reports_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No working ``flock`` must not silently mean "a supervisor is alive".

        That is the permanent wedge this module exists to remove, and an
        NFS mount without a lock daemon would reinstate it.
        """

        def _no_locks(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOLCK, "no locks available")

        monkeypatch.setattr(carmel.services.recovery.fcntl, "flock", _no_locks)
        report = probe_run_liveness(tmp_path)
        assert report.liveness == RunLiveness.UNKNOWN
        assert not report.is_finished


class TestAbandonT3Run:
    def test_a_supervised_run_is_refused(self, tmp_path: Path) -> None:
        ws = _running_workspace(tmp_path)
        with supervise_run(ws, "act-1"), pytest.raises(RunStillLiveError):
            abandon_t3_run(ws, load_campaign(ws))
        assert load_state(ws).state == CampaignStateValue.RUNNING_T3

    def test_a_stale_campaign_is_failed(self, tmp_path: Path) -> None:
        ws = _running_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        state, report = abandon_t3_run(ws, load_campaign(ws))
        assert state.state == CampaignStateValue.FAILED
        assert state.failed_from == CampaignStateValue.RUNNING_T3
        assert report.liveness == RunLiveness.UNSUPERVISED
        assert load_active_run(ws) is None

    def test_the_abandoned_run_is_recorded_under_its_own_failure_code(self, tmp_path: Path) -> None:
        """Abandoning is not the same as observing the tool fail."""
        ws = _running_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        abandon_t3_run(ws, load_campaign(ws))
        records = list((ws / "runs").glob("*.json"))
        assert len(records) == 1
        record = RunRecord.model_validate(json.loads(records[0].read_text()))
        assert record.status == RunStatus.FAILED
        assert record.failure_code == FailureCode.ABANDONED
        assert record.action_id == "act-1"

    def test_a_campaign_with_no_record_at_all_is_still_recoverable(self, tmp_path: Path) -> None:
        ws = _running_workspace(tmp_path)
        state, report = abandon_t3_run(ws, load_campaign(ws))
        assert state.state == CampaignStateValue.FAILED
        assert report.liveness == RunLiveness.NO_RECORD
        assert not list((ws / "runs").glob("*.json")), "no run was recorded, so none should be invented"

    def test_an_orphaned_tool_is_stopped_before_the_run_is_called_over(self, tmp_path: Path) -> None:
        """The defect this must not reintroduce, one layer up.

        Marking the campaign FAILED while T3 and RMG keep writing into the
        workspace is the same lie the process-tree kill exists to prevent.
        """
        ws = _running_workspace(tmp_path)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            state, report = abandon_t3_run(ws, load_campaign(ws))
            assert report.liveness == RunLiveness.ORPHANED
            assert state.state == CampaignStateValue.FAILED
            assert not _is_running(tree.leader_pid)
            assert not _is_running(tree.grandchild_pid)

    def test_a_tool_that_cannot_be_stopped_leaves_the_campaign_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing to stop the tool must not be rounded down to success."""
        ws = _running_workspace(tmp_path)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            monkeypatch.setattr(carmel.services.execution, "kill_process_group", lambda *a, **k: False)
            with pytest.raises(RunStillLiveError, match="may still be running"):
                abandon_t3_run(ws, load_campaign(ws))
            assert load_state(ws).state == CampaignStateValue.RUNNING_T3
            assert load_active_run(ws) is not None, "the record must survive a failed abandon"
            monkeypatch.undo()
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_abandoning_is_recorded_in_the_decision_log(self, tmp_path: Path) -> None:
        ws = _running_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        abandon_t3_run(ws, load_campaign(ws))
        events = [e["event"] for e in read_events(ws / "decision_log.jsonl")]
        assert "t3_run_abandoned" in events

    def test_a_run_that_cannot_be_accounted_for_is_refused(self, tmp_path: Path) -> None:
        """Refusing is the whole point: the alternative corrupts a live run.

        A tool whose launcher died is unidentifiable but very much alive.
        Abandoning it would write a terminal run record and move the
        campaign to FAILED while T3 keeps writing into the workspace.
        """
        ws = _ready_workspace(tmp_path)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            update_state(ws, CampaignStateValue.RUNNING_T3)
            os.kill(tree.leader_pid, signal.SIGKILL)
            tree.proc.wait(timeout=15)

            with pytest.raises(RunStillLiveError):
                abandon_t3_run(ws, load_campaign(ws))

            assert load_state(ws).state == CampaignStateValue.RUNNING_T3
            assert _is_running(tree.grandchild_pid), "the live tool must be untouched"
            os.killpg(tree.pgid, signal.SIGKILL)

    def test_a_campaign_that_is_not_running_cannot_be_abandoned(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        with pytest.raises(InvalidTransitionError):
            abandon_t3_run(ws, load_campaign(ws))


class TestRunsAreSupervised:
    def test_a_real_run_records_and_then_clears_its_supervision(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        seen: list[ActiveRun | None] = []

        class _Observing:
            submission_mode = SubmissionMode.SUBPROCESS

            def run(
                self,
                workspace_root: Path,
                campaign: Campaign,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
            ) -> tuple[RunRecord, DiagnosticsV1 | None]:
                assert on_process_start is not None
                on_process_start(4242, ["conda", "run"])
                seen.append(load_active_run(workspace_root))
                now = datetime.now(UTC)
                return RunRecord(
                    run_id="observed",
                    action_id=action.action_id,
                    tool_name=T3_TOOL_NAME,
                    status=RunStatus.FAILED,
                    failure_code=FailureCode.SUBPROCESS_ERROR,
                    started_at=now,
                    ended_at=now,
                    submission_mode=SubmissionMode.SUBPROCESS,
                    error_message="nope",
                ), None

        execute_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_Observing())
        assert seen[0] is not None, "the run must be recorded as in flight while it runs"
        assert seen[0].process_group_id == 4242
        assert load_active_run(ws) is None, "a finished run must leave no in-flight record"
        assert not supervisor_is_alive(ws)

    def test_the_lock_is_held_before_the_campaign_ever_reads_as_running(self, tmp_path: Path) -> None:
        """No instant may exist where a campaign is RUNNING_T3 and unsupervised.

        The transition happens in the caller's thread and the run in a
        background one. If the lock were taken by the background thread, a
        probe landing between the two would find a RUNNING_T3 campaign
        with no supervisor and no in-flight record — read that as
        NO_RECORD, conclude nothing ever started, and offer to abandon a
        run that was about to launch.

        Blocking the background thread before it does anything makes that
        window the whole test, rather than something to race against.
        """
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        # Stop the background thread from ever running. Whatever the
        # caller has done by the time it returns is then the entire
        # observable state, so this cannot pass by winning a race: if the
        # lock were taken by the thread, it would never be taken at all.
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(threading.Thread, "start", lambda self: None)
        try:
            # Held, not discarded: the lock lives as long as the thread
            # that owns the supervision does, and in production that is
            # the thread actually running T3.
            thread = start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
            assert thread is not None
            assert load_state(ws).state == CampaignStateValue.RUNNING_T3
            report = probe_run_liveness(ws)
            assert report.liveness == RunLiveness.SUPERVISED
            assert not report.is_finished, "an about-to-launch run must never read as finished"
        finally:
            monkeypatch.undo()

    def test_a_second_run_is_refused_while_the_first_holds_the_lock(self, tmp_path: Path) -> None:
        """Two POSTs must not both get a run; the lock decides, not the state."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with supervise_run(ws, "someone-else"):
            with pytest.raises(RunAlreadySupervisedError):
                start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
            assert load_state(ws).state != CampaignStateValue.RUNNING_T3, (
                "a refused run must not have moved the campaign"
            )

    def test_a_run_whose_thread_cannot_start_fails_rather_than_wedging(self, tmp_path: Path) -> None:
        """A thread that never starts must not strand the campaign in RUNNING_T3.

        By the time ``thread.start()`` runs the campaign is already
        RUNNING_T3, and if it raises — ``can't start new thread`` on a
        loaded box — nothing will ever run ``_finish_t3_run`` to release
        the lock or record an outcome: the exact permanent wedge this path
        exists to avoid, one layer up. The run must release its supervision
        and fail, so the campaign reads as recoverable rather than forever
        supervised.
        """
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)

        def _cannot_start(_self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(threading.Thread, "start", _cannot_start)
        try:
            with pytest.raises(RuntimeError, match="can't start new thread"):
                start_t3_action(ws, load_campaign(ws), plan.actions[0], adapter=_SuccessAdapter())
        finally:
            monkeypatch.undo()

        assert load_state(ws).state == CampaignStateValue.FAILED
        assert load_active_run(ws) is None, "the in-flight record must be cleared"
        assert probe_run_liveness(ws).liveness == RunLiveness.NO_RECORD, "the run lock must be released"


def _running_arc_workspace(tmp_path: Path) -> Path:
    """Create a workspace sitting in RUNNING_ARC with no live supervisor."""
    ws = _ready_workspace(tmp_path)
    update_state(ws, CampaignStateValue.RUNNING_ARC)
    return ws


class TestAbandonArcRun:
    """ARC mirror of TestAbandonT3Run.

    RUNNING_ARC is a first-class long-running subprocess state, so the
    killed-supervisor wedge the T3 abandon path recovers must be
    recoverable for ARC through the same supervision contract.
    """

    def test_a_supervised_run_is_refused(self, tmp_path: Path) -> None:
        ws = _running_arc_workspace(tmp_path)
        with supervise_run(ws, "act-1"), pytest.raises(RunStillLiveError):
            abandon_arc_run(ws, load_campaign(ws))
        assert load_state(ws).state == CampaignStateValue.RUNNING_ARC

    def test_a_stale_campaign_is_failed_under_arcs_own_name(self, tmp_path: Path) -> None:
        ws = _running_arc_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        state, report = abandon_arc_run(ws, load_campaign(ws))
        assert state.state == CampaignStateValue.FAILED
        assert state.failed_from == CampaignStateValue.RUNNING_ARC
        assert report.liveness == RunLiveness.UNSUPERVISED
        assert load_active_run(ws) is None
        records = list((ws / "runs").glob("*.json"))
        assert len(records) == 1
        record = RunRecord.model_validate(json.loads(records[0].read_text()))
        assert record.status == RunStatus.FAILED
        assert record.failure_code == FailureCode.ABANDONED
        assert record.tool_name == ARC_TOOL_NAME
        assert record.action_id == "act-1"

    def test_an_orphaned_tool_is_stopped_before_the_run_is_called_over(self, tmp_path: Path) -> None:
        """Marking the campaign FAILED while ARC and its QM children keep
        writing into the workspace would be the same lie the process-tree
        kill exists to prevent."""
        ws = _running_arc_workspace(tmp_path)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            state, report = abandon_arc_run(ws, load_campaign(ws))
            assert report.liveness == RunLiveness.ORPHANED
            assert state.state == CampaignStateValue.FAILED
            assert not _is_running(tree.leader_pid)
            assert not _is_running(tree.grandchild_pid)

    def test_abandoning_is_recorded_in_the_decision_log(self, tmp_path: Path) -> None:
        ws = _running_arc_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        abandon_arc_run(ws, load_campaign(ws))
        events = [e["event"] for e in read_events(ws / "decision_log.jsonl")]
        assert "arc_run_abandoned" in events

    def test_a_campaign_that_is_not_running_cannot_be_abandoned(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        with pytest.raises(InvalidTransitionError):
            abandon_arc_run(ws, load_campaign(ws))

    def test_the_arc_abandon_path_refuses_a_t3_run(self, tmp_path: Path) -> None:
        """The two abandon paths must not cross: each guards its own state."""
        ws = _running_workspace(tmp_path)
        with pytest.raises(InvalidTransitionError):
            abandon_arc_run(ws, load_campaign(ws))
        assert load_state(ws).state == CampaignStateValue.RUNNING_T3

    def test_the_t3_abandon_path_refuses_an_arc_run(self, tmp_path: Path) -> None:
        ws = _running_arc_workspace(tmp_path)
        with pytest.raises(InvalidTransitionError):
            abandon_t3_run(ws, load_campaign(ws))
        assert load_state(ws).state == CampaignStateValue.RUNNING_ARC

    def test_an_abandoned_arc_campaign_can_retry(self, tmp_path: Path) -> None:
        """RECOVERY_TARGETS must map RUNNING_ARC back to APPROVED_FOR_EXECUTION."""
        ws = _running_arc_workspace(tmp_path)
        _strand_active_run(ws, process_group_id=None, command=None)
        abandon_arc_run(ws, load_campaign(ws))
        state = load_state(ws)
        assert can_transition(state.state, CampaignStateValue.APPROVED_FOR_EXECUTION, state.failed_from)


class TestArcRunsAreSupervised:
    """ARC mirror of TestRunsAreSupervised: execute_arc_action owns a
    RunSupervision exactly like the T3 path does."""

    def test_a_real_run_records_and_then_clears_its_supervision(self, tmp_path: Path) -> None:
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        seen: list[ActiveRun | None] = []

        class _Observing:
            submission_mode = SubmissionMode.SUBPROCESS

            def run(
                self,
                workspace_root: Path,
                campaign: Campaign,
                action: PlannedAction,
                on_process_start: Callable[[int, list[str]], None] | None = None,
            ) -> tuple[RunRecord, DiagnosticsV1 | None]:
                assert on_process_start is not None
                on_process_start(4242, ["conda", "run"])
                seen.append(load_active_run(workspace_root))
                now = datetime.now(UTC)
                return RunRecord(
                    run_id="observed",
                    action_id=action.action_id,
                    tool_name=ARC_TOOL_NAME,
                    status=RunStatus.FAILED,
                    failure_code=FailureCode.SUBPROCESS_ERROR,
                    started_at=now,
                    ended_at=now,
                    submission_mode=SubmissionMode.SUBPROCESS,
                    error_message="nope",
                ), None

        execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_Observing())
        assert seen[0] is not None, "the run must be recorded as in flight while it runs"
        assert seen[0].process_group_id == 4242
        assert load_active_run(ws) is None, "a finished run must leave no in-flight record"
        assert not supervisor_is_alive(ws)

    def test_a_second_run_is_refused_while_the_lock_is_held(self, tmp_path: Path) -> None:
        """The lock decides, not the state — same contract as T3."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        with supervise_run(ws, "someone-else"):
            with pytest.raises(RunAlreadySupervisedError):
                execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
            assert load_state(ws).state != CampaignStateValue.RUNNING_ARC, (
                "a refused run must not have moved the campaign"
            )

    def test_a_refused_transition_releases_the_lock(self, tmp_path: Path) -> None:
        """Supervision is taken before the RUNNING_ARC transition; if the
        transition is refused, the lock must not stay held."""
        ws = _ready_workspace(tmp_path)
        plan = load_plan(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        with pytest.raises(InvalidTransitionError):
            execute_arc_action(ws, load_campaign(ws), plan.actions[0], adapter=_ARCSuccessAdapter())
        assert not supervisor_is_alive(ws), "a refused run must release its supervision"
        assert load_active_run(ws) is None
