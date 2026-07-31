"""Tests for the multi-action dispatcher: cursor, states, approvals, locks."""

import hashlib
import json
import os
import shutil
import socket
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from carmel.agents.budget import BudgetLedger, session_budget
from carmel.agents.models import MockModel
from carmel.agents.tools.fetch import MockFetchTool
from carmel.agents.tools.search import MockSearchTool
from carmel.config import AgentConfig
from carmel.schemas import (
    ActionExecutionStatus,
    ActionKind,
    ActionOutcome,
    ApprovalPolicy,
    ApprovalRequirement,
    ApprovalStatus,
    Budgets,
    CampaignInput,
    CampaignStateValue,
    DiagnosticsV1,
    FailureCode,
    InitialMixture,
    MixtureComponent,
    Plan,
    PlannedAction,
    ReactorSystem,
    ReactorType,
    RunRecord,
    RunStatus,
    SubmissionMode,
    TargetObservable,
)
from carmel.schemas.campaign import Campaign
from carmel.schemas.literature import LiteraturePassMode
from carmel.services import execution
from carmel.services.campaigns import (
    create_campaign,
    load_campaign,
    maybe_start_literature_at_creation,
)
from carmel.services.decision_log import read_events
from carmel.services.dispatcher import (
    ActionResult,
    UnsupportedActionKindError,
    default_handlers,
    execute_action,
    execute_next_action,
    recover_workspace,
    validate_plan_shape,
)
from carmel.services.literature import (
    LITERATURE_REPORT_NAME,
    LiteratureDeps,
    ReportSchemaTooNewError,
)
from carmel.services.plan_progress import (
    DISPATCH_LOCK_DIR_NAME,
    ActionInFlightError,
    attempt_result_path,
    init_progress,
    load_progress,
    mark_finished,
    mark_running,
    publish_lock_info,
    record_attempt_result,
    set_approval,
)
from carmel.services.planner import save_plan
from carmel.services.state_machine import load_state, update_state


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    session_budget().reset()
    yield
    session_budget().reset()


def _make_input(name: str = "dispatch-test") -> CampaignInput:
    return CampaignInput(
        workspace_name=name,
        initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
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


def _action(
    action_id: str,
    kind: ActionKind = ActionKind.T3_RUN,
    requirement: ApprovalRequirement = ApprovalRequirement.AUTO_APPROVED,
    blocking: bool = True,
) -> PlannedAction:
    return PlannedAction(
        action_id=action_id,
        kind=kind,
        description=f"action {action_id}",
        estimated_cpu_hours=1.0,
        blocking=blocking,
        rationale="testing",
        approval_requirement=requirement,
    )


def _plan(actions: list[PlannedAction], campaign_id: str) -> Plan:
    return Plan(
        plan_id=str(uuid4()),
        campaign_id=campaign_id,
        created_at=datetime.now(UTC),
        actions=actions,
        rationale="test plan",
        total_estimated_cpu_hours=sum(a.estimated_cpu_hours for a in actions),
        requires_approval=any(a.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL for a in actions),
    )


def _ready_campaign(tmp_path: Path, actions: list[PlannedAction]) -> tuple[Path, Campaign]:
    """A campaign at APPROVED_FOR_EXECUTION with the given plan persisted."""
    ws = tmp_path / "ws"
    campaign = create_campaign(ws, _make_input())
    plan = _plan(actions, campaign.campaign_id)
    save_plan(ws, plan)
    init_progress(ws, plan)
    for target in [
        CampaignStateValue.VALIDATED,
        CampaignStateValue.READY_FOR_PLANNING,
        CampaignStateValue.PLAN_PENDING_APPROVAL,
        CampaignStateValue.APPROVED_FOR_EXECUTION,
    ]:
        update_state(ws, target)
    return ws, campaign


def _run_record(action: PlannedAction, status: RunStatus = RunStatus.SUCCEEDED) -> RunRecord:
    now = datetime.now(UTC)
    return RunRecord(
        run_id=uuid4().hex,
        action_id=action.action_id,
        tool_name="fake",
        status=status,
        failure_code=FailureCode.NONE if status == RunStatus.SUCCEEDED else FailureCode.UNKNOWN,
        started_at=now,
        ended_at=now,
        submission_mode=SubmissionMode.LOCAL,
    )


class _FakeHandler:
    """Recording fake handler returning a fixed outcome per action kind."""

    def __init__(self, outcome: ActionOutcome = ActionOutcome.SUCCEEDED) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    def __call__(self, workspace_root: Path, campaign: Campaign, action: PlannedAction) -> ActionResult:
        self.calls.append(action.action_id)
        status = (
            RunStatus.SUCCEEDED
            if self.outcome in (ActionOutcome.SUCCEEDED, ActionOutcome.NO_GROUNDED_FINDINGS)
            else RunStatus.FAILED
        )
        return ActionResult(
            action_id=action.action_id,
            kind=action.kind,
            run_record=_run_record(action, status),
            outcome=self.outcome,
        )


def _handlers(t3: _FakeHandler | None = None, lit: _FakeHandler | None = None) -> dict[ActionKind, Any]:
    return {
        ActionKind.T3_RUN: t3 or _FakeHandler(),
        ActionKind.LITERATURE_SEARCH: lit or _FakeHandler(),
    }


def _dispatch(ws: Path, campaign: Campaign, *, handlers: dict[ActionKind, Any]) -> ActionResult | None:
    """Start the next action and WAIT for its background half to finish.

    The dispatcher now starts the tool run on a background thread and
    returns a DispatchTicket (adopting main's background execution model);
    these tests exercise whole-dispatch semantics, so they join the ticket
    and assert on the persisted outcome exactly as before.
    """
    ticket = execute_next_action(ws, campaign, handlers=handlers)
    if ticket is None:
        return None
    result = ticket.wait(timeout=60)
    assert ticket.thread is not None and not ticket.thread.is_alive(), "background dispatch did not finish"
    return result


# --------------------------- validate_plan_shape ------------------------------


class TestValidatePlanShape:
    def test_valid_shapes_pass(self) -> None:
        assert validate_plan_shape(_plan([_action("t3")], "c")) == []
        assert validate_plan_shape(_plan([_action("lit", kind=ActionKind.LITERATURE_SEARCH), _action("t3")], "c")) == []

    def test_arc_action_accepted_even_though_this_dispatcher_cannot_run_it(self) -> None:
        # ARC executes on the single-action path in carmel.services.execution, not
        # through this dispatcher's handler registry. This test previously asserted the
        # opposite, because when it was written ARC did not exist on main; judging plan
        # validity by "has a handler HERE" then made save_plan reject every ARC plan.
        assert validate_plan_shape(_plan([_action("arc", kind=ActionKind.ARC_RUN)], "c")) == []

    def test_experiment_action_rejected(self) -> None:
        # EXPERIMENT genuinely has no execution path anywhere -- the distinction the
        # ARC case above turns on.
        problems = validate_plan_shape(_plan([_action("exp", kind=ActionKind.EXPERIMENT)], "c"))
        assert any("nothing can execute" in p for p in problems)

    def test_two_t3_actions_rejected(self) -> None:
        problems = validate_plan_shape(_plan([_action("t3a"), _action("t3b")], "c"))
        assert any("at most one" in p for p in problems)

    def test_literature_after_t3_rejected(self) -> None:
        problems = validate_plan_shape(_plan([_action("t3"), _action("lit", kind=ActionKind.LITERATURE_SEARCH)], "c"))
        assert any("must precede" in p for p in problems)

    def test_duplicate_action_ids_rejected(self) -> None:
        problems = validate_plan_shape(_plan([_action("dup", kind=ActionKind.LITERATURE_SEARCH), _action("dup")], "c"))
        assert any("not unique" in p for p in problems)

    def test_two_literature_actions_rejected(self) -> None:
        """Informational #4: at most one LITERATURE_SEARCH action is allowed,
        mirroring the existing at-most-one-T3_RUN check above.
        """
        problems = validate_plan_shape(
            _plan(
                [
                    _action("lit_a", kind=ActionKind.LITERATURE_SEARCH),
                    _action("lit_b", kind=ActionKind.LITERATURE_SEARCH),
                ],
                "c",
            )
        )
        assert any("at most one" in p for p in problems)

    def test_action_after_t3_rejected_even_when_not_literature(self) -> None:
        """Informational #4: T3_RUN must be LAST among executable actions.

        Literature-before-T3 alone would not catch this: ARC_RUN is also in
        EXECUTABLE_ACTION_KINDS and could otherwise follow T3_RUN without
        tripping the literature-precedes-T3 check.
        """
        problems = validate_plan_shape(_plan([_action("t3"), _action("arc", kind=ActionKind.ARC_RUN)], "c"))
        assert any("must be the last executable action" in p for p in problems)

    def test_save_plan_rejects_bad_shape(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        bad = _plan([_action("exp", kind=ActionKind.EXPERIMENT)], campaign.campaign_id)
        with pytest.raises(ValueError, match="not executable"):
            save_plan(ws, bad)


# --------------------------- execute_action -----------------------------------


class TestExecuteAction:
    def test_routes_by_kind(self, tmp_path: Path) -> None:
        t3, lit = _FakeHandler(), _FakeHandler()
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        execute_action(ws, campaign, _action("t3"), handlers=_handlers(t3=t3, lit=lit))
        assert t3.calls == ["t3"]
        assert lit.calls == []

    def test_unregistered_kind_raises(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        with pytest.raises(UnsupportedActionKindError):
            execute_action(ws, campaign, _action("arc", kind=ActionKind.ARC_RUN))

    def test_default_t3_handler_asserts_its_kind(self, tmp_path: Path) -> None:
        """Fail closed: a mis-routed action must never reach the wrong executor."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        handlers = default_handlers()
        with pytest.raises(UnsupportedActionKindError):
            handlers[ActionKind.T3_RUN](ws, campaign, _action("lit", kind=ActionKind.LITERATURE_SEARCH))
        with pytest.raises(UnsupportedActionKindError):
            handlers[ActionKind.LITERATURE_SEARCH](ws, campaign, _action("t3"))


# --------------------------- multi-action lifecycle ---------------------------


class TestExecuteNextAction:
    def test_two_action_plan_runs_in_order(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        t3, lit = _FakeHandler(), _FakeHandler()

        first = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert first is not None and first.action_id == "lit"
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY
        assert load_progress(ws).cursor == 1

        second = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert second is not None and second.action_id == "t3"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        assert lit.calls == ["lit"]
        assert t3.calls == ["t3"]

        third = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert third is None
        assert t3.calls == ["t3"]  # nothing re-ran

    def test_auto_approved_plan_runs_without_human_step(self, tmp_path: Path) -> None:
        """P0-2 regression guard: seeding PENDING would deadlock this plan."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        t3 = _FakeHandler()
        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert result is not None
        assert t3.calls == ["t3"]
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_pending_action_never_executes(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3", requirement=ApprovalRequirement.REQUIRES_APPROVAL)])
        t3 = _FakeHandler()
        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert result is None
        assert t3.calls == []
        events = read_events(ws / "decision_log.jsonl")
        refused = [e for e in events if e.get("event") == "dispatch.refused"]
        assert refused and "pending" in refused[0]["reason"]

    def test_rejected_action_is_skipped_and_next_runs(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        from carmel.services.plan_progress import set_approval

        set_approval(ws, "lit", ApprovalStatus.REJECTED)
        t3, lit = _FakeHandler(), _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))

        assert result is not None and result.action_id == "t3"
        assert lit.calls == []
        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.SKIPPED
        assert progress.actions[0].outcome == ActionOutcome.REJECTED
        # Rejecting one action of two does not blanket-BLOCK the campaign.
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_all_actions_rejected_blocks(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        from carmel.services.plan_progress import set_approval

        set_approval(ws, "t3", ApprovalStatus.REJECTED)
        result = _dispatch(ws, campaign, handlers=_handlers())
        assert result is None
        assert load_state(ws).state == CampaignStateValue.BLOCKED

    @pytest.mark.parametrize(
        "outcome",
        [
            ActionOutcome.FAILED_NONBLOCKING,
            ActionOutcome.BUDGET_EXCEEDED,
            ActionOutcome.NO_GROUNDED_FINDINGS,
        ],
    )
    def test_nonblocking_literature_outcome_never_stops_t3(self, tmp_path: Path, outcome: ActionOutcome) -> None:
        """P1-11: EVERY non-blocking literature outcome maps to LITERATURE_READY."""
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        t3, lit = _FakeHandler(), _FakeHandler(outcome=outcome)

        first = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert first is not None and first.outcome == outcome
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY

        second = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert second is not None and second.action_id == "t3"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_blocking_failure_requires_explicit_retry_transition(self, tmp_path: Path) -> None:
        """A blocking failure never silently re-runs.

        FAILED is no longer terminal — main added a guarded
        ``FAILED -> APPROVED_FOR_EXECUTION`` retry edge — but WITHOUT that
        explicit operator transition the dispatcher must still refuse to
        touch the failed campaign (updated from the old "no retry in
        increment 1" rule, which main's retry edge superseded).
        """
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        t3 = _FakeHandler(outcome=ActionOutcome.FAILED_BLOCKING)

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert result is not None and result.outcome == ActionOutcome.FAILED_BLOCKING
        assert load_state(ws).state == CampaignStateValue.FAILED

        retry = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert retry is None
        assert t3.calls == ["t3"]  # never re-ran
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_explicit_retry_edge_reruns_the_failed_action_once(self, tmp_path: Path) -> None:
        """Main's guarded retry edge: FAILED (from RUNNING_T3) -> APPROVED_FOR_EXECUTION.

        After the operator takes the explicit retry transition, ``reconcile``
        resets the blocking-failed action to PENDING (keeping its attempt
        history) and the dispatcher runs it once more.
        """
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        t3 = _FakeHandler(outcome=ActionOutcome.FAILED_BLOCKING)
        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert result is not None and result.outcome == ActionOutcome.FAILED_BLOCKING
        assert load_state(ws).state == CampaignStateValue.FAILED
        first_attempts = len(load_progress(ws).actions[0].attempt_ids)

        # The explicit, guarded operator retry (legal: failed from RUNNING_T3).
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="retry")

        t3_ok = _FakeHandler()  # succeeds this time
        retried = _dispatch(ws, campaign, handlers=_handlers(t3=t3_ok))
        assert retried is not None and retried.outcome == ActionOutcome.SUCCEEDED
        assert t3_ok.calls == ["t3"]
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.SUCCEEDED
        assert len(progress.actions[0].attempt_ids) == first_attempts + 1  # history kept

    def test_blocking_failure_mid_plan_stops_later_actions(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=True), _action("t3")],
        )
        t3 = _FakeHandler()
        lit = _FakeHandler(outcome=ActionOutcome.FAILED_BLOCKING)

        _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert load_state(ws).state == CampaignStateValue.FAILED

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))
        assert result is None
        assert t3.calls == []

    def test_completed_action_not_rerun_after_crash(self, tmp_path: Path) -> None:
        """Crash window 2 end-to-end: finished but cursor not advanced."""
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        # Simulate: literature finished, post-transition applied, crash before advance_cursor.
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        update_state(ws, CampaignStateValue.LITERATURE_READY)
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        t3, lit = _FakeHandler(), _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))

        assert result is not None and result.action_id == "t3"
        assert lit.calls == []  # the completed action must never re-run
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_handler_exception_marks_failed_and_records_error(self, tmp_path: Path) -> None:
        """A handler crash marks the attempt FAILED and surfaces on the ticket.

        Under the background execution model the exception can no longer
        re-raise into the caller (nobody is waiting on the thread); it is
        captured on the DispatchTicket instead, and the bookkeeping — FAILED
        attempt, FAILED campaign, released lease — is unchanged.
        """
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])

        def broken(workspace_root: Path, campaign: Campaign, action: PlannedAction) -> ActionResult:
            raise RuntimeError("kaboom")

        ticket = execute_next_action(
            ws, campaign, handlers={ActionKind.T3_RUN: broken, ActionKind.LITERATURE_SEARCH: broken}
        )
        assert ticket is not None
        assert ticket.wait(timeout=60) is None
        assert isinstance(ticket.error, RuntimeError) and "kaboom" in str(ticket.error)
        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED
        assert progress.actions[0].outcome == ActionOutcome.FAILED_BLOCKING
        assert load_state(ws).state == CampaignStateValue.FAILED
        assert not (ws / DISPATCH_LOCK_DIR_NAME).exists()  # lease released

    def test_plan_with_unsupported_kind_refused(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        # Corrupt the plan on disk to contain an ARC action (bypassing save_plan).
        from carmel.services.artifacts import read_json, write_json

        raw = read_json(ws / "plan.json")
        raw["actions"][0]["kind"] = "arc_run"
        write_json(ws / "plan.json", raw)
        with pytest.raises(UnsupportedActionKindError):
            execute_next_action(ws, campaign, handlers=_handlers())

    def test_phase1_plan_json_migrates(self, tmp_path: Path) -> None:
        """A v1 plan.json (no schema_version/blocking/spend) still executes."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        v1_plan = {
            "plan_id": "phase1-plan",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "actions": [
                {
                    "action_id": "a1",
                    "kind": "t3_run",
                    "description": "Initial T3 handshake",
                    "estimated_cpu_hours": 3.0,
                    "estimated_cost": 0.0,
                    "rationale": "Phase 1 baseline",
                    "approval_requirement": "auto_approved",
                    "parameters": {},
                }
            ],
            "rationale": "Deterministic Phase 1 baseline plan",
            "total_estimated_cpu_hours": 3.0,
            "requires_approval": False,
        }
        (ws / "plan.json").write_text(json.dumps(v1_plan))
        for target in [
            CampaignStateValue.VALIDATED,
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
        ]:
            update_state(ws, target)
        t3 = _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))

        assert result is not None and result.action_id == "a1"
        assert t3.calls == ["a1"]  # migration seeded AUTO_APPROVED, not PENDING
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_recorded_human_rejection_blocks_launch_even_when_plan_progress_disagrees(self, tmp_path: Path) -> None:
        """Informational #6: a human REJECTED decision in the decision log
        must veto a launch, even when it diverges from
        ``plan_progress.json``'s own ``approval_status`` (here left
        AUTO_APPROVED). ``has_effective_human_rejection`` is consulted
        precisely because the decision log, not the plan-progress snapshot,
        is authoritative for an explicit human decision (see
        ``execute_next_action``'s "Step 3b" comment on this dispatcher's
        launch gate).

        Uses LITERATURE_SEARCH, not T3_RUN: LITERATURE_SEARCH has no
        execution envelope, so it takes the ``has_effective_human_rejection``
        branch of the gate rather than the full budget-authorization branch
        that every other T3_RUN test in this file already exercises.
        """
        from carmel.services.approvals import record_decision
        from carmel.services.authorization import BudgetExceededError
        from carmel.services.plan_progress import load_progress

        ws, campaign = _ready_campaign(tmp_path, [_action("lit", kind=ActionKind.LITERATURE_SEARCH)])
        assert load_progress(ws).actions[0].approval_status == ApprovalStatus.AUTO_APPROVED
        record_decision(ws, "lit", ApprovalStatus.REJECTED, decided_by="human")

        with pytest.raises(BudgetExceededError, match="rejected"):
            execute_next_action(ws, campaign, handlers=_handlers())

        # The veto fired before any state change or lock was taken.
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_progress(ws).actions[0].approval_status == ApprovalStatus.AUTO_APPROVED


class TestDispatchedRunReservationVisibleInSpend:
    def test_a_dispatched_t3_runs_reservation_is_visible_in_compute_spend_while_in_flight(self, tmp_path: Path) -> None:
        """Finding P1-13: the dispatcher must pass ``estimated_cpu_hours`` to
        ``start_supervision``, matching every other launch site in
        execution.py. Omitting it (the pre-fix behavior) defaults the
        reservation to 0.0, and ``spend.compute_spend`` derives the
        in-flight reservation from exactly that field
        (``reserved = active.estimated_cpu_hours if active is not None else
        0.0``) -- so a dispatcher-launched run in flight would read as
        reserving nothing, silently blinding the dispatcher's own launch
        gate (which itself consults ``compute_spend``) to every run it
        started.

        Exercises this directly: dispatch a T3 action whose handler blocks
        on a ``threading.Event`` until released, so the run is
        provably still in flight (the background thread has not finished)
        when ``compute_spend`` is checked, then release the handler and
        let the dispatch finish cleanly.
        """
        from carmel.services.spend import compute_spend

        action = _action("t3")
        ws, campaign = _ready_campaign(tmp_path, [action])
        assert action.estimated_cpu_hours != 0.0  # else the assertion below would be vacuous

        entered = threading.Event()
        release = threading.Event()

        def _blocking_t3(
            workspace_root: Path,
            campaign: Campaign,
            action: PlannedAction,
            *,
            supervision: Any = None,
        ) -> ActionResult:
            # Advertises `wants_supervision` below, so `execute_action` hands
            # the run lock straight through instead of closing it here as an
            # "unwanted" lock (see execute_action's docstring) -- exactly
            # like the real T3 handler, this is what keeps the reservation
            # open for the run's actual duration rather than for a sliver of
            # a background-thread scheduling race.
            entered.set()
            assert release.wait(timeout=10), "test never released the blocked handler"
            if supervision is not None:
                supervision.close()
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                run_record=_run_record(action, RunStatus.SUCCEEDED),
                outcome=ActionOutcome.SUCCEEDED,
            )

        _blocking_t3.wants_supervision = True  # type: ignore[attr-defined]

        ticket = execute_next_action(ws, campaign, handlers=_handlers(t3=_blocking_t3))

        assert ticket is not None
        assert entered.wait(timeout=10), "background dispatch never started"
        try:
            # The run is still in flight: the handler is parked on `release`
            # and the background thread has not finished. The reservation
            # must already be visible here -- this is exactly the window
            # the pre-fix 0.0 default would have blinded.
            spend = compute_spend(ws)
            assert spend.reserved_cpu_hours == action.estimated_cpu_hours
            assert spend.reserved_cpu_hours != 0.0
        finally:
            release.set()
            result = ticket.wait(timeout=10)

        assert result is not None and result.outcome == ActionOutcome.SUCCEEDED
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1


class TestDispatchTicketWaitTimeout:
    def test_wait_with_timeout_raises_instead_of_returning_none_while_still_running(self, tmp_path: Path) -> None:
        """Informational #5: ``DispatchTicket.wait(timeout=...)`` must raise
        ``TimeoutError`` when the background run has not finished, rather
        than returning ``None``. ``None`` is also the return value for a run
        that finished but *failed* to produce a result, so a caller passing
        a timeout would otherwise have no way to distinguish "still running"
        (unsafe to inspect ``self.error``/workspace state yet) from "failed"
        (safe to inspect both) -- see ``DispatchTicket.wait``'s own
        docstring.

        Exercises this directly: dispatch a T3 action whose handler blocks
        indefinitely (until released), call ``wait`` with a short timeout
        while it is still blocked, and confirm ``TimeoutError`` is raised
        rather than ``None`` being returned.
        """
        action = _action("t3")
        ws, campaign = _ready_campaign(tmp_path, [action])

        entered = threading.Event()
        release = threading.Event()

        def _blocking_t3(
            workspace_root: Path,
            campaign: Campaign,
            action: PlannedAction,
            *,
            supervision: Any = None,
        ) -> ActionResult:
            entered.set()
            assert release.wait(timeout=10), "test never released the blocked handler"
            if supervision is not None:
                supervision.close()
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                run_record=_run_record(action, RunStatus.SUCCEEDED),
                outcome=ActionOutcome.SUCCEEDED,
            )

        _blocking_t3.wants_supervision = True  # type: ignore[attr-defined]

        ticket = execute_next_action(ws, campaign, handlers=_handlers(t3=_blocking_t3))

        assert ticket is not None
        assert entered.wait(timeout=10), "background dispatch never started"
        try:
            with pytest.raises(TimeoutError, match=action.action_id):
                ticket.wait(timeout=0.05)
        finally:
            release.set()
            result = ticket.wait(timeout=10)

        assert result is not None and result.outcome == ActionOutcome.SUCCEEDED


# --------------------------- crash recovery (defects 1-3) ---------------------


class TestCrashRecoveryEndToEnd:
    def test_campaign_recovers_and_completes_after_post_transition_crash(self, tmp_path: Path) -> None:
        """Defect 1 e2e: state left at RUNNING_LITERATURE must not wedge /run.

        Simulates a crash after ``mark_finished`` but BEFORE the dispatcher's
        post-transition by manipulating the persisted state/progress files
        directly, then asserts the campaign RECOVERS and completes: previously
        the next T3 pre-transition (RUNNING_LITERATURE -> RUNNING_T3) was
        illegal and every /run raised or returned None forever.
        """
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        # crash here: no post-transition, no advance_cursor
        t3, lit = _FakeHandler(), _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3, lit=lit))

        assert result is not None and result.action_id == "t3"
        assert lit.calls == []  # the finished literature action never re-ran
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_t3_post_transition_crash_finalises_on_next_run(self, tmp_path: Path) -> None:
        """Defect 1 e2e, T3 flavour: RUNNING_T3 + SUCCEEDED T3 completes, never wedges."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        update_state(ws, CampaignStateValue.RUNNING_T3)
        mark_finished(ws, "t3", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        t3 = _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))

        assert result is None  # the plan is complete...
        assert t3.calls == []  # ...nothing re-ran...
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1  # ...and it is finalised
        assert _dispatch(ws, campaign, handlers=_handlers(t3=t3)) is None

    def test_mid_publication_lock_is_not_broken(self, tmp_path: Path) -> None:
        """Defect 2 e2e: a lock dir without info.json (peer mid-publication) is LIVE."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        (ws / DISPATCH_LOCK_DIR_NAME).mkdir()  # peer crashed the narrow window open
        t3 = _FakeHandler()

        with pytest.raises(ActionInFlightError):
            execute_next_action(ws, campaign, handlers=_handlers(t3=t3))

        assert t3.calls == []
        assert (ws / DISPATCH_LOCK_DIR_NAME).exists()  # and the lock was not broken

    def test_lock_info_published_immediately_with_pid_start(self, tmp_path: Path) -> None:
        """Defect 2: info.json (incl. pid_start) is on disk while the handler runs."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        seen: list[dict[str, Any]] = []

        def observing(workspace_root: Path, campaign_: Campaign, action: PlannedAction) -> ActionResult:
            seen.append(json.loads((workspace_root / DISPATCH_LOCK_DIR_NAME / "info.json").read_text()))
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                run_record=_run_record(action),
                outcome=ActionOutcome.SUCCEEDED,
            )

        _dispatch(ws, campaign, handlers={ActionKind.T3_RUN: observing, ActionKind.LITERATURE_SEARCH: observing})

        assert len(seen) == 1
        assert seen[0]["pid"] == os.getpid()
        assert "pid_start" in seen[0]

    def test_default_t3_handler_stamps_attempt_marker(self, tmp_path: Path) -> None:
        """Defect 3: the handler records the attempt->run mapping when it persists."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])

        result = _dispatch(ws, campaign, handlers=default_handlers(t3_adapter=_SuccessAdapter()))

        assert result is not None
        attempt_id = load_progress(ws).actions[0].attempt_ids[-1]
        marker = json.loads(attempt_result_path(ws, attempt_id).read_text(encoding="utf-8"))
        assert marker["run_id"] == result.run_record.run_id
        assert marker["action_id"] == "t3"
        assert marker["status"] == "succeeded"

    def test_literature_handler_stamps_attempt_marker(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        result = _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert result is not None
        attempt_id = load_progress(ws).actions[0].attempt_ids[-1]
        marker = json.loads(attempt_result_path(ws, attempt_id).read_text(encoding="utf-8"))
        assert marker["run_id"] == result.run_record.run_id
        assert marker["outcome"] == "no_grounded_findings"

    def test_adopted_result_prevents_rerun_and_completes(self, tmp_path: Path) -> None:
        """Defect 3 e2e: a crash after success adopts the persisted run, no re-run."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        update_state(ws, CampaignStateValue.RUNNING_T3)
        mark_running(ws, "t3", "a1")
        record_attempt_result(
            ws,
            action_id="t3",
            attempt_id="a1",
            run_id="r1",
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )
        # crash here: mark_finished never ran
        t3 = _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))

        assert result is None  # the adopted result completed the plan
        assert t3.calls == []  # the succeeded work never re-ran
        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.SUCCEEDED
        assert progress.actions[0].run_id == "r1"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_concurrent_approve_during_skip_does_not_strand_action(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review finding 8 e2e: a racing /approve must not strand an action.

        The skip loop marks a REJECTED action SKIPPED, then advances the
        cursor past it. If a concurrent ``POST /approve`` reopens that SAME
        action (SKIPPED -> PENDING) in the window between ``mark_finished``
        and ``advance_cursor``, the OLD blind ``cursor += 1`` would leave it
        PENDING at an index the cursor has already passed -- stranded
        forever, since the dispatcher only ever looks at ``progress.cursor``
        going forward. The identity-based guard refuses to advance when the
        action at the cursor no longer matches the action_id it just
        finished, or is no longer terminal, so the reopened action is picked
        up on this very dispatch pass instead.
        """
        import carmel.services.dispatcher as dispatcher_module

        ws, campaign = _ready_campaign(
            tmp_path,
            [
                _action(
                    "a",
                    kind=ActionKind.LITERATURE_SEARCH,
                    requirement=ApprovalRequirement.REQUIRES_APPROVAL,
                    blocking=False,
                ),
                _action("b"),
            ],
        )
        set_approval(ws, "a", ApprovalStatus.REJECTED)

        real_mark_finished = dispatcher_module.mark_finished

        def racy_mark_finished(*args: Any, **kwargs: Any) -> Any:
            result = real_mark_finished(*args, **kwargs)
            # Simulates a concurrent POST /approve landing in the exact
            # window between mark_finished and advance_cursor.
            set_approval(ws, "a", ApprovalStatus.APPROVED)
            return result

        monkeypatch.setattr(dispatcher_module, "mark_finished", racy_mark_finished)

        handler = _FakeHandler()
        result = _dispatch(ws, campaign, handlers=_handlers(lit=handler))

        # "a" was reopened to PENDING/APPROVED by the racing approve, and
        # since the cursor was correctly refused an advance past it, THIS
        # dispatch runs "a" -- it is never left stranded behind the cursor.
        assert result is not None and result.action_id == "a"
        assert handler.calls == ["a"]
        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.SUCCEEDED
        assert progress.cursor == 1


class TestRecoverWorkspace:
    """Review finding 1 (dispatcher half): a callable crash-recovery entry point.

    ``reconcile`` used to be reachable only through ``execute_next_action``,
    which every UI route refuses to call while the campaign sits in a
    ``RUNNING_*`` state -- so a crash mid-run could only be repaired by
    hand-editing the persisted JSON. ``recover_workspace`` is the callable
    entry point a UI route can invoke directly while the campaign is still
    ``RUNNING_*``.
    """

    def test_recovers_workspace_stuck_in_running_literature(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        mark_running(ws, "lit", "a1")
        # crash here: no live literature run lock, no dispatch lock, nothing
        # ever persisted a terminal result for this attempt.

        progress = recover_workspace(ws, stale_after_s=0.0)

        assert progress.actions[0].action_id == "lit"
        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED
        assert progress.actions[0].outcome == ActionOutcome.FAILED_NONBLOCKING
        # The missing post-transition was replayed and the terminal
        # projection re-run: a non-blocking literature failure maps to
        # LITERATURE_READY (never stuck at RUNNING_LITERATURE), and T3 can
        # now proceed.
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY

        t3 = _FakeHandler()
        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))
        assert result is not None and result.action_id == "t3"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_refuses_when_attempt_is_still_genuinely_live(self, tmp_path: Path) -> None:
        ws, _campaign = _ready_campaign(tmp_path, [_action("t3")])
        update_state(ws, CampaignStateValue.RUNNING_T3)
        mark_running(ws, "t3", "a1")

        with pytest.raises(ActionInFlightError):
            # stale_after_s large enough that the lease-age fallback still
            # treats the just-created attempt as live.
            recover_workspace(ws, stale_after_s=3600.0)


# --------------------------- dispatcher lock ----------------------------------


class TestDispatcherLock:
    def _write_lock(self, ws: Path, pid: int) -> Path:
        lock = ws / DISPATCH_LOCK_DIR_NAME
        lock.mkdir()
        (lock / "info.json").write_text(
            json.dumps(
                {
                    "pid": pid,
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        return lock

    def test_second_concurrent_dispatch_refused(self, tmp_path: Path) -> None:
        """P0-3: two /run clicks must not both execute the same action."""
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        self._write_lock(ws, os.getppid())  # a live pid that is not ours
        t3 = _FakeHandler()
        with pytest.raises(ActionInFlightError):
            execute_next_action(ws, campaign, handlers=_handlers(t3=t3))
        assert t3.calls == []

    def test_stale_dead_pid_lock_broken_and_logged(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        self._write_lock(ws, 2**22 - 1)  # extremely unlikely to be alive
        t3 = _FakeHandler()

        result = _dispatch(ws, campaign, handlers=_handlers(t3=t3))

        assert result is not None
        assert t3.calls == ["t3"]
        events = read_events(ws / "decision_log.jsonl")
        assert any(e.get("event") == "dispatch.lock_broken" for e in events)

    def test_lock_released_after_run(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        _dispatch(ws, campaign, handlers=_handlers())
        assert not (ws / DISPATCH_LOCK_DIR_NAME).exists()

    def test_reentrant_dispatch_from_handler_refused(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        inner_error: list[Exception] = []

        def reentrant(workspace_root: Path, campaign_: Campaign, action: PlannedAction) -> ActionResult:
            try:
                execute_next_action(workspace_root, campaign_, handlers=_handlers())
            except ActionInFlightError as e:  # our own live lock blocks the inner call
                inner_error.append(e)
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                run_record=_run_record(action),
                outcome=ActionOutcome.SUCCEEDED,
            )

        # Even within the same pid, the live dispatch lock refuses re-entry.
        _dispatch(ws, campaign, handlers={ActionKind.T3_RUN: reentrant, ActionKind.LITERATURE_SEARCH: reentrant})
        assert len(inner_error) == 1

    def test_thread_start_failure_releases_lease_and_fails_action(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 3: a thread.start() failure must not wedge the dispatch lock.

        Before the fix, ``started_background`` was set True BEFORE the call
        that can fail, so the ``finally`` cleanup skipped releasing the
        lease on a start failure — this process's own still-live pid would
        then read as holding the lock forever. Here we make
        ``threading.Thread.start`` raise and assert the lease is released
        and the action is repaired to FAILED instead of left dangling.
        """
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])

        def _raise_on_start(self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", _raise_on_start)

        with pytest.raises(RuntimeError, match="can't start new thread"):
            execute_next_action(ws, campaign, handlers=_handlers())

        # The lease must be released, not leaked behind this process's pid.
        assert not (ws / DISPATCH_LOCK_DIR_NAME).exists()

        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED
        assert progress.actions[0].outcome == ActionOutcome.FAILED_BLOCKING
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_losing_the_steal_race_reacquires_from_scratch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 7 coverage gap: losing the atomic-rename steal race.

        Simulates two racers breaking the SAME stale lock: this frame's
        ``Path.rename`` is intercepted so that, on its first call against
        the dispatch lock dir, a "peer" wins first — moving the stale lock
        dir aside itself and publishing its own fresh, live lock — before
        this frame's rename lands. Our rename must then observe
        ``FileNotFoundError`` (the source is already gone) and MUST treat
        that as losing the race: looping back to re-evaluate from scratch
        rather than assuming it performed the steal. Only one racer may ever
        believe it holds the lease; the loser must see the winner's fresh
        lock as live and refuse with ``ActionInFlightError`` instead of
        deleting it.
        """
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        self._write_lock(ws, 2**22 - 1)  # extremely unlikely to be alive: looks stale
        lock_dir = ws / DISPATCH_LOCK_DIR_NAME

        original_rename = Path.rename
        state = {"intercepted": False}

        def racy_rename(self_path: Path, target: Path, *args: object, **kwargs: object) -> object:
            if self_path == lock_dir and not state["intercepted"]:
                state["intercepted"] = True
                # A peer wins the steal race first: it moves the same stale
                # lock dir aside under its OWN process-unique name and
                # immediately publishes a fresh, live lock in its place —
                # all before our rename call (below) can land.
                peer_target = lock_dir.with_name(f"{lock_dir.name}.stale.peer")
                original_rename(self_path, peer_target)
                lock_dir.mkdir()
                publish_lock_info(lock_dir)
                # Mirrors _acquire_dispatch_lock's own cleanup of the
                # renamed-aside inode it now exclusively owns, so no stray
                # ".stale.peer" directory is left behind by the "winner".
                shutil.rmtree(peer_target, ignore_errors=True)
                raise FileNotFoundError("lock dir already moved by a peer")
            return original_rename(self_path, target, *args, **kwargs)

        monkeypatch.setattr(Path, "rename", racy_rename)

        t3 = _FakeHandler()
        with pytest.raises(ActionInFlightError):
            execute_next_action(ws, campaign, handlers=_handlers(t3=t3))
        assert t3.calls == []
        # Exactly one lock dir exists afterwards (the peer's) — no stray
        # renamed-aside directory left over from either racer.
        assert lock_dir.exists()
        assert not lock_dir.with_name(f"{lock_dir.name}.stale.peer").exists()
        leftovers = list(ws.glob(f"{lock_dir.name}.stale.*"))
        assert leftovers == []


# --------------------------- default handlers ---------------------------------


def _success_diagnostics(campaign_id: str, run_id: str) -> DiagnosticsV1:
    from carmel.schemas import PDepNetworkSelection, ReactionSelection, SpeciesSelection

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
    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple:
        run_id = str(uuid4())
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name="t3",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=now,
            ended_at=now,
            submission_mode=SubmissionMode.SUBPROCESS,
            level_of_theory="b3lyp/6-31g(d,p)",
        )
        return record, _success_diagnostics(campaign.campaign_id, run_id)


class _FailureAdapter:
    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
        on_process_start: Callable[[int, list[str]], None] | None = None,
    ) -> tuple:
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=str(uuid4()),
            action_id=action.action_id,
            tool_name="t3",
            status=RunStatus.FAILED,
            failure_code=FailureCode.SUBPROCESS_ERROR,
            started_at=now,
            ended_at=now,
            submission_mode=SubmissionMode.SUBPROCESS,
            error_message="boom",
        )
        return record, None


class TestDefaultT3Handler:
    def test_t3_success_completes_only_via_dispatcher(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-7: the T3 handler drives _finish_t3_run, never execute_t3_action.

        execute_t3_action is monkeypatched to a tripwire: if any dispatcher
        code path called the wrapper, this test would raise. (Sharing
        _finish_t3_run itself is safe: validate_plan_shape guarantees T3 is
        the LAST action, so its COMPLETED_PHASE1 cannot land while later
        actions are pending.)
        """

        def tripwire(*args: object, **kwargs: object) -> None:
            raise AssertionError("dispatcher must never call execute_t3_action")

        monkeypatch.setattr(execution, "execute_t3_action", tripwire)
        import carmel.services.dispatcher as dispatcher_module

        assert not hasattr(dispatcher_module, "execute_t3_action")

        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        handlers = default_handlers(t3_adapter=_SuccessAdapter())

        result = _dispatch(ws, campaign, handlers=handlers)

        assert result is not None and result.outcome == ActionOutcome.SUCCEEDED
        assert result.diagnostics is not None
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        # Diagnostics + SVGs persisted by the core:
        assert (ws / "diagnostics.json").exists()
        assert (ws / "models" / "species_selection.svg").exists()

    def test_t3_failure_maps_to_failed_blocking(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        handlers = default_handlers(t3_adapter=_FailureAdapter())

        result = _dispatch(ws, campaign, handlers=handlers)

        assert result is not None and result.outcome == ActionOutcome.FAILED_BLOCKING
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_t3_events_still_emitted_via_dispatcher(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [_action("t3")])
        _dispatch(ws, campaign, handlers=default_handlers(t3_adapter=_SuccessAdapter()))
        events = [e.get("event") for e in read_events(ws / "decision_log.jsonl")]
        assert "t3_run_started" in events
        assert "t3_run_finished" in events


def _literature_deps(responses: list[dict[str, Any]]) -> LiteratureDeps:
    config = AgentConfig()
    return LiteratureDeps(
        config=config,
        model=MockModel(responses),
        search=MockSearchTool({}),
        fetch=MockFetchTool({}),
        ledger=BudgetLedger(config.budget),
        verifier_model=MockModel(responses),
    )


class TestDefaultLiteratureHandler:
    def test_no_config_yields_nonblocking_failure_not_spend(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        handlers = default_handlers()  # no agent config, no deps

        result = _dispatch(ws, campaign, handlers=handlers)

        assert result is not None
        assert result.outcome == ActionOutcome.FAILED_NONBLOCKING
        assert result.run_record.failure_code == FailureCode.AGENT_ERROR
        assert (ws / "runs" / f"{result.run_record.run_id}.json").exists()
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY  # T3 not stopped

    def test_barren_run_yields_no_grounded_findings(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])
        handlers = default_handlers(literature_deps=deps)

        result = _dispatch(ws, campaign, handlers=handlers)

        assert result is not None
        assert result.outcome == ActionOutcome.NO_GROUNDED_FINDINGS
        assert result.literature_report is not None
        assert (ws / LITERATURE_REPORT_NAME).exists()
        assert (ws / "runs" / f"{result.run_record.run_id}.json").exists()
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY

    def test_agent_error_yields_nonblocking_failure(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        deps = _literature_deps([])  # exhausted MockModel -> AgentBridgeError -> StopReason.ERROR
        handlers = default_handlers(literature_deps=deps)

        result = _dispatch(ws, campaign, handlers=handlers)

        assert result is not None
        assert result.outcome == ActionOutcome.FAILED_NONBLOCKING
        assert result.run_record.failure_code == FailureCode.AGENT_ERROR
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY


class _SpySearchTool:
    """A search tool that records every call instead of answering one.

    ``MockSearchTool`` returns canned results and keeps no record, so a corpus
    pass that searched would look identical to one that did not. This records
    the queries so "the corpus pass never searches" is an assertion rather than
    an assumption.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 10) -> list[Any]:
        self.queries.append(query)
        return []


def _corpus_deps(responses: list[dict[str, Any]], search: _SpySearchTool | None = None) -> LiteratureDeps:
    """Like ``_literature_deps`` but with an observable search tool."""
    config = AgentConfig()
    return LiteratureDeps(
        config=config,
        model=MockModel(responses),
        search=search or _SpySearchTool(),  # type: ignore[arg-type]
        fetch=MockFetchTool({}),
        ledger=BudgetLedger(config.budget),
        verifier_model=MockModel(responses),
    )


def _store_document(workspace_root: Path, *, text: str, url: str) -> str:
    """Put one real artifact into the workspace's evidence store."""
    from carmel.agents.tools.extract import extract_text
    from carmel.agents.tools.fetch import FetchedArtifact
    from carmel.schemas.literature import ArtifactProvenance
    from carmel.services.evidence import store_artifact

    data = text.encode()
    stored = store_artifact(
        workspace_root,
        data=data,
        artifact=FetchedArtifact(
            url=url,
            final_url=url,
            sha256=hashlib.sha256(data).hexdigest(),
            content_type="text/plain",
            n_bytes=len(data),
            fetched_at=datetime.now(UTC),
        ),
        extracted=extract_text(data, "text/plain"),
        provenance=ArtifactProvenance.MANUAL,
        max_bytes=10_000_000,
    )
    return stored.sha256


_CORPUS_DOC = (
    "A shock tube study of syngas ignition\n"
    "J. Smith and R. Jones (2020)\n"
    "doi: 10.1000/corpus.doi\n\n"
    "Results\n\n"
    "The measured ignition delay time was 1.25 ms at 1000 K behind reflected shock waves.\n"
)


class TestCorpusVersusSearchRouting:
    """The handler must run the pass the action's KIND names.

    ``make_literature_handler`` serves both literature kinds and picks the pass
    from ``action.kind`` on a single line. Nothing pinned that line: mutating it
    to always call ``run_literature_research`` left the whole suite green, which
    would silently convert the offline, reproducible, network-free corpus pass
    into a live search pass -- the PR's central claim.
    """

    def _corpus_action(self, action_id: str = "corpus") -> PlannedAction:
        # A positive token budget is the operator's authorisation; the handler
        # refuses a corpus action without one before it ever picks a pass.
        return _action(action_id, kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False).model_copy(
            update={"estimated_tokens": 50_000}
        )

    def test_a_corpus_action_runs_a_corpus_pass_not_a_search(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(tmp_path, [self._corpus_action(), _action("t3")])
        _store_document(ws, text=_CORPUS_DOC, url="https://example.org/corpus.pdf")
        # A CorpusProposal, which forbids the `queries` channel a search pass uses.
        deps = _corpus_deps([{"findings": [], "done": True}])

        result = _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert result is not None
        assert result.literature_report is not None
        assert result.literature_report.latest.mode == LiteraturePassMode.CORPUS

    def test_a_corpus_dispatch_reaches_neither_search_nor_fetch(self, tmp_path: Path) -> None:
        """Guards the other direction: a search added *inside* the corpus loop.

        The mode assertion above catches mis-routing. This catches the corpus
        loop itself growing a network call, which mode alone would not show.
        """
        ws, campaign = _ready_campaign(tmp_path, [self._corpus_action(), _action("t3")])
        _store_document(ws, text=_CORPUS_DOC, url="https://example.org/corpus.pdf")
        search = _SpySearchTool()
        deps = _corpus_deps([{"findings": [], "done": True}], search=search)

        result = _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert result is not None and result.literature_report is not None
        assert search.queries == [], f"corpus pass searched: {search.queries}"
        # Read usage off the report: the handler REBUILDS the ledger from the
        # action's token budget, so the injected ledger records nothing.
        assert result.literature_report.latest.usage.fetches == 0

    def test_a_corpus_action_without_a_budget_never_reaches_a_pass(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("corpus", kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False), _action("t3")],
        )
        deps = _corpus_deps([{"findings": [], "done": True}])

        result = _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert result is not None
        assert result.outcome == ActionOutcome.FAILED_NONBLOCKING
        assert "names no positive budget" in (result.run_record.error_message or "")
        assert not (ws / LITERATURE_REPORT_NAME).exists()  # nothing ran


class TestAReportFromANewerCarmel:
    """``ReportSchemaTooNewError`` exists to be caught HERE, and nothing checked.

    The subclass is the whole contract: an incompatible-version report is an
    operator-actionable refusal ("upgrade Carmel") surfaced as a typed unrunnable
    action, while genuine corruption -- also a ``ValueError`` -- must keep failing
    loudly. Only the raise site was exercised; the catch, and therefore the
    distinction the subclass exists for, was unverified.
    """

    def _write_report(self, ws: Path, payload: dict[str, Any]) -> None:
        (ws / LITERATURE_REPORT_NAME).write_text(json.dumps(payload), encoding="utf-8")

    def test_it_is_a_typed_refusal_rather_than_a_handler_crash(self, tmp_path: Path) -> None:
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        self._write_report(ws, {"schema_version": 99, "report_id": "r1", "campaign_id": campaign.campaign_id})
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        result = _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert result is not None, "an incompatible report must not crash the dispatch"
        assert result.outcome == ActionOutcome.FAILED_NONBLOCKING
        assert result.run_record.failure_code == FailureCode.AGENT_ERROR
        message = result.run_record.error_message or ""
        assert "schema version 99" in message and "Upgrade Carmel" in message
        # T3 is not stopped by a report this Carmel cannot read.
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY

    def test_the_newer_report_is_never_overwritten(self, tmp_path: Path) -> None:
        """The refusal exists to protect the file; a rewrite would defeat it."""
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        payload = {"schema_version": 99, "report_id": "r1", "campaign_id": campaign.campaign_id}
        self._write_report(ws, payload)
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        _dispatch(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert json.loads((ws / LITERATURE_REPORT_NAME).read_text(encoding="utf-8")) == payload

    def test_genuine_corruption_still_fails_loudly(self, tmp_path: Path) -> None:
        """A ``ValueError`` that is NOT the typed refusal must propagate.

        If the handler caught plain ``ValueError`` instead of the subclass, a
        corrupt report would be reported as a tidy "cannot run" and the operator
        would never learn their evidence file is damaged.
        """
        ws, campaign = _ready_campaign(
            tmp_path,
            [_action("lit", kind=ActionKind.LITERATURE_SEARCH, blocking=False), _action("t3")],
        )
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION

        # Current version, so migration passes it straight through to a
        # validator that cannot make sense of it.
        self._write_report(ws, {"schema_version": CURRENT_REPORT_SCHEMA_VERSION, "passes": "not-a-list"})
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        ticket = execute_next_action(ws, campaign, handlers=default_handlers(literature_deps=deps))

        assert ticket is not None
        assert ticket.wait(timeout=60) is None, "corruption must not produce a tidy result"
        assert isinstance(ticket.error, ValueError)
        assert not isinstance(ticket.error, ReportSchemaTooNewError)


# --------------------------- campaign-creation hook ---------------------------


class TestLiteratureAtCreation:
    def test_no_agent_config_means_no_auto_run(self, tmp_path: Path) -> None:
        """P1-12: creating a campaign must never spend as a side effect."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        assert load_state(ws).state == CampaignStateValue.DRAFT
        assert not (ws / "plan.json").exists()
        assert not (ws / LITERATURE_REPORT_NAME).exists()
        assert maybe_start_literature_at_creation(ws, campaign, None) is None

    def test_toggle_off_means_no_auto_run(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        config = AgentConfig(literature_at_campaign_start=False)
        create_campaign(ws, _make_input(), agent_config=config)
        assert load_state(ws).state == CampaignStateValue.DRAFT
        assert not (ws / "plan.json").exists()

    def test_auto_run_with_injected_deps(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        result = maybe_start_literature_at_creation(ws, campaign, AgentConfig(), deps=deps)

        assert result is not None
        assert result.kind == ActionKind.LITERATURE_SEARCH
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY
        assert (ws / LITERATURE_REPORT_NAME).exists()
        # The T3 action is still pending execution:
        progress = load_progress(ws)
        assert progress.actions[1].execution_status == ActionExecutionStatus.PENDING

    def test_auto_run_defers_when_plan_requires_approval(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        policy = ApprovalPolicy(require_approval_for_literature=True)
        campaign = create_campaign(ws, _make_input(), approval_policy=policy)
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])

        result = maybe_start_literature_at_creation(ws, campaign, AgentConfig(), deps=deps)

        assert result is None  # no spend before consent
        assert load_state(ws).state == CampaignStateValue.PLAN_PENDING_APPROVAL
        assert not (ws / LITERATURE_REPORT_NAME).exists()

    def test_create_campaign_with_config_runs_literature(self, tmp_path: Path) -> None:
        # Default AgentConfig -> MockModel with no canned responses -> a typed
        # ERROR literature report (never a crash, never network).
        ws = tmp_path / "ws"
        create_campaign(ws, _make_input(), agent_config=AgentConfig())
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY
        assert (ws / LITERATURE_REPORT_NAME).exists()

    def test_hook_reuses_existing_plan_when_next_is_literature(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        deps = _literature_deps([{"queries": [], "findings": [], "done": True}])
        first = maybe_start_literature_at_creation(ws, campaign, AgentConfig(), deps=deps)
        assert first is not None
        # Second invocation (e.g. the CLI): next action is T3, so nothing runs.
        deps2 = _literature_deps([{"queries": [], "findings": [], "done": True}])
        second = maybe_start_literature_at_creation(ws, campaign, AgentConfig(), deps=deps2)
        assert second is None

    def test_load_campaign_roundtrip_still_works(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        created = create_campaign(ws, _make_input(), agent_config=AgentConfig())
        loaded = load_campaign(ws)
        assert loaded.campaign_id == created.campaign_id


# --------------------------- workspace_lock deadlock guard --------------------


class TestWorkspaceLockNoReentrantDeadlock:
    """main's ``workspace_lock`` (fcntl.flock, re-opened per call) is NOT
    re-entrant: acquiring it twice in one process blocks forever. Every helper
    that takes it internally (``update_state``, ``append_event``, the plan
    progress mutators, ``reconcile``) must therefore never be called while the
    lock is already held. These tests fail loudly, within a bounded timeout,
    if that invariant is ever broken — instead of hanging CI forever.
    """

    def test_workspace_lock_blocks_cross_acquisition(self, tmp_path: Path) -> None:
        """The hazard is real: a second acquisition blocks until release."""
        import threading

        from carmel.services.decision_log import append_event
        from carmel.services.state_machine import workspace_lock

        entered = threading.Event()
        finished = threading.Event()

        def _append_under_own_lock() -> None:
            entered.set()
            append_event(tmp_path / "decision_log.jsonl", {"event": "lock-probe"})
            finished.set()

        with workspace_lock(tmp_path):
            thread = threading.Thread(target=_append_under_own_lock, daemon=True)
            thread.start()
            assert entered.wait(timeout=10)
            # While we hold the lock the append must be blocked on it.
            assert not finished.wait(timeout=0.5)
        assert finished.wait(timeout=10), "append_event never acquired the released lock"
        thread.join(timeout=10)

    def test_full_dispatch_traverses_every_lock_taker_without_deadlock(self, tmp_path: Path) -> None:
        """A whole dispatch — reconcile, pre-transition, mark_running, the
        background finish (mark_finished, post-transition, advance_cursor,
        repair, decision-log events) — runs to completion under a watchdog.
        If any of those code paths is ever changed to call a lock-taking
        helper while already holding ``workspace_lock``, this dispatch wedges
        and the watchdog fails the test in seconds.
        """
        import threading

        ws, campaign = _ready_campaign(tmp_path, [_action("lit", kind=ActionKind.LITERATURE_SEARCH), _action("t3")])
        done = threading.Event()
        errors: list[BaseException] = []

        def _run_whole_plan() -> None:
            try:
                first = _dispatch(ws, campaign, handlers=_handlers())
                assert first is not None and first.outcome == ActionOutcome.SUCCEEDED
                second = _dispatch(ws, campaign, handlers=_handlers())
                assert second is not None and second.outcome == ActionOutcome.SUCCEEDED
            except BaseException as e:  # noqa: BLE001 -- surfaced via `errors` below
                errors.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=_run_whole_plan, daemon=True)
        thread.start()
        assert done.wait(timeout=60), "dispatch deadlocked: workspace_lock re-acquired re-entrantly"
        thread.join(timeout=10)
        assert not errors, f"dispatch failed: {errors}"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
