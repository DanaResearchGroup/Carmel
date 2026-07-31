"""Tests for per-action plan progress: seeding, approval, crash recovery."""

import json
import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from carmel.schemas import (
    ActionExecutionStatus,
    ActionKind,
    ActionOutcome,
    ApprovalRequirement,
    ApprovalStatus,
    Budgets,
    CampaignInput,
    CampaignStateValue,
    InitialMixture,
    MixtureComponent,
    Plan,
    PlannedAction,
    ReactorSystem,
    ReactorType,
    TargetObservable,
)
from carmel.services.campaigns import create_campaign
from carmel.services.decision_log import read_events
from carmel.services.plan_progress import (
    DISPATCH_LOCK_DIR_NAME,
    LITERATURE_RUN_LOCK_DIR,
    PLAN_PROGRESS_NAME,
    ActionInFlightError,
    advance_cursor,
    aggregate_state,
    attempt_result_path,
    init_progress,
    load_or_init_progress,
    load_progress,
    lock_is_live,
    mark_finished,
    mark_running,
    publish_lock_info,
    reconcile,
    record_attempt_result,
    save_progress,
    set_approval,
)
from carmel.services.recovery import supervise_run
from carmel.services.state_machine import load_state, update_state

OLD = datetime(2020, 1, 1, tzinfo=UTC)


def _make_input(name: str = "progress-test") -> CampaignInput:
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


def _plan(actions: list[PlannedAction], plan_id: str = "p1") -> Plan:
    return Plan(
        plan_id=plan_id,
        campaign_id="c1",
        created_at=datetime.now(UTC),
        actions=actions,
        rationale="test plan",
        total_estimated_cpu_hours=sum(a.estimated_cpu_hours for a in actions),
        requires_approval=any(a.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL for a in actions),
    )


def _two_action_plan(
    lit_requirement: ApprovalRequirement = ApprovalRequirement.AUTO_APPROVED,
    t3_requirement: ApprovalRequirement = ApprovalRequirement.AUTO_APPROVED,
) -> Plan:
    return _plan(
        [
            _action("lit", kind=ActionKind.LITERATURE_SEARCH, requirement=lit_requirement, blocking=False),
            _action("t3", requirement=t3_requirement),
        ]
    )


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    create_campaign(workspace, _make_input())
    return workspace


class TestInitProgress:
    def test_seeds_approval_from_requirement(self, ws: Path) -> None:
        """P0-2 regression guard: AUTO_APPROVED must NOT be seeded as PENDING."""
        plan = _two_action_plan(t3_requirement=ApprovalRequirement.REQUIRES_APPROVAL)
        progress = init_progress(ws, plan)
        assert progress.actions[0].approval_status == ApprovalStatus.AUTO_APPROVED
        assert progress.actions[1].approval_status == ApprovalStatus.PENDING

    def test_seeds_blocking_and_kind(self, ws: Path) -> None:
        progress = init_progress(ws, _two_action_plan())
        assert progress.actions[0].blocking is False
        assert progress.actions[0].kind == ActionKind.LITERATURE_SEARCH
        assert progress.actions[1].blocking is True

    def test_persists_and_roundtrips(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        assert (ws / PLAN_PROGRESS_NAME).exists()
        loaded = load_progress(ws)
        assert loaded.plan_id == "p1"
        assert loaded.cursor == 0
        assert [a.action_id for a in loaded.actions] == ["lit", "t3"]

    def test_load_or_init_initialises_missing(self, ws: Path) -> None:
        progress = load_or_init_progress(ws, _two_action_plan())
        assert (ws / PLAN_PROGRESS_NAME).exists()
        assert progress.actions[0].approval_status == ApprovalStatus.AUTO_APPROVED

    def test_load_or_init_keeps_existing(self, ws: Path) -> None:
        plan = _two_action_plan()
        init_progress(ws, plan)
        advance_cursor(ws)
        progress = load_or_init_progress(ws, plan)
        assert progress.cursor == 1  # not re-initialised

    def test_load_or_init_reinitialises_on_plan_mismatch(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        advance_cursor(ws)
        new_plan = _plan([_action("t3-only")], plan_id="p2")
        progress = load_or_init_progress(ws, new_plan)
        assert progress.plan_id == "p2"
        assert progress.cursor == 0


class TestPlanProgressHelpers:
    def test_next_action_id_and_completion(self, ws: Path) -> None:
        progress = init_progress(ws, _two_action_plan())
        assert progress.next_action_id() == "lit"
        assert not progress.is_complete()
        advance_cursor(ws)
        progress = advance_cursor(ws)
        assert progress.next_action_id() is None
        assert progress.is_complete()

    def test_has_executable_remaining(self, ws: Path) -> None:
        progress = init_progress(ws, _two_action_plan())
        assert progress.has_executable_remaining()
        progress = set_approval(ws, "lit", ApprovalStatus.REJECTED)
        assert progress.has_executable_remaining()  # t3 still executable
        progress = set_approval(ws, "t3", ApprovalStatus.REJECTED)
        assert not progress.has_executable_remaining()


class TestSetApproval:
    def test_sets_status(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan(t3_requirement=ApprovalRequirement.REQUIRES_APPROVAL))
        progress = set_approval(ws, "t3", ApprovalStatus.APPROVED)
        assert progress.actions[1].approval_status == ApprovalStatus.APPROVED

    def test_unknown_action_raises(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        with pytest.raises(KeyError):
            set_approval(ws, "nope", ApprovalStatus.APPROVED)

    def test_approving_rejected_action_unskips_and_rewinds_cursor(self, ws: Path) -> None:
        """P0-5: a rejected-then-approved action must not be stranded behind the cursor."""
        init_progress(ws, _two_action_plan())
        set_approval(ws, "lit", ApprovalStatus.REJECTED)
        mark_finished(
            ws,
            "lit",
            status=ActionExecutionStatus.SKIPPED,
            outcome=ActionOutcome.REJECTED,
        )
        advance_cursor(ws)
        assert load_progress(ws).cursor == 1

        progress = set_approval(ws, "lit", ApprovalStatus.APPROVED)
        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.PENDING
        assert action.outcome == ActionOutcome.NONE
        assert progress.cursor == 0  # rewound to the un-skipped action

    def test_approving_pending_action_does_not_rewind(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan(t3_requirement=ApprovalRequirement.REQUIRES_APPROVAL))
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws)
        progress = set_approval(ws, "t3", ApprovalStatus.APPROVED)
        assert progress.cursor == 1


class TestMarkRunningFinished:
    def test_mark_running_records_attempt(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        progress = mark_running(ws, "lit", "attempt-1")
        assert progress.actions[0].execution_status == ActionExecutionStatus.RUNNING
        assert progress.actions[0].attempt_ids == ["attempt-1"]

    def test_mark_running_twice_raises_in_flight(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "attempt-1")
        with pytest.raises(ActionInFlightError):
            mark_running(ws, "lit", "attempt-2")

    def test_mark_running_terminal_raises(self, ws: Path) -> None:
        """A completed action must never re-run."""
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        with pytest.raises(ValueError, match="never re-run"):
            mark_running(ws, "lit", "attempt-2")

    def test_mark_finished_records_fields(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        progress = mark_finished(
            ws,
            "t3",
            status=ActionExecutionStatus.FAILED,
            outcome=ActionOutcome.FAILED_BLOCKING,
            run_id="r9",
            notes="boom",
        )
        action = progress.actions[1]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.outcome == ActionOutcome.FAILED_BLOCKING
        assert action.run_id == "r9"
        assert action.notes == "boom"

    def test_advance_cursor_stops_at_end(self, ws: Path) -> None:
        init_progress(ws, _plan([_action("only")]))
        progress = advance_cursor(ws)
        assert progress.cursor == 1
        progress = advance_cursor(ws)
        assert progress.cursor == 1  # never past len(actions)

    def test_advance_cursor_with_matching_action_id_advances(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        progress = advance_cursor(ws, "lit")
        assert progress.cursor == 1

    def test_advance_cursor_with_stale_action_id_does_not_advance(self, ws: Path) -> None:
        """Finding 8: an identity check that no longer matches must not advance blindly.

        This is what closes the interleaving window: if the action that was
        at the cursor when the caller finished it is no longer the action at
        the cursor (e.g. a concurrent approval spliced a different, or the
        same, action back to PENDING in between), a blind increment would
        skip past it and strand it behind the cursor forever.
        """
        init_progress(ws, _two_action_plan())
        progress = advance_cursor(ws, "t3")  # "t3" is not at the cursor (cursor is at "lit")
        assert progress.cursor == 0

    def test_advance_cursor_without_action_id_preserves_old_blind_behavior(self, ws: Path) -> None:
        """Callers that have not migrated keep the old unconditional advance."""
        init_progress(ws, _two_action_plan())
        progress = advance_cursor(ws)
        assert progress.cursor == 1

    def test_advance_cursor_is_idempotent_for_same_action_id(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        progress = advance_cursor(ws, "lit")
        assert progress.cursor == 1
        # Calling again with the SAME action_id must not advance a second
        # time: "lit" is no longer at the cursor ("t3" is now).
        progress = advance_cursor(ws, "lit")
        assert progress.cursor == 1

    def test_advance_cursor_stranding_scenario_is_prevented(self, ws: Path) -> None:
        """Reproduces finding 8's interleaving: a concurrent re-approval must
        not be skipped over by a blind cursor advance.

        Plan [a0, a1], cursor at 0. The dispatcher marks a0 SKIPPED (as
        rejected) in one lock window. Before the dispatcher calls
        advance_cursor, a concurrent approve for a0 resets it to PENDING +
        APPROVED (set_approval does not take the dispatch lock). With the
        OLD blind advance_cursor(ws), the cursor would move to 1 and strand
        a0 PENDING behind the cursor forever.

        The action_id at the cursor is unchanged by this race ("a0" both
        times, same index), so identity alone would not catch it — this is
        exactly why the fix also requires the action at the cursor to still
        be terminal: a0 is PENDING again by the time advance_cursor(ws,
        "a0") runs, so the cursor must not move.
        """
        init_progress(ws, _plan([_action("a0", blocking=False), _action("a1")]))
        set_approval(ws, "a0", ApprovalStatus.REJECTED)
        mark_finished(ws, "a0", status=ActionExecutionStatus.SKIPPED, outcome=ActionOutcome.REJECTED)
        # Concurrent re-approval race: resets a0 back to PENDING + APPROVED.
        set_approval(ws, "a0", ApprovalStatus.APPROVED)
        # The dispatcher's advance_cursor call believes it just finished "a0".
        progress = advance_cursor(ws, "a0")
        # a0 is PENDING again (not terminal): the cursor must NOT have advanced
        # past it, or it would be stranded behind the cursor forever.
        assert progress.cursor == 0
        assert progress.actions[0].execution_status == ActionExecutionStatus.PENDING
        assert progress.actions[0].approval_status == ApprovalStatus.APPROVED


class TestAggregateState:
    def _finished(self, ws: Path, action_id: str, outcome: ActionOutcome) -> None:
        status = (
            ActionExecutionStatus.SUCCEEDED
            if outcome in (ActionOutcome.SUCCEEDED, ActionOutcome.NO_GROUNDED_FINDINGS)
            else ActionExecutionStatus.FAILED
        )
        mark_finished(ws, action_id, status=status, outcome=outcome)
        advance_cursor(ws)

    def test_in_flight_returns_none(self, ws: Path) -> None:
        progress = init_progress(ws, _two_action_plan())
        assert aggregate_state(progress) is None

    def test_all_succeeded_completes(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        self._finished(ws, "lit", ActionOutcome.SUCCEEDED)
        self._finished(ws, "t3", ActionOutcome.SUCCEEDED)
        assert aggregate_state(load_progress(ws)) == CampaignStateValue.COMPLETED_PHASE1

    def test_blocking_failure_is_failed(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        self._finished(ws, "lit", ActionOutcome.SUCCEEDED)
        self._finished(ws, "t3", ActionOutcome.FAILED_BLOCKING)
        assert aggregate_state(load_progress(ws)) == CampaignStateValue.FAILED

    def test_every_action_skipped_is_not_completed(self, ws: Path) -> None:
        """P1-10: a plan whose every action was skipped must NOT report success."""
        init_progress(ws, _two_action_plan())
        set_approval(ws, "lit", ApprovalStatus.REJECTED)
        set_approval(ws, "t3", ApprovalStatus.REJECTED)
        for action_id in ("lit", "t3"):
            mark_finished(ws, action_id, status=ActionExecutionStatus.SKIPPED, outcome=ActionOutcome.REJECTED)
            advance_cursor(ws)
        state = aggregate_state(load_progress(ws))
        assert state != CampaignStateValue.COMPLETED_PHASE1
        assert state == CampaignStateValue.BLOCKED

    def test_one_rejected_one_succeeded_still_completes(self, ws: Path) -> None:
        """Rejecting one action of two does not blanket-BLOCK the campaign."""
        init_progress(ws, _two_action_plan())
        set_approval(ws, "lit", ApprovalStatus.REJECTED)
        mark_finished(ws, "lit", status=ActionExecutionStatus.SKIPPED, outcome=ActionOutcome.REJECTED)
        advance_cursor(ws)
        self._finished(ws, "t3", ActionOutcome.SUCCEEDED)
        assert aggregate_state(load_progress(ws)) == CampaignStateValue.COMPLETED_PHASE1

    def test_cursor_past_end_but_nothing_succeeded_returns_none(self, ws: Path) -> None:
        """A reconcile bug is surfaced (None), not hidden behind a terminal state."""
        progress = init_progress(ws, _plan([_action("t3-only")]))
        progress.cursor = 1  # corrupt: past the end with the action still PENDING
        save_progress(ws, progress)
        assert aggregate_state(load_progress(ws)) is None

    def test_nonblocking_failure_still_completes(self, ws: Path) -> None:
        init_progress(ws, _two_action_plan())
        self._finished(ws, "lit", ActionOutcome.FAILED_NONBLOCKING)
        self._finished(ws, "t3", ActionOutcome.SUCCEEDED)
        assert aggregate_state(load_progress(ws)) == CampaignStateValue.COMPLETED_PHASE1


def _to_approved_for_execution(ws: Path) -> None:
    for target in [
        CampaignStateValue.VALIDATED,
        CampaignStateValue.READY_FOR_PLANNING,
        CampaignStateValue.PLAN_PENDING_APPROVAL,
        CampaignStateValue.APPROVED_FOR_EXECUTION,
    ]:
        update_state(ws, target)


def _age_action(ws: Path, index: int) -> None:
    """Backdate one action's lease far past any staleness horizon."""
    progress = load_progress(ws)
    progress.actions[index] = progress.actions[index].model_copy(update={"updated_at": OLD})
    save_progress(ws, progress)


class TestReconcile:
    def test_crash_window_running_stale_is_repaired(self, ws: Path) -> None:
        """Window 1: progress left RUNNING by a crashed attempt."""
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        _age_action(ws, 0)

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.outcome == ActionOutcome.FAILED_NONBLOCKING  # blocking=False
        assert action.notes == "recovered from interrupted attempt"
        events = read_events(ws / "decision_log.jsonl")
        assert any(e.get("event") == "dispatch.attempt_recovered" for e in events)

    def test_crash_window_running_blocking_fails_blocking(self, ws: Path) -> None:
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)

        progress = reconcile(ws)

        assert progress.actions[0].outcome == ActionOutcome.FAILED_BLOCKING
        # Terminal projection FAILED is legal from RUNNING_T3 and gets applied.
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_live_attempt_raises_in_flight(self, ws: Path) -> None:
        """A recent lease is a live attempt: never a silent second run."""
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")  # updated_at is now -> lease is fresh

        with pytest.raises(ActionInFlightError):
            reconcile(ws)

    def test_live_literature_lock_raises_even_in_dispatch_lock(self, ws: Path) -> None:
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        _age_action(ws, 0)
        lock_dir = ws / LITERATURE_RUN_LOCK_DIR
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": os.getppid(),  # a live pid that is not ours
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        with pytest.raises(ActionInFlightError):
            reconcile(ws, in_dispatch_lock=True)

    def test_live_literature_lock_covers_a_corpus_pass_too(self, ws: Path) -> None:
        """Spar round 7, P1. The liveness check named LITERATURE_SEARCH explicitly, so
        a corpus pass -- which takes the very same literature run lock -- was invisible
        to it. A running corpus pass could therefore be judged dead, marked FAILED, and
        re-run while the first was still writing: the silent second run reconcile
        exists to prevent.
        """
        _to_approved_for_execution(ws)
        init_progress(
            ws,
            _plan(
                [
                    _action(
                        "corpus",
                        kind=ActionKind.LITERATURE_CORPUS_PASS,
                        requirement=ApprovalRequirement.AUTO_APPROVED,
                        blocking=False,
                    ),
                    _action("t3"),
                ]
            ),
        )
        mark_running(ws, "corpus", "a1")
        _age_action(ws, 0)
        lock_dir = ws / LITERATURE_RUN_LOCK_DIR
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": os.getppid(),  # a live pid that is not ours
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        with pytest.raises(ActionInFlightError):
            reconcile(ws, in_dispatch_lock=True)

    def test_in_dispatch_lock_repairs_fresh_lease_without_live_lock(self, ws: Path) -> None:
        """Holding the exclusive dispatch lock proves no dispatcher attempt is live."""
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")  # fresh lease

        progress = reconcile(ws, in_dispatch_lock=True)

        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED

    def test_live_t3_supervision_raises_even_in_dispatch_lock(self, ws: Path) -> None:
        """Finding 17: reconcile must consult real T3 liveness, not just the

        dispatch lock. ``in_dispatch_lock=True`` is the ONLY production call
        site (the dispatcher already holds the exclusive workspace dispatch
        lock while calling reconcile), so before this fix every branch of
        ``_attempt_is_live`` was skipped for a T3_RUN action and it always
        returned False -- reconcile would mark a live T3 run FAILED while
        the process tree was still writing into the workspace, exactly the
        lie carmel.services.recovery exists to prevent.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)  # stale lease: only real T3 supervision should save it now

        with supervise_run(ws, "t3-only"), pytest.raises(ActionInFlightError):
            reconcile(ws, in_dispatch_lock=True)

    def test_live_t3_supervision_raises_without_dispatch_lock_too(self, ws: Path) -> None:
        """Same guard applies on the (currently unused) non-dispatch-lock path."""
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)

        with supervise_run(ws, "t3-only"), pytest.raises(ActionInFlightError):
            reconcile(ws)

    def test_stale_t3_lease_without_real_supervision_is_still_recovered(self, ws: Path) -> None:
        """Regression guard: with no live T3 supervisor at all, a stale RUNNING

        T3 action is still recovered (marked FAILED) exactly as before --
        the new T3-liveness check must not make reconcile MORE conservative
        than it needs to be when nothing is actually running.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)

        progress = reconcile(ws, in_dispatch_lock=True)

        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED
        assert progress.actions[0].outcome == ActionOutcome.FAILED_BLOCKING

    def test_crash_window_finished_but_cursor_not_advanced(self, ws: Path) -> None:
        """Window 2: the cursor is advanced past a terminal action (no re-run)."""
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        # crash before advance_cursor: cursor still 0

        progress = reconcile(ws)

        assert progress.cursor == 1
        assert progress.actions[0].execution_status == ActionExecutionStatus.SUCCEEDED

    def test_crash_window_post_transition_done_but_cursor_not_advanced(self, ws: Path) -> None:
        """Window 3: post-transition applied, then crash before advance_cursor."""
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        update_state(ws, CampaignStateValue.DIAGNOSTICS_READY)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_finished(ws, "t3-only", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)

        progress = reconcile(ws)

        assert progress.cursor == 1
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_crash_window_cursor_past_end_state_not_terminal(self, ws: Path) -> None:
        """Window 4: cursor past the end but the campaign state was never finalised."""
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        update_state(ws, CampaignStateValue.DIAGNOSTICS_READY)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_finished(ws, "t3-only", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws)

        reconcile(ws)

        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_crash_between_mark_finished_and_post_transition_recovers(self, ws: Path) -> None:
        """Defect 1 regression: a crash after ``mark_finished`` but before the
        post-transition must NOT wedge the campaign.

        Persisted state is still RUNNING_T3 while the T3 action is already
        SUCCEEDED. Reconcile replays the missing post-transition
        (RUNNING_T3 -> DIAGNOSTICS_READY) and the terminal projection then
        completes the campaign — previously this logged a mismatch and every
        future /run returned None forever.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_finished(ws, "t3-only", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        # crash here: no post-transition, no advance_cursor, no terminal projection

        progress = reconcile(ws)

        assert progress.cursor == 1
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        events = read_events(ws / "decision_log.jsonl")
        assert not any(e.get("event") == "dispatch.state_mismatch" for e in events)
        # Idempotent: a second reconcile changes nothing.
        second = reconcile(ws)
        assert second.cursor == 1
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_crash_window_literature_post_transition_replayed(self, ws: Path) -> None:
        """Defect 1, literature flavour: RUNNING_LITERATURE + finished lit action.

        Without the replay, the next T3 pre-transition
        (RUNNING_LITERATURE -> RUNNING_T3) is illegal and the campaign wedges.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)

        progress = reconcile(ws)

        assert progress.cursor == 1  # advanced past the finished lit action
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY

    def test_crash_window_corpus_pass_post_transition_replayed(self, ws: Path) -> None:
        """Spar round 7, P1. The state-to-kind map named LITERATURE_SEARCH alone, so a
        crashed CORPUS pass left the campaign in RUNNING_LITERATURE with no finished
        action the replay would accept. It wedged there permanently: the next T3
        pre-transition is illegal from RUNNING_LITERATURE, and nothing else ever
        revisits it.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        init_progress(
            ws,
            _plan(
                [
                    _action(
                        "corpus",
                        kind=ActionKind.LITERATURE_CORPUS_PASS,
                        requirement=ApprovalRequirement.AUTO_APPROVED,
                        blocking=False,
                    ),
                    _action("t3"),
                ]
            ),
        )
        mark_finished(ws, "corpus", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)

        progress = reconcile(ws)

        assert progress.cursor == 1, "the cursor never advanced past the finished corpus pass"
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY, (
            "the campaign is still wedged in RUNNING_LITERATURE"
        )

    def test_post_transition_replay_loses_a_race_without_crashing(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``_replay_missing_post_transition`` checks ``can_transition`` and then calls
        ``update_state`` with no lock held across the two -- a concurrent repairer (or
        dispatch) can legally move the persisted state in between. This must be caught
        the same way ``repair_campaign_state``'s sibling guard catches it, not crash the
        whole reconcile pass.

        Deterministically injects the interleaving: monkeypatch ``update_state`` (as
        imported into ``plan_progress``) so that, on the call ``_replay_missing_post_
        transition`` makes, a "concurrent" writer moves the persisted state to FAILED
        (a legal ``RUNNING_LITERATURE -> FAILED`` edge) immediately before the real
        transition is attempted -- so the real call now targets an now-illegal edge
        (``FAILED -> LITERATURE_READY`` does not exist) and raises.
        """
        import carmel.services.plan_progress as plan_progress_module
        from carmel.services.state_machine import update_state as real_update_state

        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        init_progress(ws, _two_action_plan())
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)

        raced = {"done": False}

        def racing_update_state(workspace_root: Path, target: CampaignStateValue, notes: str | None = None):
            if not raced["done"] and target == CampaignStateValue.LITERATURE_READY:
                raced["done"] = True
                # Simulate a concurrent winner landing between our stale
                # ``can_transition`` check and this call.
                real_update_state(workspace_root, CampaignStateValue.FAILED, notes="concurrent winner")
            return real_update_state(workspace_root, target, notes=notes)

        monkeypatch.setattr(plan_progress_module, "update_state", racing_update_state)

        with caplog.at_level("WARNING"):
            reconcile(ws)

        assert raced["done"], "the race was never actually exercised"
        assert load_state(ws).state == CampaignStateValue.FAILED, "the winner's state must stand"
        assert any("lost a race" in message for message in caplog.messages)

    def test_truly_illegal_mismatch_records_warning_not_forced(self, ws: Path) -> None:
        """When NO legal path to the projection exists, a warning is logged, not forced."""
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        update_state(ws, CampaignStateValue.DIAGNOSTICS_READY)
        update_state(ws, CampaignStateValue.COMPLETED_PHASE1)
        init_progress(ws, _plan([_action("t3-only")]))
        # Corrupt the progress: the action FAILED although the state is terminal
        # COMPLETED_PHASE1, from which no transition exists at all.
        mark_finished(ws, "t3-only", status=ActionExecutionStatus.FAILED, outcome=ActionOutcome.FAILED_BLOCKING)
        advance_cursor(ws)

        reconcile(ws)

        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1  # not forced
        events = read_events(ws / "decision_log.jsonl")
        mismatches = [e for e in events if e.get("event") == "dispatch.state_mismatch"]
        assert mismatches
        assert mismatches[0]["projected_state"] == "failed"

    def test_reconcile_adopts_persisted_success(self, ws: Path) -> None:
        """Defect 3 regression: a persisted successful result is ADOPTED, not discarded.

        Crash after the handler persisted its work (attempt marker written)
        but before ``mark_finished``: reconcile must mark the action SUCCEEDED
        with the real run_id instead of FAILED.
        """
        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        record_attempt_result(
            ws,
            action_id="t3-only",
            attempt_id="a1",
            run_id="r1",
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.SUCCEEDED
        assert action.outcome == ActionOutcome.SUCCEEDED
        assert action.run_id == "r1"
        assert progress.cursor == 1
        # Replay + terminal projection finish the campaign from the adopted result.
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        events = read_events(ws / "decision_log.jsonl")
        assert any(e.get("event") == "dispatch.attempt_adopted" for e in events)
        assert not any(e.get("event") == "dispatch.attempt_recovered" for e in events)

    def test_reconcile_adopts_persisted_failure_with_real_run_id(self, ws: Path) -> None:
        """A persisted FAILED result is adopted too — the real run_id is kept."""
        _to_approved_for_execution(ws)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        record_attempt_result(
            ws,
            action_id="t3-only",
            attempt_id="a1",
            run_id="r2",
            status=ActionExecutionStatus.FAILED,
            outcome=ActionOutcome.FAILED_BLOCKING,
        )

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.outcome == ActionOutcome.FAILED_BLOCKING
        assert action.run_id == "r2"

    def test_marker_for_wrong_action_is_ignored(self, ws: Path) -> None:
        """A marker whose action_id does not match falls back to the FAILED repair."""
        _to_approved_for_execution(ws)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        record_attempt_result(
            ws,
            action_id="some-other-action",
            attempt_id="a1",
            run_id="r3",
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.notes == "recovered from interrupted attempt"

    def test_marker_for_older_attempt_is_ignored(self, ws: Path) -> None:
        """Only the LATEST attempt's marker may be adopted."""
        _to_approved_for_execution(ws)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        record_attempt_result(
            ws,
            action_id="t3-only",
            attempt_id="a0",  # not the attempt that is being reconciled
            run_id="r0",
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )
        assert attempt_result_path(ws, "a0").exists()

        progress = reconcile(ws)

        assert progress.actions[0].execution_status == ActionExecutionStatus.FAILED

    def test_reconcile_ignores_corrupt_attempt_result_marker(self, ws: Path) -> None:
        """A truncated/corrupt marker file must not crash reconcile.

        ``read_attempt_result`` is best-effort by design (``OSError`` and
        ``json.JSONDecodeError`` both return None); this hand-writes a marker
        that isn't valid JSON at all, standing in for a crash mid-write, and
        checks reconcile falls back to the ordinary FAILED-recovery path
        instead of propagating the parse error.
        """
        _to_approved_for_execution(ws)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        marker_path = attempt_result_path(ws, "a1")
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("{not valid json", encoding="utf-8")

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.notes == "recovered from interrupted attempt"

    def test_reconcile_ignores_attempt_result_marker_with_invalid_enum_values(self, ws: Path) -> None:
        """A syntactically valid marker with a nonsense status/outcome must not crash.

        ``_adopt_attempt_result`` constructs ``ActionExecutionStatus``/``ActionOutcome``
        from the recorded strings and catches ``KeyError``/``ValueError`` from that
        construction; this writes a marker whose ``status`` is not a member of the
        enum at all (e.g. a stale marker from a since-renamed status) and checks
        reconcile falls back to FAILED-recovery instead of raising.
        """
        _to_approved_for_execution(ws)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_running(ws, "t3-only", "a1")
        _age_action(ws, 0)
        marker_path = attempt_result_path(ws, "a1")
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "action_id": "t3-only",
                    "attempt_id": "a1",
                    "run_id": "r-bogus",
                    "status": "not_a_real_status",
                    "outcome": "not_a_real_outcome",
                    "recorded_at": OLD.isoformat(),
                }
            ),
            encoding="utf-8",
        )

        progress = reconcile(ws)

        action = progress.actions[0]
        assert action.execution_status == ActionExecutionStatus.FAILED
        assert action.notes == "recovered from interrupted attempt"

    def test_reconcile_is_idempotent(self, ws: Path) -> None:
        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        _age_action(ws, 0)

        first = reconcile(ws)
        second = reconcile(ws)

        assert second.model_dump(exclude={"updated_at"}) == first.model_dump(exclude={"updated_at"})
        events = read_events(ws / "decision_log.jsonl")
        assert sum(1 for e in events if e.get("event") == "dispatch.attempt_recovered") == 1


class TestRepairCampaignState:
    def test_multi_step_repair_loses_a_race_without_crashing(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``repair_campaign_state`` walks a multi-step path with no lock held across
        the individual ``update_state`` calls -- a concurrent repairer (or dispatch)
        can legally move the persisted state in the middle of that walk. This is the
        sibling of ``test_post_transition_replay_loses_a_race_without_crashing`` but
        exercises ``repair_campaign_state`` itself (its own ``InvalidTransitionError``
        guard around the ``for step in path`` loop), not ``_replay_missing_post_
        transition``.

        Deterministically injects the interleaving: the projected terminal state is
        ``COMPLETED_PHASE1``, reached from ``RUNNING_T3`` via the two-step legal path
        ``RUNNING_T3 -> DIAGNOSTICS_READY -> COMPLETED_PHASE1``. The patched
        ``update_state`` lets the first step land normally, then -- immediately before
        the second step -- a "concurrent" winner moves the state to FAILED (a legal
        ``DIAGNOSTICS_READY -> FAILED`` edge), so the real second step now targets an
        illegal ``FAILED -> COMPLETED_PHASE1`` edge and raises.
        """
        import carmel.services.plan_progress as plan_progress_module
        from carmel.services.state_machine import update_state as real_update_state

        _to_approved_for_execution(ws)
        update_state(ws, CampaignStateValue.RUNNING_T3)
        init_progress(ws, _plan([_action("t3-only")]))
        mark_finished(ws, "t3-only", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws)
        progress = load_progress(ws)
        assert aggregate_state(progress) == CampaignStateValue.COMPLETED_PHASE1

        raced = {"done": False}

        def racing_update_state(workspace_root: Path, target: CampaignStateValue, notes: str | None = None):
            if not raced["done"] and target == CampaignStateValue.COMPLETED_PHASE1:
                raced["done"] = True
                # Simulate a concurrent winner landing between the first and
                # second steps of the repair's multi-step walk.
                real_update_state(workspace_root, CampaignStateValue.FAILED, notes="concurrent winner")
            return real_update_state(workspace_root, target, notes=notes)

        monkeypatch.setattr(plan_progress_module, "update_state", racing_update_state)

        with caplog.at_level("WARNING"):
            plan_progress_module.repair_campaign_state(ws, progress)

        assert raced["done"], "the race was never actually exercised"
        assert load_state(ws).state == CampaignStateValue.FAILED, "the winner's state must stand"
        assert any("lost a race" in message for message in caplog.messages)


class TestLockIsLive:
    def _write_info(self, lock_dir: Path, **overrides: object) -> None:
        info = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        info.update(overrides)
        lock_dir.mkdir(parents=True, exist_ok=True)
        (lock_dir / "info.json").write_text(json.dumps(info))

    def test_missing_lock_not_live(self, tmp_path: Path) -> None:
        assert not lock_is_live(tmp_path / DISPATCH_LOCK_DIR_NAME, stale_after_s=10.0)

    def test_own_pid_is_live(self, tmp_path: Path) -> None:
        """Two dispatch attempts in one process must not break each other's lock."""
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        self._write_info(lock)
        assert lock_is_live(lock, stale_after_s=10.0)

    def test_dead_pid_not_live(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        self._write_info(lock, pid=2**22 - 1)  # extremely unlikely to be alive
        assert not lock_is_live(lock, stale_after_s=10.0)

    def test_live_other_pid_is_live(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        self._write_info(lock, pid=os.getppid())
        assert lock_is_live(lock, stale_after_s=0.0)  # live pid wins over age

    def test_foreign_host_recent_is_live(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        self._write_info(lock, hostname="another-host")
        assert lock_is_live(lock, stale_after_s=3600.0)

    def test_foreign_host_old_is_stale(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        old = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        self._write_info(lock, hostname="another-host", started_at=old)
        assert not lock_is_live(lock, stale_after_s=3600.0)

    def _backdate_dir(self, lock_dir: Path, seconds: float) -> None:
        """Age the lock directory's mtime past the publication grace period."""
        old = datetime.now(UTC).timestamp() - seconds
        os.utime(lock_dir, (old, old))

    def test_unreadable_info_within_grace_is_live(self, tmp_path: Path) -> None:
        """Defect 2 regression: malformed metadata on a FRESH lock fails CLOSED.

        A peer that just ran ``mkdir`` may not have finished publishing
        ``info.json`` yet; treating that window as stale let two processes
        both acquire the lock and double-run the same action.
        """
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        (lock / "info.json").write_text("{not json")
        assert lock_is_live(lock, stale_after_s=3600.0)

    def test_missing_info_within_grace_is_live(self, tmp_path: Path) -> None:
        """A bare lock dir (mkdir done, info.json not yet written) is LIVE."""
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        assert lock_is_live(lock, stale_after_s=3600.0)

    def test_unreadable_info_past_grace_is_stale(self, tmp_path: Path) -> None:
        """Past the grace period, unreadable metadata may be treated as stale."""
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        (lock / "info.json").write_text("{not json")
        self._backdate_dir(lock, 120.0)  # default grace is 30 s
        assert not lock_is_live(lock, stale_after_s=3600.0)

    def test_missing_info_past_grace_is_stale(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        self._backdate_dir(lock, 120.0)
        assert not lock_is_live(lock, stale_after_s=3600.0)

    def test_grace_window_is_configurable(self, tmp_path: Path) -> None:
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        self._backdate_dir(lock, 120.0)
        assert lock_is_live(lock, stale_after_s=3600.0, lock_grace_s=300.0)

    def test_pid_reuse_after_reboot_not_live(self, tmp_path: Path) -> None:
        """Defect 2: a live pid with a DIFFERENT start time is a reused pid.

        The recorded holder died (e.g. the machine rebooted) and an unrelated
        process now wears its pid — the lock must not look live forever.
        """
        if not Path(f"/proc/{os.getpid()}/stat").exists():
            pytest.skip("/proc not available on this platform")
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        # Our own (verifiably live) pid, but an impossible recorded start time.
        self._write_info(lock, pid=os.getpid(), pid_start=1)
        assert not lock_is_live(lock, stale_after_s=3600.0)

    def test_publish_lock_info_records_pid_start_and_is_live(self, tmp_path: Path) -> None:
        """The shared publisher writes pid/hostname/started_at/pid_start and reads back live."""
        lock = tmp_path / DISPATCH_LOCK_DIR_NAME
        lock.mkdir(parents=True)
        publish_lock_info(lock, extra={"action_id": "a1"})
        info = json.loads((lock / "info.json").read_text(encoding="utf-8"))
        assert info["pid"] == os.getpid()
        assert info["hostname"] == socket.gethostname()
        assert "pid_start" in info
        assert info["action_id"] == "a1"
        assert lock_is_live(lock, stale_after_s=10.0)


class TestConcurrentCorpusPassAppend:
    """Spar round 7, P1. ``append_corpus_pass_action`` loaded, edited and saved the
    plan outside any lock, while progress took its own lock internally.

    Two operators appending at once would therefore each read the same plan, and the
    second ``save_plan`` would drop the first's action -- while progress, correctly
    serialized, kept both. The two files then described different plans, with progress
    naming an action the plan did not contain. Holding one lock across the whole
    read-modify-write of both files is what makes them agree.
    """

    def test_concurrent_appends_leave_plan_and_progress_aligned(self, ws: Path) -> None:
        import threading

        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import append_corpus_pass_action
        from carmel.services.planner import load_plan as _load_plan
        from carmel.services.planner import save_plan as _save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        _save_plan(ws, plan)
        init_progress(ws, plan)

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def _append() -> None:
            try:
                barrier.wait(timeout=5)
                append_corpus_pass_action(ws, budget_tokens=100_000)
            except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=_append) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # Assert they actually finished. Without this a self-deadlock regression --
        # the exact failure the non-reentrant workspace lock makes possible -- would
        # time out here and then surface below as "an append was lost from the plan",
        # sending a reader after a phantom correctness bug instead of the hang.
        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} append thread(s) did not finish within 10s: deadlock"

        # Exactly one append wins. The other is refused by the already-queued guard
        # (spar round 8) rather than lost to a race: a refusal the operator can read is
        # a correct outcome, a silently dropped action is not. Any OTHER exception is a
        # real failure.
        unexpected = [e for e in errors if not isinstance(e, ValueError)]
        assert not unexpected, f"an append failed for the wrong reason: {unexpected}"
        assert len(errors) <= 1, f"both appends were refused: {errors}"
        plan_ids = [a.action_id for a in _load_plan(ws).actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS]
        progress_ids = [a.action_id for a in load_progress(ws).actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS]
        assert len(plan_ids) == 2 - len(errors), f"an append was lost from the plan: {plan_ids}"
        assert sorted(plan_ids) == sorted(progress_ids), (
            f"plan and progress disagree: plan={sorted(plan_ids)} progress={sorted(progress_ids)}"
        )
        assert [a.action_id for a in _load_plan(ws).actions] == [a.action_id for a in load_progress(ws).actions], (
            "plan and progress are no longer index-aligned"
        )


class TestCorpusPassAppendIsNotRepeatable:
    """Spar round 8, P1. Appending happens BEFORE the run, and the run can legitimately
    not reach the new action -- the dispatcher executes the plan's next action, which
    may be an earlier one still pending.

    An operator who retries the command then appends another pass every time. Each one
    eventually runs, and since findings are deliberately never deduped across passes,
    each adds its findings to the accumulated report again.
    """

    def test_a_second_append_is_refused_while_one_is_still_queued(self, ws: Path) -> None:
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        first = append_corpus_pass_action(ws, budget_tokens=100_000)

        with pytest.raises(ValueError, match="already queued"):
            append_corpus_pass_action(ws, budget_tokens=100_000)

        kinds = [a.kind for a in load_progress(ws).actions]
        assert kinds.count(ActionKind.LITERATURE_CORPUS_PASS) == 1, "a duplicate pass was queued"
        assert first.action_id in [a.action_id for a in load_progress(ws).actions]

    def test_a_new_append_is_allowed_once_the_previous_one_has_run(self, ws: Path) -> None:
        """The guard must bound duplicates, not prevent a legitimate second pass: an
        operator re-reading the corpus after dropping new papers is the normal case."""
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        first = append_corpus_pass_action(ws, budget_tokens=100_000)
        mark_running(ws, first.action_id, "a1")
        mark_finished(
            ws,
            first.action_id,
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )

        second = append_corpus_pass_action(ws, budget_tokens=100_000)

        assert second.action_id != first.action_id
        kinds = [a.kind for a in load_progress(ws).actions]
        assert kinds.count(ActionKind.LITERATURE_CORPUS_PASS) == 2


class TestTheCorpusPassObeysTheApprovalPolicy:
    """The corpus pass was created with a hardcoded AUTO_APPROVED and never re-evaluated.

    An operator who set `require_approval_for_literature: true` to hold every
    literature action for review got a corpus pass that ran immediately -- having set
    the exact option meant to prevent it. The corpus pass is also the more expensive of
    the two literature kinds (it is the one an operator names a budget for), so it was
    the wrong one to exempt.

    Two layers had to change together: the action is now re-evaluated against the
    policy, AND `evaluate_action` matches both literature kinds. Matching only
    LITERATURE_SEARCH sent a corpus pass to the catch-all, where it required approval
    unconditionally -- inert policy in the other direction.
    """

    def test_require_approval_for_literature_holds_a_corpus_pass(self, ws: Path) -> None:
        from carmel.schemas.approval import ApprovalPolicy, ApprovalRequirement
        from carmel.services.approvals import save_policy
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        save_policy(ws, ApprovalPolicy(require_approval_for_literature=True))

        action = append_corpus_pass_action(ws, budget_tokens=100_000, model_name="gemini-2.5-flash")

        assert action.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL

    def test_a_pass_under_the_threshold_is_auto_approved(self, ws: Path) -> None:
        """The other direction: the policy must be able to let one through, or the fix
        would simply have replaced one hardcoded verdict with the opposite one."""
        from carmel.schemas.approval import ApprovalPolicy, ApprovalRequirement
        from carmel.services.approvals import save_policy
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        save_policy(ws, ApprovalPolicy(require_approval_for_literature=False, auto_approve_literature_under_usd=1e6))

        action = append_corpus_pass_action(ws, budget_tokens=1_000, model_name="gemini-2.5-flash")

        assert action.approval_requirement == ApprovalRequirement.AUTO_APPROVED

    def test_an_unpriceable_pass_fails_closed(self, ws: Path) -> None:
        """No model configured means no dollar estimate, and `estimated_spend_usd` is
        then 0.0 -- below every threshold. "We cannot price this" must not reach the
        policy as "this is free"."""
        from carmel.schemas.approval import ApprovalPolicy, ApprovalRequirement
        from carmel.services.approvals import save_policy
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        save_policy(ws, ApprovalPolicy(require_approval_for_literature=False, auto_approve_literature_under_usd=1e6))

        action = append_corpus_pass_action(ws, budget_tokens=100_000, model_name=None)

        assert action.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL


class TestAnInsertionNeverDisplacesARunningAction:
    """F17. Refusing only *strictly* behind the cursor is not enough.

    ``at == cursor`` passed that guard even when the action sitting there was
    RUNNING. Inserting there shifts the running action to ``at+1`` while the cursor
    stays put, and its own ``advance_cursor`` then refuses to move (it requires the
    id at the cursor to match) -- so the inserted action inherits the slot and is
    scheduled AFTER the run it was inserted to precede. The caller's ordering intent
    is inverted, and nothing raises.
    """

    def _corpus_action(self, action_id: str = "corpus") -> PlannedAction:
        return _action(action_id, kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False)

    def test_inserting_where_a_running_action_sits_is_refused(self, ws: Path) -> None:
        from carmel.services.plan_progress import append_action_to_progress_locked
        from carmel.services.state_machine import workspace_lock

        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws, "lit")
        mark_running(ws, "t3", "a2")  # the cursor now sits on a RUNNING action
        assert load_progress(ws).cursor == 1

        with workspace_lock(ws), pytest.raises(ValueError, match="RUNNING"):
            append_action_to_progress_locked(ws, self._corpus_action(), index=1)

        # Untouched: the running action still owns its slot.
        progress = load_progress(ws)
        assert [state.action_id for state in progress.actions] == ["lit", "t3"]
        assert progress.cursor == 1

    def test_inserting_at_the_cursor_is_still_allowed_when_nothing_is_running(self, ws: Path) -> None:
        """The guard must not over-refuse: inserting ahead of a PENDING action is the
        normal case, and is exactly how a corpus pass gets placed before the T3 run."""
        from carmel.services.plan_progress import append_action_to_progress_locked
        from carmel.services.state_machine import workspace_lock

        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws, "lit")
        assert load_progress(ws).cursor == 1

        with workspace_lock(ws):
            append_action_to_progress_locked(ws, self._corpus_action(), index=1)

        progress = load_progress(ws)
        assert [state.action_id for state in progress.actions] == ["lit", "corpus", "t3"]
        assert progress.cursor == 1  # the corpus pass runs next, ahead of T3


class TestThePublicAppendWrapper:
    """``append_action_to_progress`` takes the lock itself and had no caller and no
    test -- production goes through the ``_locked`` variant. An untested public entry
    point is one an operator script can reach and nobody has run; it also has to stay
    honest about the workspace lock, which is NOT re-entrant (calling it while holding
    the lock blocks forever)."""

    def test_it_takes_the_lock_itself_and_inserts(self, ws: Path) -> None:
        from carmel.services.plan_progress import append_action_to_progress

        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())

        progress = append_action_to_progress(
            ws, _action("corpus", kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False), index=1
        )

        assert [state.action_id for state in progress.actions] == ["lit", "corpus", "t3"]
        assert load_progress(ws).actions[1].action_id == "corpus"

    def test_it_appends_at_the_end_when_no_index_is_given(self, ws: Path) -> None:
        from carmel.services.plan_progress import append_action_to_progress

        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())

        progress = append_action_to_progress(
            ws, _action("corpus", kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False)
        )

        assert [state.action_id for state in progress.actions] == ["lit", "t3", "corpus"]

    def test_it_refuses_an_insertion_behind_the_cursor(self, ws: Path) -> None:
        from carmel.services.plan_progress import append_action_to_progress

        _to_approved_for_execution(ws)
        init_progress(ws, _two_action_plan())
        mark_running(ws, "lit", "a1")
        mark_finished(ws, "lit", status=ActionExecutionStatus.SUCCEEDED, outcome=ActionOutcome.SUCCEEDED)
        advance_cursor(ws, "lit")

        with pytest.raises(ValueError, match="already progressed"):
            append_action_to_progress(
                ws, _action("corpus", kind=ActionKind.LITERATURE_CORPUS_PASS, blocking=False), index=0
            )


class TestAppendingACorpusPassIsAllOrNothing:
    """F15. The workspace lock excludes concurrent writers; it does not make two
    writes one.

    Progress was written, then the plan, with no rollback. A ``save_plan`` failure
    left progress permanently naming an action the plan does not contain, and every
    later ``execute_next_action`` then refuses the workspace outright. There is no
    repair path short of hand-editing: ``load_or_init_progress`` re-initialises only
    when ``plan_id`` differs, and the edited copy preserves it.
    """

    def test_a_failed_plan_write_leaves_progress_untouched(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.services import planner
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)
        before = load_progress(ws)

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(planner, "save_plan", _boom)

        with pytest.raises(OSError, match="no space left"):
            append_corpus_pass_action(ws, budget_tokens=100_000, model_name="gemini-2.5-flash")

        after = load_progress(ws)
        assert [state.action_id for state in after.actions] == [state.action_id for state in before.actions]
        assert after.cursor == before.cursor

    def test_the_workspace_is_still_dispatchable_afterwards(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the rollback, stated as the symptom it prevents."""
        from carmel.services import planner
        from carmel.services.campaigns import load_campaign
        from carmel.services.dispatcher import default_handlers, execute_next_action
        from carmel.services.planner import append_corpus_pass_action, save_plan

        _to_approved_for_execution(ws)
        plan = _plan([_action("t3")])
        save_plan(ws, plan)
        init_progress(ws, plan)

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(planner, "save_plan", _boom)
        with pytest.raises(OSError):
            append_corpus_pass_action(ws, budget_tokens=100_000, model_name="gemini-2.5-flash")
        monkeypatch.undo()

        # Before the fix this raised UnsupportedActionKindError: "progress references
        # action ... missing from the plan".
        ticket = execute_next_action(ws, load_campaign(ws), handlers=default_handlers())
        assert ticket is not None
        ticket.wait(timeout=60)
