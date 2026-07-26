"""Tests for Phase 1 service modules."""

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
    DIAGNOSTICS_FILE_NAME,
    execute_t3_action,
    load_diagnostics,
    save_diagnostics,
    save_run_record,
)
from carmel.services.intake import StubIntakeParser, write_intake_review
from carmel.services.planner import (
    estimate_t3_cpu_hours,
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
