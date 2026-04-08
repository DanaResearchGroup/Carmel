"""Tests for Phase 1 service modules."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from carmel.schemas import (
    ActionKind,
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

    def test_find_workspace_missing(self, tmp_path: Path) -> None:
        assert find_campaign_workspace(tmp_path, "missing-id") is None


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
