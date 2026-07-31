"""Tests for Phase 1 pydantic schemas."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from carmel.schemas import (
    PLAN_SCHEMA_VERSION,
    ActionExecutionStatus,
    ActionKind,
    ActionOutcome,
    ActionState,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequirement,
    ApprovalStatus,
    Budgets,
    Campaign,
    CampaignInput,
    CampaignState,
    CampaignStateValue,
    DiagnosticsV1,
    EntryMode,
    FailureCode,
    InitialMixture,
    MixtureComponent,
    ObservableSummary,
    PDepNetworkSelection,
    Plan,
    PlannedAction,
    PlanProgress,
    ReactionSelection,
    ReactorSystem,
    ReactorType,
    RunRecord,
    RunStatus,
    SensitivityEntry,
    SpeciesSelection,
    SubmissionMode,
    TargetObservable,
)


def _make_input() -> CampaignInput:
    return CampaignInput(
        workspace_name="ethanol-test",
        initial_mixture=InitialMixture(
            components=[
                MixtureComponent(species="C2H5OH", mole_fraction=0.05),
                MixtureComponent(species="O2", mole_fraction=0.20),
                MixtureComponent(species="N2", mole_fraction=0.75),
            ]
        ),
        target_observables=[TargetObservable(name="ignition_delay")],
        target_reactor_systems=[
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, 5.0),
                residence_time_s=1.0,
            )
        ],
        budgets=Budgets(cpu_hours=20.0, experiment_budget=0.0),
    )


class TestMixtureComponent:
    def test_valid(self) -> None:
        c = MixtureComponent(species="O2", mole_fraction=0.21)
        assert c.species == "O2"

    def test_zero_fraction_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixtureComponent(species="O2", mole_fraction=0.0)

    def test_over_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixtureComponent(species="O2", mole_fraction=1.5)

    def test_empty_species_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixtureComponent(species="", mole_fraction=0.5)

    def test_nan_mole_fraction_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixtureComponent(species="O2", mole_fraction=float("nan"))


class TestInitialMixture:
    def test_valid(self) -> None:
        m = InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)])
        assert len(m.components) == 1

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InitialMixture(components=[])


class TestReactorSystem:
    def test_valid(self) -> None:
        r = ReactorSystem(
            reactor_type=ReactorType.JSR,
            temperature_range_K=(500.0, 1500.0),
            pressure_range_bar=(1.0, 10.0),
        )
        assert r.reactor_type == ReactorType.JSR

    def test_inverted_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(1500.0, 500.0),
                pressure_range_bar=(1.0, 10.0),
            )

    def test_negative_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.PFR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(-1.0, 5.0),
            )

    def test_nan_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(float("nan"), 1200.0),
                pressure_range_bar=(1.0, 5.0),
            )

    def test_positive_infinity_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, float("inf")),
                pressure_range_bar=(1.0, 5.0),
            )

    def test_negative_infinity_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(float("-inf"), 5.0),
            )

    def test_positive_infinity_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, float("inf")),
            )


class TestCampaignInput:
    def test_minimal_valid(self) -> None:
        ci = _make_input()
        assert ci.workspace_name == "ethanol-test"
        assert ci.entry_mode == EntryMode.BUILD_FROM_SCRATCH

    def test_empty_observables_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CampaignInput(
                workspace_name="x",
                initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
                target_observables=[],
                target_reactor_systems=[
                    ReactorSystem(
                        reactor_type=ReactorType.JSR,
                        temperature_range_K=(800.0, 1200.0),
                        pressure_range_bar=(1.0, 5.0),
                    )
                ],
                budgets=Budgets(cpu_hours=10.0, experiment_budget=0.0),
            )

    def test_empty_reactors_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CampaignInput(
                workspace_name="x",
                initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
                target_observables=[TargetObservable(name="ignition_delay")],
                target_reactor_systems=[],
                budgets=Budgets(cpu_hours=10.0, experiment_budget=0.0),
            )

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CampaignInput(
                workspace_name="x",
                initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
                target_observables=[TargetObservable(name="ignition_delay")],
                target_reactor_systems=[
                    ReactorSystem(
                        reactor_type=ReactorType.JSR,
                        temperature_range_K=(800.0, 1200.0),
                        pressure_range_bar=(1.0, 5.0),
                    )
                ],
                budgets=Budgets(cpu_hours=10.0, experiment_budget=0.0),
                surprise="value",
            )


class TestBudgets:
    def test_valid(self) -> None:
        b = Budgets(cpu_hours=10.0, experiment_budget=500.0)
        assert b.cpu_hours == 10.0

    def test_zero_cpu_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Budgets(cpu_hours=0.0, experiment_budget=0.0)

    def test_negative_experiment_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Budgets(cpu_hours=1.0, experiment_budget=-1.0)


class TestCampaign:
    def test_valid(self) -> None:
        now = datetime.now(UTC)
        c = Campaign(
            campaign_id="abc",
            workspace_root=Path("/tmp/test"),
            input=_make_input(),
            created_at=now,
            updated_at=now,
        )
        assert c.campaign_id == "abc"


class TestApprovalPolicy:
    def test_defaults(self) -> None:
        p = ApprovalPolicy()
        assert p.auto_approve_t3_under_cpu_hours == 10.0
        assert p.require_approval_for_experiments is True

    def test_custom(self) -> None:
        p = ApprovalPolicy(auto_approve_t3_under_cpu_hours=50.0)
        assert p.auto_approve_t3_under_cpu_hours == 50.0

    def test_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApprovalPolicy(unknown_field=1)  # type: ignore[call-arg]


class TestApprovalDecision:
    def test_valid(self) -> None:
        d = ApprovalDecision(
            decision_id="d1",
            action_id="a1",
            status=ApprovalStatus.APPROVED,
            decided_at=datetime.now(UTC),
            decided_by="user",
        )
        assert d.status == ApprovalStatus.APPROVED


class TestCampaignState:
    def test_valid(self) -> None:
        s = CampaignState(
            campaign_id="abc",
            state=CampaignStateValue.DRAFT,
            updated_at=datetime.now(UTC),
        )
        assert s.state == CampaignStateValue.DRAFT

    def test_failed_from_defaults_to_none(self) -> None:
        s = CampaignState(
            campaign_id="abc",
            state=CampaignStateValue.DRAFT,
            updated_at=datetime.now(UTC),
        )
        assert s.failed_from is None

    def test_failed_from_round_trips(self) -> None:
        s = CampaignState(
            campaign_id="abc",
            state=CampaignStateValue.FAILED,
            updated_at=datetime.now(UTC),
            failed_from=CampaignStateValue.RUNNING_T3,
        )
        loaded = CampaignState.model_validate_json(s.model_dump_json())
        assert loaded.failed_from == CampaignStateValue.RUNNING_T3

    def test_extra_field_forbidden_with_failed_from_present(self) -> None:
        with pytest.raises(ValidationError):
            CampaignState(
                campaign_id="abc",
                state=CampaignStateValue.FAILED,
                updated_at=datetime.now(UTC),
                failed_from=CampaignStateValue.RUNNING_T3,
                surprise="nope",
            )


class TestPlan:
    def test_valid(self) -> None:
        action = PlannedAction(
            action_id="a1",
            kind=ActionKind.T3_RUN,
            description="run T3",
            estimated_cpu_hours=5.0,
            rationale="baseline",
            approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        )
        p = Plan(
            plan_id="p1",
            campaign_id="c1",
            created_at=datetime.now(UTC),
            actions=[action],
            rationale="initial",
            total_estimated_cpu_hours=5.0,
            requires_approval=False,
        )
        assert len(p.actions) == 1

    def test_schema_version_defaults_to_current(self) -> None:
        assert PLAN_SCHEMA_VERSION == 2
        action = PlannedAction(
            action_id="a1",
            kind=ActionKind.T3_RUN,
            description="run T3",
            estimated_cpu_hours=5.0,
            rationale="baseline",
            approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        )
        p = Plan(
            plan_id="p1",
            campaign_id="c1",
            created_at=datetime.now(UTC),
            actions=[action],
            rationale="initial",
            total_estimated_cpu_hours=5.0,
            requires_approval=False,
        )
        assert p.schema_version == PLAN_SCHEMA_VERSION

    def test_planned_action_new_fields_have_defaults(self) -> None:
        action = PlannedAction(
            action_id="a1",
            kind=ActionKind.T3_RUN,
            description="run T3",
            estimated_cpu_hours=5.0,
            rationale="baseline",
            approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        )
        assert action.blocking is True
        assert action.estimated_spend_usd == 0.0

    def test_negative_spend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlannedAction(
                action_id="a1",
                kind=ActionKind.LITERATURE_SEARCH,
                description="lit",
                estimated_cpu_hours=0.0,
                estimated_spend_usd=-1.0,
                rationale="x",
                approval_requirement=ApprovalRequirement.AUTO_APPROVED,
            )

    def test_phase1_plan_dict_still_validates(self) -> None:
        """A v1 plan.json (no schema_version/blocking/spend) survives extra='forbid'."""
        v1 = {
            "plan_id": "p1",
            "campaign_id": "c1",
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
        plan = Plan.model_validate(v1)
        assert plan.actions[0].blocking is True
        assert plan.actions[0].estimated_spend_usd == 0.0
        assert plan.schema_version == PLAN_SCHEMA_VERSION


class TestActionState:
    def _state(self, **overrides: object) -> ActionState:
        data: dict[str, object] = {
            "action_id": "a1",
            "kind": ActionKind.T3_RUN,
            "updated_at": datetime.now(UTC),
        }
        data.update(overrides)
        return ActionState.model_validate(data)

    def test_defaults(self) -> None:
        s = self._state()
        assert s.approval_status == ApprovalStatus.PENDING
        assert s.execution_status == ActionExecutionStatus.PENDING
        assert s.outcome == ActionOutcome.NONE
        assert s.attempt_ids == []
        assert s.blocking is True

    def test_is_terminal(self) -> None:
        assert not self._state().is_terminal()
        assert not self._state(execution_status=ActionExecutionStatus.RUNNING).is_terminal()
        for status in (
            ActionExecutionStatus.SUCCEEDED,
            ActionExecutionStatus.FAILED,
            ActionExecutionStatus.SKIPPED,
        ):
            assert self._state(execution_status=status).is_terminal()

    def test_is_executable(self) -> None:
        assert self._state().is_executable()
        assert not self._state(approval_status=ApprovalStatus.REJECTED).is_executable()
        assert not self._state(execution_status=ActionExecutionStatus.SUCCEEDED).is_executable()


class TestPlanProgressSchema:
    def _progress(self, cursor: int = 0) -> PlanProgress:
        now = datetime.now(UTC)
        return PlanProgress(
            plan_id="p1",
            campaign_id="c1",
            actions=[
                ActionState(action_id="a1", kind=ActionKind.LITERATURE_SEARCH, updated_at=now),
                ActionState(action_id="a2", kind=ActionKind.T3_RUN, updated_at=now),
            ],
            cursor=cursor,
            updated_at=now,
        )

    def test_next_action_id(self) -> None:
        assert self._progress().next_action_id() == "a1"
        assert self._progress(cursor=1).next_action_id() == "a2"
        assert self._progress(cursor=2).next_action_id() is None

    def test_is_complete(self) -> None:
        assert not self._progress().is_complete()
        assert self._progress(cursor=2).is_complete()

    def test_negative_cursor_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlanProgress.model_validate(self._progress().model_dump() | {"cursor": -1})


class TestNewCampaignStates:
    def test_literature_states_exist(self) -> None:
        assert CampaignStateValue.RUNNING_LITERATURE.value == "running_literature"
        assert CampaignStateValue.LITERATURE_READY.value == "literature_ready"


class TestApprovalPolicyLiterature:
    def test_default_spend_threshold(self) -> None:
        assert ApprovalPolicy().auto_approve_literature_under_usd == 2.0

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApprovalPolicy(auto_approve_literature_under_usd=-0.5)


class TestRunRecord:
    def test_valid(self) -> None:
        r = RunRecord(
            run_id="r1",
            action_id="a1",
            tool_name="t3",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=datetime.now(UTC),
            submission_mode=SubmissionMode.SUBPROCESS,
        )
        assert r.tool_name == "t3"
        assert r.failure_code == FailureCode.NONE

    def test_stdout_and_stderr_paths_round_trip(self) -> None:
        r = RunRecord(
            run_id="r1",
            action_id="a1",
            tool_name="t3",
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=datetime.now(UTC),
            submission_mode=SubmissionMode.SUBPROCESS,
            stdout_path=Path("/tmp/run/carmel_stdout.log"),
            stderr_path=Path("/tmp/run/carmel_stderr.log"),
        )
        dumped = r.model_dump(mode="json")
        restored = RunRecord.model_validate(dumped)
        assert restored.stdout_path == Path("/tmp/run/carmel_stdout.log")
        assert restored.stderr_path == Path("/tmp/run/carmel_stderr.log")


class TestDiagnosticsV1:
    def test_valid_minimal(self) -> None:
        d = DiagnosticsV1(
            campaign_id="c1",
            run_id="r1",
            generated_at=datetime.now(UTC),
        )
        assert d.species_to_compute == []
        assert d.pdep_sensitivity_flag is False

    def test_full(self) -> None:
        d = DiagnosticsV1(
            campaign_id="c1",
            run_id="r1",
            level_of_theory="CCSD(T)/CBS",
            generated_at=datetime.now(UTC),
            observable_summaries=[
                ObservableSummary(
                    observable="ignition_delay",
                    top_sensitive_rates=[SensitivityEntry(label="rxn1", value=0.5)],
                    top_sensitive_thermo=[SensitivityEntry(label="OH", value=0.3, species="OH")],
                )
            ],
            species_to_compute=[SpeciesSelection(label="OH", smiles="[OH]", reason="high sensitivity")],
            reactions_to_compute=[ReactionSelection(label="r1", reactants=["A", "B"], products=["C", "D"])],
            pdep_networks_to_compute=[PDepNetworkSelection(network_id="N1", species=["A", "B"], reactions=["r1"])],
            pdep_sensitivity_flag=True,
        )
        assert d.level_of_theory == "CCSD(T)/CBS"
        assert len(d.species_to_compute) == 1
