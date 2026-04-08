"""Tests for Phase 1 pydantic schemas."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from carmel.schemas import (
    ActionKind,
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
