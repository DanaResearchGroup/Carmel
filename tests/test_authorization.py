"""Tests for the symmetric, per-adapter execution envelope."""

from __future__ import annotations

from pathlib import Path

from carmel.schemas import ActionKind, ApprovalRequirement, PlannedAction
from carmel.schemas.approval import ApprovalPolicy
from carmel.schemas.campaign import Budgets
from carmel.services.authorization import (
    DEFAULT_ARC_ENVELOPE,
    DEFAULT_ENVELOPES,
    DEFAULT_T3_ENVELOPE,
    ExecutionEnvelope,
    authorize,
    authorize_action,
    decide_requirement,
    envelope_for,
    record_authorization,
)
from carmel.services.decision_log import read_events


def _action(kind: ActionKind, cpu_hours: float, **params: object) -> PlannedAction:
    return PlannedAction(
        action_id="a1",
        kind=kind,
        description="test",
        estimated_cpu_hours=cpu_hours,
        rationale="test",
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        parameters=dict(params),
    )


class TestEnvelopeDefaults:
    def test_t3_envelope_larger_than_arc(self) -> None:
        # A T3 loop legitimately spends more than a single standalone ARC job.
        assert DEFAULT_T3_ENVELOPE.cpu_hours_per_action > DEFAULT_ARC_ENVELOPE.cpu_hours_per_action

    def test_envelope_for_maps_kinds(self) -> None:
        assert envelope_for(ActionKind.T3_RUN) is DEFAULT_T3_ENVELOPE
        assert envelope_for(ActionKind.ARC_RUN) is DEFAULT_ARC_ENVELOPE

    def test_default_envelopes_cover_both_adapters(self) -> None:
        assert set(DEFAULT_ENVELOPES) == {ActionKind.T3_RUN, ActionKind.ARC_RUN}


class TestAuthorizeSymmetry:
    def test_arc_within_envelope_and_budget_auto_approves(self) -> None:
        action = _action(ActionKind.ARC_RUN, 3.0)
        result = authorize(action, DEFAULT_ARC_ENVELOPE, remaining_cpu_hours=10.0)
        assert result.requirement == ApprovalRequirement.AUTO_APPROVED
        assert result.within_envelope and result.within_budget

    def test_arc_over_envelope_escalates(self) -> None:
        # 8 cpu-h exceeds the ARC envelope (4) even though budget (100) is ample.
        action = _action(ActionKind.ARC_RUN, 8.0)
        result = authorize(action, DEFAULT_ARC_ENVELOPE, remaining_cpu_hours=100.0)
        assert result.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert not result.within_envelope
        assert result.within_budget

    def test_same_cost_gates_differently_across_adapters(self) -> None:
        # 8 cpu-h: over ARC's envelope (escalate) but within T3's (auto).
        arc = authorize(_action(ActionKind.ARC_RUN, 8.0), DEFAULT_ARC_ENVELOPE, 100.0)
        t3 = authorize(_action(ActionKind.T3_RUN, 8.0), DEFAULT_T3_ENVELOPE, 100.0)
        assert arc.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert t3.requirement == ApprovalRequirement.AUTO_APPROVED

    def test_over_budget_escalates_even_within_envelope(self) -> None:
        # 3 cpu-h within ARC envelope (4) but over remaining budget (1).
        action = _action(ActionKind.ARC_RUN, 3.0)
        result = authorize(action, DEFAULT_ARC_ENVELOPE, remaining_cpu_hours=1.0)
        assert result.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert result.within_envelope
        assert not result.within_budget

    def test_wrong_kind_for_envelope_escalates(self) -> None:
        # An ARC action measured against the T3 envelope is out of envelope.
        result = authorize(_action(ActionKind.ARC_RUN, 1.0), DEFAULT_T3_ENVELOPE, 100.0)
        assert result.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert not result.within_envelope

    def test_allowed_levels_restrict(self) -> None:
        env = ExecutionEnvelope(
            adapter="arc",
            cpu_hours_per_action=4.0,
            allowed_action_kinds=[ActionKind.ARC_RUN],
            allowed_levels=["wb97xd/def2tzvp"],
        )
        ok = authorize(_action(ActionKind.ARC_RUN, 1.0, level_of_theory="wb97xd/def2tzvp"), env, 100.0)
        bad = authorize(_action(ActionKind.ARC_RUN, 1.0, level_of_theory="hf/sto-3g"), env, 100.0)
        assert ok.requirement == ApprovalRequirement.AUTO_APPROVED
        assert bad.requirement == ApprovalRequirement.REQUIRES_APPROVAL

    def test_allowed_levels_require_a_declared_level(self) -> None:
        # An action with no level_of_theory runs at the adapter's default, which
        # is outside the bound the allowlist exists to enforce — so it must fail
        # closed rather than slip through as "no level, nothing to check".
        env = ExecutionEnvelope(
            adapter="arc",
            cpu_hours_per_action=4.0,
            allowed_action_kinds=[ActionKind.ARC_RUN],
            allowed_levels=["wb97xd/def2tzvp"],
        )
        result = authorize(_action(ActionKind.ARC_RUN, 1.0), env, 100.0)
        assert result.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert not result.within_envelope
        assert "declares no level_of_theory" in result.rationale

    def test_unrestricted_envelope_allows_missing_level(self) -> None:
        # The guard above must not leak into envelopes that set no allowlist.
        result = authorize(_action(ActionKind.ARC_RUN, 1.0), DEFAULT_ARC_ENVELOPE, 100.0)
        assert DEFAULT_ARC_ENVELOPE.allowed_levels is None
        assert result.requirement == ApprovalRequirement.AUTO_APPROVED


class TestAuthorizeAction:
    def test_selects_envelope_by_kind(self) -> None:
        result = authorize_action(_action(ActionKind.ARC_RUN, 3.0), remaining_cpu_hours=10.0)
        assert result.adapter == "arc"
        assert result.requirement == ApprovalRequirement.AUTO_APPROVED

    def test_unmapped_kind_escalates(self) -> None:
        result = authorize_action(_action(ActionKind.EXPERIMENT, 1.0), remaining_cpu_hours=10.0)
        assert result.requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert result.adapter == "unknown"


class TestDecideRequirement:
    """The ONE combined gate: policy AND envelope+budget must both clear."""

    def test_auto_only_when_both_paths_auto_approve(self) -> None:
        requirement, rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 3.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=10.0,
        )
        assert requirement == ApprovalRequirement.AUTO_APPROVED
        assert "auto-approved" in rationale

    def test_policy_only_escalation_forces_approval(self) -> None:
        # 12 cpu-h: within the T3 envelope (24) and budget (100), but over the
        # policy threshold (10) — the envelope path alone would auto-approve.
        requirement, rationale = decide_requirement(
            _action(ActionKind.T3_RUN, 12.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=100.0,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert "approval policy" in rationale

    def test_envelope_only_escalation_forces_approval(self) -> None:
        # 4.5 cpu-h: under the ARC policy threshold (5) but over the ARC
        # envelope cap (4) — the policy path alone would auto-approve.
        requirement, rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 4.5),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=100.0,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert "envelope cap" in rationale

    def test_remaining_budget_escalation_forces_approval(self) -> None:
        # 3 cpu-h clears both thresholds but not the remaining budget (1).
        requirement, rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 3.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=1.0,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert "remaining budget" in rationale

    def test_both_escalations_are_reported_together(self) -> None:
        requirement, rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 8.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=1.0,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL
        assert "envelope cap" in rationale
        assert "approval policy also requires approval" in rationale

    def test_declared_budget_flows_to_the_policy_path(self, tmp_path: Path) -> None:
        # Over the declared budget: the policy path escalates and records the
        # violation, even though remaining_cpu_hours alone would not bind.
        requirement, _rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 3.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=3.0,
            budgets=Budgets(cpu_hours=2.0, experiment_budget=0.0),
            workspace_root=tmp_path,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL
        events = read_events(tmp_path / "decision_log.jsonl")
        assert any("budget exceeded" in str(e.get("rationale", "")) for e in events)

    def test_workspace_records_the_envelope_authorization(self, tmp_path: Path) -> None:
        decide_requirement(
            _action(ActionKind.ARC_RUN, 3.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=10.0,
            workspace_root=tmp_path,
        )
        events = read_events(tmp_path / "decision_log.jsonl")
        assert [e["event"] for e in events] == ["execution_envelope_authorization"]

    def test_custom_envelopes_are_honored(self) -> None:
        tight = {
            ActionKind.ARC_RUN: ExecutionEnvelope(
                adapter="arc",
                cpu_hours_per_action=0.5,
                allowed_action_kinds=[ActionKind.ARC_RUN],
            )
        }
        requirement, _rationale = decide_requirement(
            _action(ActionKind.ARC_RUN, 1.0),
            policy=ApprovalPolicy(),
            remaining_cpu_hours=10.0,
            envelopes=tight,
        )
        assert requirement == ApprovalRequirement.REQUIRES_APPROVAL


class TestRecordAuthorization:
    def test_writes_decision_log_entry(self, tmp_path: Path) -> None:
        action = _action(ActionKind.ARC_RUN, 3.0)
        result = authorize_action(action, remaining_cpu_hours=10.0)
        record_authorization(tmp_path, action, result)
        events = read_events(tmp_path / "decision_log.jsonl")
        assert len(events) == 1
        assert events[0]["event"] == "execution_envelope_authorization"
        assert events[0]["adapter"] == "arc"
        assert events[0]["requirement"] == "auto_approved"
