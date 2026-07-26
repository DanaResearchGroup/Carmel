# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Deterministic rule-based planner for Phase 1."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from carmel.schemas.approval import ActionKind, ApprovalPolicy, ApprovalRequirement
from carmel.schemas.campaign import Campaign
from carmel.schemas.plan import Plan, PlannedAction
from carmel.services.approvals import evaluate_action, load_policy
from carmel.services.artifacts import read_json, write_json, write_text
from carmel.services.authorization import ExecutionEnvelope, authorize_action, record_authorization

PLAN_JSON_NAME = "plan.json"
PLAN_MD_NAME = "plan.md"


def estimate_t3_cpu_hours(campaign: Campaign) -> float:
    """Estimate CPU hours for the initial T3 handshake.

    A simple deterministic estimate based on the number of reactor systems
    and observables. Real cost will depend on the actual mechanism, but this
    estimate is sufficient for triggering the approval gate.

    Args:
        campaign: The campaign to estimate for.

    Returns:
        Estimated CPU hours.
    """
    n_reactors = len(campaign.input.target_reactor_systems)
    n_observables = len(campaign.input.target_observables)
    return float(2 * n_reactors + n_observables)


def generate_initial_plan(
    campaign: Campaign,
    policy: ApprovalPolicy,
    workspace_root: Path | None = None,
) -> Plan:
    """Generate the deterministic Phase 1 initial plan.

    The Phase 1 initial plan always contains exactly one action: a T3
    handshake to produce a baseline mechanism and diagnostics.

    The campaign's declared budgets are always passed to the approval
    evaluation. Without that, ``Budgets`` would have no read site anywhere
    in the codebase and a user declaring a 0.5 CPU-hour budget would still
    have a 3 CPU-hour action auto-approved with no human in the loop —
    silently violating the "all expensive actions gated by budget checks"
    invariant.

    Args:
        campaign: The campaign to plan for.
        policy: The active approval policy.
        workspace_root: The campaign workspace root. When given, a budget
            violation is recorded to the decision log as it is detected.

    Returns:
        A Plan with a single T3-handshake action.
    """
    estimated = estimate_t3_cpu_hours(campaign)
    action = PlannedAction(
        action_id=str(uuid4()),
        kind=ActionKind.T3_RUN,
        description="Initial T3 handshake — generate baseline mechanism and diagnostics",
        estimated_cpu_hours=estimated,
        estimated_cost=0.0,
        rationale=(
            f"Phase 1 baseline: build a mechanism for {len(campaign.input.target_observables)} "
            f"target observable(s) across {len(campaign.input.target_reactor_systems)} reactor system(s)."
        ),
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        parameters={},
    )
    requirement = evaluate_action(
        action,
        policy,
        budgets=campaign.input.budgets,
        workspace_root=workspace_root,
    )
    action = action.model_copy(update={"approval_requirement": requirement})
    return Plan(
        plan_id=str(uuid4()),
        campaign_id=campaign.campaign_id,
        created_at=datetime.now(UTC),
        actions=[action],
        rationale="Deterministic Phase 1 baseline plan",
        total_estimated_cpu_hours=estimated,
        requires_approval=requirement == ApprovalRequirement.REQUIRES_APPROVAL,
    )


def estimate_arc_cpu_hours(species: list[dict[str, Any]], reactions: list[dict[str, Any]]) -> float:
    """Estimate CPU hours for a standalone ARC job.

    A simple deterministic estimate: one unit per species plus two per reaction
    (a reaction additionally needs a TS search). Sufficient to drive the
    execution-envelope gate.
    """
    return float(max(1, len(species)) + 2 * len(reactions))


def generate_arc_plan(
    campaign: Campaign,
    species: list[dict[str, Any]] | None = None,
    reactions: list[dict[str, Any]] | None = None,
    level_of_theory: str | None = None,
    job_types: dict[str, bool] | None = None,
    envelopes: dict[ActionKind, ExecutionEnvelope] | None = None,
    workspace_root: Path | None = None,
) -> Plan:
    """Generate a single-action ``run_arc`` plan, gated by the execution envelope.

    Peer to :func:`generate_initial_plan`. The action's approval requirement is
    set by the shared, adapter-agnostic
    :func:`carmel.services.authorization.authorize_action` against the ARC
    envelope **and** the campaign's remaining ``cpu_hours`` — the same symmetric
    gate applied to T3.

    Args:
        campaign: The campaign to plan for.
        species: Species to compute (``[{label, smiles}]``); defaults to the
            campaign's initial mixture when omitted.
        reactions: Optional reactions to compute (``[{label}]``).
        level_of_theory: Optional level of theory (a ``mock``-containing level
            routes ARC to its Mockter adapter).
        job_types: Optional ARC job-type profile.
        envelopes: Optional envelope override (defaults to the conservative
            per-adapter defaults).
        workspace_root: Optional campaign workspace. When given, the envelope
            authorization decision is appended to that workspace's decision log,
            so the gate that set ``approval_requirement`` is auditable.

    Returns:
        A Plan with a single ``run_arc`` action.
    """
    species = species or [
        {"label": c.species, **({"smiles": c.smiles} if c.smiles else {})}
        for c in campaign.input.initial_mixture.components
    ]
    reactions = reactions or []
    estimated = estimate_arc_cpu_hours(species, reactions)

    parameters: dict[str, Any] = {"species": species}
    if reactions:
        parameters["reactions"] = reactions
    if level_of_theory:
        parameters["level_of_theory"] = level_of_theory
    if job_types:
        parameters["job_types"] = job_types

    action = PlannedAction(
        action_id=str(uuid4()),
        kind=ActionKind.ARC_RUN,
        description="Standalone ARC job — compute thermochemistry/rates for selected species",
        estimated_cpu_hours=estimated,
        estimated_cost=0.0,
        rationale=(
            f"Refine {len(species)} species"
            + (f" and {len(reactions)} reaction(s)" if reactions else "")
            + " with a single ARC job."
        ),
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        parameters=parameters,
    )

    result = authorize_action(action, campaign.input.budgets.cpu_hours, envelopes)
    action = action.model_copy(update={"approval_requirement": result.requirement})
    if workspace_root is not None:
        record_authorization(workspace_root, action, result)
    return Plan(
        plan_id=str(uuid4()),
        campaign_id=campaign.campaign_id,
        created_at=datetime.now(UTC),
        actions=[action],
        rationale=f"ARC standalone plan ({result.rationale})",
        total_estimated_cpu_hours=estimated,
        requires_approval=result.requirement == ApprovalRequirement.REQUIRES_APPROVAL,
    )


def render_plan_markdown(plan: Plan) -> str:
    """Render a plan as a human-readable markdown summary.

    Args:
        plan: The plan to render.

    Returns:
        Markdown content.
    """
    lines = [
        f"# Plan {plan.plan_id}",
        "",
        f"- **Campaign:** `{plan.campaign_id}`",
        f"- **Created:** {plan.created_at.isoformat()}",
        f"- **Total estimated CPU hours:** {plan.total_estimated_cpu_hours:.1f}",
        f"- **Requires approval:** {'yes' if plan.requires_approval else 'no'}",
        "",
        f"**Rationale:** {plan.rationale}",
        "",
        "## Actions",
        "",
    ]
    for i, action in enumerate(plan.actions, 1):
        lines.extend(
            [
                f"### {i}. {action.description}",
                "",
                f"- **Action ID:** `{action.action_id}`",
                f"- **Kind:** {action.kind.value}",
                f"- **Estimated CPU hours:** {action.estimated_cpu_hours:.1f}",
                f"- **Approval:** {action.approval_requirement.value}",
                "",
                f"_{action.rationale}_",
                "",
            ]
        )
    return "\n".join(lines)


def save_plan(workspace_root: Path, plan: Plan) -> None:
    """Persist plan.json and plan.md."""
    write_json(workspace_root / PLAN_JSON_NAME, plan)
    write_text(workspace_root / PLAN_MD_NAME, render_plan_markdown(plan))


def load_plan(workspace_root: Path) -> Plan:
    """Load the current plan from disk."""
    return Plan.model_validate(read_json(workspace_root / PLAN_JSON_NAME))


def plan_and_save(workspace_root: Path, campaign: Campaign) -> Plan:
    """Generate the Phase 1 plan and persist it.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to plan for.

    Returns:
        The generated and saved Plan.
    """
    policy = load_policy(workspace_root)
    plan = generate_initial_plan(campaign, policy, workspace_root)
    save_plan(workspace_root, plan)
    return plan
