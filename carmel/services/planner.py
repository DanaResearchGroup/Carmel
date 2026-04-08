"""Deterministic rule-based planner for Phase 1."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from carmel.schemas.approval import ActionKind, ApprovalPolicy, ApprovalRequirement
from carmel.schemas.campaign import Campaign
from carmel.schemas.plan import Plan, PlannedAction
from carmel.services.approvals import evaluate_action, load_policy
from carmel.services.artifacts import read_json, write_json, write_text

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


def generate_initial_plan(campaign: Campaign, policy: ApprovalPolicy) -> Plan:
    """Generate the deterministic Phase 1 initial plan.

    The Phase 1 initial plan always contains exactly one action: a T3
    handshake to produce a baseline mechanism and diagnostics.

    Args:
        campaign: The campaign to plan for.
        policy: The active approval policy.

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
    requirement = evaluate_action(action, policy)
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
    plan = generate_initial_plan(campaign, policy)
    save_plan(workspace_root, plan)
    return plan
