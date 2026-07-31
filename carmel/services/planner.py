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
from carmel.services.authorization import ExecutionEnvelope, decide_requirement
from carmel.services.plan_progress import init_progress
from carmel.services.spend import compute_spend

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


def _remaining_cpu_hours(campaign: Campaign, workspace_root: Path | None) -> float:
    """Return the campaign's remaining CPU-hour budget for the gate.

    With a workspace, remaining is the declared budget minus consumed and
    reserved spend, so the campaign *total* binds cumulatively rather
    than every action being compared to the full declared budget. Without
    one (pure unit construction, nothing can have been spent), the full
    declared budget is used.
    """
    if workspace_root is None:
        return campaign.input.budgets.cpu_hours
    return compute_spend(workspace_root).remaining(campaign.input.budgets.cpu_hours)


def generate_initial_plan(
    campaign: Campaign,
    policy: ApprovalPolicy,
    workspace_root: Path | None = None,
    *,
    include_literature: bool = False,
) -> Plan:
    """Generate the deterministic Phase 1 initial plan.

    By default the plan contains exactly one action: a T3 handshake to
    produce a baseline mechanism and diagnostics. With
    ``include_literature=True`` a non-blocking LITERATURE_SEARCH action
    precedes the T3 action.

    The action is gated by the single combined gate
    (:func:`carmel.services.authorization.decide_requirement`): the policy
    thresholds, the declared budget, the per-adapter execution envelope,
    and the campaign's *remaining* budget (declared minus spend already
    consumed or reserved in this workspace) must all clear for the action
    to be auto-approved. Without the remaining-budget term, ``Budgets``
    would only ever bind per action and a 20 CPU-hour campaign could
    auto-approve ten 4-hour actions one by one.

    Args:
        campaign: The campaign to plan for.
        policy: The active approval policy.
        workspace_root: The campaign workspace root. When given, spend
            already recorded there reduces the remaining budget, and the
            gate's decisions are recorded to the decision log.
        include_literature: Whether to prepend a literature-search action.

    Returns:
        The generated Plan.
    """
    estimated = estimate_t3_cpu_hours(campaign)
    actions: list[PlannedAction] = []

    if include_literature:
        literature = PlannedAction(
            action_id=str(uuid4()),
            kind=ActionKind.LITERATURE_SEARCH,
            description="Literature search — collect grounded experimental/model/QM findings",
            estimated_cpu_hours=0.0,
            estimated_cost=0.0,
            estimated_spend_usd=policy.auto_approve_literature_under_usd,
            blocking=False,
            rationale=(
                "Advisory context only: grounded literature findings inform the scientist "
                "reviewing this campaign. In this increment they do NOT parameterise the "
                "T3 action."
            ),
            approval_requirement=ApprovalRequirement.AUTO_APPROVED,
            parameters={},
        )
        # Deliberately the POLICY-only gate (`evaluate_action`), not the combined
        # `decide_requirement` used for T3 below.
        #
        # `decide_requirement` adds a per-adapter ExecutionEnvelope check, and
        # `authorize_action` conservatively escalates any kind with no registered
        # envelope. DEFAULT_ENVELOPES maps T3_RUN and ARC_RUN only, so sending a
        # LITERATURE_SEARCH action through it would return REQUIRES_APPROVAL for every
        # campaign and silently disable the literature auto-run.
        #
        # That escalation is right for what envelopes measure and wrong for this action:
        # an envelope bounds an adapter's CPU-hours per subprocess run, and a literature
        # search launches no subprocess and consumes 0.0 CPU-hours. Its real resource is
        # money-and-tokens, bounded by the agent BudgetLedger at run time and gated here
        # by the policy's `auto_approve_literature_under_usd` threshold.
        #
        # If LITERATURE_SEARCH ever gains an envelope in carmel/services/authorization.py
        # (owned by the ARC workstream), this should move to `decide_requirement` so
        # there is once again exactly one gate.
        literature = literature.model_copy(update={"approval_requirement": evaluate_action(literature, policy)})
        actions.append(literature)

    t3_action = PlannedAction(
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
    t3_requirement, _rationale = decide_requirement(
        t3_action,
        policy=policy,
        remaining_cpu_hours=_remaining_cpu_hours(campaign, workspace_root),
        budgets=campaign.input.budgets,
        workspace_root=workspace_root,
    )
    t3_action = t3_action.model_copy(update={"approval_requirement": t3_requirement})
    actions.append(t3_action)

    requires_approval = any(a.approval_requirement == ApprovalRequirement.REQUIRES_APPROVAL for a in actions)
    return Plan(
        plan_id=str(uuid4()),
        campaign_id=campaign.campaign_id,
        created_at=datetime.now(UTC),
        actions=actions,
        rationale="Deterministic Phase 1 baseline plan",
        total_estimated_cpu_hours=sum(a.estimated_cpu_hours for a in actions),
        requires_approval=requires_approval,
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
    policy: ApprovalPolicy | None = None,
) -> Plan:
    """Generate a single-action ``run_arc`` plan, gated by the combined gate.

    Peer to :func:`generate_initial_plan`. The action's approval requirement is
    set by the single combined gate
    (:func:`carmel.services.authorization.decide_requirement`) against the
    approval policy (``auto_approve_arc_under_cpu_hours``), the ARC envelope,
    **and** the campaign's remaining ``cpu_hours`` (declared budget minus spend
    consumed or reserved in the workspace) — the same symmetric gate applied
    to T3.

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
        workspace_root: Optional campaign workspace. When given, spend already
            recorded there reduces the remaining budget, and the gate's
            decisions are appended to that workspace's decision log, so the
            gate that set ``approval_requirement`` is auditable.
        policy: Optional approval policy. Defaults to the workspace's
            persisted policy when a workspace is given, else to the default
            :class:`~carmel.schemas.approval.ApprovalPolicy`.

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

    if policy is None:
        policy = load_policy(workspace_root) if workspace_root is not None else ApprovalPolicy()
    requirement, rationale = decide_requirement(
        action,
        policy=policy,
        remaining_cpu_hours=_remaining_cpu_hours(campaign, workspace_root),
        budgets=campaign.input.budgets,
        envelopes=envelopes,
        workspace_root=workspace_root,
    )
    action = action.model_copy(update={"approval_requirement": requirement})
    return Plan(
        plan_id=str(uuid4()),
        campaign_id=campaign.campaign_id,
        created_at=datetime.now(UTC),
        actions=[action],
        rationale=f"ARC standalone plan ({rationale})",
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
    """Persist plan.json and plan.md.

    Raises:
        ValueError: If the plan fails
            :func:`carmel.services.dispatcher.validate_plan_shape` — an
            unexecutable plan is rejected at save time, fail closed.
    """
    # Function-level import: the dispatcher imports this module for load_plan.
    from carmel.services.dispatcher import validate_plan_shape

    problems = validate_plan_shape(plan)
    if problems:
        raise ValueError("plan shape is not executable: " + "; ".join(problems))
    write_json(workspace_root / PLAN_JSON_NAME, plan)
    write_text(workspace_root / PLAN_MD_NAME, render_plan_markdown(plan))


def load_plan(workspace_root: Path) -> Plan:
    """Load the current plan from disk."""
    return Plan.model_validate(read_json(workspace_root / PLAN_JSON_NAME))


def plan_and_save(workspace_root: Path, campaign: Campaign, *, include_literature: bool = False) -> Plan:
    """Generate the Phase 1 plan, persist it, and initialise plan progress.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to plan for.
        include_literature: Whether to prepend a literature-search action.

    Returns:
        The generated and saved Plan.
    """
    policy = load_policy(workspace_root)
    plan = generate_initial_plan(campaign, policy, workspace_root, include_literature=include_literature)
    save_plan(workspace_root, plan)
    init_progress(workspace_root, plan)
    return plan


def plan_and_save_arc(
    workspace_root: Path,
    campaign: Campaign,
    level_of_theory: str | None = None,
) -> Plan:
    """Generate a single-action ``run_arc`` plan and persist it.

    ARC peer of :func:`plan_and_save`, and the production entry point the
    UI's plan route uses when the operator asks for an ARC plan instead of
    the T3 baseline. Delegates to :func:`generate_arc_plan` with the
    workspace attached, so the combined gate judges the action against the
    workspace's persisted policy and *remaining* budget, and records its
    authorization to the decision log.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to plan for.
        level_of_theory: Optional level of theory for the ARC job (a
            ``mock``-containing level routes ARC to its Mockter adapter).
            Species default to the campaign's initial mixture.

    Returns:
        The generated and saved Plan.
    """
    plan = generate_arc_plan(campaign, level_of_theory=level_of_theory, workspace_root=workspace_root)
    save_plan(workspace_root, plan)
    return plan
