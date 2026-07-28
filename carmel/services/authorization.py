"""Symmetric, per-adapter execution envelope — the bounded-autonomy gate.

This single-purpose module turns the *declared-but-unenforced*
``Budgets.cpu_hours`` (:class:`carmel.schemas.campaign.Budgets`) into an
**enforced, symmetric, per-adapter** authorization gate.

Each adapter (T3, ARC) has an :class:`ExecutionEnvelope` describing what a single
action of that adapter is allowed to spend and which action kinds/levels it may
run. A shared, **adapter-agnostic** :func:`authorize` compares a planned action's
estimated cost against both its envelope **and** the campaign's remaining
``cpu_hours``:

- within envelope **and** within remaining budget  -> ``AUTO_APPROVED``
- over either                                        -> ``REQUIRES_APPROVAL`` (escalate)

The envelope is **symmetric across adapters** — the same gate for T3 and ARC —
but T3's envelope is *larger* than ARC's, because a T3 loop legitimately spends
more than a single standalone ARC job.

This module only wires the *decision* (it sets ``ApprovalRequirement`` /
``plan.requires_approval``); the existing auto-approve / user-approve paths and
decision-log records in :mod:`carmel.services.approvals` and
:mod:`carmel.ui.app` carry it out unchanged.

The numeric defaults below are **conservative starting points for the retreat
demo** and are an open item for Alon to confirm (per adapter, T3 > ARC).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from carmel.schemas.approval import ActionKind, ApprovalPolicy, ApprovalRequirement
from carmel.schemas.campaign import Budgets
from carmel.schemas.plan import PlannedAction
from carmel.services.approvals import evaluate_action
from carmel.services.decision_log import append_event


class BudgetExceededError(RuntimeError):
    """Raised when a launch is refused by the live authorization re-check.

    Raised *before* any supervision or state transition, so nothing is
    taken and nothing can wedge. The launch paths raise it when the live
    combined gate (:func:`decide_requirement`, re-evaluated against the
    campaign's *remaining* budget at launch time) escalates to
    ``REQUIRES_APPROVAL`` and no effective human approval is recorded for
    the action.
    """


class ExecutionEnvelope(BaseModel):
    """Per-adapter bound on what one action may spend and run.

    Symmetric across adapters (same shape, same gate); the numbers differ —
    T3's envelope is larger than ARC's.
    """

    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    cpu_hours_per_action: float = Field(gt=0)
    max_concurrent_jobs: int = Field(ge=1)
    allowed_action_kinds: list[ActionKind]
    # None means "any level allowed". A list restricts to the named levels of
    # theory, and the action must then *declare* a level in that list — an
    # action with no level_of_theory would otherwise run at the adapter's
    # default, outside the bound the envelope exists to enforce.
    allowed_levels: list[str] | None = None


class AuthorizationResult(BaseModel):
    """The outcome of authorizing a single action against its envelope."""

    model_config = ConfigDict(extra="forbid")

    adapter: str
    requirement: ApprovalRequirement
    within_envelope: bool
    within_budget: bool
    estimated_cpu_hours: float
    remaining_cpu_hours: float
    rationale: str


# --- Conservative default envelopes (propose; Alon confirms — see spec §8) ----
#
# ARC: one standalone QM job. T3: a whole generation/refinement loop, which
# legitimately spends more, so its per-action cap is larger.
DEFAULT_ARC_ENVELOPE = ExecutionEnvelope(
    adapter="arc",
    cpu_hours_per_action=4.0,
    max_concurrent_jobs=1,
    allowed_action_kinds=[ActionKind.ARC_RUN],
)

DEFAULT_T3_ENVELOPE = ExecutionEnvelope(
    adapter="t3",
    cpu_hours_per_action=24.0,
    max_concurrent_jobs=1,
    allowed_action_kinds=[ActionKind.T3_RUN],
)

DEFAULT_ENVELOPES: dict[ActionKind, ExecutionEnvelope] = {
    ActionKind.T3_RUN: DEFAULT_T3_ENVELOPE,
    ActionKind.ARC_RUN: DEFAULT_ARC_ENVELOPE,
}


def envelope_for(
    kind: ActionKind,
    envelopes: dict[ActionKind, ExecutionEnvelope] | None = None,
) -> ExecutionEnvelope | None:
    """Return the execution envelope for an action kind, or None if unmapped."""
    return (envelopes or DEFAULT_ENVELOPES).get(kind)


def authorize(
    action: PlannedAction,
    envelope: ExecutionEnvelope,
    remaining_cpu_hours: float,
) -> AuthorizationResult:
    """Authorize a single action against its envelope and the remaining budget.

    Adapter-agnostic: the action's cost is read from
    ``action.estimated_cpu_hours`` (which the planner sets from the owning
    adapter's ``estimate_cost``). Within envelope **and** within remaining
    campaign ``cpu_hours`` -> auto-approve; over either -> escalate to the user.

    Args:
        action: The planned action to authorize.
        envelope: The per-adapter execution envelope.
        remaining_cpu_hours: The campaign's remaining CPU-hour budget.

    Returns:
        An :class:`AuthorizationResult` carrying the decision and rationale.
    """
    cost = float(action.estimated_cpu_hours)

    kind_ok = action.kind in envelope.allowed_action_kinds
    cpu_ok = cost <= envelope.cpu_hours_per_action
    level = action.parameters.get("level_of_theory")
    level_ok = envelope.allowed_levels is None or (level is not None and level in envelope.allowed_levels)
    within_envelope = kind_ok and cpu_ok and level_ok
    within_budget = cost <= remaining_cpu_hours

    if within_envelope and within_budget:
        requirement = ApprovalRequirement.AUTO_APPROVED
        rationale = (
            f"auto-approved: {cost:.1f} cpu-h within {envelope.adapter} envelope "
            f"({envelope.cpu_hours_per_action:.1f} cpu-h/action) and within remaining "
            f"budget ({remaining_cpu_hours:.1f} cpu-h)"
        )
    else:
        requirement = ApprovalRequirement.REQUIRES_APPROVAL
        reasons: list[str] = []
        if not kind_ok:
            reasons.append(f"kind {action.kind.value} not allowed by {envelope.adapter} envelope")
        if not cpu_ok:
            reasons.append(f"{cost:.1f} cpu-h exceeds envelope cap {envelope.cpu_hours_per_action:.1f} cpu-h")
        if not level_ok:
            reasons.append(
                f"action declares no level_of_theory but {envelope.adapter} envelope restricts levels"
                if level is None
                else f"level {level!r} not in envelope allowed_levels"
            )
        if not within_budget:
            reasons.append(f"{cost:.1f} cpu-h exceeds remaining budget {remaining_cpu_hours:.1f} cpu-h")
        rationale = "escalate to user: " + "; ".join(reasons)

    return AuthorizationResult(
        adapter=envelope.adapter,
        requirement=requirement,
        within_envelope=within_envelope,
        within_budget=within_budget,
        estimated_cpu_hours=cost,
        remaining_cpu_hours=remaining_cpu_hours,
        rationale=rationale,
    )


def authorize_action(
    action: PlannedAction,
    remaining_cpu_hours: float,
    envelopes: dict[ActionKind, ExecutionEnvelope] | None = None,
) -> AuthorizationResult:
    """Authorize an action by selecting its per-adapter envelope by kind.

    Symmetric entry point used for **both** T3 and ARC actions. Unmapped kinds
    are conservatively escalated to the user.
    """
    envelope = envelope_for(action.kind, envelopes)
    if envelope is None:
        return AuthorizationResult(
            adapter="unknown",
            requirement=ApprovalRequirement.REQUIRES_APPROVAL,
            within_envelope=False,
            within_budget=float(action.estimated_cpu_hours) <= remaining_cpu_hours,
            estimated_cpu_hours=float(action.estimated_cpu_hours),
            remaining_cpu_hours=remaining_cpu_hours,
            rationale=f"escalate to user: no execution envelope registered for kind {action.kind.value}",
        )
    return authorize(action, envelope, remaining_cpu_hours)


def decide_requirement(
    action: PlannedAction,
    *,
    policy: ApprovalPolicy,
    remaining_cpu_hours: float,
    budgets: Budgets | None = None,
    envelopes: dict[ActionKind, ExecutionEnvelope] | None = None,
    workspace_root: Path | None = None,
) -> tuple[ApprovalRequirement, str]:
    """Run the ONE authoritative approval gate for a planned action.

    Combines the two previously-split approval systems so they can no
    longer stamp conflicting truth:

    * the **policy** path (:func:`carmel.services.approvals.evaluate_action`
      — per-kind thresholds such as ``auto_approve_arc_under_cpu_hours``,
      plus the declared-budget check), and
    * the **envelope + remaining budget** path (:func:`authorize_action`
      — per-adapter envelope caps and the campaign's *remaining*
      ``cpu_hours``, i.e. budget minus consumed minus reserved spend).

    The action is ``AUTO_APPROVED`` only if **both** paths auto-approve;
    if either escalates, the result is ``REQUIRES_APPROVAL``.

    Args:
        action: The planned action to gate.
        policy: The active approval policy.
        remaining_cpu_hours: The campaign's remaining CPU-hour budget
            (declared budget minus :class:`carmel.services.spend.Spend`).
        budgets: The campaign's declared budgets, forwarded to the policy
            path's declared-budget check.
        envelopes: Optional per-adapter envelope override.
        workspace_root: When given, the envelope authorization (and any
            policy budget violation) is recorded to that workspace's
            decision log, making the gate auditable.

    Returns:
        Tuple of (the combined requirement, a human-readable rationale).
    """
    policy_requirement = evaluate_action(action, policy, budgets=budgets, workspace_root=workspace_root)
    result = authorize_action(action, remaining_cpu_hours, envelopes)
    if workspace_root is not None:
        record_authorization(workspace_root, action, result)

    if (
        policy_requirement == ApprovalRequirement.AUTO_APPROVED
        and result.requirement == ApprovalRequirement.AUTO_APPROVED
    ):
        return ApprovalRequirement.AUTO_APPROVED, result.rationale

    if result.requirement == ApprovalRequirement.REQUIRES_APPROVAL:
        rationale = result.rationale
        if policy_requirement == ApprovalRequirement.REQUIRES_APPROVAL:
            rationale += "; the approval policy also requires approval"
    else:
        rationale = (
            f"escalate to user: the approval policy requires approval for "
            f"{action.kind.value} at {float(action.estimated_cpu_hours):.1f} cpu-h"
        )
    return ApprovalRequirement.REQUIRES_APPROVAL, rationale


def record_authorization(
    workspace_root: Path,
    action: PlannedAction,
    result: AuthorizationResult,
) -> None:
    """Append the envelope authorization decision to the decision log.

    This reuses the shared append-only decision log; the auto/user-approve paths
    that consume ``requirement`` are unchanged.
    """
    append_event(
        workspace_root / "decision_log.jsonl",
        {
            "event": "execution_envelope_authorization",
            "action_id": action.action_id,
            "action_kind": action.kind.value,
            "adapter": result.adapter,
            "requirement": result.requirement.value,
            "within_envelope": result.within_envelope,
            "within_budget": result.within_budget,
            "estimated_cpu_hours": result.estimated_cpu_hours,
            "remaining_cpu_hours": result.remaining_cpu_hours,
            "rationale": result.rationale,
        },
    )
