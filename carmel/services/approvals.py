# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Approval policy evaluation and decision recording."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from carmel.schemas.action_state import PlanProgress

from carmel.schemas.approval import (
    ActionKind,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequirement,
    ApprovalStatus,
)
from carmel.schemas.campaign import Budgets
from carmel.schemas.plan import PlannedAction
from carmel.services.artifacts import read_yaml, write_yaml
from carmel.services.decision_log import append_event, read_events

POLICY_FILE_NAME = "approval_policy.yaml"


def load_policy(workspace_root: Path) -> ApprovalPolicy:
    """Load the persisted approval policy."""
    return ApprovalPolicy.model_validate(read_yaml(workspace_root / POLICY_FILE_NAME))


def save_policy(workspace_root: Path, policy: ApprovalPolicy) -> None:
    """Persist an approval policy."""
    write_yaml(workspace_root / POLICY_FILE_NAME, policy)


def evaluate_action(
    action: PlannedAction,
    policy: ApprovalPolicy,
    budgets: Budgets | None = None,
    workspace_root: Path | None = None,
) -> ApprovalRequirement:
    """Evaluate whether an action requires human approval under the policy.

    Args:
        action: The planned action to evaluate.
        policy: The active approval policy.
        budgets: The campaign's declared budgets, if available. When given,
            a compute-consuming action (``T3_RUN``/``ARC_RUN``) whose
            ``estimated_cpu_hours`` exceeds ``budgets.cpu_hours`` is never
            auto-approved, regardless of the policy threshold.
        workspace_root: The campaign workspace root. When given together
            with ``budgets``, a budget violation is recorded to the
            decision log immediately.

    Returns:
        ``AUTO_APPROVED`` if the action is below the threshold for its kind
        and within the declared budget (if any), ``REQUIRES_APPROVAL``
        otherwise.
    """
    if (
        budgets is not None
        and action.kind in (ActionKind.T3_RUN, ActionKind.ARC_RUN)
        and action.estimated_cpu_hours > budgets.cpu_hours
    ):
        if workspace_root is not None:
            record_decision(
                workspace_root,
                action.action_id,
                ApprovalStatus.PENDING,
                decided_by="system",
                rationale=(
                    f"budget exceeded: estimated {action.estimated_cpu_hours:.2f} cpu_hours "
                    f"> declared budget {budgets.cpu_hours:.2f} cpu_hours"
                ),
            )
        return ApprovalRequirement.REQUIRES_APPROVAL
    if action.kind == ActionKind.T3_RUN:
        if action.estimated_cpu_hours <= policy.auto_approve_t3_under_cpu_hours:
            return ApprovalRequirement.AUTO_APPROVED
        return ApprovalRequirement.REQUIRES_APPROVAL
    if action.kind == ActionKind.ARC_RUN:
        if action.estimated_cpu_hours <= policy.auto_approve_arc_under_cpu_hours:
            return ApprovalRequirement.AUTO_APPROVED
        return ApprovalRequirement.REQUIRES_APPROVAL
    if action.kind == ActionKind.EXPERIMENT:
        if policy.require_approval_for_experiments:
            return ApprovalRequirement.REQUIRES_APPROVAL
        return ApprovalRequirement.AUTO_APPROVED
    if action.kind == ActionKind.LITERATURE_SEARCH:
        if policy.require_approval_for_literature:
            return ApprovalRequirement.REQUIRES_APPROVAL
        if action.estimated_spend_usd > policy.auto_approve_literature_under_usd:
            return ApprovalRequirement.REQUIRES_APPROVAL
        return ApprovalRequirement.AUTO_APPROVED
    return ApprovalRequirement.REQUIRES_APPROVAL


def has_effective_human_approval(workspace_root: Path, action_id: str) -> bool:
    """Report whether a human's standing approval authorizes this action.

    Effective means: the *latest* human decision (``APPROVED`` or
    ``REJECTED``) recorded for the action is ``APPROVED``. Non-human
    statuses are deliberately ignored on both sides:

    * ``AUTO_APPROVED`` records do not count — an auto-approval is only
      valid while the live gate still auto-approves, so it can never
      authorize a launch the live gate escalates (e.g. a stale
      auto-approval from before the budget ran out).
    * ``PENDING`` records (written when a budget violation is detected at
      planning time) do not revoke a human's explicit approval — a human
      who approved an over-budget action has overridden the budget, and a
      retry of that action must still launch.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The action being launched.

    Returns:
        True if the latest human decision for the action is ``APPROVED``.
    """
    return _latest_human_decision_status(workspace_root, action_id) == ApprovalStatus.APPROVED.value


def has_effective_human_rejection(workspace_root: Path, action_id: str) -> bool:
    """Report whether a human's standing decision refuses this action.

    Mirror of :func:`has_effective_human_approval`: effective means the
    *latest* human decision (``APPROVED`` or ``REJECTED``) recorded for the
    action is ``REJECTED``. As with the approval side, ``AUTO_APPROVED``
    and ``PENDING`` records are ignored on both sides — only an explicit
    human decision can flip this, and a later ``APPROVED`` always
    supersedes an earlier ``REJECTED`` for the same action.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The action being launched.

    Returns:
        True if the latest human decision for the action is ``REJECTED``.
    """
    return _latest_human_decision_status(workspace_root, action_id) == ApprovalStatus.REJECTED.value


def _latest_human_decision_status(workspace_root: Path, action_id: str) -> str | None:
    """Return the latest human (``APPROVED``/``REJECTED``) decision status for an action.

    ``AUTO_APPROVED`` and ``PENDING`` records are ignored: they are not
    human decisions, so they can neither authorize nor revoke one.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The action being decided on.

    Returns:
        The latest human decision's status value, or ``None`` if no human
        decision has been recorded for this action.
    """
    human_statuses = {ApprovalStatus.APPROVED.value, ApprovalStatus.REJECTED.value}
    last_human: str | None = None
    for event in read_events(workspace_root / "decision_log.jsonl"):
        if event.get("event") != "approval_decision" or event.get("action_id") != action_id:
            continue
        status = event.get("status")
        if status in human_statuses:
            last_human = str(status)
    return last_human


def record_decision(
    workspace_root: Path,
    action_id: str,
    status: ApprovalStatus,
    decided_by: str,
    rationale: str | None = None,
) -> ApprovalDecision:
    """Create and append an approval decision to the decision log.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The action being decided on.
        status: The decision status.
        decided_by: ``"auto"`` or a username.
        rationale: Optional rationale for the decision.

    Returns:
        The recorded ApprovalDecision.
    """
    decision = ApprovalDecision(
        decision_id=str(uuid4()),
        action_id=action_id,
        status=status,
        decided_at=datetime.now(UTC),
        decided_by=decided_by,
        rationale=rationale,
    )
    append_event(
        workspace_root / "decision_log.jsonl",
        {
            "event": "approval_decision",
            **decision.model_dump(mode="json"),
        },
    )
    return decision


def record_action_decision(
    workspace_root: Path,
    action_id: str,
    status: ApprovalStatus,
    decided_by: str,
    rationale: str | None = None,
) -> tuple[ApprovalDecision, PlanProgress]:
    """Record an approval decision AND apply it to the persisted plan progress.

    This is :func:`record_decision` plus
    :func:`carmel.services.plan_progress.set_approval` — the per-action
    entry point for multi-action plans (approving a previously-rejected
    action also un-skips it and rewinds the cursor; see ``set_approval``).

    Returns:
        Tuple of (recorded decision, updated PlanProgress).
    """
    from carmel.services.plan_progress import set_approval

    decision = record_decision(workspace_root, action_id, status, decided_by, rationale)
    progress = set_approval(workspace_root, action_id, status)
    return decision, progress
