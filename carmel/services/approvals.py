"""Approval policy evaluation and decision recording."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from carmel.schemas.approval import (
    ActionKind,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequirement,
    ApprovalStatus,
)
from carmel.schemas.plan import PlannedAction
from carmel.services.artifacts import read_yaml, write_yaml
from carmel.services.decision_log import append_event

POLICY_FILE_NAME = "approval_policy.yaml"


def load_policy(workspace_root: Path) -> ApprovalPolicy:
    """Load the persisted approval policy."""
    return ApprovalPolicy.model_validate(read_yaml(workspace_root / POLICY_FILE_NAME))


def save_policy(workspace_root: Path, policy: ApprovalPolicy) -> None:
    """Persist an approval policy."""
    write_yaml(workspace_root / POLICY_FILE_NAME, policy)


def evaluate_action(action: PlannedAction, policy: ApprovalPolicy) -> ApprovalRequirement:
    """Evaluate whether an action requires human approval under the policy.

    Args:
        action: The planned action to evaluate.
        policy: The active approval policy.

    Returns:
        ``AUTO_APPROVED`` if the action is below the threshold for its kind,
        ``REQUIRES_APPROVAL`` otherwise.
    """
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
        return ApprovalRequirement.AUTO_APPROVED
    return ApprovalRequirement.REQUIRES_APPROVAL


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
