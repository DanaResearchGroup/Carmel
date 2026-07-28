# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Campaign lifecycle services: creation, loading, listing."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from carmel.config import AgentConfig
from carmel.logger import get_logger
from carmel.paths import init_workspace
from carmel.schemas.approval import ApprovalPolicy, ApprovalStatus
from carmel.schemas.campaign import Campaign, CampaignInput
from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.approvals import record_decision, save_policy
from carmel.services.artifacts import read_yaml, write_yaml
from carmel.services.decision_log import append_event
from carmel.services.provenance import record
from carmel.services.state_machine import load_state, save_state, update_state

if TYPE_CHECKING:
    from carmel.services.dispatcher import ActionResult
    from carmel.services.literature import LiteratureDeps

CAMPAIGN_FILE_NAME = "campaign.yaml"

_log = get_logger("services.campaigns")


def create_campaign(
    workspace_root: Path,
    campaign_input: CampaignInput,
    approval_policy: ApprovalPolicy | None = None,
    agent_config: AgentConfig | None = None,
) -> Campaign:
    """Create a new campaign workspace and write canonical artifacts.

    Args:
        workspace_root: Where to create the campaign workspace.
        campaign_input: The validated user-provided input.
        approval_policy: Optional explicit policy. Defaults to ``ApprovalPolicy()``.
        agent_config: Optional agentic-layer configuration. ``None`` means NO
            literature auto-run (spar round 3, P1-12) — creating a campaign
            must never incur network traffic or spend as a side effect of a
            missing config.

    Returns:
        The created Campaign.
    """
    workspace_root = Path(workspace_root)
    init_workspace(workspace_root)

    campaign_id = str(uuid4())
    now = datetime.now(UTC)
    campaign = Campaign(
        campaign_id=campaign_id,
        workspace_root=workspace_root,
        input=campaign_input,
        created_at=now,
        updated_at=now,
    )
    write_yaml(workspace_root / CAMPAIGN_FILE_NAME, campaign)

    policy = approval_policy or ApprovalPolicy()
    save_policy(workspace_root, policy)

    state = CampaignState(
        campaign_id=campaign_id,
        state=CampaignStateValue.DRAFT,
        updated_at=now,
    )
    save_state(workspace_root, state)

    append_event(
        workspace_root / "decision_log.jsonl",
        {
            "event": "campaign_created",
            "campaign_id": campaign_id,
            "workspace_name": campaign_input.workspace_name,
        },
    )
    record(
        workspace_root,
        "campaign_created",
        {"campaign_id": campaign_id, "workspace_name": campaign_input.workspace_name},
    )
    _log.info("Created campaign %s in %s", campaign_id, workspace_root)

    maybe_start_literature_at_creation(workspace_root, campaign, agent_config)
    return campaign


def maybe_start_literature_at_creation(
    workspace_root: Path,
    campaign: Campaign,
    config: AgentConfig | None,
    deps: LiteratureDeps | None = None,
) -> ActionResult | None:
    """Single owner of the literature-at-creation auto-run.

    Invoked from campaign creation (and from ``carmel literature``, which
    calls this same service rather than owning a copy). Returns None — and
    performs NO spend-incurring work — when:

    - ``config`` is None (spar round 3, P1-12: the default-on toggle
      ``literature_at_campaign_start`` applies only when an ``AgentConfig``
      was explicitly provided);
    - ``config.literature_at_campaign_start`` is False;
    - the generated plan requires human approval (no spend before consent);
    - the campaign is in a state from which the literature step cannot
      legally start, or the next plan action is not a literature search.

    Otherwise it advances the campaign to ``APPROVED_FOR_EXECUTION``,
    generates and saves a ``[LITERATURE_SEARCH, T3_RUN]`` plan, and runs
    exactly the literature action through the dispatcher.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to run literature for.
        config: The agentic-layer configuration, or None for no auto-run.
        deps: Optional injected literature dependencies (tests).

    Returns:
        The dispatcher's ActionResult for the literature action, or None.
    """
    if config is None or not config.literature_at_campaign_start:
        return None

    # Heavy imports stay function-level: campaign creation without an agent
    # config must not touch the dispatcher/agents stack at import time.
    from carmel.schemas.approval import ActionKind
    from carmel.services.dispatcher import default_handlers, execute_next_action
    from carmel.services.plan_progress import load_or_init_progress
    from carmel.services.planner import PLAN_JSON_NAME, load_plan, plan_and_save

    state = load_state(workspace_root).state
    if state == CampaignStateValue.DRAFT:
        update_state(workspace_root, CampaignStateValue.VALIDATED, notes="auto: literature at creation")
        state = update_state(workspace_root, CampaignStateValue.READY_FOR_PLANNING).state
    elif state == CampaignStateValue.VALIDATED:
        state = update_state(workspace_root, CampaignStateValue.READY_FOR_PLANNING).state

    if not (workspace_root / PLAN_JSON_NAME).exists():
        if state != CampaignStateValue.READY_FOR_PLANNING:
            _log.warning("cannot start literature from state %s without a plan", state.value)
            return None
        plan = plan_and_save(workspace_root, campaign, include_literature=True)
        state = update_state(workspace_root, CampaignStateValue.PLAN_PENDING_APPROVAL).state
        if plan.requires_approval:
            _log.info("literature-at-creation deferred: plan requires human approval")
            return None
        for action in plan.actions:
            record_decision(workspace_root, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")
        state = update_state(workspace_root, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="auto-approved").state
    else:
        plan = load_plan(workspace_root)

    if state != CampaignStateValue.APPROVED_FOR_EXECUTION:
        _log.warning("cannot start literature from state %s", state.value)
        return None

    progress = load_or_init_progress(workspace_root, plan)
    next_id = progress.next_action_id()
    next_action = next((a for a in plan.actions if a.action_id == next_id), None)
    if next_action is None or next_action.kind != ActionKind.LITERATURE_SEARCH:
        _log.info("next plan action is not a literature search; nothing to auto-run")
        return None

    handlers = default_handlers(agent_config=config, literature_deps=deps)
    ticket = execute_next_action(workspace_root, campaign, handlers=handlers)
    if ticket is None:
        return None
    # The dispatcher starts the action on a background thread; this hook is a
    # synchronous convenience for campaign creation and the CLI, so wait for
    # the run to finish and return its persisted result.
    return ticket.wait()


def load_campaign(workspace_root: Path) -> Campaign:
    """Load a campaign from its canonical workspace file.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        The loaded Campaign.
    """
    return Campaign.model_validate(read_yaml(workspace_root / CAMPAIGN_FILE_NAME))


def list_campaigns(workspaces_root: Path) -> list[Campaign]:
    """List all campaigns under a parent workspaces directory.

    Args:
        workspaces_root: A directory whose immediate children may be campaign workspaces.

    Returns:
        Loaded Campaign objects, one per discovered workspace. Workspaces
        without a valid ``campaign.yaml`` are skipped.
    """
    workspaces_root = Path(workspaces_root)
    if not workspaces_root.exists():
        return []
    campaigns: list[Campaign] = []
    for child in sorted(workspaces_root.iterdir()):
        if not child.is_dir():
            continue
        campaign_file = child / CAMPAIGN_FILE_NAME
        if not campaign_file.exists():
            continue
        try:
            campaigns.append(load_campaign(child))
        except (ValueError, OSError) as e:
            _log.warning("Skipping invalid campaign at %s: %s", child, e)
    return campaigns


def find_campaign_workspace(workspaces_root: Path, campaign_id: str) -> Path | None:
    """Find the workspace directory for a campaign by ID.

    Returns the directory the campaign was actually discovered in, never
    the ``workspace_root`` recorded inside ``campaign.yaml`` — that value
    is untrusted user-editable data and callers (including code that
    deletes files under the returned path) must not be steered by it. A
    mismatch between the two is logged as a warning.

    Args:
        workspaces_root: The parent workspaces directory.
        campaign_id: The campaign ID to find.

    Returns:
        The discovered workspace directory, or None if not found.
    """
    workspaces_root = Path(workspaces_root)
    if not workspaces_root.exists():
        return None
    for child in sorted(workspaces_root.iterdir()):
        if not child.is_dir() or not (child / CAMPAIGN_FILE_NAME).exists():
            continue
        try:
            campaign = load_campaign(child)
        except (ValueError, OSError) as e:
            _log.warning("Skipping invalid campaign at %s: %s", child, e)
            continue
        if campaign.campaign_id != campaign_id:
            continue
        if Path(campaign.workspace_root).resolve() != child.resolve():
            _log.warning(
                "campaign %s recorded workspace_root %s but was discovered at %s; using the discovered path",
                campaign_id,
                campaign.workspace_root,
                child,
            )
        return child
    return None
