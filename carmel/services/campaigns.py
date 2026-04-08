"""Campaign lifecycle services: creation, loading, listing."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from carmel.logger import get_logger
from carmel.paths import init_workspace
from carmel.schemas.approval import ApprovalPolicy
from carmel.schemas.campaign import Campaign, CampaignInput
from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.approvals import save_policy
from carmel.services.artifacts import read_yaml, write_yaml
from carmel.services.decision_log import append_event
from carmel.services.provenance import record
from carmel.services.state_machine import save_state

CAMPAIGN_FILE_NAME = "campaign.yaml"

_log = get_logger("services.campaigns")


def create_campaign(
    workspace_root: Path,
    campaign_input: CampaignInput,
    approval_policy: ApprovalPolicy | None = None,
) -> Campaign:
    """Create a new campaign workspace and write canonical artifacts.

    Args:
        workspace_root: Where to create the campaign workspace.
        campaign_input: The validated user-provided input.
        approval_policy: Optional explicit policy. Defaults to ``ApprovalPolicy()``.

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
    return campaign


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

    Args:
        workspaces_root: The parent workspaces directory.
        campaign_id: The campaign ID to find.

    Returns:
        The workspace directory, or None if not found.
    """
    for campaign in list_campaigns(workspaces_root):
        if campaign.campaign_id == campaign_id:
            return campaign.workspace_root
    return None
