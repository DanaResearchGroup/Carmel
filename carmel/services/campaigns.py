# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Campaign lifecycle services: creation, loading, listing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from carmel.config import AgentConfig, CarmelConfig
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


class CampaignWorkspaceConflictError(ValueError):
    """Raised by :func:`create_campaign` when ``workspace_root`` already holds a
    DIFFERENT campaign.

    Without this check, a second ``new-campaign`` run against the same workspace
    silently overwrote ``campaign.yaml``/``campaign_state.json`` in place with the new
    campaign's identity while leaving the first campaign's
    ``literature_requests/manifest.json`` and ``provenance/`` records behind -- the new
    campaign then inherited a stranger's manual-acquisition queue, and evidence dropped
    later against that inherited request would be admitted into a campaign that never
    asked for it. Failing closed here (raise before anything is created or written) is
    cheap; a corrupted or split campaign identity discovered later is not.
    """

    def __init__(self, workspace_root: Path, existing_campaign_id: str) -> None:
        self.workspace_root = workspace_root
        self.existing_campaign_id = existing_campaign_id
        super().__init__(
            f"{workspace_root} already holds campaign {existing_campaign_id!r}. Creating a new "
            "campaign here would overwrite that campaign's identity in place (campaign.yaml, "
            "campaign_state.json) while leaving its literature_requests/manifest.json and "
            "provenance records behind for the new campaign to inherit. Choose a different "
            f"workspace_name, or remove {workspace_root} first if it is no longer needed."
        )


class MissingCampaignConfigError(ValueError):
    """Raised by :func:`create_campaign_from_config` when ``config.campaign`` is unset.

    A typed, named error rather than letting ``config.campaign`` be used and fail as
    an ``AttributeError`` on ``None`` -- an operator who forgot the section needs to
    be told exactly what to add to their YAML, not shown a stack trace pointing at
    ``NoneType has no attribute ...`` deep inside this function.
    """

    def __init__(self, config_path_hint: str | None = None) -> None:
        location = f" ({config_path_hint})" if config_path_hint else ""
        super().__init__(
            f"config{location} has no 'campaign:' section -- add one describing "
            "initial_mixture, target_observables, target_reactor_systems, and "
            "budgets (see CampaignConfig in carmel/config.py) before calling "
            "create_campaign_from_config"
        )


#: Bound on how long :func:`start_literature_at_creation` will synchronously wait for
#: the dispatched literature run before giving up on waiting (the run itself is NOT
#: cancelled -- see the ``TimeoutError`` handling there). Generous but finite: a
#: provider hang must not be able to wedge a caller that asked for a synchronous
#: result forever. Callers that must never block at all (the HTTP API) pass ``None``
#: instead, which skips waiting entirely rather than waiting up to this bound.
DEFAULT_LITERATURE_WAIT_TIMEOUT_S = 3600.0


def create_campaign(
    workspace_root: Path,
    campaign_input: CampaignInput,
    approval_policy: ApprovalPolicy | None = None,
    agent_config: AgentConfig | None = None,
    *,
    literature_wait_timeout_s: float | None = DEFAULT_LITERATURE_WAIT_TIMEOUT_S,
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
        literature_wait_timeout_s: How long the auto-started literature run (if
            any) is allowed to block this call. The default is a bounded wait
            (see :data:`DEFAULT_LITERATURE_WAIT_TIMEOUT_S`) rather than an
            unbounded one — a request-serving caller (the HTTP API) MUST pass
            ``None`` here to dispatch the run and return immediately instead
            of tying up a request worker until a provider call finishes or
            times out.

    Returns:
        The created Campaign.
    """
    workspace_root = Path(workspace_root)
    campaign_id = str(uuid4())

    # Fail closed BEFORE creating or writing anything: a workspace that already holds
    # a different campaign's `campaign.yaml` must never be silently overwritten (see
    # CampaignWorkspaceConflictError). A campaign.yaml that fails to parse is treated
    # the same way -- an unreadable file is not evidence the workspace is free to use.
    existing_file = workspace_root / CAMPAIGN_FILE_NAME
    if existing_file.exists():
        try:
            existing_campaign_id = load_campaign(workspace_root).campaign_id
        except (ValueError, OSError) as exc:
            raise CampaignWorkspaceConflictError(workspace_root, f"<unreadable campaign.yaml: {exc}>") from exc
        if existing_campaign_id != campaign_id:
            raise CampaignWorkspaceConflictError(workspace_root, existing_campaign_id)

    init_workspace(workspace_root)

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

    maybe_start_literature_at_creation(workspace_root, campaign, agent_config, wait_timeout_s=literature_wait_timeout_s)
    return campaign


def create_campaign_from_config(config: CarmelConfig, *, workspaces_root: Path | None = None) -> Campaign:
    """Build a :class:`CampaignInput` from ``config.campaign`` and create the campaign.

    This is the single function an operator needs to go from a YAML config file to a
    running campaign, replacing the private hand-maintained "mock of CampaignInput"
    scripts operators previously had to write and keep in sync with this API by hand.

    Args:
        config: A loaded :class:`~carmel.config.CarmelConfig`. Must have a ``campaign``
            section (see :class:`~carmel.config.CampaignConfig`); ``config.agents`` is
            threaded through unchanged so the same config controls whether
            literature-at-creation runs, and against which provider/tier.
        workspaces_root: Where to create the campaign workspace. If ``None``, falls
            back to ``config.workspace_root`` -- the explicit argument wins so
            callers (e.g. a server handling many operators' configs against one
            shared workspaces directory) can override where campaigns land without
            editing every operator's config file.

    Returns:
        The created Campaign.

    Raises:
        MissingCampaignConfigError: If ``config.campaign`` is ``None``.
    """
    if config.campaign is None:
        raise MissingCampaignConfigError()

    campaign_input = config.campaign.to_campaign_input(config.workspace_name)
    root = workspaces_root if workspaces_root is not None else config.workspace_root
    return create_campaign(root, campaign_input, agent_config=config.agents)


class LiteratureStartSkipped(StrEnum):
    """Why :func:`start_literature_at_creation` did not run a literature action.

    Exists because the bare ``None`` this used to return collapsed six distinct
    conditions into one, and the CLI could then only GUESS at the cause. It guessed
    wrong in the most confusing way possible: a run that started and died on a provider
    503 was reported as "the config toggle is off, the plan requires approval, or the
    campaign state does not allow it" -- three statements that were all false. A
    diagnostic that names causes it has not checked is worse than one that admits it
    does not know.
    """

    DISABLED_IN_CONFIG = "disabled_in_config"
    CAMPAIGN_STATE_NOT_READY = "campaign_state_not_ready"
    PLAN_REQUIRES_APPROVAL = "plan_requires_approval"
    NEXT_ACTION_IS_NOT_LITERATURE = "next_action_is_not_literature"
    NO_ACTION_DISPATCHED = "no_action_dispatched"


#: Operator-facing explanation per skip reason. Each says what happened and what to do.
LITERATURE_SKIP_EXPLANATIONS: dict[LiteratureStartSkipped, str] = {
    LiteratureStartSkipped.DISABLED_IN_CONFIG: (
        "the agents config has literature_at_campaign_start=false (or no agents section "
        "was supplied), so no literature run was attempted"
    ),
    LiteratureStartSkipped.CAMPAIGN_STATE_NOT_READY: (
        "the campaign is not in a state a literature run can start from (a completed "
        "literature step leaves it in literature_ready; a workspace reused across two "
        "campaigns leaves a plan from the earlier one)"
    ),
    LiteratureStartSkipped.PLAN_REQUIRES_APPROVAL: (
        "the generated plan requires human approval, and nothing is allowed to spend "
        "money before that approval is given -- approve the plan, then re-run"
    ),
    LiteratureStartSkipped.NEXT_ACTION_IS_NOT_LITERATURE: (
        "the next action in the plan is not a literature search, so there was nothing "
        "to run -- the literature step has most likely already completed"
    ),
    LiteratureStartSkipped.NO_ACTION_DISPATCHED: (
        "the dispatcher declined to start the action; see the campaign decision log for the action-level reason"
    ),
}


@dataclass(frozen=True)
class LiteratureStartOutcome:
    """Either a completed literature action, a still-running one, or a NAMED reason there wasn't one."""

    result: ActionResult | None = None
    skip_reason: LiteratureStartSkipped | None = None
    detail: str | None = None
    """The specific observed value behind the reason (e.g. the actual campaign state).

    Carried separately from the generic explanation so the message can state a FACT the
    code checked, rather than the most likely story. Naming a plausible cause as though
    it were the observed one is the exact habit this whole change exists to remove.
    """
    dispatched_action_id: str | None = None
    """Set when the literature action was dispatched but this call did not wait for it
    to finish (either the caller asked not to wait at all, or the bounded wait it asked
    for elapsed first). Distinct from ``skip_reason``: nothing was skipped here -- the
    run is genuinely in flight, and a caller can poll campaign state to observe it.
    """
    dispatched_attempt_id: str | None = None
    """The dispatcher's attempt id for ``dispatched_action_id``, when set."""

    def explain(self) -> str:
        """Return an operator-facing sentence for a skipped or still-running start."""
        if self.skip_reason is not None:
            explanation = LITERATURE_SKIP_EXPLANATIONS[self.skip_reason]
            return f"{explanation} [{self.detail}]" if self.detail else explanation
        if self.dispatched_action_id is not None:
            base = f"literature action {self.dispatched_action_id} was dispatched but not waited for"
            return f"{base} ({self.detail})" if self.detail else base
        return "the literature action ran"


def maybe_start_literature_at_creation(
    workspace_root: Path,
    campaign: Campaign,
    config: AgentConfig | None,
    deps: LiteratureDeps | None = None,
    *,
    wait_timeout_s: float | None = DEFAULT_LITERATURE_WAIT_TIMEOUT_S,
) -> ActionResult | None:
    """Run the literature-at-creation step, returning only its ActionResult.

    Thin wrapper over :func:`start_literature_at_creation` for callers that do not need
    to know WHY a run was skipped. Callers that report to a human should use that
    function instead and say what actually happened.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to run literature for.
        config: The agentic-layer configuration, or None for no auto-run.
        deps: Optional injected literature dependencies (tests).
        wait_timeout_s: See :func:`start_literature_at_creation`. Note that when the
            run is dispatched but not waited for (``None``, or a bounded wait that
            elapsed), this wrapper's ``None`` return is indistinguishable from every
            other "nothing to report" case -- callers that need to tell those apart
            must call :func:`start_literature_at_creation` directly.

    Returns:
        The dispatcher's ActionResult for the literature action, or None.
    """
    return start_literature_at_creation(workspace_root, campaign, config, deps, wait_timeout_s=wait_timeout_s).result


def start_literature_at_creation(
    workspace_root: Path,
    campaign: Campaign,
    config: AgentConfig | None,
    deps: LiteratureDeps | None = None,
    *,
    wait_timeout_s: float | None = DEFAULT_LITERATURE_WAIT_TIMEOUT_S,
) -> LiteratureStartOutcome:
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
    generates and saves a ``[LITERATURE_SEARCH, T3_RUN]`` plan, and dispatches
    exactly the literature action through the dispatcher.

    Args:
        workspace_root: The campaign workspace root.
        campaign: The campaign to run literature for.
        config: The agentic-layer configuration, or None for no auto-run.
        deps: Optional injected literature dependencies (tests).
        wait_timeout_s: How long to synchronously wait for the dispatched run.
            ``None`` means do not wait at all -- dispatch and return immediately,
            reporting a ``dispatched_action_id`` (the HTTP path: a request handler
            must not block on a background provider call). A numeric value waits up
            to that many seconds and, on timeout, reports the same "dispatched, not
            waited for" outcome rather than raising -- the run keeps going in the
            background either way; only this call's patience differs. Defaults to
            :data:`DEFAULT_LITERATURE_WAIT_TIMEOUT_S`, a bounded (not unbounded) wait,
            so the CLI's direct call (``Carmel.py``'s ``carmel literature``) cannot
            hang forever on a wedged provider without any caller-side change.

    Returns:
        A :class:`LiteratureStartOutcome` carrying either the dispatcher's ActionResult,
        a still-running dispatch, or a named :class:`LiteratureStartSkipped` reason --
        never an unexplained None.
    """
    if config is None or not config.literature_at_campaign_start:
        return LiteratureStartOutcome(skip_reason=LiteratureStartSkipped.DISABLED_IN_CONFIG)

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
            return LiteratureStartOutcome(
                skip_reason=LiteratureStartSkipped.CAMPAIGN_STATE_NOT_READY,
                detail=f"campaign state is {state.value!r} and the workspace has no plan",
            )
        plan = plan_and_save(workspace_root, campaign, include_literature=True)
        state = update_state(workspace_root, CampaignStateValue.PLAN_PENDING_APPROVAL).state
        if plan.requires_approval:
            _log.info("literature-at-creation deferred: plan requires human approval")
            return LiteratureStartOutcome(skip_reason=LiteratureStartSkipped.PLAN_REQUIRES_APPROVAL)
        for action in plan.actions:
            record_decision(workspace_root, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")
        state = update_state(workspace_root, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="auto-approved").state
    else:
        plan = load_plan(workspace_root)

    if state != CampaignStateValue.APPROVED_FOR_EXECUTION:
        _log.warning("cannot start literature from state %s", state.value)
        return LiteratureStartOutcome(
            skip_reason=LiteratureStartSkipped.CAMPAIGN_STATE_NOT_READY,
            detail=f"campaign state is {state.value!r}",
        )

    progress = load_or_init_progress(workspace_root, plan)
    next_id = progress.next_action_id()
    next_action = next((a for a in plan.actions if a.action_id == next_id), None)
    if next_action is None or next_action.kind != ActionKind.LITERATURE_SEARCH:
        _log.info("next plan action is not a literature search; nothing to auto-run")
        detail = (
            "the plan has no further actions" if next_action is None else f"next action is {next_action.kind.value!r}"
        )
        return LiteratureStartOutcome(skip_reason=LiteratureStartSkipped.NEXT_ACTION_IS_NOT_LITERATURE, detail=detail)

    handlers = default_handlers(agent_config=config, literature_deps=deps)
    ticket = execute_next_action(workspace_root, campaign, handlers=handlers)
    if ticket is None:
        return LiteratureStartOutcome(skip_reason=LiteratureStartSkipped.NO_ACTION_DISPATCHED)

    if wait_timeout_s is None:
        # HTTP path: the dispatcher already runs the action on its own background
        # thread (see DispatchTicket); a request handler must not block on it too,
        # or a slow/hanging provider call ties up a request worker indefinitely.
        # Report that the run was dispatched -- the caller polls campaign state for
        # progress instead of getting a synchronous result here.
        return LiteratureStartOutcome(dispatched_action_id=ticket.action_id, dispatched_attempt_id=ticket.attempt_id)
    try:
        return LiteratureStartOutcome(result=ticket.wait(timeout=wait_timeout_s))
    except TimeoutError:
        # The run is NOT abandoned -- only this call's patience ran out (no
        # cancellation exists here, or is warranted: the dispatcher's thread keeps
        # going and will still persist its result to the workspace). Report the
        # same "dispatched, not waited for" outcome as the no-wait branch above
        # rather than letting the timeout escape as a bare exception to a caller
        # that only asked for a bounded wait, not a guarantee of completion.
        return LiteratureStartOutcome(
            dispatched_action_id=ticket.action_id,
            dispatched_attempt_id=ticket.attempt_id,
            detail=f"did not finish within {wait_timeout_s:.0f}s; still running in the background",
        )


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
