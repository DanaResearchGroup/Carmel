# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Flask application factory and routes for Carmel.

The UI layer is intentionally thin: route handlers parse form data,
delegate to service modules, and render templates. All business logic
lives in :mod:`carmel.services`.
"""

from __future__ import annotations

import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Flask, abort, flash, redirect, render_template, request, url_for
from werkzeug.wrappers.response import Response

from carmel.logger import get_logger
from carmel.schemas.approval import ActionKind, ApprovalStatus
from carmel.schemas.campaign import (
    Budgets,
    CampaignInput,
    EntryMode,
    InitialMixture,
    MixtureComponent,
    ReactorSystem,
    ReactorType,
    TargetObservable,
)
from carmel.schemas.plan import Plan
from carmel.schemas.state import CampaignState, CampaignStateValue
from carmel.services.approvals import (
    record_decision,
)
from carmel.services.authorization import BudgetExceededError
from carmel.services.campaigns import (
    create_campaign,
    find_campaign_workspace,
    list_campaigns,
    load_campaign,
)
from carmel.services.decision_log import append_event, read_events
from carmel.services.execution import (
    ARC_DIAGNOSTICS_FILE_NAME,
    ARC_MODELS_SUBDIR_NAME,
    DIAGNOSTICS_FILE_NAME,
    MODELS_DIR_NAME,
    RunStillLiveError,
    abandon_arc_run,
    abandon_t3_run,
    load_arc_diagnostics,
    load_diagnostics,
    start_arc_action,
    start_t3_action,
)
from carmel.services.intake import StubIntakeParser, write_intake_review
from carmel.services.planner import load_plan, plan_and_save, plan_and_save_arc
from carmel.services.recovery import (
    LockStateUnknownError,
    RunAlreadySupervisedError,
    probe_run_liveness,
)
from carmel.services.state_machine import InvalidTransitionError, can_transition, load_state, update_state
from carmel.ui.csrf import init_csrf

_log = get_logger("ui")


def _resolve_workspaces_root(workspaces_root: Path | None) -> Path:
    """Resolve the workspaces root directory.

    Preference order:
        1. explicit ``workspaces_root`` argument
        2. ``$CARMEL_WORKSPACES`` env var
        3. ``~/carmel_workspaces`` (user-level default, repo-independent)
    """
    if workspaces_root is not None:
        return Path(workspaces_root).expanduser()
    env = os.environ.get("CARMEL_WORKSPACES")
    if env:
        return Path(env).expanduser()
    return Path.home() / "carmel_workspaces"


def _safe_workspace_dirname(name: str) -> str:
    """Convert a free-form name into a safe directory name."""
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name).strip("-")
    return cleaned or f"campaign-{uuid4().hex[:8]}"


def _build_input_from_form(form: dict[str, Any]) -> CampaignInput:
    """Translate a posted HTML form into a validated CampaignInput.

    Form fields expected:
        - workspace_name
        - mixture_components (textarea, one per line: ``species,fraction[,smiles]``)
        - observables (textarea, one per line: ``name[,species[,smiles]]``)
        - reactors (textarea, one per line: ``type,Tmin,Tmax,Pmin,Pmax[,residence_s]``)
        - cpu_hours
        - experiment_budget
        - notes (optional)
    """
    components: list[MixtureComponent] = []
    for line in (form.get("mixture_components", "") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ValueError(f"mixture component line must be 'species,fraction[,smiles]': {line!r}")
        species = parts[0]
        try:
            mole_fraction = float(parts[1])
        except ValueError as e:
            raise ValueError(f"invalid mole fraction in {line!r}") from e
        smiles = parts[2] if len(parts) > 2 and parts[2] else None
        components.append(MixtureComponent(species=species, mole_fraction=mole_fraction, smiles=smiles))

    observables: list[TargetObservable] = []
    for line in (form.get("observables", "") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        species = parts[1] if len(parts) > 1 and parts[1] else None
        smiles = parts[2] if len(parts) > 2 and parts[2] else None
        observables.append(TargetObservable(name=parts[0], species=species, smiles=smiles))

    reactors: list[ReactorSystem] = []
    for line in (form.get("reactors", "") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            raise ValueError(f"reactor line must be 'type,Tmin,Tmax,Pmin,Pmax[,residence_s]': {line!r}")
        residence = float(parts[5]) if len(parts) > 5 and parts[5] else None
        reactors.append(
            ReactorSystem(
                reactor_type=ReactorType(parts[0].lower()),
                temperature_range_K=(float(parts[1]), float(parts[2])),
                pressure_range_bar=(float(parts[3]), float(parts[4])),
                residence_time_s=residence,
            )
        )

    return CampaignInput(
        workspace_name=form["workspace_name"].strip(),
        entry_mode=EntryMode.BUILD_FROM_SCRATCH,
        initial_mixture=InitialMixture(components=components),
        target_observables=observables,
        target_reactor_systems=reactors,
        budgets=Budgets(
            cpu_hours=float(form["cpu_hours"]),
            experiment_budget=float(form["experiment_budget"]),
        ),
        notes=form.get("notes") or None,
    )


def _plan_tool(plan: Plan | None) -> str | None:
    """Name the tool the current plan's action runs: ``"t3"``, ``"arc"``, or None.

    Used for tool-correct button labels and flashes. A plan is Phase 1
    single-action, so the first action's kind is the plan's tool.
    """
    if plan is None or not plan.actions:
        return None
    return "arc" if plan.actions[0].kind == ActionKind.ARC_RUN else "t3"


_ARC_RESULT_STATES = frozenset({CampaignStateValue.RUNNING_ARC, CampaignStateValue.RESULTS_READY})
_T3_RESULT_STATES = frozenset({CampaignStateValue.RUNNING_T3, CampaignStateValue.DIAGNOSTICS_READY})


def _results_tool(state: CampaignState, plan: Plan | None) -> str | None:
    """Name the tool whose results the campaign's *current* state came from.

    Both tools persist diagnostics under their own file (``diagnostics.json``
    for T3, ``arc_diagnostics.json`` for ARC), and a workspace can hold both
    at once — a campaign that failed out of a T3 flow and was re-planned
    through ARC keeps the stale T3 file on disk. The dashboard must show the
    diagnostics belonging to the tool that actually produced the state it is
    describing, so the choice is keyed on the state (and ``failed_from``
    when FAILED), never on which file happens to exist:

    * ``RUNNING_ARC`` / ``RESULTS_READY`` → ``"arc"``; ``RUNNING_T3`` /
      ``DIAGNOSTICS_READY`` → ``"t3"``.
    * ``FAILED`` → decided by ``failed_from`` the same way. A campaign that
      failed before any run (or from an unrecorded origin) has no results
      tool.
    * ``COMPLETED_PHASE1`` → the state alone cannot say which tool finished
      the phase, so it falls back to the plan that ran (terminal state, so
      the plan cannot change from under it).
    * Any earlier state → None: nothing this campaign's current life has
      produced results yet.
    """
    if state.state in _ARC_RESULT_STATES:
        return "arc"
    if state.state in _T3_RESULT_STATES:
        return "t3"
    if state.state == CampaignStateValue.FAILED:
        if state.failed_from in _ARC_RESULT_STATES:
            return "arc"
        if state.failed_from in _T3_RESULT_STATES:
            return "t3"
        return None
    if state.state == CampaignStateValue.COMPLETED_PHASE1:
        return _plan_tool(plan)
    return None


def create_app(workspaces_root: Path | None = None) -> Flask:
    """Create and configure the Carmel Flask application.

    Args:
        workspaces_root: Optional override for the parent workspaces directory.
            Defaults to ``$CARMEL_WORKSPACES`` or ``./workspaces``.

    Returns:
        A configured Flask app.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    # Without CARMEL_SECRET_KEY a fresh random key is generated per process,
    # which invalidates sessions (and CSRF tokens) across restarts — correct
    # and acceptable for a single-user local tool.
    app.secret_key = os.environ.get("CARMEL_SECRET_KEY") or secrets.token_hex(32)
    init_csrf(app)
    workspaces = _resolve_workspaces_root(workspaces_root)
    workspaces.mkdir(parents=True, exist_ok=True)
    app.config["WORKSPACES_ROOT"] = workspaces

    @app.route("/")
    def index() -> str:
        campaigns = list_campaigns(workspaces)
        return render_template("index.html", campaigns=campaigns)

    @app.route("/favicon.ico")
    def favicon() -> Response:
        return redirect(url_for("static", filename="favicon.svg"), code=301)

    @app.route("/campaigns/new", methods=["GET", "POST"])
    def campaign_new() -> str | Response:
        if request.method == "GET":
            return render_template("campaign_create.html")
        try:
            campaign_input = _build_input_from_form(request.form)
        except (KeyError, ValueError) as e:
            flash(f"Invalid campaign input: {e}", "error")
            return render_template("campaign_create.html", form=request.form), 400  # type: ignore[return-value]
        ws = workspaces / _safe_workspace_dirname(campaign_input.workspace_name)
        if ws.exists() and (ws / "campaign.yaml").exists():
            flash(f"A campaign already exists at {ws}", "error")
            return render_template("campaign_create.html", form=request.form), 400  # type: ignore[return-value]
        campaign = create_campaign(ws, campaign_input)
        update_state(ws, CampaignStateValue.VALIDATED, notes="form-validated")
        update_state(ws, CampaignStateValue.READY_FOR_PLANNING)
        return redirect(url_for("campaign_dashboard", campaign_id=campaign.campaign_id))

    @app.route("/campaigns/<campaign_id>")
    def campaign_dashboard(campaign_id: str) -> str:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        campaign = load_campaign(ws)
        state = load_state(ws)
        plan_path = ws / "plan.json"
        plan = load_plan(ws) if plan_path.exists() else None
        # Diagnostics are keyed on the tool that produced the current state
        # (see _results_tool), never loaded unconditionally: a workspace can
        # hold a stale diagnostics.json from an earlier T3 life next to the
        # arc_diagnostics.json of the ARC run that owns the current state,
        # and showing the leftover file would attribute one tool's results
        # to the other.
        results_tool = _results_tool(state, plan)
        diagnostics = load_diagnostics(ws) if results_tool == "t3" else None
        arc_diagnostics = load_arc_diagnostics(ws) if results_tool == "arc" else None
        events = read_events(ws / "decision_log.jsonl")
        latest_run_path = None
        runs_dir = ws / "runs"
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob("*.json"))
            latest_run_path = run_files[-1] if run_files else None
        # Only probed while the campaign claims to be running: it is the
        # one state whose truth cannot be read off disk, and the answer
        # decides whether the page shows progress or a way out.
        running_states = (CampaignStateValue.RUNNING_T3, CampaignStateValue.RUNNING_ARC)
        liveness = probe_run_liveness(ws) if state.state in running_states else None
        return render_template(
            "campaign_dashboard.html",
            campaign=campaign,
            state=state,
            plan=plan,
            plan_tool=_plan_tool(plan),
            results_tool=results_tool,
            diagnostics=diagnostics,
            arc_diagnostics=arc_diagnostics,
            diagnostics_on_disk=(ws / DIAGNOSTICS_FILE_NAME).exists(),
            arc_diagnostics_on_disk=(ws / ARC_DIAGNOSTICS_FILE_NAME).exists(),
            events=events[-20:],
            latest_run_path=latest_run_path.name if latest_run_path else None,
            workspace_root=str(ws),
            liveness=liveness,
        )

    @app.route("/campaigns/<campaign_id>/plan", methods=["POST"])
    def campaign_plan(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        if not can_transition(state.state, CampaignStateValue.PLAN_PENDING_APPROVAL, state.failed_from):
            abort(409, description=f"Cannot plan a campaign in state {state.state.value!r}.")
        campaign = load_campaign(ws)
        tool = request.form.get("tool", "t3")
        if tool == "t3":
            plan = plan_and_save(ws, campaign)
        elif tool == "arc":
            plan = plan_and_save_arc(ws, campaign, level_of_theory=request.form.get("level_of_theory") or None)
        else:
            abort(400, description=f"Unknown planning tool {tool!r}; expected 't3' or 'arc'.")
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        if not plan.requires_approval:
            update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="auto-approved")
            for action in plan.actions:
                record_decision(ws, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/approve", methods=["POST"])
    def campaign_approve(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        if state.state != CampaignStateValue.PLAN_PENDING_APPROVAL:
            abort(409, description=f"Cannot approve a campaign in state {state.state.value!r}.")
        plan = load_plan(ws)
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="user-approved")
        for action in plan.actions:
            record_decision(
                ws,
                action.action_id,
                ApprovalStatus.APPROVED,
                decided_by="user",
                rationale="approved via UI",
            )
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/reject", methods=["POST"])
    def campaign_reject(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        if state.state != CampaignStateValue.PLAN_PENDING_APPROVAL:
            abort(409, description=f"Cannot reject a campaign in state {state.state.value!r}.")
        plan = load_plan(ws)
        update_state(ws, CampaignStateValue.BLOCKED, notes="user-rejected")
        for action in plan.actions:
            record_decision(
                ws,
                action.action_id,
                ApprovalStatus.REJECTED,
                decided_by="user",
                rationale="rejected via UI",
            )
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/run", methods=["POST"])
    def campaign_run(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        campaign = load_campaign(ws)
        plan = load_plan(ws)
        if not plan.actions:
            abort(400)
        action = plan.actions[0]
        decisions = [
            event
            for event in read_events(ws / "decision_log.jsonl")
            if event.get("event") == "approval_decision" and event.get("action_id") == action.action_id
        ]
        approved = {ApprovalStatus.APPROVED.value, ApprovalStatus.AUTO_APPROVED.value}
        if not decisions or decisions[-1].get("status") not in approved:
            abort(409, description="Action has no recorded approval decision.")
        # The plan decides the tool: an approved run_arc action must start
        # ARC, never T3. Both branches share the guards below — approval
        # check above, transition preflight, and the identical launch-time
        # error mapping — so neither tool path is softer than the other.
        if action.kind == ActionKind.ARC_RUN:
            running_state = CampaignStateValue.RUNNING_ARC
            start_action = start_arc_action
        elif action.kind == ActionKind.T3_RUN:
            running_state = CampaignStateValue.RUNNING_T3
            start_action = start_t3_action
        else:
            abort(409, description=f"Action kind {action.kind.value!r} is not executable from the UI.")
        # Preflight the transition rather than letting it raise out of the
        # service layer as a 500. Re-clicking Run — which the auto-refreshing
        # running dashboard now makes easy to do — must read as a conflict,
        # not a crash.
        current = load_state(ws).state
        if not can_transition(current, running_state):
            abort(409, description=f"Cannot start a run for a campaign in state {current.value!r}.")
        try:
            start_action(ws, campaign, action)
        except BudgetExceededError as e:
            # The launch-time re-check found the campaign's remaining budget
            # (or the live gate generally) no longer covers this action and
            # no human approval stands for it. Nothing was started; the
            # dashboard state is unchanged — a conflict, not a crash.
            abort(409, description=str(e))
        except InvalidTransitionError, RunAlreadySupervisedError:
            # The preflight above is a read, so it cannot be authoritative:
            # a concurrent POST can win the race between it and the locked
            # transition inside the start call. Whether the loser trips the
            # state check or the run lock first depends only on timing, so
            # both must read as a conflict rather than a 500.
            abort(409, description="A run for this campaign was started concurrently.")
        except LockStateUnknownError as e:
            # The workspace's filesystem cannot answer whether a run lock is
            # held (no working flock). That is an environment fault, not the
            # caller's — a 503, not a 500 or a 409.
            abort(503, description=f"Cannot determine run-lock state for this workspace: {e}")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/replan", methods=["POST"])
    def campaign_replan(campaign_id: str) -> Response:
        """Send a failed or rejected campaign back to planning.

        The universal way out of a wedged campaign. It is always available
        because it bypasses nothing: the plan is regenerated and re-judged
        against the approval policy from scratch.
        """
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        if not can_transition(state.state, CampaignStateValue.READY_FOR_PLANNING, state.failed_from):
            abort(409, description=f"Cannot re-plan a campaign in state {state.state.value!r}.")
        update_state(ws, CampaignStateValue.READY_FOR_PLANNING, notes="recovered: re-planning")
        append_event(
            ws / "decision_log.jsonl",
            {"event": "campaign_recovered", "recovery": "replan", "from_state": state.state.value},
        )
        flash("Campaign returned to planning. Generate a new plan to continue.", "info")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/retry", methods=["POST"])
    def campaign_retry(campaign_id: str) -> Response:
        """Re-arm a campaign whose already-approved tool run failed.

        Reachable when the campaign failed from ``RUNNING_T3`` or from
        ``APPROVED_FOR_EXECUTION`` (a run that failed after approval but
        before it launched) — the two origins ``RECOVERY_TARGETS`` maps
        back to ``APPROVED_FOR_EXECUTION``. Either way the approval this
        returns to is one a human (or the policy) already gave for this
        very plan, so retry keeps it rather than discarding it through a
        re-plan.
        """
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        if not can_transition(state.state, CampaignStateValue.APPROVED_FOR_EXECUTION, state.failed_from):
            abort(409, description=f"Cannot retry a campaign in state {state.state.value!r}.")
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="recovered: retrying the run")
        append_event(
            ws / "decision_log.jsonl",
            {"event": "campaign_recovered", "recovery": "retry", "from_state": state.state.value},
        )
        # The re-armed plan may be an ARC one; naming the wrong tool here
        # would tell the user to press a button the dashboard does not show.
        plan = load_plan(ws) if (ws / "plan.json").exists() else None
        tool_label = "ARC" if _plan_tool(plan) == "arc" else "T3"
        flash(f"Run re-armed. Use Run {tool_label} to start it again.", "info")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/finalize", methods=["POST"])
    def campaign_finalize(campaign_id: str) -> Response:
        """Complete a campaign from diagnostics already persisted on disk.

        Covers the run that produced real diagnostics and then failed
        while being recorded as complete. Re-running a multi-hour T3 or
        ARC job to recover output already sitting in the workspace would
        be absurd, so this adopts it instead. The state resumed through —
        ``DIAGNOSTICS_READY`` (T3) or ``RESULTS_READY`` (ARC) — is decided
        by where the campaign failed from, and each is gated on its own
        tool's persisted diagnostics.
        """
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        state = load_state(ws)
        resuming = state.state == CampaignStateValue.FAILED
        if resuming:
            if can_transition(state.state, CampaignStateValue.DIAGNOSTICS_READY, state.failed_from):
                ready_state = CampaignStateValue.DIAGNOSTICS_READY
            elif can_transition(state.state, CampaignStateValue.RESULTS_READY, state.failed_from):
                ready_state = CampaignStateValue.RESULTS_READY
            else:
                abort(409, description=f"Cannot complete a campaign in state {state.state.value!r}.")
        elif state.state in (CampaignStateValue.DIAGNOSTICS_READY, CampaignStateValue.RESULTS_READY):
            ready_state = state.state
        else:
            abort(409, description=f"Cannot complete a campaign in state {state.state.value!r}.")
        load_persisted = load_arc_diagnostics if ready_state == CampaignStateValue.RESULTS_READY else load_diagnostics
        if load_persisted(ws) is None:
            abort(409, description="There are no persisted diagnostics to complete this campaign from.")
        if resuming:
            update_state(ws, ready_state, notes="recovered: adopting persisted diagnostics")
        update_state(ws, CampaignStateValue.COMPLETED_PHASE1, notes="recovered: completed from persisted diagnostics")
        append_event(
            ws / "decision_log.jsonl",
            {"event": "campaign_recovered", "recovery": "finalize", "from_state": state.state.value},
        )
        flash("Campaign completed from its persisted diagnostics.", "info")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/abandon", methods=["POST"])
    def campaign_abandon(campaign_id: str) -> Response:
        """End a run whose supervising Carmel process died.

        Refuses while anything is still executing — see
        :func:`~carmel.services.execution.abandon_t3_run` /
        :func:`~carmel.services.execution.abandon_arc_run`, which stop an
        orphaned tool tree before they let the campaign be called failed.
        """
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        campaign = load_campaign(ws)
        state = load_state(ws)
        if state.state == CampaignStateValue.RUNNING_T3:
            abandon_run = abandon_t3_run
        elif state.state == CampaignStateValue.RUNNING_ARC:
            abandon_run = abandon_arc_run
        else:
            abort(409, description=f"Cannot abandon a run for a campaign in state {state.state.value!r}.")
        try:
            _, report = abandon_run(ws, campaign)
        except RunStillLiveError as e:
            abort(409, description=str(e))
        except InvalidTransitionError:
            abort(409, description="The campaign changed state while it was being abandoned.")
        flash(f"Run abandoned. {report.detail}", "info")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/free-text", methods=["POST"])
    def campaign_free_text(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        free_text = request.form.get("free_text", "")
        parser = StubIntakeParser()
        review = parser.parse(free_text)
        write_intake_review(ws, review)
        flash("Free-text review saved as intake_review.md (advisory only).", "info")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/svg/<artifact>")
    def campaign_svg(campaign_id: str, artifact: str) -> str | tuple[str, int]:
        """Serve a persisted SVG artifact for the dashboard graphical view.

        ``?tool=arc`` serves the ARC run's selection SVGs from
        ``models/arc/``; the default (``t3``) keeps serving T3's from
        ``models/``. The two trees are separate on disk for the same reason
        the diagnostics files are: one tool's artifacts must never be
        presented as the other's.
        """
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        allowed = {"species_selection.svg", "reactions_selection.svg", "pdep_networks_selection.svg"}
        if artifact not in allowed:
            abort(404)
        tool = request.args.get("tool", "t3")
        if tool == "t3":
            models_dir = ws / MODELS_DIR_NAME
        elif tool == "arc":
            models_dir = ws / MODELS_DIR_NAME / ARC_MODELS_SUBDIR_NAME
        else:
            abort(404)
        path = models_dir / artifact
        if not path.exists():
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40">'
                '<text x="10" y="25" font-family="sans-serif" font-size="12" fill="#888">'
                "no diagnostics yet</text></svg>"
            ), 200
        return path.read_text(encoding="utf-8")

    @app.template_filter("format_datetime")
    def format_datetime(value: Any) -> str:
        """Render a datetime or ISO string as a readable timestamp."""
        if isinstance(value, datetime):
            return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                return value
        return str(value)

    return app
