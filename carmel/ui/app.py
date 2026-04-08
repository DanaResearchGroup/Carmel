"""Flask application factory and routes for Carmel.

The UI layer is intentionally thin: route handlers parse form data,
delegate to service modules, and render templates. All business logic
lives in :mod:`carmel.services`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Flask, abort, flash, redirect, render_template, request, url_for
from werkzeug.wrappers.response import Response

from carmel.logger import get_logger
from carmel.schemas.approval import ApprovalStatus
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
from carmel.schemas.state import CampaignStateValue
from carmel.services.approvals import (
    record_decision,
)
from carmel.services.campaigns import (
    create_campaign,
    find_campaign_workspace,
    list_campaigns,
    load_campaign,
)
from carmel.services.decision_log import read_events
from carmel.services.execution import execute_t3_action, load_diagnostics
from carmel.services.intake import StubIntakeParser, write_intake_review
from carmel.services.planner import load_plan, plan_and_save
from carmel.services.state_machine import load_state, update_state

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
        - observables (textarea, one per line: ``name[,species]``)
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
        observables.append(TargetObservable(name=parts[0], species=species))

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
    app.secret_key = os.environ.get("CARMEL_SECRET_KEY", "carmel-dev-secret-do-not-use-in-prod")
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
        diagnostics = load_diagnostics(ws)
        events = read_events(ws / "decision_log.jsonl")
        latest_run_path = None
        runs_dir = ws / "runs"
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob("*.json"))
            latest_run_path = run_files[-1] if run_files else None
        return render_template(
            "campaign_dashboard.html",
            campaign=campaign,
            state=state,
            plan=plan,
            diagnostics=diagnostics,
            events=events[-20:],
            latest_run_path=latest_run_path.name if latest_run_path else None,
            workspace_root=str(ws),
        )

    @app.route("/campaigns/<campaign_id>/plan", methods=["POST"])
    def campaign_plan(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        campaign = load_campaign(ws)
        plan = plan_and_save(ws, campaign)
        if plan.requires_approval:
            update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        else:
            update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
            update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="auto-approved")
            for action in plan.actions:
                record_decision(ws, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/approve", methods=["POST"])
    def campaign_approve(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        plan = load_plan(ws)
        for action in plan.actions:
            record_decision(
                ws,
                action.action_id,
                ApprovalStatus.APPROVED,
                decided_by="user",
                rationale="approved via UI",
            )
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="user-approved")
        return redirect(url_for("campaign_dashboard", campaign_id=campaign_id))

    @app.route("/campaigns/<campaign_id>/reject", methods=["POST"])
    def campaign_reject(campaign_id: str) -> Response:
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        plan = load_plan(ws)
        for action in plan.actions:
            record_decision(
                ws,
                action.action_id,
                ApprovalStatus.REJECTED,
                decided_by="user",
                rationale="rejected via UI",
            )
        update_state(ws, CampaignStateValue.BLOCKED, notes="user-rejected")
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
        execute_t3_action(ws, campaign, plan.actions[0])
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
        """Serve a persisted SVG artifact for the dashboard graphical view."""
        ws = find_campaign_workspace(workspaces, campaign_id)
        if ws is None:
            abort(404)
        allowed = {"species_selection.svg", "reactions_selection.svg", "pdep_networks_selection.svg"}
        if artifact not in allowed:
            abort(404)
        path = ws / "models" / artifact
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
