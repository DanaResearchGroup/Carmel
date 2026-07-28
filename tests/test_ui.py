"""Tests for the Carmel Flask UI."""

import logging
import os
import signal
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from carmel.schemas import CampaignStateValue, DiagnosticsV1
from carmel.schemas.approval import ActionKind, ApprovalPolicy, ApprovalStatus
from carmel.schemas.run import FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.services.approvals import record_decision, save_policy
from carmel.services.campaigns import find_campaign_workspace
from carmel.services.decision_log import read_events
from carmel.services.execution import (
    load_arc_diagnostics,
    save_arc_diagnostics,
    save_diagnostics,
    save_run_record,
)
from carmel.services.planner import load_plan
from carmel.services.recovery import load_active_run, supervise_run
from carmel.services.state_machine import load_state, update_state
from carmel.ui import create_app
from carmel.ui.app import _resolve_workspaces_root
from tests.helpers import _strand_active_run, _tool_tree

ARC_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "arc" / "sample_project"


@pytest.fixture
def client(tmp_path: Path) -> FlaskClient:
    app = create_app(workspaces_root=tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def workspaces_root(tmp_path: Path) -> Path:
    return tmp_path


def _csrf_token(client: FlaskClient) -> str:
    """Render a form page to prime the session and return its CSRF token."""
    assert client.get("/campaigns/new").status_code == 200
    with client.session_transaction() as session:
        token = session["csrf_token"]
    assert isinstance(token, str) and token
    return token


def _post(client: FlaskClient, path: str, data: dict[str, str] | None = None, **kwargs: Any) -> TestResponse:
    """POST with a valid CSRF token injected, mirroring a real form submit."""
    payload: dict[str, str] = dict(data or {})
    payload.setdefault("csrf_token", _csrf_token(client))
    return client.open(path, method="POST", data=payload, **kwargs)


def _create_via_form(client: FlaskClient, name: str = "ethanol") -> str:
    response = _post(
        client,
        "/campaigns/new",
        data={
            "workspace_name": name,
            "mixture_components": "CH4,0.05\nO2,0.20\nN2,0.75",
            "observables": "ignition_delay",
            "reactors": "jsr,800,1200,1.0,5.0,1.0",
            "cpu_hours": "20",
            "experiment_budget": "0",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    location = response.headers["Location"]
    return location.rsplit("/", 1)[-1]


class TestIndex:
    def test_loads(self, client: FlaskClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Campaigns" in response.data

    def test_empty_state(self, client: FlaskClient) -> None:
        response = client.get("/")
        assert b"No campaigns" in response.data


class TestCampaignNew:
    def test_get_loads(self, client: FlaskClient) -> None:
        response = client.get("/campaigns/new")
        assert response.status_code == 200
        assert b"New Campaign" in response.data

    def test_post_creates(self, client: FlaskClient) -> None:
        campaign_id = _create_via_form(client)
        response = client.get(f"/campaigns/{campaign_id}")
        assert response.status_code == 200

    def test_invalid_form_returns_400(self, client: FlaskClient) -> None:
        response = _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "x",
                "mixture_components": "incomplete-line",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200,1,5",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 400


class TestCampaignDashboard:
    def test_dashboard_renders(self, client: FlaskClient) -> None:
        cid = _create_via_form(client)
        response = client.get(f"/campaigns/{cid}")
        assert response.status_code == 200
        assert b"ethanol" in response.data
        assert b"ready_for_planning" in response.data

    def test_dashboard_unknown_404(self, client: FlaskClient) -> None:
        response = client.get("/campaigns/unknown-id")
        assert response.status_code == 404


class TestPlanFlow:
    def test_generate_plan(self, client: FlaskClient) -> None:
        cid = _create_via_form(client)
        response = _post(client, f"/campaigns/{cid}/plan", follow_redirects=False)
        assert response.status_code == 302
        dashboard = client.get(f"/campaigns/{cid}").data
        assert b"baseline" in dashboard.lower() or b"Plan" in dashboard

    def test_approve_action(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        # Generate a plan that requires approval (high cpu) — set policy via direct call
        from carmel.schemas.approval import ApprovalPolicy
        from carmel.services.approvals import save_policy
        from carmel.services.campaigns import find_campaign_workspace

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        save_policy(ws, ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.1))
        _post(client, f"/campaigns/{cid}/plan")
        response = _post(client, f"/campaigns/{cid}/approve", follow_redirects=False)
        assert response.status_code == 302
        from carmel.services.state_machine import load_state

        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_reject_action(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        from carmel.schemas.approval import ApprovalPolicy
        from carmel.services.approvals import save_policy
        from carmel.services.campaigns import find_campaign_workspace

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        save_policy(ws, ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.1))
        _post(client, f"/campaigns/{cid}/plan")
        response = _post(client, f"/campaigns/{cid}/reject", follow_redirects=False)
        assert response.status_code == 302
        from carmel.services.state_machine import load_state

        assert load_state(ws).state == CampaignStateValue.BLOCKED


class TestFreeTextIntake:
    def test_free_text_creates_review(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        from carmel.services.campaigns import find_campaign_workspace

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        response = _post(
            client,
            f"/campaigns/{cid}/free-text",
            data={"free_text": "we want a methane mechanism"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        review = ws / "intake_review.md"
        assert review.exists()
        assert "methane mechanism" in review.read_text()


class TestResolveWorkspacesRoot:
    """Tests for workspaces-root resolution precedence."""

    def test_explicit_argument_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARMEL_WORKSPACES", str(tmp_path / "from-env"))
        assert _resolve_workspaces_root(tmp_path / "explicit") == tmp_path / "explicit"

    def test_env_var_used_when_no_argument(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARMEL_WORKSPACES", str(tmp_path / "from-env"))
        assert _resolve_workspaces_root(None) == tmp_path / "from-env"

    def test_env_var_expands_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARMEL_WORKSPACES", "~/some_workspaces")
        assert _resolve_workspaces_root(None) == Path.home() / "some_workspaces"

    def test_falls_back_to_home_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARMEL_WORKSPACES", "")
        assert _resolve_workspaces_root(None) == Path.home() / "carmel_workspaces"

    def test_falls_back_to_home_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARMEL_WORKSPACES", raising=False)
        assert _resolve_workspaces_root(None) == Path.home() / "carmel_workspaces"


class TestFormParsing:
    """Tests for translating posted form text into a CampaignInput."""

    def test_blank_lines_are_ignored(self, client: FlaskClient) -> None:
        response = _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "blanks",
                "mixture_components": "CH4,0.5\n\n  \nO2,0.5",
                "observables": "ignition_delay\n\n  \n",
                "reactors": "jsr,800,1200,1.0,5.0\n\n  \n",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
            follow_redirects=False,
        )
        assert response.status_code == 302

    def test_non_numeric_mole_fraction_returns_400(self, client: FlaskClient) -> None:
        response = _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "bad-fraction",
                "mixture_components": "CH4,not-a-number",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200,1.0,5.0",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 400
        assert b"invalid mole fraction" in response.data

    def test_short_reactor_line_returns_400(self, client: FlaskClient) -> None:
        response = _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "bad-reactor",
                "mixture_components": "CH4,1.0",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 400
        assert b"reactor line must be" in response.data

    def test_missing_required_field_returns_400(self, client: FlaskClient) -> None:
        response = _post(
            client,
            "/campaigns/new",
            data={
                "mixture_components": "CH4,1.0",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200,1.0,5.0",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 400

    def test_observable_species_column_parsed(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.services.campaigns import find_campaign_workspace, load_campaign

        _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "with-species",
                "mixture_components": "CH4,1.0",
                "observables": "species_profile,OH",
                "reactors": "jsr,800,1200,1.0,5.0",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        campaigns = list(workspaces_root.glob("with-species"))
        assert campaigns
        campaign = load_campaign(campaigns[0])
        assert campaign.input.target_observables[0].species == "OH"
        assert find_campaign_workspace(workspaces_root, campaign.campaign_id) == campaigns[0]

    def test_duplicate_workspace_returns_400(self, client: FlaskClient) -> None:
        _create_via_form(client, "dupe")
        response = _post(
            client,
            "/campaigns/new",
            data={
                "workspace_name": "dupe",
                "mixture_components": "CH4,1.0",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200,1.0,5.0",
                "cpu_hours": "10",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 400
        assert b"already exists" in response.data


class TestFavicon:
    def test_redirects_to_svg(self, client: FlaskClient) -> None:
        response = client.get("/favicon.ico", follow_redirects=False)
        assert response.status_code == 301
        assert response.headers["Location"].endswith("favicon.svg")


class TestUnknownCampaign404:
    """Every campaign-scoped route must 404 on an unknown campaign id."""

    @pytest.mark.parametrize(
        "path",
        [
            "/campaigns/nope/plan",
            "/campaigns/nope/approve",
            "/campaigns/nope/reject",
            "/campaigns/nope/run",
            "/campaigns/nope/free-text",
        ],
    )
    def test_post_routes_404(self, client: FlaskClient, path: str) -> None:
        assert _post(client, path).status_code == 404

    def test_svg_route_404(self, client: FlaskClient) -> None:
        assert client.get("/campaigns/nope/svg/species_selection.svg").status_code == 404


class TestCampaignRun:
    def test_run_redirects_to_dashboard(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.ui import app as ui_app

        called: dict[str, object] = {}

        # The run route dispatches through execute_next_action now (the
        # dispatcher drives the same background T3 lifecycle start_t3_action
        # exposes); the observable contract is unchanged: dispatch, redirect.
        def _fake_execute(ws: Path, campaign: object, **kwargs: object) -> None:
            called["campaign_id"] = getattr(campaign, "campaign_id", None)
            return None

        monkeypatch.setattr(ui_app, "execute_next_action", _fake_execute)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["Location"].endswith(cid)
        assert called["campaign_id"] == cid

    def test_run_without_plan_actions_returns_400(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.planner import load_plan, save_plan

        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = load_plan(ws)
        save_plan(ws, plan.model_copy(update={"actions": []}))
        assert _post(client, f"/campaigns/{cid}/run").status_code == 400


class TestDashboardLatestRun:
    def test_latest_run_file_is_surfaced(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.services.campaigns import find_campaign_workspace

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        runs = ws / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "aaa.json").write_text("{}")
        (runs / "zzz.json").write_text("{}")
        response = client.get(f"/campaigns/{cid}")
        assert response.status_code == 200
        assert b"zzz.json" in response.data

    def test_empty_runs_dir_is_tolerated(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.services.campaigns import find_campaign_workspace

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        (ws / "runs").mkdir(parents=True, exist_ok=True)
        assert client.get(f"/campaigns/{cid}").status_code == 200

    def test_missing_runs_dir_is_tolerated(self, client: FlaskClient, workspaces_root: Path) -> None:
        import shutil

        from carmel.services.campaigns import find_campaign_workspace

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        shutil.rmtree(ws / "runs", ignore_errors=True)
        assert client.get(f"/campaigns/{cid}").status_code == 200


class TestFormatDatetimeFilter:
    """Tests for the ``format_datetime`` Jinja filter."""

    @pytest.fixture
    def format_datetime(self, tmp_path: Path) -> Any:
        app = create_app(workspaces_root=tmp_path)
        return app.jinja_env.filters["format_datetime"]

    def test_formats_datetime(self, format_datetime: Any) -> None:
        value = datetime(2026, 4, 8, 9, 15, tzinfo=UTC)
        assert format_datetime(value) == "2026-04-08 09:15 UTC"

    def test_formats_iso_string(self, format_datetime: Any) -> None:
        assert format_datetime("2026-04-08T09:15:00+00:00") == "2026-04-08 09:15 UTC"

    def test_returns_unparseable_string_unchanged(self, format_datetime: Any) -> None:
        assert format_datetime("not a timestamp") == "not a timestamp"

    def test_stringifies_other_types(self, format_datetime: Any) -> None:
        assert format_datetime(42) == "42"
        assert format_datetime(None) == "None"


class TestSvgArtifacts:
    def test_svg_route_renders_diagnostics(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        from datetime import datetime

        from carmel.schemas import (
            DiagnosticsV1,
            PDepNetworkSelection,
            ReactionSelection,
            SpeciesSelection,
        )
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.drawing import write_selection_svgs

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        d = DiagnosticsV1(
            campaign_id=cid,
            run_id="r1",
            generated_at=datetime.now(UTC),
            species_to_compute=[SpeciesSelection(label="OH")],
            reactions_to_compute=[ReactionSelection(label="r1", reactants=["A"], products=["B"])],
            pdep_networks_to_compute=[PDepNetworkSelection(network_id="N1", species=["A"])],
        )
        save_diagnostics(ws, d)
        write_selection_svgs(
            ws / "models",
            d.species_to_compute,
            d.reactions_to_compute,
            d.pdep_networks_to_compute,
        )
        for art in ("species_selection.svg", "reactions_selection.svg", "pdep_networks_selection.svg"):
            response = client.get(f"/campaigns/{cid}/svg/{art}")
            assert response.status_code == 200
            assert b"<svg" in response.data

    def test_svg_route_unknown_artifact_404(self, client: FlaskClient) -> None:
        cid = _create_via_form(client)
        response = client.get(f"/campaigns/{cid}/svg/unknown.svg")
        assert response.status_code == 404

    def test_svg_route_returns_placeholder_when_missing(self, client: FlaskClient) -> None:
        cid = _create_via_form(client)
        response = client.get(f"/campaigns/{cid}/svg/species_selection.svg")
        assert response.status_code == 200
        assert b"<svg" in response.data


def _plan_pending_approval(client: FlaskClient, workspaces_root: Path, name: str = "gated") -> tuple[str, Path]:
    """Create a campaign whose plan requires human approval and generate it."""
    cid = _create_via_form(client, name)
    ws = find_campaign_workspace(workspaces_root, cid)
    assert ws is not None
    save_policy(ws, ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.1))
    assert _post(client, f"/campaigns/{cid}/plan").status_code == 302
    return cid, ws


class TestPlanStateGuard:
    """The /plan route must validate the state transition before writing plan.json."""

    def test_replan_after_user_approval_conflicts_and_preserves_plan(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws = _plan_pending_approval(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/approve").status_code == 302
        plan_before = (ws / "plan.json").read_bytes()
        action_id_before = load_plan(ws).actions[0].action_id
        response = _post(client, f"/campaigns/{cid}/plan")
        assert response.status_code == 409
        assert (ws / "plan.json").read_bytes() == plan_before
        assert load_plan(ws).actions[0].action_id == action_id_before
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_replan_after_auto_approval_conflicts_and_preserves_plan(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid = _create_via_form(client, "auto-gated")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert _post(client, f"/campaigns/{cid}/plan").status_code == 302
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        plan_before = (ws / "plan.json").read_bytes()
        response = _post(client, f"/campaigns/{cid}/plan")
        assert response.status_code == 409
        assert (ws / "plan.json").read_bytes() == plan_before


class TestApprovalAuditIntegrity:
    """Approval/rejection events must only be logged for operations that took effect."""

    def test_double_approve_records_exactly_one_event_per_action(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws = _plan_pending_approval(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/approve").status_code == 302
        assert _post(client, f"/campaigns/{cid}/approve").status_code == 409
        action_id = load_plan(ws).actions[0].action_id
        approvals = [
            e
            for e in read_events(ws / "decision_log.jsonl")
            if e.get("event") == "approval_decision"
            and e.get("action_id") == action_id
            and e.get("status") == "approved"
        ]
        assert len(approvals) == 1

    def test_double_reject_records_exactly_one_event_per_action(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws = _plan_pending_approval(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/reject").status_code == 302
        assert _post(client, f"/campaigns/{cid}/reject").status_code == 409
        action_id = load_plan(ws).actions[0].action_id
        rejections = [
            e
            for e in read_events(ws / "decision_log.jsonl")
            if e.get("event") == "approval_decision"
            and e.get("action_id") == action_id
            and e.get("status") == "rejected"
        ]
        assert len(rejections) == 1
        assert load_state(ws).state == CampaignStateValue.BLOCKED

    def test_approve_on_blocked_campaign_is_refused_and_not_runnable(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.ui import app as ui_app

        executed: dict[str, object] = {}

        def _fake_execute(ws: Path, campaign: object, **kwargs: object) -> None:
            executed["campaign_id"] = getattr(campaign, "campaign_id", None)
            return None

        monkeypatch.setattr(ui_app, "execute_next_action", _fake_execute)
        cid, ws = _plan_pending_approval(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/reject").status_code == 302
        assert _post(client, f"/campaigns/{cid}/approve").status_code == 409
        assert load_state(ws).state == CampaignStateValue.BLOCKED
        events = read_events(ws / "decision_log.jsonl")
        assert not any(e.get("event") == "approval_decision" and e.get("status") == "approved" for e in events)
        assert _post(client, f"/campaigns/{cid}/run").status_code == 409
        assert executed == {}


class TestRunApprovalGate:
    """The /run route must refuse actions with no effective approval decision."""

    def test_run_without_recorded_approval_returns_409(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.ui import app as ui_app

        executed: dict[str, object] = {}

        def _fake_execute(ws: Path, campaign: object, **kwargs: object) -> None:
            executed["campaign_id"] = getattr(campaign, "campaign_id", None)
            return None

        monkeypatch.setattr(ui_app, "execute_next_action", _fake_execute)
        cid, ws = _plan_pending_approval(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/run").status_code == 409
        assert executed == {}


class TestCsrfProtection:
    """Every state-changing POST must carry the session CSRF token."""

    _FORM = {
        "workspace_name": "csrf-check",
        "mixture_components": "CH4,1.0",
        "observables": "ignition_delay",
        "reactors": "jsr,800,1200,1.0,5.0",
        "cpu_hours": "10",
        "experiment_budget": "0",
    }

    def test_post_without_token_returns_400(self, client: FlaskClient, workspaces_root: Path) -> None:
        response = client.post("/campaigns/new", data=self._FORM)
        assert response.status_code == 400
        assert not (workspaces_root / "csrf-check").exists()

    def test_post_with_wrong_token_of_correct_length_returns_400(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        token = _csrf_token(client)
        wrong = ("a" if token[0] != "a" else "b") + token[1:]
        assert wrong != token and len(wrong) == len(token)
        response = client.post("/campaigns/new", data={**self._FORM, "csrf_token": wrong})
        assert response.status_code == 400
        assert not (workspaces_root / "csrf-check").exists()

    def test_post_with_token_but_unprimed_session_returns_400(self, client: FlaskClient) -> None:
        response = client.post("/campaigns/new", data={**self._FORM, "csrf_token": "attacker-guess"})
        assert response.status_code == 400

    def test_post_with_valid_token_succeeds(self, client: FlaskClient) -> None:
        token = _csrf_token(client)
        response = client.post("/campaigns/new", data={**self._FORM, "csrf_token": token})
        assert response.status_code == 302

    def test_token_is_stable_within_a_session(self, client: FlaskClient) -> None:
        assert _csrf_token(client) == _csrf_token(client)

    def test_get_requests_do_not_require_token(self, client: FlaskClient) -> None:
        assert client.get("/").status_code == 200


class TestCsrfFieldRendered:
    """Every rendered POST form must embed the hidden CSRF field."""

    @staticmethod
    def _assert_every_post_form_has_token(html: str, expected_forms: int) -> None:
        assert html.count('method="POST"') == expected_forms
        assert html.count('name="csrf_token"') == expected_forms

    def test_create_form(self, client: FlaskClient) -> None:
        html = client.get("/campaigns/new").data.decode()
        self._assert_every_post_form_has_token(html, expected_forms=1)

    def test_dashboard_ready_for_planning(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "render-ready")
        html = client.get(f"/campaigns/{cid}").data.decode()
        self._assert_every_post_form_has_token(html, expected_forms=3)  # plan (T3) + plan (ARC) + free-text

    def test_dashboard_pending_approval(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ = _plan_pending_approval(client, workspaces_root, "render-pending")
        html = client.get(f"/campaigns/{cid}").data.decode()
        # per-action approve + per-action reject + approve all + reject all + free-text
        self._assert_every_post_form_has_token(html, expected_forms=5)

    def test_dashboard_approved(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "render-approved")
        assert _post(client, f"/campaigns/{cid}/plan").status_code == 302
        html = client.get(f"/campaigns/{cid}").data.decode()
        # per-action reject + run + free-text
        self._assert_every_post_form_has_token(html, expected_forms=3)


class TestSecretKey:
    """Secret-key resolution: env var honored, otherwise random per process."""

    def test_unset_env_yields_distinct_random_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARMEL_SECRET_KEY", raising=False)
        app_one = create_app(workspaces_root=tmp_path / "one")
        app_two = create_app(workspaces_root=tmp_path / "two")
        assert app_one.secret_key != app_two.secret_key
        assert app_one.secret_key != "carmel-dev-secret-do-not-use-in-prod"

    def test_env_value_is_honored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARMEL_SECRET_KEY", "configured-secret")
        app = create_app(workspaces_root=tmp_path)
        assert app.secret_key == "configured-secret"


class TestRunIsAsynchronous:
    """POST /run must return while T3 is still running.

    Executed inline, a run holds the request open for
    ``estimated_cpu_hours * 3600 + 600`` seconds — 3.2 hours for a minimal
    campaign. The browser times out long before that, so the redirect to
    the auto-refreshing running dashboard never arrives and the whole
    ``RUNNING_T3`` UX is unreachable from the tab that started the run.

    These tests therefore assert on what the *user* gets: a response that
    arrives while the adapter is still blocked mid-run.
    """

    class _BlockingAdapter:
        """Blocks inside ``run`` until released, like a real T3 invocation."""

        submission_mode = SubmissionMode.SUBPROCESS

        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def run(
            self,
            workspace_root: Path,
            campaign: Any,
            action: Any,
            on_process_start: Callable[[int, list[str]], None] | None = None,
        ) -> tuple[RunRecord, None]:
            self.entered.set()
            assert self.release.wait(timeout=60), "the test never released the adapter"
            return RunRecord(
                run_id="blocking-run",
                action_id=action.action_id,
                tool_name="t3",
                status=RunStatus.FAILED,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                submission_mode=SubmissionMode.SUBPROCESS,
                error_message="released by the test",
            ), None

    def _arrange(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[str, _BlockingAdapter, list[threading.Thread]]:
        from carmel.services import execution as execution_module
        from carmel.ui import app as ui_app

        adapter = self._BlockingAdapter()
        monkeypatch.setattr(execution_module, "_default_adapter", lambda: adapter)

        # The run route dispatches through execute_next_action, which starts
        # the same background T3 lifecycle start_t3_action exposes; capture
        # the background threads via the returned tickets.
        threads: list[threading.Thread] = []
        real_execute = ui_app.execute_next_action

        def _capture(ws: Path, campaign: Any, **kwargs: Any) -> Any:
            ticket = real_execute(ws, campaign, **kwargs)
            if ticket is not None and ticket.thread is not None:
                threads.append(ticket.thread)
            return ticket

        monkeypatch.setattr(ui_app, "execute_next_action", _capture)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        return cid, adapter, threads

    def test_run_returns_while_t3_is_still_running(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cid, adapter, threads = self._arrange(client, monkeypatch)
        try:
            response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
            assert response.status_code == 302
            # The response is in hand and the adapter has not been released:
            # inline execution could not have produced this.
            assert adapter.entered.wait(timeout=30), "the background run never started"
            assert not adapter.release.is_set()

            ws = find_campaign_workspace(workspaces_root, cid)
            assert ws is not None
            assert load_state(ws).state == CampaignStateValue.RUNNING_T3
        finally:
            adapter.release.set()
            for thread in threads:
                thread.join(timeout=60)

    def test_background_run_completes_the_state_machine(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returning early must not mean the run is forgotten."""
        cid, adapter, threads = self._arrange(client, monkeypatch)
        _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert adapter.entered.wait(timeout=30)
        adapter.release.set()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive()

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert load_state(ws).state == CampaignStateValue.FAILED
        events = read_events(ws / "decision_log.jsonl")
        assert any(event.get("event") == "t3_run_finished" for event in events)

    def test_second_run_while_running_is_a_conflict_not_a_crash(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The running dashboard auto-refreshes, so re-clicking Run is easy."""
        cid, adapter, threads = self._arrange(client, monkeypatch)
        try:
            assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 302
            assert adapter.entered.wait(timeout=30)
            assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 409
        finally:
            adapter.release.set()
            for thread in threads:
                thread.join(timeout=60)
        assert len(threads) == 1, "the rejected second submit still started a background run"

    def test_losing_a_concurrent_race_is_a_409_not_a_500(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The preflight is a read, so it cannot be authoritative.

        Two POSTs can both pass `can_transition` and then race the locked
        pre-transition inside the dispatcher. The loser must see a conflict,
        not a traceback.
        """
        from carmel.services.state_machine import InvalidTransitionError
        from carmel.ui import app as ui_app

        def _lost_the_race(ws: Path, campaign: Any, **kwargs: Any) -> Any:
            raise InvalidTransitionError("approved_for_execution -> running_t3 is not permitted")

        monkeypatch.setattr(ui_app, "execute_next_action", _lost_the_race)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 409

    def test_a_run_lost_to_a_live_lock_is_a_409_not_a_500(self, client: FlaskClient, workspaces_root: Path) -> None:
        """The real lock-first ordering, not a monkeypatched stand-in.

        Since supervision is taken before the transition, the loser of two
        racing runs trips the *run lock*, not ``can_transition`` —
        ``RunAlreadySupervisedError``, a different type than the state race
        raises. Holding the real lock proves the route treats it as a
        conflict rather than 500-ing on an unhandled exception.
        """
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        with supervise_run(ws, "act-holds-the-lock"):
            assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 409

    def test_a_run_whose_lock_state_is_unknowable_is_a_503(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A workspace whose filesystem cannot ``flock`` is a 503, not a 500.

        No working lock is an environment fault the operator can act on, not
        a crash. It must not surface as an opaque server error, and must not
        be mistaken for the 409 a genuine concurrent run gets.
        """
        import errno

        from carmel.services import recovery

        def _no_locks(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOLCK, "no locks available")

        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        monkeypatch.setattr(recovery.fcntl, "flock", _no_locks)
        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 503


def _failed_campaign(
    client: FlaskClient,
    workspaces_root: Path,
    failed_from: CampaignStateValue,
    name: str = "wedged",
) -> tuple[str, Path]:
    """Drive a campaign into FAILED from a chosen origin.

    Every step goes through ``update_state`` rather than writing the state
    file, so ``failed_from`` is recorded the way a real failure records it.
    """
    cid = _create_via_form(client, name)
    ws = find_campaign_workspace(workspaces_root, cid)
    assert ws is not None
    route = {
        CampaignStateValue.READY_FOR_PLANNING: [],
        CampaignStateValue.PLAN_PENDING_APPROVAL: [CampaignStateValue.PLAN_PENDING_APPROVAL],
        CampaignStateValue.APPROVED_FOR_EXECUTION: [
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
        ],
        CampaignStateValue.RUNNING_T3: [
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
        ],
        CampaignStateValue.DIAGNOSTICS_READY: [
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.DIAGNOSTICS_READY,
        ],
        CampaignStateValue.RUNNING_ARC: [
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_ARC,
        ],
        CampaignStateValue.RESULTS_READY: [
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_ARC,
            CampaignStateValue.RESULTS_READY,
        ],
    }[failed_from]
    for step in route:
        update_state(ws, step)
    update_state(ws, CampaignStateValue.FAILED, notes="something went wrong")
    assert load_state(ws).failed_from == failed_from
    return cid, ws


class TestReplanRoute:
    def test_a_failed_campaign_returns_to_planning(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.READY_FOR_PLANNING)
        assert _post(client, f"/campaigns/{cid}/replan", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.READY_FOR_PLANNING

    def test_it_works_from_every_origin_a_campaign_can_fail_from(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """The wedge, exercised end-to-end through the UI.

        Before recovery edges existed each of these returned 500 from
        every button the dashboard offered.
        """
        origins = [
            CampaignStateValue.READY_FOR_PLANNING,
            CampaignStateValue.PLAN_PENDING_APPROVAL,
            CampaignStateValue.APPROVED_FOR_EXECUTION,
            CampaignStateValue.RUNNING_T3,
            CampaignStateValue.DIAGNOSTICS_READY,
            CampaignStateValue.RUNNING_ARC,
            CampaignStateValue.RESULTS_READY,
        ]
        for index, origin in enumerate(origins):
            cid, ws = _failed_campaign(client, workspaces_root, origin, name=f"wedged-{index}")
            assert _post(client, f"/campaigns/{cid}/replan", follow_redirects=False).status_code == 302
            assert load_state(ws).state == CampaignStateValue.READY_FOR_PLANNING

    def test_a_rejected_plan_can_be_re_planned(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _plan_pending_approval(client, workspaces_root, "rejected")
        assert _post(client, f"/campaigns/{cid}/reject").status_code == 302
        assert load_state(ws).state == CampaignStateValue.BLOCKED
        assert _post(client, f"/campaigns/{cid}/replan", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.READY_FOR_PLANNING

    def test_a_running_campaign_cannot_be_re_planned(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "busy")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        assert _post(client, f"/campaigns/{cid}/replan", follow_redirects=False).status_code == 409

    def test_an_unknown_campaign_is_a_404(self, client: FlaskClient) -> None:
        assert _post(client, "/campaigns/nope/replan", follow_redirects=False).status_code == 404


class TestRetryRoute:
    def test_a_failed_run_can_be_retried(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_T3)
        assert _post(client, f"/campaigns/{cid}/retry", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_a_run_that_failed_before_launching_can_be_retried(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """A run that never reached RUNNING_T3 keeps the approval it holds.

        A campaign failed from ``approved_for_execution`` — its runner
        thread never started, say — already carries an approval for this
        exact plan, so retry returns it there rather than discarding the
        approval through re-planning.
        """
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.APPROVED_FOR_EXECUTION)
        assert _post(client, f"/campaigns/{cid}/retry", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_a_failed_arc_run_can_be_retried(self, client: FlaskClient, workspaces_root: Path) -> None:
        """RUNNING_ARC is a first-class running state: retry works for it too."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_ARC)
        assert _post(client, f"/campaigns/{cid}/retry", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_a_failure_before_approval_cannot_be_retried(self, client: FlaskClient, workspaces_root: Path) -> None:
        """Retrying from an unapproved failure would skip the HITL gate."""
        for index, origin in enumerate(
            [
                CampaignStateValue.READY_FOR_PLANNING,
                CampaignStateValue.PLAN_PENDING_APPROVAL,
                CampaignStateValue.DIAGNOSTICS_READY,
            ]
        ):
            cid, ws = _failed_campaign(client, workspaces_root, origin, name=f"unapproved-{index}")
            assert _post(client, f"/campaigns/{cid}/retry", follow_redirects=False).status_code == 409
            assert load_state(ws).state == CampaignStateValue.FAILED

    def test_an_unknown_campaign_is_a_404(self, client: FlaskClient) -> None:
        assert _post(client, "/campaigns/nope/retry", follow_redirects=False).status_code == 404


class TestFinalizeRoute:
    @staticmethod
    def _with_diagnostics(ws: Path) -> None:
        save_diagnostics(
            ws,
            DiagnosticsV1(
                run_id="recovered-run",
                campaign_id="c",
                generated_at=datetime.now(UTC),
                species_to_compute=[],
                reactions_to_compute=[],
                pdep_networks_to_compute=[],
            ),
        )

    def test_a_campaign_that_failed_while_finalizing_can_adopt_its_diagnostics(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.DIAGNOSTICS_READY)
        self._with_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_completing_without_diagnostics_is_refused(self, client: FlaskClient, workspaces_root: Path) -> None:
        """Nothing may claim a phase is complete on output that is not there."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.DIAGNOSTICS_READY)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 409
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_a_campaign_with_diagnostics_ready_can_be_completed(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid = _create_via_form(client, "diag-ready")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        for step in [CampaignStateValue.RUNNING_T3, CampaignStateValue.DIAGNOSTICS_READY]:
            update_state(ws, step)
        self._with_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    @staticmethod
    def _with_arc_diagnostics(ws: Path) -> None:
        save_arc_diagnostics(
            ws,
            DiagnosticsV1(
                run_id="recovered-arc-run",
                campaign_id="c",
                generated_at=datetime.now(UTC),
                species_to_compute=[],
                reactions_to_compute=[],
                pdep_networks_to_compute=[],
            ),
        )

    def test_a_campaign_that_failed_while_finalizing_arc_results_can_adopt_them(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """ARC mirror: FAILED-from-RESULTS_READY finalizes via RESULTS_READY."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RESULTS_READY)
        self._with_arc_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_completing_arc_results_without_arc_diagnostics_is_refused(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """T3 diagnostics must not stand in for ARC's: each path gates on its own."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RESULTS_READY)
        self._with_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 409
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_a_campaign_with_results_ready_can_be_completed(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "results-ready")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        for step in [CampaignStateValue.RUNNING_ARC, CampaignStateValue.RESULTS_READY]:
            update_state(ws, step)
        self._with_arc_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_a_failure_from_anywhere_else_cannot_be_finalized(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_T3)
        self._with_diagnostics(ws)
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 409

    def test_an_unknown_campaign_is_a_404(self, client: FlaskClient) -> None:
        assert _post(client, "/campaigns/nope/finalize", follow_redirects=False).status_code == 404


class TestAbandonRoute:
    @staticmethod
    def _wedged(client: FlaskClient, workspaces_root: Path, name: str = "orphaned") -> tuple[str, Path]:
        cid = _create_via_form(client, name)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        return cid, ws

    @staticmethod
    def _wedged_arc(client: FlaskClient, workspaces_root: Path, name: str = "orphaned-arc") -> tuple[str, Path]:
        cid = _create_via_form(client, name)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        return cid, ws

    def test_a_run_nobody_is_supervising_can_be_abandoned(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = self._wedged(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.FAILED

    def test_an_arc_run_nobody_is_supervising_can_be_abandoned(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """The abandon route dispatches on the running state, T3 or ARC."""
        cid, ws = self._wedged_arc(client, workspaces_root)
        assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 302
        state = load_state(ws)
        assert state.state == CampaignStateValue.FAILED
        assert state.failed_from == CampaignStateValue.RUNNING_ARC

    def test_an_arc_run_still_in_progress_is_refused(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = self._wedged_arc(client, workspaces_root, "live-arc")
        with supervise_run(ws, "act-1"):
            assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 409
        assert load_state(ws).state == CampaignStateValue.RUNNING_ARC

    def test_a_run_still_in_progress_is_refused(self, client: FlaskClient, workspaces_root: Path) -> None:
        """Abandoning a live run would race the supervisor's own ending."""
        cid, ws = self._wedged(client, workspaces_root, "live")
        with supervise_run(ws, "act-1"):
            assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 409
        assert load_state(ws).state == CampaignStateValue.RUNNING_T3

    def test_a_campaign_that_is_not_running_is_refused(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "not-running")
        assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 409

    def test_an_unknown_campaign_is_a_404(self, client: FlaskClient) -> None:
        assert _post(client, "/campaigns/nope/abandon", follow_redirects=False).status_code == 404


class TestRecoveryIsVisibleOnTheDashboard:
    """The `failed_from` mechanism had no UI at all: retrying meant hand-editing JSON."""

    def test_a_failed_run_offers_a_retry(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_T3)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert f"/campaigns/{cid}/retry" in html
        assert f"/campaigns/{cid}/replan" in html

    def test_a_run_that_failed_before_launching_offers_a_retry(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """The retry edge the state machine allows must have a button too.

        A campaign failed from ``approved_for_execution`` can retry — the
        route accepts it and the approval is still valid — so the dashboard
        must offer it, or the only exit is a re-plan that throws the
        approval away.
        """
        cid, _ws = _failed_campaign(client, workspaces_root, CampaignStateValue.APPROVED_FOR_EXECUTION)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert f"/campaigns/{cid}/retry" in html
        assert f"/campaigns/{cid}/replan" in html

    def test_a_failure_before_approval_offers_only_re_planning(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """The dashboard must not offer a button the state machine refuses."""
        cid, _ws = _failed_campaign(client, workspaces_root, CampaignStateValue.PLAN_PENDING_APPROVAL)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert f"/campaigns/{cid}/retry" not in html
        assert f"/campaigns/{cid}/replan" in html

    def test_a_rejected_plan_offers_re_planning(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ws = _plan_pending_approval(client, workspaces_root, "blocked-ui")
        _post(client, f"/campaigns/{cid}/reject")
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert f"/campaigns/{cid}/replan" in html
        assert "rejected" in html.lower()

    def test_the_failure_reason_is_shown(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_T3)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert "something went wrong" in html

    def test_a_healthy_campaign_shows_no_recovery_card(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "healthy")
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert "Recovery" not in html


class TestTheDashboardTellsTheTruthAboutRunningCampaigns:
    def test_a_dead_run_stops_promising_progress_and_offers_a_way_out(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """The forever-refreshing page was the whole user-visible symptom.

        Auto-refresh on a run whose supervisor died reloads a stale promise
        every five seconds and never offers anything else.
        """
        cid = _create_via_form(client, "dead-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert 'http-equiv="refresh"' not in html
        assert "not being supervised" in html
        assert f"/campaigns/{cid}/abandon" in html

    def test_a_dead_arc_run_stops_promising_progress_and_offers_a_way_out(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """RUNNING_ARC gets the same liveness probe and abandon affordance."""
        cid = _create_via_form(client, "dead-arc-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        html = client.get(f"/campaigns/{cid}").data.decode()
        assert 'http-equiv="refresh"' not in html
        assert "not being supervised" in html
        assert f"/campaigns/{cid}/abandon" in html

    def test_a_live_arc_run_still_refreshes_and_offers_no_escape_hatch(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid = _create_via_form(client, "live-arc-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        with supervise_run(ws, "act-1"):
            html = client.get(f"/campaigns/{cid}").data.decode()
        assert 'http-equiv="refresh"' in html
        assert "Run in progress" in html
        assert f"/campaigns/{cid}/abandon" not in html

    def test_a_live_run_still_refreshes_and_offers_no_escape_hatch(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid = _create_via_form(client, "live-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        with supervise_run(ws, "act-1"):
            html = client.get(f"/campaigns/{cid}").data.decode()
        assert 'http-equiv="refresh"' in html
        assert "Run in progress" in html
        assert f"/campaigns/{cid}/abandon" not in html

    def test_an_unaccountable_tool_is_reported_without_an_abandon_button(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """Offering an action that can only ever 409 is worse than offering none.

        A tool that outlived its launcher cannot be identified, so
        abandoning is refused by the service. The dashboard has to say so
        and point at the process group, rather than render a button whose
        only possible outcome is an error.
        """
        cid = _create_via_form(client, "unaccountable-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            os.kill(tree.leader_pid, signal.SIGKILL)
            tree.proc.wait(timeout=15)

            html = client.get(f"/campaigns/{cid}").data.decode()
            assert 'http-equiv="refresh"' not in html
            assert "not being supervised" in html
            assert str(tree.pgid) in html
            assert f"/campaigns/{cid}/abandon" not in html, (
                "the dashboard must not offer an abandon the service will refuse"
            )

            os.killpg(tree.pgid, signal.SIGKILL)

    def test_an_orphaned_tool_is_named_on_the_dashboard(self, client: FlaskClient, workspaces_root: Path) -> None:
        """The identifiable case still gets the button, and names the tree."""
        cid = _create_via_form(client, "orphaned-run")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        with _tool_tree() as tree:
            _strand_active_run(ws, tree.pgid, tree.command)
            html = client.get(f"/campaigns/{cid}").data.decode()
            assert "not being supervised" in html
            assert str(tree.pgid) in html
            assert f"/campaigns/{cid}/abandon" in html
            os.killpg(tree.pgid, signal.SIGKILL)


class TestBudgetGateOnRunRoute:
    """The launch-time budget re-check surfaces as a 409, and only on /run."""

    @staticmethod
    def _spent(ws: Path, cpu_hours: float) -> None:
        save_run_record(
            ws,
            RunRecord(
                run_id=f"spent-{cpu_hours}",
                action_id="earlier-action",
                tool_name="t3",
                status=RunStatus.SUCCEEDED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                estimated_cpu_hours=cpu_hours,
                actual_cpu_hours=cpu_hours,
                submission_mode=SubmissionMode.SUBPROCESS,
            ),
        )

    def test_running_an_action_the_budget_no_longer_covers_is_a_409(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """An auto-approved plan gone stale must read as a conflict, not a 500.

        The plan auto-approved when the budget (20) was untouched; by run
        time earlier runs have consumed 19 of it, so the live re-check
        escalates and no human approval stands.
        """
        cid = _create_via_form(client, "budget-gate")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        self._spent(ws, 19.0)

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 409
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_running_a_rejected_action_is_a_409(self, client: FlaskClient, workspaces_root: Path) -> None:
        """A human REJECTED decision must refuse the launch, even well within budget.

        The plan auto-approves (budget is untouched), so the live gate alone
        would launch this — but a human explicitly rejected the action, and
        that must surface as the same 409 a budget conflict would, not a
        silent launch.
        """
        cid = _create_via_form(client, "budget-gate-rejected")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        action = load_plan(ws).actions[0]
        record_decision(ws, action.action_id, ApprovalStatus.REJECTED, decided_by="alon")

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 409
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_finalize_is_never_budget_blocked(self, client: FlaskClient, workspaces_root: Path) -> None:
        """Adopting already-persisted diagnostics spends nothing new."""
        cid = _create_via_form(client, "budget-finalize")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        for step in [CampaignStateValue.RUNNING_T3, CampaignStateValue.DIAGNOSTICS_READY]:
            update_state(ws, step)
        save_diagnostics(
            ws,
            DiagnosticsV1(run_id="r", campaign_id="c", generated_at=datetime.now(UTC)),
        )
        self._spent(ws, 100.0)  # far over budget

        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 302
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1


class TestRecoveryRoutesRequireCsrf:
    @pytest.mark.parametrize("route", ["replan", "retry", "finalize", "abandon"])
    def test_a_post_without_a_token_is_rejected(self, client: FlaskClient, workspaces_root: Path, route: str) -> None:
        cid, _ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RUNNING_T3, name=f"csrf-{route}")
        response = client.post(f"/campaigns/{cid}/{route}", data={})
        assert response.status_code == 400


class TestRecoveryRouteEdgeCases:
    def test_a_campaign_mid_flow_cannot_be_finalized(self, client: FlaskClient) -> None:
        """Neither failed nor holding diagnostics: there is nothing to adopt."""
        cid = _create_via_form(client, "mid-flow")
        assert _post(client, f"/campaigns/{cid}/finalize", follow_redirects=False).status_code == 409

    def test_losing_a_race_while_abandoning_is_a_409_not_a_500(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The state can change between the route's read and the transition."""
        from carmel.services.state_machine import InvalidTransitionError
        from carmel.ui import app as ui_app

        cid = _create_via_form(client, "abandon-race")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)

        def _lost_the_race(_ws: Path, _campaign: Any) -> None:
            raise InvalidTransitionError("running_t3 -> failed is not permitted")

        monkeypatch.setattr(ui_app, "abandon_t3_run", _lost_the_race)
        assert _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False).status_code == 409


# ---------------------------------------------------------------------------
# ARC through the UI (M8 / issue #12)
# ---------------------------------------------------------------------------


def _create_arc_campaign_via_form(client: FlaskClient, name: str = "arc-e2e") -> str:
    """Create a campaign whose mixture matches the golden ARC fixture (OH + CH3)."""
    response = _post(
        client,
        "/campaigns/new",
        data={
            "workspace_name": name,
            "mixture_components": "OH,0.05,[OH]\nCH3,0.20,[CH3]",
            "observables": "ignition_delay",
            "reactors": "jsr,800,1200,1.0,5.0,1.0",
            "cpu_hours": "20",
            "experiment_budget": "0",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    return response.headers["Location"].rsplit("/", 1)[-1]


def _arc_planned(
    client: FlaskClient,
    workspaces_root: Path,
    name: str = "arc-plan",
    level_of_theory: str | None = None,
) -> tuple[str, Path]:
    """Create an ARC campaign and generate its auto-approved run_arc plan."""
    cid = _create_arc_campaign_via_form(client, name)
    data = {"tool": "arc"}
    if level_of_theory:
        data["level_of_theory"] = level_of_theory
    assert _post(client, f"/campaigns/{cid}/plan", data=data).status_code == 302
    ws = find_campaign_workspace(workspaces_root, cid)
    assert ws is not None
    return cid, ws


class TestArcPlanRoute:
    """POST /plan with tool=arc is the production caller of generate_arc_plan."""

    def test_tool_arc_generates_a_run_arc_plan(self, client: FlaskClient, workspaces_root: Path) -> None:
        _cid, ws = _arc_planned(client, workspaces_root)
        plan = load_plan(ws)
        assert len(plan.actions) == 1
        assert plan.actions[0].kind == ActionKind.ARC_RUN
        # 2 mixture species -> 2 cpu-h: inside the ARC envelope and budget,
        # so the small action stays auto-approved and immediately runnable.
        assert not plan.requires_approval
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

    def test_level_of_theory_reaches_the_plan_parameters(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.adapters.arc import MOCK_LEVEL_OF_THEORY

        _cid, ws = _arc_planned(client, workspaces_root, level_of_theory=MOCK_LEVEL_OF_THEORY)
        assert load_plan(ws).actions[0].parameters["level_of_theory"] == MOCK_LEVEL_OF_THEORY

    def test_the_default_tool_is_still_t3(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "default-t3")
        assert _post(client, f"/campaigns/{cid}/plan").status_code == 302
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert load_plan(ws).actions[0].kind == ActionKind.T3_RUN

    def test_an_unknown_tool_is_a_400_and_writes_no_plan(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "bad-tool")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert _post(client, f"/campaigns/{cid}/plan", data={"tool": "quantum"}).status_code == 400
        assert not (ws / "plan.json").exists()
        assert load_state(ws).state == CampaignStateValue.READY_FOR_PLANNING


class TestRunDispatchesOnActionKind:
    """/run must start the tool the approved plan names — never the other one."""

    @staticmethod
    def _recorders(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
        """Record which execution path the /run route took.

        Records ``arc`` (the single-action path) and ``dispatch`` (the multi-action
        dispatcher) rather than the old ``t3``/``arc`` pair: the route no longer calls
        ``start_t3_action`` at all, so patching that symbol would only ever record an
        empty list and every "did not start T3" assertion would pass vacuously.
        """
        from carmel.ui import app as ui_app

        started: dict[str, list[str]] = {"dispatch": [], "arc": []}

        def _fake_arc(ws: Path, campaign: Any, action: Any) -> None:
            started["arc"].append(action.action_id)

        def _fake_dispatch(ws: Path, campaign: Any, **kwargs: Any) -> None:
            started["dispatch"].append(str(ws))
            return None

        monkeypatch.setattr(ui_app, "start_arc_action", _fake_arc)
        monkeypatch.setattr(ui_app, "execute_next_action", _fake_dispatch)
        return started

    def test_an_arc_plan_starts_arc_not_t3(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = self._recorders(monkeypatch)
        cid, ws = _arc_planned(client, workspaces_root, name="dispatch-arc")
        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 302
        assert started["arc"] == [load_plan(ws).actions[0].action_id]
        # An ARC plan must never be handed to the multi-action dispatcher, which has no
        # handler for it.
        assert started["dispatch"] == []

    def test_a_t3_plan_runs_through_the_dispatcher_and_never_starts_arc(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # T3 no longer reaches `start_t3_action` from this route: the agentic layer
        # moved it onto the multi-action dispatcher, which drives the SAME underlying
        # finish path (`execution._finish_t3_run`). What must still hold -- and is the
        # actual point of this test -- is that a T3 plan never starts ARC.
        started = self._recorders(monkeypatch)

        cid = _create_via_form(client, "dispatch-t3")
        _post(client, f"/campaigns/{cid}/plan")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 302
        assert started["dispatch"], "the T3 plan did not reach the dispatcher"
        assert started["arc"] == []

    def test_rerunning_a_running_arc_campaign_is_a_conflict(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The preflight must check RUNNING_ARC for an ARC plan, not RUNNING_T3."""
        started = self._recorders(monkeypatch)
        cid, ws = _arc_planned(client, workspaces_root, name="arc-conflict")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 409
        assert started == {"dispatch": [], "arc": []}

    def test_a_kind_no_tool_runs_can_never_even_be_persisted(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plan holding a non-runnable kind must start nothing.

        The defence moved EARLIER than this test originally assumed: `save_plan` now
        validates plan shape, so an EXPERIMENT action cannot reach disk at all and the
        run route never gets the chance to mis-dispatch it. Asserting the refusal at the
        write is strictly stronger than asserting a 409 at the run.
        """
        from carmel.services.planner import save_plan

        started = self._recorders(monkeypatch)
        _cid, ws = _arc_planned(client, workspaces_root, name="odd-kind")
        plan = load_plan(ws)
        action = plan.actions[0].model_copy(update={"kind": ActionKind.EXPERIMENT})

        with pytest.raises(ValueError, match="nothing can execute"):
            save_plan(ws, plan.model_copy(update={"actions": [action]}))

        assert started == {"dispatch": [], "arc": []}


class TestArcSvgRoute:
    """?tool=arc serves the ARC run's SVGs from models/arc/, never T3's."""

    def test_each_tool_gets_its_own_svg_tree(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "svg-trees")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        (ws / "models").mkdir(parents=True, exist_ok=True)
        (ws / "models" / "species_selection.svg").write_text("<svg>from-t3</svg>", encoding="utf-8")
        (ws / "models" / "arc").mkdir(parents=True, exist_ok=True)
        (ws / "models" / "arc" / "species_selection.svg").write_text("<svg>from-arc</svg>", encoding="utf-8")

        arc = client.get(f"/campaigns/{cid}/svg/species_selection.svg?tool=arc")
        assert arc.status_code == 200
        assert b"from-arc" in arc.data
        assert b"from-t3" not in arc.data

        t3 = client.get(f"/campaigns/{cid}/svg/species_selection.svg")
        assert b"from-t3" in t3.data

    def test_a_missing_arc_svg_returns_the_placeholder(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "svg-missing")
        response = client.get(f"/campaigns/{cid}/svg/species_selection.svg?tool=arc")
        assert response.status_code == 200
        assert b"no diagnostics yet" in response.data

    def test_an_unknown_tool_is_a_404(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "svg-bad-tool")
        assert client.get(f"/campaigns/{cid}/svg/species_selection.svg?tool=rmg").status_code == 404


def _diagnostics(run_id: str) -> DiagnosticsV1:
    return DiagnosticsV1(
        run_id=run_id,
        campaign_id="c",
        generated_at=datetime.now(UTC),
        species_to_compute=[],
        reactions_to_compute=[],
        pdep_networks_to_compute=[],
    )


class TestDashboardArcResults:
    """The dashboard attributes results to the tool that produced them."""

    def test_arc_results_show_arc_diagnostics_and_hide_stale_t3_ones(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """A workspace can hold both files; only the state-owning tool's shows.

        This is the stale-diagnostics cross-contamination case: a campaign
        whose earlier life left a T3 diagnostics.json behind and whose
        current state came from ARC must not present the T3 file.
        """
        cid, ws = _arc_planned(client, workspaces_root, name="arc-results")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        update_state(ws, CampaignStateValue.RESULTS_READY)
        save_arc_diagnostics(ws, _diagnostics("arc-run-live"))
        save_diagnostics(ws, _diagnostics("t3-run-stale"))

        page = client.get(f"/campaigns/{cid}").data
        assert b"arc-run-live" in page
        assert b"t3-run-stale" not in page
        assert "Diagnostics · ARC".encode() in page

    def test_the_finalize_button_is_offered_at_results_ready(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _arc_planned(client, workspaces_root, name="arc-finalize")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        update_state(ws, CampaignStateValue.RESULTS_READY)
        save_arc_diagnostics(ws, _diagnostics("arc-run-ready"))
        page = client.get(f"/campaigns/{cid}").data
        assert b"Complete phase 1" in page

    def test_t3_results_hide_stale_arc_diagnostics(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "t3-results")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/plan")
        update_state(ws, CampaignStateValue.RUNNING_T3)
        update_state(ws, CampaignStateValue.DIAGNOSTICS_READY)
        save_diagnostics(ws, _diagnostics("t3-run-live"))
        save_arc_diagnostics(ws, _diagnostics("arc-run-stale"))

        page = client.get(f"/campaigns/{cid}").data
        assert b"t3-run-live" in page
        assert b"arc-run-stale" not in page
        assert "Diagnostics · T3".encode() in page

    def test_a_campaign_with_no_results_shows_neither_tools_leftovers(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """Before any run, files on disk (however they got there) stay unattributed."""
        cid = _create_via_form(client, "no-results")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        save_diagnostics(ws, _diagnostics("t3-run-old"))
        save_arc_diagnostics(ws, _diagnostics("arc-run-old"))
        page = client.get(f"/campaigns/{cid}").data
        assert b"t3-run-old" not in page
        assert b"arc-run-old" not in page
        assert b"No diagnostics yet" in page

    def test_a_campaign_that_failed_from_arc_results_still_shows_its_arc_diagnostics(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """FAILED-from-RESULTS_READY is an ARC outcome: the dashboard shows the
        persisted ARC diagnostics (and the recovery card offers to complete
        from them), not a stale T3 file."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.RESULTS_READY, name="failed-arc")
        save_arc_diagnostics(ws, _diagnostics("arc-run-failed-late"))
        save_diagnostics(ws, _diagnostics("t3-run-stale"))
        page = client.get(f"/campaigns/{cid}").data
        assert b"arc-run-failed-late" in page
        assert b"t3-run-stale" not in page
        assert b"Complete from persisted diagnostics" in page

    def test_a_campaign_that_failed_before_running_shows_neither_tools_leftovers(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """FAILED from a pre-run origin has no results tool to attribute to."""
        cid, ws = _failed_campaign(client, workspaces_root, CampaignStateValue.READY_FOR_PLANNING, name="failed-early")
        save_diagnostics(ws, _diagnostics("t3-run-old"))
        save_arc_diagnostics(ws, _diagnostics("arc-run-old"))
        page = client.get(f"/campaigns/{cid}").data
        assert b"t3-run-old" not in page
        assert b"arc-run-old" not in page

    def test_the_run_button_names_the_planned_tool(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ws = _arc_planned(client, workspaces_root, name="arc-button")
        assert b"Run ARC" in client.get(f"/campaigns/{cid}").data

        t3_cid = _create_via_form(client, "t3-button")
        _post(client, f"/campaigns/{t3_cid}/plan")
        assert b"Run T3" in client.get(f"/campaigns/{t3_cid}").data

    def test_the_retry_flash_names_the_planned_tool(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws = _arc_planned(client, workspaces_root, name="arc-retry")
        update_state(ws, CampaignStateValue.RUNNING_ARC)
        update_state(ws, CampaignStateValue.FAILED, notes="arc run failed")
        response = _post(client, f"/campaigns/{cid}/retry", follow_redirects=True)
        assert b"Use Run ARC" in response.data


class TestBudgetGateOnArcRunRoute:
    """The M9 launch-time re-check guards the ARC path exactly like T3's."""

    def test_an_over_budget_arc_launch_is_a_409_and_wedges_nothing(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws = _arc_planned(client, workspaces_root, name="arc-budget")
        # The plan auto-approved against an untouched budget of 20; by run
        # time 19 are consumed, so remaining (1) < the action's estimate (2).
        save_run_record(
            ws,
            RunRecord(
                run_id="spent-earlier",
                action_id="earlier-action",
                tool_name="t3",
                status=RunStatus.SUCCEEDED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                estimated_cpu_hours=19.0,
                actual_cpu_hours=19.0,
                submission_mode=SubmissionMode.SUBPROCESS,
            ),
        )
        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 409
        # Nothing was taken: state untouched, no in-flight record, lock free.
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION
        assert load_active_run(ws) is None
        with supervise_run(ws, "prove-the-lock-is-free"):
            pass


class TestMockterThroughCarmelEndToEnd:
    """ARC driven through the full Carmel stack via the Mockter level of theory.

    The user-visible arc, exercised through the same Flask routes a browser
    hits: create → plan (run_arc, Mockter) → auto-approve → /run →
    RESULTS_READY → COMPLETED_PHASE1 → dashboard shows ARC diagnostics.

    This fast variant replays the golden Mockter fixture
    (tests/fixtures/arc/sample_project/, captured from a real Mockter run)
    through the REAL ARCAdapter — only the subprocess boundary is stubbed —
    so input building, output normalization, run-record persistence, the
    background execution service, and the UI wiring are all real. The
    @requires_arc peer that launches the real Mockter subprocess lives in
    tests/test_arc_adapter.py::TestARCAdapterRealSubprocess, where the CI
    tools lane refuses to let it skip.
    """

    @staticmethod
    def _replay_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil

        import yaml

        from carmel.adapters import arc as arc_module
        from carmel.adapters.arc import arc_info_filename

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        class _Completed:
            returncode = 0

        class _ProbeCompleted:
            returncode = 1  # the version probe (cwd=None) returns None harmlessly

        def _fake_run(command: list[str], cwd: Path | None = None, **kwargs: object) -> object:
            if cwd is None:
                return _ProbeCompleted()
            run_dir = Path(cwd)
            payload = yaml.safe_load((run_dir / "input.yml").read_text())
            shutil.copy(ARC_FIXTURE_ROOT / "carmel_mock_opt_info.yml", run_dir / arc_info_filename(payload))
            (run_dir / "output").mkdir(exist_ok=True)
            shutil.copy(ARC_FIXTURE_ROOT / "output" / "output.yml", run_dir / "output" / "output.yml")
            return _Completed()

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _fake_run)

    @staticmethod
    def _capture_arc_threads(monkeypatch: pytest.MonkeyPatch) -> list[threading.Thread]:
        from carmel.ui import app as ui_app

        threads: list[threading.Thread] = []
        real_start = ui_app.start_arc_action

        def _capture(ws: Path, campaign: Any, action: Any) -> threading.Thread:
            thread = real_start(ws, campaign, action)
            threads.append(thread)
            return thread

        monkeypatch.setattr(ui_app, "start_arc_action", _capture)
        return threads

    def test_mockter_plan_approve_run_results_dashboard(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters.arc import MOCK_LEVEL_OF_THEORY

        self._replay_fixture(monkeypatch)
        threads = self._capture_arc_threads(monkeypatch)

        cid, ws = _arc_planned(client, workspaces_root, name="mockter-e2e", level_of_theory=MOCK_LEVEL_OF_THEORY)
        plan = load_plan(ws)
        action = plan.actions[0]
        assert action.kind == ActionKind.ARC_RUN
        assert action.parameters["level_of_theory"] == MOCK_LEVEL_OF_THEORY
        # The small Mockter action clears the combined gate: auto-approved,
        # so /run needs no human decision and M9 enforcement does not block.
        assert not plan.requires_approval
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

        assert _post(client, f"/campaigns/{cid}/run", follow_redirects=False).status_code == 302
        assert threads, "the run route never started an ARC action"
        for thread in threads:
            thread.join(timeout=120)
            assert not thread.is_alive()

        # COMPLETED_PHASE1 is only reachable through RESULTS_READY, so the
        # terminal state proves the whole ARC state arc was walked.
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        events = read_events(ws / "decision_log.jsonl")
        finished = [e for e in events if e.get("event") == "arc_run_finished"]
        assert finished and finished[-1].get("status") == "succeeded"

        diagnostics = load_arc_diagnostics(ws)
        assert diagnostics is not None
        assert sorted(s.label for s in diagnostics.species_to_compute) == ["CH3", "OH"]
        assert diagnostics.level_of_theory == MOCK_LEVEL_OF_THEORY

        page = client.get(f"/campaigns/{cid}").data
        assert b"completed_phase1" in page
        assert "Diagnostics · ARC".encode() in page
        assert diagnostics.run_id.encode() in page

        svg = client.get(f"/campaigns/{cid}/svg/species_selection.svg?tool=arc")
        assert svg.status_code == 200
        assert b"<svg" in svg.data
        assert b"no diagnostics yet" not in svg.data


# --------------------- multi-action dispatch via the UI ---------------------


def _fake_handlers_factory(outcome_by_kind=None, calls=None):
    """Build a stand-in for carmel.ui.app.default_handlers using fake handlers."""
    from datetime import datetime
    from uuid import uuid4

    from carmel.schemas import (
        ActionKind,
        ActionOutcome,
        FailureCode,
        RunRecord,
        RunStatus,
        SubmissionMode,
    )
    from carmel.services.dispatcher import ActionResult

    outcome_by_kind = outcome_by_kind or {}
    calls = calls if calls is not None else []

    def handler(workspace_root, campaign, action):
        calls.append(action.action_id)
        outcome = outcome_by_kind.get(action.kind, ActionOutcome.SUCCEEDED)
        status = (
            RunStatus.SUCCEEDED
            if outcome in (ActionOutcome.SUCCEEDED, ActionOutcome.NO_GROUNDED_FINDINGS)
            else RunStatus.FAILED
        )
        now = datetime.now(UTC)
        record = RunRecord(
            run_id=uuid4().hex,
            action_id=action.action_id,
            tool_name="fake",
            status=status,
            failure_code=FailureCode.NONE if status == RunStatus.SUCCEEDED else FailureCode.UNKNOWN,
            started_at=now,
            ended_at=now,
            submission_mode=SubmissionMode.LOCAL,
        )
        return ActionResult(action_id=action.action_id, kind=action.kind, run_record=record, outcome=outcome)

    def factory(**kwargs):
        return {ActionKind.T3_RUN: handler, ActionKind.LITERATURE_SEARCH: handler}

    return factory, calls


def _join_background(ws: Path, timeout: float = 60.0) -> None:
    """Wait for the background half of a dispatch to finish.

    The dispatcher holds the workspace dispatch lease until its background
    thread completes the bookkeeping, so the lease vanishing is the
    "run finished" signal a UI test can await deterministically.
    """
    import time

    from carmel.services.plan_progress import DISPATCH_LOCK_DIR_NAME

    lease = ws / DISPATCH_LOCK_DIR_NAME
    deadline = time.monotonic() + timeout
    while lease.exists():
        assert time.monotonic() < deadline, "background dispatch did not finish in time"
        time.sleep(0.01)


class TestRunRoute:
    def test_run_executes_next_action_to_completion(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import carmel.ui.app as app_module
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.state_machine import load_state

        factory, calls = _fake_handlers_factory()
        monkeypatch.setattr(app_module, "default_handlers", factory)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")  # auto-approved under default policy
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert load_state(ws).state == CampaignStateValue.APPROVED_FOR_EXECUTION

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        assert response.status_code == 302
        _join_background(ws)
        assert calls  # the dispatcher ran the T3 action through the fake handler
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1

    def test_run_with_literature_plan_runs_both_steps(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import carmel.ui.app as app_module
        from carmel.services.campaigns import find_campaign_workspace, load_campaign
        from carmel.services.planner import plan_and_save
        from carmel.services.state_machine import load_state, update_state

        factory, calls = _fake_handlers_factory()
        monkeypatch.setattr(app_module, "default_handlers", factory)
        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = plan_and_save(ws, load_campaign(ws), include_literature=True)
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION, notes="auto-approved")
        # /run refuses an action with no recorded approval decision (main's
        # HITL guard), so record the auto-approvals the plan route would have.
        from carmel.schemas.approval import ApprovalStatus
        from carmel.services.approvals import record_decision

        for action in plan.actions:
            record_decision(ws, action.action_id, ApprovalStatus.AUTO_APPROVED, decided_by="auto")

        _post(client, f"/campaigns/{cid}/run")
        _join_background(ws)
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY
        _post(client, f"/campaigns/{cid}/run")
        _join_background(ws)
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        assert len(calls) == 2

    def test_run_without_approved_action_is_a_conflict(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unapproved next action must 409, per main's /run approval guard.

        (Previously this flashed an info message; main's stricter recorded-
        decision guard supersedes that and refuses outright.)
        """
        import carmel.ui.app as app_module
        from carmel.schemas.approval import ApprovalPolicy
        from carmel.services.approvals import save_policy
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.state_machine import load_state

        factory, calls = _fake_handlers_factory()
        monkeypatch.setattr(app_module, "default_handlers", factory)
        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        # Plan requires approval, so /run has nothing approved to execute.
        save_policy(ws, ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.1))
        _post(client, f"/campaigns/{cid}/plan")

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        assert response.status_code == 409
        assert calls == []
        assert load_state(ws).state == CampaignStateValue.PLAN_PENDING_APPROVAL

    def test_run_when_plan_complete_flashes_info(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import carmel.ui.app as app_module
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.state_machine import load_state

        factory, calls = _fake_handlers_factory()
        monkeypatch.setattr(app_module, "default_handlers", factory)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")  # auto-approved under default policy
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        _post(client, f"/campaigns/{cid}/run")
        _join_background(ws)
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1
        runs_so_far = len(calls)

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=True)

        assert response.status_code == 200
        assert b"Nothing to run" in response.data
        assert len(calls) == runs_so_far

    def test_run_unknown_campaign_404(self, client: FlaskClient) -> None:
        response = _post(client, "/campaigns/unknown/run")
        assert response.status_code == 404


class TestPerActionApproval:
    def _pending_two_action_plan(self, client: FlaskClient, workspaces_root: Path):
        from carmel.schemas.approval import ApprovalPolicy
        from carmel.services.approvals import save_policy
        from carmel.services.campaigns import find_campaign_workspace, load_campaign
        from carmel.services.planner import plan_and_save
        from carmel.services.state_machine import update_state

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        save_policy(
            ws,
            ApprovalPolicy(auto_approve_t3_under_cpu_hours=0.1, require_approval_for_literature=True),
        )
        plan = plan_and_save(ws, load_campaign(ws), include_literature=True)
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        return cid, ws, plan

    def test_approve_single_action(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.schemas import ApprovalStatus
        from carmel.services.plan_progress import load_progress

        cid, ws, plan = self._pending_two_action_plan(client, workspaces_root)
        t3_id = plan.actions[1].action_id

        response = _post(client, f"/campaigns/{cid}/approve", data={"action_id": t3_id}, follow_redirects=False)

        assert response.status_code == 302
        progress = load_progress(ws)
        assert progress.actions[1].approval_status == ApprovalStatus.APPROVED
        assert progress.actions[0].approval_status == ApprovalStatus.PENDING  # untouched

    def test_reject_one_action_does_not_blanket_block(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.schemas import ApprovalStatus
        from carmel.services.plan_progress import load_progress
        from carmel.services.state_machine import load_state

        cid, ws, plan = self._pending_two_action_plan(client, workspaces_root)
        lit_id = plan.actions[0].action_id

        _post(client, f"/campaigns/{cid}/reject", data={"action_id": lit_id})

        progress = load_progress(ws)
        assert progress.actions[0].approval_status == ApprovalStatus.REJECTED
        assert progress.has_executable_remaining()
        assert load_state(ws).state != CampaignStateValue.BLOCKED

    def test_reject_all_blocks(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.services.state_machine import load_state

        cid, ws, _plan = self._pending_two_action_plan(client, workspaces_root)
        _post(client, f"/campaigns/{cid}/reject")
        assert load_state(ws).state == CampaignStateValue.BLOCKED

    def test_approve_unknown_action_404(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ws, _plan = self._pending_two_action_plan(client, workspaces_root)
        response = _post(client, f"/campaigns/{cid}/approve", data={"action_id": "nope"})
        assert response.status_code == 404


class TestDashboardAgenticCards:
    def test_dashboard_shows_literature_report(self, client: FlaskClient, workspaces_root: Path) -> None:
        from datetime import datetime

        from carmel.agents.budget import BudgetUsage
        from carmel.schemas.literature import LiteratureReport, StopReason
        from carmel.services.campaigns import find_campaign_workspace

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        report = LiteratureReport(
            report_id="rep1",
            campaign_id=cid,
            action_id="lit1",
            run_id="run1",
            created_at=datetime.now(UTC),
            stop_reason=StopReason.SELF_TERMINATED,
            usage=BudgetUsage(model_calls=0, tokens=0, cost_usd=0.0, fetches=0, fetch_bytes=0, elapsed_s=0.0),
        )
        (ws / "literature_report.json").write_text(report.model_dump_json())

        response = client.get(f"/campaigns/{cid}")

        assert response.status_code == 200
        assert b"Literature report" in response.data
        assert b"self_terminated" in response.data

    def test_dashboard_shows_progress_state_mismatch_warning(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.schemas import ActionExecutionStatus, ActionOutcome
        from carmel.services.campaigns import find_campaign_workspace, load_campaign
        from carmel.services.plan_progress import advance_cursor, mark_finished
        from carmel.services.planner import plan_and_save

        cid = _create_via_form(client)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = plan_and_save(ws, load_campaign(ws))
        # Progress says the plan completed, but the campaign state was never
        # finalised (still ready_for_planning): the dashboard must surface it.
        mark_finished(
            ws,
            plan.actions[0].action_id,
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )
        advance_cursor(ws)

        response = client.get(f"/campaigns/{cid}")

        assert response.status_code == 200
        assert b"mismatch" in response.data
        assert b"completed_phase1" in response.data

    def test_dashboard_shows_per_action_progress(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        response = client.get(f"/campaigns/{cid}")
        assert response.status_code == 200
        assert b"auto_approved" in response.data


class TestCreateAppWithAgentConfig:
    def test_campaign_creation_auto_runs_literature(self, tmp_path: Path) -> None:
        """create_app(agent_config=...) wires the literature auto-run; the
        TEST-tier MockModel path never touches the network."""
        from carmel.config import AgentConfig

        app = create_app(workspaces_root=tmp_path, agent_config=AgentConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        cid = _create_via_form(client, name="auto-lit")

        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.state_machine import load_state

        ws = find_campaign_workspace(tmp_path, cid)
        assert ws is not None
        assert load_state(ws).state == CampaignStateValue.LITERATURE_READY
        assert (ws / "literature_report.json").exists()
        dashboard = client.get(f"/campaigns/{cid}").data
        assert b"Literature report" in dashboard

    def test_no_agent_config_means_no_auto_run(self, client: FlaskClient, workspaces_root: Path) -> None:
        from carmel.services.campaigns import find_campaign_workspace

        cid = _create_via_form(client, name="no-auto")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        assert not (ws / "literature_report.json").exists()
        from carmel.services.state_machine import load_state

        assert load_state(ws).state == CampaignStateValue.READY_FOR_PLANNING


class TestLiteratureCrashRecovery:
    """Finding 1: a crash mid literature-run must not wedge the campaign.

    Every UI route refuses while the campaign reads ``RUNNING_*`` (there is
    no ``RUNNING_LITERATURE -> RUNNING_LITERATURE`` self-edge), so without a
    recovery path a crashed literature action could only be fixed by
    hand-editing ``plan_progress.json``. ``/run`` now calls ``reconcile()``
    ahead of its preflight (self-heal) and ``/abandon`` now handles
    ``RUNNING_LITERATURE`` explicitly (manual escape hatch) instead of only
    ``RUNNING_T3``.
    """

    @staticmethod
    def _wedged_literature(
        client: FlaskClient, workspaces_root: Path, name: str = "wedged-lit", *, stale: bool
    ) -> tuple[str, Path, str]:
        """A campaign at RUNNING_LITERATURE with a RUNNING literature action.

        ``stale=True`` backdates the action's lease past
        ``DEFAULT_STALE_AFTER_S`` so ``reconcile`` treats it as no longer
        live (no lock, and the lease-age fallback used when
        ``in_dispatch_lock=False`` no longer vouches for it) — modelling a
        crash. ``stale=False`` leaves the lease fresh, modelling a run that
        is still genuinely in progress.
        """
        from carmel.services.campaigns import load_campaign
        from carmel.services.plan_progress import DEFAULT_STALE_AFTER_S, load_progress, mark_running, save_progress
        from carmel.services.planner import plan_and_save

        cid = _create_via_form(client, name)
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = plan_and_save(ws, load_campaign(ws), include_literature=True)
        lit_id = plan.actions[0].action_id
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)
        update_state(ws, CampaignStateValue.RUNNING_LITERATURE)
        mark_running(ws, lit_id, "attempt-1")
        if stale:
            progress = load_progress(ws)
            idx = next(i for i, a in enumerate(progress.actions) if a.action_id == lit_id)
            stale_time = datetime.now(UTC) - timedelta(seconds=DEFAULT_STALE_AFTER_S + 1)
            progress.actions[idx] = progress.actions[idx].model_copy(update={"updated_at": stale_time})
            save_progress(ws, progress)
        return cid, ws, lit_id

    def test_a_stale_running_literature_action_self_heals_on_run(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        from carmel.schemas import ActionExecutionStatus
        from carmel.services.plan_progress import load_progress

        cid, ws, lit_id = self._wedged_literature(client, workspaces_root, stale=True)

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        # Not a 409: the stale RUNNING lease was reconciled away before the
        # preflight ever looked at campaign state, so the campaign is no
        # longer wedged.
        assert response.status_code == 302
        progress = load_progress(ws)
        lit_state = next(a for a in progress.actions if a.action_id == lit_id)
        assert lit_state.execution_status != ActionExecutionStatus.RUNNING
        assert load_state(ws).state != CampaignStateValue.RUNNING_LITERATURE

    def test_a_live_running_literature_action_is_left_alone_by_run(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """A genuinely in-progress attempt must not be reconciled away."""
        from carmel.schemas import ActionExecutionStatus

        cid, ws, _lit_id = self._wedged_literature(client, workspaces_root, stale=False)

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        assert response.status_code == 409
        assert load_state(ws).state == CampaignStateValue.RUNNING_LITERATURE
        from carmel.services.plan_progress import load_progress

        progress = load_progress(ws)
        assert progress.actions[0].execution_status == ActionExecutionStatus.RUNNING

    def test_a_stale_running_literature_action_can_be_abandoned(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        cid, ws, _lit_id = self._wedged_literature(client, workspaces_root, stale=True)

        response = _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False)

        assert response.status_code == 302
        assert load_state(ws).state != CampaignStateValue.RUNNING_LITERATURE

    def test_a_live_running_literature_action_refuses_abandon(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, ws, _lit_id = self._wedged_literature(client, workspaces_root, stale=False)

        response = _post(client, f"/campaigns/{cid}/abandon", follow_redirects=False)

        assert response.status_code == 409
        assert load_state(ws).state == CampaignStateValue.RUNNING_LITERATURE


class TestLiteratureReportLoading:
    """Finding 24: the dashboard uses ``load_literature_report``, not a
    mirrored literal + hand-rolled read, and logs a distinct signal for a
    corrupt report versus an absent one.
    """

    def test_dashboard_degrades_gracefully_with_no_report(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client, "no-lit-report")
        response = client.get(f"/campaigns/{cid}")
        assert response.status_code == 200
        assert b"No literature report yet." in response.data

    def test_a_corrupt_report_logs_a_distinct_warning_and_still_degrades(
        self, client: FlaskClient, workspaces_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cid = _create_via_form(client, "corrupt-lit-report")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        (ws / "literature_report.json").write_text("{not valid json")

        # carmel.logger.configure_logging() sets propagate=False on the "carmel"
        # logger, so once any earlier test has configured logging, records never
        # reach caplog's root handler. Restore propagation for this assertion only,
        # otherwise the test passes alone and fails in a full-suite run.
        carmel_logger = logging.getLogger("carmel")
        previous_propagate = carmel_logger.propagate
        carmel_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="carmel.ui"):
                response = client.get(f"/campaigns/{cid}")
        finally:
            carmel_logger.propagate = previous_propagate

        assert response.status_code == 200
        assert b"No literature report yet." in response.data
        # getMessage(), not .message: the latter is only populated once a Formatter
        # has run, and these records are logged with lazy %-style interpolation.
        assert any("corrupt" in record.getMessage() for record in caplog.records)


class TestRunApprovalIsProgressBased:
    """Finding 18: ``/run``'s preflight authorizes off ``plan_progress.json``,
    not the append-only decision log — the log stays an audit trail only.
    """

    def test_run_succeeds_purely_from_progress_approval_with_an_empty_decision_log(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        from carmel.services.campaigns import find_campaign_workspace, load_campaign
        from carmel.services.planner import plan_and_save

        cid = _create_via_form(client, "progress-authorized")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = plan_and_save(ws, load_campaign(ws))  # single auto-approved T3 action
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        update_state(ws, CampaignStateValue.APPROVED_FOR_EXECUTION)
        # No APPROVAL decision was ever logged for the action -- authorization comes
        # purely from plan_progress.json's AUTO_APPROVED seeding.
        #
        # Checks `approval_decision` specifically rather than "any event mentioning this
        # action": planning now also records an `execution_envelope_authorization` audit
        # entry for the action, which is exactly the sort of log record this test exists
        # to prove is NOT what authorizes the run.
        events = read_events(ws / "decision_log.jsonl")
        assert not any(
            e.get("event") == "approval_decision" and e.get("action_id") == plan.actions[0].action_id for e in events
        )

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        assert response.status_code == 302
        assert plan.actions  # sanity: there was something to authorize

    def test_run_refuses_when_progress_disagrees_with_a_stale_decision_log_entry(
        self, client: FlaskClient, workspaces_root: Path
    ) -> None:
        """A decision-log entry claiming approval must not override progress.

        Simulates the two sources of truth disagreeing (the scenario Finding
        18 flags as reachable under a concurrent approve/reject): the log
        says APPROVED, but ``plan_progress.json`` — the single source of
        truth for authorization — still says PENDING.
        """
        from carmel.schemas import ApprovalStatus
        from carmel.services.campaigns import find_campaign_workspace, load_campaign
        from carmel.services.decision_log import append_event
        from carmel.services.planner import plan_and_save

        cid = _create_via_form(client, "log-disagrees")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = plan_and_save(ws, load_campaign(ws), include_literature=True)
        update_state(ws, CampaignStateValue.PLAN_PENDING_APPROVAL)
        lit_id = plan.actions[0].action_id
        append_event(
            ws / "decision_log.jsonl",
            {"event": "decision", "action_id": lit_id, "status": ApprovalStatus.APPROVED.value},
        )
        # plan_progress.json itself was never updated to APPROVED for lit_id
        # (its requirement seeded it PENDING/AUTO_APPROVED via the policy;
        # force it to PENDING to make the disagreement explicit).
        from carmel.services.plan_progress import load_progress, save_progress

        progress = load_progress(ws)
        idx = next(i for i, a in enumerate(progress.actions) if a.action_id == lit_id)
        progress.actions[idx] = progress.actions[idx].model_copy(update={"approval_status": ApprovalStatus.PENDING})
        save_progress(ws, progress)

        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)

        assert response.status_code == 409


class TestAcquisitionQueuePanel:
    """The acquisition queue is the only part of a literature run that asks the USER to
    act, so it has to be visible on the dashboard. Most combustion papers cannot be
    fetched, so a run that queues several and grounds none is the normal outcome -- and
    without this panel it would read on screen as "the agent found nothing"."""

    def test_pending_papers_are_shown_with_the_exact_filename_to_use(self, client: FlaskClient, tmp_path: Path) -> None:
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import record_request

        cid = _create_via_form(client)
        ws = find_campaign_workspace(tmp_path, cid)
        record_request(
            ws,
            title="High pressure shock tube ignition delay measurements",
            doi="10.1115/1.4036254",
            landing_url="https://doi.org/10.1115/1.4036254",
            reason=AcquisitionReason.PAYWALLED,
            detail="HTTP 403",
        )

        body = client.get(f"/campaigns/{cid}").data.decode()

        assert "Papers Carmel needs you to obtain" in body
        assert "10.1115-1.4036254.pdf" in body
        assert "High pressure shock tube ignition delay measurements" in body

    def test_no_panel_is_rendered_when_nothing_is_queued(self, client: FlaskClient) -> None:
        cid = _create_via_form(client)
        body = client.get(f"/campaigns/{cid}").data.decode()
        assert "Papers Carmel needs you to obtain" not in body
