"""Tests for the Carmel Flask UI."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from carmel.schemas import CampaignStateValue
from carmel.schemas.approval import ApprovalPolicy
from carmel.services.approvals import save_policy
from carmel.services.campaigns import find_campaign_workspace
from carmel.services.decision_log import read_events
from carmel.services.execution import save_diagnostics
from carmel.services.planner import load_plan
from carmel.services.state_machine import load_state
from carmel.ui import create_app
from carmel.ui.app import _resolve_workspaces_root


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

        def _fake_execute(ws: Path, campaign: object, action: object) -> tuple[None, None]:
            called["action_id"] = getattr(action, "action_id", None)
            return None, None

        monkeypatch.setattr(ui_app, "execute_t3_action", _fake_execute)
        cid = _create_via_form(client)
        _post(client, f"/campaigns/{cid}/plan")
        response = _post(client, f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["Location"].endswith(cid)
        assert called["action_id"] is not None

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

        def _fake_execute(ws: Path, campaign: object, action: object) -> tuple[None, None]:
            executed["action_id"] = getattr(action, "action_id", None)
            return None, None

        monkeypatch.setattr(ui_app, "execute_t3_action", _fake_execute)
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

        def _fake_execute(ws: Path, campaign: object, action: object) -> tuple[None, None]:
            executed["action_id"] = getattr(action, "action_id", None)
            return None, None

        monkeypatch.setattr(ui_app, "execute_t3_action", _fake_execute)
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
        self._assert_every_post_form_has_token(html, expected_forms=2)  # plan + free-text

    def test_dashboard_pending_approval(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid, _ = _plan_pending_approval(client, workspaces_root, "render-pending")
        html = client.get(f"/campaigns/{cid}").data.decode()
        self._assert_every_post_form_has_token(html, expected_forms=3)  # approve + reject + free-text

    def test_dashboard_approved(self, client: FlaskClient) -> None:
        cid = _create_via_form(client, "render-approved")
        assert _post(client, f"/campaigns/{cid}/plan").status_code == 302
        html = client.get(f"/campaigns/{cid}").data.decode()
        self._assert_every_post_form_has_token(html, expected_forms=2)  # run + free-text


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
