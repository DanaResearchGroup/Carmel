"""Tests for the Carmel Flask UI."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from carmel.schemas import CampaignStateValue
from carmel.services.execution import save_diagnostics
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


def _create_via_form(client: FlaskClient, name: str = "ethanol") -> str:
    response = client.post(
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
        response = client.post(
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
        response = client.post(f"/campaigns/{cid}/plan", follow_redirects=False)
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
        client.post(f"/campaigns/{cid}/plan")
        response = client.post(f"/campaigns/{cid}/approve", follow_redirects=False)
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
        client.post(f"/campaigns/{cid}/plan")
        response = client.post(f"/campaigns/{cid}/reject", follow_redirects=False)
        assert response.status_code == 302
        from carmel.services.state_machine import load_state

        assert load_state(ws).state == CampaignStateValue.BLOCKED


class TestFreeTextIntake:
    def test_free_text_creates_review(self, client: FlaskClient, workspaces_root: Path) -> None:
        cid = _create_via_form(client)
        from carmel.services.campaigns import find_campaign_workspace

        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        response = client.post(
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
        response = client.post(
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
        response = client.post(
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
        response = client.post(
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
        response = client.post(
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

        client.post(
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
        response = client.post(
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
        assert client.post(path).status_code == 404

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
        client.post(f"/campaigns/{cid}/plan")
        response = client.post(f"/campaigns/{cid}/run", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["Location"].endswith(cid)
        assert called["action_id"] is not None

    def test_run_without_plan_actions_returns_400(
        self, client: FlaskClient, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.planner import load_plan, save_plan

        cid = _create_via_form(client)
        client.post(f"/campaigns/{cid}/plan")
        ws = find_campaign_workspace(workspaces_root, cid)
        assert ws is not None
        plan = load_plan(ws)
        save_plan(ws, plan.model_copy(update={"actions": []}))
        assert client.post(f"/campaigns/{cid}/run").status_code == 400


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
