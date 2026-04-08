"""Tests for the Carmel Flask UI."""

from datetime import UTC
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from carmel.schemas import CampaignStateValue
from carmel.services.execution import save_diagnostics
from carmel.ui import create_app


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
