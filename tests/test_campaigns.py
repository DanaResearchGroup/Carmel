"""Tests for carmel.services.campaigns: the literature-at-creation hook.

Focused on the campaigns.py:282 finding: ``start_literature_at_creation`` used to end
in an unbounded ``ticket.wait()`` -- fine for the CLI (a human is watching a foreground
process anyway) but wrong for the HTTP path (``carmel/ui/app.py`` calling
``create_campaign``), where blocking a request worker on an arbitrarily slow provider
call ties it up indefinitely. These tests exercise the resulting ``wait_timeout_s``
branching directly, by faking ``execute_next_action``'s dispatcher ticket rather than
running a real literature agent -- the branching under test lives entirely in
``campaigns.py``, so a fake ticket keeps the test deterministic and fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carmel.config import AgentConfig, CampaignConfig, CarmelConfig
from carmel.schemas import (
    Budgets,
    CampaignInput,
    InitialMixture,
    MixtureComponent,
    ReactorSystem,
    ReactorType,
    TargetObservable,
)
from carmel.services.campaigns import (
    DEFAULT_LITERATURE_WAIT_TIMEOUT_S,
    MissingCampaignConfigError,
    create_campaign,
    create_campaign_from_config,
    start_literature_at_creation,
)


def _make_input(name: str = "campaigns-test") -> CampaignInput:
    return CampaignInput(
        workspace_name=name,
        initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
        target_observables=[TargetObservable(name="ignition_delay")],
        target_reactor_systems=[
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, 5.0),
            )
        ],
        budgets=Budgets(cpu_hours=20.0, experiment_budget=0.0),
    )


class _FakeTicket:
    """Stand-in for ``DispatchTicket``: records every ``wait()`` call instead of
    actually running anything on a background thread."""

    def __init__(self, action_id: str = "lit-1", attempt_id: str = "attempt-1", *, on_wait: Any = None) -> None:
        self.action_id = action_id
        self.attempt_id = attempt_id
        self.wait_calls: list[float | None] = []
        self._on_wait = on_wait

    def wait(self, timeout: float | None = None) -> Any:
        self.wait_calls.append(timeout)
        if self._on_wait is not None:
            return self._on_wait(timeout)
        return None


def _patch_dispatch(monkeypatch: pytest.MonkeyPatch, ticket: _FakeTicket) -> None:
    # ``start_literature_at_creation`` imports ``execute_next_action`` from
    # ``carmel.services.dispatcher`` inside its own function body (a deliberate,
    # documented lazy import so campaign creation without an agent config never
    # touches the dispatcher/agents stack) -- patching the source module's
    # attribute is what a fresh ``from ... import`` on every call will pick up.
    monkeypatch.setattr(
        "carmel.services.dispatcher.execute_next_action",
        lambda workspace_root, campaign, *, handlers=None: ticket,
    )


class TestWaitTimeoutBranching:
    def test_none_dispatches_and_returns_without_waiting_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP path: creation must not block on the literature run finishing."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())  # no agent_config: no plan/state work yet
        ticket = _FakeTicket()
        _patch_dispatch(monkeypatch, ticket)

        outcome = start_literature_at_creation(ws, campaign, AgentConfig(), wait_timeout_s=None)

        assert outcome.result is None
        assert outcome.dispatched_action_id == "lit-1"
        assert outcome.dispatched_attempt_id == "attempt-1"
        assert ticket.wait_calls == [], "wait() must never be called when wait_timeout_s is None"
        assert "dispatched" in outcome.explain()

    def test_bounded_timeout_elapsing_reports_dispatched_not_finished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI path with a slow run: the wait gives up, but is not a crash."""
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())

        def _raise_timeout(_timeout: float | None) -> Any:
            raise TimeoutError

        ticket = _FakeTicket(on_wait=_raise_timeout)
        _patch_dispatch(monkeypatch, ticket)

        outcome = start_literature_at_creation(ws, campaign, AgentConfig(), wait_timeout_s=5.0)

        assert outcome.result is None
        assert outcome.dispatched_action_id == "lit-1"
        assert outcome.dispatched_attempt_id == "attempt-1"
        assert ticket.wait_calls == [5.0]
        assert "still running" in outcome.explain()

    def test_default_timeout_still_synchronously_waits_for_a_quick_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI path, no explicit argument: behavior stays synchronous by default."""
        sentinel = object()
        ticket = _FakeTicket(on_wait=lambda _timeout: sentinel)
        ws = tmp_path / "ws"
        campaign = create_campaign(ws, _make_input())
        _patch_dispatch(monkeypatch, ticket)

        outcome = start_literature_at_creation(ws, campaign, AgentConfig())

        assert outcome.result is sentinel
        assert outcome.dispatched_action_id is None
        assert ticket.wait_calls == [DEFAULT_LITERATURE_WAIT_TIMEOUT_S]


class TestCreateCampaignThreadsWaitTimeout:
    def test_literature_wait_timeout_none_does_not_block_campaign_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The HTTP route's call shape: ``create_campaign(..., literature_wait_timeout_s=None)``."""
        ticket = _FakeTicket()
        _patch_dispatch(monkeypatch, ticket)
        ws = tmp_path / "ws"

        campaign = create_campaign(ws, _make_input(), agent_config=AgentConfig(), literature_wait_timeout_s=None)

        assert campaign is not None
        assert ticket.wait_calls == []

    def test_default_create_campaign_call_still_waits_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI's call shape (``Carmel.py``'s ``carmel literature`` calls
        ``start_literature_at_creation`` directly, not through ``create_campaign``, but
        both must default to the same bounded-synchronous behavior)."""
        sentinel = object()
        ticket = _FakeTicket(on_wait=lambda _timeout: sentinel)
        _patch_dispatch(monkeypatch, ticket)
        ws = tmp_path / "ws"

        create_campaign(ws, _make_input(), agent_config=AgentConfig())

        assert ticket.wait_calls == [DEFAULT_LITERATURE_WAIT_TIMEOUT_S]


def _make_campaign_config() -> CampaignConfig:
    """A minimal, physically valid ``campaign:`` section, built from objects
    rather than dicts since this file already imports the campaign schema types."""
    return CampaignConfig(
        initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
        target_observables=[TargetObservable(name="ignition_delay")],
        target_reactor_systems=[
            ReactorSystem(
                reactor_type=ReactorType.JSR,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, 5.0),
            )
        ],
        budgets=Budgets(cpu_hours=20.0, experiment_budget=0.0),
    )


class TestCreateCampaignFromConfig:
    """``create_campaign_from_config``: the config -> ``CampaignInput`` -> Campaign path."""

    def test_missing_campaign_section_raises_typed_error(self, tmp_path: Path) -> None:
        """An operator who forgot the section gets a message naming the fix, not an AttributeError."""
        config = CarmelConfig(workspace_name="no-campaign", workspace_root=tmp_path / "ws")

        with pytest.raises(MissingCampaignConfigError, match="campaign"):
            create_campaign_from_config(config)

    def test_builds_campaign_from_config_section(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        config = CarmelConfig(workspace_name="cfg-campaign", workspace_root=ws, campaign=_make_campaign_config())

        campaign = create_campaign_from_config(config)

        assert campaign.input.workspace_name == "cfg-campaign"
        assert campaign.input.initial_mixture == config.campaign.initial_mixture  # type: ignore[union-attr]
        assert ws.exists()

    def test_workspaces_root_override_wins_over_config(self, tmp_path: Path) -> None:
        default_ws = tmp_path / "default-ws"
        override_ws = tmp_path / "override-ws"
        config = CarmelConfig(
            workspace_name="cfg-campaign", workspace_root=default_ws, campaign=_make_campaign_config()
        )

        campaign = create_campaign_from_config(config, workspaces_root=override_ws)

        assert campaign.workspace_root == override_ws
        assert override_ws.exists()
        assert not default_ws.exists()

    def test_agent_config_is_threaded_through_to_literature_at_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agent_config=config.agents`` must reach ``create_campaign`` so the same
        config controls literature-at-creation. Uses the TEST/mock tier (the
        ``AgentConfig()`` default) and fakes the dispatcher ticket -- never a real
        model or network, per the no-network rule for this test suite."""
        ticket = _FakeTicket()
        _patch_dispatch(monkeypatch, ticket)
        config = CarmelConfig(
            workspace_name="cfg-campaign",
            workspace_root=tmp_path / "ws",
            agents=AgentConfig(),  # TEST tier, MOCK provider -- no network
            campaign=_make_campaign_config(),
        )

        create_campaign_from_config(config)

        # A dispatch only happens if agent_config reached create_campaign's
        # maybe_start_literature_at_creation call -- confirming the threading.
        assert ticket.wait_calls == [DEFAULT_LITERATURE_WAIT_TIMEOUT_S]
