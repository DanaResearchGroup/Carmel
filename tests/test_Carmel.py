"""Tests for Carmel CLI."""

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from Carmel import main
from carmel.paths import WORKSPACE_SUBDIRS
from carmel.version import __version__
from tests.test_acquisition import _matching_body, _patch_text_sniff_to_pdf


class TestVersionCommand:
    """Tests for 'carmel version'."""

    def test_exit_code_zero(self) -> None:
        assert main(["version"]) == 0

    def test_output_contains_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["version"])
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_output_contains_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["version"])
        captured = capsys.readouterr()
        assert "carmel" in captured.out


class TestValidateConfigCommand:
    """Tests for 'carmel validate-config'."""

    def test_valid_config_exit_code(self, valid_config_file: Path) -> None:
        assert main(["validate-config", str(valid_config_file)]) == 0

    def test_valid_config_stdout(self, valid_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["validate-config", str(valid_config_file)])
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower()

    def test_missing_file_exit_code(self) -> None:
        assert main(["validate-config", "/nonexistent/config.yaml"]) == 1

    def test_missing_file_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["validate-config", "/nonexistent/config.yaml"])
        captured = capsys.readouterr()
        assert "failed" in captured.err.lower()

    def test_invalid_yaml_exit_code(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not_a_mapping")
        assert main(["validate-config", str(path)]) == 1

    def test_incomplete_config_exit_code(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.dump({"logging_level": "INFO"}))
        assert main(["validate-config", str(path)]) == 1

    def test_invalid_config_stderr_has_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({}))
        main(["validate-config", str(path)])
        captured = capsys.readouterr()
        assert "failed" in captured.err.lower()

    def test_missing_arg_exits_with_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["validate-config"])
        assert exc_info.value.code == 2


class TestInitWorkspaceCommand:
    """Tests for 'carmel init-workspace'."""

    def test_exit_code_zero(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        assert main(["init-workspace", str(ws)]) == 0

    def test_creates_directory(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        main(["init-workspace", str(ws)])
        assert ws.is_dir()

    def test_creates_all_subdirs(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        main(["init-workspace", str(ws)])
        for subdir in WORKSPACE_SUBDIRS:
            assert (ws / subdir).is_dir()

    def test_output_message(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ws = tmp_path / "workspace"
        main(["init-workspace", str(ws)])
        captured = capsys.readouterr()
        assert "initialized" in captured.out.lower()

    def test_idempotent(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        assert main(["init-workspace", str(ws)]) == 0
        assert main(["init-workspace", str(ws)]) == 0

    def test_missing_arg_exits_with_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["init-workspace"])
        assert exc_info.value.code == 2


class TestInitWorkspaceFailure:
    """Tests for the init-workspace failure path."""

    def test_unwritable_parent_returns_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Carmel

        def _raise(_directory: Path) -> Path:
            raise OSError("Read-only file system")

        monkeypatch.setattr(Carmel, "init_workspace", _raise)
        assert main(["init-workspace", str(tmp_path / "ws")]) == 1

    def test_unwritable_parent_reports_to_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import Carmel

        def _raise(_directory: Path) -> Path:
            raise OSError("Read-only file system")

        monkeypatch.setattr(Carmel, "init_workspace", _raise)
        main(["init-workspace", str(tmp_path / "ws")])
        captured = capsys.readouterr()
        assert "Failed to initialize workspace" in captured.err
        assert "Read-only file system" in captured.err
        assert captured.out == ""


class TestServeCommand:
    """Tests for the serve command.

    ``app.run`` is stubbed out so the test never binds a socket.
    """

    def test_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("flask.Flask.run", lambda *args, **kwargs: None)
        assert main(["serve", "--workspaces", str(tmp_path)]) == 0

    def test_announces_bound_address(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("flask.Flask.run", lambda *args, **kwargs: None)
        main(["serve", "--workspaces", str(tmp_path), "--host", "0.0.0.0", "--port", "8080"])
        assert "http://0.0.0.0:8080" in capsys.readouterr().out

    def test_passes_host_port_and_debug_to_flask(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: dict[str, object] = {}

        def _fake_run(_self: object, **kwargs: object) -> None:
            recorded.update(kwargs)

        monkeypatch.setattr("flask.Flask.run", _fake_run)
        main(["serve", "--workspaces", str(tmp_path), "--host", "127.0.0.1", "--port", "5555", "--debug"])
        assert recorded == {"host": "127.0.0.1", "port": 5555, "debug": True}

    def test_uses_given_workspaces_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("flask.Flask.run", lambda *args, **kwargs: None)
        target = tmp_path / "nested" / "workspaces"
        assert main(["serve", "--workspaces", str(target)]) == 0
        assert target.is_dir()

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
    def test_debug_on_non_loopback_host_is_rejected(
        self, host: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("flask.Flask.run", lambda *args, **kwargs: None)
        exit_code = main(["serve", "--workspaces", str(tmp_path), "--host", host, "--debug"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "--debug" in captured.err
        assert host in captured.err
        assert captured.out == ""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_debug_on_loopback_host_is_allowed(
        self, host: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("flask.Flask.run", lambda *args, **kwargs: None)
        assert main(["serve", "--workspaces", str(tmp_path), "--host", host, "--debug"]) == 0


class TestCliEntryPoint:
    """Tests for the console-script entry point."""

    def test_exits_with_main_return_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import Carmel

        monkeypatch.setattr(sys, "argv", ["carmel"])
        with pytest.raises(SystemExit) as exc_info:
            Carmel.cli()
        assert exc_info.value.code == 1

    def test_exits_zero_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import Carmel

        monkeypatch.setattr(sys, "argv", ["carmel", "version"])
        with pytest.raises(SystemExit) as exc_info:
            Carmel.cli()
        assert exc_info.value.code == 0


class TestNoCommand:
    """Tests for CLI with no command or unknown input."""

    def test_no_args_returns_one(self) -> None:
        assert main([]) == 1

    def test_no_args_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        captured = capsys.readouterr()
        assert "carmel" in captured.out.lower()


class TestLiteratureCommand:
    """Tests for the `carmel literature` subcommand."""

    def test_no_config_exits_one_with_message(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        code = main(["literature", "--campaign", "some-id", "--workspaces", str(tmp_path)])
        assert code == 1
        assert "agent config" in capsys.readouterr().err.lower()

    def test_config_without_agents_section_exits_one(
        self, tmp_path: Path, valid_config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from Carmel import main

        code = main(
            [
                "literature",
                "--campaign",
                "some-id",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(valid_config_file),
            ]
        )
        assert code == 1
        assert "agent config" in capsys.readouterr().err.lower()

    def test_unknown_campaign_exits_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "workspace_name": "lit-cli",
                    "workspace_root": str(tmp_path / "root"),
                    "agents": {"tier": "test", "provider": "mock"},
                }
            )
        )
        code = main(
            [
                "literature",
                "--campaign",
                "missing-id",
                "--workspaces",
                str(tmp_path / "workspaces"),
                "--config",
                str(config_file),
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_invalid_config_exits_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        code = main(
            [
                "literature",
                "--campaign",
                "x",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(tmp_path / "missing.yaml"),
            ]
        )
        assert code == 1
        assert "Failed to load config" in capsys.readouterr().err

    def test_runs_literature_for_existing_campaign(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Mock-tier end-to-end: the CLI drives the same service hook."""
        import yaml

        from Carmel import main
        from carmel.schemas import (
            Budgets,
            CampaignInput,
            CampaignStateValue,
            InitialMixture,
            MixtureComponent,
            ReactorSystem,
            ReactorType,
            TargetObservable,
        )
        from carmel.services.campaigns import create_campaign
        from carmel.services.state_machine import load_state

        workspaces = tmp_path / "workspaces"
        campaign = create_campaign(
            workspaces / "cli-lit",
            CampaignInput(
                workspace_name="cli-lit",
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
            ),
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "workspace_name": "cli-lit",
                    "workspace_root": str(workspaces / "cli-lit"),
                    "agents": {"tier": "test", "provider": "mock"},
                }
            )
        )

        code = main(
            [
                "literature",
                "--campaign",
                campaign.campaign_id,
                "--workspaces",
                str(workspaces),
                "--config",
                str(config_file),
            ]
        )

        assert code == 0
        assert "outcome" in capsys.readouterr().out
        assert load_state(workspaces / "cli-lit").state == CampaignStateValue.LITERATURE_READY


class TestLiteratureSkipDiagnostics:
    """When a literature run does NOT start, the CLI must say why -- accurately.

    The message used to name three candidate causes it had never checked ("the config
    toggle is off, the plan requires approval, or the campaign state does not allow
    it"). A run that started and died on a provider 503 was reported that way, with all
    three statements false, sending the operator to look at configuration that was fine.
    A wrong diagnosis costs more than no diagnosis.
    """

    @staticmethod
    def _campaign(workspaces: Path) -> Any:
        from carmel.schemas import (
            Budgets,
            CampaignInput,
            InitialMixture,
            MixtureComponent,
            ReactorSystem,
            ReactorType,
            TargetObservable,
        )
        from carmel.services.campaigns import create_campaign

        return create_campaign(
            workspaces / "cli-skip",
            CampaignInput(
                workspace_name="cli-skip",
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
            ),
        )

    def test_toggle_off_is_named_and_nothing_else_is_blamed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import yaml

        from Carmel import main

        workspaces = tmp_path / "workspaces"
        campaign = self._campaign(workspaces)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "workspace_name": "cli-skip",
                    "workspace_root": str(workspaces / "cli-skip"),
                    "agents": {
                        "tier": "test",
                        "provider": "mock",
                        "literature_at_campaign_start": False,
                    },
                }
            )
        )

        code = main(
            [
                "literature",
                "--campaign",
                campaign.campaign_id,
                "--workspaces",
                str(workspaces),
                "--config",
                str(config_file),
            ]
        )
        err = capsys.readouterr().err

        assert code == 1
        assert "literature_at_campaign_start" in err
        # The two causes that were NOT the problem must not be asserted.
        assert "requires approval" not in err
        assert "campaign state" not in err

    def test_running_the_module_as_a_script_actually_runs_it(self) -> None:
        """`python Carmel.py ...` must DO something.

        Carmel.py had no ``if __name__ == "__main__"`` block, so running it directly
        parsed nothing, ran nothing, printed nothing, and exited 0 -- indistinguishable
        from a successful run that produced no output. Only the installed console script
        worked, which made the most obvious way to invoke a checkout also the most
        quietly misleading one.
        """
        import subprocess

        root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            [sys.executable, str(root / "Carmel.py"), "version"],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )

        assert completed.returncode == 0
        assert completed.stdout.strip(), "running Carmel.py directly produced no output at all"

    def test_state_skip_reports_the_observed_state_not_a_likely_story(self) -> None:
        from carmel.services.campaigns import LiteratureStartOutcome, LiteratureStartSkipped

        outcome = LiteratureStartOutcome(
            skip_reason=LiteratureStartSkipped.CAMPAIGN_STATE_NOT_READY,
            detail="campaign state is 'literature_ready'",
        )

        # The observed fact has to appear, or the operator is again reading a guess.
        assert "literature_ready" in outcome.explain()

    def test_every_skip_reason_has_an_operator_facing_explanation(self) -> None:
        # A reason added without an explanation would fall back to a bare enum name --
        # i.e. straight back to a diagnostic that tells the operator nothing.
        from carmel.services.campaigns import LITERATURE_SKIP_EXPLANATIONS, LiteratureStartSkipped

        assert set(LITERATURE_SKIP_EXPLANATIONS) == set(LiteratureStartSkipped)
        for text in LITERATURE_SKIP_EXPLANATIONS.values():
            assert len(text) > 40


class TestServeConfigOption:
    def test_serve_invalid_config_exits_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        code = main(["serve", "--config", str(tmp_path / "missing.yaml")])
        assert code == 1
        assert "Failed to load config" in capsys.readouterr().err


CAMPAIGN_YAML = """\
workspace_name: nh3-cli
workspace_root: {root}
agents:
  tier: test
  provider: mock
  external_provider_consent: false
  literature_at_campaign_start: false
  budget:
    max_model_calls: 5
    max_tokens: 50000
    max_fetches: 5
    max_cost_usd: 1.0
campaign:
  initial_mixture:
    components:
      - species: NH3
        mole_fraction: 0.01
      - species: O2
        mole_fraction: 0.0075
      - species: AR
        mole_fraction: 0.9825
  target_observables:
    - name: ignition_delay_time
  target_reactor_systems:
    - reactor_type: shock_tube
      temperature_range_K: [1400.0, 2000.0]
      pressure_range_bar: [1.0, 30.0]
  budgets:
    cpu_hours: 10.0
    experiment_budget: 0.0
"""


def _write_campaign_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "nh3.yaml"
    cfg.write_text(CAMPAIGN_YAML.format(root=tmp_path / "ws"), encoding="utf-8")
    return cfg


class TestNewCampaignCommand:
    """`carmel new-campaign` replaces the hand-written setup script an operator would
    otherwise maintain outside the repo -- a private mock of Carmel's own API that rots
    against the real one."""

    def test_creates_a_campaign_from_the_config_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        out = capsys.readouterr().out
        assert "Campaign ID" in out
        assert (tmp_path / "ws" / "campaign.yaml").exists()

    def test_a_config_without_a_campaign_section_explains_what_to_add(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "bare.yaml"
        cfg.write_text(f"workspace_name: bare\nworkspace_root: {tmp_path / 'ws2'}\n", encoding="utf-8")

        from Carmel import main

        assert main(["new-campaign", "--config", str(cfg)]) == 1
        assert "campaign" in capsys.readouterr().out.lower()

    def test_workspaces_is_the_parent_directory_so_requests_can_find_the_campaign(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--workspaces` must mean the same thing in every command: the PARENT holding
        one subdirectory per campaign.

        This used to be passed straight through as the campaign's own workspace root, so
        `new-campaign --workspaces D` put the campaign directly in ``D`` while
        `requests --workspaces D` scanned ``D``'s subdirectories -- and could never find
        the campaign that had just been created there.
        """
        from Carmel import main
        from carmel.config import load_config

        parent = tmp_path / "workspaces"
        cfg = _write_campaign_config(tmp_path)
        name = load_config(cfg).workspace_name

        assert main(["new-campaign", "--config", str(cfg), "--workspaces", str(parent)]) == 0
        assert (parent / name / "campaign.yaml").exists()
        capsys.readouterr()

        from carmel.services.campaigns import load_campaign

        campaign_id = load_campaign(parent / name).campaign_id
        assert main(["requests", "--campaign", campaign_id, "--workspaces", str(parent)]) == 0
        assert "not found" not in capsys.readouterr().out.lower()

    def test_a_workspace_name_that_escapes_the_parent_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``workspace_name`` is only validated as non-blank, so it can carry ``..``."""
        cfg = tmp_path / "escape.yaml"
        cfg.write_text(
            f"workspace_name: ../escaped\nworkspace_root: {tmp_path / 'ws3'}\n",
            encoding="utf-8",
        )

        from Carmel import main

        assert main(["new-campaign", "--config", str(cfg), "--workspaces", str(tmp_path / "parent")]) == 1
        assert not (tmp_path / "escaped").exists()


class TestRequestsCommand:
    """The manual-acquisition step. Listing must print the exact command to run next,
    and admitting must report the identity verdict immediately -- waiting for a whole
    literature run to learn whether a file was accepted is what made this painful."""

    @pytest.fixture(autouse=True)
    def _pdf_sniff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # This class's fixtures write plain-text bodies to stand in for "the right
        # paper" -- these tests are about CLI wiring (exit codes, --collect/--add
        # reporting), not about format gating, which is exercised on its own in
        # TestPlainTextLandingPageRefused. Without this seam every "accepted" case here
        # would now be REJECTED by the new text/plain admission guard.
        _patch_text_sniff_to_pdf(monkeypatch)

    def _campaign_with_request(self, tmp_path: Path) -> tuple[str, Path]:
        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import record_request
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        record_request(
            ws,
            title="Shock tube study of ammonia oxidation ignition delay times",
            doi="10.1016/j.test.2019.01.001",
            landing_url="https://doi.org/10.1016/j.test.2019.01.001",
            reason=AcquisitionReason.PAYWALLED,
            detail="HTTP 403",
        )
        return load_campaign(ws).campaign_id, ws

    def test_config_workspace_root_is_honoured_without_an_explicit_workspaces_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--config` names a workspace_root; this command used to ignore it.

        It resolved the root before the config was even loaded, then read the config
        only for max_artifact_bytes -- so it scanned the DEFAULT workspaces directory
        while the operator had just handed it a config naming a different one. The
        failure is quiet in the worst way: "No campaign <id> under <dir>", naming a
        directory the operator never asked about, for a campaign that exists.

        `new-campaign --workspaces` already advertises "default: the config's
        workspace_root"; only this command disagreed.
        """
        cid, _ = self._campaign_with_request(tmp_path)
        from Carmel import main

        # Point the default elsewhere and leave it empty: if the config's
        # workspace_root were ignored, the campaign could only be found by accident.
        empty_default = tmp_path / "not-here"
        empty_default.mkdir()
        monkeypatch.setenv("CARMEL_WORKSPACES", str(empty_default))

        code = main(["requests", "--campaign", cid, "--config", str(tmp_path / "nh3.yaml")])

        assert code == 0, "the campaign named by the config's workspace_root was not found"
        assert "Shock tube" in capsys.readouterr().out

    def test_an_explicit_workspaces_flag_still_wins_over_the_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Precedence must not invert: an operator who types --workspaces means it."""
        cid, _ = self._campaign_with_request(tmp_path)
        from Carmel import main

        code = main(
            ["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--config", str(tmp_path / "nh3.yaml")]
        )

        assert code == 0
        assert "Shock tube" in capsys.readouterr().out

    def test_listing_prints_the_next_command_for_each_pending_paper(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, _ = self._campaign_with_request(tmp_path)

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "awaiting a human" in out
        assert "--add" in out and "--slug" in out

    def test_the_listing_never_prints_a_raw_reason_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The "why" line must render the operator-facing phrase, never the enum value.

        A raw value reaches the operator as ``oa_lookup_not_attempted``, underscores
        intact -- exactly the leak the phrase table exists to prevent, and a leak that
        widens silently every time a member is added. Asserting the ABSENCE of the raw
        value is the load-bearing half: asserting only that the phrase is present would
        still pass if both were printed.
        """
        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import reason_phrase, record_request
        from carmel.services.campaigns import load_campaign

        cid, ws = self._campaign_with_request(tmp_path)
        record_request(
            ws,
            title="Laminar burning velocity of syngas at elevated pressure",
            doi="10.1016/j.test.2020.02.002",
            landing_url="https://doi.org/10.1016/j.test.2020.02.002",
            reason=AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED,
            detail="no open-access resolver is configured, so no lookup could run",
        )
        assert load_campaign(ws).campaign_id == cid

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED.value not in out
        assert reason_phrase(AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED) in out

    def test_the_wrong_paper_is_rejected_and_nothing_is_admitted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, ws = self._campaign_with_request(tmp_path)
        wrong = tmp_path / "wrong.txt"
        wrong.write_text("Laminar burning velocities of methane air mixtures\n", encoding="utf-8")

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(wrong)]) == 1
        out = capsys.readouterr().out
        assert "REJECTED" in out
        assert "Nothing was admitted" in out

    def test_the_right_paper_is_accepted_and_says_so(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, _ = self._campaign_with_request(tmp_path)
        right = tmp_path / "right.txt"
        right.write_text(
            _matching_body(
                "Abstract: we report ignition delay times.\n",
                title="Shock tube study of ammonia oxidation ignition delay times",
                doi="10.1016/j.test.2019.01.001",
            ),
            encoding="utf-8",
        )

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(right)]) == 0
        assert "ACCEPTED" in capsys.readouterr().out

    def test_a_rejected_drop_can_be_retried_with_the_correct_paper(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rejection must leave the request on the worklist, not retire it."""
        cid, _ = self._campaign_with_request(tmp_path)
        wrong = tmp_path / "wrong.txt"
        wrong.write_text("Something else entirely\n", encoding="utf-8")
        right = tmp_path / "right.txt"
        right.write_text(
            _matching_body(
                title="Shock tube study of ammonia oxidation ignition delay times",
                doi="10.1016/j.test.2019.01.001",
            ),
            encoding="utf-8",
        )

        from Carmel import main

        main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(wrong)])
        capsys.readouterr()
        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(right)]) == 0
        assert "ACCEPTED" in capsys.readouterr().out

    def test_collect_admits_a_good_dropped_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """`--collect` calls the existing service and reports its verdict -- no
        reimplementation of the sweep in the CLI layer."""
        cid, ws = self._campaign_with_request(tmp_path)
        from carmel.services.acquisition import drop_path_for, pending_requests

        slug = pending_requests(ws)[0].slug
        drop = drop_path_for(ws, slug, suffix=".txt")
        drop.parent.mkdir(parents=True, exist_ok=True)
        drop.write_text(
            _matching_body(
                "Abstract: we report ignition delay times.\n",
                title="Shock tube study of ammonia oxidation ignition delay times",
                doi="10.1016/j.test.2019.01.001",
            ),
            encoding="utf-8",
        )

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--collect"]) == 0
        out = capsys.readouterr().out
        assert "ACCEPTED" in out
        assert "1 accepted, 0 rejected" in out

    def test_collect_rejects_a_wrong_dropped_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, ws = self._campaign_with_request(tmp_path)
        from carmel.services.acquisition import drop_path_for, pending_requests

        slug = pending_requests(ws)[0].slug
        drop = drop_path_for(ws, slug, suffix=".txt")
        drop.parent.mkdir(parents=True, exist_ok=True)
        drop.write_text("Laminar burning velocities of methane air mixtures\n", encoding="utf-8")

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--collect"]) == 1
        out = capsys.readouterr().out
        assert "REJECTED" in out
        assert "0 accepted, 1 rejected" in out

    def test_collect_with_empty_inbox_tells_the_operator_where_to_drop_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, ws = self._campaign_with_request(tmp_path)
        from Carmel import main
        from carmel.services.acquisition import inbox_dir

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--collect"]) == 0
        out = capsys.readouterr().out
        assert "Nothing new in the inbox" in out
        assert str(inbox_dir(ws)) in out

    def test_collect_and_add_together_are_refused(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, _ = self._campaign_with_request(tmp_path)
        right = tmp_path / "right.txt"
        right.write_text("Shock tube study\n", encoding="utf-8")

        from Carmel import main

        assert (
            main(
                [
                    "requests",
                    "--campaign",
                    cid,
                    "--workspaces",
                    str(tmp_path),
                    "--collect",
                    "--add",
                    str(right),
                ]
            )
            == 1
        )

    def test_listing_tells_the_operator_about_collect(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, _ = self._campaign_with_request(tmp_path)

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "--collect" in out


class TestRequestsCommandAddDirectory:
    """`--add` pointed at a directory of publisher-named downloads. The operator's real
    pain: filenames like ``Experimental studies of the fundamental flame speeds of
    syngas (H2-CO)-air mixtures.pdf`` need shell quoting for every single paper when
    typed one at a time. Handing Carmel the whole folder must match by CONTENT
    (`_infer_slug`), never by filename, and one bad file must never abort the batch.
    """

    _FIRST_TITLE = "Experimental studies of the fundamental flame speeds of syngas H2 CO air mixtures"
    _FIRST_DOI = "10.1016/j.combustflame.2010.09.004"
    _SECOND_TITLE = "Shock tube study of ammonia oxidation ignition delay times"
    _SECOND_DOI = "10.1016/j.test.2019.01.001"

    @pytest.fixture(autouse=True)
    def _pdf_sniff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same seam as TestRequestsCommand: these fixtures write plain-text bodies
        # (named .pdf to mirror real publisher filenames, since sniffing is by content,
        # never filename) and this class is about directory-add wiring, not format
        # gating.
        _patch_text_sniff_to_pdf(monkeypatch)

    def _campaign_with_requests(self, tmp_path: Path, titles_dois: list[tuple[str, str]]) -> tuple[str, Path]:
        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import record_request
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        for title, doi in titles_dois:
            record_request(
                ws,
                title=title,
                doi=doi,
                landing_url=f"https://doi.org/{doi}",
                reason=AcquisitionReason.PAYWALLED,
            )
        return load_campaign(ws).campaign_id, ws

    def test_a_directory_of_matching_papers_is_all_admitted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, ws = self._campaign_with_requests(
            tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI), (self._SECOND_TITLE, self._SECOND_DOI)]
        )
        lit = tmp_path / "lit"
        lit.mkdir()
        # Real publisher filenames: spaces, parentheses, hyphens -- the whole point is
        # that these are irrelevant to matching, which happens on document content.
        (lit / "Experimental studies of the fundamental flame speeds of syngas (H2-CO)-air mixtures.pdf").write_text(
            _matching_body("Abstract: measurements follow.\n", title=self._FIRST_TITLE, doi=self._FIRST_DOI),
            encoding="utf-8",
        )
        (lit / "Shock tube study (ammonia oxidation) - ignition delay times [2019].pdf").write_text(
            _matching_body(
                "Abstract: we report ignition delay times.\n", title=self._SECOND_TITLE, doi=self._SECOND_DOI
            ),
            encoding="utf-8",
        )

        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionStatus
        from carmel.services.acquisition import load_manifest

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 0
        out = capsys.readouterr().out
        assert "2 accepted, 0 rejected, 0 already acquired, 0 skipped" in out
        assert out.count("ACCEPTED") == 2

        manifest = load_manifest(ws)
        assert all(r.status == AcquisitionStatus.FULFILLED for r in manifest.requests)

    def test_one_bad_file_does_not_abort_the_batch(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """With a single pending request, an unrelated file still resolves to that one
        slug (nothing else to infer) and is REJECTED by the identity check itself --
        exercising the `else` branch of the per-file loop."""
        cid, ws = self._campaign_with_requests(tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI)])
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / "good.pdf").write_text(
            _matching_body("Abstract: measurements follow.\n", title=self._FIRST_TITLE, doi=self._FIRST_DOI),
            encoding="utf-8",
        )
        (lit / "bad.pdf").write_text("An entirely unrelated document about catalytic converters.\n", encoding="utf-8")

        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionStatus
        from carmel.services.acquisition import load_manifest

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 1
        out = capsys.readouterr().out
        assert "ACCEPTED  good.pdf" in out
        assert "REJECTED  bad.pdf" in out
        assert "1 accepted, 1 rejected, 0 already acquired, 0 skipped" in out

        manifest = load_manifest(ws)
        assert any(r.status == AcquisitionStatus.FULFILLED for r in manifest.requests)

    def test_a_file_matching_no_pending_request_does_not_abort_the_batch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With two-or-more pending requests, a file matching neither makes
        `_infer_slug` raise `ValueError` (zero candidates) instead of resolving to a
        request at all -- exercising the `except (OSError, ValueError)` branch of the
        per-file loop, distinct from the identity-mismatch case above."""
        cid, ws = self._campaign_with_requests(
            tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI), (self._SECOND_TITLE, self._SECOND_DOI)]
        )
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / "good.pdf").write_text(
            _matching_body("Abstract: measurements follow.\n", title=self._FIRST_TITLE, doi=self._FIRST_DOI),
            encoding="utf-8",
        )
        (lit / "bad.pdf").write_text("An entirely unrelated document about catalytic converters.\n", encoding="utf-8")

        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionStatus
        from carmel.services.acquisition import load_manifest

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 1
        out = capsys.readouterr().out
        assert "ACCEPTED  good.pdf" in out
        assert "REJECTED  bad.pdf: cannot tell which pending request this file is for" in out
        assert "1 accepted, 1 rejected, 0 already acquired, 0 skipped" in out

        manifest = load_manifest(ws)
        assert any(r.status == AcquisitionStatus.FULFILLED for r in manifest.requests)

    def test_subdirectories_dotfiles_and_partial_downloads_are_skipped_and_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, _ = self._campaign_with_requests(tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI)])
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / "good.pdf").write_text(
            _matching_body("Abstract: measurements follow.\n", title=self._FIRST_TITLE, doi=self._FIRST_DOI),
            encoding="utf-8",
        )
        (lit / "subdir").mkdir()
        (lit / "subdir" / "nested.pdf").write_text("should never be seen\n", encoding="utf-8")
        (lit / ".hidden.pdf").write_text("should never be seen\n", encoding="utf-8")
        (lit / "downloading.pdf.part").write_text("partial\n", encoding="utf-8")
        (lit / "downloading.pdf.crdownload").write_text("partial\n", encoding="utf-8")
        (lit / "scratch.tmp").write_text("partial\n", encoding="utf-8")

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 0
        out = capsys.readouterr().out
        assert "1 accepted, 0 rejected, 0 already acquired, 5 skipped" in out
        assert "SKIPPED   subdir (subdirectory" in out
        assert "SKIPPED   .hidden.pdf (dotfile)" in out
        assert "SKIPPED   downloading.pdf.part (partial download)" in out
        assert "SKIPPED   downloading.pdf.crdownload (partial download)" in out
        assert "SKIPPED   scratch.tmp (partial download)" in out

    def test_slug_with_a_directory_is_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, _ = self._campaign_with_requests(tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI)])
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / "good.pdf").write_text(_matching_body(title=self._FIRST_TITLE, doi=self._FIRST_DOI), encoding="utf-8")

        from Carmel import main

        assert (
            main(
                ["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit), "--slug", "whatever"]
            )
            == 1
        )
        out = capsys.readouterr().out
        assert "--slug" in out
        assert "directory" in out

    def test_single_file_add_is_unchanged(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Regression: pointing `--add` at a plain file must behave exactly as before
        directory support was added."""
        cid, _ = self._campaign_with_requests(tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI)])
        right = tmp_path / "right.pdf"
        right.write_text(
            _matching_body("Abstract: measurements follow.\n", title=self._FIRST_TITLE, doi=self._FIRST_DOI),
            encoding="utf-8",
        )

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(right)]) == 0
        out = capsys.readouterr().out
        assert "ACCEPTED" in out
        assert "Re-run `carmel literature`" in out

    def test_empty_directory_is_reported_not_silently_successful(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, _ = self._campaign_with_requests(tmp_path, [(self._FIRST_TITLE, self._FIRST_DOI)])
        lit = tmp_path / "lit"
        lit.mkdir()

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 1
        out = capsys.readouterr().out
        assert "empty" in out.lower()
        assert "nothing to admit" in out.lower()


class TestRequestsCommandReIngest:
    """Re-running an ingest over the same download folder must be a clean no-op.

    Reproduces the live shape that motivated this: an operator re-runs the folder
    ingest, and the papers accepted on the first pass come back as "REJECTED", some of
    them with a note accusing a correct paper of not looking like itself. Every outcome
    was right and the report was wrong, which is the failure mode that teaches an
    operator to stop reading the verdicts.
    """

    _TITLE = "Shock tube ammonia oxidation ignition delay times"
    _DOI = "10.1016/j.test.2019.01.001"

    @pytest.fixture(autouse=True)
    def _pdf_sniff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same seam as TestRequestsCommand: this class is about re-ingest idempotency
        # (SKIPPED vs REJECTED reporting), not format gating.
        _patch_text_sniff_to_pdf(monkeypatch)

    def _campaign(self, tmp_path: Path) -> tuple[str, Path]:
        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import record_request
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        record_request(
            ws,
            title=self._TITLE,
            doi=self._DOI,
            landing_url=f"https://doi.org/{self._DOI}",
            reason=AcquisitionReason.PAYWALLED,
            detail="HTTP 403",
        )
        return load_campaign(ws).campaign_id, ws

    def _folder(self, tmp_path: Path) -> Path:
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / "publisher-named-download.txt").write_text(
            _matching_body("Abstract: ignition delay times.\n", title=self._TITLE, doi=self._DOI), encoding="utf-8"
        )
        return lit

    def test_the_second_ingest_reports_skipped_not_rejected_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from Carmel import main

        cid, _ = self._campaign(tmp_path)
        lit = self._folder(tmp_path)
        args = ["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]

        assert main(args) == 0
        capsys.readouterr()

        # Exit 0 is the load-bearing half: a folder in which every paper is already held
        # is a success. Requiring a fresh acceptance would fail the second run of an
        # ingest that fully succeeded the first time.
        assert main(args) == 0
        out = capsys.readouterr().out
        assert "SKIPPED" in out
        assert "already acquired" in out
        assert "REJECTED" not in out

    def test_a_single_add_of_an_already_held_paper_also_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from Carmel import main

        cid, _ = self._campaign(tmp_path)
        paper = self._folder(tmp_path) / "publisher-named-download.txt"
        args = ["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(paper)]

        assert main(args) == 0
        capsys.readouterr()

        assert main(args) == 0
        out = capsys.readouterr().out
        assert "already acquired" in out
        assert "REJECTED" not in out


class TestRequestsCommandDirectoryExitCode:
    """A run that admitted nothing must not look like one that worked.

    `skipped` covers junk -- dotfiles, partial downloads, subdirectories. Letting it
    stand in for an acceptance would make a folder of pure noise exit 0, which is the
    signal a script or a CI lane reads as "the papers are in".
    """

    def _campaign(self, tmp_path: Path) -> str:
        from Carmel import main
        from carmel.schemas.acquisition import AcquisitionReason
        from carmel.services.acquisition import record_request
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        record_request(
            ws,
            title="Shock tube ammonia oxidation ignition delay times",
            doi="10.1016/j.test.2019.01.001",
            landing_url="https://doi.org/10.1016/j.test.2019.01.001",
            reason=AcquisitionReason.PAYWALLED,
            detail="HTTP 403",
        )
        return load_campaign(ws).campaign_id

    def test_a_directory_of_only_junk_exits_nonzero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from Carmel import main

        cid = self._campaign(tmp_path)
        lit = tmp_path / "lit"
        lit.mkdir()
        (lit / ".hidden.pdf").write_text("x", encoding="utf-8")
        (lit / "half.crdownload").write_text("x", encoding="utf-8")
        (lit / "subdir").mkdir()

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(lit)]) == 1
        out = capsys.readouterr().out
        assert "0 accepted" in out


class TestCorpusPassCommand:
    """The operator-facing half of the second pass.

    Appending is the authorisation: it names an explicit budget, and it is recorded
    in the plan whether or not the run then succeeds.
    """

    def _campaign(self, tmp_path: Path, *, dispatchable: bool = False) -> tuple[str, Path]:
        """A campaign with a plan; optionally advanced to a state that can dispatch.

        ``dispatchable`` walks the campaign up the real state ladder to
        APPROVED_FOR_EXECUTION. A corpus pass is a SECOND pass over papers a campaign
        already holds, so by the time an operator appends one the campaign has long
        since left DRAFT -- a draft campaign has not run the literature search whose
        results the corpus pass re-reads. Tests that only append (``--dry-run``) do not
        need it, because appending is authorisation and deliberately works regardless
        of whether the run then succeeds.
        """
        from Carmel import main
        from carmel.schemas import CampaignStateValue
        from carmel.services.campaigns import load_campaign
        from carmel.services.planner import plan_and_save
        from carmel.services.state_machine import update_state

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        campaign = load_campaign(ws)
        plan_and_save(ws, campaign, include_literature=True)
        if dispatchable:
            for target in (
                CampaignStateValue.VALIDATED,
                CampaignStateValue.READY_FOR_PLANNING,
                CampaignStateValue.PLAN_PENDING_APPROVAL,
                CampaignStateValue.APPROVED_FOR_EXECUTION,
            ):
                update_state(ws, target)
        return campaign.campaign_id, ws

    def test_a_campaign_with_no_plan_is_told_what_is_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The raw failure is 'JSON file not found: .../plan.json', which tells an
        operator nothing about the remedy."""
        from Carmel import main
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        cid = load_campaign(tmp_path / "ws").campaign_id

        code = main(
            ["corpus-pass", "--campaign", cid, "--budget-tokens", "100000", "--workspaces", str(tmp_path), "--dry-run"]
        )

        assert code == 1
        err = capsys.readouterr().err
        assert "no plan yet" in err
        assert "plan.json" not in err

    def test_dry_run_appends_the_action_before_the_t3_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """T3 must stay the last executable action, and the ordering is the honest
        one anyway: corpus findings are context for the T3 run."""
        from Carmel import main
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import load_plan

        cid, ws = self._campaign(tmp_path)

        code = main(
            ["corpus-pass", "--campaign", cid, "--budget-tokens", "250000", "--workspaces", str(tmp_path), "--dry-run"]
        )

        assert code == 0
        out = capsys.readouterr().out
        # The cap is reported in the unit that binds. With no --config there is no
        # model to price against, so the dollar cost is declared unavailable rather
        # than defaulted to a number that would read as free.
        assert "250,000 tokens" in out
        assert "cannot be estimated" in out
        kinds = [a.kind for a in load_plan(ws).actions]
        assert ActionKind.LITERATURE_CORPUS_PASS in kinds
        assert kinds.index(ActionKind.LITERATURE_CORPUS_PASS) < kinds.index(ActionKind.T3_RUN)

    def test_the_dollar_cost_is_reported_as_an_estimate_beside_the_binding_token_cap(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Tokens are the authorisation unit; dollars are derived and REPORTED.

        The operator still needs to know roughly what a token number costs, or the
        cap is unusable in practice -- but the moment a dollar figure is printed it
        invites being read as the ceiling. That is precisely the confusion that cost a
        live run 87% of its authorised budget: it stopped on a token cap nobody had
        looked at while the dollar figure sat there looking authoritative.

        So the output must carry BOTH, and must say which one binds. Asserting on the
        disambiguating phrase, not merely on the number, is the point of this test.
        """
        from Carmel import main

        cid, _ = self._campaign(tmp_path)
        config_path = tmp_path / "agents.yaml"
        config_path.write_text(
            (tmp_path / "nh3.yaml").read_text(encoding="utf-8")
            + "\nagents:\n  tier: test\n  external_provider_consent: false\n",
            encoding="utf-8",
        )

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "250000",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(config_path),
                "--dry-run",
            ]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "250,000 tokens" in out, "the binding cap must be stated in the unit that binds"
        assert "at most ~$" in out, "an operator cannot use a token cap without knowing its cost"
        assert "estimated ~$" not in out, (
            "the figure is a worst-case bound (unpriced models get a punitive fallback rate), "
            "so printing it as a point estimate would mislead in the same way a non-binding "
            "dollar ceiling did"
        )
        assert "the token cap is what binds" in out, (
            "a bare dollar figure beside a token cap reads as the ceiling; it must say which one binds"
        )

    def test_progress_gains_the_action_at_the_same_position_as_the_plan(self, tmp_path: Path) -> None:
        """Plan and progress are index-aligned; drifting them apart would make the
        cursor point at a different action than the plan says."""
        from Carmel import main
        from carmel.services.plan_progress import load_progress
        from carmel.services.planner import load_plan

        cid, ws = self._campaign(tmp_path)
        assert (
            main(
                [
                    "corpus-pass",
                    "--campaign",
                    cid,
                    "--budget-tokens",
                    "100000",
                    "--workspaces",
                    str(tmp_path),
                    "--dry-run",
                ]
            )
            == 0
        )

        plan_ids = [a.action_id for a in load_plan(ws).actions]
        progress_ids = [a.action_id for a in load_progress(ws).actions]
        assert plan_ids == progress_ids

    def test_a_non_positive_budget_is_refused(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The budget is the authorisation. A zero or negative one authorises
        nothing, and silently appending an action that can never run would be worse
        than refusing."""
        from Carmel import main
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import load_plan

        cid, ws = self._campaign(tmp_path)

        code = main(
            ["corpus-pass", "--campaign", cid, "--budget-tokens", "0", "--workspaces", str(tmp_path), "--dry-run"]
        )

        assert code == 1
        assert "must be positive" in capsys.readouterr().err
        assert ActionKind.LITERATURE_CORPUS_PASS not in [a.kind for a in load_plan(ws).actions]

    def test_an_unknown_campaign_is_reported_not_crashed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from Carmel import main

        self._campaign(tmp_path)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                "no-such-id",
                "--budget-tokens",
                "100000",
                "--workspaces",
                str(tmp_path),
                "--dry-run",
            ]
        )

        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_it_refuses_rather_than_dispatching_a_different_action(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`corpus-pass` dispatches the plan's NEXT runnable action, which is not
        necessarily the corpus pass it just appended.

        With a LITERATURE_SEARCH still pending ahead of it, the command used to run
        THAT -- an outward-facing pass that reaches the network and spends money -- and
        only report the mismatch afterwards, once it had already happened. The operator
        asked for an offline pass over papers already held, under a budget named for
        that pass. Reporting after the fact is not a guard.

        Asserting the handler was never built is the load-bearing part: an assertion on
        the message alone would pass even if the search had run.
        """
        import carmel.services.dispatcher as dispatcher
        from Carmel import main

        cid, _ = self._campaign(tmp_path, dispatchable=True)
        config_path = tmp_path / "agents.yaml"
        config_path.write_text(
            (tmp_path / "nh3.yaml").read_text(encoding="utf-8")
            + "\nagents:\n  tier: test\n  external_provider_consent: false\n",
            encoding="utf-8",
        )
        called = False

        def _spy(**kwargs: Any) -> Any:
            nonlocal called
            called = True
            return dispatcher.default_handlers(**kwargs)

        monkeypatch.setattr(dispatcher, "default_handlers", _spy)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "100000",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(config_path),
            ]
        )

        assert code == 1
        assert called is False, "an action other than the corpus pass was dispatched"
        assert "Refusing to dispatch" in capsys.readouterr().err

    def test_the_agent_config_reaches_the_handler(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: the command loaded --config and then dispatched without it, so
        the literature handler was built with nothing to run on and returned a typed
        "no agent config available" failure. That reads as a failed RUN rather than a
        command that never started one, and no test using injected deps could see it
        -- only the real dispatch path does.
        """
        import carmel.services.dispatcher as dispatcher
        from Carmel import main
        from carmel.schemas import ActionExecutionStatus, ActionOutcome
        from carmel.services.plan_progress import advance_cursor, load_progress, mark_finished, mark_running

        cid, ws = self._campaign(tmp_path, dispatchable=True)
        # Retire the literature search sitting ahead of the corpus pass. The command
        # now refuses to dispatch when the cursor points at a different action, so
        # without this it never reaches the handler at all.
        search = load_progress(ws).actions[0]
        mark_running(ws, search.action_id, "attempt-1")
        mark_finished(
            ws,
            search.action_id,
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )
        advance_cursor(ws, search.action_id)
        config_path = tmp_path / "agents.yaml"
        config_path.write_text(
            (tmp_path / "nh3.yaml").read_text(encoding="utf-8")
            + "\nagents:\n  tier: test\n  external_provider_consent: false\n",
            encoding="utf-8",
        )

        seen: dict[str, object] = {}
        real_default_handlers = dispatcher.default_handlers

        def _spy(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return real_default_handlers(**kwargs)

        monkeypatch.setattr(dispatcher, "default_handlers", _spy)

        main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "100000",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(config_path),
            ]
        )

        assert "agent_config" in seen, "the command dispatched without handing over a handler registry"
        assert seen["agent_config"] is not None, "the loaded --config never reached the handler"

    def test_the_run_advances_the_plan_cursor_past_the_corpus_action(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spar round 7, P1. The command used to call execute_action, the by-kind
        router, which runs the handler and NOTHING else -- no approval gate, no state
        transitions, no attempt record, and no cursor advance.

        A corpus pass therefore completed and wrote its report while plan_progress.json
        still showed the action pending, so the next dispatcher run would execute it a
        second time. Findings are deliberately never deduped across passes, so that
        silently doubles them in the accumulated report.

        Asserting on the CURSOR rather than on the outcome is deliberate: the run
        itself fails here (consent is off, so there is no model to call), and the
        bookkeeping must be correct regardless of how the run turns out. An assertion
        that only held for a successful run would not have caught this.
        """
        from Carmel import main
        from carmel.schemas import ActionExecutionStatus, ActionKind, ActionOutcome
        from carmel.services.plan_progress import (
            advance_cursor,
            load_progress,
            mark_finished,
            mark_running,
        )

        cid, ws = self._campaign(tmp_path, dispatchable=True)
        # Widen the auto-approve threshold. Since the corpus pass started going through
        # the approval gate, the default $2 would hold this action for review: the mock
        # model has no pricing entry and is charged a punitive fallback rate, so the
        # worst-case estimate for this token cap lands far above it. That is the gate
        # working; this test is about the cursor, so let the action through.
        from carmel.schemas.approval import ApprovalPolicy
        from carmel.services.approvals import save_policy

        save_policy(ws, ApprovalPolicy(auto_approve_literature_under_usd=10_000.0))
        # Retire the literature search that precedes the corpus pass in the plan, so
        # the cursor actually reaches the appended action. Without this the dispatcher
        # correctly runs the SEARCH instead -- which the command reports, and which is
        # covered by test_it_refuses_rather_than_dispatching_a_different_action above.
        search = load_progress(ws).actions[0]
        mark_running(ws, search.action_id, "attempt-1")
        mark_finished(
            ws,
            search.action_id,
            status=ActionExecutionStatus.SUCCEEDED,
            outcome=ActionOutcome.SUCCEEDED,
        )
        advance_cursor(ws, search.action_id)
        config_path = tmp_path / "agents.yaml"
        config_path.write_text(
            (tmp_path / "nh3.yaml").read_text(encoding="utf-8")
            + "\nagents:\n  tier: test\n  external_provider_consent: false\n",
            encoding="utf-8",
        )

        main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "100000",
                "--workspaces",
                str(tmp_path),
                "--config",
                str(config_path),
            ]
        )

        progress = load_progress(ws)
        corpus = next(a for a in progress.actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS)
        assert corpus.execution_status != ActionExecutionStatus.PENDING, (
            "the corpus pass ran but progress still shows it PENDING, so the next "
            "dispatcher run would execute it a second time and double the findings"
        )
        assert corpus.attempt_ids, "the dispatcher recorded no attempt for the corpus pass"


class TestDispatchingAnAlreadyQueuedCorpusPass:
    """The escape from a dead end where two correct guards contradicted each other.

    A corpus pass that requires approval is appended by one command and can only
    run once a human approves it. But the dispatcher's own advice was "approve it
    and dispatch again", while re-running the plain command correctly refused --
    it would have queued a second identical pass. There was no third command, so
    an approval-gated corpus pass could never be run at all (found by live run
    2026.08.01, invisible to the suite because nothing re-ran the command after a
    refusal). ``--dispatch-queued`` is that third command.
    """

    def test_the_refusal_names_the_command_that_actually_dispatches(self, tmp_path: Path) -> None:
        """A refusal that does not name the way forward is the dead end itself."""
        from carmel.services.planner import append_corpus_pass_action

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)
        append_corpus_pass_action(ws, budget_tokens=100_000, model_name="m")

        with pytest.raises(ValueError) as excinfo:
            append_corpus_pass_action(ws, budget_tokens=100_000, model_name="m")

        assert "--dispatch-queued" in str(excinfo.value)
        assert cid  # the campaign id is what the operator passes to that command

    def test_dispatching_when_nothing_is_queued_is_a_typed_refusal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """'I ran nothing' must never be indistinguishable from 'I ran your pass'."""
        from Carmel import main

        cid, _ = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(["corpus-pass", "--campaign", cid, "--workspaces", str(tmp_path), "--dispatch-queued", "--dry-run"])

        assert code == 1
        assert "No corpus pass is queued" in capsys.readouterr().err

    def test_a_budget_may_not_be_renamed_at_dispatch_time(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The approver agreed to a specific cap; running under another defeats that."""
        from Carmel import main

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--dispatch-queued",
                "--budget-tokens",
                "999999",
            ]
        )

        assert code == 1
        assert "cannot be combined" in capsys.readouterr().err
        assert ws  # nothing was appended under the rejected budget

    def test_a_plain_run_still_requires_an_explicit_budget(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Making --budget-tokens optional for --dispatch-queued must not make it
        optional for the append path, where it is the operator's authorisation."""
        from Carmel import main

        cid, _ = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(["corpus-pass", "--campaign", cid, "--workspaces", str(tmp_path), "--dry-run"])

        assert code == 1
        assert "--budget-tokens is required" in capsys.readouterr().err

    def test_allow_unauthenticated_legacy_roots_is_recorded_on_the_queued_action(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """C1. The flag must actually reach the queued action's parameters, not
        merely be accepted and dropped -- that is the operator's authorisation for
        the next dispatch, recorded the same way ``reread_all`` is."""
        from Carmel import main
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import load_plan

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "250000",
                "--workspaces",
                str(tmp_path),
                "--allow-unauthenticated-legacy-roots",
                "--dry-run",
            ]
        )

        assert code == 0
        capsys.readouterr()
        actions = load_plan(ws).actions
        corpus_action = next(a for a in actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS)
        assert corpus_action.parameters.get("allow_unauthenticated_legacy_roots") is True

    def test_the_key_is_absent_without_the_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """C2. Absent, not present-and-False -- matching ``reread_all``'s own
        convention, so a downstream ``.get(..., False)`` reads the fail-closed
        default rather than an explicit but redundant False."""
        from Carmel import main
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import load_plan

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(
            ["corpus-pass", "--campaign", cid, "--budget-tokens", "250000", "--workspaces", str(tmp_path), "--dry-run"]
        )

        assert code == 0
        capsys.readouterr()
        actions = load_plan(ws).actions
        corpus_action = next(a for a in actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS)
        assert "allow_unauthenticated_legacy_roots" not in corpus_action.parameters

    def test_the_planner_grants_no_permission_a_caller_did_not_ask_for(self, tmp_path: Path) -> None:
        """C7. The DEFAULTS, pinned at the API boundary rather than through the CLI.

        Every CLI path passes both flags explicitly, so a mutation flipping either
        default to True survives the whole suite -- found by a raise-guard audit, not
        by review. The default is what a programmatic caller that never considered the
        question gets, and for a permission that must be the refusal: a caller who did
        not ask to read unauthenticated text has not authorised reading it.
        """
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import append_corpus_pass_action, load_plan

        _cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        append_corpus_pass_action(ws, budget_tokens=100_000, model_name="m")

        action = next(a for a in load_plan(ws).actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS)
        assert action.parameters == {}, (
            "a call that named neither flag must grant neither permission -- an empty "
            "parameters dict is the only shape that says 'nothing was authorised here'"
        )

    def test_allow_unauthenticated_legacy_roots_composes_with_reread_all(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """C3. The composition trap: two independent boolean flags, each recorded
        under its own key, must not let one clobber the other's dict entry."""
        from Carmel import main
        from carmel.schemas.approval import ActionKind
        from carmel.services.planner import load_plan

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--budget-tokens",
                "250000",
                "--workspaces",
                str(tmp_path),
                "--allow-unauthenticated-legacy-roots",
                "--reread-all",
                "--dry-run",
            ]
        )

        assert code == 0
        capsys.readouterr()
        actions = load_plan(ws).actions
        corpus_action = next(a for a in actions if a.kind == ActionKind.LITERATURE_CORPUS_PASS)
        assert corpus_action.parameters.get("allow_unauthenticated_legacy_roots") is True
        assert corpus_action.parameters.get("reread_all") is True

    def test_allow_unauthenticated_legacy_roots_may_not_be_combined_with_dispatch_queued(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """C4. The queued pass already carries the parameters it was approved
        under; bolting a fresh permission on at dispatch time would defeat the
        approval -- exactly the reasoning that already refuses --budget-tokens
        there."""
        from Carmel import main

        cid, ws = TestCorpusPassCommand()._campaign(tmp_path)

        code = main(
            [
                "corpus-pass",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--dispatch-queued",
                "--allow-unauthenticated-legacy-roots",
            ]
        )

        assert code == 1
        assert "cannot be combined" in capsys.readouterr().err
        assert ws  # nothing was appended under the rejected authorisation


class TestReextractCommand:
    """The operator-facing verb for re-parsing a stored artifact's raw.bin and
    appending a new, separately-addressed extraction record.

    Dry run is the default (opposite polarity from ``corpus-pass --dry-run``): a
    plain ``carmel reextract --sha ...`` must never write, only ``--apply`` does.
    """

    def _campaign(self, tmp_path: Path) -> tuple[str, Path]:
        from Carmel import main
        from carmel.services.campaigns import load_campaign

        assert main(["new-campaign", "--config", str(_write_campaign_config(tmp_path))]) == 0
        ws = tmp_path / "ws"
        return load_campaign(ws).campaign_id, ws

    def _config(self, tmp_path: Path) -> Path:
        # Any config with an 'agents' section works: --config is only there to supply
        # budget.max_artifact_bytes, no default is invented in Carmel.py itself.
        return _write_campaign_config(tmp_path)

    def test_neither_sha_nor_all_is_a_usage_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cid, _ = self._campaign(tmp_path)
        config = str(self._config(tmp_path))

        code = main(["reextract", "--campaign", cid, "--workspaces", str(tmp_path), "--config", config])

        assert code == 1
        assert "--sha <raw_sha256> or --all is required" in capsys.readouterr().err

    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tests.test_reextraction import _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Dry run source text"))

        from carmel.services.extraction_record import list_extraction_records

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--sha",
                raw_sha256,
            ]
        )

        assert code == 0
        assert list_extraction_records(ws, raw_sha256=raw_sha256) == []

    def test_dry_run_reports_the_would_be_sha(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from carmel.services.reextraction import preview_reextraction
        from tests.test_reextraction import MAX_BYTES, _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Reported source text"))
        expected_sha256, _ = preview_reextraction(ws, raw_sha256=raw_sha256, max_bytes=MAX_BYTES)

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--sha",
                raw_sha256,
            ]
        )

        assert code == 0
        assert expected_sha256 in capsys.readouterr().out

    def test_apply_writes_a_new_extraction_record(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from carmel.services.extraction_record import list_extraction_records
        from tests.test_reextraction import _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Apply source text"))

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--sha",
                raw_sha256,
                "--apply",
            ]
        )

        assert code == 0
        assert len(list_extraction_records(ws, raw_sha256=raw_sha256)) == 1
        assert "WRITTEN" in capsys.readouterr().out

    def test_apply_twice_is_a_no_op(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from carmel.services.extraction_record import list_extraction_records
        from tests.test_reextraction import _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Idempotent apply source text"))
        config = str(self._config(tmp_path))

        args = [
            "reextract",
            "--campaign",
            cid,
            "--workspaces",
            str(tmp_path),
            "--config",
            config,
            "--sha",
            raw_sha256,
            "--apply",
        ]
        assert main(args) == 0
        capsys.readouterr()
        assert main(args) == 0
        out = capsys.readouterr().out

        assert len(list_extraction_records(ws, raw_sha256=raw_sha256)) == 1
        assert "ALREADY-PRESENT" in out

    def test_all_reports_the_bad_artifact_and_still_processes_the_good_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from carmel.services.extraction_record import list_extraction_records
        from tests.test_reextraction import _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        good_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Good artifact source text"))
        bad_sha256 = _store_synthetic_artifact(ws, b"not a pdf at all, no header")

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--all",
                "--apply",
            ]
        )

        out = capsys.readouterr().out
        assert code == 1
        assert f"REFUSED         {bad_sha256}" in out
        assert "does not sniff as a PDF" in out
        assert f"WRITTEN         {good_sha256}" in out
        assert len(list_extraction_records(ws, raw_sha256=good_sha256)) == 1

    def test_apply_refuses_a_bogus_record_directory_instead_of_reporting_already_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """"Already present" must mean an AUTHENTICATED record, never merely a
        directory existing at the computed address. A directory occupying that
        address without authenticating to it (here: empty) is a distinct, fatal
        collision -- ``--apply`` must neither report it as ALREADY-PRESENT (a
        false success) nor silently exit 0; it must surface an explicit refusal,
        same as any other ``REFUSED`` case."""
        from carmel.services.extraction_record import extraction_record_dir
        from carmel.services.reextraction import preview_reextraction
        from tests.test_reextraction import MAX_BYTES, _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Bogus record dir CLI source text"))
        extraction_sha256, _ = preview_reextraction(ws, raw_sha256=raw_sha256, max_bytes=MAX_BYTES)
        bogus_dir = extraction_record_dir(ws, raw_sha256, extraction_sha256)
        bogus_dir.mkdir(parents=True)  # exists, but authenticates to nothing: empty.

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--sha",
                raw_sha256,
                "--apply",
            ]
        )

        out = capsys.readouterr().out
        assert code == 1
        assert "ALREADY-PRESENT" not in out
        assert f"REFUSED         {raw_sha256}" in out
        assert "does not authenticate as a stored extraction record" in out

    def test_no_consumer_notice_appears_in_run_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tests.test_reextraction import _build_tiny_pdf, _store_synthetic_artifact

        cid, ws = self._campaign(tmp_path)
        raw_sha256 = _store_synthetic_artifact(ws, _build_tiny_pdf(b"Notice source text"))

        code = main(
            [
                "reextract",
                "--campaign",
                cid,
                "--workspaces",
                str(tmp_path),
                "--config",
                str(self._config(tmp_path)),
                "--sha",
                raw_sha256,
            ]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "no consumer reads extraction records yet" in out.lower()

    def test_no_consumer_notice_appears_in_help_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["reextract", "--help"])
        out = capsys.readouterr().out
        # argparse line-wraps the description to the terminal width, so compare on
        # whitespace-normalized text rather than requiring the phrase on one line.
        normalized = " ".join(out.lower().split())
        assert "no consumer reads extraction records yet" in normalized
