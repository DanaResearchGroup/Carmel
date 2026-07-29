"""Tests for Carmel CLI."""

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from Carmel import main
from carmel.paths import WORKSPACE_SUBDIRS
from carmel.version import __version__


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

    def test_listing_prints_the_next_command_for_each_pending_paper(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cid, _ = self._campaign_with_request(tmp_path)

        from Carmel import main

        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "awaiting a human" in out
        assert "--add" in out and "--slug" in out

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
            "Shock tube study of ammonia oxidation ignition delay times\n"
            "DOI: 10.1016/j.test.2019.01.001\nAbstract: we report ignition delay times.\n",
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
            "Shock tube study of ammonia oxidation ignition delay times\nDOI: 10.1016/j.test.2019.01.001\n",
            encoding="utf-8",
        )

        from Carmel import main

        main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(wrong)])
        capsys.readouterr()
        assert main(["requests", "--campaign", cid, "--workspaces", str(tmp_path), "--add", str(right)]) == 0
        assert "ACCEPTED" in capsys.readouterr().out
