"""Tests for Carmel CLI."""

import sys
from pathlib import Path

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
