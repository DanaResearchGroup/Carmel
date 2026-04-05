"""Tests for Carmel CLI."""

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


class TestNoCommand:
    """Tests for CLI with no command or unknown input."""

    def test_no_args_returns_one(self) -> None:
        assert main([]) == 1

    def test_no_args_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        captured = capsys.readouterr()
        assert "carmel" in captured.out.lower()
