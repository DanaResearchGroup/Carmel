# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""A conda-selected T3/ARC env runs its own interpreter, never Carmel's venv ``python``.

These tests use a fake ``conda`` shell script that reproduces the real
``conda run`` behaviour that matters here: it sets ``CONDA_PREFIX`` to the
named env and execs the command with ``PATH`` unchanged. A fake venv
``python`` sits first on ``PATH`` (with ``VIRTUAL_ENV`` set), exactly as
under ``uv run``, and leaves a marker file if it is ever executed.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from carmel.adapters import _launcher
from carmel.adapters import arc as arc_module
from carmel.adapters import t3 as t3_module
from carmel.schemas import FailureCode, RunStatus
from tests import test_arc_adapter, test_t3_adapter

_TOOL_ENV_VARS = ("T3_CONDA_ENV", "T3_PYTHON", "T3_PATH", "ARC_CONDA_ENV", "ARC_PYTHON", "ARC_PATH")


@dataclass(frozen=True)
class _Tool:
    """How one adapter is driven: its conda env var and its public entry points."""

    name: str
    conda_env_var: str
    python_command: Callable[[], list[str]]
    conda_env_error: Callable[[], str | None]
    is_importable: Callable[[], bool]
    version: Callable[[], str | None]
    find_executable: Callable[[], list[str] | None]
    run: Callable[[Path], Any]


def _run_t3(ws: Path) -> Any:
    """Run the T3 adapter once in workspace *ws*."""
    return t3_module.T3Adapter().run(
        workspace_root=ws, campaign=test_t3_adapter._campaign(ws), action=test_t3_adapter._action()
    )


def _run_arc(ws: Path) -> Any:
    """Run the ARC adapter once in workspace *ws*."""
    return arc_module.ARCAdapter().run(
        workspace_root=ws, campaign=test_arc_adapter._campaign(ws), action=test_arc_adapter._action()
    )


_TOOLS = [
    _Tool(
        "t3",
        "T3_CONDA_ENV",
        t3_module._t3_python_command,
        t3_module._t3_conda_env_error,
        t3_module.is_t3_importable,
        t3_module._t3_version,
        t3_module._find_t3_executable,
        _run_t3,
    ),
    _Tool(
        "arc",
        "ARC_CONDA_ENV",
        arc_module._arc_python_command,
        arc_module._arc_conda_env_error,
        arc_module.is_arc_importable,
        arc_module._arc_version,
        arc_module._find_arc_executable,
        _run_arc,
    ),
]


def _write_script(path: Path, body: str) -> None:
    """Write an executable ``/bin/sh`` script at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@dataclass(frozen=True)
class _Layout:
    """Paths of the fake conda install, the fake venv, and the run markers."""

    envs_root: Path
    venv_marker: Path
    env_marker: Path

    def make_env(self, name: str, *, with_python: bool = True) -> Path:
        """Create conda env *name*; give it a marker-writing ``bin/python`` unless told not to."""
        prefix = self.envs_root / name
        prefix.mkdir(parents=True)
        if with_python:
            _write_script(prefix / "bin" / "python", f'touch "{self.env_marker}"\nexit 0\n')
        return prefix


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Layout:
    """Put a fake venv ``python`` first on PATH and a fake ``conda`` after it."""
    for var in _TOOL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    envs_root = tmp_path / "conda" / "envs"
    envs_root.mkdir(parents=True)
    venv_marker = tmp_path / "venv-ran"
    env_marker = tmp_path / "env-ran"
    venv_bin = tmp_path / "venv" / "bin"
    _write_script(venv_bin / "python", f'touch "{venv_marker}"\nexit 1\n')
    conda_bin = tmp_path / "conda" / "bin"
    _write_script(
        conda_bin / "conda",
        '[ "$1" = run ] && [ "$2" = -n ] && [ "$4" = --no-capture-output ] || exit 2\n'
        f'prefix="{envs_root}/$3"\n'
        "shift 4\n"
        'if [ ! -d "$prefix" ]; then\n'
        '  echo "EnvironmentLocationNotFound: Not a conda environment: $prefix" >&2\n'
        "  exit 1\n"
        "fi\n"
        'echo "activate.d hook noise"\n'
        'CONDA_PREFIX="$prefix" "$@"\n'
        "rc=$?\n"
        'echo "deactivate hook noise"\n'
        "exit $rc\n",
    )
    monkeypatch.setenv("PATH", f"{venv_bin}{os.pathsep}{conda_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_bin.parent))
    return _Layout(envs_root=envs_root, venv_marker=venv_marker, env_marker=env_marker)


@pytest.mark.parametrize("tool", _TOOLS, ids=lambda t: t.name)
class TestCondaEnvRunsItsOwnInterpreter:
    """The env's ``bin/python`` runs; the venv ``python`` first on PATH never does."""

    def test_import_probe_runs_env_interpreter(
        self, tool: _Tool, layout: _Layout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prefix = layout.make_env("tool_env")
        monkeypatch.setenv(tool.conda_env_var, "tool_env")
        assert tool.python_command()[-1] == str(prefix / "bin" / "python")
        assert tool.conda_env_error() is None
        assert tool.is_importable() is True
        assert layout.env_marker.exists()
        assert not layout.venv_marker.exists()

    def test_missing_env_refuses(
        self, tool: _Tool, layout: _Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tool.conda_env_var, "no_such_env")
        error = tool.conda_env_error()
        assert error is not None
        assert tool.conda_env_var in error
        assert "no_such_env" in error
        assert "EnvironmentLocationNotFound" in error
        with pytest.raises(_launcher.ToolEnvironmentError):
            tool.python_command()
        assert tool.is_importable() is False
        assert tool.version() is None
        assert tool.find_executable() is None
        ws = tmp_path / "ws"
        ws.mkdir()
        run, _diagnostics = tool.run(ws)
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert not layout.venv_marker.exists()

    def test_env_without_interpreter_refuses(
        self, tool: _Tool, layout: _Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layout.make_env("hollow_env", with_python=False)
        monkeypatch.setenv(tool.conda_env_var, "hollow_env")
        error = tool.conda_env_error()
        assert error is not None
        assert "no executable bin/python" in error
        assert tool.is_importable() is False
        ws = tmp_path / "ws"
        ws.mkdir()
        run, _diagnostics = tool.run(ws)
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert not layout.venv_marker.exists()


def _runner_returning(stdout: str = "", returncode: int = 0) -> Any:
    """Build a stub runner that answers every call with one fixed result."""

    def _run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=returncode, stdout=stdout, stderr="")

    return _run


class TestResolveCondaInterpreter:
    """Output that is not one absolute prefix holding ``bin/python`` is refused."""

    def _resolve(self, runner: Any) -> str:
        return _launcher.resolve_conda_interpreter(
            "/opt/conda/bin/conda", "env", env_var="T3_CONDA_ENV", runner=runner, timeout=1.0
        )

    def test_asks_conda_for_the_prefix(self, tmp_path: Path) -> None:
        _write_script(tmp_path / "bin" / "python", "exit 0\n")
        calls: list[list[str]] = []

        def _run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            stdout = f"CARMEL_CONDA_PREFIX={tmp_path}\n"
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

        assert self._resolve(_run) == str(tmp_path / "bin" / "python")
        assert calls == [
            ["/opt/conda/bin/conda", "run", "-n", "env", "--no-capture-output", "sh", "-c", _launcher._PREFIX_COMMAND]
        ]

    def test_prefix_command_prints_the_marker_line(self, tmp_path: Path) -> None:
        completed = subprocess.run(
            ["sh", "-c", _launcher._PREFIX_COMMAND],
            env={**os.environ, "CONDA_PREFIX": str(tmp_path)},
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout == f"CARMEL_CONDA_PREFIX={tmp_path}\n"

    def test_hook_noise_around_the_marker_is_ignored(self, tmp_path: Path) -> None:
        _write_script(tmp_path / "bin" / "python", "exit 0\n")
        stdout = f"hook says hi\n/not/a/prefix\nCARMEL_CONDA_PREFIX={tmp_path}\nanother hook line\n"
        assert self._resolve(_runner_returning(stdout)) == str(tmp_path / "bin" / "python")

    @pytest.mark.parametrize(
        "stdout",
        [
            "",
            "/abs/prefix/without/marker\n",
            "CARMEL_CONDA_PREFIX=relative/prefix\n",
            "CARMEL_CONDA_PREFIX=/a\nCARMEL_CONDA_PREFIX=/b\n",
            "CARMEL_CONDA_PREFIX=\n",
        ],
        ids=["empty", "no-marker", "relative", "two-markers", "empty-prefix"],
    )
    def test_malformed_output_refuses(self, stdout: str) -> None:
        with pytest.raises(_launcher.ToolEnvironmentError, match="single absolute prefix"):
            self._resolve(_runner_returning(stdout))

    def test_nonzero_exit_refuses(self) -> None:
        with pytest.raises(_launcher.ToolEnvironmentError, match="exit 1"):
            self._resolve(_runner_returning(returncode=1))

    @pytest.mark.parametrize("exc", [OSError("boom"), subprocess.TimeoutExpired(["conda"], 1.0)])
    def test_runner_failure_refuses(self, exc: Exception) -> None:
        def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise exc

        with pytest.raises(_launcher.ToolEnvironmentError, match="could not run it"):
            self._resolve(_raise)


class TestToolChildEnvironment:
    """The T3/ARC child environment carries no trace of Carmel's own venv."""

    def test_strips_venv_bin_and_virtual_env(self, tmp_path: Path) -> None:
        venv = tmp_path / "venv"
        _write_script(venv / "bin" / "python", "exit 0\n")
        conda_bin = tmp_path / "conda" / "bin"
        _write_script(conda_bin / "python", "exit 0\n")
        other = tmp_path / "other"
        path = os.pathsep.join([str(venv / "bin"), str(other), str(conda_bin), "/usr/bin"])
        environ = {"PATH": path, "VIRTUAL_ENV": str(venv), "CONDA_PREFIX": "/c", "CONDA_SHLVL": "1"}
        env = _launcher.tool_child_environment(environ, prefix=str(venv), base_prefix="/usr")
        assert "VIRTUAL_ENV" not in env
        assert env["PATH"].split(os.pathsep) == [str(other), str(conda_bin), "/usr/bin"]
        assert env["CONDA_PREFIX"] == "/c"
        assert env["CONDA_SHLVL"] == "1"
        assert shutil.which("python", path=env["PATH"]) == str(conda_bin / "python")
        assert environ["PATH"] == path

    def test_running_venv_stripped_without_virtual_env(self) -> None:
        environ = {"PATH": os.pathsep.join(["/p/venv/bin", "/usr/bin"])}
        env = _launcher.tool_child_environment(environ, prefix="/p/venv", base_prefix="/usr")
        assert env["PATH"] == "/usr/bin"

    def test_virtual_env_stripped_when_running_outside_it(self) -> None:
        environ = {"PATH": os.pathsep.join(["/v/bin", "/usr/bin"]), "VIRTUAL_ENV": "/v"}
        env = _launcher.tool_child_environment(environ, prefix="/usr", base_prefix="/usr")
        assert "VIRTUAL_ENV" not in env
        assert env["PATH"] == "/usr/bin"

    def test_no_venv_changes_nothing(self) -> None:
        environ = {"PATH": os.pathsep.join(["/opt/conda/envs/x/bin", "/usr/bin"]), "CONDA_PREFIX": "/opt/conda"}
        env = _launcher.tool_child_environment(environ, prefix="/opt/conda", base_prefix="/opt/conda")
        assert env == environ

    def test_defaults_read_the_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIRTUAL_ENV", "/nowhere/venv")
        env = _launcher.tool_child_environment()
        assert "VIRTUAL_ENV" not in env

    def test_launched_child_and_its_nested_lookup_miss_the_venv(
        self, layout: _Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A child started by the launcher, and a bare ``python`` it runs, never reach the venv."""
        conda_bin = tmp_path / "conda" / "bin"
        _write_script(conda_bin / "python", 'echo "VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"\n')
        completed = t3_module._run_in_process_group(
            ["sh", "-c", "python"], timeout=10.0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert completed.returncode == 0
        assert completed.stdout.strip() == "VIRTUAL_ENV=unset"
        assert not layout.venv_marker.exists()
