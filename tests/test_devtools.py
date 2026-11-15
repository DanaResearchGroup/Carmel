"""Tests for the devtools installers and the Makefile that drives them.

These guard the two properties that are easy to break by accident and
expensive to discover in CI: that the required lanes never invoke the heavy
chemistry-stack install, and that no installer decides what to do from a
cache-hit flag rather than from what is on disk.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS = REPO_ROOT / "devtools"
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TOOLS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tools.yml"
DOCS_WORKFLOWS = [REPO_ROOT / ".github" / "workflows" / name for name in ("docs.yml", "docs-deploy.yml")]
TOOLS_FILE = "carmel-tools" + ".env"

SCRIPTS = sorted(DEVTOOLS.glob("*.sh"))


def makefile_targets() -> set[str]:
    """Return the names of every target declared in the Makefile.

    Returns:
        Target names, e.g. ``{"install", "test", ...}``.
    """
    return {match.group(1) for match in re.finditer(r"^([a-zA-Z][\w-]*):", MAKEFILE.read_text(), flags=re.MULTILINE)}


ARGUMENT_TAKING_FLAGS = {"-C", "-f", "-j", "-I", "-o", "-W"}


def targets_in_command(command: str) -> set[str]:
    """Return the Makefile targets a shell command invokes.

    Handles the forms that a naive ``make (\\w+)`` regex misses: leading
    variable assignments, flags, flags that consume an argument, and several
    targets in one invocation.

    Args:
        command: A single shell command line.

    Returns:
        The set of target names invoked.
    """
    tokens = command.replace("&&", " ").replace("||", " ").replace(";", " ").replace("|", " ").split()
    targets: set[str] = set()
    index = 0
    while index < len(tokens):
        if tokens[index] != "make":
            index += 1
            continue
        index += 1
        while index < len(tokens):
            token = tokens[index]
            if token in ARGUMENT_TAKING_FLAGS:
                index += 2
            elif token.startswith("-") or "=" in token:
                index += 1
            elif token == "make":
                break  # a second invocation; let the outer loop take it
            elif re.fullmatch(r"[a-zA-Z][\w-]*", token):
                targets.add(token)
                index += 1
            else:
                break
    return targets


def make_targets_invoked_by(workflow: Path) -> set[str]:
    """Return the Makefile targets a workflow invokes.

    Every non-comment line is scanned, not just single-line ``run:`` steps —
    a ``run: |`` block is the obvious way to smuggle the wrong target past a
    guard that only reads one line.

    Args:
        workflow: Path to the workflow file.

    Returns:
        The set of target names invoked.
    """
    targets: set[str] = set()
    for line in workflow.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("run:"):
            stripped = stripped[len("run:") :]
        targets |= targets_in_command(stripped)
    return targets


class TestTargetsInCommand:
    """The parser the workflow guards depend on."""

    def test_empty_command(self) -> None:
        assert targets_in_command("") == set()

    def test_command_without_make(self) -> None:
        assert targets_in_command("conda activate t3_env") == set()

    def test_plain_invocation(self) -> None:
        assert targets_in_command("make install") == {"install"}

    def test_leading_variable_assignment(self) -> None:
        assert targets_in_command('CARMEL_STACK_ROOT="$HOME" make install') == {"install"}

    def test_flag_that_consumes_an_argument(self) -> None:
        assert targets_in_command("make -C . install-dev") == {"install-dev"}

    def test_several_targets_in_one_invocation(self) -> None:
        assert targets_in_command("make install-dev install") == {"install-dev", "install"}

    def test_chained_invocations(self) -> None:
        assert targets_in_command("make lint && make typecheck") == {"lint", "typecheck"}

    def test_stops_at_a_non_target_token(self) -> None:
        assert targets_in_command("make install > /dev/null") == {"install"}


class TestWorkflowScanning:
    """`run: |` blocks are the obvious way past a one-line guard."""

    def test_block_run_steps_are_scanned(self, tmp_path: Path) -> None:
        workflow = tmp_path / "block.yml"
        workflow.write_text(
            "jobs:\n  x:\n    steps:\n      - run: |\n          conda activate t3_env\n          make install\n"
        )
        assert make_targets_invoked_by(workflow) == {"install"}

    def test_commented_out_targets_are_ignored(self, tmp_path: Path) -> None:
        workflow = tmp_path / "commented.yml"
        workflow.write_text("jobs:\n  x:\n    steps:\n      # make install\n      - run: make install-dev\n")
        assert make_targets_invoked_by(workflow) == {"install-dev"}


class TestScriptsAreWellFormed:
    """The installers themselves."""

    def test_the_expected_installers_exist(self) -> None:
        assert {path.name for path in SCRIPTS} == {
            "common.sh",
            "install_all.sh",
            "install_carmel.sh",
            "install_rmg.sh",
            "install_stack.sh",
            "install_t3.sh",
        }

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_script_is_syntactically_valid(self, script: Path) -> None:
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script.name}: {result.stderr}"

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_script_carries_the_licence_header(self, script: Path) -> None:
        header = script.read_text()[:400]
        assert "Copyright 2026 Dana Research Group" in header
        assert "SPDX-License-Identifier: Apache-2.0" in header

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_script_sets_strict_mode(self, script: Path) -> None:
        """A half-finished install must fail loudly, not report success."""
        text = script.read_text()
        if script.name == "common.sh":
            pytest.skip("sourced, so it inherits the caller's shell options")
        assert "set -euo pipefail" in text


class TestStateNotFlags:
    """Installers must branch on what is on disk, never on a CI cache flag.

    `b14a712` exists because a workflow gated ARC's extension build on a
    cache-hit output while ARC's source tree was re-cloned every run — so on a
    warm cache the extensions were never built, `import arc` failed, every real
    test skipped and the lane reported green.
    """

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_no_installer_reads_a_cache_hit_flag(self, script: Path) -> None:
        text = script.read_text().lower()
        assert "cache-hit" not in text
        assert "cache_hit" not in text

    def test_the_tools_lane_gates_no_step_on_a_cache_hit(self) -> None:
        """Parsed as YAML, so a folded or multi-line `if:` cannot slip past."""
        workflow = yaml.safe_load(TOOLS_WORKFLOW.read_text())
        for job_name, job in workflow["jobs"].items():
            assert "cache-hit" not in str(job.get("if", "")), f"job {job_name} gated on a cache flag"
            for step in job["steps"]:
                condition = str(step.get("if", ""))
                assert "cache-hit" not in condition, (
                    f"{job_name} / {step.get('name', step.get('uses'))} gated on a cache flag: {condition}"
                )

    def test_the_tools_lane_never_reads_a_cache_hit_output(self) -> None:
        """A shell `if` inside `run:` is a step gate the YAML `if:` check misses."""
        assert "outputs.cache-hit" not in TOOLS_WORKFLOW.read_text()


def run_install_carmel(tmp_path: Path, *, with_t3: bool = True, **overrides: str) -> subprocess.CompletedProcess[str]:
    """Run ``install_carmel.sh`` against throwaway stack dirs with a fake ``uv`` on PATH.

    The fake records its arguments and exits 0, so the run exercises only the
    script's own logic: the sync call it makes and the tool-path file it writes.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(f'#!/usr/bin/env bash\necho "$*" >> {tmp_path / "uv-calls.log"}\n')
    fake_uv.chmod(0o755)
    stack = tmp_path / "stack"
    for name in ("RMG-Py", "RMG-database", "ARC") + (("T3",) if with_t3 else ()):
        (stack / name).mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CARMEL_STACK_ROOT": str(stack),
        "CARMEL_TOOLS_ENV_FILE": str(tmp_path / TOOLS_FILE),
        **overrides,
    }
    return subprocess.run(
        ["bash", str(DEVTOOLS / "install_carmel.sh")], env=env, capture_output=True, text=True, check=False
    )


class TestTheToolPathsFile:
    """``install_carmel.sh`` records the tool paths in ``carmel-tools.env``.

    Carmel's own environment is a uv ``.venv``, which has no activation hooks, so
    the file is the one place a user (``uv run --env-file``) and the tools lane
    (``source``) read the paths back from.
    """

    def test_the_installer_syncs_from_the_lock(self, tmp_path: Path) -> None:
        result = run_install_carmel(tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "uv-calls.log").read_text().split("\n")[0] == "sync --locked --extra dev"

    def test_every_tool_path_is_written_single_quoted(self, tmp_path: Path) -> None:
        result = run_install_carmel(tmp_path)
        assert result.returncode == 0, result.stderr
        text = (tmp_path / TOOLS_FILE).read_text()
        exports = dict(re.findall(r"^export (\w+)='([^']*)'$", text, flags=re.MULTILINE))
        assert set(exports) == {"T3_CONDA_ENV", "T3_PATH", "ARC_PATH", "RMG_PATH", "RMG_DB_PATH"}
        assert exports["T3_PATH"] == str(tmp_path / "stack" / "T3")
        assert exports["T3_CONDA_ENV"] == "t3_env"
        assert not (tmp_path / f"{TOOLS_FILE}.tmp").exists()

    def test_a_path_with_a_single_quote_is_refused_and_nothing_is_written(self, tmp_path: Path) -> None:
        quoted = tmp_path / "it's"
        quoted.mkdir()
        result = run_install_carmel(tmp_path, RMG_PATH=str(quoted))
        assert result.returncode != 0
        assert "single quote" in result.stderr
        assert not (tmp_path / TOOLS_FILE).exists()
        assert not (tmp_path / f"{TOOLS_FILE}.tmp").exists()

    def test_a_stale_file_is_removed_when_t3_is_gone(self, tmp_path: Path) -> None:
        (tmp_path / TOOLS_FILE).write_text("export T3_PATH='/gone'\n")
        result = run_install_carmel(tmp_path, with_t3=False)
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / TOOLS_FILE).exists()

    def test_the_installer_writes_atomically(self) -> None:
        script = (DEVTOOLS / "install_carmel.sh").read_text()
        assert 'mv -f "$CARMEL_TOOLS_ENV_FILE.tmp" "$CARMEL_TOOLS_ENV_FILE"' in script

    def test_the_file_is_gitignored(self) -> None:
        lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
        assert TOOLS_FILE in lines

    @pytest.fixture
    def written_file(self, tmp_path: Path) -> Path:
        """A real file from the installer, with a path holding a space and ``$(...)``."""
        odd = tmp_path / "a b $(touch pwned)"
        (odd / "T3").mkdir(parents=True)
        (odd / "RMG-database").mkdir()
        result = run_install_carmel(tmp_path, T3_PATH=str(odd / "T3"), RMG_DB_PATH=str(odd / "RMG-database"))
        assert result.returncode == 0, result.stderr
        return tmp_path / TOOLS_FILE

    def test_bash_source_reads_the_file_back_verbatim(self, written_file: Path, tmp_path: Path) -> None:
        result = subprocess.run(
            ["bash", "-c", 'set -a; source "$1"; set +a; printf "%s" "$T3_PATH"', "_", str(written_file)],
            env={"PATH": "/usr/bin:/bin"},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == str(tmp_path / "a b $(touch pwned)" / "T3")
        assert not (tmp_path / "pwned").exists()

    def test_uv_env_file_reads_the_file_back_verbatim(self, written_file: Path, tmp_path: Path) -> None:
        """Run in a clean environment: ``uv run --env-file`` never overrides a variable already set."""
        uv = shutil.which("uv")
        assert uv, "uv is not on PATH; Carmel's environment is managed by uv"
        result = subprocess.run(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "--env-file",
                str(written_file),
                "python",
                "-c",
                "import os; print(os.environ['T3_PATH'], end='')",
            ],
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == str(tmp_path / "a b $(touch pwned)" / "T3")


class TestMakefileWiring:
    """The Makefile is the interface both users and CI go through."""

    def test_every_script_the_makefile_calls_exists(self) -> None:
        called = re.findall(r"\$\(DEVTOOLS_DIR\)/([\w.]+)", MAKEFILE.read_text())
        assert called, "the Makefile no longer drives devtools/"
        for name in called:
            assert (DEVTOOLS / name).is_file(), f"Makefile calls missing script {name}"

    def test_default_goal_is_help(self) -> None:
        """A bare `make` must not start a 40-minute build."""
        assert ".DEFAULT_GOAL := help" in MAKEFILE.read_text()

    def test_help_lists_every_install_target(self) -> None:
        help_text = subprocess.run(
            ["make", "-s", "help"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        for target in ("install", "install-dev", "install-stack", "install-rmg", "install-t3", "install-carmel"):
            assert re.search(rf"^\s+{re.escape(target)}\s", help_text, flags=re.MULTILINE), (
                f"{target} is undocumented in `make help`"
            )


class TestWorkflowsUseTheMakefile:
    """CI must install the way the documentation tells users to."""

    def test_required_lanes_never_run_the_heavy_install(self) -> None:
        """`make install` builds the whole chemistry stack.

        The required lanes lint and test pure Python; pointing them at
        `install` turns a 20-second job into a 40-minute one on every PR.
        """
        invoked = make_targets_invoked_by(CI_WORKFLOW)
        assert "install" not in invoked
        assert "install-dev" in invoked

    def test_the_tools_lane_installs_through_make_install(self) -> None:
        assert "install" in make_targets_invoked_by(TOOLS_WORKFLOW)

    @pytest.mark.parametrize("workflow", [CI_WORKFLOW, TOOLS_WORKFLOW], ids=lambda p: p.name)
    def test_every_target_a_workflow_invokes_is_defined(self, workflow: Path) -> None:
        undefined = make_targets_invoked_by(workflow) - makefile_targets()
        assert not undefined, f"{workflow.name} calls undefined target(s): {sorted(undefined)}"


class TestTheLaneProvesSomething:
    """The tools lane must not be able to pass without running real T3.

    A skipped `@requires_t3` test is silent, so the lane asserts on its JUnit
    report. That assertion names a test class, which is a link that can rot —
    hence the second test here.
    """

    def test_the_lane_asserts_the_real_test_ran(self) -> None:
        workflow = yaml.safe_load(TOOLS_WORKFLOW.read_text())
        steps = workflow["jobs"]["tools"]["steps"]
        assert any("TestT3AdapterRealSubprocess" in str(step.get("run", "")) for step in steps), (
            "no step asserts that the real T3 execution test ran"
        )

    def test_the_test_class_the_lane_names_still_exists(self) -> None:
        suite = (REPO_ROOT / "tests" / "test_t3_adapter.py").read_text()
        assert "class TestT3AdapterRealSubprocess" in suite, (
            "the tools lane asserts on TestT3AdapterRealSubprocess; renaming it silently turns that gate into a no-op"
        )

    def test_the_lane_asserts_the_real_arc_test_ran(self) -> None:
        workflow = yaml.safe_load(TOOLS_WORKFLOW.read_text())
        steps = workflow["jobs"]["tools"]["steps"]
        assert any("TestARCAdapterRealSubprocess" in str(step.get("run", "")) for step in steps), (
            "no step asserts that the real ARC execution test ran"
        )

    def test_the_arc_test_class_the_lane_names_still_exists(self) -> None:
        suite = (REPO_ROOT / "tests" / "test_arc_adapter.py").read_text()
        assert "class TestARCAdapterRealSubprocess" in suite, (
            "the tools lane asserts on TestARCAdapterRealSubprocess; renaming it silently turns that gate into a no-op"
        )


class TestThePythonFloorIsDeclaredOnce:
    """Every declaration of the Python floor must agree with pyproject.toml.

    Six locations once disagreed across three minor versions, with
    `requires-python = ">=3.14"` making the package uninstallable on the
    3.12 the README advertised. Agreement is only stable if something
    checks it, so this reads the floor from the one source of truth and
    holds every restatement against it.
    """

    @staticmethod
    def floor() -> tuple[int, int]:
        """Return the (major, minor) floor from ``requires-python``."""
        text = (REPO_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^requires-python\s*=\s*"[><=~!]*\s*(\d+)\.(\d+)', text, flags=re.MULTILINE)
        assert match, "pyproject.toml declares no requires-python"
        return int(match.group(1)), int(match.group(2))

    def test_mypy_targets_the_declared_floor(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^python_version\s*=\s*"(\d+)\.(\d+)"', text, flags=re.MULTILINE)
        assert match, "the mypy config no longer pins python_version"
        assert (int(match.group(1)), int(match.group(2))) == self.floor()

    def test_ruff_does_not_restate_the_floor(self) -> None:
        """Ruff derives target-version from requires-python; restating it drifts."""
        text = (REPO_ROOT / "pyproject.toml").read_text()
        assert not re.search(r"^target-version\s*=", text, flags=re.MULTILINE), (
            "remove target-version: ruff reads requires-python, and a second copy of the floor drifts from it"
        )

    def test_the_lockfile_matches_the_floor(self) -> None:
        text = (REPO_ROOT / "uv.lock").read_text()
        match = re.search(r'^requires-python\s*=\s*">=\s*(\d+)\.(\d+)"', text, flags=re.MULTILINE)
        assert match, "uv.lock records no requires-python"
        assert (int(match.group(1)), int(match.group(2))) == self.floor()

    def test_the_conftest_interpreter_guard_matches_the_floor(self) -> None:
        """A stale guard admits the exact interpreter it exists to refuse.

        `tests/conftest.py` refuses to collect below the floor, because a bare `pytest`
        with no environment activated never consults `requires-python` at all. Raising
        the floor without raising the guard leaves that refusal silently one version
        behind, which reads as protection and is not.
        """
        text = (REPO_ROOT / "tests" / "conftest.py").read_text()
        match = re.search(r"^if sys\.version_info < \((\d+), (\d+)\):", text, flags=re.MULTILINE)
        assert match, "tests/conftest.py no longer guards the interpreter version"
        assert (int(match.group(1)), int(match.group(2))) == self.floor()

    @pytest.mark.parametrize("workflow", ["ci.yml", "docs.yml", "docs-deploy.yml"], ids=lambda name: name)
    def test_every_workflow_python_matches_the_floor(self, workflow: str) -> None:
        """A lane on an older Python tests something users cannot install."""
        major, minor = self.floor()
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        declared = re.findall(r'python-version:\s*\[?\s*"(\d+\.\d+)"', text)
        assert declared, f"{workflow} declares no python-version"
        for value in declared:
            assert value == f"{major}.{minor}", f"{workflow} runs on {value}, but the floor is {major}.{minor}"


REQUIRED_JOB_NAMES = {
    "lint": "Lint + format + typecheck (required)",
    "test": "Tests + packaging smoke (required)",
    "agents": "Agents extra — tests + typecheck (required)",
}


class TestCiInstallsThroughUv:
    """Carmel's lanes install from ``uv.lock`` through uv, never through pip.

    The required job names are literal status-check contexts in the repo's
    branch-protection ruleset, so they are pinned here too.
    """

    @pytest.mark.parametrize("workflow", [CI_WORKFLOW, *DOCS_WORKFLOWS], ids=lambda p: p.name)
    def test_every_python_job_sets_up_uv_pinned_by_sha(self, workflow: Path) -> None:
        jobs = yaml.safe_load(workflow.read_text())["jobs"]
        checked = 0
        for job_name, job in jobs.items():
            uses = [str(step.get("uses", "")) for step in job.get("steps", [])]
            runs = " ".join(str(step.get("run", "")) for step in job.get("steps", []))
            assert not any(u.startswith("actions/setup-python") for u in uses), f"{job_name} still sets up pip"
            assert "pip install" not in runs.replace("uv pip install", ""), f"{job_name} still installs through pip"
            if not any(word in runs for word in ("make ", "uv ", "mkdocs")):
                continue
            checked += 1
            setup = [u for u in uses if u.startswith("astral-sh/setup-uv@")]
            assert setup, f"{workflow.name} / {job_name} runs Python tooling without astral-sh/setup-uv"
            assert re.fullmatch(r"astral-sh/setup-uv@[0-9a-f]{40}", setup[0]), f"{setup[0]} is not pinned by SHA"
        assert checked, f"{workflow.name} has no job running Python tooling; this test checked nothing"

    @pytest.mark.parametrize("workflow", [CI_WORKFLOW, *DOCS_WORKFLOWS], ids=lambda p: p.name)
    def test_every_sync_is_locked(self, workflow: Path) -> None:
        syncs = re.findall(r"uv sync[^\n]*", workflow.read_text())
        assert syncs or workflow == CI_WORKFLOW, f"{workflow.name} never syncs"
        for line in syncs:
            assert "--locked" in line, f"{workflow.name}: {line!r} may rewrite the lock"

    def test_the_lint_job_fails_on_a_stale_lock(self) -> None:
        steps = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["lint"]["steps"]
        assert any(str(step.get("run", "")).strip() == "uv lock --check" for step in steps)

    def test_the_required_job_names_are_unchanged(self) -> None:
        jobs = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]
        assert {job_id: job["name"] for job_id, job in jobs.items()} == REQUIRED_JOB_NAMES

    def test_the_test_job_keeps_its_matrix_suffix(self) -> None:
        """The ruleset context is ``Tests + packaging smoke (required) (3.14)``."""
        matrix = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["test"]["strategy"]["matrix"]
        assert matrix == {"python-version": ["3.14"]}

    def test_the_packaging_smoke_builds_with_uv(self) -> None:
        text = CI_WORKFLOW.read_text()
        assert "uv build --wheel --out-dir dist" in text
        assert "dist/*.whl" in text

    def test_the_tools_lane_sets_up_uv_before_make_install(self) -> None:
        steps = yaml.safe_load(TOOLS_WORKFLOW.read_text())["jobs"]["tools"]["steps"]
        uses = [i for i, step in enumerate(steps) if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")]
        install = [i for i, step in enumerate(steps) if "make install" in str(step.get("run", ""))]
        assert uses and install and uses[0] < install[0]


class TestTheMakefileUsesUv:
    """Carmel's targets go through uv; the conda tool-env targets do not move."""

    @staticmethod
    def recipe(target: str) -> list[str]:
        """Return the recipe lines of one Makefile target, leading tab stripped."""
        pattern = rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)+)"
        match = re.search(pattern, MAKEFILE.read_text(), flags=re.MULTILINE)
        assert match, f"no recipe for {target}"
        return [line[1:] for line in match.group(1).splitlines()]

    def test_the_conda_tool_env_targets_are_unchanged(self) -> None:
        assert self.recipe("install-rmg") == ["bash $(DEVTOOLS_DIR)/install_rmg.sh"]
        assert self.recipe("install-t3") == ["bash $(DEVTOOLS_DIR)/install_t3.sh"]

    def test_install_targets_sync_from_the_lock(self) -> None:
        assert self.recipe("install-dev") == ["uv sync --locked --extra dev"]
        assert self.recipe("install-agents-dev") == ["uv sync --locked --extra dev --extra agents"]

    def test_uv_run_syncs_the_dev_extra(self) -> None:
        match = re.search(r"^UV_RUN := (.*)$", MAKEFILE.read_text(), flags=re.MULTILINE)
        assert match, "no UV_RUN in the Makefile"
        assert match.group(1).split() == ["uv", "run", "--locked", "--extra", "dev"]

    @pytest.mark.parametrize("target", ["test", "lint", "typecheck", "format"])
    def test_tool_targets_run_inside_the_uv_env(self, target: str) -> None:
        commands = [line for line in self.recipe(target) if not line.startswith("\t")]
        assert commands
        for line in commands:
            assert line.lstrip("@").startswith("$(UV_RUN) "), f"{target}: {line!r} runs outside the uv env"
