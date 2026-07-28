"""Tests for the devtools installers and the Makefile that drives them.

These guard the two properties that are easy to break by accident and
expensive to discover in CI: that the required lanes never invoke the heavy
chemistry-stack install, and that no installer decides what to do from a
cache-hit flag rather than from what is on disk.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS = REPO_ROOT / "devtools"
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TOOLS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tools.yml"

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
        assert targets_in_command("conda activate crml_env") == set()

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
            "jobs:\n  x:\n    steps:\n      - run: |\n          conda activate crml_env\n          make install\n"
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


class TestTheActivationHook:
    """`conda run` deactivates the current environment before activating the target.

    So a `deactivate.d` hook unsetting the tool paths strips them from T3's
    environment at the moment Carmel launches T3 — which is what T3 needs them
    for. It cost a red CI lane once; the variables must outlive deactivation.
    """

    def test_no_deactivate_hook_is_written(self) -> None:
        script = (DEVTOOLS / "install_carmel.sh").read_text()
        assert 'rm -f "$deactivate_hook"' in script, "a stale deactivate hook must be removed"
        assert '>"$deactivate_hook' not in script, (
            "writing a deactivate.d hook unsets the tool paths inside `conda run -n t3_env`"
        )

    def test_hook_values_are_shell_quoted(self) -> None:
        """A path with a space would truncate a variable; one with $(...) would run."""
        script = (DEVTOOLS / "install_carmel.sh").read_text()
        exports = re.findall(r"^\s*(?:printf|echo).*export (\w+)", script, flags=re.MULTILINE)
        assert set(exports) == {"T3_CONDA_ENV", "T3_PATH", "ARC_PATH", "RMG_PATH", "RMG_DB_PATH"}
        assert "%q" in script


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

    def test_the_conda_environment_matches_the_floor(self) -> None:
        text = (REPO_ROOT / "environment.yml").read_text()
        match = re.search(r"python\s*>=\s*(\d+)\.(\d+)", text)
        assert match, "environment.yml no longer pins a python floor for crml_env"
        assert (int(match.group(1)), int(match.group(2))) == self.floor()

    @pytest.mark.parametrize("workflow", ["ci.yml", "docs.yml"], ids=lambda name: name)
    def test_every_workflow_python_matches_the_floor(self, workflow: str) -> None:
        """A lane on an older Python tests something users cannot install."""
        major, minor = self.floor()
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        declared = re.findall(r'python-version:\s*\[?\s*"(\d+\.\d+)"', text)
        assert declared, f"{workflow} declares no python-version"
        for value in declared:
            assert value == f"{major}.{minor}", f"{workflow} runs on {value}, but the floor is {major}.{minor}"
