"""Tests for the ARC adapter — real contract + golden fixture.

Parity with ``tests/test_t3_adapter.py``: the pure-Python helpers (input
building, output walking, normalization, level-of-theory extraction) are tested
unconditionally against:
  1. unit fixtures defined inline
  2. a captured real ARC/Mockter fixture under tests/fixtures/arc/sample_project/

The actual subprocess-execution path is exercised only when ARC is truly
importable in the current environment (``is_arc_importable()``). That path runs
in the heavy CI lane and is skipped locally when the upstream ARC distutils
blocker (Python 3.12) is in effect.

Discovery/launch parity with T3: ARC is now launched the same way T3 is —
via a resolved conda-env/python command run in a *subprocess*
(:func:`carmel.adapters.arc._arc_python_command`,
:func:`carmel.adapters.arc._find_arc_executable`), never an in-process
``import arc`` and never a bare ``"python"``. The precedence-order and
conda-env-error tests below mirror ``tests/test_t3_adapter.py``'s equivalents
exactly, extended for ARC's four-source fallback chain (``$ARC_CONDA_ENV`` /
``$ARC_PYTHON`` falling back to ``$T3_CONDA_ENV`` / ``$T3_PYTHON``, since ARC
and T3 may share a single environment).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from carmel.adapters.arc import (
    ARC_GUARDRAILS,
    ARC_LAYOUT,
    DEFAULT_JOB_TYPES,
    MOCK_LEVEL_OF_THEORY,
    ARCAdapter,
    _arc_conda_env_error,
    _arc_python_command,
    _arc_version,
    _coerce_reaction_entry,
    _coerce_species_entry,
    _converged_species_labels,
    _count_converged,
    _find_arc_executable,
    _requested_labels,
    _success_labels,
    arc_info_filename,
    build_arc_input,
    extract_level_of_theory,
    is_arc_importable,
    is_arc_installed,
    normalize_arc_outputs,
    read_arc_info_file,
    read_arc_output_file,
    resolve_project_info_file,
    write_arc_input_file,
)
from carmel.schemas import (
    ActionKind,
    ApprovalRequirement,
    Budgets,
    Campaign,
    CampaignInput,
    CampaignStateValue,
    FailureCode,
    InitialMixture,
    MixtureComponent,
    PlannedAction,
    ReactorSystem,
    ReactorType,
    RunStatus,
    SubmissionMode,
    TargetObservable,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "arc" / "sample_project"

# Captured at import time, before the autouse fixture below can ever run, so
# TestARCAdapterRealSubprocess can restore the ambient CI launcher environment
# explicitly even though every other test in this module starts from a
# cleared one. Mirrors tests/test_t3_adapter.py's _AMBIENT_T3_ENV.
_AMBIENT_ARC_ENV = {
    name: os.environ[name]
    for name in ("ARC_CONDA_ENV", "ARC_PYTHON", "ARC_PATH", "T3_CONDA_ENV", "T3_PYTHON")
    if name in os.environ
}


def _fixture_input() -> dict[str, Any]:
    """The captured ARC input the golden fixture was produced from."""
    return dict(yaml.safe_load((FIXTURE_ROOT / "input.yml").read_text()))


# Evaluated at collection time (module import), deliberately *before* the
# autouse fixture below can clear anything — this must keep reading the real
# ambient environment so the skip decision matches what
# TestARCAdapterRealSubprocess will actually see. See
# tests/test_t3_adapter.py's requires_t3 for why this ordering matters.
requires_arc = pytest.mark.skipif(
    not is_arc_importable(),
    reason="ARC not actually importable (likely distutils blocker on Python 3.12)",
)


@pytest.fixture(autouse=True)
def _clear_ambient_arc_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ARC/T3 launcher env vars so every test starts from a known-empty env.

    Mirrors tests/test_t3_adapter.py's ``_clear_ambient_t3_env``: every test
    in this module that deliberately breaks or monkeypatches one part of the
    resolution chain must not have its intent overridden by ambient
    ``$ARC_CONDA_ENV``/``$ARC_PYTHON``/``$ARC_PATH``/``$T3_CONDA_ENV``/
    ``$T3_PYTHON`` values winning at a higher precedence step. Each test here
    starts from a cleared environment and opts in explicitly to whatever env
    var it needs.

    ``TestARCAdapterRealSubprocess`` is exempted: it genuinely needs the
    ambient CI launcher environment, which is why it's captured into
    ``_AMBIENT_ARC_ENV`` at module import time (before this fixture could
    ever clear it) and restored here explicitly for that class only.
    """
    for name in ("ARC_CONDA_ENV", "ARC_PYTHON", "ARC_PATH", "T3_CONDA_ENV", "T3_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    if request.cls is not None and request.cls.__name__ == "TestARCAdapterRealSubprocess":
        for name, value in _AMBIENT_ARC_ENV.items():
            monkeypatch.setenv(name, value)


def _campaign(workspace_root: Path) -> Campaign:
    return Campaign(
        campaign_id="test-id",
        workspace_root=workspace_root,
        input=CampaignInput(
            workspace_name="ethanol_combustion",
            initial_mixture=InitialMixture(
                components=[
                    MixtureComponent(species="OH", mole_fraction=0.05, smiles="[OH]"),
                    MixtureComponent(species="CH3", mole_fraction=0.20, smiles="[CH3]"),
                ]
            ),
            target_observables=[TargetObservable(name="ignition_delay")],
            target_reactor_systems=[
                ReactorSystem(
                    reactor_type=ReactorType.JSR,
                    temperature_range_K=(800.0, 1200.0),
                    pressure_range_bar=(1.0, 5.0),
                    residence_time_s=1.0,
                )
            ],
            budgets=Budgets(cpu_hours=10.0, experiment_budget=0.0),
        ),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _action(**params: object) -> PlannedAction:
    return PlannedAction(
        action_id="action-1",
        kind=ActionKind.ARC_RUN,
        description="Standalone ARC job",
        estimated_cpu_hours=1.0,
        rationale="test",
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        parameters=dict(params),
    )


# ---------------------------------------------------------------------------
# ARC discovery
# ---------------------------------------------------------------------------


class TestARCDiscovery:
    def test_is_arc_importable_returns_bool(self) -> None:
        assert isinstance(is_arc_importable(), bool)

    def test_importable_and_installed_probe_different_environments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``is_arc_importable() => is_arc_installed()`` is not a real invariant.

        Mirrors ``tests/test_t3_adapter.py``'s
        ``test_importable_and_installed_probe_different_environments``:
        ``is_arc_importable`` probes the *resolved ARC interpreter* (which may
        live in a separate conda env named by ``$ARC_CONDA_ENV``/``$ARC_PYTHON``
        or T3's), while ``is_arc_installed`` calls
        ``importlib.util.find_spec("arc")`` in *Carmel's own process*. Under
        the three-env deployment model these can and do diverge.
        """
        from carmel.adapters import arc as arc_module

        # `is_arc_installed` is isolated from whatever happens to be on this
        # machine's own crml_env sys.path (which may genuinely have `arc`
        # installed, e.g. under the collapsed single-env deployment) so the
        # divergence demonstrated here is deterministic either way.
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: None)
        (tmp_path / "arc.py").write_text("__version__ = '9.9.9'\n")
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # only the *resolved* subprocess sees this
        assert is_arc_importable() is True
        assert is_arc_installed() is False

    def test_find_executable_returns_list_or_none(self) -> None:
        result = _find_arc_executable()
        prefix = _arc_python_command()
        assert result is None or (isinstance(result, list) and result[: len(prefix)] == prefix)

    def test_is_arc_installed_true_when_spec_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: object())
        assert is_arc_installed() is True

    def test_is_arc_installed_false_when_spec_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: None)
        assert is_arc_installed() is False

    def test_is_arc_importable_true_when_resolved_interpreter_can_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "arc.py").write_text("__version__ = '1.2.3'\n")
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert is_arc_importable() is True

    def test_is_arc_importable_false_when_resolved_interpreter_cannot_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # empty dir: no arc module here
        assert is_arc_importable() is False

    def test_is_arc_importable_false_when_interpreter_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = tmp_path / "no-such-interpreter"
        monkeypatch.setenv("ARC_PYTHON", str(missing))
        assert is_arc_importable() is False

    def test_is_arc_importable_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        def _timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="arc-probe", timeout=1.0)

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _timeout)
        assert is_arc_importable() is False

    def test_arc_version_returns_version_string(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "arc.py").write_text("__version__ = '9.9.9'\n")
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert _arc_version() == "9.9.9"

    def test_arc_version_none_when_module_has_no_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "arc.py").write_text("# no __version__ attribute\n")
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert _arc_version() is None

    def test_arc_version_none_when_import_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # empty dir: no arc module here
        assert _arc_version() is None


class TestFindARCExecutable:
    """Precedence order of ``_find_arc_executable``, each branch isolated."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.delenv("ARC_PATH", raising=False)
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: None)
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)

    def test_env_path_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT).write_text("# ARC")
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        assert _find_arc_executable() == [sys.executable, str(tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_env_path_ignored_when_script_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        assert _find_arc_executable() is None

    def test_falls_back_to_package_sibling_script(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        repo_root = tmp_path / "ARC"
        pkg = repo_root / "arc"
        pkg.mkdir(parents=True)
        (repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT).write_text("# ARC")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: spec)
        assert _find_arc_executable() == [sys.executable, str(repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_package_sibling_skipped_when_script_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        pkg = tmp_path / "ARC" / "arc"
        pkg.mkdir(parents=True)
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: spec)
        assert _find_arc_executable() is None

    def test_falls_back_to_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: "/usr/local/bin/ARC.py")
        assert _find_arc_executable() == [sys.executable, "/usr/local/bin/ARC.py"]

    def test_returns_none_when_nothing_found(self) -> None:
        assert _find_arc_executable() is None

    def test_no_module_fallback_even_when_arc_is_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression test: unlike T3, ARC has no ``-m`` invocation mode (no
        ``arc/__main__.py``, no console_scripts entry point) — the real
        DanaResearchGroup/ARC repo ships only ``ARC.py`` at the repo root.
        ``_find_arc_executable`` must therefore return ``None`` when nothing
        else is found, *regardless* of what ``is_arc_importable()`` reports —
        never fabricate a phantom ``python -m arc`` command.
        """
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "is_arc_importable", lambda: True)
        assert _find_arc_executable() is None

    def test_env_path_carries_conda_run_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_arc_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "arc_env",
            "--no-capture-output",
            "python",
            str(script),
        ]

    def test_conda_env_set_ignores_carmel_own_package_sibling_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test: with $ARC_CONDA_ENV set, Carmel must not discover
        an ARC.py sitting next to *Carmel's own* importable ``arc`` package
        and then execute that script inside the named conda environment — the
        two may be entirely different checkouts. Since ARC has no ``-m``
        fallback, the named conda env being authoritative here means
        discovery correctly comes up empty (``None``), not a wrong-checkout
        script.
        """
        from carmel.adapters import arc as arc_module

        repo_root = tmp_path / "carmel_side_ARC"
        pkg = repo_root / "arc"
        pkg.mkdir(parents=True)
        (repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT).write_text("# wrong-checkout ARC")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: spec)
        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_arc_executable() is None

    def test_conda_env_set_ignores_carmel_own_which_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Companion regression test to the one above, for the
        ``shutil.which(ARC.py)`` discovery step rather than the
        ``importlib.util.find_spec`` one.
        """
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: (
                "/opt/conda/bin/conda"
                if name == "conda"
                else ("/usr/local/bin/ARC.py" if name == ARC_LAYOUT.EXECUTABLE_SCRIPT else None)
            ),
        )
        assert _find_arc_executable() is None

    def test_t3_conda_env_fallback_used_when_arc_conda_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ARC_SOURCES falls back to $T3_CONDA_ENV when $ARC_CONDA_ENV is
        unset, since ARC and T3 may share a single environment."""
        from carmel.adapters import arc as arc_module

        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_arc_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "shared_env",
            "--no-capture-output",
            "python",
            str(script),
        ]

    def test_arc_conda_env_wins_over_t3_conda_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both are set, $ARC_CONDA_ENV takes precedence over
        $T3_CONDA_ENV — ARC-specific configuration wins over the shared
        fallback."""
        from carmel.adapters import arc as arc_module

        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_arc_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "arc_env",
            "--no-capture-output",
            "python",
            str(script),
        ]

    def test_arc_python_wins_over_t3_python(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both are set (and valid), $ARC_PYTHON takes precedence over $T3_PYTHON."""
        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        arc_python = tmp_path / "arc-python3"
        arc_python.write_text("#!/bin/sh\n")
        arc_python.chmod(0o755)
        t3_python = tmp_path / "t3-python3"
        t3_python.write_text("#!/bin/sh\n")
        t3_python.chmod(0o755)
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("ARC_PYTHON", str(arc_python))
        monkeypatch.setenv("T3_PYTHON", str(t3_python))
        assert _find_arc_executable() == [str(arc_python), str(script)]

    def test_t3_python_used_as_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """$T3_PYTHON is used when none of the ARC-specific/conda sources are set."""
        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        t3_python = tmp_path / "t3-python3"
        t3_python.write_text("#!/bin/sh\n")
        t3_python.chmod(0o755)
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("T3_PYTHON", str(t3_python))
        assert _find_arc_executable() == [str(t3_python), str(script)]

    def test_arc_python_invalid_falls_through_to_valid_t3_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set-but-invalid $ARC_PYTHON is skipped in favor of a valid $T3_PYTHON.

        Regression test for the P1-a/b fix: ``launch_command`` used to
        return a set ``"python"``-kind source unconditionally, with no
        isfile/X_OK check, so a broken $ARC_PYTHON would be selected and
        handed straight to the caller instead of falling through.
        """
        script = tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# ARC")
        t3_python = tmp_path / "t3-python3"
        t3_python.write_text("#!/bin/sh\n")
        t3_python.chmod(0o755)
        monkeypatch.setenv("ARC_PATH", str(tmp_path))
        monkeypatch.setenv("ARC_PYTHON", str(tmp_path / "no-such-arc-python"))
        monkeypatch.setenv("T3_PYTHON", str(t3_python))
        assert _find_arc_executable() == [str(t3_python), str(script)]

    def test_host_discovery_allowed_when_no_overrides_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1-c: with no python/conda overrides, the resolved command is
        ``[sys.executable]`` so Carmel's own ``find_spec``/``which``
        discovery is allowed to run and can find a script.
        """
        from carmel.adapters import arc as arc_module

        repo_root = tmp_path / "ARC"
        pkg = repo_root / "arc"
        pkg.mkdir(parents=True)
        (repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT).write_text("# ARC")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: spec)
        assert _arc_python_command() == [sys.executable]
        assert _find_arc_executable() == [sys.executable, str(repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_host_discovery_disabled_when_arc_python_resolves_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1-c: a resolved command other than ``[sys.executable]`` disables
        Carmel's own host discovery, even though no ``ARC_PATH`` override is
        set and ``$T3_CONDA_ENV`` is *also* set (i.e. this is NOT merely
        "no conda var set" — a set-and-valid ``$ARC_PYTHON`` winning over an
        inherited ``$T3_CONDA_ENV`` must still disable host discovery, since
        ARC launches under a non-Carmel interpreter either way).
        """
        from carmel.adapters import arc as arc_module

        repo_root = tmp_path / "ARC"
        pkg = repo_root / "arc"
        pkg.mkdir(parents=True)
        (repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT).write_text("# ARC")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: spec)
        other_python = tmp_path / "other-python3"
        other_python.write_text("#!/bin/sh\n")
        other_python.chmod(0o755)
        monkeypatch.setenv("ARC_PYTHON", str(other_python))
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        assert _arc_python_command() == [str(other_python)]
        assert _arc_python_command() != [sys.executable]
        # Host discovery would otherwise find the sibling script above; it
        # must be skipped entirely, and ARC has no "-m" fallback, so the
        # result is None rather than a wrong-interpreter script invocation.
        assert _find_arc_executable() is None


# ---------------------------------------------------------------------------
# Input building (real ARC contract)
# ---------------------------------------------------------------------------


class TestBuildARCInput:
    def test_top_level_keys_match_real_arc_schema(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert ARC_LAYOUT.INPUT_PROJECT_KEY in payload
        assert ARC_LAYOUT.INPUT_LOT_KEY in payload
        assert ARC_LAYOUT.INPUT_SPECIES_KEY in payload

    def test_project_defaults_to_workspace_name(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert payload[ARC_LAYOUT.INPUT_PROJECT_KEY] == "ethanol_combustion"

    def test_species_derive_from_mixture_by_default(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        labels = [s["label"] for s in payload[ARC_LAYOUT.INPUT_SPECIES_KEY]]
        assert labels == ["OH", "CH3"]
        assert all("smiles" in s for s in payload[ARC_LAYOUT.INPUT_SPECIES_KEY])

    def test_action_species_override_mixture(self, tmp_path: Path) -> None:
        action = _action(species=[{"label": "H2O2", "smiles": "OO"}])
        payload = build_arc_input(_campaign(tmp_path), action)
        assert [s["label"] for s in payload[ARC_LAYOUT.INPUT_SPECIES_KEY]] == ["H2O2"]

    def test_level_of_theory_from_action(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action(level_of_theory=MOCK_LEVEL_OF_THEORY))
        assert payload[ARC_LAYOUT.INPUT_LOT_KEY] == MOCK_LEVEL_OF_THEORY

    def test_no_reactions_disables_ts_search(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert payload[ARC_LAYOUT.INPUT_TS_ADAPTERS_KEY] == []
        assert ARC_LAYOUT.INPUT_REACTIONS_KEY not in payload

    def test_reactions_passed_through(self, tmp_path: Path) -> None:
        action = _action(reactions=[{"label": "OH + CH3 <=> CH4 + O"}])
        payload = build_arc_input(_campaign(tmp_path), action)
        assert payload[ARC_LAYOUT.INPUT_REACTIONS_KEY][0]["label"] == "OH + CH3 <=> CH4 + O"

    def test_default_job_types_are_opt_only(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert payload[ARC_LAYOUT.INPUT_JOB_TYPES_KEY] == DEFAULT_JOB_TYPES

    def test_mixture_species_without_smiles_omits_smiles_key(self, tmp_path: Path) -> None:
        campaign = _campaign(tmp_path).model_copy(
            update={
                "input": _campaign(tmp_path).input.model_copy(
                    update={
                        "initial_mixture": InitialMixture(
                            components=[MixtureComponent(species="AR", mole_fraction=1.0)]
                        )
                    }
                )
            }
        )
        payload = build_arc_input(campaign, _action())
        assert payload[ARC_LAYOUT.INPUT_SPECIES_KEY] == [{"label": "AR"}]

    def test_action_species_with_one_malformed_entry_raises(self, tmp_path: Path) -> None:
        """A present ``species`` list is the exact requested target set: a
        single malformed entry must raise, not silently retarget onto the
        valid subset (that would run a different, smaller job than asked)."""
        action = _action(
            species=[
                {"label": "OH"},  # valid, no smiles
                {"smiles": "[CH3]"},  # missing label -> malformed
                "not-a-dict",  # malformed
                {"label": "H2O2", "smiles": "OO"},
            ]
        )
        with pytest.raises(ValueError, match="contains malformed entries") as excinfo:
            build_arc_input(_campaign(tmp_path), action)
        assert "not-a-dict" in str(excinfo.value)

    def test_action_species_all_invalid_raises_and_surfaces_dropped(self, tmp_path: Path) -> None:
        """F9: a present-but-unusable ``species`` must not silently fall back
        to the full mixture — the caller asked for specific targets and would
        otherwise get a different (much larger) job."""
        action = _action(species=[{"smiles": "[CH3]"}, "not-a-dict"])
        with pytest.raises(ValueError, match="contains malformed entries") as excinfo:
            build_arc_input(_campaign(tmp_path), action)
        assert "not-a-dict" in str(excinfo.value)

    def test_action_species_empty_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'species' is present but empty"):
            build_arc_input(_campaign(tmp_path), _action(species=[]))

    def test_action_species_absent_falls_back_to_mixture(self, tmp_path: Path) -> None:
        """The 'species' key absent entirely is the only case that falls back
        to the campaign mixture — this must keep working unmodified."""
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert [s["label"] for s in payload[ARC_LAYOUT.INPUT_SPECIES_KEY]] == ["OH", "CH3"]

    def test_action_reactions_all_invalid_raises_and_surfaces_dropped(self, tmp_path: Path) -> None:
        action = _action(reactions=[{"reactants": ["A"]}, "not-a-dict"])
        with pytest.raises(ValueError, match="contains malformed entries") as excinfo:
            build_arc_input(_campaign(tmp_path), action)
        assert "not-a-dict" in str(excinfo.value)

    @pytest.mark.parametrize("bad", ["../evil", "a/b", "a\\b", ".."])
    def test_unsafe_project_name_rejected(self, tmp_path: Path, bad: str) -> None:
        """F10: the project name flows into path joins (``<project>_info.yml``),
        so a separator or ``..`` must be rejected up front."""
        with pytest.raises(ValueError, match="must not contain path separators"):
            build_arc_input(_campaign(tmp_path), _action(project=bad))

    def test_action_reactions_with_one_malformed_entry_raises(self, tmp_path: Path) -> None:
        """A non-empty 'reactions' list is the exact requested target set: a
        single malformed entry must raise, not silently drop it."""
        action = _action(
            reactions=[
                {"label": "A => B"},
                {"reactants": ["A"]},  # missing label -> malformed
                "not-a-dict",  # malformed
            ]
        )
        with pytest.raises(ValueError, match="contains malformed entries") as excinfo:
            build_arc_input(_campaign(tmp_path), action)
        assert "not-a-dict" in str(excinfo.value)

    def test_action_reactions_absent_yields_no_reactions(self, tmp_path: Path) -> None:
        payload = build_arc_input(_campaign(tmp_path), _action())
        assert ARC_LAYOUT.INPUT_REACTIONS_KEY not in payload
        assert payload[ARC_LAYOUT.INPUT_TS_ADAPTERS_KEY] == []

    def test_action_reactions_present_empty_is_valid_species_only_job(self, tmp_path: Path) -> None:
        """reactions=[] is a legitimate species-only thermo job: no raise,
        and no reactions are sent to ARC."""
        payload = build_arc_input(_campaign(tmp_path), _action(reactions=[]))
        assert ARC_LAYOUT.INPUT_REACTIONS_KEY not in payload
        assert payload[ARC_LAYOUT.INPUT_TS_ADAPTERS_KEY] == []


class TestWriteARCInputFile:
    def test_writes_atomically(self, tmp_path: Path) -> None:
        payload = {"project": "x", "level_of_theory": "mock", "species": []}
        path = write_arc_input_file(tmp_path, payload)
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["project"] == "x"


# ---------------------------------------------------------------------------
# <project>_info.yml
# ---------------------------------------------------------------------------


class TestReadARCInfoFile:
    def test_reads_real_fixture(self) -> None:
        info = read_arc_info_file(resolve_project_info_file(FIXTURE_ROOT, _fixture_input()))
        labels = [s["label"] for s in info["species"]]
        assert labels == ["OH", "CH3"]
        assert info["reactions"] == []

    def test_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_arc_info_file(tmp_path / "nope.yml")

    def test_non_mapping_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("- list\n- only\n")
        with pytest.raises(ValueError, match="mapping"):
            read_arc_info_file(path)

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """F11: yaml.YAMLError is not a ValueError; without the wrap it would
        escape the adapter's typed INVALID_OUTPUT contract."""
        path = tmp_path / "bad.yml"
        path.write_text("species: [unterminated\n")
        with pytest.raises(ValueError, match="not valid YAML"):
            read_arc_info_file(path)

    def test_defaults_missing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yml"
        path.write_text("project: x\n")
        info = read_arc_info_file(path)
        assert info["species"] == []
        assert info["reactions"] == []


class TestReadARCOutputFile:
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_arc_output_file(tmp_path) is None

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """F11: same wrap as the info file — corrupt YAML must surface as the
        typed ValueError, not an unhandled yaml.YAMLError."""
        out = tmp_path / "output"
        out.mkdir()
        (out / "output.yml").write_text("arc_version: [unterminated\n")
        with pytest.raises(ValueError, match="not valid YAML"):
            read_arc_output_file(tmp_path)


class TestCoerceEntries:
    def test_species_entry(self) -> None:
        sel = _coerce_species_entry({"label": "OH", "smiles": "[OH]", "success": True})
        assert sel is not None
        assert sel.label == "OH"
        assert sel.smiles == "[OH]"
        assert "success=True" in (sel.reason or "")

    def test_species_entry_missing_label(self) -> None:
        assert _coerce_species_entry({"smiles": "[OH]"}) is None

    def test_species_entry_non_dict(self) -> None:
        assert _coerce_species_entry("nope") is None

    def test_reaction_entry_with_labels(self) -> None:
        sel = _coerce_reaction_entry({"label": "A => B", "reactants": ["A"], "products": ["B"], "success": True})
        assert sel is not None
        assert sel.label == "A => B"
        assert sel.reactants == ["A"]
        assert sel.products == ["B"]

    def test_reaction_entry_missing_label(self, caplog: pytest.LogCaptureFixture) -> None:
        # Dropped, but not silently: ARC writes a label for every reaction it
        # reports, so an entry without one is surprising enough to warn about.
        with caplog.at_level(logging.WARNING, logger="carmel.adapters.arc"):
            assert _coerce_reaction_entry({"reactants": ["A"]}) is None
        assert any("no 'label'" in message for message in caplog.messages)

    def test_reaction_entry_non_dict(self) -> None:
        assert _coerce_reaction_entry("nope") is None


class TestCountConverged:
    def test_none_output_returns_zero(self) -> None:
        assert _count_converged(None) == 0

    def test_empty_output_returns_zero(self) -> None:
        assert _count_converged({}) == 0


class TestRequestedLabels:
    def test_non_dict_entry_is_skipped(self) -> None:
        assert _requested_labels(["not-a-dict", {"label": "OH"}]) == ["OH"]

    def test_entry_missing_label_is_skipped(self) -> None:
        assert _requested_labels([{"smiles": "[OH]"}, {"label": "OH"}]) == ["OH"]

    def test_entry_with_empty_label_is_skipped(self) -> None:
        assert _requested_labels([{"label": ""}, {"label": "OH"}]) == ["OH"]


class TestSuccessLabels:
    def test_non_list_returns_empty_frozenset(self) -> None:
        assert _success_labels(None) == frozenset()
        assert _success_labels({"label": "OH", "success": True}) == frozenset()


class TestConvergedSpeciesLabels:
    def test_falsy_output_dict_returns_empty_frozenset(self) -> None:
        assert _converged_species_labels(None) == frozenset()
        assert _converged_species_labels({}) == frozenset()


class TestArcInfoFilename:
    def test_derives_from_project_key(self) -> None:
        assert arc_info_filename({"project": "my_project"}) == "my_project_info.yml"

    def test_matches_the_real_fixture(self) -> None:
        # The fixture is a real ARC capture; if this drifts, one of the two is wrong.
        assert arc_info_filename(_fixture_input()) == "carmel_mock_opt_info.yml"
        assert (FIXTURE_ROOT / "carmel_mock_opt_info.yml").is_file()

    @pytest.mark.parametrize("bad", [{}, {"project": ""}, {"project": None}])
    def test_missing_project_raises(self, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="no non-empty 'project' key"):
            arc_info_filename(bad)


class TestResolveProjectInfoFile:
    def test_finds_real_fixture(self) -> None:
        info = resolve_project_info_file(FIXTURE_ROOT, _fixture_input())
        assert info is not None
        assert info.name == "carmel_mock_opt_info.yml"

    def test_none_for_missing_dir(self, tmp_path: Path) -> None:
        assert resolve_project_info_file(tmp_path / "missing", {"project": "p"}) is None

    def test_none_when_no_info_file(self, tmp_path: Path) -> None:
        assert resolve_project_info_file(tmp_path, {"project": "p"}) is None

    def test_ignores_a_foreign_capture(self, tmp_path: Path) -> None:
        # A glob over *_info.yml would return this stale file and Carmel would
        # report another project's results as this run's. Resolving the exact
        # name ARC was told to write makes a missing file read as missing.
        (tmp_path / "someone_elses_info.yml").write_text("species: [{label: GHOST}]\n")
        assert resolve_project_info_file(tmp_path, {"project": "p"}) is None

    def test_ignores_a_directory_of_the_right_name(self, tmp_path: Path) -> None:
        (tmp_path / "p_info.yml").mkdir()
        assert resolve_project_info_file(tmp_path, {"project": "p"}) is None


# ---------------------------------------------------------------------------
# Level-of-theory
# ---------------------------------------------------------------------------


class TestExtractLevelOfTheory:
    def test_extracts_from_input(self) -> None:
        assert extract_level_of_theory({"level_of_theory": "wb97xd/def2tzvp"}) == "wb97xd/def2tzvp"

    def test_none_when_absent(self) -> None:
        assert extract_level_of_theory({"project": "x"}) is None


# ---------------------------------------------------------------------------
# Golden fixture — real captured ARC/Mockter output tree
# ---------------------------------------------------------------------------


class TestGoldenFixture:
    @pytest.fixture
    def input_dict(self) -> dict[str, Any]:
        return _fixture_input()

    def test_normalize_finds_species(self, input_dict: dict[str, object]) -> None:
        diag, quality = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert sorted(s.label for s in diag.species_to_compute) == ["CH3", "OH"]
        assert quality.ok
        assert quality.warnings == []

    def test_normalize_records_success_in_reason(self, input_dict: dict[str, object]) -> None:
        diag, _quality = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        by_label = {s.label: s for s in diag.species_to_compute}
        assert "success=True" in (by_label["OH"].reason or "")

    def test_normalize_no_pdep_networks(self, input_dict: dict[str, object]) -> None:
        diag, _quality = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.pdep_networks_to_compute == []

    def test_normalize_extracts_lot(self, input_dict: dict[str, object]) -> None:
        diag, _quality = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.level_of_theory == MOCK_LEVEL_OF_THEORY

    def test_normalize_records_metadata(self, input_dict: dict[str, object]) -> None:
        diag, quality = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.tool_metadata["adapter"] == "arc"
        assert diag.tool_metadata["species_count"] == 2
        assert diag.tool_metadata["converged_species_count"] == 2
        assert diag.tool_metadata["arc_version"] == "1.1.0"
        assert diag.tool_metadata["requested_count"] == 2
        assert diag.tool_metadata["succeeded_count"] == 2
        assert diag.tool_metadata["failed_labels"] == []
        assert diag.tool_metadata["missing_output"] is False
        assert diag.tool_metadata["count_mismatch"] is False
        assert quality.failed_labels == []

    def test_normalize_raises_when_no_output(self, tmp_path: Path, input_dict: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="No ARC info file"):
            normalize_arc_outputs(tmp_path, input_dict, campaign_id="c", run_id="r")


class TestNormalizeArcOutputsInline:
    """Edge cases the golden fixture cannot exercise (it has no reactions)."""

    def test_reactions_and_invalid_species_entries_are_handled(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "proj_info.yml").write_text(
            yaml.safe_dump(
                {
                    "species": [{"label": "OH", "success": True}, {"no_label": "bad"}],
                    "reactions": [
                        {"no_label": "bad"},
                        {"label": "OH + CH3 => CH4 + O", "success": True},
                    ],
                }
            )
        )
        # Reaction quality depends only on `success`, so it needs no output.yml,
        # but species quality also requires convergence — write one so OH's
        # inclusion in species_to_compute exercises the invalid-entry filtering
        # this test is actually about, rather than the (separately-tested)
        # missing-output path.
        (project / "output").mkdir()
        (project / "output" / "output.yml").write_text(
            yaml.safe_dump({"species": [{"label": "OH", "converged": True}]})
        )
        diag, quality = normalize_arc_outputs(project, {"project": "proj"}, campaign_id="c", run_id="r")
        assert [s.label for s in diag.species_to_compute] == ["OH"]
        assert [r.label for r in diag.reactions_to_compute] == ["OH + CH3 => CH4 + O"]
        assert diag.tool_metadata["reaction_count"] == 1
        # No species/reactions were requested via the ARC *input* dict here
        # (it is a bare {"project": ...}), so ARCQuality has nothing to
        # cross-check against and `ok` reads as False — this test is only
        # about entry coercion/filtering, not the requested-vs-actual gate.
        assert quality.requested_labels == frozenset()
        assert not quality.ok

    def test_output_only_without_info_file_raises(self, tmp_path: Path) -> None:
        """F12: an output.yml with no ``<project>_info.yml`` is not a success.
        ARC writes the info file unconditionally at the end of a run, so its
        absence means the run died — returning SUCCESS with empty
        species/reactions would silently lose every result."""
        project = tmp_path / "proj"
        (project / "output").mkdir(parents=True)
        (project / "output" / "output.yml").write_text(yaml.safe_dump({"arc_version": "0.0.1", "species": []}))
        with pytest.raises(ValueError, match="No ARC info file"):
            normalize_arc_outputs(project, {"project": "proj"}, campaign_id="c", run_id="r")


class TestARCQuality:
    """Unit tests for the requested-vs-actual quality gate (ARCQuality)."""

    def _input(self, species: list[str], reactions: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"project": "proj", "species": [{"label": lbl} for lbl in species]}
        if reactions:
            payload["reactions"] = [{"label": lbl} for lbl in reactions]
        return payload

    def test_golden_fixture_is_ok_with_no_warnings(self) -> None:
        diag, quality = normalize_arc_outputs(FIXTURE_ROOT, _fixture_input(), campaign_id="c", run_id="r")
        assert quality.ok
        assert quality.warnings == []
        assert diag.warnings == []

    def test_one_species_failed_excludes_it_and_warns(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "proj_info.yml").write_text(
            yaml.safe_dump(
                {
                    "species": [{"label": "OH", "success": True}, {"label": "CH3", "success": False}],
                    "reactions": [],
                }
            )
        )
        (project / "output").mkdir()
        (project / "output" / "output.yml").write_text(
            yaml.safe_dump({"species": [{"label": "OH", "converged": True}, {"label": "CH3", "converged": False}]})
        )
        diag, quality = normalize_arc_outputs(project, self._input(["OH", "CH3"]), campaign_id="c", run_id="r")
        assert not quality.ok
        assert quality.failed_labels == ["CH3"]
        assert [s.label for s in diag.species_to_compute] == ["OH"]
        assert any("CH3" in w for w in diag.warnings)

    def test_missing_output_file_fails_with_warning(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "proj_info.yml").write_text(
            yaml.safe_dump({"species": [{"label": "OH", "success": True}], "reactions": []})
        )
        diag, quality = normalize_arc_outputs(project, self._input(["OH"]), campaign_id="c", run_id="r")
        assert not quality.ok
        assert quality.missing_output
        assert diag.species_to_compute == []
        assert any("output" in w.lower() for w in diag.warnings)

    def test_count_mismatch_fails_with_warning(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "proj_info.yml").write_text(
            yaml.safe_dump(
                {
                    "species": [{"label": "OH", "success": True}, {"label": "CH3", "success": True}],
                    "reactions": [],
                }
            )
        )
        (project / "output").mkdir()
        # Info reports 2 successful species; output.yml reports only 1 converged.
        (project / "output" / "output.yml").write_text(
            yaml.safe_dump({"species": [{"label": "OH", "converged": True}, {"label": "CH3", "converged": False}]})
        )
        diag, quality = normalize_arc_outputs(project, self._input(["OH", "CH3"]), campaign_id="c", run_id="r")
        assert not quality.ok
        assert quality.count_mismatch
        assert any("mismatch" not in w and "reports" in w for w in diag.warnings)

    def test_requested_reaction_not_in_info_file_fails_with_warning(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "proj_info.yml").write_text(
            yaml.safe_dump(
                {
                    "species": [{"label": "OH", "success": True}],
                    "reactions": [{"label": "A <=> B", "success": True}],
                }
            )
        )
        (project / "output").mkdir()
        (project / "output" / "output.yml").write_text(
            yaml.safe_dump({"species": [{"label": "OH", "converged": True}]})
        )
        diag, quality = normalize_arc_outputs(
            project, self._input(["OH"], ["A <=> B", "OH + CH3 <=> CH4 + O"]), campaign_id="c", run_id="r"
        )
        assert not quality.ok
        # The succeeded reaction is not marked as failed; only the missing one is.
        assert quality.failed_labels == ["OH + CH3 <=> CH4 + O"]
        assert any("OH + CH3 <=> CH4 + O" in w and "did not succeed" in w for w in diag.warnings)
        assert not any("A <=> B" in w for w in diag.warnings)


# ---------------------------------------------------------------------------
# Adapter failure paths (deterministic, no ARC needed)
# ---------------------------------------------------------------------------


class TestARCCondaEnvError:
    """Unit tests for ``_arc_conda_env_error()`` in isolation.

    Mirrors ``tests/test_t3_adapter.py``'s ``TestT3CondaEnvError`` exactly,
    for ``$ARC_CONDA_ENV`` (falling back to ``$T3_CONDA_ENV``, since ARC and
    T3 may share a single environment).
    """

    def test_env_var_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARC_CONDA_ENV", raising=False)
        monkeypatch.delenv("T3_CONDA_ENV", raising=False)
        assert _arc_conda_env_error() is None

    def test_conda_missing_from_path_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)
        error = _arc_conda_env_error()
        assert error is not None
        assert "ARC_CONDA_ENV" in error
        assert "conda" in error.lower()

    def test_named_env_does_not_exist_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "no_such_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="EnvironmentLocationNotFound: could not find environment"
        )
        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", lambda *a, **k: completed)
        error = _arc_conda_env_error()
        assert error is not None
        assert "no_such_env" in error
        assert "EnvironmentLocationNotFound" in error

    def test_conda_run_raises_oserror_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: "/usr/bin/conda")

        def _raise(*args: object, **kwargs: object) -> None:
            raise OSError("boom")

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _raise)
        error = _arc_conda_env_error()
        assert error is not None
        assert "arc_env" in error

    def test_env_usable_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", lambda *a, **k: completed)
        assert _arc_conda_env_error() is None

    def test_python_source_valid_file_short_circuits_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set, executable ``$ARC_PYTHON`` (no conda source set) is accepted outright.

        Exercises the ``"python"``-kind branch of the shared
        ``_launcher.conda_env_error`` (unlike ``"conda"``-kind sources, a
        python-kind source is never probed by running anything — only
        checked for existence and the execute bit).
        """
        monkeypatch.setenv("ARC_PYTHON", sys.executable)
        assert _arc_conda_env_error() is None

    def test_invalid_python_source_falls_through_to_next_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set-but-invalid ``$ARC_PYTHON`` is not fatal; resolution keeps walking sources.

        Unlike a broken ``"conda"``-kind source, a broken ``"python"``-kind
        source does not short-circuit with an error — it merely fails to
        short-circuit with success, and the walk continues to the next
        source in precedence order (here, ``$T3_CONDA_ENV``).
        """
        monkeypatch.setenv("ARC_PYTHON", "/nonexistent/carmel-not-a-binary")
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)
        error = _arc_conda_env_error()
        assert error is not None
        assert "T3_CONDA_ENV" in error

    def test_t3_conda_env_used_as_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """$T3_CONDA_ENV is consulted when $ARC_CONDA_ENV is unset."""
        from carmel.adapters import arc as arc_module

        monkeypatch.delenv("ARC_CONDA_ENV", raising=False)
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)
        error = _arc_conda_env_error()
        assert error is not None
        assert "T3_CONDA_ENV" in error

    def test_conda_missing_message_names_arc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2-a: the ``tool_label`` wired into ``_arc_conda_env_error`` is
        ``"ARC"`` (not the shared-launcher-generic "the tool"), so the
        "refusing to silently launch" message reads naturally for ARC.
        """
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)
        error = _arc_conda_env_error()
        assert error is not None
        assert "refusing to silently launch ARC under a different interpreter" in error

    def test_launch_command_conda_missing_with_no_next_source_names_tool_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither T3_SOURCES nor ARC_SOURCES ever end on a "conda" entry
        (both end on a "python" entry), so the ``next_var is None`` branch
        of ``launch_command`` — a broken conda source with nothing after it
        in *sources* — is only reachable by calling the shared launcher
        directly with a custom sources list. Covers the "trying the next
        %s source" (tool_label) wording used when there is no more specific
        env-var hint available.
        """
        from carmel.adapters import _launcher

        monkeypatch.setenv("SOME_CONDA_ENV", "an_env")
        result = _launcher.launch_command(
            [("conda", "SOME_CONDA_ENV")],
            logger=_launcher._log,
            tool_label="ARC",
            which=lambda _name: None,
        )
        assert result == [sys.executable]

    def test_invalid_arc_python_agrees_with_launch_command_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-a/b: ``_arc_conda_env_error`` and ``_arc_python_command`` must
        agree on which source is selected when a set ``$ARC_PYTHON`` is
        invalid and ``$T3_CONDA_ENV`` is a valid fallback — both walk past
        the broken python source to the same conda source.
        """
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_PYTHON", "/nonexistent/carmel-not-a-binary")
        monkeypatch.setenv("T3_CONDA_ENV", "shared_env")
        monkeypatch.setattr(
            arc_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", lambda *a, **k: completed)

        # conda_env_error walks past the invalid ARC_PYTHON and finds the
        # T3_CONDA_ENV conda source usable -> None (no error).
        assert _arc_conda_env_error() is None
        # launch_command must select that same T3_CONDA_ENV conda source,
        # not the invalid ARC_PYTHON, and not silently fall through further
        # than conda_env_error agrees is safe.
        assert _arc_python_command() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "shared_env",
            "--no-capture-output",
            "python",
        ]


class TestARCAdapterCondaEnvFailures:
    """Adapter-level behavior when ``$ARC_CONDA_ENV`` is set but unusable.

    Mirrors ``tests/test_t3_adapter.py``'s ``TestT3AdapterCondaEnvFailures``:
    both scenarios must be a clean, typed ``FailureCode.TOOL_NOT_FOUND``
    failure rather than a silent fallback onto Carmel's own interpreter.
    """

    def test_conda_missing_from_path_records_tool_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert run.input_path is not None
        assert run.input_path.exists()
        assert "ARC_CONDA_ENV" in (run.error_message or "")
        assert "conda" in (run.error_message or "").lower()

    def test_named_env_does_not_exist_records_tool_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "no_such_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="EnvironmentLocationNotFound: could not find environment"
        )
        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", lambda *a, **k: completed)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert "no_such_env" in (run.error_message or "")

    def test_conda_preflight_failure_does_not_probe_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prelaunch conda-env failure must NOT probe ARC's version.

        Probing there would fall back through the lenient launcher to the
        wrong interpreter (stamping a misleading ``tool_version``) and add a
        redundant, potentially 120s subprocess to an already-decided
        prelaunch failure. Mirrors T3's ``probe_version=False`` guard.
        """
        from carmel.adapters import arc as arc_module

        monkeypatch.setenv("ARC_CONDA_ENV", "arc_env")
        monkeypatch.setattr(arc_module.shutil, "which", lambda _name: None)

        def _boom() -> str:
            raise AssertionError("_arc_version must not be probed for a prelaunch failure")

        monkeypatch.setattr(arc_module, "_arc_version", _boom)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, _diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert run.tool_version is None


class TestARCAdapterFailures:
    def test_tool_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: None)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert run.input_path is not None
        assert run.input_path.exists()

    def test_input_build_error_is_typed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        def _raise(*_a: object, **_k: object) -> dict[str, Any]:
            raise ValueError("boom")

        monkeypatch.setattr(arc_module, "build_arc_input", _raise)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INPUT_BUILD_ERROR
        assert diagnostics is None

    def test_run_dir_creation_failure_is_typed(self, tmp_path: Path) -> None:
        """F8: an OSError from run_dir.mkdir must return a typed
        SUBPROCESS_ERROR record, not escape the adapter (mirrors T3)."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "runs").write_text("not a directory")
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "Could not create run directory" in (run.error_message or "")
        assert diagnostics is None

    def test_input_write_os_error_is_typed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """F8: write_arc_input_file does mkdir+write_text, which raise OSError —
        that must map to INPUT_BUILD_ERROR like the T3 input-build path."""
        from carmel.adapters import arc as arc_module

        def _raise(*_a: object, **_k: object) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr(arc_module, "write_arc_input_file", _raise)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INPUT_BUILD_ERROR
        assert "disk full" in (run.error_message or "")
        assert diagnostics is None

    def test_species_present_but_all_invalid_is_input_build_error(self, tmp_path: Path) -> None:
        """F9 end-to-end: the ValueError from _resolve_species must surface as
        a typed INPUT_BUILD_ERROR, never as a silently retargeted job."""
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(
            workspace_root=ws, campaign=_campaign(ws), action=_action(species=["not-a-dict"])
        )
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INPUT_BUILD_ERROR
        assert "malformed entries" in (run.error_message or "")
        assert diagnostics is None

    def test_subprocess_raises_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        def _raise_os_error(*_a: object, **_k: object) -> None:
            raise OSError("no such executable")

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _raise_os_error)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert diagnostics is None

    def test_the_adapter_passes_the_recorder_through_to_the_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adapter must actually forward ``on_process_start``, not accept it.

        ARC mirror of the T3 test of the same name: a test of the shared
        launcher alone passes happily when the adapter takes the argument
        and drops it on the floor. Every real run would then launch ARC
        with no recorded process group, and every service-level test would
        still be green, because those use adapter doubles that call the
        recorder themselves.
        """
        from carmel.adapters import arc as arc_module

        seen: list[object] = []

        def _capture(*_args: object, **kwargs: object) -> None:
            seen.append(kwargs.get("on_process_start"))
            raise OSError("stop here, the launch is all this checks")

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])
        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _capture)
        ws = tmp_path / "ws"
        ws.mkdir()

        def _recorder(_pgid: int, _argv: list[str]) -> None:  # pragma: no cover -- never invoked
            raise AssertionError("the stub never launches anything")

        ARCAdapter().run(
            workspace_root=ws,
            campaign=_campaign(ws),
            action=_action(),
            on_process_start=_recorder,
        )
        # Later entries are the version probe, which is a separate launch
        # and deliberately unrecorded; the first is ARC itself.
        assert seen[0] is _recorder, "the adapter dropped the recorder instead of forwarding it"

    def test_subprocess_nonzero_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["false"])
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "exited with code" in (run.error_message or "")
        assert diagnostics is None

    def test_subprocess_succeeds_but_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        # `true` exits 0 but writes nothing — normalize_arc_outputs should fail.
        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["true"])
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INVALID_OUTPUT
        assert diagnostics is None

    def test_timeout_is_typed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["sleep"])

        def _raise_timeout(*a: object, **k: object) -> None:
            raise subprocess.TimeoutExpired(cmd="arc", timeout=1)

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _raise_timeout)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TIMEOUT
        assert diagnostics is None

    def test_success_path_normalizes_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate ARC writing its real output tree, then assert the success path."""
        import shutil

        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        class _Completed:
            returncode = 0

        class _ProbeCompleted:
            returncode = 1  # non-zero: _arc_version's probe call returns None, never touching .stdout

        def _fake_run(command: list[str], cwd: Path | None = None, **kwargs: object) -> object:
            # The same monkeypatched symbol also backs the version-probe call
            # inside _failed_record/_arc_version, which never passes `cwd`
            # (see _launcher.probe_version) — treat that as a harmless no-op
            # rather than the real ARC invocation.
            if cwd is None:
                return _ProbeCompleted()
            # ARC writes its output tree into the run dir (cwd); mirror the golden
            # fixture. Crucially, name the info file the way real ARC does — from
            # the project in the input Carmel just wrote — rather than hardcoding
            # the fixture's own project name. A stand-in that writes a name the
            # adapter never asked for tests nothing.
            run_dir = Path(cwd)
            payload = yaml.safe_load((run_dir / "input.yml").read_text())
            shutil.copy(FIXTURE_ROOT / "carmel_mock_opt_info.yml", run_dir / arc_info_filename(payload))
            (run_dir / "output").mkdir(exist_ok=True)
            shutil.copy(FIXTURE_ROOT / "output" / "output.yml", run_dir / "output" / "output.yml")
            return _Completed()

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _fake_run)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(
            workspace_root=ws, campaign=_campaign(ws), action=_action(level_of_theory=MOCK_LEVEL_OF_THEORY)
        )
        assert run.status == RunStatus.SUCCEEDED
        assert run.failure_code == FailureCode.NONE
        assert run.tool_name == "arc"
        assert run.actual_cpu_hours is not None
        assert diagnostics is not None
        assert sorted(s.label for s in diagnostics.species_to_compute) == ["CH3", "OH"]
        assert diagnostics.level_of_theory == MOCK_LEVEL_OF_THEORY

    def test_non_converged_species_yields_failed_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: returncode==0 but one requested species never converged.

        ARCAdapter.run must not report SUCCEEDED purely off the subprocess exit
        code -- it has to gate on ARCQuality. Diagnostics must still come back
        (with the good subset + warnings) rather than None, so the partial
        result stays inspectable even though the run is FAILED."""
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        class _Completed:
            returncode = 0

        class _ProbeCompleted:
            returncode = 1  # non-zero: _arc_version's probe call returns None, never touching .stdout

        def _fake_run(command: list[str], cwd: Path | None = None, **kwargs: object) -> object:
            if cwd is None:  # version-probe call inside _failed_record; see above
                return _ProbeCompleted()
            run_dir = Path(cwd)
            payload = yaml.safe_load((run_dir / "input.yml").read_text())
            (run_dir / arc_info_filename(payload)).write_text(
                yaml.safe_dump(
                    {
                        "species": [
                            {"label": "OH", "success": True},
                            {"label": "CH3", "success": False},
                        ],
                        "reactions": [],
                    }
                )
            )
            (run_dir / "output").mkdir(exist_ok=True)
            (run_dir / "output" / "output.yml").write_text(
                yaml.safe_dump(
                    {
                        "species": [
                            {"label": "OH", "converged": True},
                            {"label": "CH3", "converged": False},
                        ]
                    }
                )
            )
            return _Completed()

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _fake_run)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(
            workspace_root=ws, campaign=_campaign(ws), action=_action(level_of_theory=MOCK_LEVEL_OF_THEORY)
        )
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INVALID_OUTPUT
        assert diagnostics is not None
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert any("CH3" in w for w in diagnostics.warnings)

    def test_corrupt_info_yaml_is_typed_invalid_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """F11 end-to-end: a corrupt ``<project>_info.yml`` must produce a typed
        INVALID_OUTPUT record — yaml.YAMLError must not escape run()."""
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        class _Completed:
            returncode = 0

        class _ProbeCompleted:
            returncode = 1  # non-zero: _arc_version's probe call returns None, never touching .stdout

        def _fake_run(command: list[str], cwd: Path | None = None, **kwargs: object) -> object:
            if cwd is None:  # version-probe call inside _failed_record; see above
                return _ProbeCompleted()
            run_dir = Path(cwd)
            payload = yaml.safe_load((run_dir / "input.yml").read_text())
            (run_dir / arc_info_filename(payload)).write_text("species: [unterminated\n")
            return _Completed()

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _fake_run)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INVALID_OUTPUT
        assert "not valid YAML" in (run.error_message or "")
        assert diagnostics is None

    def test_estimate_cost_uses_declared_estimate(self) -> None:
        assert ARCAdapter().estimate_cost(_action()) == 1.0

    def test_estimate_cost_falls_back_to_species_count(self) -> None:
        action = _action(species=[{"label": "A"}, {"label": "B"}], reactions=[{"label": "A => B"}])
        action = action.model_copy(update={"estimated_cpu_hours": 0.0})
        assert ARCAdapter().estimate_cost(action) == 4.0  # 2 species + 2*1 reaction

    def test_estimate_cost_with_campaign_reflects_mixture_fallback(self, tmp_path: Path) -> None:
        """F13: with no explicit species, the estimate must reflect the mixture
        the input actually resolves to, not a single-species floor."""
        action = _action().model_copy(update={"estimated_cpu_hours": 0.0})
        adapter = ARCAdapter()
        assert adapter.estimate_cost(action) == 1.0  # without the campaign: blind floor
        assert adapter.estimate_cost(action, _campaign(tmp_path)) == 2.0  # OH + CH3

    def test_estimate_cost_with_campaign_counts_resolved_species_and_reactions(self, tmp_path: Path) -> None:
        action = _action(species=[{"label": "A"}, {"label": "B"}], reactions=[{"label": "A => B"}])
        action = action.model_copy(update={"estimated_cpu_hours": 0.0})
        assert ARCAdapter().estimate_cost(action, _campaign(tmp_path)) == 4.0

    def test_estimate_cost_with_campaign_survives_unresolvable_parameters(self, tmp_path: Path) -> None:
        """estimate_cost must never raise (it runs inside _failed_record too):
        unresolvable parameters fall back to the raw-parameter estimate."""
        action = _action(species=[{"smiles": "no-label"}]).model_copy(update={"estimated_cpu_hours": 0.0})
        assert ARCAdapter().estimate_cost(action, _campaign(tmp_path)) == 1.0

    def test_timeout_budget_reflects_resolved_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """F13 end-to-end: the subprocess wall-clock must be derived from the
        resolved input size (2 mixture species), not the absent species param."""
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])
        captured: dict[str, object] = {}

        class _Completed:
            returncode = 1

        def _fake_run(command: list[str], cwd: Path | None = None, **kwargs: object) -> _Completed:
            if cwd is None:  # version-probe call inside _failed_record; see above
                return _Completed()
            captured.update(kwargs)
            return _Completed()

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", _fake_run)
        ws = tmp_path / "ws"
        ws.mkdir()
        action = _action().model_copy(update={"estimated_cpu_hours": 0.0})
        ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=action)
        assert captured["timeout"] == int(2 * 3600 + ARC_GUARDRAILS.subprocess_grace_seconds)

    def test_failed_record_estimate_reflects_resolved_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed run record must carry the same resolved-input estimate as
        timeout/success records, not the blind single-species floor — even
        though the estimate is only used for bookkeeping once ARC has failed."""
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        class _Completed:
            returncode = 1

        monkeypatch.setattr(arc_module, "_arc_run_in_process_group", lambda *a, **k: _Completed())
        ws = tmp_path / "ws"
        ws.mkdir()
        action = _action().model_copy(update={"estimated_cpu_hours": 0.0})
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=action)
        assert run.status == RunStatus.FAILED
        assert diagnostics is None
        # 2 mixture species (OH + CH3) resolved via the campaign, not the blind
        # single-species floor that ignores the campaign entirely.
        assert run.estimated_cpu_hours == 2.0


class TestArcTerminateProcessTree:
    """``_arc_terminate_process_tree``'s call-time default grace period.

    Mirrors ``tests/test_t3_adapter.py``'s real-subprocess timeout coverage:
    ``_launcher.run_in_process_group`` invokes its ``terminate`` callback
    with just the process (no ``grace_period_s``) on a real timeout, which
    is the only way to exercise ``_arc_terminate_process_tree``'s
    ``if grace_period_s is None: grace_period_s = _KILL_GRACE_PERIOD_S``
    branch — every other test in this module drives ARC's subprocess layer
    through a monkeypatched ``_arc_run_in_process_group``, which never
    reaches the real termination path at all.
    """

    def test_real_timeout_uses_module_level_grace_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_KILL_GRACE_PERIOD_S", 0.2)
        with pytest.raises(subprocess.TimeoutExpired):
            arc_module._arc_run_in_process_group([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)

    def test_explicit_grace_period_is_not_overridden_by_the_default(self) -> None:
        """The ``grace_period_s is not None`` branch: an explicit caller value wins."""
        from carmel.adapters import arc as arc_module

        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        proc.wait()
        arc_module._arc_terminate_process_tree(proc, grace_period_s=0.01)


class TestARCProcessTreeTermination:
    """ARC wires into the shared ``_launcher`` process-tree kill — proven live.

    The full termination machinery (grandchild death, write liveness, pgid
    reporting, recorder failure) is exercised by
    ``tests/test_t3_adapter.py::TestProcessTreeTermination`` against the
    shared implementation in ``carmel.adapters._launcher``. This class does
    not duplicate that suite; it proves the one thing T3's tests cannot:
    that ARC's own wrapper (``_arc_run_in_process_group`` →
    ``_arc_terminate_process_tree``) actually delegates a timeout into that
    shared path, grandchildren included — with ARC's grace period, not just
    the direct child.
    """

    # Parent spawns a grandchild that outlives it, then blocks forever —
    # the same tree shape as T3's regression test for the orphan.
    _TREE = (
        "import subprocess, sys, time, pathlib;"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(gc.pid));"
        "time.sleep(300)"
    )

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    @classmethod
    def _died_within(cls, pid: int, timeout_s: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not cls._alive(pid):
                return True
            time.sleep(0.05)
        return False

    def test_timeout_kills_the_grandchild_via_the_shared_launcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_KILL_GRACE_PERIOD_S", 1.0)
        pid_file = tmp_path / "grandchild.pid"
        command = [sys.executable, "-c", self._TREE, str(pid_file)]

        with pytest.raises(subprocess.TimeoutExpired):
            arc_module._arc_run_in_process_group(command, timeout=3.0)

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (pid_file.exists() and pid_file.read_text().strip()):
            time.sleep(0.05)
        assert pid_file.exists() and pid_file.read_text().strip(), "the child never reported a grandchild pid"
        grandchild = int(pid_file.read_text().strip())
        assert self._died_within(grandchild), (
            f"grandchild {grandchild} survived an ARC timeout — the ARC wrapper is not "
            "delegating to the shared _launcher process-tree kill"
        )


class TestARCAdapterRealSubprocess:
    """End-to-end subprocess tests, only when ARC is actually importable."""

    @requires_arc
    def test_run_does_not_crash(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        adapter = ARCAdapter(submission_mode=SubmissionMode.SUBPROCESS)
        run, diagnostics = adapter.run(
            workspace_root=ws,
            campaign=_campaign(ws),
            action=_action(level_of_theory=MOCK_LEVEL_OF_THEORY),
        )
        # A no-op run also "does not crash", and the tools lane's gate only
        # knows this test was not skipped -- so assert concrete evidence that
        # ARC was actually launched as a subprocess, not just a plausible
        # status. TOOL_NOT_FOUND means discovery failed *before* launch (e.g.
        # the environment did not put ARC on ARC_PATH), which is exactly the
        # silent "green gate, zero ARC executed" this lane exists to prevent.
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED)
        assert run.failure_code != FailureCode.TOOL_NOT_FOUND, (
            f"ARC was never launched (failure_code={run.failure_code}, "
            f"error={run.error_message!r}); is ARC on ARC_PATH in this environment?"
        )
        assert run.input_path is not None
        assert run.input_path.exists()
        # The stdout/stderr capture files are opened immediately before the
        # subprocess is spawned, so their presence is on-disk proof ARC really
        # ran as a subprocess (they do not exist on a prelaunch failure). They
        # live in the run directory alongside the input file the adapter wrote
        # (workspace/runs/<run_id>/), so derive it from input_path rather than
        # re-deriving the layout here.
        run_dir = run.input_path.parent
        assert (run_dir / ARC_LAYOUT.CARMEL_STDOUT_FILENAME).exists()
        assert (run_dir / ARC_LAYOUT.CARMEL_STDERR_FILENAME).exists()
        if run.status == RunStatus.SUCCEEDED:
            assert diagnostics is not None
            assert diagnostics.run_id == run.run_id

    @requires_arc
    def test_mockter_through_carmel_ui_reaches_completed_phase1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ARC end to end THROUGH CARMEL, not just through the adapter.

        The whole production path a browser drives, with the real Mockter
        subprocess underneath: create a campaign → generate a ``run_arc``
        plan (Mockter level of theory, auto-approved by the combined gate)
        → POST /run (dispatches on the action kind to ``start_arc_action``)
        → background thread runs ARC under supervision → ``RESULTS_READY``
        → ``COMPLETED_PHASE1`` → the dashboard shows the ARC diagnostics.

        Deliberately strict where ``test_run_does_not_crash`` is tolerant:
        the golden fixture proves Mockter converges these two species, so
        anything short of ``COMPLETED_PHASE1`` here means some layer between
        the UI and ARC — not ARC itself — dropped the result.

        Lives in this class so the tools lane's must-not-skip gate covers
        it and the ambient launcher environment is restored for it.
        """
        import threading

        from carmel.services.campaigns import find_campaign_workspace
        from carmel.services.execution import load_arc_diagnostics
        from carmel.services.planner import load_plan
        from carmel.services.state_machine import load_state
        from carmel.ui import app as ui_app
        from carmel.ui import create_app

        app = create_app(workspaces_root=tmp_path)
        app.config["TESTING"] = True
        client = app.test_client()

        threads: list[threading.Thread] = []
        real_start = ui_app.start_arc_action

        def _capture(ws: Path, campaign: Campaign, action: PlannedAction) -> threading.Thread:
            thread = real_start(ws, campaign, action)
            threads.append(thread)
            return thread

        monkeypatch.setattr(ui_app, "start_arc_action", _capture)

        def _post(path: str, data: dict[str, str] | None = None) -> Any:
            assert client.get("/campaigns/new").status_code == 200
            with client.session_transaction() as session:
                token = session["csrf_token"]
            payload = dict(data or {})
            payload.setdefault("csrf_token", token)
            return client.post(path, data=payload)

        response = _post(
            "/campaigns/new",
            {
                "workspace_name": "mockter-ui-e2e",
                "mixture_components": "OH,0.05,[OH]\nCH3,0.20,[CH3]",
                "observables": "ignition_delay",
                "reactors": "jsr,800,1200,1.0,5.0,1.0",
                "cpu_hours": "20",
                "experiment_budget": "0",
            },
        )
        assert response.status_code == 302
        cid = response.headers["Location"].rsplit("/", 1)[-1]
        ws = find_campaign_workspace(tmp_path, cid)
        assert ws is not None

        planned = _post(f"/campaigns/{cid}/plan", {"tool": "arc", "level_of_theory": MOCK_LEVEL_OF_THEORY})
        assert planned.status_code == 302
        plan = load_plan(ws)
        assert plan.actions[0].kind == ActionKind.ARC_RUN
        assert not plan.requires_approval, "the small Mockter action must stay auto-approved"

        assert _post(f"/campaigns/{cid}/run").status_code == 302
        assert threads, "the run route never started an ARC action"
        for thread in threads:
            thread.join(timeout=1800)
            assert not thread.is_alive(), "the background ARC run never finished"

        run_files = sorted((ws / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        detail = run_files[-1].read_text(encoding="utf-8") if run_files else "no run record was written"
        assert load_state(ws).state == CampaignStateValue.COMPLETED_PHASE1, detail

        diagnostics = load_arc_diagnostics(ws)
        assert diagnostics is not None
        assert sorted(s.label for s in diagnostics.species_to_compute) == ["CH3", "OH"]

        page = client.get(f"/campaigns/{cid}").data
        assert "Diagnostics · ARC".encode() in page
        assert diagnostics.run_id.encode() in page
