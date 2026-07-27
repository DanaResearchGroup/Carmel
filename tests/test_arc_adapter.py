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
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from carmel.adapters.arc import (
    ARC_LAYOUT,
    DEFAULT_JOB_TYPES,
    MOCK_LEVEL_OF_THEORY,
    ARCAdapter,
    _arc_version,
    _coerce_reaction_entry,
    _coerce_species_entry,
    _count_converged,
    _find_arc_executable,
    arc_info_filename,
    build_arc_input,
    extract_level_of_theory,
    is_arc_importable,
    is_arc_installed,
    normalize_arc_outputs,
    read_arc_info_file,
    resolve_project_info_file,
    write_arc_input_file,
)
from carmel.schemas import (
    ActionKind,
    ApprovalRequirement,
    Budgets,
    Campaign,
    CampaignInput,
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


def _fixture_input() -> dict[str, Any]:
    """The captured ARC input the golden fixture was produced from."""
    return dict(yaml.safe_load((FIXTURE_ROOT / "input.yml").read_text()))


requires_arc = pytest.mark.skipif(
    not is_arc_importable(),
    reason="ARC not actually importable (likely distutils blocker on Python 3.12)",
)


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

    def test_importable_implies_installed(self) -> None:
        if is_arc_importable():
            assert is_arc_installed()

    def test_find_executable_returns_list_or_none(self) -> None:
        result = _find_arc_executable()
        assert result is None or (isinstance(result, list) and result[0] == "python")

    def test_is_arc_installed_true_when_spec_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: object())
        assert is_arc_installed() is True

    def test_is_arc_installed_false_when_spec_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib.util, "find_spec", lambda _name: None)
        assert is_arc_installed() is False

    def test_is_arc_importable_true_when_import_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib, "import_module", lambda _name: object())
        assert is_arc_importable() is True

    def test_is_arc_importable_false_when_import_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        def _raise(_name: str) -> None:
            raise ImportError("no arc")

        monkeypatch.setattr(arc_module.importlib, "import_module", _raise)
        assert is_arc_importable() is False

    def test_arc_version_returns_version_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib, "import_module", lambda _name: SimpleNamespace(__version__="9.9.9"))
        assert _arc_version() == "9.9.9"

    def test_arc_version_none_when_module_has_no_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module.importlib, "import_module", lambda _name: SimpleNamespace())
        assert _arc_version() is None

    def test_arc_version_none_when_import_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        def _raise(_name: str) -> None:
            raise ImportError("no arc")

        monkeypatch.setattr(arc_module.importlib, "import_module", _raise)
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
        assert _find_arc_executable() == ["python", str(tmp_path / ARC_LAYOUT.EXECUTABLE_SCRIPT)]

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
        assert _find_arc_executable() == ["python", str(repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT)]

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
        assert _find_arc_executable() == ["python", "/usr/local/bin/ARC.py"]

    def test_returns_none_when_nothing_found(self) -> None:
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

    def test_action_species_skips_invalid_entries(self, tmp_path: Path) -> None:
        action = _action(
            species=[
                {"label": "OH"},  # valid, no smiles
                {"smiles": "[CH3]"},  # missing label -> skipped
                "not-a-dict",  # skipped
                {"label": "H2O2", "smiles": "OO"},
            ]
        )
        payload = build_arc_input(_campaign(tmp_path), action)
        assert payload[ARC_LAYOUT.INPUT_SPECIES_KEY] == [
            {"label": "OH"},
            {"label": "H2O2", "smiles": "OO"},
        ]

    def test_action_species_all_invalid_falls_back_to_mixture(self, tmp_path: Path) -> None:
        action = _action(species=[{"smiles": "[CH3]"}, "not-a-dict"])
        payload = build_arc_input(_campaign(tmp_path), action)
        labels = [s["label"] for s in payload[ARC_LAYOUT.INPUT_SPECIES_KEY]]
        assert labels == ["OH", "CH3"]

    def test_action_reactions_skips_invalid_entries(self, tmp_path: Path) -> None:
        action = _action(
            reactions=[
                {"label": "A => B"},
                {"reactants": ["A"]},  # missing label -> skipped
                "not-a-dict",  # skipped
            ]
        )
        payload = build_arc_input(_campaign(tmp_path), action)
        assert payload[ARC_LAYOUT.INPUT_REACTIONS_KEY] == [{"label": "A => B"}]


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

    def test_defaults_missing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yml"
        path.write_text("project: x\n")
        info = read_arc_info_file(path)
        assert info["species"] == []
        assert info["reactions"] == []


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
        sel = _coerce_reaction_entry(
            {"label": "A => B", "reactant_labels": ["A"], "product_labels": ["B"], "success": True}
        )
        assert sel is not None
        assert sel.label == "A => B"
        assert sel.reactants == ["A"]
        assert sel.products == ["B"]

    def test_reaction_entry_missing_label(self) -> None:
        assert _coerce_reaction_entry({"reactants": ["A"]}) is None

    def test_reaction_entry_non_dict(self) -> None:
        assert _coerce_reaction_entry("nope") is None


class TestCountConverged:
    def test_none_output_returns_zero(self) -> None:
        assert _count_converged(None) == 0

    def test_empty_output_returns_zero(self) -> None:
        assert _count_converged({}) == 0


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
        diag = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert sorted(s.label for s in diag.species_to_compute) == ["CH3", "OH"]

    def test_normalize_records_success_in_reason(self, input_dict: dict[str, object]) -> None:
        diag = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        by_label = {s.label: s for s in diag.species_to_compute}
        assert "success=True" in (by_label["OH"].reason or "")

    def test_normalize_no_pdep_networks(self, input_dict: dict[str, object]) -> None:
        diag = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.pdep_networks_to_compute == []

    def test_normalize_extracts_lot(self, input_dict: dict[str, object]) -> None:
        diag = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.level_of_theory == MOCK_LEVEL_OF_THEORY

    def test_normalize_records_metadata(self, input_dict: dict[str, object]) -> None:
        diag = normalize_arc_outputs(FIXTURE_ROOT, input_dict, campaign_id="c", run_id="r")
        assert diag.tool_metadata["adapter"] == "arc"
        assert diag.tool_metadata["species_count"] == 2
        assert diag.tool_metadata["converged_species_count"] == 2
        assert diag.tool_metadata["arc_version"] == "1.1.0"

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
        diag = normalize_arc_outputs(project, {"project": "proj"}, campaign_id="c", run_id="r")
        assert [s.label for s in diag.species_to_compute] == ["OH"]
        assert [r.label for r in diag.reactions_to_compute] == ["OH + CH3 => CH4 + O"]
        assert diag.tool_metadata["reaction_count"] == 1

    def test_output_only_without_info_file(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        (project / "output").mkdir(parents=True)
        (project / "output" / "output.yml").write_text(yaml.safe_dump({"arc_version": "0.0.1", "species": []}))
        diag = normalize_arc_outputs(project, {"project": "proj"}, campaign_id="c", run_id="r")
        assert diag.species_to_compute == []
        assert diag.tool_metadata["arc_version"] == "0.0.1"


# ---------------------------------------------------------------------------
# Adapter failure paths (deterministic, no ARC needed)
# ---------------------------------------------------------------------------


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

    def test_subprocess_raises_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import arc as arc_module

        monkeypatch.setattr(arc_module, "_find_arc_executable", lambda: ["arc-stub"])

        def _raise_os_error(*_a: object, **_k: object) -> None:
            raise OSError("no such executable")

        monkeypatch.setattr(arc_module.subprocess, "run", _raise_os_error)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = ARCAdapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert diagnostics is None

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

        monkeypatch.setattr(arc_module.subprocess, "run", _raise_timeout)
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

        def _fake_run(command: list[str], cwd: Path, **kwargs: object) -> _Completed:
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

        monkeypatch.setattr(arc_module.subprocess, "run", _fake_run)
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

    def test_estimate_cost_uses_declared_estimate(self) -> None:
        assert ARCAdapter().estimate_cost(_action()) == 1.0

    def test_estimate_cost_falls_back_to_species_count(self) -> None:
        action = _action(species=[{"label": "A"}, {"label": "B"}], reactions=[{"label": "A => B"}])
        action = action.model_copy(update={"estimated_cpu_hours": 0.0})
        assert ARCAdapter().estimate_cost(action) == 4.0  # 2 species + 2*1 reaction


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
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED)
        if run.status == RunStatus.SUCCEEDED:
            assert diagnostics is not None
            assert diagnostics.run_id == run.run_id
