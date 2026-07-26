"""Tests for the T3 adapter — real contract + golden fixture.

The pure-Python helpers (input building, output walking, normalization,
LOT extraction) are tested unconditionally against:
    1. unit fixtures defined inline
    2. a captured real T3 fixture under tests/fixtures/t3/sample_project/

The actual subprocess execution path is exercised only when T3 is
truly importable in the current environment (``is_t3_importable()``).
That path runs in the heavy CI lane and is skipped locally if the
upstream ARC distutils blocker is in effect.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from carmel.adapters.t3 import (
    T3_LAYOUT,
    T3Adapter,
    _coerce_reaction_entry,
    _coerce_species_entry,
    _discover_pdep_networks,
    _find_t3_executable,
    _resolve_t3_python,
    _t3_conda_env_error,
    _t3_python_command,
    _t3_version,
    _walk_iterations,
    arc_info_filename,
    build_t3_input,
    extract_level_of_theory,
    is_t3_importable,
    is_t3_installed,
    normalize_t3_outputs,
    read_t3_info_file,
    write_t3_input_file,
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

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "t3" / "sample_project"

requires_t3 = pytest.mark.skipif(
    not is_t3_importable(),
    reason="T3 not actually importable (likely the ARC distutils blocker on Python 3.12)",
)


def _campaign(workspace_root: Path) -> Campaign:
    return Campaign(
        campaign_id="test-id",
        workspace_root=workspace_root,
        input=CampaignInput(
            workspace_name="ethanol_combustion",
            initial_mixture=InitialMixture(
                components=[
                    MixtureComponent(species="C2H5OH", mole_fraction=0.05, smiles="CCO"),
                    MixtureComponent(species="O2", mole_fraction=0.20, smiles="[O][O]"),
                    MixtureComponent(species="N2", mole_fraction=0.75, smiles="N#N"),
                ]
            ),
            target_observables=[
                TargetObservable(name="ignition_delay"),
                TargetObservable(name="species_profile", species="OH"),
            ],
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


def _action() -> PlannedAction:
    return PlannedAction(
        action_id="action-1",
        kind=ActionKind.T3_RUN,
        description="run T3",
        estimated_cpu_hours=1.0,
        rationale="test",
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
    )


# ---------------------------------------------------------------------------
# Discovery / availability helpers
# ---------------------------------------------------------------------------


class TestT3Discovery:
    def test_is_t3_installed_returns_bool(self) -> None:
        assert isinstance(is_t3_installed(), bool)

    def test_is_t3_importable_returns_bool(self) -> None:
        assert isinstance(is_t3_importable(), bool)

    def test_importable_implies_installed(self) -> None:
        if is_t3_importable():
            assert is_t3_installed()

    def test_find_executable_returns_list_or_none(self) -> None:
        result = _find_t3_executable()
        prefix = _t3_python_command()
        assert result is None or (isinstance(result, list) and result[: len(prefix)] == prefix)


# ---------------------------------------------------------------------------
# Input building (real T3 contract)
# ---------------------------------------------------------------------------


class TestBuildT3Input:
    def test_top_level_keys_match_real_t3_schema(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        assert set(payload.keys()) >= {
            T3_LAYOUT.INPUT_PROJECT_KEY,
            T3_LAYOUT.INPUT_T3_KEY,
            T3_LAYOUT.INPUT_RMG_KEY,
            T3_LAYOUT.INPUT_QM_KEY,
        }

    def test_project_is_workspace_name(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        assert payload[T3_LAYOUT.INPUT_PROJECT_KEY] == "ethanol_combustion"

    def test_species_use_real_t3_keys(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        species = payload[T3_LAYOUT.INPUT_RMG_KEY]["species"]
        assert all("label" in s and "concentration" in s for s in species)
        assert all("smiles" in s for s in species)

    def test_no_observable_flag_when_species_not_in_mixture(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        species = payload[T3_LAYOUT.INPUT_RMG_KEY]["species"]
        assert all(not s.get("SA_observable", False) for s in species)

    def test_observable_species_flag_when_in_mixture(self, tmp_path: Path) -> None:
        c = _campaign(tmp_path)
        c.input.target_observables[0] = TargetObservable(name="ignition_delay", species="O2")
        payload = build_t3_input(c)
        species = payload[T3_LAYOUT.INPUT_RMG_KEY]["species"]
        o2 = next(s for s in species if s["label"] == "O2")
        assert o2["SA_observable"] is True

    def test_reactor_uses_real_t3_type(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        reactors = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"]
        assert reactors[0]["type"] == "gas batch constant T P"

    def test_reactor_temperature_is_range_list(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        reactor = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert reactor["T"] == [800.0, 1200.0]
        assert reactor["P"] == [1.0, 5.0]

    def test_reactor_termination_time_present(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        reactor = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert reactor["termination_time"] == [1.0, "s"]

    def test_qm_block_has_lot_and_adapter(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        qm = payload[T3_LAYOUT.INPUT_QM_KEY]
        assert T3_LAYOUT.QM_LOT_KEY in qm
        assert T3_LAYOUT.QM_ADAPTER_KEY in qm
        assert qm[T3_LAYOUT.QM_ADAPTER_KEY] == "ARC"

    def test_t3_block_has_sensitivity(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        t3_block = payload[T3_LAYOUT.INPUT_T3_KEY]
        assert "sensitivity" in t3_block
        assert "options" in t3_block


class TestWriteT3InputFile:
    def test_writes_atomically(self, tmp_path: Path) -> None:
        payload = {"project": "x", "t3": {}, "rmg": {}, "qm": {}}
        path = write_t3_input_file(tmp_path, payload)
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["project"] == "x"


# ---------------------------------------------------------------------------
# Reading and parsing <project>_info.yml
# ---------------------------------------------------------------------------


class TestReadT3InfoFile:
    def test_reads_real_iteration_1_fixture(self) -> None:
        info = read_t3_info_file(FIXTURE_ROOT / "iteration_1" / "ARC" / "functional_2_thermo_info.yml")
        assert info["reactions"] == []
        assert info["species"][0]["label"] == "Imipramine_1_peroxy_0"
        assert info["species"][0]["success"] is True

    def test_reads_real_iteration_2_fixture(self) -> None:
        info = read_t3_info_file(FIXTURE_ROOT / "iteration_2" / "ARC" / "functional_2_thermo_info.yml")
        assert len(info["species"]) == 2
        labels = [s["label"] for s in info["species"]]
        assert "imipramine_ol_2_ket_4" in labels
        assert "imipramine_ol_2_ket_5" in labels

    def test_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_t3_info_file(tmp_path / "nope.yml")

    def test_non_mapping_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("- list\n- only\n")
        with pytest.raises(ValueError, match="mapping"):
            read_t3_info_file(path)

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("species: [unterminated\n")
        with pytest.raises(ValueError, match="not valid YAML"):
            read_t3_info_file(path)


class TestArcInfoFilename:
    def test_resolves_from_top_level_project(self) -> None:
        assert arc_info_filename({"project": "myproj"}) == "myproj_info.yml"

    def test_qm_project_overrides_top_level_project(self) -> None:
        assert arc_info_filename({"project": "myproj", "qm": {"project": "override"}}) == "override_info.yml"

    def test_falls_back_to_top_level_when_qm_project_empty(self) -> None:
        assert arc_info_filename({"project": "myproj", "qm": {"project": ""}}) == "myproj_info.yml"

    def test_raises_when_no_project_resolvable(self) -> None:
        with pytest.raises(ValueError, match="Cannot resolve ARC info filename"):
            arc_info_filename({})

    def test_defaults_added_for_missing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yml"
        path.write_text("project: x\n")
        info = read_t3_info_file(path)
        assert info["species"] == []
        assert info["reactions"] == []


class TestCoerceEntries:
    def test_species_entry_with_success(self) -> None:
        sel = _coerce_species_entry({"label": "OH", "success": True}, iteration=2)
        assert sel is not None
        assert sel.label == "OH"
        assert "iteration 2" in (sel.reason or "")
        assert "success=True" in (sel.reason or "")

    def test_species_entry_with_smiles(self) -> None:
        sel = _coerce_species_entry({"label": "OH", "smiles": "[OH]"}, iteration=1)
        assert sel is not None
        assert sel.smiles == "[OH]"

    def test_species_entry_missing_label(self) -> None:
        assert _coerce_species_entry({"success": True}, iteration=1) is None

    def test_species_entry_not_dict(self) -> None:
        assert _coerce_species_entry("not a dict", iteration=1) is None

    def test_reaction_entry_with_label(self) -> None:
        sel = _coerce_reaction_entry(
            {"label": "r1", "reactants": ["A"], "products": ["B"], "success": False},
            iteration=1,
        )
        assert sel is not None
        assert sel.reactants == ["A"]
        assert sel.products == ["B"]

    def test_reaction_entry_uses_equation_fallback(self) -> None:
        sel = _coerce_reaction_entry({"equation": "A => B"}, iteration=1)
        assert sel is not None
        assert sel.label == "A => B"

    def test_reaction_entry_missing_label(self) -> None:
        assert _coerce_reaction_entry({"reactants": ["A"]}, iteration=1) is None


# ---------------------------------------------------------------------------
# Project walking and pdep discovery
# ---------------------------------------------------------------------------


class TestWalkIterations:
    def test_walks_real_fixture(self) -> None:
        iters = _walk_iterations(FIXTURE_ROOT)
        assert len(iters) == 2
        assert [p.name for p in iters] == ["iteration_1", "iteration_2"]

    def test_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert _walk_iterations(tmp_path / "missing") == []

    def test_sorted_by_iteration_index(self, tmp_path: Path) -> None:
        for i in [3, 1, 10, 2]:
            (tmp_path / f"iteration_{i}").mkdir()
        iters = _walk_iterations(tmp_path)
        assert [p.name for p in iters] == [
            "iteration_1",
            "iteration_2",
            "iteration_3",
            "iteration_10",
        ]


class TestDiscoverPdepNetworks:
    def test_finds_real_pdep_files(self) -> None:
        nets = _discover_pdep_networks(_walk_iterations(FIXTURE_ROOT))
        ids = sorted(n.network_id for n in nets)
        assert ids == ["network1_1", "network4_1"]

    def test_empty_when_no_pdep_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "iteration_1" / "ARC").mkdir(parents=True)
        assert _discover_pdep_networks(_walk_iterations(tmp_path)) == []


# ---------------------------------------------------------------------------
# Level-of-theory extraction
# ---------------------------------------------------------------------------


class TestExtractLevelOfTheory:
    def test_extracts_from_qm_block(self) -> None:
        d = {"qm": {"level_of_theory": "b3lyp/6-31g(d,p)"}}
        assert extract_level_of_theory(d) == "b3lyp/6-31g(d,p)"

    def test_returns_none_when_no_qm(self) -> None:
        assert extract_level_of_theory({"project": "x"}) is None

    def test_returns_none_when_no_lot_key(self) -> None:
        assert extract_level_of_theory({"qm": {"adapter": "ARC"}}) is None

    def test_returns_none_when_qm_not_mapping(self) -> None:
        assert extract_level_of_theory({"qm": "not a mapping"}) is None


# ---------------------------------------------------------------------------
# Golden fixture: end-to-end normalize_t3_outputs against real T3 data
# ---------------------------------------------------------------------------


class TestGoldenFixture:
    """Validate the parser/normalizer against a captured real T3 project."""

    @pytest.fixture
    def input_dict(self) -> dict[str, object]:
        return yaml.safe_load((FIXTURE_ROOT / "input.yml").read_text())

    def test_input_yaml_loadable(self, input_dict: dict[str, object]) -> None:
        assert input_dict[T3_LAYOUT.INPUT_PROJECT_KEY] == "functional_2_thermo"

    def test_input_yaml_has_qm_lot(self, input_dict: dict[str, object]) -> None:
        lot = extract_level_of_theory(input_dict)
        assert lot == "gfn2"

    def test_normalize_aggregates_species_across_iterations(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        labels = sorted(s.label for s in diag.species_to_compute)
        assert labels == [
            "Imipramine_1_peroxy_0",
            "imipramine_ol_2_ket_4",
            "imipramine_ol_2_ket_5",
        ]

    def test_normalize_records_iteration_in_reason(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        by_label = {s.label: s for s in diag.species_to_compute}
        assert "iteration 1" in (by_label["Imipramine_1_peroxy_0"].reason or "")
        assert "iteration 2" in (by_label["imipramine_ol_2_ket_4"].reason or "")
        assert "iteration 2" in (by_label["imipramine_ol_2_ket_5"].reason or "")
        assert "success=False" in (by_label["imipramine_ol_2_ket_5"].reason or "")

    def test_normalize_finds_pdep_networks(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        ids = sorted(n.network_id for n in diag.pdep_networks_to_compute)
        assert ids == ["network1_1", "network4_1"]

    def test_normalize_extracts_lot(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        assert diag.level_of_theory == "gfn2"

    def test_normalize_records_iteration_count_in_metadata(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        assert diag.tool_metadata["iteration_count"] == 2
        assert diag.tool_metadata["pdep_network_count"] == 2

    def test_normalize_no_reactions_in_fixture(self, input_dict: dict[str, object]) -> None:
        diag = normalize_t3_outputs(
            project_dir=FIXTURE_ROOT,
            input_dict=input_dict,
            campaign_id="cgold",
            run_id="rgold",
        )
        assert diag.reactions_to_compute == []

    def test_normalize_raises_for_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No T3 iteration"):
            normalize_t3_outputs(
                project_dir=tmp_path,
                input_dict={},
                campaign_id="c",
                run_id="r",
            )


# ---------------------------------------------------------------------------
# Adapter failure paths (no live T3 needed)
# ---------------------------------------------------------------------------


class TestT3AdapterFailures:
    def test_tool_not_found_records_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: None)
        ws = tmp_path / "ws"
        ws.mkdir()
        adapter = T3Adapter()
        run, diagnostics = adapter.run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert run.input_path is not None
        assert run.input_path.exists()

    def test_subprocess_nonzero_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["false"])
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "exited with code" in (run.error_message or "")
        assert diagnostics is None

    def test_subprocess_succeeds_but_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        # `true` exits 0 but writes nothing — normalize_t3_outputs should fail
        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["true"])
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INVALID_OUTPUT
        assert diagnostics is None


class TestT3CondaEnvError:
    """Unit tests for ``_t3_conda_env_error()`` in isolation."""

    def test_env_var_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("T3_CONDA_ENV", raising=False)
        assert _t3_conda_env_error() is None

    def test_conda_missing_from_path_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: None)
        error = _t3_conda_env_error()
        assert error is not None
        assert "T3_CONDA_ENV" in error
        assert "conda" in error.lower()

    def test_named_env_does_not_exist_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "no_such_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="EnvironmentLocationNotFound: could not find environment"
        )
        monkeypatch.setattr(t3_module.subprocess, "run", lambda *a, **k: completed)
        error = _t3_conda_env_error()
        assert error is not None
        assert "no_such_env" in error
        assert "EnvironmentLocationNotFound" in error

    def test_conda_run_raises_oserror_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/bin/conda")

        def _raise(*args: object, **kwargs: object) -> None:
            raise OSError("boom")

        monkeypatch.setattr(t3_module.subprocess, "run", _raise)
        error = _t3_conda_env_error()
        assert error is not None
        assert "t3_env" in error

    def test_env_usable_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        monkeypatch.setattr(t3_module.subprocess, "run", lambda *a, **k: completed)
        assert _t3_conda_env_error() is None


class TestT3AdapterCondaEnvFailures:
    """Adapter-level behavior when ``$T3_CONDA_ENV`` is set but unusable.

    Both scenarios must be a clean, typed ``FailureCode.TOOL_NOT_FOUND``
    failure rather than a silent fallback onto Carmel's own interpreter
    (finding 1) or an indistinguishable "T3 not importable" result that
    would otherwise let a broken CI conda env pass as a harmless skip
    (finding 2).
    """

    def test_conda_missing_from_path_records_tool_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: None)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert run.input_path is not None
        assert run.input_path.exists()
        assert "T3_CONDA_ENV" in (run.error_message or "")
        assert "conda" in (run.error_message or "").lower()

    def test_named_env_does_not_exist_records_tool_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "no_such_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="EnvironmentLocationNotFound: could not find environment"
        )
        monkeypatch.setattr(t3_module.subprocess, "run", lambda *a, **k: completed)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TOOL_NOT_FOUND
        assert diagnostics is None
        assert "no_such_env" in (run.error_message or "")


class TestT3VersionProbe:
    """Tests for the importability and version probes.

    Both probes must interrogate the *resolved T3 interpreter* in a
    subprocess (never Carmel's own in-process ``importlib``), since under
    the three-env deployment model Carmel's own interpreter never has
    ``t3`` installed. Pointing ``T3_PYTHON`` at ``sys.executable`` collapses
    this back to the single-env case, letting these tests craft the probed
    module via ``sys.path`` manipulation without a real T3 checkout.
    """

    def test_importable_true_when_resolved_interpreter_can_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "t3.py").write_text("__version__ = '1.2.3'\n")
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert is_t3_importable() is True

    def test_importable_false_when_resolved_interpreter_cannot_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # empty dir: no t3 module here
        assert is_t3_importable() is False

    def test_importable_false_when_interpreter_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "no-such-interpreter"
        monkeypatch.setattr("carmel.adapters.t3._resolve_t3_python", lambda: str(missing))
        assert is_t3_importable() is False

    def test_importable_false_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        def _timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="t3-probe", timeout=1.0)

        monkeypatch.setattr(t3_module.subprocess, "run", _timeout)
        assert is_t3_importable() is False

    def test_version_returns_module_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "t3.py").write_text("__version__ = '1.2.3'\n")
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert _t3_version() == "1.2.3"

    def test_version_none_when_module_has_no_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "t3.py").write_text("# no __version__ attribute\n")
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        assert _t3_version() is None

    def test_version_none_when_not_importable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # empty dir: no t3 module here
        assert _t3_version() is None

    def test_version_none_when_interpreter_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "no-such-interpreter"
        monkeypatch.setattr("carmel.adapters.t3._resolve_t3_python", lambda: str(missing))
        assert _t3_version() is None


class TestFindT3Executable:
    """Tests for T3 executable discovery precedence."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.delenv("T3_PATH", raising=False)
        monkeypatch.delenv("T3_PYTHON", raising=False)
        monkeypatch.delenv("T3_CONDA_ENV", raising=False)
        monkeypatch.setattr(t3_module.importlib.util, "find_spec", lambda _name: None)
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: None)
        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: False)

    def test_env_path_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text("# T3")
        monkeypatch.setenv("T3_PATH", str(tmp_path))
        assert _find_t3_executable() == [sys.executable, str(tmp_path / T3_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_env_path_ignored_when_script_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("T3_PATH", str(tmp_path))
        assert _find_t3_executable() is None

    def test_falls_back_to_package_sibling_script(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        repo_root = tmp_path / "T3"
        pkg = repo_root / "t3"
        pkg.mkdir(parents=True)
        (repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text("# T3")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(t3_module.importlib.util, "find_spec", lambda _name: spec)
        assert _find_t3_executable() == [sys.executable, str(repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_package_sibling_skipped_when_script_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        pkg = tmp_path / "T3" / "t3"
        pkg.mkdir(parents=True)
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(t3_module.importlib.util, "find_spec", lambda _name: spec)
        assert _find_t3_executable() is None

    def test_falls_back_to_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/local/bin/T3.py")
        assert _find_t3_executable() == [sys.executable, "/usr/local/bin/T3.py"]

    def test_falls_back_to_module_invocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: True)
        assert _find_t3_executable() == [sys.executable, "-m", T3_LAYOUT.EXECUTABLE_MODULE]

    def test_returns_none_when_nothing_found(self) -> None:
        assert _find_t3_executable() is None

    def test_env_path_uses_t3_python_over_sys_executable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text("# T3")
        monkeypatch.setenv("T3_PATH", str(tmp_path))
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        result = _find_t3_executable()
        assert result == [sys.executable, str(tmp_path / T3_LAYOUT.EXECUTABLE_SCRIPT)]
        assert result is not None and result[0] != "python"  # sanity: it's a real path, not the fallback name

    def test_package_sibling_uses_t3_python_over_sys_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        repo_root = tmp_path / "T3"
        pkg = repo_root / "t3"
        pkg.mkdir(parents=True)
        (repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text("# T3")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(t3_module.importlib.util, "find_spec", lambda _name: spec)
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        assert _find_t3_executable() == [sys.executable, str(repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT)]

    def test_which_uses_t3_python_over_sys_executable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/local/bin/T3.py")
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        assert _find_t3_executable() == [sys.executable, "/usr/local/bin/T3.py"]

    def test_module_invocation_uses_t3_python_over_sys_executable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: True)
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        assert _find_t3_executable() == [sys.executable, "-m", T3_LAYOUT.EXECUTABLE_MODULE]

    def test_env_path_carries_conda_run_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        script = tmp_path / T3_LAYOUT.EXECUTABLE_SCRIPT
        script.write_text("# T3")
        monkeypatch.setenv("T3_PATH", str(tmp_path))
        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(
            t3_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_t3_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "t3_env",
            "--no-capture-output",
            "python",
            str(script),
        ]

    def test_module_invocation_carries_conda_run_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: True)
        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(
            t3_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_t3_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "t3_env",
            "--no-capture-output",
            "python",
            "-m",
            T3_LAYOUT.EXECUTABLE_MODULE,
        ]

    def test_conda_env_set_ignores_carmel_own_package_sibling_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test: with $T3_CONDA_ENV set, Carmel must not discover
        a T3.py sitting next to *Carmel's own* importable ``t3`` package
        and then execute that script inside the named conda environment —
        the two may be entirely different checkouts. The named conda
        environment is authoritative; only $T3_PATH or `-m T3` inside it
        may be used.
        """
        from carmel.adapters import t3 as t3_module

        repo_root = tmp_path / "carmel_side_T3"
        pkg = repo_root / "t3"
        pkg.mkdir(parents=True)
        (repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text("# wrong-checkout T3")
        spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
        monkeypatch.setattr(t3_module.importlib.util, "find_spec", lambda _name: spec)
        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: True)
        monkeypatch.setattr(
            t3_module.shutil,
            "which",
            lambda name: "/opt/conda/bin/conda" if name == "conda" else None,
        )
        assert _find_t3_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "t3_env",
            "--no-capture-output",
            "python",
            "-m",
            T3_LAYOUT.EXECUTABLE_MODULE,
        ]

    def test_conda_env_set_ignores_carmel_own_which_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Companion regression test to the one above, for the
        ``shutil.which(T3.py)`` discovery step rather than the
        ``importlib.util.find_spec`` one.
        """
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module, "is_t3_importable", lambda: True)
        monkeypatch.setattr(
            t3_module.shutil,
            "which",
            lambda name: (
                "/opt/conda/bin/conda"
                if name == "conda"
                else ("/usr/local/bin/T3.py" if name == T3_LAYOUT.EXECUTABLE_SCRIPT else None)
            ),
        )
        assert _find_t3_executable() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "t3_env",
            "--no-capture-output",
            "python",
            "-m",
            T3_LAYOUT.EXECUTABLE_MODULE,
        ]


class TestResolveT3Python:
    """Tests for the ``$T3_PYTHON`` resolution helper directly."""

    def test_unset_falls_back_to_sys_executable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("T3_PYTHON", raising=False)
        assert _resolve_t3_python() == sys.executable

    def test_set_to_real_executable_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        assert _resolve_t3_python() == sys.executable

    def test_set_to_nonexistent_path_falls_back_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        warnings: list[tuple[Any, ...]] = []
        monkeypatch.setattr(t3_module._log, "warning", lambda *args: warnings.append(args))
        missing = tmp_path / "no-such-interpreter"
        monkeypatch.setenv("T3_PYTHON", str(missing))
        assert _resolve_t3_python() == sys.executable
        assert len(warnings) == 1
        assert warnings[0][0] % warnings[0][1:] == (
            f"T3_PYTHON is set to '{missing}' but is not an existing executable file; falling back to {sys.executable}"
        )

    def test_set_to_non_executable_file_falls_back_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        warnings: list[tuple[Any, ...]] = []
        monkeypatch.setattr(t3_module._log, "warning", lambda *args: warnings.append(args))
        not_executable = tmp_path / "not-executable"
        not_executable.write_text("not a real interpreter\n")
        not_executable.chmod(0o644)
        monkeypatch.setenv("T3_PYTHON", str(not_executable))
        assert _resolve_t3_python() == sys.executable
        assert len(warnings) == 1
        assert warnings[0][0] % warnings[0][1:] == (
            f"T3_PYTHON is set to '{not_executable}' but is not an existing "
            f"executable file; falling back to {sys.executable}"
        )


class TestT3PythonCommand:
    """Tests for the ``$T3_CONDA_ENV``-aware python command resolution."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("T3_CONDA_ENV", raising=False)
        monkeypatch.delenv("T3_PYTHON", raising=False)

    def test_conda_env_set_and_conda_on_path_uses_conda_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda name: "/opt/conda/bin/conda" if name == "conda" else None)
        assert _t3_python_command() == [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "t3_env",
            "--no-capture-output",
            "python",
        ]

    def test_conda_env_set_but_conda_not_on_path_falls_back_with_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_t3_python_command()`` itself stays non-raising and still falls
        back to ``[_resolve_t3_python()]`` in this scenario — it is
        low-level plumbing that must never throw. That fallback is *not*
        what actually runs T3 under the wrong interpreter, though: this
        exact scenario (``$T3_CONDA_ENV`` set, no ``conda`` on PATH) is
        independently detected by ``_t3_conda_env_error()``, which
        ``T3Adapter.run()`` checks before ever calling ``_t3_python_command()``
        for real, converting it into an explicit ``FailureCode.TOOL_NOT_FOUND``
        instead of silently launching Carmel's own interpreter as "T3" — see
        TestT3AdapterCondaEnvFailures.test_conda_missing_from_path_records_tool_not_found.
        """
        from carmel.adapters import t3 as t3_module

        # carmel's own logger sets `propagate = False` once `configure_logging()`
        # has run anywhere in the process (see carmel/logger.py), which would
        # silently swallow this warning from pytest's `caplog` (attached to the
        # root logger); assert on the log call directly instead, matching the
        # precedent in TestResolveT3Python.
        warnings: list[tuple[Any, ...]] = []
        monkeypatch.setattr(t3_module._log, "warning", lambda *args: warnings.append(args))
        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.delenv("T3_PYTHON", raising=False)
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: None)
        result = _t3_python_command()
        assert result == [_resolve_t3_python()]
        assert len(warnings) == 1
        message = warnings[0][0] % warnings[0][1:]
        assert "T3_CONDA_ENV" in message
        assert "conda" in message.lower()

        # The higher-level, adapter-facing check catches the same scenario
        # explicitly rather than relying on the silent low-level fallback.
        conda_error = _t3_conda_env_error()
        assert conda_error is not None
        assert "T3_CONDA_ENV" in conda_error
        assert "conda" in conda_error.lower()

    def test_conda_env_set_to_empty_string_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("T3_CONDA_ENV", "")
        monkeypatch.delenv("T3_PYTHON", raising=False)
        assert _t3_python_command() == [sys.executable]

    def test_conda_env_unset_and_t3_python_set_uses_t3_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        assert _t3_python_command() == [sys.executable]

    def test_conda_env_and_t3_python_both_unset_uses_sys_executable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("T3_PYTHON", raising=False)
        assert _t3_python_command() == [sys.executable]


class TestBuildT3InputOptionalFields:
    """Optional campaign fields must be omitted rather than emitted as null."""

    def test_reactor_without_residence_time_omits_termination(self, tmp_path: Path) -> None:
        campaign = _campaign(tmp_path)
        campaign.input.target_reactor_systems[0].residence_time_s = None
        reactor = build_t3_input(campaign)[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert "termination_time" not in reactor

    def test_species_without_smiles_omits_smiles(self, tmp_path: Path) -> None:
        campaign = _campaign(tmp_path)
        campaign.input.initial_mixture.components[0].smiles = None
        species = build_t3_input(campaign)[T3_LAYOUT.INPUT_RMG_KEY]["species"][0]
        assert "smiles" not in species


class TestAggregateEdgeCases:
    """ARC/ subdir presence is the discriminator for whether ARC ran an iteration.

    No ARC/ subdir at all is a legitimate skip (ARC simply did not run that
    iteration). An ARC/ subdir that exists but is missing or has an
    unparseable info file means ARC was launched but never finished writing
    its output — that is invalid output and must raise, not be swallowed
    into a warning.
    """

    def test_reaction_entry_not_dict(self) -> None:
        assert _coerce_reaction_entry("not-a-dict", 1) is None

    def test_walk_iterations_tolerates_non_numeric_suffix(self, tmp_path: Path) -> None:
        for name in ("iteration_2", "iteration_final", "iteration_1"):
            (tmp_path / name).mkdir()
        names = [p.name for p in _walk_iterations(tmp_path)]
        assert names == ["iteration_final", "iteration_1", "iteration_2"]

    def test_iteration_without_arc_subdir_is_skipped_silently(self, tmp_path: Path) -> None:
        """No ARC/ subdir at all means ARC legitimately did not run this iteration.

        This must be a silent, warning-free skip: T3 only creates
        ``iteration_N/ARC/`` inside ``run_arc()``, and only calls that when
        additional ARC calculations are actually required for that
        iteration, so an iteration with no ARC/ subdir is normal (e.g. a
        converged terminal iteration).
        """
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- label: OH\n  success: true\nreactions: []\n"
        )
        (tmp_path / "iteration_2").mkdir()
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert diagnostics.warnings == []

    def test_arc_subdir_without_info_file_raises(self, tmp_path: Path) -> None:
        """An ARC/ subdir with no info file means ARC crashed before saving it.

        ARC writes ``<project>_info.yml`` very late and unconditionally via
        ``save_project_info_file()``, outside any try/except, so its
        absence means the ARC run itself is broken.
        """
        (tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="iteration_1"):
            normalize_t3_outputs(project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r")

    def test_arc_info_file_non_mapping_raises(self, tmp_path: Path) -> None:
        """An ARC info file that parses to a bare list (not a mapping) must raise."""
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text("- this is a list, not a mapping\n")
        with pytest.raises(ValueError, match="iteration_1"):
            normalize_t3_outputs(project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r")

    def test_arc_info_file_malformed_yaml_raises(self, tmp_path: Path) -> None:
        """An ARC info file with malformed (unparseable) YAML must raise."""
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text("species: [unterminated\n")
        with pytest.raises(ValueError, match="iteration_1"):
            normalize_t3_outputs(project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r")

    def test_arc_info_file_with_empty_lists_is_legitimate(self, tmp_path: Path) -> None:
        """An info file that parses fine but has empty species/reactions contributes nothing."""
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text("species: []\nreactions: []\n")
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert diagnostics.species_to_compute == []
        assert diagnostics.reactions_to_compute == []
        assert diagnostics.warnings == []

    def test_trailing_rmg_only_iteration_has_no_arc_and_still_counts_networks(self, tmp_path: Path) -> None:
        """A trailing RMG-only iteration (no ARC/ dir) is skipped but its pdep networks still count."""
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- label: OH\n  success: true\nreactions: []\n"
        )
        pdep = tmp_path / "iteration_2" / T3_LAYOUT.RMG_SUBDIR / T3_LAYOUT.PDEP_SUBDIR
        pdep.mkdir(parents=True)
        (pdep / "network1_1.py").write_text("# network\n")
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert [n.network_id for n in diagnostics.pdep_networks_to_compute] == ["network1_1"]
        assert diagnostics.warnings == []

    def test_iteration_0_arc_only_no_rmg(self, tmp_path: Path) -> None:
        """An iteration_0-shaped dir with only an ARC/ subdir (no RMG/) parses correctly."""
        arc = tmp_path / "iteration_0" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- label: OH\n  success: true\nreactions: []\n"
        )
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert diagnostics.pdep_networks_to_compute == []

    def test_non_numeric_iteration_name_still_parsed(self, tmp_path: Path) -> None:
        arc = tmp_path / "iteration_final" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- label: OH\n  success: true\nreactions: []\n"
        )
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert "iteration 0" in (diagnostics.species_to_compute[0].reason or "")

    def test_unlabelled_entries_are_dropped(self, tmp_path: Path) -> None:
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- success: true\n- label: OH\n  success: true\n"
            "reactions:\n- success: true\n- label: r1\n  success: true\n"
        )
        diagnostics = normalize_t3_outputs(
            project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r"
        )
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert [r.label for r in diagnostics.reactions_to_compute] == ["r1"]

    def test_old_t3_info_filename_is_not_recognized(self, tmp_path: Path) -> None:
        """Regression guard: the old hardcoded 'T3_info.yml' name must not be accepted.

        ARC really writes ``<project>_info.yml``; a file left at the
        previously-invented literal name must be treated as absent and hard-fail.
        """
        arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        arc.mkdir(parents=True)
        (arc / "T3_info.yml").write_text("species:\n- label: OH\n  success: true\nreactions: []\n")
        with pytest.raises(FileNotFoundError, match="iteration_1"):
            normalize_t3_outputs(project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r")

    def test_bad_iteration_after_good_one_raises_rather_than_warns(self, tmp_path: Path) -> None:
        """A later iteration with ARC/ but no info file must raise even if an earlier one parsed fine.

        Partial ARC-output loss is invalid output for the whole run, not a
        soft warning tacked onto an otherwise-successful result.
        """
        good_arc = tmp_path / "iteration_1" / T3_LAYOUT.ARC_SUBDIR
        good_arc.mkdir(parents=True)
        (good_arc / arc_info_filename({"project": "myproj"})).write_text(
            "species:\n- label: OH\n  success: true\nreactions: []\n"
        )
        (tmp_path / "iteration_2" / T3_LAYOUT.ARC_SUBDIR).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="iteration_2"):
            normalize_t3_outputs(project_dir=tmp_path, input_dict={"project": "myproj"}, campaign_id="c", run_id="r")


class TestT3AdapterInputBuildFailure:
    def test_unbuildable_input_records_input_build_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        def _raise(_campaign: Campaign) -> dict[str, object]:
            raise ValueError("cannot build input")

        monkeypatch.setattr(t3_module, "build_t3_input", _raise)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INPUT_BUILD_ERROR
        assert "cannot build input" in (run.error_message or "")
        assert run.input_path is None
        assert diagnostics is None

    def test_write_input_oserror_records_input_build_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        def _raise(_run_dir: Path, _payload: dict[str, object]) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr(t3_module, "write_t3_input_file", _raise)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.INPUT_BUILD_ERROR
        assert "disk full" in (run.error_message or "")
        assert diagnostics is None

    def test_run_dir_mkdir_oserror_records_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        ws = tmp_path / "ws"
        ws.mkdir(parents=True)

        def _raise(self: Path, *args: object, **kwargs: object) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(t3_module.Path, "mkdir", _raise)
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "permission denied" in (run.error_message or "")
        assert run.input_path is None
        assert diagnostics is None


class TestT3AdapterSubprocessErrors:
    def test_timeout_records_timeout_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        def _timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="T3.py", timeout=1.0)

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["true"])
        monkeypatch.setattr(t3_module.subprocess, "run", _timeout)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TIMEOUT
        assert run.command is not None
        assert diagnostics is None

    def test_oserror_records_subprocess_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        def _oserror(*args: object, **kwargs: object) -> None:
            raise OSError("exec format error")

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["true"])
        monkeypatch.setattr(t3_module.subprocess, "run", _oserror)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "exec format error" in (run.error_message or "")
        assert diagnostics is None


class TestT3AdapterSuccessPath:
    """Drives the adapter's success path with a stand-in for the T3 executable.

    The stand-in writes the same ``iteration_N/ARC/<project>_info.yml`` layout
    real ARC produces (for the ``ethanol_combustion`` project used by
    ``_campaign()``), so the adapter's parse-and-record path is exercised end
    to end without requiring the full RMG/ARC stack. Per real ARC output,
    reaction entries carry only ``label`` (an equation string) and
    ``success`` — no ``reactants``/``products`` keys.
    """

    @staticmethod
    def _fake_t3_command() -> list[str]:
        script = (
            "import pathlib, sys\n"
            "print('hello from stdout')\n"
            "print('hello from stderr', file=sys.stderr)\n"
            "arc = pathlib.Path('iteration_1/ARC')\n"
            "arc.mkdir(parents=True, exist_ok=True)\n"
            "arc.joinpath('ethanol_combustion_info.yml').write_text(\n"
            "    'species:\\n- label: OH\\n  success: true\\n"
            'reactions:\\n- label: "C2H5OH + OH <=> C2H5O + H2O"\\n  success: true\\n\'\n'
            ")\n"
            "pdep = pathlib.Path('iteration_1/RMG/pdep')\n"
            "pdep.mkdir(parents=True, exist_ok=True)\n"
            "pdep.joinpath('network1_1.py').write_text('# network')\n"
        )
        return [sys.executable, "-c", script]

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_find_t3_executable", self._fake_t3_command)
        ws = tmp_path / "ws"
        ws.mkdir()
        return T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())

    def test_status_is_succeeded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run, diagnostics = self._run(tmp_path, monkeypatch)
        assert run.status == RunStatus.SUCCEEDED
        assert run.failure_code == FailureCode.NONE
        assert diagnostics is not None

    def test_diagnostics_link_back_to_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run, diagnostics = self._run(tmp_path, monkeypatch)
        assert diagnostics is not None
        assert diagnostics.run_id == run.run_id
        assert diagnostics.campaign_id == "test-id"

    def test_parsed_selections_reach_diagnostics(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _run_record, diagnostics = self._run(tmp_path, monkeypatch)
        assert diagnostics is not None
        assert [s.label for s in diagnostics.species_to_compute] == ["OH"]
        assert [r.label for r in diagnostics.reactions_to_compute] == ["C2H5OH + OH <=> C2H5O + H2O"]
        assert len(diagnostics.pdep_networks_to_compute) == 1

    def test_run_record_paths_and_timing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run, _diagnostics = self._run(tmp_path, monkeypatch)
        assert run.input_path is not None and run.input_path.exists()
        assert run.stdout_path is not None and run.stdout_path.exists()
        assert run.stderr_path is not None and run.stderr_path.exists()
        assert run.output_path is not None and run.output_path.is_dir()
        assert run.actual_cpu_hours is not None and run.actual_cpu_hours >= 0
        assert run.ended_at >= run.started_at

    def test_level_of_theory_carried_from_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run, diagnostics = self._run(tmp_path, monkeypatch)
        assert diagnostics is not None
        assert run.level_of_theory == diagnostics.level_of_theory
        assert run.level_of_theory is not None

    def test_stdout_and_stderr_captured_under_carmel_owned_filenames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard for C3: Carmel must never write its subprocess
        capture to ``t3.log`` — that path is T3's own log file, which T3's
        logger archives and unlinks on startup, silently discarding
        anything Carmel wrote there."""
        run, _diagnostics = self._run(tmp_path, monkeypatch)
        assert run.stdout_path is not None
        assert run.stderr_path is not None
        assert run.stdout_path.name != T3_LAYOUT.LOG_FILENAME
        assert run.stderr_path.name != T3_LAYOUT.LOG_FILENAME
        assert run.stdout_path != run.stderr_path
        assert "hello from stdout" in run.stdout_path.read_text(encoding="utf-8")
        assert "hello from stderr" in run.stderr_path.read_text(encoding="utf-8")
        assert "hello from stderr" not in run.stdout_path.read_text(encoding="utf-8")


class TestT3AdapterStreamContract:
    """Behavioral test for the stream contract that ``conda run
    --no-capture-output`` provides in production: a nonzero child exit
    code propagates through exactly, and stdout/stderr land in separate
    files without cross-contamination. Driven through the adapter's own
    ``_find_t3_executable`` seam with a stub command so it never requires
    a real conda environment.
    """

    @staticmethod
    def _stub_command(exit_code: int) -> list[str]:
        script = (
            "import sys\n"
            "print('stub stdout line')\n"
            "print('stub stderr line', file=sys.stderr)\n"
            f"sys.exit({exit_code})\n"
        )
        return [sys.executable, "-c", script]

    def test_nonzero_exit_propagates_and_streams_stay_separate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: self._stub_command(17))
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())

        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.SUBPROCESS_ERROR
        assert "17" in (run.error_message or "")
        assert diagnostics is None

        assert run.stdout_path is not None and run.stdout_path.exists()
        assert run.stderr_path is not None and run.stderr_path.exists()
        stdout_text = run.stdout_path.read_text(encoding="utf-8")
        stderr_text = run.stderr_path.read_text(encoding="utf-8")
        assert "stub stdout line" in stdout_text
        assert "stub stderr line" not in stdout_text
        assert "stub stderr line" in stderr_text
        assert "stub stdout line" not in stderr_text


class TestT3AdapterRealSubprocess:
    """End-to-end subprocess tests — only run when T3 is actually importable."""

    @requires_t3
    def test_run_does_not_crash(self, tmp_path: Path) -> None:
        """T3 needs the full RMG stack to actually converge, so we don't
        assert SUCCEEDED here — a ground-truth local run against real T3
        launched RMG, ran it for ~50s to non-convergence, and then T3
        itself raised inside its own Cantera-fixing step (no
        ``chem_annotated.yaml`` to fix), which Carmel correctly records as
        FAILED. That is the expected *healthy* outcome of this test: T3 was
        genuinely launched and ran a real iteration. What must not happen
        is a tautological pass where nothing was actually launched (e.g.
        ``conda run`` failing immediately, or the wrong interpreter being
        used) yet the adapter still reports FAILED and this test still
        goes green — so we assert concrete evidence of a real T3
        invocation, not just a plausible-looking status.
        """
        ws = tmp_path / "ws"
        ws.mkdir()
        adapter = T3Adapter(submission_mode=SubmissionMode.SUBPROCESS)
        run, diagnostics = adapter.run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED)

        assert run.input_path is not None
        assert run.input_path.exists()
        run_dir = run.input_path.parent

        # T3 actually started an iteration and wrote its own log — proof
        # that a real T3 process ran, not just that some command exited.
        assert (run_dir / "iteration_1").exists()
        assert (run_dir / T3_LAYOUT.LOG_FILENAME).exists()

        combined_output = ""
        if run.stdout_path is not None and run.stdout_path.exists():
            combined_output += run.stdout_path.read_text(encoding="utf-8")
        if run.stderr_path is not None and run.stderr_path.exists():
            combined_output += run.stderr_path.read_text(encoding="utf-8")
        for signature in (
            "No module named",
            "can't open file",
            "command not found",
            "Not a conda environment",
        ):
            assert signature not in combined_output, (
                f"launch-failure signature {signature!r} found in captured output — T3 was never actually launched"
            )

        if run.status == RunStatus.SUCCEEDED:
            assert diagnostics is not None
            assert diagnostics.run_id == run.run_id
