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

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
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
    _run_in_process_group,
    _t3_conda_env_error,
    _t3_python_command,
    _t3_version,
    _terminate_process_tree,
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

# Captured at import time, before the autouse fixture below can ever run, so
# TestT3AdapterRealSubprocess can restore the ambient CI launcher environment
# ($T3_CONDA_ENV / $T3_PYTHON / $T3_PATH) explicitly even though every other
# test in this module starts from a cleared one.
_AMBIENT_T3_ENV = {name: os.environ[name] for name in ("T3_CONDA_ENV", "T3_PYTHON", "T3_PATH") if name in os.environ}

# Evaluated at collection time (module import), deliberately *before* the
# autouse fixture below can clear anything — this must keep reading the real
# ambient environment so the skip decision matches what
# TestT3AdapterRealSubprocess will actually see. Do not "fix" this by moving
# it after the fixture or by reading _AMBIENT_T3_ENV instead: pytest marks
# are evaluated once, at collection, outside of any test's fixture scope, so
# there is no fixture ordering to fight here.
requires_t3 = pytest.mark.skipif(
    not is_t3_importable(),
    reason="T3 not actually importable (likely the ARC distutils blocker on Python 3.12)",
)


@pytest.fixture(autouse=True)
def _clear_ambient_t3_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear T3 launcher env vars so every test starts from a known-empty env.

    In CI, the workflow exports ``$T3_CONDA_ENV``, ``$T3_PYTHON``, and
    ``$T3_PATH`` into the job environment so the real T3 subprocess test can
    launch T3. Every *other* test in this module that deliberately breaks or
    monkeypatches one part of the resolution chain (e.g. forcing
    ``shutil.which`` to return None, or pointing ``$T3_PYTHON`` at a missing
    interpreter) must not have its intent overridden by those ambient values
    winning at a higher precedence step. So each test here starts from a
    cleared environment and opts in explicitly to whatever env var it needs.

    ``TestT3AdapterRealSubprocess`` is exempted: it genuinely needs the
    ambient CI launcher environment, which is why it's captured into
    ``_AMBIENT_T3_ENV`` at module import time (before this fixture could ever
    clear it) and restored here explicitly for that class only.
    """
    monkeypatch.delenv("T3_CONDA_ENV", raising=False)
    monkeypatch.delenv("T3_PYTHON", raising=False)
    monkeypatch.delenv("T3_PATH", raising=False)
    if request.cls is not None and request.cls.__name__ == "TestT3AdapterRealSubprocess":
        for name, value in _AMBIENT_T3_ENV.items():
            monkeypatch.setenv(name, value)


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
                TargetObservable(name="species_profile", species="OH", smiles="[OH]"),
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

    def test_importable_and_installed_probe_different_environments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``is_t3_importable() => is_t3_installed()`` is not a real invariant.

        This test used to assert that implication. Under the three-env
        deployment model it is false by design, not a bug: ``is_t3_importable``
        probes whether ``t3`` imports inside the *resolved T3 interpreter*
        (which may live in a separate conda env named by ``$T3_CONDA_ENV`` or
        pointed to by ``$T3_PYTHON``), while ``is_t3_installed`` calls
        ``importlib.util.find_spec("t3")`` in *Carmel's own process*. In the
        normal three-env deployment, T3 is importable in its own env while
        genuinely absent from Carmel's env, so the old invariant would fail
        honestly, not spuriously.

        Rather than delete the test outright, this rewrites it to demonstrate
        the real, useful fact: the two probes can and do diverge, because
        they answer questions about different environments. If a future
        reader is tempted to "restore" the implication, don't — it contradicts
        the three-env model by construction.
        """
        (tmp_path / "t3.py").write_text("__version__ = '9.9.9'\n")
        monkeypatch.setenv("T3_PYTHON", sys.executable)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # only the *resolved* subprocess sees this
        assert is_t3_importable() is True
        assert is_t3_installed() is False

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
        mixture_labels = {"C2H5OH", "O2", "N2"}
        assert all("smiles" in s for s in species if s["label"] in mixture_labels)

    def test_observable_not_in_mixture_emitted_as_new_species(self, tmp_path: Path) -> None:
        """An observable species that is not an initial-mixture component (e.g. a
        product) must still surface as an SA observable, emitted as an additional
        zero-concentration species entry — the default campaign's
        ``species_profile`` observable targets OH, which is not one of the
        C2H5OH/O2/N2 mixture components.
        """
        payload = build_t3_input(_campaign(tmp_path))
        species = payload[T3_LAYOUT.INPUT_RMG_KEY]["species"]
        oh = next(s for s in species if s["label"] == "OH")
        assert oh["SA_observable"] is True
        assert oh["concentration"] == 0.0
        mixture_labels = {"C2H5OH", "O2", "N2"}
        assert all(not s.get("SA_observable", False) for s in species if s["label"] in mixture_labels)

    def test_blank_observable_species_raises(self, tmp_path: Path) -> None:
        c = _campaign(tmp_path)
        c.input.target_observables[1] = TargetObservable(name="species_profile", species="   ")
        with pytest.raises(ValueError, match="blank species"):
            build_t3_input(c)

    def test_observable_not_in_mixture_without_smiles_raises(self, tmp_path: Path) -> None:
        """An observable species that is not an initial-mixture component and
        carries no smiles has no structural descriptor T3/RMG could resolve
        (bare labels are rejected — see ``t3/utils/writer.py``), so Carmel
        must fail at input-build time with a clear message instead of
        emitting an input T3 will refuse."""
        c = _campaign(tmp_path)
        c.input.target_observables[1] = TargetObservable(name="species_profile", species="OH")
        with pytest.raises(ValueError, match="OH.*smiles"):
            build_t3_input(c)

    def test_mixture_component_without_smiles_raises(self, tmp_path: Path) -> None:
        """A mixture component with no smiles has no structural descriptor
        either, so it must fail the same way rather than reach T3."""
        c = _campaign(tmp_path)
        c.input.initial_mixture.components[0] = MixtureComponent(species="C2H5OH", mole_fraction=0.05)
        with pytest.raises(ValueError, match="C2H5OH.*smiles"):
            build_t3_input(c)

    def test_observable_species_flag_when_in_mixture(self, tmp_path: Path) -> None:
        c = _campaign(tmp_path)
        c.input.target_observables[0] = TargetObservable(name="ignition_delay", species="O2")
        payload = build_t3_input(c)
        species = payload[T3_LAYOUT.INPUT_RMG_KEY]["species"]
        o2 = next(s for s in species if s["label"] == "O2")
        assert o2["SA_observable"] is True

    def test_reactor_uses_real_t3_type(self, tmp_path: Path) -> None:
        """T3/RMG only accept the literal strings 'gas batch constant T P' or
        'liquid batch constant T V' (t3/schema.py RMGReactor.check_reactor_type).
        Carmel's ReactorSystem is gas-phase only (it always carries
        pressure_range_bar), so every non-FLAME ReactorType maps onto the gas
        string — reactor_type must still be genuinely read, not hardcoded (see
        test_all_gas_reactor_types_map_to_real_t3_type and
        test_flame_reactor_type_raises for the discriminating coverage).
        """
        payload = build_t3_input(_campaign(tmp_path))
        reactors = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"]
        assert reactors[0]["type"] == "gas batch constant T P"

    @pytest.mark.parametrize(
        "reactor_type",
        [ReactorType.JSR, ReactorType.PFR, ReactorType.BATCH, ReactorType.SHOCK_TUBE, ReactorType.RCM],
    )
    def test_all_gas_reactor_types_map_to_real_t3_type(self, tmp_path: Path, reactor_type: ReactorType) -> None:
        campaign = _campaign(tmp_path)
        campaign.input.target_reactor_systems[0].reactor_type = reactor_type
        payload = build_t3_input(campaign)
        reactor = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert reactor["type"] == "gas batch constant T P"

    def test_flame_reactor_type_raises(self, tmp_path: Path) -> None:
        campaign = _campaign(tmp_path)
        campaign.input.target_reactor_systems[0].reactor_type = ReactorType.FLAME
        with pytest.raises(ValueError, match="reactor_type"):
            build_t3_input(campaign)

    def test_reactor_temperature_is_range_list(self, tmp_path: Path) -> None:
        payload = build_t3_input(_campaign(tmp_path))
        reactor = payload[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert reactor["T"] == [800.0, 1200.0]
        assert reactor["P"] == [1.0, 5.0]

    def test_reactor_scalar_when_range_collapses(self, tmp_path: Path) -> None:
        """A degenerate (min == max) temperature/pressure range must be emitted
        as a bare float, not a redundant [x, x] list."""
        campaign = _campaign(tmp_path)
        campaign.input.target_reactor_systems[0].temperature_range_K = (900.0, 900.0)
        campaign.input.target_reactor_systems[0].pressure_range_bar = (2.0, 2.0)
        reactor = build_t3_input(campaign)[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert reactor["T"] == 900.0
        assert reactor["P"] == 2.0

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

    def test_reaction_entry_matches_real_arc_contract(self) -> None:
        """ARC's real <project>_info.yml reaction entries only ever carry
        'label' and 'success' (see arc/main.py save_project_info_file) — never
        reactants/products. Confirm the real shape parses cleanly, resolving
        to empty reactant/product lists rather than erroring."""
        sel = _coerce_reaction_entry({"label": "r1", "success": True}, iteration=1)
        assert sel is not None
        assert sel.reactants == []
        assert sel.products == []


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

    def test_reason_names_iteration_not_rmg(self) -> None:
        """net_file is .../iteration_N/RMG/pdep/networkX.py — the discovery
        reason must name the iteration directory, not the intermediate 'RMG'
        directory."""
        nets = _discover_pdep_networks(_walk_iterations(FIXTURE_ROOT))
        net = next(n for n in nets if n.network_id == "network1_1")
        assert net.reason == "discovered at iteration_1"
        assert all(n.reason == "discovered at iteration_1" for n in nets)


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
        monkeypatch.setattr(t3_module, "_run_in_process_group", lambda *a, **k: completed)
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

        monkeypatch.setattr(t3_module, "_run_in_process_group", _raise)
        error = _t3_conda_env_error()
        assert error is not None
        assert "t3_env" in error

    def test_env_usable_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        monkeypatch.setenv("T3_CONDA_ENV", "t3_env")
        monkeypatch.setattr(t3_module.shutil, "which", lambda _name: "/usr/bin/conda")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        monkeypatch.setattr(t3_module, "_run_in_process_group", lambda *a, **k: completed)
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
        monkeypatch.setattr(t3_module, "_run_in_process_group", lambda *a, **k: completed)
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

        monkeypatch.setattr(t3_module, "_run_in_process_group", _timeout)
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
        """RMG requires at least one termination criterion (rmgpy/rmg/input.py
        raises if none is given) — omitting termination_time must not leave a
        reactor with zero criteria; it must fall back to termination_rate_ratio."""
        campaign = _campaign(tmp_path)
        campaign.input.target_reactor_systems[0].residence_time_s = None
        reactor = build_t3_input(campaign)[T3_LAYOUT.INPUT_RMG_KEY]["reactors"][0]
        assert "termination_time" not in reactor
        assert reactor["termination_rate_ratio"] == 0.1

    def test_species_without_smiles_raises(self, tmp_path: Path) -> None:
        """A mixture component with no smiles has no structural descriptor
        T3/RMG could resolve, so build_t3_input must fail loudly instead of
        silently omitting the field (which produced a T3 input rejected
        with ``ValueError: A species must have either an adjlist, smiles,
        or inchi descriptor.``)."""
        campaign = _campaign(tmp_path)
        campaign.input.initial_mixture.components[0].smiles = None
        with pytest.raises(ValueError, match="smiles"):
            build_t3_input(campaign)


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
        monkeypatch.setattr(t3_module, "_run_in_process_group", _timeout)
        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TIMEOUT
        assert run.command is not None
        assert diagnostics is None

    def test_the_adapter_passes_the_recorder_through_to_the_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adapter must actually forward ``on_process_start``, not accept it.

        Tested here, at :meth:`T3Adapter.run`, rather than only against
        ``_run_in_process_group``: a test of the helper alone passes
        happily when the adapter takes the argument and drops it on the
        floor. Every real run would then launch T3 with no recorded
        process group, and every service-level test would still be green,
        because those use adapter doubles that call the recorder
        themselves.
        """
        from carmel.adapters import t3 as t3_module

        seen: list[object] = []

        def _capture(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.append(kwargs.get("on_process_start"))
            raise OSError("stop here, the launch is all this checks")

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["true"])
        monkeypatch.setattr(t3_module, "_run_in_process_group", _capture)
        ws = tmp_path / "ws"
        ws.mkdir()

        def _recorder(_pgid: int, _argv: list[str]) -> None:  # pragma: no cover -- never invoked
            raise AssertionError("the stub never launches anything")

        T3Adapter().run(
            workspace_root=ws,
            campaign=_campaign(ws),
            action=_action(),
            on_process_start=_recorder,
        )
        # Later entries are the version probe, which is a separate launch
        # and deliberately unrecorded; the first is T3 itself.
        assert seen[0] is _recorder, "the adapter dropped the recorder instead of forwarding it"

    def test_oserror_records_subprocess_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        def _oserror(*args: object, **kwargs: object) -> None:
            raise OSError("exec format error")

        monkeypatch.setattr(t3_module, "_find_t3_executable", lambda: ["true"])
        monkeypatch.setattr(t3_module, "_run_in_process_group", _oserror)
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


def _t3_diagnostic(run: Any, run_dir: Path) -> str:
    """Build a diagnostic message explaining a real-T3-subprocess assertion failure.

    Used as the second argument to every assertion in
    ``TestT3AdapterRealSubprocess`` so a CI failure is self-explanatory
    instead of a bare ``assert False``. Includes the resolved command, the
    recorded ``failure_code``, a directory listing of the run directory, and
    the last ~2000 characters of both captured output streams.

    Args:
        run: The ``RunRecord`` returned by ``T3Adapter.run``.
        run_dir: The run's working directory (``run.input_path.parent``).

    Returns:
        A formatted, human-readable diagnostic string.
    """

    def _tail(path: Path | None, label: str) -> str:
        if path is None:
            return f"--- {label}: <no path recorded> ---"
        if not path.exists():
            return f"--- {label} ({path}): <file does not exist> ---"
        text = path.read_text(encoding="utf-8", errors="replace")
        tail = text[-2000:]
        return f"--- {label} ({path}, last {len(tail)} of {len(text)} chars) ---\n{tail}"

    if run_dir.exists():
        try:
            listing = "\n".join(sorted(p.name for p in run_dir.iterdir()))
        except OSError as e:
            listing = f"<could not list run dir: {e}>"
    else:
        listing = "<run dir does not exist>"

    return (
        f"command: {run.command}\n"
        f"failure_code: {run.failure_code}\n"
        f"--- run_dir listing ({run_dir}) ---\n{listing}\n"
        f"{_tail(run.stdout_path, 'carmel_stdout.log')}\n"
        f"{_tail(run.stderr_path, 'carmel_stderr.log')}"
    )


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
        run_dir_guess = ws / "runs"  # best-effort location for diagnostics if input_path was never set
        adapter = T3Adapter(submission_mode=SubmissionMode.SUBPROCESS)
        run, diagnostics = adapter.run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        run_dir = run.input_path.parent if run.input_path is not None else run_dir_guess
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED), _t3_diagnostic(run, run_dir)

        assert run.input_path is not None, _t3_diagnostic(run, run_dir)
        assert run.input_path.exists(), _t3_diagnostic(run, run_dir)
        run_dir = run.input_path.parent

        # T3 actually started an iteration and wrote its own log — proof
        # that a real T3 process ran, not just that some command exited.
        assert (run_dir / "iteration_1").exists(), _t3_diagnostic(run, run_dir)
        assert (run_dir / T3_LAYOUT.LOG_FILENAME).exists(), _t3_diagnostic(run, run_dir)

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
                f"launch-failure signature {signature!r} found in captured output — "
                f"T3 was never actually launched\n{_t3_diagnostic(run, run_dir)}"
            )

        if run.status == RunStatus.SUCCEEDED:
            assert diagnostics is not None, _t3_diagnostic(run, run_dir)
            assert diagnostics.run_id == run.run_id, _t3_diagnostic(run, run_dir)


class TestProcessTreeTermination:
    """A timeout must kill T3's children, not just the wrapper Carmel holds.

    These tests launch a real process tree and assert on the *grandchild*.
    Asserting that the call returned, or that the direct child died, is
    exactly what the old code already satisfied while T3 and RMG kept
    running: `subprocess.run(timeout=...)` killed the `conda run` wrapper
    and orphaned everything underneath it.
    """

    # parent spawns a grandchild that outlives it, then blocks forever.
    _TREE = (
        "import subprocess, sys, time, pathlib;"
        "gc = subprocess.Popen([sys.executable, '-c',"
        ' "import sys,time,pathlib;\\n"'
        ' "p = pathlib.Path(sys.argv[1]);\\n"'
        ' "[ (p.write_text(str(i)), time.sleep(0.05)) for i in range(2000) ]",'
        " sys.argv[1]]);"
        "pathlib.Path(sys.argv[2]).write_text(str(gc.pid));"
        "time.sleep(300)"
    )

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover -- not reachable as the same user
            return True
        return True

    @classmethod
    def _died_within(cls, pid: int, timeout_s: float = 15.0) -> bool:
        """Poll until *pid* is gone, or *timeout_s* elapses.

        Killing a tree is asynchronous in a way a single sample cannot
        capture: once the direct child is reaped, a descendant is
        reparented to init and stays a zombie — for which ``kill(pid, 0)``
        still succeeds — until init reaps it. Sampling instantly makes the
        assertion flaky; waiting does not weaken it, because a survivor
        never dies and still fails.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not cls._alive(pid):
                return True
            time.sleep(0.05)
        return False

    def _launch_tree(self, tmp_path: Path) -> tuple[Path, Path]:
        heartbeat = tmp_path / "heartbeat"
        pid_file = tmp_path / "grandchild.pid"
        return heartbeat, pid_file

    def _grandchild_pid(self, pid_file: Path, timeout_s: float = 20.0) -> int:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if pid_file.exists() and pid_file.read_text().strip():
                return int(pid_file.read_text().strip())
            time.sleep(0.05)
        raise AssertionError(f"the child never reported a grandchild pid at {pid_file}")

    def test_timeout_kills_the_grandchild(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression test for the orphan. Fails against the old code."""
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_KILL_GRACE_PERIOD_S", 1.0)
        heartbeat, pid_file = self._launch_tree(tmp_path)
        command = [sys.executable, "-c", self._TREE, str(heartbeat), str(pid_file)]

        with pytest.raises(subprocess.TimeoutExpired):
            _run_in_process_group(command, timeout=3.0)

        grandchild = self._grandchild_pid(pid_file)
        assert self._died_within(grandchild), (
            f"grandchild {grandchild} survived the timeout — Carmel would have recorded "
            "TIMEOUT while the real work kept running"
        )

    def test_timeout_stops_the_grandchild_writing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Liveness, not just the process table: the writes must actually stop."""
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module, "_KILL_GRACE_PERIOD_S", 1.0)
        heartbeat, pid_file = self._launch_tree(tmp_path)
        command = [sys.executable, "-c", self._TREE, str(heartbeat), str(pid_file)]

        with pytest.raises(subprocess.TimeoutExpired):
            _run_in_process_group(command, timeout=3.0)

        before = heartbeat.read_text() if heartbeat.exists() else ""
        time.sleep(1.0)
        after = heartbeat.read_text() if heartbeat.exists() else ""
        assert before == after, "the grandchild was still writing after the tree was killed"

    def test_the_launched_process_group_is_reported(self) -> None:
        """A supervisor needs the pgid to reap the tree if it is killed itself."""
        seen: list[tuple[int, list[str]]] = []
        command = [sys.executable, "-c", "print('ok')"]
        completed = _run_in_process_group(
            command,
            timeout=60.0,
            stdout=subprocess.PIPE,
            text=True,
            on_process_start=lambda pgid, argv: seen.append((pgid, argv)),
        )
        assert completed.returncode == 0
        assert len(seen) == 1
        pgid, argv = seen[0]
        assert argv == command
        assert pgid > 0

    def test_a_failing_recorder_stops_the_tree_it_could_not_record(self) -> None:
        """An untrackable tree is stopped, not left running unrecorded.

        This assertion was once the opposite, on the reasoning that a
        hypothetical future recovery should not cost a certain present
        run. It inverts once that recovery exists: a tree whose process
        group went unrecorded reads afterwards exactly like a run that
        never launched, so recovery offers to abandon a campaign whose T3
        is still writing into it.
        """
        seen_pid = 0

        def _explode(pgid: int, _argv: list[str]) -> None:
            nonlocal seen_pid
            seen_pid = pgid
            raise RuntimeError("could not record")

        with pytest.raises(RuntimeError, match="could not record"):
            _run_in_process_group(
                [sys.executable, "-c", "import time; time.sleep(300)"],
                timeout=60.0,
                stdout=subprocess.PIPE,
                text=True,
                on_process_start=_explode,
            )

        assert seen_pid > 0
        assert self._died_within(seen_pid), "the tree outlived the recorder that could not track it"

    def test_a_registered_tree_is_forgotten_even_when_recording_fails(self) -> None:
        """The early exit must not leak an entry into ``_LIVE_TREES``.

        Shutdown walks that set to kill whatever is still running, so a
        stale entry has it signalling a pid it no longer owns.
        """

        from carmel.adapters import t3 as t3_module

        def _explode(_pgid: int, _argv: list[str]) -> None:
            raise RuntimeError("could not record")

        before = set(t3_module._LIVE_TREES)
        with pytest.raises(RuntimeError, match="could not record"):
            _run_in_process_group(
                [sys.executable, "-c", "import time; time.sleep(300)"],
                timeout=60.0,
                stdout=subprocess.PIPE,
                text=True,
                on_process_start=_explode,
            )
        assert set(t3_module._LIVE_TREES) == before

    def test_normal_completion_is_unaffected(self, tmp_path: Path) -> None:
        completed = _run_in_process_group(
            [sys.executable, "-c", "print('ok')"],
            timeout=60.0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert completed.returncode == 0
        assert completed.stdout.strip() == "ok"

    def test_nonzero_exit_is_returned_not_raised(self) -> None:
        completed = _run_in_process_group([sys.executable, "-c", "raise SystemExit(3)"], timeout=60.0)
        assert completed.returncode == 3

    def test_child_runs_in_its_own_process_group(self, tmp_path: Path) -> None:
        """Without this, killpg would take down Carmel and its web server."""
        out = tmp_path / "pgid"
        _run_in_process_group(
            [sys.executable, "-c", f"import os, pathlib; pathlib.Path({str(out)!r}).write_text(str(os.getpgid(0)))"],
            timeout=60.0,
        )
        assert int(out.read_text()) != os.getpgid(0)

    def test_output_files_receive_the_child_output(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out.log"
        err_path = tmp_path / "err.log"
        with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
            _run_in_process_group(
                [sys.executable, "-c", "import sys; print('to-stdout'); print('to-stderr', file=sys.stderr)"],
                timeout=60.0,
                stdout=out,
                stderr=err,
            )
        assert "to-stdout" in out_path.read_text(encoding="utf-8")
        assert "to-stderr" in err_path.read_text(encoding="utf-8")

    def test_missing_executable_raises_oserror(self) -> None:
        with pytest.raises(OSError):
            _run_in_process_group(["/nonexistent/carmel-not-a-binary"], timeout=60.0)

    def test_terminate_is_safe_on_an_already_dead_child(self) -> None:
        """Reaping twice, or signalling a group that is already gone, must not raise."""
        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        proc.wait()
        _terminate_process_tree(proc)

    def test_sigkill_escalation_when_sigterm_is_ignored(self, tmp_path: Path) -> None:
        """A tree that traps SIGTERM must still not survive the timeout.

        The grandchild ignores SIGTERM too, on purpose: a version of this
        test with a childless child passes even against a kill that only
        ever reaches the direct child.
        """
        pid_file = tmp_path / "grandchild.pid"
        ignores_sigterm = (
            "import signal, subprocess, sys, time, pathlib;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "gc = subprocess.Popen([sys.executable, '-c',"
            ' "import signal, sys, time, pathlib\\n"'
            ' "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"'
            " \"pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid()))\\n\""
            ' "time.sleep(300)\\n",'
            " sys.argv[1]]);"
            "time.sleep(300)"
        )
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", ignores_sigterm, str(pid_file)],
            start_new_session=True,
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (pid_file.exists() and pid_file.read_text().strip()):
            time.sleep(0.05)
        grandchild = int(pid_file.read_text().strip())

        _terminate_process_tree(proc, grace_period_s=1.0)
        assert proc.poll() is not None, "a SIGTERM-ignoring child survived _terminate_process_tree"
        assert self._died_within(proc.pid)
        assert self._died_within(grandchild), "a SIGTERM-ignoring GRANDCHILD survived the escalation"


class TestAdapterTimeoutKillsTheToolTree:
    """End-to-end through T3Adapter.run, with a stub standing in for T3.py.

    The helper tests cover the mechanism; this covers the wiring — that the
    adapter actually routes its real invocation through it, with the real
    command building, cwd, and output files. The stub is a real script in a
    real subprocess that spawns a real grandchild, exactly as T3 spawns RMG.
    """

    _STUB_T3 = """
import os, subprocess, sys, time, pathlib
here = pathlib.Path(__file__).parent
grandchild = subprocess.Popen([
    sys.executable, "-c",
    "import sys, time, pathlib\\n"
    "p = pathlib.Path(sys.argv[1])\\n"
    "[(p.write_text(str(i)), time.sleep(0.05)) for i in range(2000)]\\n",
    str(here / "grandchild_heartbeat"),
])
(here / "grandchild.pid").write_text(str(grandchild.pid))
time.sleep(300)
"""

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    @staticmethod
    def _instant_action() -> PlannedAction:
        """Zero estimated hours, so the timeout is the buffer alone."""
        return PlannedAction(
            action_id="action-timeout",
            kind=ActionKind.T3_RUN,
            description="run T3",
            estimated_cpu_hours=0.0,
            rationale="test",
            approval_requirement=ApprovalRequirement.AUTO_APPROVED,
        )

    def test_timeout_leaves_no_surviving_tool_process(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.adapters import t3 as t3_module

        t3_home = tmp_path / "fake_t3"
        t3_home.mkdir()
        (t3_home / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text(self._STUB_T3, encoding="utf-8")

        monkeypatch.setenv("T3_PATH", str(t3_home))
        monkeypatch.delenv(T3_LAYOUT.T3_CONDA_ENV_VAR, raising=False)
        monkeypatch.delenv(T3_LAYOUT.T3_PYTHON_ENV_VAR, raising=False)
        monkeypatch.setattr(t3_module, "_RUN_TIMEOUT_BUFFER_S", 3.0)
        monkeypatch.setattr(t3_module, "_KILL_GRACE_PERIOD_S", 1.0)

        ws = tmp_path / "ws"
        ws.mkdir()
        run, diagnostics = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=self._instant_action())

        assert run.status == RunStatus.FAILED
        assert run.failure_code == FailureCode.TIMEOUT
        assert diagnostics is None

        pid_file = t3_home / "grandchild.pid"
        assert pid_file.exists(), "the stub never got far enough to spawn a grandchild"
        grandchild = int(pid_file.read_text())
        assert not self._alive(grandchild), f"grandchild {grandchild} outlived the run Carmel recorded as TIMEOUT"

        heartbeat = t3_home / "grandchild_heartbeat"
        before = heartbeat.read_text() if heartbeat.exists() else ""
        time.sleep(0.75)
        assert (heartbeat.read_text() if heartbeat.exists() else "") == before

    def test_timeout_message_says_the_tree_was_killed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The record must not claim more, or less, than what happened."""
        from carmel.adapters import t3 as t3_module

        t3_home = tmp_path / "fake_t3"
        t3_home.mkdir()
        (t3_home / T3_LAYOUT.EXECUTABLE_SCRIPT).write_text(self._STUB_T3, encoding="utf-8")
        monkeypatch.setenv("T3_PATH", str(t3_home))
        monkeypatch.delenv(T3_LAYOUT.T3_CONDA_ENV_VAR, raising=False)
        monkeypatch.delenv(T3_LAYOUT.T3_PYTHON_ENV_VAR, raising=False)
        monkeypatch.setattr(t3_module, "_RUN_TIMEOUT_BUFFER_S", 3.0)
        monkeypatch.setattr(t3_module, "_KILL_GRACE_PERIOD_S", 1.0)

        ws = tmp_path / "ws"
        ws.mkdir()
        run, _ = T3Adapter().run(workspace_root=ws, campaign=_campaign(ws), action=self._instant_action())
        assert "killed" in (run.error_message or "")


@contextlib.contextmanager
def monkeypatched_grace(module: Any, seconds: float) -> Any:
    """Temporarily shorten the SIGTERM->SIGKILL grace period."""
    original = module._KILL_GRACE_PERIOD_S
    module._KILL_GRACE_PERIOD_S = seconds
    try:
        yield
    finally:
        module._KILL_GRACE_PERIOD_S = original


class TestShutdownKillsLiveTrees:
    """A run executes on a daemon thread, which interpreter exit does not join.

    Without the atexit sweep, stopping the server leaves T3 and RMG running
    with no supervisor — the same orphaning this module exists to prevent,
    one layer up. It cannot help when Carmel is SIGKILLed; nothing can.
    """

    def test_a_running_tree_is_registered_and_then_forgotten(self, tmp_path: Path) -> None:
        from carmel.adapters import t3 as t3_module

        seen: list[int] = []
        real_register = t3_module._register_live_tree

        def _spy(proc: subprocess.Popen[Any]) -> None:
            seen.append(proc.pid)
            real_register(proc)

        t3_module._register_live_tree = _spy  # type: ignore[assignment]
        try:
            _run_in_process_group([sys.executable, "-c", "pass"], timeout=60.0)
        finally:
            t3_module._register_live_tree = real_register  # type: ignore[assignment]

        assert seen, "the process tree was never registered for shutdown cleanup"
        assert not t3_module._LIVE_TREES, "a finished tree is still registered"

    def test_shutdown_kills_a_tree_that_is_still_running(self) -> None:
        from carmel.adapters import t3 as t3_module

        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", "import time; time.sleep(300)"],
            start_new_session=True,
        )
        t3_module._register_live_tree(proc)
        try:
            with monkeypatched_grace(t3_module, 1.0):
                t3_module._terminate_live_trees()
            assert proc.poll() is not None, "shutdown left a live T3 process tree behind"
        finally:
            t3_module._forget_live_tree(proc)
            if proc.poll() is None:  # pragma: no cover -- only if the sweep failed
                proc.kill()
                proc.wait()

    def test_shutdown_ignores_trees_that_already_finished(self) -> None:
        """A finished process must not be signalled again during shutdown."""
        from carmel.adapters import t3 as t3_module

        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        proc.wait()
        t3_module._register_live_tree(proc)
        try:
            t3_module._terminate_live_trees()
        finally:
            t3_module._forget_live_tree(proc)

    def test_falls_back_to_killing_the_child_when_killpg_does_not_land(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A group signal can fail (EPERM, a setsid'ed descendant).

        The calling thread must not be left waiting forever on a child that
        the group signal never reached.
        """
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module.os, "killpg", lambda pgid, sig: None)
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", "import time; time.sleep(300)"],
            start_new_session=True,
        )
        try:
            t3_module._terminate_process_tree(proc, grace_period_s=1.0)
            assert proc.poll() is not None, "the child outlived a killpg that never landed"
        finally:
            if proc.poll() is None:  # pragma: no cover -- only if the fallback failed
                proc.kill()
                proc.wait()

    def test_an_unreapable_child_is_reported_rather_than_hung_on(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Last resort: say so loudly, but return. Never block forever."""
        from carmel.adapters import t3 as t3_module

        monkeypatch.setattr(t3_module.os, "killpg", lambda pgid, sig: None)
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", "import time; time.sleep(300)"],
            start_new_session=True,
        )
        monkeypatch.setattr(proc, "kill", lambda: None)

        emitter = logging.getLogger("carmel.adapters.t3")
        emitter.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.ERROR):
                t3_module._terminate_process_tree(proc, grace_period_s=0.5)
            assert any("Could not reap process" in record.message for record in caplog.records)
        finally:
            emitter.removeHandler(caplog.handler)
            # NOT proc.kill(): it is monkeypatched to a no-op for this test,
            # so waiting on it would block until the child's own sleep ends.
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
