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

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from carmel.adapters.t3 import (
    T3_LAYOUT,
    T3Adapter,
    _coerce_reaction_entry,
    _coerce_species_entry,
    _discover_pdep_networks,
    _find_t3_executable,
    _walk_iterations,
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
        assert result is None or (isinstance(result, list) and result[0] == "python")


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
# Reading and parsing T3_info.yml
# ---------------------------------------------------------------------------


class TestReadT3InfoFile:
    def test_reads_real_iteration_1_fixture(self) -> None:
        info = read_t3_info_file(FIXTURE_ROOT / "iteration_1" / "ARC" / "T3_info.yml")
        assert info["reactions"] == []
        assert info["species"][0]["label"] == "Imipramine_1_peroxy_0"
        assert info["species"][0]["success"] is True

    def test_reads_real_iteration_2_fixture(self) -> None:
        info = read_t3_info_file(FIXTURE_ROOT / "iteration_2" / "ARC" / "T3_info.yml")
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


class TestT3AdapterRealSubprocess:
    """End-to-end subprocess tests — only run when T3 is actually importable."""

    @requires_t3
    def test_run_does_not_crash(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        adapter = T3Adapter(submission_mode=SubmissionMode.SUBPROCESS)
        run, diagnostics = adapter.run(workspace_root=ws, campaign=_campaign(ws), action=_action())
        # We don't assert success because real T3 needs the full RMG stack;
        # we just assert that the adapter produced a typed RunRecord.
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED)
        if run.status == RunStatus.SUCCEEDED:
            assert diagnostics is not None
            assert diagnostics.run_id == run.run_id
