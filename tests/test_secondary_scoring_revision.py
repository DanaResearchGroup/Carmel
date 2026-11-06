"""Network-free integrity, augmentation and target-score boundary checks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carmel.benchmarks import cantera_worker as worker
from carmel.benchmarks import holdout, holdout_idt, n_heptane, n_pentane
from carmel.benchmarks.schema import validate_contract_json

ROOT = Path(__file__).resolve().parents[1]
# Opt-in directory of local pinned evidence; when unset, the real-data tests skip.
LOCAL_RUNS = os.environ.get("CARMEL_LOCAL_RUNS", "")


def local_evidence(relative: str) -> Path:
    return Path(LOCAL_RUNS, relative)


GROUND = ["CH", "O2", "CO", "C2H", "O", "CO2", "H", "OH", "H2O", "N2", "H2", "CH4", "AR", "HE"]


def mechanism(native: bool = False) -> dict[str, Any]:
    species = [{"name": name, "composition": {"H": 1}} for name in GROUND]
    reactions = [{"equation": "H + O2 <=> O + OH", "rate-constant": {"A": 7, "b": 0, "Ea": 0}}]
    if native:
        species += [
            {"name": "OHV", "composition": {"O": 1, "H": 1}},
            {"name": "CHV", "composition": {"C": 1, "H": 1}},
        ]
        reactions += [{"equation": "CH + O2 <=> CO + OHV", "rate-constant": {"A": 99, "b": 1, "Ea": 4}}]
    return {
        "species": species,
        "reactions": reactions,
        "phases": [{"name": "gas", "species": [s["name"] for s in species]}],
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
    }


@pytest.mark.parametrize(
    "case,loader", [("n-heptane-low-t", n_heptane.load_contract), ("n-pentane-low-t", n_pentane.load_contract)]
)
def test_contract_revision_is_required_and_entire_document_is_frozen(case: str, loader: Any, tmp_path: Path) -> None:
    path = ROOT / "benchmarks" / case / "contract.json"
    document = json.loads(path.read_bytes())
    assert validate_contract_json(path.read_bytes()).revision.number == 2
    assert loader(path).revision.number == 2
    previous = copy.deepcopy(document)
    revision = previous.pop("revision")
    previous["scoring"].pop("target_chemistry")
    previous["scoring"].pop("excited_state_submechanism_sha256", None)
    assert (
        hashlib.sha256((json.dumps(previous, indent=2) + "\n").encode()).hexdigest()
        == revision["previous_contract_sha256"]
    )
    for field in ("reason", "previous_contract_sha256"):
        modified = copy.deepcopy(document)
        modified["revision"][field] = "unrecorded"
        with pytest.raises(ValueError, match="UNRECORDED"):
            validate_contract_json(json.dumps(modified))
    for key in ("scoring", "identity"):
        modified = copy.deepcopy(document)
        field = "locking" if key == "scoring" else "historical_identity_caveat"
        modified[key][field] += " edited"
        path = tmp_path / "edited.json"
        path.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match="UNRECORDED"):
            loader(path)
    del document["revision"]
    with pytest.raises(ValueError, match="revision"):
        validate_contract_json(json.dumps(document))


def test_pinned_chemistry_is_identical_for_old_new_and_candidate_and_idempotent() -> None:
    signatures = []
    for parent in (mechanism(), mechanism(native=True), mechanism()):
        before = copy.deepcopy(parent)
        augmented, metadata = worker.append_excited_submechanism(parent)
        assert parent == before
        ground = [s for s in augmented["species"] if s["name"] not in worker.EXCITED_ALIASES]
        assert ground == before["species"][: len(GROUND)]
        assert augmented["reactions"][0] == before["reactions"][0]
        assert metadata["ground_reactions_preserved"] == 1
        assert metadata["replaced_species"] == (["OHV", "CHV"] if len(before["species"]) > len(GROUND) else [])
        assert len(metadata["replaced_reactions"]) == (1 if metadata["replaced_species"] else 0)
        assert augmented["phases"][0]["species"] == GROUND + ["OH*", "CH*"]
        assert len(augmented["reactions"]) == 26
        signatures.append(json.dumps([augmented["species"][-2:], augmented["reactions"][1:]], sort_keys=True).encode())
        again, _ = worker.append_excited_submechanism(augmented)
        assert again == augmented
    assert signatures[0] == signatures[1] == signatures[2]
    pinned = worker.load_excited_submechanism()
    assert len(pinned["reactions"]) == 25
    assert {s["name"] for s in pinned["species"]} == {"OH*", "CH*"}
    assert all(r["units"] == pinned["units"] for r in augmented["reactions"][1:])


@pytest.mark.parametrize(
    "defect", ["composition", "missing", "efficiencies", "collider", "explicit_collider", "phases"]
)
def test_replacement_refuses_ground_state_changes(defect: str) -> None:
    parent = mechanism(native=True)
    if defect == "composition":
        parent["species"][-1]["composition"] = {"C": 2}
    elif defect == "missing":
        parent["species"] = parent["species"][1:]
    elif defect == "efficiencies":
        parent["reactions"][0]["efficiencies"] = {"OHV": 2}
    elif defect == "collider":
        parent["reactions"][0]["equation"] = "H + O2 (+OHV) <=> O + OH (+OHV)"
    elif defect == "explicit_collider":
        # Ground-state chemistry with an unchanged excited third body must not be deleted silently.
        parent["reactions"][0]["equation"] = "H + O2 + OHV <=> HO2 + OHV"
    else:
        parent["phases"] = []
    before = copy.deepcopy(parent)
    with pytest.raises(ValueError, match="NEEDS-INPUT"):
        worker.append_excited_submechanism(parent)
    assert parent == before


def test_submechanism_data_tampering_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "FFCM1_EXCITED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="PIN CHANGED"):
        worker.load_excited_submechanism()


@pytest.mark.parametrize(
    "target,names,expected",
    [
        ("OH*", ["OHV"], "OHV"),
        ("CH*", ["CHV"], "CHV"),
        ("OHV", ["OH*"], "OH*"),
        ("CHV", ["CH*"], "CH*"),
        ("OH*", ["OH*"], "OH*"),
        ("UNKNOWN*", ["OHV", "CHV"], None),
        ("OH*", ["OH"], None),
        ("pressure", [], "pressure"),
    ],
)
def test_only_declared_aliases_resolve(target: str, names: list[str], expected: str | None) -> None:
    assert worker.resolve_ignition_target(target, names) == expected


def test_frozen_temperature_projection_selects_exactly_124_points() -> None:
    contract = n_heptane.load_contract(ROOT / "benchmarks/n-heptane-low-t/contract.json")
    projection = json.loads((ROOT / "tests/fixtures/idt_temperature_projection.json").read_bytes())["files"]
    pins = {f.path: f for study in contract.split.studies if study.set == "holdout" for f in study.files}
    assert set(projection) == set(pins)
    rows = []
    for path, item in projection.items():
        assert item["sha256"] == pins[path].sha256
        assert len(item["temperature_k"]) == pins[path].points
        for index, temperature in enumerate(item["temperature_k"]):
            rows.append(
                {
                    "row_id": f"{path}:{index}",
                    "temperature_k": temperature,
                    "group": item["group"],
                    "experimental_s": 1,
                    "ignition_delay_s": 10,
                    "ignition_target": "CH*",
                }
            )
    score = holdout_idt.target_idt_metric(rows)
    assert len(rows) == 189
    assert score["total_count"] == score["scored_count"] == 124
    assert score["overall"] == 1
    assert score["groups"]["shock-tube"]["scored_count"] == 109
    assert score["groups"]["RCM-ideal"]["scored_count"] == 15
    assert sum(600 <= r["temperature_k"] <= 900 for r in rows) == 89
    assert sum(900 < r["temperature_k"] <= 1000 for r in rows) == 35
    edge_rows = [{**rows[0], "row_id": str(t), "temperature_k": t} for t in [599.99, 600, 1000, 1000.01]]
    assert [r["row_id"] for r in holdout_idt.target_idt_metric(edge_rows)["scored"]] == ["600", "1000"]


def test_target_jsr_preserves_existing_metric_and_selects_exactly_five_ethers() -> None:
    contract = n_pentane.load_contract(ROOT / "benchmarks/n-pentane-low-t/contract.json")
    assert dict(contract.scoring.target_chemistry.species) == holdout.TARGET_ETHERS
    assert len(holdout.TARGET_ETHERS) == 5
    rows = [
        {"species": species, "row_id": str(i), "temperature_k": t, "experimental": 1, "simulated": 10}
        for species in [*holdout.TARGET_ETHERS, "CH4"]
        for i, t in enumerate([550, 800, 950])
    ]
    before = holdout.score_metric(rows)
    target = holdout.target_jsr_metric(rows)
    assert target["overall"] == 1
    assert target["total_observations"] == 15
    assert set(target["per_species"]) == set(holdout.TARGET_ETHERS)
    assert all(r["scored_rows"] == 3 for r in target["per_species"].values())
    assert holdout.score_metric(rows) == before
    rows[0].update(simulated=0, experimental=0)
    rows[1]["refusal"] = "solver_failure"
    assert holdout.target_jsr_metric(rows)["per_species"][rows[0]["species"]]["scored_rows"] == 2
    assert holdout.target_jsr_metric(rows)["per_species"][rows[0]["species"]]["detection_limit"] == 0.5
    # An ether with no scorable row keeps a zero-count entry instead of disappearing from the report.
    missing = next(iter(holdout.TARGET_ETHERS))
    for row in rows:
        if row["species"] == missing:
            row["simulated"] = None
    report = holdout.target_jsr_metric(rows)
    assert list(report["per_species"]) == list(holdout.TARGET_ETHERS)
    assert report["per_species"][missing]["score"] is None
    assert report["per_species"][missing]["scored_rows"] == 0
    assert report["overall"] == 1


@pytest.mark.parametrize(
    "equation,expected",
    [
        ("H + O2 + OHV <=> HO2 + OHV", {"OHV"}),
        ("2CH* <=> 2CH*", {"CH*"}),
        ("OH* + N2 <=> OH + N2", set()),
        ("2OHV <=> OHV + OH", set()),
        ("OH* + AR => OH + AR", set()),
        ("H + O2 (+OHV) <=> HO2 (+OHV)", set()),
        ("not an equation", set()),
    ],
)
def test_explicit_collider_needs_equal_stoichiometry_on_both_sides(equation: str, expected: set[str]) -> None:
    assert worker._explicit_colliders(equation, set(worker.EXCITED_ALIASES)) == expected


def test_non_solver_refusal_gate_is_strictly_more_than_one_third() -> None:
    rows = [
        {
            "row_id": str(i),
            "group": "RCM-ideal" if i < 15 else "shock-tube",
            "experimental_s": 1,
            "ignition_delay_s": 1,
            "ignition_target": "OH*",
        }
        for i in range(189)
    ]
    for row in rows[15:73]:
        row["reason"] = "ignition_not_reached"
    assert holdout_idt.idt_coverage_gate(rows)["passed"]
    rows[73]["reason"] = "target_species_absent"
    assert not holdout_idt.idt_coverage_gate(rows)["passed"]
    for reason in ("solver_failure: numerical error", "solver_crash", "max_steps_exceeded"):
        rows[73]["reason"] = reason
        assert holdout_idt.idt_coverage_gate(rows)["passed"]


@pytest.mark.parametrize("pin", [worker.FFCM1_EXCITED_SHA256, "unknown"])
def test_worker_applies_requested_pin_or_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str) -> None:
    import yaml

    converted = tmp_path / "converted.yaml"
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "observable": "idt",
                "excited_state_submechanism_sha256": pin,
                "mechanism_path": "fake",
                "thermo_path": "fake",
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "rows": [{"row_id": "p", "ignition_target": "OH*", "pressure_pa": 101325}],
            }
        )
    )

    def convert(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted.write_text(yaml.safe_dump(mechanism(native=True)))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(worker.subprocess, "run", convert)
    monkeypatch.setitem(
        sys.modules,
        "cantera",
        SimpleNamespace(__version__="test", Solution=lambda _: SimpleNamespace(species_names=GROUND + ["OH*", "CH*"])),
    )

    def execute(rows: list[dict[str, Any]], **kwargs: Any) -> tuple[dict[str, Any], int]:
        assert rows == [{"row_id": "p", "ignition_target": "OH*", "pressure_pa": 101325}]
        assert kwargs["initial_results"] == {}
        augmented = yaml.safe_load(Path(kwargs["trimmed_path"]).read_text())
        assert augmented["phases"][0]["species"][-2:] == ["OH*", "CH*"]
        assert len(augmented["reactions"]) == 26
        return {"p": {"row_id": "p", "status": "ok", "ignition_delay_s": 0.1}}, 0

    monkeypatch.setattr(worker, "execute_rows", execute)
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(response)])
    assert worker.main() == 0
    result = json.loads(response.read_bytes())
    if pin == "unknown":
        assert result["refusal"] == "excited_state_submechanism_refused"
    else:
        assert result["status"] == "ok"
        assert result["excited_state_submechanism"]["replaced_species"] == ["OHV", "CHV"]


def test_pinned_data_and_aliases_invalidate_checkpoint_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = worker.simulation_lock_sha256(3600, "test")
    monkeypatch.setattr(worker, "FFCM1_EXCITED_SHA256", "different")
    assert worker.simulation_lock_sha256(3600, "test") != before


def test_typed_idt_requests_always_pin_chemistry_and_jsr_rejects_it(tmp_path: Path) -> None:
    args = {
        name: tmp_path / name
        for name in ("mechanism_path", "thermo_path", "converted_yaml_path", "trimmed_yaml_path", "checkpoint_path")
    }
    request = holdout.WorkerRequest(**args, observable="idt", rows=(), reaction_source_lines={})
    assert request.excited_state_submechanism_sha256 == worker.FFCM1_EXCITED_SHA256
    with pytest.raises(ValueError, match="only defined for IDT"):
        holdout.WorkerRequest(
            **args,
            observable="jsr",
            rows=(),
            reaction_source_lines={},
            excited_state_submechanism_sha256=worker.FFCM1_EXCITED_SHA256,
        )


ARCHIVE = local_evidence("i116-alkane-census/evidence/pentane_indirect_v2.5.zip")


@pytest.mark.skipif(not LOCAL_RUNS or not ARCHIVE.is_file(), reason="local pinned archive unavailable")
def test_real_frozen_holdout_contains_exactly_five_c5_cyclic_ethers() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    rows = holdout._load_rows(n_pentane.load_contract(ROOT / "benchmarks/n-pentane-low-t/contract.json"), ARCHIVE)
    labels = {s for row in rows for s in row["observations"]}
    ethers = {}
    for label in labels:
        smiles = holdout.observable_smiles(label)
        assert smiles is not None
        mol = Chem.MolFromSmiles(smiles)
        if rdMolDescriptors.CalcMolFormula(mol) == "C5H10O" and mol.GetRingInfo().NumRings():
            ethers[label] = smiles
    assert ethers == holdout.TARGET_ETHERS
    assert sum(s in ethers for row in rows for s in row["observations"]) == 600


REAL_SPLIT = local_evidence("i112-rcm-probe/chemked")


@pytest.mark.skipif(not LOCAL_RUNS or not REAL_SPLIT.is_dir(), reason="local pinned split unavailable")
def test_projection_reproduces_exact_frozen_source_temperatures() -> None:
    contract = n_heptane.load_contract(ROOT / "benchmarks/n-heptane-low-t/contract.json")
    rows = holdout_idt.load_idt_rows(contract, REAL_SPLIT)
    projection = json.loads((ROOT / "tests/fixtures/idt_temperature_projection.json").read_bytes())["files"]
    for row in rows:
        index = int(row["row_id"].rsplit(":", 1)[1])
        assert row["temperature_k"] == projection[row["record"]]["temperature_k"][index]
        assert row["group"] == projection[row["record"]]["group"]
    assert holdout_idt.target_idt_metric(rows)["total_count"] == 124


HISTORICAL = local_evidence("i131-holdout-idt")


@pytest.mark.skipif(
    not LOCAL_RUNS or not (HISTORICAL / "v_new/converted.yaml").is_file(),
    reason="local converted mechanisms unavailable",
)
def test_real_mechanisms_preserve_ground_records_and_receive_identical_additions() -> None:
    import yaml

    additions = []
    for key in ("v_old", "v_new"):
        parent = yaml.safe_load((HISTORICAL / key / "converted.yaml").read_text())
        result, metadata = worker.append_excited_submechanism(parent)
        aliases = set(worker.EXCITED_ALIASES)
        assert result["species"][:-2] == [s for s in parent["species"] if s["name"] not in aliases]
        assert result["reactions"][:-25] == [
            r for r in parent["reactions"] if not worker._equation_species(r["equation"]) & aliases
        ]
        assert len(metadata["replaced_reactions"]) == (25 if key == "v_new" else 0)
        additions.append(json.dumps([result["species"][-2:], result["reactions"][-25:]], sort_keys=True).encode())
    candidate, _ = worker.append_excited_submechanism(mechanism())
    assert (
        additions[0]
        == additions[1]
        == json.dumps([candidate["species"][-2:], candidate["reactions"][-25:]], sort_keys=True).encode()
    )


def test_integrator_samples_the_aliased_species() -> None:
    looked_up = []
    gas = SimpleNamespace(
        species_names=["OHV"], P=101325, X=[0.1], species_index=lambda name: looked_up.append(name) or 0
    )
    reactor = SimpleNamespace(phase=gas, T=800)

    class Network:
        time = 0.0
        solver_stats = {"steps": 0}

        def step(self) -> None:
            self.time = 1.0
            self.solver_stats = {"steps": 1}

    ct = SimpleNamespace(IdealGasReactor=lambda *a, **k: reactor, ReactorNet=lambda *a: Network())
    times, values, _, _ = worker.integrate_idt_trace(ct, gas, {"ignition_target": "OH*"})
    assert looked_up == ["OHV"]
    assert times == [0, 1]
    assert values == [0.1, 0.1]


def _load_n_pentane_reproducer() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "n_pentane_reproduce", ROOT / "benchmarks/n-pentane-low-t/reproduce.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_n_pentane_reproducer_emits_the_recorded_revision_fields() -> None:
    pytest.importorskip("rdkit")  # the reproducer imports the structure toolkit at load time
    reproducer = _load_n_pentane_reproducer()
    frozen = json.loads((ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes())
    assert frozen["revision"] == reproducer.REVISION
    assert dict(reproducer.TARGET_ETHERS) == frozen["scoring"]["target_chemistry"]["species"]


NC5_EVIDENCE = local_evidence("i116-alkane-census/evidence")


@pytest.mark.skipif(
    not LOCAL_RUNS or not (NC5_EVIDENCE / "pentane_indirect_v2.5.zip").is_file(),
    reason="local pinned n-pentane evidence unavailable",
)
def test_n_pentane_reproducer_rewrites_the_frozen_contract_byte_for_byte(tmp_path: Path) -> None:
    written = tmp_path / "contract.json"
    command = [
        sys.executable,
        str(ROOT / "benchmarks/n-pentane-low-t/reproduce.py"),
        "--write-contract",
        "--contract",
        str(written),
        "--v-old",
        str(NC5_EVIDENCE / "nc5-49.txt"),
        "--v-new",
        str(NC5_EVIDENCE / "c5-2015.txt"),
        "--archive",
        str(NC5_EVIDENCE / "pentane_indirect_v2.5.zip"),
    ]
    subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
    assert written.read_bytes() == (ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes()
