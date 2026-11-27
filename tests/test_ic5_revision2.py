"""Case-3 revision 2: the 600-900 K JSR holdout window as target chemistry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from carmel.benchmarks import holdout, iso_pentane, n_heptane, n_pentane
from carmel.benchmarks.revision import FROZEN_REVISIONS, JSRWindowTarget
from carmel.benchmarks.schema import validate_contract_json
from tests.ic5_acceptance import REVISION2_CANONICAL, reverse_a2

ROOT = Path(__file__).resolve().parents[1]
CASE3 = ROOT / "benchmarks/iso-pentane-low-t/contract.json"
A1_CANONICAL_SHA256 = "2bcc848601a39b86d26038c82c87e240f78add3f3a7941efb47355b1acf714b3"
LOCAL_RUNS = os.environ.get("CARMEL_LOCAL_RUNS", "")
ARCHIVE = Path(LOCAL_RUNS, "i116-alkane-census/evidence/pentane_indirect_v2.5.zip")
CASE2_REPORTS = Path(LOCAL_RUNS, "i139-deterministic-row-budget/case_2")
TARGET = JSRWindowTarget(
    temperature_min_k=600,
    temperature_max_k=900,
    inclusive=True,
    frozen_rows=31,
    frozen_scalars=434,
    metric="existing_species_balanced_jsr_metric",
)


def canonical_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def a1_document() -> dict[str, Any]:
    """Recover A-1 by reversing A-2 and removing the two revision-2 keys."""
    previous = reverse_a2(json.loads(CASE3.read_bytes()))
    del previous["revision"]
    del previous["scoring"]["target_chemistry"]
    return previous


def test_only_revision_keys_differ_from_the_a1_contract() -> None:
    """Reconstructed revision 2 differs from A-1 only in its two declared additions."""
    previous = a1_document()
    assert canonical_sha256(previous) == A1_CANONICAL_SHA256
    revised = reverse_a2(json.loads(CASE3.read_bytes()))
    assert (
        revised["revision"]["previous_contract_sha256"]
        == hashlib.sha256((json.dumps(previous, indent=2) + "\n").encode()).hexdigest()
    )
    changed = sorted(key for key in revised.keys() | previous.keys() if revised.get(key) != previous.get(key))
    assert changed == ["revision", "scoring"]
    scoring_changed = sorted(
        key
        for key in revised["scoring"].keys() | previous["scoring"].keys()
        if revised["scoring"].get(key) != previous["scoring"].get(key)
    )
    assert scoring_changed == ["target_chemistry"]
    assert canonical_sha256(revised) == REVISION2_CANONICAL != A1_CANONICAL_SHA256
    assert FROZEN_REVISIONS["iso-pentane-low-t"] == canonical_sha256(json.loads(CASE3.read_bytes()))


def test_reproducer_rebuilds_the_committed_contract_byte_for_byte() -> None:
    """The same revision-2 transformation preserves both historical and current splits."""
    pytest.importorskip("rdkit")  # the reproducer loads the case-1/2 reproducers, which import it
    reproducer = runpy.run_path(str(ROOT / "benchmarks/iso-pentane-low-t/reproduce.py"))
    previous = json.loads(CASE3.read_bytes())
    del previous["revision"]
    del previous["scoring"]["target_chemistry"]
    before = copy.deepcopy(previous)
    revised = reproducer["apply_revision"](previous)
    assert previous == before
    rebuilt = iso_pentane.Contract.model_validate(revised).model_dump_json(indent=2) + "\n"
    assert rebuilt.encode() == CASE3.read_bytes()
    historical = a1_document()
    assert reproducer["apply_revision"](historical) == reverse_a2(json.loads(CASE3.read_bytes()))


def test_target_literals_equal_the_frozen_split_totals() -> None:
    contract = iso_pentane.load_contract(CASE3)
    target = contract.scoring.target_chemistry
    assert contract.revision.number == 2
    assert (target.frozen_rows, target.frozen_scalars) == (
        iso_pentane.Contract.expected_totals["holdout_low_temperature_rows"],
        iso_pentane.Contract.expected_totals["holdout_low_temperature_observations"],
    )
    assert (target.temperature_min_k, target.temperature_max_k, target.inclusive) == (600, 900, True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frozen_rows", 30),
        ("frozen_scalars", 435),
        ("temperature_max_k", 1000),
        ("inclusive", False),
        ("metric", "mean_abs_log10_delay_ratio"),
    ],
)
def test_wrong_target_literal_is_refused(field: str, value: object) -> None:
    document = json.loads(CASE3.read_bytes())
    document["scoring"]["target_chemistry"][field] = value
    with pytest.raises(ValidationError):
        validate_contract_json(json.dumps(document))


@pytest.mark.parametrize("edit", ["missing_revision", "missing_target", "holdout_text", "target_extra"])
def test_edited_case3_contract_is_refused(edit: str, tmp_path: Path) -> None:
    document = json.loads(CASE3.read_bytes())
    if edit == "missing_revision":
        del document["revision"]
    elif edit == "missing_target":
        del document["scoring"]["target_chemistry"]
    elif edit == "holdout_text":
        document["split"]["studies"][-1]["bibliography"] += " edited"
    else:
        document["scoring"]["target_chemistry"]["note"] = "extra"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    with pytest.raises(ValueError):
        iso_pentane.load_contract(path)
    with pytest.raises(ValueError):
        validate_contract_json(path.read_bytes())


def window_rows() -> list[dict[str, Any]]:
    """31 in-window rows of 14 species (434 scalars), plus out-of-window and edge rows."""
    species = [f"S{i:02d}" for i in range(14)]
    temperatures = [600.0, 900.0, *[700.0 + i for i in range(29)]]
    rows = [
        {
            "row_id": f"w{index}",
            "species": label,
            "temperature_k": temperature,
            "experimental": 1.0,
            "simulated": 10.0,
        }
        for index, temperature in enumerate(temperatures)
        for label in species
    ]
    for row in rows:
        if row["species"] == "S13":
            row["refusal"] = "unmapped_species"
    for row_id, temperature in (("edge-low", 599.99), ("edge-high", 900.01), ("hot", 950.0)):
        rows += [
            {"row_id": row_id, "species": label, "temperature_k": temperature, "experimental": 1.0, "simulated": 1.0}
            for label in ("S00", "HOT_ONLY")
        ]
    return rows


def test_window_metric_reuses_the_low_temperature_subset_and_keeps_zero_counts() -> None:
    rows = window_rows()
    report = holdout.target_window_metric(rows, TARGET)
    assert (report["total_rows"], report["total_observations"]) == (31, 434)
    assert report["overall"] == 1
    subset = holdout.score_metric(rows)["low_temperature_600_900_k"]
    for label, entry in subset["per_species"].items():
        assert report["per_species"][label] == entry
    assert report["per_species"]["S00"]["scored_rows"] == 31
    assert list(report["per_species"]) == sorted({row["species"] for row in rows})
    assert report["per_species"]["HOT_ONLY"] == {"score": None, "scored_rows": 0, "detection_limit": 0.5}
    assert report["per_species"]["S13"] == {"score": None, "scored_rows": 0, "detection_limit": 0.5}
    assert report["refusals"] == subset["refusals"]


def test_window_metric_refuses_counts_that_differ_from_the_frozen_target() -> None:
    rows = window_rows()
    with pytest.raises(ValueError, match="30 rows and 420 scalars"):
        holdout.target_window_metric([row for row in rows if row["row_id"] != "w5"], TARGET)
    with pytest.raises(ValueError, match="31 rows and 433 scalars"):
        holdout.target_window_metric(rows[1:], TARGET)


def test_harness_loader_and_target_dispatch_per_case() -> None:
    case3 = holdout.load_contract(CASE3.read_bytes())
    assert isinstance(case3, iso_pentane.Contract)
    case2_bytes = (ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes()
    case2 = holdout.load_contract(case2_bytes)
    assert isinstance(case2, n_pentane.Contract)
    with pytest.raises(ValueError, match="does not score case n-heptane-low-t"):
        holdout.load_contract((ROOT / "benchmarks/n-heptane-low-t/contract.json").read_bytes())
    rows = window_rows()
    assert holdout._target_chemistry(case3, rows) == holdout.target_window_metric(rows, TARGET)
    ether_rows = [{**row, "species": next(iter(holdout.TARGET_ETHERS))} for row in rows]
    assert holdout._target_chemistry(case2, ether_rows) == holdout.target_jsr_metric(ether_rows)


@pytest.mark.parametrize("case", ["n-heptane-low-t", "n-pentane-low-t"])
def test_cases_1_and_2_are_untouched(case: str) -> None:
    raw = (ROOT / "benchmarks" / case / "contract.json").read_bytes()
    assert canonical_sha256(json.loads(raw)) == FROZEN_REVISIONS[case]
    loader = n_heptane.load_contract if case == "n-heptane-low-t" else n_pentane.load_contract
    assert loader(ROOT / "benchmarks" / case / "contract.json").revision.number == 2


@pytest.mark.skipif(not LOCAL_RUNS or not ARCHIVE.is_file(), reason="local pinned archive unavailable")
def test_real_case3_window_holds_the_frozen_counts() -> None:
    contract = iso_pentane.load_contract(CASE3)
    rows = holdout._load_rows(contract, ARCHIVE)
    metric_rows = [
        {"row_id": row["row_id"], "species": label, "temperature_k": row["temperature_k"], "experimental": value}
        for row in rows
        for label, value in row["observations"].items()
    ]
    report = holdout.target_window_metric(metric_rows, contract.scoring.target_chemistry)
    assert (report["total_rows"], report["total_observations"]) == (31, 434)
    assert len(report["per_species"]) == 17
    assert {entry["scored_rows"] for entry in report["per_species"].values()} == {0}


@pytest.mark.skipif(
    not LOCAL_RUNS or not (CASE2_REPORTS / "v_new/report.json").is_file(), reason="stored case-2 reports unavailable"
)
@pytest.mark.parametrize("key", ["v_new", "v_old_loaded"])
def test_stored_case2_scores_rescore_byte_identically(key: str) -> None:
    report = json.loads((CASE2_REPORTS / key / "report.json").read_bytes())
    contract = n_pentane.load_contract(ROOT / "benchmarks/n-pentane-low-t/contract.json")
    rows = report["metric_rows"]
    metric = holdout.score_metric(rows, metric_limits=holdout._metric_limits(rows))
    assert json.dumps(metric, sort_keys=True) == json.dumps(report["metric"], sort_keys=True)
    assert holdout._target_chemistry(contract, rows) == holdout.target_jsr_metric(rows)


def test_harness_refuses_a_drifted_window_before_any_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    paths = {name: tmp_path / name for name in ("contract.json", "mechanism.inp", "thermo.dat", "split.zip")}
    for path in paths.values():
        path.write_bytes(b"x\n")
    contract = SimpleNamespace(
        mechanisms={"v_old": SimpleNamespace(verify=lambda _: None)},
        scoring=SimpleNamespace(target_chemistry=TARGET),
    )
    row = {"row_id": "r:1", "temperature_k": 800.0, "observations": {"CO": 0.1}}
    monkeypatch.setattr(holdout, "load_contract", lambda _: contract)
    monkeypatch.setattr(holdout, "_load_rows", lambda *_: [row])

    def no_simulation(*_: object, **__: object) -> None:
        raise AssertionError("the worker must not run when the target window drifted")

    monkeypatch.setattr(holdout, "invoke_cantera_worker", no_simulation)
    with pytest.raises(ValueError, match="1 rows and 1 scalars, not the frozen 31 and 434"):
        holdout.run_harness(
            contract_path=paths["contract.json"],
            mechanism_path=paths["mechanism.inp"],
            mechanism_key="v_old",
            thermo_path=paths["thermo.dat"],
            thermo_url="https://example.test/thermo.dat",
            thermo_retrieval_date="2026-11-15",
            thermo_sha256=hashlib.sha256(b"x\n").hexdigest(),
            archive_path=paths["split.zip"],
            run_dir=tmp_path / "run",
            cantera_python=Path("fake-python"),
        )


def test_harness_scores_a_held_window_after_a_metadata_only_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case-3 window holding its frozen counts passes a metadata-only preflight, then simulates and scores."""
    from types import SimpleNamespace

    from carmel.benchmarks.cantera_worker import THREAD_ENV, simulation_lock, simulation_lock_sha256

    paths = {name: tmp_path / name for name in ("contract.json", "mechanism.inp", "thermo.dat", "split.zip")}
    for path in paths.values():
        path.write_bytes(b"x\n")
    products = [f"P{index:02d}" for index in range(14)]
    contract = SimpleNamespace(
        case="iso-pentane-low-t",
        mechanisms={"v_old": SimpleNamespace(verify=lambda _: None)},
        scoring=SimpleNamespace(target_chemistry=TARGET),
        split=SimpleNamespace(
            archive=SimpleNamespace(sha256="f" * 64),
            model_dump=lambda **_: {"archive": {"sha256": "f" * 64}, "studies": []},
        ),
        old_species=(SimpleNamespace(label="FUEL", smiles="C"),),
        new_species=tuple(SimpleNamespace(label=label, smiles=label) for label in products),
    )
    temperatures = [600.0 + 10.0 * index for index in range(31)] + [1000.0]
    rows = [
        {
            "row_id": f"r:{index}",
            "record": "synthetic.xml",
            "temperature_k": temperature,
            "pressure_pa": 101325.0,
            "residence_time_s": 2.0,
            "volume_m3": 8.5e-5,
            "inlet": {"FUEL": 1.0},
            "observations": {label: 0.01 for label in products},
            "record_structures": {},
        }
        for index, temperature in enumerate(temperatures)
    ]
    events: list[tuple[str, list[dict[str, Any]]]] = []
    real_target_chemistry = holdout._target_chemistry

    def recording_target_chemistry(contract: Any, metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
        events.append(("target", copy.deepcopy(metric_rows)))
        return real_target_chemistry(contract, metric_rows)

    def fake_worker(request: holdout.WorkerRequest, **_: Any) -> holdout.WorkerResponse:
        events.append(("worker", []))
        return holdout.WorkerResponse.model_validate(
            {
                "schema_version": 1,
                "status": "ok",
                "rows": [
                    {
                        "row_id": row.row_id,
                        "status": "ok",
                        "mole_fractions": {label: 0.02 for label in products},
                        "wall_time_s": 0.25,
                        "integrator_steps": 42,
                    }
                    for row in request.rows
                ],
                "converter": {"events": [], "refusals": [], "returncode": 0, "log": ""},
                "trimmed_yaml_sha256": "d" * 64,
                "mechanism_semantic_sha256": "e" * 64,
                "cantera_version": "3.2.0",
                "row_workers": 1,
                "configured_workers": 1,
                "row_timeout_s": 12.0,
                "checkpoint_path": str(tmp_path / "run" / "rows.jsonl"),
                "thread_env": THREAD_ENV,
                "row_process_start_method": "spawn",
                "load_average_at_start": [1.0, 2.0, 3.0],
                "simulation_lock": simulation_lock(12.0, "3.2.0"),
                "simulation_lock_sha256": simulation_lock_sha256(12.0, "3.2.0"),
            }
        )

    monkeypatch.setattr(holdout, "canonical", lambda value: value)
    monkeypatch.setattr(holdout, "load_contract", lambda _: contract)
    monkeypatch.setattr(holdout, "_load_rows", lambda *_: rows)
    monkeypatch.setattr(holdout, "mechanism", lambda _: (["FUEL", *products], [{"equation": "FUEL=P00", "line": 7}]))
    monkeypatch.setattr(holdout, "_target_chemistry", recording_target_chemistry)
    monkeypatch.setattr(holdout, "invoke_cantera_worker", fake_worker)
    report = holdout.run_harness(
        contract_path=paths["contract.json"],
        mechanism_path=paths["mechanism.inp"],
        mechanism_key="v_old",
        thermo_path=paths["thermo.dat"],
        thermo_url="https://example.test/thermo.dat",
        thermo_retrieval_date="2026-11-23",
        thermo_sha256=hashlib.sha256(b"x\n").hexdigest(),
        archive_path=paths["split.zip"],
        run_dir=tmp_path / "run",
        cantera_python=Path("fake-python"),
        row_timeout_s=12.0,
        workers=1,
    )

    assert [kind for kind, _ in events] == ["target", "worker", "target"]
    preflight = events[0][1]
    assert all(set(entry) == {"row_id", "species", "temperature_k"} for entry in preflight)
    assert preflight == [
        {"row_id": row["row_id"], "species": label, "temperature_k": row["temperature_k"]}
        for row in rows
        for label in row["observations"]
    ]
    assert len(preflight) == 32 * 14
    assert report["rows"]["converged"] == 32
    assert report["target_chemistry"] == holdout.target_window_metric(report["metric_rows"], TARGET)
    assert (report["target_chemistry"]["total_rows"], report["target_chemistry"]["total_observations"]) == (31, 434)
