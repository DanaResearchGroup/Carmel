"""Data-driven warning reviews preserve exact pins and temperature conditions."""

import dataclasses
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from carmel.benchmarks import cantera_worker as worker

SOURCE = (Path(__file__).parent / "data/reviewed_ec6_thermo.dat").read_bytes()
WARNING = "NasaPoly2::validate: For species EC6H13O2, discontinuity in s/R detected at Tmid = 1388"


def test_existing_case2_simulation_lock_is_byte_identical():
    assert worker.simulation_lock_sha256(600.0, "3.2.0") == (
        "72ece5b5ba96271f06ef0668fe5d2b64f098d53a740e05a1078e4e5a66f93497"
    )


@pytest.mark.parametrize("temperature", [None, 0, 1388, 1400, float("nan"), float("inf")])
def test_temperature_dependent_review_requires_strict_upper_bound(temperature):
    assert worker.cantera_validation_refusals([WARNING], (SOURCE,), holdout_max_temperature_k=temperature) == [WARNING]


def test_ec6_review_accepts_exact_pinned_card_and_bound():
    review = worker.REVIEWED_NASA_WARNINGS[2]
    assert worker.nasa7_record_digests((SOURCE,))["EC6H13O2"] == review.numeric_field_sha256
    assert worker.cantera_validation_refusals([WARNING], (SOURCE,), holdout_max_temperature_k=1100) == []
    assert review.source_line == 2987
    assert review.source_file == "prf_v3_therm_dat.txt"
    assert "0.64%" in review.reason


@pytest.mark.parametrize("change", ["species", "quantity", "midpoint", "numeric", "missing", "extra"])
def test_ec6_review_refuses_every_unreviewed_dimension(change):
    source, warning = SOURCE, WARNING
    if change == "species":
        warning = warning.replace("EC6H13O2", "OTHER")
    elif change == "quantity":
        warning = warning.replace("s/R", "cp/R")
    elif change == "midpoint":
        warning = warning.replace("1388", "1389")
    elif change == "numeric":
        source = source.replace(b"1388.000", b"1389.000")
    elif change == "missing":
        source = b""
    else:
        warning += "\nunreviewed additional warning"
    assert worker.cantera_validation_refusals([warning], (source,), holdout_max_temperature_k=1100) == [warning]


def test_registry_quantity_is_data_not_a_special_case(monkeypatch):
    review = dataclasses.replace(worker.REVIEWED_NASA_WARNINGS[2], quantity="cp/R")
    monkeypatch.setattr(worker, "REVIEWED_NASA_WARNINGS", (review,))
    warning = WARNING.replace("s/R", "cp/R")
    assert worker.cantera_validation_refusals([warning], (SOURCE,), holdout_max_temperature_k=1100) == []
    assert worker.cantera_validation_refusals([WARNING], (SOURCE,), holdout_max_temperature_k=1100) == [WARNING]


def test_changed_first_ec6_record_cannot_use_later_approved_duplicate():
    changed = SOURCE.replace(b"1388.000", b"1389.000")
    assert worker.cantera_validation_refusals([WARNING], (changed, SOURCE), holdout_max_temperature_k=1100) == [WARNING]
    assert worker.cantera_validation_refusals([WARNING], (SOURCE, changed), holdout_max_temperature_k=1100) == []


def _run_worker(tmp_path, monkeypatch, observable, temperatures):
    """Drive worker.main() with Cantera, the converter and row execution stubbed out."""
    mechanism, thermo = tmp_path / "mechanism.inp", tmp_path / "thermo.dat"
    mechanism.write_bytes(b"SPECIES\nEC6H13O2\nEND\n")
    thermo.write_bytes(SOURCE)
    converted = tmp_path / "converted.yaml"
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    rows = [
        {"row_id": f"r{i}", "pressure_pa": 101325.0, "temperature_k": t, "ignition_target": "pressure"}
        for i, t in enumerate(temperatures)
    ]
    request.write_text(
        json.dumps(
            {
                "observable": observable,
                "mechanism_path": str(mechanism),
                "thermo_path": str(thermo),
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "checkpoint_path": str(tmp_path / "rows.jsonl"),
                "workers": 1,
                "rows": rows,
            }
        )
    )

    def convert(*args, **kwargs):
        converted.write_text("phases: []\nspecies: []\nreactions: []\n")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def solution(_):
        warnings.warn(WARNING, stacklevel=2)
        return SimpleNamespace(species_names=["EC6H13O2"])

    executed = []

    def execute_rows(rows, **kwargs):
        executed.extend(rows)
        return {row["row_id"]: {"row_id": row["row_id"], "status": "ok"} for row in rows}, 0

    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(__version__="test", Solution=solution))
    monkeypatch.setattr(worker.subprocess, "run", convert)
    monkeypatch.setattr(worker, "execute_rows", execute_rows)
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request), str(response)])
    assert worker.main() == 0
    return json.loads(response.read_text()), executed


@pytest.mark.parametrize(
    ("observable", "temperatures", "accepted"),
    [
        ("jsr", [700.0, 1100.0], True),
        ("jsr", [700.0, 1388.0], False),
        ("jsr", [1400.0], False),
        ("jsr", [], False),
        ("idt", [700.0, 1100.0], False),
    ],
)
def test_worker_bounds_the_ec6_review_by_scored_jsr_rows_only(
    tmp_path, monkeypatch, observable, temperatures, accepted
):
    result, executed = _run_worker(tmp_path, monkeypatch, observable, temperatures)
    assert result["validation_warnings"] == [WARNING]
    if accepted:
        assert result["status"] == "ok"
        assert [row["row_id"] for row in executed] == ["r0", "r1"]
    else:
        assert result["status"] == "refused"
        assert result["refusal"] == "unallowlisted_cantera_validation_warning"
        assert executed == []
