"""Hand-checkable paired row uncertainty, with no Cantera dependency."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from carmel.benchmarks import cantera_worker, holdout, holdout_intervals
from carmel.benchmarks.holdout import score_metric
from carmel.benchmarks.holdout_intervals import idt_separation, jsr_intervals, paired_interval, rescore_saved_jsr

ROOT = Path(__file__).resolve().parents[1]


def test_pairing_and_species_balance_survive_row_resampling() -> None:
    errors = {("a", "A"): (1.0, 0.5), ("b", "A"): (1.0, 0.5), ("c", "A"): (1.0, 0.5), ("a", "B"): (9.0, 8.5)}
    result = paired_interval(errors)
    assert result["v_old"] == 5.0
    assert result["v_new"] == 4.5
    assert result["gap"] == 0.5
    assert result["interval_95"] == [0.5, 0.5]
    assert result["row_count"] == 3
    assert result["observable_count"] == 4
    assert result["bootstrap_replicates"] == 10000
    assert json.dumps(result, sort_keys=True) == json.dumps(paired_interval(errors), sort_keys=True)


def test_percentile_interval_for_two_hand_known_points() -> None:
    result = paired_interval({("a", "idt"): (0.0, 1.0), ("b", "idt"): (3.0, 0.0)})
    # Two draws can yield gaps -1, 1, 3 with probabilities 1/4, 1/2, 1/4.
    assert result["gap"] == 1.0
    assert result["interval_95"] == [-1.0, 3.0]
    assert paired_interval({})["interval_95"] is None
    with pytest.raises(ValueError, match="finite"):
        paired_interval({("a", "idt"): (math.nan, 0.0)})


def _idt_report() -> dict[str, Any]:
    return {
        "simulation_lock_sha256": "lock",
        "holdout_fingerprint": {"sha256": "frozen"},
        "row_results": [
            {
                "row_id": "a",
                "ignition_target": "pressure",
                "ignition_type": "d/dt max",
                "experimental_s": 1.0,
                "ignition_delay_s": 2.0,
                "temperature_k": 750.0,
                "temperature_band": "750_to_900",
                "group": "RCM-ideal",
                "reason": None,
            },
            {
                "row_id": "b",
                "ignition_target": "CH*",
                "ignition_type": "d/dt max",
                "experimental_s": 1.0,
                "temperature_k": 700.0,
                "temperature_band": "below_750",
                "group": "shock-tube",
                "reason": "target_species_absent",
            },
        ],
    }


def test_idt_intersection_and_empty_shock_diagnostic() -> None:
    old = _idt_report()
    new = copy.deepcopy(old)
    new["row_results"][0]["ignition_delay_s"] = 1.0
    result = idt_separation(old, new)
    assert result["primary"]["gap"] == pytest.approx(math.log10(2))
    assert result["diagnostics"]["groups"]["shock-tube"]["row_count"] == 0
    assert result["coverage"]["v_new"]["refusals"][0]["ignition_target"] == "CH*"
    new["row_results"][0]["temperature_k"] = 900.0
    with pytest.raises(ValueError, match="observation"):
        idt_separation(old, new)
    new = copy.deepcopy(old)
    new["simulation_lock_sha256"] = "different"
    with pytest.raises(ValueError, match="locks"):
        idt_separation(old, new)
    new = copy.deepcopy(old)
    new["holdout_fingerprint"]["sha256"] = "different"
    with pytest.raises(ValueError, match="holdout"):
        idt_separation(old, new)
    new = copy.deepcopy(old)
    new["row_results"].append(new["row_results"][0])
    with pytest.raises(ValueError, match="duplicate"):
        idt_separation(old, new)


CASE2_SHA256 = hashlib.sha256((ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes()).hexdigest()


def _jsr_report() -> dict[str, Any]:
    rows = [
        {
            "row_id": str(i),
            "species": s,
            "experimental": exp,
            "simulated": sim,
            "temperature_k": t,
            "detection_limit": None,
        }
        for i, t in enumerate((600.0, 900.0, 1000.0))
        for s, exp, sim in (("CH4", 0.01, 0.02), ("O2", 0.1, 0.2))
    ]
    return {
        "case": "n-pentane-low-t",
        "simulation_lock_sha256": "lock",
        "holdout_fingerprint": {"sha256": "frozen", "contract_sha256": CASE2_SHA256},
        "metric_rows": rows,
        "metric": score_metric(rows),
        "species_mapping": {s: {"mechanism_label": s} for s in ("CH4", "O2")},
        "row_results": [
            {"row_id": str(i), "status": "ok", "reason": None, "temperature_k": t}
            for i, t in enumerate((600.0, 900.0, 1000.0))
        ],
    }


def _write_saved_jsr(
    tmp_path: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    temperatures: tuple[float, ...] = (600.0, 900.0, 1000.0),
) -> Path:
    """Serialize the historical checkpoint identity independently of the worker."""
    report.setdefault("simulation_lock", {"schema_version": 1, "jsr": {"steady_state_timeout_s": 600.0}})
    report.update(inputs={"mechanism_semantic_sha256": "saved-mechanism"}, plog_validation_sha256="saved-plog")
    requests = [{"row_id": str(i), "temperature_k": t} for i, t in enumerate(temperatures)]
    (tmp_path / "request.json").write_text(json.dumps({"rows": requests}))
    entries = []
    for row in rows:
        request = next((r for r in requests if r["row_id"] == row["row_id"]), {"row_id": row["row_id"]})
        identity = {
            "mechanism_semantic_sha256": "saved-mechanism",
            "simulation_lock_sha256": report["simulation_lock_sha256"],
            "plog_validation_sha256": "saved-plog",
            "row": request,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        entries.append({"fingerprint": fingerprint, "result": row})
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def test_jsr_reuses_clamps_and_reports_requested_groups() -> None:
    old = _jsr_report()
    new = copy.deepcopy(old)
    for row in new["metric_rows"]:
        row["simulated"] = row["experimental"]
    new["metric"] = score_metric(new["metric_rows"])
    result = jsr_intervals(old, new)
    assert result["primary"]["v_old"] == pytest.approx(result["separation"]["primary"]["v_old"]["all"]["overall"])
    assert result["primary"]["gap"] == pytest.approx(math.log10(2))
    assert result["diagnostics"]["600_to_900_k"]["row_count"] == 2
    assert result["diagnostics"]["species_groups"]["O2"]["interval_95"] == pytest.approx([math.log10(2)] * 2)
    assert result["diagnostics"]["species_groups"]["C5_cyclic_ethers"]["interval_95"] is None
    assert json.dumps(result, sort_keys=True) == json.dumps(jsr_intervals(old, new), sort_keys=True)
    new["metric_rows"][0]["experimental"] = 0.1
    with pytest.raises(ValueError, match="observation"):
        jsr_intervals(old, new)


def test_saved_checkpoint_rescore_is_read_only_and_rejects_disagreement(tmp_path: Path) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    path = _write_saved_jsr(tmp_path, report, rows)
    checkpoint = tmp_path / "rows.jsonl"
    before = path.read_bytes(), checkpoint.read_bytes()
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["metric"] == report["metric"]
    assert before == (path.read_bytes(), checkpoint.read_bytes())
    assert rebuilt["rescore_inputs"]["rows_sha256"]
    report["metric_rows"][0]["simulated"] = 0.03
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="simulation differ"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("duplicate", "conflicting saved"),
        ("missing", "different rows"),
        ("foreign", "different rows"),
        ("refusal", "status differ"),
    ],
)
def test_saved_checkpoint_rejects_inconsistent_rows(tmp_path: Path, defect: str, message: str) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    if defect == "duplicate":
        rows.append({**rows[0], "mole_fractions": {"CH4": 0.03, "O2": 0.2}})
    elif defect == "missing":
        rows.pop()
    elif defect == "foreign":
        rows[0]["row_id"] = "foreign"
    else:
        rows[0].update(status="refused", reason="steady_state_timeout")
    path = _write_saved_jsr(tmp_path, report, rows)
    checkpoint = tmp_path / "rows.jsonl"
    before = path.read_bytes(), checkpoint.read_bytes()
    with pytest.raises(ValueError, match=message):
        rescore_saved_jsr(path)
    assert before == (path.read_bytes(), checkpoint.read_bytes())


def test_saved_checkpoint_preserves_unmapped_observations_and_matching_refusals(tmp_path: Path) -> None:
    report = _jsr_report()
    del report["species_mapping"]["O2"]
    rows = [{"row_id": str(i), "status": "refused", "reason": "steady_state_timeout"} for i in range(3)]
    for row in report["row_results"]:
        row.update(status="refused", reason="steady_state_timeout")
    for row in report["metric_rows"]:
        if row["species"] == "CH4":
            row.update(simulated=None, refusal="steady_state_timeout")
        else:
            # The harness writes an unmapped observation as a refusal with no value.
            del row["simulated"]
            row["refusal"] = "unmapped_species"
    report["metric"] = score_metric(report["metric_rows"])
    path = _write_saved_jsr(tmp_path, report, rows)
    checkpoint = tmp_path / "rows.jsonl"
    before = path.read_bytes(), checkpoint.read_bytes()
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["metric_rows"] == report["metric_rows"]
    assert rebuilt["metric"] == score_metric(report["metric_rows"])
    assert before == (path.read_bytes(), checkpoint.read_bytes())


@pytest.mark.parametrize(
    "unmapped",
    [{"simulated": 0.2}, {"simulated": 0.2, "refusal": "unmapped_species"}, {"refusal": "steady_state_timeout"}],
)
def test_rescore_rejects_unmapped_observation_without_its_refusal(tmp_path: Path, unmapped: dict[str, Any]) -> None:
    report = _jsr_report()
    del report["species_mapping"]["O2"]
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02}} for i in range(3)]
    for row in report["metric_rows"]:
        if row["species"] == "O2":
            del row["simulated"]
            row.update(unmapped)
    path = _write_saved_jsr(tmp_path, report, rows)
    with pytest.raises(ValueError, match="invalid unmapped observation"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize("stale_first", [True, False])
def test_rescore_selects_saved_identity_before_checking_duplicates(tmp_path: Path, stale_first: bool) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    path = _write_saved_jsr(tmp_path, report, rows)
    checkpoint = tmp_path / "rows.jsonl"
    current = checkpoint.read_text()
    stale = (
        json.dumps(
            {
                "fingerprint": "stale-lock-or-mechanism",
                "result": {"row_id": "0", "status": "refused", "reason": "steady_state_timeout"},
            }
        )
        + "\n"
    )
    checkpoint.write_text(stale + current if stale_first else current + stale)
    before = path.read_bytes(), checkpoint.read_bytes()
    assert rescore_saved_jsr(path)["metric"] == report["metric"]
    assert before == (path.read_bytes(), checkpoint.read_bytes())
    # Identical repeated completion records are not conflicts within an identity.
    checkpoint.write_text(checkpoint.read_text() + current.splitlines()[0] + "\n")
    assert rescore_saved_jsr(path)["metric"] == report["metric"]


@pytest.mark.parametrize("checkpoint_status", ["ok", "refused"])
def test_rescore_rejects_report_status_disagreement_even_when_metrics_match(
    tmp_path: Path, checkpoint_status: str
) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    if checkpoint_status == "ok":
        report["row_results"][0].update(status="refused", reason="steady_state_timeout")
    else:
        rows[0].update(status="refused", reason="steady_state_timeout")
        for row in report["metric_rows"]:
            if row["row_id"] == "0":
                row.update(simulated=None, refusal="steady_state_timeout")
    path = _write_saved_jsr(tmp_path, report, rows)
    with pytest.raises(ValueError, match="status differ"):
        rescore_saved_jsr(path)


def test_rescore_preserves_validated_pre_dispatch_refusals_without_checkpoint(tmp_path: Path) -> None:
    report = _jsr_report()
    report["row_results"][0].update(status="refused", reason="unmapped_inlet_species")
    report["unmapped_species"] = ["unknown_inlet", "O2"]
    del report["species_mapping"]["O2"]
    for row in report["metric_rows"]:
        if row["row_id"] == "0":
            row.pop("simulated")
            row["refusal"] = "unmapped_species" if row["species"] == "O2" else "unmapped_inlet_species"
        elif row["species"] == "O2":
            row.pop("simulated")
            row["refusal"] = "unmapped_species"
    report["metric"] = score_metric(report["metric_rows"])
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in (1, 2)]
    path = _write_saved_jsr(tmp_path, report, rows)
    request_path = tmp_path / "request.json"
    request = json.loads(request_path.read_text())
    request["rows"] = request["rows"][1:]
    request_path.write_text(json.dumps(request))
    before = {p: p.read_bytes() for p in (path, request_path, tmp_path / "rows.jsonl")}
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["row_results"] == report["row_results"]
    assert rebuilt["metric_rows"] == report["metric_rows"]
    assert rebuilt["metric"] == score_metric(report["metric_rows"])
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize(("reason", "omitted"), [(None, False), (None, True), ("", False), ("   ", False)])
def test_rescore_rejects_empty_checkpoint_refusal_reason(tmp_path: Path, reason: str | None, omitted: bool) -> None:
    report = _jsr_report()
    report["row_results"][0].update(status="refused", reason=reason)
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    rows[0].update(status="refused", reason=reason)
    if omitted:
        rows[0].pop("reason")
    for row in report["metric_rows"]:
        if row["row_id"] == "0":
            row["refusal"] = reason
    path = _write_saved_jsr(tmp_path, report, rows)
    with pytest.raises(ValueError, match="nonempty.*refusal reason"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize("component", ["mechanism", "lock", "plog", "row"])
def test_rescore_never_falls_back_to_row_id_when_saved_identity_changes(tmp_path: Path, component: str) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    path = _write_saved_jsr(tmp_path, report, rows)
    if component == "row":
        request_path = tmp_path / "request.json"
        request = json.loads(request_path.read_text())
        request["rows"][0]["temperature_k"] = 610.0
        request_path.write_text(json.dumps(request))
    else:
        if component == "mechanism":
            report["inputs"]["mechanism_semantic_sha256"] = "changed"
        else:
            report["simulation_lock_sha256" if component == "lock" else "plog_validation_sha256"] = "changed"
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checkpoint fingerprints do not match"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("request_duplicate", "duplicate saved request"),
        ("report_duplicate", "duplicate saved report"),
        ("report_missing", "different rows"),
        ("unknown_status", "status differ"),
        ("report_reason", "reported refusal differ"),
        ("metric_reason", "reported refusal differ"),
    ],
)
def test_rescore_rejects_conflicting_saved_provenance(tmp_path: Path, defect: str, message: str) -> None:
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    if defect == "report_duplicate":
        report["row_results"].append(report["row_results"][0])
    elif defect == "report_missing":
        report["row_results"].pop()
    elif defect == "unknown_status":
        report["row_results"][0]["status"] = rows[0]["status"] = "unknown"
    elif defect in ("report_reason", "metric_reason"):
        rows[0].update(status="refused", reason="steady_state_timeout")
        report["row_results"][0].update(
            status="refused", reason="steady_state_timeout" if defect == "metric_reason" else "solver_crash"
        )
        for row in report["metric_rows"]:
            if row["row_id"] == "0":
                row.update(
                    simulated=None, refusal="solver_crash" if defect == "metric_reason" else "steady_state_timeout"
                )
    path = _write_saved_jsr(tmp_path, report, rows)
    if defect == "request_duplicate":
        request_path = tmp_path / "request.json"
        request = json.loads(request_path.read_text())
        request["rows"].append(request["rows"][0])
        request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match=message):
        rescore_saved_jsr(path)


@pytest.mark.parametrize(
    "defect", ["status", "reason", "evidence", "metric_reason", "simulation", "unknown_metric_row"]
)
def test_rescore_rejects_unverified_pre_dispatch_refusals(tmp_path: Path, defect: str) -> None:
    report = _jsr_report()
    report["row_results"][0].update(status="refused", reason="unmapped_inlet_species")
    report["unmapped_species"] = ["unknown_inlet"]
    for row in report["metric_rows"]:
        if row["row_id"] == "0":
            row.update(simulated=None, refusal="unmapped_inlet_species")
    if defect == "status":
        report["row_results"][0]["status"] = "ok"
    elif defect == "reason":
        report["row_results"][0]["reason"] = "steady_state_timeout"
    elif defect == "evidence":
        report["unmapped_species"] = []
    elif defect == "metric_reason":
        report["metric_rows"][0]["refusal"] = "steady_state_timeout"
    elif defect == "simulation":
        report["metric_rows"][0]["simulated"] = 0.02
    else:
        report["metric_rows"][0]["row_id"] = "unknown"
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in (1, 2)]
    path = _write_saved_jsr(tmp_path, report, rows)
    request_path = tmp_path / "request.json"
    request = json.loads(request_path.read_text())
    request["rows"] = request["rows"][1:]
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="invalid pre-dispatch"):
        rescore_saved_jsr(path)


def test_rescore_all_pre_dispatch_refusals_needs_no_checkpoint_file(tmp_path: Path) -> None:
    report = _jsr_report()
    report["unmapped_species"] = ["unknown_inlet"]
    for row in report["row_results"]:
        row.update(status="refused", reason="unmapped_inlet_species")
    for row in report["metric_rows"]:
        row.update(simulated=None, refusal="unmapped_inlet_species")
    path = _write_saved_jsr(tmp_path, report, [])
    (tmp_path / "request.json").write_text(json.dumps({"rows": []}))
    (tmp_path / "rows.jsonl").unlink()
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["row_results"] == report["row_results"]
    assert rebuilt["metric"] == score_metric(report["metric_rows"])
    assert rebuilt["rescore_inputs"]["rows_sha256"] is None
    assert not (tmp_path / "rows.jsonl").exists()


@pytest.mark.parametrize(
    ("defect", "message"), [("limits", "different frozen detection limits"), ("duplicate", "duplicate metric")]
)
def test_jsr_intervals_reject_incompatible_observations(defect: str, message: str) -> None:
    old = _jsr_report()
    new = copy.deepcopy(old)
    if defect == "limits":
        new["metric_rows"][0]["detection_limit"] = 0.001
    else:
        new["metric_rows"].append(copy.deepcopy(new["metric_rows"][0]))
    with pytest.raises(ValueError, match=message):
        jsr_intervals(old, new)


def test_jsr_intervals_exclude_failed_rows_and_invalid_observables() -> None:
    old = _jsr_report()
    new = copy.deepcopy(old)
    new["row_results"][0].update(status="refused", reason="steady_state_timeout")
    new["metric_rows"][2].update(simulated=None, refusal="species_absent")
    result = jsr_intervals(old, new)
    assert result["primary"]["row_ids"] == ["1", "2"]
    assert result["primary"]["observable_count"] == 3
    assert result["primary"]["gap"] == 0.0
    assert result["primary"]["interval_95"] == [0.0, 0.0]


@pytest.mark.parametrize("lock", [{}, {"schema_version": 2}, {"schema_version": 1}])
def test_saved_timeout_requires_historical_lock(tmp_path: Path, lock: dict[str, Any]) -> None:
    report = _jsr_report()
    report["simulation_lock"] = lock
    rows = [{"row_id": str(i), "status": "refused", "reason": "steady_state_timeout"} for i in range(3)]
    for row in report["row_results"]:
        row.update(status="refused", reason="steady_state_timeout")
    path = _write_saved_jsr(tmp_path, report, rows)
    with pytest.raises(ValueError, match="historical simulation lock"):
        rescore_saved_jsr(path)


CASE3_CONTRACT = ROOT / "benchmarks/iso-pentane-low-t/contract.json"
CASE2_PREVIOUS_SHA256 = json.loads((ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes())["revision"][
    "previous_contract_sha256"
]
CASE2_REVISION0_SHA256 = "ef3755d08c60f6aba4e272601f6082307c60f7392841c8ecff28fa48b0af25ad"
LOCAL_RUNS = os.environ.get("CARMEL_LOCAL_RUNS", "")
CASE2_REPORTS = Path(LOCAL_RUNS, "i139-deterministic-row-budget/case_2")


def _species_report(
    case: str, contract_sha256: str, species: tuple[str, ...], temperatures: tuple[float, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a saved report and its checkpoint rows over every species at every temperature."""
    rows = [
        {
            "row_id": str(i),
            "species": name,
            "experimental": 0.01 * (k + 1),
            "simulated": 0.02 * (k + 1) * (1 + i % 3),
            "temperature_k": t,
            "detection_limit": None,
        }
        for i, t in enumerate(temperatures)
        for k, name in enumerate(species)
    ]
    report = {
        "case": case,
        "simulation_lock_sha256": "lock",
        "holdout_fingerprint": {"sha256": "frozen", "contract_sha256": contract_sha256},
        "metric_rows": rows,
        "metric": score_metric(rows),
        "species_mapping": {name: {"mechanism_label": name} for name in species},
        "row_results": [
            {"row_id": str(i), "status": "ok", "reason": None, "temperature_k": t} for i, t in enumerate(temperatures)
        ],
    }
    checkpoint = [
        {
            "row_id": str(i),
            "status": "ok",
            "mole_fractions": {row["species"]: row["simulated"] for row in rows if row["row_id"] == str(i)},
        }
        for i in range(len(temperatures))
    ]
    return report, checkpoint


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def test_case2_rescore_reproduces_the_stored_target_byte_for_byte(tmp_path: Path) -> None:
    report, checkpoint = _species_report(
        "n-pentane-low-t", CASE2_SHA256, ("CH4", "C5H10O1-4", "C5H10O2-3"), (600.0, 900.0, 1000.0)
    )
    report["target_chemistry"] = holdout.target_jsr_metric(report["metric_rows"])
    path = _write_saved_jsr(tmp_path, report, checkpoint)
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["target_chemistry"]["total_observations"] == 6
    assert _canonical(rebuilt["target_chemistry"]) == _canonical(report["target_chemistry"])
    assert rebuilt["target_chemistry_contract_sha256"] == CASE2_SHA256


def test_case3_rescore_scores_its_temperature_window(tmp_path: Path) -> None:
    species = tuple(f"S{k:02d}" for k in range(14))
    temperatures = tuple(600.0 + 10.0 * i for i in range(31)) + (950.0, 1000.0)
    contract_bytes = CASE3_CONTRACT.read_bytes()
    report, checkpoint = _species_report(
        "iso-pentane-low-t", hashlib.sha256(contract_bytes).hexdigest(), species, temperatures
    )
    report["target_chemistry"] = holdout.target_jsr_metric(report["metric_rows"])
    path = _write_saved_jsr(tmp_path, report, checkpoint, temperatures)
    rebuilt = rescore_saved_jsr(path)
    target = holdout.load_contract(contract_bytes).scoring.target_chemistry
    expected = holdout.target_window_metric(report["metric_rows"], target)
    assert (expected["total_rows"], expected["total_observations"]) == (31, 434)
    assert _canonical(rebuilt["target_chemistry"]) == _canonical(expected)
    assert rebuilt["target_chemistry_contract_sha256"] == hashlib.sha256(contract_bytes).hexdigest()


def test_rescore_recomputes_a_target_its_rows_no_longer_support(tmp_path: Path) -> None:
    report, checkpoint = _species_report("n-pentane-low-t", CASE2_SHA256, ("CH4", "C5H10O1-4"), (600.0, 900.0, 1000.0))
    stale = holdout.target_jsr_metric(report["metric_rows"])
    stale["overall"] = 99.0
    report["target_chemistry"] = stale
    path = _write_saved_jsr(tmp_path, report, checkpoint)
    rebuilt = rescore_saved_jsr(path)
    assert rebuilt["target_chemistry"] == holdout.target_jsr_metric(report["metric_rows"])
    assert rebuilt["target_chemistry"]["overall"] != 99.0


def test_rescore_accepts_one_recorded_revision_back_and_keeps_the_recorded_digest(tmp_path: Path) -> None:
    report = _jsr_report()
    report["holdout_fingerprint"]["contract_sha256"] = CASE2_PREVIOUS_SHA256
    checkpoint = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    rebuilt = rescore_saved_jsr(_write_saved_jsr(tmp_path, report, checkpoint))
    assert "target_chemistry" not in report
    assert rebuilt["target_chemistry"] == holdout.target_jsr_metric(report["metric_rows"])
    assert rebuilt["target_chemistry_contract_sha256"] == CASE2_SHA256
    assert rebuilt["holdout_fingerprint"]["contract_sha256"] == CASE2_PREVIOUS_SHA256


@pytest.mark.parametrize("recorded", [CASE2_REVISION0_SHA256, "0" * 64])
def test_rescore_refuses_a_digest_outside_one_revision_hop(tmp_path: Path, recorded: str) -> None:
    report = _jsr_report()
    report["holdout_fingerprint"]["contract_sha256"] = recorded
    path = _write_saved_jsr(tmp_path, report, [])
    with pytest.raises(ValueError) as refused:
        rescore_saved_jsr(path)
    message = str(refused.value)
    assert recorded in message and CASE2_SHA256 in message and CASE2_PREVIOUS_SHA256 in message


@pytest.mark.parametrize(
    "strip",
    [
        lambda report: report.pop("case"),
        lambda report: report.pop("holdout_fingerprint"),
        lambda report: report["holdout_fingerprint"].pop("contract_sha256"),
        lambda report: report.update(case=7),
        lambda report: report.update(holdout_fingerprint="frozen"),
        lambda report: report["holdout_fingerprint"].update(contract_sha256=None),
    ],
)
def test_rescore_refuses_a_report_without_contract_identity(tmp_path: Path, strip: Any) -> None:
    report = _jsr_report()
    strip(report)
    path = _write_saved_jsr(tmp_path, report, [])
    with pytest.raises(ValueError, match="does not record its case"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize("case", ["../n-pentane-low-t", "unknown-case"])
def test_rescore_refuses_a_case_without_a_committed_contract(tmp_path: Path, case: str) -> None:
    report = _jsr_report()
    report["case"] = case
    path = _write_saved_jsr(tmp_path, report, [])
    with pytest.raises(ValueError, match="has no committed contract"):
        rescore_saved_jsr(path)


def test_rescore_refuses_a_non_jsr_case(tmp_path: Path) -> None:
    report = _jsr_report()
    report["case"] = "n-heptane-low-t"
    path = _write_saved_jsr(tmp_path, report, [])
    with pytest.raises(ValueError, match="does not score case n-heptane-low-t"):
        rescore_saved_jsr(path)


def test_rescore_of_a_contract_without_revision_accepts_only_its_own_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = holdout.load_contract((ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes())
    monkeypatch.setattr(holdout_intervals, "load_contract", lambda _: contract.model_copy(update={"revision": None}))
    report = _jsr_report()
    report["holdout_fingerprint"]["contract_sha256"] = CASE2_PREVIOUS_SHA256
    path = _write_saved_jsr(tmp_path, report, [])
    with pytest.raises(ValueError, match="recorded predecessor None"):
        rescore_saved_jsr(path)


@pytest.mark.skipif(
    not LOCAL_RUNS or not (CASE2_REPORTS / "v_new/report.json").is_file(), reason="stored case-2 reports unavailable"
)
@pytest.mark.parametrize("key", ["v_new", "v_old", "v_old_loaded"])
def test_stored_case2_reports_rescore_byte_identically(key: str) -> None:
    path = CASE2_REPORTS / key / "report.json"
    stored = json.loads(path.read_bytes())
    rebuilt = rescore_saved_jsr(path)
    assert _canonical(rebuilt["metric"]) == _canonical(stored["metric"])
    assert _canonical(rebuilt["target_chemistry"]) == _canonical(holdout.target_jsr_metric(stored["metric_rows"]))
    assert rebuilt["target_chemistry_contract_sha256"] == CASE2_SHA256
    assert rebuilt["holdout_fingerprint"] == stored["holdout_fingerprint"]


LEGACY_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
LEGACY_START = "forkserver"


def _legacy_fingerprint(row: dict[str, Any], thread_env: Any, start_method: Any) -> str:
    """Hash the pre-lock checkpoint identity independently of the worker."""
    identity = {
        "mechanism_semantic_sha256": "saved-mechanism",
        "thread_env": thread_env,
        "process_start_method": start_method,
        "row": row,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_legacy_jsr(
    tmp_path: Path, count: int = 3, response: dict[str, Any] | None = None, *, write_response: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Write a saved JSR run whose checkpoint is keyed wholly under the pre-lock form."""
    temperatures = tuple(600.0 + i for i in range(count))
    metric_rows = [
        {
            "row_id": str(i),
            "species": s,
            "experimental": exp,
            "simulated": sim,
            "temperature_k": t,
            "detection_limit": None,
        }
        for i, t in enumerate(temperatures)
        for s, exp, sim in (("CH4", 0.01, 0.02), ("O2", 0.1, 0.2))
    ]
    report = _jsr_report()
    report.update(
        metric_rows=metric_rows,
        metric=score_metric(metric_rows),
        row_results=[
            {"row_id": str(i), "status": "ok", "reason": None, "temperature_k": t} for i, t in enumerate(temperatures)
        ],
    )
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(count)]
    path = _write_saved_jsr(tmp_path, report, rows, temperatures)
    report.update(thread_env=LEGACY_ENV, row_process_start_method=LEGACY_START)
    path.write_text(json.dumps(report))
    requests = {r["row_id"]: r for r in json.loads((tmp_path / "request.json").read_text())["rows"]}
    entries = [
        {"fingerprint": _legacy_fingerprint(requests[r["row_id"]], LEGACY_ENV, LEGACY_START), "result": r} for r in rows
    ]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    if write_response:
        recorded = {"thread_env": LEGACY_ENV, "row_process_start_method": LEGACY_START, "simulation_lock_sha256": "run"}
        identity = {"mechanism_semantic_sha256": "saved-mechanism", "rows": rows}
        (tmp_path / "response.json").write_text(
            json.dumps({**identity, **(recorded if response is None else response)})
        )
    return path, report


def _inputs(tmp_path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir())}


def test_rescore_accepts_whole_file_pre_lock_v0_and_labels_it(tmp_path: Path) -> None:
    """A checkpoint keyed wholly under pre_lock_v0 rescores, labelled lock-unbound, with the run's own lock."""
    path, report = _write_legacy_jsr(tmp_path)
    assert LEGACY_ENV != cantera_worker.THREAD_ENV and LEGACY_START != cantera_worker.ROW_PROCESS_START_METHOD
    before = _inputs(tmp_path)
    rebuilt = rescore_saved_jsr(path)
    assert before == _inputs(tmp_path)
    assert rebuilt["metric"] == report["metric"]
    inputs = rebuilt["rescore_inputs"]
    assert inputs["checkpoint_fingerprint_form"] == "pre_lock_v0"
    assert "bound to no simulation lock" in inputs["simulation_lock_binding"]
    assert "simulation_lock_sha256" not in rebuilt and "simulation_lock" not in rebuilt
    response = tmp_path / "response.json"
    assert inputs["run_recorded_lock"] == {
        "simulation_lock_sha256": "run",
        "source_path": str(response),
        "source_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("response", [None, {"thread_env": LEGACY_ENV}])
def test_rescore_pre_lock_v0_omits_run_lock_without_a_run_written_source(
    tmp_path: Path, response: dict[str, Any] | None
) -> None:
    """With no response.json, or one that records no lock, no run_recorded_lock appears."""
    path, _ = _write_legacy_jsr(tmp_path, response=response, write_response=response is not None)
    inputs = rescore_saved_jsr(path)["rescore_inputs"]
    assert inputs["checkpoint_fingerprint_form"] == "pre_lock_v0"
    assert "run_recorded_lock" not in inputs


def test_rescore_refuses_pre_lock_v0_file_with_one_altered_line(tmp_path: Path) -> None:
    """One altered line among 144 pre_lock_v0 lines refuses the whole file."""
    path, _ = _write_legacy_jsr(tmp_path, count=144)
    checkpoint = tmp_path / "rows.jsonl"
    entries = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    entries[71]["fingerprint"] = "0" * 64
    checkpoint.write_text("".join(json.dumps(e) + "\n" for e in entries))
    with pytest.raises(ValueError, match=r"^1/144 checkpoint fingerprints do not match the pre_lock_v0 form"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize(
    "defect",
    ["no_thread_env", "no_start_method", "thread_env_list", "thread_env_empty", "thread_env_int", "start_blank"],
)
def test_rescore_pre_lock_v0_refuses_without_recorded_run_fields(tmp_path: Path, defect: str) -> None:
    """A missing or ill-typed thread_env or start method refuses; no constant or default stands in."""
    path, report = _write_legacy_jsr(tmp_path)
    if defect == "no_thread_env":
        del report["thread_env"]
    elif defect == "no_start_method":
        del report["row_process_start_method"]
    elif defect == "start_blank":
        report["row_process_start_method"] = " "
    else:
        report["thread_env"] = {"thread_env_list": ["1"], "thread_env_empty": {}, "thread_env_int": {"A": 1}}[defect]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=r"^3/3 checkpoint fingerprints do not match; pre_lock_v0 not checked"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize("field", ["thread_env", "row_process_start_method"])
def test_rescore_pre_lock_v0_refuses_when_response_disagrees(tmp_path: Path, field: str) -> None:
    """The report's run fields must agree with the run's own response.json."""
    response = {"thread_env": LEGACY_ENV, "row_process_start_method": LEGACY_START}
    response[field] = {"OMP_NUM_THREADS": "1"} if field == "thread_env" else "spawn"
    path, _ = _write_legacy_jsr(tmp_path, response=response)
    with pytest.raises(ValueError, match=f"record different {field}"):
        rescore_saved_jsr(path)


@pytest.mark.parametrize("majority", ["today", "pre_lock_v0"])
def test_rescore_refuses_mixed_fingerprint_forms(tmp_path: Path, majority: str) -> None:
    """A checkpoint holding lines of both forms refuses, whichever form holds most lines."""
    path, report = _write_legacy_jsr(tmp_path)
    checkpoint = tmp_path / "rows.jsonl"
    legacy = checkpoint.read_text().splitlines()
    results = [json.loads(line)["result"] for line in legacy]
    _write_saved_jsr(tmp_path, copy.deepcopy(report), results, (600.0, 601.0, 602.0))
    path.write_text(json.dumps(report))
    today = checkpoint.read_text().splitlines()
    lines = today[:2] + legacy[2:] if majority == "today" else today[:1] + legacy[1:]
    checkpoint.write_text("".join(line + "\n" for line in lines))
    with pytest.raises(ValueError, match="mixed checkpoint fingerprint forms"):
        rescore_saved_jsr(path)


def test_rescore_counts_fingerprint_mismatches(tmp_path: Path) -> None:
    """When every row is present but no fingerprint matches, the refusal says so with real counts."""
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    path = _write_saved_jsr(tmp_path, report, rows)
    report["plog_validation_sha256"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=r"^3/3 checkpoint fingerprints do not match"):
        rescore_saved_jsr(path)


def test_today_form_rescore_output_carries_no_legacy_labels(tmp_path: Path) -> None:
    """A today's-form rescore keeps its lock fields and its original four rescore_inputs keys."""
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    report.update(thread_env=LEGACY_ENV, row_process_start_method=LEGACY_START)
    rebuilt = rescore_saved_jsr(_write_saved_jsr(tmp_path, report, rows))
    assert set(rebuilt["rescore_inputs"]) == {"report_path", "report_sha256", "rows_path", "rows_sha256"}
    assert rebuilt["simulation_lock_sha256"] == report["simulation_lock_sha256"]
    assert rebuilt["simulation_lock"] == report["simulation_lock"]


@pytest.mark.parametrize("position", ["v_old", "v_new"])
def test_jsr_intervals_refuse_a_pre_lock_v0_rescore(tmp_path: Path, position: str) -> None:
    """Pairing refuses a pre_lock_v0 rescore, which carries no simulation lock to pair on."""
    path, _ = _write_legacy_jsr(tmp_path)
    legacy = rescore_saved_jsr(path)
    other = _jsr_report()
    pair = (legacy, other) if position == "v_old" else (other, legacy)
    with pytest.raises(ValueError, match=f"{position} is a pre_lock_v0 rescore: pairing needs a simulation lock"):
        jsr_intervals(*pair)


def _schema1_lock(*, timeout: bool = True) -> dict[str, Any]:
    """Return a schema-1 lock built from the committed lock, with or without the JSR timeout."""
    lock = cantera_worker.simulation_lock(600.0, "3.1.0")
    lock["schema_version"] = 1
    if timeout:
        lock["jsr"]["steady_state_timeout_s"] = 600.0
    return lock


def _write_legacy_timeout_jsr(
    tmp_path: Path, restamp: dict[str, Any], response: dict[str, Any] | None
) -> tuple[Path, dict[str, Any]]:
    """Write a pre_lock_v0 run whose row 0 is a matching steady_state_timeout refusal."""
    path, report = _write_legacy_jsr(tmp_path, response=response, write_response=response is not None)
    report["row_results"][0].update(status="refused", reason="steady_state_timeout")
    for row in report["metric_rows"]:
        if row["row_id"] == "0":
            row.update(simulated=None, refusal="steady_state_timeout")
    report["metric"] = score_metric(report["metric_rows"])
    report["simulation_lock"] = restamp
    path.write_text(json.dumps(report))
    checkpoint = tmp_path / "rows.jsonl"
    entries = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    entries[0]["result"] = {"row_id": "0", "status": "refused", "reason": "steady_state_timeout"}
    checkpoint.write_text("".join(json.dumps(e) + "\n" for e in entries))
    if response is not None:
        recorded = json.loads((tmp_path / "response.json").read_text())
        recorded["rows"][0] = entries[0]["result"]
        (tmp_path / "response.json").write_text(json.dumps(recorded))
    return path, report


def _bound_response(lock: dict[str, Any]) -> dict[str, Any]:
    """Return a response.json body that records the lock bound to its own digest."""
    return {
        "thread_env": LEGACY_ENV,
        "row_process_start_method": LEGACY_START,
        "simulation_lock": lock,
        "simulation_lock_sha256": holdout_intervals._lock_sha256(lock),
    }


def test_lock_sha256_matches_the_worker_digest() -> None:
    """The rescore's lock digest equals the digest the worker records for the same lock."""
    lock = cantera_worker.simulation_lock(600.0, "3.1.0")
    assert hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == cantera_worker.simulation_lock_sha256(600.0, "3.1.0")


def test_rescore_pre_lock_v0_checks_timeouts_against_the_run_lock(tmp_path: Path) -> None:
    """A schema-2 restamp does not refuse a timeout row whose run lock is a bound schema-1 lock."""
    restamp = {**_schema1_lock(), "schema_version": 2}
    path, report = _write_legacy_timeout_jsr(tmp_path, restamp, _bound_response(_schema1_lock()))
    before = _inputs(tmp_path)
    rebuilt = rescore_saved_jsr(path)
    assert before == _inputs(tmp_path)
    assert rebuilt["metric"] == report["metric"]
    assert rebuilt["rescore_inputs"]["checkpoint_fingerprint_form"] == "pre_lock_v0"
    assert "simulation_lock" not in rebuilt


@pytest.mark.parametrize(
    ("defect", "problem"),
    [
        ("absent", "no response.json"),
        ("no_lock", "holds no simulation_lock object"),
        ("lock_not_object", "holds no simulation_lock object"),
        ("no_digest", "does not bind to its simulation_lock_sha256"),
        ("wrong_digest", "does not bind to its simulation_lock_sha256"),
    ],
)
def test_rescore_pre_lock_v0_refuses_timeouts_without_a_bound_run_lock(
    tmp_path: Path, defect: str, problem: str
) -> None:
    """A timeout row on the pre_lock_v0 path refuses when the run's lock is missing or unbound."""
    response: dict[str, Any] | None = _bound_response(_schema1_lock())
    assert response is not None
    if defect == "absent":
        response = None
    elif defect == "no_lock":
        del response["simulation_lock"]
    elif defect == "lock_not_object":
        response["simulation_lock"] = "lock"
    elif defect == "no_digest":
        del response["simulation_lock_sha256"]
    else:
        response["simulation_lock"]["jsr"]["rtol"] = 1.0
    path, _ = _write_legacy_timeout_jsr(tmp_path, {**_schema1_lock(), "schema_version": 2}, response)
    with pytest.raises(ValueError, match=f"^steady_state_timeout requires its historical simulation lock: .*{problem}"):
        rescore_saved_jsr(path)


def test_rescore_pre_lock_v0_restamp_never_rescues_a_timeout(tmp_path: Path) -> None:
    """A valid schema-1 restamp cannot accept a timeout the bound run lock does not support."""
    path, _ = _write_legacy_timeout_jsr(tmp_path, _schema1_lock(), _bound_response(_schema1_lock(timeout=False)))
    with pytest.raises(ValueError, match=r"^steady_state_timeout requires its historical simulation lock$"):
        rescore_saved_jsr(path)


def test_today_form_timeout_still_checks_the_report_lock(tmp_path: Path) -> None:
    """Today's form judges a timeout by the report's lock and ignores a bound response.json lock."""
    report = _jsr_report()
    report["simulation_lock"] = {**_schema1_lock(), "schema_version": 2}
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    rows[0].update(status="refused", reason="steady_state_timeout")
    report["row_results"][0].update(status="refused", reason="steady_state_timeout")
    path = _write_saved_jsr(tmp_path, report, rows)
    (tmp_path / "response.json").write_text(json.dumps(_bound_response(_schema1_lock())))
    with pytest.raises(ValueError, match=r"^steady_state_timeout requires its historical simulation lock$"):
        rescore_saved_jsr(path)


def test_rescore_pre_lock_v0_accepts_a_matching_response(tmp_path: Path) -> None:
    """A response.json recording this run's mechanism and saved rows is used for the lock and the label."""
    path, report = _write_legacy_timeout_jsr(tmp_path, _schema1_lock(), _bound_response(_schema1_lock()))
    before = _inputs(tmp_path)
    rebuilt = rescore_saved_jsr(path)
    assert before == _inputs(tmp_path)
    assert rebuilt["metric"] == report["metric"]
    response = tmp_path / "response.json"
    assert rebuilt["rescore_inputs"]["run_recorded_lock"] == {
        "simulation_lock_sha256": holdout_intervals._lock_sha256(_schema1_lock()),
        "source_path": str(response),
        "source_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
    }


def _edit_response(tmp_path: Path, edit: Any) -> None:
    """Apply ``edit`` to the saved response.json body in place."""
    response = tmp_path / "response.json"
    body = json.loads(response.read_text())
    edit(body)
    response.write_text(json.dumps(body))


@pytest.mark.parametrize(
    ("mechanism", "message"),
    [
        ("other-mechanism", "^saved report and response.json record different mechanism_semantic_sha256$"),
        (None, "^response.json lacks a mechanism_semantic_sha256$"),
        (7, "^response.json lacks a mechanism_semantic_sha256$"),
    ],
)
def test_rescore_pre_lock_v0_refuses_another_runs_mechanism(tmp_path: Path, mechanism: Any, message: str) -> None:
    """A response.json with another, missing or ill-typed mechanism refuses, even with a bound schema-1 lock."""
    path, _ = _write_legacy_timeout_jsr(tmp_path, _schema1_lock(), _bound_response(_schema1_lock()))
    rescore_saved_jsr(path)

    def edit(body: dict[str, Any]) -> None:
        if mechanism is None:
            del body["mechanism_semantic_sha256"]
        else:
            body["mechanism_semantic_sha256"] = mechanism

    _edit_response(tmp_path, edit)
    with pytest.raises(ValueError, match=message):
        rescore_saved_jsr(path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("row_ok", "^saved row 0 differs between rows.jsonl and response.json$"),
        ("row_missing", "^response.json lacks saved row 1$"),
        ("no_rows", "^response.json holds no rows list keyed by row_id$"),
        ("rows_object", "^response.json holds no rows list keyed by row_id$"),
        ("row_without_id", "^response.json holds no rows list keyed by row_id$"),
        ("duplicate", "^duplicate response.json row ID: 2$"),
    ],
)
def test_rescore_pre_lock_v0_refuses_a_response_row_mismatch(tmp_path: Path, defect: str, message: str) -> None:
    """A response.json whose rows do not record every saved checkpoint result refuses."""
    path, _ = _write_legacy_timeout_jsr(tmp_path, _schema1_lock(), _bound_response(_schema1_lock()))
    rescore_saved_jsr(path)

    def edit(body: dict[str, Any]) -> None:
        rows = body["rows"]
        if defect == "row_ok":
            rows[0] = {"row_id": "0", "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}}
        elif defect == "row_missing":
            del rows[1]
        elif defect == "no_rows":
            del body["rows"]
        elif defect == "rows_object":
            body["rows"] = {row["row_id"]: row for row in rows}
        elif defect == "row_without_id":
            del rows[2]["row_id"]
        else:
            rows.append(dict(rows[2]))

    _edit_response(tmp_path, edit)
    with pytest.raises(ValueError, match=message):
        rescore_saved_jsr(path)


def _rerun_dir(source: Path, target: Path, rebuilt: dict[str, Any]) -> Path:
    """Copy a run's request, checkpoint and response beside a rescore output written as its report."""
    target.mkdir()
    for name in ("request.json", "rows.jsonl", "response.json"):
        if (source / name).exists():
            (target / name).write_bytes((source / name).read_bytes())
    path = target / "report.json"
    path.write_text(json.dumps(rebuilt))
    return path


def test_rescore_of_a_pre_lock_v0_rescore_is_accepted_again(tmp_path: Path) -> None:
    """A saved pre_lock_v0 rescore, rescored again from the same run files, stays pre_lock_v0 with an equal metric."""
    first_dir = tmp_path / "run"
    first_dir.mkdir()
    path, report = _write_legacy_jsr(first_dir)
    first = rescore_saved_jsr(path)
    assert "simulation_lock_sha256" not in first
    second_path = _rerun_dir(first_dir, tmp_path / "again", first)
    before = _inputs(second_path.parent)
    second = rescore_saved_jsr(second_path)
    assert before == _inputs(second_path.parent)
    assert second["rescore_inputs"]["checkpoint_fingerprint_form"] == "pre_lock_v0"
    assert second["metric"] == first["metric"] == report["metric"]
    assert second["rescore_inputs"]["report_sha256"] == hashlib.sha256(second_path.read_bytes()).hexdigest()
    assert second["rescore_inputs"]["run_recorded_lock"]["simulation_lock_sha256"] == "run"


@pytest.mark.parametrize("missing", ["thread_env", "row_process_start_method"])
def test_rescore_refuses_a_lockless_report_without_legacy_inputs(tmp_path: Path, missing: str) -> None:
    """A report lacking its lock and a legacy input is refused with a ValueError naming both gaps."""
    path, report = _write_legacy_jsr(tmp_path)
    del report["simulation_lock_sha256"], report[missing]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=rf"^report lacks simulation_lock_sha256.*report lacks a recorded {missing}"):
        rescore_saved_jsr(path)


def test_rescore_refuses_a_lockless_report_with_an_empty_checkpoint(tmp_path: Path) -> None:
    """A lock-less report whose checkpoint has no lines is refused, never accepted under today's form."""
    path, report = _write_legacy_jsr(tmp_path)
    del report["simulation_lock_sha256"]
    path.write_text(json.dumps(report))
    (tmp_path / "rows.jsonl").write_text("")
    with pytest.raises(ValueError, match=r"^report lacks simulation_lock_sha256.*the checkpoint has no lines"):
        rescore_saved_jsr(path)


def test_rescore_of_a_timeout_rescore_names_the_absent_report_lock(tmp_path: Path) -> None:
    """Rescoring a timeout rescore whose run lock is gone names the report's absent lock in the refusal."""
    first_dir = tmp_path / "run"
    first_dir.mkdir()
    path, _ = _write_legacy_timeout_jsr(
        first_dir, {**_schema1_lock(), "schema_version": 2}, _bound_response(_schema1_lock())
    )
    second_path = _rerun_dir(first_dir, tmp_path / "again", rescore_saved_jsr(path))
    (second_path.parent / "response.json").unlink()
    with pytest.raises(ValueError, match=r"no response\.json; report lacks simulation_lock_sha256$"):
        rescore_saved_jsr(second_path)


def test_today_form_report_with_lock_fields_is_accepted_as_before(tmp_path: Path) -> None:
    """A today's-form report rescores to the report itself plus only the recomputed target and rescore inputs."""
    report = _jsr_report()
    rows = [{"row_id": str(i), "status": "ok", "mole_fractions": {"CH4": 0.02, "O2": 0.2}} for i in range(3)]
    path = _write_saved_jsr(tmp_path, report, rows)
    stored = json.loads(path.read_bytes())
    rebuilt = rescore_saved_jsr(path)
    added = {"target_chemistry", "target_chemistry_contract_sha256", "rescore_inputs"}
    assert {k: v for k, v in rebuilt.items() if k not in added} == stored
    assert rebuilt["rescore_inputs"] == {
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows_path": str(tmp_path / "rows.jsonl"),
        "rows_sha256": hashlib.sha256((tmp_path / "rows.jsonl").read_bytes()).hexdigest(),
    }
