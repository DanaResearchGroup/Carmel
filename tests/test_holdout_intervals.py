"""Hand-checkable paired row uncertainty, with no Cantera dependency."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carmel.benchmarks.holdout import score_metric
from carmel.benchmarks.holdout_intervals import idt_separation, jsr_intervals, paired_interval, rescore_saved_jsr


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
        "simulation_lock_sha256": "lock",
        "holdout_fingerprint": {"sha256": "frozen"},
        "metric_rows": rows,
        "metric": score_metric(rows),
        "species_mapping": {s: {"mechanism_label": s} for s in ("CH4", "O2")},
        "row_results": [
            {"row_id": str(i), "status": "ok", "reason": None, "temperature_k": t}
            for i, t in enumerate((600.0, 900.0, 1000.0))
        ],
    }


def _write_saved_jsr(tmp_path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Serialize the historical checkpoint identity independently of the worker."""
    report.update(inputs={"mechanism_semantic_sha256": "saved-mechanism"}, plog_validation_sha256="saved-plog")
    requests = [{"row_id": str(i), "temperature_k": t} for i, t in enumerate((600.0, 900.0, 1000.0))]
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
    with pytest.raises(ValueError, match="different rows"):
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
