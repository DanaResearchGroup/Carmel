"""Paired row bootstrap diagnostics for frozen IDT and JSR separation scores."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carmel.benchmarks.cantera_worker import (
    checkpoint_fingerprint,
    checkpoint_fingerprint_pre_lock_v0,
    validate_row_refusal,
)
from carmel.benchmarks.contract import read_pinned_file
from carmel.benchmarks.holdout import (
    _metric_limits,
    _target_chemistry,
    build_separation_report,
    load_contract,
    score_metric,
)
from carmel.benchmarks.holdout_idt import idt_metric
from carmel.benchmarks.iso_pentane import Contract as IsoPentaneContract
from carmel.benchmarks.n_pentane import Contract as PentaneContract
from carmel.benchmarks.revision import FROZEN_REVISIONS

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261018
JSR_SPECIES_GROUPS = {
    "C0-C2": ("CH4", "C2H4", "C2H6", "CH2O", "CH3CHO", "CO", "CO2"),
    "fuel_and_alkenes": ("NC5H12", "C5H10-1", "C5H10-2", "C3H6"),
    "C5_cyclic_ethers": ("C5H10O1-2", "C5H10O1-3", "C5H10O1-4", "C5H10O2-3", "C5H10O2-4"),
    "O2": ("O2",),
}
PairErrors = dict[tuple[str, str], tuple[float, float]]
BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
PRE_LOCK_V0 = "pre_lock_v0"
PRE_LOCK_V0_LOCK_BINDING = (
    "none: pre_lock_v0 checkpoint fingerprints bind the mechanism, the row, thread_env and the start "
    "method; these rows are bound to no simulation lock and no PLOG validation"
)


def paired_interval(errors: PairErrors, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Resample whole row IDs with paired weights, then average species means."""
    row_ids = sorted({key[0] for key in errors})
    species = sorted({key[1] for key in errors})
    result: dict[str, Any] = {
        "row_count": len(row_ids),
        "row_ids": row_ids,
        "observable_count": len(errors),
        "species": species,
        "v_old": None,
        "v_new": None,
        "gap": None,
        "interval_95": None,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }
    if not errors:
        return result
    positions = {row_id: i for i, row_id in enumerate(row_ids)}
    columns = {name: j for j, name in enumerate(species)}
    old = np.zeros((len(row_ids), len(species)))
    new = np.zeros_like(old)
    present = np.zeros_like(old)
    for (row_id, name), (old_error, new_error) in errors.items():
        if not all(math.isfinite(v) and v >= 0 for v in (old_error, new_error)):
            raise ValueError("bootstrap errors must be finite and nonnegative")
        i, j = positions[row_id], columns[name]
        old[i, j], new[i, j], present[i, j] = old_error, new_error, 1.0
    counts = present.sum(axis=0)
    old_score, new_score = float(np.mean(old.sum(axis=0) / counts)), float(np.mean(new.sum(axis=0) / counts))
    rng = np.random.Generator(np.random.PCG64(seed))
    # Multinomial counts are exactly resampling N whole row IDs with replacement.
    weights = rng.multinomial(len(row_ids), np.full(len(row_ids), 1.0 / len(row_ids)), size=BOOTSTRAP_REPLICATES)
    sampled_counts = weights @ present
    sampled_sum = weights @ (old - new)
    represented = sampled_counts > 0
    sampled_means = np.divide(sampled_sum, sampled_counts, out=np.zeros_like(sampled_sum), where=represented)
    gaps = sampled_means.sum(axis=1) / represented.sum(axis=1)
    interval = np.percentile(gaps, [2.5, 97.5], method="linear").tolist()
    result.update(v_old=old_score, v_new=new_score, gap=old_score - new_score, interval_95=interval)
    return result


def _check_pair(old: dict[str, Any], new: dict[str, Any]) -> None:
    if old["simulation_lock_sha256"] != new["simulation_lock_sha256"]:
        raise ValueError("different simulation locks")
    if old["holdout_fingerprint"]["sha256"] != new["holdout_fingerprint"]["sha256"]:
        raise ValueError("different frozen holdout inputs")


def idt_separation(old: dict[str, Any], new: dict[str, Any], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Compare common valid IDT IDs, retaining all refusal and coverage diagnostics."""
    _check_pair(old, new)
    old_rows = {r["row_id"]: r for r in old["row_results"]}
    new_rows = {r["row_id"]: r for r in new["row_results"]}
    if len(old_rows) != len(old["row_results"]) or len(new_rows) != len(new["row_results"]):
        raise ValueError("duplicate IDT row IDs")
    a = {r["row_id"]: r["error"] for r in idt_metric(old["row_results"])["scored"]}
    b = {r["row_id"]: r["error"] for r in idt_metric(new["row_results"])["scored"]}
    errors = {(key, "idt"): (a[key], b[key]) for key in sorted(a.keys() & b.keys())}
    for key, _ in errors:
        if any(
            old_rows[key][field] != new_rows[key][field]
            for field in (
                "experimental_s",
                "temperature_k",
                "group",
                "temperature_band",
                "ignition_target",
                "ignition_type",
            )
        ):
            raise ValueError(f"different frozen IDT observation: {key}")
    groups = {
        group: paired_interval(
            {key: value for key, value in errors.items() if old_rows[key[0]]["group"] == group}, seed=seed
        )
        for group in ("shock-tube", "RCM-ideal")
    }
    bands = {
        band: paired_interval(
            {key: value for key, value in errors.items() if old_rows[key[0]]["temperature_band"] == band}, seed=seed
        )
        for band in ("below_750", "750_to_900", "above_900")
    }
    return {
        "schema_version": 1,
        "case": "n-heptane-low-t",
        "simulation_lock_sha256": old["simulation_lock_sha256"],
        "holdout_fingerprint": old["holdout_fingerprint"],
        "bootstrap": bootstrap_definition(seed),
        "primary": paired_interval(errors, seed=seed),
        "diagnostics": {"groups": groups, "temperature_bands": bands},
        "coverage": {"v_old": idt_metric(old["row_results"]), "v_new": idt_metric(new["row_results"])},
    }


def bootstrap_definition(seed: int) -> dict[str, Any]:
    return {
        "method": "paired_percentile",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "interval_probability": 0.95,
        "quantile_method": "linear",
        "rng": "numpy.PCG64",
        "numpy_version": np.__version__,
        "resampling_unit": "whole_row_id_with_replacement",
        "gap_sign": "v_old_error_minus_v_new_error",
        "diagnostics_are_not_contract_metric_replacements": True,
    }


def saved_report_contract(report: dict[str, Any]) -> tuple[PentaneContract | IsoPentaneContract, str]:
    """Find and verify the committed contract that a saved JSR report was scored against.

    The report must record its ``case`` and ``holdout_fingerprint.contract_sha256``. The case
    names a committed ``benchmarks/<case>/contract.json``, which is validated by
    ``holdout.load_contract`` (schema and frozen revision digest), as ``run_harness`` does. The
    recorded digest must equal the sha256 of those bytes or, one revision back, the contract's
    ``revision.previous_contract_sha256``; any other digest is refused, naming both accepted
    digests. Returns the contract and the sha256 of its committed bytes.
    """
    case = report.get("case")
    fingerprint = report.get("holdout_fingerprint")
    recorded = fingerprint.get("contract_sha256") if isinstance(fingerprint, dict) else None
    if not isinstance(case, str) or not isinstance(recorded, str):
        raise ValueError("saved report does not record its case and holdout_fingerprint.contract_sha256")
    if case not in FROZEN_REVISIONS:
        raise ValueError(f"saved report names case {case!r}, which has no committed contract")
    contract_bytes = read_pinned_file(BENCHMARKS_DIR / case / "contract.json")
    contract = load_contract(contract_bytes)
    committed = hashlib.sha256(contract_bytes).hexdigest()
    revision = getattr(contract, "revision", None)
    previous = revision.previous_contract_sha256 if revision is not None else None
    if recorded not in (committed, previous):
        raise ValueError(
            f"saved report contract {recorded} is neither the committed {case} contract {committed} "
            f"nor its recorded predecessor {previous}"
        )
    return contract, committed


def rescore_saved_jsr(report_path: Path) -> dict[str, Any]:
    """Rebuild a saved JSR report's simulated values and scores from its checkpoint.

    Nothing is written. Every reported simulated value must match the original checkpoint row
    with the same fingerprint. ``metric`` is rescored from the rebuilt rows, and
    ``target_chemistry`` is recomputed through the same dispatch as a fresh score, using the
    committed contract found by :func:`saved_report_contract`; its digest is recorded as
    ``target_chemistry_contract_sha256``. A stored target is never copied through.

    Checkpoint lines are matched under today's fingerprint form first. Only when no line matches
    it, the legacy ``pre_lock_v0`` form is tried, from the report's recorded ``thread_env`` and
    ``row_process_start_method``, and it must match every line. Such a rescore drops the report's
    lock fields, is labelled lock-unbound, and records the run's own lock from ``response.json``
    when that file records one. That file is used only when it records the report's mechanism
    and, for every saved checkpoint row, the same result.
    """
    report_bytes = report_path.read_bytes()
    report: dict[str, Any] = json.loads(report_bytes)
    contract, contract_sha256 = saved_report_contract(report)
    request = json.loads(report_path.with_name("request.json").read_bytes())
    mechanism_sha256 = report["inputs"]["mechanism_semantic_sha256"]
    has_lock = "simulation_lock_sha256" in report
    expected = (
        {
            row["row_id"]: checkpoint_fingerprint(
                mechanism_sha256,
                row,
                report["simulation_lock_sha256"],
                report["plog_validation_sha256"],
            )
            for row in request["rows"]
        }
        if has_lock
        else {}
    )
    if len({row["row_id"] for row in request["rows"]}) != len(request["rows"]):
        raise ValueError("duplicate saved request row IDs")
    checkpoint_path = report_path.with_name("rows.jsonl")
    checkpoint = checkpoint_path.read_bytes() if request["rows"] or checkpoint_path.exists() else None
    entries = [json.loads(line) for line in (checkpoint or b"").splitlines()]
    legacy_inputs, legacy_gap = _pre_lock_v0_inputs(report)
    legacy = (
        {
            row["row_id"]: checkpoint_fingerprint_pre_lock_v0(mechanism_sha256, row, *legacy_inputs)
            for row in request["rows"]
        }
        if legacy_inputs is not None
        else {}
    )
    unmatched = [entry for entry in entries if expected.get(entry["result"]["row_id"]) != entry["fingerprint"]]
    form = None
    response_path, response_bytes, response = report_path.with_name("response.json"), None, None
    if entries and len(unmatched) == len(entries) and legacy_inputs is not None:
        form = PRE_LOCK_V0
        response_path, response_bytes, response = _read_run_response(report_path, legacy_inputs)
        expected = legacy
        unmatched = [entry for entry in entries if expected.get(entry["result"]["row_id"]) != entry["fingerprint"]]
        if unmatched:
            raise ValueError(
                f"{len(unmatched)}/{len(entries)} checkpoint fingerprints do not match the pre_lock_v0 form, "
                "which must match every line"
            )
    elif any(legacy.get(entry["result"]["row_id"]) == entry["fingerprint"] for entry in unmatched):
        raise ValueError("mixed checkpoint fingerprint forms")
    if not has_lock and form != PRE_LOCK_V0:
        gap = f"report lacks {legacy_gap}" if legacy_gap else "the checkpoint has no lines"
        raise ValueError(
            "report lacks simulation_lock_sha256, so no checkpoint line can match today's form; "
            f"pre_lock_v0 not checked: {gap}"
        )
    results: dict[str, dict[str, Any]] = {}
    for entry in entries:
        row = entry["result"]
        if row["row_id"] not in expected or entry["fingerprint"] != expected[row["row_id"]]:
            continue
        if row["row_id"] in results and row != results[row["row_id"]]:
            raise ValueError(f"conflicting saved checkpoint row: {row['row_id']}")
        results[row["row_id"]] = row
    if form == PRE_LOCK_V0 and response is not None:
        _check_run_response_identity(response, mechanism_sha256, results)
    run_lock, run_lock_problem = _run_recorded_lock(response) if form == PRE_LOCK_V0 else (None, None)
    reported = {row["row_id"]: row for row in report["row_results"]}
    if len(reported) != len(report["row_results"]):
        raise ValueError("duplicate saved report row IDs")
    if set(results) != set(expected) and set(expected) <= {entry["result"]["row_id"] for entry in entries}:
        gap = f"; pre_lock_v0 not checked: report lacks {legacy_gap}" if legacy_gap and len(results) == 0 else ""
        raise ValueError(f"{len(unmatched)}/{len(entries)} checkpoint fingerprints do not match{gap}")
    if set(results) != set(expected) or not set(expected) <= set(reported):
        raise ValueError("checkpoint and report cover different rows")
    for row_id in reported.keys() - expected.keys():
        row = reported[row_id]
        if (
            row["status"] != "refused"
            or row.get("reason") != "unmapped_inlet_species"
            or not report.get("unmapped_species")
        ):
            raise ValueError(f"invalid pre-dispatch refusal: {row_id}")
    for row_id, result in results.items():
        if result["status"] not in ("ok", "refused") or result["status"] != reported[row_id]["status"]:
            raise ValueError(f"checkpoint and reported status differ: {row_id}")
        if result["status"] == "refused":
            reason = result.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"checkpoint requires a nonempty refusal reason: {row_id}")
            if form != PRE_LOCK_V0:
                validate_row_refusal(reason, report.get("simulation_lock", {}))
            else:
                try:
                    validate_row_refusal(reason, run_lock or {})
                except ValueError as error:
                    if run_lock_problem is None:
                        raise
                    lockless = "" if has_lock else "; report lacks simulation_lock_sha256"
                    raise ValueError(
                        f"{error}: the run's recorded lock is unusable: {run_lock_problem}{lockless}"
                    ) from error
            if reported[row_id].get("reason") != reason:
                raise ValueError(f"checkpoint and reported refusal differ: {row_id}")
    rebuilt = copy.deepcopy(report)
    for row in rebuilt["metric_rows"]:
        target = report["species_mapping"].get(row["species"])
        if row["row_id"] not in expected:
            refusal = "unmapped_species" if target is None else "unmapped_inlet_species"
            if row["row_id"] not in reported or row.get("refusal") != refusal or row.get("simulated") is not None:
                raise ValueError("invalid pre-dispatch metric refusal")
            continue
        result = results[row["row_id"]]
        if target is None:
            # No mechanism species backs this observation, so no checkpoint value
            # can verify it: it must stay an unmapped refusal with no value.
            if row.get("refusal") != "unmapped_species" or row.get("simulated") is not None:
                raise ValueError(f"invalid unmapped observation: {row['row_id']}")
            continue
        if result["status"] == "ok":
            value = result["mole_fractions"].get(target["mechanism_label"])
            if value != row.get("simulated") or row.get("refusal"):
                raise ValueError("checkpoint and reported simulation differ")
            row["simulated"] = value
        elif result.get("reason") != row.get("refusal"):
            raise ValueError("checkpoint and reported refusal differ")
    rebuilt["metric"] = score_metric(rebuilt["metric_rows"])
    rebuilt["target_chemistry"] = _target_chemistry(contract, rebuilt["metric_rows"])
    rebuilt["target_chemistry_contract_sha256"] = contract_sha256
    rebuilt["rescore_inputs"] = {
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "rows_path": str(report_path.with_name("rows.jsonl")),
        "rows_sha256": hashlib.sha256(checkpoint).hexdigest() if checkpoint is not None else None,
    }
    if form == PRE_LOCK_V0:
        rebuilt.pop("simulation_lock_sha256", None)
        rebuilt.pop("simulation_lock", None)
        rebuilt["rescore_inputs"]["checkpoint_fingerprint_form"] = PRE_LOCK_V0
        rebuilt["rescore_inputs"]["simulation_lock_binding"] = PRE_LOCK_V0_LOCK_BINDING
        run_lock = response.get("simulation_lock_sha256") if response is not None else None
        if isinstance(run_lock, str) and response_bytes is not None:
            rebuilt["rescore_inputs"]["run_recorded_lock"] = {
                "simulation_lock_sha256": run_lock,
                "source_path": str(response_path),
                "source_sha256": hashlib.sha256(response_bytes).hexdigest(),
            }
    return rebuilt


def _pre_lock_v0_inputs(report: dict[str, Any]) -> tuple[tuple[dict[str, str], str] | None, str | None]:
    """Read the run's thread environment and start method from the report's own fields.

    Returns the two inputs, or ``None`` with the name of the field that is missing or ill-typed.
    """
    thread_env = report.get("thread_env")
    if (
        not isinstance(thread_env, dict)
        or not thread_env
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in thread_env.items())
    ):
        return None, "a recorded thread_env"
    start_method = report.get("row_process_start_method")
    if not isinstance(start_method, str) or not start_method.strip():
        return None, "a recorded row_process_start_method"
    return (thread_env, start_method), None


def _lock_sha256(lock: dict[str, Any]) -> str:
    """Hash a simulation lock's canonical bytes, as the worker hashes the lock it records."""
    return hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run_recorded_lock(response: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """Return the run's recorded simulation lock when it binds to its recorded digest.

    Otherwise return ``None`` with the reason the lock cannot be used.
    """
    if response is None:
        return None, "no response.json"
    lock = response.get("simulation_lock")
    if not isinstance(lock, dict):
        return None, "response.json holds no simulation_lock object"
    if response.get("simulation_lock_sha256") != _lock_sha256(lock):
        return None, "simulation_lock does not bind to its simulation_lock_sha256"
    return lock, None


def _read_run_response(
    report_path: Path, legacy_inputs: tuple[dict[str, str], str]
) -> tuple[Path, bytes | None, dict[str, Any] | None]:
    """Read the run directory's ``response.json`` and refuse if it disagrees with the report."""
    path = report_path.with_name("response.json")
    if not path.exists():
        return path, None, None
    raw = path.read_bytes()
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise ValueError("saved response.json is not an object")
    thread_env, start_method = legacy_inputs
    for field, value in (("thread_env", thread_env), ("row_process_start_method", start_method)):
        if field in response and response[field] != value:
            raise ValueError(f"saved report and response.json record different {field}")
    return path, raw, response


def _check_run_response_identity(
    response: dict[str, Any], mechanism_sha256: str, results: dict[str, dict[str, Any]]
) -> None:
    """Refuse a ``response.json`` that does not record this report's mechanism and saved rows."""
    recorded = response.get("mechanism_semantic_sha256")
    if not isinstance(recorded, str):
        raise ValueError("response.json lacks a mechanism_semantic_sha256")
    if recorded != mechanism_sha256:
        raise ValueError("saved report and response.json record different mechanism_semantic_sha256")
    rows = response.get("rows")
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("row_id"), str) for r in rows):
        raise ValueError("response.json holds no rows list keyed by row_id")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["row_id"] in by_id:
            raise ValueError(f"duplicate response.json row ID: {row['row_id']}")
        by_id[row["row_id"]] = row
    for row_id, result in results.items():
        if row_id not in by_id:
            raise ValueError(f"response.json lacks saved row {row_id}")
        if by_id[row_id] != result:
            raise ValueError(f"saved row {row_id} differs between rows.jsonl and response.json")


def jsr_intervals(old: dict[str, Any], new: dict[str, Any], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Paired row bootstrap of the existing species-balanced intersection metric."""
    for name, report in (("v_old", old), ("v_new", new)):
        if (report.get("rescore_inputs") or {}).get("checkpoint_fingerprint_form") == PRE_LOCK_V0:
            raise ValueError(
                f"{name} is a {PRE_LOCK_V0} rescore: pairing needs a simulation lock this report does not carry"
            )
    separation = build_separation_report(old, new)
    common = set(separation["primary"]["row_ids"])
    limits = _metric_limits(old["metric_rows"])
    if limits[0] != _metric_limits(new["metric_rows"])[0]:
        raise ValueError("different frozen detection limits")
    data = []
    metadata: dict[tuple[str, str], tuple[float, float]] = {}
    for report in (old, new):
        errors = {}
        seen = set()
        for row in report["metric_rows"]:
            key = (row["row_id"], row["species"])
            if key in seen:
                raise ValueError(f"duplicate metric observation: {key}")
            seen.add(key)
            if row["row_id"] not in common:
                continue
            metric = score_metric([row], metric_limits=limits)["all"]["per_species"].get(row["species"])
            if metric is None:
                continue
            current = (row["temperature_k"], row["experimental"])
            if key in metadata and metadata[key] != current:
                raise ValueError(f"different frozen JSR observation: {key}")
            metadata[key] = current
            errors[key] = metric["score"]
        data.append(errors)
    paired = {key: (data[0][key], data[1][key]) for key in sorted(data[0].keys() & data[1].keys())}
    return {
        "schema_version": 1,
        "case": "n-pentane-low-t",
        "bootstrap": bootstrap_definition(seed),
        "simulation_lock_sha256": old["simulation_lock_sha256"],
        "holdout_fingerprint": old["holdout_fingerprint"],
        "rescore_inputs": {"v_old": old.get("rescore_inputs"), "v_new": new.get("rescore_inputs")},
        "primary": paired_interval(paired, seed=seed),
        "diagnostics": {
            "600_to_900_k": paired_interval(
                {key: v for key, v in paired.items() if 600 <= metadata[key][0] <= 900}, seed=seed
            ),
            "species_groups": {
                group: paired_interval({key: v for key, v in paired.items() if key[1] in species}, seed=seed)
                for group, species in JSR_SPECIES_GROUPS.items()
            },
        },
        "separation": separation,
    }
