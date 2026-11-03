"""Paired row bootstrap diagnostics for frozen IDT and JSR separation scores."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carmel.benchmarks.cantera_worker import checkpoint_fingerprint, validate_row_refusal
from carmel.benchmarks.holdout import _metric_limits, build_separation_report, score_metric
from carmel.benchmarks.holdout_idt import idt_metric

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261018
JSR_SPECIES_GROUPS = {
    "C0-C2": ("CH4", "C2H4", "C2H6", "CH2O", "CH3CHO", "CO", "CO2"),
    "fuel_and_alkenes": ("NC5H12", "C5H10-1", "C5H10-2", "C3H6"),
    "C5_cyclic_ethers": ("C5H10O1-2", "C5H10O1-3", "C5H10O1-4", "C5H10O2-3", "C5H10O2-4"),
    "O2": ("O2",),
}
PairErrors = dict[tuple[str, str], tuple[float, float]]


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


def rescore_saved_jsr(report_path: Path) -> dict[str, Any]:
    """Rebuild simulated values from the original checkpoint without writing to it."""
    report_bytes = report_path.read_bytes()
    report: dict[str, Any] = json.loads(report_bytes)
    request = json.loads(report_path.with_name("request.json").read_bytes())
    expected = {
        row["row_id"]: checkpoint_fingerprint(
            report["inputs"]["mechanism_semantic_sha256"],
            row,
            report["simulation_lock_sha256"],
            report["plog_validation_sha256"],
        )
        for row in request["rows"]
    }
    if len(expected) != len(request["rows"]):
        raise ValueError("duplicate saved request row IDs")
    checkpoint_path = report_path.with_name("rows.jsonl")
    checkpoint = checkpoint_path.read_bytes() if expected or checkpoint_path.exists() else None
    results: dict[str, dict[str, Any]] = {}
    for line in (checkpoint or b"").splitlines():
        entry = json.loads(line)
        row = entry["result"]
        if row["row_id"] not in expected or entry["fingerprint"] != expected[row["row_id"]]:
            continue
        if row["row_id"] in results and row != results[row["row_id"]]:
            raise ValueError(f"conflicting saved checkpoint row: {row['row_id']}")
        results[row["row_id"]] = row
    reported = {row["row_id"]: row for row in report["row_results"]}
    if len(reported) != len(report["row_results"]):
        raise ValueError("duplicate saved report row IDs")
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
            validate_row_refusal(reason, report.get("simulation_lock", {}))
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
    rebuilt["rescore_inputs"] = {
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "rows_path": str(report_path.with_name("rows.jsonl")),
        "rows_sha256": hashlib.sha256(checkpoint).hexdigest() if checkpoint is not None else None,
    }
    return rebuilt


def jsr_intervals(old: dict[str, Any], new: dict[str, Any], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Paired row bootstrap of the existing species-balanced intersection metric."""
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
