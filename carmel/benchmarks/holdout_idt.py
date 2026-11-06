"""Frozen n-heptane ignition-delay adapter for the shared subprocess harness."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from carmel.benchmarks.cantera_worker import FFCM1_EXCITED_SHA256
from carmel.benchmarks.chemkin import mechanism, normalize_for_ck2yaml
from carmel.benchmarks.contract import read_pinned_file
from carmel.benchmarks.holdout import (
    DEFAULT_ROW_TIMEOUT_S,
    DEFAULT_WORKERS,
    IDTWorkerRequestRow,
    WorkerRequest,
    _base_unit_value,
    _canonical_sha256,
    _source_line_keys,
    _structure_mapping,
    _thermo_duplicates,
    invoke_cantera_worker,
)
from carmel.benchmarks.n_heptane import Contract, git_blob, load_contract, load_split
from carmel.schemas.datasets import Absent, Composition
from carmel.services.artifacts import write_bytes
from carmel.services.chemked import parse_idt_record


def temperature_band(temperature_k: float) -> str:
    """Use the frozen contract's inclusive 750–900 K middle band."""
    return "below_750" if temperature_k < 750 else "750_to_900" if temperature_k <= 900 else "above_900"


def load_idt_rows(contract: Contract, root: Path) -> list[dict[str, Any]]:
    """Verify every pinned split blob and export all exact source definitions."""
    load_split(contract, root)
    rows = []
    for study in contract.split.studies:
        if study.set != "holdout":
            continue
        for item in study.files:
            raw = read_pinned_file(root / item.path)
            if hashlib.sha256(raw).hexdigest() != item.sha256 or git_blob(raw) != item.git_blob:
                raise ValueError(f"SPLIT BLOB CHANGED: {item.path}")
            record = parse_idt_record(raw, item.path, item.sha256)
            for index, point in enumerate(record.envelope.series[0].points):
                composition = (
                    point.composition if isinstance(point.composition, Composition) else record.envelope.composition
                )
                if not isinstance(composition, Composition) or composition.basis != "mole_fraction":
                    raise ValueError(f"unsupported inlet composition: {item.path}:{index}")
                conditions = {v.axis_id: _base_unit_value(v.value) for v in point.coordinates}
                observed = next(v.value for v in point.observations if v.axis_id == "ignition_delay")
                if isinstance(observed, Absent):
                    raise ValueError(f"missing ignition delay: {item.path}:{index}")
                rows.append(
                    {
                        "row_id": f"{item.path}:{index}",
                        "record": item.path,
                        "source_sha256": item.sha256,
                        "source_git_blob": item.git_blob,
                        "group": "RCM-ideal" if index in item.rcm_ideal_points else "shock-tube",
                        "temperature_k": conditions["temperature"],
                        "pressure_pa": conditions["pressure"],
                        "temperature_band": temperature_band(conditions["temperature"]),
                        "experimental_s": _base_unit_value(observed),
                        "ignition_target": record.ignition.target_raw.raw,
                        "ignition_type": record.ignition.criterion_raw.raw,
                        "inlet": {
                            component.species_raw_name: _base_unit_value(component.amount)
                            for component in composition.components
                        },
                    }
                )
    return rows


def idt_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-point mean |log10(tau_sim/tau_exp)| with explicit per-point coverage."""
    scored = []
    refused = []
    for row in rows:
        reason = row.get("reason")
        exp, sim = row.get("experimental_s"), row.get("ignition_delay_s")
        if not reason and not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (exp, sim)):
            reason = "invalid_ignition_delay"
        if reason:
            refused.append({"row_id": row["row_id"], "reason": reason, "ignition_target": row["ignition_target"]})
        else:
            assert isinstance(sim, (float, int)) and isinstance(exp, (float, int))
            scored.append({"row_id": row["row_id"], "error": abs(math.log10(sim / exp))})
    return {
        "overall": fmean(r["error"] for r in scored) if scored else None,
        "scored_count": len(scored),
        "total_count": len(rows),
        "scored": scored,
        "refusals": refused,
    }


def idt_coverage_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow at most one third non-solver shock refusals; retain the RCM coverage gate."""
    shock = [r for r in rows if r["group"] == "shock-tube"]
    rcm = [r for r in rows if r["group"] == "RCM-ideal"]
    rcm_count = idt_metric(rcm)["scored_count"]
    unexpected = [
        r["row_id"]
        for r in idt_metric(shock)["refusals"]
        if not r["reason"].startswith("solver_failure") and r["reason"] not in ("solver_crash", "max_steps_exceeded")
    ]
    return {
        "passed": len(rows) == 189
        and len(shock) == 174
        and len(rcm) == 15
        and len(unexpected) <= 58
        and rcm_count >= 12,
        "unexpected_shock_rows": unexpected,
        "rcm_scored": rcm_count,
        "minimum_rcm_scored": 12,
        "maximum_non_solver_shock_refusals": 58,
    }


def target_idt_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The predeclared inclusive 600–1000 K window, with equal point weights."""
    selected = [r for r in rows if 600 <= r["temperature_k"] <= 1000]
    return {
        "temperature_min_k": 600,
        "temperature_max_k": 1000,
        **idt_metric(selected),
        "groups": {
            group: idt_metric([r for r in selected if r["group"] == group]) for group in ("shock-tube", "RCM-ideal")
        },
    }


def run_idt_harness(
    *,
    contract_path: Path,
    mechanism_path: Path,
    mechanism_key: Literal["v_old", "v_new"],
    thermo_path: Path,
    thermo_url: str,
    thermo_retrieval_date: str,
    thermo_sha256: str,
    split_root: Path,
    run_dir: Path,
    cantera_python: Path,
    row_timeout_s: float = DEFAULT_ROW_TIMEOUT_S,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Score the pinned case-1 holdout using the existing worker/lock/checkpoint boundary."""
    contract_bytes = read_pinned_file(contract_path)
    contract = load_contract(contract_path)
    raw_mechanism, raw_thermo = read_pinned_file(mechanism_path), read_pinned_file(thermo_path)
    contract.mechanisms[mechanism_key].verify(raw_mechanism)
    if hashlib.sha256(raw_thermo).hexdigest() != thermo_sha256:
        raise ValueError("PIN CHANGED: thermo bytes differ from supplied pin")
    if mechanism_key == "v_old":
        contract.thermo_source.verify(raw_thermo)
    rows = load_idt_rows(contract, split_root)
    counts = dict(Counter(row["group"] for row in rows))
    if len(rows) != 189 or counts != {"shock-tube": 174, "RCM-ideal": 15}:
        raise ValueError(f"holdout export gate: expected 189 = 174 + 15; got {len(rows)} {counts}")
    kinetics = normalize_for_ck2yaml(raw_mechanism, file=mechanism_path.name)
    thermo = normalize_for_ck2yaml(raw_thermo, file=thermo_path.name)
    labels, reactions = mechanism(kinetics.data)
    # The existing structural mapper resolves both source and mechanism labels by structure.
    inlet_structures = {"nC7H16": "CCCCCCC", "NC7H16": "CCCCCCC", "AR": "[Ar]", "Ar": "[Ar]"}
    mapping, unmapped = _structure_mapping(
        {label for row in rows for label in row["inlet"]},
        labels,
        observed_structures=inlet_structures,
        contract_structures=inlet_structures,
    )
    if unmapped:
        raise ValueError(f"unmapped inlet species: {unmapped}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mechanism_input, thermo_input = run_dir / "mechanism.normalized.inp", run_dir / "thermo.normalized.dat"
    write_bytes(mechanism_input, kinetics.data)
    write_bytes(thermo_input, thermo.data)
    requests = tuple(
        IDTWorkerRequestRow(
            row_id=row["row_id"],
            temperature_k=row["temperature_k"],
            pressure_pa=row["pressure_pa"],
            composition={mapping[label]: amount for label, amount in row["inlet"].items()},
            ignition_target=row["ignition_target"],
            ignition_type=row["ignition_type"],
        )
        for row in rows
    )
    request = WorkerRequest(
        observable="idt",
        excited_state_submechanism_sha256=FFCM1_EXCITED_SHA256,
        mechanism_path=mechanism_input,
        thermo_path=thermo_input,
        converted_yaml_path=run_dir / "converted.yaml",
        trimmed_yaml_path=run_dir / "trimmed.yaml",
        checkpoint_path=run_dir / "rows.jsonl",
        reaction_source_lines=_source_line_keys(kinetics.data, reactions),
        rows=requests,
        row_timeout_s=row_timeout_s,
        workers=workers,
    )
    response = invoke_cantera_worker(request, run_dir=run_dir, cantera_python=cantera_python)
    if response.status != "ok" or response.observable != "idt":
        detail = response.validation_error or response.construction_error or response.observable
        raise ValueError(f"Cantera subprocess refused: {response.refusal}: {detail}")
    if (
        response.excited_state_submechanism is None
        or response.excited_state_submechanism.get("sha256") != FFCM1_EXCITED_SHA256
    ):
        raise ValueError("worker omitted or changed the pinned excited-state submechanism")
    simulated = {item.row_id: item for item in response.rows}
    if len(simulated) != len(response.rows) or set(simulated) != {r["row_id"] for r in rows}:
        raise ValueError("worker response row IDs differ from the frozen request")
    results = []
    for row in rows:
        item = simulated[row["row_id"]]
        if item.ignition_target is not None and item.ignition_target != row["ignition_target"]:
            raise ValueError(f"worker changed ignition target: {item.row_id}")
        reason = item.reason if item.status == "refused" else None
        if item.status == "refused" and not reason:
            reason = "unspecified_worker_refusal"
        results.append(
            {
                **row,
                "status": item.status,
                "reason": reason,
                "ignition_delay_s": item.ignition_delay_s,
                "wall_time_s": item.wall_time_s,
                "integrator_steps": item.integrator_steps,
                "signal": item.signal,
                "fault_log": item.fault_log,
                "progress": item.progress,
            }
        )
    return {
        "schema_version": 1,
        "case": contract.case,
        "observable": "idt",
        "mechanism_key": mechanism_key,
        "holdout_fingerprint": {
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "sha256": _canonical_sha256({"contract": hashlib.sha256(contract_bytes).hexdigest(), "rows": rows}),
        },
        "export_counts": {"total": len(rows), **counts},
        "inputs": {
            "mechanism": {
                "path": str(mechanism_path),
                "raw_sha256": kinetics.raw_sha256,
                "normalized_sha256": kinetics.normalized_sha256,
                "rewrites": [r.model_dump() for r in kinetics.rewrites],
            },
            "thermo": {
                "path": str(thermo_path),
                "url": thermo_url,
                "retrieval_date": thermo_retrieval_date,
                "raw_sha256": thermo.raw_sha256,
                "normalized_sha256": thermo.normalized_sha256,
                "rewrites": [r.model_dump() for r in thermo.rewrites],
            },
            "trimmed_yaml_sha256": response.trimmed_yaml_sha256,
            "mechanism_semantic_sha256": response.mechanism_semantic_sha256,
        },
        "species_mapping": mapping,
        "thermo_duplicates": _thermo_duplicates(raw_thermo),
        "row_results": results,
        "metric": idt_metric(results),
        "target_chemistry": target_idt_metric(results),
        "diagnostics": {
            "groups": {
                group: idt_metric([r for r in results if r["group"] == group])
                for group in contract.scoring.secondary_groups
            },
            "temperature_bands": {
                band: idt_metric([r for r in results if r["temperature_band"] == band])
                for band in ("below_750", "750_to_900", "above_900")
            },
        },
        "coverage_gate": idt_coverage_gate(results),
        **{
            key: value
            for key, value in response.model_dump(mode="json").items()
            if key not in ("rows", "schema_version", "observable", "status")
        },
    }
