"""Cantera subprocess worker and pre-validation mechanism transformations.

This module imports Cantera only inside :func:`main`, so Carmel and its CI remain
Cantera-free. Pure transformation helpers are tested in Carmel's normal environment.
"""

from __future__ import annotations

import copy
import faulthandler
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

PLOG_VALIDATION_TEMPERATURES_K = (300.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
_R_CAL = 1.98720425864083
_PRESSURE_UNITS = {"pa": 1.0, "bar": 1.0e5, "atm": 101325.0}
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ACTIVATION_ENERGY_TO_CAL_PER_MOL = {
    "cal/mol": 1.0,
    "kcal/mol": 1.0e3,
    "j/mol": 1.0 / 4.184,
    "kj/mol": 1.0e3 / 4.184,
    "cal/kmol": 1.0e-3,
    "kcal/kmol": 1.0,
    "j/kmol": 1.0 / 4184.0,
    "kj/kmol": 1.0 / 4.184,
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ROW_PROCESS_START_METHOD = "spawn"
JSR_RTOL = 1.0e-9
JSR_ATOL = 1.0e-15
JSR_MAX_STEPS = 100000
PRESSURE_CONTROLLER_K = 1.0e-5
Simulation = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
_ROW_PROGRESS_PATH: Path | None = None
_OBSERVABLE_SIMULATORS: dict[str, Simulation] = {}


def write_text(path: Path, content: str) -> None:
    """Atomically and durably write UTF-8 text without importing Carmel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def simulation_lock(row_timeout_s: float, cantera_version: str | None) -> dict[str, Any]:
    """Return the complete locked simulation and conversion regime."""
    return {
        "schema_version": 1,
        "simulator": {"cantera_version": cantera_version},
        "jsr": {
            "rtol": JSR_RTOL,
            "atol": JSR_ATOL,
            "max_steps": JSR_MAX_STEPS,
            "steady_state_timeout_s": float(row_timeout_s),
            "initial_state": "row_inlet",
            "temperature_continuation": False,
            "energy": "off",
            "pressure_controller_K": PRESSURE_CONTROLLER_K,
        },
        "process": {
            "thread_env": THREAD_ENV,
            "row_start_method": ROW_PROCESS_START_METHOD,
        },
        "ck2yaml": {
            "flags": ["--permissive", "--verbose", "--no-validate"],
            "eless_exponent_normalization": "whole_numeric_tokens_outside_bang_comments_v1",
            "warning_allowlist": [
                "redundant_species_declaration",
                "redundant_thermo_entry_first_wins",
                "comment_only_unparsable_thermo_block",
            ],
            "plog_rule": "trim_nonpositive_levels_only_outside_scored_interpolation_intervals_v1",
            "plog_validation_temperatures_k": list(PLOG_VALIDATION_TEMPERATURES_K),
        },
    }


def simulation_lock_sha256(row_timeout_s: float, cantera_version: str | None) -> str:
    """Hash the canonical bytes of the locked simulation regime."""
    canonical = json.dumps(
        simulation_lock(row_timeout_s, cantera_version), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def mechanism_semantic_sha256(mechanism: dict[str, Any]) -> str:
    """Hash simulation content while excluding ck2yaml's volatile generation date."""
    semantic = copy.deepcopy(mechanism)
    semantic.pop("date", None)
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def plog_validation_sha256(trims: list[dict[str, Any]], pressure_refusals: list[dict[str, Any]]) -> str:
    """Hash the canonical pre-trim PLOG validation result used for row safety."""

    def canonical_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))

    canonical = json.dumps(
        {
            "trims": canonical_items(trims),
            "pressure_refusals": canonical_items(pressure_refusals),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class RunLevelSimulationError(RuntimeError):
    """A child exposed an adapter defect rather than a condition-level failure."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("construction_error") or result.get("refusal")))
        self.result = result


class PlogValidationError(ValueError):
    """A PLOG rate cannot be validated without guessing its units."""

    reason = "unsupported_plog_activation_energy_unit"


def _write_progress(
    phase: str,
    *,
    last_reactor_time_s: float | None = None,
    temperature_k: float | None = None,
    integrator_steps: int | None = None,
) -> None:
    """Persist the latest row phase so a fatal signal can be classified honestly."""
    if _ROW_PROGRESS_PATH is None:
        return
    payload = {
        "phase": phase,
        "last_reactor_time_s": last_reactor_time_s,
        "temperature_k": temperature_k,
        "integrator_steps": integrator_steps,
    }
    write_text(_ROW_PROGRESS_PATH, json.dumps(payload, sort_keys=True))


def cantera_validation_refusals(messages: list[str]) -> list[str]:
    """Allow only the two reviewed NASA discontinuity warnings."""
    return [
        message
        for message in messages
        if not ("NasaPoly2::validate" in message and ("For species OH*," in message or "For species CH*," in message))
    ]


def parse_ck2yaml_log(log: str) -> dict[str, list[Any]]:
    """Classify verbose ck2yaml output and fail closed on unknown warnings."""
    events: list[dict[str, Any]] = []
    refusals: list[str] = []
    lines = log.splitlines()
    index = 0
    ordinary = (
        "Wrote YAML mechanism file to ",
        "Mechanism contains ",
        "Skipping unexpected species ",
    )
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("Ignoring redundant declaration for species "):
            events.append({"kind": "redundant_species", "message": line})
        elif line.startswith("Ignoring redundant thermo data for species "):
            events.append({"kind": "redundant_thermo", "message": line})
        elif line.startswith("Unparsable lines while reading thermo data in "):
            start = line
            index += 1
            if index >= len(lines) or lines[index] != '"""':
                refusals.append(f"malformed ck2yaml unparsable block: {start}")
                continue
            index += 1
            content: list[str] = []
            while index < len(lines) and lines[index] != '"""':
                content.append(lines[index])
                index += 1
            bad = [item for item in content if item.strip(" !\t")]
            if bad:
                refusals.append(f"unallowlisted unparsable thermo content: {bad[0]}")
            else:
                events.append({"kind": "comment_only_unparsable", "message": start, "content": content})
            if index + 1 < len(lines) and lines[index + 1] == "Lines could not be parsed as a NASA7 entry.":
                index += 1
        elif line.startswith(ordinary):
            events.append({"kind": "information", "message": line})
        else:
            refusals.append(f"unallowlisted ck2yaml output: {line}")
        index += 1
    return {"events": events, "refusals": refusals}


def _pressure_pa(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(rf"\s*({_NUMBER_PATTERN})\s*([A-Za-z]+)\s*", str(value))
    if match is None or match[2].lower() not in _PRESSURE_UNITS:
        raise ValueError(f"unsupported PLOG pressure {value!r}")
    return float(match[1]) * _PRESSURE_UNITS[match[2].lower()]


def _number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    match = re.match(rf"\s*({_NUMBER_PATTERN})", str(value))
    if match is None:
        raise ValueError(f"unsupported numeric value {value!r}")
    return float(match[1])


def _activation_energy_cal_per_mol(value: object, inherited_unit: str) -> float:
    if isinstance(value, int | float):
        number = float(value)
        unit = inherited_unit
    else:
        match = re.fullmatch(rf"\s*({_NUMBER_PATTERN})(?:\s*([A-Za-z]+\s*/\s*[A-Za-z]+))?\s*", str(value))
        if match is None:
            raise PlogValidationError(f"unsupported PLOG activation energy {value!r}")
        number = float(match[1])
        unit = match[2] or inherited_unit
    normalized_unit = unit.replace(" ", "").lower()
    try:
        factor = _ACTIVATION_ENERGY_TO_CAL_PER_MOL[normalized_unit]
    except KeyError as exc:
        raise PlogValidationError(f"unsupported PLOG activation-energy unit {unit!r}") from exc
    return number * factor


def _arrhenius(expression: dict[str, object], temperature: float, activation_energy_unit: str = "cal/mol") -> float:
    a = _number(expression["A"])
    b = _number(expression["b"])
    ea = _activation_energy_cal_per_mol(expression["Ea"], activation_energy_unit)
    try:
        return float(a * temperature**b * math.exp(-ea / (_R_CAL * temperature)))
    except OverflowError:
        return math.copysign(math.inf, a)


def _affected(pressure: float, lower: float | None, upper: float | None) -> bool:
    if lower is None:
        assert upper is not None
        return pressure < upper
    if upper is None:
        return pressure > lower
    return lower < pressure < upper


def trim_unsound_plog_levels(
    mechanism: dict[str, Any], scored_pressures_pa: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Trim invalid PLOG levels only outside all scored interpolation intervals."""
    output = copy.deepcopy(mechanism)
    activation_energy_unit = str(output.get("units", {}).get("activation-energy", "cal/mol"))
    trims: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for reaction in output.get("reactions", []):
        if reaction.get("type") != "pressure-dependent-Arrhenius":
            continue
        rates = reaction["rate-constants"]
        grouped: dict[float, list[dict[str, object]]] = {}
        for expression in rates:
            grouped.setdefault(_pressure_pa(expression["P"]), []).append(expression)
        failing: dict[float, list[float]] = {}
        for pressure, expressions in grouped.items():
            bad = [
                temperature
                for temperature in PLOG_VALIDATION_TEMPERATURES_K
                if not (sum(_arrhenius(item, temperature, activation_energy_unit) for item in expressions) > 0.0)
            ]
            if bad:
                failing[pressure] = bad
        if not failing:
            continue
        if len(grouped) == len(failing):
            raise ValueError(f"refusing to remove whole PLOG reaction {reaction['equation']}")
        kept = sorted(set(grouped) - set(failing))
        for pressure in sorted(failing):
            lower = max((item for item in kept if item < pressure), default=None)
            upper = min((item for item in kept if item > pressure), default=None)
            affected = [item for item in scored_pressures_pa if _affected(item, lower, upper)]
            for item in affected:
                refusals.append(
                    {
                        "pressure_pa": item,
                        "reaction": reaction["equation"],
                        "reason": "mechanism_rate_unsound_at_condition",
                        "removed_pressure_pa": pressure,
                    }
                )
            trims.append(
                {
                    "reaction": reaction["equation"],
                    "source_line": reaction.get("source_line"),
                    "removed_pressure_pa": pressure,
                    "expressions": copy.deepcopy(grouped[pressure]),
                    "kept_bracket_pa": [lower, upper],
                    "failing_temperatures_k": failing[pressure],
                }
            )
        reaction["rate-constants"] = [item for item in rates if _pressure_pa(item["P"]) not in failing]
    return output, trims, refusals


_ARROW = re.compile(r"(<=>|=>|=)")
_COLLIDER = re.compile(r"\(\+([^()]+)\)")
_COEFFICIENT = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(.+)")


def _canonical_side(side: str) -> str:
    collider = ""
    found = _COLLIDER.search(side)
    if found is not None:
        collider = f" (+{found[1]})"
        side = side[: found.start()] + side[found.end() :]
    counts: dict[str, float] = {}
    third_body = False
    for token in side.split("+"):
        if token == "M":
            third_body = True
            continue
        if not token:
            continue
        coefficient = _COEFFICIENT.fullmatch(token)
        amount, species = (float(coefficient[1]), coefficient[2]) if coefficient else (1.0, token)
        counts[species] = counts.get(species, 0.0) + amount
    terms = [species if counts[species] == 1 else f"{counts[species]:g} {species}" for species in sorted(counts)]
    if third_body:
        terms.append("M")
    return " + ".join(terms) + collider


def canonical_reaction_key(equation: str) -> str:
    """Return a spacing-, order- and coefficient-style-invariant key for a reaction equation.

    Used by both the CHEMKIN source side and the converted-YAML side so keys agree by construction.
    """
    compact = re.sub(r"\s+", "", equation)
    parts = _ARROW.split(compact, maxsplit=1)
    if len(parts) != 3:
        return compact
    left, arrow, right = parts
    direction = "=>" if arrow == "=>" else "<=>"
    return f"{_canonical_side(left)} {direction} {_canonical_side(right)}"


def _annotate_reaction_source_lines(mechanism: dict[str, Any], source_lines: dict[str, list[int]]) -> None:
    """Attach occurrence-specific source lines without guessing on count mismatches."""
    reactions = mechanism.get("reactions", [])
    counts: dict[str, int] = {}
    for reaction in reactions:
        key = canonical_reaction_key(str(reaction["equation"]))
        counts[key] = counts.get(key, 0) + 1
    seen: dict[str, int] = {}
    for reaction in reactions:
        key = canonical_reaction_key(str(reaction["equation"]))
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        candidates = source_lines.get(key, [])
        if len(candidates) == counts[key]:
            reaction["source_line"] = candidates[occurrence]
        else:
            reaction["source_line"] = {
                "status": "ambiguous",
                "candidate_lines": candidates,
            }


def _simulate_row(trimmed_path: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build and solve one independent JSR condition in an isolated process."""
    started = time.monotonic()
    _write_progress(
        "construction",
        temperature_k=float(row["temperature_k"]),
        integrator_steps=0,
    )
    import cantera as ct  # type: ignore[import-not-found]

    try:
        gas = ct.Solution(trimmed_path)
        gas.TPX = float(row["temperature_k"]), float(row["pressure_pa"]), row["composition"]
        inlet = ct.Reservoir(gas, clone=True)
        reactor = ct.IdealGasReactor(
            gas, clone=True, energy="off", volume=float(row["volume_m3"]), name=str(row["row_id"])
        )
        outlet = ct.Reservoir(gas, clone=True)
        mass_flow = ct.MassFlowController(inlet, reactor, mdot=reactor.mass / float(row["residence_time_s"]))
        _pressure = ct.PressureController(reactor, outlet, primary=mass_flow, K=PRESSURE_CONTROLLER_K)
        network = ct.ReactorNet([reactor])
        network.rtol = JSR_RTOL
        network.atol = JSR_ATOL
        network.max_steps = JSR_MAX_STEPS
    except Exception as exc:
        _write_progress(
            "construction_failed",
            temperature_k=float(row["temperature_k"]),
            integrator_steps=0,
        )
        return "run", {
            "refusal": "reactor_construction_failed",
            "construction_error": str(exc),
            "wall_time_s": time.monotonic() - started,
        }
    try:
        _write_progress(
            "integration",
            last_reactor_time_s=float(network.time),
            temperature_k=float(reactor.T),
            integrator_steps=int(network.solver_stats.get("steps", 0)),
        )
        network.advance_to_steady_state(max_steps=JSR_MAX_STEPS)
        steps = int(network.solver_stats.get("steps", 0))
        _write_progress(
            "complete",
            last_reactor_time_s=float(network.time),
            temperature_k=float(reactor.T),
            integrator_steps=steps,
        )
        return "row", {
            "row_id": row["row_id"],
            "status": "ok",
            "mole_fractions": {
                species: float(reactor.phase.X[reactor.phase.species_index(species)])
                for species in row["output_species"]
            },
            "wall_time_s": time.monotonic() - started,
            "integrator_steps": steps,
        }
    except Exception as exc:
        steps = int(network.solver_stats.get("steps", 0))
        _write_progress(
            "solver_failure",
            last_reactor_time_s=float(network.time),
            temperature_k=float(reactor.T),
            integrator_steps=steps,
        )
        return "row", {
            "row_id": row["row_id"],
            "status": "refused",
            "reason": f"solver_failure: {exc}",
            "wall_time_s": time.monotonic() - started,
            "integrator_steps": steps,
        }


def _row_fingerprint(
    mechanism_sha256: str,
    row: dict[str, Any],
    row_timeout_s: float,
    cantera_version: str,
    plog_validation_identity: str = "",
) -> str:
    payload = json.dumps(
        {
            "mechanism_semantic_sha256": mechanism_sha256,
            "plog_validation_sha256": plog_validation_identity,
            "simulation_lock_sha256": simulation_lock_sha256(row_timeout_s, cantera_version),
            "row": row,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    mechanism_sha256: str,
    row_timeout_s: float,
    cantera_version: str,
    plog_validation_identity: str = "",
    checkpoint_recoveries: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    expected = {
        str(row["row_id"]): _row_fingerprint(
            mechanism_sha256,
            row,
            row_timeout_s,
            cantera_version,
            plog_validation_identity,
        )
        for row in rows
    }
    results: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return results
    raw_checkpoint = checkpoint_path.read_bytes()
    lines = raw_checkpoint.splitlines(keepends=True)
    valid_bytes = 0
    for line_number, raw_line in enumerate(lines, start=1):
        terminated = raw_line.endswith(b"\n")
        line_bytes = raw_line.rstrip(b"\r\n")
        if not line_bytes.strip():
            valid_bytes += len(raw_line)
            continue
        try:
            line = line_bytes.decode("utf-8")
            entry = json.loads(line)
            result = entry["result"]
            row_id = str(result["row_id"])
            fingerprint = str(entry["fingerprint"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(lines) and not terminated:
                with checkpoint_path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                if checkpoint_recoveries is not None:
                    checkpoint_recoveries.append(
                        {
                            "kind": "recovered_truncated_checkpoint_tail",
                            "byte_length": len(raw_line),
                        }
                    )
                break
            raise ValueError(f"malformed row checkpoint at line {line_number}") from exc
        valid_bytes += len(raw_line)
        if expected.get(row_id) == fingerprint:
            if row_id in results and results[row_id] != result:
                raise ValueError(f"conflicting row checkpoint entries for {row_id}")
            results[row_id] = result
    return results


def _append_checkpoint(
    checkpoint_path: Path,
    mechanism_sha256: str,
    row: dict[str, Any],
    result: dict[str, Any],
    row_timeout_s: float,
    cantera_version: str,
    plog_validation_identity: str = "",
) -> None:
    entry = {
        "fingerprint": _row_fingerprint(
            mechanism_sha256,
            row,
            row_timeout_s,
            cantera_version,
            plog_validation_identity,
        ),
        "result": result,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _row_child(
    simulation: Simulation,
    trimmed_path: str,
    row: dict[str, Any],
    result_path: str,
) -> None:
    global _ROW_PROGRESS_PATH
    output_path = Path(result_path)
    _ROW_PROGRESS_PATH = output_path.with_suffix(".progress.json")
    fault_path = output_path.with_suffix(".fault.log")
    with fault_path.open("w") as fault_stream:
        faulthandler.enable(file=fault_stream, all_threads=True)
        try:
            scope, result = simulation(trimmed_path, row)
            payload = {"scope": scope, "result": result}
        except Exception as exc:
            payload = {
                "scope": "run",
                "result": {"refusal": "simulation_worker_failed", "construction_error": repr(exc)},
            }
        finally:
            faulthandler.disable()
    write_text(output_path, json.dumps(payload, allow_nan=False))


def _stop_processes(active: dict[Any, dict[str, Any]]) -> None:
    for process in active:
        if process.is_alive():
            process.terminate()
    for process in active:
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()


def execute_rows(
    rows: list[dict[str, Any]],
    *,
    trimmed_path: str,
    mechanism_sha256: str,
    checkpoint_path: Path,
    workers: int,
    row_timeout_s: float,
    cantera_version: str,
    plog_validation_sha256: str = "",
    simulation: Simulation = _simulate_row,
    poll_interval_s: float = 0.05,
    initial_results: dict[str, dict[str, Any]] | None = None,
    checkpoint_recoveries: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Run rows in individually killable processes and durably checkpoint each result."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if row_timeout_s <= 0:
        raise ValueError("row timeout must be positive")
    checkpointed = _load_checkpoint(
        checkpoint_path,
        rows,
        mechanism_sha256,
        row_timeout_s,
        cantera_version,
        plog_validation_sha256,
        checkpoint_recoveries,
    )
    resumed = len(checkpointed)
    by_id = {str(row["row_id"]): row for row in rows}
    results = dict(checkpointed)
    for row_id, result in (initial_results or {}).items():
        if row_id in results:
            continue
        result.setdefault("wall_time_s", 0.0)
        result.setdefault("integrator_steps", 0)
        _append_checkpoint(
            checkpoint_path,
            mechanism_sha256,
            by_id[row_id],
            result,
            row_timeout_s,
            cantera_version,
            plog_validation_sha256,
        )
        results[row_id] = result

    pending = [row for row in rows if str(row["row_id"]) not in results]
    context: Any = get_context(ROW_PROCESS_START_METHOD)
    result_dir = checkpoint_path.parent / ".row-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    active: dict[Any, dict[str, Any]] = {}
    try:
        while pending or active:
            while pending and len(active) < workers:
                row = pending.pop(0)
                fingerprint = _row_fingerprint(
                    mechanism_sha256,
                    row,
                    row_timeout_s,
                    cantera_version,
                    plog_validation_sha256,
                )
                result_path = result_dir / f"{fingerprint}.json"
                progress_path = result_path.with_suffix(".progress.json")
                fault_path = result_path.with_suffix(".fault.log")
                for stale_path in (result_path, progress_path, fault_path):
                    stale_path.unlink(missing_ok=True)
                process = context.Process(
                    target=_row_child,
                    args=(simulation, trimmed_path, row, str(result_path)),
                )
                process.start()
                active[process] = {
                    "row": row,
                    "started": time.monotonic(),
                    "result_path": result_path,
                    "progress_path": progress_path,
                    "fault_path": fault_path,
                }

            progressed = False
            expired = {
                process: time.monotonic() - float(task["started"])
                for process, task in active.items()
                if process.is_alive() and time.monotonic() - float(task["started"]) >= row_timeout_s
            }
            for process in expired:
                process.terminate()
            for process, task in list(active.items()):
                elapsed = time.monotonic() - float(task["started"])
                row = task["row"]
                result_path = task["result_path"]
                progress_path = task["progress_path"]
                fault_path = task["fault_path"]
                if process in expired:
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    del active[process]
                    progressed = True
                    for stale_path in (result_path, progress_path, fault_path):
                        stale_path.unlink(missing_ok=True)
                    result = {
                        "row_id": row["row_id"],
                        "status": "refused",
                        "reason": "steady_state_timeout",
                        "wall_time_s": expired[process],
                        "integrator_steps": None,
                    }
                    _append_checkpoint(
                        checkpoint_path,
                        mechanism_sha256,
                        row,
                        result,
                        row_timeout_s,
                        cantera_version,
                        plog_validation_sha256,
                    )
                    results[str(row["row_id"])] = result
                elif not process.is_alive():
                    process.join()
                    del active[process]
                    progressed = True
                    if process.exitcode != 0 or not result_path.is_file():
                        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else None
                        if (
                            process.exitcode is not None
                            and process.exitcode < 0
                            and progress is not None
                            and progress.get("phase") == "integration"
                        ):
                            result = {
                                "row_id": row["row_id"],
                                "status": "refused",
                                "reason": "solver_crash",
                                "signal": -process.exitcode,
                                "wall_time_s": elapsed,
                                "integrator_steps": progress.get("integrator_steps"),
                                "progress": progress,
                                "fault_log": str(fault_path),
                            }
                            _append_checkpoint(
                                checkpoint_path,
                                mechanism_sha256,
                                row,
                                result,
                                row_timeout_s,
                                cantera_version,
                                plog_validation_sha256,
                            )
                            results[str(row["row_id"])] = result
                            result_path.unlink(missing_ok=True)
                            continue
                        raise RunLevelSimulationError(
                            {
                                "refusal": "reactor_construction_failed",
                                "construction_error": (
                                    f"row {row['row_id']} crashed during construction with exit "
                                    f"{process.exitcode}; progress={progress!r}"
                                ),
                                "signal": -process.exitcode
                                if process.exitcode is not None and process.exitcode < 0
                                else None,
                                "fault_log": str(fault_path) if fault_path.is_file() else None,
                            }
                        )
                    payload = json.loads(result_path.read_text())
                    result_path.unlink(missing_ok=True)
                    progress_path.unlink(missing_ok=True)
                    if not fault_path.is_file() or not fault_path.read_text():
                        fault_path.unlink(missing_ok=True)
                    if payload.get("scope") != "row":
                        raise RunLevelSimulationError(payload["result"])
                    result = payload["result"]
                    result.setdefault("wall_time_s", elapsed)
                    result.setdefault("integrator_steps", None)
                    _append_checkpoint(
                        checkpoint_path,
                        mechanism_sha256,
                        row,
                        result,
                        row_timeout_s,
                        cantera_version,
                        plog_validation_sha256,
                    )
                    results[str(row["row_id"])] = result
            if active and not progressed:
                time.sleep(poll_interval_s)
    except BaseException:
        _stop_processes(active)
        raise
    return results, resumed


_OBSERVABLE_SIMULATORS["jsr"] = _simulate_row


def _write_response(path: Path, response: dict[str, Any]) -> None:
    write_text(path, json.dumps(response, indent=2) + "\n")


def main() -> int:
    """Convert, validate, and simulate one typed JSON request."""
    if len(sys.argv) != 3:
        print("usage: cantera_worker.py REQUEST.json RESPONSE.json", file=sys.stderr)
        return 2
    request_path, response_path = map(Path, sys.argv[1:])
    os.environ.update(THREAD_ENV)
    request = json.loads(request_path.read_text())
    observable = str(request.get("observable", "jsr"))
    simulation = _OBSERVABLE_SIMULATORS.get(observable)
    if simulation is None:
        cantera_version = None
    else:
        import cantera as ct

        cantera_version = str(ct.__version__)
    locked = simulation_lock(float(request.get("row_timeout_s", 600.0)), cantera_version)
    response: dict[str, Any] = {
        "schema_version": 1,
        "status": "refused",
        "rows": [],
        "converter": {},
        "thread_env": {key: os.environ[key] for key in THREAD_ENV},
        "row_process_start_method": ROW_PROCESS_START_METHOD,
        "load_average_at_start": list(os.getloadavg()),
        "simulation_lock": locked,
        "simulation_lock_sha256": simulation_lock_sha256(float(request.get("row_timeout_s", 600.0)), cantera_version),
        "cantera_version": cantera_version,
        "observable": observable,
    }
    if simulation is None:
        response["refusal"] = f"unsupported_observable: {observable}"
        _write_response(response_path, response)
        return 0
    assert cantera_version is not None
    command = [
        sys.executable,
        "-m",
        "cantera.ck2yaml",
        "--permissive",
        "--verbose",
        "--no-validate",
        "--input",
        request["mechanism_path"],
        "--thermo",
        request["thermo_path"],
        "--output",
        request["converted_yaml_path"],
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    converter_log = completed.stdout + completed.stderr
    response["converter"] = {"command": command, "returncode": completed.returncode, "log": converter_log}
    parsed_log = parse_ck2yaml_log(converter_log)
    response["converter"].update(parsed_log)
    if completed.returncode or parsed_log["refusals"]:
        response["refusal"] = "ck2yaml_conversion_refused"
        _write_response(response_path, response)
        return 0

    import yaml

    mechanism = yaml.safe_load(Path(request["converted_yaml_path"]).read_text())
    source_lines = request.get("reaction_source_lines", {})
    _annotate_reaction_source_lines(mechanism, source_lines)
    try:
        trimmed, trims, pressure_refusals = trim_unsound_plog_levels(
            mechanism, [float(row["pressure_pa"]) for row in request["rows"]]
        )
    except PlogValidationError as exc:
        response["refusal"] = exc.reason
        response["validation_error"] = str(exc)
        _write_response(response_path, response)
        return 0
    plog_validation_identity = plog_validation_sha256(trims, pressure_refusals)
    for reaction in trimmed.get("reactions", []):
        reaction.pop("source_line", None)
    trimmed_path = Path(request["trimmed_yaml_path"])
    write_text(trimmed_path, yaml.safe_dump(trimmed, sort_keys=False))
    response["plog_trims"] = trims
    response["pressure_refusals"] = pressure_refusals
    response["plog_validation_sha256"] = plog_validation_identity
    response["trimmed_yaml_sha256"] = hashlib.sha256(trimmed_path.read_bytes()).hexdigest()
    response["mechanism_semantic_sha256"] = mechanism_semantic_sha256(trimmed)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ct.Solution(trimmed_path)
    except Exception as exc:
        response["refusal"] = "trimmed_mechanism_validation_failed"
        response["validation_error"] = str(exc)
        _write_response(response_path, response)
        return 0
    response["validation_warnings"] = [str(item.message) for item in caught]
    unexpected = cantera_validation_refusals(response["validation_warnings"])
    if unexpected:
        response["refusal"] = "unallowlisted_cantera_validation_warning"
        _write_response(response_path, response)
        return 0

    affected_pressures = {item["pressure_pa"] for item in pressure_refusals}
    initial_results: dict[str, dict[str, Any]] = {}
    for row in request["rows"]:
        if float(row["pressure_pa"]) in affected_pressures:
            initial_results[row["row_id"]] = {
                "row_id": row["row_id"],
                "status": "refused",
                "reason": "mechanism_rate_unsound_at_condition",
            }
    requested_workers = int(request.get("workers", max(1, (os.cpu_count() or 1) - 2)))
    response["configured_workers"] = requested_workers
    response["row_workers"] = min(requested_workers, len(request["rows"])) if request["rows"] else 0
    response["row_timeout_s"] = float(request.get("row_timeout_s", 600.0))
    checkpoint_path = Path(request.get("checkpoint_path", response_path.parent / "rows.jsonl"))
    response["checkpoint_path"] = str(checkpoint_path)
    checkpoint_recoveries: list[dict[str, Any]] = []
    response["checkpoint_recoveries"] = checkpoint_recoveries
    try:
        results, resumed = execute_rows(
            request["rows"],
            trimmed_path=str(trimmed_path),
            mechanism_sha256=response["mechanism_semantic_sha256"],
            checkpoint_path=checkpoint_path,
            workers=requested_workers,
            row_timeout_s=response["row_timeout_s"],
            cantera_version=cantera_version,
            plog_validation_sha256=plog_validation_identity,
            simulation=simulation,
            initial_results=initial_results,
            checkpoint_recoveries=checkpoint_recoveries,
        )
    except RunLevelSimulationError as exc:
        response.update(exc.result)
        response["rows"] = []
        _write_response(response_path, response)
        return 0
    response["resumed_rows"] = resumed
    response["rows"] = [results[row["row_id"]] for row in request["rows"]]
    response["status"] = "ok"
    _write_response(response_path, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
