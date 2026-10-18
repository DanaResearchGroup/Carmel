"""Frozen-contract simulation harness with a Cantera subprocess boundary."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from carmel.adapters.t3 import T3_LAYOUT
from carmel.benchmarks.cantera_worker import THREAD_ENV, canonical_reaction_key
from carmel.benchmarks.chemkin import mechanism, normalize_for_ck2yaml
from carmel.benchmarks.contract import FrozenModel, read_pinned_file
from carmel.benchmarks.n_pentane import Contract, load_contract
from carmel.benchmarks.structure import canonical
from carmel.schemas.datasets import Absent
from carmel.services.artifacts import write_bytes, write_text
from carmel.services.respecth_series import SpeciesIdentity, parse_series_record

DEFAULT_ROW_TIMEOUT_S = 600.0
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 2)


class WorkerRequestRow(FrozenModel):
    """One JSR condition sent across the Cantera process boundary."""

    row_id: str
    temperature_k: float
    pressure_pa: float
    residence_time_s: float
    volume_m3: float
    composition: dict[str, float]
    output_species: tuple[str, ...]


class WorkerRequest(FrozenModel):
    """Versioned request accepted by the self-contained Cantera worker."""

    schema_version: Literal[1] = 1
    observable: Literal["jsr"] = "jsr"
    mechanism_path: Path
    thermo_path: Path
    converted_yaml_path: Path
    trimmed_yaml_path: Path
    checkpoint_path: Path
    reaction_source_lines: dict[str, tuple[int, ...]]
    rows: tuple[WorkerRequestRow, ...]
    row_timeout_s: float = DEFAULT_ROW_TIMEOUT_S
    workers: int = DEFAULT_WORKERS


class WorkerResponseRow(FrozenModel):
    """A converged result or an explicit condition-level refusal."""

    row_id: str
    status: Literal["ok", "refused"]
    mole_fractions: dict[str, float] | None = None
    reason: str | None = None
    wall_time_s: float
    integrator_steps: int | None
    signal: int | None = None
    fault_log: str | None = None
    progress: dict[str, Any] | None = None


class WorkerResponse(FrozenModel):
    """Versioned response validated when it re-enters the Cantera-free process."""

    schema_version: Literal[1]
    observable: str = "jsr"
    status: Literal["ok", "refused"]
    rows: tuple[WorkerResponseRow, ...]
    converter: dict[str, Any]
    plog_trims: tuple[dict[str, Any], ...] = ()
    pressure_refusals: tuple[dict[str, Any], ...] = ()
    plog_validation_sha256: str | None = None
    trimmed_yaml_sha256: str | None = None
    mechanism_semantic_sha256: str | None = None
    cantera_version: str | None = None
    validation_warnings: tuple[str, ...] = ()
    row_workers: int = 0
    configured_workers: int = 0
    row_timeout_s: float = DEFAULT_ROW_TIMEOUT_S
    checkpoint_path: str | None = None
    checkpoint_recoveries: tuple[dict[str, Any], ...] = ()
    resumed_rows: int = 0
    thread_env: dict[str, str]
    row_process_start_method: str
    load_average_at_start: tuple[float, float, float]
    simulation_lock: dict[str, Any]
    simulation_lock_sha256: str
    refusal: str | None = None
    validation_error: str | None = None
    construction_error: str | None = None
    wall_time_s: float | None = None
    signal: int | None = None
    fault_log: str | None = None


def _base_unit_value(value: Any) -> float:
    return float(value.converted_to_base().exact)


def _cyclic_ether(label: str) -> str | None:
    import re

    from rdkit import Chem

    match = re.fullmatch(r"C5H10O([1-5])-([1-5])", label)
    if match is None or match[1] == match[2]:
        return None
    left, right = int(match[1]) - 1, int(match[2]) - 1
    graph = Chem.RWMol()
    carbons = [graph.AddAtom(Chem.Atom("C")) for _ in range(5)]
    for first, second in zip(carbons, carbons[1:], strict=False):
        graph.AddBond(first, second, Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(carbons[left], oxygen, Chem.BondType.SINGLE)
    graph.AddBond(oxygen, carbons[right], Chem.BondType.SINGLE)
    molecule = graph.GetMol()
    Chem.SanitizeMol(molecule)
    return str(Chem.MolToSmiles(molecule, isomericSmiles=False))


def observable_smiles(label: str) -> str | None:
    """Resolve the documented labels used by the frozen JSR records/mechanisms."""
    known = {
        "O2": "O=O",
        "N2": "N#N",
        "HE": "[He]",
        "NC5H12": "CCCCC",
        "C2H4": "C=C",
        "C2H6": "CC",
        "C3H6": "CC=C",
        "C5H10-1": "CCCC=C",
        "C5H10-2": "CCC=CC",
        "CH2O": "C=O",
        "CH4": "C",
        "CO": "[C-]#[O+]",
        "CO2": "O=C=O",
        "CH3CHO": "CC=O",
    }
    smiles = known.get(label) or _cyclic_ether(label)
    return canonical(smiles) if smiles is not None else None


def _record_smiles(identity: SpeciesIdentity) -> str | None:
    if not isinstance(identity.smiles, Absent):
        return canonical(identity.smiles.raw)
    if not isinstance(identity.inchi, Absent):
        from rdkit import Chem

        molecule = Chem.MolFromInchi(identity.inchi.raw)  # type: ignore[no-untyped-call]
        if molecule is not None:
            return str(Chem.MolToSmiles(molecule, isomericSmiles=False))
    return None


_DETECTION_PROPERTY = re.compile(r"<\s*property\b[^>]*>", re.IGNORECASE)
_DETECTION_PROPERTY_LABEL = re.compile(r"""\b(?:name|label)\s*=\s*(["'])(?P<value>.*?)\1""", re.IGNORECASE | re.DOTALL)
_DETECTION_PROPERTY_TEXT = re.compile(r"detection[ _-]limit|limit[ _]of[ _]detection|\bLOD\b", re.IGNORECASE)
_DETECTION_NAMED = re.compile(r"<\s*detection_?limit\w*[\s/>]|\bdetection_?limit\w*\s*=", re.IGNORECASE)


def _declares_detection_limit(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="replace")
    if _DETECTION_NAMED.search(text):
        return True
    for tag in _DETECTION_PROPERTY.finditer(text):
        for label in _DETECTION_PROPERTY_LABEL.finditer(tag.group(0)):
            if _DETECTION_PROPERTY_TEXT.search(label.group("value")):
                return True
    return False


def _load_rows(contract: Contract, archive_path: Path) -> list[dict[str, Any]]:
    archive_bytes = read_pinned_file(archive_path)
    contract.split.archive.verify(archive_bytes)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for study in contract.split.studies:
            if study.set != "holdout":
                continue
            for item in study.files:
                raw = archive.read(item.path)
                if hashlib.sha256(raw).hexdigest() != item.sha256:
                    raise ValueError(f"SPLIT BLOB CHANGED: {item.path}")
                if _declares_detection_limit(raw):
                    raise ValueError(
                        f"{item.path} declares a stated detection limit: the stated-limit branch of the contract "
                        "zero_and_detection_limit_policy cannot be honoured because the ReSpecTh parser carries no "
                        "detection-limit field, so the run is refused rather than silently using the "
                        "half-minimum-positive fallback"
                    )
                record = parse_series_record(raw, contract.split.archive.to_pinned_archive(), item.path)
                series = record.envelope.series[0]
                constants = {entry.axis_id: entry.value for entry in series.constants}
                if isinstance(record.envelope.composition, Absent):
                    raise ValueError(f"unsupported reactor condition: {item.path} has no inlet composition")
                inlet = {
                    component.species_raw_name: _base_unit_value(component.amount)
                    for component in record.envelope.composition.components
                }
                species = {column.axis_id: column.species for column in record.species_columns}
                structures = {
                    identity.display: structure
                    for identity in species.values()
                    if (structure := _record_smiles(identity)) is not None
                }
                for point in series.points:
                    conditions = {**constants, **{entry.axis_id: entry.value for entry in point.coordinates}}
                    observations = {
                        species[entry.axis_id].display: _base_unit_value(entry.value)
                        for entry in point.observations
                        if not isinstance(entry.value, Absent)
                    }
                    rows.append(
                        {
                            "row_id": f"{item.path}:{point.point_id}",
                            "record": item.path,
                            "temperature_k": _base_unit_value(conditions["temperature"]),
                            "pressure_pa": _base_unit_value(conditions["pressure"]),
                            "residence_time_s": _base_unit_value(conditions["residence_time"]),
                            "volume_m3": _base_unit_value(conditions["volume"]),
                            "inlet": inlet,
                            "observations": observations,
                            "record_structures": structures,
                        }
                    )
    return rows


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _frozen_holdout_fingerprint(
    contract_bytes: bytes, contract: Contract, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    split = contract.split.model_dump(mode="json")
    scored_rows = [
        {
            "row_id": row["row_id"],
            "temperature_k": row["temperature_k"],
            "pressure_pa": row["pressure_pa"],
            "residence_time_s": row["residence_time_s"],
            "volume_m3": row["volume_m3"],
            "inlet": row["inlet"],
            "observations": row["observations"],
        }
        for row in sorted(rows, key=lambda item: str(item["row_id"]))
    ]
    identity = {
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "archive_sha256": contract.split.archive.sha256,
        "split_sha256": _canonical_sha256(split),
        "scored_rows_sha256": _canonical_sha256(scored_rows),
    }
    return {**identity, "sha256": _canonical_sha256(identity)}


def _structure_mapping(
    labels: set[str],
    mechanism_labels: list[str],
    *,
    observed_structures: dict[str, str] | None = None,
    contract_structures: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    observed_structures = observed_structures or {}
    contract_structures = contract_structures or {}

    def resolve(label: str, *, measured: bool) -> str | None:
        candidates = {
            item
            for item in (
                observed_structures.get(label) if measured else None,
                contract_structures.get(label),
                observable_smiles(label),
            )
            if item is not None
        }
        if len(candidates) > 1:
            raise ValueError(f"conflicting structures for species {label}")
        return next(iter(candidates), None)

    structural: dict[str, list[str]] = defaultdict(list)
    for candidate in mechanism_labels:
        smiles = resolve(candidate, measured=False)
        if smiles is not None:
            structural[smiles].append(candidate)
    mapped: dict[str, str] = {}
    refused: list[str] = []
    for label in sorted(labels):
        smiles = resolve(label, measured=True)
        matches = [] if smiles is None else structural.get(smiles, [])
        if len(matches) == 1:
            mapped[label] = matches[0]
        else:
            refused.append(label)
    return mapped, refused


_MetricLimits = tuple[dict[str, float], list[dict[str, Any]], dict[str, str]]


def _metric_limits(rows: list[dict[str, Any]]) -> _MetricLimits:
    """Return (limits, refusals, sources); sources maps species to "stated" or "half_min_positive"."""
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_species[row["species"]].append(row)
    limits: dict[str, float] = {}
    refusals: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for species, items in by_species.items():
        stated = [float(item["detection_limit"]) for item in items if item.get("detection_limit") is not None]
        valid_stated = [value for value in stated if math.isfinite(value) and value > 0]
        positives = [
            float(item["experimental"])
            for item in items
            if item.get("experimental") is not None
            and math.isfinite(float(item["experimental"]))
            and float(item["experimental"]) > 0
        ]
        if stated:
            if valid_stated:
                limits[species] = valid_stated[0]
                sources[species] = "stated"
            else:
                refusals.append({"species": species, "reason": "invalid_detection_limit"})
        elif positives:
            limits[species] = min(positives) / 2.0
            sources[species] = "half_min_positive"
        else:
            refusals.append({"species": species, "reason": "no_detection_limit_or_positive_observation"})
    return limits, refusals, sources


def score_metric(
    rows: list[dict[str, Any]],
    *,
    metric_limits: _MetricLimits | None = None,
) -> dict[str, Any]:
    """Compute the contract's clamped, species-balanced absolute log metric."""
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_species[row["species"]].append(row)
    limits, refusals, _ = metric_limits or _metric_limits(rows)

    def subset(low_temperature: bool) -> dict[str, Any]:
        per_species: dict[str, Any] = {}
        local_refusals = list(refusals)
        for species, items in sorted(by_species.items()):
            selected = [item for item in items if not low_temperature or 600 <= float(item["temperature_k"]) <= 900]
            errors: list[float] = []
            for item in selected:
                raw_experimental = item.get("experimental")
                simulated = item.get("simulated")
                reason = item.get("refusal")
                if reason:
                    local_refusals.append({"species": species, "row_id": item["row_id"], "reason": reason})
                    continue
                if raw_experimental is None or not math.isfinite(float(raw_experimental)):
                    local_refusals.append(
                        {
                            "species": species,
                            "row_id": item["row_id"],
                            "reason": "missing_or_nonfinite_experimental",
                        }
                    )
                    continue
                experimental = float(raw_experimental)
                if experimental < 0:
                    local_refusals.append(
                        {"species": species, "row_id": item["row_id"], "reason": "negative_experimental"}
                    )
                    continue
                if species not in limits:
                    continue
                if simulated is None or not math.isfinite(float(simulated)):
                    local_refusals.append(
                        {"species": species, "row_id": item["row_id"], "reason": "missing_or_nonfinite_simulated"}
                    )
                    continue
                if float(simulated) < 0:
                    local_refusals.append(
                        {"species": species, "row_id": item["row_id"], "reason": "negative_simulated"}
                    )
                    continue
                limit = limits[species]
                errors.append(abs(math.log10(max(float(simulated), limit) / max(experimental, limit))))
            if errors:
                per_species[species] = {
                    "score": fmean(errors),
                    "scored_rows": len(errors),
                    "detection_limit": limits[species],
                }
        scores = [item["score"] for item in per_species.values()]
        return {"overall": fmean(scores) if scores else None, "per_species": per_species, "refusals": local_refusals}

    return {"all": subset(False), "low_temperature_600_900_k": subset(True)}


def _temperature_band(temperature_k: float) -> str:
    if temperature_k < 600.0:
        return "below_600_k"
    if temperature_k <= 900.0:
        return "600_900_k"
    return "above_900_k"


def _refusal_class(reason: str | None) -> str:
    if reason is None:
        return "unspecified"
    return reason.split(":", maxsplit=1)[0]


def _refusal_summary(report: dict[str, Any]) -> dict[str, Any]:
    refused = [row for row in report["row_results"] if row["status"] != "ok"]
    by_class: dict[str, int] = defaultdict(int)
    by_band: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in refused:
        refusal = _refusal_class(row.get("reason"))
        band = _temperature_band(float(row["temperature_k"]))
        by_class[refusal] += 1
        by_band[band][refusal] += 1
    return {
        "total": len(refused),
        "by_class": dict(sorted(by_class.items())),
        "by_temperature_band": {band: dict(sorted(classes.items())) for band, classes in sorted(by_band.items())},
        "rows": refused,
    }


def build_separation_report(
    v_old: dict[str, Any],
    v_new: dict[str, Any],
    *,
    identity_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare locked reports primarily on their shared converged row intersection."""
    old_lock = v_old["simulation_lock_sha256"]
    new_lock = v_new["simulation_lock_sha256"]
    if old_lock != new_lock:
        raise ValueError("refusing separation across different simulation locks")
    old_holdout = v_old.get("holdout_fingerprint")
    new_holdout = v_new.get("holdout_fingerprint")
    if (
        not isinstance(old_holdout, dict)
        or not isinstance(new_holdout, dict)
        or old_holdout.get("sha256") != new_holdout.get("sha256")
    ):
        raise ValueError("refusing separation across different frozen holdout inputs")
    old_ok = {row["row_id"] for row in v_old["row_results"] if row["status"] == "ok"}
    new_ok = {row["row_id"] for row in v_new["row_results"] if row["status"] == "ok"}
    intersection = old_ok & new_ok
    metric_limits = _metric_limits(v_old["metric_rows"])
    limits = metric_limits[0]

    def observable_key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row["row_id"]), str(row["species"])

    def is_valid(row: dict[str, Any]) -> bool:
        if row["row_id"] not in intersection or row.get("refusal") or row["species"] not in limits:
            return False
        experimental = row.get("experimental")
        simulated = row.get("simulated")
        return (
            experimental is not None
            and simulated is not None
            and math.isfinite(float(experimental))
            and math.isfinite(float(simulated))
            and float(experimental) >= 0.0
            and float(simulated) >= 0.0
        )

    old_rows = [row for row in v_old["metric_rows"] if row["row_id"] in intersection]
    new_rows = [row for row in v_new["metric_rows"] if row["row_id"] in intersection]
    old_valid = {observable_key(row) for row in old_rows if is_valid(row)}
    new_valid = {observable_key(row) for row in new_rows if is_valid(row)}
    shared_observables = old_valid & new_valid

    def excluded(valid: set[tuple[str, str]]) -> dict[str, Any]:
        all_observables = {observable_key(row) for row in [*old_rows, *new_rows]}
        missing = all_observables - valid
        by_species: dict[str, int] = defaultdict(int)
        for _, species in missing:
            by_species[species] += 1
        return {"total": len(missing), "by_species": dict(sorted(by_species.items()))}

    def intersection_metric(report: dict[str, Any]) -> dict[str, Any]:
        return score_metric(
            [row for row in report["metric_rows"] if observable_key(row) in shared_observables],
            metric_limits=metric_limits,
        )

    old_refusals = _refusal_summary(v_old)
    new_refusals = _refusal_summary(v_new)
    return {
        "schema_version": 1,
        "simulation_lock_sha256": old_lock,
        "holdout_fingerprint": old_holdout,
        "primary": {
            "row_basis": "rows_both_mechanisms_converged_and_observables_valid_in_both",
            "row_count": len(intersection),
            "row_ids": sorted(intersection),
            "observable_count": len(shared_observables),
            "excluded_observables": {
                "v_old": excluded(old_valid),
                "v_new": excluded(new_valid),
            },
            "v_old": intersection_metric(v_old),
            "v_new": intersection_metric(v_new),
        },
        "own_converged_rows": {
            "v_old": {"row_count": len(old_ok), "metric": v_old["metric"]},
            "v_new": {"row_count": len(new_ok), "metric": v_new["metric"]},
        },
        "refusals": {"v_old": old_refusals, "v_new": new_refusals},
        "v_old_timeout_temperatures_k": [
            row["temperature_k"] for row in old_refusals["rows"] if row.get("reason") == "steady_state_timeout"
        ],
        "identity_check": identity_check,
    }


def _thermo_duplicates(source: bytes) -> list[dict[str, Any]]:
    entries: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    lines = source.decode(errors="replace").splitlines()
    for index in range(len(lines) - 3):
        if lines[index].rstrip().endswith("1") and all(
            lines[index + offset].rstrip().endswith(str(offset + 1)) for offset in range(1, 4)
        ):
            label = lines[index][:18].split()[0] if lines[index][:18].split() else ""
            if label:
                entries[label].append((index + 1, lines[index : index + 4]))
    return [
        {
            "species": label,
            "first_header": items[0][1][0],
            "later_headers": [item[1][0] for item in items[1:]],
            "first_used": True,
            "coefficients_differ": any(item[1][1:] != items[0][1][1:] for item in items[1:]),
        }
        for label, items in sorted(entries.items())
        if len(items) > 1
    ]


def resolve_cantera_python(explicit: Path | None) -> Path:
    """Pick the Cantera interpreter: ``--cantera-python``, else ``$T3_PYTHON``, else refuse.

    Never falls back to ``sys.executable``: Carmel's own environment has no Cantera.
    """
    env_var = T3_LAYOUT.T3_PYTHON_ENV_VAR
    if explicit is not None:
        chosen, source = explicit, "--cantera-python"
    elif os.environ.get(env_var):
        chosen, source = Path(os.environ[env_var]), f"${env_var}"
    else:
        raise ValueError(
            f"no Cantera interpreter: pass --cantera-python or set {env_var} to a Python "
            "interpreter that has Cantera installed"
        )
    if not (chosen.is_file() and os.access(chosen, os.X_OK)):
        raise ValueError(f"Cantera interpreter from {source} is not an existing executable file: {chosen}")
    return chosen


def invoke_cantera_worker(
    request: WorkerRequest,
    *,
    run_dir: Path,
    cantera_python: Path,
) -> WorkerResponse:
    """Run and validate the versioned Cantera subprocess exchange."""
    request_path, response_path = run_dir / "request.json", run_dir / "response.json"
    write_text(request_path, request.model_dump_json(indent=2) + "\n")
    response_path.unlink(missing_ok=True)
    worker = Path(__file__).with_name("cantera_worker.py")
    worker_env = {**os.environ, **THREAD_ENV}
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        completed = subprocess.run(  # noqa: S603
            [str(cantera_python), str(worker), str(request_path), str(response_path)],
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=worker_env,
        )
    if completed.returncode or not response_path.exists():
        raise ValueError(f"Cantera subprocess failed with exit code {completed.returncode}")
    return WorkerResponse.model_validate_json(response_path.read_text())


def _rev_reaction_lines(mechanism_bytes: bytes, reaction_lines: set[int]) -> set[int]:
    """Return reaction line numbers followed by a REV auxiliary line before the next reaction or END."""
    found: set[int] = set()
    current: int | None = None
    for number, raw in enumerate(mechanism_bytes.decode(errors="replace").splitlines(), 1):
        code = raw.split("!", 1)[0].strip()
        if not code:
            continue
        if number in reaction_lines:
            current = number
        elif code.split()[0].upper() == "END":
            current = None
        elif current is not None and re.match(r"REV\s*/", code, re.IGNORECASE):
            found.add(current)
    return found


def _source_line_keys(mechanism_bytes: bytes, reactions: list[Any]) -> dict[str, list[int]]:
    """Key CHEMKIN reaction lines as ck2yaml will emit them; REV splits into two irreversible reactions."""
    rev_lines = _rev_reaction_lines(mechanism_bytes, {reaction["line"] for reaction in reactions})
    source_lines: dict[str, list[int]] = defaultdict(list)
    for reaction in reactions:
        equation = reaction["equation"]
        if reaction["line"] in rev_lines:
            left, _, right = re.split(r"(<=>|=>|=)", equation, maxsplit=1)
            source_lines[canonical_reaction_key(f"{left}=>{right}")].append(reaction["line"])
            source_lines[canonical_reaction_key(f"{right}=>{left}")].append(reaction["line"])
        else:
            source_lines[canonical_reaction_key(equation)].append(reaction["line"])
    return source_lines


def run_harness(
    *,
    contract_path: Path,
    mechanism_path: Path,
    mechanism_key: Literal["v_old", "v_new"],
    thermo_path: Path,
    thermo_url: str,
    thermo_retrieval_date: str,
    thermo_sha256: str,
    archive_path: Path,
    run_dir: Path,
    cantera_python: Path,
    row_timeout_s: float = DEFAULT_ROW_TIMEOUT_S,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Score one pinned mechanism against a frozen contract holdout."""
    contract_bytes = read_pinned_file(contract_path)
    contract = load_contract(contract_path)
    mechanism_bytes = read_pinned_file(mechanism_path)
    contract.mechanisms[mechanism_key].verify(mechanism_bytes)
    thermo_bytes = read_pinned_file(thermo_path)
    if hashlib.sha256(thermo_bytes).hexdigest() != thermo_sha256:
        raise ValueError("PIN CHANGED: thermo bytes differ from the supplied pin")
    normalized_mechanism = normalize_for_ck2yaml(mechanism_bytes, file=mechanism_path.name)
    normalized_thermo = normalize_for_ck2yaml(thermo_bytes, file=thermo_path.name)
    rows = _load_rows(contract, archive_path)
    holdout_fingerprint = _frozen_holdout_fingerprint(contract_bytes, contract, rows)
    mechanism_labels, reactions = mechanism(normalized_mechanism.data)
    labels = {label for row in rows for label in (*row["inlet"], *row["observations"])}
    observed_structures: dict[str, str] = {}
    for row in rows:
        for label, structure in row["record_structures"].items():
            if label in observed_structures and observed_structures[label] != structure:
                raise ValueError(f"conflicting record structures for species {label}")
            observed_structures[label] = structure
    contract_structures: dict[str, str] = {}
    for species in (*contract.old_species, *contract.new_species):
        if species.smiles is None:
            continue
        structure = canonical(species.smiles)
        if species.label in contract_structures and contract_structures[species.label] != structure:
            raise ValueError(f"conflicting contract structures for species {species.label}")
        contract_structures[species.label] = structure
    mapping, unmapped = _structure_mapping(
        labels,
        mechanism_labels,
        observed_structures=observed_structures,
        contract_structures=contract_structures,
    )
    measured = {label for row in rows for label in row["observations"]}
    if len(measured - set(unmapped)) / len(measured) < 0.5:
        raise ValueError("fewer than half the measured species map structurally")

    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_mechanism_path = run_dir / "mechanism.normalized.inp"
    normalized_thermo_path = run_dir / "thermo.normalized.dat"
    write_bytes(normalized_mechanism_path, normalized_mechanism.data)
    write_bytes(normalized_thermo_path, normalized_thermo.data)
    source_lines = _source_line_keys(normalized_mechanism.data, reactions)
    request_rows: list[WorkerRequestRow] = []
    for row in rows:
        if any(label not in mapping for label in row["inlet"]):
            continue
        request_rows.append(
            WorkerRequestRow(
                row_id=row["row_id"],
                temperature_k=row["temperature_k"],
                pressure_pa=row["pressure_pa"],
                residence_time_s=row["residence_time_s"],
                volume_m3=row["volume_m3"],
                composition={mapping[label]: amount for label, amount in row["inlet"].items()},
                output_species=tuple(sorted({mapping[label] for label in row["observations"] if label in mapping})),
            )
        )
    request = WorkerRequest(
        mechanism_path=normalized_mechanism_path,
        thermo_path=normalized_thermo_path,
        converted_yaml_path=run_dir / "converted.yaml",
        trimmed_yaml_path=run_dir / "trimmed.yaml",
        checkpoint_path=run_dir / "rows.jsonl",
        reaction_source_lines=source_lines,
        rows=tuple(request_rows),
        row_timeout_s=row_timeout_s,
        workers=workers,
    )
    response = invoke_cantera_worker(request, run_dir=run_dir, cantera_python=cantera_python)
    if response.status != "ok":
        detail = response.validation_error or response.construction_error or ""
        raise ValueError(f"Cantera subprocess refused: {response.refusal}: {detail}")
    simulated = {item.row_id: item for item in response.rows}
    row_refusals: dict[str, str] = {}
    for row in rows:
        result = simulated.get(row["row_id"])
        if any(label not in mapping for label in row["inlet"]):
            row_refusals[row["row_id"]] = "unmapped_inlet_species"
        elif result is None:
            row_refusals[row["row_id"]] = "unsupported_reactor_condition_or_solver_failure"
        elif result.status != "ok":
            row_refusals[row["row_id"]] = result.reason or "unsupported_reactor_condition_or_solver_failure"
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        result = simulated.get(row["row_id"])
        for species, experimental in row["observations"].items():
            entry = {
                "row_id": row["row_id"],
                "species": species,
                "temperature_k": row["temperature_k"],
                "experimental": experimental,
                "detection_limit": None,  # stated limits are refused at load (_declares_detection_limit)
            }
            if species not in mapping:
                entry["refusal"] = "unmapped_species"
            elif row["row_id"] in row_refusals:
                entry["refusal"] = row_refusals[row["row_id"]]
            else:
                assert result is not None
                entry["simulated"] = (result.mole_fractions or {}).get(mapping[species])
            metric_rows.append(entry)
    converged = sum(item.status == "ok" for item in response.rows)
    convergence_fraction = converged / len(rows)
    temperature_by_id = {str(row["row_id"]): float(row["temperature_k"]) for row in rows}
    row_table: list[dict[str, Any]] = []
    for row in rows:
        item = simulated.get(row["row_id"])
        reason = row_refusals.get(row["row_id"])
        row_table.append(
            {
                "row_id": row["row_id"],
                "temperature_k": temperature_by_id[row["row_id"]],
                "status": "refused" if reason is not None else "ok",
                "reason": reason,
                "wall_time_s": item.wall_time_s if item is not None else None,
                "integrator_steps": item.integrator_steps if item is not None else None,
                "signal": item.signal if item is not None else None,
                "fault_log": item.fault_log if item is not None else None,
                "progress": item.progress if item is not None else None,
            }
        )
    metric_limits = _metric_limits(metric_rows)
    return {
        "schema_version": 1,
        "case": contract.case,
        "mechanism_key": mechanism_key,
        "inputs": {
            "ck2yaml_version": response.cantera_version,
            "mechanism": {
                "path": str(mechanism_path),
                "raw_sha256": normalized_mechanism.raw_sha256,
                "normalized_sha256": normalized_mechanism.normalized_sha256,
                "rewrites": [item.model_dump() for item in normalized_mechanism.rewrites],
            },
            "thermo": {
                "path": str(thermo_path),
                "url": thermo_url,
                "retrieval_date": thermo_retrieval_date,
                "raw_sha256": normalized_thermo.raw_sha256,
                "normalized_sha256": normalized_thermo.normalized_sha256,
                "rewrites": [item.model_dump() for item in normalized_thermo.rewrites],
            },
            "trimmed_yaml_sha256": response.trimmed_yaml_sha256,
            "mechanism_semantic_sha256": response.mechanism_semantic_sha256,
        },
        "species_mapping": {
            label: {
                "mechanism_label": target,
                "smiles": observed_structures.get(label) or contract_structures.get(label) or observable_smiles(label),
            }
            for label, target in mapping.items()
        },
        "unmapped_species": unmapped,
        "thermo_duplicates": _thermo_duplicates(thermo_bytes),
        "converter": response.converter,
        "cantera_version": response.cantera_version,
        "row_workers": response.row_workers,
        "configured_workers": response.configured_workers,
        "row_timeout_s": response.row_timeout_s,
        "checkpoint_path": response.checkpoint_path,
        "checkpoint_recoveries": response.checkpoint_recoveries,
        "resumed_rows": response.resumed_rows,
        "thread_env": response.thread_env,
        "row_process_start_method": response.row_process_start_method,
        "load_average_at_start": response.load_average_at_start,
        "simulation_lock": response.simulation_lock,
        "simulation_lock_sha256": response.simulation_lock_sha256,
        "holdout_fingerprint": holdout_fingerprint,
        "validation_warnings": response.validation_warnings,
        "plog_trims": response.plog_trims,
        "plog_validation_sha256": response.plog_validation_sha256,
        "rows": {"total": len(rows), "converged": converged, "refused": len(rows) - converged},
        "row_results": row_table,
        "convergence_gate": {
            "threshold": 0.7,
            "fraction": convergence_fraction,
            "passed": convergence_fraction >= 0.7,
        },
        "detection_limits": metric_limits[0],
        "detection_limit_sources": metric_limits[2],
        "metric": score_metric(metric_rows, metric_limits=metric_limits),
        "metric_rows": metric_rows,
    }
