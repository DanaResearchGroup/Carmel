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
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

FFCM1_EXCITED_SHA256 = "49d61ec0c538eca831336d2ab316a0dbed212b6f4d44fa10082ab7ca071e918b"
EXCITED_ALIASES = {"OH*": "OH*", "OHV": "OH*", "CH*": "CH*", "CHV": "CH*"}


def resolve_ignition_target(target: str, species_names: list[str]) -> str | None:
    """Resolve only the two declared excited-state aliases; never guess a label."""
    if target in ("pressure", "temperature") or target in species_names:
        return target
    canonical = EXCITED_ALIASES.get(target)
    matches = [name for name in species_names if canonical is not None and EXCITED_ALIASES.get(name) == canonical]
    return matches[0] if len(matches) == 1 else None


def load_excited_submechanism() -> dict[str, Any]:
    raw = (Path(__file__).parents[1] / "data" / "ffcm1_excited.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FFCM1_EXCITED_SHA256:
        raise ValueError("PIN CHANGED: FFCM-1 excited-state submechanism")
    data: dict[str, Any] = json.loads(raw)
    return data


def _equation_species(equation: str) -> set[str]:
    compact = re.sub(r"\s+", "", equation)
    compact = re.sub(r"\(\+[^)]+\)", "", compact)
    return {re.sub(r"^\d+(?:\.\d+)?", "", term) for term in re.split(r"<=>|=>|=|\+", compact)} - {"M", ""}


def _side_counts(side: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for term in side.split("+"):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)?(.+)", term)
        if match:
            counts[match[2]] = counts.get(match[2], 0.0) + float(match[1] or 1)
    return counts


def _explicit_colliders(equation: str, labels: set[str]) -> set[str]:
    """Labels written with the same stoichiometry on both sides: an unchanged third body."""
    compact = re.sub(r"\(\+[^)]+\)", "", re.sub(r"\s+", "", equation))
    sides = re.split(r"<=>|=>|=", compact, maxsplit=1)
    if len(sides) != 2:
        return set()
    left, right = map(_side_counts, sides)
    return {label for label in labels if left.get(label) and left.get(label) == right.get(label)}


def append_excited_submechanism(mechanism: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace native excited species/reactions, preserving all ground-state records.

    This is shared by every IDT worker request, including baseline and candidates.
    Local units on the appended reactions prevent parent-unit reinterpretation.
    """
    pinned = load_excited_submechanism()
    result = copy.deepcopy(mechanism)
    aliases = set(EXCITED_ALIASES)
    native = [s for s in result["species"] if s["name"] in aliases]
    for species in native:
        expected = {"O" if EXCITED_ALIASES[species["name"]] == "OH*" else "C": 1, "H": 1}
        if species["composition"] != expected:
            raise ValueError("NEEDS-INPUT: excited label identifies a ground-state species with different composition")
    ground_species = [s for s in result["species"] if s["name"] not in aliases]
    names = {s["name"] for s in ground_species} | {"OH*", "CH*"}
    required = set().union(*(_equation_species(r["equation"]) for r in pinned["reactions"]))
    if required - names:
        raise ValueError(f"NEEDS-INPUT: missing ground-state dependencies: {sorted(required - names)}")
    removed, kept = [], []
    for reaction in result["reactions"]:
        if _explicit_colliders(reaction["equation"], aliases):
            # e.g. H + O2 + OHV <=> HO2 + OHV is ground-state chemistry with an excited third body.
            raise ValueError("NEEDS-INPUT: replacement would alter a ground-state reaction collider")
        if _equation_species(reaction["equation"]) & aliases:
            removed.append(reaction)
        else:
            if aliases & set(reaction.get("efficiencies", {})) or any(
                f"(+{label})" in reaction["equation"].replace(" ", "") for label in aliases
            ):
                raise ValueError("NEEDS-INPUT: replacement would alter a ground-state reaction collider")
            kept.append(reaction)
    if len(result["phases"]) != 1 or not isinstance(result["phases"][0]["species"], list):
        raise ValueError("NEEDS-INPUT: expected one explicit gas-phase species list")
    phase = result["phases"][0]
    phase["species"] = [name for name in phase["species"] if name not in aliases] + ["OH*", "CH*"]
    result["species"] = ground_species + pinned["species"]
    appended = [{**reaction, "units": pinned["units"]} for reaction in pinned["reactions"]]
    result["reactions"] = kept + appended
    return result, {
        "sha256": FFCM1_EXCITED_SHA256,
        "source": pinned["source"],
        "replaced_species": [s["name"] for s in native],
        "replaced_reactions": [r["equation"] for r in removed],
        "appended_species": ["OH*", "CH*"],
        "appended_reactions": [r["equation"] for r in appended],
        "ground_species_preserved": len(ground_species),
        "ground_reactions_preserved": len(kept),
    }


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
JSR_STEPS_PER_ITERATION = 10
IDT_MAX_STEPS = 1000000
DEFAULT_ROW_WALL_CEILING_S = 3600.0
PRESSURE_CONTROLLER_K = 1.0e-5
IDT_RTOL = 1.0e-8
IDT_ATOL = 1.0e-16
IDT_HORIZON_S = 1.0
IDT_TEMPERATURE_RISE_K = 50.0
# Reviewed case-2 v_new/thermo.normalized.dat headers: OH* line 25, CH* line 145.
# SHA-256 of the three 10-byte temperature fields followed by fourteen 15-byte
# coefficient fields, in CHEMKIN card order, with padding retained.
REVIEWED_OH_NASA_SHA256 = "85ab13a7ce62508df7daf3411313e27f5e650008f665ec790e04c194aa7fb2d7"
REVIEWED_CH_NASA_SHA256 = "4a3c493e51e59ada6dfba5d198e487ab5e78b907d4bd0ba32f217df460ae880f"


@dataclass(frozen=True)
class ReviewedNasaWarning:
    """One reviewed warning and its exact source, independent of label spelling."""

    labels: tuple[str, ...]
    quantity: Literal["h/RT", "s/R", "cp/R"]
    tmid_k: int
    numeric_field_sha256: str
    source_file: str
    source_line: int
    reason: str
    requires_below_tmid: bool = False


REVIEWED_NASA_WARNINGS = (
    ReviewedNasaWarning(
        ("OH*", "OHV"),
        "h/RT",
        1000,
        REVIEWED_OH_NASA_SHA256,
        "nuigmech_c5_july2015.therm",
        25,
        "Existing case-2 reviewed excited-OH enthalpy warning; exact numeric fields preserved.",
    ),
    ReviewedNasaWarning(
        ("CH*", "CHV"),
        "h/RT",
        1000,
        REVIEWED_CH_NASA_SHA256,
        "nuigmech_c5_july2015.therm",
        145,
        "Existing case-2 reviewed excited-CH enthalpy warning; exact numeric fields preserved.",
    ),
    ReviewedNasaWarning(
        ("EC6H13O2",),
        "s/R",
        1388,
        "d1d4928748f6538395afde16ef286c6a78bc4cf4994c1f979f2290b1e6efb0b6",
        "prf_v3_therm_dat.txt",
        2987,
        "0.64% s/R jump (107.806 vs 107.116) at 1388 K, above the case-3 holdout maximum "
        "temperature (1100 K), so the high-T polynomial is never evaluated at a scored condition.",
        requires_below_tmid=True,
    ),
)
# Preserve the existing case-2 simulation-lock serialization and fingerprint.
_REVIEWED_NASA_LABELS = {
    label: review.numeric_field_sha256
    for review in REVIEWED_NASA_WARNINGS
    if not review.requires_below_tmid
    for label in review.labels
}
_REVIEWED_NASA_WARNING = re.compile(
    r"NasaPoly2::validate:\s*For species ([^,\s]+), "
    r"discontinuity in (h/RT|s/R|cp/R) detected at Tmid = (\d+(?:\.0*)?)\s*"
    rf"(?:Value computed using low-temperature polynomial:\s*{_NUMBER_PATTERN}\s*"
    rf"Value computed using high-temperature polynomial:\s*{_NUMBER_PATTERN}\s*)?"
)
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
        "schema_version": 2,
        "simulator": {"cantera_version": cantera_version},
        "jsr": {
            "rtol": JSR_RTOL,
            "atol": JSR_ATOL,
            "max_steps": JSR_MAX_STEPS,
            "steady_state_max_iterations": JSR_MAX_STEPS,
            "steps_per_iteration": JSR_STEPS_PER_ITERATION,
            "max_accepted_steps": JSR_STEPS_PER_ITERATION * JSR_MAX_STEPS,
            "initial_state": "row_inlet",
            "temperature_continuation": False,
            "energy": "off",
            "pressure_controller_K": PRESSURE_CONTROLLER_K,
        },
        "idt": {
            "excited_state_submechanism_sha256": FFCM1_EXCITED_SHA256,
            "target_aliases": dict(EXCITED_ALIASES),
            "rtol": IDT_RTOL,
            "atol": IDT_ATOL,
            "horizon_s": IDT_HORIZON_S,
            "max_accepted_steps": IDT_MAX_STEPS,
            "minimum_temperature_rise_k": IDT_TEMPERATURE_RISE_K,
            "reactor": "adiabatic_constant_volume_IdealGasReactor",
            "initial_state": "stored_temperature_pressure_composition",
            "criterion": "exact_target_and_source_definition_t3_v1",
            "gradient": "numpy.gradient_default_edge_order_1",
            "resolution": "t3_source_defined_idt_is_resolved_excited_relative_span_v3",
            "plog_pressure_guard": "initial_and_each_step_crossing_removed_open_brackets_v1",
        },
        "process": {
            "row_wall_ceiling_s": float(row_timeout_s),
            "wall_ceiling_policy": "abort_run_without_row_refusal",
            "thread_env": THREAD_ENV,
            "row_start_method": ROW_PROCESS_START_METHOD,
        },
        "ck2yaml": {
            "flags": ["--permissive", "--verbose", "--no-validate"],
            "eless_exponent_normalization": "nasa_fixed_width_and_reaction_rate_fields_only_v2",
            "nasa7_fourth_card_normalization": "restore_absent_or_lone_column_81_index_after_valid_cards_v2",
            "warning_allowlist": [
                "redundant_species_declaration",
                "redundant_thermo_entry_first_wins",
                "comment_only_unparsable_thermo_block",
            ],
            "validation_warning_allowlist": "reviewed_nasa_numeric_field_sha256_v2",
            "reviewed_nasa_sha256_by_label": dict(_REVIEWED_NASA_LABELS),
            "plog_rule": "trim_nonpositive_levels_only_outside_scored_interpolation_intervals_v1",
            "plog_validation_temperatures_k": list(PLOG_VALIDATION_TEMPERATURES_K),
        },
    }


def validate_row_refusal(reason: str | None, lock: dict[str, Any]) -> None:
    """Accept wall-time row refusals only with their historical version-1 lock."""
    legacy = {"steady_state_timeout": "jsr", "ignition_timeout": "idt"}
    if reason in legacy and (lock.get("schema_version") != 1 or reason + "_s" not in lock.get(legacy[reason], {})):
        raise ValueError(f"{reason} requires its historical simulation lock")


class MaxStepsExceeded(RuntimeError):
    """A deterministic solver budget was exhausted before reaching the target."""

    def __init__(self, integrator_steps: int) -> None:
        super().__init__("max_steps_exceeded")
        self.integrator_steps = integrator_steps


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
    """An evaluator defect or safety ceiling prevents scoring the run."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("construction_error") or result.get("refusal")))
        self.result = result


class PlogValidationError(ValueError):
    """A PLOG rate cannot be validated without guessing its units."""

    reason = "unsupported_plog_activation_energy_unit"


class PlogTrimmedIntervalEntered(RuntimeError):
    """An IDT trace reached kinetics altered by PLOG trimming."""

    def __init__(self, integrator_steps: int) -> None:
        super().__init__("plog_trimmed_interval_entered")
        self.integrator_steps = integrator_steps


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


def nasa7_record_digests(sources: tuple[bytes, ...]) -> dict[str, str | None]:
    """Hash fixed-width numeric records; first entries win as in ck2yaml.

    Sources are ordered as the converter reads them: kinetics, then thermo.
    An invalid first record cannot be rescued by an approved later duplicate.
    """
    digests: dict[str, str | None] = {}
    for source in sources:
        lines = [line.partition(b"!")[0] for line in source.splitlines()]
        in_thermo = False
        for index, line in enumerate(lines):
            head = line.split()[0].upper() if line.split() else b""
            if head in (b"THERMO", b"THERM"):
                in_thermo = True
            elif head == b"END":
                in_thermo = False
            if not in_thermo or line[79:80] != b"1":
                continue
            label = line[:18].split()[0].decode("ascii") if line[:18].split() else ""
            if label in digests:
                continue
            digests[label] = None
            cards = lines[index : index + 4]
            if len(cards) != 4 or any(card[79:80] != str(i).encode() for i, card in enumerate(cards, 1)):
                continue
            fields = [line[start : start + 10] for start in (45, 55, 65)]
            fields += [
                card[start : start + 15]
                for card, count in zip(cards[1:], (5, 5, 4), strict=True)
                for start in range(0, 15 * count, 15)
            ]
            if not all(re.fullmatch(_NUMBER_PATTERN, field.decode("ascii").strip()) for field in fields):
                continue
            digests[label] = hashlib.sha256(b"".join(fields)).hexdigest()
    return digests


def reviewed_nasa_warning(message: str) -> ReviewedNasaWarning | None:
    """Find the exact reviewed label, quantity and midpoint; do not approve bytes."""
    match = _REVIEWED_NASA_WARNING.fullmatch(message.strip())
    if match is None:
        return None
    return next(
        (
            review
            for review in REVIEWED_NASA_WARNINGS
            if match[1] in review.labels and match[2] == review.quantity and float(match[3]) == review.tmid_k
        ),
        None,
    )


def cantera_validation_refusals(
    messages: list[str],
    thermo_sources: tuple[bytes, ...] = (),
    *,
    holdout_max_temperature_k: float | None = None,
) -> list[str]:
    """Match every review field and digest; enforce any temperature-dependent review."""
    digests = nasa7_record_digests(thermo_sources)
    refused = []
    for message in messages:
        match = _REVIEWED_NASA_WARNING.fullmatch(message.strip())
        review = reviewed_nasa_warning(message)
        if (
            match is None
            or review is None
            or digests.get(match[1]) != review.numeric_field_sha256
            or (
                review.requires_below_tmid
                and not (holdout_max_temperature_k is not None and 0 < holdout_max_temperature_k < review.tmid_k)
            )
        ):
            refused.append(message)
    return refused


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
        # Restrict classification to this locked, valid-argument call. The same
        # undefined name also occurs on unrelated Cantera error paths.
        try:
            network.advance_to_steady_state(max_steps=JSR_MAX_STEPS)
        except Exception as exc:
            steps = int(network.solver_stats.get("steps", 0))
            signature = (type(exc) is NameError and str(exc) == "name 'CanteraError' is not defined") or (
                isinstance(exc, getattr(ct, "CanteraError", ()))
                and str(exc) == "Maximum number of steps reached before convergence below maximum residual"
            )
            if signature and steps == JSR_STEPS_PER_ITERATION * JSR_MAX_STEPS:
                raise MaxStepsExceeded(steps) from exc
            raise
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
            "reason": str(exc) if isinstance(exc, MaxStepsExceeded) else f"solver_failure: {exc}",
            "wall_time_s": time.monotonic() - started,
            "integrator_steps": steps,
        }


def source_defined_idt(times: list[float], values: list[float], criterion_type: str) -> float | None:
    """T3 experimental event semantics, including its nonuniform numerical gradient."""
    import numpy as np

    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if (
        t.ndim != 1
        or y.ndim != 1
        or len(t) != len(y)
        or len(t) < 2
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(t) <= 0)
    ):
        return None
    peak = int(np.argmax(y))
    if criterion_type == "max":
        return float(t[peak])
    if criterion_type == "1/2 max":
        threshold = 0.5 * float(y[peak])
        crossings = np.flatnonzero(y[: peak + 1] >= threshold)
        if not len(crossings):
            return None
        upper = int(crossings[0])
        if upper == 0:
            return float(t[0])
        lower = upper - 1
        delta = y[upper] - y[lower]
        if delta == 0:
            return float(t[upper])
        return float(t[lower] + (threshold - y[lower]) / delta * (t[upper] - t[lower]))
    slopes = np.gradient(y, t)
    index = int(np.argmax(slopes))
    slope = float(slopes[index])
    if not math.isfinite(slope) or slope <= 0:
        return None
    if criterion_type == "d/dt max":
        return float(t[index])
    if criterion_type == "d/dt max extrapolated":
        intersection = float(t[index] - (y[index] - y[0]) / slope)
        return intersection if math.isfinite(intersection) else None
    raise ValueError(f"unknown source-defined ignition criterion: {criterion_type!r}")


def idt_trace_is_resolved(times: list[float], values: list[float], criterion_type: str, target: str) -> bool:
    """T3 history-free experimental resolution check (including the settled tail)."""
    import numpy as np

    if source_defined_idt(times, values, criterion_type) is None or len(times) < 3:
        return False
    y = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(y))
    minimum_span = max(abs(float(y[0])), 1.0) * 1e-6 if target in ("pressure", "temperature") else 1e-12
    if target in EXCITED_ALIASES:
        # Emission mole fractions can be far below the ground-species floor.
        amplitude = float(np.max(np.abs(y)))
        if amplitude <= 0:
            return False
        minimum_span = 1e-6 * amplitude
    if not math.isfinite(span) or span <= minimum_span:
        return False
    tail_length = max(3, len(y) // 10)
    peak = int(np.argmax(y))
    # Settled-tail floors: absolute machine epsilon for ground-state targets; for excited targets,
    # epsilon scaled to the signal, since an absolute floor would pass a still-rising faint trace.
    excited = target in EXCITED_ALIASES
    eps = float(np.finfo(float).eps)
    if criterion_type in ("max", "1/2 max"):
        tail_floor = eps * float(np.max(np.abs(y))) if excited else eps
        return peak < len(y) - 1 or float(np.ptp(y[-tail_length:])) <= max(span * 1e-3, tail_floor)
    slopes = np.gradient(y, np.asarray(times, dtype=np.float64))
    index = int(np.argmax(slopes))
    slope_floor = eps * float(np.max(np.abs(slopes))) if excited else eps
    return bool(
        index < len(y) - tail_length
        and float(np.max(np.abs(slopes[-tail_length:]))) <= max(abs(float(slopes[index])) * 0.1, slope_floor)
    )


def integrate_idt_trace(ct: Any, gas: Any, row: dict[str, Any]) -> tuple[list[float], list[float], list[float], int]:
    """Integrate a fixed one-second, adiabatic constant-volume trace from t=0."""
    try:
        reactor = ct.IdealGasReactor(gas, energy="on")
        network = ct.ReactorNet([reactor])
        network.atol, network.rtol = IDT_ATOL, IDT_RTOL
    except Exception as exc:
        raise RunLevelSimulationError(
            {"refusal": "reactor_construction_failed", "construction_error": str(exc)}
        ) from exc
    target = row["ignition_target"]
    if target in EXCITED_ALIASES:
        target = resolve_ignition_target(target, gas.species_names)
    species_index = gas.species_index(target) if target not in ("pressure", "temperature") else None

    def value() -> float:
        if target == "pressure":
            return float(reactor.phase.P)
        if target == "temperature":
            return float(reactor.T)
        return float(reactor.phase.X[species_index])

    times, values, temperatures = [0.0], [value()], [float(reactor.T)]
    brackets = row.get("plog_trimmed_brackets_pa", [])
    previous_pressure = float(reactor.phase.P)

    def check_pressure(pressure: float) -> None:
        # A continuous pressure path can cross an entire removed bracket between
        # samples; checking only the endpoints would miss that entry. Kept
        # boundary levels themselves remain valid, hence the strict inequalities.
        for lower, upper in brackets:
            if (lower is None or max(previous_pressure, pressure) > lower) and (
                upper is None or min(previous_pressure, pressure) < upper
            ):
                raise PlogTrimmedIntervalEntered(int(network.solver_stats.get("steps", 0)))

    check_pressure(previous_pressure)
    _write_progress("integration", last_reactor_time_s=0.0, temperature_k=float(reactor.T), integrator_steps=0)
    steps = 0
    while network.time < IDT_HORIZON_S:
        if steps >= IDT_MAX_STEPS:
            raise MaxStepsExceeded(steps)
        network.step()
        steps = int(network.solver_stats["steps"])
        pressure = float(reactor.phase.P)
        check_pressure(pressure)
        previous_pressure = pressure
        times.append(float(network.time))
        values.append(value())
        temperatures.append(float(reactor.T))
        # Sidecars stay bounded in cost while still identifying integration crashes.
        if len(times) % 100 == 0:
            _write_progress(
                "integration",
                last_reactor_time_s=times[-1],
                temperature_k=temperatures[-1],
                integrator_steps=int(network.solver_stats.get("steps", 0)),
            )
    steps = int(network.solver_stats.get("steps", 0))
    _write_progress("complete", last_reactor_time_s=times[-1], temperature_k=temperatures[-1], integrator_steps=steps)
    return times, values, temperatures, steps


def _simulate_idt_row(trimmed_path: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    import cantera as ct

    started = time.monotonic()
    target = row["ignition_target"]
    result: dict[str, Any] = {
        "row_id": row["row_id"],
        "status": "refused",
        "ignition_target": target,
        "integrator_steps": 0,
        "wall_time_s": 0.0,
    }
    try:
        _write_progress("construction")
        gas = ct.Solution(trimmed_path)
        if resolve_ignition_target(target, gas.species_names) is None:
            result["reason"] = "target_species_absent"
            return "row", result
        gas.TPX = row["temperature_k"], row["pressure_pa"], row["composition"]
    except Exception as exc:
        return "run", {"refusal": "reactor_construction_failed", "construction_error": str(exc)}
    try:
        times, values, temperatures, steps = integrate_idt_trace(ct, gas, row)
        result["integrator_steps"] = steps
        delay = source_defined_idt(times, values, row["ignition_type"])
        if delay is None:
            result["reason"] = "no_ignition_within_horizon"
        elif not math.isfinite(delay):
            result["reason"] = "source_delay_nonfinite"
        elif not 0 < delay <= IDT_HORIZON_S:
            result["reason"] = "source_delay_outside_horizon"
        elif max(temperatures) - temperatures[0] < IDT_TEMPERATURE_RISE_K:
            result["reason"] = "temperature_rise_below_minimum"
        elif not idt_trace_is_resolved(times, values, row["ignition_type"], target):
            result["reason"] = "trace_unresolved"
        else:
            result.update(status="ok", ignition_delay_s=delay)
    except RunLevelSimulationError as exc:
        return "run", exc.result
    except (PlogTrimmedIntervalEntered, MaxStepsExceeded) as exc:
        result["reason"] = str(exc)
        result["integrator_steps"] = exc.integrator_steps
    except Exception as exc:
        result["reason"] = f"solver_failure: {exc}"
        result["integrator_steps"] = None
    result["wall_time_s"] = time.monotonic() - started
    return "row", result


def _row_fingerprint(
    mechanism_sha256: str,
    row: dict[str, Any],
    row_timeout_s: float,
    cantera_version: str,
    plog_validation_identity: str = "",
) -> str:
    return checkpoint_fingerprint(
        mechanism_sha256, row, simulation_lock_sha256(row_timeout_s, cantera_version), plog_validation_identity
    )


def checkpoint_fingerprint(
    mechanism_sha256: str, row: dict[str, Any], simulation_lock_identity: str, plog_validation_identity: str
) -> str:
    """Identify a row using its recorded lock, including historical saved runs."""
    payload = json.dumps(
        {
            "mechanism_semantic_sha256": mechanism_sha256,
            "plog_validation_sha256": plog_validation_identity,
            "simulation_lock_sha256": simulation_lock_identity,
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
            validate_row_refusal(result.get("reason"), simulation_lock(row_timeout_s, cantera_version))
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
    if not math.isfinite(row_timeout_s) or row_timeout_s <= 0:
        raise ValueError("row wall ceiling must be finite and positive")
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
            for process, task in list(active.items()):
                elapsed = time.monotonic() - float(task["started"])
                row = task["row"]
                result_path = task["result_path"]
                progress_path = task["progress_path"]
                fault_path = task["fault_path"]
                if process.is_alive() and elapsed >= row_timeout_s:
                    # The outer exception handler stops every active child. Keep
                    # diagnostic sidecars, but never checkpoint a timed-out row.
                    raise RunLevelSimulationError(
                        {
                            "refusal": "row_wall_ceiling_exceeded",
                            "construction_error": f"row {row['row_id']} exceeded wall ceiling {row_timeout_s:g} s",
                            "wall_time_s": elapsed,
                        }
                    )
                if not process.is_alive():
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
_OBSERVABLE_SIMULATORS["idt"] = _simulate_idt_row


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
    locked = simulation_lock(float(request.get("row_timeout_s", DEFAULT_ROW_WALL_CEILING_S)), cantera_version)
    response: dict[str, Any] = {
        "schema_version": 1,
        "status": "refused",
        "rows": [],
        "converter": {},
        "thread_env": {key: os.environ[key] for key in THREAD_ENV},
        "row_process_start_method": ROW_PROCESS_START_METHOD,
        "load_average_at_start": list(os.getloadavg()),
        "simulation_lock": locked,
        "simulation_lock_sha256": simulation_lock_sha256(
            float(request.get("row_timeout_s", DEFAULT_ROW_WALL_CEILING_S)), cantera_version
        ),
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
    requested_submechanism = request.get("excited_state_submechanism_sha256")
    if requested_submechanism is not None:
        try:
            if observable != "idt" or requested_submechanism != FFCM1_EXCITED_SHA256:
                raise ValueError("PIN CHANGED: unknown or incompatible excited-state model request")
            mechanism, response["excited_state_submechanism"] = append_excited_submechanism(mechanism)
        except ValueError as exc:
            response["refusal"] = "excited_state_submechanism_refused"
            response["validation_error"] = str(exc)
            _write_response(response_path, response)
            return 0
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
            validated_gas = ct.Solution(trimmed_path)
    except Exception as exc:
        response["refusal"] = "trimmed_mechanism_validation_failed"
        response["validation_error"] = str(exc)
        _write_response(response_path, response)
        return 0
    response["validation_warnings"] = [str(item.message) for item in caught]
    thermo_sources: tuple[bytes, ...] = ()
    if any(_REVIEWED_NASA_WARNING.fullmatch(message.strip()) for message in response["validation_warnings"]):
        thermo_sources = (Path(request["mechanism_path"]).read_bytes(), Path(request["thermo_path"]).read_bytes())
    unexpected = cantera_validation_refusals(
        response["validation_warnings"],
        thermo_sources,
        holdout_max_temperature_k=(
            max((float(row["temperature_k"]) for row in request["rows"]), default=None) if observable == "jsr" else None
        ),
    )
    if unexpected:
        response["refusal"] = "unallowlisted_cantera_validation_warning"
        _write_response(response_path, response)
        return 0

    affected_pressures = {item["pressure_pa"] for item in pressure_refusals}
    initial_results: dict[str, dict[str, Any]] = {}
    for row in request["rows"]:
        target = row.get("ignition_target")
        if (
            observable == "idt"
            and target not in ("pressure", "temperature")
            and resolve_ignition_target(target, validated_gas.species_names) is None
        ):
            initial_results[row["row_id"]] = {
                "row_id": row["row_id"],
                "status": "refused",
                "reason": "target_species_absent",
                "ignition_target": target,
            }
        elif float(row["pressure_pa"]) in affected_pressures:
            initial_results[row["row_id"]] = {
                "row_id": row["row_id"],
                "status": "refused",
                "reason": "mechanism_rate_unsound_at_condition",
            }
    requested_workers = int(request.get("workers", max(1, (os.cpu_count() or 1) - 2)))
    response["configured_workers"] = requested_workers
    response["row_workers"] = min(requested_workers, len(request["rows"])) if request["rows"] else 0
    response["row_timeout_s"] = float(request.get("row_timeout_s", DEFAULT_ROW_WALL_CEILING_S))
    checkpoint_path = Path(request.get("checkpoint_path", response_path.parent / "rows.jsonl"))
    response["checkpoint_path"] = str(checkpoint_path)
    checkpoint_recoveries: list[dict[str, Any]] = []
    response["checkpoint_recoveries"] = checkpoint_recoveries
    execution_rows = request["rows"]
    if observable == "idt" and trims:
        execution_rows = [
            {**row, "plog_trimmed_brackets_pa": [trim["kept_bracket_pa"] for trim in trims]} for row in execution_rows
        ]
    try:
        results, resumed = execute_rows(
            execution_rows,
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
