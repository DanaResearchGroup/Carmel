"""Replay-preserving curated IDT selection and T3 version-1 YAML export."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml

from carmel.schemas.datasets import (
    Absent,
    ComponentRole,
    Composition,
    MeasuredValue,
    Uncertainty,
    UncertaintyBasis,
    YamlPathLocator,
)
from carmel.services import artifacts, chem, units
from carmel.services.archive_unpack import _is_absolute_member_name
from carmel.services.chemked import ChemkedIdtRecord, ChemkedRefusal, parse_idt_record, replay_idt_record
from carmel.services.chemked_archive import fetch_file, load_manifest
from carmel.services.rcm_history import HistoryRefusal, RcmHistory, RcmState, in_si_decimal
from carmel.services.rcm_thermo import check_initial_temperature, isentropic_eoc, solver_composition
from carmel.services.respecth import (
    IgnitionCriterion,
    IgnitionTarget,
    RespecthIdtRecord,
    RespecthRefusal,
    _check_plausible_kelvin,
    _linked_points,
)
from carmel.services.respecth import parse_idt_record as parse_respecth
from carmel.services.respecth import replay_idt_record as replay_respecth
from carmel.services.respecth_archive import fetch_archive, iter_xml_members
from carmel.services.respecth_archive import load_manifest as respecth_manifest

IdtRecord = ChemkedIdtRecord | RespecthIdtRecord

_EXISTING_SOURCE_SUM_TOLERANCE = Decimal("0.000001")
_ROUNDING_SOURCE_SUM_TOLERANCE = Decimal("0.00001")


class ExportReason(StrEnum):
    NO_SMILES = "no_confident_smiles"
    IGNITION_DEFINITION = "inexpressible_ignition_definition"
    COMPOSITION = "invalid_composition"
    THERMO = "rcm_thermo_unavailable"
    IMPLAUSIBLE_TEMPERATURE = "implausible_ignition_temperature"
    STATE = "missing_stated_state"
    T3_CONSTRAINT = "t3_constraint"
    UNSAFE_LOCATOR = "unsafe_source_locator"


class ExportRefusal(ValueError):
    def __init__(self, reason: ExportReason, detail: str):
        self.reason = reason
        super().__init__(detail)


def record_path(record: IdtRecord) -> str:
    return record.path if isinstance(record, ChemkedIdtRecord) else record.archive.member_path


def record_source(record: IdtRecord) -> str:
    return "chemked" if isinstance(record, ChemkedIdtRecord) else "respecth"


def _selected(record: IdtRecord, fuel: str | None, studies: tuple[str, ...]) -> bool:
    path = record_path(record)
    identifiers = {path, PurePosixPath(path).name, record.citation_doi}
    if studies and not identifiers.intersection(studies):
        return False
    if fuel is None:
        return True
    compositions = [record.envelope.composition, *(point.composition for point in record.envelope.series[0].points)]
    fuels = {
        component.species_raw_name
        for composition in compositions
        if isinstance(composition, Composition)
        for component in composition.components
        if component.role is ComponentRole.FUEL
    }
    return fuel in fuels or (isinstance(record, ChemkedIdtRecord) and path.split("/")[0] == fuel)


def point_history(record: IdtRecord, index: int) -> tuple[RcmHistory | None, RcmState | None]:
    if isinstance(record, ChemkedIdtRecord):
        return (record.rcm_histories[index], record.rcm_states[index]) if record.rcm_histories else (None, None)
    if not record.rcm_states:
        return None, None
    assert not isinstance(record.rcm_conditions, Absent)
    history = next(
        item.history
        for item in record.rcm_conditions.histories
        if index + 1 in _linked_points(item.point_link.raw, len(record.envelope.series[0].points))
    )
    return history, record.rcm_states[index]


def _float_value(exact: Decimal, reason: ExportReason, detail: str) -> float:
    """Preserve source sign/zero and refuse nonzero float underflow or overflow."""
    if exact <= 0:
        raise ExportRefusal(reason, detail)
    value = float(exact)
    if not math.isfinite(value) or value == 0:
        raise ExportRefusal(reason, detail)
    return value


def _base_decimal(value: MeasuredValue, reason: ExportReason) -> Decimal:
    """Keep conversion failures at the same per-point boundary as admission."""
    try:
        return in_si_decimal(value)
    except Exception as exc:
        raise ExportRefusal(reason, f"cannot convert {value.quantity_kind.value} to its base unit: {exc}") from exc


def _composition(record: IdtRecord, index: int) -> tuple[list[dict[str, Any]], dict[str, Decimal], dict[str, Any]]:
    point = record.envelope.series[0].points[index]
    composition = point.composition if isinstance(point.composition, Composition) else record.envelope.composition
    if not isinstance(composition, Composition):
        raise ExportRefusal(ExportReason.COMPOSITION, "no resolved mole-fraction composition")
    material: list[tuple[str, str, Decimal]] = []
    exact_total = Decimal(0)
    identity_lookups: set[tuple[str, str, str]] = set()
    identifiers = record.species_identifiers
    if isinstance(record, ChemkedIdtRecord):
        # parse_idt_record emits one series point per YAML row, in source order.
        prefix = f"datapoints[{index}].composition.species["
        identifiers = tuple(
            item
            for item in identifiers
            if isinstance(item[2].ref.locator, YamlPathLocator) and item[2].ref.locator.path.startswith(prefix)
        )
    for component in composition.components:
        exact = _base_decimal(component.amount, ExportReason.COMPOSITION)
        if exact.is_zero():
            continue  # T3 requires positive entries; exact zero carries no material.
        if exact > 1:
            raise ExportRefusal(ExportReason.COMPOSITION, "source mole fractions must not exceed one")
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            exact_total += exact
        smiles_set = set()
        for name, kind, identifier in identifiers:
            if name != component.species_raw_name:
                continue
            if kind == "smiles":
                smiles = chem.canonical_smiles(identifier.raw)
            elif chem.INCHIKEY_PATTERN.fullmatch(identifier.raw):
                inchi = chem.inchi_from_inchikey(identifier.raw)
                smiles = chem.smiles_from_inchi(inchi) if inchi is not None else None
                if smiles is not None and inchi is not None:
                    identity_lookups.add((component.species_raw_name, identifier.raw, inchi))
            else:
                smiles = chem.smiles_from_inchi(identifier.raw)
            if smiles:
                smiles_set.add(smiles)
        if len(smiles_set) != 1:
            raise ExportRefusal(
                ExportReason.NO_SMILES, f"{component.species_raw_name}: absent or conflicting source identity"
            )
        (smiles,) = smiles_set
        if any(existing == smiles for existing, _, _ in material):
            raise ExportRefusal(ExportReason.COMPOSITION, "positive unique SMILES mole fractions required")
        material.append((smiles, component.species_raw_name, exact))
    with localcontext() as ctx:
        ctx.prec = units._CONVERT_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        source_deviation = abs(exact_total - Decimal(1))
    if not material or source_deviation > _ROUNDING_SOURCE_SUM_TOLERANCE:
        raise ExportRefusal(ExportReason.COMPOSITION, "mole fractions must sum to one")
    renormalized = source_deviation > _EXISTING_SOURCE_SUM_TOLERANCE
    exact_amounts = [amount for _, _, amount in material]
    if renormalized:
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            exact_amounts = [amount / exact_total for amount in exact_amounts]
            exact_amounts[-1] += Decimal(1) - sum(exact_amounts, Decimal(0))
    entries: dict[str, float] = {}
    thermo: dict[str, Decimal] = {}
    for (smiles, species, _), exact in zip(material, exact_amounts, strict=True):
        entries[smiles] = _float_value(
            exact, ExportReason.COMPOSITION, "source fraction is not representable as a positive T3 fraction"
        )
        thermo[species] = exact
    # A separate transport check, after source admission: T3 consumes these floats.
    with localcontext() as ctx:
        ctx.prec = units._CONVERT_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        represented_total = Decimal.from_float(sum(entries.values()))
        outside_t3_tolerance = abs(represented_total - Decimal(1)) > Decimal.from_float(1e-6)
    if outside_t3_tolerance:
        raise ExportRefusal(ExportReason.COMPOSITION, "T3 fraction representation must sum to one")
    provenance: dict[str, Any] = {}
    if renormalized:
        provenance["composition"] = {"source_total": format(exact_total, "f"), "renormalized": True}
    if identity_lookups:
        provenance["identity_lookups"] = [
            {"species": species, "inchikey": inchikey, "inchi": inchi}
            for species, inchikey, inchi in sorted(identity_lookups)
        ]
    return [{"smiles": key, "mole_fraction": value} for key, value in sorted(entries.items())], thermo, provenance


def _quantity(value: MeasuredValue, units: str) -> dict[str, Any]:
    return {
        "value": _float_value(
            _base_decimal(value, ExportReason.STATE), ExportReason.STATE, "stated T/P must be positive finite T3 values"
        ),
        "units": units,
    }


def _point(record: IdtRecord, index: int, include_derived_labels: bool) -> tuple[dict[str, Any], str]:
    path = record_path(record)
    if "\\" in path or _is_absolute_member_name(path) or ".." in PurePosixPath(path).parts:
        raise ExportRefusal(ExportReason.UNSAFE_LOCATOR, "source file locator must be a contained relative member name")
    ignition = record.ignition
    if ignition.target is IgnitionTarget.OHEX or ignition.criterion is IgnitionCriterion.RELATIVE_CONCENTRATION:
        raise ExportRefusal(ExportReason.IGNITION_DEFINITION, f"{ignition.target.value}/{ignition.criterion.value}")
    composition, thermo, source_provenance = _composition(record, index)
    series = record.envelope.series[0]
    point = series.points[index]
    values = {c.axis_id: c.value for c in (*series.constants, *point.coordinates)}
    observation = point.observations[0]
    if isinstance(observation.value, Absent):
        raise ExportRefusal(ExportReason.T3_CONSTRAINT, "ignition delay is absent")
    idt_exact = _base_decimal(observation.value, ExportReason.T3_CONSTRAINT)
    if idt_exact > 10:
        raise ExportRefusal(ExportReason.T3_CONSTRAINT, "T3 requires a positive ignition delay no greater than 10 s")
    idt = _float_value(idt_exact, ExportReason.T3_CONSTRAINT, "T3 requires a positive finite ignition delay")
    sha = record.sha256 if isinstance(record, ChemkedIdtRecord) else record.member_sha256
    locator = f"{record_source(record)}:{path};datapoint={index};sha256={sha}"
    if isinstance(record, RespecthIdtRecord):
        locator += f";archive={record.archive.archive_name};archive_sha256={record.archive.archive_sha256}"
    apparatus = (
        "rapid compression machine"
        if (isinstance(record, RespecthIdtRecord) and record.apparatus.device_class.value == "rcm")
        else "shock tube"
    )
    if isinstance(record, ChemkedIdtRecord):
        apparatus = str(record.apparatus)
    output: dict[str, Any] = {
        "composition": composition,
        "apparatus": apparatus,
        "ignition_definition": {
            "target": "pressure" if ignition.target is IgnitionTarget.PRESSURE else ignition.target.value,
            "type": ignition.criterion.value,
        },
        "idt": {"value": idt, "units": "s"},
        "source": {"doi": record.citation_doi, "record": locator},
    }
    history, state = point_history(record, index)
    basis = "stated"
    if history is not None:
        assert state is not None  # Record validators require paired histories/states.
        times, volumes = history.times, history.volumes
        first, last = (
            _base_decimal(history.time[0], ExportReason.T3_CONSTRAINT),
            _base_decimal(history.time[-1], ExportReason.T3_CONSTRAINT),
        )
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            duration = last - first
            exact_times = tuple(_base_decimal(value, ExportReason.T3_CONSTRAINT) for value in history.time)
            exact_volumes = tuple(_base_decimal(value, ExportReason.T3_CONSTRAINT) for value in history.volume)
            slopes = tuple(
                (b - a) / (t2 - t1)
                for a, b, t1, t2 in zip(exact_volumes, exact_volumes[1:], exact_times, exact_times[1:], strict=False)
            )
        if duration > 10 or any(not math.isfinite(float(slope)) for slope in slopes):
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "history exceeds T3's duration or finite-slope bound")
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            represented_times = tuple(Decimal.from_float(value) for value in times)
            represented_volumes = tuple(Decimal.from_float(value) for value in volumes)
            eoc = Decimal.from_float(history.compression_time)
            index = bisect_left(represented_times, eoc)
            end_volume = represented_volumes[index]
            if represented_times[index] != eoc:
                fraction = (eoc - represented_times[index - 1]) / (
                    represented_times[index] - represented_times[index - 1]
                )
                end_volume = (1 - fraction) * represented_volumes[index - 1] + fraction * end_volume
            represented_ratio = represented_volumes[0] / end_volume
            represented_duration = represented_times[-1] - represented_times[0]
            represented_slopes = tuple(
                (b - a) / (t2 - t1)
                for a, b, t1, t2 in zip(
                    represented_volumes, represented_volumes[1:], represented_times, represented_times[1:], strict=False
                )
            )
        if represented_duration > 10 or any(not math.isfinite(float(slope)) for slope in represented_slopes):
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "float history representation exceeds T3's bounds")
        if (
            history.compression_time_stated is not None
            and not first <= _base_decimal(history.compression_time_stated, ExportReason.T3_CONSTRAINT) <= last
        ):
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "stated compression time is outside the source history")
        if history.compression_time_decimal not in exact_times and history.compression_time in times:
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "EOC collapses onto a sample in float representation")
        extrapolated = False
        if state.eoc_basis != "stated" and (include_derived_labels or isinstance(record, RespecthIdtRecord)):
            try:
                extrapolated = check_initial_temperature(
                    _base_decimal(state.initial_temperature, ExportReason.THERMO),
                    thermo,
                    volume_ratio=history.volume_ratio_decimal,
                )
            except ValueError as exc:
                raise ExportRefusal(ExportReason.THERMO, str(exc)) from exc
        if represented_ratio <= 1:
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "float history must describe compression")
        output.update(
            initial_temperature=_quantity(state.initial_temperature, "K"),
            initial_pressure=_quantity(state.initial_pressure, "Pa"),
            volume_history={
                "time": {"values": list(times), "units": "s"},
                "volume": {"values": list(volumes), "units": "m3"},
                "compression_time": {"value": history.compression_time, "units": "s"},
            },
        )
        output["source"]["record"] += ";compression_time=" + (
            "derived-minimum" if history.compression_time_derived else "stated"
        )
        if state.eoc_basis != "stated":
            basis = "derived" if include_derived_labels else "unlabelled"
            if include_derived_labels:
                try:
                    temperature, pressure = isentropic_eoc(
                        float(_base_decimal(state.initial_temperature, ExportReason.THERMO)),
                        float(_base_decimal(state.initial_pressure, ExportReason.THERMO)),
                        history.volume_ratio,
                        solver_composition(thermo),
                    )
                except HistoryRefusal as exc:
                    raise ExportRefusal(ExportReason.T3_CONSTRAINT, str(exc)) from exc
                except ValueError as exc:
                    raise ExportRefusal(ExportReason.THERMO, str(exc)) from exc
                try:
                    _check_plausible_kelvin(temperature, f"derived {temperature} K")
                except RespecthRefusal as exc:
                    raise ExportRefusal(ExportReason.IMPLAUSIBLE_TEMPERATURE, exc.detail) from exc
                output.update(
                    temperature={"value": temperature, "units": "K"}, pressure={"value": pressure, "units": "Pa"}
                )
                output["source"]["record"] += ";eoc=derived-isentropic"
            elif extrapolated:
                output["source"]["record"] += ";eoc=derived-isentropic"
            if extrapolated:
                output["source"]["record"] += ";thermo=extrapolated-below-300K"
        else:
            # RcmState validates that stated labels contain both quantities.
            assert state.temperature is not None and state.pressure is not None
            output.update(temperature=_quantity(state.temperature, "K"), pressure=_quantity(state.pressure, "Pa"))
    else:
        output.update(temperature=_quantity(values["temperature"], "K"), pressure=_quantity(values["pressure"], "Pa"))
    uncertainty = observation.uncertainty
    if isinstance(uncertainty, Uncertainty):
        # Preserve asymmetric uncertainty as a refusal: T3 has only a scalar.
        if (
            not isinstance(uncertainty.upper, MeasuredValue)
            or not isinstance(uncertainty.lower, MeasuredValue)
            or isinstance(uncertainty.basis, Absent)
        ):
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "T3 requires quantified uncertainty")
        lower, upper = (
            _base_decimal(uncertainty.lower, ExportReason.T3_CONSTRAINT),
            _base_decimal(uncertainty.upper, ExportReason.T3_CONSTRAINT),
        )
        if lower != upper:
            raise ExportRefusal(ExportReason.T3_CONSTRAINT, "T3 cannot express asymmetric IDT uncertainty")
        detail = "nonzero uncertainty must be representable as a positive finite T3 value"
        _float_value(upper, ExportReason.T3_CONSTRAINT, detail)
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            exact_error = upper * idt_exact if uncertainty.basis is UncertaintyBasis.RELATIVE else upper
        error = _float_value(exact_error, ExportReason.T3_CONSTRAINT, detail)
        output["uncertainty"] = {"value": error, "units": "s"}
    if composition_provenance := source_provenance.get("composition"):
        output["source"]["record"] += f";composition=renormalized;source_total={composition_provenance['source_total']}"
    for lookup in source_provenance.get("identity_lookups", []):
        species = quote(lookup["species"], safe="")
        output["source"]["record"] += f";identity=inchikey:{species}:{lookup['inchikey']}"
    return output, basis


def export_idt(
    records: Iterable[IdtRecord],
    *,
    fuel: str | None = None,
    studies: tuple[str, ...] = (),
    history_only: bool = False,
    include_derived_labels: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic T3 YAML data and a per-point refusal report.

    History points omit derived end-of-compression labels by default. Callers
    can opt in to explicitly marked labels; source-stated labels are always kept.
    """
    records = tuple(records)
    available = {
        identifier
        for record in records
        for identifier in (record_path(record), PurePosixPath(record_path(record)).name, record.citation_doi)
    }
    if set(studies) - available:
        raise ValueError(f"requested studies/files are not mapped: {sorted(set(studies) - available)}")
    points = []
    failures = []
    reasons: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda r: (record_source(r), record_path(r))):
        if not _selected(record, fuel, studies):
            continue
        for index in range(len(record.envelope.series[0].points)):
            if history_only and point_history(record, index)[0] is None:
                continue
            try:
                point, basis = _point(record, index, include_derived_labels)
            except ExportRefusal as exc:
                reasons[exc.reason.value] += 1
                failures.append(
                    {
                        "source": record_source(record),
                        "file": record_path(record),
                        "datapoint": index,
                        "reason": exc.reason.value,
                        "detail": str(exc),
                    }
                )
            else:
                points.append(point)
                counts[basis] += 1
    return {"version": 1, "points": points}, {
        "exported": len(points),
        "stated": counts["stated"],
        "derived": counts["derived"],
        "unlabelled": counts["unlabelled"],
        "refused": dict(sorted(reasons.items())),
        "refusals": failures,
    }


def load_records(
    *, source: str, cache_root: Path, download: bool, fuel: str | None = None
) -> tuple[tuple[IdtRecord, ...], dict[str, int]]:
    """Count parser refusals, but abort the whole load on any source replay failure."""
    if source not in {"chemked", "respecth", "all"}:
        raise ValueError("source must be chemked, respecth or all")
    records: list[IdtRecord] = []
    refusals: Counter[str] = Counter()
    if source in {"chemked", "all"}:
        manifest = load_manifest()
        fuel_directories = {item.path.split("/")[0] for item in manifest.files}
        for item in manifest.files:
            if fuel in fuel_directories and item.path.split("/")[0] != fuel:
                continue
            raw = fetch_file(item, manifest, cache_root, download=download)
            try:
                record = parse_idt_record(raw, item.path, item.sha256)
            except ChemkedRefusal as exc:
                refusals["chemked:" + exc.reason.value] += 1
                continue
            replay_idt_record(record, raw)
            records.append(record)
    if source in {"respecth", "all"}:
        respecth = respecth_manifest()
        for archive in respecth.archives:
            blob = fetch_archive(archive, manifest=respecth, cache_root=cache_root, download=download)
            for member, raw in iter_xml_members(blob):
                try:
                    mapped = parse_respecth(raw, archive, member)
                except RespecthRefusal as exc:
                    if exc.reason.value != "not_ignition_delay":
                        refusals["respecth:" + exc.reason.value] += 1
                    continue
                if not replay_respecth(mapped, raw).verified:
                    raise ValueError(f"source replay failed for {member}")
                records.append(mapped)
    return tuple(records), dict(sorted(refusals.items()))


def write_export(payload: dict[str, Any], report: dict[str, Any], output: Path) -> Path:
    report_path = output.with_suffix(".report.json")
    if output.name.endswith(".report.json"):
        raise ValueError("output must not use the report's .report.json suffix")
    artifacts.write_text(output, yaml.safe_dump(payload, sort_keys=False))
    artifacts.write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path
