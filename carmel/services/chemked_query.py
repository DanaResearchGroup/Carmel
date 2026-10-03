# ruff: noqa: E501
"""ChemKED loading and condition filtering, sharing ReSpecTh query semantics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from carmel.services import units
from carmel.services.chemked import ChemkedIdtRecord, ChemkedRefusal, parse_idt_record
from carmel.services.chemked_archive import ChemkedManifest, fetch_file
from carmel.services.respecth_query import ConditionWindow

__all__ = [
    "ChemkedIdtMatch",
    "ChemkedLoadResult",
    "find_idt",
    "load_idt_records",
    "point_conditions",
]


@dataclass(frozen=True)
class ChemkedLoadResult:
    records: tuple[ChemkedIdtRecord, ...]
    refusals: Counter[str]


@dataclass(frozen=True)
class ChemkedIdtMatch:
    """One ChemKED record with at least one point inside all requested windows."""

    record: ChemkedIdtRecord
    matched_points: int
    total_points: int
    temperature_k: tuple[Decimal, Decimal]
    pressure_bar: tuple[Decimal, Decimal]
    conditions: tuple[tuple[Decimal, Decimal], ...]


def load_idt_records(manifest: ChemkedManifest, *, cache_root: Path, download: bool = True) -> ChemkedLoadResult:
    records: list[ChemkedIdtRecord] = []
    refusals: Counter[str] = Counter()
    for item in manifest.files:
        try:
            records.append(
                parse_idt_record(fetch_file(item, manifest, cache_root, download=download), item.path, item.sha256)
            )
        except ChemkedRefusal as exc:
            refusals[exc.reason.value] += 1
    return ChemkedLoadResult(tuple(records), refusals)


def point_conditions(record: ChemkedIdtRecord) -> tuple[tuple[Decimal, Decimal], ...]:
    """Return every point as exact ``(temperature K, pressure bar)`` decimals."""
    conditions: list[tuple[Decimal, Decimal]] = []
    for index, point in enumerate(record.envelope.series[0].points):
        values = {coordinate.axis_id: coordinate.value for coordinate in point.coordinates}
        state = record.rcm_states[index] if record.rcm_states else None
        if state is not None and state.eoc_basis != "stated":
            continue
        temperature_value = state.temperature if state is not None else values["temperature"]
        pressure_value = state.pressure if state is not None else values["pressure"]
        assert temperature_value is not None and pressure_value is not None
        temperature = Decimal(
            units.convert(
                temperature_value.canonical_decimal_value,
                quantity=temperature_value.quantity_kind,
                from_unit=temperature_value.unit_normalized,
                to_unit="K",
                table=units.table_for_sha(temperature_value.conversion_table_sha256),
            ).exact
        )
        pressure = Decimal(
            units.convert(
                pressure_value.canonical_decimal_value,
                quantity=pressure_value.quantity_kind,
                from_unit=pressure_value.unit_normalized,
                to_unit="Pa",
                table=units.table_for_sha(pressure_value.conversion_table_sha256),
            ).exact
        ) / Decimal(100000)
        conditions.append((temperature, pressure))
    return tuple(conditions)


def find_idt(
    records: tuple[ChemkedIdtRecord, ...],
    *,
    fuel: str | None = None,
    temperature_k: ConditionWindow | None = None,
    pressure_bar: ConditionWindow | None = None,
) -> list[ChemkedIdtMatch]:
    result: list[ChemkedIdtMatch] = []
    for record in records:
        if fuel and fuel not in record.fuels:
            continue
        conditions = point_conditions(record)
        matched = tuple(
            (temperature, pressure)
            for temperature, pressure in conditions
            if (temperature_k is None or temperature_k.contains(temperature))
            and (pressure_bar is None or pressure_bar.contains(pressure))
        )
        if matched:
            temperatures = tuple(temperature for temperature, _ in conditions)
            pressures = tuple(pressure for _, pressure in conditions)
            result.append(
                ChemkedIdtMatch(
                    record=record,
                    matched_points=len(matched),
                    total_points=len(conditions),
                    temperature_k=(min(temperatures), max(temperatures)),
                    pressure_bar=(min(pressures), max(pressures)),
                    conditions=conditions,
                )
            )
    return sorted(result, key=lambda match: match.record.citation_doi)
