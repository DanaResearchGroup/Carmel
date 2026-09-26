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


@dataclass(frozen=True)
class ChemkedLoadResult:
    records: tuple[ChemkedIdtRecord, ...]
    refusals: Counter[str]


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


def find_idt(
    records: tuple[ChemkedIdtRecord, ...],
    *,
    fuel: str | None,
    temperature_k: ConditionWindow | None,
    pressure_bar: ConditionWindow | None,
) -> list[ChemkedIdtRecord]:
    result: list[ChemkedIdtRecord] = []
    for record in records:
        if fuel and fuel not in record.fuels:
            continue
        conditions: list[tuple[Decimal, Decimal]] = []
        for point in record.envelope.series[0].points:
            values = {coordinate.axis_id: coordinate.value for coordinate in point.coordinates}
            temperature = Decimal(
                units.convert(
                    values["temperature"].canonical_decimal_value,
                    quantity=values["temperature"].quantity_kind,
                    from_unit=values["temperature"].unit_normalized,
                    to_unit="K",
                    table=units.TABLE_V3,
                ).exact
            )
            pressure = Decimal(
                units.convert(
                    values["pressure"].canonical_decimal_value,
                    quantity=values["pressure"].quantity_kind,
                    from_unit=values["pressure"].unit_normalized,
                    to_unit="Pa",
                    table=units.TABLE_V3,
                ).exact
            ) / Decimal(100000)
            conditions.append((temperature, pressure))
        if any(
            (temperature_k is None or temperature_k.contains(t)) and (pressure_bar is None or pressure_bar.contains(p))
            for t, p in conditions
        ):
            result.append(record)
    return sorted(result, key=lambda item: item.citation_doi)
