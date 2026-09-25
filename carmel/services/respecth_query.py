# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Load every pinned ReSpecTh ignition-delay record and filter them by condition window.

Backs ``carmel data find --kind idt``. Loading goes through the pinned cache
(:func:`carmel.services.respecth_archive.fetch_archive`) and the native parser
(:func:`carmel.services.respecth.parse_idt_record`); a member the parser refuses is COUNTED
by reason, never silently dropped and never partially listed. Filtering compares exact
decimals after converting each point's temperature to K and pressure to bar through the
same conversion table the record was bound under.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from carmel.schemas.datasets import Absent, ComponentRole, Coordinate, MeasuredValue
from carmel.services import units
from carmel.services.respecth import RespecthIdtRecord, RespecthRefusal, RespecthRefusalReason, parse_idt_record
from carmel.services.respecth_archive import RespecthManifest, fetch_archive, iter_xml_members

__all__ = [
    "ConditionWindow",
    "IdtLoadResult",
    "IdtMatch",
    "find_idt",
    "load_idt_records",
    "parse_window",
]

_PA_PER_BAR = Decimal(100000)


@dataclass(frozen=True)
class ConditionWindow:
    """A closed interval ``[low, high]``."""

    low: Decimal
    high: Decimal

    def contains(self, value: Decimal) -> bool:
        return self.low <= value <= self.high


def parse_window(text: str) -> ConditionWindow:
    """Parse ``"LOW:HIGH"`` into a :class:`ConditionWindow`.

    Raises:
        ValueError: Not two decimals separated by one colon, or ``LOW > HIGH``.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"expected LOW:HIGH, got {text!r}")
    try:
        low, high = (Decimal(part.strip()) for part in parts)
    except ArithmeticError:
        raise ValueError(f"expected two numbers in LOW:HIGH, got {text!r}") from None
    if not (low.is_finite() and high.is_finite()) or low > high:
        raise ValueError(f"expected finite LOW <= HIGH, got {text!r}")
    return ConditionWindow(low=low, high=high)


@dataclass(frozen=True)
class IdtLoadResult:
    """Every mapped record, plus how many IDT members were refused and why."""

    records: tuple[RespecthIdtRecord, ...]
    refusals: Counter[str]


def load_idt_records(manifest: RespecthManifest, *, cache_root: Path, download: bool = True) -> IdtLoadResult:
    """Parse every ignition-delay member of every pinned archive.

    Members of another experiment type are skipped (they are not IDT records); every IDT
    member either maps or is counted under its refusal reason.

    Raises:
        RespecthError: An archive failed to fetch or verify -- the whole load is refused,
            since a partial listing would read as a complete one.
    """
    records: list[RespecthIdtRecord] = []
    refusals: Counter[str] = Counter()
    for archive in manifest.archives:
        data = fetch_archive(archive, manifest=manifest, cache_root=cache_root, download=download)
        for member_path, member_bytes in iter_xml_members(data):
            try:
                records.append(parse_idt_record(member_bytes, archive, member_path))
            except RespecthRefusal as exc:
                if exc.reason is not RespecthRefusalReason.NOT_IGNITION_DELAY:
                    refusals[exc.reason.value] += 1
    return IdtLoadResult(records=tuple(records), refusals=refusals)


@dataclass(frozen=True)
class IdtMatch:
    """One record with at least one point inside every requested window."""

    record: RespecthIdtRecord
    fuels: tuple[str, ...]
    matched_points: int
    total_points: int
    temperature_k: tuple[Decimal, Decimal]
    pressure_bar: tuple[Decimal, Decimal]


def _in_base_unit(value: MeasuredValue, base_unit: str) -> Decimal:
    table = units.table_for_sha(value.conversion_table_sha256)
    converted = units.convert(
        value.canonical_decimal_value,
        quantity=value.quantity_kind,
        from_unit=value.unit_normalized,
        to_unit=base_unit,
        table=table,
    )
    return Decimal(converted.exact)


def _point_conditions(record: RespecthIdtRecord) -> list[tuple[Decimal, Decimal]]:
    """``(T in K, P in bar)`` for every point, constants filled in."""
    series = record.envelope.series[0]
    constants = {constant.axis_id: constant for constant in series.constants}
    conditions: list[tuple[Decimal, Decimal]] = []
    for point in series.points:
        by_axis: dict[str, Coordinate] = {**constants, **{c.axis_id: c for c in point.coordinates}}
        temperature = _in_base_unit(by_axis["temperature"].value, "K")
        pressure = _in_base_unit(by_axis["pressure"].value, "Pa") / _PA_PER_BAR
        conditions.append((temperature, pressure))
    return conditions


def _fuels(record: RespecthIdtRecord) -> tuple[str, ...]:
    composition = record.envelope.composition
    if isinstance(composition, Absent):
        return ()
    return tuple(c.species_raw_name for c in composition.components if c.role is ComponentRole.FUEL)


def find_idt(
    records: tuple[RespecthIdtRecord, ...],
    *,
    fuel: str | None = None,
    temperature_k: ConditionWindow | None = None,
    pressure_bar: ConditionWindow | None = None,
) -> list[IdtMatch]:
    """Records whose mixture has ``fuel`` as a fuel component and with at least one point
    inside both windows, sorted by the record's ReSpecTh DOI."""
    matches: list[IdtMatch] = []
    for record in records:
        fuels = _fuels(record)
        if fuel is not None and fuel not in fuels:
            continue
        conditions = _point_conditions(record)
        matched = [
            (t, p)
            for t, p in conditions
            if (temperature_k is None or temperature_k.contains(t))
            and (pressure_bar is None or pressure_bar.contains(p))
        ]
        if not matched:
            continue
        temperatures = [t for t, _ in conditions]
        pressures = [p for _, p in conditions]
        matches.append(
            IdtMatch(
                record=record,
                fuels=fuels,
                matched_points=len(matched),
                total_points=len(conditions),
                temperature_k=(min(temperatures), max(temperatures)),
                pressure_bar=(min(pressures), max(pressures)),
            )
        )
    return sorted(matches, key=lambda match: match.record.citation_doi)
