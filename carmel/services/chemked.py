# ruff: noqa: E501
"""Native, fail-closed ChemKED ignition-delay records.

ChemKED files are fetched as individually pinned raw YAML files. Values use a
dedicated backwards-compatible YAML key-path locator, and replay re-parses the
pinned bytes with :func:`yaml.safe_load` before following every path.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import yaml

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    MeasuredValue,
    Observation,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    ValueOrigin,
    YamlPathLocator,
)
from carmel.services import units
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.respecth import _measured
from carmel.services.units import QuantityKind

__all__ = ["ChemkedIdtRecord", "ChemkedRefusal", "ChemkedRefusalReason", "parse_idt_record", "replay_idt_record"]


class ChemkedRefusalReason(StrEnum):
    MALFORMED_YAML = "malformed_yaml"
    SCHEMA_REJECTED = "schema_rejected"
    UNKNOWN_EXPERIMENT_TYPE = "unknown_experiment_type"
    UNMAPPED_APPARATUS = "unmapped_apparatus"
    UNMAPPED_UNIT = "unmapped_unit"
    UNMAPPED_IGNITION_DEFINITION = "unmapped_ignition_definition"
    INCOMPLETE_RECORD = "incomplete_record"
    UNRESOLVABLE_PATH = "unresolvable_yaml_path"
    RCM_PRE_COMPRESSION_CONDITIONS = "rcm_pre_compression_conditions"


class ChemkedRefusal(ValueError):
    def __init__(self, reason: ChemkedRefusalReason, detail: str):
        super().__init__(f"{reason.value}: {detail}")
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class ChemkedIdtRecord:
    path: str
    sha256: str
    citation_doi: str
    fuels: tuple[str, ...]
    envelope: DatasetEnvelope


_PART = re.compile(r"([^\[.]+)(?:\[(\d+)\])?")


def yaml_value(document: object, path: str) -> object:
    """Follow a fully positional ``a[0].b`` path, refusing absent paths."""
    base: str
    marker: str | None
    if "#" in path:
        base, marker = path.rsplit("#", 1)
    else:
        base, marker = path, None
    value = document
    for part in base.split("."):
        match = _PART.fullmatch(part)
        if match is None or not isinstance(value, dict) or match.group(1) not in value:
            raise KeyError(path)
        value = value[match.group(1)]
        if match.group(2) is not None:
            index = int(match.group(2))
            if not isinstance(value, list) or index >= len(value):
                raise KeyError(path)
            value = value[index]
    if marker == "key":
        return base.rsplit(".", 1)[-1].split("[", 1)[0]
    if marker in {"value", "unit"}:
        if not isinstance(value, str) or " " not in value.strip():
            raise KeyError(path)
        return value.strip().split(None, 1)[0 if marker == "value" else 1]
    if marker is not None:
        raise KeyError(path)
    return value


def _ref(path: str) -> SourceRef:
    return SourceRef(node_id="record", locator=YamlPathLocator(path=path))


def _quantity(raw: object, path: str, quantity: QuantityKind, unit_path: str | None = None) -> MeasuredValue:
    if not isinstance(raw, str) or " " not in raw.strip():
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_UNIT, f"{path}: expected '<value> <unit>', got {raw!r}")
    value, unit = raw.strip().split(None, 1)
    try:
        return _measured(
            type("T", (), {"raw": value, "ref": _ref(path + "#value")})(),
            type("T", (), {"raw": unit, "ref": _ref((unit_path or path) + ("" if unit_path else "#unit"))})(),
            quantity,
            units.TABLE_V3,
        )
    except Exception as exc:
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_UNIT, f"{path}: {exc}") from exc


def _dimensionless(raw: object, path: str, kind_path: str) -> MeasuredValue:
    return _measured(
        type("T", (), {"raw": str(raw), "ref": _ref(path)})(),
        type("T", (), {"raw": "mole fraction", "ref": _ref(kind_path)})(),
        QuantityKind.MOLE_FRACTION,
        units.TABLE_V3,
    )


def parse_idt_record(raw_bytes: bytes, path: str, expected_sha256: str | None = None) -> ChemkedIdtRecord:
    actual = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: sha256 mismatch")
    try:
        doc = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ChemkedRefusal(ChemkedRefusalReason.MALFORMED_YAML, f"{path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ChemkedRefusal(ChemkedRefusalReason.MALFORMED_YAML, f"{path}: document is not a mapping")
    if doc.get("experiment-type") != "ignition delay":
        raise ChemkedRefusal(ChemkedRefusalReason.UNKNOWN_EXPERIMENT_TYPE, f"{path}: {doc.get('experiment-type')!r}")
    _validate_subset(doc, path)
    apparatus = doc.get("apparatus")
    if not isinstance(apparatus, dict) or apparatus.get("kind") not in {"shock tube", "rapid compression machine"}:
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_APPARATUS, f"{path}: {apparatus!r}")
    doi = doc.get("reference", {}).get("doi") if isinstance(doc.get("reference"), dict) else None
    if not isinstance(doi, str) or not doi:
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: missing reference DOI")
    rows = doc.get("datapoints")
    if not isinstance(rows, list) or not rows:
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: no datapoints")
    if apparatus["kind"] == "rapid compression machine":
        for row in rows:
            history = row.get("volume-history") if isinstance(row, dict) else None
            values = history.get("values") if isinstance(history, dict) else None
            if isinstance(values, list) and values:
                volumes = [pair[1] for pair in values if isinstance(pair, list) and len(pair) >= 2]
                if volumes and float(volumes[0]) != min(float(volume) for volume in volumes):
                    raise ChemkedRefusal(
                        ChemkedRefusalReason.RCM_PRE_COMPRESSION_CONDITIONS,
                        f"{path}: RCM volume history begins before its minimum volume",
                    )
    components: list[CompositionComponent] = []
    first = rows[0]
    composition = first.get("composition") if isinstance(first, dict) else None
    species = composition.get("species") if isinstance(composition, dict) else None
    if not isinstance(species, list) or not isinstance(composition, dict) or composition.get("kind") != "mole fraction":
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: unmapped composition")
    fuels: list[str] = []
    for index, item in enumerate(species):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("species-name"), str)
            or not isinstance(item.get("amount"), list)
        ):
            raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: malformed species")
        name = item["species-name"]
        role = (
            ComponentRole.OXIDIZER
            if name == "O2"
            else ComponentRole.DILUENT
            if name in {"N2", "Ar", "He"}
            else ComponentRole.FUEL
        )
        if role is ComponentRole.FUEL:
            fuels.append(name)
        components.append(
            CompositionComponent(
                species_raw_name=name,
                amount=_dimensionless(
                    item["amount"][0],
                    f"datapoints[0].composition.species[{index}].amount[0]",
                    "datapoints[0].composition.kind",
                ),
                role=role,
            )
        )
    axes = (
        AxisDeclaration(
            axis_id="ignition_delay",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.TIME,
            label_raw="ignition-delay",
            label_ref=_ref("datapoints[0].ignition-delay#key"),
        ),
        AxisDeclaration(
            axis_id="pressure",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.PRESSURE,
            label_raw="pressure",
            label_ref=_ref("datapoints[0].pressure#key"),
        ),
        AxisDeclaration(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_raw="temperature",
            label_ref=_ref("datapoints[0].temperature#key"),
        ),
    )
    points: list[DataPoint] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: point {index} is not a mapping")
        ignition = row.get("ignition-type") or doc.get("common-properties", {}).get("ignition-type")
        if (
            not isinstance(ignition, dict)
            or ignition.get("target") not in {"OH", "OH*", "p"}
            or ignition.get("type") not in {"d/dt max", "max"}
        ):
            raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION, f"{path}: point {index}")
        points.append(
            DataPoint(
                point_id=f"p{index + 1:04d}",
                coordinates=(
                    Coordinate(
                        axis_id="pressure",
                        value=_quantity(row["pressure"][0], f"datapoints[{index}].pressure[0]", QuantityKind.PRESSURE),
                        uncertainty=Absent(reason=AbsenceReason.UNKNOWN),
                    ),
                    Coordinate(
                        axis_id="temperature",
                        value=_quantity(
                            row["temperature"][0], f"datapoints[{index}].temperature[0]", QuantityKind.TEMPERATURE
                        ),
                        uncertainty=Absent(reason=AbsenceReason.UNKNOWN),
                    ),
                ),
                observations=(
                    Observation(
                        axis_id="ignition_delay",
                        value=_quantity(
                            row["ignition-delay"][0], f"datapoints[{index}].ignition-delay[0]", QuantityKind.TIME
                        ),
                        uncertainty=Absent(reason=AbsenceReason.UNKNOWN),
                    ),
                ),
                composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
            )
        )
    envelope = DatasetEnvelope(
        source_graph=SourceGraph(
            nodes=(
                SourceNode(
                    node_id="record",
                    kind=SourceNodeKind.DATABASE_RECORD,
                    sha256=actual,
                    parent_node_id=None,
                    origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                    extraction=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                    glyph_health=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                    verification=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                    crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                    document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                ),
            )
        ),
        composition=Composition(
            raw_name="initial composition",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.UNKNOWN),
            components=tuple(sorted(components, key=lambda x: x.species_raw_name)),
        ),
        series=(
            Series(
                series_id="ignition_delay",
                source_form=SourceForm.STRUCTURED_RECORD,
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=axes,
                constants=(),
                points=tuple(points),
                digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            ),
        ),
        conversion_tables=(
            EmbeddedConversionTable(
                sha256=units.TABLE_V3.sha256,
                canonical_json=canonical_json_bytes(units.TABLE_V3.identity_payload()).decode(),
            ),
        ),
        table_inventories=(),
        ooxml_table_inventories=(),
        figure_digitizations=(),
    )
    return ChemkedIdtRecord(path=path, sha256=actual, citation_doi=doi, fuels=tuple(fuels), envelope=envelope)


def _validate_subset(doc: dict[object, object], path: str) -> None:
    """Offline subset of the BSD-3 PyKED schema for the fields this lane maps.

    PyKED's high-level validator calls Crossref and ORCID. The committed suite
    must be network-free, so this validator deliberately checks only the
    schema facts this native mapper consumes.
    """
    authors = doc.get("file-authors")
    reference = doc.get("reference")
    rows = doc.get("datapoints")
    if not isinstance(authors, list) or not authors:
        raise ChemkedRefusal(ChemkedRefusalReason.SCHEMA_REJECTED, f"{path}: file-authors is required")
    if not isinstance(reference, dict) or not isinstance(reference.get("doi"), str):
        raise ChemkedRefusal(ChemkedRefusalReason.SCHEMA_REJECTED, f"{path}: reference DOI is required")
    if not isinstance(rows, list) or not rows:
        raise ChemkedRefusal(ChemkedRefusalReason.SCHEMA_REJECTED, f"{path}: datapoints is required")


def replay_idt_record(record: ChemkedIdtRecord, raw_bytes: bytes) -> None:
    """Raise a typed refusal unless every YAML locator and bytes pin replay."""
    if hashlib.sha256(raw_bytes).hexdigest() != record.sha256:
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{record.path}: sha256 mismatch on replay")
    doc = yaml.safe_load(raw_bytes)
    for ref, expected in _values_with_refs(record.envelope):
        if not isinstance(ref.locator, YamlPathLocator):
            continue
        try:
            actual = yaml_value(doc, ref.locator.path)
        except KeyError as exc:
            raise ChemkedRefusal(ChemkedRefusalReason.UNRESOLVABLE_PATH, f"{record.path}: {ref.locator.path}") from exc
        if str(actual) != expected:
            raise ChemkedRefusal(
                ChemkedRefusalReason.UNRESOLVABLE_PATH,
                f"{record.path}: {ref.locator.path} -> {actual!r}, expected {expected!r}",
            )


def _values_with_refs(envelope: DatasetEnvelope) -> Iterator[tuple[SourceRef, str]]:
    for series in envelope.series:
        for axis in series.axes:
            yield axis.label_ref, axis.label_raw
        for point in series.points:
            for coordinate in point.coordinates:
                if isinstance(coordinate.value, MeasuredValue):
                    if isinstance(coordinate.value.value_ref, SourceRef) and isinstance(coordinate.value.raw_text, str):
                        yield coordinate.value.value_ref, coordinate.value.raw_text
                    if isinstance(coordinate.value.unit_ref, SourceRef) and isinstance(coordinate.value.unit_raw, str):
                        yield coordinate.value.unit_ref, coordinate.value.unit_raw
            for observation in point.observations:
                if isinstance(observation.value, MeasuredValue):
                    if isinstance(observation.value.value_ref, SourceRef) and isinstance(
                        observation.value.raw_text, str
                    ):
                        yield observation.value.value_ref, observation.value.raw_text
                    if isinstance(observation.value.unit_ref, SourceRef) and isinstance(
                        observation.value.unit_raw, str
                    ):
                        yield observation.value.unit_ref, observation.value.unit_raw
    composition = envelope.composition
    if isinstance(composition, Composition):
        for component in composition.components:
            if isinstance(component.amount, MeasuredValue):
                if isinstance(component.amount.value_ref, SourceRef) and isinstance(component.amount.raw_text, str):
                    yield component.amount.value_ref, component.amount.raw_text
                if isinstance(component.amount.unit_ref, SourceRef) and isinstance(component.amount.unit_raw, str):
                    yield component.amount.unit_ref, component.amount.unit_raw
