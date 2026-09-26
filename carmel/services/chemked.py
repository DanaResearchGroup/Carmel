# ruff: noqa: E501
"""Native, fail-closed ChemKED ignition-delay records.

ChemKED files are fetched as individually pinned raw YAML files. Values use a
dedicated backwards-compatible YAML key-path locator, and replay re-parses the
pinned bytes with :func:`yaml.safe_load` before following every path.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
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
    UnitProvenance,
    ValueOrigin,
    YamlPathLocator,
)
from carmel.services import units
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.respecth import (
    IgnitionCriterion,
    IgnitionDefinition,
    IgnitionTarget,
    RecordText,
    RespecthRefusal,
    _measured,
)
from carmel.services.units import QuantityKind

__all__ = [
    "CHEMKED_IGNITION_CRITERIA",
    "CHEMKED_IGNITION_TARGETS",
    "ChemkedIdtRecord",
    "ChemkedRefusal",
    "ChemkedRefusalReason",
    "parse_idt_record",
    "replay_idt_record",
]


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
    ignition: IgnitionDefinition
    envelope: DatasetEnvelope


#: Explicit translation from the target vocabulary in PyKED's
#: ``chemked_schema.yaml`` to Carmel's spelling-preserving shared model. Schema
#: members absent here (currently ``temperature``) remain typed refusals.
CHEMKED_IGNITION_TARGETS: Mapping[str, IgnitionTarget] = {
    "pressure": IgnitionTarget.PRESSURE,
    "OH": IgnitionTarget.OH,
    "OH*": IgnitionTarget.OH_STAR,
    "CH": IgnitionTarget.CH,
    "CH*": IgnitionTarget.CH_STAR,
}

#: Explicit translation from PyKED's ignition criterion vocabulary. Schema
#: member ``min`` is deliberately absent and therefore remains a typed refusal.
CHEMKED_IGNITION_CRITERIA: Mapping[str, IgnitionCriterion] = {
    "d/dt max": IgnitionCriterion.MAX_SLOPE,
    "max": IgnitionCriterion.PEAK,
    "1/2 max": IgnitionCriterion.HALF_MAX,
    "d/dt max extrapolated": IgnitionCriterion.EXTRAPOLATED_MAX_SLOPE,
}

#: Roles grounded by the pinned collection's fuel families and unambiguous species identities.
#: A species absent here stays explicitly role-less; in particular, CO2, H2O and NO2 are not
#: guessed to be diluents/oxidizers merely because they are not the primary fuel.
_SPECIES_ROLES: Mapping[str, ComponentRole] = {
    "1-butanol": ComponentRole.FUEL,
    "2-butanol": ComponentRole.FUEL,
    "C5H12": ComponentRole.FUEL,
    "C6H12O2": ComponentRole.FUEL,
    "Methyl Decanoate": ComponentRole.FUEL,
    "i-butanol": ComponentRole.FUEL,
    "iso-butanol": ComponentRole.FUEL,
    "n-butanol": ComponentRole.FUEL,
    "n-heptane": ComponentRole.FUEL,
    "nC7H16": ComponentRole.FUEL,
    "t-butanol": ComponentRole.FUEL,
    "toluene": ComponentRole.FUEL,
    "O2": ComponentRole.OXIDIZER,
    "AR": ComponentRole.DILUENT,
    "Ar": ComponentRole.DILUENT,
    "HE": ComponentRole.DILUENT,
    "He": ComponentRole.DILUENT,
    "N2": ComponentRole.DILUENT,
}


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
            try:
                index = int(match.group(2))
            except ValueError as exc:
                raise KeyError(path) from exc
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
            RecordText(raw=value, ref=_ref(path + "#value")),
            RecordText(raw=unit, ref=_ref((unit_path or path) + ("" if unit_path else "#unit"))),
            quantity,
            units.TABLE_V4,
        )
    except Exception as exc:
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_UNIT, f"{path}: {exc}") from exc


def _dimensionless(raw: object, path: str, kind_path: str) -> MeasuredValue:
    try:
        return _measured(
            RecordText(raw=str(raw), ref=_ref(path)),
            RecordText(raw="mole fraction", ref=_ref(kind_path)),
            QuantityKind.MOLE_FRACTION,
            units.TABLE_V4,
        )
    except RespecthRefusal as exc:
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_UNIT, f"{path}: {exc}") from exc


def _equivalence_ratio(raw: object, path: str) -> MeasuredValue:
    """Ground a unitless ratio whose source prints no unit token."""
    try:
        measured = _measured(
            RecordText(raw=str(raw), ref=_ref(path)),
            RecordText(raw="-", ref=_ref(path)),
            QuantityKind.EQUIVALENCE_RATIO,
            units.TABLE_V4,
        )
        return MeasuredValue(
            raw_text=measured.raw_text,
            canonical_decimal_value=measured.canonical_decimal_value,
            repairs=measured.repairs,
            repair_dependency=measured.repair_dependency,
            quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            unit_provenance=UnitProvenance.NOT_PRINTED_IN_SOURCE,
            unit_raw=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            unit_normalized="1",
            conversion_table_sha256=measured.conversion_table_sha256,
            value_ref=_ref(path),
            unit_ref=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
    except (RespecthRefusal, ValueError) as exc:
        raise ChemkedRefusal(ChemkedRefusalReason.SCHEMA_REJECTED, f"{path}: {exc}") from exc


def _composition(row: object, row_index: int, path: str) -> Composition:
    if not isinstance(row, dict):
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: point {row_index} is not a mapping")
    composition = row.get("composition")
    species = composition.get("species") if isinstance(composition, dict) else None
    if (
        not isinstance(species, list)
        or not species
        or not isinstance(composition, dict)
        or composition.get("kind") != "mole fraction"
    ):
        raise ChemkedRefusal(
            ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: datapoints[{row_index}] has unmapped composition"
        )
    components: list[CompositionComponent] = []
    for species_index, item in enumerate(species):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("species-name"), str)
            or not isinstance(item.get("amount"), list)
            or len(item["amount"]) != 1
        ):
            raise ChemkedRefusal(
                ChemkedRefusalReason.INCOMPLETE_RECORD,
                f"{path}: datapoints[{row_index}].composition.species[{species_index}] is malformed",
            )
        name = item["species-name"]
        role = _SPECIES_ROLES.get(name)
        components.append(
            CompositionComponent(
                species_raw_name=name,
                amount=_dimensionless(
                    item["amount"][0],
                    f"datapoints[{row_index}].composition.species[{species_index}].amount[0]",
                    f"datapoints[{row_index}].composition.kind",
                ),
                role=role
                if role is not None
                else Absent(
                    reason=AbsenceReason.UNKNOWN,
                    note="the record states no role and the species is not in the lane's explicit role table",
                ),
            )
        )
    return Composition(
        raw_name="initial composition",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=_equivalence_ratio(row["equivalence-ratio"], f"datapoints[{row_index}].equivalence-ratio")
        if "equivalence-ratio" in row
        else Absent(reason=AbsenceReason.UNKNOWN),
        components=tuple(sorted(components, key=lambda item: item.species_raw_name)),
    )


def _composition_signature(composition: Composition) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    ratio = composition.equivalence_ratio
    return (
        None if isinstance(ratio, Absent) else ratio.canonical_decimal_value,
        tuple(
            (component.species_raw_name, component.amount.canonical_decimal_value)
            for component in composition.components
        ),
    )


def _ignition_definition(doc: dict[object, object], rows: list[object], path: str) -> IgnitionDefinition:
    """Map the file's single effective definition and retain its raw YAML spellings."""
    common = doc.get("common-properties")
    common_ignition = common.get("ignition-type") if isinstance(common, dict) else None
    definitions: list[IgnitionDefinition] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: point {index} is not a mapping")
        if "ignition-type" in row:
            ignition = row["ignition-type"]
            ignition_path = f"datapoints[{index}].ignition-type"
        else:
            ignition = common_ignition
            ignition_path = "common-properties.ignition-type"
        target_raw = ignition.get("target") if isinstance(ignition, dict) else None
        criterion_raw = ignition.get("type") if isinstance(ignition, dict) else None
        target = CHEMKED_IGNITION_TARGETS.get(target_raw) if isinstance(target_raw, str) else None
        criterion = CHEMKED_IGNITION_CRITERIA.get(criterion_raw) if isinstance(criterion_raw, str) else None
        if target is None or criterion is None:
            raise ChemkedRefusal(
                ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION,
                f"{path}: point {index} has target={target_raw!r}, type={criterion_raw!r}",
            )
        definitions.append(
            IgnitionDefinition(
                target=target,
                target_raw=RecordText(raw=target_raw, ref=_ref(f"{ignition_path}.target")),
                criterion=criterion,
                criterion_raw=RecordText(raw=criterion_raw, ref=_ref(f"{ignition_path}.type")),
                amount=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                amount_units=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            )
        )
    first = definitions[0]
    if any((item.target, item.criterion) != (first.target, first.criterion) for item in definitions[1:]):
        raise ChemkedRefusal(
            ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION,
            f"{path}: points do not share one ignition definition",
        )
    return first


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
    doi, rows = _validate_subset(doc, path)
    apparatus = doc.get("apparatus")
    if not isinstance(apparatus, dict) or apparatus.get("kind") not in {"shock tube", "rapid compression machine"}:
        raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_APPARATUS, f"{path}: {apparatus!r}")
    if not doi:
        raise ChemkedRefusal(ChemkedRefusalReason.INCOMPLETE_RECORD, f"{path}: missing reference DOI")
    ignition_definition = _ignition_definition(doc, rows, path)
    if apparatus["kind"] == "rapid compression machine":
        for row_index, row in enumerate(rows):
            history = row.get("volume-history") if isinstance(row, dict) else None
            if history is not None and not isinstance(history, dict):
                raise ChemkedRefusal(
                    ChemkedRefusalReason.SCHEMA_REJECTED,
                    f"{path}: datapoints[{row_index}].volume-history must be a mapping",
                )
            values = history.get("values") if isinstance(history, dict) else None
            if values is not None and not isinstance(values, list):
                raise ChemkedRefusal(
                    ChemkedRefusalReason.SCHEMA_REJECTED,
                    f"{path}: datapoints[{row_index}].volume-history.values must be a list",
                )
            if values:
                try:
                    numeric_pairs: list[tuple[float, float]] = []
                    for entry_index, pair in enumerate(values):
                        if (
                            not isinstance(pair, list)
                            or len(pair) != 2
                            or any(isinstance(value, bool) for value in pair)
                        ):
                            raise ValueError(
                                f"entry {entry_index} must be exactly two numbers [time, volume], got {pair!r}"
                            )
                        numeric_pair = (float(pair[0]), float(pair[1]))
                        if not all(math.isfinite(value) for value in numeric_pair):
                            raise ValueError(f"entry {entry_index} contains a non-finite value")
                        numeric_pairs.append(numeric_pair)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ChemkedRefusal(
                        ChemkedRefusalReason.SCHEMA_REJECTED,
                        f"{path}: RCM volume-history values must be finite [time, volume] pairs: {exc}",
                    ) from exc
                numeric_volumes = [pair[1] for pair in numeric_pairs]
                if numeric_volumes[0] != min(numeric_volumes):
                    raise ChemkedRefusal(
                        ChemkedRefusalReason.RCM_PRE_COMPRESSION_CONDITIONS,
                        f"{path}: RCM volume history begins before its minimum volume",
                    )
    compositions = tuple(_composition(row, index, path) for index, row in enumerate(rows))
    composition_is_constant = all(
        _composition_signature(composition) == _composition_signature(compositions[0])
        for composition in compositions[1:]
    )
    fuels = tuple(
        dict.fromkeys(
            component.species_raw_name
            for composition in compositions
            for component in composition.components
            if component.role is ComponentRole.FUEL
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
    for index, (row, composition) in enumerate(zip(rows, compositions, strict=True)):
        assert isinstance(row, dict)
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
                composition=Absent(reason=AbsenceReason.SAME_AS_DATASET) if composition_is_constant else composition,
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
        composition=compositions[0]
        if composition_is_constant
        else Absent(
            reason=AbsenceReason.NOT_APPLICABLE,
            note="the record's composition varies by point; each DataPoint carries its own grounded composition",
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
                sha256=units.TABLE_V4.sha256,
                canonical_json=canonical_json_bytes(units.TABLE_V4.identity_payload()).decode(),
            ),
        ),
        table_inventories=(),
        ooxml_table_inventories=(),
        figure_digitizations=(),
    )
    return ChemkedIdtRecord(
        path=path,
        sha256=actual,
        citation_doi=doi,
        fuels=fuels,
        ignition=ignition_definition,
        envelope=envelope,
    )


def _validate_subset(doc: dict[object, object], path: str) -> tuple[str, list[object]]:
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
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for field in ("pressure", "temperature", "ignition-delay"):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or not values
                or isinstance(values[0], dict)
                or any(not isinstance(metadata, dict) for metadata in values[1:])
            ):
                raise ChemkedRefusal(
                    ChemkedRefusalReason.SCHEMA_REJECTED,
                    f"{path}: datapoints[{index}].{field} must contain exactly one scalar value, "
                    "followed only by metadata mappings",
                )
    return reference["doi"], rows


def _first_difference(recorded: object, derived: object, field: str) -> str | None:
    """Name the first field whose stored value differs from its re-derived value."""
    if isinstance(recorded, dict) and isinstance(derived, dict):
        if recorded.keys() != derived.keys():
            return f"{field}: recorded keys {tuple(recorded)}; re-derived keys {tuple(derived)}"
        for key in derived:
            difference = _first_difference(recorded[key], derived[key], f"{field}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(recorded, (list, tuple)) and isinstance(derived, (list, tuple)):
        if len(recorded) != len(derived):
            return f"{field}: recorded length {len(recorded)}; re-derived length {len(derived)}"
        for index, (recorded_item, derived_item) in enumerate(zip(recorded, derived, strict=True)):
            difference = _first_difference(recorded_item, derived_item, f"{field}[{index}]")
            if difference is not None:
                return difference
        return None
    if recorded != derived:
        return f"{field}: recorded {recorded!r}; re-derived {derived!r}"
    return None


def replay_idt_record(record: ChemkedIdtRecord, raw_bytes: bytes) -> None:
    """Re-parse pinned bytes and refuse the first stored field that does not re-derive."""
    derived = parse_idt_record(raw_bytes, record.path, record.sha256)
    fields = (
        (record.citation_doi, derived.citation_doi, "citation_doi"),
        (record.fuels, derived.fuels, "fuels"),
        (record.ignition.model_dump(mode="json"), derived.ignition.model_dump(mode="json"), "ignition"),
        (record.envelope.model_dump(mode="json"), derived.envelope.model_dump(mode="json"), "envelope"),
    )
    for recorded, expected, field in fields:
        difference = _first_difference(recorded, expected, field)
        if difference is not None:
            raise ChemkedRefusal(ChemkedRefusalReason.UNRESOLVABLE_PATH, f"{record.path}: {difference}")
