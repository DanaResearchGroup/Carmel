# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""ReSpecTh flame-speed and speciation records -> :class:`DatasetEnvelope`, and back.

The second RKD route, over the same two pinned archives as the ignition-delay lane
(:mod:`carmel.services.respecth`) and on the same terms: one member XML is parsed natively into
a :class:`RespecthSeriesRecord` -- a :class:`DatasetEnvelope` plus the record-level facts it has
no slot for -- with ONE ``DATABASE_RECORD`` source node and a positional :class:`XPathLocator`
behind every stored value, unit, label and fact. :func:`replay_series_record` re-hashes the
member, re-evaluates every path, and re-derives the whole record from the bytes.

Four ``experimentType`` values map (:data:`EXPERIMENT_KINDS`):

- laminar burning velocity -> ``lbv``: the velocity is the series' observation; T and P are
  constants or coordinates; the mixture is either the common ``initial composition`` (the
  envelope's composition) or per-point ``composition`` columns, which become one mole-fraction
  COORDINATE axis per species AND each point's own :class:`Composition`.
- jet stirred reactor, outlet concentration, concentration time profile -> ``jsr``, ``outlet``,
  ``profile``: every ``composition`` column is one mole-fraction OBSERVATION axis, all sharing
  the swept coordinate(s) -- temperature, residence time, or (profile) time. The series stays
  ONE series; a time-resolved profile is a series whose coordinate is ``time``.

The envelope has no species slot on an axis, so each species axis's identity (preferred key,
InChI, CAS, SMILES, name -- whichever the member gives) is a grounded record-level fact
(:class:`SpeciesColumn`). A column that names no identifier is refused, as are two columns naming
one species.

Everything this lane does not map is a typed :class:`RespecthRefusal` with no partial output:
an experiment type outside :data:`EXPERIMENT_KINDS` (burner-stabilized flame speciation among
them, see :data:`REFUSED_EXPERIMENT_TYPES`), an apparatus outside
:data:`APPARATUS_DEVICE_CLASSES`, a unit :data:`carmel.services.units.TABLE_V3` cannot bind
(``Torr``, ``mPa``), a property, column, top-level element or timeshift the lane does not know,
and a condition stated both as a constant and as a column. No quality flag is carried: the
records have none.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from carmel.schemas.campaign import ReactorType
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    AxisDeclaration,
    AxisRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    Maybe,
    MeasuredValue,
    Observation,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal
from carmel.services.respecth import (
    _SPECIES_ROLES,
    _SUPPORTED_RKD_MAJOR,
    RECORD_NODE_ID,
    ArchivePin,
    PaperReference,
    RecordReplayReport,
    RecordText,
    ReferenceDoiTrust,
    RespecthRefusal,
    RespecthRefusalReason,
    _composition,
    _Doc,
    _embedded_table,
    _measured,
    _one,
    _paper,
    _parse_xml,
    check_record_node,
    replay_grounded_text,
)
from carmel.services.respecth_archive import PinnedArchive
from carmel.services.units import QuantityKind

__all__ = [
    "APPARATUS_DEVICE_CLASSES",
    "EXPERIMENT_KINDS",
    "REFUSED_EXPERIMENT_TYPES",
    "PointUncertainty",
    "ReportedUncertaintyColumn",
    "RespecthExperimentKind",
    "RespecthSeriesRecord",
    "SeriesApparatus",
    "SpeciesColumn",
    "SpeciesIdentity",
    "Timeshift",
    "TimeshiftType",
    "UncertaintyStatement",
    "parse_series_record",
    "read_experiment_type",
    "replay_series_record",
]

#: TABLE_V2 plus the exact ``mm/s`` velocity scale some flame-speed columns use.
_TABLE = units.TABLE_V3


class RespecthExperimentKind(StrEnum):
    """The experiment kinds this module maps; the value is the ``carmel data find --kind`` name."""

    LAMINAR_BURNING_VELOCITY = "lbv"
    JET_STIRRED_REACTOR = "jsr"
    OUTLET_CONCENTRATION = "outlet"
    CONCENTRATION_TIME_PROFILE = "profile"


_LBV = RespecthExperimentKind.LAMINAR_BURNING_VELOCITY
_JSR = RespecthExperimentKind.JET_STIRRED_REACTOR
_OUTLET = RespecthExperimentKind.OUTLET_CONCENTRATION
_PROFILE = RespecthExperimentKind.CONCENTRATION_TIME_PROFILE

#: RKD ``experimentType`` -> kind. Anything else is refused as ``unmapped_experiment_type``.
EXPERIMENT_KINDS: Mapping[str, RespecthExperimentKind] = {
    "laminar burning velocity measurement": _LBV,
    "jet stirred reactor measurement": _JSR,
    "outlet concentration measurement": _OUTLET,
    "concentration time profile measurement": _PROFILE,
}

#: Experiment types present in the pinned archives that are refused on purpose, with why.
REFUSED_EXPERIMENT_TYPES: Mapping[str, str] = {
    "burner stabilized flame speciation measurement": (
        "its boundary condition is a mass flux (flow rate in g cm-2 s-1), a quantity the unit table has "
        "no kind for, and its temperature column is a flame-temperature profile beside the inlet "
        "temperature constant; the spatial axis itself (distance) would map as a length"
    ),
}

#: The explicit apparatus table: (kind, ``apparatus/kind``, the set of ``apparatus/mode``s) ->
#: device class. Anything not listed is refused.
APPARATUS_DEVICE_CLASSES: Mapping[tuple[RespecthExperimentKind, str, frozenset[str]], ReactorType] = {
    **{
        (_LBV, "flame", frozenset({"premixed", "laminar", *extra})): ReactorType.FLAME
        for extra in (
            (),
            ("spherical",),
            ("cylindrical",),
            ("counterflow", "twin flat"),
            ("HFM",),
            ("OPF",),
            ("CTF",),
        )
    },
    (_JSR, "stirred reactor", frozenset()): ReactorType.JSR,
    (_OUTLET, "flow reactor", frozenset()): ReactorType.PFR,
    (_OUTLET, "flow reactor", frozenset({"turbulent"})): ReactorType.PFR,
    (_OUTLET, "shock tube", frozenset({"reflected shock"})): ReactorType.SHOCK_TUBE,
    (_PROFILE, "flow reactor", frozenset()): ReactorType.PFR,
    (_PROFILE, "flow reactor", frozenset({"turbulent"})): ReactorType.PFR,
}

#: RKD condition property name -> (series axis id, quantity). A condition is a constant in
#: ``commonProperties`` or a coordinate column in the data group -- never both.
_CONDITIONS: Mapping[str, tuple[str, QuantityKind]] = {
    "temperature": ("temperature", QuantityKind.TEMPERATURE),
    "pressure": ("pressure", QuantityKind.PRESSURE),
    "residence time": ("residence_time", QuantityKind.TIME),
    "time": ("time", QuantityKind.TIME),
    "volume": ("volume", QuantityKind.VOLUME),
}
_LBV_NAME = "laminar burning velocity"
_LBV_AXIS_ID = "laminar_burning_velocity"
_SPECIES_COLUMN = "composition"
_STD_DEV = "evaluated standard deviation"
_UNCERTAINTY = "uncertainty"
_PLUS_MINUS = "plusminus"
_SERIES_IDS: Mapping[RespecthExperimentKind, str] = {
    _LBV: "laminar_burning_velocity",
    _JSR: "mole_fractions",
    _OUTLET: "mole_fractions",
    _PROFILE: "mole_fractions",
}
_TOP_LEVEL_TAGS = frozenset(
    {
        "fileAuthor",
        "fileDOI",
        "fileVersion",
        "ReSpecThVersion",
        "firstPublicationDate",
        "lastModificationDate",
        "bibliographyLink",
        "experimentType",
        "apparatus",
        "commonProperties",
        "dataGroup",
        "comment",
        "timeshift",
    }
)
#: ``speciesLink`` attributes that identify a species, and the ones that may accompany them.
_SPECIES_IDENTIFIERS = ("preferredKey", "InChI", "CAS", "SMILES")
_SPECIES_LINK_ATTRIBUTES = frozenset({*_SPECIES_IDENTIFIERS, "chemName"})
_EQUIVALENCE_RATIO_NOTE = "RKD records state no equivalence ratio; not derived"


# --------------------------------------------------------------------------- record model


class SpeciesIdentity(BaseModel):
    """The identifiers a ``speciesLink`` gives, verbatim. At least one of preferred key, InChI,
    CAS or SMILES is present -- a chemical name alone does not identify a species."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preferred_key: Maybe[RecordText]
    inchi: Maybe[RecordText]
    cas: Maybe[RecordText]
    smiles: Maybe[RecordText]
    chem_name: Maybe[RecordText]

    @model_validator(mode="after")
    def _identified(self) -> SpeciesIdentity:
        if all(isinstance(value, Absent) for value in (self.preferred_key, self.inchi, self.cas, self.smiles)):
            raise ValueError("a species needs a preferred key, InChI, CAS or SMILES")
        return self

    @property
    def key(self) -> tuple[str | None, ...]:
        """``(preferredKey, InChI, CAS, SMILES)`` raw, ``None`` where absent -- the identity two
        mentions of a species must agree on exactly."""
        return tuple(
            None if isinstance(value, Absent) else value.raw
            for value in (self.preferred_key, self.inchi, self.cas, self.smiles)
        )

    @property
    def display(self) -> str:
        """The most readable identifier present."""
        return next(raw for raw in self.key if raw is not None)


class SpeciesColumn(BaseModel):
    """Which species one mole-fraction axis of the series is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    species: SpeciesIdentity


class SeriesApparatus(BaseModel):
    """The apparatus the record names -- every ``mode`` it states, in order -- and the device
    class the explicit table maps it to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_class: ReactorType
    kind_raw: RecordText
    modes_raw: tuple[RecordText, ...]


class UncertaintyStatement(BaseModel):
    """The words a common ``evaluated standard deviation`` or ``uncertainty`` property states,
    grounding the kind and basis of the envelope :class:`Uncertainty` on ``axis_id``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    name_raw: RecordText
    """``evaluated standard deviation`` -> ``STD_DEV``; ``uncertainty`` -> ``UNKNOWN`` (a stated
    +- bound whose statistic the record does not name)."""
    basis_raw: RecordText
    """``@kind``: ``absolute`` or ``relative``."""
    reference_raw: RecordText
    method_raw: Maybe[RecordText]
    bound_raw: Maybe[RecordText]
    species: Maybe[SpeciesIdentity]
    """For a species deviation, the species it is stated for (matched to its column exactly)."""


class PointUncertainty(BaseModel):
    """One point's value of a per-point uncertainty column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point_id: str
    value: MeasuredValue


class ReportedUncertaintyColumn(BaseModel):
    """A per-point ``uncertainty`` column of the data group (the source paper's own +- bars).

    Kept beside the envelope rather than in it: the observation's :class:`Uncertainty` slot
    carries the record's ``evaluated standard deviation``, as in the ignition-delay lane, and
    one slot cannot hold both."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    """The observation axis the column's ``@reference`` names."""
    label: RecordText
    basis_raw: RecordText
    reference_raw: RecordText
    bound_raw: RecordText
    values: tuple[PointUncertainty, ...] = Field(min_length=1)


class TimeshiftType(StrEnum):
    """How a concentration profile's time axis is anchored (``timeshift/@type``)."""

    HALF_DECREASE = "half decrease"
    """t is shifted so the target's concentration falls to half its initial value at the
    measured instant."""
    RELATIVE_DECREASE = "relative decrease"
    """As above, at ``amount`` times its initial value; ``amount`` is required."""


class Timeshift(BaseModel):
    """A concentration profile's time-axis anchoring: time values are relative to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_raw: RecordText
    type: TimeshiftType
    type_raw: RecordText
    amount: Maybe[RecordText]


class RespecthSeriesRecord(BaseModel):
    """One RKD flame-speed or speciation member, mapped. ``envelope`` holds every number; the
    rest are the record-level facts a :class:`DatasetEnvelope` has no slot for, each grounded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["respecth"] = "respecth"
    kind: RespecthExperimentKind
    experiment_type: RecordText
    archive: ArchivePin
    record_doi: RecordText
    """The ReSpecTh ``fileDOI`` -- the citable, stable id of THIS record."""
    paper: Maybe[PaperReference]
    apparatus: SeriesApparatus
    species_columns: tuple[SpeciesColumn, ...]
    uncertainty_statements: tuple[UncertaintyStatement, ...]
    reported_uncertainties: tuple[ReportedUncertaintyColumn, ...]
    timeshift: Maybe[Timeshift]
    """Present exactly when a concentration-time profile states one."""
    column_source_types: tuple[tuple[str, RecordText], ...]
    """Per series axis, the RKD ``@sourcetype`` (``reported``/``digitized``/...)."""
    envelope: DatasetEnvelope

    @model_validator(mode="after")
    def _node_matches_pin(self) -> RespecthSeriesRecord:
        check_record_node(self.envelope, self.archive)
        return self

    @property
    def member_sha256(self) -> str:
        """sha256 of the member bytes every locator addresses."""
        return self.envelope.source_graph.node(RECORD_NODE_ID).sha256

    @property
    def citation_doi(self) -> str:
        """The id to cite this record by: always the ReSpecTh fileDOI."""
        return self.record_doi.raw

    @property
    def paper_doi(self) -> str | None:
        """The source paper's DOI, or ``None`` when absent or only a sorting placeholder."""
        if isinstance(self.paper, Absent) or self.paper.trust is not ReferenceDoiTrust.CITED:
            return None
        return self.paper.doi.raw

    def species(self, axis_id: str) -> SpeciesIdentity:
        """The species a mole-fraction axis is."""
        return next(column.species for column in self.species_columns if column.axis_id == axis_id)


# --------------------------------------------------------------------------- mapping helpers


def _optional_attribute(doc: _Doc, element: ElementTree.Element, name: str) -> Maybe[RecordText]:
    if (element.get(name) or "").strip():
        return doc.attribute(element, name)
    return Absent(reason=AbsenceReason.NOT_REPORTED_HERE)


def _species_identity(doc: _Doc, owner: ElementTree.Element) -> SpeciesIdentity:
    """The one ``speciesLink`` under ``owner``, or refuse as unidentified."""
    unidentified = RespecthRefusalReason.UNIDENTIFIED_SPECIES
    links = owner.findall("speciesLink")
    where = doc.path(owner)
    if len(links) != 1:
        raise RespecthRefusal(unidentified, f"{where} carries {len(links)} speciesLink(s), not exactly one")
    (link,) = links
    extra = set(link.attrib) - _SPECIES_LINK_ATTRIBUTES
    if extra:
        raise RespecthRefusal(unidentified, f"{where} speciesLink has unmapped attributes {sorted(extra)}")
    if not any((link.get(name) or "").strip() for name in _SPECIES_IDENTIFIERS):
        raise RespecthRefusal(unidentified, f"{where} names no species identifier (preferredKey, InChI, CAS or SMILES)")
    return SpeciesIdentity(
        preferred_key=_optional_attribute(doc, link, "preferredKey"),
        inchi=_optional_attribute(doc, link, "InChI"),
        cas=_optional_attribute(doc, link, "CAS"),
        smiles=_optional_attribute(doc, link, "SMILES"),
        chem_name=_optional_attribute(doc, link, "chemName"),
    )


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _species_axis_id(species: SpeciesIdentity, column_id: str) -> str:
    """``x_<preferred key>`` slugged to the schema's id grammar, else ``x_col_<column id>``."""
    if isinstance(species.preferred_key, RecordText):
        return "x_" + _SLUG_RE.sub("_", species.preferred_key.raw.strip().lower())
    return "x_col_" + _SLUG_RE.sub("_", column_id.strip().lower())


def _apparatus(doc: _Doc, kind: RespecthExperimentKind) -> SeriesApparatus:
    apparatus = _one(doc.root, "apparatus")
    kind_raw = doc.text(_one(apparatus, "kind"))
    modes = tuple(doc.text(mode) for mode in apparatus.findall("mode") if (mode.text or "").strip())
    mode_set = frozenset(mode.raw.strip() for mode in modes)
    extra = {child.tag for child in apparatus} - {"kind", "mode"}
    device_class = APPARATUS_DEVICE_CLASSES.get((kind, kind_raw.raw.strip(), mode_set))
    if device_class is None or extra or len(mode_set) != len(modes):
        raise RespecthRefusal(
            RespecthRefusalReason.UNMAPPED_APPARATUS,
            f"{kind.value} apparatus kind={kind_raw.raw!r} modes={[mode.raw for mode in modes]} "
            f"(extra fields {sorted(extra)}) is not in the apparatus table",
        )
    return SeriesApparatus(device_class=device_class, kind_raw=kind_raw, modes_raw=modes)


def _timeshift(doc: _Doc, kind: RespecthExperimentKind) -> Maybe[Timeshift]:
    elements = doc.root.findall("timeshift")
    if not elements:
        return Absent(reason=AbsenceReason.NOT_REPORTED_HERE if kind is _PROFILE else AbsenceReason.NOT_APPLICABLE)
    unmapped = RespecthRefusalReason.UNMAPPED_TIMESHIFT
    if kind is not _PROFILE or len(elements) != 1:
        raise RespecthRefusal(unmapped, f"{len(elements)} timeshift element(s) on a {kind.value} record")
    (element,) = elements
    type_raw = doc.attribute(element, "type")
    extra = set(element.attrib) - {"target", "type", "amount"}
    if type_raw.raw not in {member.value for member in TimeshiftType} or extra:
        raise RespecthRefusal(unmapped, f"timeshift type {type_raw.raw!r} (extra {sorted(extra)}) is not mapped")
    mapped = TimeshiftType(type_raw.raw)
    amount: Maybe[RecordText] = Absent(reason=AbsenceReason.NOT_APPLICABLE)
    if mapped is TimeshiftType.RELATIVE_DECREASE:
        amount = doc.attribute(element, "amount")
        try:
            fraction = Decimal(canonical_decimal(amount.raw))
        except CanonicalDecimalError as exc:
            raise RespecthRefusal(unmapped, f"timeshift amount {amount.raw!r} is not a decimal") from exc
        if not 0 < fraction <= 1:
            raise RespecthRefusal(unmapped, f"timeshift amount {amount.raw!r} is not a fraction in (0, 1]")
    elif "amount" in element.attrib:
        raise RespecthRefusal(unmapped, f"timeshift type {type_raw.raw!r} carries an amount it does not define")
    return Timeshift(target_raw=doc.attribute(element, "target"), type=mapped, type_raw=type_raw, amount=amount)


# --------------------------------------------------------------------------- the data group


@dataclass
class _Column:
    """One data-group column and what it maps to."""

    column_id: str
    element: ElementTree.Element
    axis_id: str
    role: AxisRole
    quantity: QuantityKind
    units: RecordText
    species: SpeciesIdentity | None = None


@dataclass
class _Mapping:
    """Everything read off ``commonProperties`` and the data group, before assembly."""

    constants: dict[str, tuple[ElementTree.Element, QuantityKind]] = field(default_factory=dict)
    initial_composition: Composition | None = None
    std_devs: list[ElementTree.Element] = field(default_factory=list)
    condition_uncertainties: list[ElementTree.Element] = field(default_factory=list)
    columns: list[_Column] = field(default_factory=list)
    uncertainty_columns: list[ElementTree.Element] = field(default_factory=list)
    source_types: dict[str, RecordText] = field(default_factory=dict)


def _read_common(doc: _Doc, mapping: _Mapping) -> None:
    unmapped = RespecthRefusalReason.UNMAPPED_PROPERTY
    seen: set[str] = set()
    for prop in _one(doc.root, "commonProperties"):
        name = prop.get("name", "")
        if prop.tag != "property":
            raise RespecthRefusal(unmapped, f"unexpected <{prop.tag}> in commonProperties")
        if name == _STD_DEV:
            mapping.std_devs.append(prop)
            continue
        if name == _UNCERTAINTY:
            mapping.condition_uncertainties.append(prop)
            continue
        if name in seen:
            raise RespecthRefusal(unmapped, f"common property {name!r} is stated twice")
        seen.add(name)
        if name in _CONDITIONS:
            axis_id, quantity = _CONDITIONS[name]
            mapping.constants[axis_id] = (prop, quantity)
            if prop.get("sourcetype"):
                mapping.source_types[axis_id] = doc.attribute(prop, "sourcetype")
        elif name == "initial composition":
            mapping.initial_composition = _composition(doc, prop, _TABLE, _EQUIVALENCE_RATIO_NOTE)
        else:
            raise RespecthRefusal(unmapped, f"common property {name!r} ({prop.get('units')!r}) is not mapped")


def _read_columns(doc: _Doc, kind: RespecthExperimentKind, group: ElementTree.Element, mapping: _Mapping) -> None:
    unmapped = RespecthRefusalReason.UNMAPPED_PROPERTY
    seen_column_ids: set[str] = set()
    for prop in group.findall("property"):
        name = prop.get("name", "")
        column_id = doc.attribute(prop, "id").raw
        if column_id in seen_column_ids:
            raise RespecthRefusal(
                RespecthRefusalReason.INCOMPLETE_RECORD,
                f"data columns reuse id {column_id!r}",
            )
        seen_column_ids.add(column_id)
        species: SpeciesIdentity | None = None
        if name == _UNCERTAINTY:
            mapping.uncertainty_columns.append(prop)
            continue
        if name in _CONDITIONS:
            axis_id, quantity = _CONDITIONS[name]
            role = AxisRole.COORDINATE
        elif name == _LBV_NAME and kind is _LBV:
            axis_id, quantity, role = _LBV_AXIS_ID, QuantityKind.VELOCITY, AxisRole.OBSERVATION
        elif name == _SPECIES_COLUMN:
            species = _species_identity(doc, prop)
            axis_id, quantity = _species_axis_id(species, column_id), QuantityKind.MOLE_FRACTION
            role = AxisRole.COORDINATE if kind is _LBV else AxisRole.OBSERVATION
        else:
            raise RespecthRefusal(unmapped, f"{kind.value} data column {name!r} is not mapped")
        if axis_id in mapping.constants:
            raise RespecthRefusal(
                unmapped, f"{name!r} is stated both as a common constant and as data column {column_id!r}"
            )
        clash = next((column for column in mapping.columns if column.axis_id == axis_id), None)
        if clash is not None:
            reason = RespecthRefusalReason.UNIDENTIFIED_SPECIES if species is not None else unmapped
            raise RespecthRefusal(reason, f"columns {clash.column_id!r} and {column_id!r} both map to axis {axis_id!r}")
        if prop.get("sourcetype"):
            mapping.source_types[axis_id] = doc.attribute(prop, "sourcetype")
        mapping.columns.append(
            _Column(
                column_id=column_id,
                element=prop,
                axis_id=axis_id,
                role=role,
                quantity=quantity,
                units=doc.attribute(prop, "units"),
                species=species,
            )
        )


def _check_required(kind: RespecthExperimentKind, mapping: _Mapping) -> None:
    incomplete = RespecthRefusalReason.INCOMPLETE_RECORD
    axis_ids = set(mapping.constants) | {column.axis_id for column in mapping.columns}
    for required in ("temperature", "pressure"):
        if required not in axis_ids:
            raise RespecthRefusal(incomplete, f"the record states no {required}")
    species_columns = [column for column in mapping.columns if column.species is not None]
    if kind is _LBV:
        if _LBV_AXIS_ID not in axis_ids:
            raise RespecthRefusal(incomplete, "the flame-speed record has no laminar burning velocity column")
        if (mapping.initial_composition is None) == (not species_columns):
            raise RespecthRefusal(
                incomplete,
                "the mixture must be stated exactly once, as the initial composition or as per-point composition "
                f"columns (initial composition {'absent' if mapping.initial_composition is None else 'present'}, "
                f"{len(species_columns)} composition column(s))",
            )
        return
    if not species_columns:
        raise RespecthRefusal(incomplete, f"the {kind.value} record has no species column")
    if mapping.initial_composition is None:
        raise RespecthRefusal(incomplete, "the record states no initial composition")
    if kind is _PROFILE and "time" not in {column.axis_id for column in mapping.columns}:
        raise RespecthRefusal(incomplete, "the concentration profile has no time column")


# --------------------------------------------------------------------------- uncertainties


def _basis(doc: _Doc, prop: ElementTree.Element) -> tuple[UncertaintyBasis, RecordText]:
    basis_raw = doc.attribute(prop, "kind")
    if basis_raw.raw not in {member.value for member in UncertaintyBasis}:
        raise RespecthRefusal(
            RespecthRefusalReason.UNMAPPED_PROPERTY, f"{doc.path(prop)} states uncertainty kind {basis_raw.raw!r}"
        )
    return UncertaintyBasis(basis_raw.raw), basis_raw


def _bound_quantity(basis: UncertaintyBasis, quantity: QuantityKind) -> QuantityKind:
    return quantity if basis is UncertaintyBasis.ABSOLUTE else QuantityKind.RELATIVE_UNCERTAINTY


def _std_dev_target(
    doc: _Doc, kind: RespecthExperimentKind, prop: ElementTree.Element, mapping: _Mapping
) -> tuple[_Column, Maybe[SpeciesIdentity]]:
    """The observation column an ``evaluated standard deviation`` is of."""
    reference = prop.get("reference")
    observations = [column for column in mapping.columns if column.role is AxisRole.OBSERVATION]
    if kind is _LBV and reference == _LBV_NAME and prop.find("speciesLink") is None:
        (column,) = (column for column in observations if column.axis_id == _LBV_AXIS_ID)
        return column, Absent(reason=AbsenceReason.NOT_APPLICABLE)
    if kind is not _LBV and reference == _SPECIES_COLUMN:
        species = _species_identity(doc, prop)
        match = next(
            (column for column in observations if column.species is not None and column.species.key == species.key),
            None,
        )
        if match is None:
            near = [column.species.key for column in observations if column.species is not None]
            raise RespecthRefusal(
                RespecthRefusalReason.UNIDENTIFIED_SPECIES,
                f"{doc.path(prop)} is stated for species {species.key}, whose identifiers match no column exactly "
                f"(columns carry {near})",
            )
        return match, species
    raise RespecthRefusal(
        RespecthRefusalReason.UNMAPPED_PROPERTY,
        f"{kind.value} evaluated standard deviation with reference={reference!r} is not mapped",
    )


def _uncertainties(
    doc: _Doc, kind: RespecthExperimentKind, mapping: _Mapping
) -> tuple[dict[str, Uncertainty], tuple[UncertaintyStatement, ...]]:
    """Every common uncertainty property -> ``{axis_id: Uncertainty}`` plus its grounded words."""
    unmapped = RespecthRefusalReason.UNMAPPED_PROPERTY
    by_axis: dict[str, Uncertainty] = {}
    statements: list[UncertaintyStatement] = []
    quantities = {column.axis_id: column.quantity for column in mapping.columns}
    quantities |= {axis_id: quantity for axis_id, (_, quantity) in mapping.constants.items()}
    for prop in mapping.std_devs + mapping.condition_uncertainties:
        basis, basis_raw = _basis(doc, prop)
        species: Maybe[SpeciesIdentity] = Absent(reason=AbsenceReason.NOT_APPLICABLE)
        if prop.get("name") == _STD_DEV:
            column, species = _std_dev_target(doc, kind, prop, mapping)
            axis_id, uncertainty_kind = column.axis_id, UncertaintyKind.STD_DEV
        else:
            reference = prop.get("reference", "")
            if reference not in _CONDITIONS or _CONDITIONS[reference][0] not in quantities:
                raise RespecthRefusal(
                    unmapped, f"common uncertainty with reference={reference!r} names no condition of this record"
                )
            if prop.get("bound") != _PLUS_MINUS:
                raise RespecthRefusal(unmapped, f"common uncertainty bound {prop.get('bound')!r} is not mapped")
            axis_id, uncertainty_kind = _CONDITIONS[reference][0], UncertaintyKind.UNKNOWN
        if axis_id in by_axis:
            raise RespecthRefusal(unmapped, f"two uncertainties are stated for {axis_id!r}")
        bound = _measured(
            doc.text(_one(prop, "value")),
            doc.attribute(prop, "units"),
            _bound_quantity(basis, quantities[axis_id]),
            _TABLE,
        )
        by_axis[axis_id] = Uncertainty(
            kind=uncertainty_kind,
            basis=basis,
            scale=Absent(reason=AbsenceReason.UNKNOWN, note="the record does not say linear or log"),
            upper=bound,
            lower=bound,
        )
        statements.append(
            UncertaintyStatement(
                axis_id=axis_id,
                name_raw=doc.attribute(prop, "name"),
                basis_raw=basis_raw,
                reference_raw=doc.attribute(prop, "reference"),
                method_raw=_optional_attribute(doc, prop, "method"),
                bound_raw=_optional_attribute(doc, prop, "bound"),
                species=species,
            )
        )
    return by_axis, tuple(sorted(statements, key=lambda statement: statement.axis_id))


_Rows = list[tuple[str, dict[str, ElementTree.Element]]]


def _reported_uncertainty_columns(doc: _Doc, mapping: _Mapping, rows: _Rows) -> tuple[ReportedUncertaintyColumn, ...]:
    out: list[ReportedUncertaintyColumn] = []
    observations = [column for column in mapping.columns if column.role is AxisRole.OBSERVATION]
    for prop in mapping.uncertainty_columns:
        reference = doc.attribute(prop, "reference")
        target = next(
            (
                column
                for column in observations
                if reference.raw in {column.element.get("label"), column.element.get("name")}
            ),
            None,
        )
        basis, basis_raw = _basis(doc, prop)
        bound = doc.attribute(prop, "bound")
        if target is None or bound.raw != _PLUS_MINUS:
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_PROPERTY,
                f"uncertainty column {doc.path(prop)} (reference {reference.raw!r}, bound {bound.raw!r}) is not mapped",
            )
        units_raw = doc.attribute(prop, "units")
        quantity = _bound_quantity(basis, target.quantity)
        column_id = doc.attribute(prop, "id").raw
        values = tuple(
            PointUncertainty(
                point_id=point_id, value=_measured(doc.text(cells[column_id]), units_raw, quantity, _TABLE)
            )
            for point_id, cells in rows
        )
        out.append(
            ReportedUncertaintyColumn(
                axis_id=target.axis_id,
                label=doc.attribute(prop, "label") if prop.get("label") else doc.attribute(prop, "name"),
                basis_raw=basis_raw,
                reference_raw=reference,
                bound_raw=bound,
                values=values,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- assembly


def _no_uncertainty() -> Absent:
    return Absent(reason=AbsenceReason.UNKNOWN, note="the record states no uncertainty for this value")


def _point_composition(doc: _Doc, columns: list[_Column], cells: dict[str, ElementTree.Element]) -> Composition:
    components: list[CompositionComponent] = []
    for column in columns:
        species = column.species
        if species is None:
            raise RespecthRefusal(
                RespecthRefusalReason.UNIDENTIFIED_SPECIES,
                f"composition column {column.column_id!r} has no species identifier to name its mixture component",
            )
        name_value = next(
            (
                value
                for value in (species.preferred_key, species.inchi, species.cas, species.smiles)
                if isinstance(value, RecordText)
            ),
            None,
        )
        if name_value is None:
            raise RespecthRefusal(
                RespecthRefusalReason.UNIDENTIFIED_SPECIES,
                f"composition column {column.column_id!r} has no species identifier to name its mixture component",
            )
        name = name_value.raw
        role = _SPECIES_ROLES.get(species.preferred_key.raw) if isinstance(species.preferred_key, RecordText) else None
        components.append(
            CompositionComponent(
                species_raw_name=name,
                amount=_measured(doc.text(cells[column.column_id]), column.units, QuantityKind.MOLE_FRACTION, _TABLE),
                role=role
                if role is not None
                else Absent(
                    reason=AbsenceReason.UNKNOWN,
                    note="the record states no role and the species is not in the lane's role table",
                ),
            )
        )
    return Composition(
        raw_name=_SPECIES_COLUMN,
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=Absent(reason=AbsenceReason.UNKNOWN, note=_EQUIVALENCE_RATIO_NOTE),
        components=tuple(sorted(components, key=lambda component: component.species_raw_name)),
    )


def _axis(
    doc: _Doc, column_or_axis: _Column | str, quantity: QuantityKind, prop: ElementTree.Element
) -> AxisDeclaration:
    """A species column is labelled by its ``@label`` (``[H2O]``); everything else by ``@name``."""
    if isinstance(column_or_axis, _Column):
        axis_id, role = column_or_axis.axis_id, column_or_axis.role
        species_label = column_or_axis.species is not None and bool(prop.get("label"))
    else:
        axis_id, role, species_label = column_or_axis, AxisRole.CONSTANT, False
    label = doc.attribute(prop, "label" if species_label else "name")
    return AxisDeclaration(axis_id=axis_id, role=role, quantity_kind=quantity, label_raw=label.raw, label_ref=label.ref)


def _rows(doc: _Doc, group: ElementTree.Element, column_ids: set[str]) -> _Rows:
    rows = group.findall("dataPoint")
    if not rows:
        raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, "the data group has no data points")
    width = max(4, len(str(len(rows))))
    out: _Rows = []
    for number, row in enumerate(rows, start=1):
        cells = {cell.tag: cell for cell in row}
        if set(cells) != column_ids or len(cells) != len(row) or row.attrib:
            raise RespecthRefusal(
                RespecthRefusalReason.INCOMPLETE_RECORD,
                f"{doc.path(row)} holds {sorted(cell.tag for cell in row)}, "
                f"not exactly the columns {sorted(column_ids)}",
            )
        out.append((f"p{number:0{width}d}", cells))
    return out


def _series(
    doc: _Doc, kind: RespecthExperimentKind, mapping: _Mapping, rows: _Rows, uncertainties: Mapping[str, Uncertainty]
) -> Series:
    axes = [_axis(doc, column, column.quantity, column.element) for column in mapping.columns]
    constants: list[Coordinate] = []
    for axis_id, (prop, quantity) in mapping.constants.items():
        axes.append(_axis(doc, axis_id, quantity, prop))
        value = _measured(doc.text(_one(prop, "value")), doc.attribute(prop, "units"), quantity, _TABLE)
        constants.append(
            Coordinate(axis_id=axis_id, value=value, uncertainty=uncertainties.get(axis_id, _no_uncertainty()))
        )
    mixture_columns = [column for column in mapping.columns if kind is _LBV and column.species is not None]
    points: list[DataPoint] = []
    for point_id, cells in rows:
        coordinates: list[Coordinate] = []
        observations: list[Observation] = []
        for column in mapping.columns:
            value = _measured(doc.text(cells[column.column_id]), column.units, column.quantity, _TABLE)
            uncertainty = uncertainties.get(column.axis_id, _no_uncertainty())
            if column.role is AxisRole.OBSERVATION:
                observations.append(Observation(axis_id=column.axis_id, value=value, uncertainty=uncertainty))
            else:
                coordinates.append(Coordinate(axis_id=column.axis_id, value=value, uncertainty=uncertainty))
        points.append(
            DataPoint(
                point_id=point_id,
                coordinates=tuple(sorted(coordinates, key=lambda c: c.axis_id)),
                observations=tuple(sorted(observations, key=lambda o: o.axis_id)),
                composition=_point_composition(doc, mixture_columns, cells)
                if mixture_columns
                else Absent(reason=AbsenceReason.SAME_AS_DATASET),
            )
        )
    return Series(
        series_id=_SERIES_IDS[kind],
        source_form=SourceForm.STRUCTURED_RECORD,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=tuple(sorted(axes, key=lambda axis: axis.axis_id)),
        constants=tuple(sorted(constants, key=lambda c: c.axis_id)),
        points=tuple(points),
        digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _check_format(doc: _Doc) -> RecordText:
    if doc.root.tag != "experiment":
        raise RespecthRefusal(
            RespecthRefusalReason.MALFORMED_XML, f"root element is <{doc.root.tag}>, not <experiment>"
        )
    major = (_one(_one(doc.root, "ReSpecThVersion"), "major").text or "").strip()
    if major != _SUPPORTED_RKD_MAJOR:
        raise RespecthRefusal(
            RespecthRefusalReason.UNSUPPORTED_FORMAT_VERSION, f"ReSpecThVersion major {major!r}; only 2.x is parsed"
        )
    return doc.text(_one(doc.root, "experimentType"))


def read_experiment_type(member_bytes: bytes) -> str:
    """The member's ``experimentType``, stripped, after the same well-formedness gate the
    parsers apply.

    Raises:
        RespecthRefusal: The bytes are not a readable RKD 2.x ``<experiment>``.
    """
    return _check_format(_Doc.of(_parse_xml(member_bytes))).raw.strip()


def _build(member_bytes: bytes, pin: ArchivePin) -> RespecthSeriesRecord:
    doc = _Doc.of(_parse_xml(member_bytes))
    experiment_type = _check_format(doc)
    kind = EXPERIMENT_KINDS.get(experiment_type.raw.strip())
    if kind is None:
        why = REFUSED_EXPERIMENT_TYPES.get(experiment_type.raw.strip(), "no parser of this lane maps it")
        raise RespecthRefusal(
            RespecthRefusalReason.UNMAPPED_EXPERIMENT_TYPE, f"experimentType {experiment_type.raw!r}: {why}"
        )
    unknown = sorted({child.tag for child in doc.root} - _TOP_LEVEL_TAGS)
    if unknown:
        raise RespecthRefusal(RespecthRefusalReason.UNMAPPED_PROPERTY, f"unmapped top-level element(s) {unknown}")
    try:
        apparatus = _apparatus(doc, kind)
        timeshift = _timeshift(doc, kind)
        mapping = _Mapping()
        _read_common(doc, mapping)
        groups = doc.root.findall("dataGroup")
        if len(groups) != 1:
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_DATA_GROUP, f"expected one data group, found {len(groups)}"
            )
        _read_columns(doc, kind, groups[0], mapping)
        _check_required(kind, mapping)
        column_ids = {column.column_id for column in mapping.columns}
        column_ids |= {doc.attribute(prop, "id").raw for prop in mapping.uncertainty_columns}
        rows = _rows(doc, groups[0], column_ids)
        uncertainties, statements = _uncertainties(doc, kind, mapping)
        series = _series(doc, kind, mapping, rows, uncertainties)
        node = SourceNode(
            node_id=RECORD_NODE_ID,
            kind=SourceNodeKind.DATABASE_RECORD,
            sha256=hashlib.sha256(member_bytes).hexdigest(),
            parent_node_id=None,
            origin=ArchiveOrigin(archive_sha256=pin.archive_sha256, member_display_path=pin.member_path),
            extraction=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            glyph_health=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            verification=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
        envelope = DatasetEnvelope(
            source_graph=SourceGraph(nodes=(node,)),
            composition=mapping.initial_composition
            if mapping.initial_composition is not None
            else Absent(reason=AbsenceReason.NOT_APPLICABLE, note="the mixture is stated per point"),
            series=(series,),
            conversion_tables=(_embedded_table(_TABLE),),
            table_inventories=(),
            ooxml_table_inventories=(),
            figure_digitizations=(),
        )
        species_columns = (
            SpeciesColumn(axis_id=column.axis_id, species=column.species)
            for column in mapping.columns
            if column.species is not None
        )
        return RespecthSeriesRecord(
            kind=kind,
            experiment_type=experiment_type,
            archive=pin,
            record_doi=doc.text(_one(doc.root, "fileDOI")),
            paper=_paper(doc),
            apparatus=apparatus,
            species_columns=tuple(sorted(species_columns, key=lambda column: column.axis_id)),
            uncertainty_statements=statements,
            reported_uncertainties=_reported_uncertainty_columns(doc, mapping, rows),
            timeshift=timeshift,
            column_source_types=tuple(sorted(mapping.source_types.items())),
            envelope=envelope,
        )
    except ValidationError as exc:
        raise RespecthRefusal(RespecthRefusalReason.SCHEMA_REJECTED, str(exc)) from exc


def parse_series_record(member_bytes: bytes, archive: PinnedArchive, member_path: str) -> RespecthSeriesRecord:
    """Map one RKD flame-speed or speciation member into a :class:`RespecthSeriesRecord`.

    Args:
        member_bytes: The member's exact bytes, as read (and verified) out of ``archive``.
        archive: The pinned archive the member came from.
        member_path: The member's path inside the archive (display only; the sha is identity).

    Raises:
        RespecthRefusal: Anything this lane cannot map without guessing -- including an
            ignition-delay member (that is :func:`carmel.services.respecth.parse_idt_record`'s)
            and any experiment type outside :data:`EXPERIMENT_KINDS`. Nothing is returned
            partially.
    """
    pin = ArchivePin(
        archive_name=archive.name,
        osf_file_id=archive.osf_file_id,
        osf_version=archive.osf_version,
        archive_sha256=archive.sha256,
        member_path=member_path,
    )
    return _build(member_bytes, pin)


# --------------------------------------------------------------------------- replay


def replay_series_record(record: RespecthSeriesRecord, member_bytes: bytes) -> RecordReplayReport:
    """Re-derive ``record`` from ``member_bytes``.

    Checks, in order: the bytes hash to the record node's sha256; every grounded string
    re-evaluates by XPath to exactly its recorded text
    (:func:`~carmel.services.respecth.replay_grounded_text`); and re-mapping the bytes under the
    record's own archive pin yields exactly this record -- so a mapped fact (device class,
    species, uncertainty kind, timeshift type, axis role) its text does not support is a finding
    too. ``verified`` is true only when at least one pair was checked and nothing disagreed.
    """
    actual_sha = hashlib.sha256(member_bytes).hexdigest()
    if actual_sha != record.member_sha256:
        return RecordReplayReport(
            verified=False,
            checked=0,
            findings=(f"member bytes hash to {actual_sha}, not the recorded {record.member_sha256}",),
        )
    try:
        root = _parse_xml(member_bytes)
        rebuilt = _build(member_bytes, record.archive)
    except RespecthRefusal as exc:
        return RecordReplayReport(verified=False, checked=0, findings=(f"the member does not re-map: {exc}",))
    checked, findings = replay_grounded_text(record, root)
    findings.extend(
        f"{name} does not re-derive from the member"
        for name in type(record).model_fields
        if getattr(rebuilt, name) != getattr(record, name)
    )
    return RecordReplayReport(verified=checked > 0 and not findings, checked=checked, findings=tuple(findings))
