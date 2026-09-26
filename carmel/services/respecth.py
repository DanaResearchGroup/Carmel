# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""ReSpecTh Kinetics Data (RKD v2.x) ignition-delay records -> :class:`DatasetEnvelope`, and back.

Carmel's first curated-database route. One RKD member XML (see
:mod:`carmel.services.respecth_archive` for how the archive holding it is pinned) is parsed
NATIVELY -- no ``respth2ck``/``ck2respth``, which the ReSpecTh maintainers document as lossy in
both directions -- into a :class:`RespecthIdtRecord`: a :class:`DatasetEnvelope` plus the
record-level facts the envelope has no slot for.

Provenance: the envelope's source graph holds ONE ``DATABASE_RECORD`` node whose ``sha256`` is
the member's bytes and whose :class:`ArchiveOrigin` names the pinned zip. Every stored value,
unit and label -- and every record-level fact -- carries a :class:`SourceRef` with a positional
:class:`XPathLocator` into that member (``/experiment/dataGroup[1]/dataPoint[3]/x2[1]``,
``.../property[2]/@units``). :func:`replay_idt_record` re-hashes the member bytes, re-evaluates
every path, and compares the text, so "replay or refuse" needs no locator beyond XPath.

Everything this lane does not map is a typed :class:`RespecthRefusal` with no partial output --
never a guess:

- apparatus: a shock tube in REFLECTED-shock mode, or a rapid compression machine with no mode
  (:data:`APPARATUS_DEVICE_CLASSES`). A shock tube that states NO mode is mapped with reflected
  shock ASSUMED (operator ruling, :data:`ASSUMED_APPARATUS_MODES`): the record carries
  ``mode_basis=ASSUMED`` and an Absent ``mode_raw``, never a stated mode, and
  :func:`replay_idt_record` lists the assumption. Any other stated mode (incident shock) is refused.
- ignition definition: target ``p``/``OH``/``OH*``/``OHEX`` x criterion ``d/dt max``/``max``/
  ``relative concentration`` (the last with its ``amount`` in ``unitless``).
- units: whatever :data:`carmel.services.units.TABLE_V2` cannot bind (e.g. ``Torr``).
- properties and data groups: anything outside T, P, initial composition, the ignition delay,
  and its evaluated relative standard deviation -- except an RCM volume-history group
  (``time`` + ``volume``), which is recorded as skipped rather than ingested.

No quality/"unreliable" flag is carried: no member of either pinned archive has one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    Maybe,
    MeasuredValue,
    Observation,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    ValueOrigin,
    XPathLocator,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal, canonical_json_bytes
from carmel.services.numeric import GlyphHealth, SourceContext, Unresolvable, normalize_numeric_span
from carmel.services.respecth_archive import PinnedArchive, RespecthError
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import QuantityKind

__all__ = [
    "APPARATUS_DEVICE_CLASSES",
    "ASSUMED_APPARATUS_MODES",
    "MIN_IGNITION_TEMPERATURE_K",
    "RECORD_NODE_ID",
    "Apparatus",
    "ApparatusModeBasis",
    "ArchivePin",
    "IgnitionCriterion",
    "IgnitionDefinition",
    "IgnitionTarget",
    "PaperReference",
    "RecordReplayReport",
    "RcmConditions",
    "RcmVolumeHistory",
    "RecordText",
    "ReferenceDoiTrust",
    "RespecthIdtRecord",
    "RespecthRefusal",
    "RespecthRefusalReason",
    "SkippedDataGroup",
    "UncertaintyDefinition",
    "evaluate_xpath",
    "parse_idt_record",
    "replay_idt_record",
]

RECORD_NODE_ID = "record"
"""The single source-graph node every RKD envelope holds."""

#: The conversion table this lane binds against: TABLE_V1 plus the RKD spellings
#: ``"mole fraction"`` and ``"unitless"``.
_TABLE = units.TABLE_V2

#: A database field is read as-is: no glyph damage is possible in parsed XML text, so the
#: healthy (all-false) assessment is the honest one. Any repair the numeral grammar would
#: still apply is refused below rather than recorded.
_CLEAN_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)

_SUPPORTED_RKD_MAJOR = "2"
_EXPERIMENT_TYPE = "ignition delay measurement"
_PLACEHOLDER_DOI_RE = re.compile(r"referenceDOI\s+is\s+just\s+for\s+sorting\s+purposes", re.IGNORECASE)


class RespecthRefusalReason(StrEnum):
    """Why a member was refused. Every reason is a refusal with zero partial output."""

    MALFORMED_XML = "malformed_xml"
    UNSUPPORTED_FORMAT_VERSION = "unsupported_format_version"
    NOT_IGNITION_DELAY = "not_ignition_delay"
    INCOMPLETE_RECORD = "incomplete_record"
    UNMAPPED_APPARATUS = "unmapped_apparatus"
    UNMAPPED_IGNITION_DEFINITION = "unmapped_ignition_definition"
    UNMAPPED_UNIT = "unmapped_unit"
    UNMAPPED_PROPERTY = "unmapped_property"
    UNMAPPED_DATA_GROUP = "unmapped_data_group"
    UNREADABLE_VALUE = "unreadable_value"
    SCHEMA_REJECTED = "schema_rejected"
    RCM_PRE_COMPRESSION_CONDITIONS = "rcm_pre_compression_conditions"
    """The RCM record's P/T start a volume history that compresses: they are pre-compression
    conditions, and the ignition state needs a volume-history simulation (slice S4)."""
    RCM_CONDITIONS_UNIDENTIFIED = "rcm_conditions_unidentified"
    """No volume history covers every point, so P/T cannot be told apart from pre-compression ones."""
    IMPLAUSIBLE_IGNITION_TEMPERATURE = "implausible_ignition_temperature"
    """A condition temperature below :data:`MIN_IGNITION_TEMPERATURE_K` -- a backstop, whatever the apparatus."""


class RespecthRefusal(RespecthError):
    """A typed refusal: this member cannot be mapped without guessing."""

    def __init__(self, reason: RespecthRefusalReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


class IgnitionTarget(StrEnum):
    """The observable an RKD ``ignitionType/@target`` names (verbatim spelling as value)."""

    PRESSURE = "p"
    OH = "OH"
    OH_STAR = "OH*"
    OHEX = "OHEX"
    """ReSpecTh's species key for electronically excited OH -- kept distinct from ``OH*``
    because the records spell them differently and this lane does not merge spellings."""


class IgnitionCriterion(StrEnum):
    """How the target's trace defines the ignition instant (``ignitionType/@type``)."""

    MAX_SLOPE = "d/dt max"
    PEAK = "max"
    RELATIVE_CONCENTRATION = "relative concentration"
    """The instant the target reaches ``amount`` times its peak; ``amount`` is required."""


#: The explicit apparatus table: (``apparatus/kind``, ``apparatus/mode`` or ``None``) -> device
#: class. Anything not listed is refused.
APPARATUS_DEVICE_CLASSES: Mapping[tuple[str, str | None], ReactorType] = {
    ("shock tube", "reflected shock"): ReactorType.SHOCK_TUBE,
    ("shock tube", None): ReactorType.SHOCK_TUBE,
    ("rapid compression machine", None): ReactorType.RCM,
}

#: Keys of :data:`APPARATUS_DEVICE_CLASSES` whose record states no mode but is mapped as if it
#: stated this one -- an operator ruling (the ReSpecTh shock-tube IDT records that omit the mode
#: are taken as reflected-shock measurements), carried as ``ApparatusModeBasis.ASSUMED``.
ASSUMED_APPARATUS_MODES: Mapping[tuple[str, None], str] = {
    ("shock tube", None): "reflected shock",
}

#: Species whose mixture role is fixed by what the species IS in this fuel family. A species
#: absent here keeps an explicit ``Absent`` role rather than a guessed one.
_SPECIES_ROLES: Mapping[str, ComponentRole] = {
    "H2": ComponentRole.FUEL,
    "CO": ComponentRole.FUEL,
    "O2": ComponentRole.OXIDIZER,
    "Ar": ComponentRole.DILUENT,
    "He": ComponentRole.DILUENT,
    "N2": ComponentRole.DILUENT,
}

#: RKD property name -> (series axis id, quantity).
_CONDITION_PROPERTIES: Mapping[str, tuple[str, QuantityKind]] = {
    "temperature": ("temperature", QuantityKind.TEMPERATURE),
    "pressure": ("pressure", QuantityKind.PRESSURE),
}
_IGNITION_DELAY = "ignition delay"
_IGNITION_AXIS_ID = "ignition_delay"
_SERIES_ID = "ignition_delay"
_VOLUME_HISTORY_COLUMNS = frozenset({"time", "volume"})

#: Below this, a condition temperature is refused as an ignition state (a pre-compression RCM
#: state that slipped through would sit near room temperature).
MIN_IGNITION_TEMPERATURE_K = Decimal(500)


# --------------------------------------------------------------------------- record model


class RecordText(BaseModel):
    """A verbatim string read from the member, with where it was read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: str = Field(min_length=1)
    ref: SourceRef


class ReferenceDoiTrust(StrEnum):
    """Whether the record's ``referenceDOI`` may be emitted as the paper's citation."""

    CITED = "cited"
    """No disclaimer in the record: the DOI is what the record says its data came from."""

    PLACEHOLDER = "placeholder"
    """The record's own comment says the DOI is only for sorting inside ReSpecTh. It is kept
    for audit and NEVER emitted as a citation."""


class PaperReference(BaseModel):
    """The paper DOI a record cites, with an explicit trust marker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doi: RecordText
    trust: ReferenceDoiTrust
    placeholder_evidence: Maybe[RecordText]
    """The comment that marks the DOI a placeholder -- present exactly when ``trust`` is
    ``PLACEHOLDER``, so the marker is itself replayable."""

    @model_validator(mode="after")
    def _evidence_matches_trust(self) -> PaperReference:
        has_evidence = not isinstance(self.placeholder_evidence, Absent)
        if has_evidence != (self.trust is ReferenceDoiTrust.PLACEHOLDER):
            raise ValueError("placeholder_evidence must be present exactly when trust is PLACEHOLDER")
        return self


class IgnitionDefinition(BaseModel):
    """The record's ignition definition, mapped and verbatim side by side."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: IgnitionTarget
    target_raw: RecordText
    criterion: IgnitionCriterion
    criterion_raw: RecordText
    amount: Maybe[RecordText]
    """The fraction of the target's peak, for ``RELATIVE_CONCENTRATION`` only."""
    amount_units: Maybe[RecordText]


class ApparatusModeBasis(StrEnum):
    """Where the apparatus mode the device class rests on comes from."""

    STATED = "stated"
    """``mode_raw`` grounds it: the record states the mode."""

    ASSUMED = "assumed"
    """The record states NO mode; ``assumed_mode`` is an operator ruling, not a reading."""

    NOT_APPLICABLE = "not_applicable"
    """The device has no mode (an RCM)."""


class Apparatus(BaseModel):
    """The apparatus the record names, and the device class the explicit table maps it to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_class: ReactorType
    kind_raw: RecordText
    mode_raw: Maybe[RecordText]
    """The mode the record states -- Absent whenever it states none, including when one is assumed."""
    mode_basis: ApparatusModeBasis
    assumed_mode: Maybe[str]
    """The mode mapped WITHOUT being stated -- present exactly when ``mode_basis`` is ``ASSUMED``."""

    @model_validator(mode="after")
    def _basis_matches_mode(self) -> Apparatus:
        stated = not isinstance(self.mode_raw, Absent)
        assumed = not isinstance(self.assumed_mode, Absent)
        expected = (
            ApparatusModeBasis.STATED
            if stated and not assumed
            else ApparatusModeBasis.ASSUMED
            if assumed and not stated
            else ApparatusModeBasis.NOT_APPLICABLE
            if not (stated or assumed)
            else None
        )
        if self.mode_basis is not expected:
            raise ValueError(
                "mode_basis must be STATED exactly when mode_raw is present, ASSUMED exactly when only "
                "assumed_mode is, and NOT_APPLICABLE when neither is"
            )
        return self

    @property
    def assumption(self) -> str | None:
        """The assumption this mapping rests on, in words, or ``None`` when nothing is assumed."""
        if isinstance(self.assumed_mode, Absent):
            return None
        return f"apparatus mode {self.assumed_mode!r} is ASSUMED: {self.apparatus_xpath} states no mode"

    @property
    def apparatus_xpath(self) -> str:
        """The ``<apparatus>`` element's path, read off the ``kind`` ref."""
        locator = self.kind_raw.ref.locator
        return locator.xpath.rsplit("/", 1)[0] if isinstance(locator, XPathLocator) else str(locator)


class UncertaintyDefinition(BaseModel):
    """The words the record states its ignition-delay uncertainty in -- what grounds the
    envelope's ``STD_DEV``/``RELATIVE`` claim, which :class:`Uncertainty` cannot cite itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name_raw: RecordText
    """``@name``: ``evaluated standard deviation`` -- grounds ``kind=STD_DEV``."""
    basis_raw: RecordText
    """``@kind``: ``relative`` -- grounds ``basis=RELATIVE``."""
    reference_raw: RecordText
    """``@reference``: ``ignition delay`` -- which quantity the deviation is of."""
    method_raw: Maybe[RecordText]
    """``@method``: ``statistical scatter`` (measured on these data) or ``generic
    uncertainty`` (assigned by the database)."""


class ArchivePin(BaseModel):
    """Which pinned archive the member was read from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_name: str = Field(min_length=1)
    osf_file_id: str = Field(min_length=1)
    osf_version: int = Field(ge=1)
    archive_sha256: str = Field(min_length=64, max_length=64)
    member_path: str = Field(min_length=1)


class RcmVolumeHistory(BaseModel):
    """One volume-time history, and the two cells its verdict rests on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: RecordText
    point_link: RecordText
    """``@dataPointLink``: ``all``, or the 1-based points it applies to (``1;2;``)."""
    first_volume: RecordText
    minimum_volume: RecordText
    """The smallest volume anywhere in the history; not below ``first_volume`` means no compression."""


class RcmConditions(BaseModel):
    """Why an RCM record's P/T are taken as END-of-compression conditions (RKD v2.5: they are
    the state at the start of the volume history, which may begin before compression or at its
    end). Mapped only when every point is covered by a history with no compression phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["end_of_compression"] = "end_of_compression"
    histories: tuple[RcmVolumeHistory, ...] = Field(min_length=1)


class SkippedDataGroup(BaseModel):
    """A data group the record carries but this lane does not ingest (an RCM volume history)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    column_names: tuple[str, ...]


class RespecthIdtRecord(BaseModel):
    """One RKD ignition-delay member, mapped. ``envelope`` holds every number; the rest are
    the record-level facts a :class:`DatasetEnvelope` has no slot for, each grounded the same
    way."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["respecth"] = "respecth"
    archive: ArchivePin
    record_doi: RecordText
    """The ReSpecTh ``fileDOI`` -- the citable, stable id of THIS record."""
    paper: Maybe[PaperReference]
    apparatus: Apparatus
    ignition: IgnitionDefinition
    uncertainty: Maybe[UncertaintyDefinition]
    column_source_types: tuple[tuple[str, RecordText], ...]
    """Per series axis, the RKD ``@sourcetype`` (``reported``/``digitized``/...)."""
    rcm_conditions: Maybe[RcmConditions]
    """Present exactly for an RCM: the evidence its P/T are end-of-compression conditions."""
    skipped_data_groups: tuple[SkippedDataGroup, ...]
    envelope: DatasetEnvelope

    @model_validator(mode="after")
    def _node_matches_pin(self) -> RespecthIdtRecord:
        node = self.envelope.source_graph.node(RECORD_NODE_ID)
        if node.kind is not SourceNodeKind.DATABASE_RECORD or not isinstance(node.origin, ArchiveOrigin):
            raise ValueError("the record node must be a DATABASE_RECORD with an ArchiveOrigin")
        if (node.origin.archive_sha256, node.origin.member_display_path) != (
            self.archive.archive_sha256,
            self.archive.member_path,
        ):
            raise ValueError("the record node's ArchiveOrigin disagrees with the archive pin")
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


# --------------------------------------------------------------------------- XML + XPath

_NAME = r"[A-Za-z_][A-Za-z0-9_.\-]*"
_XPATH_RE = re.compile(rf"/(?P<root>{_NAME})(?P<steps>(?:/{_NAME}\[[1-9][0-9]*\])*)(?:/@(?P<attr>{_NAME}))?")
_STEP_RE = re.compile(rf"/({_NAME})\[([1-9][0-9]*)\]")


_ENCODING_DECL_RE = re.compile(rb"""^(?:\xef\xbb\xbf)?<\?xml[^>]*?\bencoding\s*=\s*["']([^"']*)["']""")


def _parse_xml(data: bytes) -> ElementTree.Element:
    """Parse member bytes, refusing a DOCTYPE (no entity expansion is ever possible).

    The DOCTYPE/ENTITY scan is a byte scan, so it is only sound on UTF-8: anything else --
    a UTF-16/32 BOM or NUL-interleaved text, bytes that do not decode, or an encoding
    declaration naming another charset -- is refused before the scan, never re-decoded.
    """
    declared = _ENCODING_DECL_RE.match(data)
    if declared is not None and declared.group(1).lower() not in {b"utf-8", b"utf8"}:
        raise RespecthRefusal(
            RespecthRefusalReason.MALFORMED_XML,
            f"the member declares encoding {declared.group(1).decode('ascii', 'replace')!r}; only UTF-8 is read",
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, f"the member is not UTF-8: {exc}") from exc
    if b"\x00" in data:
        raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, "the member holds NUL bytes (UTF-16/32?); refused")
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, "the member declares a DOCTYPE/ENTITY; refused")
    try:
        return ElementTree.fromstring(data)  # noqa: S314 - DOCTYPE/ENTITY refused above
    except (ElementTree.ParseError, ValueError) as exc:
        raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, f"the member is not well-formed XML: {exc}") from exc


def evaluate_xpath(root: ElementTree.Element, xpath: str) -> str | None:
    """Evaluate the lane's positional XPath subset; the addressed text, or ``None`` if nothing.

    The grammar is ``/root(/tag[n])*(/@attr)?`` with 1-based same-tag positions -- a subset
    of XPath 1.0 that addresses exactly one element or attribute. An element's text is its
    ``.text`` (``""`` when empty).
    """
    match = _XPATH_RE.fullmatch(xpath)
    if match is None or match.group("root") != root.tag:
        return None
    element = root
    for tag, position in _STEP_RE.findall(match.group("steps")):
        same_tag = [child for child in element if child.tag == tag]
        index = int(position) - 1
        if index >= len(same_tag):
            return None
        element = same_tag[index]
    attribute = match.group("attr")
    if attribute is not None:
        return element.get(attribute)
    return element.text or ""


@dataclass(frozen=True)
class _Doc:
    """A parsed member plus what is needed to address any node in it."""

    root: ElementTree.Element
    parents: Mapping[ElementTree.Element, ElementTree.Element]

    @classmethod
    def of(cls, root: ElementTree.Element) -> _Doc:
        return cls(root=root, parents={child: parent for parent in root.iter() for child in parent})

    def path(self, element: ElementTree.Element) -> str:
        steps: list[str] = []
        while element is not self.root:
            parent = self.parents[element]
            same_tag = [child for child in parent if child.tag == element.tag]
            steps.append(f"{element.tag}[{same_tag.index(element) + 1}]")
            element = parent
        return "/" + "/".join([self.root.tag, *reversed(steps)])

    def ref(self, element: ElementTree.Element, attribute: str | None = None) -> SourceRef:
        xpath = self.path(element) if attribute is None else f"{self.path(element)}/@{attribute}"
        return SourceRef(node_id=RECORD_NODE_ID, locator=XPathLocator(xpath=xpath))

    def text(self, element: ElementTree.Element) -> RecordText:
        raw = element.text or ""
        if not raw.strip():
            raise RespecthRefusal(
                RespecthRefusalReason.INCOMPLETE_RECORD, f"{self.path(element)} is empty; nothing to ground"
            )
        return RecordText(raw=raw, ref=self.ref(element))

    def attribute(self, element: ElementTree.Element, name: str) -> RecordText:
        raw = element.get(name)
        if raw is None or not raw.strip():
            raise RespecthRefusal(
                RespecthRefusalReason.INCOMPLETE_RECORD, f"{self.path(element)} has no @{name}; nothing to ground"
            )
        return RecordText(raw=raw, ref=self.ref(element, name))


def _one(parent: ElementTree.Element, tag: str) -> ElementTree.Element:
    found = parent.findall(tag)
    if len(found) != 1:
        raise RespecthRefusal(
            RespecthRefusalReason.INCOMPLETE_RECORD,
            f"expected exactly one <{tag}> under <{parent.tag}>, found {len(found)}",
        )
    return found[0]


def _optional_one(parent: ElementTree.Element, tag: str) -> ElementTree.Element | None:
    found = parent.findall(tag)
    if len(found) > 1:
        raise RespecthRefusal(
            RespecthRefusalReason.INCOMPLETE_RECORD,
            f"expected at most one <{tag}> under <{parent.tag}>, found {len(found)}",
        )
    return found[0] if found else None


# --------------------------------------------------------------------------- mapping


def _repair_dependency() -> SemanticDependencyUse:
    return SemanticDependencyUse(
        dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
        content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
        input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _measured(
    value: RecordText, unit: RecordText, quantity: QuantityKind, table: units.ConversionTable = _TABLE
) -> MeasuredValue:
    """Bind a verbatim numeral to its verbatim unit under :data:`_TABLE`, or refuse."""
    try:
        unit_normalized = units.normalize_unit(quantity, unit.raw, table=table)
    except units.UnitError as exc:
        raise RespecthRefusal(
            RespecthRefusalReason.UNMAPPED_UNIT, f"unit {unit.raw!r} at {unit.ref.locator} for {quantity.value}: {exc}"
        ) from exc
    normalized = normalize_numeric_span(
        value.raw, source_context=SourceContext.OPERATOR_RAW, glyph_health=_CLEAN_GLYPH_HEALTH
    )
    if isinstance(normalized, Unresolvable) or normalized.repairs:
        reason = normalized.reason if isinstance(normalized, Unresolvable) else f"needs repairs {normalized.repairs}"
        raise RespecthRefusal(
            RespecthRefusalReason.UNREADABLE_VALUE,
            f"value {value.raw!r} at {value.ref.locator} is not a clean numeral: {reason}",
        )
    try:
        canonical = canonical_decimal(normalized.text)
    except CanonicalDecimalError as exc:
        raise RespecthRefusal(
            RespecthRefusalReason.UNREADABLE_VALUE, f"value {value.raw!r} at {value.ref.locator}: {exc}"
        ) from exc
    return MeasuredValue(
        raw_text=value.raw,
        canonical_decimal_value=canonical,
        repairs=(),
        repair_dependency=_repair_dependency(),
        quantity_kind=quantity,
        unit_raw=unit.raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=table.sha256,
        value_ref=value.ref,
        unit_ref=unit.ref,
    )


def _apparatus(doc: _Doc) -> Apparatus:
    apparatus = _one(doc.root, "apparatus")
    kind = doc.text(_one(apparatus, "kind"))
    mode_element = _optional_one(apparatus, "mode")
    mode: RecordText | None = None
    if mode_element is not None and (mode_element.text or "").strip():
        mode = doc.text(mode_element)
    extra = {child.tag for child in apparatus} - {"kind", "mode"}
    key = (kind.raw.strip(), mode.raw.strip() if mode is not None else None)
    device_class = APPARATUS_DEVICE_CLASSES.get(key)
    if device_class is None or extra:
        raise RespecthRefusal(
            RespecthRefusalReason.UNMAPPED_APPARATUS,
            f"apparatus kind={key[0]!r} mode={key[1]!r} (extra fields {sorted(extra)}) is not in the apparatus "
            f"table {sorted(APPARATUS_DEVICE_CLASSES, key=str)}",
        )
    if mode is not None:
        return Apparatus(
            device_class=device_class,
            kind_raw=kind,
            mode_raw=mode,
            mode_basis=ApparatusModeBasis.STATED,
            assumed_mode=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
    assumed = ASSUMED_APPARATUS_MODES.get((key[0], None))
    if assumed is not None:
        return Apparatus(
            device_class=device_class,
            kind_raw=kind,
            mode_raw=Absent(
                reason=AbsenceReason.NOT_REPORTED_HERE, note=f"no mode stated; {assumed!r} is assumed, not read"
            ),
            mode_basis=ApparatusModeBasis.ASSUMED,
            assumed_mode=assumed,
        )
    return Apparatus(
        device_class=device_class,
        kind_raw=kind,
        mode_raw=Absent(reason=AbsenceReason.NOT_APPLICABLE, note="no mode stated"),
        mode_basis=ApparatusModeBasis.NOT_APPLICABLE,
        assumed_mode=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _ignition(doc: _Doc) -> IgnitionDefinition:
    element = _one(doc.root, "ignitionType")
    target = doc.attribute(element, "target")
    criterion = doc.attribute(element, "type")
    targets = [part.strip() for part in target.raw.split(";") if part.strip()]
    unmapped = RespecthRefusalReason.UNMAPPED_IGNITION_DEFINITION
    if len(targets) != 1 or targets[0] not in {member.value for member in IgnitionTarget}:
        raise RespecthRefusal(
            unmapped, f"ignition target {target.raw!r} is not one of {[m.value for m in IgnitionTarget]}"
        )
    if criterion.raw not in {member.value for member in IgnitionCriterion}:
        raise RespecthRefusal(
            unmapped, f"ignition type {criterion.raw!r} is not one of {[m.value for m in IgnitionCriterion]}"
        )
    mapped_criterion = IgnitionCriterion(criterion.raw)
    extra = set(element.attrib) - {"target", "type", "amount", "units"}
    if extra:
        raise RespecthRefusal(unmapped, f"ignitionType carries unmapped attributes {sorted(extra)}")
    amount: Maybe[RecordText] = Absent(reason=AbsenceReason.NOT_APPLICABLE)
    amount_units: Maybe[RecordText] = Absent(reason=AbsenceReason.NOT_APPLICABLE)
    if mapped_criterion is IgnitionCriterion.RELATIVE_CONCENTRATION:
        amount = doc.attribute(element, "amount")
        amount_units = doc.attribute(element, "units")
        try:
            fraction = canonical_decimal(amount.raw)
        except CanonicalDecimalError as exc:
            raise RespecthRefusal(unmapped, f"relative-concentration amount {amount.raw!r} is not a decimal") from exc
        if amount_units.raw != "unitless" or not 0 < float(fraction) <= 1:
            raise RespecthRefusal(
                unmapped,
                f"relative-concentration amount {amount.raw!r} {amount_units.raw!r} is not a unitless fraction",
            )
    elif "amount" in element.attrib or "units" in element.attrib:
        raise RespecthRefusal(unmapped, f"ignition type {criterion.raw!r} carries an amount it does not define")
    return IgnitionDefinition(
        target=IgnitionTarget(targets[0]),
        target_raw=target,
        criterion=mapped_criterion,
        criterion_raw=criterion,
        amount=amount,
        amount_units=amount_units,
    )


def _paper(doc: _Doc) -> Maybe[PaperReference]:
    link = _one(doc.root, "bibliographyLink")
    doi_element = _optional_one(link, "referenceDOI")
    if doi_element is None or not (doi_element.text or "").strip():
        return Absent(reason=AbsenceReason.NOT_REPORTED_HERE, note="the record names no referenceDOI")
    doi = doc.text(doi_element)
    evidence = next((doc.text(comment) for comment in _placeholder_comments(doc.root)), None)
    if evidence is None:
        return PaperReference(
            doi=doi, trust=ReferenceDoiTrust.CITED, placeholder_evidence=Absent(reason=AbsenceReason.NOT_APPLICABLE)
        )
    return PaperReference(doi=doi, trust=ReferenceDoiTrust.PLACEHOLDER, placeholder_evidence=evidence)


def _placeholder_comments(root: ElementTree.Element) -> Iterator[ElementTree.Element]:
    """Top-level comments saying the referenceDOI is only a sorting key inside ReSpecTh."""
    return (comment for comment in root.findall("comment") if _PLACEHOLDER_DOI_RE.search(comment.text or ""))


@dataclass(frozen=True)
class _Common:
    constants: Mapping[str, tuple[ElementTree.Element, QuantityKind]]
    composition: Composition
    uncertainty: Maybe[Uncertainty]
    uncertainty_definition: Maybe[UncertaintyDefinition]
    source_types: Mapping[str, RecordText]


def _composition(doc: _Doc, prop: ElementTree.Element) -> Composition:
    components: list[CompositionComponent] = []
    for component in prop:
        if component.tag != "component":
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_PROPERTY, f"unexpected <{component.tag}> in composition"
            )
        species = doc.attribute(_one(component, "speciesLink"), "preferredKey")
        amount_element = _one(component, "amount")
        amount = _measured(doc.text(amount_element), doc.attribute(amount_element, "units"), QuantityKind.MOLE_FRACTION)
        if amount.unit_normalized != "1":
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_UNIT, f"component amount unit {amount.unit_raw!r} is not a mole fraction"
            )
        role = _SPECIES_ROLES.get(species.raw)
        components.append(
            CompositionComponent(
                species_raw_name=species.raw,
                amount=amount,
                role=role
                if role is not None
                else Absent(
                    reason=AbsenceReason.UNKNOWN,
                    note="the record states no role and the species is not in the lane's role table",
                ),
            )
        )
    if not components:
        raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, "initial composition lists no components")
    return Composition(
        raw_name=prop.get("name", "initial composition"),
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=Absent(
            reason=AbsenceReason.UNKNOWN, note="RKD ignition-delay records state no equivalence ratio; not derived"
        ),
        components=tuple(sorted(components, key=lambda component: component.species_raw_name)),
    )


def _uncertainty(doc: _Doc, prop: ElementTree.Element) -> tuple[Uncertainty, UncertaintyDefinition]:
    unmapped = RespecthRefusalReason.UNMAPPED_PROPERTY
    if prop.get("reference") != _IGNITION_DELAY or prop.get("kind") != "relative":
        raise RespecthRefusal(
            unmapped,
            f"evaluated standard deviation with reference={prop.get('reference')!r} kind={prop.get('kind')!r}; only a "
            "relative deviation of the ignition delay is mapped",
        )
    bound = _measured(doc.text(_one(prop, "value")), doc.attribute(prop, "units"), QuantityKind.RELATIVE_UNCERTAINTY)
    definition = UncertaintyDefinition(
        name_raw=doc.attribute(prop, "name"),
        basis_raw=doc.attribute(prop, "kind"),
        reference_raw=doc.attribute(prop, "reference"),
        method_raw=doc.attribute(prop, "method")
        if prop.get("method")
        else Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
    )
    return (
        Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.RELATIVE,
            scale=Absent(reason=AbsenceReason.UNKNOWN, note="the record does not say linear or log"),
            upper=bound,
            lower=bound,
        ),
        definition,
    )


def _common(doc: _Doc) -> _Common:
    common = _one(doc.root, "commonProperties")
    constants: dict[str, tuple[ElementTree.Element, QuantityKind]] = {}
    source_types: dict[str, RecordText] = {}
    composition: Composition | None = None
    uncertainty: Maybe[Uncertainty] = Absent(reason=AbsenceReason.UNKNOWN, note="the record states no uncertainty")
    definition: Maybe[UncertaintyDefinition] = Absent(reason=AbsenceReason.NOT_APPLICABLE)
    seen: set[str] = set()
    for prop in common:
        name = prop.get("name", "")
        if prop.tag != "property" or name in seen:
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_PROPERTY, f"unexpected or repeated common property {name!r}"
            )
        seen.add(name)
        if name in _CONDITION_PROPERTIES:
            axis_id, quantity = _CONDITION_PROPERTIES[name]
            constants[axis_id] = (prop, quantity)
            if prop.get("sourcetype"):
                source_types[axis_id] = doc.attribute(prop, "sourcetype")
        elif name == "initial composition":
            composition = _composition(doc, prop)
        elif name == "evaluated standard deviation":
            uncertainty, definition = _uncertainty(doc, prop)
        else:
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_PROPERTY,
                f"common property {name!r} ({prop.get('units')!r}) is not mapped",
            )
    if composition is None:
        raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, "the record states no initial composition")
    return _Common(
        constants=constants,
        composition=composition,
        uncertainty=uncertainty,
        uncertainty_definition=definition,
        source_types=source_types,
    )


def _data_groups(doc: _Doc) -> tuple[ElementTree.Element, tuple[SkippedDataGroup, ...]]:
    idt_groups: list[ElementTree.Element] = []
    skipped: list[SkippedDataGroup] = []
    for group in doc.root.findall("dataGroup"):
        names = tuple(prop.get("name", "") for prop in group.findall("property"))
        if _IGNITION_DELAY in names:
            idt_groups.append(group)
        elif frozenset(names) == _VOLUME_HISTORY_COLUMNS:
            skipped.append(SkippedDataGroup(group_id=group.get("id", ""), column_names=names))
        else:
            raise RespecthRefusal(
                RespecthRefusalReason.UNMAPPED_DATA_GROUP, f"data group with columns {names} is not mapped"
            )
    if len(idt_groups) != 1:
        raise RespecthRefusal(
            RespecthRefusalReason.INCOMPLETE_RECORD, f"expected one ignition-delay data group, found {len(idt_groups)}"
        )
    return idt_groups[0], tuple(skipped)


def _rcm_conditions(doc: _Doc, idt_group: ElementTree.Element) -> RcmConditions:
    """Identify an RCM record's P/T as end-of-compression conditions, or refuse."""
    point_count = len(idt_group.findall("dataPoint"))
    covered: set[int] = set()
    histories: list[RcmVolumeHistory] = []
    for group in doc.root.findall("dataGroup"):
        names = {prop.get("id", ""): prop.get("name", "") for prop in group.findall("property")}
        if frozenset(names.values()) != _VOLUME_HISTORY_COLUMNS:
            continue
        (volume_id,) = (column_id for column_id, name in names.items() if name == "volume")
        cells = [_one(row, volume_id) for row in group.findall("dataPoint")]
        if not cells:
            raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, f"{doc.path(group)} holds no data points")
        volumes = [_decimal_cell(doc, cell) for cell in cells]
        smallest = min(range(len(volumes)), key=volumes.__getitem__)
        group_id = doc.attribute(group, "id")
        if volumes[smallest] < volumes[0]:
            raise RespecthRefusal(
                RespecthRefusalReason.RCM_PRE_COMPRESSION_CONDITIONS,
                f"volume history {group_id.raw!r} compresses ({cells[0].text!r} down to {cells[smallest].text!r}), "
                "so P/T are pre-compression conditions; the ignition state needs a volume-history simulation (S4)",
            )
        link = doc.attribute(group, "dataPointLink")
        covered |= _linked_points(link.raw, point_count)
        histories.append(
            RcmVolumeHistory(
                group_id=group_id,
                point_link=link,
                first_volume=doc.text(cells[0]),
                minimum_volume=doc.text(cells[smallest]),
            )
        )
    missing = sorted(set(range(1, point_count + 1)) - covered)
    if missing:
        raise RespecthRefusal(
            RespecthRefusalReason.RCM_CONDITIONS_UNIDENTIFIED,
            f"no volume history covers point(s) {missing}, so their P/T cannot be told apart from "
            "pre-compression conditions",
        )
    return RcmConditions(histories=tuple(histories))


def _decimal_cell(doc: _Doc, cell: ElementTree.Element) -> Decimal:
    try:
        value = Decimal((cell.text or "").strip())
    except InvalidOperation:
        value = Decimal("NaN")
    if not value.is_finite():
        raise RespecthRefusal(RespecthRefusalReason.UNREADABLE_VALUE, f"{doc.path(cell)} reads {cell.text!r}")
    return value


def _linked_points(link: str, point_count: int) -> set[int]:
    if link.strip() == "all":
        return set(range(1, point_count + 1))
    parts = [part.strip() for part in link.split(";") if part.strip()]
    if not parts or not all(part.isdigit() for part in parts):
        raise RespecthRefusal(RespecthRefusalReason.UNREADABLE_VALUE, f"dataPointLink {link!r} is not readable")
    return {int(part) for part in parts}


def _check_plausible_temperature(series: Series) -> None:
    """Backstop: refuse any condition temperature below :data:`MIN_IGNITION_TEMPERATURE_K`."""
    values = [c.value for c in series.constants if c.axis_id == "temperature"]
    values += [c.value for point in series.points for c in point.coordinates if c.axis_id == "temperature"]
    for value in values:
        kelvin = units.convert(
            value.canonical_decimal_value,
            quantity=QuantityKind.TEMPERATURE,
            from_unit=value.unit_normalized,
            to_unit="K",
            table=_TABLE,
        )
        if Decimal(kelvin.exact) < MIN_IGNITION_TEMPERATURE_K:
            raise RespecthRefusal(
                RespecthRefusalReason.IMPLAUSIBLE_IGNITION_TEMPERATURE,
                f"condition temperature {value.raw_text} {value.unit_raw} is below {MIN_IGNITION_TEMPERATURE_K} K, "
                "implausible as an ignition condition",
            )


def _series(doc: _Doc, group: ElementTree.Element, common: _Common) -> tuple[Series, dict[str, RecordText]]:
    source_types = dict(common.source_types)
    columns: dict[str, tuple[str, QuantityKind, RecordText]] = {}
    axes: list[AxisDeclaration] = []
    for prop in group.findall("property"):
        name = prop.get("name", "")
        if name == _IGNITION_DELAY:
            axis_id, quantity, role = _IGNITION_AXIS_ID, QuantityKind.TIME, AxisRole.OBSERVATION
        elif name in _CONDITION_PROPERTIES:
            axis_id, quantity = _CONDITION_PROPERTIES[name]
            role = AxisRole.COORDINATE
        else:
            raise RespecthRefusal(RespecthRefusalReason.UNMAPPED_PROPERTY, f"data column {name!r} is not mapped")
        if axis_id in columns or axis_id in common.constants:
            raise RespecthRefusal(RespecthRefusalReason.UNMAPPED_PROPERTY, f"{name!r} is stated more than once")
        column_id = doc.attribute(prop, "id").raw
        label = doc.attribute(prop, "name")
        columns[column_id] = (axis_id, quantity, doc.attribute(prop, "units"))
        if prop.get("sourcetype"):
            source_types[axis_id] = doc.attribute(prop, "sourcetype")
        axes.append(
            AxisDeclaration(
                axis_id=axis_id, role=role, quantity_kind=quantity, label_raw=label.raw, label_ref=label.ref
            )
        )
    if not any(axis.role is AxisRole.COORDINATE for axis in axes):
        raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, "no condition varies across the data points")

    constants: list[Coordinate] = []
    for axis_id, (prop, quantity) in common.constants.items():
        label = doc.attribute(prop, "name")
        axes.append(
            AxisDeclaration(
                axis_id=axis_id,
                role=AxisRole.CONSTANT,
                quantity_kind=quantity,
                label_raw=label.raw,
                label_ref=label.ref,
            )
        )
        value = _measured(doc.text(_one(prop, "value")), doc.attribute(prop, "units"), quantity)
        constants.append(Coordinate(axis_id=axis_id, value=value, uncertainty=_no_uncertainty()))
    for required in ("temperature", "pressure"):
        if not any(axis.axis_id == required for axis in axes):
            raise RespecthRefusal(RespecthRefusalReason.INCOMPLETE_RECORD, f"the record states no {required}")

    rows = group.findall("dataPoint")
    if not rows:
        raise RespecthRefusal(
            RespecthRefusalReason.INCOMPLETE_RECORD, "the ignition-delay data group has no data points"
        )
    width = max(4, len(str(len(rows))))
    points: list[DataPoint] = []
    for number, row in enumerate(rows, start=1):
        cells = {cell.tag: cell for cell in row}
        if set(cells) != set(columns) or len(cells) != len(row) or row.attrib:
            raise RespecthRefusal(
                RespecthRefusalReason.INCOMPLETE_RECORD,
                f"{doc.path(row)} holds {sorted(cell.tag for cell in row)}, not exactly the columns {sorted(columns)}",
            )
        coordinates: list[Coordinate] = []
        observations: list[Observation] = []
        for column_id, (axis_id, quantity, unit) in columns.items():
            value = _measured(doc.text(cells[column_id]), unit, quantity)
            if axis_id == _IGNITION_AXIS_ID:
                observations.append(Observation(axis_id=axis_id, value=value, uncertainty=common.uncertainty))
            else:
                coordinates.append(Coordinate(axis_id=axis_id, value=value, uncertainty=_no_uncertainty()))
        points.append(
            DataPoint(
                point_id=f"p{number:0{width}d}",
                coordinates=tuple(sorted(coordinates, key=lambda c: c.axis_id)),
                observations=tuple(observations),
                composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
            )
        )
    series = Series(
        series_id=_SERIES_ID,
        source_form=SourceForm.STRUCTURED_RECORD,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=tuple(sorted(axes, key=lambda axis: axis.axis_id)),
        constants=tuple(sorted(constants, key=lambda c: c.axis_id)),
        points=tuple(points),
        digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    return series, source_types


def _no_uncertainty() -> Absent:
    return Absent(reason=AbsenceReason.UNKNOWN, note="the record states no uncertainty for this condition")


def _check_format(doc: _Doc) -> None:
    version = _one(doc.root, "ReSpecThVersion")
    major = (_one(version, "major").text or "").strip()
    if major != _SUPPORTED_RKD_MAJOR:
        raise RespecthRefusal(
            RespecthRefusalReason.UNSUPPORTED_FORMAT_VERSION, f"ReSpecThVersion major {major!r}; only 2.x is parsed"
        )
    experiment_type = (_one(doc.root, "experimentType").text or "").strip()
    if experiment_type != _EXPERIMENT_TYPE:
        raise RespecthRefusal(RespecthRefusalReason.NOT_IGNITION_DELAY, f"experimentType is {experiment_type!r}")


def parse_idt_record(member_bytes: bytes, archive: PinnedArchive, member_path: str) -> RespecthIdtRecord:
    """Map one RKD ignition-delay member into a :class:`RespecthIdtRecord`.

    Args:
        member_bytes: The member's exact bytes, as read (and verified) out of ``archive``.
        archive: The pinned archive the member came from.
        member_path: The member's path inside the archive (display only; the sha is identity).

    Raises:
        RespecthRefusal: Anything this lane cannot map without guessing. Nothing is returned
            partially -- a refusal means no record at all.
    """
    doc = _Doc.of(_parse_xml(member_bytes))
    if doc.root.tag != "experiment":
        raise RespecthRefusal(
            RespecthRefusalReason.MALFORMED_XML, f"root element is <{doc.root.tag}>, not <experiment>"
        )
    _check_format(doc)
    try:
        apparatus = _apparatus(doc)
        ignition = _ignition(doc)
        common = _common(doc)
        group, skipped = _data_groups(doc)
        series, source_types = _series(doc, group, common)
        rcm_conditions: RcmConditions | Absent = (
            _rcm_conditions(doc, group)
            if apparatus.device_class is ReactorType.RCM
            else Absent(reason=AbsenceReason.NOT_APPLICABLE)
        )
        _check_plausible_temperature(series)
        node = SourceNode(
            node_id=RECORD_NODE_ID,
            kind=SourceNodeKind.DATABASE_RECORD,
            sha256=hashlib.sha256(member_bytes).hexdigest(),
            parent_node_id=None,
            origin=ArchiveOrigin(archive_sha256=archive.sha256, member_display_path=member_path),
            extraction=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            glyph_health=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            verification=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
        envelope = DatasetEnvelope(
            source_graph=SourceGraph(nodes=(node,)),
            composition=common.composition,
            series=(series,),
            conversion_tables=(_embedded_table(),),
            table_inventories=(),
            ooxml_table_inventories=(),
            figure_digitizations=(),
        )
        return RespecthIdtRecord(
            archive=ArchivePin(
                archive_name=archive.name,
                osf_file_id=archive.osf_file_id,
                osf_version=archive.osf_version,
                archive_sha256=archive.sha256,
                member_path=member_path,
            ),
            record_doi=doc.text(_one(doc.root, "fileDOI")),
            paper=_paper(doc),
            apparatus=apparatus,
            ignition=ignition,
            uncertainty=common.uncertainty_definition,
            column_source_types=tuple(sorted(source_types.items())),
            rcm_conditions=rcm_conditions,
            skipped_data_groups=skipped,
            envelope=envelope,
        )
    except ValidationError as exc:
        raise RespecthRefusal(RespecthRefusalReason.SCHEMA_REJECTED, str(exc)) from exc


def _embedded_table() -> EmbeddedConversionTable:
    return EmbeddedConversionTable(
        sha256=_TABLE.sha256, canonical_json=canonical_json_bytes(_TABLE.identity_payload()).decode("utf-8")
    )


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class RecordReplayReport:
    """The outcome of replaying every grounded string of a record against member bytes."""

    verified: bool
    checked: int
    """How many (ref, expected text) pairs were re-evaluated and matched or not."""
    findings: tuple[str, ...]
    """Every disagreement or inability, verbatim. Empty exactly when ``verified``."""
    assumptions: tuple[str, ...] = ()
    """What the record maps WITHOUT the member stating it (an assumed apparatus mode). Not
    findings -- a verified record may rest on them -- but always shown."""


def _grounded_pairs(obj: object, path: str = "") -> Iterator[tuple[str, SourceRef, str]]:
    """Every ``(ref path, ref, expected text)`` a record grounds, in the path format
    :func:`~carmel.schemas.datasets.iter_source_refs` uses, so coverage can be compared."""

    def join(name: str) -> str:
        return f"{path}.{name}" if path else name

    if isinstance(obj, MeasuredValue):
        yield join("value_ref"), obj.value_ref, obj.raw_text
        if isinstance(obj.unit_ref, SourceRef) and isinstance(obj.unit_raw, str):
            yield join("unit_ref"), obj.unit_ref, obj.unit_raw
        return
    if isinstance(obj, AxisDeclaration):
        yield join("label_ref"), obj.label_ref, obj.label_raw
        return
    if isinstance(obj, RecordText):
        yield join("ref"), obj.ref, obj.raw
        return
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from _grounded_pairs(getattr(obj, name), join(name))
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _grounded_pairs(value, f"{path}[{index}]")


def _replay_conditions(record: RespecthIdtRecord, root: ElementTree.Element) -> list[str]:
    """Re-derive the RCM end-of-compression verdict and the temperature backstop from the bytes."""
    findings: list[str] = []
    try:
        _check_plausible_temperature(record.envelope.series[0])
        doc = _Doc.of(root)
        group, _ = _data_groups(doc)
        expected: RcmConditions | None = (
            _rcm_conditions(doc, group) if record.apparatus.device_class is ReactorType.RCM else None
        )
    except RespecthRefusal as exc:
        return [f"conditions do not re-map: {exc.reason.value}: {exc.detail}"]
    recorded = None if isinstance(record.rcm_conditions, Absent) else record.rcm_conditions
    if recorded != expected:
        findings.append(f"rcm_conditions {recorded!r} do not re-derive (the member gives {expected!r})")
    return findings


def replay_idt_record(record: RespecthIdtRecord, member_bytes: bytes) -> RecordReplayReport:
    """Re-derive every grounded string of ``record`` from ``member_bytes``.

    Checks, in order: the bytes hash to the record node's sha256; every
    :class:`SourceRef` anywhere in the record (walked by
    :func:`~carmel.schemas.datasets.iter_source_refs`, not hand-listed) is paired with the
    text it claims and targets the record node by XPath; each path re-evaluates to exactly
    that text; and the mapped apparatus and ignition definition re-map from their raw text.
    ``verified`` is true only when at least one pair was checked and nothing disagreed.
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
    except RespecthRefusal as exc:
        return RecordReplayReport(verified=False, checked=0, findings=(str(exc),))

    findings: list[str] = []
    pairs = list(_grounded_pairs(record))
    paired_paths = {path for path, _, _ in pairs}
    for path, _ in iter_source_refs(record):
        if path not in paired_paths:
            findings.append(f"{path}: a SourceRef with no text this replayer knows to compare")
    checked = 0
    for path, ref, expected in pairs:
        if ref.node_id != RECORD_NODE_ID or not isinstance(ref.locator, XPathLocator):
            findings.append(f"{path}: does not address the record node by XPath")
            continue
        actual = evaluate_xpath(root, ref.locator.xpath)
        checked += 1
        if actual != expected:
            findings.append(f"{path}: {ref.locator.xpath} reads {actual!r}, recorded {expected!r}")

    kind = record.apparatus.kind_raw.raw.strip()
    mode = None if isinstance(record.apparatus.mode_raw, Absent) else record.apparatus.mode_raw.raw.strip()
    if APPARATUS_DEVICE_CLASSES.get((kind, mode)) is not record.apparatus.device_class:
        findings.append(f"apparatus ({kind!r}, {mode!r}) does not re-map to {record.apparatus.device_class.value!r}")
    expected_assumption = ASSUMED_APPARATUS_MODES.get((kind, None)) if mode is None else None
    recorded_assumption = None if isinstance(record.apparatus.assumed_mode, Absent) else record.apparatus.assumed_mode
    if recorded_assumption != expected_assumption:
        findings.append(
            f"apparatus mode assumption {recorded_assumption!r} does not re-map "
            f"(the table gives {expected_assumption!r})"
        )
    stated_mode = evaluate_xpath(root, f"{record.apparatus.apparatus_xpath}/mode[1]")
    if recorded_assumption is not None and (stated_mode or "").strip():
        findings.append(f"apparatus mode is recorded as assumed, but the member states {stated_mode!r}")
    assumption = record.apparatus.assumption
    findings.extend(_replay_conditions(record, root))
    targets = [part.strip() for part in record.ignition.target_raw.raw.split(";") if part.strip()]
    if (
        targets != [record.ignition.target.value]
        or record.ignition.criterion_raw.raw != record.ignition.criterion.value
    ):
        findings.append("the ignition definition does not re-map from its raw text")
    if not isinstance(record.uncertainty, Absent) and (
        record.uncertainty.name_raw.raw,
        record.uncertainty.basis_raw.raw,
        record.uncertainty.reference_raw.raw,
    ) != ("evaluated standard deviation", "relative", _IGNITION_DELAY):
        findings.append("the uncertainty definition does not re-map to a relative standard deviation of the delay")
    if not isinstance(record.paper, Absent):
        says_placeholder = next(_placeholder_comments(root), None) is not None
        if says_placeholder != (record.paper.trust is ReferenceDoiTrust.PLACEHOLDER):
            findings.append(f"the member's comments do not support referenceDOI trust {record.paper.trust.value!r}")
    return RecordReplayReport(
        verified=checked > 0 and not findings,
        checked=checked,
        findings=tuple(findings),
        assumptions=(assumption,) if assumption is not None else (),
    )
