# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Schema primitives for literature-extracted experimental kinetics datasets.

This module builds the primitives listed for milestone M-D2a (explicit
absence states, coordinate frames/bboxes, measured values with per-field
unit binding, uncertainty, and composition), M-D2b part c (the source graph,
:class:`SourceGraph`, and the dataset envelope, :class:`DatasetEnvelope`,
that ties every embedded :class:`SourceRef` back to a node the graph
actually contains), and M-D2b part a (the dataset "series" aggregate,
:class:`Series`, built from axes/constants/points -- see its own docstring).
A registry across envelopes (M-D2b part b) remains out of scope here.

Cardinal rule this module exists to serve: every load-bearing number in a
Carmel dataset must be grounded against stored bytes and auditable, and the
schema itself must make fabrication structurally impossible rather than
merely discouraged. Three empirically-measured failure modes drove the
design here, each documented at the model that addresses it:

- Absence: a missing field must never be representable as a plain ``None``,
  because ``None`` carries no reason and is trivially confusable with
  "inherits from parent" (see :class:`AbsenceReason`/:class:`Absent`).
- Units are inconsistent WITHIN a single paper (narrative prose in cm/s, a
  table column in m/s) -- so a value and its unit must carry independent
  provenance (see :class:`MeasuredValue`).
- "air" is an unresolved token in most of this corpus's papers, and no paper
  restates its O2:N2 ratio numerically -- so composition must be able to
  represent an unresolved named mixture with NO components, rather than
  force an extractor to fabricate the 0.21/0.79 split (see
  :class:`Composition`).

Numeric facts here are canonical decimal STRINGS (via
:func:`carmel.services.dataset_store.canonical_decimal`), never floats --
see that function's docstring for why floats are rejected outright.

Every model in this module is constructed with ``frozen=True``: a value that
has passed validation must not be mutable afterward, because a content
address (see :class:`MeasuredValue`'s ``conversion_table_sha256``, and the
sha256 identity of :class:`~carmel.services.units.ConversionTable`) is a
claim about a specific, already-validated payload -- an in-place attribute
assignment after construction would let that payload silently drift out from
under an address that was computed before the mutation, with nothing here to
notice. Plain attribute assignment (``instance.field = ...``) now raises.
This is NOT an absolute immutability guarantee: ``Model.model_construct()``
bypasses validation entirely (including this check) by design, and
``instance.model_copy(update={...})`` deliberately builds a new, independent
instance rather than mutating the original -- neither is closed off by
``frozen=True``, and both remain the correct escape hatches for code that
genuinely needs to construct or derive a payload without going through
``__init__`` validation.

``frozen=True`` only blocks attribute REASSIGNMENT (``instance.field = ...``);
it does nothing to stop IN-PLACE mutation of a mutable container an instance
happens to hold (``instance.field.append(...)``, ``.clear()``, ``[...] =
...``). A field typed ``list[...]``/``dict[...]``/``set[...]`` would reopen
exactly the hole this paragraph describes -- an already-validated payload
(e.g. ``Composition.components``, which every RESOLVED_COMPONENTS/
UNRESOLVED_NAMED_MIXTURE invariant is checked against at construction time)
could be mutated after the fact with no validator ever re-running. Every
container-typed field in this module is therefore declared with an immutable
element type -- ``tuple[..., ...]`` in place of ``list[...]`` (see
``MeasuredValue.repairs`` and ``Composition.components``) -- and this is
exhaustive only because every field is checked this way; a future field that
introduces a bare ``list``/``dict``/``set`` would silently reopen the hole
frozen=True was meant to close.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from carmel.services import units
from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal, canonical_json_bytes
from carmel.services.numeric import (
    REPAIR_NAMES,
    GlyphHealth,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    InputPolicy,
    UnknownSemanticDependencyError,
    current_sha_for,
    dependency_for_sha,
)
from carmel.services.units import QuantityKind

__all__ = [
    "AbsenceReason",
    "Absent",
    "ArchiveOrigin",
    "AxisDeclaration",
    "AxisRole",
    "BBox",
    "BBoxLocator",
    "CaptionLabelKey",
    "ComponentRole",
    "Composition",
    "CompositionBasis",
    "CompositionComponent",
    "CompositionResolution",
    "Coordinate",
    "CoordinateFrame",
    "DataPoint",
    "DatasetEnvelope",
    "EmbeddedConversionTable",
    "Maybe",
    "MeasuredValue",
    "MemberSheetKey",
    "Observation",
    "QuantityKind",
    "SemanticDependencyUse",
    "Series",
    "SourceForm",
    "SourceGraph",
    "SourceLocator",
    "SourceNode",
    "SourceNodeKind",
    "SourceRef",
    "TableCellLocator",
    "TableKey",
    "TableKeyKind",
    "Uncertainty",
    "UncertaintyBasis",
    "UncertaintyKind",
    "UncertaintyScale",
    "ValueOrigin",
    "XPathLocator",
    "iter_measured_values",
    "iter_source_refs",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# UNTRUSTED-CANONICAL-JSON-RESOURCE-GUARD: bound on EmbeddedConversionTable.canonical_json's
# length, in UTF-8 bytes-ish characters. canonical_json arrives embedded in a stored dataset
# file -- untrusted input, not something this process generated -- and is parsed with
# json.loads in _validate_canonical_json_reconstructs_to_sha256 below. TABLE_V1's own
# canonical rendering is ~4KB (3986 bytes, measured directly); 1 MiB gives roughly 250x
# headroom over the largest table shipped today while still bounding how much memory/CPU a
# single hostile or corrupted embedding can force this process to spend on json.loads.
# Mirrors the same defensive posture as carmel.services.dataset_store's own untrusted-input
# guards (_MAX_JSON_DEPTH, _raw_bytes_nest_too_deeply) -- see that module's
# _read_verified_canonical_dict for the precedent this follows.
_MAX_EMBEDDED_CANONICAL_JSON_LENGTH = 1_048_576

# carmel.services.dataset_store._HEALTHY_GLYPH_HEALTH is module-private (leading
# underscore) -- reaching across a module boundary to import another module's private
# name would tie this schema to dataset_store's internals, so this constant is
# constructed here identically instead. Same rationale as there: a stored
# MeasuredValue's raw_text carries no surrounding document, so there is nothing to run
# assess_glyph_health() on, and SourceContext.OPERATOR_RAW (used below) is the one
# SourceContext that never inherits or implies a document's dash-corruption quarantine
# state -- exactly right here, since there is no document at all.
_HEALTHY_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)


def _require_canonical_decimal(value: str, *, field_name: str) -> str:
    """Validate ``value`` is already in canonical decimal form.

    Shared by every model field that stores a canonical decimal string
    (:class:`MeasuredValue`'s ``canonical_decimal_value``,
    :class:`Uncertainty`'s ``upper``/``lower``): re-canonicalizing must be a
    no-op (``canonical_decimal`` is idempotent by construction -- see its own
    docstring), so any mismatch means the caller handed this field a value
    that was never run through :func:`canonical_decimal` at all, or was
    hand-typed with different apparent precision than it claims.
    """
    try:
        canonical = canonical_decimal(value)
    except CanonicalDecimalError as exc:
        raise ValueError(f"{field_name}={value!r} is not a valid canonical decimal string: {exc}") from exc
    if canonical != value:
        raise ValueError(
            f"{field_name}={value!r} is not already canonical (canonical form is {canonical!r}); "
            "store the output of canonical_decimal(), not a hand-formatted string"
        )
    return value


def _require_finite_as_float(value: str, *, field_name: str) -> Decimal:
    """Require ``value`` (already a canonical decimal string) to evaluate to a
    finite ``float``, and return it as a :class:`Decimal`.

    This closes a real leak across two deliberately different layers in
    :mod:`carmel.services.numeric`: ``normalize_numeric_span`` accepts
    ``"1E+400"`` as a well-formed, exactly-representable decimal numeral (its
    docstring explains why -- textual FORM and float EVALUATION are different
    concerns there), while ``parse_numeric_span`` refuses it because
    ``float("1E+400")`` is ``inf``. ``canonical_decimal`` -- which every
    canonical-decimal field here is built on -- inherits the permissive,
    textual-only side of that split (deliberately: it also canonicalizes bbox
    coordinates and conversion factors, none of which should ever gain a
    finiteness opinion). Left unchecked, that permissiveness leaks straight
    through into any field that stores the OUTCOME of a real-world
    measurement: a ``MeasuredValue`` with ``canonical_decimal_value="1E+400"``
    validates today, silently breaking every downstream consumer that
    assumes "schema-valid implies convertible to a finite float".

    Fix it HERE, at the model boundary that actually means "this is a
    measured quantity" -- never in ``canonical_decimal`` itself, which must
    stay permissive for its other, non-measurement callers. No measurement in
    this domain is ``1E+400``.
    """
    decimal_value = Decimal(value)
    if not math.isfinite(float(decimal_value)):
        raise ValueError(
            f"{field_name}={value!r} is a well-formed canonical decimal string but does not evaluate to "
            "a finite float; no real measurement in this domain has that magnitude, and a consumer that "
            "assumes 'schema-valid implies convertible to a finite float' would silently break on it"
        )
    return decimal_value


class AbsenceReason(StrEnum):
    """Why a field is absent, as a first-class fact rather than a bare ``None``.

    ``unknown`` is deliberately NOT a dumping ground: ``not_reported_here`` and
    ``not_extracted_yet`` are different facts with different remedies (one is
    a property of the source paper; the other is a bug in Carmel's own
    extraction) and must never be collapsible into each other or into
    ``unknown``.
    """

    NOT_APPLICABLE = "not_applicable"
    """The field does not apply to this source at all (e.g. reactor geometry
    for a pure-simulation source that has no physical reactor)."""

    NOT_REPORTED_HERE = "not_reported_here"
    """The paper itself defers this fact to a companion paper; it is a
    property of the source, not a gap in Carmel's extraction."""

    NOT_EXTRACTED_YET = "not_extracted_yet"
    """Carmel has not yet extracted this field -- extractor incompleteness,
    i.e. OUR gap, not the source's. Must never be conflated with
    ``not_reported_here``: one is fixed by re-running extraction, the other
    never can be."""

    CONFLICTING_SOURCES = "conflicting_sources"
    """Two locations in the paper (or paper vs. SI) disagree on this value,
    and no arbitration has been performed."""

    UNKNOWN = "unknown"
    """Genuinely indeterminate -- neither a companion-paper deferral nor an
    extraction gap nor a conflict; the fact simply cannot be determined from
    available sources."""

    SAME_AS_DATASET = "same_as_dataset"
    """Explicitly inherits the parent dataset's value for this field. This
    is itself an explicit statement (never the default), so an "inherits"
    relationship can never be confused with an unexamined field."""


class Absent(BaseModel):
    """An explicit "this value is absent, and here is why" marker.

    Every field that can be missing is typed ``T | Absent`` (see
    :func:`Maybe`) rather than ``T | None``: a bare ``None`` carries no reason
    and is trivially confusable with "nobody looked at this field yet",
    whereas an ``Absent`` always states which of the six
    :class:`AbsenceReason` values applies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: AbsenceReason
    note: str | None = None
    """Optional free-text elaboration (e.g. which companion paper, which two
    locations conflicted). Never required -- the ``reason`` alone is always
    sufficient to distinguish the six cases."""


type Maybe[T] = T | Absent
"""The type of a field that may be explicitly absent: ``Maybe[SomeType]`` means
"a ``SomeType``, or an explicit :class:`Absent` stating why it's missing".

Deliberately NOT ``T | None``: see :class:`Absent` for why a bare ``None``
must never be usable in its place. A PEP 695 generic type alias (not a
factory function or a wrapper model) so it stays a plain union at both
runtime and under static type checking; pydantic's "smart" union mode picks
the right member on validation, and passing ``None`` for a ``Maybe[T]``
field matches neither member and is rejected.
"""


class CoordinateFrame(BaseModel):
    """The rendering context a :class:`BBox`'s coordinates are defined against.

    Page NUMBER is deliberately NOT a field here, and must never be added as
    a provenance key: empirical probing of this corpus's extractors found
    pdfminer and pypdf silently drop pages in 3 of 8 stored documents, so a
    page number produced by one tool does not reliably address the same
    physical page as the same number from another tool (or a later
    re-render). ``render_fingerprint`` is the actual identity here -- a hash
    or equivalent fingerprint of the rendered page content -- so provenance
    survives even when page numbering does not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    render_fingerprint: str = Field(min_length=1)
    """Fingerprint of the rendered page (e.g. a hash of the rendered image or
    text layer). The actual provenance key -- see class docstring."""
    cropbox: tuple[str, str, str, str]
    """``(x0, y0, x1, y1)`` as canonical decimal strings, never floats -- see
    module docstring. Must be non-degenerate under the same convention as
    :class:`BBox` (enforced below): a cropbox with ``x0 >= x1`` or
    ``y0 >= y1`` describes no actual page area."""
    mediabox: tuple[str, str, str, str]
    """Same convention and constraints as ``cropbox``."""
    rotation: int
    """Page rotation in degrees; constrained to exactly ``{0, 90, 180, 270}``
    (not merely "a multiple of 90") -- see the validator below for why."""
    units: str = Field(min_length=1)
    dpi: Maybe[str]
    """Canonical decimal string when present, strictly positive (enforced
    below) -- a bare ``None`` would be exactly the un-reasoned absence this
    module's ``Absent`` machinery exists to forbid (see module docstring), and
    a dpi cannot legitimately measure zero or negative."""
    render_settings: Maybe[str]
    """Free-text description of the renderer and its settings/version, for
    reproducing the exact render this frame's coordinates were measured
    against. This is render PROVENANCE -- an asserted fact about how the page
    was rendered, not decoration -- so a bare ``None`` must never stand in for
    "the extractor didn't record this"; see :class:`Absent`."""

    @field_validator("cropbox", "mediabox")
    @classmethod
    def _validate_box_coords(cls, value: tuple[str, str, str, str], info: ValidationInfo) -> tuple[str, str, str, str]:
        field_name = info.field_name or "box"
        x0, y0, x1, y1 = (_require_canonical_decimal(v, field_name=f"{field_name}[{i}]") for i, v in enumerate(value))
        # Compare as Decimal, never as strings -- string comparison would call
        # "10" less than "9". A degenerate or inverted box describes no actual
        # page area.
        if not (Decimal(x0) < Decimal(x1)):
            raise ValueError(f"{field_name}: x0={x0!r} must be strictly less than x1={x1!r}")
        if not (Decimal(y0) < Decimal(y1)):
            raise ValueError(f"{field_name}: y0={y0!r} must be strictly less than y1={y1!r}")
        return (x0, y0, x1, y1)

    @field_validator("dpi")
    @classmethod
    def _validate_dpi(cls, value: str | Absent) -> str | Absent:
        if isinstance(value, Absent):
            return value
        canonical = _require_canonical_decimal(value, field_name="dpi")
        if not (Decimal(canonical) > 0):
            raise ValueError(f"dpi={canonical!r} must be strictly positive -- a render cannot have zero/negative dpi")
        return canonical

    @field_validator("rotation")
    @classmethod
    def _validate_rotation(cls, value: int) -> int:
        # Constrained to exactly {0, 90, 180, 270}, not merely "a multiple of
        # 90": that would admit -90, which describes the same physical page
        # as 270 but serializes to different bytes. Admitting both would give
        # one physical fact two different content addresses -- an
        # address-stability violation, not just a cosmetic one.
        if value not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be one of 0, 90, 180, 270 degrees, got {value}")
        return value


class BBox(BaseModel):
    """A bounding box on a rendered page.

    ``frame`` is a required field with no default: a bbox's coordinates are
    meaningless without knowing the frame (cropbox, rotation, DPI, ...) they
    were measured against, so a bbox cannot be constructed at all without one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: CoordinateFrame
    x0: str
    """Canonical decimal string, never a float -- see module docstring."""
    y0: str
    x1: str
    y1: str

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def _validate_coord(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "coordinate"
        return _require_canonical_decimal(value, field_name=field_name)

    @model_validator(mode="after")
    def _validate_non_degenerate(self) -> BBox:
        # Compare as Decimal, never as strings -- string comparison would
        # call "10" less than "9". A degenerate or inverted box locates
        # nothing, so impossible provenance must be rejected here rather than
        # stored: bad provenance is worse than absent provenance, because it
        # reads as verified.
        if not (Decimal(self.x0) < Decimal(self.x1)):
            raise ValueError(f"BBox requires x0 < x1, got x0={self.x0!r}, x1={self.x1!r}")
        if not (Decimal(self.y0) < Decimal(self.y1)):
            raise ValueError(f"BBox requires y0 < y1, got y0={self.y0!r}, y1={self.y1!r}")
        return self


class SourceNodeKind(StrEnum):
    """Kinds of artifacts that can appear as a node in a dataset's source graph."""

    PAPER_PDF = "paper_pdf"
    JATS_XML = "jats_xml"
    SI_MEMBER = "si_member"
    FIGURE_CROP = "figure_crop"


class ArchiveOrigin(BaseModel):
    """Which archive (e.g. an SI zip) a :class:`SourceNode` was extracted
    from, and where within it -- an artifact's ORIGIN, not a location within
    currently-referenced content (that's what :class:`SourceLocator` is for;
    see :class:`SourceNode`'s ``origin`` field for the boundary between the
    two).

    ``archive_sha256`` is identity; ``member_display_path`` is display-only.
    This split is deliberate: archive paths collide under normalization
    (e.g. ``"./a/b"`` and ``"a/b"`` name the same path; a path may use ``/``
    or ``\\`` as its separator) in ways that carry no information about the
    member's actual bytes, and a path string can be adversarially crafted
    (path traversal sequences, homoglyphs) to *look* like it identifies one
    member while actually addressing another. Treating a display path as
    identity would let two such adversarial paths collide silently;
    ``archive_sha256`` is the only identity-bearing field here for exactly
    that reason, and ``member_display_path`` exists purely so a human
    reviewing provenance has something readable to look at.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_sha256: str = Field(min_length=64, max_length=64)
    member_display_path: str | None = None
    """Human-readable ONLY -- never identity. See class docstring."""

    @field_validator("archive_sha256")
    @classmethod
    def _validate_archive_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(f"invalid archive_sha256: expected 64 lowercase hex chars, got {value!r}")
        return value


class SourceNode(BaseModel):
    """One artifact in a dataset's source graph.

    A dataset's points routinely come from several different artifacts (the
    main PDF, a supplementary-information spreadsheet member, a figure crop
    taken from one page of that PDF) -- a single root-level
    ``source_artifact_sha`` cannot represent that by construction, so the
    source graph is the primitive instead. ``parent_node_id`` lets a node
    point back to the artifact it was derived from (an SI member's parent
    paper; a figure crop's parent page/PDF).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    kind: SourceNodeKind
    sha256: str = Field(min_length=64, max_length=64)
    parent_node_id: str | None = None
    origin: Maybe[ArchiveOrigin]
    """Which archive this node was extracted from, if any. ``Maybe``-typed,
    with no default, for the same reason documented on
    :class:`CompositionComponent`'s ``role`` field: a mandatory default
    would force every node -- including a ``PAPER_PDF`` that plainly didn't
    come out of a zip -- to state one way or the other, when for most nodes
    the honest, structural answer is "the concept does not apply here" (see
    the validator below, which requires exactly that for every non
    ``SI_MEMBER`` node) rather than an unreasoned absence."""

    @model_validator(mode="after")
    def _validate_origin_only_for_si_member(self) -> SourceNode:
        """Only an ``SI_MEMBER`` node may carry a concrete (non-``Absent``)
        :class:`ArchiveOrigin`.

        A paper PDF didn't come out of a zip -- nor did a JATS/XML document,
        nor a figure crop (both derived some other way) -- so any other
        kind claiming a concrete origin is describing a provenance
        relationship that cannot actually exist.
        """
        if self.kind != SourceNodeKind.SI_MEMBER and not isinstance(self.origin, Absent):
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which cannot carry a concrete "
                "ArchiveOrigin -- only an SI_MEMBER node can, since only an SI_MEMBER node was ever "
                "extracted from an archive; origin must be Absent(...) here"
            )
        return self

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(f"invalid sha256: {value!r} (expected 64 lowercase hex characters)")
        return value


class LocatorKind(StrEnum):
    """Discriminator for :data:`SourceLocator`."""

    BBOX = "bbox"
    TABLE_CELL = "table_cell"
    XPATH = "xpath"


class BBoxLocator(BaseModel):
    """Locates a reference at a bounding box on a rendered page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.BBOX] = LocatorKind.BBOX
    bbox: BBox


class TableKeyKind(StrEnum):
    """Discriminator for :data:`TableKey`."""

    CAPTION_LABEL = "caption_label"
    MEMBER_SHEET = "member_sheet"


class CaptionLabelKey(BaseModel):
    """Identifies a table by the label printed on its caption (e.g. ``"Table 2"``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[TableKeyKind.CAPTION_LABEL] = TableKeyKind.CAPTION_LABEL
    label: str = Field(min_length=1)
    """The caption label verbatim, e.g. ``"Table 2"`` or ``"Table S1"``."""


class MemberSheetKey(BaseModel):
    """Identifies a table by the sheet name of an ``SI_MEMBER`` spreadsheet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[TableKeyKind.MEMBER_SHEET] = TableKeyKind.MEMBER_SHEET
    sheet_name: str = Field(min_length=1)
    """The workbook sheet name verbatim."""


TableKey = Annotated[CaptionLabelKey | MemberSheetKey, Field(discriminator="kind")]
"""Which table, within the targeted node, a :class:`TableCellLocator` addresses.

Required rather than optional: a node (e.g. a multi-table SI member or a PDF
page rendering several tables) can hold more than one table, and ``row``/
``col`` alone are meaningless without saying which table they index into.
"""


class TableCellLocator(BaseModel):
    """Locates a reference at a specific table cell.

    ``table_key`` disambiguates WITHIN THE TARGETED NODE only -- it makes no
    global-uniqueness claim across the dataset's whole source graph, and it
    is not itself an independent locator: two different nodes may each
    legitimately hold a "Table 2". A content-addressed digest of the actual
    table REGION (as opposed to this caption/sheet-name key) is deliberately
    deferred to M-C/M1: it is circular to require now, before any extractor
    exists that defines what bytes constitute "the region" to hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.TABLE_CELL] = LocatorKind.TABLE_CELL
    table_key: TableKey
    row: int = Field(ge=0)
    """0-indexed row; a negative row locates no real table cell."""
    col: int = Field(ge=0)
    """0-indexed column; a negative col locates no real table cell."""


class XPathLocator(BaseModel):
    """Locates a reference in JATS/XML via an XPath expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.XPATH] = LocatorKind.XPATH
    xpath: str = Field(min_length=1)


SourceLocator = Annotated[
    BBoxLocator | TableCellLocator | XPathLocator,
    Field(discriminator="kind"),
]


class SourceRef(BaseModel):
    """A reference INTO a dataset's source graph: which node, and where in it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    locator: SourceLocator


def iter_source_refs(obj: object, _path: str = "") -> Iterator[tuple[str, SourceRef]]:
    """Recursively walk ``obj``, yielding ``(dotted_path, ref)`` for every
    :class:`SourceRef` reachable from it.

    This is the single choke point every "does this payload cite something
    real" check (:class:`DatasetEnvelope`'s V1/V2 validators, and any future
    consumer) runs through, and it is deliberately GENERIC over the payload
    shape rather than hand-listing "the composition's equivalence_ratio, the
    composition's components' amounts, ..." field by field: a hand-written
    list silently goes stale the moment a new SourceRef-bearing field is
    added anywhere in the tree (the series aggregate, M-D2b part a, is the
    concrete next case), which would let a dangling or decorative ref hide
    from validation with nothing here to notice. Walking pydantic
    ``BaseModel`` fields, ``list``/``tuple`` elements, and ``dict`` values
    covers every shape this schema currently uses to nest a payload, so
    adding a field of any of those container shapes is automatically
    covered with zero changes here -- see
    ``TestRefWalkCannotBeOutgrown`` in the test suite, which pins exactly
    this property.

    Deliberately does NOT recurse into a :class:`SourceRef`'s own fields
    (``node_id``, ``locator``): a ``SourceRef`` is a leaf of this walk, not a
    container to look inside of -- its ``locator`` may itself be a
    ``BaseModel`` (e.g. :class:`BBoxLocator` wrapping a :class:`BBox`), but
    that nested structure is the locator's OWN geometry, not another
    reference to chase.

    ``_path`` is an internal accumulator (leading underscore, not part of the
    public two-argument contract) used for the recursive descent; callers
    always invoke this with a single argument. Path segments join field
    names with ``.`` and index list/tuple elements or dict keys with
    ``[...]`` (e.g. ``"composition.components[0].amount.value_ref"``),
    matching the format asserted throughout the test suite.
    """
    if isinstance(obj, SourceRef):
        yield _path, obj
        return
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            child_path = f"{_path}.{name}" if _path else name
            yield from iter_source_refs(value, child_path)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_source_refs(value, f"{_path}[{key}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_source_refs(value, f"{_path}[{index}]")
        return
    return


def iter_measured_values(obj: object, _path: str = "") -> Iterator[tuple[str, MeasuredValue]]:
    """Recursively walk ``obj``, yielding ``(dotted_path, value)`` for every
    :class:`MeasuredValue` reachable from it.

    Mirrors :func:`iter_source_refs` exactly in style and for the same
    reason: it is the choke point :class:`DatasetEnvelope`'s T2 validator
    (``conversion_tables`` must cover exactly the set of tables actually
    cited) runs through, and it is deliberately GENERIC over payload shape
    rather than a hand-written list of "the composition's components'
    amounts, a point's coordinates' values, an observation's uncertainty
    bounds, ..." -- a hand-written list would silently go stale the moment a
    new ``MeasuredValue``-bearing field is added anywhere in the tree,
    letting a cited-but-unembedded table hide from T2 with nothing here to
    notice. Walking pydantic ``BaseModel`` fields, ``list``/``tuple``
    elements, and ``dict`` values covers every shape this schema currently
    uses to nest a payload, so a future field of any of those container
    shapes is automatically covered with zero changes here.

    Deliberately does NOT recurse into a :class:`MeasuredValue`'s own fields:
    a ``MeasuredValue`` is a leaf of this walk, not a container to look
    inside of -- its ``value_ref``/``unit_ref`` are followed by
    :func:`iter_source_refs`, a separate walk for a separate purpose.

    ``_path`` is an internal accumulator, exactly as in
    :func:`iter_source_refs`; callers always invoke this with a single
    argument.
    """
    if isinstance(obj, MeasuredValue):
        yield _path, obj
        return
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            child_path = f"{_path}.{name}" if _path else name
            yield from iter_measured_values(value, child_path)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_measured_values(value, f"{_path}[{key}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_measured_values(value, f"{_path}[{index}]")
        return
    return


class SourceGraph(BaseModel):
    """A validated DAG of :class:`SourceNode`\\ s: a dataset's whole provenance graph.

    A single node cannot express "the main PDF, plus an SI spreadsheet
    member, plus a figure crop taken from one of the PDF's pages, all
    provenance for the SAME dataset" -- so the graph, not the individual
    node, is the top-level provenance primitive an envelope holds. Every
    invariant below runs in a FIXED order via a single ``model_validator``,
    each with a distinctive, greppable error message, so a test can pin any
    one of them independently of the others.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[SourceNode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_invariants(self) -> SourceGraph:
        nodes_by_id: dict[str, SourceNode] = {}
        # I1: every node_id must be unique -- a duplicate id would make
        # `graph.node(id)` and every SourceRef targeting it ambiguous about
        # which node's provenance is actually meant.
        for node in self.nodes:
            if node.node_id in nodes_by_id:
                raise ValueError(f"SourceGraph contains a duplicate node_id: {node.node_id!r} appears more than once")
            nodes_by_id[node.node_id] = node

        # I2: a stated parent_node_id must resolve to a node actually present
        # in this graph -- an unresolvable parent is a dangling provenance
        # claim, no different in kind from a SourceRef naming a node that
        # doesn't exist.
        for node in self.nodes:
            if node.parent_node_id is not None and node.parent_node_id not in nodes_by_id:
                raise ValueError(
                    f"node {node.node_id!r} names parent_node_id={node.parent_node_id!r}, which is not "
                    "the id of any node present in this SourceGraph"
                )

        # I3: the parent_node_id chain must be acyclic. This MUST run before
        # I4 (kind/parent rules), and deliberately raises a distinct
        # "cycle"-bearing message: I4 happens to forbid every cycle the
        # current SourceNodeKind hierarchy can express (no kind lists itself,
        # or any kind reachable from itself, among its own valid parent
        # kinds), so if I4 ran first it would always win the race and I3
        # would never be reachable -- its own test would then pass only by
        # accident, via the neighbouring guard, leaving I3 itself
        # unpinned. Running I3 first, with its own message, is what makes it
        # independently testable.
        resolved: set[str] = set()
        for start_id in nodes_by_id:
            if start_id in resolved:
                continue
            path: list[str] = []
            position: dict[str, int] = {}
            current_id: str | None = start_id
            while current_id is not None and current_id not in resolved:
                if current_id in position:
                    cycle_ids = path[position[current_id] :]
                    raise ValueError(
                        f"SourceGraph contains a cycle among node ids {cycle_ids!r}: following "
                        "parent_node_id from each of these nodes loops back on itself instead of "
                        "terminating at a parentless root"
                    )
                position[current_id] = len(path)
                path.append(current_id)
                current_id = nodes_by_id[current_id].parent_node_id
            resolved.update(path)

        # I4: which SourceNodeKinds may/must have a parent, and of what kind.
        # PAPER_PDF/JATS_XML are the only artifacts that can stand on their
        # own (a top-level document); SI_MEMBER and FIGURE_CROP are always
        # DERIVED from something else, so an orphan of either kind, or one
        # whose parent is the wrong kind, describes a provenance relationship
        # that cannot actually exist.
        for node in self.nodes:
            if node.kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML):
                if node.parent_node_id is not None:
                    parent_kind = nodes_by_id[node.parent_node_id].kind
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r}, which must be a parentless "
                        f"root, but names parent_node_id={node.parent_node_id!r} (kind={parent_kind.value!r})"
                    )
            elif node.kind == SourceNodeKind.SI_MEMBER:
                if node.parent_node_id is None:
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r}, which requires a parent of "
                        "kind PAPER_PDF or JATS_XML, but has no parent"
                    )
                parent_kind = nodes_by_id[node.parent_node_id].kind
                if parent_kind not in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML):
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r}, which requires a parent of "
                        f"kind PAPER_PDF or JATS_XML, but its parent {node.parent_node_id!r} has "
                        f"kind={parent_kind.value!r}"
                    )
            elif node.kind == SourceNodeKind.FIGURE_CROP:
                if node.parent_node_id is None:
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r}, which requires a parent of "
                        "kind PAPER_PDF, JATS_XML or SI_MEMBER, but has no parent"
                    )
                parent_kind = nodes_by_id[node.parent_node_id].kind
                if parent_kind not in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER):
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r}, which requires a parent of "
                        f"kind PAPER_PDF, JATS_XML or SI_MEMBER, but its parent {node.parent_node_id!r} has "
                        f"kind={parent_kind.value!r}"
                    )

        # I5: no two nodes may be exact duplicates of each other -- same
        # kind, same bytes, same parent. The same bytes appearing under a
        # genuinely different role (a different kind, or a different parent)
        # stays legal: e.g. a PAPER_PDF root and a FIGURE_CROP taken from
        # that same PDF share a sha256 but are not duplicates of each other.
        # This triple deliberately does NOT include `origin`: two
        # byte-identical SI_MEMBER files belonging to the same paper are now
        # DISTINGUISHABLE from each other via `origin` (different archives,
        # or different member_display_paths within the same archive), but
        # this invariant does not treat that distinction as enough on its
        # own to admit both -- widening the triple to include origin is a
        # deliberate future change, not an oversight here.
        seen_triples: set[tuple[SourceNodeKind, str, str | None]] = set()
        for node in self.nodes:
            triple = (node.kind, node.sha256, node.parent_node_id)
            if triple in seen_triples:
                raise ValueError(
                    f"node {node.node_id!r} duplicates an earlier node exactly: another node already "
                    f"has kind={node.kind.value!r}, sha256={node.sha256!r}, "
                    f"parent_node_id={node.parent_node_id!r}; the same bytes in a genuinely different "
                    "role (a different kind or a different parent) remains legal, but an exact repeat "
                    "adds nothing"
                )
            seen_triples.add(triple)

        return self

    @property
    def node_ids(self) -> frozenset[str]:
        """Every node id present in this graph."""
        return frozenset(node.node_id for node in self.nodes)

    def node(self, node_id: str) -> SourceNode:
        """Return the node with id ``node_id``, or raise ``KeyError(node_id)``."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def ancestors(self, node_id: str) -> tuple[SourceNode, ...]:
        """Return ``node_id``'s parent chain, from immediate parent to root.

        Empty for a root node (``parent_node_id is None``). I3 forbids cycles
        at construction time for a graph built through normal validation, but
        ``SourceGraph.model_construct()`` is a documented escape hatch that
        bypasses validation (including I3) and can produce one -- so this
        walk keeps its own visited-set and raises ``ValueError`` naming the
        cycle rather than looping forever if it ever revisits a node_id.
        """
        nodes_by_id = {node.node_id: node for node in self.nodes}
        chain: list[SourceNode] = []
        visited: set[str] = {node_id}
        current = nodes_by_id[node_id]
        while current.parent_node_id is not None:
            if current.parent_node_id in visited:
                raise ValueError(
                    f"cycle detected in source graph while walking ancestors of {node_id!r}: "
                    f"node_id {current.parent_node_id!r} was already visited"
                )
            visited.add(current.parent_node_id)
            current = nodes_by_id[current.parent_node_id]
            chain.append(current)
        return tuple(chain)


class SemanticDependencyUse(BaseModel):
    """A RECORD that a specific :class:`~carmel.services.semantic_deps.SemanticDependencyDefinition`
    was applied to produce some other field on the record that embeds this one.

    This is deliberately NOT the same thing as
    :class:`~carmel.services.semantic_deps.SemanticDependencyDefinition` itself.
    That class (in the services layer, a frozen stdlib dataclass) names WHAT a
    versioned heuristic IS -- one entry per historical code version, living in
    an append-only registry. This class instead records THIS APPLICATION of
    that heuristic to a specific input -- e.g. "this particular
    ``MeasuredValue.repairs`` was produced by running
    ``carmel.numeric.context_free_span_repair`` at content address
    ``content_sha256``." Conflating the two would make it impossible to tell
    "the logic exists" from "the logic was actually run to produce this
    field," which is exactly the gap this model closes.

    ``content_sha256`` is validated against
    :func:`~carmel.services.semantic_deps.dependency_for_sha`'s registry --
    never against "the current version of the dependency" -- mirroring
    :meth:`MeasuredValue._validate_unit_normalization_against_the_recorded_table`'s
    own framing: a ``SemanticDependencyUse`` is validated against the
    dependency version it RECORDS, and an unresolvable ``content_sha256`` is
    refused rather than silently re-interpreted against whatever dependency
    happens to be current today.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dependency_id: str = Field(min_length=1)
    """The stable slug naming WHAT dependency this is (see
    :attr:`~carmel.services.semantic_deps.SemanticDependencyDefinition.dependency_id`).
    Enforced below to equal the RESOLVED definition's own ``dependency_id`` --
    an attacker (or a typo) supplying a ``content_sha256`` that resolves to
    one dependency while claiming a different ``dependency_id`` is a forgery
    attempt, not a harmless inconsistency, and is rejected as such."""
    content_sha256: str = Field(min_length=1)
    """The content address of the EXACT dependency version that was applied.
    Must resolve via :func:`~carmel.services.semantic_deps.dependency_for_sha`;
    an unresolvable value is refused rather than silently accepted, exactly
    like :attr:`MeasuredValue.conversion_table_sha256`."""
    input_sha256: Maybe[str]
    """The digest of the external input the dependency was applied to, when
    the resolved dependency's ``input_policy`` is
    :attr:`~carmel.services.semantic_deps.InputPolicy.EXTERNAL_DIGEST_REQUIRED`.
    Enforced below to be present if and only if the resolved dependency's
    ``input_policy`` demands it -- never a free-floating optional field whose
    presence is left to the caller's discretion."""

    @field_validator("input_sha256")
    @classmethod
    def _validate_input_sha256_shape(cls, value: Maybe[str]) -> Maybe[str]:
        if isinstance(value, Absent):
            return value
        if not _SHA256_RE.match(value):
            raise ValueError(f"invalid input_sha256: {value!r} (expected 64 lowercase hex characters)")
        return value

    @model_validator(mode="after")
    def _validate_against_the_resolved_dependency(self) -> SemanticDependencyUse:
        """Resolve ``content_sha256`` once and check every cross-field invariant against it.

        Three distinct checks share a single ``dependency_for_sha`` lookup
        (rather than three separate model validators each re-resolving it)
        because the registry lookup is monkeypatched in tests -- three
        independent lookups could observe three different registry states if
        anything ever mutated the registry mid-validation, so resolving once
        and reusing the result is the only way to guarantee all three checks
        agree on what ``content_sha256`` actually names.
        """
        try:
            definition = dependency_for_sha(self.content_sha256)
        except UnknownSemanticDependencyError as exc:
            raise ValueError(
                f"content_sha256={self.content_sha256!r} does not name any known semantic "
                "dependency; a SemanticDependencyUse is validated against the dependency "
                "version it RECORDS, never against 'the current dependency', so an "
                f"unresolvable sha is refused rather than silently re-interpreted: {exc}"
            ) from exc

        if self.dependency_id != definition.dependency_id:
            raise ValueError(
                f"dependency_id={self.dependency_id!r} disagrees with the dependency that "
                f"content_sha256={self.content_sha256!r} actually resolves to "
                f"(dependency_id={definition.dependency_id!r}); a SemanticDependencyUse whose "
                "dependency_id does not match its own content_sha256 is a forgery attempt, "
                "not a typo, and is rejected as such"
            )

        is_present = not isinstance(self.input_sha256, Absent)
        if definition.input_policy is InputPolicy.EXTERNAL_DIGEST_REQUIRED:
            if not is_present:
                raise ValueError(
                    f"content_sha256={self.content_sha256!r} resolves to a dependency whose "
                    "input_policy is EXTERNAL_DIGEST_REQUIRED, so input_sha256 must be present; "
                    "got Absent"
                )
        else:
            if is_present:
                raise ValueError(
                    f"content_sha256={self.content_sha256!r} resolves to a dependency whose "
                    f"input_policy is {definition.input_policy.value!r}, so input_sha256 must be "
                    f"Absent; got a present value {self.input_sha256!r}"
                )
        return self


class MeasuredValue(BaseModel):
    """A single numeric fact, bound to its unit with independently-verifiable provenance.

    This is the schema's most important model. Binding a number to its unit
    is the standing predicted failure mode for this project, and its real
    mechanism was measured directly in this corpus: **units are inconsistent
    WITHIN a single paper** (e.g. narrative prose reporting cm/s while a
    table column in the same paper reports m/s). A single value-level source
    ref cannot catch that: it can "verify" that the NUMBER is grounded while
    the unit silently came from the wrong column, from narrative prose, or
    from a stale assumption. So the value and the unit each carry their own
    :class:`SourceRef`, and both are required -- a value whose unit has no
    independent provenance cannot be constructed.

    The addressed payload stores only what the SOURCE said; conversion is a
    DERIVED, reproducible computation, never a stored claim. ``unit_raw`` is
    preserved verbatim, and ``quantity_kind`` (together with the recorded
    ``conversion_table_sha256``) is what makes a unit binding SOUND rather
    than decorative -- a unit pair alone does not identify a conversion
    (``"s"`` is time but ``"1/s"`` is a strain rate; ``"%"`` is a relative
    uncertainty or a volume percent; ``"1"`` is an equivalence ratio or a
    mole fraction). ``unit_normalized`` is the ONE table-derived claim that
    still lives in the payload, because it is a real interpretation the
    extractor made at extraction time (which spelling of ``unit_raw`` the
    table recognizes) and must be pinned rather than silently re-derived
    later if aliases change; it is emphatically not a converted value -- see
    that field's own docstring. Everything past spelling -- the actual
    numeric conversion to a quantity's base unit -- is recomputed on demand
    by :meth:`converted_to_base`, never stored: storing a converted value
    would manufacture precision into the content-addressed store (``1.23``
    atm times ``101325`` is ``124629.75`` -- eight digits from a
    three-significant-figure measurement), and a stored conversion that
    later disagreed with the table would be exactly the silent
    reinterpretation the table's sha256 exists to prevent.

    ``raw_text`` and ``canonical_decimal_value`` are deliberately allowed to
    DIFFER, via an explicit, auditable ``repairs`` chain -- this is not a gap,
    it is the point. Measured directly on this corpus: real source text is
    routinely glyph-corrupted in ways that hide a genuine numeric fact behind
    a substituted character -- ``/C0`` standing in for a minus sign in 7 of 8
    real papers, ``þ`` (U+00FE) standing in for a ``+`` in an exponent, U+2212
    or a leading U+2013 also standing in for a minus. A schema that required
    ``canonical_decimal_value == canonical_decimal(raw_text)`` verbatim could
    never represent the (very common) case where the paper's own printed
    minus sign survived only as a corrupted glyph -- it would force a choice
    between rejecting most of the real corpus's negative numbers outright, or
    writing the REPAIRED text into ``raw_text`` and destroying the evidence.
    Neither is acceptable, so instead: ``raw_text`` keeps the corrupted
    source span byte-for-byte (it IS the evidence), ``canonical_decimal_value``
    holds the canonicalization of the REPAIRED text, and ``repairs`` is the
    explicit, checked claim that bridges them -- validated below to be an
    exact, ordered match against what :func:`~carmel.services.numeric.normalize_numeric_span`
    itself reports needing, so neither under-claiming (a repair happened but
    was not recorded) nor over-claiming (a recorded repair the text never
    needed) can pass.

    A real boundary this leaves open, stated plainly rather than papered
    over: this schema's validator checks DERIVABILITY, a context-free
    textual property of ``raw_text`` alone. The dash-corruption QUARANTINE
    rule in :mod:`carmel.services.numeric` (a bare lowercase ``e`` exponent
    token that might actually encode a corrupted en-dash range) is
    document-level and extraction-time: it depends on whether the SOURCE
    DOCUMENT is suspected of that corruption, a fact this schema calls with a
    healthy, hand-constructed :class:`~carmel.services.numeric.GlyphHealth`
    because a stored ``MeasuredValue`` carries no surrounding document to
    assess. So this validator can confirm "this repair chain is internally
    consistent" but it can never re-run the document-level quarantine that
    decided whether this span was allowed to become a ``MeasuredValue`` at
    all -- that decision must already have been made correctly upstream, at
    extraction time, and this schema is not a complete substitute for it.

    To state that plainly rather than leave it implied: this validator
    confirms CONTEXT-FREE DERIVABILITY only, and it is NOT an admission
    check. The document-level dash-corruption quarantine is decided at
    EXTRACTION time, against the source document; a stored ``MeasuredValue``
    carries no document-health context at all, so a span the extractor
    should have quarantined (e.g. ``"2e50"`` read out of a suspect flat-PDF
    text layer whose minus/en-dash glyphs are known to be lost) is still
    representable here -- this validator has no way to know the document was
    ever suspect. Recording document glyph health belongs one level up, in
    the storage envelope, and is deliberately NOT stamped onto every value:
    a document-level fact repeated per point would let two values extracted
    from the SAME document assert different glyph health with no way to
    arbitrate between them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str = Field(min_length=1)
    """The exact numeric span as it appears in the source, VERBATIM --
    including any glyph corruption (e.g. ``"/C0 1.0"``, ``"7.000Eþ17"``). This
    is the evidence; it is never rewritten to the repaired form, even when
    ``repairs`` records exactly what repair would be needed to read it."""
    canonical_decimal_value: str = Field(min_length=1)
    """The canonicalization (via
    :func:`carmel.services.dataset_store.canonical_decimal`) of ``raw_text``
    AFTER applying the glyph repairs listed in ``repairs`` -- never a float,
    see that function's docstring for why. Must already be in canonical form.
    Enforced below to be exactly ``canonical_decimal`` of ``raw_text`` as
    repaired by the claimed ``repairs`` chain -- never asserted
    independently, and never silently equal to ``canonical_decimal(raw_text)``
    verbatim when a repair was actually needed.

    Also enforced below (see :func:`_require_finite_as_float`) to evaluate to
    a finite ``float``. ``canonical_decimal`` itself stays permissive and
    accepts e.g. ``"1E+400"`` -- it also canonicalizes bbox coordinates and
    conversion factors, which must never gain a finiteness opinion -- but a
    ``MeasuredValue`` is a MEASURED QUANTITY, and no measurement in this
    domain is ``1E+400``, so this field requires finiteness even though the
    string it stores is otherwise a valid canonical decimal."""
    repairs: tuple[str, ...] = ()
    """Names of the glyph repairs (each a member of
    :data:`carmel.services.numeric.REPAIR_NAMES` -- never free text, enforced
    below) applied to ``raw_text`` to derive ``canonical_decimal_value``. The
    empty tuple (the default) is the normal case: it means the source span
    was already clean and needed no repair. When non-empty, this is a claim
    about the evidence that must be EXACTLY true -- validated below against
    what :func:`~carmel.services.numeric.normalize_numeric_span` itself
    reports needing, in the same order."""
    repair_dependency: SemanticDependencyUse
    """WHICH version of the repair heuristic produced ``repairs``. Required,
    no default: this closes the defect where a stored ``MeasuredValue``
    recorded no version identity for the heuristic that produced its
    ``repairs``, so a future change to :mod:`carmel.services.numeric`'s
    regex/logic would make old, correctly-recorded data fail validation
    indistinguishably from forged data. Enforced below to name exactly
    :data:`~carmel.services.semantic_deps.CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID`
    -- the only repair heuristic this validator chain re-runs -- and its
    ``content_sha256`` is what the repair-chain validator below compares
    against the CURRENT sha to decide whether it may re-run the heuristic at
    all."""
    quantity_kind: QuantityKind
    """Which physical (or dimensionless-bookkeeping) quantity this value
    measures. Required, no default: a unit pair alone does not identify a
    conversion -- ``"s"`` is time but ``"1/s"`` is a strain rate, ``"%"`` is
    a relative uncertainty or a volume percent, ``"1"`` is an equivalence
    ratio or a mole fraction -- so the quantity kind is what makes the unit
    binding SOUND rather than decorative. ``QuantityKind.OTHER`` is the
    honest state for a quantity this table deliberately does not model; it
    permits identity conversion only (see :mod:`carmel.services.units`)."""
    unit_raw: str = Field(min_length=1)
    """The unit exactly as printed in the source."""
    unit_normalized: str = Field(min_length=1)
    """The recorded table's canonical SPELLING of ``unit_raw`` -- the SAME
    physical unit, never a converted target. ``"°C"`` normalizes to ``"C"``,
    which is still Celsius; the comparable target unit ``"K"`` is a
    different fact and is NOT stored here -- it is
    ``table.base_unit(quantity_kind)``, derived on demand (see
    :meth:`converted_to_base`). Conflating "the same unit, respelled" with
    "the value converted to another unit" into a single field was the
    defect this split replaced."""
    conversion_table_sha256: str = Field(min_length=1)
    """The content address (sha256) of the :class:`~carmel.services.units.ConversionTable`
    that ``unit_normalized`` was validated against. A version STRING is not
    identity: the constant behind a name like ``"v1"`` could be edited later
    while every already-addressed payload still says ``"v1"``, silently
    reinterpreting stored data. The sha256 makes that a migration (a new
    table, a new sha) instead of a silent reinterpretation."""
    value_ref: SourceRef
    """Provenance for the NUMBER. Required -- see class docstring."""
    unit_ref: SourceRef
    """Provenance for the UNIT, independent of ``value_ref``. Required -- see
    class docstring."""

    @field_validator("conversion_table_sha256")
    @classmethod
    def _validate_conversion_table_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(f"invalid conversion_table_sha256: {value!r} (expected 64 lowercase hex characters)")
        return value

    @field_validator("repairs")
    @classmethod
    def _validate_repair_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for name in value:
            if name not in REPAIR_NAMES:
                raise ValueError(
                    f"repairs contains {name!r}, which is not a member of "
                    f"carmel.services.numeric.REPAIR_NAMES ({sorted(REPAIR_NAMES)!r}); "
                    "repairs must never contain free text -- only the core's own repair names"
                )
        return value

    @field_validator("repair_dependency")
    @classmethod
    def _validate_repair_dependency_names_the_context_free_span_repair(
        cls, value: SemanticDependencyUse
    ) -> SemanticDependencyUse:
        if value.dependency_id != CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID:
            raise ValueError(
                f"repair_dependency.dependency_id={value.dependency_id!r} is not the repair "
                f"heuristic this validator chain re-runs ({CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID!r}); "
                "a MeasuredValue's repair_dependency must name exactly the one dependency its own "
                "repair-chain validator knows how to re-run"
            )
        return value

    @model_validator(mode="after")
    def _validate_repair_chain_agrees_with_raw_text(self) -> MeasuredValue:
        """Reject a ``repairs``/``canonical_decimal_value`` pair that disagrees
        with what ``raw_text`` itself actually needs.

        Replaces a plain string-equality check (``canonical_decimal(raw_text)
        == canonical_decimal_value``) that could never represent a repaired
        value -- see the class docstring for why that was wrong for this
        corpus. This validator checks the REPAIR CHAIN instead of string
        equality, but ONLY when ``repair_dependency`` names the CURRENT
        version of the heuristic:

        0. If ``repair_dependency.content_sha256`` is not the CURRENT sha for
           :data:`~carmel.services.semantic_deps.CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID`,
           this record names a REGISTERED-BUT-SUPERSEDED version of the
           heuristic. There is no runnable validator for a superseded
           version, so this validator refuses to re-run the CURRENT
           heuristic against a record that does not claim to have used it --
           doing so would silently re-interpret old data against logic it was
           never produced by, exactly the failure mode
           :mod:`carmel.services.semantic_deps` exists to prevent. There is
           NO "accept without re-running" middle path: a superseded record is
           rejected outright, never silently passed through.
        1. Otherwise (current version), ``raw_text`` must itself be
           derivable at all (rejects with the core's own reason if not).
        2. ``repairs`` must be an EXACT, ORDERED match for what
           :func:`~carmel.services.numeric.normalize_numeric_span` reports
           needing -- both under-claiming (a repair happened but was not
           recorded) and over-claiming (a recorded repair the text never
           needed) are rejected.
        3. ``canonical_decimal_value`` must be exactly
           ``canonical_decimal`` of the REPAIRED text -- never asserted
           independently.

        (An unresolvable ``repair_dependency.content_sha256`` never reaches
        this validator at all: it is rejected by ``SemanticDependencyUse``'s
        own validator during construction of the nested model, which
        completes -- including any raising -- before this outer model's own
        ``model_validator``s run.)
        """
        current_sha = current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID)
        if self.repair_dependency.content_sha256 != current_sha:
            raise ValueError(
                f"repair_dependency.content_sha256={self.repair_dependency.content_sha256!r} names a "
                f"registered but SUPERSEDED version of {CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID!r} "
                f"(current content_sha256 is {current_sha!r}); no runnable validator is registered for "
                "a superseded version, so this record cannot be re-validated by re-running the CURRENT "
                "heuristic against it -- there is no 'accept without re-running' path, so it is rejected "
                "outright rather than silently passed through"
            )
        normalized = normalize_numeric_span(
            self.raw_text,
            source_context=SourceContext.OPERATOR_RAW,
            glyph_health=_HEALTHY_GLYPH_HEALTH,
        )
        if isinstance(normalized, Unresolvable):
            raise ValueError(f"raw_text={self.raw_text!r} is not derivable into a numeral: {normalized.reason}")
        if tuple(self.repairs) != normalized.repairs:
            raise ValueError(
                f"repairs={self.repairs!r} disagrees with the repair(s) raw_text={self.raw_text!r} "
                f"actually needs ({normalized.repairs!r}); repairs is a claim about the evidence and "
                "must be exactly true -- neither under-claimed (a repair happened but was not "
                "recorded) nor over-claimed (a recorded repair the text never needed) is accepted"
            )
        try:
            expected = canonical_decimal(normalized.text)
        except CanonicalDecimalError as exc:
            raise ValueError(
                f"raw_text={self.raw_text!r} repaired via {self.repairs!r} to {normalized.text!r}, "
                f"which is not itself a valid canonical decimal string: {exc}"
            ) from exc
        if expected != self.canonical_decimal_value:
            raise ValueError(
                f"canonical_decimal_value={self.canonical_decimal_value!r} disagrees with raw_text="
                f"{self.raw_text!r} as repaired via {self.repairs!r} (canonical_decimal(repaired) == "
                f"{expected!r}); a MeasuredValue's canonical form must be derived from its own "
                "repaired raw_text, never asserted independently"
            )
        # canonical_decimal() itself stays permissive (see _require_finite_as_float's
        # docstring for why); a MeasuredValue is a MEASURED QUANTITY, so require its
        # canonical value to evaluate to a finite float here instead.
        _require_finite_as_float(self.canonical_decimal_value, field_name="canonical_decimal_value")
        return self

    @model_validator(mode="after")
    def _validate_unit_normalization_against_the_recorded_table(self) -> MeasuredValue:
        """Reject a ``unit_normalized`` that is not the RECORDED table's own answer.

        A ``MeasuredValue`` is validated against the table it RECORDS
        (``conversion_table_sha256``), never against "the current table" --
        an unresolvable sha is refused rather than silently re-interpreted
        against whatever table happens to be live today. And within that
        recorded table, ``unit_normalized`` must be exactly
        :func:`~carmel.services.units.normalize_unit`'s own answer for
        ``(quantity_kind, unit_raw)`` -- this is the check that keeps a
        spelling normalization (``"°C"`` -> ``"C"``, still Celsius) from
        silently becoming a unit CONVERSION (``"°C"`` -> ``"K"``, a
        different fact) as an unverified claim in the payload.
        """
        try:
            table = units.table_for_sha(self.conversion_table_sha256)
        except units.UnknownConversionTableError as exc:
            raise ValueError(
                f"conversion_table_sha256={self.conversion_table_sha256!r} does not name any known "
                "conversion table; a MeasuredValue is validated against the table it RECORDS, never "
                f"against 'the current table', so an unresolvable sha is refused rather than silently "
                f"re-interpreted: {exc}"
            ) from exc
        try:
            expected = units.normalize_unit(self.quantity_kind, self.unit_raw, table=table)
        except units.UnknownUnitError as exc:
            raise ValueError(
                f"unit_raw={self.unit_raw!r} is not a known unit or alias of quantity_kind="
                f"{self.quantity_kind.value!r} in conversion table {self.conversion_table_sha256!r}: {exc}"
            ) from exc
        if expected != self.unit_normalized:
            raise ValueError(
                f"unit_normalized={self.unit_normalized!r} disagrees with the recorded table's own "
                f"normalization of unit_raw={self.unit_raw!r} for quantity_kind="
                f"{self.quantity_kind.value!r}, which is {expected!r}; unit_normalized must be exactly "
                "the table's answer, never asserted independently"
            )
        return self

    def converted_to_base(self) -> units.Converted:
        """Convert ``canonical_decimal_value`` to this quantity's base unit.

        This is what replaced the stored ``conversion_factor``/
        ``unit_canonical``: the conversion is recomputed from recorded facts
        (``conversion_table_sha256``, ``quantity_kind``, ``unit_normalized``)
        on demand, so it can never drift out of step with the table, and the
        manufactured-precision problem (see the class docstring) stays out
        of the addressed store. For ``QuantityKind.OTHER``, which has no
        base unit, this converts to ``unit_normalized`` itself -- i.e. an
        identity conversion. See :func:`~carmel.services.units.convert` for
        the pinned rounding policy; the returned :class:`~carmel.services.units.Converted`
        carries both the ``exact`` and the ``rounded`` result.
        """
        table = units.table_for_sha(self.conversion_table_sha256)
        if self.quantity_kind is QuantityKind.OTHER:
            target_unit = self.unit_normalized
        else:
            target_unit = table.base_unit(self.quantity_kind)
        return units.convert(
            self.canonical_decimal_value,
            quantity=self.quantity_kind,
            from_unit=self.unit_normalized,
            to_unit=target_unit,
            table=table,
        )


class UncertaintyKind(StrEnum):
    """Kind of uncertainty a reported value carries."""

    STD_DEV = "std_dev"
    CI_95 = "ci_95"
    INSTRUMENT_ERROR = "instrument_error"
    UNSPECIFIED_PERCENTAGE = "unspecified_percentage"
    """A bare percentage figure (e.g. "+-5%") whose statistical method was
    NOT stated by the source. This kind's entire meaning is the absence of a
    method, so it BLOCKS statistical interpretation (see
    :attr:`Uncertainty.blocks_statistical_interpretation`) exactly like
    :attr:`UNKNOWN`, even when ``basis``, ``scale``, and a bound are all
    present -- a consumer still cannot know whether that "+-5%" is a standard
    deviation, a 95% confidence interval, or an instrument spec. Measured
    over the 8 real corpus papers: an explicit uncertainty KIND is stated in
    only 1 of 8, so this (like ``UNKNOWN``) is the common case, not an edge
    case."""
    UNKNOWN = "unknown"
    """First-class, NOT a lower-quality fallback: measured directly on this
    corpus, only 1 of 8 real papers states its uncertainty kind, making
    ``unknown`` the COMMON case, not an exceptional one. A faithfully
    extracted value whose paper never stated an uncertainty kind is not
    itself lower quality -- what changes is that downstream statistical
    interpretation (e.g. computing a weighted fit) should be blocked/flagged
    for it, which is exactly what :attr:`Uncertainty.blocks_statistical_interpretation`
    encodes, deliberately kept separate from any notion of extraction quality."""


class UncertaintyBasis(StrEnum):
    """Whether an uncertainty bound is absolute or relative to the value."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class UncertaintyScale(StrEnum):
    """Whether an uncertainty bound is expressed on a linear or log scale."""

    LINEAR = "linear"
    LOG = "log"


class Uncertainty(BaseModel):
    """An uncertainty on a :class:`MeasuredValue`, asymmetric by construction.

    ``upper``/``lower`` are two independent :class:`MeasuredValue` bounds
    (not a single ``+-`` figure and not bare strings), because a
    symmetric-only model cannot represent an asymmetric confidence interval
    or log-scale error bar without lossy averaging, AND a magnitude with no
    unit and no provenance reintroduces the exact unit-binding failure
    :class:`MeasuredValue` exists to prevent (a bare ``"0.05"`` could be 0.05
    cm/s, 0.05 m/s, or 5%, and nothing would point at where it came from).
    Relative uncertainty is expressed as a ``MeasuredValue`` whose unit is
    ``%`` or ``-``.

    ``basis``, ``scale``, ``upper`` and ``lower`` are all ``Maybe[...]``-typed
    for the same reason ``kind`` includes :attr:`UncertaintyKind.UNKNOWN`:
    measured directly on this corpus, basis/scale are almost never stated
    alongside a bare reported error, and a schema that required concrete
    values for them would force an extractor to invent a basis or scale the
    paper never gave -- exactly the fabrication this project exists to
    prevent. A paper reporting a bare "+-5%" with no stated method must be
    representable with its magnitude recorded and its basis/scale explicitly
    absent, not forced into ``ABSOLUTE``/``LINEAR`` as an invented default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: UncertaintyKind
    basis: Maybe[UncertaintyBasis]
    scale: Maybe[UncertaintyScale]
    upper: Maybe[MeasuredValue]
    """Upper bound magnitude, with its own unit and provenance. Absent if the
    source never states it (e.g. a symmetric-only report, or no bound at all)."""
    lower: Maybe[MeasuredValue]
    """Lower bound magnitude, with its own unit and provenance. Absent if the
    source never states it."""

    @model_validator(mode="after")
    def _validate_bounds_are_positive_magnitudes(self) -> Uncertainty:
        """Reject an ``upper``/``lower`` bound whose magnitude is zero or negative.

        An uncertainty bound is a DISTANCE from the reported value, never a
        signed quantity: an uncertainty of ``-5`` has no meaning (a bound
        cannot be "5 less than the true value" -- that would just be a
        different reported value). Asymmetric uncertainty is represented by
        ``upper``/``lower`` differing in SIZE (e.g. upper=8, lower=3), never
        by one of them being negative. ``MeasuredValue.canonical_decimal_value``
        being finite (enforced there) is not enough on its own -- this
        additionally rejects a bound that is finite but zero or negative.
        """
        for bound_name, bound in (("upper", self.upper), ("lower", self.lower)):
            if isinstance(bound, MeasuredValue) and Decimal(bound.canonical_decimal_value) <= 0:
                raise ValueError(
                    f"{bound_name}.canonical_decimal_value={bound.canonical_decimal_value!r} must be "
                    "strictly positive: an uncertainty bound is a magnitude (a distance), never "
                    "negative or zero -- asymmetric uncertainty is represented by upper/lower "
                    "differing in size, never by a negative bound"
                )
        return self

    @property
    def blocks_statistical_interpretation(self) -> bool:
        """False iff this uncertainty is FULLY quantified; True otherwise.

        "Fully quantified" requires ALL FOUR of: ``kind`` is known (i.e.
        neither :attr:`UncertaintyKind.UNKNOWN` nor
        :attr:`UncertaintyKind.UNSPECIFIED_PERCENTAGE` -- ``kind`` has no
        ``Absent`` state of its own; those two members are its "not stated"
        sentinels), ``basis`` is known (not :class:`Absent`), ``scale`` is
        known (not :class:`Absent`), AND at least one of ``upper``/``lower``
        is present (not :class:`Absent`). A known ``kind`` alone is NOT
        sufficient: a known kind with basis, scale, and both bounds all
        :class:`Absent` is "known kind, no usable magnitude", which is just
        as statistically useless as an unknown kind -- treating it as usable
        would let a paper's bare "+-5%, method unstated" be silently read as
        a fully quantified standard deviation, the exact failure this
        property exists to prevent. Nor is a fully-populated
        ``UNSPECIFIED_PERCENTAGE`` sufficient on its own: that kind's entire
        meaning is that the statistical method was NOT stated, so even with
        ``basis``, ``scale``, and a bound all present, a consumer still
        cannot know whether the figure is a standard deviation, a 95%
        confidence interval, or an instrument spec -- it must block exactly
        as ``UNKNOWN`` does. Deliberately NOT a quality signal -- see
        :attr:`UncertaintyKind.UNKNOWN`'s docstring. This flag tells a
        downstream consumer "do not compute a weighted statistic from this
        bound", it does not say "this extraction is worse than one with a
        stated kind"."""
        if self.kind in (UncertaintyKind.UNKNOWN, UncertaintyKind.UNSPECIFIED_PERCENTAGE):
            return True
        if isinstance(self.basis, Absent) or isinstance(self.scale, Absent):
            return True
        return isinstance(self.upper, Absent) and isinstance(self.lower, Absent)


class CompositionResolution(StrEnum):
    """Discriminates whether a :class:`Composition` has resolved components."""

    RESOLVED_COMPONENTS = "resolved_components"
    UNRESOLVED_NAMED_MIXTURE = "unresolved_named_mixture"
    """The source names a mixture (e.g. "air") without stating its numeric
    composition. See :class:`Composition` for why this must carry NO
    components rather than a default split."""


class CompositionBasis(StrEnum):
    """The basis explicit component fractions are expressed in."""

    MOLE_FRACTION = "mole_fraction"
    VOLUME_PERCENT = "volume_percent"
    MASS_FRACTION = "mass_fraction"
    PPM = "ppm"


class ComponentRole(StrEnum):
    """The role a composition component plays in the mixture."""

    FUEL = "fuel"
    OXIDIZER = "oxidizer"
    DILUENT = "diluent"
    BATH_GAS = "bath_gas"
    BALANCE = "balance"


def _component_role_sort_key(role: Maybe[ComponentRole]) -> tuple[int, str]:
    """A TOTAL, deterministic ordering over ``Maybe[ComponentRole]``, used to both
    sort and de-duplicate-key :class:`CompositionComponent`.``role``.

    ``role`` can be an actual :class:`ComponentRole` or an :class:`Absent` marker,
    and the two are not otherwise comparable (``Absent`` carries no ordering of its
    own, and mixing a ``StrEnum`` with an arbitrary ``BaseModel`` in one sort has no
    default behavior Python would pick consistently). Rather than leave that
    ambiguous, one explicit order is pinned here: every ``Absent`` role sorts
    BEFORE every present ``ComponentRole``, and all ``Absent`` roles compare equal
    to each other regardless of ``reason``/``note`` (an unstated role carries no
    further distinguishing information to sort or de-duplicate on). Among present
    roles, ordering is by the enum's own string value. This is address-bearing --
    :class:`Composition`'s component order feeds ``identity_payload()`` -- so it
    must be exactly this one stable order, not "whatever Python's default
    comparison happens to do" (which would in fact be a ``TypeError``, since
    ``Absent`` instances and ``ComponentRole`` members are not ``<``-comparable).
    """
    if isinstance(role, Absent):
        return (0, "")
    return (1, role.value)


class CompositionComponent(BaseModel):
    """One resolved component of a :class:`Composition`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species_raw_name: str = Field(min_length=1)
    amount: MeasuredValue
    role: Maybe[ComponentRole]
    """The component's role in the mixture (fuel, oxidizer, diluent, ...).
    ``Maybe``-typed, with no default, for the same reason documented on
    :class:`Uncertainty`'s basis/scale fields: measured directly on this
    corpus, ``"air"`` appears as an unresolved token in roughly 5 of 8 real
    papers, no paper ever restates the O2:N2 ratio numerically, and
    components routinely appear in the source with no stated role at all
    (e.g. a bare species list). A mandatory concrete :class:`ComponentRole`
    would force the extractor to invent ``fuel``/``diluent``/etc. for a
    component whose role the paper never states -- exactly the fabrication
    this schema exists to make structurally impossible -- so this field must
    be able to say "not stated" via an explicit :class:`Absent` rather than
    silently defaulting to a guess."""


class Composition(BaseModel):
    """A gas mixture composition, unresolved-mixture-safe by construction.

    Measured directly on this corpus: "air" is an unresolved token in ~5 of 8
    papers, and NO paper restates the O2:N2 ratio numerically. If this schema
    required a resolved composition for every source, an extractor faced with
    "air" WOULD expand it to 0.21/0.79 mole fraction and fabricate a number
    the paper never stated -- exactly the failure mode this project exists to
    prevent. So ``resolution=UNRESOLVED_NAMED_MIXTURE`` is first-class and is
    enforced (see the validator below) to carry NO ``components`` at all. A
    downstream simulator adapter may still apply a dry-air default split, but
    that is a downstream assumption applied at simulation time, never
    something this schema lets get stored as dataset truth.

    ``basis`` and ``equivalence_ratio`` are both ``Maybe[...]``-typed because
    each is itself frequently unstated in the source (a resolution basis is
    meaningless for an unresolved mixture; an equivalence ratio is routinely
    absent even for resolved compositions) -- using the six explicit
    :class:`AbsenceReason` values here keeps "not stated" distinguishable from
    "stated as zero" or "inherits from the parent dataset".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_name: str = Field(min_length=1)
    """Verbatim mixture name/description as printed in the source (e.g.
    ``"air"``, ``"synthetic air"``, or a descriptive phrase)."""
    resolution: CompositionResolution
    basis: Maybe[CompositionBasis]
    equivalence_ratio: Maybe[MeasuredValue]
    components: tuple[CompositionComponent, ...] = ()

    @model_validator(mode="after")
    def _enforce_unresolved_has_no_components(self) -> Composition:
        if self.resolution == CompositionResolution.UNRESOLVED_NAMED_MIXTURE and self.components:
            raise ValueError(
                "a Composition with resolution=UNRESOLVED_NAMED_MIXTURE must not carry components -- "
                "doing so would fabricate a numeric split (e.g. air -> 0.21/0.79) that the source "
                "never stated; components may only be attached to a RESOLVED_COMPONENTS composition"
            )
        return self

    @model_validator(mode="after")
    def _enforce_resolved_has_components(self) -> Composition:
        if self.resolution == CompositionResolution.RESOLVED_COMPONENTS and not self.components:
            raise ValueError(
                "a Composition with resolution=RESOLVED_COMPONENTS must carry at least one component -- "
                "an empty components list would let a downstream consumer read this as a resolved "
                "composition with no actual composition data; a composition with nothing resolved "
                "must be represented as UNRESOLVED_NAMED_MIXTURE instead"
            )
        return self

    @model_validator(mode="after")
    def _enforce_no_duplicate_component_species(self) -> Composition:
        """Reject a repeated ``(species_raw_name, role)`` pair among ``components``.

        Keying on ``species_raw_name`` alone would make a legitimate source shape
        unrepresentable: a paper can list the same species twice in DIFFERENT roles
        before mixture aggregation -- e.g. N2 as the oxidizer-diluent implicit
        within "air", and N2 again as a separately-added diluent. Those are two
        distinct, individually-sourced components that happen to share a species,
        not a duplicate. Keying on the pair instead only rejects the case that is
        actually ambiguous: the SAME species in the SAME role listed twice, which
        would make "which component's amount is THE amount for this species in
        this role" ambiguous for any downstream consumer that indexes components
        by (species, role) -- the same failure mode S1/S6 close one level down,
        for axes and points within a :class:`Series`.

        ``role`` is compared via :func:`_component_role_sort_key` (see its
        docstring for why ``Absent`` needs an explicit, total ordering here) so
        that two components with the same species and both an ``Absent`` role are
        still caught as a duplicate -- an unstated role carries no information
        that would distinguish them.
        """
        seen: set[tuple[str, tuple[int, str]]] = set()
        for component in self.components:
            key = (component.species_raw_name, _component_role_sort_key(component.role))
            if key in seen:
                raise ValueError(
                    f"Composition(raw_name={self.raw_name!r}): duplicate (species_raw_name, role) "
                    f"{(component.species_raw_name, component.role)!r} in components"
                )
            seen.add(key)
        return self

    @model_validator(mode="after")
    def _enforce_components_sorted_by_species(self) -> Composition:
        """``components`` must be sorted ascending by ``(species_raw_name, role)``.

        Pinning one canonical ordering here (rather than leaving component
        order to whatever sequence an extractor happened to emit) is what
        makes ``identity_payload()`` produce the same content address for
        logically-identical mixtures regardless of input order -- the same
        rationale as S2/S7/E1b for ``axes``/``points``/``series``. ``role`` is
        included in the key (not just ``species_raw_name``) now that the same
        species may legitimately appear more than once in different roles; see
        :func:`_component_role_sort_key` for the ``Absent``-inclusive total order
        this uses.
        """
        expected = tuple(
            sorted(
                self.components,
                key=lambda component: (component.species_raw_name, _component_role_sort_key(component.role)),
            )
        )
        if self.components != expected:
            raise ValueError(
                f"Composition(raw_name={self.raw_name!r}): components must be sorted ascending by "
                "(species_raw_name, role)"
            )
        return self

    @model_validator(mode="after")
    def _enforce_basis_matches_component_quantity_kind(self) -> Composition:
        """Reject a ``basis`` that disagrees with what its components actually measure.

        ``basis`` names the arithmetic space a resolved composition's
        fractions live in (mole fraction, mass fraction, ...); each
        component's ``amount.quantity_kind`` is the independently-validated
        physical quantity that value was bound to at extraction time (see
        :class:`MeasuredValue`). Without this check, ``basis`` is a free-text
        label a downstream consumer must trust blind -- a component amount
        that is actually a mass fraction could be labeled
        ``basis=MOLE_FRACTION`` and nothing here would ever notice, silently
        corrupting every arithmetic operation (e.g. summing a mixture to 1)
        that trusts the label.

        ``CompositionBasis.VOLUME_PERCENT`` is a judgment call, not a fact
        this codebase measured: :class:`QuantityKind` has no dedicated
        volume-fraction member, so this validator maps it onto
        ``QuantityKind.MOLE_FRACTION``. That mapping is only exact for an
        ideal gas (Amagat's law: volume fraction equals mole fraction for
        ideal-gas mixtures at the same temperature and pressure), which is
        the standard assumption combustion sources make when they report a
        gas mixture by volume percent. It is NOT exact for a liquid mixture
        or a real (non-ideal) gas; this schema has no way to distinguish
        those cases from the ``basis`` value alone, so it accepts the
        combustion-corpus-typical case rather than rejecting every
        ``VOLUME_PERCENT`` composition outright.

        ``CompositionBasis.PPM`` additionally requires
        ``amount.unit_normalized == "ppm"`` specifically (not merely a
        MOLE_FRACTION/MASS_FRACTION quantity_kind): both quantities also
        accept unit ``"1"`` (a bare fraction), and a composition whose basis
        is explicitly stated as parts-per-million must not silently accept a
        component recorded as a bare fraction instead -- that would let a
        components-summed-to-1 mixture and a components-summed-to-1e6
        mixture both claim ``basis=PPM``.
        """
        if self.resolution != CompositionResolution.RESOLVED_COMPONENTS:
            return self
        if isinstance(self.basis, Absent):
            return self
        basis = self.basis
        expected_quantity_kind = {
            CompositionBasis.MOLE_FRACTION: QuantityKind.MOLE_FRACTION,
            CompositionBasis.MASS_FRACTION: QuantityKind.MASS_FRACTION,
            CompositionBasis.VOLUME_PERCENT: QuantityKind.MOLE_FRACTION,
            CompositionBasis.PPM: None,
        }[basis]
        for component in self.components:
            amount = component.amount
            if basis == CompositionBasis.PPM:
                if amount.quantity_kind not in (QuantityKind.MOLE_FRACTION, QuantityKind.MASS_FRACTION):
                    raise ValueError(
                        f"component {component.species_raw_name!r} has amount.quantity_kind="
                        f"{amount.quantity_kind!r}, but basis=PPM requires a MOLE_FRACTION or "
                        "MASS_FRACTION quantity_kind (ppm is representable under either)"
                    )
                if amount.unit_normalized != "ppm":
                    raise ValueError(
                        f"component {component.species_raw_name!r} has amount.unit_normalized="
                        f"{amount.unit_normalized!r}, but basis=PPM requires unit_normalized == 'ppm' "
                        "specifically -- a bare fraction (unit '1') under the same quantity_kind is a "
                        "different fact and must not be labeled PPM"
                    )
            elif amount.quantity_kind is not expected_quantity_kind:
                raise ValueError(
                    f"component {component.species_raw_name!r} has amount.quantity_kind="
                    f"{amount.quantity_kind!r}, but basis={basis!r} requires "
                    f"quantity_kind={expected_quantity_kind!r}"
                )
            elif basis == CompositionBasis.VOLUME_PERCENT and amount.unit_normalized != "%":
                # Same rule as PPM above, for the same reason, and it must not be
                # weaker just because VOLUME_PERCENT shares MOLE_FRACTION's quantity
                # kind: constraining the KIND alone still admits a bare fraction.
                # A basis of VOLUME_PERCENT holding `0.21` reads as 0.21% to any
                # consumer that trusts the basis, and as 21% to one that trusts the
                # unit -- a silent factor of 100 in a composition, which is exactly
                # the class of corruption this binding exists to prevent. Percent is
                # a real unit in the table (it scales to the base by 0.01), so the
                # honest representation of "21 vol%" is `21` with unit `%`, never
                # `0.21` with unit `1` under a percent basis.
                raise ValueError(
                    f"component {component.species_raw_name!r} has amount.unit_normalized="
                    f"{amount.unit_normalized!r}, but basis=VOLUME_PERCENT requires "
                    "unit_normalized == '%' specifically -- a bare fraction (unit '1') "
                    "under the same quantity_kind is the same number meaning something "
                    "100x different, and must not be labeled a percent basis"
                )
        return self


class AxisRole(StrEnum):
    """What role an axis plays within a :class:`Series`."""

    COORDINATE = "coordinate"
    """Varies point to point; together with the series' other COORDINATE
    axes, it locates the point among its siblings (e.g. equivalence ratio in
    a burning-velocity-vs-phi series)."""

    OBSERVATION = "observation"
    """What was measured or computed at that located point (e.g. burning
    velocity itself)."""

    CONSTANT = "constant"
    """Fixed for the whole series and cited once via :attr:`Series.constants`,
    never repeated per point (e.g. the pressure a whole burning-velocity
    sweep was run at)."""


class SourceForm(StrEnum):
    """WHERE, physically, a series' numbers were read from in the source.

    Orthogonal to :class:`ValueOrigin` (which is about HOW the numbers were
    produced): a simulation's results can still be read off a table
    (TABULAR) or off a plotted figure (DIGITIZED) -- the two axes vary
    independently. See :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS`-adjacent
    :class:`DatasetEnvelope` validator V4 for what each member constrains.
    """

    TABULAR = "tabular"
    DIGITIZED = "digitized"
    TEXTUAL = "textual"


class ValueOrigin(StrEnum):
    """HOW a series' numbers were produced, as ASSERTED by the extractor.

    This is an extractor ASSERTION, not a grounded claim the way a
    :class:`SourceRef` is: nothing in this schema layer verifies that a
    series claiming ``EXPERIMENTAL`` truly reports a measurement rather than
    a simulation the source itself mislabels, or that a ``DERIVED``
    quantity's derivation was performed correctly -- there is no
    machine-checkable ground truth for "how was this number produced" the
    way there is for "does this SourceRef resolve". Turning this assertion,
    combined with everything else known about the source, into an actual
    TRUST score is M-D3's job, not this schema's; this field only records
    what the extractor claims.
    """

    EXPERIMENTAL = "experimental"
    SIMULATION = "simulation"
    DERIVED = "derived"


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_identifier(value: str, *, field_name: str) -> str:
    """Reject an id that is not a plain lowercase ASCII identifier.

    ``axis_id``/``series_id``/``point_id`` values appear inside dotted
    diagnostic paths (see :func:`iter_source_refs`) and inside the
    content-addressed payload itself: an id containing ``.``, ``[``, ``]``,
    a quote character, or a Unicode confusable would be indistinguishable
    from path-parsing syntax (a ``.`` inside an id looks exactly like a
    field-name separator) or could collide/fail-to-collide unpredictably
    under normalization. Restricting every such id to
    ``^[a-z][a-z0-9_]*$`` removes the entire class at construction time
    rather than leaving it to whatever code later parses a diagnostic path.
    """
    if not _IDENTIFIER_PATTERN.match(value):
        raise ValueError(
            f"{field_name}={value!r} must match {_IDENTIFIER_PATTERN.pattern!r} -- ids appear inside "
            "dotted diagnostic paths and the content-addressed payload, where a character other than "
            "[a-z0-9_] (or a Unicode confusable) would poison path parsing or collide unpredictably"
        )
    return value


def _validate_uncertainty_bound_quantity_kind(
    *, uncertainty: Maybe[Uncertainty], value: Maybe[MeasuredValue], where: str
) -> None:
    """S13: an :class:`Uncertainty` bound's ``quantity_kind`` must agree with
    its ``basis``, on BOTH :class:`Coordinate` and :class:`Observation`.

    Factored into one module-level helper (rather than duplicated on both
    models) so there is a SINGLE implementation of this rule: an
    ``ABSOLUTE`` bound is a magnitude in the same physical quantity as the
    value itself (e.g. an absolute burning-velocity uncertainty is itself a
    velocity), so its ``quantity_kind`` must equal ``value.quantity_kind``;
    a ``RELATIVE`` bound is a fraction/percentage OF the value, so its
    ``quantity_kind`` must be ``QuantityKind.RELATIVE_UNCERTAINTY``
    specifically, never the value's own quantity_kind. Skipped entirely when
    ``uncertainty``, ``uncertainty.basis``, or ``value`` is ``Absent``:
    there is nothing to cross-check a bound's quantity against when either
    side of the comparison was never stated.
    """
    if isinstance(uncertainty, Absent) or isinstance(value, Absent):
        return
    basis = uncertainty.basis
    if isinstance(basis, Absent):
        return
    expected = value.quantity_kind if basis == UncertaintyBasis.ABSOLUTE else QuantityKind.RELATIVE_UNCERTAINTY
    for bound_name, bound in (("upper", uncertainty.upper), ("lower", uncertainty.lower)):
        if isinstance(bound, Absent):
            continue
        if bound.quantity_kind is not expected:
            raise ValueError(
                f"{where}: uncertainty bound quantity -- {bound_name}.quantity_kind="
                f"{bound.quantity_kind!r} does not match the required {expected!r} for basis={basis!r}"
            )


class AxisDeclaration(BaseModel):
    """One axis of a :class:`Series`: its role, physical quantity, and where
    its printed label was read from.

    ``label_ref`` grounds the axis HEADER itself (e.g. the table column
    header or figure axis label): the header is a claim about what the
    column/axis means just as much as any individual value is a claim about
    a number, and this project's cardinal rule (see the module docstring)
    applies to it identically -- a fabricated or unverifiable axis label
    would silently mislabel every value recorded under it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    role: AxisRole
    quantity_kind: QuantityKind
    label_raw: str = Field(min_length=1)
    """Verbatim header/axis text as printed in the source (e.g. ``"phi"``,
    ``"S_L (cm/s)"``)."""
    label_ref: SourceRef

    @field_validator("axis_id")
    @classmethod
    def _validate_axis_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="axis_id")


class Coordinate(BaseModel):
    """A value that is always present: either a per-point locating
    coordinate, or one of a :class:`Series`' whole-series constants.

    Unlike :class:`Observation`, ``value`` here is a plain
    :class:`MeasuredValue` with no ``Absent`` option: a coordinate that
    could be absent would not locate the point it is attached to, and a
    constant that could be absent is not actually constant for the series --
    either case belongs to :class:`Observation` instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    value: MeasuredValue
    uncertainty: Maybe[Uncertainty]

    @field_validator("axis_id")
    @classmethod
    def _validate_axis_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="axis_id")

    @model_validator(mode="after")
    def _validate_uncertainty_bound_quantity(self) -> Coordinate:
        _validate_uncertainty_bound_quantity_kind(
            uncertainty=self.uncertainty, value=self.value, where=f"Coordinate(axis_id={self.axis_id!r})"
        )
        return self


class Observation(BaseModel):
    """A value that may be explicitly absent: what was measured/computed at
    a point along one ``OBSERVATION`` axis.

    ``value`` is ``Maybe[MeasuredValue]`` (unlike :class:`Coordinate`,
    which is never absent) because a point can genuinely lack an observed
    value the source never reported for it (e.g. one row of a table left a
    column blank) -- see :class:`Absent` for why that must be an explicit,
    reasoned state rather than a bare ``None`` or an omitted slot (also see
    S9 on :class:`Series`, which requires the slot to exist regardless).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_id: str
    value: Maybe[MeasuredValue]
    uncertainty: Maybe[Uncertainty]

    @field_validator("axis_id")
    @classmethod
    def _validate_axis_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="axis_id")

    @model_validator(mode="after")
    def _validate_uncertainty_without_value(self) -> Observation:
        """S12: an absent ``value`` implies an absent ``uncertainty``.

        An uncertainty bound is meaningless without the value it bounds --
        "the error bar on a number that was never reported" is not a fact
        this schema can represent, so a populated ``uncertainty`` alongside
        ``value=Absent(...)`` is rejected outright rather than silently
        accepted as orphaned metadata.
        """
        if isinstance(self.value, Absent) and not isinstance(self.uncertainty, Absent):
            raise ValueError(
                f"Observation(axis_id={self.axis_id!r}): uncertainty without a value -- an uncertainty "
                "bound on a value that was never reported is not a representable fact; uncertainty "
                "must be Absent whenever value is Absent"
            )
        return self

    @model_validator(mode="after")
    def _validate_uncertainty_bound_quantity(self) -> Observation:
        _validate_uncertainty_bound_quantity_kind(
            uncertainty=self.uncertainty, value=self.value, where=f"Observation(axis_id={self.axis_id!r})"
        )
        return self


class DataPoint(BaseModel):
    """One located point of a :class:`Series`: its coordinates, its
    observations, and an optional per-point composition override."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point_id: str
    coordinates: tuple[Coordinate, ...]
    observations: tuple[Observation, ...]
    composition: Maybe[Composition]
    """A per-point composition override. ``Maybe``-typed with no default:
    most points inherit the dataset's single :class:`DatasetEnvelope`
    composition unchanged, but a series that sweeps composition itself
    (rather than sweeping equivalence ratio at one fixed composition) needs
    a way to say so per point, and an unstated per-point composition must
    stay distinguishable from an explicit ``SAME_AS_DATASET`` claim -- see
    :class:`AbsenceReason`."""

    @field_validator("point_id")
    @classmethod
    def _validate_point_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="point_id")


class Series(BaseModel):
    """One aggregate series of data points extracted from a source: a set
    of declared axes, whole-series constants, and the points themselves.

    This is the M-D2b(a) aggregate: the structural container that lets a
    multi-point table or digitized plot be recorded as ONE grounded object
    (one ``source_form``, one ``value_origin``, one fixed axis schema)
    rather than as N independent, uncorrelated :class:`MeasuredValue`
    fragments with no shared structure tying them together as one dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    series_id: str
    source_form: SourceForm
    value_origin: ValueOrigin
    axes: tuple[AxisDeclaration, ...]
    constants: tuple[Coordinate, ...]
    points: tuple[DataPoint, ...]

    @field_validator("series_id")
    @classmethod
    def _validate_series_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="series_id")

    @model_validator(mode="after")
    def _validate_no_duplicate_axis_id(self) -> Series:
        """S1: reject a repeated ``axis_id`` among ``axes``.

        Every downstream lookup (S3-S5, S8, S9, S11, and V4) indexes axes BY
        ``axis_id``; a duplicate id would make "which axis declaration does
        this coordinate/observation's axis_id refer to" ill-defined.
        """
        seen: set[str] = set()
        for axis in self.axes:
            if axis.axis_id in seen:
                raise ValueError(
                    f"Series(series_id={self.series_id!r}): duplicate axis_id {axis.axis_id!r} in axes"
                )
            seen.add(axis.axis_id)
        return self

    @model_validator(mode="after")
    def _validate_axes_sorted_and_nonempty(self) -> Series:
        """S2: ``axes`` must be sorted ascending by ``axis_id`` and
        non-empty.

        Deliberately checked here rather than via ``Field(min_length=1)``,
        so that BOTH the sortedness failure and the non-emptiness failure
        raise through the same marker-bearing message: a bare
        ``Field(min_length=1)`` failure carries pydantic's own generic
        "too_short" text, which a mutation test could satisfy whether or
        not this validator's actual sortedness check exists at all (the
        exact mutation-testing failure mode this spec calls out).
        """
        expected = tuple(sorted(self.axes, key=lambda axis: axis.axis_id))
        if not self.axes or self.axes != expected:
            raise ValueError(
                f"Series(series_id={self.series_id!r}): axes must be sorted ascending by axis_id and "
                "must contain at least one axis"
            )
        return self

    @model_validator(mode="after")
    def _validate_has_coordinate_axis(self) -> Series:
        """S3: a series must declare at least one ``COORDINATE`` axis.

        A series with zero locating axes cannot place a point among its
        siblings at all -- locating a point is the entire purpose a
        ``COORDINATE`` axis serves.
        """
        if not any(axis.role == AxisRole.COORDINATE for axis in self.axes):
            raise ValueError(
                f"Series(series_id={self.series_id!r}) must declare at least one coordinate axis"
            )
        return self

    @model_validator(mode="after")
    def _validate_has_observation_axis(self) -> Series:
        """S4: a series must declare at least one ``OBSERVATION`` axis.

        A series with no observation axis records locations but nothing
        measured at them -- it would be a table of coordinates with no
        data, not a dataset.
        """
        if not any(axis.role == AxisRole.OBSERVATION for axis in self.axes):
            raise ValueError(
                f"Series(series_id={self.series_id!r}) must declare at least one observation axis"
            )
        return self

    @model_validator(mode="after")
    def _validate_constants_cover_constant_axes(self) -> Series:
        """S5: ``constants`` must cover EXACTLY the series' ``CONSTANT``
        axes, sorted by ``axis_id``, with no duplicates.

        Reuses the idiom from
        :meth:`Composition._enforce_basis_matches_component_quantity_kind`:
        a single list-equality check against a canonically sorted expected
        id list simultaneously enforces set-coverage (every CONSTANT axis
        has a value and nothing extra is present), sortedness, AND
        uniqueness in one comparison.
        """
        constant_axis_ids = sorted(axis.axis_id for axis in self.axes if axis.role == AxisRole.CONSTANT)
        actual = [constant.axis_id for constant in self.constants]
        if actual != constant_axis_ids:
            raise ValueError(
                f"Series(series_id={self.series_id!r}): constants must cover exactly the series' "
                f"CONSTANT axes, sorted by axis_id with no duplicates (expected {constant_axis_ids!r}, "
                f"got {actual!r})"
            )
        return self

    @model_validator(mode="after")
    def _validate_no_duplicate_point_id(self) -> Series:
        """S6: reject a repeated ``point_id`` among ``points``."""
        seen: set[str] = set()
        for point in self.points:
            if point.point_id in seen:
                raise ValueError(
                    f"Series(series_id={self.series_id!r}): duplicate point_id {point.point_id!r} in "
                    "points"
                )
            seen.add(point.point_id)
        return self

    @model_validator(mode="after")
    def _validate_points_sorted_and_nonempty(self) -> Series:
        """S7: ``points`` must be sorted ascending by ``point_id`` and
        non-empty -- same rationale as S2 for bundling both checks under
        one custom message rather than relying on ``Field(min_length=1)``.
        """
        expected = tuple(sorted(self.points, key=lambda point: point.point_id))
        if not self.points or self.points != expected:
            raise ValueError(
                f"Series(series_id={self.series_id!r}): points must be sorted ascending by point_id "
                "and must contain at least one point"
            )
        return self

    @model_validator(mode="after")
    def _validate_coordinates_cover_coordinate_axes(self) -> Series:
        """S8: each point's ``coordinates`` must cover EXACTLY the series'
        ``COORDINATE`` axes, sorted by ``axis_id``, with no duplicates.

        A point missing a coordinate is unlocated -- it cannot be placed
        among its siblings -- so this is checked per point, not merely
        summed across the series.
        """
        coordinate_axis_ids = sorted(axis.axis_id for axis in self.axes if axis.role == AxisRole.COORDINATE)
        for point in self.points:
            actual = [coord.axis_id for coord in point.coordinates]
            if actual != coordinate_axis_ids:
                raise ValueError(
                    f"Series(series_id={self.series_id!r}) point {point.point_id!r}: coordinates must "
                    f"cover exactly the series' COORDINATE axes, sorted by axis_id with no duplicates "
                    f"(expected {coordinate_axis_ids!r}, got {actual!r}) -- a point missing a "
                    "coordinate is unlocated"
                )
        return self

    @model_validator(mode="after")
    def _validate_observations_cover_observation_axes(self) -> Series:
        """S9: each point's ``observations`` must cover EXACTLY the series'
        ``OBSERVATION`` axes, sorted by ``axis_id``, with no duplicates.

        The slot must exist for every ``OBSERVATION`` axis regardless of
        whether the source actually reported a value there -- absence is
        expressed by ``value=Absent(...)``, never by omitting the slot
        itself (the round of review that settled on "distinct states,
        never a missing field" for exactly this reason).
        """
        observation_axis_ids = sorted(axis.axis_id for axis in self.axes if axis.role == AxisRole.OBSERVATION)
        for point in self.points:
            actual = [obs.axis_id for obs in point.observations]
            if actual != observation_axis_ids:
                raise ValueError(
                    f"Series(series_id={self.series_id!r}) point {point.point_id!r}: observations must "
                    f"cover exactly the series' OBSERVATION axes, sorted by axis_id with no duplicates "
                    f"(expected {observation_axis_ids!r}, got {actual!r}) -- absence is expressed by "
                    "value=Absent(...), never by omitting the slot"
                )
        return self

    @model_validator(mode="after")
    def _validate_points_record_observed_value(self) -> Series:
        """S10: every point must carry at least one observation whose
        ``value`` is a present :class:`MeasuredValue`.

        A point where every single observation is ``Absent`` records a
        location and nothing else -- it contributes no actual data to the
        series, which is indistinguishable from the point not existing at
        all except that it silently pads the point count.
        """
        for point in self.points:
            if not any(isinstance(obs.value, MeasuredValue) for obs in point.observations):
                raise ValueError(
                    f"Series(series_id={self.series_id!r}) point {point.point_id!r} records no "
                    "observed value -- at least one observation must carry a present MeasuredValue"
                )
        return self

    @model_validator(mode="after")
    def _validate_quantity_kind_matches_axis(self) -> Series:
        """S11: every present coordinate/observation/constant value's
        ``quantity_kind`` must equal the ``quantity_kind`` its declared
        axis carries.

        Without this, ``axis_id`` is just a free-text label a downstream
        consumer must trust blind -- a value physically bound (see
        :class:`MeasuredValue`) to ``QuantityKind.TEMPERATURE`` could sit
        under an axis declared ``QuantityKind.PRESSURE`` and nothing would
        ever notice, exactly the label/physical-quantity mismatch
        :class:`Composition`'s basis check exists to prevent one layer up.
        Runs AFTER S8/S9 (declaration order), so every ``axis_id`` looked
        up here is already known to name a real axis.

        This check does NOT validate the printed HEADER text against the
        axis's ``quantity_kind`` -- ``AxisDeclaration.label_raw`` is verbatim
        source text and is never inspected here at all; what this compares
        is two already-structured fields (the axis's own ``quantity_kind``
        and the value's ``MeasuredValue.quantity_kind``), and
        ``MeasuredValue`` itself RE-DERIVES its ``unit_normalized`` from the
        recorded conversion table rather than trusting a printed unit
        string, so this guard inherits that same distance from the source
        text. Two live escape hatches remain even with this check in place:
        (1) ``QuantityKind.OTHER`` is a legitimate value on both the axis and
        the ``MeasuredValue``, so a genuinely mismatched physical quantity
        can still agree at ``OTHER`` on both sides with no unit binding to
        catch it either, and (2) nothing here re-derives or re-checks the
        header text itself against either quantity_kind. Closing that
        remaining gap between printed header text and a trusted
        quantity_kind is M-D3's job (trust computation over the extracted
        payload), not this validator's.
        """
        axis_by_id = {axis.axis_id: axis for axis in self.axes}

        def _check(axis_id: str, value: Maybe[MeasuredValue], where: str) -> None:
            if isinstance(value, Absent):
                return
            expected = axis_by_id[axis_id].quantity_kind
            if value.quantity_kind is not expected:
                raise ValueError(
                    f"{where}: quantity_kind disagrees with its axis -- value.quantity_kind="
                    f"{value.quantity_kind!r} but axis {axis_id!r} declares quantity_kind={expected!r}"
                )

        for constant in self.constants:
            _check(constant.axis_id, constant.value, f"Series(series_id={self.series_id!r}) constant")
        for point in self.points:
            for coord in point.coordinates:
                _check(
                    coord.axis_id,
                    coord.value,
                    f"Series(series_id={self.series_id!r}) point {point.point_id!r} coordinate",
                )
            for obs in point.observations:
                _check(
                    obs.axis_id,
                    obs.value,
                    f"Series(series_id={self.series_id!r}) point {point.point_id!r} observation",
                )
        return self


class EmbeddedConversionTable(BaseModel):
    """A :class:`~carmel.services.units.ConversionTable`'s own canonical
    identity-payload JSON, embedded VERBATIM in a :class:`DatasetEnvelope` so
    that a non-Carmel consumer holding only this envelope's bytes can learn
    what a cited ``conversion_table_sha256`` means (e.g. that ``atm -> Pa``
    is ``101325``) without needing :mod:`carmel.services.units` at all.

    ``canonical_json`` is a ``str``, deliberately never a ``dict``:
    ``frozen=True`` (see ``model_config`` below) is a claim about attribute
    REASSIGNMENT, not about the mutability of an object a validated,
    frozen model happens to hold -- a ``dict`` field here could be mutated
    in place after validation with nothing here to notice, which has
    already shipped as a defect twice in this project. A ``str`` is
    immutable, closing that hole structurally rather than by convention.

    SCOPE OF WHAT VALIDATION HERE PROVES (read before trusting a table):
    :meth:`_validate_canonical_json_reconstructs_to_sha256` (T1) proves only
    that ``canonical_json`` is *internally coherent* -- it parses, it
    reconstructs a structurally valid ``ConversionTable`` whose own
    ``__post_init__`` invariants all hold, and re-canonicalizing that
    reconstruction reproduces ``canonical_json`` byte-for-byte. None of that
    checks whether the table's numbers are SCIENTIFICALLY correct: a table
    that declares ``atm -> Pa`` with ``scale="1"`` would pass every one of
    these checks while being physically wrong (the true factor is 101325).
    Reconstruction can only catch INTERNAL contradictions (a malformed
    decimal, a duplicate rule, an unreachable base unit); it has no way to
    know what the real conversion factor between two physical units is.
    Actual trust that a table's numbers are right comes from a table's
    membership in :data:`carmel.services.units.TABLES_BY_SHA` -- the
    hand-reviewed, shipped registry -- not from this validator succeeding.
    An ``EmbeddedConversionTable`` whose ``sha256`` is absent from
    ``TABLES_BY_SHA`` may be internally coherent nonsense that no human ever
    checked against reality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- must equal the sha256 of the table
    that ``canonical_json`` decodes to; see
    :meth:`_validate_canonical_json_reconstructs_to_sha256` (T1)."""

    canonical_json: str = Field(min_length=1, max_length=_MAX_EMBEDDED_CANONICAL_JSON_LENGTH)
    """The table's canonical identity-payload JSON, verbatim, as a str --
    see the class docstring for why this is never a ``dict``.

    Bounded by ``_MAX_EMBEDDED_CANONICAL_JSON_LENGTH`` -- see that constant's
    comment for the resource-exhaustion guard this length bound enforces --
    ``canonical_json`` arrives from a stored file (untrusted input), and an
    unbounded string handed to ``json.loads`` is a resource-exhaustion
    vector, not just a parsing convenience."""

    @field_validator("sha256")
    @classmethod
    def _validate_sha256_shape(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(
                f"EmbeddedConversionTable.sha256 {value!r} is not 64 lowercase hex characters"
            )
        return value

    @model_validator(mode="after")
    def _validate_canonical_json_reconstructs_to_sha256(self) -> EmbeddedConversionTable:
        """T1: ``canonical_json`` must be the CANONICAL rendering of a
        conversion table whose sha256 is exactly ``self.sha256`` -- verified
        by full RECONSTRUCTION, not merely by re-hashing the stored bytes (a
        byte-hash check alone proves only self-addressing: arbitrary garbage
        can be honestly hashed against itself).

        Four steps, each with its own marker phrase so a test can tell
        exactly which one fired:

        1. parse ``canonical_json`` as JSON at all;
        2. reconstruct via ``ConversionTable.from_identity_payload`` -- this
           re-runs every ``__post_init__`` invariant on the reconstructed
           table, including the scale/offset checks an opaque string
           bypasses;
        3. the reconstructed table's own ``.sha256`` must equal the
           declared ``self.sha256``;
        4. re-projecting the reconstructed table's ``identity_payload()``
           through ``canonical_json_bytes`` must reproduce
           ``canonical_json`` byte-for-byte -- this is what pins that the
           embedded bytes are the CANONICAL rendering, not merely some JSON
           that happens to parse to an equivalent object.

        All four steps together prove internal shape/invariant COHERENCE
        only -- NOT scientific or domain correctness of the table's
        conversion factors. See the class docstring's "SCOPE OF WHAT
        VALIDATION HERE PROVES" section for why (short version: an
        internally coherent table can still be physically wrong, and actual
        trust comes from membership in the shipped ``TABLES_BY_SHA``
        registry, not from this validator passing).
        """
        try:
            parsed = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            # canonical_json is untrusted input (it arrives embedded in a stored file), so a
            # deeply-nested payload can blow the interpreter's own call stack with a bare
            # RecursionError from inside json.loads itself, before this validator gets a
            # chance to react -- mirrors the same broadened except used around
            # dataset_store._read_verified_canonical_dict's own json.loads call for the
            # identical reason. The length bound on canonical_json (see
            # _MAX_EMBEDDED_CANONICAL_JSON_LENGTH) is the primary defence; this catch is the
            # backstop for the exceedingly deep-but-short payload the length bound alone would
            # not catch.
            raise ValueError(
                f"EmbeddedConversionTable(sha256={self.sha256!r}): canonical_json does not parse as JSON: "
                f"{exc}"
            ) from exc
        try:
            reconstructed = units.ConversionTable.from_identity_payload(parsed)
        except units.ConversionTableInvariantError as exc:
            raise ValueError(
                f"EmbeddedConversionTable(sha256={self.sha256!r}): canonical_json does not decode to a "
                f"structurally valid ConversionTable: {exc}"
            ) from exc
        if reconstructed.sha256 != self.sha256:
            raise ValueError(
                f"EmbeddedConversionTable(sha256={self.sha256!r}): canonical_json decodes to a table whose "
                f"own sha256 is {reconstructed.sha256!r}, not the declared sha256"
            )
        recanonicalized = canonical_json_bytes(reconstructed.identity_payload())
        if recanonicalized != self.canonical_json.encode("utf-8"):
            raise ValueError(
                f"EmbeddedConversionTable(sha256={self.sha256!r}): canonical_json is not the canonical "
                "rendering of the table it decodes to -- re-serializing the reconstructed table's own "
                "identity_payload() through canonical_json_bytes produced different bytes"
            )
        return self


def _check_source_form_for_ref(
    *, source_form: SourceForm, ref: SourceRef, node_kind: SourceNodeKind, where: str
) -> None:
    """V4: constrain a ``value_ref`` to match the ``source_form`` claimed
    for the :class:`Series` it belongs to.

    Every branch below is constrained -- there is deliberately no
    unconstrained escape branch, so a ``source_form`` this table has no
    entry for would be a loud crash (a missing dict-style ``elif``/``else``
    match) rather than a silently unchecked case; see
    :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS` for the same design
    philosophy applied to locator/node compatibility.
    """
    if source_form == SourceForm.TABULAR:
        if ref.locator.kind is not LocatorKind.TABLE_CELL:
            raise ValueError(
                f"{where}: source_form=TABULAR requires value_ref.locator.kind=TABLE_CELL, got "
                f"{ref.locator.kind!r}"
            )
    elif source_form == SourceForm.DIGITIZED:
        if node_kind is not SourceNodeKind.FIGURE_CROP:
            raise ValueError(
                f"{where}: source_form=DIGITIZED requires value_ref to target a FIGURE_CROP node, got "
                f"node kind={node_kind!r}"
            )
    elif source_form == SourceForm.TEXTUAL:
        if ref.locator.kind is LocatorKind.TABLE_CELL or node_kind is SourceNodeKind.FIGURE_CROP:
            raise ValueError(
                f"{where}: source_form=TEXTUAL requires value_ref to be neither a TABLE_CELL locator "
                f"nor a reference to a FIGURE_CROP node, got locator kind={ref.locator.kind!r} node "
                f"kind={node_kind!r}"
            )
    else:  # pragma: no cover - exhaustiveness guard, see docstring
        raise AssertionError(f"unhandled source_form={source_form!r}; every SourceForm member must be handled above")


_LOCATOR_KIND_COMPATIBLE_NODE_KINDS: dict[LocatorKind, frozenset[SourceNodeKind]] = {
    LocatorKind.BBOX: frozenset({SourceNodeKind.PAPER_PDF, SourceNodeKind.SI_MEMBER, SourceNodeKind.FIGURE_CROP}),
    LocatorKind.TABLE_CELL: frozenset({SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER}),
    LocatorKind.XPATH: frozenset({SourceNodeKind.JATS_XML}),
}
"""Which :class:`SourceNodeKind`\\ s a given :class:`LocatorKind` may target.

A SINGLE explicit table, keyed by every ``LocatorKind`` member, rather than a
chain of per-kind ``if`` branches: a dict lookup on a key that is missing
raises ``KeyError`` immediately (see the assertion right below, which pins
that the table stays exhaustive), so a newly added ``LocatorKind`` with no
entry here is a loud crash the next time :class:`DatasetEnvelope` validates
anything, not a locator/kind pair that silently sails through unchecked --
which is exactly what an ``if``/``elif`` chain with no matching branch (and
no final ``else: raise``) would do instead.

The compatibility itself reflects what each locator actually addresses:
``XPathLocator`` only makes sense against a JATS/XML document; a bounding
box only makes sense against something that was actually RENDERED to a page
(a PDF, an SI member that is itself a rendered document, or a figure crop);
a table cell locator makes sense against anything that can carry a table (a
PDF, JATS/XML, or an SI spreadsheet member) but not a figure crop, which has
no tabular structure of its own. Which ARCHIVE a node was extracted from is
not a locator concern at all -- see :class:`ArchiveOrigin` on
:class:`SourceNode` -- so no ``LocatorKind`` addresses that here.
"""

assert set(_LOCATOR_KIND_COMPATIBLE_NODE_KINDS) == set(LocatorKind), (
    "_LOCATOR_KIND_COMPATIBLE_NODE_KINDS must have an entry for every LocatorKind member -- see its "
    "docstring for why an omission must be a loud failure rather than a silent pass"
)

_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS: dict[TableKeyKind, frozenset[SourceNodeKind]] = {
    TableKeyKind.CAPTION_LABEL: frozenset(
        {SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER}
    ),
    TableKeyKind.MEMBER_SHEET: frozenset({SourceNodeKind.SI_MEMBER}),
}
"""Which :class:`SourceNodeKind`\\ s a given :class:`TableKeyKind` may target.

Same idiom as :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS` right above, and for
the same reason: a single explicit table keyed by every ``TableKeyKind``
member, checked exhaustive by the assertion below, so a newly added
``TableKeyKind`` with no entry here is a loud ``KeyError`` the next time
:class:`DatasetEnvelope` validates anything, not a table-key/node-kind pair
that silently sails through unchecked.

A ``MEMBER_SHEET`` key names a workbook SHEET NAME -- that concept only
exists for a spreadsheet SI member, so it may only target ``SI_MEMBER``. A
``CAPTION_LABEL`` key names a caption string printed in a rendered document
-- a PDF page, a JATS/XML document, or an SI member that is itself such a
document -- so it may target any of those three, but a ``TableCellLocator``
against those node kinds is already required by
:data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS` to exclude ``FIGURE_CROP``, so
that exclusion is not repeated here.

``SI_MEMBER`` compatible with BOTH key kinds is a known remaining gap, not an
oversight: today ``SI_MEMBER`` is a single ``SourceNodeKind`` covering a csv,
an xlsx, a PDF, and a zip member all alike, so this table cannot yet tell
"this SI member is a spreadsheet, so only MEMBER_SHEET makes sense against
it" from "this SI member is a rendered document, so only CAPTION_LABEL makes
sense against it" -- both keys are accepted against SI_MEMBER because the
node kind itself does not yet distinguish those cases. M-E (SI retrieval) is
the closer for this gap -- that is the milestone where an SI member's media
type first becomes known at all: once an SI member's actual media type is
represented, this table can narrow to reject a ``MEMBER_SHEET`` key against
an SI member that is not actually a spreadsheet.
"""

assert set(_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS) == set(TableKeyKind), (
    "_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS must have an entry for every TableKeyKind member -- see its "
    "docstring for why an omission must be a loud failure rather than a silent pass"
)


def _absent_identity_payload(value: Absent) -> dict[str, Any]:
    """Project an :class:`Absent` marker to a shape no present value can produce.

    ``{"__absent__": True, "reason": ..., "note": ...}``: no other projector
    in this module ever emits a ``"__absent__"`` key, so this dict can never
    collide with the projection of a present value of any ``Maybe[T]`` --
    not a present ``str`` (projects as a bare string), not a present model
    (projects as a dict keyed on that model's own field names, never
    ``"__absent__"``). ``reason`` is unwrapped to its ``.value`` like every
    other enum in this projection; ``note`` is carried through as-is (it is
    already ``str | None``).
    """
    return {"__absent__": True, "reason": value.reason.value, "note": value.note}


def _project_maybe(value: Any, project: Callable[[Any], Any] = lambda value: value) -> Any:
    """Project a ``Maybe[T]`` field: ``Absent`` becomes an uncollidable marker,
    a present value is handed to ``project`` (identity by default, for the
    ``Maybe[str]``/``Maybe[<StrEnum>]`` fields whose present projection is
    already a bare JSON-able scalar).
    """
    if isinstance(value, Absent):
        return _absent_identity_payload(value)
    return project(value)


def _coordinate_frame_identity_payload(frame: CoordinateFrame) -> dict[str, Any]:
    return {
        "render_fingerprint": frame.render_fingerprint,
        "cropbox": list(frame.cropbox),
        "mediabox": list(frame.mediabox),
        "rotation": frame.rotation,
        "units": frame.units,
        "dpi": _project_maybe(frame.dpi),
        "render_settings": _project_maybe(frame.render_settings),
    }


def _bbox_identity_payload(bbox: BBox) -> dict[str, Any]:
    return {
        "frame": _coordinate_frame_identity_payload(bbox.frame),
        "x0": bbox.x0,
        "y0": bbox.y0,
        "x1": bbox.x1,
        "y1": bbox.y1,
    }


def _table_key_identity_payload(table_key: CaptionLabelKey | MemberSheetKey) -> dict[str, Any]:
    if isinstance(table_key, CaptionLabelKey):
        return {"kind": table_key.kind.value, "label": table_key.label}
    if isinstance(table_key, MemberSheetKey):
        return {"kind": table_key.kind.value, "sheet_name": table_key.sheet_name}
    raise TypeError(f"_table_key_identity_payload: unhandled TableKey variant {table_key!r}")


def _source_locator_identity_payload(
    locator: BBoxLocator | TableCellLocator | XPathLocator,
) -> dict[str, Any]:
    if isinstance(locator, BBoxLocator):
        return {"kind": locator.kind.value, "bbox": _bbox_identity_payload(locator.bbox)}
    if isinstance(locator, TableCellLocator):
        return {
            "kind": locator.kind.value,
            "table_key": _table_key_identity_payload(locator.table_key),
            "row": locator.row,
            "col": locator.col,
        }
    if isinstance(locator, XPathLocator):
        return {"kind": locator.kind.value, "xpath": locator.xpath}
    raise TypeError(f"_source_locator_identity_payload: unhandled SourceLocator variant {locator!r}")


def _source_ref_identity_payload(ref: SourceRef) -> dict[str, Any]:
    return {"node_id": ref.node_id, "locator": _source_locator_identity_payload(ref.locator)}


def _archive_origin_identity_payload(origin: ArchiveOrigin) -> dict[str, Any]:
    # member_display_path is deliberately NOT projected here -- see
    # ArchiveOrigin's class docstring and its entry in _UNADDRESSED_FIELDS
    # below. It is display-only by contract; projecting it would make two
    # envelopes that differ only in a cosmetic display path address
    # differently, even though archive_sha256 (the actual identity-bearing
    # field) is unchanged.
    return {"archive_sha256": origin.archive_sha256}


def _source_node_identity_payload(node: SourceNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "sha256": node.sha256,
        "parent_node_id": node.parent_node_id,
        "origin": _project_maybe(node.origin, _archive_origin_identity_payload),
    }


def _source_graph_identity_payload(graph: SourceGraph) -> dict[str, Any]:
    return {"nodes": [_source_node_identity_payload(node) for node in graph.nodes]}


def _semantic_dependency_use_identity_payload(use: SemanticDependencyUse) -> dict[str, Any]:
    return {
        "dependency_id": use.dependency_id,
        "content_sha256": use.content_sha256,
        "input_sha256": _project_maybe(use.input_sha256),
    }


def _measured_value_identity_payload(value: MeasuredValue) -> dict[str, Any]:
    return {
        "raw_text": value.raw_text,
        "canonical_decimal_value": value.canonical_decimal_value,
        "repairs": list(value.repairs),
        "repair_dependency": _semantic_dependency_use_identity_payload(value.repair_dependency),
        "quantity_kind": value.quantity_kind.value,
        "unit_raw": value.unit_raw,
        "unit_normalized": value.unit_normalized,
        "conversion_table_sha256": value.conversion_table_sha256,
        "value_ref": _source_ref_identity_payload(value.value_ref),
        "unit_ref": _source_ref_identity_payload(value.unit_ref),
    }


def _uncertainty_identity_payload(uncertainty: Uncertainty) -> dict[str, Any]:
    return {
        "kind": uncertainty.kind.value,
        "basis": _project_maybe(uncertainty.basis, lambda basis: basis.value),
        "scale": _project_maybe(uncertainty.scale, lambda scale: scale.value),
        "upper": _project_maybe(uncertainty.upper, _measured_value_identity_payload),
        "lower": _project_maybe(uncertainty.lower, _measured_value_identity_payload),
    }


def _composition_component_identity_payload(component: CompositionComponent) -> dict[str, Any]:
    return {
        "species_raw_name": component.species_raw_name,
        "amount": _measured_value_identity_payload(component.amount),
        "role": _project_maybe(component.role, lambda role: role.value),
    }


def _composition_identity_payload(composition: Composition) -> dict[str, Any]:
    return {
        "raw_name": composition.raw_name,
        "resolution": composition.resolution.value,
        "basis": _project_maybe(composition.basis, lambda basis: basis.value),
        "equivalence_ratio": _project_maybe(composition.equivalence_ratio, _measured_value_identity_payload),
        "components": [_composition_component_identity_payload(c) for c in composition.components],
    }


def _axis_declaration_identity_payload(axis: AxisDeclaration) -> dict[str, Any]:
    return {
        "axis_id": axis.axis_id,
        "role": axis.role.value,
        "quantity_kind": axis.quantity_kind.value,
        "label_raw": axis.label_raw,
        "label_ref": _source_ref_identity_payload(axis.label_ref),
    }


def _coordinate_identity_payload(coordinate: Coordinate) -> dict[str, Any]:
    return {
        "axis_id": coordinate.axis_id,
        "value": _measured_value_identity_payload(coordinate.value),
        "uncertainty": _project_maybe(coordinate.uncertainty, _uncertainty_identity_payload),
    }


def _observation_identity_payload(observation: Observation) -> dict[str, Any]:
    return {
        "axis_id": observation.axis_id,
        "value": _project_maybe(observation.value, _measured_value_identity_payload),
        "uncertainty": _project_maybe(observation.uncertainty, _uncertainty_identity_payload),
    }


def _data_point_identity_payload(point: DataPoint) -> dict[str, Any]:
    return {
        "point_id": point.point_id,
        "coordinates": [_coordinate_identity_payload(c) for c in point.coordinates],
        "observations": [_observation_identity_payload(o) for o in point.observations],
        "composition": _project_maybe(point.composition, _composition_identity_payload),
    }


def _series_identity_payload(series: Series) -> dict[str, Any]:
    return {
        "series_id": series.series_id,
        "source_form": series.source_form.value,
        "value_origin": series.value_origin.value,
        "axes": [_axis_declaration_identity_payload(a) for a in series.axes],
        "constants": [_coordinate_identity_payload(c) for c in series.constants],
        "points": [_data_point_identity_payload(p) for p in series.points],
    }


def _embedded_conversion_table_identity_payload(table: EmbeddedConversionTable) -> dict[str, Any]:
    return {"sha256": table.sha256, "canonical_json": table.canonical_json}


_UNADDRESSED_FIELDS: Mapping[tuple[str, str], str] = {
    ("ArchiveOrigin", "member_display_path"): (
        "Display-only by contract (see ArchiveOrigin's class docstring): "
        "archive paths collide under normalization and can be adversarially "
        "crafted to *look* like they identify one archive member while "
        "actually addressing another, so this field carries no information "
        "that can safely be treated as identity. archive_sha256 is the sole "
        "identity-bearing field on this model and is projected instead. "
        "This is the model entry for a field that is genuinely not "
        "addressable, not a shortcut for having forgotten to project it --"
        "see the completeness meta-test for how that distinction is "
        "enforced."
    ),
}
"""``(model_name, field_name) -> reason`` registry of fields NOT covered by
:meth:`DatasetEnvelope.identity_payload`'s hand-written projection.

Every OTHER field of every pydantic model reachable from
:class:`DatasetEnvelope` (``SourceGraph``, ``SourceNode``, ``ArchiveOrigin``,
``SourceRef`` and its locator/table-key discriminated-union arms, ``BBox``,
``CoordinateFrame``, ``MeasuredValue``, ``SemanticDependencyUse``,
``Uncertainty``, ``Composition``, ``CompositionComponent``, ``Series``,
``AxisDeclaration``, ``Coordinate``, ``Observation``, ``DataPoint``,
``EmbeddedConversionTable``, and ``Absent`` itself) is projected by one of
the ``_*_identity_payload`` helpers above.
This registry exists so that a
FUTURE field added to any of those models and left unprojected is caught
by
the completeness meta-test as a loud failure -- entering it here is the
escape hatch for a field that is genuinely not addressable (e.g. a derived
``@property``), not a shortcut for "forgot to project it". Do not add an
entry to make the meta-test pass; add the projection instead.
"""


class DatasetEnvelope(BaseModel):
    """The top-level payload for one literature-extracted dataset: a source
    graph plus the extracted content that cites it.

    This is the model that makes fabrication structurally impossible at the
    WHOLE-PAYLOAD level, not just within a single :class:`MeasuredValue`:
    every :class:`SourceRef` embedded anywhere in this envelope (found via
    :func:`iter_source_refs`, which is shape-agnostic -- see its docstring)
    is checked against ``source_graph``, and the graph itself is checked for
    having nothing extra hanging off it. The validators below run in a fixed
    order (V1 through V5), each addressing a distinct, independently measured
    failure mode; see each validator's own docstring for the concrete failure
    it closes.

    A former V0 (`"the envelope must cite at least one SourceRef"`) has been
    DELETED, not merely renumbered: it was unreachable and untestable. Once
    ``series`` became a required (``min_length=1``) field whose ``axes`` are
    themselves non-empty (S2) and whose ``AxisDeclaration.label_ref`` is a
    required ``SourceRef`` (not ``Maybe``), no ``DatasetEnvelope`` can be
    constructed at all without at least one ``SourceRef`` reachable via
    :func:`iter_source_refs` -- grounding is now enforced STRUCTURALLY by
    that chain of required fields, so V0's own runtime check could never
    actually fire, and a negative test for it could only ever be rejected by
    one of these earlier structural requirements instead. Keeping a guard
    that can never independently fail is worse than deleting it: it invites
    the false confidence of a "passing" test that in fact pins some other
    validator's message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_graph: SourceGraph
    composition: Maybe[Composition]
    series: tuple[Series, ...] = Field(min_length=1)
    """Series aggregates (M-D2b part a) extracted from this source, each a
    self-contained set of axes/constants/points -- see :class:`Series`.

    REQUIRED, with no default and at least one entry: a dataset envelope that
    carries no series carries no dataset. Such an envelope would be the
    "audit-shaped artifact" this whole milestone exists to eliminate -- it
    has a validated source graph and possibly a composition, so it LOOKS like
    grounded extracted data, while containing nothing a model could ever be
    compared against.

    An earlier revision gave this field a ``()`` default, justified as
    backward compatibility and as avoiding a JSON round-trip failure in
    ``TestFunctionalRealisticEnvelope``. Both reasons were wrong. Nothing has
    ever been stored under this schema (the branch is unreleased), so there
    is no compatibility to preserve; and the round-trip failure was a signal
    that the FIXTURE was modelling an envelope that should not exist, not a
    reason to weaken the field. Reshaping the schema so an existing test
    keeps passing is precisely the "build to pass the check" antipattern --
    the fixture was fixed instead.
    """
    conversion_tables: tuple[EmbeddedConversionTable, ...]
    """Every :class:`~carmel.services.units.ConversionTable` cited (by
    ``conversion_table_sha256``) by any :class:`MeasuredValue` reachable in
    this envelope, embedded verbatim via its own canonical JSON -- see
    :class:`EmbeddedConversionTable`. Required so a stored dataset's bytes
    are self-contained: a non-Carmel consumer never needs
    :mod:`carmel.services.units` to learn what e.g. ``atm -> Pa`` means.

    Deliberately NOT constrained to "exactly one table" or "one table per
    series": a series citing both ``TABLE_V1`` and a future ``TABLE_V2`` is
    a PROVENANCE SMELL worth surfacing, not corruption -- forbidding it
    would block a legitimate re-extraction that spans a table revision.
    Detecting and scoring mixed-table use within one envelope/series is
    deliberately NOT enforced here; it is deferred to M-D3's trust
    computation. This field only enforces that the embedded set is EXACTLY
    the set actually cited (T2, below) and that it has one canonical order
    (T3, below).
    """

    @model_validator(mode="after")
    def _validate_conversion_tables_cover_cited_tables(self) -> DatasetEnvelope:
        """T2: ``conversion_tables`` must cover EXACTLY the set of
        ``conversion_table_sha256`` values cited by every :class:`MeasuredValue`
        reachable in this envelope (via :func:`iter_measured_values`) -- no
        fewer, no more.

        A table cited but not embedded (missing) would leave a
        ``MeasuredValue`` whose conversion a non-Carmel consumer cannot
        interpret at all -- exactly the payload this field exists to
        prevent. A table embedded but never cited by anything (decorative)
        is unearned provenance, the same failure class V2 closes for source
        graph nodes. These are two distinct, independently measured failure
        modes, so they get two distinct error messages -- a test must be
        able to tell which one fired, not merely that "something" was
        wrong.
        """
        cited = {value.conversion_table_sha256 for _, value in iter_measured_values(self)}
        embedded = {table.sha256 for table in self.conversion_tables}
        missing = cited - embedded
        if missing:
            raise ValueError(
                f"DatasetEnvelope.conversion_tables is missing table(s) {sorted(missing)!r} cited by a "
                "MeasuredValue -- every cited conversion table must be embedded"
            )
        decorative = embedded - cited
        if decorative:
            raise ValueError(
                f"DatasetEnvelope.conversion_tables embeds decorative table(s) {sorted(decorative)!r} that "
                "no MeasuredValue actually cites -- an embedded table nothing needs is unearned provenance"
            )
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_no_duplicate_sha256(self) -> DatasetEnvelope:
        """DUPLICATE-CONVERSION-TABLE-SHA256-GUARD: ``conversion_tables`` must
        not embed the same ``sha256`` more than once.

        T2 (above) compares SETS of embedded sha256s against the set of
        cited sha256s, so a tuple like ``(V1, V1)`` has exactly the same
        embedded set as ``(V1,)`` and passes T2 unchanged; T3's adjacent-pair
        sort check likewise accepts two equal, already-sorted entries. Left
        unchecked, two envelopes that differ only in whether a table is
        embedded once or twice have the SAME logical content but different
        bytes, and therefore different content addresses -- an
        address-uniqueness bug. This guard closes that gap directly, by
        sha256 identity rather than by set arithmetic.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        for table in self.conversion_tables:
            if table.sha256 in seen:
                duplicates.add(table.sha256)
            seen.add(table.sha256)
        if duplicates:
            raise ValueError(
                f"DatasetEnvelope.conversion_tables embeds duplicate sha256(s) {sorted(duplicates)!r} -- "
                "each cited conversion table must be embedded exactly once"
            )
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_sorted(self) -> DatasetEnvelope:
        """T3: ``conversion_tables`` must be sorted ascending by ``sha256``,
        so exactly one legal ordering exists -- matching the S2/S7/E1b idiom
        already in this module (a canonical order pins one, and only one,
        addressable representation)."""
        expected = tuple(sorted(self.conversion_tables, key=lambda table: table.sha256))
        if self.conversion_tables != expected:
            raise ValueError("DatasetEnvelope.conversion_tables must be sorted ascending by sha256")
        return self

    @model_validator(mode="after")
    def _validate_refs_resolve(self) -> DatasetEnvelope:
        """V1: every embedded :class:`SourceRef` must name a node this
        envelope's ``source_graph`` actually contains.

        Without this, a ``SourceRef`` is just a free-floating claim -- it
        looks like provenance (it has a ``node_id`` and a locator) but there
        is no guarantee the node it names was ever validated, or even
        exists. This is the check that makes a ``SourceRef`` actually mean
        something.
        """
        node_ids = self.source_graph.node_ids
        for path, ref in iter_source_refs(self):
            if ref.node_id not in node_ids:
                raise ValueError(
                    f"SourceRef at {path!r} names node_id={ref.node_id!r}, which is not present in "
                    f"source_graph (known node ids: {sorted(node_ids)!r})"
                )
        return self

    @model_validator(mode="after")
    def _validate_series_single_root_artifact(self) -> DatasetEnvelope:
        """V5: every :class:`SourceRef` within a single :class:`Series` must
        resolve to a node under the SAME parentless root artifact.

        Runs AFTER V1 (``_validate_refs_resolve``), so every ``ref.node_id``
        below is already known to resolve against ``self.source_graph`` --
        this validator only has to reason about which root each resolved
        node sits under, not whether it resolves at all.

        A series is meant to be one coherent measured quantity extracted
        from one place in the literature; a series whose refs span two
        different root artifacts (e.g. half its points grounded in the main
        PDF, the other half in an entirely different paper's PDF pulled into
        the same source graph) is not one dataset series, it is two series
        silently stitched into one -- exactly the kind of fabrication this
        module exists to make structurally impossible. This is checked at
        the ROOT level, not the node level: different nodes under the SAME
        root (e.g. a table cell in the main PDF and a caption label in an SI
        member of that same paper) are legitimate -- unit inconsistency
        WITHIN one paper is a real, expected shape (see the module
        docstring's second empirically-measured failure mode), not a
        fabrication signal, so refs to different nodes sharing one root must
        stay accepted.

        A node's root is itself (empty ``ancestors()``) if it is parentless,
        otherwise ``ancestors(node_id)[-1]`` -- ``ancestors()`` returns the
        chain from immediate parent up to root, so the ROOT is the LAST
        entry, not the first.
        """
        for series in self.series:
            roots: dict[str, str] = {}
            for _, ref in iter_source_refs(series):
                node_id = ref.node_id
                ancestor_chain = self.source_graph.ancestors(node_id)
                root_id = ancestor_chain[-1].node_id if ancestor_chain else node_id
                roots[root_id] = node_id
            if len(roots) > 1:
                raise ValueError(
                    f"Series(series_id={series.series_id!r}) spans multiple root artifacts: refs resolve "
                    f"to nodes under root artifacts {sorted(roots)!r} -- a series must be grounded under "
                    "a single root artifact"
                )
        return self

    @model_validator(mode="after")
    def _validate_no_decorative_nodes(self) -> DatasetEnvelope:
        """V2: every node in ``source_graph`` must be targeted by some
        ``SourceRef``, or be an ancestor of a node that is.

        An unreferenced node pads the graph with authoritative-looking
        provenance that nothing in the payload actually relies on -- an
        "audit-shaped" artifact rather than a real one: it LOOKS like the
        dataset is grounded against that artifact, but no extracted fact
        actually cites it. That is precisely the failure class this
        milestone exists to close, so a node that nothing (directly or via
        an ancestor of something) targets is rejected, not merely unused.
        Ancestors of a targeted node stay legal: an SI member's PAPER_PDF
        parent is real provenance context for that member even if nothing
        cites the parent directly.
        """
        referenced_ids = {ref.node_id for _, ref in iter_source_refs(self)}
        covered_ids: set[str] = set()
        for node_id in referenced_ids:
            covered_ids.add(node_id)
            covered_ids.update(ancestor.node_id for ancestor in self.source_graph.ancestors(node_id))
        for node in self.source_graph.nodes:
            if node.node_id not in covered_ids:
                raise ValueError(
                    f"node {node.node_id!r} is not targeted by any SourceRef, nor is it an ancestor of a "
                    "targeted node -- an unreferenced node is decorative provenance that nothing in this "
                    "envelope actually relies on"
                )
        return self

    @model_validator(mode="after")
    def _validate_locator_kind_compatibility(self) -> DatasetEnvelope:
        """V3: a :class:`SourceRef`'s locator kind must be compatible with
        the kind of node it targets, and (for a ``TableCellLocator``) its
        ``table_key`` kind must ALSO be compatible with that node kind.

        See :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS` for the
        locator/node-kind compatibility table and why it exists. Without
        this check, e.g. an ``XPathLocator`` could target a ``PAPER_PDF``
        node -- an XPath into a PDF is not a real locator, it is a claim
        about a document that was never parsed as XML at all.

        The ``table_key`` check is a second, narrower compatibility axis
        layered on top: ``LocatorKind.TABLE_CELL`` alone says the target can
        carry SOME table, but a ``MemberSheetKey`` names a workbook SHEET --
        a concept meaningless against anything but an ``SI_MEMBER`` -- while
        a ``CaptionLabelKey`` names a printed caption, meaningless against a
        node that carries no rendered caption at all. See
        :data:`_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS` for the compatibility
        table and its documented remaining gap (``SI_MEMBER`` too broad).
        """
        for path, ref in iter_source_refs(self):
            node = self.source_graph.node(ref.node_id)
            locator_kind = ref.locator.kind
            compatible_kinds = _LOCATOR_KIND_COMPATIBLE_NODE_KINDS[locator_kind]
            if node.kind not in compatible_kinds:
                raise ValueError(
                    f"SourceRef at {path!r} uses locator kind={locator_kind.value!r} against node "
                    f"{node.node_id!r} of kind={node.kind.value!r}, but {locator_kind.value!r} may only "
                    f"target nodes of kind {sorted(kind.value for kind in compatible_kinds)!r}"
                )
            if isinstance(ref.locator, TableCellLocator):
                table_key_kind = ref.locator.table_key.kind
                compatible_table_key_kinds = _TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS[table_key_kind]
                if node.kind not in compatible_table_key_kinds:
                    raise ValueError(
                        f"SourceRef at {path!r} uses table_key kind={table_key_kind.value!r} against node "
                        f"{node.node_id!r} of kind={node.kind.value!r}, but table_key kind="
                        f"{table_key_kind.value!r} may only target nodes of kind "
                        f"{sorted(kind.value for kind in compatible_table_key_kinds)!r}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_no_duplicate_series_id(self) -> DatasetEnvelope:
        """E1a: reject a repeated ``series_id`` among ``series``.

        A duplicate ``series_id`` would make "which series does this id
        refer to" ambiguous for any downstream consumer that indexes series
        by id -- the same failure mode S1/S6 close one level down, for axes
        and points within a single series.
        """
        seen: set[str] = set()
        for series in self.series:
            if series.series_id in seen:
                raise ValueError(f"DatasetEnvelope: duplicate series_id {series.series_id!r} in series")
            seen.add(series.series_id)
        return self

    @model_validator(mode="after")
    def _validate_series_sorted(self) -> DatasetEnvelope:
        """E1b: ``series`` must be sorted ascending by ``series_id``.

        Deliberately does NOT also assert non-emptiness the way S2/S7 do on
        :class:`Series` (``axes``/``points``): here that non-emptiness is
        already enforced structurally by ``Field(min_length=1)`` on
        :attr:`series` itself (see that field's own docstring for why an
        empty ``series`` tuple was made unconstructible), so this validator
        has exactly one job -- pin ascending ``series_id`` order -- and does
        not need to duplicate a check pydantic already runs before this
        validator ever executes.
        """
        expected = tuple(sorted(self.series, key=lambda series: series.series_id))
        if self.series != expected:
            raise ValueError("DatasetEnvelope: series must be sorted ascending by series_id")
        return self

    @model_validator(mode="after")
    def _validate_source_form_constrains_value_refs(self) -> DatasetEnvelope:
        """V4: a series' ``source_form`` constrains where its points' values
        may be grounded -- see :func:`_check_source_form_for_ref` for the
        per-branch rule and why every branch is constrained with no escape
        hatch.

        Only ``Coordinate.value.value_ref`` and a present
        ``Observation.value.value_ref`` are constrained -- NOT
        ``unit_ref``, ``label_ref``, or any composition ref: this validator
        is about where the NUMBER itself was read from, not about where its
        unit or axis label came from (a TABULAR series can still cite its
        unit spelling from running prose, for instance).

        Runs AFTER V1 (``_validate_refs_resolve``, declaration order), so
        every ``ref.node_id`` looked up here via ``self.source_graph.node``
        is already known to resolve.
        """
        for series in self.series:
            for point in series.points:
                for coord in point.coordinates:
                    ref = coord.value.value_ref
                    node = self.source_graph.node(ref.node_id)
                    _check_source_form_for_ref(
                        source_form=series.source_form,
                        ref=ref,
                        node_kind=node.kind,
                        where=(
                            f"Series(series_id={series.series_id!r}) point {point.point_id!r} "
                            f"coordinate axis_id={coord.axis_id!r}"
                        ),
                    )
                for obs in point.observations:
                    if isinstance(obs.value, Absent):
                        continue
                    ref = obs.value.value_ref
                    node = self.source_graph.node(ref.node_id)
                    _check_source_form_for_ref(
                        source_form=series.source_form,
                        ref=ref,
                        node_kind=node.kind,
                        where=(
                            f"Series(series_id={series.series_id!r}) point {point.point_id!r} "
                            f"observation axis_id={obs.axis_id!r}"
                        ),
                    )
        return self

    def identity_payload(self) -> dict[str, Any]:
        """Project this envelope to its canonical-JSON identity payload.

        Feeds :func:`carmel.services.dataset_store.canonical_json_bytes` and
        :func:`carmel.services.dataset_store.compute_dataset_sha` -- mirrors
        :meth:`carmel.services.units.ConversionTable.identity_payload`, the
        in-repo precedent for exactly this shape (a hand-written projection
        sitting next to the model it inverts).

        Deliberately NOT ``self.model_dump()`` / ``self.model_dump_json()`` /
        ``dict(self)``. ``compute_dataset_sha``'s own docstring refuses to
        accept or unwrap a pydantic model for one reason: an unrelated change
        to this schema's field shape, order, or defaults must never be able
        to silently re-address (rename the content-addressed identity of)
        every dataset already in the store. Routing this method through
        ``model_dump`` would reintroduce exactly that hazard one layer up --
        pydantic's default dump shape is an implementation detail of the
        model, not a stable on-the-wire contract, and it would change out
        from under callers the instant a field is renamed, reordered, or
        given a new default. Every nested model is instead projected by an
        explicit, hand-written ``_*_identity_payload`` helper (see the
        module-level helpers directly above this class) that names every
        field it emits, so a future field is either projected here on
        purpose or is a loud test failure via ``_UNADDRESSED_FIELDS`` --
        never a silent, unreviewed change to every stored dataset's address.

        Output is plain JSON-able data only: ``dict``/``list``/``str``/
        ``int``/``bool``/``None``, enums unwrapped to ``.value``, tuples
        projected as lists, and never a ``float`` at any depth (every
        numeric fact in this schema is already a canonical decimal
        ``str`` -- see the module docstring). A ``Maybe[T]`` field holding
        ``Absent`` projects via ``_absent_identity_payload`` to a
        ``{"__absent__": True, "reason": ..., "note": ...}`` dict that no
        present value's projection can ever collide with, so an envelope
        with ``composition=Absent(...)`` addresses differently from one
        with a real ``Composition``.

        Returns a freshly built dict on every call: nothing here is shared
        with ``self`` or with a previous call's return value, so mutating
        the result affects neither this envelope nor a subsequent call.
        """
        return {
            "source_graph": _source_graph_identity_payload(self.source_graph),
            "composition": _project_maybe(self.composition, _composition_identity_payload),
            "series": [_series_identity_payload(series) for series in self.series],
            "conversion_tables": [
                _embedded_conversion_table_identity_payload(table) for table in self.conversion_tables
            ],
        }
