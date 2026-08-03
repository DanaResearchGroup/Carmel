# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Schema primitives for literature-extracted experimental kinetics datasets.

This module builds the primitives listed for milestone M-D2a (explicit
absence states, coordinate frames/bboxes, measured values with per-field
unit binding, uncertainty, and composition) plus M-D2b part c: the source
graph (:class:`SourceGraph`) and the dataset envelope
(:class:`DatasetEnvelope`) that ties every embedded :class:`SourceRef` back
to a node the graph actually contains. It deliberately does NOT yet build a
dataset "series" aggregate or a registry -- those remain M-D2b part a and
part b respectively.

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

import math
import re
from collections.abc import Iterator
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from carmel.services import units
from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal
from carmel.services.numeric import (
    REPAIR_NAMES,
    GlyphHealth,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)
from carmel.services.units import QuantityKind

__all__ = [
    "AbsenceReason",
    "Absent",
    "ArchiveOrigin",
    "BBox",
    "BBoxLocator",
    "ComponentRole",
    "Composition",
    "CompositionBasis",
    "CompositionComponent",
    "CompositionResolution",
    "CoordinateFrame",
    "DatasetEnvelope",
    "Maybe",
    "MeasuredValue",
    "QuantityKind",
    "SourceGraph",
    "SourceLocator",
    "SourceNode",
    "SourceNodeKind",
    "SourceRef",
    "TableCellLocator",
    "Uncertainty",
    "UncertaintyBasis",
    "UncertaintyKind",
    "UncertaintyScale",
    "XPathLocator",
    "iter_source_refs",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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


class TableCellLocator(BaseModel):
    """Locates a reference at a specific table cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.TABLE_CELL] = LocatorKind.TABLE_CELL
    row: int = Field(ge=0)
    """0-indexed row; a negative row locates no real table cell."""
    col: int = Field(ge=0)
    """0-indexed column; a negative col locates no real table cell."""
    table_label: str | None = None


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

    @model_validator(mode="after")
    def _validate_repair_chain_agrees_with_raw_text(self) -> MeasuredValue:
        """Reject a ``repairs``/``canonical_decimal_value`` pair that disagrees
        with what ``raw_text`` itself actually needs.

        Replaces a plain string-equality check (``canonical_decimal(raw_text)
        == canonical_decimal_value``) that could never represent a repaired
        value -- see the class docstring for why that was wrong for this
        corpus. This validator checks the REPAIR CHAIN instead of string
        equality:

        1. ``raw_text`` must itself be derivable at all (rejects with the
           core's own reason if not).
        2. ``repairs`` must be an EXACT, ORDERED match for what
           :func:`~carmel.services.numeric.normalize_numeric_span` reports
           needing -- both under-claiming (a repair happened but was not
           recorded) and over-claiming (a recorded repair the text never
           needed) are rejected.
        3. ``canonical_decimal_value`` must be exactly
           ``canonical_decimal`` of the REPAIRED text -- never asserted
           independently.
        """
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


class DatasetEnvelope(BaseModel):
    """The top-level payload for one literature-extracted dataset: a source
    graph plus the extracted content that cites it.

    This is the model that makes fabrication structurally impossible at the
    WHOLE-PAYLOAD level, not just within a single :class:`MeasuredValue`:
    every :class:`SourceRef` embedded anywhere in this envelope (found via
    :func:`iter_source_refs`, which is shape-agnostic -- see its docstring)
    is checked against ``source_graph``, and the graph itself is checked for
    having nothing extra hanging off it. The four validators below run in a
    fixed order (V0 through V3), each addressing a distinct, independently
    measured failure mode; see each validator's own docstring for the
    concrete failure it closes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_graph: SourceGraph
    composition: Maybe[Composition]

    @model_validator(mode="after")
    def _validate_grounded(self) -> DatasetEnvelope:
        """V0: the envelope must contain at least one :class:`SourceRef`.

        An envelope that cites nothing is, by definition, ungrounded -- and
        making that impossible to construct is precisely why this project's
        schema layer exists at all (see the module docstring's "cardinal
        rule"). This currently makes ``composition=Absent(...)``
        unconstructible: today ``composition`` is the ONLY field that can
        carry a ``SourceRef``, so an envelope with no composition carries
        none anywhere, and this validator refuses it. That is a real,
        temporary limitation of this milestone (M-D2b part c), not a design
        claim that ``composition`` must always be present: once the series
        aggregate (M-D2b part a) adds its own ref-bearing payload alongside
        ``composition``, an envelope with ``composition=Absent(...)`` but a
        populated series becomes constructible again -- the ``Absent``
        branch is a real future state this validator will naturally admit
        once there is another field to ground the envelope through, not a
        dead branch kept around for cosmetic completeness.
        """
        if next(iter_source_refs(self), None) is None:
            raise ValueError(
                "DatasetEnvelope contains no SourceRef anywhere in its payload; an envelope that cites "
                "no evidence is ungrounded by definition and cannot be constructed -- see this "
                "validator's docstring for the composition=Absent(...) case specifically"
            )
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
        the kind of node it targets.

        See :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS` for the compatibility
        table and why it exists. Without this check, e.g. an
        ``XPathLocator`` could target a ``PAPER_PDF`` node -- an XPath into a
        PDF is not a real locator, it is a claim about a document that was
        never parsed as XML at all.
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
        return self
