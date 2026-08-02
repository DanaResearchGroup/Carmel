# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Schema primitives for literature-extracted experimental kinetics datasets.

This module builds ONLY the primitives listed for milestone M-D2a: explicit
absence states, coordinate frames/bboxes, the source graph, measured values
with per-field unit binding, uncertainty, and composition. It deliberately
does NOT build a dataset "series", the dataset envelope, or a registry --
those are M-D2b.

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
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal
from carmel.services.numeric import (
    REPAIR_NAMES,
    GlyphHealth,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)

__all__ = [
    "AbsenceReason",
    "Absent",
    "ArchiveMemberLocator",
    "BBox",
    "BBoxLocator",
    "ComponentRole",
    "Composition",
    "CompositionBasis",
    "CompositionComponent",
    "CompositionResolution",
    "CoordinateFrame",
    "Maybe",
    "MeasuredValue",
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
    (:class:`MeasuredValue`'s ``canonical_decimal``/``conversion_factor``,
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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    kind: SourceNodeKind
    sha256: str = Field(min_length=64, max_length=64)
    parent_node_id: str | None = None

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
    ARCHIVE_MEMBER = "archive_member"


class BBoxLocator(BaseModel):
    """Locates a reference at a bounding box on a rendered page."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[LocatorKind.BBOX] = LocatorKind.BBOX
    bbox: BBox


class TableCellLocator(BaseModel):
    """Locates a reference at a specific table cell."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[LocatorKind.TABLE_CELL] = LocatorKind.TABLE_CELL
    row: int = Field(ge=0)
    """0-indexed row; a negative row locates no real table cell."""
    col: int = Field(ge=0)
    """0-indexed column; a negative col locates no real table cell."""
    table_label: str | None = None


class XPathLocator(BaseModel):
    """Locates a reference in JATS/XML via an XPath expression."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[LocatorKind.XPATH] = LocatorKind.XPATH
    xpath: str = Field(min_length=1)


class ArchiveMemberLocator(BaseModel):
    """Locates a reference at a member of an archive (e.g. an SI zip).

    ``member_sha256`` is identity; ``display_path`` is display-only. This
    split is deliberate: member paths collide after normalization (e.g.
    ``"./a/b"`` vs ``"a/b"`` vs a path using backslashes) and, for archives
    extracted from untrusted PDFs' supplementary information, can be
    adversarially crafted (path traversal, homoglyphs). A locator that used
    the path as identity would let two different bytes silently look like
    the same reference, or vice versa.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[LocatorKind.ARCHIVE_MEMBER] = LocatorKind.ARCHIVE_MEMBER
    member_sha256: str = Field(min_length=64, max_length=64)
    display_path: str | None = None
    """Human-readable path for display ONLY -- never used for identity or
    equality. See class docstring."""

    @field_validator("member_sha256")
    @classmethod
    def _validate_member_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(f"invalid member_sha256: {value!r} (expected 64 lowercase hex characters)")
        return value


SourceLocator = Annotated[
    BBoxLocator | TableCellLocator | XPathLocator | ArchiveMemberLocator,
    Field(discriminator="kind"),
]


class SourceRef(BaseModel):
    """A reference INTO a dataset's source graph: which node, and where in it."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    locator: SourceLocator


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

    Unit conversion is never silent: ``unit_raw`` is preserved verbatim,
    ``unit_canonical`` is the normalized form, and ``conversion_factor``
    together with ``conversion_table_version`` record exactly what was
    applied and against which version of the conversion table, so the
    conversion is always reversible and auditable.

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
    """

    model_config = ConfigDict(extra="forbid")

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
    verbatim when a repair was actually needed."""
    repairs: tuple[str, ...] = ()
    """Names of the glyph repairs (each a member of
    :data:`carmel.services.numeric.REPAIR_NAMES` -- never free text, enforced
    below) applied to ``raw_text`` to derive ``canonical_decimal_value``. The
    empty tuple (the default) is the normal case: it means the source span
    was already clean and needed no repair. When non-empty, this is a claim
    about the evidence that must be EXACTLY true -- validated below against
    what :func:`~carmel.services.numeric.normalize_numeric_span` itself
    reports needing, in the same order."""
    unit_raw: str = Field(min_length=1)
    """The unit exactly as printed in the source."""
    unit_canonical: str = Field(min_length=1)
    """The normalized unit."""
    conversion_factor: str = Field(min_length=1)
    """Canonical decimal factor applied to convert ``unit_raw`` to
    ``unit_canonical``. ``"1"`` when no conversion was needed -- never
    omitted, so "no conversion happened" is always an explicit, auditable
    fact rather than an assumption."""
    conversion_table_version: str = Field(min_length=1)
    """Identifies which version of the unit-conversion table produced
    ``conversion_factor``, so a later change to that table can never
    silently reinterpret an already-stored value."""
    value_ref: SourceRef
    """Provenance for the NUMBER. Required -- see class docstring."""
    unit_ref: SourceRef
    """Provenance for the UNIT, independent of ``value_ref``. Required -- see
    class docstring."""

    @field_validator("conversion_factor")
    @classmethod
    def _validate_conversion_factor(cls, value: str) -> str:
        return _require_canonical_decimal(value, field_name="conversion_factor")

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
        return self


class UncertaintyKind(StrEnum):
    """Kind of uncertainty a reported value carries."""

    STD_DEV = "std_dev"
    CI_95 = "ci_95"
    INSTRUMENT_ERROR = "instrument_error"
    UNSPECIFIED_PERCENTAGE = "unspecified_percentage"
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

    model_config = ConfigDict(extra="forbid")

    kind: UncertaintyKind
    basis: Maybe[UncertaintyBasis]
    scale: Maybe[UncertaintyScale]
    upper: Maybe[MeasuredValue]
    """Upper bound magnitude, with its own unit and provenance. Absent if the
    source never states it (e.g. a symmetric-only report, or no bound at all)."""
    lower: Maybe[MeasuredValue]
    """Lower bound magnitude, with its own unit and provenance. Absent if the
    source never states it."""

    @property
    def blocks_statistical_interpretation(self) -> bool:
        """False iff this uncertainty is FULLY quantified; True otherwise.

        "Fully quantified" requires ALL FOUR of: ``kind`` is known (i.e. not
        :attr:`UncertaintyKind.UNKNOWN` -- ``kind`` has no ``Absent`` state
        of its own; ``UNKNOWN`` is its "not stated" sentinel), ``basis`` is
        known (not :class:`Absent`), ``scale`` is known (not :class:`Absent`),
        AND at least one of ``upper``/``lower`` is present (not
        :class:`Absent`). A known ``kind`` alone is NOT sufficient: a known kind with basis,
        scale, and both bounds all :class:`Absent` is "known kind, no usable
        magnitude", which is just as statistically useless as an unknown
        kind -- treating it as usable would let a paper's bare "+-5%, method
        unstated" be silently read as a fully quantified standard deviation,
        the exact failure this property exists to prevent. Deliberately NOT
        a quality signal -- see :attr:`UncertaintyKind.UNKNOWN`'s docstring.
        This flag tells a downstream consumer "do not compute a weighted
        statistic from this bound", it does not say "this extraction is
        worse than one with a stated kind"."""
        if self.kind == UncertaintyKind.UNKNOWN:
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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    raw_name: str = Field(min_length=1)
    """Verbatim mixture name/description as printed in the source (e.g.
    ``"air"``, ``"synthetic air"``, or a descriptive phrase)."""
    resolution: CompositionResolution
    basis: Maybe[CompositionBasis]
    equivalence_ratio: Maybe[MeasuredValue]
    components: list[CompositionComponent] = Field(default_factory=list)

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
