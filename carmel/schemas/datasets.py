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

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from carmel.services import units
from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal, canonical_json_bytes
from carmel.services.extraction_record import (
    _PYPDF_DEPENDENT_EXTRACTORS,
    _build_identity_payload,
    compute_extraction_sha,
)
from carmel.services.figure_digitization_record import (
    DIGITIZATION_PAYLOAD_KEYS,
    DIGITIZATION_PAYLOAD_VERSION,
    UNREADABLE_PAYLOAD,
    FigureCoverage,
    FigureDigitization,
    MarkerCensus,
)
from carmel.services.member_table_record import (
    MEMBER_INVENTORY_PAYLOAD_KEYS,
    MEMBER_INVENTORY_PAYLOAD_VERSION,
)
from carmel.services.numeric import (
    REPAIR_NAMES,
    GlyphHealth,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)
from carmel.services.pdf_table_record import (
    INVENTORY_PAYLOAD_KEYS,
    INVENTORY_PAYLOAD_VERSION,
    footprint_unreadable_reason,
    refusal_reasons_of,
)
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    GLYPH_HEALTH_DEPENDENCY_ID,
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
    "ConditionAttribution",
    "ConditionSetEnvelope",
    "Coordinate",
    "CoordinateFrame",
    "DataPoint",
    "DatasetEnvelope",
    "DatasetEnvelopeParseError",
    "DeviceClassDeclaration",
    "EmbeddedConversionTable",
    "EmbeddedFigureDigitization",
    "EmbeddedMemberTableInventory",
    "EmbeddedTableInventory",
    "ExtractedTextVerification",
    "ExtractionBinding",
    "GlyphHealthAssessment",
    "GroundedCategoricalClaim",
    "GroundedScalarClaim",
    "Maybe",
    "MeasuredValue",
    "MemberSheetKey",
    "Observation",
    "QuantityKind",
    "RawArtifactVerification",
    "RootSidecarVerification",
    "SemanticDependencyUse",
    "Series",
    "SiMemberDocumentKind",
    "SourceForm",
    "SourceGraph",
    "SourceLocator",
    "SourceNode",
    "SourceNodeKind",
    "SourceRef",
    "SourceVerification",
    "SubjectRefusalReason",
    "TableCellLocator",
    "TableKey",
    "TableKeyKind",
    "Uncertainty",
    "UncertaintyBasis",
    "UncertaintyKind",
    "UncertaintyScale",
    "UnextractedConditionStatement",
    "UnextractedReason",
    "UnitProvenance",
    "UnresolvedSubject",
    "ValueOrigin",
    "XPathLocator",
    "iter_measured_values",
    "iter_uncertainties",
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


class DatasetEnvelopeParseError(ValueError):
    """Raised by :meth:`DatasetEnvelope.from_identity_payload` when a payload
    cannot be reconstructed into a :class:`DatasetEnvelope` that reproduces
    it exactly.

    Covers three distinct failure modes, all reported through this one type
    so callers can catch it and distinguish it from an unrelated
    ``ValueError``:

    * a malformed ``__absent__`` marker (wrong shape, ``__absent__`` not
      ``True``, unknown ``reason``, non-``str``/``None`` ``note``) found
      while rehydrating the payload, before validation is even attempted;
    * an ordinary pydantic ``ValidationError`` raised by
      ``DatasetEnvelope.model_validate`` on the rehydrated payload (e.g. a
      cross-field invariant such as sorted-``series`` is violated);
    * the parsed envelope validates, but re-projecting it via
      :meth:`DatasetEnvelope.identity_payload` does not reproduce the input
      payload byte-for-byte -- a parser/projector disagreement, not a claim
      that the input itself is corrupt.
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


class SiMemberDocumentKind(StrEnum):
    """The kind of document a supplementary-information member IS, once that
    has been determined -- the fact that lets the schema tell a ``.docx`` from
    a ``.pdf`` from an ``.xlsx`` and DECIDE where it used to REFUSE.

    :class:`SourceNodeKind` deliberately carries a single ``SI_MEMBER`` value
    covering a csv, an xlsx, a PDF, and a Word document all alike (see that
    enum, and the "SI_MEMBER too broad" gap documented on
    :data:`_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS`): the node kind says an
    artifact came out of an archive, never what medium it is. That single
    kind is why two validators had to REFUSE a caption-labelled SI-member
    cell both ways -- a PDF cell has fragment geometry an inventory can
    describe, a word-processor cell does not, and the node kind could not
    tell them apart. This enum is that missing distinction, declared as a
    first-class stored fact rather than sniffed from a filename suffix or the
    bytes.

    Exactly three members, and the split is load-bearing for BOTH validators
    to agree (V3 and V8): a ``PDF`` and a ``WORD_PROCESSOR`` document are
    RENDERED, carry PRINTED CAPTIONS, and have no workbook sheets; a
    ``SPREADSHEET`` has workbook SHEETS and no printed captions. The two
    rendered kinds are kept distinct rather than merged because a PDF cell
    CAN have fragment geometry a cell inventory describes while a
    word-processor cell cannot -- the very distinction V8 turns on.
    """

    PDF = "pdf"
    """A rendered PDF supplementary member. Its cells CAN carry PDF fragment
    geometry, so a cell inventory over it is meaningful in principle -- though
    verifying one is follow-on work (see
    :func:`_validate_table_cell_inventory_citation`)."""

    WORD_PROCESSOR = "word_processor"
    """A word-processor document (e.g. ``.docx``). Rendered and caption-bearing
    like a PDF, but its cells have NO PDF fragment geometry a cell inventory
    could describe -- the case V8 now ACCEPTS with an absent citation."""

    SPREADSHEET = "spreadsheet"
    """A spreadsheet (e.g. ``.xlsx``/``.csv``). Carries workbook SHEETS and no
    printed captions, so a ``MemberSheetKey`` addresses it and a
    ``CaptionLabelKey`` does not (V3)."""


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
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through.
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid archive_sha256: expected 64 lowercase hex chars, got {value!r}")
        return value


class ExtractionBinding(BaseModel):
    """Binds a :class:`SourceNode`'s raw bytes to the extracted text derived
    from them, closing a gap where the two were connected only by
    convention.

    ``extracted_sha256`` is the integrity anchor: the digest of the stored
    ``extracted.json`` FILE BYTES, computed by
    :func:`carmel.services.evidence` AFTER writing that file to disk (never
    re-serialized from the in-memory model), so it describes exactly the
    bytes a replayer would read back. ``extracted_text_sha256`` is the
    digest of ``extracted.text`` (UTF-8 encoded) as recorded in that same
    verified ``extracted.json`` -- deliberately NOT a digest of the
    evidence store's ``text.txt``, because ``text.txt`` is only ever
    presence-checked on disk (``text_path.exists()`` in
    ``evidence._artifact_intact``), never digest-checked, so it cannot serve
    as an integrity anchor. A replayer wanting to confirm
    ``extracted_text_sha256`` must recompute it from a VERIFIED
    ``extracted.json`` (one whose bytes hash to ``extracted_sha256``), not
    from ``text.txt``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_raw_sha256: str = Field(min_length=64, max_length=64)
    """sha256 of the raw artifact this extraction was derived from -- must equal
    the owning :class:`SourceNode`'s own ``sha256`` (enforced by
    ``SourceNode._validate_extraction_parent_matches_node_sha256``). Named to
    match :class:`carmel.services.extraction_record.ExtractionRecordMeta`'s
    field of the same name: this is half of the resolvable extraction address
    that field pairs with ``extraction_sha256`` below."""
    extraction_sha256: str = Field(min_length=64, max_length=64)
    """The content address of the specific extraction record this binding
    describes, i.e. the ``extraction_sha256`` computed by
    :func:`carmel.services.extraction_record.compute_extraction_sha` and
    returned by :func:`carmel.services.extraction_record.store_extraction_record`.

    The evidence store now allows MANY extraction records per raw-bytes
    directory (``evidence/literature/<raw sha256>/extractions/<extraction
    sha256>/``, see :mod:`carmel.services.extraction_record`) -- a ``pypdf``
    upgrade, or a change to Carmel's own extraction code, produces a NEW
    extraction record alongside any earlier ones rather than replacing them.
    This field is what makes a binding resolvable to exactly ONE of those
    records rather than assuming there is only ever one to find."""
    extracted_sha256: str = Field(min_length=64, max_length=64)
    """Digest of the stored ``extracted.json`` file's bytes -- the integrity anchor."""
    extracted_text_sha256: str = Field(min_length=64, max_length=64)
    """Digest of ``extracted.text.encode("utf-8")`` as recorded in that ``extracted.json``."""
    extractor: str = Field(min_length=1)
    """The extractor string the addressed record claims produced its text
    (e.g. ``"pdf:pypdf"``, ``"html"``, ``"text"`` -- see
    :class:`carmel.agents.tools.extract.ExtractedText` for the authoritative
    vocabulary). Identity-bearing: it is folded into the extraction address,
    and it decides whether ``pypdf_version`` below is part of that address at
    all."""
    extractor_code_sha256: str = Field(min_length=64, max_length=64)
    """Carmel's own extraction/normalization code identity at the time the
    addressed record was stored -- see
    :func:`carmel.services.semantic_deps.current_sha_for` for
    ``EXTRACT_TEXT_DEPENDENCY_ID``. Identity-bearing (folded into the
    extraction address)."""
    identity_payload_version: str = Field(min_length=1)
    """The identity-payload SHAPE version the addressed record's address was
    computed under -- see ``_IDENTITY_PAYLOAD_VERSION`` in
    :mod:`carmel.services.extraction_record`. Carried so the address stays
    recomputable from this binding alone even after the shape version moves
    on."""
    pypdf_version: Maybe[str]
    """The installed ``pypdf`` version at extraction time, iff ``extractor``
    is one of the ``pypdf``-dependent extractors named by
    ``_PYPDF_DEPENDENT_EXTRACTORS`` in
    :mod:`carmel.services.extraction_record` -- REQUIRED for those extractors
    (their identity genuinely depends on it) and FORBIDDEN for every other
    (their address provably does not fold it in, so a binding claiming one
    would assert an identity fact its own address does not carry; see the
    model validator below, which mirrors that module's rule rather than
    inventing a new one).

    DELIBERATELY HAS NO DEFAULT, like every other ``Maybe[...]`` field in
    this module: "the producer forgot to say" and "the concept does not apply
    to this extractor" must stay distinguishable, so callers state the
    absence explicitly (reason ``NOT_APPLICABLE`` -- the only honest reason,
    enforced below) or they do not build the binding.
    """

    @field_validator(
        "parent_raw_sha256",
        "extraction_sha256",
        "extracted_sha256",
        "extracted_text_sha256",
        "extractor_code_sha256",
    )
    @classmethod
    def _validate_sha256_shape(cls, value: str) -> str:
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through. Guards all
        # five sha256 fields above (this validator body runs once per field).
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid sha256: {value!r} (expected 64 lowercase hex characters)")
        return value

    @field_validator("pypdf_version")
    @classmethod
    def _validate_pypdf_version_shape(cls, value: Maybe[str]) -> Maybe[str]:
        if isinstance(value, Absent):
            # NOT_APPLICABLE is the only honest reason: an absent
            # pypdf_version on a valid binding means exactly "this
            # extractor's identity does not involve pypdf" (the model
            # validator below rejects absence outright for the
            # pypdf-dependent extractors). UNKNOWN would claim the version
            # existed but was lost -- a binding like that could never
            # recompute its own address and must not exist at all.
            if value.reason is not AbsenceReason.NOT_APPLICABLE:
                raise ValueError(
                    f"pypdf_version may only be Absent for reason "
                    f"{AbsenceReason.NOT_APPLICABLE.value!r}, not {value.reason.value!r}: absence "
                    "here means the extractor's identity does not involve pypdf at all, never that "
                    "a version existed but went unrecorded"
                )
            return value
        if not value:
            raise ValueError("pypdf_version, when present, must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate_extraction_sha256_recomputes_from_identity_fields(self) -> ExtractionBinding:
        """This binding is INTERNALLY COHERENT, not self-authenticating: it
        carries every field that
        :func:`carmel.services.extraction_record.compute_extraction_sha`
        folds into an extraction record's content address, and this
        validator recomputes that address (via the same
        ``_build_identity_payload``/``compute_extraction_sha`` pair the
        store itself uses -- never a second implementation that could
        drift) and requires it to equal ``extraction_sha256``. A binding
        whose ``extraction_sha256`` is inconsistent with its own identity
        fields -- one field edited, the address left stale -- never comes
        into existence.

        What this does NOT prove: this recomputes the address from the
        binding's OWN fields, so it cannot detect a CONSISTENTLY forged
        binding -- one whose identity fields were edited and whose address
        was recomputed to match. It buys internal coherence, not
        authenticity. The actual authenticity check happens at replay time,
        when the binding's identity fields are compared against the
        extraction record's own ``meta.json`` as loaded from the store (see
        ``carmel/services/dataset_replay.py``); that comparison is what
        makes forgery detectable, because the store's record is not under
        the envelope author's control.
        """
        if self.extractor in _PYPDF_DEPENDENT_EXTRACTORS:
            if isinstance(self.pypdf_version, Absent):
                raise ValueError(
                    f"pypdf_version must be present for extractor {self.extractor!r}: that "
                    "extractor's identity depends on the installed pypdf version, so an address "
                    "computed without one is not recomputable and the binding cannot authenticate "
                    "itself"
                )
            pypdf_version = self.pypdf_version
        else:
            if not isinstance(self.pypdf_version, Absent):
                raise ValueError(
                    f"pypdf_version must be Absent (reason "
                    f"{AbsenceReason.NOT_APPLICABLE.value!r}) for extractor {self.extractor!r}: "
                    "that extractor's address does not fold a pypdf version in (see "
                    "_PYPDF_DEPENDENT_EXTRACTORS in carmel.services.extraction_record), so a "
                    "concrete value here would be an identity claim the address provably does not "
                    "carry"
                )
            # _build_identity_payload drops pypdf_version from the payload
            # entirely for a non-pypdf-dependent extractor, so this
            # placeholder provably never reaches the hashed payload -- it
            # only satisfies the helper's signature.
            pypdf_version = "not-applicable"
        try:
            recomputed = compute_extraction_sha(
                _build_identity_payload(
                    identity_payload_version=self.identity_payload_version,
                    raw_sha256=self.parent_raw_sha256,
                    extractor=self.extractor,
                    extractor_code_sha256=self.extractor_code_sha256,
                    pypdf_version=pypdf_version,
                    extracted_sha256=self.extracted_sha256,
                    extracted_text_sha256=self.extracted_text_sha256,
                )
            )
        except ValueError as exc:
            # Includes UnknownPypdfVersionError: a pdf:pypdf binding whose
            # pypdf_version is the "could not determine" sentinel can never
            # authenticate and is refused an existence, mirroring
            # store_extraction_record's own refusal to mint such an address.
            raise ValueError(
                f"ExtractionBinding's identity fields do not form an addressable extraction identity payload: {exc}"
            ) from exc
        if recomputed != self.extraction_sha256:
            raise ValueError(
                f"extraction_sha256={self.extraction_sha256!r} does not recompute from this "
                f"binding's own identity fields (recomputed {recomputed!r}); a binding that cannot "
                "authenticate its own address must never exist -- either the address or the "
                "identity fields are wrong, and there is no way to say which"
            )
        return self


class GlyphHealthAssessment(BaseModel):
    """A stored record of a document's glyph-health assessment (see
    :class:`~carmel.services.numeric.GlyphHealth`), attributed to the
    dependency version that produced it.

    HONEST SCOPE: validating a stored ``GlyphHealthAssessment`` can check
    registry identity (``assessor.dependency_id`` genuinely names the glyph
    -health dependency), digest shape, and -- once embedded in a
    :class:`SourceNode` -- the node's extraction binding. It CANNOT re-run
    :func:`~carmel.services.numeric.assess_glyph_health`, because its input
    (the whole extracted document text) is out-of-payload here. A stored
    assessment is therefore UNVERIFIED-BY-CONSTRUCTION until the replayer
    milestone lands; never describe one as "verified" anywhere.

    All five booleans on ``health`` are recorded even though only
    ``suspects_dash_corruption`` is consumed by parsing logic today
    (:mod:`carmel.services.numeric`). That is acceptable ONLY because
    ``assessor`` identity travels with them: if a later heuristic version
    disagrees with an earlier one about, say, ``has_thorn_plus_marker``, the
    disagreement reads as "a different heuristic version produced this," not
    as an unexplained bug.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    health: GlyphHealth
    assessor: SemanticDependencyUse

    @model_validator(mode="after")
    def _validate_assessor_names_the_glyph_health_dependency(self) -> GlyphHealthAssessment:
        if self.assessor.dependency_id != GLYPH_HEALTH_DEPENDENCY_ID:
            raise ValueError(
                f"assessor.dependency_id={self.assessor.dependency_id!r} is not the glyph-health "
                f"dependency this assessment claims to record ({GLYPH_HEALTH_DEPENDENCY_ID!r}); a "
                "GlyphHealthAssessment whose assessor does not name that dependency is a forgery "
                "attempt (mis-stamping a health record with the wrong heuristic's identity), not a "
                "harmless inconsistency, and is rejected as such"
            )
        return self


class RawArtifactVerification(StrEnum):
    """What was checked about a node's ``raw.bin`` when its envelope was produced.

    One member only, deliberately. The producer refuses outright when the raw
    bytes do not re-hash to the node's ``sha256``, so no envelope can exist
    carrying any weaker claim -- and this module DELETES unreachable guards
    rather than shipping them (see :class:`DatasetEnvelope`'s note on the
    removed V0 validator). An unreachable enum member is the data equivalent.
    The enum exists rather than a bare ``bool`` so that a future producer path
    which checks something different must add a member and say so, instead of
    silently reusing a field that already reads as "the bytes were
    authenticated".
    """

    RAW_SHA256_DIGEST_AUTHENTICATED = "raw_sha256_digest_authenticated"
    """``raw.bin`` was re-read from the store and re-hashed, and its digest
    equalled the node's own ``sha256``."""


class ExtractedTextVerification(StrEnum):
    """Which tier of the evidence store the grounded text was authenticated against.

    Read the member names EXACTLY: every one of them is a claim about DIGEST
    AUTHENTICATION, never about derivation. Nothing in this codebase proves
    that any stored extracted text was genuinely re-derived from ``raw.bin`` --
    not the extraction record, not the replayer, not this field. An extraction
    record is bytes plus a self-consistent identity payload, and a caller that
    can write to the store can mint one from text of its choosing. What these
    members do assert is that the bytes grounding this envelope hashed to the
    digest recorded for them, at the tier named.
    """

    EXTRACTION_RECORD_DIGEST_AUTHENTICATED = "extraction_record_digest_authenticated"
    """The grounded text came from an extraction record under
    ``extractions/<extraction_sha256>/`` whose stored bytes hashed to the
    digest its own address folds in. This is the only tier the producer will
    ground against: the root sidecar is never used as grounding input, because
    text read from it is exactly what the corpus gate refuses without an
    explicit operator opt-in."""


class RootSidecarVerification(StrEnum):
    """What, if anything, was established about the ROOT ``extracted.json``
    sidecar -- the legacy, unauthenticated tier that predates the extraction
    -record store.

    This is the one tier whose answer genuinely varies with the artifact, and
    it is mostly a NEGATIVE claim: its job is to stop a reader inferring that
    a record-grounded envelope also carries root-level verification. It never
    describes the text this envelope grounds against (that is
    :class:`ExtractedTextVerification`); the root sidecar is not an input to
    production at all.
    """

    ROOT_SIDECAR_DIGEST_AUTHENTICATED = "root_sidecar_digest_authenticated"
    """The root ``meta.json`` records an ``extracted_sha256``, and the sidecar's
    bytes on disk hashed to it at production time.

    This says nothing about the text this envelope grounds against -- that came
    from the extraction record either way. It is recorded because the
    alternative was a ``NOT_CHECKED`` member, and an unfalsifiable claim has no
    business in persisted evidence: ``NOT_CHECKED`` would have described a
    producer CHOICE rather than a fact about the store, so no consumer could
    ever contradict it, and a claim nobody can refute is indistinguishable from
    a claim nobody made. Checking costs one hash and buys a value replay can
    put to the test."""

    ROOT_SIDECAR_DIGEST_MISMATCH = "root_sidecar_digest_mismatch"
    """The root ``meta.json`` records an ``extracted_sha256`` and the sidecar's
    bytes did NOT hash to it: the legacy tier of this artifact is damaged.

    Recorded, deliberately, rather than refused. The root sidecar is not an
    input to production -- refusing over it would re-erect exactly the gate this
    design removed, and would block a dataset whose raw bytes and grounded text
    are both fully authenticated. Surfacing the damage in the envelope is the
    honest handling; deciding what to do about it belongs to whoever reads the
    envelope, not to the producer.

    DO NOT READ THIS AS BENIGN VERSION DRIFT. It is tempting to explain a
    mismatch away as "a different pypdf wrote that sidecar", and Codex round 73
    was right to push back: root sidecars are never rewritten, so a mismatch
    means the recorded digest does not match the bytes sitting there NOW --
    damage or tampering at the root tier, not a vintage difference. It is
    recorded rather than raised only because that tier is not the one this
    envelope's evidence rests on. Any downstream policy that maps a whole
    :class:`SourceVerification` onto a single "verified" boolean will get this
    wrong; the tier has to be read on its own."""

    NO_RECORDED_DIGEST = "no_recorded_digest"
    """The artifact's root ``meta.json`` records ``extracted_sha256=None``: it
    was stored before that field existed, so its sidecar carries no digest and
    CANNOT be authenticated by anyone, now or later.

    Distinct from ``ROOT_SIDECAR_DIGEST_MISMATCH``, which reports a check that
    RAN and disagreed. Inability-to-check and demonstrated-disagreement are
    different facts, and this codebase has already had to unwind that
    conflation three times on the acquisition side (PAYWALLED vs
    NO_OPEN_ACCESS_COPY, then NO_OPEN_ACCESS_COPY vs OA_LOOKUP_INCOMPLETE)."""


class SourceVerification(BaseModel):
    """What was ACTUALLY verified about one source node, tier by tier.

    Three orthogonal claims, because "verified" without a tier is the field
    that quietly changes meaning between envelope vintages. A dataset produced
    against a legacy artifact -- raw bytes authenticated, text authenticated
    against a genuinely stored extraction record, root sidecar never
    authenticable at all -- is not the same evidence as one produced against
    an artifact verified at every tier, and an envelope that says only
    "verified" cannot tell the two apart.

    NONE of this is self-asserted trust. Every claim here is independently
    FALSIFIABLE by :mod:`carmel.services.dataset_replay` against the store, and
    the replayer does exactly that: a claim that disagrees with what the store
    can support is positive evidence the envelope was altered outside validated
    construction, and is reported FAILED. A claim no consumer can refute would
    be decoration, and this codebase has already carried provenance nobody read
    for long enough to find the same stale "no consumer reads this yet" comment
    in four separate places.

    FALSIFIABLE IS NOT THE SAME AS ALWAYS CHECKED, and this docstring used to
    blur the two (Codex round 73). ``raw_artifact`` and ``extracted_text`` are
    re-derived from bytes on every replay, so those are always checked. The
    ``root_sidecar`` claim is checked only when the root tier can actually be
    read: replay never treats an unreadable root ``meta.json`` as a failure,
    because its verification of the DATA is root-independent by design. What it
    no longer does is stay SILENT about it -- a claim it could not check is
    reported on the replay report as a
    :class:`~carmel.services.dataset_replay.UncheckedStoreClaim`, orthogonal to
    ``evidence_outcome`` and folded into ``overall_outcome``. A reader who wants
    "every carried claim held" reads ``overall_outcome``, which is exactly that
    question; it no longer has to be paired with a side list, because an
    uncheckable claim is precisely what keeps that verdict off VERIFIED.

    Note what is NOT here: any claim that the extracted text was derived from
    the raw bytes. See :class:`ExtractedTextVerification` for why no component
    in this system can honestly make that claim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_artifact: RawArtifactVerification
    extracted_text: ExtractedTextVerification
    root_sidecar: RootSidecarVerification


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
    extraction: Maybe[ExtractionBinding]
    """Binds this node's raw bytes to the extracted text derived from them,
    if any has been recorded. ``Maybe``-typed, with no default, for the same
    "no unreasoned absence" reasoning as ``origin`` above. Deliberately NOT
    inferred from ``kind``: a ``SI_MEMBER`` is a mixed bucket (some members
    are spreadsheets with no flat text layer at all; others are text-bearing
    documents), and even ``PAPER_PDF`` does not itself assert that a flat
    text layer has been extracted -- so ``kind`` alone cannot distinguish
    "extraction does not apply to this node" (``NOT_APPLICABLE``) from
    "extraction has not happened yet" (``NOT_EXTRACTED_YET``); only an
    explicit ``Absent(reason=...)`` can say which."""
    glyph_health: Maybe[GlyphHealthAssessment]
    """This node's stored glyph-health assessment, if any has been recorded.
    ``Maybe``-typed, with no default, for the same reason as ``extraction``
    above -- and for the same reason, NOT inferred from ``kind`` either."""
    verification: Maybe[SourceVerification]
    """Tier-by-tier record of what was ACTUALLY verified about this node when
    the envelope was produced -- see :class:`SourceVerification`.

    ``Maybe``-typed, with no default, for the same "no unreasoned absence"
    reasoning as the three fields above, and bound to ``extraction`` by the
    validator below: present exactly when this node carries an extraction,
    absent otherwise. A node with no extracted text has no verification story
    this system can honestly state, and inventing one would be exactly the
    unearned provenance the surrounding validators exist to prevent."""
    crop_region: Maybe[BBox]
    """WHERE, on the page it was cut from, a ``FIGURE_CROP`` sits -- the
    crop's own addressing identity. Required for a ``FIGURE_CROP``, forbidden
    for every other kind (I7, below).

    A ``FIGURE_CROP`` previously had no way to say WHICH figure it was.
    ``node_id`` is a label a producer picks, not an address; a crop carries no
    extracted text by construction (I6); so the only field left to tell two
    crops apart was ``sha256`` -- which a producer holding no crop bytes of
    its own would borrow from the parent PDF. Two crops of two DIFFERENT
    figures in one paper were then identical in every field
    :class:`SourceGraph`'s I5 duplicate rule inspects, and the second was
    refused as an exact repeat of the first: the graph could not represent
    two figures from one paper at all. This field is the identity that tells
    them apart, and I8 forbids the borrowed ``sha256`` that made them
    collide. I5's ``(kind, sha256, parent_node_id)`` triple is deliberately
    UNTOUCHED by all of this -- the fix is that a crop now has an identity of
    its own, not that duplicate detection was loosened to admit two crops
    that don't.

    A whole :class:`BBox` rather than a page number plus four coordinates,
    because a ``BBox`` carries the :class:`CoordinateFrame` that makes
    coordinates mean anything at all: ``cropbox``/``mediabox``, ``rotation``,
    the render ``dpi`` and ``render_settings``, and -- as the page's identity
    -- ``render_fingerprint``. A page NUMBER is deliberately NOT recorded
    here, and must not be added: :class:`CoordinateFrame`'s docstring rules
    one out as a provenance key outright, because this corpus's own
    extractors silently drop pages, so a number produced by one tool does not
    reliably address the same physical page as that number from another.

    THE COORDINATES ARE IN THE PARENT'S FRAME, never the crop's own. A
    :class:`BBoxLocator` targeting a ``FIGURE_CROP`` node addresses a region
    WITHIN that crop's render (see
    :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS`); this field addresses the
    crop itself within the page it was cut from. Two different rectangles in
    two different frames -- never interchangeable.

    ``Maybe``-typed with no default, for the same "no unreasoned absence"
    reasoning as ``origin``/``extraction``/``glyph_health`` above. WHICH
    absences are legal is decided by I7, not by this field."""
    document_kind: Maybe[SiMemberDocumentKind]
    """WHICH kind of document a ``SI_MEMBER`` node IS -- a PDF, a
    word-processor document, or a spreadsheet (see
    :class:`SiMemberDocumentKind`). Meaningful ONLY for an ``SI_MEMBER``;
    forbidden (``Absent(NOT_APPLICABLE)``) for every other kind, exactly as
    ``origin`` is (see :meth:`_validate_document_kind`, below).

    A DECLARED value is a first-class stored fact, projected into the node's
    identity: it is what lets :func:`_validate_table_cell_inventory_citation`
    (V8) and :func:`_validate_locator_kind_compatibility` (V3) DECIDE a
    caption-labelled SI-member cell where a single ``SI_MEMBER`` kind forced
    them to REFUSE. It is DELIBERATELY never inferred from a filename suffix
    or the member's bytes -- both are author-controlled and the whole point
    is a fact the schema can stand behind, not a guess it sniffed.

    ``Maybe``-typed with no default, for the same "no unreasoned absence"
    reasoning as ``origin``/``extraction`` above, and -- like ``extraction``
    -- NOT inferable from ``kind``: an SI member HAS a document kind whether
    or not we have determined it, so an undeclared member is an explicit
    ``Absent(reason=NOT_EXTRACTED_YET)`` (the honest "the document has a kind,
    we have not determined which"), never a silent default and never
    ``NOT_APPLICABLE`` (which would falsely claim the concept does not apply
    to a document). WHICH absences are legal is decided by
    :meth:`_validate_document_kind`, not by this field."""

    @model_validator(mode="after")
    def _validate_verification_binds_to_extraction(self) -> SourceNode:
        """``verification`` must be present exactly when ``extraction`` is.

        The two directions fail differently, so they are reported differently.
        A node carrying a :class:`SourceVerification` but no extraction is
        claiming an ``extracted_text`` tier for text it does not have -- the
        same unearned-provenance failure as a ``glyph_health`` assessment with
        nothing to attribute it to. A node carrying an extraction but no
        verification is the more dangerous direction: it is the envelope that
        grounds against real text while declining to say what was checked
        about it, which is precisely the "verified means whatever the reader
        assumes" ambiguity this field was added to remove.
        """
        has_extraction = not isinstance(self.extraction, Absent)
        has_verification = not isinstance(self.verification, Absent)
        if has_verification and not has_extraction:
            raise ValueError(
                f"node {self.node_id!r} carries a SourceVerification but no extraction binding "
                "(extraction is Absent); a verification record states what was checked about this "
                "node's extracted text, and there is no extracted text here to have checked"
            )
        if has_extraction and not has_verification:
            raise ValueError(
                f"node {self.node_id!r} carries an extraction binding but no SourceVerification "
                "(verification is Absent); an envelope that grounds against extracted text must state "
                "tier by tier what was actually verified about it, or 'verified' means only whatever "
                "the reader assumes"
            )
        return self

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

    @model_validator(mode="after")
    def _validate_document_kind(self) -> SourceNode:
        """``document_kind`` is meaningful ONLY for an ``SI_MEMBER``, and its
        absence reason is PINNED so the conditional projection's inverse can
        restore the exact marker from ``kind`` alone.

        This is stricter than :meth:`_validate_origin_only_for_si_member`
        above, which only forbids a non-``SI_MEMBER`` from carrying a concrete
        value and leaves the absence reason free. Here the reason is
        load-bearing: :data:`_CONDITIONALLY_PROJECTED_FIELDS` omits
        ``document_kind`` whenever it is ``Absent`` and
        :func:`_restore_unprojected_document_kinds` reconstructs the marker
        from the payload's ``kind`` -- so an unpinned reason would not
        round-trip. Exactly as I7 pins ``crop_region``'s reason to
        ``NOT_APPLICABLE``, this pins ``document_kind``'s two legal absences:

        - ``kind != SI_MEMBER``: the concept does not apply (a paper PDF, a
          JATS document, a figure crop are not supplementary members with a
          media type to declare), so ``document_kind`` MUST be
          ``Absent(reason=NOT_APPLICABLE)``. A concrete value, or an ``Absent``
          with any other reason, is refused.
        - ``kind == SI_MEMBER``: the member IS a document with a kind, so the
          value is EITHER a concrete :class:`SiMemberDocumentKind` (declared)
          OR ``Absent(reason=NOT_EXTRACTED_YET)`` (undeclared: the document has
          a kind, we have not determined which). ``NOT_APPLICABLE`` is refused
          because it would falsely claim the concept does not apply to a
          document, and every other absence reason is refused for the same
          "unreasoned absence" reason the field's ``Maybe`` typing exists to
          prevent.
        """
        if self.kind != SourceNodeKind.SI_MEMBER:
            if not isinstance(self.document_kind, Absent):
                raise ValueError(
                    f"node {self.node_id!r} has kind={self.kind.value!r}, which cannot carry a "
                    "concrete SiMemberDocumentKind -- only an SI_MEMBER node IS a supplementary "
                    "document with a media type to declare; document_kind must be "
                    f"Absent(reason={AbsenceReason.NOT_APPLICABLE.value!r}) here, not "
                    f"{self.document_kind!r}"
                )
            if self.document_kind.reason != AbsenceReason.NOT_APPLICABLE:
                raise ValueError(
                    f"node {self.node_id!r} has kind={self.kind.value!r}, for which document_kind "
                    f"does not apply, so it must be Absent(reason="
                    f"{AbsenceReason.NOT_APPLICABLE.value!r}) -- reason "
                    f"{self.document_kind.reason.value!r} would claim some other absence for a "
                    "concept that simply does not apply to this kind"
                )
            return self
        if isinstance(self.document_kind, Absent) and self.document_kind.reason != AbsenceReason.NOT_EXTRACTED_YET:
            raise ValueError(
                f"node {self.node_id!r} is an SI_MEMBER, which IS a document with a kind, so an "
                "undeclared document_kind must be Absent(reason="
                f"{AbsenceReason.NOT_EXTRACTED_YET.value!r}) -- reason "
                f"{self.document_kind.reason.value!r} is wrong: {AbsenceReason.NOT_APPLICABLE.value!r} "
                "would falsely claim the concept does not apply to a document, and any other reason "
                "invents an absence the schema does not stand behind; declare a concrete "
                "SiMemberDocumentKind or leave it NOT_EXTRACTED_YET"
            )
        return self

    @model_validator(mode="after")
    def _validate_glyph_health_binds_to_this_nodes_extraction(self) -> SourceNode:
        """A ``glyph_health`` assessment must be attributable to THIS node's
        extracted text, not merely present alongside it.

        Requires ``extraction`` to be present whenever ``glyph_health`` is
        (an assessment of "the extracted text" with no recorded extraction
        to point at is meaningless), and requires
        ``glyph_health.assessor.input_sha256`` to both be present and equal
        ``extraction.extracted_text_sha256`` -- an assessment whose input
        digest does not match this node's extracted text cannot be
        attributed to this node, whatever else it might be valid evidence
        of.
        """
        if isinstance(self.glyph_health, Absent):
            return self
        if isinstance(self.extraction, Absent):
            raise ValueError(
                f"node {self.node_id!r} carries a glyph_health assessment but no extraction "
                "binding; an assessment of 'the extracted text' cannot be attributed to this node "
                "without a recorded extraction to point at -- extraction must be present whenever "
                "glyph_health is"
            )
        input_sha256 = self.glyph_health.assessor.input_sha256
        if isinstance(input_sha256, Absent) or input_sha256 != self.extraction.extracted_text_sha256:
            raise ValueError(
                f"node {self.node_id!r} carries a glyph_health assessment whose assessor.input_sha256="
                f"{input_sha256!r} does not equal this node's extraction.extracted_text_sha256="
                f"{self.extraction.extracted_text_sha256!r}; an assessment whose input digest doesn't "
                "match the node's extracted text cannot be attributed to that node"
            )
        return self

    @model_validator(mode="after")
    def _validate_extraction_parent_matches_node_sha256(self) -> SourceNode:
        """A recorded ``extraction`` must address a record derived from THIS
        node's own raw bytes, not merely be present alongside them.

        ``extraction.parent_raw_sha256`` is the sha256 of the raw artifact the
        addressed extraction record was actually derived from -- it must equal
        this node's own ``sha256``, or the binding names an extraction of some
        OTHER node's bytes while claiming to describe this one.
        """
        if isinstance(self.extraction, Absent):
            return self
        if self.extraction.parent_raw_sha256 != self.sha256:
            raise ValueError(
                f"node {self.node_id!r} has sha256={self.sha256!r} but its extraction binding names "
                f"parent_raw_sha256={self.extraction.parent_raw_sha256!r}; an extraction record derived "
                "from different raw bytes cannot be attributed to this node"
            )
        return self

    @model_validator(mode="after")
    def _validate_figure_crop_has_no_extraction(self) -> SourceNode:
        """I6: a ``FIGURE_CROP`` node's ``extraction`` (and, if present,
        ``glyph_health``) must be exactly ``Absent(reason=NOT_APPLICABLE)``.

        A ``FIGURE_CROP`` is an image region -- there is no flat text layer
        for it to have been extracted from, which is exactly why no
        ``CharSpanLocator`` may target one (see
        ``_LOCATOR_KIND_COMPATIBLE_NODE_KINDS`` and
        ``_validate_locator_kind_compatibility``). ``NOT_EXTRACTED_YET``
        would misstate that: it says extraction merely hasn't happened yet,
        implying it validly could later, when for an image region it never
        will -- there is no text to extract. ``NOT_APPLICABLE`` is the only
        reason that actually describes a crop, and since these ``reason``
        values are recorded evidence (what a future extraction run, or an
        auditor, should conclude from them) rather than decoration, the
        wrong one is a real defect, not a cosmetic one.

        This also closes a sha256-sharing hole. A crop USED to be allowed to
        share its parent PAPER_PDF's ``sha256`` -- ``SourceGraph``'s I8 now
        forbids exactly that -- and a crop that carried a PRESENT
        ``ExtractionBinding`` could then name that same parent's extraction
        address, claiming the parent's extracted text as its own when no
        locator on a crop can legitimately slice any of it. This validator
        remains the guard for that claim on its own terms: it holds at NODE
        level, without needing a parent in scope, and it forbids a crop
        naming ANY extraction address, not merely an ancestor's.

        ``glyph_health`` is folded into the same requirement for the same
        reason: it assesses the quality of extracted OCR'd text, and a crop
        has none to assess, so an Absent ``glyph_health`` on a crop must
        also carry ``NOT_APPLICABLE`` rather than ``NOT_EXTRACTED_YET``.
        (``_validate_glyph_health_binds_to_this_nodes_extraction``, above,
        already forces ``glyph_health`` to be ``Absent`` whenever
        ``extraction`` is; this validator only additionally pins down WHICH
        ``AbsenceReason`` is legal for a crop.)
        """
        if self.kind != SourceNodeKind.FIGURE_CROP:
            return self
        if not isinstance(self.extraction, Absent) or self.extraction.reason != AbsenceReason.NOT_APPLICABLE:
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which is an image region with no "
                "extracted text to bind -- extraction must be Absent(reason=AbsenceReason.NOT_APPLICABLE), "
                f"not {self.extraction!r}"
            )
        if not isinstance(self.glyph_health, Absent) or self.glyph_health.reason != AbsenceReason.NOT_APPLICABLE:
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which has no extracted text for a "
                "glyph-health assessment to describe -- glyph_health must be "
                f"Absent(reason=AbsenceReason.NOT_APPLICABLE), not {self.glyph_health!r}"
            )
        if not isinstance(self.verification, Absent) or self.verification.reason != AbsenceReason.NOT_APPLICABLE:
            # Folded in for the same reason as glyph_health above, and it pins the
            # same distinction: the iff-rule in
            # `_validate_verification_binds_to_extraction` already forces a crop's
            # `verification` to be SOME Absent (because its extraction is), so all
            # this validator decides is WHICH AbsenceReason -- and for an image
            # region NOT_EXTRACTED_YET would be a false promise that a verification
            # story could arrive later. It never can; there is no text to verify.
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which has no extracted text for a "
                "verification record to describe -- verification must be "
                f"Absent(reason=AbsenceReason.NOT_APPLICABLE), not {self.verification!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_crop_region_addresses_a_figure_crop(self) -> SourceNode:
        """I7: a ``FIGURE_CROP`` must carry a concrete ``crop_region``; every
        other kind must carry ``Absent(reason=NOT_APPLICABLE)``.

        An UNADDRESSABLE CROP IS REFUSED, not admitted with a note. A crop
        that cannot say which figure it is has no identity of its own -- it
        is distinguishable from its siblings only by a ``node_id`` its own
        producer chose -- and every consumer downstream of it (a
        ``DIGITIZED`` value citing it as the thing its number was read off, a
        replayer resolving it, a human auditing that citation) would have
        nothing to check the claim against. No :class:`AbsenceReason`
        rescues that: an image region ALWAYS has a location on the page it
        was cut from, so ``NOT_APPLICABLE`` would be false; and
        ``NOT_EXTRACTED_YET``/``UNKNOWN`` promise an address that arrives
        later, for a node already being cited as provenance now. Absence is
        therefore rejected outright here rather than narrowed to one legal
        reason the way I6 narrows a crop's extraction.

        The other direction mirrors ``_validate_origin_only_for_si_member``:
        only a crop was ever cut out of a page, so any other kind claiming a
        crop region describes a relationship that cannot exist. Its absence
        reason is pinned to ``NOT_APPLICABLE`` for the same reason I6 pins a
        crop's: these reasons are recorded evidence about what a later run
        should conclude, and "this PAPER_PDF has no crop region YET" is a
        false promise rather than a cosmetic slip.
        """
        if self.kind == SourceNodeKind.FIGURE_CROP:
            if isinstance(self.crop_region, Absent):
                raise ValueError(
                    f"node {self.node_id!r} has kind={self.kind.value!r} but cannot address itself: "
                    f"crop_region is {self.crop_region!r}. A crop with no region names no figure -- "
                    "it is distinguishable from another crop of the same paper only by a node_id its "
                    "own producer chose -- so nothing downstream could ever check a value digitized "
                    "off it against the figure it claims to come from; no AbsenceReason makes that "
                    "honest, because an image region always has a location on the page it was cut from"
                )
            return self
        if not isinstance(self.crop_region, Absent):
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which was not cut out of a page "
                "and so cannot carry a crop_region -- only a FIGURE_CROP can; crop_region must be "
                f"Absent(reason=AbsenceReason.NOT_APPLICABLE) here, not {self.crop_region!r}"
            )
        if self.crop_region.reason != AbsenceReason.NOT_APPLICABLE:
            raise ValueError(
                f"node {self.node_id!r} has kind={self.kind.value!r}, which was not cut out of a page, "
                f"so crop_region must be Absent(reason={AbsenceReason.NOT_APPLICABLE.value!r}) -- "
                f"reason {self.crop_region.reason.value!r} would promise an address that could arrive "
                "later, and for a node that is not a crop none ever can"
            )
        return self

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through.
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid sha256: {value!r} (expected 64 lowercase hex characters)")
        return value


class TextSpace(StrEnum):
    """Which string a :class:`CharSpanLocator`'s offsets index into.

    Exactly one member today. :class:`~carmel.agents.tools.extract.ExtractedText`
    carries BOTH a ``text`` and a ``normalized`` string
    (``carmel/agents/tools/extract.py:143``), and the grounding gate
    deliberately matches in NORMALIZED space but then maps the offsets it
    RECORDS back to RAW ``text`` indices (``carmel/services/grounding.py``,
    ``carmel/services/literature.py:985``). Offsets into the wrong one of
    those two strings are silently, invisibly wrong -- same character count,
    different characters at each index -- so the space a locator's offsets
    index into is named explicitly here rather than assumed.
    """

    EXTRACTED_TEXT = "extracted_text"


class LocatorKind(StrEnum):
    """Discriminator for :data:`SourceLocator`."""

    BBOX = "bbox"
    TABLE_CELL = "table_cell"
    XPATH = "xpath"
    CHAR_SPAN = "char_span"


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
    table REGION was deliberately deferred here "to M-C/M1: it is circular to
    require now, before any extractor exists that defines what bytes
    constitute 'the region' to hash". That extractor now exists
    (:mod:`carmel.services.pdf_tables` derives a grid,
    :mod:`carmel.services.pdf_table_record` addresses it), so the deferral is
    closed by ``pdf_table_inventory_sha256`` below.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.TABLE_CELL] = LocatorKind.TABLE_CELL
    table_key: TableKey
    row: int = Field(ge=0)
    """0-indexed row; a negative row locates no real table cell."""
    col: int = Field(ge=0)
    """0-indexed column; a negative col locates no real table cell."""
    pdf_table_inventory_sha256: Maybe[str]
    """The content address of the PDF cell inventory that DEFINES the grid
    ``row``/``col`` index into -- i.e. the ``inventory_sha256`` computed by
    :func:`carmel.services.pdf_table_record.compute_inventory_sha`.

    Without it, one document holding several tables leaves nothing able to
    say WHICH derived grid justified a cell, and a replayer would have to
    rescan and guess a footprint. ``table_key`` cannot stand in: it is a
    printed caption label, and its own docstring disclaims uniqueness.

    NAMED ``pdf_``-specifically on purpose. A ``TableCellLocator`` may also
    target ``JATS_XML`` and ``SI_MEMBER`` nodes (see
    :data:`_LOCATOR_KIND_COMPATIBLE_NODE_KINDS`), whose cells have no PDF
    fragment geometry at all; a general name would invite laundering an XML
    or workbook citation through a field that only ever means "a grid derived
    from PDF text fragments".

    ``Maybe``-typed with NO default, for the same "no unreasoned absence"
    reasoning as :class:`SourceNode`'s ``origin``/``extraction``: whether a
    cell can cite an inventory is a structural fact about its target, and
    every locator must state which case it is in. WHICH absences are legal is
    NOT a property of this field -- a locator does not know its own node --
    so it is enforced at envelope level by
    :func:`_validate_table_cell_inventory_citation`, which permits exactly
    one ``AbsenceReason`` and refuses ``Absent`` entirely for a PDF node.

    Deliberately NOT keyed on ``SourceNode.extraction``: the fragment lane
    (``pdf_fragments.extract_fragments(data: bytes)``) reads RAW BYTES and
    needs no extraction record, so a PDF node whose text extraction is
    ``Absent`` can still have an inventory. Treating a missing extraction as
    "no inventory applies" would make exactly that node the bypass."""

    @field_validator("pdf_table_inventory_sha256")
    @classmethod
    def _validate_inventory_sha256_shape(cls, value: Maybe[str]) -> Maybe[str]:
        # fullmatch, never match: `$` also matches just BEFORE a trailing newline, so
        # match would let "a" * 64 + "\n" through -- same reasoning as
        # EmbeddedConversionTable._validate_sha256_shape.
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            raise ValueError(
                f"TableCellLocator.pdf_table_inventory_sha256 {value!r} is not 64 lowercase hex characters"
            )
        return value


class XPathLocator(BaseModel):
    """Locates a reference in JATS/XML via an XPath expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.XPATH] = LocatorKind.XPATH
    xpath: str = Field(min_length=1)


class CharSpanLocator(BaseModel):
    """Locates a reference by a half-open character span ``[start, end)``
    into a node's extracted text.

    This is the ONE positional primitive the shipped pipeline actually
    emits: nothing in this codebase renders a page (so no
    :class:`CoordinateFrame` a :class:`BBoxLocator` could honestly cite),
    the pypdf text extractor emits no table cells, and XPath is JATS-only
    while the corpus is PDFs. A character offset into
    ``EvidenceRef.quote_start``/``quote_end``
    (``carmel/schemas/literature.py:244-246``, populated at
    ``carmel/services/literature.py:985``) is the locator that names what
    the runtime can actually produce AND what a replayer can actually
    verify: re-extract, check the text digest, slice ``text[start:end]``,
    compare.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[LocatorKind.CHAR_SPAN] = LocatorKind.CHAR_SPAN
    text_space: TextSpace
    """Which string ``start``/``end`` index into -- see :class:`TextSpace`.

    DELIBERATELY HAS NO DEFAULT, like every other field in this module for
    which "the producer forgot to say" and "the producer positively meant
    this" must stay distinguishable (see
    :attr:`ExtractionBinding.pypdf_version`'s docstring for the same
    argument made at length). :class:`TextSpace` has exactly one member
    today, so no caller can choose WRONG -- but a default would mean that
    once a second space is added, every producer that simply forgot to say
    which space it meant would be silently recorded as having POSITIVELY
    CLAIMED ``extracted_text``. Requiring the value now, while there is
    only one honest answer, costs nothing; it is the only way to make a
    future omission a loud crash instead of a silent misattribution.
    """
    start: int = Field(ge=0)
    """Inclusive start offset into the named text space."""
    end: int = Field(ge=0)
    """Exclusive end offset into the named text space."""

    @model_validator(mode="after")
    def _validate_span_nonempty(self) -> CharSpanLocator:
        """The span is HALF-OPEN ``[start, end)`` over the named text
        space, matching Python slicing and :attr:`TextSection.start`/
        :attr:`TextSection.end`. ``end == start`` locates zero characters
        and so cannot ground anything; ``end < start`` locates nothing at
        all (a reversed range). Both are rejected -- only a positive-width
        span is a locator that could ever verify against real text.
        """
        if self.end <= self.start:
            raise ValueError(
                f"CharSpanLocator: end={self.end!r} must be strictly greater than start={self.start!r} -- "
                "the span [start, end) is half-open, so end == start locates zero characters and "
                "end < start locates nothing at all"
            )
        return self


SourceLocator = Annotated[
    BBoxLocator | TableCellLocator | XPathLocator | CharSpanLocator,
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


def iter_uncertainties(obj: object, _path: str = "") -> Iterator[tuple[str, Uncertainty]]:
    """Recursively walk ``obj``, yielding ``(dotted_path, uncertainty)`` for every
    :class:`Uncertainty` reachable from it.

    Mirrors :func:`iter_measured_values` exactly, and exists for the same
    reason a second walk exists anywhere in this module: it is the INDEPENDENT
    side of a duplication check. ``carmel.services.dataset_replay`` enumerates
    the uncertainty sites an envelope carries BY HAND, because deciding which
    fields are assertions about the paper (``kind``, ``basis``, ``scale``) and
    which are self-describing machinery is a semantic judgment no generic walk
    can make. Reconciling that hand-written inventory against this walk is what
    stops the inventory silently going stale when a new ``Uncertainty``-bearing
    field is added -- and the reconciliation is only meaningful because the two
    are written separately. Deriving either from the other would make it a
    tautology that always passes.

    Unlike :func:`iter_measured_values`, this walk DOES recurse into the yielded
    object: an ``Uncertainty``'s ``upper``/``lower`` are :class:`MeasuredValue`
    bounds and are not uncertainties themselves, so there is no risk of yielding
    a nested ``Uncertainty`` twice, and stopping here would be an arbitrary
    difference from the sibling walk rather than a considered one.
    """
    if isinstance(obj, Uncertainty):
        yield _path, obj
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            child_path = f"{_path}.{name}" if _path else name
            yield from iter_uncertainties(value, child_path)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_uncertainties(value, f"{_path}[{key}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_uncertainties(value, f"{_path}[{index}]")
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
        # stays legal: e.g. a PAPER_PDF root and an SI_MEMBER child that is a
        # byte-identical copy of that same paper share a sha256 but are not
        # duplicates of each other. (This example used to be a PAPER_PDF and a
        # FIGURE_CROP taken from it; I8 below now refuses that pairing outright
        # -- not because I5 changed, but because a crop borrowing its parent's
        # bytes was never an honest identity in the first place.)
        # This triple deliberately does NOT include `origin`: two
        # byte-identical SI_MEMBER files belonging to the same paper are now
        # DISTINGUISHABLE from each other via `origin` (different archives,
        # or different member_display_paths within the same archive), but
        # this invariant does not treat that distinction as enough on its
        # own to admit both -- widening the triple to include origin is a
        # deliberate future change, not an oversight here.
        # I5c: two nodes that name the SAME extraction address -- the pair
        # (extraction.parent_raw_sha256, extraction.extraction_sha256) -- must
        # agree on their whole ExtractionBinding, when both have one present.
        # This is the STRICTER within-one-graph case: even when two nodes
        # agree on which extraction record they are naming (see
        # carmel.services.extraction_record:
        # evidence/literature/<raw sha256>/extractions/<extraction sha256>/
        # {extracted.json,text.txt,meta.json}), they must also agree on what
        # that record actually contains -- at most one of two disagreeing
        # bindings for the same address can match what is actually stored
        # there, so the other is guaranteed unresolvable by any replayer.
        # (Whether two nodes may name DIFFERENT extraction addresses for the
        # SAME raw sha256 at all is a separate question, answered NO within
        # one graph -- see I5d below, which is the actual guard for that
        # case.)
        # This compares the WHOLE ExtractionBinding via `==` (the model is
        # frozen, so this is value equality over every field) rather than
        # field-by-field, so that a future field added to ExtractionBinding is
        # automatically covered here without needing to be remembered. An
        # Absent extraction on either side is deliberately NOT a conflict --
        # only compare when BOTH nodes have a present binding; silence is not
        # a contradiction, exactly as I5b (below) does not treat an Absent
        # glyph_health as disagreeing with anything. This check is not gated
        # on glyph_health in any way and must fire even when NEITHER node has
        # any glyph_health at all -- that is the widest part of the hole it
        # closes.
        extraction_by_address: dict[tuple[str, str], tuple[str, ExtractionBinding]] = {}
        for node in self.nodes:
            if isinstance(node.extraction, Absent):
                continue
            address = (node.extraction.parent_raw_sha256, node.extraction.extraction_sha256)
            if address in extraction_by_address:
                other_node_id, other_extraction = extraction_by_address[address]
                if other_extraction != node.extraction:
                    raise ValueError(
                        f"node {node.node_id!r} and node {other_node_id!r} both name extraction address "
                        f"(parent_raw_sha256={address[0]!r}, extraction_sha256={address[1]!r}) but their "
                        "recorded ExtractionBinding values disagree; this is a CONFLICT, not a legal "
                        "'different role' duplicate: at most one of two disagreeing bindings for the "
                        "same extraction address can match what is actually stored under "
                        "evidence/literature/<raw sha256>/extractions/<extraction sha256>/ -- the other "
                        "is guaranteed unresolvable and must be reconciled before this graph can validate"
                    )
                continue
            extraction_by_address[address] = (node.node_id, node.extraction)

        # I5d: all PRESENT ExtractionBindings within this one SourceGraph
        # that share a parent_raw_sha256 must name the SAME extraction_sha256.
        #
        # This is deliberately narrower than "the evidence store may hold
        # many extraction records per raw document" (see I5c above) -- that
        # breadth is a property of the STORE ACROSS TIME: a pypdf upgrade, or
        # a change to Carmel's own extraction code, mints a new extraction
        # record beside an older one for the same raw bytes, and every
        # docstring in this codebase that justifies multiple extraction
        # records per raw sha256 makes exactly that temporal-supersession
        # argument. A single SourceGraph is not the store across time -- it
        # is the output of ONE producer run against ONE extraction of each
        # raw document. If two nodes in that one graph name the same
        # parent_raw_sha256 but DIFFERENT extraction_sha256 values, they are
        # claiming this graph's evidence for that document was read from two
        # different texts. A CharSpanLocator's offsets index into the
        # extracted text identified by extraction_sha256, not into the raw
        # bytes -- so two different extraction addresses for one raw
        # document mean their character offsets index into two DIFFERENT
        # texts. An envelope could then carry two claims, each individually
        # "verified" against its own extraction address, that are
        # nonetheless grounded in mutually inconsistent readings of the same
        # underlying document. That is not the legal
        # independently-resolvable-extraction-record case I5c's address
        # keying protects; it is a coherence violation this graph must
        # reject.
        #
        # A previous change re-keyed I5c onto the (raw sha256,
        # extraction_sha256) pair and, in doing so, made "same raw bytes,
        # different extraction addresses" legal *within one graph* -- that
        # was over-broad: it imported the store's cross-time breadth into a
        # single graph, where it does not belong. This invariant restores
        # what the original single-extraction-per-raw-sha256 check actually
        # protected, with the correct justification this time.
        #
        # Absent extractions do not participate: silence is not a
        # contradiction, exactly as I5c (above) and I5b (below) do not treat
        # an Absent extraction/glyph_health as disagreeing with anything.
        # This is SEPARATE from, and composes with, I5c's same-address
        # full-binding equality check above: I5c fires when two nodes name
        # the IDENTICAL address and disagree on its contents; I5d fires when
        # two nodes name DIFFERENT addresses for the same raw document at
        # all, regardless of what their bindings say.
        extraction_sha_by_raw_sha: dict[str, tuple[str, str]] = {}
        for node in self.nodes:
            if isinstance(node.extraction, Absent):
                continue
            raw_sha256 = node.extraction.parent_raw_sha256
            extraction_sha256 = node.extraction.extraction_sha256
            if raw_sha256 in extraction_sha_by_raw_sha:
                other_node_id, other_extraction_sha256 = extraction_sha_by_raw_sha[raw_sha256]
                if other_extraction_sha256 != extraction_sha256:
                    raise ValueError(
                        f"node {node.node_id!r} and node {other_node_id!r} both name raw document "
                        f"parent_raw_sha256={raw_sha256!r} but recorded DIFFERENT extraction_sha256 "
                        f"values ({extraction_sha256!r} vs {other_extraction_sha256!r}) within this "
                        "one SourceGraph; a CharSpanLocator's offsets index into the text identified "
                        "by extraction_sha256, so two extraction addresses for one raw document mean "
                        "an envelope could carry two claims each individually 'verified' against "
                        "mutually inconsistent readings of the same document -- this is a CONFLICT, "
                        "not the legal cross-time case of many extraction records for one raw "
                        "document (that breadth belongs to the evidence store across time, not to a "
                        "single graph produced by one producer run)"
                    )
                continue
            extraction_sha_by_raw_sha[raw_sha256] = (node.node_id, extraction_sha256)

        # I5b: two nodes that name the SAME extraction address (see I5c
        # above) must agree on glyph health, if both have any recorded. The
        # health assessment describes THE EXTRACTED TEXT that address names,
        # not the raw bytes it was derived from. (Two nodes sharing a raw
        # sha256 while naming DIFFERENT extraction addresses -- which would
        # let them legitimately disagree, since they'd be assessments of
        # different extracted text -- cannot occur within one graph at all:
        # I5d below forbids that combination outright. So by the time this
        # loop runs, any two nodes sharing a raw sha256 are already known to
        # share one extraction address too, and keying on the extraction
        # address here is equivalent to keying on the raw sha256.) Keying
        # this on the extraction address rather than raw sha256 is safe: `SourceNode.
        # _validate_glyph_health_binds_to_this_nodes_extraction` already
        # guarantees `extraction` is present whenever `glyph_health` is, so
        # every node reaching this loop with a glyph_health also has an
        # extraction to key on. Two nodes naming the SAME address can only
        # actually have one glyph-health story, so a disagreement there is
        # not a legal "different role" case like the duplicate check below
        # (which is about kind/parent, not about the underlying extracted
        # text) -- it is a CONFLICT between two assessments of the same
        # text, and must be rejected as such rather than silently admitted.
        #
        # THE ORDER OF THESE CHECKS IS LOAD-BEARING, not stylistic.
        # Nodes sharing a (kind, sha256, parent_node_id) triple AND disagreeing
        # on glyph health satisfy both checks at once. Running the duplicate
        # check first therefore reports them as "an exact repeat" that "adds
        # nothing" -- which is precisely false, because they differ on health,
        # and it sends the reader looking for a redundant node instead of an
        # irreconcilable assessment. The conflict is the more specific and more
        # actionable diagnosis, so it must be raised first. A regression test
        # covers same-triple-disagreeing-health for exactly this reason; if it
        # starts reporting a duplicate, these blocks have been reordered.
        #
        # I5c (above) must run before I5b, for the same reason one level up:
        # when two same-address nodes disagree on BOTH extraction and glyph
        # health, the extraction disagreement is the ROOT CAUSE (different
        # extracted text naturally produces different health) and the health
        # disagreement is only a downstream SYMPTOM of it -- reporting the
        # symptom first misdirects the reader away from the actual conflict.
        # This exact class of bug already bit this file once: the I5b branch
        # was written correctly but placed after the duplicate check, so for
        # the case that mattered it was unreachable. Do not recreate that
        # mistake by placing I5c after I5b.
        health_by_address: dict[tuple[str, str], tuple[str, GlyphHealth]] = {}
        for node in self.nodes:
            if isinstance(node.glyph_health, Absent):
                continue
            assert not isinstance(node.extraction, Absent)  # guaranteed above; see docstring
            address = (node.extraction.parent_raw_sha256, node.extraction.extraction_sha256)
            health = node.glyph_health.health
            if address in health_by_address:
                other_node_id, other_health = health_by_address[address]
                if other_health != health:
                    raise ValueError(
                        f"node {node.node_id!r} and node {other_node_id!r} both name extraction address "
                        f"(parent_raw_sha256={address[0]!r}, extraction_sha256={address[1]!r}) but their "
                        "recorded glyph_health assessments disagree; the same extracted text can only "
                        "have one true glyph-health story, so this is a CONFLICT between two "
                        "assessments of the same text -- not a legal 'different role' duplicate -- and "
                        "must be reconciled before this graph can validate"
                    )
                continue
            health_by_address[address] = (node.node_id, health)

        # I5e: two nodes naming the SAME extraction address must also agree on
        # what was VERIFIED about it. Same reasoning as I5b one block up, on a
        # different field: a SourceVerification describes the artifact and the
        # extracted text that address names, and those have exactly one true
        # verification story. Left unchecked, a graph could carry one node
        # claiming its root sidecar authenticated and a second node on the very
        # same record claiming it has no recorded digest -- a contradiction
        # replay would then report twice, once per node, without either
        # finding naming the fact that the ENVELOPE disagrees with itself.
        #
        # Placed after I5b for the same ordering reason I5c is placed before
        # it: an extraction-binding conflict is the root cause and a
        # verification disagreement is downstream of it, so the more specific
        # diagnosis has already had its chance to fire.
        verification_by_address: dict[tuple[str, str], tuple[str, SourceVerification]] = {}
        for node in self.nodes:
            if isinstance(node.verification, Absent):
                continue
            assert not isinstance(node.extraction, Absent)  # SourceNode's iff-rule guarantees this
            address = (node.extraction.parent_raw_sha256, node.extraction.extraction_sha256)
            if address in verification_by_address:
                other_node_id, other_verification = verification_by_address[address]
                if other_verification != node.verification:
                    raise ValueError(
                        f"node {node.node_id!r} and node {other_node_id!r} both name extraction address "
                        f"(parent_raw_sha256={address[0]!r}, extraction_sha256={address[1]!r}) but their "
                        "recorded SourceVerifications disagree; the same artifact and extracted text can "
                        "only have one true verification story, so this envelope contradicts itself and "
                        "must be reconciled before this graph can validate"
                    )
                continue
            verification_by_address[address] = (node.node_id, node.verification)

        # I8: a FIGURE_CROP may not carry the sha256 of any node in its own
        # ancestor chain. A crop is a strict sub-region of the thing it was cut
        # from, so its bytes are never that thing's bytes; a crop naming them
        # is not addressing itself at all, it is pointing at the whole parent
        # and calling the result an identity. That borrowed sha256 used to be
        # explicitly legal here, and it is what made two crops of two DIFFERENT
        # figures in one paper collide under the duplicate rule below: with
        # kind, sha256 and parent_node_id all identical, the second crop was
        # refused as an exact repeat of the first. Closing it is what lets
        # `SourceNode.crop_region` (I7) be a real identity rather than
        # decoration -- and it leaves the duplicate triple below untouched.
        #
        # The WHOLE ancestor chain, not just the immediate parent: a crop under
        # an SI member that names the root paper's bytes makes the same false
        # claim one level further up. I2 and I3 above already guarantee every
        # parent_node_id resolves and the chain terminates, so this walk cannot
        # loop or KeyError.
        #
        # THE ORDER OF THESE CHECKS IS LOAD-BEARING, for the same reason it is
        # between I5b and I5c. Two crops that both borrow their parent's sha256
        # satisfy this invariant AND the duplicate rule at once, and the
        # duplicate diagnosis ("an exact repeat adds nothing") is precisely
        # wrong for them: nothing here is redundant, and the reader sent
        # looking for a superfluous node will not find one -- what is actually
        # wrong is that NEITHER crop has an identity of its own. This must
        # therefore run first, and a regression test pins that message.
        for node in self.nodes:
            if node.kind != SourceNodeKind.FIGURE_CROP:
                continue
            ancestor_id = node.parent_node_id
            while ancestor_id is not None:
                ancestor = nodes_by_id[ancestor_id]
                if ancestor.sha256 == node.sha256:
                    raise ValueError(
                        f"node {node.node_id!r} has kind={node.kind.value!r} but borrows "
                        f"sha256={node.sha256!r} from its ancestor {ancestor_id!r} "
                        f"(kind={ancestor.kind.value!r}); a crop is a sub-region of what it was cut "
                        "from, so it never holds that whole artifact's bytes -- a crop naming them "
                        "has no identity of its own, and two crops doing it are indistinguishable "
                        "from each other by anything but the node_id their producer chose"
                    )
                ancestor_id = ancestor.parent_node_id

        # I9: two FIGURE_CROP nodes under the SAME parent may not name the same
        # crop_region. One rectangle, measured against one render of one page,
        # is one figure -- so two nodes claiming it are two names for a single
        # thing, and if their sha256 values differ they are additionally two
        # different sets of bytes each claiming to BE that thing, a
        # contradiction no consumer can arbitrate. Keyed on the parent because
        # the region is expressed in the PARENT's frame (see
        # `SourceNode.crop_region`): the same coordinates under a different
        # parent address a different document's page.
        #
        # Placed here, before the duplicate rule, for the same ordering reason
        # as I8 above: two crops naming one region are a specific, actionable
        # collision, and "an exact repeat adds nothing" would describe neither
        # what happened nor what to do about it.
        crop_region_by_parent: dict[tuple[str, BBox], str] = {}
        for node in self.nodes:
            if node.kind != SourceNodeKind.FIGURE_CROP or isinstance(node.crop_region, Absent):
                # A crop with an Absent region cannot reach here (I7 rejects it
                # at node level), and a non-crop cannot carry one at all; the
                # guard is what makes that structural fact explicit to a reader
                # rather than an assumption this loop silently depends on.
                continue
            assert node.parent_node_id is not None  # I4 above: a crop always has a parent
            key = (node.parent_node_id, node.crop_region)
            if key in crop_region_by_parent:
                raise ValueError(
                    f"node {node.node_id!r} and node {crop_region_by_parent[key]!r} are both "
                    f"FIGURE_CROPs of parent {node.parent_node_id!r} and name the same crop_region "
                    "(same rectangle, same CoordinateFrame); one rectangle on one render of one page "
                    "is one figure, so these are two names for a single figure -- and where their "
                    "sha256 values differ, two different sets of bytes each claiming to be it"
                )
            crop_region_by_parent[key] = node.node_id

        # Keyed by the same triple as ever, and admitting exactly the same
        # graphs; the dict (rather than a set) only remembers WHICH earlier node
        # collided, so the message below can name it and, for the one case
        # I7/I8/I9 made expressible, say what was actually detected.
        first_by_triple: dict[tuple[SourceNodeKind, str, str | None], SourceNode] = {}
        for node in self.nodes:
            triple = (node.kind, node.sha256, node.parent_node_id)
            earlier = first_by_triple.get(triple)
            if earlier is not None:
                # Two FIGURE_CROPs of one parent with identical bytes but
                # DIFFERENT regions are not an exact repeat -- they name two
                # different figures that happen to render to the same pixels (a
                # repeated inset, a duplicated axis panel). The triple cannot
                # see that difference, and widening it to include crop_region
                # would weaken duplicate detection for every node kind to admit
                # a rare case, so the graph still refuses them -- but it must
                # refuse them for the reason that is true, not by claiming a
                # redundancy that is not there. (The equal-region case never
                # reaches here: I9 above refuses it first, with its own
                # message.)
                if (
                    node.kind == SourceNodeKind.FIGURE_CROP
                    and earlier.kind == SourceNodeKind.FIGURE_CROP
                    and node.crop_region != earlier.crop_region
                ):
                    raise ValueError(
                        f"node {node.node_id!r} and node {earlier.node_id!r} are both FIGURE_CROPs of "
                        f"parent {node.parent_node_id!r} carrying the SAME sha256={node.sha256!r} while "
                        "naming DIFFERENT crop_regions -- two figures that render to identical bytes. "
                        "They are not an exact repeat of each other, but node identity here is "
                        "(kind, sha256, parent_node_id), which cannot tell them apart, so this graph "
                        "cannot hold both: give each crop the bytes of its own region, or keep one node "
                        "and cite the region through a BBoxLocator instead"
                    )
                raise ValueError(
                    f"node {node.node_id!r} duplicates an earlier node exactly: another node already "
                    f"has kind={node.kind.value!r}, sha256={node.sha256!r}, "
                    f"parent_node_id={node.parent_node_id!r}; the same bytes in a genuinely different "
                    "role (a different kind or a different parent) remains legal, but an exact repeat "
                    "adds nothing"
                )
            first_by_triple[triple] = node

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
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through.
        if not _SHA256_RE.fullmatch(value):
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


class UnitProvenance(StrEnum):
    """Whether this value's unit was PRINTED in the source, or is a declared
    dimensionless unit the source never printed at all.

    A first-class stored fact, never an absence and never a comment: requirement
    I-060 is that a reader of the stored data must be able to tell a unit that was
    NEVER PRINTED from one merely unrecorded. The absence of ``unit_raw`` cannot
    carry that distinction on its own -- an absence says "not here", not "the
    source printed nothing" -- so this marker states it explicitly.
    """

    PRINTED_IN_SOURCE = "printed_in_source"
    """The ordinary, universal case: the unit was printed in the document and is
    grounded by ``unit_raw`` (the verbatim token) and ``unit_ref`` (its location).
    Every value in the corpus is this unless it is the one narrow exception
    below."""

    NOT_PRINTED_IN_SOURCE = "not_printed_in_source"
    """The source printed NO unit token for this value, and the dimensionless unit
    is asserted from the quantity alone. Permitted for
    :attr:`~carmel.services.units.QuantityKind.EQUIVALENCE_RATIO` ONLY -- the sole
    dimensionless quantity whose canonical unit (``"1"``) carries no SCALE, so
    asserting it rescales nothing. The contrast is load-bearing and is the whole
    reason this is not a class-wide dimensionless exemption: the conversion table
    scales ``%`` -> ``1`` by 0.01 and ``ppm`` -> ``1`` by 1e-6 for MOLE_FRACTION,
    MASS_FRACTION and RELATIVE_UNCERTAINTY, so declaring one of THOSE without a
    printed unit would silently rescale a stored magnitude by 100x or 1,000,000x.
    "Carries no information" and "carries no scale" are different claims;
    equivalence ratio is safe on the second, not the first.

    In this state ``unit_raw`` and ``unit_ref`` are :class:`Absent` -- there is no
    printed token to quote and no location to cite -- and ``unit_normalized`` is
    the quantity's dimensionless base unit ``"1"``. That the column IS this
    quantity is NOT established by this marker: it is a separate unchecked claim
    replay files (the header bridge), because grounding a label proves LOCATION,
    never MEANING."""


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
    ever suspect. Recording document glyph health belongs one level down, on
    the per-document :class:`SourceNode` (see its ``glyph_health`` field),
    NOT on the multi-document ``DatasetEnvelope``: an envelope holds a whole
    :class:`SourceGraph` of documents, so stamping the fact at envelope level
    would reintroduce the exact arbitration failure it must avoid -- two
    nodes in the SAME envelope could then disagree on glyph health with no
    way to say which document a given assessment actually describes. It is
    still deliberately NOT stamped onto every value: a document-level fact
    repeated per point would let two values extracted from the SAME document
    assert different glyph health with no way to arbitrate between them.
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
    unit_provenance: UnitProvenance = UnitProvenance.PRINTED_IN_SOURCE
    """Whether the unit was PRINTED in the source (the default, universal case) or
    is a dimensionless unit the source never printed
    (:attr:`UnitProvenance.NOT_PRINTED_IN_SOURCE`, EQUIVALENCE_RATIO only). A
    first-class field, not an inference from ``unit_raw`` being absent -- see the
    enum's own docstring. Enforced below: NOT_PRINTED_IN_SOURCE requires
    ``quantity_kind == EQUIVALENCE_RATIO``, ``unit_raw``/``unit_ref`` both
    :class:`Absent`, and ``unit_normalized == "1"``; PRINTED_IN_SOURCE requires
    both present, exactly as before this field existed."""
    unit_raw: Maybe[str]
    """The unit exactly as printed in the source, or :class:`Absent` when
    ``unit_provenance`` is :attr:`UnitProvenance.NOT_PRINTED_IN_SOURCE` (the
    source printed no unit token). A present value must be non-empty -- enforced
    below rather than by ``Field(min_length=1)``, which a ``Maybe`` union does not
    reliably apply to its ``str`` member."""
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
    unit_ref: Maybe[SourceRef]
    """Provenance for the UNIT, independent of ``value_ref``. Present in the
    ordinary case; :class:`Absent` when ``unit_provenance`` is
    :attr:`UnitProvenance.NOT_PRINTED_IN_SOURCE`, because a unit the source never
    printed has no location to cite. The class docstring's "both are required"
    invariant is preserved for every PRINTED_IN_SOURCE value (enforced below); it
    is the not-printed marker, not a silent omission, that relaxes it for the one
    narrow dimensionless case."""

    @field_validator("conversion_table_sha256")
    @classmethod
    def _validate_conversion_table_sha256(cls, value: str) -> str:
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through.
        if not _SHA256_RE.fullmatch(value):
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
    def _validate_unit_provenance(self) -> MeasuredValue:
        """Enforce the ``unit_provenance`` contract, gated NARROWLY to equivalence
        ratio (I-060).

        The whole safety of declaring a dimensionless unit the source never
        printed rests on this gate. :attr:`UnitProvenance.NOT_PRINTED_IN_SOURCE`
        is allowed ONLY for :attr:`~carmel.services.units.QuantityKind.EQUIVALENCE_RATIO`,
        whose canonical unit ``"1"`` carries no scale. The three sibling
        dimensionless kinds -- MOLE_FRACTION, MASS_FRACTION, RELATIVE_UNCERTAINTY --
        are refused here, because their conversion table scales ``%``/``ppm`` to
        ``1`` and asserting an un-printed ``"1"`` for one of them would silently
        rescale a stored magnitude. In the not-printed state ``unit_raw`` and
        ``unit_ref`` MUST be :class:`Absent` (there is nothing to quote or cite)
        and ``unit_normalized`` MUST be the dimensionless base ``"1"``. In the
        ordinary PRINTED_IN_SOURCE state both MUST be present and ``unit_raw``
        non-empty -- restoring, exactly, the "a value whose unit has no
        independent provenance cannot be constructed" invariant the class
        docstring states.
        """
        unit_raw_absent = isinstance(self.unit_raw, Absent)
        unit_ref_absent = isinstance(self.unit_ref, Absent)
        if self.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE:
            if self.quantity_kind is not QuantityKind.EQUIVALENCE_RATIO:
                raise ValueError(
                    f"unit_provenance={self.unit_provenance.value!r} is permitted for "
                    f"quantity_kind={QuantityKind.EQUIVALENCE_RATIO.value!r} ONLY, not "
                    f"{self.quantity_kind.value!r}: equivalence ratio is the sole dimensionless "
                    "quantity whose unit '1' carries no scale, so asserting an unprinted unit "
                    "rescales nothing; the table scales %/ppm to 1 for the fraction kinds, so "
                    "declaring one of those without a printed unit would silently rescale a "
                    "stored magnitude"
                )
            if not unit_raw_absent:
                raise ValueError(
                    f"unit_provenance={self.unit_provenance.value!r} requires unit_raw to be Absent "
                    f"(the source printed no unit token), not {self.unit_raw!r}"
                )
            if not unit_ref_absent:
                raise ValueError(
                    f"unit_provenance={self.unit_provenance.value!r} requires unit_ref to be Absent "
                    "(a unit the source never printed has no location to cite)"
                )
            if self.unit_normalized != "1":
                raise ValueError(
                    f"unit_provenance={self.unit_provenance.value!r} requires unit_normalized == '1' "
                    f"(the dimensionless base unit of equivalence ratio), not {self.unit_normalized!r}"
                )
        else:
            if unit_raw_absent or unit_ref_absent:
                raise ValueError(
                    f"unit_provenance={self.unit_provenance.value!r} requires unit_raw and unit_ref "
                    "to be present: a value whose unit has no independent provenance cannot be "
                    "constructed (the not-printed marker, not a silent omission, is the only path to "
                    "an absent unit)"
                )
            # Narrowed by the guard above: not Absent, so a str.
            assert isinstance(self.unit_raw, str)
            if not self.unit_raw.strip():
                raise ValueError("unit_raw must be a non-empty, non-blank unit token when it is present")
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
        if self.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE:
            # No printed ``unit_raw`` to normalize (guaranteed Absent by
            # _validate_unit_provenance). ``unit_normalized`` was already pinned
            # to "1" there; the remaining check is that "1" IS the recorded
            # table's dimensionless base for equivalence ratio, so an edited table
            # that ever moved that base surfaces as a migration here rather than a
            # silent reinterpretation.
            base = table.base_unit(self.quantity_kind)
            if self.unit_normalized != base:
                raise ValueError(
                    f"unit_normalized={self.unit_normalized!r} disagrees with the recorded table's "
                    f"base unit {base!r} for quantity_kind={self.quantity_kind.value!r}; a not-printed "
                    "dimensionless unit is the quantity's base unit, never asserted independently"
                )
            return self
        # Past the not-printed early return above, _validate_unit_provenance (which
        # runs first) has already guaranteed a present str unit_raw for a printed
        # value; narrow it for the type checker.
        assert isinstance(self.unit_raw, str)
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

    PARTLY LIVE: a ``Coordinate`` is now constructed from a TABLE by
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
    whose values are ``TABLE_CELL`` locators; the CHAR-SPAN route stays refused.
    The char-span ``produce_envelope_from_artifact``
    (:mod:`carmel.services.dataset_producer`) still refuses
    unconditionally -- see its docstring for the full argument. In
    outline: this runtime can only locate a value with a
    :class:`CharSpanLocator` into extracted running text, and
    :meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`
    (V7) refuses a char span as the source of a series value, which a
    ``Coordinate`` always is. What makes a ``Coordinate`` specifically
    dormant rather than merely unused: even though its own field shape
    (a required, never-absent :class:`MeasuredValue`) is exactly right for a
    locating axis value, there is still no way to prove that the coordinate
    and the observation it locates come from the same structured row --
    that proof needs a ``TABLE_CELL`` or ``FIGURE_CROP`` locator. The
    ``TABLE_CELL`` one now exists, so the tabular producer fills this class;
    the ``FIGURE_CROP`` one does not. The replayer/validators exercise it.
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

    PARTLY LIVE: an ``Observation`` is now constructed from a TABLE by
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
    whose values are ``TABLE_CELL`` locators; the CHAR-SPAN route stays refused.
    The char-span ``produce_envelope_from_artifact``
    (:mod:`carmel.services.dataset_producer`) still refuses
    unconditionally -- see its docstring for the full argument. What
    makes an ``Observation`` specifically dormant: it is the field V7
    (:meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`)
    is actually about -- an observation's ``value`` is the series VALUE that
    validator forbids a :class:`CharSpanLocator` from grounding, because a
    char span in running text proves the location of a quoted number but
    never proves it pairs with the coordinate that locates the same point.
    The ``Maybe``-typed absence handling above is real schema machinery, not
    dead flexibility -- it is exercised by the replayer and by V7 itself --
    and the tabular producer now populates a real ``Observation`` via a
    ``TABLE_CELL`` locator (the table parser); a ``FIGURE_CROP`` node (a
    figure digitizer) does not yet exist.
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
    observations, and an optional per-point composition override.

    PARTLY LIVE: a ``DataPoint`` is now constructed from a TABLE by
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
    whose values are ``TABLE_CELL`` locators; the CHAR-SPAN route stays refused.
    The char-span ``produce_envelope_from_artifact``
    (:mod:`carmel.services.dataset_producer`) still refuses
    unconditionally -- see its docstring for the full argument. A
    ``DataPoint`` is specifically the object that ASSERTS the pairing V7
    (:meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`)
    says this runtime cannot prove: bundling one or more :class:`Coordinate`
    values with one or more :class:`Observation` values into a single
    ``point_id`` is exactly the "structured row" claim that a char span into
    running text carries no evidence for. The table parser (emitting
    ``TABLE_CELL`` locators) now fills this class via the tabular producer; a
    figure digitizer (emitting ``FIGURE_CROP`` nodes) does not yet exist. The
    replayer exercises it.
    """

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

    PARTLY LIVE: a ``Series`` is now constructed from a TABLE by
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
    whose values are ``TABLE_CELL`` locators; the CHAR-SPAN route stays refused.
    The char-span ``produce_envelope_from_artifact``
    (:mod:`carmel.services.dataset_producer`) still refuses
    unconditionally -- see its docstring for the full argument. In
    outline: this runtime can only locate a value with a
    :class:`CharSpanLocator` into extracted running text, and
    :meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`
    (V7) rejects a char span as the source of a series VALUE. A ``Series``
    is precisely the aggregate that makes that structural claim -- it
    asserts a fixed axis schema and a set of points that instantiate it, and
    running text carries no row structure from which that pairing could be
    proven, no matter how many points are declared. The table parser
    (``TABLE_CELL`` locators) now fills this class via the tabular producer; a
    figure digitizer (``FIGURE_CROP`` nodes) does not yet exist. The replayer
    and validators exercise it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    series_id: str
    source_form: SourceForm
    value_origin: ValueOrigin
    axes: tuple[AxisDeclaration, ...]
    constants: tuple[Coordinate, ...]
    points: tuple[DataPoint, ...]
    digitization_sha256: Maybe[str]
    """The content address of the :class:`EmbeddedFigureDigitization` whose
    record recovered this series from a figure -- i.e. the
    ``digitization_sha256`` computed by
    :func:`carmel.services.figure_digitization_record.compute_digitization_sha`.

    This is the figure lane's counterpart to
    :attr:`TableCellLocator.pdf_table_inventory_sha256`, and it exists to close
    the hole that lane's citation already closes one level down: a series whose
    ``source_form`` is :attr:`SourceForm.DIGITIZED` reports ``len(points)`` and
    nothing about how PARTIAL the recovery was, so without a citation it could
    be stored with no digitization record and no partialness at all -- exactly
    the gap :class:`EmbeddedFigureDigitization` was built to close, reappearing
    at series level. Whether a digitization record applies is a per-SERIES fact
    (one figure, one recovered curve), not a per-point one, so the citation
    lives HERE rather than on a locator the way the table lane's does.

    ``Maybe``-typed with NO default, for the same "no unreasoned absence"
    reasoning as :attr:`TableCellLocator.pdf_table_inventory_sha256`: every
    series must STATE which case it is in. WHICH absences are legal, and when a
    citation is mandatory, is NOT a property of this field -- a series does not
    know the envelope's ``figure_digitizations`` or ``source_graph`` -- so it is
    enforced at envelope level by
    :func:`_validate_series_digitization_citation` (V9), which requires a
    present citation for a ``DIGITIZED`` series and ``Absent(NOT_APPLICABLE)``
    for every other form.

    **A STATED CEILING, so a reader meets it here.** A resolvable citation
    proves the digitized coordinates were RECORDED and are UNALTERED -- the
    embedded record's canonical bytes hash to this address, its census balances,
    its coverage claim cannot contradict its own ledger, and the chain reaches
    the document's raw bytes. It proves NOTHING about whether those coordinates
    are TRUE: nothing here re-derives markers from the crop's pixels, so a
    verified record is not a validated measurement, and "the curve was traced
    accurately" is a claim this schema cannot make. See
    :class:`EmbeddedFigureDigitization` for the same limit stated on the record."""

    @field_validator("series_id")
    @classmethod
    def _validate_series_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="series_id")

    @field_validator("digitization_sha256")
    @classmethod
    def _validate_digitization_sha256_shape(cls, value: Maybe[str]) -> Maybe[str]:
        # fullmatch, never match: `$` also matches just BEFORE a trailing newline, so
        # match would let "a" * 64 + "\n" through -- same reasoning as
        # TableCellLocator._validate_inventory_sha256_shape.
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            raise ValueError(f"Series.digitization_sha256 {value!r} is not 64 lowercase hex characters")
        return value

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
                raise ValueError(f"Series(series_id={self.series_id!r}): duplicate axis_id {axis.axis_id!r} in axes")
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
            raise ValueError(f"Series(series_id={self.series_id!r}) must declare at least one coordinate axis")
        return self

    @model_validator(mode="after")
    def _validate_has_observation_axis(self) -> Series:
        """S4: a series must declare at least one ``OBSERVATION`` axis.

        A series with no observation axis records locations but nothing
        measured at them -- it would be a table of coordinates with no
        data, not a dataset.
        """
        if not any(axis.role == AxisRole.OBSERVATION for axis in self.axes):
            raise ValueError(f"Series(series_id={self.series_id!r}) must declare at least one observation axis")
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
                    f"Series(series_id={self.series_id!r}): duplicate point_id {point.point_id!r} in points"
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


class GroundedScalarClaim(BaseModel):
    """ONE scalar fact a source states about an experiment, label and all --
    "the initial pressure was 1 atm", "the shock-tube bore is 5.0 cm".

    Deliberately NOT a series element. It has no axis, no point, and no
    siblings; it is a standalone claim, and it exists because conditions have
    no legal home in this schema otherwise. A :class:`Series` requires at
    least one ``COORDINATE`` axis (S3), at least one ``OBSERVATION`` axis
    (S4), and at least one point (S7), and :attr:`DatasetEnvelope.series`
    carries ``MinLen(1)`` -- so a constants-only series and a series-free
    envelope are BOTH unrepresentable, and routing a standalone condition
    into :attr:`Series.constants` is impossible without weakening three
    invariants that are CORRECT for real series. Those invariants stay; this
    type is the home instead.

    ``label_raw``/``label_ref`` are the load-bearing fields, not decoration.
    Proving a number is an exact located substring of the source can never
    prove what that number MEANS, and for a standalone scalar the meaning
    lives entirely in the surrounding prose: ``"1 atm"`` is a pressure only
    because a sentence nearby called it one. So the label carries its own
    :class:`SourceRef`, independent of the value's -- exactly the split
    :class:`MeasuredValue` makes between ``value_ref`` and ``unit_ref``, for
    exactly the same reason (a single ref can "verify" the number while the
    label silently came from somewhere else entirely).

    There is deliberately NO ``quantity_kind`` field of its own, unlike
    :class:`AxisDeclaration`. An axis declares a quantity separately from the
    per-point values that must match it (S14), because one axis governs many
    points; a scalar claim's declaration and value co-locate, so a second copy
    would be one fact stored twice in a content-addressed payload, and every
    way the two could disagree would be an error class the duplication itself
    created. Read it from ``value.quantity_kind``.

    Also deliberately absent: ``value_origin`` and ``source_form``. Both are
    facts about a whole extracted set (was this an experiment or a
    simulation; was it read from a table, a figure, or prose), not about an
    individual scalar, and stamping them per claim would let two claims from
    the same reported run disagree with no way to arbitrate -- the same
    reasoning that puts ``glyph_health`` on :class:`SourceNode` rather than on
    every :class:`MeasuredValue`.

    **What this type does NOT prove, stated plainly rather than implied:** it
    proves a label and a number were each LOCATED in a source. It does not
    prove the label describes the number. That is a semantic relation between
    two spans, and no element model -- holding no document, and no text --
    can check it. A claim whose label reads "laminar burning velocity" over a
    value in atm is constructible here and must be caught by the prose-local
    scalar rule in the extraction gate, which has the text this model does
    not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    label_raw: str = Field(min_length=1)
    """The name the SOURCE gives this quantity, verbatim (e.g. ``"initial
    pressure"``, ``"P1"``, ``"bore"``) -- never a normalized or invented
    label."""
    label_ref: SourceRef
    """Provenance for the LABEL, independent of the value's refs. Required and
    never :class:`Absent`: an ungrounded label is not a weaker claim to record
    honestly, it is no claim at all."""
    value: MeasuredValue
    """The number and its unit. Plain, never ``Maybe`` -- mirroring
    :class:`Coordinate` rather than :class:`Observation`. An observation may
    honestly be absent (a paper plotted a point it never tabulated), but a
    condition the source never stated is not a condition with a missing
    number; it is simply not a claim, and must not occupy a ``claim_id``."""
    uncertainty: Maybe[Uncertainty]

    @field_validator("claim_id")
    @classmethod
    def _validate_claim_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="claim_id")

    @model_validator(mode="after")
    def _validate_uncertainty_bound_quantity(self) -> GroundedScalarClaim:
        _validate_uncertainty_bound_quantity_kind(
            uncertainty=self.uncertainty,
            value=self.value,
            where=f"GroundedScalarClaim(claim_id={self.claim_id!r})",
        )
        return self


class GroundedCategoricalClaim(BaseModel):
    """ONE condition a source states as a NAME rather than a number -- "the
    diluent is CO2", "the bath gas is argon", "the reactor is a heat-flux
    burner", "the tubing is Teflon".

    Sibling to :class:`GroundedScalarClaim`, not a variant of it: the two
    share a shape -- a labeled claim with two independently-grounded halves
    -- because that is the right shape for a labeled claim, not because one
    specializes the other. No base class joins them, deliberately. Envelope
    identity here is a hand-written projection naming its keys, so a subclass
    inheriting one would address two different payloads identically, and in a
    write-once store that collision is permanent; the seven shared envelope
    validators above are shared by CALL for the same reason.

    This is not a completeness nicety. A probe of the eight real corpus
    papers found the diluent IDENTITY -- N2 vs CO2 vs He vs Ar -- is the
    independent variable in 4 of 8, with the numeric dilution fraction
    secondary to it. A conditions model that holds only numbers would drop
    the actual subject of half that corpus, or worse, launder it: record a
    fraction and silently lose which gas it was a fraction OF.

    ``label_raw``/``label_ref`` and ``token_raw``/``token_ref`` carry
    INDEPENDENT refs for exactly the reason :class:`MeasuredValue` splits
    ``value_ref`` from ``unit_ref``: proving the string ``"CO2"`` sits at
    some located offset can never prove it is the diluent, rather than a
    product species, a cylinder label printed in the same figure, or another
    lab's experiment quoted in the discussion. Only an independent ref on
    the label closes that gap, and even then only as far as "these two spans
    exist" -- see below for what it still does not prove.

    There is deliberately NO normalized/resolved token field. A
    :class:`MeasuredValue` may normalize a printed unit because an auditable
    conversion table with a content hash (``conversion_table_sha256``) backs
    that normalization end to end. Nothing comparable backs chemical
    identity: there is no content-addressed table mapping the printed token
    ``"CO2"`` to a canonical species identity, and inventing one here would
    fabricate exactly the kind of unearned authority this project exists to
    refuse. This codebase already has the right home for that problem --
    :class:`CompositionResolution`, which exists precisely because ``"air"``
    is an unresolved token in roughly 5 of the 8 corpus papers. Resolving a
    printed token to a real identity is a separate, auditable step with its
    own evidence trail; this model is the evidence atom that records only
    what was PRINTED, and where.

    **What this type does NOT prove, stated plainly rather than implied:**
    that ``token_raw`` names a real chemical species or apparatus, that
    ``label_raw`` actually describes ``token_raw``, or that either belongs to
    THIS experiment rather than one the source merely cites. All three need
    the extraction gate, which holds the document this model does not; a
    label and a token located anywhere in a source, in any relation to each
    other, still construct here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    label_raw: str = Field(min_length=1)
    """The name the SOURCE gives this slot, verbatim (e.g. ``"diluent"``,
    ``"bath gas"``, ``"carrier gas"``, ``"reactor"``) -- never a normalized
    or invented label."""
    label_ref: SourceRef
    """Provenance for the LABEL, independent of the token's. Required and
    never :class:`Absent`: an ungrounded label is not a weaker claim to
    record honestly, it is no claim at all."""
    token_raw: str = Field(min_length=1)
    """The categorical value as PRINTED in the source (e.g. ``"CO2"``,
    ``"Ar"``, ``"heat flux method"``, ``"Teflon"``) -- verbatim, never
    normalized, resolved, or expanded. See class docstring for why no
    normalized counterpart exists."""
    token_ref: SourceRef
    """Provenance for the TOKEN, independent of the label's -- same split,
    same reason, as ``label_ref``. Required and never :class:`Absent`, for
    the same reason as ``label_ref``: an ungrounded token is no claim at
    all."""

    @field_validator("claim_id")
    @classmethod
    def _validate_claim_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="claim_id")


class UnextractedReason(StrEnum):
    """Why a located condition statement did NOT become a
    :class:`GroundedScalarClaim` or :class:`GroundedCategoricalClaim`.

    This is a THIRD vocabulary, distinct from and never to be conflated with
    this project's two existing failure vocabularies. ``FAILED`` means a
    check RAN and the artifact was lost. ``UNVERIFIABLE`` means the check
    could NOT run at all. Neither applies here: a member of this enum means
    the statement was located SUCCESSFULLY -- the extractor found it, and can
    say exactly where -- and is simply outside what the claim models can
    currently represent. A scope boundary, not a defect.

    ``AbsenceReason`` was deliberately NOT reused for this. ``AbsenceReason``
    answers "why is this FIELD missing" -- a fact about a field on an
    already-existing record. This enum answers "why did a LOCATED STATEMENT
    not become a claim" -- a fact about an extraction decision made before
    any record with fields even existed. Overloading one enum to answer both
    questions would blur two different kinds of fact into one vocabulary,
    exactly the conflation ``AbsenceReason``'s own docstring warns against
    for its own members.
    """

    MULTI_VALUED_SWEEP = "multi_valued_sweep"
    """The statement gives several discrete values for one quantity (a list,
    e.g. "0.6, 1.2 and 2.0")."""

    VALUE_RANGE = "value_range"
    """The statement gives an interval rather than a value (e.g. "0.4 to
    5.0")."""

    ONE_SIDED_BOUND = "one_sided_bound"
    """The statement constrains the quantity with an inequality rather than
    stating it (e.g. "Re < 2000")."""

    QUALITATIVE_ONLY = "qualitative_only"
    """The statement names a condition with no number at all (e.g.
    "atmospheric pressure", "ambient temperature")."""

    COMPOSITE_VALUE = "composite_value"
    """The statement gives a tuple/ratio that loses meaning if split into
    scalars (e.g. a three-component blend ratio)."""

    ATTRIBUTION_UNCLEAR = "attribution_unclear"
    """A single value, located, but it could not be established that it
    belongs to THIS experiment rather than to a simulation or to a cited
    third party.

    The odd member out: the other five are shape facts about the statement
    itself (is it a list, a range, a bound, ...); this one is instead a fact
    about the EXTRACTOR'S CONFIDENCE in attribution. Included because the
    probe found simulation conditions, cited third-party conditions, and
    own-experiment conditions are not separable by wording alone -- one paper
    in the corpus writes "the initial temperature was set to 298 K and
    initial pressure was fixed at 1 atm" about a CHEMKIN run, with numbers
    that coincide with its real experimental values stated pages earlier."""


class UnextractedConditionStatement(BaseModel):
    """A record that a condition statement WAS found and located in a source,
    and was deliberately NOT turned into a claim.

    Exists for coverage honesty. A probe of the eight real corpus papers
    found that ~70-80% of stated conditions are not single-valued: the
    equivalence ratio is a range or a list in 8 of 8 papers ("0.4 to 5.0";
    "0.6, 1.2 and 2.0"), pressure is in 7 of 8 ("1, 5 and 10 atm"),
    composition is in 8 of 8 -- and some conditions carry no number at all
    ("atmospheric pressure", "ambient temperature", in 8 of 8 papers, and in
    one paper that qualitative statement is the ONLY mention of temperature
    anywhere). Carmel extracts only genuinely single-valued conditions today.
    Without this type, a dataset that holds three apparatus dimensions and no
    pressure claim is indistinguishable from one where a pressure sweep was
    seen and deliberately skipped -- the two look byte-identical, and any
    coverage number computed from claims alone silently overstates itself in
    exactly the cases where real coverage is worst.

    ``statement_ref`` is required and never :class:`Maybe`, for the same
    reason ``label_ref``/``value_ref`` are required on
    :class:`GroundedScalarClaim`: a refusal that does not say WHERE the
    refused statement lives is indistinguishable from a guess. This is what
    makes the record auditable -- a human, or a later extraction pass, can go
    look at the exact span this type declined to turn into a claim.

    Holds the statement's verbatim words (``statement_raw``) and their span
    (``statement_ref``), but NO PARSE of them -- no range endpoints, no list
    members, no inequality operator. Parsing a sweep into its endpoints is most
    of the actual work of SUPPORTING sweeps as claims, which is exactly what
    this narrow slice defers; a half-parsed record here would be scope creep
    wearing a refusal's clothes. The words are a string the source really prints
    and an independent reader can re-derive; the parse is the interpretation this
    type declines to make. This type holds the span and the words, not the
    parse.

    **What this type does NOT prove, stated plainly rather than implied:**
    that ``reason`` is the CORRECT classification of the statement. This
    model records a decision the extractor already made; it cannot check
    that decision itself, because checking would require the document, and
    an element model holds no document. A record whose ``reason`` is
    ``VALUE_RANGE`` but whose statement is actually a single value still
    constructs here -- the real check for that lives upstream, in whatever
    process assigned the reason in the first place.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement_id: str
    label_raw: str = Field(min_length=1)
    """The name the SOURCE gives the quantity, verbatim -- never a normalized
    or invented label."""
    label_ref: SourceRef
    """Provenance for the LABEL. Plain, never :class:`Maybe`: an ungrounded
    label is not a weaker record to keep honestly, it is no record at all."""
    statement_raw: str = Field(min_length=1)
    """The refused statement's OWN words, verbatim -- the document's text at
    ``statement_ref``, exactly as :attr:`label_raw` carries the label's text
    beside :attr:`label_ref`. This is the string a citation was resolved
    against, never a repaired or re-spelled form: the producer grounds it with
    :func:`~carmel.services.dataset_producer.ground_quote`, which guarantees
    ``text[start:end] == statement_raw`` for a char span and validates whole-cell
    equality for a table cell, so an independent reader re-derives exactly this
    string from the source. It is NOT a parse: no range endpoints, no list
    members, no operator -- just the words (``"0.6-1.0"``, ``"1, 5 and 10 atm"``)
    the extractor declined to flatten. Its purpose is to let replay COMPARE the
    refused text against the source rather than merely locate it: without a
    recorded counterpart beside ``statement_ref``, the text was verified once by
    the producer and never again by anything independent."""
    statement_ref: SourceRef
    """Provenance for the LOCATED STATEMENT itself. Plain, never
    :class:`Maybe`: this is what makes the record auditable -- a refusal
    that doesn't say WHERE is indistinguishable from a guess. Paired with
    :attr:`statement_raw` (the words read at this span), so replay checks the
    text as a grounding pair, not merely as a located-but-unexplained span."""
    reason: UnextractedReason
    quantity_kind: Maybe[QuantityKind]
    """What quantity the statement appears to concern, where the extractor
    could tell. Honestly ``Absent`` when it could not -- a statement whose
    quantity is itself unclear must still be recordable, not silently
    dropped."""

    @field_validator("statement_id")
    @classmethod
    def _validate_statement_id(cls, value: str) -> str:
        return _require_identifier(value, field_name="statement_id")


class SubjectRefusalReason(StrEnum):
    """Why a condition set's SUBJECT could not be resolved to a device-class
    declaration -- the refusal half of the subject sum on
    :class:`ConditionSetEnvelope`.

    A FOURTH vocabulary, sibling to :class:`UnextractedReason` and, like it,
    never to be conflated with the project's two failure vocabularies
    (``FAILED``: a check ran and the artifact was lost; ``UNVERIFIABLE``: the
    check could not run). A member here means the extractor read the source
    successfully and is reporting, with a located span as evidence, that the
    source itself does not support naming a subject. A scope boundary of the
    SOURCE, not a defect of the extraction.

    :class:`UnextractedReason` was deliberately not reused, even though both
    enums describe extraction refusals: that enum classifies why one located
    CONDITION STATEMENT did not become a claim; this one classifies why the
    envelope-level SUBJECT -- a fact about the whole condition set, not
    about any one statement -- could not be declared. Overloading one enum
    to answer both would blur a per-statement fact into a per-envelope one,
    exactly the vocabulary conflation ``AbsenceReason``'s own docstring
    warns against.
    """

    MULTIPLE_INDISTINGUISHABLE_DEVICES = "multiple_indistinguishable_devices"
    """Two or more physical devices that the source never names apart. The
    motivating case, found in a survey of eight real combustion-kinetics
    papers: one paper's "bomb" is two physically different vessels the text
    calls only "The first vessel" and "The other vessel", with one
    conditions table covering both under one caption -- and one of the two
    vessels physically cannot reach the highest temperature in that table.
    A required device NAME there would produce a record that LOOKS grounded
    and is wrong; this member is the honest alternative."""

    ASSIGNMENT_DEPENDS_ON_RESULT = "assignment_depends_on_result"
    """Which device was used is decided by the MEASURED value (e.g. "runs
    above 5 atm used the second vessel"): the subject of a condition row
    cannot be resolved without already knowing the measurement outcome, so
    any fixed declaration would be a guess wearing a name."""

    DEVICE_UNNAMED = "device_unnamed"
    """The source never names a device class at all -- conditions are stated
    with no apparatus noun anywhere to ground a declaration against."""

    ATTRIBUTION_UNCLEAR = "attribution_unclear"
    """It cannot be established WHOSE device the conditions describe -- this
    paper's own, a cited third party's, or a simulated one. Deliberately the
    same spelling as :attr:`UnextractedReason.ATTRIBUTION_UNCLEAR`, and
    deliberately a distinct member of a distinct enum: the parallel naming
    marks the same epistemic situation recurring at two different scopes
    (one statement there, the whole subject here), while keeping the two
    vocabularies un-mixable in typed code."""


class DeviceClassDeclaration(BaseModel):
    """The grounded declaration half of the subject sum on
    :class:`ConditionSetEnvelope`: the source names a device CLASS, and this
    record carries that name verbatim with the span it was read from.

    The field is named ``label_raw`` -- a CLASS label, never a unique
    physical apparatus -- and the class name says "class" out loud for the
    same reason: class-level granularity is the STRONGEST subject identity
    this schema is willing to assert. In a survey of eight real
    combustion-kinetics papers, one paper's single "bomb" was two physically
    different vessels the text never names apart ("The first vessel" / "The
    other vessel"), with one conditions table covering both; an
    extractor-assigned per-vessel identifier there would be a fabricated
    identity wearing an identifier, and a "required apparatus name" would
    make the record look grounded while attributing conditions to a vessel
    that physically cannot produce them. A field name that carried
    apparatus-identity semantics would invite downstream code to launder the
    class label into a device id; these names are chosen so that misreading
    requires ignoring the words, not just skipping a docstring.

    **What this type does NOT prove, stated plainly rather than implied:**
    that the labeled class is the device that produced any claim in the
    envelope, or that only one physical instance of the class exists. It
    proves one thing: this class NAME was located at this span.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_raw: str = Field(min_length=1)
    """The device-class name as the SOURCE prints it (e.g. ``"heat flux
    burner"``, ``"spherical combustion vessel"``, ``"shock tube"``) --
    verbatim, never normalized, resolved, or invented."""
    label_ref: SourceRef
    """Provenance for the class label. Plain, never :class:`Maybe`: an
    ungrounded subject declaration is not a weaker declaration to record
    honestly, it is no declaration at all -- that case must be an
    :class:`UnresolvedSubject` instead."""


class UnresolvedSubject(BaseModel):
    """The refusal half of the subject sum on :class:`ConditionSetEnvelope`:
    an explicit, grounded statement that the subject cannot be resolved even
    at device-class granularity, and why.

    This is a first-class record, not a missing field: a ``Maybe`` subject
    (or an optional one) would make "nobody resolved the subject yet" and
    "the source makes the subject unresolvable" byte-identical, and the
    second is a fact about the SOURCE that downstream consumers must be able
    to see and audit. ``reason_ref`` is required for the same reason
    ``statement_ref`` is on :class:`UnextractedConditionStatement`: a
    refusal that does not say WHERE the evidence for refusing lives is
    indistinguishable from a guess.

    **What this type does NOT prove, stated plainly rather than implied:**
    that ``reason`` is the CORRECT classification, or that the span behind
    ``reason_ref`` really shows what the reason claims. This model records a
    decision the extractor already made; checking it needs the document, and
    an element model holds no document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: SubjectRefusalReason
    reason_ref: SourceRef
    """The located span that SHOWS why the subject is unresolvable (e.g. the
    sentence introducing "The other vessel"). Plain, never
    :class:`Maybe`."""


class ConditionAttribution(StrEnum):
    """Whose conditions a :class:`ConditionSetEnvelope` asserts these are.

    An extractor ASSERTION with an auditable span
    (:attr:`ConditionSetEnvelope.attribution_ref`), never a verified fact:
    nothing in this schema checks it, and nothing can -- the probe of eight
    real corpus papers found simulation conditions, cited third-party
    conditions, and own-experiment conditions are not separable by wording
    alone (one paper writes "the initial temperature was set to 298 K and
    initial pressure was fixed at 1 atm" about a CHEMKIN run, with numbers
    that coincide with its real experimental values stated pages earlier).
    """

    OWN_EXPERIMENT = "own_experiment"
    """Asserted to describe an experiment the source's authors ran
    themselves."""

    CITED_THIRD_PARTY = "cited_third_party"
    """Asserted to describe a cited, third-party experiment reproduced in
    this source's text or tables."""

    SIMULATION = "simulation"
    """Asserted to describe a simulation (e.g. a CHEMKIN run), not a
    physical measurement."""


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
        # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
        # trailing newline, so match would let "a" * 64 + "\n" through.
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"EmbeddedConversionTable.sha256 {value!r} is not 64 lowercase hex characters")
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
                f"EmbeddedConversionTable(sha256={self.sha256!r}): canonical_json does not parse as JSON: {exc}"
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


class EmbeddedTableInventory(BaseModel):
    """One PDF cell inventory's own canonical record JSON, embedded VERBATIM
    in an envelope so a consumer holding only the envelope's bytes can see the
    grid a :class:`TableCellLocator` indexes into -- without the evidence
    store, and without re-deriving anything.

    Same rationale and same ``str``-not-``dict`` decision as
    :class:`EmbeddedConversionTable` (see its docstring: ``frozen=True`` is a
    claim about attribute REASSIGNMENT, and a ``dict`` field could be mutated
    in place after validation with nothing here to notice).

    **The store is a CACHE, never replay's source of truth.** A replayer reads
    the record from HERE and re-derives against the document's raw bytes; if
    it read the store instead, an envelope's meaning would depend on a
    directory that may have been pruned, repopulated, or never written on the
    machine doing the replay.

    SCOPE OF WHAT VALIDATION HERE PROVES -- read before trusting an inventory.
    T1 below is materially WEAKER than :class:`EmbeddedConversionTable`'s T1,
    and the difference is not an oversight. A conversion table can be
    RECONSTRUCTED (``ConversionTable.from_identity_payload`` re-runs every
    ``__post_init__`` invariant), so its embedded bytes are checked against
    the type's own rules. There is no equivalent reconstruction for an
    inventory: nothing rebuilds a ``CellInventory`` from a payload, because
    the only thing that could establish a grid is DERIVING it from the PDF
    again. So T1 proves canonical self-coherence, self-addressing, and
    REPLAY-READABILITY -- these bytes are the canonical rendering, they hash to
    the address they claim, they name the document they claim, they carry no
    refusal, they have exactly the top-level shape of a record of their
    declared ``payload_version``, and their cell ordinals are integers -- and
    NOTHING about whether the grid corresponds to any real table. A fabricated
    payload asserting a plausible 4x3 grid over a document that has none passes
    every check in this class.

    Read that limit as strictly as it is written. In particular ``raw_sha256``
    matching a node's ``sha256`` proves the author NAMED that document, never
    that a PDF parser ever ran on it -- the entire payload is author-controlled.
    An earlier version of :func:`_validate_table_cell_inventory_citation`
    admitted SI members on exactly that mistaken inference; see its docstring.

    What T1 DOES buy beyond self-consistency is that the record is not
    unverifiable by construction: a payload with a stray key can never
    reproduce, and one without ``footprint`` cannot be read by the verifier at
    all, so both would be citations no future replay could confirm or deny.
    Rejecting them is the last thing a reader holding no document can check.

    What closes the remaining gap is
    :func:`carmel.services.pdf_table_record.verify_inventory_record`, which
    re-derives from the raw PDF bytes. Schema validation deliberately does NOT
    call it, read ``raw.bin``, or touch the store: a schema that performs I/O
    makes an envelope's validity depend on the filesystem it is validated on.
    An ``EmbeddedTableInventory`` that has never been through that verifier is
    a well-formed CITATION, not evidence that the grid is real.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inventory_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- the address a
    :attr:`TableCellLocator.pdf_table_inventory_sha256` cites, and the sha256
    of ``canonical_json``'s bytes (T1)."""

    raw_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- the document these bytes were derived
    from. Declared HERE, redundantly with the payload's own ``raw_sha256``
    (T1 requires them equal), so the envelope-level join to
    :attr:`SourceNode.sha256` can be made without parsing JSON in the
    validator that performs it."""

    canonical_json: str = Field(min_length=1, max_length=_MAX_EMBEDDED_CANONICAL_JSON_LENGTH)
    """The inventory record's canonical JSON, verbatim, as a str.

    Bounded by ``_MAX_EMBEDDED_CANONICAL_JSON_LENGTH`` for the same
    resource-exhaustion reason as :attr:`EmbeddedConversionTable.canonical_json`
    -- and it binds harder here, because an inventory record carries one entry
    per CELL. A record too large to embed is a rejected envelope, never a
    reason to embed a projection instead: a projection does not hash to
    ``inventory_sha256``, so it could not be the artifact being cited."""

    _cell_index: frozenset[tuple[int, int]] = PrivateAttr(default=frozenset())
    """Every ``(row, col)`` the record's grid contains, built by T1.

    Private and derived, never an input: it is not a field, so it takes no part
    in serialization, in equality, or in the address. The default is empty
    rather than None because the only way to hold an instance whose T1 did not
    run is to have bypassed construction entirely -- and an empty index answers
    every ``has_cell`` with False, which is the fail-closed direction."""

    _cell_text_index: dict[tuple[int, int], str] = PrivateAttr(default_factory=dict)
    """Maps each ``(row, col)`` to the record's OWN text for that cell, built by
    T1 in the same loop that built ``_cell_index`` so the two cannot disagree.

    Populated only for cells whose ``"text"`` is a genuine ``str`` -- a cell that
    omits it, or carries a non-string, is left out, so :meth:`cell_text` answers
    ``None`` for it (the fail-closed direction). Private and derived for the same
    reason as ``_cell_index``: it takes no part in serialization, equality, or the
    content address. It exists so a PRODUCER can enforce that a cell citation it
    emits names a cell whose whole text equals the whole grounded string -- the
    settled exact-equality matching contract -- WITHOUT re-parsing ``canonical_json``
    on every lookup."""

    @field_validator("inventory_sha256", "raw_sha256")
    @classmethod
    def _validate_sha256_shape(cls, value: str, info: ValidationInfo) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"EmbeddedTableInventory.{info.field_name} {value!r} is not 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _validate_canonical_json_coheres(self) -> EmbeddedTableInventory:
        """T1: ``canonical_json`` must be the canonical rendering of a record
        that addresses to ``inventory_sha256``, names ``raw_sha256``, is of a
        payload version this code can read, and carries no refusal.

        Five steps, each with its own marker phrase so a test can tell which
        one fired. See the class docstring for what these five do NOT prove.
        """
        try:
            parsed = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            # Same broadened except, for the same reason, as
            # EmbeddedConversionTable's: canonical_json is untrusted input from a stored
            # file, and a short-but-deeply-nested payload blows the interpreter's own
            # stack from inside json.loads before this validator can react.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json does "
                f"not parse as JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json is not "
                f"a JSON object, it decodes to {type(parsed).__name__}"
            )
        recanonicalized = canonical_json_bytes(parsed)
        if recanonicalized != self.canonical_json.encode("utf-8"):
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json is not "
                "the canonical rendering of what it decodes to -- re-serializing through "
                "canonical_json_bytes produced different bytes"
            )
        # The address is over the canonical bytes, exactly as compute_inventory_sha
        # defines it (sha256 of inventory_record_bytes). Recomputed from the
        # RE-canonicalized bytes, which the check above has already pinned equal to the
        # stored ones -- so this cannot be satisfied by honestly hashing arbitrary bytes
        # against themselves in some other rendering.
        actual = hashlib.sha256(recanonicalized).hexdigest()
        if actual != self.inventory_sha256:
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json's bytes "
                f"hash to {actual!r}, so this record does not live at the address it claims"
            )
        version = parsed.get("payload_version")
        if version != INVENTORY_PAYLOAD_VERSION:
            # A reader that does not know a shape must not guess at it -- the same rule
            # the record's own INVENTORY_PAYLOAD_VERSION comment states. "I cannot read
            # this" and "this does not reproduce" are different facts.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): payload_version "
                f"{version!r} is not the readable version {INVENTORY_PAYLOAD_VERSION!r}"
            )
        # Having pinned the version, the shape that version names is knowable EXACTLY. A
        # record with an extra key can never reproduce (verify_inventory_record rebuilds the
        # payload and compares canonical bytes, and the rebuilt one will not carry it); a
        # record without `footprint` cannot even be READ by the verifier, which returns
        # PAYLOAD_UNREADABLE before it looks at the document at all. Either way the citation
        # would be unverifiable BY CONSTRUCTION -- a claim nothing can ever confirm or deny,
        # which is precisely what this schema exists to make impossible.
        keys = set(parsed)
        if keys != set(INVENTORY_PAYLOAD_KEYS):
            unexpected = sorted(keys - INVENTORY_PAYLOAD_KEYS)
            missing = sorted(INVENTORY_PAYLOAD_KEYS - keys)
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record is not "
                f"the shape of a version-{INVENTORY_PAYLOAD_VERSION} inventory (unexpected keys "
                f"{unexpected!r}, missing keys {missing!r}), so it could never be replayed against the "
                "document it names"
            )
        unreadable = footprint_unreadable_reason(parsed)
        if unreadable is not None:
            # The key-set check above proves the 'footprint' key is PRESENT. Presence was
            # only ever a proxy for the property that matters: that the verifier can read
            # the box back out and re-derive against it. `footprint={}`, a footprint that
            # is a list, a page that is a string, a coordinate that is not float.fromhex
            # readable -- each satisfies the key set and each makes the record permanently
            # unfalsifiable, since verify_inventory_record can only answer
            # PAYLOAD_UNREADABLE. That is a THIRD outcome, not a failure to reproduce, and
            # a citation nothing can ever confirm or deny is exactly what this schema
            # exists to make impossible. Asked of the record module, which performs the
            # same read, so the schema never carries its own idea of what a footprint is.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record's "
                f"footprint cannot be read back, so replay could only ever answer "
                f"PAYLOAD_UNREADABLE and the citation could never be checked: {unreadable}"
            )
        declared_raw = parsed.get("raw_sha256")
        if declared_raw != self.raw_sha256:
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record names "
                f"document {declared_raw!r}, not the declared raw_sha256 {self.raw_sha256!r}"
            )
        if not isinstance(parsed["refusals"], list):
            # refusal_reasons_of reads `payload.get("refusals") or ()`, so ANY falsy non-list
            # -- 0, "", {} -- reads as refusal-free without a single refusal ever having been
            # ruled out. The key-set check above already guarantees the key EXISTS; this is
            # about its type. "This record does not say" and "this record says none" are
            # different facts, and only the second can clear a citation.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record's "
                f"'refusals' is {type(parsed['refusals']).__name__}, not a list, so it never states whether "
                "the derivation refused -- silence is not a refusal-free claim"
            )
        try:
            refusals = refusal_reasons_of(parsed)
        except (ValueError, TypeError, KeyError) as exc:
            # canonical_json is untrusted, and refusal_reasons_of reaches entry["reason"]
            # unguarded: it raises a BARE TypeError on refusals=["x"] and a BARE KeyError on
            # refusals=[{}]. Either would leave this validator as that exception rather than
            # a pydantic ValidationError, crashing a caller that correctly catches only
            # ValidationError. Every way that indexing can fail is caught here, because the
            # helper makes no promise about which it raises.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record's "
                f"'refusals' cannot be read, so it cannot be shown refusal-free: {exc!r}"
            ) from exc
        if refusals:
            # An inventory that refused is a legitimate record to STORE -- it is the
            # honest outcome for most real tables -- but it is by construction not a
            # grid, so it can never justify a cell. Embedding one is only ever a
            # citation (T2 makes embedded and cited the same set), so refusing it here
            # closes the case structurally rather than leaving it to a caller.
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record carries "
                f"refusal(s) {sorted(reason.value for reason in refusals)!r}, and a refused derivation "
                "defines no grid a table cell could be located in"
            )
        # The ordinals a citation is checked against must be ORDINALS. JSON has no integer
        # type distinct from bool, and Python's `True == 1`, so an unchecked payload can
        # satisfy a lookup for (1, 0) with `{"row": true, "col": false}` -- a grid that
        # contains a cell it does not contain. Checked once here rather than in has_cell so
        # the guarantee belongs to the stored bytes, not to whoever happens to read them.
        cells = parsed["cells"]
        if not isinstance(cells, list):
            raise ValueError(
                f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): the record's 'cells' "
                f"is {type(cells).__name__}, not a list, so it describes no grid"
            )
        index: set[tuple[int, int]] = set()
        text_index: dict[tuple[int, int], str] = {}
        for position, cell in enumerate(cells):
            if not isinstance(cell, dict):
                raise ValueError(
                    f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): cells[{position}] "
                    f"is {type(cell).__name__}, not an object"
                )
            for axis in ("row", "col"):
                ordinal = cell.get(axis)
                # `isinstance(True, int)` is True, so bool must be excluded explicitly.
                if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                    raise ValueError(
                        f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): "
                        f"cells[{position}][{axis!r}] is {ordinal!r}, which is not an integer ordinal"
                    )
            position_key = (cell["row"], cell["col"])
            if position_key in index:
                # A coordinate that appears twice is not a grid. The two entries can carry
                # DIFFERENT text -- reproduced: a record saying (0, 0) is both `Fuel` and
                # `9999` was accepted, and `has_cell` answered `True`, because a set
                # membership bit cannot express "present, but the record does not agree
                # with itself about what is here". A citation resolving to that coordinate
                # would then be checkable and meaningless at once, which is precisely the
                # pair this schema exists to keep apart. Every other collection here
                # already refuses duplicates (`Series.points`, `Series.axes`,
                # `Composition.components`); `cells` was the one that did not.
                #
                # ALL duplicates are refused, not just disagreeing ones: `build_inventory`
                # emits one cell per (row, col) by construction, so a repeat is never
                # something a real derivation produced, and "agrees with itself" is a
                # weaker property to have to define than "appears once".
                raise ValueError(
                    f"EmbeddedTableInventory(inventory_sha256={self.inventory_sha256!r}): cells[{position}] "
                    f"repeats the coordinate (row={cell['row']}, col={cell['col']}), so the record does not "
                    "define one value at that position"
                )
            index.add(position_key)
            # The record's own text for this cell, captured in the SAME loop so it can
            # never be read from different bytes than the ordinals were. Only a genuine
            # str is recorded: a cell that omits "text" or carries a non-string is left
            # out, and `cell_text` answers None for it -- the fail-closed direction, which
            # a producer reads as "cannot confirm the exact-equality contract" and refuses.
            cell_text = cell.get("text")
            if isinstance(cell_text, str):
                text_index[position_key] = cell_text
        # Built HERE, in the loop that just proved every ordinal is an integer and unique,
        # so the index and the validation cannot be looking at different bytes. `has_cell`
        # then answers from it instead of re-parsing: a validator's work done once, rather
        # than once per lookup.
        self._cell_index = frozenset(index)
        self._cell_text_index = text_index
        return self

    def has_cell(self, *, row: int, col: int) -> bool:
        """Whether this record's grid actually contains ``(row, col)``.

        Answers from the index T1 built while validating the same bytes, so a
        bool masquerading as an ordinal cannot satisfy a lookup and the cost
        does not grow with the number of refs consulting the record. An envelope
        may cite one inventory from many refs, and the payload is an entire
        table's grid -- re-parsing it per lookup made the check quadratic in
        exactly the input an attacker chooses.
        """
        return (row, col) in self._cell_index

    def cell_text(self, *, row: int, col: int) -> str | None:
        """This record's OWN text for ``(row, col)``, or ``None`` if it has none.

        Answers from the text index T1 built while validating the same bytes, so
        it cannot disagree with :meth:`has_cell` or read different bytes than the
        validator did. ``None`` means the record either does not contain the cell
        at all or contains it with no genuine string ``"text"`` -- both fail-closed:
        a producer enforcing the exact-equality matching contract (the whole cell
        text must equal the whole grounded string) reads ``None`` as "cannot
        confirm" and REFUSES rather than emitting an unverifiable citation.

        Note the asymmetry with :meth:`has_cell`, and that it is deliberate: a cell
        can be PRESENT (``has_cell`` True) yet have no readable text here. This
        schema class never re-derives a grid from a document -- see "SCOPE OF WHAT
        VALIDATION HERE PROVES" -- so this returns the record's own claimed text and
        proves nothing about whether that text is what the table actually printed.
        Only :func:`~carmel.services.pdf_table_record.verify_inventory_record`,
        holding the real bytes, can establish that.

        A CONSUMER comparing a stored citation against this reads ``None`` the same
        fail-closed way, and must keep its two causes apart from a comparison that
        actually ran: the coordinate is absent from the grid (``has_cell`` False), or
        it is present but its ``"text"`` was not a string (``has_cell`` True). Both
        mean *"there is no verbatim string to compare against here"*, which replay
        surfaces as an INABILITY to compare -- never as a match, and never as a
        mismatch against ``""``.
        """
        return self._cell_text_index.get((row, col))

    def grid_cells(self) -> tuple[tuple[int, int, str | None], ...]:
        """Every ``(row, col, text)`` this record's grid contains, sorted by ``(row, col)``.

        ``text`` is the cell's own recorded string, or ``None`` where the record
        carries no genuine string for it -- the SAME fail-closed meaning as
        :meth:`cell_text`. Answers from the two indexes T1 built while validating
        these same bytes, so it can never disagree with :meth:`has_cell` or
        :meth:`cell_text`, and it re-parses nothing.

        This exists for a deterministic RESOLVER that must locate a column by its
        printed header text and then walk the grid's rows -- see
        :mod:`carmel.services.tabular_series_resolver`. It is read-only and, like
        everything else on this class, proves NOTHING about whether the grid is
        real: it returns the record's own claimed ordinals and text (see "SCOPE OF
        WHAT VALIDATION HERE PROVES"). Only
        :func:`~carmel.services.pdf_table_record.verify_inventory_record`, holding
        the raw bytes, can establish that.
        """
        return tuple((row, col, self._cell_text_index.get((row, col))) for row, col in sorted(self._cell_index))


class EmbeddedMemberTableInventory(BaseModel):
    """One tabulated supplementary member's inventory record, embedded VERBATIM so a
    consumer holding only the envelope's bytes can see the grid a ``MemberSheetKey``
    cell indexes into -- without the evidence store, and without re-deriving anything.

    The member-lane counterpart to :class:`EmbeddedTableInventory`. Its record is
    produced and re-derived by :mod:`carmel.services.member_table_record`, from the
    member's OWN bytes (a CSV/TSV member) rather than from PDF text fragments. It is a
    separate type on purpose: the PDF record is version-4 and PDF-shaped (``footprint``
    page geometry, ``column_bounds``, ``pypdf_version``), and a workbook sheet has no
    page geometry, so forcing a sheet through that type would mean weakening the very
    validators that make a PDF citation honest.

    SCOPE OF WHAT VALIDATION HERE PROVES -- read before trusting an inventory. Exactly
    as for :class:`EmbeddedTableInventory`, T1 proves canonical self-coherence,
    self-addressing, replay-readability (readable version, exact key set, matching
    ``member_sha256``, integer unique cell ordinals) and NOTHING about whether the grid
    corresponds to any real member. ``member_sha256`` matching a node's ``sha256``
    proves the author NAMED that member, never that a CSV parser ever ran on it -- the
    payload is author-controlled. What closes that gap is
    :func:`carmel.services.member_table_record.verify_member_inventory_record`, which
    re-reads the member's raw bytes; schema validation deliberately does NOT call it,
    so an envelope's validity never depends on the filesystem it is validated on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inventory_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- the sha256 of ``canonical_json``'s bytes (T1),
    i.e. :func:`~carmel.services.member_table_record.compute_member_inventory_sha`."""

    member_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- the member these bytes were derived from. Declared
    redundantly with the payload's own ``member_sha256`` (T1 requires them equal) so the
    join to :attr:`SourceNode.sha256` can be made without parsing JSON."""

    canonical_json: str = Field(min_length=1, max_length=_MAX_EMBEDDED_CANONICAL_JSON_LENGTH)
    """The inventory record's canonical JSON, verbatim, as a str. Bounded for the same
    resource-exhaustion reason as :attr:`EmbeddedTableInventory.canonical_json`, and it
    binds harder here too: the record carries one entry per CELL."""

    _cell_index: frozenset[tuple[int, int]] = PrivateAttr(default=frozenset())
    """Every ``(row, col)`` the record's grid contains, built by T1. Private and derived;
    an empty default answers every ``has_cell`` False, the fail-closed direction."""

    _cell_text_index: dict[tuple[int, int], str] = PrivateAttr(default_factory=dict)
    """Maps each ``(row, col)`` to the record's own text, built by T1 in the same loop as
    ``_cell_index`` so the two cannot disagree."""

    @field_validator("inventory_sha256", "member_sha256")
    @classmethod
    def _validate_sha256_shape(cls, value: str, info: ValidationInfo) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(
                f"EmbeddedMemberTableInventory.{info.field_name} {value!r} is not 64 lowercase hex characters"
            )
        return value

    @model_validator(mode="after")
    def _validate_canonical_json_coheres(self) -> EmbeddedMemberTableInventory:
        """T1: ``canonical_json`` must be the canonical rendering of a member inventory
        record that addresses to ``inventory_sha256``, names ``member_sha256``, is of a
        readable version, has exactly the record's key set, and whose cells are unique
        integer-addressed. Mirrors :meth:`EmbeddedTableInventory._validate_canonical_json_coheres`.
        """
        try:
            parsed = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json "
                f"does not parse as JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json is "
                f"not a JSON object, it decodes to {type(parsed).__name__}"
            )
        recanonicalized = canonical_json_bytes(parsed)
        if recanonicalized != self.canonical_json.encode("utf-8"):
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json is "
                "not the canonical rendering of what it decodes to -- re-serializing through "
                "canonical_json_bytes produced different bytes"
            )
        actual = hashlib.sha256(recanonicalized).hexdigest()
        if actual != self.inventory_sha256:
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): canonical_json's "
                f"bytes hash to {actual!r}, so this record does not live at the address it claims"
            )
        version = parsed.get("payload_version")
        if version != MEMBER_INVENTORY_PAYLOAD_VERSION:
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): payload_version "
                f"{version!r} is not the readable version {MEMBER_INVENTORY_PAYLOAD_VERSION!r}"
            )
        keys = set(parsed)
        if keys != set(MEMBER_INVENTORY_PAYLOAD_KEYS):
            unexpected = sorted(keys - MEMBER_INVENTORY_PAYLOAD_KEYS)
            missing = sorted(MEMBER_INVENTORY_PAYLOAD_KEYS - keys)
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): the record is not "
                f"the shape of a version-{MEMBER_INVENTORY_PAYLOAD_VERSION} inventory (unexpected keys "
                f"{unexpected!r}, missing keys {missing!r}), so it could never be replayed against the member "
                "it names"
            )
        declared_member = parsed.get("member_sha256")
        if declared_member != self.member_sha256:
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): the record names "
                f"member {declared_member!r}, not the declared member_sha256 {self.member_sha256!r}"
            )
        sheet_name = parsed.get("sheet_name")
        if not isinstance(sheet_name, str) or not sheet_name:
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): sheet_name "
                f"{sheet_name!r} is not a non-empty string, so the record names no sheet"
            )
        cells = parsed["cells"]
        if not isinstance(cells, list):
            raise ValueError(
                f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): the record's "
                f"'cells' is {type(cells).__name__}, not a list, so it describes no grid"
            )
        index: set[tuple[int, int]] = set()
        text_index: dict[tuple[int, int], str] = {}
        for position, cell in enumerate(cells):
            if not isinstance(cell, dict):
                raise ValueError(
                    f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): "
                    f"cells[{position}] is {type(cell).__name__}, not an object"
                )
            for axis in ("row", "col"):
                ordinal = cell.get(axis)
                # `isinstance(True, int)` is True, so bool must be excluded explicitly.
                if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                    raise ValueError(
                        f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): "
                        f"cells[{position}][{axis!r}] is {ordinal!r}, which is not an integer ordinal"
                    )
            position_key = (cell["row"], cell["col"])
            if position_key in index:
                raise ValueError(
                    f"EmbeddedMemberTableInventory(inventory_sha256={self.inventory_sha256!r}): "
                    f"cells[{position}] repeats the coordinate (row={cell['row']}, col={cell['col']}), so the "
                    "record does not define one value at that position"
                )
            index.add(position_key)
            cell_text = cell.get("text")
            if isinstance(cell_text, str):
                text_index[position_key] = cell_text
        self._cell_index = frozenset(index)
        self._cell_text_index = text_index
        return self

    def has_cell(self, *, row: int, col: int) -> bool:
        """Whether this record's grid actually contains ``(row, col)``. Answers from the
        index T1 built while validating the same bytes, so a bool masquerading as an
        ordinal cannot satisfy a lookup."""
        return (row, col) in self._cell_index

    def cell_text(self, *, row: int, col: int) -> str | None:
        """This record's OWN text for ``(row, col)``, or ``None`` if it has none. Proves
        nothing about whether that text is what the member printed -- only
        :func:`~carmel.services.member_table_record.verify_member_inventory_record`,
        holding the real bytes, can establish that."""
        return self._cell_text_index.get((row, col))


class EmbeddedFigureDigitization(BaseModel):
    """One figure digitization's own canonical record JSON, embedded VERBATIM so a consumer
    holding only these bytes can see how COMPLETE the digitized series actually is -- without
    the evidence store, and without inferring partialness from a point count.

    This is the figure lane's counterpart to :class:`EmbeddedTableInventory`, and it exists for
    a reason that lane does not have. A :class:`Series` recovered from a plot reports
    ``len(points)`` and nothing else about its own completeness, so a series that lost a marker
    -- straddling an axis boundary, occluded at a curve crossing, unplaceable against the axes
    -- reads exactly like one that lost none. The record these bytes carry states the
    difference instead: see :mod:`carmel.services.figure_digitization_record`.

    **Two orthogonal facts, and the reader gets both separately.**
    :attr:`coverage` answers "is anything missing"; :attr:`auditable` answers "could the
    instrument have told". :attr:`FigureCoverage.UNCHECKABLE` is what "no way to know" reads as,
    and it is deliberately NOT the same value as "nothing missing" -- collapsing them would put
    an unaudited series and a verified-whole one behind one indistinguishable answer, which is
    the failure the record was written to end.

    Same ``str``-not-``dict`` decision, and the same rationale, as
    :class:`EmbeddedConversionTable` and :class:`EmbeddedTableInventory` (see the former's
    docstring: ``frozen=True`` is a claim about attribute REASSIGNMENT, and a ``dict`` field
    could be mutated in place after validation with nothing here to notice).

    SCOPE OF WHAT VALIDATION HERE PROVES -- read before trusting a digitization. T1 (see
    :meth:`_reconstruct`) is STRONGER than :class:`EmbeddedTableInventory`'s, and weaker than it
    looks. Stronger, because a digitization record CAN be reconstructed:
    :meth:`~carmel.services.figure_digitization_record.FigureDigitization.from_payload` re-runs
    every construction invariant, so these embedded bytes are checked against the type's own
    rules the way :class:`EmbeddedConversionTable`'s are and the way an inventory's can never
    be. In particular a payload claiming ``coverage="complete"`` while carrying an omission, or
    one whose census does not balance against its recovered points, is refused HERE, on the
    bytes, with no producer involved -- and refused on every ACCESSOR READ as well as at
    construction, because both call :meth:`_reconstruct` and there is no cheaper read-time path
    that could accept a record construction rejects as INVALID. (Construction applies this
    class's field constraints on top of that, which a read does not re-apply; see
    :meth:`_validated_record` for the one asymmetry that creates and why it is in the safe
    direction.) That equality is not decoration: it is what makes "this class cannot state
    partialness incoherently" a property of reading one rather than only of building one.

    Weaker than it looks, because everything it proves is INTERNAL. Nothing in this class -- and
    nothing anywhere in this repository yet -- re-derives markers from the crop's pixels. So a
    valid record establishes that its coverage claim is consistent with its own ledger and its
    own census, and establishes NOTHING about whether a detector ever ran, whether ``detected``
    is the figure's true marker count, or whether the ledger names every marker that was
    dropped. A fabricated payload asserting a plausible 12-marker census over a figure that has
    none passes every check in this class, exactly as
    :class:`EmbeddedTableInventory`'s docstring warns for its own fabrications. Read
    ``raw_sha256`` matching a node's ``sha256`` as proof the author NAMED that document, never
    that any image was ever looked at.

    Weaker in a second, separate way: ``digitization_sha256`` ADDRESSES THE CLAIM AND NOT THE
    DIGITIZATION. The payload it hashes carries a recovered COUNT and no recovered coordinate,
    so two different digitizations of one figure agreeing on series id, crop, region, coverage,
    census and ledger share an address while holding different points. Two citations at one
    address mean two producers said the same thing about coverage; they never mean the same data
    was recovered. That collision is the citation surface's, on purpose: telling two such
    digitizations apart is what
    :func:`~carmel.services.figure_digitization_record.compute_digitization_identity` folds the
    recovered points, the calibration and the producer in to do, and it is a separate address that
    nothing embeds here.

    THE CROP'S REGION IS NOT VERIFIED, AND WILL NOT BE. A ``FIGURE_CROP`` node carries a required
    ``crop_region`` and its bytes are addressed by hash; replay confirms the crop's bytes hash to
    what the node claims. NOTHING confirms those bytes ARE that region -- or even that page -- of
    the document. A crop cut from the wrong part of the page, or the wrong page entirely, is
    hash-verified and wrong, and every figure claim resting on it reads as fully grounded. Proving
    otherwise needs a PDF renderer this repository does not have and is not getting; the check is
    ruled out of scope, so this is a knowingly accepted hole, not an oversight.

    BOTH UNVERIFIABLE FACTS STACK, and a reader who learns only one will still over-trust the
    claim. Neither the crop's region nor the coordinates recovered from it is independently
    checkable here: the region because nothing renders the page to confirm it, the coordinates
    because nothing re-derives markers from the pixels (see "Weaker than it looks" above). Attested
    points read off a crop of the wrong region, and hash-verified points read off the right region
    but never checked against the image, each reads as grounded and neither is.

    AN ENVELOPE FIELD, and cited. :attr:`DatasetEnvelope.figure_digitizations` embeds these
    verbatim, and a ``DIGITIZED`` :class:`Series` names the record that recovered it through
    :attr:`Series.digitization_sha256` -- the citation lives on the series, not on a locator,
    because whether a digitization applies is a per-SERIES fact. Four envelope validators stand
    behind that surface: :func:`_validate_series_digitization_citation` (V9) joins each series to
    its record; :func:`_validate_figure_digitizations_cover_cited` (FD1) makes the embedded set
    cover EXACTLY the cited addresses; :func:`_validate_figure_digitizations_no_duplicate_sha256`
    (FD2) forbids a repeated ``digitization_sha256``; and
    :func:`_validate_figure_digitizations_sorted` (FD3) fixes the order.

    This wiring was once described here as blocked on ``FIGURE_CROP`` crop addressing -- the worry
    being that an envelope citing this record would resolve ``figure_crop_node_id`` to a node that
    could not identify its own subject. That blocker is discharged: V9 resolves
    ``figure_crop_node_id`` to a node in the envelope's source graph, requires it to be of kind
    ``FIGURE_CROP``, and requires ``figure_crop_sha256`` to match that node's ``sha256`` -- the
    crop identifies its own subject, so the citation is usable.

    Four checks span two objects that never meet inside this class -- the record and the ``Series``
    it describes -- and so cannot be performed HERE. **All four are now enforced at envelope level
    by** :func:`_validate_series_digitization_citation` **(V9); a producer owes NONE of them:**

    - ``record.series_id == series.series_id`` -- the record is about THAT series.
    - ``record.recovered == len(series.points)`` -- the count the census balances against (D9) is
      the count the series actually has. Without this, D9 balances a number nothing else uses.
    - ``record.figure_crop_node_id`` resolves to a :class:`SourceNode` of kind
      ``FIGURE_CROP`` -- not to a page, a table or an SI member.
    - ``record.figure_crop_sha256`` equals that node's ``sha256`` -- the two halves of the crop's
      identity agree, so the record names one crop rather than one crop's id and another's bytes.

    V9 adds two joins beyond those four: every digitized point's value ref must target the SAME
    crop the record names, and ``record.raw_sha256`` must equal the crop's ROOT artifact's
    ``sha256`` rather than an intermediate one. What none of this reaches is whether the recovered
    coordinates are TRUE -- nothing here re-derives markers from the crop's pixels (see the
    "Weaker than it looks" paragraph above). What it leaves is the thing this ticket needed:
    partialness that the stored evidence STATES and cannot state incoherently, rather than
    partialness a reader infers from a number that looks the same either way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    digitization_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- this record's content address, and the sha256 of
    ``canonical_json``'s bytes (T1). The same addressing rule as
    :attr:`EmbeddedTableInventory.inventory_sha256`, computed by
    :func:`~carmel.services.figure_digitization_record.compute_digitization_sha`."""

    raw_sha256: str = Field(min_length=1)
    """64 lowercase hex characters -- the document the figure crop came from. Declared HERE,
    redundantly with the payload's own ``raw_sha256`` (T1 requires them equal), for the same
    reason :attr:`EmbeddedTableInventory.raw_sha256` is: so an envelope-level join to
    :attr:`SourceNode.sha256` can be made without parsing JSON in the validator doing it."""

    canonical_json: str = Field(min_length=1, max_length=_MAX_EMBEDDED_CANONICAL_JSON_LENGTH)
    """The digitization record's canonical JSON, verbatim, as a str.

    Bounded by ``_MAX_EMBEDDED_CANONICAL_JSON_LENGTH`` for the same resource-exhaustion reason
    as :attr:`EmbeddedTableInventory.canonical_json`, and it binds more loosely here: the
    payload holds one entry per OMITTED marker, not one per cell, and a figure with a
    megabyte's worth of omissions is not a figure anyone digitized. A record too large to embed
    is a rejected envelope, never a reason to embed a projection instead -- a projection does
    not hash to ``digitization_sha256``, so it could not be the artifact being cited."""

    # THIS CLASS HOLDS NO PRIVATE STATE AT ALL, and that is load-bearing rather than incidental.
    # Two earlier revisions cached something here -- first a digest of the public fields, then
    # the reconstructed record -- and each became the attack surface, because every route that
    # can write a public field past `frozen=True` (`object.__setattr__`, a `__dict__` write,
    # `model_construct`) can write a private one in the next line. There is nothing here to
    # write. `_validated_record` re-derives the record from `canonical_json` on every read, by
    # calling the same `_reconstruct` that construction calls.
    #
    # It costs more than caching did, and that cost was paid deliberately. Measured on this
    # machine against the final code, on a 94.8 KB payload carrying 700 omissions: see
    # `_reconstruct` for the figures. Reads really are slower than the revision this replaced,
    # and the reason to accept that is correctness, not a rounding error in the benchmark.

    @field_validator("digitization_sha256", "raw_sha256")
    @classmethod
    def _validate_sha256_shape(cls, value: str, info: ValidationInfo) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(
                f"EmbeddedFigureDigitization.{info.field_name} {value!r} is not 64 lowercase hex characters"
            )
        return value

    @model_validator(mode="after")
    def _validate_canonical_json_coheres(self) -> EmbeddedFigureDigitization:
        """T1, at construction: refuse a citation that is not well formed.

        The work is in :meth:`_reconstruct`, which the accessors run again on every read. This
        wrapper exists only so pydantic reports a malformed citation as a ``ValidationError`` at
        the point of construction.
        """
        self._reconstruct()
        return self

    def _reconstruct(self) -> FigureDigitization:
        """T1: validate these bytes end to end and return the record they describe.

        THE ONLY DEFINITION OF "ACCEPTED" IN THIS CLASS, and that is the point rather than a
        stylistic preference. Construction runs it, and so does every accessor read, so there is
        no subset of it that a read could pass while construction would refuse. An earlier
        revision had the read path check a hand-picked three facts -- address, document digest,
        and the cached record re-serializing to the bytes -- and that trio, while individually
        sound, never re-ran D1-D9. Since :class:`FigureDigitization` is a frozen DATACLASS,
        ``object.__new__`` skips its ``__post_init__`` exactly as ``object.__setattr__`` skips
        pydantic's ``frozen=True``: a record built that way with ``coverage=COMPLETE`` and a
        non-empty ledger, paired with bytes and an address minted to match it, satisfied all
        three and reported ``COMPLETE`` with ``omission_count=1`` -- a state
        :meth:`FigureDigitization.from_payload` refuses on the identical bytes. Coherence is not
        validity, so the read path stops asking for coherence and asks for the whole thing.

        Each refusal carries its own marker phrase, so a test can tell which one fired:

        - ``does not parse as JSON``
        - ``is not a JSON object``
        - ``nested too deeply to re-serialize``
        - ``is not the canonical rendering``
        - ``does not live at the address it claims``
        - ``is not the readable version``
        - ``is not the shape of a version-``
        - ``not the declared raw_sha256``
        - ``does not reconstruct``

        Listed rather than counted on purpose: this paragraph twice said a NUMBER of steps that
        the code had since moved past, and ``TestTheDocstringSaysWhatIsActuallyTrue`` now checks
        the list against the source instead of trusting the prose to have been updated.

        The canonical-rendering refusal and the reconstruction refusal are BOTH required and
        neither replaces the other: reconstruction establishes that the payload is a VALID
        record, re-serialization that these bytes are its CANONICAL rendering. Dropping the
        latter would admit non-canonical but valid bytes carrying an honestly recomputed address
        -- one logical record with as many addresses as it has JSON renderings, which defeats
        addressing itself. Dropping the former is the bug described above.

        That is a claim about renderings and no more: the check pins the JSON, not the SPELLING
        of the values inside it. Two payloads differing only in the case of one hex-float
        coordinate (``0x1.03b3333333333p+9`` against ``0X1.03B3333333333P+9``) are each the
        canonical JSON of themselves, each accepted at its own honestly computed address, and
        :meth:`FigureDigitization.from_payload` returns EQUAL records from both -- one logical
        record, two addresses, which is exactly what this paragraph says cannot happen. It
        cannot happen in practice because
        :func:`~carmel.services.figure_digitization_record._pt` emits ``float.hex()``, which is
        one spelling per value, so no producer can reach the second address. The axis is closed
        by the PRODUCER, not by this schema, and a future writer of coordinates has to keep it
        that way. Pinned by ``test_the_address_pins_the_rendering_not_the_spelling``.

        See the class docstring for what all of them together do NOT prove.

        WHAT IT COSTS, measured on this machine against this code. A read is one full T1: parse,
        re-canonicalize, hash, reconstruct. At 0.6 KB (one omission) that is ~24 us; at 94.8 KB
        (700 omissions), ~2558 us, of which roughly 1400 us is the parse-and-reconstruct and
        1150 us the re-canonicalization that pins the canonical-rendering refusal. The
        hand-picked trio this replaced cost
        ~1495 us at the same size, so reads really are about 1.7x slower than the revision
        described above -- the one that cached the reconstructed record and re-checked only
        address, document digest, and re-serialization. On a cache hit that revision genuinely
        was the faster path. Something fast WAS given up; it was given up because it was also
        wrong, admitting records
        :meth:`FigureDigitization.from_payload` refuses on the identical bytes. (The trio was a
        net loss against holding no cache at all, which is why nothing here is worth
        reintroducing -- but that is a different comparison from the one this paragraph makes,
        and it does not make the 1.7x free.)

        Do NOT memoize this to win the 1.7x back. Any memo is private state, and private state is
        writable by every route that writes a public field -- which is how both previous
        revisions were broken. If reads ever become hot, the fix is for the CALLER to hold the
        returned :class:`FigureDigitization`, which is frozen and validated, not for this class
        to hold something an attacker can rewrite.

        Raises:
            ValueError: If these bytes are not a well-formed citation of a valid record. Raised
                rather than returned so construction and read share one failure mode.
        """
        try:
            parsed = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            # Same broadened except, for the same reason, as EmbeddedTableInventory's:
            # canonical_json is untrusted input from a stored file, and a short-but-deeply-nested
            # payload blows the interpreter's own stack from inside json.loads before this
            # validator can react.
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                f"canonical_json does not parse as JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                f"canonical_json is not a JSON object, it decodes to {type(parsed).__name__}"
            )
        try:
            recanonicalized = canonical_json_bytes(parsed)
        except RecursionError as exc:
            # The same threat as the broadened except above, one call later and easy to miss:
            # `json.loads` uses a C scanner while `canonical_json_bytes` recurses in Python, so a
            # payload nested deeply enough to exhaust the interpreter HERE can still have parsed
            # cleanly. Reproduced under a lowered recursion limit; at the default limit the
            # re-canonicalization survived every depth tried. Guarded regardless, because the
            # stack available at this call is a property of the CALLER's depth, not of the bytes,
            # and an unguarded RecursionError leaves an accessor raising something no caller is
            # told to catch.
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                f"canonical_json is nested too deeply to re-serialize: {exc}"
            ) from exc
        if recanonicalized != self.canonical_json.encode("utf-8"):
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                "canonical_json is not the canonical rendering of what it decodes to -- re-serializing "
                "through canonical_json_bytes produced different bytes"
            )
        # The address is over the canonical bytes, exactly as compute_digitization_sha defines
        # it. Recomputed from the RE-canonicalized bytes, which the check above has already
        # pinned equal to the stored ones -- so this cannot be satisfied by honestly hashing
        # arbitrary bytes against themselves in some other rendering.
        actual = hashlib.sha256(recanonicalized).hexdigest()
        if actual != self.digitization_sha256:
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                f"canonical_json's bytes hash to {actual!r}, so this record does not live at the address "
                "it claims"
            )
        # Version and key set are checked by `from_payload` too, but are checked HERE as well so
        # the failure a reader sees names the version it could not read rather than surfacing as
        # a missing key three frames down. A reader that does not know a shape must not guess at
        # it -- "I cannot read this" and "this is incoherent" are different facts.
        version = parsed.get("payload_version")
        if version != DIGITIZATION_PAYLOAD_VERSION:
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): "
                f"payload_version {version!r} is not the readable version {DIGITIZATION_PAYLOAD_VERSION!r}"
            )
        keys = set(parsed)
        if keys != set(DIGITIZATION_PAYLOAD_KEYS):
            unexpected = sorted(keys - DIGITIZATION_PAYLOAD_KEYS)
            missing = sorted(DIGITIZATION_PAYLOAD_KEYS - keys)
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): the record "
                f"is not the shape of a version-{DIGITIZATION_PAYLOAD_VERSION} digitization (unexpected "
                f"keys {unexpected!r}, missing keys {missing!r})"
            )
        declared_raw = parsed.get("raw_sha256")
        if declared_raw != self.raw_sha256:
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): the record "
                f"names document {declared_raw!r}, not the declared raw_sha256 {self.raw_sha256!r}"
            )
        # The reconstruction, and the reason this class's T1 is stronger than
        # EmbeddedTableInventory's. `from_payload` re-runs D1-D9, so a record claiming a
        # complete series while carrying an omission -- or one whose census does not balance
        # against its recovered points -- is refused on the BYTES, with no producer in the loop
        # and no document required. Asked of the record module rather than re-described here, so
        # the schema never carries its own second idea of what a coherent digitization is.
        try:
            return FigureDigitization.from_payload(parsed)
        except UNREADABLE_PAYLOAD as exc:
            raise ValueError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}): the record "
                f"does not reconstruct, so its coverage claim is not one anything could act on: {exc!r}"
            ) from exc

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> EmbeddedFigureDigitization:
        """Copy this citation, re-running T1 whenever a field changes.

        Pydantic's ``model_copy(update=...)`` deliberately runs NO validation and copies private
        attributes verbatim, which on an addressed model is a hole rather than a convenience:
        ``model_copy(update={"canonical_json": "{}"})`` produced an object reporting the
        ORIGINAL's ``coverage``, carrying different bytes, and declaring a
        ``digitization_sha256`` that hashed neither. Everything downstream that trusts the
        address would then have been verifying a stale answer and reporting success.

        So a copy that changes anything is rebuilt through full validation. It does NOT
        recompute the address to match new bytes: silently re-deriving ``digitization_sha256``
        would turn this method into a way to mint a valid citation for arbitrary content, which
        is a worse hole than the one being closed. Change the bytes and the address together and
        the copy validates; change one alone and T1 refuses it, loudly, at the copy.

        ``deep`` is accepted for signature compatibility and has no effect: all three fields are
        immutable ``str``, and there is no other state to copy shallowly or otherwise.
        """
        if not update:
            return super().model_copy(deep=deep)
        merged = {**self.model_dump(), **dict(update)}
        return type(self)(**merged)

    def _validated_record(self) -> FigureDigitization:
        """Re-run :meth:`_reconstruct` and answer from what it returns, or refuse loudly.

        NOTHING IS REMEMBERED AND NOTHING IS TRUSTED. There is no cached record and no stored
        provenance; the record is derived afresh from ``canonical_json`` on every read, and
        derived by the SAME code that construction runs. Two earlier revisions failed here in
        two different ways, and both failures were the same shape -- a read-time check that was
        cheaper than construction's:

        - A private digest of the public fields, compared per read. Forgeable in one line,
          because any route that could write ``digitization_sha256`` past ``frozen=True`` could
          recompute the digest immediately after. A guard an attacker recomputes is not a guard.
        - A hand-picked trio -- address, document digest, and a cached record re-serializing to
          the bytes -- which never re-ran D1-D9. ``FigureDigitization`` is a frozen DATACLASS, so
          ``object.__new__`` skips ``__post_init__``; a record minted that way claiming
          ``COMPLETE`` over a non-empty ledger, with bytes and an address generated to match,
          satisfied every one of the three.

        The lesson both times was that a subset of the constructor's checks is not a weaker
        version of it, it is a different and wrong predicate. So the read path stopped picking a
        subset. Whatever construction refuses as an invalid RECORD, a read refuses, because it
        is the same call -- and only that, see the asymmetry below.

        WHAT THIS DOES AND DOES NOT REFUSE. It refuses every record construction would refuse as
        invalid, which is the strongest claim this class can make about itself, and it cannot be
        talked out of it by writing to any attribute, private or public -- there is nothing left
        to write that it does not re-derive. FOUR residues, named rather than rounded up:

        1. The check runs when an ACCESSOR is called. A caller reading ``.canonical_json`` or
           ``.digitization_sha256`` as plain attributes gets whatever is there, by not asking.
        2. Validity is not truth. A hand-built, fully valid citation is a well-formed CLAIM that
           nothing has checked against a figure -- see the class docstring.
        3. A SUBCLASS overriding :attr:`coverage`, :attr:`auditable`, :attr:`omission_count` or
           this method answers whatever it likes. Python has no way to prevent that, and this
           class does not pretend to.
        4. The answer is a snapshot. A caller that stores ``coverage`` in a local holds a value
           that a later mutation of the object will not invalidate; only the next read re-checks.

        HOW A READ REFUSES, which is part of the same promise and was briefly not true. A read
        that refuses raises ``RuntimeError``, so ``except RuntimeError`` around an accessor is
        the guard a caller wants. That held only once
        :data:`~carmel.services.figure_digitization_record.UNREADABLE_PAYLOAD` grew
        ``OverflowError``: making the read path re-run the whole reconstruction imported that
        function's refusal surface wholesale, and a stored coordinate reading ``"0x1p+99999"``
        came back out of an accessor as a raw ``OverflowError``, past the ``except ValueError``
        below. The surface is now swept by a test rather than read off the code, because reading
        the code is how it was got wrong. What the sweep does NOT reach, and what no ``except``
        clause here can promise, is the interpreter running out of stack or memory partway
        through -- ``RecursionError`` is caught at the two places these bytes are walked, and a
        ``MemoryError`` on a payload large enough to cause one is not a refusal, it is the
        process failing.

        ONE ASYMMETRY WITH CONSTRUCTION, deliberate and in the safe direction. Construction also
        applies this class's FIELD constraints -- digest shape, and ``canonical_json``'s
        ``max_length`` -- which a read does not re-apply. So bytes over
        ``_MAX_EMBEDDED_CANONICAL_JSON_LENGTH`` carrying a perfectly valid record are refused at
        construction and answered on read. That is not a hole: the size bound is a
        resource-exhaustion guard on what may ENTER an envelope, not a claim about validity, and
        an object holding such bytes cannot have entered one. On the axis that matters -- is this
        a valid record, canonically rendered, at the address it claims -- construction and read
        are the same call and cannot diverge.

        Raising beats returning a default. Every question this class answers is a question about
        MISSING DATA, and a default answer to that question is the one shape of wrong answer that
        reads as reassuring.
        """
        try:
            return self._reconstruct()
        except ValueError as exc:
            raise RuntimeError(
                f"EmbeddedFigureDigitization(digitization_sha256={self.digitization_sha256!r}) no longer "
                f"validates, so nothing about it can be answered: {exc}"
            ) from exc

    @property
    def coverage(self) -> FigureCoverage:
        """Is anything missing from this series -- COMPLETE, PARTIAL, or UNCHECKABLE.

        The coverage axis ALONE. ``UNCHECKABLE`` is not a middling amount of coverage; it is the
        statement that the question was not answerable, and a caller that reads it as a
        near-``COMPLETE`` has performed exactly the collapse :attr:`auditable` exists to
        prevent.
        """
        return self._validated_record().coverage

    @property
    def auditable(self) -> bool:
        """Could the instrument have told that something was missing?

        The auditability axis ALONE, independent of :attr:`coverage`. True means a marker census
        exists for this series' plot region, so a completeness claim has something to be
        measured against. It says nothing about whether anything WAS missing.
        """
        return isinstance(self._validated_record().census, MarkerCensus)

    @property
    def omission_count(self) -> int:
        """How many markers this series is recorded as having lost.

        Zero is a completeness claim only when :attr:`auditable` is True. Under an unavailable
        census it means the ledger names nothing, which is not the same as nothing having been
        dropped -- and that pair is precisely why this is a count beside a flag rather than a
        single number a reader could interpret alone.
        """
        return len(self._validated_record().omissions)


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
                f"{where}: source_form=TABULAR requires value_ref.locator.kind=TABLE_CELL, got {ref.locator.kind!r}"
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
    LocatorKind.CHAR_SPAN: frozenset({SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER}),
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
no tabular structure of its own. A ``CharSpanLocator`` addresses the node's
EXTRACTED TEXT, so it may target any node kind that has one -- a PDF, a
JATS/XML document, or an SI member all get text extracted from them.
``FIGURE_CROP`` is deliberately excluded from that row: a figure crop is an
image region, and this project stores no OCR output or extracted text for
one, so a character offset into it would address nothing that exists. Which
ARCHIVE a node was extracted from is not a locator concern at all -- see
:class:`ArchiveOrigin` on :class:`SourceNode` -- so no ``LocatorKind``
addresses that here.
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

``SI_MEMBER`` compatible with BOTH key kinds is what this STATIC table can
say, because ``SI_MEMBER`` is a single ``SourceNodeKind`` covering a csv, an
xlsx, a PDF, and a zip member all alike -- the table keyed on node kind alone
cannot tell "this SI member is a spreadsheet, so only MEMBER_SHEET makes
sense against it" from "this SI member is a rendered document, so only
CAPTION_LABEL makes sense against it".

I-075 closes that gap for a member that DECLARES its
:attr:`SourceNode.document_kind`: :func:`_validate_locator_kind_compatibility`
reads this table AND then, for a declared member, rejects a ``MEMBER_SHEET``
key against a non-spreadsheet and a ``CAPTION_LABEL`` key against a
spreadsheet. So the gap now stays open ONLY for members that declare nothing
-- both keys are still accepted against an undeclared ``SI_MEMBER`` here,
where V8 refuses a caption cell both ways. The table itself stays as-is
(keyed on node kind, it cannot express document kind); the narrowing lives in
the validator that also holds the node's declared value.
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


_ABSENCE_MARKER_KEYS = frozenset({"__absent__", "reason", "note"})


def _is_absence_marker(value: Any) -> bool:
    """Detect a dict that is *shaped like* an absence marker, i.e. carries the
    ``"__absent__"`` key at all.

    This is deliberately a cheap, permissive test (key presence only, not full
    shape validation) -- :func:`_rehydrate_absence_marker` does the strict
    validation and is the one that raises. Using the mere presence of the key
    to decide "this dict must be a marker" is exactly what makes a malformed
    marker (bad ``__absent__`` value, extra key, unknown ``reason``) a hard
    rejection here rather than something that falls through and gets handed
    to pydantic as if it were ordinary present-value data.
    """
    return isinstance(value, dict) and "__absent__" in value


def _rehydrate_absence_marker(marker: dict[str, Any]) -> Absent:
    """Reconstruct the :class:`Absent` instance that
    :func:`_absent_identity_payload` would have projected from ``marker``.

    Every check here is load-bearing: ``Absent`` itself is
    ``extra="forbid"``, so it cannot be constructed directly from the
    marker's own dict (which carries the extra ``"__absent__"`` key) --
    this function is what strips that key back out, but only after proving
    the marker is exactly what a real projection would have produced.
    """
    if marker.get("__absent__") is not True:
        raise DatasetEnvelopeParseError(
            f"absence marker has __absent__={marker.get('__absent__')!r}, not True -- "
            "a real Absent projection always sets __absent__ to the literal True"
        )
    actual_keys = set(marker)
    if actual_keys != _ABSENCE_MARKER_KEYS:
        raise DatasetEnvelopeParseError(
            f"absence marker has keys {sorted(actual_keys)!r}, expected exactly "
            f"{sorted(_ABSENCE_MARKER_KEYS)!r} -- a real Absent projection never has more or fewer"
        )
    reason_raw = marker["reason"]
    if not isinstance(reason_raw, str):
        raise DatasetEnvelopeParseError(
            f"absence marker's reason is {reason_raw!r} ({type(reason_raw).__name__}), expected a str "
            "-- AbsenceReason always projects to its .value, a plain string"
        )
    try:
        reason = AbsenceReason(reason_raw)
    except ValueError as exc:
        raise DatasetEnvelopeParseError(
            f"absence marker's reason {reason_raw!r} is not a known AbsenceReason value"
        ) from exc
    note = marker["note"]
    if note is not None and not isinstance(note, str):
        raise DatasetEnvelopeParseError(
            f"absence marker's note is {note!r} ({type(note).__name__}), expected str or None"
        )
    return Absent(reason=reason, note=note)


def _rehydrate_identity_payload(value: Any) -> Any:
    """Recursively walk an ``identity_payload()``-shaped structure, replacing
    every absence marker with the real :class:`Absent` instance it
    represents so the result can be handed to ``DatasetEnvelope.model_validate``.

    Everything that is not an absence marker -- dicts, lists, scalars -- is
    left structurally alone; pydantic does the rest of the reconstruction
    (enums from their ``.value`` strings, tuples from lists, nested models
    from nested dicts) during ``model_validate`` itself.
    """
    if isinstance(value, dict):
        if _is_absence_marker(value):
            return _rehydrate_absence_marker(value)
        return {key: _rehydrate_identity_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_identity_payload(item) for item in value]
    return value


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
    locator: BBoxLocator | TableCellLocator | XPathLocator | CharSpanLocator,
) -> dict[str, Any]:
    if isinstance(locator, BBoxLocator):
        return {"kind": locator.kind.value, "bbox": _bbox_identity_payload(locator.bbox)}
    if isinstance(locator, TableCellLocator):
        return {
            "kind": locator.kind.value,
            "table_key": _table_key_identity_payload(locator.table_key),
            "row": locator.row,
            "col": locator.col,
            # Projected, not omitted: two locators differing ONLY in which inventory they
            # cite are different claims about which grid justified the cell, and omitting
            # the field here would collapse them onto one content address in a write-once
            # store. _project_maybe renders Absent through the same `__absent__` marker
            # _rehydrate_identity_payload already round-trips.
            "pdf_table_inventory_sha256": _project_maybe(locator.pdf_table_inventory_sha256),
        }
    if isinstance(locator, XPathLocator):
        return {"kind": locator.kind.value, "xpath": locator.xpath}
    if isinstance(locator, CharSpanLocator):
        return {
            "kind": locator.kind.value,
            "text_space": locator.text_space.value,
            "start": locator.start,
            "end": locator.end,
        }
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


def _extraction_binding_identity_payload(binding: ExtractionBinding) -> dict[str, Any]:
    return {
        "parent_raw_sha256": binding.parent_raw_sha256,
        "extraction_sha256": binding.extraction_sha256,
        "extracted_sha256": binding.extracted_sha256,
        "extracted_text_sha256": binding.extracted_text_sha256,
        "extractor": binding.extractor,
        "extractor_code_sha256": binding.extractor_code_sha256,
        "identity_payload_version": binding.identity_payload_version,
        "pypdf_version": _project_maybe(binding.pypdf_version),
    }


def _glyph_health_identity_payload(health: GlyphHealth) -> dict[str, Any]:
    return {
        "suspects_dash_corruption": health.suspects_dash_corruption,
        "has_thorn_plus_marker": health.has_thorn_plus_marker,
        "has_equals_ambiguity_marker": health.has_equals_ambiguity_marker,
        "has_slash_c0_minus_marker": health.has_slash_c0_minus_marker,
        "has_ascii6_uncertainty_marker": health.has_ascii6_uncertainty_marker,
    }


def _glyph_health_assessment_identity_payload(assessment: GlyphHealthAssessment) -> dict[str, Any]:
    return {
        "health": _glyph_health_identity_payload(assessment.health),
        "assessor": _semantic_dependency_use_identity_payload(assessment.assessor),
    }


def _source_verification_identity_payload(verification: SourceVerification) -> dict[str, Any]:
    return {
        "raw_artifact": verification.raw_artifact.value,
        "extracted_text": verification.extracted_text.value,
        "root_sidecar": verification.root_sidecar.value,
    }


def _addresses_a_crop_region(node: SourceNode) -> bool:
    """Whether ``node``'s projection carries a ``crop_region`` key -- the one
    predicate behind :data:`_CONDITIONALLY_PROJECTED_FIELDS`, and the single
    place the PROJECTION side of that decision is made. The parse side makes
    the mirror decision separately and off ``kind``, because it has no value
    to read; see :func:`_restore_unprojected_crop_regions` and the pair named
    in the registry's own docstring.

    Asks the VALUE, not the kind, and that is the whole point. "Only a
    FIGURE_CROP may hold a region" is I7's rule (see
    :meth:`SourceNode._validate_crop_region_addresses_a_figure_crop`), and
    restating it here as ``node.kind == FIGURE_CROP`` would make a second,
    hand-maintained copy of it: a kind that is later allowed to carry a region
    would have to be remembered in two places, and forgetting the second one
    silently computes an address with the region left out. Reading the value
    instead makes this a CONSEQUENCE of I7 rather than a duplicate of it --
    whatever I7 decides may hold a region, this projects.

    It also keeps a producer-bug detector that a kind-derived predicate loses.
    A node carrying a region it has no right to -- a ``PAPER_PDF`` with a real
    ``BBox``, reachable only by skipping validation via ``model_construct`` or
    ``object.__setattr__`` -- projects that region here, so its payload no
    longer parses (I7 refuses it on the way back in) and
    ``store_typed_envelope`` refuses the write. Keyed on kind, the same
    tampered node would project byte-identically to a clean one, and the write
    would go through unremarked. Neither shape puts anything untrue in the
    store, but only this one still notices.

    Both callers on the PROJECTION side -- the projection itself and the
    registry the completeness walker reads -- go through this function, so
    they cannot disagree about which nodes carry the key. The PARSE side
    cannot join them: see the caveat at the top of this docstring.
    """
    return not isinstance(node.crop_region, Absent)


def _declares_a_document_kind(node: SourceNode) -> bool:
    """Whether ``node``'s projection carries a ``document_kind`` key -- the
    projection-side predicate for that field's entry in
    :data:`_CONDITIONALLY_PROJECTED_FIELDS`, and the single place the
    PROJECTION side of that decision is made. Its exact parallel is
    :func:`_addresses_a_crop_region`, and the same reasoning applies verbatim.

    Asks the VALUE, not the kind, and that is the whole point.
    ":meth:`SourceNode._validate_document_kind` decides WHICH absences are
    legal" is that validator's rule; restating it here as
    ``node.kind == SI_MEMBER`` would make a second, hand-maintained copy of
    it. Reading the value makes this a CONSEQUENCE of the validator rather
    than a duplicate: whatever it admits as a declared kind, this projects,
    and an ``Absent`` (of either legal reason) is omitted. The parse side
    makes the mirror decision separately and off ``kind`` because it has no
    value to read; see :func:`_restore_unprojected_document_kinds`.
    """
    return not isinstance(node.document_kind, Absent)


def _source_node_identity_payload(node: SourceNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "sha256": node.sha256,
        "parent_node_id": node.parent_node_id,
        "origin": _project_maybe(node.origin, _archive_origin_identity_payload),
        "extraction": _project_maybe(node.extraction, _extraction_binding_identity_payload),
        "glyph_health": _project_maybe(node.glyph_health, _glyph_health_assessment_identity_payload),
        # Folded into the address on purpose: an envelope's content address must
        # COMMIT to the verification standard it claims. Left out, a record-only
        # envelope and a fully root-verified one would address identically, and the
        # claim could be swapped for free without breaking the address that is
        # supposed to authenticate it.
        "verification": _project_maybe(node.verification, _source_verification_identity_payload),
    }
    # The one CONDITIONAL key in either projection (registered as such in
    # _CONDITIONALLY_PROJECTED_FIELDS): emitted when this node actually holds a
    # region, omitted when it does not.
    #
    # A node holding one is a FIGURE_CROP -- but that is I7's conclusion, read
    # off the value here rather than restated as a second copy of I7's kind
    # rule. See :func:`_addresses_a_crop_region` for why the difference matters.
    #
    # For a crop the region is identity-bearing: it is what tells two figures of
    # one paper apart when their bytes cannot (two byte-identical images at two
    # page locations), so an envelope citing Figure 3 must not address like one
    # citing Figure 7. On every other kind I7 admits exactly ONE value --
    # Absent(NOT_APPLICABLE) -- so emitting it there would fold a CONSTANT into
    # the address: zero distinguishing bits, at the price of re-addressing every
    # envelope in every store, including the ones holding no crop at all.
    #
    # Omission is not a shortcut for an unaddressed field: the inverse
    # (_restore_unprojected_crop_regions) puts the marker back before
    # validation, the completeness walker requires the key present for an
    # instance the predicate includes and ABSENT for one it excludes, and the
    # round trip is byte-exact for every node kind.
    if _addresses_a_crop_region(node):
        payload["crop_region"] = _project_maybe(node.crop_region, _bbox_identity_payload)
    # The second CONDITIONAL key, registered in _CONDITIONALLY_PROJECTED_FIELDS
    # and following the crop_region precedent above exactly. Emitted when this
    # node DECLARES a document kind, omitted otherwise.
    #
    # A declared kind is identity-bearing: V3 and V8 branch on it (a declared
    # word-processor member's caption-labelled cell is ACCEPTED where a PDF's
    # or an undeclared member's is refused), so an envelope declaring
    # word_processor must not address like one that declares pdf or declares
    # nothing. When it is Absent -- NOT_APPLICABLE for a non-SI node,
    # NOT_EXTRACTED_YET for an undeclared SI member -- the marker is a
    # kind-determined CONSTANT (_validate_document_kind admits exactly one
    # reason per kind), so folding it into the address would add zero
    # distinguishing bits at the price of re-addressing every node.
    #
    # When the predicate holds the value is always a concrete
    # SiMemberDocumentKind, so its .value (a plain string) is the payload; the
    # inverse (_restore_unprojected_document_kinds) reconstructs the omitted
    # Absent marker from the payload's kind before validation, and the round
    # trip is byte-exact.
    if _declares_a_document_kind(node):
        payload["document_kind"] = _project_maybe(node.document_kind, lambda kind: kind.value)
    return payload


def _source_graph_identity_payload(graph: SourceGraph) -> dict[str, Any]:
    """Project a :class:`SourceGraph` to its identity-payload shape, with
    ``nodes`` emitted in ascending ``node_id`` order regardless of the order
    the model's tuple happens to hold them in.

    The sort is what makes this projection CANONICAL. ``SourceGraph.nodes``
    is semantically a set -- no validator constrains its order, and no
    consumer may read meaning into it -- so the same graph is legally
    constructible with its nodes in any permutation. Projected in tuple
    order, each of those permutations produced different canonical bytes,
    i.e. ONE dataset held MANY content addresses: the store's write-once
    dedup silently stopped deduplicating, and byte-level comparison of two
    stored payloads could report two "different" datasets that differ in
    nothing at all. Normalizing here, at the projection, is deliberate --
    an ordering VALIDATOR on the model field would instead outlaw
    construction orders that carry no meaning to begin with. The sort key
    is total and deterministic because ``node_id`` is unique within a graph
    (enforced by ``SourceGraph``'s own duplicate-id validator).
    """
    return {
        "nodes": [_source_node_identity_payload(node) for node in sorted(graph.nodes, key=lambda node: node.node_id)]
    }


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
        "unit_provenance": value.unit_provenance.value,
        "unit_raw": _project_maybe(value.unit_raw),
        "unit_normalized": value.unit_normalized,
        "conversion_table_sha256": value.conversion_table_sha256,
        "value_ref": _source_ref_identity_payload(value.value_ref),
        "unit_ref": _project_maybe(value.unit_ref, _source_ref_identity_payload),
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
        "digitization_sha256": _project_maybe(series.digitization_sha256),
    }


def _embedded_conversion_table_identity_payload(table: EmbeddedConversionTable) -> dict[str, Any]:
    return {"sha256": table.sha256, "canonical_json": table.canonical_json}


def _embedded_table_inventory_identity_payload(inventory: EmbeddedTableInventory) -> dict[str, Any]:
    return {
        "inventory_sha256": inventory.inventory_sha256,
        "raw_sha256": inventory.raw_sha256,
        "canonical_json": inventory.canonical_json,
    }


def _embedded_figure_digitization_identity_payload(digitization: EmbeddedFigureDigitization) -> dict[str, Any]:
    return {
        "digitization_sha256": digitization.digitization_sha256,
        "raw_sha256": digitization.raw_sha256,
        "canonical_json": digitization.canonical_json,
    }


def _grounded_scalar_claim_identity_payload(claim: GroundedScalarClaim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "label_raw": claim.label_raw,
        "label_ref": _source_ref_identity_payload(claim.label_ref),
        "value": _measured_value_identity_payload(claim.value),
        "uncertainty": _project_maybe(claim.uncertainty, _uncertainty_identity_payload),
    }


def _grounded_categorical_claim_identity_payload(claim: GroundedCategoricalClaim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "label_raw": claim.label_raw,
        "label_ref": _source_ref_identity_payload(claim.label_ref),
        "token_raw": claim.token_raw,
        "token_ref": _source_ref_identity_payload(claim.token_ref),
    }


def _unextracted_condition_statement_identity_payload(
    statement: UnextractedConditionStatement,
) -> dict[str, Any]:
    return {
        "statement_id": statement.statement_id,
        "label_raw": statement.label_raw,
        "label_ref": _source_ref_identity_payload(statement.label_ref),
        "statement_raw": statement.statement_raw,
        "statement_ref": _source_ref_identity_payload(statement.statement_ref),
        "reason": statement.reason.value,
        "quantity_kind": _project_maybe(statement.quantity_kind, lambda kind: kind.value),
    }


_SUBJECT_KIND_KEY = "subject_kind"
"""The tag key under which :func:`_condition_subject_identity_payload`
records WHICH variant of the subject sum a payload projects.

``"subject_kind"`` rather than the ``"kind"`` used by the locator/table-key
projections: those unions tag with ``"kind"`` because their models carry a
literal ``kind`` FIELD whose projection doubles as the tag. Neither subject
variant has (or should grow) such a field, so this tag is projection-only
data -- naming it after the sum itself (the subject) keeps it from ever
colliding with a real field name of either variant, today or after a future
field addition, and makes a raw payload self-describing to a reader who has
never seen this module."""

_SUBJECT_KIND_DEVICE_CLASS = "device_class"
_SUBJECT_KIND_UNRESOLVED = "unresolved"


def _device_class_declaration_identity_payload(subject: DeviceClassDeclaration) -> dict[str, Any]:
    return {
        "label_raw": subject.label_raw,
        "label_ref": _source_ref_identity_payload(subject.label_ref),
    }


def _unresolved_subject_identity_payload(subject: UnresolvedSubject) -> dict[str, Any]:
    return {
        "reason": subject.reason.value,
        "reason_ref": _source_ref_identity_payload(subject.reason_ref),
    }


def _condition_subject_identity_payload(
    subject: DeviceClassDeclaration | UnresolvedSubject,
) -> dict[str, Any]:
    """Project the subject SUM of a :class:`ConditionSetEnvelope`, TAGGED
    with :data:`_SUBJECT_KIND_KEY` so the two variants can never produce the
    same payload.

    The tag is load-bearing, not decorative. Today the two variants happen
    to have disjoint field names, so their untagged projections could not
    collide -- but "happen to" is exactly the wrong thing to hang a
    content address on: one future field addition (say, a ``label_raw`` on
    a refusal recording what the source ALMOST named) could make an
    untagged :class:`UnresolvedSubject` payload a legal
    :class:`DeviceClassDeclaration` payload, and in a write-once
    content-addressed store two different subjects addressing identically is
    a permanent collision, not a bug that can be fixed after the fact.
    :meth:`ConditionSetEnvelope.from_identity_payload` dispatches on this
    tag (see :func:`_rehydrate_condition_subject`) rather than sniffing
    field shapes, for the same reason.

    The tag is applied LAST -- ``{**variant_payload, _SUBJECT_KIND_KEY: ...}``
    -- never merged in first, so a future variant field literally named
    ``"subject_kind"`` cannot silently overwrite the tag by dict-merge
    order: the correct tag always wins the address, structurally, not by
    convention. The guard below backs that structural guarantee with a
    loud failure: it raises, before the merge, if a variant payload already
    carries the tag key, rather than silently drop that field's real value
    under the tag -- a colliding variant field is a bug in the variant
    projector, and this function is the one place positioned to catch it
    before it can ever reach the content address.
    """
    if isinstance(subject, DeviceClassDeclaration):
        variant_payload = _device_class_declaration_identity_payload(subject)
        tag = _SUBJECT_KIND_DEVICE_CLASS
    elif isinstance(subject, UnresolvedSubject):
        variant_payload = _unresolved_subject_identity_payload(subject)
        tag = _SUBJECT_KIND_UNRESOLVED
    else:
        raise TypeError(f"_condition_subject_identity_payload: unhandled subject variant {subject!r}")
    if _SUBJECT_KIND_KEY in variant_payload:
        raise AssertionError(
            f"_condition_subject_identity_payload: {type(subject).__name__}'s projected payload already "
            f"contains the {_SUBJECT_KIND_KEY!r} tag key -- a variant field must never be named after the "
            "sum's own tag, or the tag and that field's value could collide in the content address"
        )
    return {**variant_payload, _SUBJECT_KIND_KEY: tag}


def _rehydrate_condition_subject(subject: Any) -> DeviceClassDeclaration | UnresolvedSubject:
    """Inverse of :func:`_condition_subject_identity_payload`: dispatch on
    the :data:`_SUBJECT_KIND_KEY` tag, strip it, and reconstruct the tagged
    variant -- never guess a variant from field shapes.

    A missing or unknown tag is a hard :class:`DatasetEnvelopeParseError`,
    not a fall-through to pydantic's union matching: the tag is what makes
    the two variants un-collidable in the store, so a payload without a
    usable tag has no trustworthy identity to reconstruct. The tag key is
    stripped before validation because both variants are ``extra="forbid"``
    -- the same reason :func:`_rehydrate_absence_marker` strips
    ``"__absent__"`` -- but only after the tag has been proven to name a
    known variant.
    """
    if not isinstance(subject, dict):
        raise DatasetEnvelopeParseError(
            f"condition-set subject payload is {type(subject).__name__}, expected a dict carrying the "
            f"{_SUBJECT_KIND_KEY!r} tag"
        )
    tag = subject.get(_SUBJECT_KIND_KEY)
    if tag == _SUBJECT_KIND_DEVICE_CLASS:
        variant: type[DeviceClassDeclaration | UnresolvedSubject] = DeviceClassDeclaration
    elif tag == _SUBJECT_KIND_UNRESOLVED:
        variant = UnresolvedSubject
    else:
        raise DatasetEnvelopeParseError(
            f"condition-set subject payload has {_SUBJECT_KIND_KEY}={tag!r}, expected "
            f"{_SUBJECT_KIND_DEVICE_CLASS!r} or {_SUBJECT_KIND_UNRESOLVED!r} -- a subject without a "
            "known tag has no trustworthy identity to reconstruct"
        )
    untagged = {key: value for key, value in subject.items() if key != _SUBJECT_KIND_KEY}
    try:
        return variant.model_validate(untagged)
    except ValidationError as exc:
        raise DatasetEnvelopeParseError(
            f"condition-set subject payload tagged {tag!r} failed validation as {variant.__name__}: {exc}"
        ) from exc


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

A field that is addressed for SOME instances and genuinely inapplicable for
others belongs in :data:`_CONDITIONALLY_PROJECTED_FIELDS` below, NOT here:
registering it here would claim it is never addressed, which for the
instances that do carry it is a lie in the opposite direction.
"""


_CONDITIONALLY_PROJECTED_FIELDS: Mapping[tuple[str, str], tuple[Callable[[Any], bool], str]] = {
    ("SourceNode", "crop_region"): (
        _addresses_a_crop_region,
        "Identity-bearing for a node that holds a region -- a FIGURE_CROP, by "
        "I7 -- and structurally inapplicable to every other kind. I7 admits "
        "exactly one value on a non-crop node, Absent(NOT_APPLICABLE), so "
        "projecting it there would fold a CONSTANT into the content address: "
        "no envelope becomes distinguishable from any other, while every "
        "stored envelope re-addresses, including ones holding no crop at all. "
        "On a crop it is the opposite: the region separates two figures of one "
        "paper whose bytes cannot separate them, so omitting it there would "
        "let an envelope citing one figure address like an envelope citing "
        "another.",
    ),
    ("SourceNode", "document_kind"): (
        _declares_a_document_kind,
        "Identity-bearing for an SI_MEMBER that DECLARES its kind -- V3 and V8 "
        "branch on it, so an envelope declaring word_processor must not address "
        "like one declaring pdf or one declaring nothing -- and a "
        "kind-determined CONSTANT otherwise. _validate_document_kind admits "
        "exactly one absence per node kind (NOT_APPLICABLE off an SI_MEMBER, "
        "NOT_EXTRACTED_YET on an undeclared SI_MEMBER), so projecting the "
        "absent marker would fold that constant into the content address: no "
        "envelope becomes distinguishable, while every stored envelope "
        "re-addresses, including the ones that declare nothing. Emitting it "
        "only when a concrete kind is declared keeps exactly the declarations "
        "that separate two members apart.",
    ),
}
"""``(model_name, field_name) -> (is_projected_for, reason)`` registry of
fields the projection emits for SOME instances of a model and omits for
others, keyed the same way as :data:`_UNADDRESSED_FIELDS`.

``is_projected_for`` takes the MODEL INSTANCE and answers whether this field
must appear in that instance's projection. The completeness meta-test uses it
in both directions FOR THIS FIELD -- required present when the predicate
holds, required ABSENT when it does not -- so a change that starts emitting a
registered key unconditionally is caught as loudly as one that stops emitting
it altogether.

That is the limit of what the walk can see, and the limit matters. The walk
is driven by ``model_fields``: it visits each FIELD and asks whether the
projection carries it. A projected key that no field backs at all -- a
phantom, a typo'd name, a key some helper adds on its own -- is invisible to
it, in this registry or out of it. The authored byte pins are what catch
that, which is exactly how one was caught during review of this change: the
completeness walk stayed green and the pins did not.

Every conditional key needs an INVERSE, because ``from_identity_payload``
hands its payload straight to pydantic and these fields are required with no
default: see :func:`_restore_unprojected_crop_regions` for ``crop_region``
and :func:`_restore_unprojected_document_kinds` for ``document_kind``, each
restoring its own entry here. A conditional projection without an inverse is
a payload that projects cleanly and cannot be parsed back.

THE TWO SIDES ARE NOT ONE PIECE OF CODE, and for this entry they are not
even written against the same thing. The projection side asks the VALUE
(:func:`_addresses_a_crop_region`); the inverse asks the payload's ``kind``,
because it runs before validation, on bytes where the value it would consult
is exactly what was omitted. So these are the two places that must move
together when I7 changes which kinds may hold a region:

1. :func:`_addresses_a_crop_region` -- the projection-side predicate, and
   the registry entry above that reads it.
2. :func:`_restore_unprojected_crop_regions` -- the parse-side guard,
   keyed on ``kind``.

Left un-unified deliberately, with the cost measured rather than assumed:
if the two disagree, a node whose region should have been restored is left
without one and pydantic reports ``Field required`` where I7 would have said
"cannot address itself" -- an error-message downgrade on an already-refused
payload, never a wrong address, because the projection side alone decides
what gets addressed. (An inverse that restored the marker UNCONDITIONALLY
would need no ``kind`` test at all and would upgrade that message rather
than downgrade it; it is a behaviour change, so it is recorded as future
work rather than smuggled in beside a docstring fix.)
"""


_ENVELOPE_TYPE_KEY = "envelope_type"
"""Top-level identity-payload key naming which envelope class projected the
payload. Deliberately NOT in the store's reserved ``_carmel_`` namespace:
the store injects and strips reserved-prefixed keys itself and refuses
payloads that already carry them, so a discriminator the ENVELOPE must own
end-to-end (it is part of the addressed bytes) has to live under a plain
name."""

_IDENTITY_PAYLOAD_VERSION_KEY = "identity_payload_version"
"""Top-level identity-payload key carrying the projection-schema version.
Distinct from :attr:`ExtractionBinding.identity_payload_version` (a nested
FIELD of one projected model); this key versions the ENVELOPE projection
itself."""

_DATASET_ENVELOPE_TYPE = "dataset"
"""``envelope_type`` value emitted by :meth:`DatasetEnvelope.identity_payload`."""

_CONDITION_SET_ENVELOPE_TYPE = "condition_set"
"""``envelope_type`` value emitted by
:meth:`ConditionSetEnvelope.identity_payload`."""

_SUPPORTED_IDENTITY_PAYLOAD_VERSION = 6
"""The one envelope-projection version this module can parse. A payload
carrying any other version was projected by code this module has never
seen, so parsing it here could only produce a silently reinterpreted
envelope.

Version history:

1. The projection before :attr:`SourceNode.crop_region` existed.
2. ``crop_region`` projected for a node that HOLDS a region -- which, by
   I7, is a ``FIGURE_CROP`` -- and omitted for one that does not. See
   :func:`_addresses_a_crop_region` for why the condition is written
   against the value rather than the kind, and
   :data:`_CONDITIONALLY_PROJECTED_FIELDS` for the registry entry.
3. :attr:`Series.digitization_sha256` and
   :attr:`DatasetEnvelope.figure_digitizations` added -- the figure lane's
   citation surface (V9/FD1/FD2/FD3), the counterpart to the table lane's
   :attr:`TableCellLocator.pdf_table_inventory_sha256` and
   :attr:`DatasetEnvelope.table_inventories`. The version is a single shared
   projection-schema number, so a :class:`ConditionSetEnvelope` -- whose own
   shape is unchanged, since a condition set carries no series -- also stamps
   3: the projection MODULE changed, and a consumer keyed on this number must
   be told so. Nothing has been stored under any version (the branch is
   unreleased and no runtime constructs an envelope), so re-addressing every
   projection carries no migration cost.
4. :attr:`UnextractedConditionStatement.statement_raw` added -- the refused
   statement's own verbatim words, projected beside its ``statement_ref`` so a
   condition set's ``unextracted`` payload now carries the string as well as the
   span. This changes only the CONDITION-SET shape, but the version is a single
   shared projection-schema number, so a :class:`DatasetEnvelope` -- whose own
   shape is unchanged, since it carries no ``unextracted`` -- also stamps 4, by
   the same rule entry 3 records: the projection MODULE changed, and a consumer
   keyed on this number must be told. Nothing has been stored under any version,
   so re-addressing every projection still carries no migration cost.
5. :attr:`MeasuredValue.unit_provenance` added, and
   :attr:`MeasuredValue.unit_raw`/:attr:`~MeasuredValue.unit_ref` made
   ``Maybe`` (I-060): a dimensionless coordinate may now DECLARE
   ``quantity_kind=EQUIVALENCE_RATIO`` with the unit marked not-printed-in-source
   instead of falling back to ``OTHER`` with a corrupted header. This changes the
   projected shape of EVERY ``MeasuredValue`` -- the new ``unit_provenance`` key,
   and ``unit_raw``/``unit_ref`` now projected through the absence machinery --
   so BOTH a :class:`DatasetEnvelope` (series values) and a
   :class:`ConditionSetEnvelope` (scalar-claim values) re-address at 5, by the
   same shared-number rule entries 3 and 4 record.

   Unlike entries 3 and 4, the "nothing has been stored, so re-addressing carries
   no migration cost" premise is NO LONGER TRUE, and that was checked rather than
   assumed: a live workspace holds a v4 dataset (the flagship flame-speed sweep)
   AND a v4 condition set. The bump ORPHANS both -- the version gate below refuses
   to LOAD a v4 payload, fail-closed, so old bytes are refused rather than
   silently reinterpreted, which is the behaviour this key exists to produce. The
   operator authorised the bump knowing this; neither artifact is migrated or
   deleted (re-producing them under 5 is a separate, explicit run), so the
   orphaning is an accepted, recorded consequence, never a silent one.
6. :attr:`SourceNode.document_kind` added (I-075): a supplementary-member node
   (``SI_MEMBER``) may now DECLARE whether it is a PDF, a word-processor
   document, or a spreadsheet (:class:`SiMemberDocumentKind`), which is what
   lets V8 and V3 DECIDE a caption-labelled SI-member cell where a single
   ``SI_MEMBER`` kind forced them to refuse. Projected CONDITIONALLY like
   ``crop_region`` -- emitted only when a concrete kind is declared, omitted
   when the kind-determined absence holds (``NOT_APPLICABLE`` off a non-SI
   node, ``NOT_EXTRACTED_YET`` on an undeclared SI member), with the marker
   restored from the payload's ``kind`` by
   :func:`_restore_unprojected_document_kinds`. This changes only the NODE
   shape, but the version is a single shared projection-schema number, so BOTH
   a :class:`DatasetEnvelope` and a :class:`ConditionSetEnvelope` stamp 6, by
   the same rule entries 3-5 record.

   The migration reality here differs from entry 5's and was checked, not
   assumed: the two v4 artifacts entry 5 names were ALREADY orphaned by the v5
   gate and remain the only stored envelopes anywhere; NOTHING is stored at v5.
   So this 5->6 bump orphans nothing NEW -- re-addressing every projection
   carries no additional migration cost, and no artifact changes state that
   entry 5 did not already change.

A version-1 payload is REFUSED here, never migrated. Refusing is the whole
reason this key exists: a payload written under a projection this code has
never seen must fail with ONE message naming the version mismatch, rather
than reaching pydantic and producing a pile of field-level errors that name
a symptom (``crop_region Field required``, once per node, plus whatever
cascades behind it) instead of the cause. Inferring the missing field
instead -- ``NOT_APPLICABLE`` for the non-crop nodes, and a guess for the
crops -- would contradict that contract outright: it would silently
re-address stored envelopes under a schema their bytes never claimed.
Accepting version 1 again is therefore a deliberate migration decision
carrying its own explicit inverse, never a default that arrives by
omission."""


def _check_identity_payload_discriminator(
    payload: dict[str, Any], *, expected_envelope_type: str, class_name: str
) -> None:
    """Refuse a payload whose self-description does not name
    ``expected_envelope_type`` at :data:`_SUPPORTED_IDENTITY_PAYLOAD_VERSION`
    -- the gate both ``from_identity_payload`` classmethods run BEFORE any
    rehydration or model validation.

    The failure mode this closes: without a discriminator in the stored
    bytes, nothing says whether a payload is a dataset or a condition set,
    so a condition-set payload handed to
    :meth:`DatasetEnvelope.from_identity_payload` (or the reverse) is
    interpreted purely by field-shape luck -- today the two schemas happen
    to reject each other, but that is an accident of their current fields,
    not a guarantee, and the error it produces is an incomprehensible
    field-level ``ValidationError`` rather than the actual problem ("this
    is not a dataset at all"). Checking BEFORE model validation is the
    point: the type mismatch must be reported as a type mismatch, never
    laundered into (or, worse, silently absorbed by) field validation.

    The version check is the same refusal aimed at time rather than type: a
    payload projected under a future projection schema must be refused
    loudly, not parsed by a module that cannot know what its bytes mean.
    ``bool`` is explicitly excluded even though ``True == 1`` in Python --
    a payload carrying ``true`` was not written by any version of this
    projector, and equality alone would wave it through this gate.

    Raises:
        DatasetEnvelopeParseError: the ``envelope_type`` key is missing or
            names something other than ``expected_envelope_type``, or the
            ``identity_payload_version`` key is missing or is not exactly
            :data:`_SUPPORTED_IDENTITY_PAYLOAD_VERSION`. Every message
            names both what was expected and what was found.
    """
    if _ENVELOPE_TYPE_KEY not in payload:
        raise DatasetEnvelopeParseError(
            f"{class_name}.from_identity_payload: payload carries no {_ENVELOPE_TYPE_KEY!r} key -- "
            f"expected {_ENVELOPE_TYPE_KEY}={expected_envelope_type!r}, found none; a payload that "
            "does not say what it is cannot be trusted to be this envelope type"
        )
    found_type = payload[_ENVELOPE_TYPE_KEY]
    if found_type != expected_envelope_type:
        raise DatasetEnvelopeParseError(
            f"{class_name}.from_identity_payload: payload declares "
            f"{_ENVELOPE_TYPE_KEY}={found_type!r} but this parser expected "
            f"{expected_envelope_type!r} -- refusing to reinterpret one envelope type as another"
        )
    if _IDENTITY_PAYLOAD_VERSION_KEY not in payload:
        raise DatasetEnvelopeParseError(
            f"{class_name}.from_identity_payload: payload carries no "
            f"{_IDENTITY_PAYLOAD_VERSION_KEY!r} key -- expected version "
            f"{_SUPPORTED_IDENTITY_PAYLOAD_VERSION!r}, found none"
        )
    found_version = payload[_IDENTITY_PAYLOAD_VERSION_KEY]
    if isinstance(found_version, bool) or found_version != _SUPPORTED_IDENTITY_PAYLOAD_VERSION:
        raise DatasetEnvelopeParseError(
            f"{class_name}.from_identity_payload: payload declares "
            f"{_IDENTITY_PAYLOAD_VERSION_KEY}={found_version!r} but this parser supports exactly "
            f"version {_SUPPORTED_IDENTITY_PAYLOAD_VERSION!r} -- a payload projected under any "
            "other version cannot be parsed here without silently reinterpreting it"
        )


def _strip_identity_payload_discriminator(rehydrated: dict[str, Any]) -> dict[str, Any]:
    """Return ``rehydrated`` without the two discriminator keys, which are
    projection-only data: they are emitted by ``identity_payload()`` and
    verified by :func:`_check_identity_payload_discriminator`, but they are
    not model fields, and both envelope classes are ``extra="forbid"`` --
    handed through unstripped, ``model_validate`` would reject every
    well-formed payload. The stage-2 byte comparison still covers them,
    because the RE-projection puts them back before comparing against the
    original input."""
    return {
        key: value
        for key, value in rehydrated.items()
        if key not in (_ENVELOPE_TYPE_KEY, _IDENTITY_PAYLOAD_VERSION_KEY)
    }


def _restore_unprojected_crop_regions(rehydrated: dict[str, Any]) -> dict[str, Any]:
    """Put back the ``crop_region`` marker that
    :func:`_source_node_identity_payload` omits for a node holding no region
    -- the explicit inverse of the one entry in
    :data:`_CONDITIONALLY_PROJECTED_FIELDS`, run by both
    ``from_identity_payload`` classmethods between rehydration and
    ``model_validate``.

    THIS GUARD IS KEYED ON ``kind``, and that is a second, hand-maintained
    statement of I7's rule -- the projection side derives its condition from
    the field's VALUE (see :func:`_addresses_a_crop_region`), and this side
    cannot: it runs on a payload dict before validation, where the value it
    would consult is precisely the thing that is missing. ``kind`` is the
    only signal in those bytes. The pair is named in
    :data:`_CONDITIONALLY_PROJECTED_FIELDS`'s docstring as the two places
    that must move together, with the cost of their drifting: an
    error-message downgrade, never a wrong address.

    Needed because ``SourceNode.crop_region`` is required with NO default:
    a payload that legitimately omits the key would otherwise fail
    validation with ``Field required``, so the projection would be
    unparseable by its own inverse. Restoring it here rather than giving
    the field a pydantic default is the point -- a default would also
    accept a payload that omits the key on a CROP, silently manufacturing
    an address for a figure nobody located, whereas this restores exactly
    what the projection dropped and nothing else.

    NOT_APPLICABLE is a restoration, not an inference: I7 admits that one
    value and no other on a non-crop node, so the marker is fully
    determined by the ``kind`` the payload itself carries. A node whose
    payload says ``kind="figure_crop"`` is deliberately left alone -- if it
    is missing its region, that is a malformed payload, and I7's "cannot
    address itself" (or pydantic's ``Field required``, if the key is
    missing entirely) is the honest answer. Anything not shaped like a node
    list is also left untouched, so a malformed payload still fails where
    it would have failed, with its own error rather than one from here.

    A node that DOES carry the key on a non-crop kind is likewise left
    alone: it validates only if it holds Absent(NOT_APPLICABLE), and then
    fails the caller's byte-for-byte round-trip check, because
    re-projecting drops the key again. That is correct -- the projection
    shape is canonical, and a payload in a different shape addresses
    nothing this projector would ever have written.
    """
    graph = rehydrated.get("source_graph")
    if not isinstance(graph, dict):
        return rehydrated
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return rehydrated
    restored = [
        {**node, "crop_region": Absent(reason=AbsenceReason.NOT_APPLICABLE)}
        if isinstance(node, dict) and "crop_region" not in node and node.get("kind") != SourceNodeKind.FIGURE_CROP.value
        else node
        for node in nodes
    ]
    return {**rehydrated, "source_graph": {**graph, "nodes": restored}}


def _restore_unprojected_document_kinds(rehydrated: dict[str, Any]) -> dict[str, Any]:
    """Put back the ``document_kind`` marker that
    :func:`_source_node_identity_payload` omits for a node that DECLARES no
    kind -- the explicit inverse of the ``("SourceNode", "document_kind")``
    entry in :data:`_CONDITIONALLY_PROJECTED_FIELDS`, run by both
    ``from_identity_payload`` classmethods between rehydration and
    ``model_validate``, and the exact structural twin of
    :func:`_restore_unprojected_crop_regions` above.

    THIS SIDE AND THE PROJECTION SIDE ARE NOT ONE PIECE OF CODE, and must
    move together. The projection side derives its condition from the field's
    VALUE (see :func:`_declares_a_document_kind`) and cannot be reused here,
    because this runs on a payload dict before validation, where the value it
    would consult is precisely the marker that is missing. ``kind`` is the
    only signal left in those bytes, so this guard KEYS ON ``kind`` -- a
    second, hand-maintained statement of :meth:`SourceNode._validate_document_kind`'s
    rule. The two are named together in
    :data:`_CONDITIONALLY_PROJECTED_FIELDS`'s docstring as the pair that
    drifts together; the cost of their drifting is an error-message downgrade
    (a ``Field required`` in place of a targeted message), never a wrong
    address.

    Needed because ``SourceNode.document_kind`` is required with NO default:
    a payload that legitimately omits the key would otherwise fail with
    ``Field required``, so the projection would be unparseable by its own
    inverse. Restoring exactly what the projection dropped -- rather than
    giving the field a pydantic default -- is the point: a default would also
    swallow a payload that omits the key on a declared member, silently
    manufacturing an "undeclared" fact the author never recorded.

    The restored reason is fully determined by the payload's own ``kind``,
    which is why this is a restoration and not an inference:
    :meth:`SourceNode._validate_document_kind` admits exactly ``NOT_APPLICABLE``
    off a non-SI node and exactly ``NOT_EXTRACTED_YET`` on an SI member, so a
    node whose payload omits ``document_kind``:

    - with ``kind == "si_member"`` had an undeclared SI member, restored to
      ``Absent(NOT_EXTRACTED_YET)``;
    - with any other ``kind`` had the concept not apply, restored to
      ``Absent(NOT_APPLICABLE)``.

    A node that DOES carry the key, and anything not shaped like a node list,
    is left untouched -- a present declaration passes straight through, and a
    malformed payload still fails where it would have failed, with its own
    error rather than one manufactured here. A node carrying the key with a
    mismatched shape validates only if it is self-consistent and then fails
    the caller's byte-for-byte round-trip, because re-projecting drops the
    key again: the projection shape stays canonical.
    """
    graph = rehydrated.get("source_graph")
    if not isinstance(graph, dict):
        return rehydrated
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return rehydrated
    restored = [
        {
            **node,
            "document_kind": Absent(
                reason=AbsenceReason.NOT_EXTRACTED_YET
                if node.get("kind") == SourceNodeKind.SI_MEMBER.value
                else AbsenceReason.NOT_APPLICABLE
            ),
        }
        if isinstance(node, dict) and "document_kind" not in node
        else node
        for node in nodes
    ]
    return {**rehydrated, "source_graph": {**graph, "nodes": restored}}


class _SourceGraphEnvelope(Protocol):
    """Structural type for the provenance state shared by more than one
    concrete envelope class.

    Named for the two attributes the seven module-level helpers below
    actually read (``source_graph``, ``conversion_tables``), not for any
    particular envelope class: :class:`DatasetEnvelope` is today's only
    caller, but a second, unrelated envelope class is expected to call
    these same helpers directly. A nominal base class was deliberately
    rejected as the sharing mechanism -- a subclass silently inheriting a
    base ``identity_payload()`` that omits its own distinguishing field
    would let two different payloads collide on ONE address in a
    write-once immutable store -- so this Protocol, not inheritance, is
    what lets both envelope classes satisfy the same helper signatures.
    The walkers ``iter_source_refs``/``iter_measured_values`` themselves
    take ``object``, so passing the whole envelope to them is fine either
    way.
    """

    @property
    def source_graph(self) -> SourceGraph: ...

    @property
    def conversion_tables(self) -> tuple[EmbeddedConversionTable, ...]: ...

    @property
    def table_inventories(self) -> tuple[EmbeddedTableInventory, ...]: ...


# SHARED-PROVENANCE-VALIDATORS: the seven helpers below implement provenance logic that reads
# only `source_graph`, `conversion_tables`, and whatever `iter_source_refs`/`iter_measured_values`
# reach -- nothing series-specific. They are factored out to module level (rather than left as
# DatasetEnvelope methods) so a second, unrelated envelope class can call them directly without
# inheriting from DatasetEnvelope -- see `_SourceGraphEnvelope` above for why inheritance was
# rejected as the sharing mechanism.
def _validate_conversion_tables_cover_cited_tables(envelope: _SourceGraphEnvelope) -> None:
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
    cited = {value.conversion_table_sha256 for _, value in iter_measured_values(envelope)}
    embedded = {table.sha256 for table in envelope.conversion_tables}
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


def _validate_conversion_tables_no_duplicate_sha256(envelope: _SourceGraphEnvelope) -> None:
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
    for table in envelope.conversion_tables:
        if table.sha256 in seen:
            duplicates.add(table.sha256)
        seen.add(table.sha256)
    if duplicates:
        raise ValueError(
            f"DatasetEnvelope.conversion_tables embeds duplicate sha256(s) {sorted(duplicates)!r} -- "
            "each cited conversion table must be embedded exactly once"
        )


def _validate_conversion_tables_sorted(envelope: _SourceGraphEnvelope) -> None:
    """T3: ``conversion_tables`` must be sorted ascending by ``sha256``,
    so exactly one legal ordering exists -- matching the S2/S7/E1b idiom
    already in this module (a canonical order pins one, and only one,
    addressable representation)."""
    expected = tuple(sorted(envelope.conversion_tables, key=lambda table: table.sha256))
    if envelope.conversion_tables != expected:
        raise ValueError("DatasetEnvelope.conversion_tables must be sorted ascending by sha256")


def _validate_refs_resolve(envelope: _SourceGraphEnvelope) -> None:
    """V1: every embedded :class:`SourceRef` must name a node this
    envelope's ``source_graph`` actually contains.

    Without this, a ``SourceRef`` is just a free-floating claim -- it
    looks like provenance (it has a ``node_id`` and a locator) but there
    is no guarantee the node it names was ever validated, or even
    exists. This is the check that makes a ``SourceRef`` actually mean
    something.
    """
    node_ids = envelope.source_graph.node_ids
    for path, ref in iter_source_refs(envelope):
        if ref.node_id not in node_ids:
            raise ValueError(
                f"SourceRef at {path!r} names node_id={ref.node_id!r}, which is not present in "
                f"source_graph (known node ids: {sorted(node_ids)!r})"
            )


def _validate_no_decorative_nodes(envelope: _SourceGraphEnvelope) -> None:
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
    referenced_ids = {ref.node_id for _, ref in iter_source_refs(envelope)}
    covered_ids: set[str] = set()
    for node_id in referenced_ids:
        covered_ids.add(node_id)
        covered_ids.update(ancestor.node_id for ancestor in envelope.source_graph.ancestors(node_id))
    for node in envelope.source_graph.nodes:
        if node.node_id not in covered_ids:
            raise ValueError(
                f"node {node.node_id!r} is not targeted by any SourceRef, nor is it an ancestor of a "
                "targeted node -- an unreferenced node is decorative provenance that nothing in this "
                "envelope actually relies on"
            )


def _validate_locator_kind_compatibility(envelope: _SourceGraphEnvelope) -> None:
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
    :data:`_TABLE_KEY_KIND_COMPATIBLE_NODE_KINDS` for the static compatibility
    table, which keeps BOTH keys compatible with ``SI_MEMBER`` (it cannot
    express media type). I-075 adds a third axis below for a member that
    DECLARES its :attr:`SourceNode.document_kind`: a ``MemberSheetKey`` needs a
    spreadsheet's workbook sheets, a ``CaptionLabelKey`` needs a rendered
    document's printed caption. The "``SI_MEMBER`` too broad" gap that table
    documents thus CLOSES for declared members and stays open only for
    undeclared ones (where V8 still refuses a caption cell both ways).
    """
    for path, ref in iter_source_refs(envelope):
        node = envelope.source_graph.node(ref.node_id)
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
            # I-075: the static map above cannot express document kind, so it keeps BOTH
            # table keys compatible with SI_MEMBER -- the "SI_MEMBER too broad" gap, which
            # stays open for members that declare nothing. For a member that DOES declare
            # its document_kind, close the gap here: a MemberSheetKey needs workbook sheets
            # (only a spreadsheet has them) and a CaptionLabelKey needs a printed caption
            # (a spreadsheet has none). This runs before V8, so a spreadsheet caption cell
            # is stopped here and never reaches V8's finer PDF/word-processor distinction.
            if node.kind is SourceNodeKind.SI_MEMBER and not isinstance(node.document_kind, Absent):
                document_kind = node.document_kind
                is_spreadsheet = document_kind is SiMemberDocumentKind.SPREADSHEET
                if table_key_kind is TableKeyKind.MEMBER_SHEET and not is_spreadsheet:
                    raise ValueError(
                        f"SourceRef at {path!r} uses table_key kind={table_key_kind.value!r} against "
                        f"{document_kind.value!r} SI_MEMBER node {node.node_id!r}, but a MemberSheetKey names a "
                        "workbook sheet and only a spreadsheet has sheets"
                    )
                if table_key_kind is TableKeyKind.CAPTION_LABEL and is_spreadsheet:
                    raise ValueError(
                        f"SourceRef at {path!r} uses table_key kind={table_key_kind.value!r} against "
                        f"{document_kind.value!r} SI_MEMBER node {node.node_id!r}, but a CaptionLabelKey names a "
                        "printed caption and a spreadsheet has none"
                    )


def _validate_char_span_requires_extraction(envelope: _SourceGraphEnvelope) -> None:
    """V6: a :class:`CharSpanLocator` addresses a node's EXTRACTED TEXT,
    so its target node must actually HAVE one -- i.e.
    ``SourceNode.extraction`` must be present, not :class:`Absent`.

    Runs AFTER V1 (``_validate_refs_resolve``, declaration order), so
    every ``ref.node_id`` looked up here via ``envelope.source_graph.node``
    is already known to resolve.

    This rule is scoped to ``CHAR_SPAN`` ONLY -- it is deliberately NOT
    generalised to the other three locator kinds. A ``BBoxLocator``
    addresses the node's RENDERED RAW bytes, an ``XPathLocator``
    addresses its RAW XML, and a ``TableCellLocator`` addresses the
    document's own table structure; none of those three is a claim
    about extracted text, so requiring ``extraction`` for them would
    reject legitimate graphs (e.g. a bounding box against a PDF that was
    never text-extracted at all is still a real, verifiable locator).
    """
    for path, ref in iter_source_refs(envelope):
        if not isinstance(ref.locator, CharSpanLocator):
            continue
        node = envelope.source_graph.node(ref.node_id)
        if isinstance(node.extraction, Absent):
            raise ValueError(
                f"SourceRef at {path!r} uses a CharSpanLocator against node {node.node_id!r}, but "
                f"that node's extraction is Absent ({node.extraction.reason.value!r}) -- a character "
                "offset into text that was never extracted addresses nothing"
            )


def _validate_table_cell_inventory_citation(envelope: _SourceGraphEnvelope) -> None:
    """V8: every ``TABLE_CELL`` locator must state whether a PDF cell
    inventory defines the grid it indexes into -- and a PDF one must cite an
    inventory this envelope actually embeds, which actually contains the cell.

    Runs AFTER V1, so every ``ref.node_id`` already resolves.

    **Why the rule is keyed on node kind and table-key kind, and on nothing
    else.** The obvious discriminator -- ``SourceNode.extraction.extractor``
    beginning ``"pdf:"`` -- is WRONG twice over, and both ways were measured
    rather than reasoned:

    * ``pdf_fragments.extract_fragments(data: bytes)`` reads RAW BYTES. A PDF
      node whose text extraction is :class:`Absent` can still have a perfectly
      good inventory, so a rule keyed on ``extraction`` would classify exactly
      that node as "no inventory applies" -- turning the missing-extraction
      case into the bypass for the whole field.
    * ``"pdf:unavailable"`` also starts with ``"pdf:"`` and is emitted
      precisely when pypdf did NOT run (see
      ``extraction_record._PYPDF_DEPENDENT_EXTRACTORS``, which documents the
      same trap for its own purposes).

    Text-extraction provenance is simply not the authority on table-inventory
    provenance.

    The four cases, all fail-closed:

    * ``PAPER_PDF``: the citation MUST be present. **No** ``AbsenceReason`` is
      legal -- particularly not ``NOT_EXTRACTED_YET``, which is precisely the
      escape hatch that would make this field decorative.
    * ``JATS_XML``: MUST be ``Absent(NOT_APPLICABLE)``. XML table cells have
      no PDF fragment geometry, and no other absence reason is true.
    * ``SI_MEMBER`` + :class:`MemberSheetKey`: MUST be
      ``Absent(NOT_APPLICABLE)``. A workbook SHEET has no fragment geometry,
      structurally.
    * ``SI_MEMBER`` + :class:`CaptionLabelKey`: now BRANCHES on the member's
      declared :attr:`SourceNode.document_kind` (I-075) -- the media type the
      node kind alone could never carry, the gap
      :func:`_validate_locator_kind_compatibility` documents as "``SI_MEMBER``
      too broad":

      - declared ``WORD_PROCESSOR``: a word-processor cell has NO PDF fragment
        geometry, so ``Absent(NOT_APPLICABLE)`` is the true absence and is
        ACCEPTED (the case this change opens); any other absence reason, and a
        present citation, are refused exactly as any non-PDF node is.
      - declared ``PDF``: STAYS refused both ways, as strictly as the
        undeclared case was before. A PDF cell DOES have fragment geometry, so
        ``NOT_APPLICABLE`` is false; and a present citation cannot be verified
        here because :class:`EmbeddedTableInventory` never re-derives the grid
        from the document's bytes (see its "SCOPE OF WHAT VALIDATION HERE
        PROVES") -- the whole payload, ``raw_sha256`` included, is
        author-controlled, so citing one asserts a grid rather than
        establishing it. Only :func:`~carmel.services.pdf_table_record.
        verify_inventory_record`, holding the real bytes, can establish that;
        SI-member PDF-inventory verification is follow-on work.
      - UNDECLARED (``Absent(NOT_EXTRACTED_YET)``): STAYS refused both ways,
        and the message KEEPS the "this schema cannot tell which" explanation
        -- still literally true, because the member has not declared whether it
        is a PDF or a word-processor document. This is the guard the whole
        change rests on: an undeclared member must never pass as a declared
        one.
      - declared ``SPREADSHEET``: refused upstream by
        :func:`_validate_locator_kind_compatibility` (V3 runs before V8), which
        rejects a caption key against a spreadsheet; V8 refuses it too if
        reached, never letting it silently pass.

    This case was first written to ACCEPT a present citation, on the argument
    that it SELF-CERTIFIES: the checks below require the embedded record's
    ``raw_sha256`` to equal this node's ``sha256``, and only a PDF yields an
    inventory, so the envelope was said to PROVE the member is a PDF. That
    argument is false, and the refutation is worth keeping because it is easy
    to re-derive: the payload is author-controlled, so matching digests prove
    only that the author NAMED this node, never that a PDF parser ever ran on
    it. A DECLARED document kind is a different thing entirely -- a first-class
    stored fact the author stands behind, projected into the node's identity --
    which is why it, and not a self-certifying citation, is what lets this arm
    decide.
    """
    embedded_by_sha = {inventory.inventory_sha256: inventory for inventory in envelope.table_inventories}
    for path, ref in iter_source_refs(envelope):
        locator = ref.locator
        if not isinstance(locator, TableCellLocator):
            continue
        node = envelope.source_graph.node(ref.node_id)
        citation = locator.pdf_table_inventory_sha256
        # I-075: the caption-labelled SI-member case, which a single SI_MEMBER
        # kind forced this validator to REFUSE both ways, now BRANCHES on the
        # member's declared document_kind. All of its sub-cases are decided
        # inside the two `is_si_caption` blocks below -- each either accepts
        # (only a declared word-processor with the true absence) or raises with
        # a message naming the REAL reason -- so none fall through to the
        # generic non-PDF arms.
        is_si_caption = node.kind is SourceNodeKind.SI_MEMBER and locator.table_key.kind is TableKeyKind.CAPTION_LABEL
        requires_citation = node.kind is SourceNodeKind.PAPER_PDF

        if isinstance(citation, Absent):
            if requires_citation:
                raise ValueError(
                    f"SourceRef at {path!r} locates a table cell in PAPER_PDF node {node.node_id!r} but its "
                    f"pdf_table_inventory_sha256 is Absent ({citation.reason.value!r}) -- a PDF table cell "
                    "must cite the inventory that defines its grid, and no absence reason is legal here"
                )
            if is_si_caption:
                document_kind = node.document_kind
                if isinstance(document_kind, Absent):
                    raise ValueError(
                        f"SourceRef at {path!r} locates a caption-labelled table cell in SI_MEMBER node "
                        f"{node.node_id!r} with pdf_table_inventory_sha256 Absent ({citation.reason.value!r}) "
                        "-- the member has not declared its document_kind, so an SI member may be a PDF or a "
                        "word-processor document and this schema cannot tell which; absence here would record "
                        "a claim nothing checked. Declare document_kind to decide it."
                    )
                if document_kind is SiMemberDocumentKind.WORD_PROCESSOR:
                    if citation.reason is AbsenceReason.NOT_APPLICABLE:
                        continue
                    raise ValueError(
                        f"SourceRef at {path!r} locates a caption-labelled cell in word-processor SI_MEMBER "
                        f"node {node.node_id!r} with pdf_table_inventory_sha256 Absent "
                        f"({citation.reason.value!r}) -- a word-processor cell has no PDF fragment geometry, so "
                        f"its only true absence is {AbsenceReason.NOT_APPLICABLE.value!r}"
                    )
                if document_kind is SiMemberDocumentKind.PDF:
                    raise ValueError(
                        f"SourceRef at {path!r} locates a caption-labelled cell in PDF SI_MEMBER node "
                        f"{node.node_id!r} with pdf_table_inventory_sha256 Absent ({citation.reason.value!r}) "
                        f"-- a PDF cell HAS fragment geometry, so {AbsenceReason.NOT_APPLICABLE.value!r} is "
                        "false; verifying an SI-member PDF's cell inventory is follow-on work, so this cell "
                        "cannot yet be grounded either way"
                    )
                raise ValueError(
                    f"SourceRef at {path!r} locates a caption-labelled cell in spreadsheet SI_MEMBER node "
                    f"{node.node_id!r} -- a spreadsheet has no printed captions, so this should have been "
                    "refused upstream by _validate_locator_kind_compatibility (V3); it is refused here too"
                )
            if citation.reason is not AbsenceReason.NOT_APPLICABLE:
                raise ValueError(
                    f"SourceRef at {path!r} locates a table cell in {node.kind.value!r} node "
                    f"{node.node_id!r} with pdf_table_inventory_sha256 Absent "
                    f"({citation.reason.value!r}) -- the only true absence for a cell that has no PDF "
                    f"fragment geometry is {AbsenceReason.NOT_APPLICABLE.value!r}"
                )
            continue

        if is_si_caption:
            # Checked BEFORE the generic non-PDF rejection so the message names the real
            # reason for THIS member's declared kind rather than the old "cannot tell which".
            document_kind = node.document_kind
            if isinstance(document_kind, Absent):
                raise ValueError(
                    f"SourceRef at {path!r} locates a caption-labelled table cell in SI_MEMBER node "
                    f"{node.node_id!r} and cites pdf_table_inventory_sha256={citation!r}, but the member has "
                    "not declared its document_kind -- an SI member may be a PDF or a word-processor document "
                    "and this schema cannot tell which; an embedded record's raw_sha256 is author-controlled, "
                    "so citing one asserts the member is a PDF rather than establishing it"
                )
            if document_kind is SiMemberDocumentKind.PDF:
                raise ValueError(
                    f"SourceRef at {path!r} locates a caption-labelled cell in PDF SI_MEMBER node "
                    f"{node.node_id!r} and cites pdf_table_inventory_sha256={citation!r}, but an embedded "
                    "record's raw_sha256 is author-controlled, so citing one asserts a grid over this PDF "
                    "rather than establishing it; SI-member PDF-inventory verification is follow-on work"
                )
            if document_kind is SiMemberDocumentKind.WORD_PROCESSOR:
                raise ValueError(
                    f"SourceRef at {path!r} locates a caption-labelled cell in word-processor SI_MEMBER node "
                    f"{node.node_id!r} and cites pdf_table_inventory_sha256={citation!r}, but a word-processor "
                    "cell has no PDF fragment geometry a cell inventory could ever describe"
                )
            raise ValueError(
                f"SourceRef at {path!r} locates a caption-labelled cell in spreadsheet SI_MEMBER node "
                f"{node.node_id!r} and cites pdf_table_inventory_sha256={citation!r} -- a spreadsheet has no "
                "printed captions, so this should have been refused upstream by "
                "_validate_locator_kind_compatibility (V3); it is refused here too"
            )

        if not requires_citation:
            # Checked BEFORE the cover check below, deliberately: an XML or workbook cell
            # citing a real, embedded PDF inventory would otherwise contribute to the cited
            # set and sail through exact cover, laundering a PDF grid into a node that has
            # none.
            raise ValueError(
                f"SourceRef at {path!r} locates a table cell in {node.kind.value!r} node {node.node_id!r} "
                f"and cites pdf_table_inventory_sha256={citation!r}, but a cell in that node has no PDF "
                "fragment geometry a cell inventory could ever describe"
            )

        inventory = embedded_by_sha.get(citation)
        if inventory is None:
            raise ValueError(
                f"SourceRef at {path!r} cites pdf_table_inventory_sha256={citation!r}, which this envelope "
                "does not embed -- a citation resolvable only against an evidence store makes the "
                "envelope's meaning depend on the machine replaying it"
            )
        if inventory.raw_sha256 != node.sha256:
            raise ValueError(
                f"SourceRef at {path!r} cites inventory {citation!r}, whose record was derived from "
                f"document {inventory.raw_sha256!r}, but its node {node.node_id!r} has "
                f"sha256={node.sha256!r} -- the grid describes a different document than the one the "
                "locator targets"
            )
        if not inventory.has_cell(row=locator.row, col=locator.col):
            raise ValueError(
                f"SourceRef at {path!r} locates row={locator.row}, col={locator.col} in inventory "
                f"{citation!r}, whose grid has no such cell -- a citation that resolves to a real "
                "inventory of the right document can still name a cell that grid never derived"
            )


def _validate_table_inventories_cover_cited_inventories(envelope: _SourceGraphEnvelope) -> None:
    """T4: ``table_inventories`` must cover EXACTLY the inventories cited by
    the ``TABLE_CELL`` locators reachable in this envelope -- no fewer, no
    more.

    The "no fewer" half is enforced per-ref by V8 above (which reports WHICH
    ref dangles, information a set difference cannot give); this states the
    same requirement over the whole envelope and adds the "no more" half. An
    embedded inventory nothing cites is unearned provenance -- the same
    failure class T2 closes for conversion tables and V2 for graph nodes.

    The cited set is computed from locators whose citation is PRESENT, so this
    rule ALONE says less than it appears to: an envelope whose PDF locators all
    carry ``Absent`` contributes an empty cited set and passes exact cover
    against an empty embedded set while citing nothing at all. V8 is what makes
    that envelope illegal.

    Declaration order (V8 first) is therefore about the DIAGNOSTIC, not the
    verdict -- measured, not assumed: pydantic runs ``mode="after"`` model
    validators in declaration order, and inverting these two leaves every test
    in ``test_table_cell_inventory_citation`` green. Both validators are pure
    and neither mutates the model, so an envelope T4 accepts vacuously is still
    rejected by V8 whichever runs first. What order buys is that an envelope
    violating both reports the missing citation rather than a confusing cover
    complaint. Do not weaken either rule on the theory that the other one
    ordered before it makes the check redundant.
    """
    cited = {
        ref.locator.pdf_table_inventory_sha256
        for _, ref in iter_source_refs(envelope)
        if isinstance(ref.locator, TableCellLocator) and isinstance(ref.locator.pdf_table_inventory_sha256, str)
    }
    embedded = {inventory.inventory_sha256 for inventory in envelope.table_inventories}
    missing = cited - embedded
    if missing:
        raise ValueError(
            f"table_inventories is missing inventory(ies) {sorted(missing)!r} cited by a TableCellLocator "
            "-- every cited inventory must be embedded"
        )
    decorative = embedded - cited
    if decorative:
        raise ValueError(
            f"table_inventories embeds decorative inventory(ies) {sorted(decorative)!r} that no "
            "TableCellLocator cites -- an embedded record nothing needs is unearned provenance"
        )


def _validate_table_inventories_no_duplicate_sha256(envelope: _SourceGraphEnvelope) -> None:
    """DUPLICATE-TABLE-INVENTORY-SHA256-GUARD: ``table_inventories`` must not
    embed the same ``inventory_sha256`` twice.

    Exactly the gap the conversion-table duplicate guard closes, for exactly
    the same reason: T4 compares SETS, so ``(I1, I1)`` has the same embedded
    set as ``(I1,)``, and T5's adjacent-pair sort check accepts two equal,
    already-sorted entries. Two envelopes with the same logical content and
    different bytes address differently -- an address-uniqueness bug.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for inventory in envelope.table_inventories:
        if inventory.inventory_sha256 in seen:
            duplicates.add(inventory.inventory_sha256)
        seen.add(inventory.inventory_sha256)
    if duplicates:
        raise ValueError(
            f"table_inventories embeds duplicate inventory_sha256(s) {sorted(duplicates)!r} -- each cited "
            "inventory must be embedded exactly once"
        )


def _validate_table_inventories_sorted(envelope: _SourceGraphEnvelope) -> None:
    """T5: ``table_inventories`` must be sorted ascending by
    ``inventory_sha256``, so exactly one legal ordering -- and therefore
    exactly one addressable representation -- exists. Same idiom as T3."""
    expected = tuple(sorted(envelope.table_inventories, key=lambda inventory: inventory.inventory_sha256))
    if envelope.table_inventories != expected:
        raise ValueError("table_inventories must be sorted ascending by inventory_sha256")


def _validate_series_digitization_citation(envelope: DatasetEnvelope) -> None:
    """V9: every ``DIGITIZED`` :class:`Series` must cite a figure digitization
    this envelope actually embeds, whose record is about THAT series, over THAT
    crop, chained to the document's raw bytes -- and every non-digitized series
    must cite none.

    The figure lane's counterpart to :func:`_validate_table_cell_inventory_citation`,
    matched to its shape rather than reinvented. It closes, at series level, the
    hole :class:`EmbeddedFigureDigitization` was built to close at record level:
    a series whose ``source_form`` is :attr:`SourceForm.DIGITIZED` reports only
    ``len(points)``, so without this join it could be stored with no digitization
    record and no partialness at all.

    Runs AFTER V1 (``_validate_refs_resolve``) so every point-value ``ref.node_id``
    already resolves, and AFTER V4 (``_validate_source_form_constrains_value_refs``)
    so a value ref that is not a ``FIGURE_CROP`` at all is reported by V4's precise
    message before this validator's same-crop join speaks.

    Two cases, both fail-closed:

    * ``DIGITIZED``: the citation MUST be present. An ``Absent`` here is exactly
      the hole -- a digitized series that cites nothing. The cited address must be
      embedded (a citation resolvable only against an evidence store makes the
      envelope's meaning depend on the replaying machine), and the embedded
      record must pass four producer checks and two further joins (below).
    * every other form: the citation MUST be ``Absent(NOT_APPLICABLE)``. A present
      citation on a ``TABULAR``/``TEXTUAL`` series would widen the requirement
      beyond digitized series; any other absence reason would record a claim
      nothing checked.

    The FOUR producer checks -- enforced here, not left for a producer to remember:

    * ``record.series_id == series.series_id`` -- the record is about THIS series.
    * ``record.recovered == len(series.points)`` -- the count the record's census
      balances against (D9) is the count the series actually has.
    * ``record.figure_crop_node_id`` resolves to a ``FIGURE_CROP`` node -- not to
      a page, a table, or an SI member, and not to nothing at all.
    * ``record.figure_crop_sha256 == that node's sha256`` -- the two halves of
      the crop's identity agree, so the record names one crop rather than one
      crop's id and another's bytes.

    The TWO joins beyond the obvious one, both required:

    * **One crop.** Every digitized point's value ref must target the SAME crop
      the record names. A series assembled from two different crops is not one
      digitization, and a record can only describe one.
    * **Root bytes.** ``record.raw_sha256`` must equal the crop's ROOT artifact's
      ``sha256`` -- the parentless node the crop descends from (a ``PAPER_PDF`` or
      ``JATS_XML``), not merely the crop's immediate parent. This is what makes
      the chain reach actual document bytes rather than stopping at an
      intermediate artifact (an SI member a crop was cut from is itself derived).

    **A STATED CEILING.** Everything this join proves is that the digitized
    coordinates were RECORDED and are UNALTERED: the record's canonical bytes
    hash to the cited address (T1), its census balances against a count this
    series really has, and the chain reaches the document's raw bytes. It proves
    NOTHING about whether those coordinates are TRUE -- nothing here re-derives
    markers from the crop's pixels, so replay can confirm a record was faithfully
    stored and never that the curve was traced accurately. A verified record is
    not a validated measurement; see :class:`EmbeddedFigureDigitization`'s "SCOPE
    OF WHAT VALIDATION HERE PROVES".
    """
    embedded_by_sha = {digitization.digitization_sha256: digitization for digitization in envelope.figure_digitizations}
    for series in envelope.series:
        citation = series.digitization_sha256
        if series.source_form is not SourceForm.DIGITIZED:
            if not isinstance(citation, Absent):
                raise ValueError(
                    f"Series(series_id={series.series_id!r}) has source_form={series.source_form.value!r} "
                    f"but cites digitization_sha256={citation!r} -- only a DIGITIZED series may cite a "
                    "figure digitization, and widening the requirement to any other form would record a "
                    "citation nothing constrains"
                )
            if citation.reason is not AbsenceReason.NOT_APPLICABLE:
                raise ValueError(
                    f"Series(series_id={series.series_id!r}) has source_form={series.source_form.value!r} "
                    f"and digitization_sha256 Absent ({citation.reason.value!r}) -- the only true absence for "
                    f"a series that was not digitized is {AbsenceReason.NOT_APPLICABLE.value!r}"
                )
            continue

        if isinstance(citation, Absent):
            raise ValueError(
                f"Series(series_id={series.series_id!r}) has source_form=DIGITIZED but its "
                f"digitization_sha256 is Absent ({citation.reason.value!r}) -- a digitized series must cite "
                "the figure digitization that recovered it, so its partialness is stated rather than "
                "silently absent, and no absence reason is legal here"
            )

        digitization = embedded_by_sha.get(citation)
        if digitization is None:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization_sha256={citation!r}, which this "
                "envelope does not embed -- a citation resolvable only against an evidence store makes the "
                "envelope's meaning depend on the machine replaying it"
            )
        record = digitization._validated_record()

        if record.series_id != series.series_id:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose record is "
                f"about series {record.series_id!r} -- a digitization must name the series it recovered"
            )
        if record.recovered != len(series.points):
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose record "
                f"recovered {record.recovered} marker(s), but the series has {len(series.points)} point(s) "
                "-- the count the record's census balances against must be the count the series actually has"
            )
        if record.figure_crop_node_id not in envelope.source_graph.node_ids:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose record names "
                f"figure_crop_node_id={record.figure_crop_node_id!r}, which is not the id of any node in this "
                "envelope's source graph -- a digitization must name a crop this envelope actually carries"
            )
        crop_node = envelope.source_graph.node(record.figure_crop_node_id)
        if crop_node.kind is not SourceNodeKind.FIGURE_CROP:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose "
                f"figure_crop_node_id {record.figure_crop_node_id!r} resolves to a {crop_node.kind.value!r} "
                "node, not a FIGURE_CROP -- a digitization is read off a figure crop and off nothing else"
            )
        if crop_node.sha256 != record.figure_crop_sha256:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose record names "
                f"figure_crop_sha256={record.figure_crop_sha256!r}, but crop node "
                f"{record.figure_crop_node_id!r} has sha256={crop_node.sha256!r} -- the two halves of the "
                "crop's identity must agree, so the record names one crop rather than one crop's id and "
                "another's bytes"
            )

        for where, ref in _iter_series_point_value_refs(series):
            if ref.node_id != record.figure_crop_node_id:
                raise ValueError(
                    f"Series(series_id={series.series_id!r}) cites digitization {citation!r} over crop "
                    f"{record.figure_crop_node_id!r}, but {where} grounds in crop {ref.node_id!r} -- a "
                    "series assembled from two different crops is not one digitization, and one record can "
                    "describe only one crop"
                )

        ancestors = envelope.source_graph.ancestors(record.figure_crop_node_id)
        root_node = ancestors[-1] if ancestors else crop_node
        if record.raw_sha256 != root_node.sha256:
            raise ValueError(
                f"Series(series_id={series.series_id!r}) cites digitization {citation!r}, whose record was "
                f"derived from document raw_sha256={record.raw_sha256!r}, but crop "
                f"{record.figure_crop_node_id!r} descends from root artifact {root_node.node_id!r} with "
                f"sha256={root_node.sha256!r} -- the record's document digest must reach the crop's root PDF "
                "bytes rather than stopping at an intermediate artifact"
            )


def _iter_series_point_value_refs(series: Series) -> Iterator[tuple[str, SourceRef]]:
    """Yield ``(where, value_ref)`` for every point coordinate and present point
    observation of ``series`` -- the exact scope V4
    (:func:`_check_source_form_for_ref`) constrains, and the scope V9's same-crop
    join checks. A whole-series ``constant`` is deliberately excluded, for the
    same reason V4 excludes it: a constant is not a point reference, so it makes
    no per-point crop claim."""
    for point in series.points:
        for coord in point.coordinates:
            yield (
                f"Series(series_id={series.series_id!r}) point {point.point_id!r} coordinate axis_id={coord.axis_id!r}",
                coord.value.value_ref,
            )
        for obs in point.observations:
            if isinstance(obs.value, Absent):
                continue
            yield (
                f"Series(series_id={series.series_id!r}) point {point.point_id!r} observation axis_id={obs.axis_id!r}",
                obs.value.value_ref,
            )


def _validate_figure_digitizations_cover_cited(envelope: DatasetEnvelope) -> None:
    """FD1: ``figure_digitizations`` must cover EXACTLY the digitizations cited
    by the ``DIGITIZED`` series in this envelope -- no fewer, no more.

    The "no fewer" half is enforced per-series by V9 above (which reports WHICH
    series dangles, information a set difference cannot give); this states the
    same requirement over the whole envelope and adds the "no more" half. An
    embedded digitization nothing cites is unearned provenance -- the same
    failure class T4 closes for table inventories. Same declaration-order note as
    T4: V9 first, so an envelope violating both reports the missing citation."""
    cited = {series.digitization_sha256 for series in envelope.series if isinstance(series.digitization_sha256, str)}
    embedded = {digitization.digitization_sha256 for digitization in envelope.figure_digitizations}
    missing = cited - embedded
    if missing:
        raise ValueError(
            f"figure_digitizations is missing digitization(s) {sorted(missing)!r} cited by a DIGITIZED "
            "series -- every cited digitization must be embedded"
        )
    decorative = embedded - cited
    if decorative:
        raise ValueError(
            f"figure_digitizations embeds decorative digitization(s) {sorted(decorative)!r} that no series "
            "cites -- an embedded record nothing needs is unearned provenance"
        )


def _validate_figure_digitizations_no_duplicate_sha256(envelope: DatasetEnvelope) -> None:
    """FD2: ``figure_digitizations`` must not embed the same
    ``digitization_sha256`` twice -- the same gap the table-inventory duplicate
    guard closes, for the same reason: FD1 compares SETS and FD3's adjacent-pair
    sort accepts two equal entries, so two envelopes with the same logical
    content and different bytes would address differently."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for digitization in envelope.figure_digitizations:
        if digitization.digitization_sha256 in seen:
            duplicates.add(digitization.digitization_sha256)
        seen.add(digitization.digitization_sha256)
    if duplicates:
        raise ValueError(
            f"figure_digitizations embeds duplicate digitization_sha256(s) {sorted(duplicates)!r} -- each "
            "cited digitization must be embedded exactly once"
        )


def _validate_figure_digitizations_sorted(envelope: DatasetEnvelope) -> None:
    """FD3: ``figure_digitizations`` must be sorted ascending by
    ``digitization_sha256``, so exactly one legal ordering -- and therefore
    exactly one addressable representation -- exists. Same idiom as T5."""
    expected = tuple(sorted(envelope.figure_digitizations, key=lambda digitization: digitization.digitization_sha256))
    if envelope.figure_digitizations != expected:
        raise ValueError("figure_digitizations must be sorted ascending by digitization_sha256")


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

    PARTLY LIVE: a ``DatasetEnvelope`` is now constructed from a TABLE by
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
    whose series values are ``TABLE_CELL`` locators. The CHAR-SPAN
    ``produce_envelope_from_artifact``
    (:mod:`carmel.services.dataset_producer`) still refuses
    unconditionally -- read its docstring for the full argument,
    which this note only summarizes. That char-span route can only locate a value
    with a :class:`CharSpanLocator` into extracted running text, and V7
    below (:meth:`_validate_no_char_span_grounds_a_series_value`) rejects a
    char span as the source of a series VALUE, because a series asserts a
    structured pairing of coordinates to observations and running text
    carries no row structure from which that pairing can be proven. Since
    every series this producer could emit is grounded by char spans alone,
    V7 closes the only route this runtime has to a populated ``series``
    field, and ``series`` is required with ``min_length=1`` (see its field
    docstring above) -- so the CHAR-SPAN route reaches no populated
    ``series`` field at all. The TABULAR route is open: the table parser
    emits ``TABLE_CELL`` locators, so
    :func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`
    constructs a populated ``DatasetEnvelope`` from a table. The DIGITIZED
    (figure) route still needs a ``FIGURE_CROP`` digitizer that does not
    exist. The replayer and every validator below exercise this class, and
    the tabular producer now fills it -- not dead code, and no longer dormant.
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
    table_inventories: tuple[EmbeddedTableInventory, ...]
    """Every PDF cell inventory cited (by
    :attr:`TableCellLocator.pdf_table_inventory_sha256`) by any
    :class:`SourceRef` reachable in this envelope, embedded verbatim -- see
    :class:`EmbeddedTableInventory`, and V8/T4/T5 for the invariants.

    Present on THIS envelope as well as on :class:`ConditionSetEnvelope`
    because a :class:`Series` point can cite a table cell too (V4 in fact
    REQUIRES it: ``source_form=TABULAR`` demands a ``TABLE_CELL`` locator).
    Embedding on the condition-set side alone would leave this class as the
    bypass surface for the whole citation rule."""
    figure_digitizations: tuple[EmbeddedFigureDigitization, ...]
    """Every figure digitization cited (by :attr:`Series.digitization_sha256`)
    by any ``DIGITIZED`` :class:`Series` in this envelope, embedded verbatim --
    see :class:`EmbeddedFigureDigitization`, and V9/FD1/FD2/FD3 for the
    invariants.

    The figure lane's counterpart to :attr:`table_inventories`, and present
    ONLY here, not on :class:`ConditionSetEnvelope`: a condition set carries no
    :class:`Series`, so no digitized series can exist there to cite one. That is
    the whole difference from the table lane, where a condition-set claim CAN
    cite a table cell and so ``table_inventories`` must live on both. Embedded,
    rather than resolvable only against an evidence store, for the same reason
    every other embedded record here is: a citation the replaying machine cannot
    resolve from the envelope's own bytes makes the envelope's meaning depend on
    a directory that may have been pruned or never written."""

    @model_validator(mode="after")
    def _validate_conversion_tables_cover_cited_tables(self) -> DatasetEnvelope:
        """T2: see :func:`_validate_conversion_tables_cover_cited_tables`."""
        _validate_conversion_tables_cover_cited_tables(self)
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_no_duplicate_sha256(self) -> DatasetEnvelope:
        """See :func:`_validate_conversion_tables_no_duplicate_sha256`."""
        _validate_conversion_tables_no_duplicate_sha256(self)
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_sorted(self) -> DatasetEnvelope:
        """T3: see :func:`_validate_conversion_tables_sorted`."""
        _validate_conversion_tables_sorted(self)
        return self

    @model_validator(mode="after")
    def _validate_refs_resolve(self) -> DatasetEnvelope:
        """V1: see :func:`_validate_refs_resolve`."""
        _validate_refs_resolve(self)
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
        """V2: see :func:`_validate_no_decorative_nodes`."""
        _validate_no_decorative_nodes(self)
        return self

    @model_validator(mode="after")
    def _validate_locator_kind_compatibility(self) -> DatasetEnvelope:
        """V3: see :func:`_validate_locator_kind_compatibility`."""
        _validate_locator_kind_compatibility(self)
        return self

    @model_validator(mode="after")
    def _validate_char_span_requires_extraction(self) -> DatasetEnvelope:
        """V6: see :func:`_validate_char_span_requires_extraction`."""
        _validate_char_span_requires_extraction(self)
        return self

    @model_validator(mode="after")
    def _validate_table_cell_inventory_citation(self) -> DatasetEnvelope:
        """V8: see :func:`_validate_table_cell_inventory_citation`. Declared
        BEFORE T4 below so that an envelope violating both reports the missing
        citation, which is the actionable half -- see T4's docstring for why
        that is a diagnostic preference and not a soundness requirement."""
        _validate_table_cell_inventory_citation(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_cover_cited_inventories(self) -> DatasetEnvelope:
        """T4: see :func:`_validate_table_inventories_cover_cited_inventories`."""
        _validate_table_inventories_cover_cited_inventories(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_no_duplicate_sha256(self) -> DatasetEnvelope:
        """See :func:`_validate_table_inventories_no_duplicate_sha256`."""
        _validate_table_inventories_no_duplicate_sha256(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_sorted(self) -> DatasetEnvelope:
        """T5: see :func:`_validate_table_inventories_sorted`."""
        _validate_table_inventories_sorted(self)
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
    def _validate_no_char_span_grounds_a_series_value(self) -> DatasetEnvelope:
        """V7 (P0-c): a series data point's VALUE may not be located by a
        :class:`CharSpanLocator`.

        A char span in running text CAN support a prose-local scalar
        statement -- that is what :class:`ConditionSetEnvelope` exists for --
        but it cannot support a series DATA POINT. A series asserts a
        structured pairing of coordinates to observations, and running text
        carries no row structure from which that pairing can be proven. What
        made this a live fabrication rather than a theoretical one: pypdf
        renders a figure's axis furniture (tick labels, axis titles) into
        ``text.txt`` as ordinary body prose, so grounding a coordinate at the
        tick ``0.7`` and an observation at the tick ``24`` succeeded, produced
        a schema-valid envelope, and replayed ``VERIFIED`` with zero findings.
        Every quote really was an exact, located substring of a verified
        document; grounding proves LOCATION and never MEANING, and nothing in
        that datum was a datum.

        **The rule is about the LOCATOR, not about ``source_form``.** Banning
        ``source_form=TEXTUAL`` outright was the first attempt and it
        over-reached: ``TEXTUAL`` is also the only form compatible with an
        ``XPATH`` value ref, and an XPath into a JATS ``<td>`` is exactly the
        structured pairing evidence a char span lacks. Refusing the char span
        refuses the thing that cannot prove a pairing, and leaves every
        locator that can (``TABLE_CELL``, ``XPATH``, ``BBOX``,
        ``FIGURE_CROP``) alone.

        **This is a statement about THIS RUNTIME, not about prose.** Prose can
        carry a real series -- "At 300, 400 and 500 K the rates were 1.2, 2.4
        and 4.8 s-1, respectively" is one. What cannot be done here is PROVE
        the pairing from a char span, so the refusal is worded as incapacity;
        a later reader who finds the counterexample should conclude the
        runtime is limited, not that the rule was wrong.

        Constrains only a POINT's ``Coordinate.value`` and a present
        ``Observation.value`` -- the same scope as V4, and for the same
        reason: this is about where a DATA POINT's number was read from.
        ``unit_ref``, ``label_ref`` and every uncertainty ref may still be
        char spans, because a series legitimately takes its unit spelling and
        axis label from running prose.

        ``Series.constants`` is deliberately NOT covered, and this is a
        decision rather than an oversight (it was originally an oversight;
        spar r94 caught it). A whole-series constant -- "the pressure was held
        at 1 atm throughout" -- is precisely a prose-local SCALAR statement,
        which is the thing a char span CAN support; it makes no
        coordinate/observation pairing claim, so the argument above does not
        reach it. Composition refs are excluded for the same reason. If that
        ever changes, ``test_a_char_span_constant_is_deliberately_still_allowed``
        is the test that must be confronted first.

        Deliberately form-agnostic rather than an extra clause on V4's
        ``TEXTUAL`` branch: today only ``TEXTUAL`` can carry a ``CHAR_SPAN``
        value (``TABULAR`` demands ``TABLE_CELL``, and ``CHAR_SPAN`` is not
        compatible with the ``FIGURE_CROP`` node ``DIGITIZED`` demands), but a
        rule buried in one branch is a rule a newly added ``SourceForm`` can
        forget.
        """
        for series in self.series:
            for point in series.points:
                values = [(coord.axis_id, "coordinate", coord.value) for coord in point.coordinates]
                values += [
                    (obs.axis_id, "observation", obs.value)
                    for obs in point.observations
                    if not isinstance(obs.value, Absent)
                ]
                for axis_id, cell_kind, value in values:
                    if value.value_ref.locator.kind is LocatorKind.CHAR_SPAN:
                        raise ValueError(
                            f"DatasetEnvelope: Series(series_id={series.series_id!r}) point "
                            f"{point.point_id!r} {cell_kind} axis_id={axis_id!r} grounds its value "
                            "in a CHAR_SPAN, but a char span in running text cannot ground a series "
                            "data point -- a series asserts a coordinate/observation pairing that "
                            "running text carries no structure to prove (a figure's axis ticks "
                            "extract as ordinary body prose and would ground perfectly). This is a "
                            "limit of what this runtime can prove, not a claim that prose never "
                            "states a series: locate series values by TABLE_CELL, XPATH or BBOX, "
                            "and put a prose-local scalar statement in a ConditionSetEnvelope"
                        )
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

    @model_validator(mode="after")
    def _validate_series_digitization_citation(self) -> DatasetEnvelope:
        """V9: see :func:`_validate_series_digitization_citation`. Declared
        AFTER V1 (refs resolve) and V4 (source_form constrains value refs) so a
        value ref that is not a FIGURE_CROP at all is reported by V4's precise
        message, and before FD1 below so an envelope violating both reports the
        missing citation, which is the actionable half."""
        _validate_series_digitization_citation(self)
        return self

    @model_validator(mode="after")
    def _validate_figure_digitizations_cover_cited(self) -> DatasetEnvelope:
        """FD1: see :func:`_validate_figure_digitizations_cover_cited`."""
        _validate_figure_digitizations_cover_cited(self)
        return self

    @model_validator(mode="after")
    def _validate_figure_digitizations_no_duplicate_sha256(self) -> DatasetEnvelope:
        """FD2: see :func:`_validate_figure_digitizations_no_duplicate_sha256`."""
        _validate_figure_digitizations_no_duplicate_sha256(self)
        return self

    @model_validator(mode="after")
    def _validate_figure_digitizations_sorted(self) -> DatasetEnvelope:
        """FD3: see :func:`_validate_figure_digitizations_sorted`."""
        _validate_figure_digitizations_sorted(self)
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

        The payload is SELF-DESCRIBING: it carries ``envelope_type`` and
        ``identity_payload_version`` alongside the model fields, so the
        stored bytes themselves say what they are and under which
        projection schema they were written. Without those keys, nothing in
        a stored payload distinguishes a dataset from a condition set --
        the wrong parser could only be caught by field-shape accident --
        and no future projection change could ever be told apart from
        corruption. Both keys are verified (and stripped) by
        ``from_identity_payload`` before model validation.
        """
        return {
            _ENVELOPE_TYPE_KEY: _DATASET_ENVELOPE_TYPE,
            _IDENTITY_PAYLOAD_VERSION_KEY: _SUPPORTED_IDENTITY_PAYLOAD_VERSION,
            "source_graph": _source_graph_identity_payload(self.source_graph),
            "composition": _project_maybe(self.composition, _composition_identity_payload),
            "series": [_series_identity_payload(series) for series in self.series],
            "conversion_tables": [
                _embedded_conversion_table_identity_payload(table) for table in self.conversion_tables
            ],
            "table_inventories": [
                _embedded_table_inventory_identity_payload(inventory) for inventory in self.table_inventories
            ],
            "figure_digitizations": [
                _embedded_figure_digitization_identity_payload(digitization)
                for digitization in self.figure_digitizations
            ],
        }

    @classmethod
    def from_identity_payload(cls, payload: dict[str, Any]) -> DatasetEnvelope:
        """Reconstruct a :class:`DatasetEnvelope` from its own
        :meth:`identity_payload` projection -- the exact inverse of that
        method.

        This is a two-stage parse, and both stages are load-bearing:

        1. **Rehydrate.** ``identity_payload()`` is not pydantic's dump
           shape: an ``Absent`` marker projects to a plain
           ``{"__absent__": True, "reason": ..., "note": ...}`` dict, and
           ``Absent`` is ``extra="forbid"`` so that dict cannot be handed
           to ``model_validate`` as-is. This stage walks the payload and
           replaces every such marker with a real ``Absent`` instance
           (see :func:`_rehydrate_identity_payload`); everything else --
           other dicts, lists, scalars -- is left structurally alone,
           because pydantic itself already knows how to turn an enum's
           ``.value`` string back into the enum and a list back into a
           tuple. This stage also restores the one CONDITIONAL key the
           projection omits -- ``crop_region``, dropped for a node holding
           no region, which by I7 is any node that is not a ``FIGURE_CROP``
           (see :func:`_restore_unprojected_crop_regions` and
           :data:`_CONDITIONALLY_PROJECTED_FIELDS`) -- which is what makes
           that omission a projection SHAPE rather than a field this
           parser cannot recover.
        2. **Validate, then prove the parse.** The rehydrated structure is
           handed to ``cls.model_validate``, and the *resulting* envelope
           is immediately re-projected via its own ``identity_payload()``
           and compared, byte-for-byte (via
           :func:`carmel.services.dataset_store.canonical_json_bytes`),
           against the original input ``payload``.

           This second half of stage 2 is not a redundant belt-and-braces
           check on top of ``model_validate`` -- it is the only thing that
           actually proves this method is the inverse of
           ``identity_payload()``, because ``identity_payload()`` *is* the
           definition of this envelope's identity (it is exactly what
           ``compute_dataset_sha`` hashes). ``model_validate`` succeeding
           only proves the rehydrated structure satisfies this schema's
           field types and cross-field invariants; it does not prove
           nothing was lost or silently reinterpreted along the way. The
           clearest case is a dict that looks like a present value but
           happens to also validate as ``Absent`` under pydantic's "smart"
           union mode (e.g. a mangled ``ArchiveOrigin`` payload missing its
           required field, which is a legal ``Absent`` shape once
           ``__absent__`` is disregarded) -- ``model_validate`` would
           accept it silently, changing what dataset this payload
           addresses, and only the round-trip byte comparison below
           catches that.

           If the byte comparison fails, the raised error says the payload
           did not survive a round trip and names it as a disagreement
           between this parser and ``identity_payload()`` -- never as a
           claim that the input ``payload`` itself is corrupt, because
           nothing here has evidence of that; all that is known is that
           the parser and the projector disagree about what the payload
           means.

        **This inverse is exact on IDENTITY, and lossy on everything
        else.** ``result == original_envelope`` is NOT guaranteed, and in
        general is false: any field registered in
        :data:`_UNADDRESSED_FIELDS` is deliberately never projected, so it
        cannot possibly come back. Today that is exactly one field --
        :attr:`ArchiveOrigin.member_display_path` -- so an envelope stored
        carrying ``member_display_path="si/data.csv"`` parses back with
        that field as ``None``, silently.

        Silently, and correctly, because the guarantee this method offers
        is the one a content-addressed store actually needs: the returned
        envelope ADDRESSES THE SAME DATASET as the one that was stored, and
        stage 2 proves exactly that, byte for byte. It does not offer
        object equality, and callers must not test for it. A caller that
        needs a display path -- or any other unaddressed, display-only
        datum -- has to carry it alongside the dataset rather than expect
        to recover it from the store: it was never in the stored bytes to
        recover. ``test_round_trip_drops_unaddressed_display_only_fields``
        in ``tests/test_dataset_bridge.py`` pins this, so that projecting a
        currently-unaddressed field later has to be a deliberate, visible
        decision instead of a silent change of what a stored sha means.

        Args:
            payload: A ``dict`` produced by (or shaped exactly like one
                produced by) some ``DatasetEnvelope.identity_payload()``
                call.

        Returns:
            The reconstructed ``DatasetEnvelope``.

        Raises:
            DatasetEnvelopeParseError: the payload is missing its
                discriminator (``envelope_type`` /
                ``identity_payload_version``), declares another envelope
                type or an unsupported version, contains a malformed
                absence marker, fails pydantic validation, or -- after
                validating -- does not reproduce byte-for-byte under
                re-projection. The discriminator checks run FIRST, before
                any rehydration or model validation, so a wrong-type
                payload is refused as a wrong-type payload -- never
                laundered into a field-level validation error.
        """
        _check_identity_payload_discriminator(
            payload, expected_envelope_type=_DATASET_ENVELOPE_TYPE, class_name=cls.__name__
        )
        rehydrated = _restore_unprojected_document_kinds(
            _restore_unprojected_crop_regions(
                _strip_identity_payload_discriminator(_rehydrate_identity_payload(payload))
            )
        )
        try:
            parsed = cls.model_validate(rehydrated)
        except ValidationError as exc:
            raise DatasetEnvelopeParseError(
                f"DatasetEnvelope payload failed validation after rehydration: {exc}"
            ) from exc
        reprojected = parsed.identity_payload()
        if canonical_json_bytes(reprojected) != canonical_json_bytes(payload):
            raise DatasetEnvelopeParseError(
                "DatasetEnvelope.from_identity_payload: the parsed envelope's re-projected "
                "identity_payload() does not byte-match the input payload -- this is a "
                "parser/projector disagreement (the parse silently changed what dataset this "
                "payload addresses), not a claim that the input payload itself is corrupt"
            )
        return parsed


class ConditionSetEnvelope(BaseModel):
    """The top-level payload for one source's extracted experimental
    CONDITIONS: a source graph, a SUBJECT, an attribution assertion, and the
    condition atoms (:class:`GroundedScalarClaim`,
    :class:`GroundedCategoricalClaim`,
    :class:`UnextractedConditionStatement`) that cite that graph.

    A SEPARATE class from :class:`DatasetEnvelope`, deliberately not a
    subclass and not sharing a base: both envelopes are content-addressed
    through hand-written ``identity_payload()`` projections, and a subclass
    silently inheriting the parent's projection would let two DIFFERENT
    payloads address identically in a write-once store -- see
    :class:`_SourceGraphEnvelope` for why the seven shared provenance
    validators are reused by CALL, never by inheritance.

    **The subject is the hard problem this container solves, and it is a
    required SUM.** A set of condition claims with no subject silently
    merges a bomb and a shock tube from one paper into one record with
    every validator green. A required apparatus NAME does not fix that
    either: in a survey of eight real combustion-kinetics papers, one
    paper's "bomb" is two physically different vessels the text never names
    apart ("The first vessel" / "The other vessel"), whose conditions table
    covers both under one caption, and one of the two vessels physically
    cannot reach the highest temperature in that table -- a required name
    there produces a record that LOOKS grounded and is wrong. So
    :attr:`subject` is either a :class:`DeviceClassDeclaration` (grounded,
    and named as a CLASS -- never a unique physical apparatus) or an
    :class:`UnresolvedSubject` (an explicit, grounded refusal to resolve
    the subject at all). There is no third state and no escape into
    ``Maybe``; the field names carry the class-not-instance semantics so
    that downstream laundering has to ignore the words, not just a
    docstring.

    Validators: the seven shared provenance helpers (T2, the duplicate-sha
    guard, T3, V1, V2, V3, V6 -- called as module-level functions), plus
    four container-specific invariants C1--C4, each with its own docstring
    below. As on :class:`DatasetEnvelope`, ``mode="after"`` validators run
    in declaration order, and C4/V2/V3/V6 rely on V1 having already proven
    every ref resolves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_graph: SourceGraph
    conversion_tables: tuple[EmbeddedConversionTable, ...]
    """Every :class:`~carmel.services.units.ConversionTable` cited (by
    ``conversion_table_sha256``) by any :class:`MeasuredValue` reachable in
    this envelope, embedded verbatim -- same contract, same T2/T3
    validators, and same self-containment rationale as
    :attr:`DatasetEnvelope.conversion_tables`. Empty exactly when the
    envelope holds no :class:`MeasuredValue` at all (e.g. a refusals-only
    condition set), because an embedded table nothing cites is unearned
    provenance (T2)."""
    table_inventories: tuple[EmbeddedTableInventory, ...]
    """Every PDF cell inventory cited by any :class:`TableCellLocator`
    reachable in this envelope, embedded verbatim -- same contract and same
    V8/T4/T5 validators as :attr:`DatasetEnvelope.table_inventories`. Empty
    exactly when no reachable locator cites one."""
    subject: DeviceClassDeclaration | UnresolvedSubject
    """The required subject SUM -- see the class docstring for why this is
    a sum and why the declaration arm is class-level. Projected TAGGED (see
    :func:`_condition_subject_identity_payload`), so the two arms can never
    collide in the content-addressed store."""
    attribution: ConditionAttribution
    """Whose conditions these are asserted to be. An extractor ASSERTION
    with an auditable span, never a verified fact -- see
    :class:`ConditionAttribution` and what the counterweight below does not
    prove."""
    attribution_ref: SourceRef
    """The located span the attribution assertion was read from. EVIDENCE
    for the assertion, never an admission gate: a real, resolving ref can
    be paired with a false ``OWN_EXPERIMENT`` (the corpus probe found
    CHEMKIN-run conditions worded exactly like experimental ones), and
    nothing here can tell. Required anyway, because an attribution with no
    span at all is not even auditable by a human."""
    scalar_claims: tuple[GroundedScalarClaim, ...]
    """Grounded single-valued numeric conditions. May be empty -- see C1:
    emptiness is judged across all three collections jointly."""
    categorical_claims: tuple[GroundedCategoricalClaim, ...]
    """Grounded name-valued conditions (diluent identity, reactor type,
    ...). May be empty -- see C1."""
    unextracted: tuple[UnextractedConditionStatement, ...]
    """Located condition statements deliberately NOT turned into claims --
    the coverage-honesty records. May be empty -- see C1 -- but a refusal
    COUNTS as a record there: refusals are first-class content, not
    padding."""

    @model_validator(mode="after")
    def _validate_conversion_tables_cover_cited_tables(self) -> ConditionSetEnvelope:
        """T2: see :func:`_validate_conversion_tables_cover_cited_tables`."""
        _validate_conversion_tables_cover_cited_tables(self)
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_no_duplicate_sha256(self) -> ConditionSetEnvelope:
        """See :func:`_validate_conversion_tables_no_duplicate_sha256`."""
        _validate_conversion_tables_no_duplicate_sha256(self)
        return self

    @model_validator(mode="after")
    def _validate_conversion_tables_sorted(self) -> ConditionSetEnvelope:
        """T3: see :func:`_validate_conversion_tables_sorted`."""
        _validate_conversion_tables_sorted(self)
        return self

    @model_validator(mode="after")
    def _validate_refs_resolve(self) -> ConditionSetEnvelope:
        """V1: see :func:`_validate_refs_resolve`."""
        _validate_refs_resolve(self)
        return self

    @model_validator(mode="after")
    def _validate_single_root_artifact(self) -> ConditionSetEnvelope:
        """C4: every :class:`SourceRef` in the WHOLE envelope must resolve
        under ONE parentless root artifact.

        Deliberately STRONGER than :class:`DatasetEnvelope`'s per-series V5:
        a dataset envelope may legitimately aggregate series from one paper
        and its JATS rendition side by side, but a condition set is one
        subject's conditions from one source -- a subject label span
        grounded in paper A with a claim value span grounded in paper B is
        two papers silently stitched into one record, exactly the
        fabrication shape this module exists to make unconstructible.
        Checked at the ROOT level, not the node level: a label in the main
        PDF and a value in that paper's SI member share a root and stay
        legal.

        Runs AFTER V1 (declaration order), so every ``ref.node_id`` here is
        already known to resolve. A node's root is itself if its
        ``ancestors()`` chain is empty, otherwise the LAST entry of that
        chain -- ``SourceGraph.ancestors`` returns immediate parent first,
        root last.
        """
        roots: dict[str, str] = {}
        for path, ref in iter_source_refs(self):
            ancestor_chain = self.source_graph.ancestors(ref.node_id)
            root_id = ancestor_chain[-1].node_id if ancestor_chain else ref.node_id
            roots[root_id] = path
        if len(roots) > 1:
            raise ValueError(
                f"ConditionSetEnvelope spans multiple root artifacts: refs resolve to nodes under root "
                f"artifacts {sorted(roots)!r} (e.g. {', '.join(sorted(roots[r] for r in roots))}) -- a "
                "condition set must be grounded under a single root artifact"
            )
        return self

    @model_validator(mode="after")
    def _validate_no_decorative_nodes(self) -> ConditionSetEnvelope:
        """V2: see :func:`_validate_no_decorative_nodes`."""
        _validate_no_decorative_nodes(self)
        return self

    @model_validator(mode="after")
    def _validate_locator_kind_compatibility(self) -> ConditionSetEnvelope:
        """V3: see :func:`_validate_locator_kind_compatibility`."""
        _validate_locator_kind_compatibility(self)
        return self

    @model_validator(mode="after")
    def _validate_char_span_requires_extraction(self) -> ConditionSetEnvelope:
        """V6: see :func:`_validate_char_span_requires_extraction`."""
        _validate_char_span_requires_extraction(self)
        return self

    @model_validator(mode="after")
    def _validate_table_cell_inventory_citation(self) -> ConditionSetEnvelope:
        """V8: see :func:`_validate_table_cell_inventory_citation`. Declared
        BEFORE T4 for the diagnostic reason T4's docstring gives -- not because
        the verdict depends on the order."""
        _validate_table_cell_inventory_citation(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_cover_cited_inventories(self) -> ConditionSetEnvelope:
        """T4: see :func:`_validate_table_inventories_cover_cited_inventories`."""
        _validate_table_inventories_cover_cited_inventories(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_no_duplicate_sha256(self) -> ConditionSetEnvelope:
        """See :func:`_validate_table_inventories_no_duplicate_sha256`."""
        _validate_table_inventories_no_duplicate_sha256(self)
        return self

    @model_validator(mode="after")
    def _validate_table_inventories_sorted(self) -> ConditionSetEnvelope:
        """T5: see :func:`_validate_table_inventories_sorted`."""
        _validate_table_inventories_sorted(self)
        return self

    @model_validator(mode="after")
    def _validate_holds_at_least_one_record(self) -> ConditionSetEnvelope:
        """C1: at least one entry across ``scalar_claims`` +
        ``categorical_claims`` + ``unextracted`` COMBINED -- refusals COUNT.

        A paper stating only "pressures from 1 to 10 atm" yields zero
        claims and one ``VALUE_RANGE`` refusal, and that envelope MUST stay
        legal: it is exactly the coverage honesty the ``unextracted``
        collection exists for. What is refused is the ALL-empty envelope --
        a validated source graph, a grounded subject, an attribution, and
        no condition content whatsoever: an audit-shaped artifact that
        LOOKS like grounded extraction while containing nothing any
        consumer could ever read a condition from. This cannot be a
        ``Field(min_length=1)`` because the requirement spans three fields
        jointly; a per-field minimum would wrongly outlaw the
        refusals-only and claims-only shapes that are each individually
        legitimate.
        """
        if not (self.scalar_claims or self.categorical_claims or self.unextracted):
            raise ValueError(
                "ConditionSetEnvelope holds no scalar_claims, no categorical_claims and no unextracted "
                "statements -- an all-empty condition set is an audit-shaped artifact, not a record; a "
                "set with nothing extractable must still carry its refusals (unextracted), which count"
            )
        return self

    @model_validator(mode="after")
    def _validate_one_id_namespace(self) -> ConditionSetEnvelope:
        """C2: ONE id namespace across the whole envelope -- no id may
        repeat across ``scalar_claims``, ``categorical_claims`` and
        ``unextracted`` JOINTLY, not merely within each collection.

        The concrete failure a per-collection check would miss: a coverage
        map keyed by logical condition id, where
        ``scalar_claims["pressure"]`` and ``unextracted["pressure"]``
        coexist and one silently overwrites the other -- turning "refused a
        range" into "extracted a scalar" (or the reverse) with no
        validator anywhere in the path. One namespace makes the collision
        unconstructible instead of merely unlikely.
        """
        owners: dict[str, str] = {}
        for collection_name, entry_id in (
            *(("scalar_claims", claim.claim_id) for claim in self.scalar_claims),
            *(("categorical_claims", claim.claim_id) for claim in self.categorical_claims),
            *(("unextracted", statement.statement_id) for statement in self.unextracted),
        ):
            if entry_id in owners:
                raise ValueError(
                    f"ConditionSetEnvelope: duplicate id {entry_id!r} appears in both "
                    f"{owners[entry_id]} and {collection_name} -- claim_ids and statement_ids share "
                    "ONE namespace across all three collections"
                )
            owners[entry_id] = collection_name
        return self

    @model_validator(mode="after")
    def _validate_collections_sorted(self) -> ConditionSetEnvelope:
        """C3: each of the three collections must be sorted ascending by its
        own id, so exactly one legal ordering -- and therefore exactly one
        content address -- exists per logical condition set; the same
        S2/S7/E1b/T3 idiom used throughout this module."""
        for collection_name, actual_ids in (
            ("scalar_claims", [claim.claim_id for claim in self.scalar_claims]),
            ("categorical_claims", [claim.claim_id for claim in self.categorical_claims]),
            ("unextracted", [statement.statement_id for statement in self.unextracted]),
        ):
            if actual_ids != sorted(actual_ids):
                raise ValueError(
                    f"ConditionSetEnvelope: {collection_name} must be sorted ascending by id, got {actual_ids!r}"
                )
        return self

    def identity_payload(self) -> dict[str, Any]:
        """Project this envelope to its canonical-JSON identity payload --
        the condition-set counterpart of
        :meth:`DatasetEnvelope.identity_payload`, with the same contract:
        hand-written field-by-field projection, never ``model_dump`` (see
        that method's docstring for why pydantic's dump shape must not
        define a content address), plain JSON-able output only, enums
        unwrapped to ``.value``, ``Maybe`` fields through
        :func:`_project_maybe`, and a freshly built dict on every call.

        The one shape unique to this envelope is the subject SUM, which
        projects TAGGED via :func:`_condition_subject_identity_payload` --
        see that helper for why the tag is load-bearing.

        Like its dataset counterpart, the payload is SELF-DESCRIBING:
        ``envelope_type`` and ``identity_payload_version`` are part of the
        addressed bytes, so a stored condition set can never be silently
        parsed as a dataset (or vice versa) -- see
        :meth:`DatasetEnvelope.identity_payload` for the full rationale.
        """
        return {
            _ENVELOPE_TYPE_KEY: _CONDITION_SET_ENVELOPE_TYPE,
            _IDENTITY_PAYLOAD_VERSION_KEY: _SUPPORTED_IDENTITY_PAYLOAD_VERSION,
            "source_graph": _source_graph_identity_payload(self.source_graph),
            "conversion_tables": [
                _embedded_conversion_table_identity_payload(table) for table in self.conversion_tables
            ],
            "table_inventories": [
                _embedded_table_inventory_identity_payload(inventory) for inventory in self.table_inventories
            ],
            "subject": _condition_subject_identity_payload(self.subject),
            "attribution": self.attribution.value,
            "attribution_ref": _source_ref_identity_payload(self.attribution_ref),
            "scalar_claims": [_grounded_scalar_claim_identity_payload(claim) for claim in self.scalar_claims],
            "categorical_claims": [
                _grounded_categorical_claim_identity_payload(claim) for claim in self.categorical_claims
            ],
            "unextracted": [
                _unextracted_condition_statement_identity_payload(statement) for statement in self.unextracted
            ],
        }

    @classmethod
    def from_identity_payload(cls, payload: dict[str, Any]) -> ConditionSetEnvelope:
        """Reconstruct a :class:`ConditionSetEnvelope` from its own
        :meth:`identity_payload` projection -- the exact inverse of that
        method, with the same two-stage parse as
        :meth:`DatasetEnvelope.from_identity_payload` (see that docstring
        for the full rationale of each stage) plus one condition-set-only
        step: the subject sum is dispatched on its
        :data:`_SUBJECT_KIND_KEY` tag via
        :func:`_rehydrate_condition_subject` BEFORE ``model_validate``,
        because both variants are ``extra="forbid"`` and the tag is
        projection-only data that must be stripped -- and because variant
        choice must follow the tag, never pydantic's field-shape sniffing.

        Stage 2's byte-for-byte re-projection comparison is what proves the
        parse preserved identity; a mismatch is raised as a
        parser/projector DISAGREEMENT, never as a claim that the input
        payload is corrupt -- nothing here has evidence of that.

        Like its ``DatasetEnvelope`` counterpart, this inverse is exact on
        IDENTITY and lossy on ``_UNADDRESSED_FIELDS`` (today:
        ``ArchiveOrigin.member_display_path`` only), and it restores the one
        CONDITIONAL key the projection omits -- ``crop_region``, dropped for
        a node holding no region, which by I7 is any node that is not a
        ``FIGURE_CROP``; see :func:`_restore_unprojected_crop_regions`. Both
        envelope classes
        share one source graph projection, so both need that same inverse;
        omitting it here would make a condition set holding no crop
        unparseable by its own projector.

        Raises:
            DatasetEnvelopeParseError: the payload is missing its
                discriminator (``envelope_type`` /
                ``identity_payload_version``), declares another envelope
                type or an unsupported version, has a malformed absence
                marker, a missing/unknown subject tag, fails validation, or
                does not reproduce byte-for-byte under re-projection. The
                discriminator checks run FIRST, before any rehydration or
                model validation -- a dataset payload handed here must be
                refused as the wrong envelope type, never half-parsed into
                a subject-tag or field-level error.
        """
        _check_identity_payload_discriminator(
            payload, expected_envelope_type=_CONDITION_SET_ENVELOPE_TYPE, class_name=cls.__name__
        )
        rehydrated = _restore_unprojected_document_kinds(
            _restore_unprojected_crop_regions(
                _strip_identity_payload_discriminator(_rehydrate_identity_payload(payload))
            )
        )
        if isinstance(rehydrated, dict) and "subject" in rehydrated:
            rehydrated = {**rehydrated, "subject": _rehydrate_condition_subject(rehydrated["subject"])}
        try:
            parsed = cls.model_validate(rehydrated)
        except ValidationError as exc:
            raise DatasetEnvelopeParseError(
                f"ConditionSetEnvelope payload failed validation after rehydration: {exc}"
            ) from exc
        reprojected = parsed.identity_payload()
        if canonical_json_bytes(reprojected) != canonical_json_bytes(payload):
            raise DatasetEnvelopeParseError(
                "ConditionSetEnvelope.from_identity_payload: the parsed envelope's re-projected "
                "identity_payload() does not byte-match the input payload -- this is a "
                "parser/projector disagreement (the parse silently changed what condition set this "
                "payload addresses), not a claim that the input payload itself is corrupt"
            )
        return parsed
