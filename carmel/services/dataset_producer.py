# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Produce a validated :class:`~carmel.schemas.datasets.DatasetEnvelope` from one
REAL stored evidence artifact.

Naming: this module is the schema-AWARE producer counterpart to the
schema-blind :mod:`carmel.services.dataset_store`, exactly as
:mod:`carmel.services.dataset_bridge` is the schema-aware store/load layer
above that same blind store. ``dataset_bridge`` moves an already-built
envelope in and out of the store; this module is the one place that BUILDS an
envelope from the other real runtime artifact this project has -- a stored
literature artifact in the evidence store (:mod:`carmel.services.evidence`).
It therefore has to know about both sides at once (the evidence store's
layout/metadata and the dataset schema), which is exactly why it is its own
module rather than a method on either side.

The cardinal rule of the dataset schema (see :mod:`carmel.schemas.datasets`)
is that every number is grounded in stored, auditable source bytes. This
producer enforces that rule MECHANICALLY, not by convention:

- Callers state only DOMAIN FACTS: "the quote ``'1023'`` is a temperature
  coordinate whose unit is printed as ``'K'``". They never supply character
  offsets. Every :class:`~carmel.schemas.datasets.CharSpanLocator` in the
  produced envelope is computed by :func:`ground_quote` actually SEARCHING
  the artifact's verified extracted text -- an offset nobody searched for
  cannot appear in the output.
- The extracted text itself is never trusted from a convenience loader:
  ``extracted.json``'s file bytes are re-read and re-verified against the
  digest recorded at store time (``StoredArtifact.extracted_sha256``) BEFORE
  they are parsed at all. Unverified bytes are never parsed.
- ``ExtractionBinding.extracted_text_sha256`` is computed from
  ``extracted.text`` (the field inside the verified, parsed
  ``ExtractedText``), NEVER from ``text.txt`` on disk -- ``text.txt`` is
  presence-checked only, never digest-checked (see that field's docstring in
  :mod:`carmel.schemas.datasets`), so hashing it would anchor the binding to
  the one file in the store nothing verifies.

Everything here is fail-closed: a quote that cannot be found, is ambiguous,
or is out of range raises; a missing/legacy/corrupt artifact raises; nothing
ever silently guesses.

WHAT GROUNDING DOES AND DOES NOT PROVE (read this before trusting an
envelope this module produces): every span this module records verifiably
slices the extracted text to exactly the string the envelope claims -- that
is what :func:`ground_quote` mechanically enforces, and (for a numeric value
quote) that the span is a maximal numeric token rather than an interior
fragment of a larger one. NOTHING here yet checks that the value, unit, and
label spans it grounds independently belong to the SAME measurement: a
caller can pass a ``value_quote`` from one sentence and a ``unit_quote``
from an unrelated one elsewhere in the document, and this module will
happily ground both and assemble a valid envelope out of them. Each
:class:`MeasurementSpec` is grounded quote-by-quote, with no adjacency,
table/row, or sentence-scoping check tying the three quotes of one spec
together. Closing that gap needs a bounded measurement context (e.g.
adjacency or table/row structure) and is its own future milestone, not
something this vertical slice attempts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    CharSpanLocator,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    ExtractionBinding,
    MeasuredValue,
    Observation,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TextSpace,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_store import (
    CanonicalDecimalError,
    canonical_decimal,
    canonical_json_bytes,
)
from carmel.services.evidence import artifact_dir, load_artifact_meta, verify_artifact
from carmel.services.numeric import (
    GlyphHealth,
    SourceContext,
    Unresolvable,
    assess_glyph_health,
    normalize_numeric_span,
)
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import QuantityKind

__all__ = [
    "DatasetProducerError",
    "MeasurementSpec",
    "QuoteGroundingError",
    "ground_quote",
    "produce_envelope_from_artifact",
]

_EXTRACTED_NAME = "extracted.json"
"""Filename of the full ``ExtractedText`` sidecar inside an artifact's
content-addressed directory. This is the evidence store's PUBLIC on-disk
contract -- ``carmel.services.evidence``'s module docstring documents the
layout as exact ("Layout, exactly::") -- not private knowledge duplicated
here."""

_ROOT_NODE_ID = "paper"
"""The single :class:`SourceNode`'s id in every produced graph. One artifact
in, one root node out -- this vertical slice models exactly that shape and
nothing more. The node's ``kind`` (``PAPER_PDF`` or ``JATS_XML``) is derived
from the artifact's own ``content_type``, never hardcoded -- see
``_CONTENT_TYPE_TO_NODE_KIND``."""

_POINT_ID = "p1"

_NUMERIC_TOKEN_RE = re.compile(r"[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?")
"""Character class defining one "numeric token" for :func:`ground_quote`'s
maximal-extent check: an optional integer part, an optional decimal point, a
mandatory digit run, and an optional exponent suffix (``e``/``E`` with an
optional sign). This governs ONLY where a numeric token STOPS -- a run of
``[0-9]``, at most one ``.``, and at most one ``[eE][+-]?[0-9]+`` suffix all
count as one token; anything else (a letter that isn't part of an exponent
marker, whitespace, punctuation) ends it. Deliberate consequence: ``"1023"``
grounded against ``"1023K"`` is accepted (``K`` is not numeric continuation,
so ``"1023"`` IS the maximal token there) but ``"1023"`` grounded against
``"11023"`` or ``"1023.5"`` or ``"0.51023"`` is rejected (in each case the
match is an interior slice of a strictly larger numeric token). A quote that
does not itself look like a numeric token (e.g. a unit or label quote such as
``"mole fraction"``) is exempt from this check entirely -- it exists only to
stop a numeral from silently grounding to a fragment of a bigger numeral."""


def _quote_looks_numeric(quote: str) -> bool:
    """True if ``quote`` itself matches :data:`_NUMERIC_TOKEN_RE` in full --
    i.e. the caller is claiming a numeral, so the maximal-token check below
    applies. A unit/label quote (``"K"``, ``"mole fraction"``) never matches
    this and is left alone: the check exists to stop a numeral fragmenting a
    larger numeral, not to constrain non-numeric quotes."""
    match = _NUMERIC_TOKEN_RE.fullmatch(quote)
    return match is not None

# Mirrors carmel.schemas.datasets._HEALTHY_GLYPH_HEALTH (module-private there,
# so restated rather than imported): MeasuredValue's own repair-chain
# validator re-runs normalize_numeric_span with SourceContext.OPERATOR_RAW and
# a healthy, hand-constructed GlyphHealth, because a stored MeasuredValue
# carries no surrounding document to assess. This producer derives
# repairs/canonical_decimal_value through the IDENTICAL call so that what it
# records is exactly what the schema's validator will re-derive -- any other
# context here would let the producer construct values its own schema then
# rejects (or worse, accepts for the wrong reason).
_CONTEXT_FREE_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)


#: Restated from :mod:`carmel.services.grounding`'s ``_PDF_EXTRACTOR_PREFIX``
#: (module-private, so per this codebase's convention it is restated here
#: rather than imported): the ``ExtractedText.extractor`` value recorded for
#: text flattened out of a PDF's text layer, and therefore the one extractor
#: family known to be subject to the dash-corruption quarantine rule.
_PDF_EXTRACTOR_PREFIX = "pdf:pypdf"


def _source_context_for(extracted: ExtractedText) -> SourceContext:
    """The strict core's :class:`SourceContext` for text extracted into
    ``extracted``. Mirrors :func:`carmel.services.grounding._source_context_for`
    exactly: derived from ``extracted.extractor`` rather than hardcoded, so
    only artifacts that actually came from a flattened PDF text layer are
    ever subject to the dash-corruption quarantine rule.

    P1-D: this producer's own :func:`_measured_value` used to normalize every
    value quote under a HARDCODED ``SourceContext.OPERATOR_RAW`` plus a
    hand-constructed always-healthy ``_CONTEXT_FREE_GLYPH_HEALTH`` --
    regardless of what the artifact actually was or whether its text showed
    any sign of corruption. That bypassed the exact quarantine
    :mod:`carmel.services.grounding` enforces for suspect PDF text, so a
    dash-corrupted bare-exponent value (e.g. ``"1023e5"`` where the source
    PDF actually had an en-dash-separated range) could sail straight through
    this producer while :func:`carmel.services.grounding` would have refused
    it. This function and :func:`_document_glyph_health` below restore that
    quarantine as an ADDITIONAL refusal-only canary check in
    :func:`_measured_value` -- the value actually stored on
    :class:`MeasuredValue` still comes from the OPERATOR_RAW/context-free
    computation, because ``MeasuredValue``'s own pydantic validator
    recomputes and requires exactly that computation's ``repairs``; this
    canary can only ever ADD a refusal, never change what gets stored.
    """
    if extracted.extractor == _PDF_EXTRACTOR_PREFIX:
        return SourceContext.FLAT_PDF_TEXT
    return SourceContext.OPERATOR_RAW


class QuoteGroundingError(ValueError):
    """Raised by :func:`ground_quote` when a quote cannot be grounded
    unambiguously: empty quote, quote not found, ambiguous quote with no
    explicit occurrence, or an occurrence out of range for the matches
    actually found. Never a silent guess."""


class DatasetProducerError(ValueError):
    """Raised by :func:`produce_envelope_from_artifact` when the stored
    artifact cannot be resolved or verified, or a caller-supplied spec cannot
    be honestly realised. Fail-closed: nothing is ever produced from
    unverified bytes or an unverifiable claim."""


def ground_quote(text: str, quote: str, *, occurrence: int | None = None) -> CharSpanLocator:
    """Locate ``quote`` in ``text`` by SEARCHING, returning a half-open
    :class:`CharSpanLocator` over ``TextSpace.EXTRACTED_TEXT``.

    ``occurrence`` is ``int | None`` with a default of ``None``, deliberately
    NOT ``int = 0``: with ``int = 0`` a caller who never thought about
    ambiguity is indistinguishable from one who explicitly chose the first
    match, so ambiguity would silently resolve to match 0 -- exactly the
    guess this function exists to refuse. ``None`` (the default) means "I
    claim this quote is unique": it succeeds only when exactly one match
    exists, and an ambiguous quote raises, stating the match count. Passing
    an explicit ``occurrence`` (0-based) is the caller's affirmative
    disambiguation and selects that match; an out-of-range value raises,
    stating how many matches were actually found.

    Matches are counted INCLUDING overlapping ones (``"aa"`` occurs twice in
    ``"aaa"``): every position the quote appears at is a distinct candidate
    grounding, and undercounting would let a genuinely ambiguous quote pass
    the uniqueness check.

    Args:
        text: The verified extracted text to search (``ExtractedText.text``,
            the RAW string every ``CharSpanLocator`` offset indexes into --
            never ``.normalized``).
        quote: The exact source substring to locate, verbatim.
        occurrence: ``None`` to require uniqueness; a 0-based index to
            explicitly select among multiple matches.

    Returns:
        A ``CharSpanLocator`` with ``text[start:end] == quote``.

    Raises:
        QuoteGroundingError: Empty quote; quote not found; ambiguous quote
            with ``occurrence=None``; or ``occurrence`` out of range.
    """
    if not quote:
        # CharSpanLocator's own validator would eventually reject the
        # zero-width span this would produce, but with a generic
        # "end must be strictly greater than start" message about a locator
        # the caller never wrote. Refuse here, earlier, naming the actual
        # mistake: an empty quote matches at every position and grounds
        # nothing.
        raise QuoteGroundingError(
            "ground_quote: quote is empty -- an empty quote matches at every position in the text "
            "and grounds nothing; supply the exact, non-empty source substring to locate"
        )
    starts: list[int] = []
    found = text.find(quote)
    while found != -1:
        starts.append(found)
        found = text.find(quote, found + 1)
    display = quote if len(quote) <= 120 else quote[:117] + "..."
    if not starts:
        raise QuoteGroundingError(f"ground_quote: quote {display!r} was not found in the supplied text")
    if occurrence is None:
        if len(starts) > 1:
            raise QuoteGroundingError(
                f"ground_quote: quote {display!r} is ambiguous -- it appears {len(starts)} times in the "
                "supplied text; ambiguity never silently resolves to the first match, so pass "
                "occurrence= explicitly to select one"
            )
        start = starts[0]
    else:
        if not 0 <= occurrence < len(starts):
            raise QuoteGroundingError(
                f"ground_quote: occurrence={occurrence} is out of range for quote {display!r} -- "
                f"{len(starts)} match(es) found, so occurrence must be in [0, {len(starts) - 1}]"
            )
        start = starts[occurrence]
    end = start + len(quote)
    if _quote_looks_numeric(quote):
        # P1-A: a numeral quote must ground to a MAXIMAL numeric token, never
        # an interior slice of a strictly larger one -- e.g. quote "1023"
        # must not silently accept the middle of "11023", "1023.5", or
        # "0.51023". ``_NUMERIC_TOKEN_RE.finditer`` walks the text producing
        # non-overlapping, greedily-maximal numeric tokens, so the token that
        # contains our chosen ``start`` is, by construction, the largest
        # numeral touching that position; if its span differs from
        # ``(start, end)`` the quote is a fragment, not the whole numeral.
        for match in _NUMERIC_TOKEN_RE.finditer(text):
            if match.start() <= start < match.end():
                if match.span() != (start, end):
                    raise QuoteGroundingError(
                        f"ground_quote: quote {display!r} is an interior fragment of the larger "
                        f"numeral {text[match.start():match.end()]!r} (span "
                        f"[{match.start()}:{match.end()}]) in the supplied text -- a grounded "
                        "numeral span must be the MAXIMAL numeric token, never a slice of a "
                        "bigger one; quote the full numeral if that is what is meant"
                    )
                break
    # Correctness self-check on this function's own arithmetic (an assert,
    # not a raise: user input was already validated above; this can only
    # fail if the search/slicing logic itself is wrong).
    assert text[start:end] == quote, "ground_quote arithmetic self-check failed: slice != quote"
    return CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end)


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """One caller-stated domain fact: what a quoted number/unit/label IS.

    The caller states only meaning -- axis identity/role, physical quantity,
    and the VERBATIM quotes for the number, its unit, and the axis label as
    printed in the source. No character offsets, ever: the producer computes
    every offset itself via :func:`ground_quote` against the verified
    extracted text. The optional ``*_occurrence`` fields are the caller's
    explicit disambiguation for a quote that appears more than once (see
    :func:`ground_quote`); ``None`` claims the quote is unique.
    """

    axis_id: str
    role: AxisRole
    quantity_kind: QuantityKind
    label_quote: str
    value_quote: str
    unit_quote: str
    label_occurrence: int | None = None
    value_occurrence: int | None = None
    unit_occurrence: int | None = None

    def __post_init__(self) -> None:
        # A plain dataclass has no field validation of its own, and this one is
        # slotted/frozen so nothing downstream re-checks its fields either. ``bool``
        # is a subclass of ``int`` in Python, so a bare ``isinstance(x, int)`` check
        # would silently accept ``True``/``False`` here as occurrence 1/0 -- almost
        # certainly a caller typo (e.g. a stray boolean flag), never a real
        # disambiguation intent -- so ``bool`` must be excluded explicitly.
        for name in ("label_occurrence", "value_occurrence", "unit_occurrence"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise DatasetProducerError(
                    f"MeasurementSpec.{name}={value!r} must be an int or None, not "
                    f"{type(value).__name__} -- bool is a subclass of int in Python and would "
                    "silently mean occurrence 0/1"
                )


def _current_repair_dependency() -> SemanticDependencyUse:
    """The repair-dependency record for a value repaired by the CURRENT
    context-free span-repair heuristic -- resolved per call (not cached at
    import) so it always reflects the live registry."""
    return SemanticDependencyUse(
        dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
        content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
        input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _measured_value(
    text: str,
    spec: MeasurementSpec,
    *,
    document_source_context: SourceContext,
    document_glyph_health: GlyphHealth,
) -> MeasuredValue:
    """Build one grounded :class:`MeasuredValue` from ``spec`` against ``text``.

    Every offset comes from :func:`ground_quote`; ``repairs`` and
    ``canonical_decimal_value`` are DERIVED from the value quote through the
    same ``normalize_numeric_span`` call the schema's own validator re-runs
    (see ``_CONTEXT_FREE_GLYPH_HEALTH``), never asserted independently.

    P1-D: ``document_source_context``/``document_glyph_health`` (the artifact's
    REAL context and glyph health, from :func:`_source_context_for` and
    :func:`carmel.services.numeric.assess_glyph_health` against the actual
    stored text -- see :func:`_source_context_for`'s docstring for why) are
    used ONLY as an additional refusal-only canary call below, run BEFORE the
    stored value is computed. It can only ever REFUSE a value the
    context-free computation would have accepted; it never changes what gets
    stored, because ``MeasuredValue``'s own validator requires the
    context-free ``repairs`` exactly.
    """
    value_locator = ground_quote(text, spec.value_quote, occurrence=spec.value_occurrence)
    unit_locator = ground_quote(text, spec.unit_quote, occurrence=spec.unit_occurrence)
    canary = normalize_numeric_span(
        spec.value_quote,
        source_context=document_source_context,
        glyph_health=document_glyph_health,
    )
    if isinstance(canary, Unresolvable):
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for axis {spec.axis_id!r} is refused under the "
            f"document's REAL source context ({document_source_context!r}) and glyph health "
            f"({document_glyph_health!r}): {canary.reason} -- this is the P1-D quarantine canary; "
            "the value may be genuinely corrupt in this document even though it would parse fine "
            "in isolation"
        )
    normalized = normalize_numeric_span(
        spec.value_quote,
        source_context=SourceContext.OPERATOR_RAW,
        glyph_health=_CONTEXT_FREE_GLYPH_HEALTH,
    )
    if isinstance(normalized, Unresolvable):
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for axis {spec.axis_id!r} is not derivable into a "
            f"numeral: {normalized.reason}"
        )
    try:
        canonical = canonical_decimal(normalized.text)
    except CanonicalDecimalError as exc:
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for axis {spec.axis_id!r} repaired to "
            f"{normalized.text!r}, which is not a valid canonical decimal string: {exc}"
        ) from exc
    try:
        unit_normalized = units.normalize_unit(spec.quantity_kind, spec.unit_quote, table=units.TABLE_V1)
    except units.UnknownUnitError as exc:
        raise DatasetProducerError(
            f"unit quote {spec.unit_quote!r} for axis {spec.axis_id!r} is not a known unit or alias "
            f"of quantity_kind={spec.quantity_kind.value!r} in TABLE_V1: {exc}"
        ) from exc
    return MeasuredValue(
        raw_text=spec.value_quote,
        canonical_decimal_value=canonical,
        repairs=normalized.repairs,
        repair_dependency=_current_repair_dependency(),
        quantity_kind=spec.quantity_kind,
        unit_raw=spec.unit_quote,
        unit_normalized=unit_normalized,
        conversion_table_sha256=units.TABLE_V1.sha256,
        value_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=value_locator),
        unit_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=unit_locator),
    )


#: Maps ``StoredArtifact.content_type`` to the :class:`SourceNodeKind` this
#: producer's single root node may honestly claim to be. Only the two content
#: types :func:`carmel.services.acquisition._sniff_content_type` actually
#: emits for a primary document are recognised: ``"application/pdf"`` and
#: ``"application/xml"`` (the latter used for JATS-style full-text XML;
#: ``"text/xml"`` is included too since it is the same media type family and
#: some sources serve it under that label). Anything else -- an HTML page, a
#: plain-text scrape, an unrecognised type -- has no ``SourceNodeKind`` this
#: producer may truthfully assert, so it is not in this table and must be
#: refused rather than guessed.
_CONTENT_TYPE_TO_NODE_KIND: dict[str, SourceNodeKind] = {
    "application/pdf": SourceNodeKind.PAPER_PDF,
    "application/xml": SourceNodeKind.JATS_XML,
    "text/xml": SourceNodeKind.JATS_XML,
}


def _load_verified_extracted_text(
    workspace_root: Path, sha256: str
) -> tuple[ExtractedText, str, str | None, str]:
    """Resolve, verify, and parse the stored extraction for ``sha256``.

    Returns ``(extracted, extracted_sha256, derivation_binding, content_type)``.
    The bytes of ``extracted.json`` are read directly from disk and their digest is
    compared against ``StoredArtifact.extracted_sha256`` BEFORE any parsing
    -- unverified bytes are never parsed. (``extracted_sha256`` really is the
    digest of those file bytes: ``evidence._write_all`` computes it as
    ``hashlib.sha256(extracted_path.read_bytes()).hexdigest()`` AFTER writing
    the file, precisely so the recorded digest describes the bytes on disk.)

    Artifact resolution goes through :func:`carmel.services.evidence.load_artifact_meta`
    (added alongside this module): a direct by-sha lookup with the read
    path's sha-shape/containment validation. The alternative --
    ``list_artifacts()`` and filter -- is O(store size) per call, silently
    skips artifacts whose ``meta.json`` is unreadable (so "absent" and
    "present but corrupt" become indistinguishable), and validates nothing
    about the caller-supplied sha string.

    P1-B: before this function existed, the producer verified ONLY
    ``extracted.json`` (below) -- ``raw.bin`` and ``meta.json`` were never
    checked at all. Two checks close that gap, both refusing with a message
    naming exactly which one failed:

    1. ``meta.sha256 == sha256``: ``load_artifact_meta`` resolves the store
       directory purely from the ``sha256`` PARAMETER (via its own
       sha-shape/containment validation) and does NOT cross-check that
       parameter against the ``sha256`` FIELD recorded inside the loaded
       ``meta.json`` -- so a ``meta.json`` whose ``sha256`` field disagrees
       with the directory it lives in (e.g. hand-edited or copied from
       elsewhere) would otherwise go undetected.
    2. :func:`carmel.services.evidence.verify_artifact` with ``deep=False``:
       confirms ``raw.bin`` exists and hashes to ``sha256`` (the parameter),
       and that ``extracted.json`` matches its recorded digest (the same
       check performed by hand below, but this call also covers ``raw.bin``,
       which nothing here otherwise touches). ``deep=False`` deliberately:
       ``deep=True`` additionally re-checks ``derivation_binding``, but it
       FAILS CLOSED for any artifact missing ``derivation_binding`` /
       ``extractor_version`` / ``extracted_sha256`` -- i.e. every legacy
       artifact -- which would silently regress this producer's existing,
       deliberate support for legacy artifacts (handled below via
       ``AbsenceReason.UNKNOWN``). Consequence, flagged rather than
       silently accepted: ``derivation_binding``'s own internal-consistency
       guarantee (see ``StoredArtifact.derivation_binding``'s docstring) is
       NOT independently re-verified by this producer for artifacts that do
       carry it; only ``meta.json``'s presence, structure, and the
       ``extracted.json``/``raw.bin`` digests it records are.
    """
    meta = load_artifact_meta(workspace_root, sha256)
    if meta is None:
        raise DatasetProducerError(
            f"no stored artifact found under sha256 {sha256!r} in this workspace's evidence store"
        )
    if meta.sha256 != sha256:
        raise DatasetProducerError(
            f"artifact meta.json at sha256 {sha256!r} records sha256={meta.sha256!r} internally -- "
            "the two disagree, so this evidence directory is not trustworthy; refusing to use it"
        )
    if not verify_artifact(workspace_root, sha256, deep=False):
        raise DatasetProducerError(
            f"artifact {sha256!r} failed verify_artifact (raw.bin missing/corrupt, or extracted.json "
            "does not match its recorded extracted_sha256); refusing to use unverified bytes"
        )
    if meta.extracted_sha256 is None:
        # A legacy artifact stored before extracted_sha256 existed carries no
        # digest for its sidecar, so its extracted.json cannot be verified at
        # all -- and this producer never parses unverified bytes.
        raise DatasetProducerError(
            f"artifact {sha256!r} predates extracted_sha256 and its extracted.json cannot be "
            "verified; refusing to parse unverified bytes"
        )
    extracted_path = artifact_dir(workspace_root, sha256) / _EXTRACTED_NAME
    try:
        raw_bytes = extracted_path.read_bytes()
    except OSError as exc:  # FileNotFoundError is an OSError subclass
        raise DatasetProducerError(
            f"artifact {sha256!r} has no readable {_EXTRACTED_NAME}: {exc}"
        ) from exc
    actual_digest = hashlib.sha256(raw_bytes).hexdigest()
    if actual_digest != meta.extracted_sha256:
        raise DatasetProducerError(
            f"artifact {sha256!r}: {_EXTRACTED_NAME} bytes on disk hash to {actual_digest!r}, not the "
            f"recorded extracted_sha256 {meta.extracted_sha256!r}; refusing to parse unverified bytes"
        )
    try:
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    except ValueError as exc:
        raise DatasetProducerError(
            f"artifact {sha256!r}: verified {_EXTRACTED_NAME} bytes do not parse as an ExtractedText: {exc}"
        ) from exc
    return extracted, meta.extracted_sha256, meta.derivation_binding, meta.content_type


def produce_envelope_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    series_id: str,
    value_origin: ValueOrigin,
    measurements: tuple[MeasurementSpec, ...],
) -> DatasetEnvelope:
    """Build a fully validated :class:`DatasetEnvelope` from ONE stored artifact.

    The vertical slice, end to end: resolve the artifact's metadata by its
    raw-bytes sha256, verify ``raw.bin``/``meta.json``/``extracted.json``
    (see :func:`_load_verified_extracted_text`), parse the verified
    extraction, ground every caller-stated quote in the extracted text via
    :func:`ground_quote`, and assemble one root node -- ``PAPER_PDF`` or
    ``JATS_XML``, derived honestly from the artifact's own ``content_type``
    (never hardcoded; an unrecognised ``content_type`` is refused, not
    guessed) -- one ``TEXTUAL`` series, and one data point into an envelope
    that passes every schema validator (construction runs pydantic's full
    validation -- nothing here uses ``model_construct``).

    ``source_form`` is fixed at ``TEXTUAL``: a :class:`CharSpanLocator` into
    extracted running text is the only locator kind this runtime can actually
    produce (the round-33 ruling that added it), and it is what every span
    here is. ``value_origin`` is the caller's assertion, passed through --
    see :class:`ValueOrigin` for why the schema records it unverified.

    WHAT GROUNDING DOES AND DOES NOT PROVE (P1-F, restated at the call site
    that matters most): every ``value_ref``/``unit_ref``/``label_ref`` this
    function emits is independently verified to be an exact, located
    substring of the one verified document -- that is the entire guarantee.
    It is NOT verified that the value, unit, and label for a given axis were
    stated TOGETHER, in the same sentence, table row, or even the same
    paragraph. A caller can supply ``value_quote="1023"`` from one part of
    the paper and ``unit_quote="K"`` from an unrelated part, and this
    function will happily ground both and produce a fully schema-valid
    envelope asserting they belong together. Closing that gap needs a bounded
    measurement-context notion (e.g. requiring value/unit/label quotes to
    fall within one caller-supplied span) that does not exist yet -- it is
    intentionally out of scope for this vertical slice and is its own future
    milestone. See ``TestGroundingIsIndependentPerQuote`` in the test suite
    for a pinning test of this exact gap: if it starts failing, this
    paragraph is stale and must be updated (or removed) alongside the fix.

    Args:
        workspace_root: Root of the campaign workspace holding the evidence
            store.
        sha256: Raw-bytes sha256 of the stored artifact to ground against.
        series_id: Id for the single produced series.
        value_origin: How the numbers were produced, as asserted by the
            caller.
        measurements: One :class:`MeasurementSpec` per axis; at least one
            ``COORDINATE`` and one ``OBSERVATION`` role are required (by the
            schema's own S3/S4 validators). ``CONSTANT`` is not supported by
            this producer.

    Returns:
        The validated envelope.

    Raises:
        DatasetProducerError: Artifact missing/legacy/corrupt, a value quote
            not derivable into a numeral, an unknown unit, or a ``CONSTANT``
            role spec.
        QuoteGroundingError: A quote that cannot be grounded unambiguously.
    """
    for spec in measurements:
        if spec.role is AxisRole.CONSTANT:
            raise DatasetProducerError(
                f"MeasurementSpec for axis {spec.axis_id!r} has role=CONSTANT, which this producer "
                "does not support -- every spec must be a per-point COORDINATE or OBSERVATION"
            )

    extracted, extracted_sha256, derivation_binding_raw, content_type = _load_verified_extracted_text(
        workspace_root, sha256
    )
    node_kind = _CONTENT_TYPE_TO_NODE_KIND.get(content_type)
    if node_kind is None:
        # P1-C: fail closed rather than guess. This producer's single root
        # node must honestly claim what kind of document it is; a
        # content_type this table does not recognise establishes nothing,
        # so no SourceNodeKind may be asserted for it.
        raise DatasetProducerError(
            f"artifact {sha256!r} has content_type={content_type!r}, which does not map to any "
            f"SourceNodeKind this producer may honestly assert (recognised: "
            f"{sorted(_CONTENT_TYPE_TO_NODE_KIND)}); refusing to guess the document kind"
        )
    text = extracted.text
    # P1-D: the artifact's REAL source context and glyph health, derived the
    # same way carmel.services.grounding derives them for the strict core --
    # never hardcoded -- used below as an additional refusal-only canary in
    # _measured_value. See _source_context_for's docstring for the full gap
    # this closes.
    document_source_context = _source_context_for(extracted)
    document_glyph_health = assess_glyph_health(text)
    # CRITICAL: hash extracted.text (the field inside the verified, parsed
    # ExtractedText), NEVER text.txt on disk -- text.txt is presence-checked
    # only, never digest-checked (see ExtractionBinding.extracted_text_sha256's
    # docstring), so a digest of it would be anchored to the one unverified
    # file in the store.
    extracted_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    derivation_binding: str | Absent
    if derivation_binding_raw is None:
        # UNKNOWN is the ONLY reason ExtractionBinding permits here (enforced
        # by its own validator): the digest predates the field and no other
        # AbsenceReason promises a remedy that exists.
        derivation_binding = Absent(reason=AbsenceReason.UNKNOWN)
    else:
        derivation_binding = derivation_binding_raw
    binding = ExtractionBinding(
        extracted_sha256=extracted_sha256,
        extracted_text_sha256=extracted_text_sha256,
        derivation_binding=derivation_binding,
    )
    root_node = SourceNode(
        node_id=_ROOT_NODE_ID,
        kind=node_kind,
        sha256=sha256,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=binding,
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
    )
    graph = SourceGraph(nodes=(root_node,))

    axes: list[AxisDeclaration] = []
    coordinates: list[Coordinate] = []
    observations: list[Observation] = []
    for spec in sorted(measurements, key=lambda spec: spec.axis_id):
        label_locator = ground_quote(text, spec.label_quote, occurrence=spec.label_occurrence)
        axes.append(
            AxisDeclaration(
                axis_id=spec.axis_id,
                role=spec.role,
                quantity_kind=spec.quantity_kind,
                label_raw=spec.label_quote,
                label_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=label_locator),
            )
        )
        value = _measured_value(
            text,
            spec,
            document_source_context=document_source_context,
            document_glyph_health=document_glyph_health,
        )
        if spec.role is AxisRole.COORDINATE:
            coordinates.append(
                Coordinate(
                    axis_id=spec.axis_id,
                    value=value,
                    uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                )
            )
        else:
            observations.append(
                Observation(
                    axis_id=spec.axis_id,
                    value=value,
                    uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                )
            )

    point = DataPoint(
        point_id=_POINT_ID,
        coordinates=tuple(coordinates),
        observations=tuple(observations),
        composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
    )
    series = Series(
        series_id=series_id,
        source_form=SourceForm.TEXTUAL,
        value_origin=value_origin,
        axes=tuple(axes),
        constants=(),
        points=(point,),
    )
    embedded_table = EmbeddedConversionTable(
        sha256=units.TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(units.TABLE_V1.identity_payload()).decode("utf-8"),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        series=(series,),
        conversion_tables=(embedded_table,),
    )
