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
    NUMERAL_CANDIDATE_RE,
    GlyphHealth,
    QuoteRole,
    SourceContext,
    Unresolvable,
    assess_glyph_health,
    enclosing_numeric_construct,
    find_numeral_extent,
    has_clean_token_boundary,
    label_boundary_violation,
    normalize_numeric_span,
    unit_boundary_violation,
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


def _unit_spellings(quantity: QuantityKind) -> frozenset[str]:
    """All registered spellings (known units plus every alias's raw and
    normalized form) for one quantity, per :data:`units.TABLE_V1`.

    Lives HERE, not in :mod:`carmel.services.numeric`, because that module is
    deliberately zero-``carmel.*``-import (see its module docstring and
    ``tests/test_semantic_deps.py::test_numeric_module_has_zero_carmel_imports``,
    which enforces this at the AST level so :func:`compute_dependency_sha`'s
    hash closure stays trustworthy). This module already legitimately imports
    and uses ``carmel.services.units``/``units.TABLE_V1`` directly elsewhere
    (e.g. :func:`units.normalize_unit`), so it is the correct home for
    computing :func:`unit_boundary_violation`'s Layer 3 vocabulary; the
    result is passed down as plain ``frozenset[str]`` data.
    """
    spellings: set[str] = set(units.TABLE_V1.known_units(quantity))
    for alias in units.TABLE_V1.aliases:
        if alias.quantity is quantity:
            spellings.add(alias.raw)
            spellings.add(alias.normalized)
    return frozenset(spellings)


#: Layer 3 admission vocabulary, keyed by the CLAIMED quantity -- a unit
#: quote must be one of ITS quantity's own registered spellings to be
#: admitted at all. Excludes ``QuantityKind.OTHER``, which has no vocabulary
#: (:func:`units.normalize_unit` returns OTHER's raw string unchanged) and is
#: refused outright by :func:`ground_quote`'s Layer 0 before
#: :func:`unit_boundary_violation` is ever consulted. Computed lazily (on
#: first use) and cached, matching the previous in-``numeric.py``
#: implementation's caching behaviour.
_unit_spellings_by_quantity_cache: dict[QuantityKind, frozenset[str]] | None = None

#: Layer 3 maximality vocabulary: the UNION of every quantity's spellings,
#: deliberately NOT scoped to the claimed quantity. Maximality must use the
#: union rather than the claimed quantity's own spellings alone, or a caller
#: could dodge the check entirely by mis-claiming quantity -- e.g. grounding
#: ``"cm"`` inside ``"cm s^-1"`` while claiming LENGTH (whose vocabulary does
#: not contain the VELOCITY alias ``"cm s^-1"``) would otherwise sail through
#: a maximality check scoped only to LENGTH's own spellings.
_unit_spellings_union_cache: frozenset[str] | None = None


def _unit_spellings_by_quantity() -> dict[QuantityKind, frozenset[str]]:
    global _unit_spellings_by_quantity_cache
    if _unit_spellings_by_quantity_cache is None:
        _unit_spellings_by_quantity_cache = {
            q: _unit_spellings(q) for q in QuantityKind if q is not QuantityKind.OTHER
        }
    return _unit_spellings_by_quantity_cache


def _unit_spellings_union() -> frozenset[str]:
    global _unit_spellings_union_cache
    if _unit_spellings_union_cache is None:
        _unit_spellings_union_cache = frozenset().union(*_unit_spellings_by_quantity().values())
    return _unit_spellings_union_cache


def ground_quote(
    text: str,
    quote: str,
    *,
    role: QuoteRole,
    occurrence: int | None = None,
    value_span: tuple[int, int] | None = None,
    quantity: QuantityKind | None = None,
) -> CharSpanLocator:
    """Locate ``quote`` in ``text`` by SEARCHING, returning a half-open
    :class:`CharSpanLocator` over ``TextSpace.EXTRACTED_TEXT``.

    ``role`` is required and keyword-only, deliberately with NO default: one
    boundary rule cannot serve every quote's job. A VALUE quote ("1023") and
    a UNIT quote ("K") glued right after it in "1023K" need the numeral to
    stay groundable and the unit to ALSO stay groundable -- but a LABEL quote
    ("CO") glued to a trailing digit in "CO2" must be refused, because there
    the digit is part of a DIFFERENT token (the species name "CO2"), not a
    value the label is reporting. Collapsing all three into one boundary rule
    either over-refuses the unit case or under-refuses the label case; there
    is no default that is safe for every caller, so every call site must say
    which job its quote is doing. See :class:`carmel.services.numeric.QuoteRole`
    for the full role docstrings.

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
        role: Which job this quote is doing (``VALUE`` / ``UNIT`` /
            ``LABEL``) -- selects which boundary rule applies.
        occurrence: ``None`` to require uniqueness; a 0-based index to
            explicitly select among multiple matches.
        value_span: Only meaningful for ``role=QuoteRole.UNIT``. The
            ``(start, end)`` span of this measurement's own already-grounded
            VALUE quote, used solely to gate UNIT's leading-edge digit-glue
            exception: a unit may abut a preceding digit run only when
            ``value_span`` is supplied and ends exactly where the unit quote
            starts, i.e. the caller is asserting that digit run IS the value
            being reported, not merely some other clean numeral (e.g. a run
            id). Omitted (``None``, the default) means the exception never
            fires -- fail closed.
        quantity: REQUIRED for ``role=QuoteRole.UNIT`` (and refused for every
            other role): the ``carmel.services.units.QuantityKind`` this unit
            quote claims to measure, used to select which quantity's spelling
            vocabulary in ``carmel.services.units.TABLE_V1`` a UNIT quote must
            belong to (D-U2's table-maximality layer). Must be a genuine
            ``QuantityKind`` member, not merely something that ``==``-compares
            to one -- a plain string like ``"temperature"`` compares equal to
            ``QuantityKind.TEMPERATURE`` in dict/set lookups (``StrEnum``) but
            is not a real member, and admitting it would silently bypass the
            vocabulary gate for whatever spellings happen to attach to that
            string. ``QuantityKind.OTHER`` is refused outright:
            ``carmel.services.units.normalize_unit`` returns OTHER's raw
            string unchanged (OTHER has no vocabulary), so there is nothing
            for this gate to verify admission against -- a deliberate, known
            coverage gap, not a bug to route around.

    Returns:
        A ``CharSpanLocator`` with ``text[start:end] == quote``.

    Raises:
        QuoteGroundingError: ``role`` is not a genuine ``QuoteRole`` member
            (checked FIRST, before any of the checks below, so it is never
            masked by a later check surfacing first); for ``role=QuoteRole.UNIT``,
            ``quantity`` is missing, is not a genuine ``QuantityKind`` member,
            or is ``QuantityKind.OTHER``; for any other role, ``quantity`` was
            supplied at all; empty quote; whitespace-padded quote; quote not
            found; ambiguous quote with ``occurrence=None``; ``occurrence``
            out of range; or a role-specific boundary violation.
    """
    if not isinstance(role, QuoteRole):
        # Exhaustive, fail-closed dispatch: EVERY call must resolve to one of
        # the three role-specific boundary rules below, never fall through to
        # the unconditional assert/return at the bottom of this function via
        # unguarded substring search. A role that is not a genuine QuoteRole
        # member -- a plain string that happens to equal a member's value
        # (e.g. "value"), None, or any other object -- has no boundary rule
        # that is safe to apply, so it is refused here, up front, rather than
        # silently skipping every check below (the round-40 regression this
        # guard exists to close for good). This guard runs FIRST, before even
        # the empty-quote check, so an invalid role is never masked by a
        # LATER check surfacing first (round 41: this guard used to run after
        # the empty/whitespace/find/ambiguity/occurrence checks, so an
        # invalid role on an ambiguous, padded, or absent quote surfaced as
        # THAT check's message instead of the role error).
        raise QuoteGroundingError(
            f"ground_quote: role={role!r} is not a carmel.services.numeric.QuoteRole "
            "member -- grounding has no default boundary rule that is safe for every "
            "quote's job, so a role that is not a genuine QuoteRole member (a plain "
            "string, None, or any other object) is refused outright rather than "
            "silently skipping every boundary check"
        )
    # D-U2 Layer 0: quantity is required for -- and only for -- role=UNIT.
    # Runs immediately after the role guard above and before every other
    # check (including the empty-quote check), for the same reason that
    # guard runs first: a caller that gets this argument wrong must see
    # THIS message, never a later, unrelated check's message surfacing
    # first and masking it.
    if role is QuoteRole.UNIT:
        if quantity is None:
            raise QuoteGroundingError(
                "ground_quote: role=QuoteRole.UNIT requires quantity= -- a genuine "
                "carmel.services.units.QuantityKind member naming which quantity this "
                "unit quote claims to measure -- but quantity was not supplied (None); "
                "UNIT grounding cannot check table maximality without knowing which "
                "quantity's vocabulary to check against"
            )
        if not isinstance(quantity, QuantityKind):
            raise QuoteGroundingError(
                f"ground_quote: quantity={quantity!r} is not a genuine "
                "carmel.services.units.QuantityKind member -- QuantityKind is a StrEnum, "
                "so a plain string that happens to equal a member's value (e.g. "
                "'temperature') compares equal in dict/set lookups but is not a real "
                "member; refused outright rather than risk silently checking the wrong "
                "(or no) vocabulary"
            )
        if quantity is QuantityKind.OTHER:
            raise QuoteGroundingError(
                "ground_quote: quantity=QuantityKind.OTHER has no unit vocabulary -- "
                "carmel.services.units.normalize_unit returns OTHER's raw string "
                "unchanged, so OTHER has no known spellings to check admission against; "
                "UNIT grounding for OTHER is refused outright as a deliberate, known "
                "coverage gap, not a bypass"
            )
    elif quantity is not None:
        raise QuoteGroundingError(
            f"ground_quote: quantity={quantity!r} was supplied but role={role!r} is not "
            "QuoteRole.UNIT -- quantity only selects a vocabulary for UNIT-role table "
            "maximality checking and has no meaning for any other role"
        )
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
    if quote != quote.strip():
        # A quote padded with leading/trailing whitespace (e.g. " K" instead
        # of "K") is not a token at all -- it is a slice that happens to
        # include the separator next to the token. Refuse before the search
        # even runs, for every role: no role's boundary rule is meant to
        # accept a quote whose own edges are whitespace.
        raise QuoteGroundingError(
            f"ground_quote: quote {display!r} has leading or trailing whitespace -- a padded "
            "quote is not a token; strip the whitespace and quote the exact token, or widen the "
            "quote to include the adjacent text as its own token if that is what is meant"
        )
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
    if role is QuoteRole.VALUE:
        # Both VALUE branches below are conditional on a property of `quote`
        # itself (is it numeral-shaped, does it already sit at a clean
        # boundary) rather than on `role`, unlike UNIT/LABEL below -- so this
        # wraps both in a single `if role is QuoteRole.VALUE:` and lets
        # neither inner branch raising mean "accept": a VALUE quote that is
        # non-numeral AND already at a clean token boundary falls through
        # both inner checks with nothing to raise, which is the intended
        # accept path, not a missing-branch bug (round 40, defect 1's fix
        # must not turn a legitimate "no violation found" into a spurious
        # refusal).
        if NUMERAL_CANDIDATE_RE.fullmatch(quote):
            # P1-A: a numeral quote must ground to a MAXIMAL numeral candidate, never
            # an interior slice of a strictly larger one, and it must sit at a genuine
            # numeral boundary at all -- e.g. quote "1023" must not silently accept the
            # middle of "11023", "1023.5", "1,023", or "0.51023", and quote "2" must not
            # silently accept the subscript digit inside the identifier "H2". This uses
            # the SAME shared primitive (:data:`carmel.services.numeric.NUMERAL_CANDIDATE_RE`
            # / :func:`carmel.services.numeric.find_numeral_extent`) that
            # :mod:`carmel.services.grounding` uses to scan an evidence window for
            # numeric corroboration -- unifying two independently-grown, and
            # independently-wrong, boundary heuristics onto one grammar. Note the
            # trigger is `fullmatch` against this CORRECTED (signed) grammar, not the
            # old, sign-less one -- so a quote like "-3" now actually reaches this
            # check, where under the old grammar it silently skipped it entirely.
            extent = find_numeral_extent(text, start)
            if extent is None:
                raise QuoteGroundingError(
                    f"ground_quote: quote {display!r} looks like a numeral but does not sit at a "
                    "clean numeral boundary in the supplied text -- it is directly adjacent to a "
                    "letter, digit, dot, or comma that disqualifies it from ever being a standalone "
                    "numeral candidate (e.g. a species subscript like 'H2', a unit power like 'cm3', "
                    "or a comma-grouped thousands digit like the '023' in '1,023')"
                )
            if extent != (start, end):
                raise QuoteGroundingError(
                    f"ground_quote: quote {display!r} is an interior fragment of the larger "
                    f"numeral {text[extent[0]:extent[1]]!r} (span "
                    f"[{extent[0]}:{extent[1]}]) in the supplied text -- a grounded "
                    "numeral span must be the MAXIMAL numeral candidate, never a slice of a "
                    "bigger one; quote the full numeral if that is what is meant"
                )
            # `extent == (start, end)` only proves the quote is a whole numeral ON
            # ITS OWN -- it does not prove the numeral is not itself just one piece
            # of a larger multi-token construct (a mangled ASCII-6 uncertainty
            # marker, a spaced range, or a flattened scientific-notation triple).
            # `enclosing_numeric_construct` answers exactly that strictly-weaker
            # question. Each construct gets its own message (not just a distinct
            # exception type) so operators -- and tests -- can tell the refusals
            # apart from message content alone; this project has hit masked,
            # indistinguishable refusals of this shape before.
            construct = enclosing_numeric_construct(text, start, end)
            if construct == "ascii6_uncertainty":
                raise QuoteGroundingError(
                    f"ground_quote: quote {display!r} is one piece of a mangled "
                    "ascii6_uncertainty marker (e.g. '307 6 10' where '±' was OCR'd as "
                    "a bare '6' between a value and its uncertainty) -- quoting only "
                    "the value, the digit '6', or the uncertainty in isolation loses "
                    "the marker's meaning; refuse rather than ground a fragment of it"
                )
            if construct == "spaced_range":
                raise QuoteGroundingError(
                    f"ground_quote: quote {display!r} is one endpoint of a "
                    "spaced_range whose dash has whitespace on at least one side "
                    "(e.g. '1000 - 1200' or '1000 – 1200') -- quoting only one "
                    "endpoint silently drops the other bound; quote the full range "
                    "if that is what is meant"
                )
            if construct == "flattened_scientific":
                raise QuoteGroundingError(
                    f"ground_quote: quote {display!r} is one piece of a "
                    "flattened_scientific notation triple (e.g. '3.94 x 10 03' where "
                    "the exponent was split from its base by OCR) -- quoting only the "
                    "base or only the exponent loses the value; quote the full triple "
                    "if that is what is meant"
                )
        elif not has_clean_token_boundary(text, start, end):
            # A VALUE quote that is not itself a numeral candidate (rare, but
            # possible if a caller passes a non-numeral VALUE quote) still needs
            # SOME boundary guard rather than plain, unguarded substring search.
            # See carmel.services.numeric.has_clean_token_boundary for the
            # per-edge letter/digit boundary rule (deliberately asymmetric: a
            # digit immediately before a letter-initial quote is fine, mirroring
            # NUMERAL_EXTENT_RE's own trailing-boundary asymmetry).
            raise QuoteGroundingError(
                f"ground_quote: quote {display!r} does not sit at a clean word/token "
                "boundary in the supplied text -- it is directly adjacent to another "
                "character of the same class (a letter next to a letter, or a digit "
                "next to a digit, '.', or ',') that makes it look like a fragment of "
                "a larger token, not a standalone quote; quote the full token if "
                "that is what is meant"
            )
        # else: quote is a clean, non-numeral VALUE token already at a clean
        # boundary -- nothing to raise, fall through to the self-check below.
    elif role is QuoteRole.UNIT:
        # A unit quote (role UNIT) uses its own three-layer boundary rule,
        # not has_clean_token_boundary above (see
        # carmel.services.numeric.unit_boundary_violation for the full
        # rule): Layer 1 partitions the character space into delimiters,
        # unit-token characters, and an unclassified bucket that fails
        # closed; Layer 2 requires each edge to already be maximal over
        # unit-token characters (with a narrow leading-edge exception for
        # a value the unit is genuinely glued to, gated by value_span
        # matching exactly, not merely "some clean numeral"); Layer 3
        # requires the quote to be a registered spelling for the claimed
        # quantity and to be maximal against the union of ALL quantities'
        # spellings (so a caller cannot dodge maximality by mis-claiming
        # quantity). Each distinguishable cause gets its own message,
        # mirroring the enclosing_numeric_construct pattern above -- this
        # project has hit masked, indistinguishable refusals of this shape
        # before.
        # Layer 0 above already guarantees quantity is a genuine, non-OTHER
        # QuantityKind for every role=UNIT call that reaches here (it raises
        # first otherwise); this assert only makes that guarantee visible to
        # mypy, which cannot correlate the two separate `role is QuoteRole.UNIT`
        # branches on its own.
        assert isinstance(quantity, QuantityKind)
        violation = unit_boundary_violation(
            text,
            start,
            end,
            value_span=value_span,
            quantity_spellings=_unit_spellings_by_quantity().get(quantity, frozenset()),
            all_spellings=_unit_spellings_union(),
        )
        if violation == "unit_leading_not_maximal":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} does not sit at a clean unit "
                "boundary in the supplied text -- its LEADING edge is directly "
                "adjacent to another unit-token character, so this quote is not "
                "maximal at that edge -- that makes it look like a fragment of a "
                "larger unit or token (e.g. 'C' inside 'mC', or 'K' inside 'mK'); "
                "quote the full unit if that is what is meant"
            )
        if violation == "unit_leading_unclassified_char":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} does not sit at a clean unit "
                "boundary in the supplied text -- its LEADING edge is directly "
                "adjacent to a character that is neither whitespace, a recognised "
                "unit-boundary delimiter, nor a unit-token character -- an "
                "unclassified neighbour is refused outright rather than guessed at; "
                "quote the full unit if that is what is meant"
            )
        if violation == "unit_trailing_not_maximal":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} does not sit at a clean unit "
                "boundary in the supplied text -- its TRAILING edge is directly "
                "adjacent to another unit-token character (beyond any trimmed "
                "trailing bare-operator run), so this quote is not maximal at that "
                "edge -- that makes it look like a fragment of a larger unit (e.g. "
                "'cm3' inside 'cm3/mol/s', or 'K' inside 'K*cm'); quote the full "
                "unit if that is what is meant"
            )
        if violation == "unit_trailing_unclassified_char":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} does not sit at a clean unit "
                "boundary in the supplied text -- its TRAILING edge is directly "
                "adjacent to a character that is neither whitespace, a recognised "
                "unit-boundary delimiter, nor a unit-token character -- an "
                "unclassified neighbour is refused outright rather than guessed at; "
                "quote the full unit if that is what is meant"
            )
        if violation == "unit_not_in_vocabulary":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is not a registered spelling "
                "(known unit or alias, raw or normalized) for the claimed quantity "
                "in carmel.services.units.TABLE_V1 -- an unrecognised unit spelling "
                "is refused rather than admitted on the strength of merely sitting "
                "at a clean boundary; register the spelling in TABLE_V1 first if it "
                "is genuinely a unit this project should understand"
            )
        if violation == "unit_not_maximal_forward":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is a registered spelling, but "
                "a LONGER registered spelling (for some quantity) starts at the same "
                "position and extends past this quote's end -- e.g. quoting 'cm' "
                "where the text actually reads the registered VELOCITY alias "
                "'cm s^-1' -- quote the full, longer unit spelling instead"
            )
        if violation == "unit_not_maximal_backward":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is a registered spelling, but "
                "a LONGER registered spelling (for some quantity) starts before this "
                "quote and ends at the same position -- this quote is a suffix "
                "fragment of that longer spelling; quote the full, longer unit "
                "spelling instead"
            )
        if violation == "unit_digit_glue_no_value_span":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is glued to a preceding digit "
                "run, but no value_span was supplied to confirm that digit run is "
                "this measurement's own value -- fail closed: without value_span the "
                "leading-edge digit-glue exception never fires, even if the digit run "
                "is itself a clean numeral (e.g. '1023K' is refused exactly like "
                "'run3K' when neither call passes value_span); pass the already-"
                "grounded VALUE locator's span to assert this digit run IS the value "
                "being reported"
            )
        if violation == "unit_digit_glue_value_span_malformed":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is glued to a preceding digit "
                "run, and the supplied value_span is malformed or out of range for "
                "the supplied text -- value_span must be a (start, end) pair of ints "
                "with 0 <= start < end <= len(text); a malformed span proves nothing "
                "about this measurement's value, so the digit-glue exception is "
                "refused rather than risk an IndexError/TypeError from inside this gate"
            )
        if violation == "unit_digit_glue_value_span_mismatch":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is glued to a preceding digit "
                "run, but the supplied value_span does not end exactly where this "
                "quote starts -- a unit may only abut a digit run on its leading edge "
                "when that digit run IS the value it is reporting (e.g. the '1023' in "
                "'1023K'), and value_span is how the caller asserts that; a value_span "
                "ending anywhere else is not a claim about THIS glued digit run"
            )
        if violation == "unit_digit_glue_value_span_not_clean_numeral":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is glued to a preceding digit "
                "run, and the supplied value_span ends exactly where this quote "
                "starts, but that span is not itself a clean, maximal numeral in "
                "place -- value_span must name a genuine numeral extent (as "
                "carmel.services.numeric.find_numeral_extent computes it), never a "
                "fabricated span drawn around an arbitrary digit run (e.g. a caller "
                "cannot claim (0, 4) over 'run3K' or (0, 6) over 'case 1K' just "
                "because the span's end lines up with the unit quote's start)"
            )
    elif role is QuoteRole.LABEL:
        # A label quote (role LABEL) is the strictest role: each edge is
        # refused unless the neighbouring character is whitespace,
        # start/end of text, or on the shared LABEL separator allowlist --
        # unlike UNIT there is no "glued value" shape a label is ever
        # allowed to sit inside, and unlike the old letter/digit-only
        # denylist this also refuses punctuation-glued fragments like the
        # 'T' inside '1/T' or the 'CO' inside 'X_CO'. See
        # carmel.services.numeric.label_boundary_violation.
        violation = label_boundary_violation(text, start, end)
        if violation == "label_leading_adjacency":
            raise QuoteGroundingError(
                f"ground_quote: label quote {display!r} does not sit at a clean "
                "label boundary in the supplied text -- its LEADING edge is directly "
                "adjacent to a character that is not whitespace, start-of-text, or a "
                "permitted separator (e.g. 'CO' inside 'X_CO', or 'T' inside '1/T') "
                "that makes it look like a fragment of a larger species/label token, "
                "not a standalone label; quote the full label if that is what is meant"
            )
        if violation == "label_trailing_adjacency":
            raise QuoteGroundingError(
                f"ground_quote: label quote {display!r} does not sit at a clean "
                "label boundary in the supplied text -- its TRAILING edge is directly "
                "adjacent to a character that is not whitespace, end-of-text, or a "
                "permitted separator (e.g. 'CO' inside 'CO2 mole fraction', or 'H2' "
                "inside 'H2/CO') that makes it look like a fragment of a larger "
                "species/label token, not a standalone label; quote the full label if "
                "that is what is meant"
            )
    else:
        # Unreachable given the isinstance(role, QuoteRole) guard above --
        # QuoteRole has exactly three members (VALUE / UNIT / LABEL), all
        # handled above. Kept as an explicit, fail-closed backstop so a
        # FUTURE QuoteRole member added without a corresponding branch here
        # raises loudly instead of silently falling through to the
        # unconditional assert/return below -- exactly the round-40 defect
        # this function exists to never repeat.
        raise QuoteGroundingError(
            f"ground_quote: role={role!r} is a QuoteRole member with no boundary rule "
            "implemented in this function -- add one before using this role, never "
            "fall through to unguarded substring search"
        )
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
    value_locator = ground_quote(
        text, spec.value_quote, role=QuoteRole.VALUE, occurrence=spec.value_occurrence
    )
    # P1: pass the VALUE locator's own span so UNIT's leading-edge digit-glue
    # exception (see carmel.services.numeric.unit_boundary_violation) can
    # require the glued digit run to be THIS measurement's own value, not
    # merely some other clean numeral (e.g. a run id) that happens to sit
    # before the unit quote.
    unit_locator = ground_quote(
        text,
        spec.unit_quote,
        role=QuoteRole.UNIT,
        occurrence=spec.unit_occurrence,
        value_span=(value_locator.start, value_locator.end),
        quantity=spec.quantity_kind,
    )
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
) -> tuple[ExtractedText, str, str, str]:
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
    checked at all. Several checks close that gap, each refusing with a
    message naming exactly which one failed:

    1. ``meta.sha256 == sha256``: ``load_artifact_meta`` resolves the store
       directory purely from the ``sha256`` PARAMETER (via its own
       sha-shape/containment validation) and does NOT cross-check that
       parameter against the ``sha256`` FIELD recorded inside the loaded
       ``meta.json`` -- so a ``meta.json`` whose ``sha256`` field disagrees
       with the directory it lives in (e.g. hand-edited or copied from
       elsewhere) would otherwise go undetected.
    2. round-36: :func:`~carmel.services.evidence.verify_artifact` with
       ``deep=False`` -- runs BEFORE the legacy carve-outs in step 3, on
       purpose. An artifact that is both legacy (predates
       ``extracted_sha256``/``derivation_binding``) AND corrupt (``raw.bin``
       no longer hashes to ``sha256``, or its sidecar no longer matches its
       recorded digest) must be refused as CORRUPT, not waved through to a
       legacy carve-out whose message reads as routine and names the wrong
       problem. Checking shallow integrity first, before either carve-out
       runs, is what makes that ordering guarantee hold.
    3. Two legacy carve-outs, each refused with its own named cause rather
       than falling through to the generic ``verify_artifact`` failure above
       (which would otherwise report the same message for both, and for
       every other kind of corruption besides): an artifact that predates
       ``extracted_sha256`` (its sidecar can never be verified at all), and
       one that predates ``derivation_binding`` but does carry
       ``extracted_sha256``. round-36: this carve-out's raise means
       :func:`produce_envelope_from_artifact` is never even reached for such
       an artifact -- an earlier version of this docstring described that
       function as handling this case "below via ``AbsenceReason.UNKNOWN``",
       which was never true: the ``None`` this function's own return type
       still admits for ``derivation_binding`` is unreachable in practice,
       because every path that would produce it raises first, right here.
    4. :func:`carmel.services.evidence.verify_artifact` with ``deep=True``:
       confirms ``raw.bin`` exists and hashes to ``sha256`` (the parameter),
       that ``extracted.json`` matches its recorded digest (the same check
       performed by hand below, but this call also covers ``raw.bin``, which
       nothing here otherwise touches), and -- because every artifact reaching
       this call already carries a non-``None`` ``extracted_sha256`` and
       ``derivation_binding`` (checks 2 above ran first) -- that the recorded
       ``derivation_binding`` is internally consistent: recomputed from
       ``meta.json``'s own ``extractor_version``/``sha256``/``extracted_sha256``
       fields, it still matches the recorded value. That closes the gap this
       function used to leave open: ``derivation_binding`` is carried verbatim
       into the produced envelope (see ``produce_envelope_from_artifact``
       below), so it is worth re-checking rather than trusted blind.

       Read exactly what this buys, and no more -- quoting
       :data:`~carmel.schemas.literature.StoredArtifact.derivation_binding`'s
       own caveat: it proves only INTERNAL CONSISTENCY of the ``meta.json``
       record -- that ``extracted_sha256`` was not changed independently of
       ``derivation_binding`` after the two were bound together at store
       time. It is NOT proof that ``extracted.json`` was actually re-derived
       from ``raw.bin``, and it is no defence against a forger who swaps the
       sidecar AND updates ``extracted_sha256`` AND recomputes
       ``derivation_binding`` to match, all together, consistently -- that
       forgery passes every check here undetected.

       Because :func:`verify_artifact` returns a plain ``bool`` with no record
       of which of its internal checks failed, a ``False`` here cannot be
       narrated as one specific cause: it might be ``raw.bin`` missing or not
       hashing to ``sha256``, ``extracted.json`` not matching its recorded
       digest, or a stale/inconsistent ``derivation_binding`` -- the refusal
       message below says so honestly rather than guessing.
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
        # round-36: this shallow integrity check MUST run before the legacy
        # carve-outs below. An artifact that is BOTH legacy (predates
        # extracted_sha256/derivation_binding) AND corrupt (raw.bin no longer
        # hashes to sha256, or its sidecar no longer matches its recorded
        # digest) used to reach a legacy carve-out first and be refused with a
        # "predates ..." message that named the wrong problem -- masking a
        # real integrity failure behind a message that reads as routine
        # legacy handling. Checking integrity first ensures corruption is
        # always named as corruption, on legacy artifacts included.
        raise DatasetProducerError(
            f"artifact {sha256!r} failed integrity verification (raw.bin missing, its bytes not "
            "hashing to sha256, meta.json unreadable, or extracted.json not matching its recorded "
            "digest -- verify_artifact reports only a plain bool, not which check failed); refusing "
            "to use unverified bytes"
        )
    if meta.extracted_sha256 is None:
        # A legacy artifact stored before extracted_sha256 existed carries no
        # digest for its sidecar, so its extracted.json cannot be verified at
        # all -- and this producer never parses unverified bytes. Checked
        # before verify_artifact(deep=True) below: that call fails closed for
        # this same artifact too, but with no way to say why.
        raise DatasetProducerError(
            f"artifact {sha256!r} predates extracted_sha256 and its extracted.json cannot be "
            "verified; refusing to parse unverified bytes"
        )
    if meta.derivation_binding is None:
        # A legacy artifact stored before derivation_binding existed (but
        # after extracted_sha256 did) has a verifiable sidecar yet nothing to
        # deep-verify -- also checked before verify_artifact(deep=True),
        # which would otherwise fail this artifact for the same underlying
        # reason as a genuinely stale binding, indistinguishably.
        raise DatasetProducerError(
            f"artifact {sha256!r} predates derivation_binding and its extractor identity cannot be "
            "bound to its extracted bytes; refusing to carry an unverifiable binding forward"
        )
    if not verify_artifact(workspace_root, sha256, deep=True):
        raise DatasetProducerError(
            f"artifact {sha256!r} failed verify_artifact (raw.bin digest, meta.json, extracted.json "
            "digest, or derivation_binding consistency -- verify_artifact reports only a plain bool, "
            "not which check failed); refusing to use unverified bytes"
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

    extracted, extracted_sha256, derivation_binding, content_type = _load_verified_extracted_text(
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
    # round-36: no Absent(reason=AbsenceReason.UNKNOWN) branch belongs here.
    # _load_verified_extracted_text already raises DatasetProducerError for
    # any artifact whose meta.derivation_binding is None (the legacy
    # carve-out at its own "predates derivation_binding" check) -- this
    # function is never reached for such an artifact at all, so
    # derivation_binding here is always the verified str the type says it is.
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
        label_locator = ground_quote(
            text, spec.label_quote, role=QuoteRole.LABEL, occurrence=spec.label_occurrence
        )
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
