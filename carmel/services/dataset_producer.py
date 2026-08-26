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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from carmel.agents.tools.extract import ExtractedText
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisRole,
    CharSpanLocator,
    DatasetEnvelope,
    EmbeddedConversionTable,
    ExtractedTextVerification,
    ExtractionBinding,
    MeasuredValue,
    RawArtifactVerification,
    RootSidecarVerification,
    SemanticDependencyUse,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SourceVerification,
    TableCellLocator,
    TextSpace,
    ValueOrigin,
)
from carmel.schemas.literature import StoredArtifact
from carmel.services import units
from carmel.services.dataset_store import (
    CanonicalDecimalError,
    canonical_decimal,
    canonical_json_bytes,
)
from carmel.services.evidence import artifact_dir, load_artifact_meta
from carmel.services.extraction_record import (
    _PYPDF_DEPENDENT_EXTRACTORS,
    CurrentSelectionKind,
    load_extraction_record,
    select_current_extraction,
)
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

_RAW_NAME = "raw.bin"
"""Filename of the stored raw bytes inside an artifact's content-addressed
directory. This is the evidence store's PUBLIC on-disk contract --
``carmel.services.evidence``'s module docstring documents the layout as exact
("Layout, exactly::") -- not private knowledge duplicated here."""

_ROOT_EXTRACTED_NAME = "extracted.json"
"""Filename of the ROOT ``ExtractedText`` sidecar, the legacy tier.

Read at exactly one place, :func:`_root_sidecar_claim`, and for exactly one
purpose: to compute a claim a reader can refute. It is NEVER grounding input --
text read from it is precisely what the corpus gate refuses without an explicit
operator opt-in, and routing it into an envelope would launder unauthenticated
text into an address the corpus treats as authenticated."""

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


@dataclass(frozen=True, slots=True)
class _ActiveTableBinding:
    """Every unit-table-derived artifact this module reads, bound to ONE
    :class:`units.ConversionTable` by construction.

    This module used to read ``units.TABLE_V1`` at four independent sites
    (the spelling vocabulary below, unit normalization, the recorded
    ``conversion_table_sha256``, and the embedded table in the produced
    envelope); those sites agreed only by coincidence of each one separately
    naming the same global. Binding all four derived artifacts into one
    frozen object, built by a single pure constructor, makes disagreement
    UNREPRESENTABLE rather than merely unlikely: there is no seam left where
    one site could reference a different table than the others.

    :meth:`derive` is the ONLY constructor path and is a pure function of its
    ``table`` argument -- it must never reference ``units.TABLE_V1`` itself,
    or the binding it built would once again be able to disagree with the
    table it claims to be derived from.
    """

    table: units.ConversionTable
    spellings_by_quantity: Mapping[QuantityKind, frozenset[str]]
    spellings_union: frozenset[str]
    whitespace_patterns: Mapping[str, re.Pattern[str]]
    embedded: EmbeddedConversionTable

    @classmethod
    def derive(cls, table: units.ConversionTable) -> _ActiveTableBinding:
        """Derive every artifact this module needs from ``table`` alone.

        Lives HERE, not in :mod:`carmel.services.numeric`, because that
        module is deliberately zero-``carmel.*``-import (see its module
        docstring and
        ``tests/test_semantic_deps.py::test_numeric_module_has_zero_carmel_imports``,
        which enforces this at the AST level so
        :func:`compute_dependency_sha`'s hash closure stays trustworthy).
        This module already legitimately imports and uses
        ``carmel.services.units`` directly elsewhere (e.g.
        :func:`units.normalize_unit`), so it is the correct home for
        computing :func:`_unit_table_boundary_violation`'s Layer 3
        vocabulary.
        """
        spellings_by_quantity_mut: dict[QuantityKind, frozenset[str]] = {}
        for quantity in QuantityKind:
            if quantity is QuantityKind.OTHER:
                # Layer 3 admission vocabulary excludes ``QuantityKind.OTHER``,
                # which has no vocabulary (:func:`units.normalize_unit`
                # returns OTHER's raw string unchanged) and is refused
                # outright by :func:`ground_quote`'s Layer 0 before
                # :func:`_unit_table_boundary_violation` is ever consulted.
                continue
            found: set[str] = set(table.known_units(quantity))
            for alias in table.aliases:
                if alias.quantity is quantity:
                    found.add(alias.raw)
                    found.add(alias.normalized)
            spellings_by_quantity_mut[quantity] = frozenset(found)
        # A frozen dataclass does not freeze its NESTED containers -- a plain
        # dict/set assigned to a frozen field would still be mutable in
        # place. Wrap in MappingProxyType/frozenset explicitly, exactly as
        # the module-level constants this replaces used to.
        spellings_by_quantity = MappingProxyType(dict(spellings_by_quantity_mut))

        # Layer 3 maximality vocabulary: the UNION of every quantity's
        # spellings, deliberately NOT scoped to the claimed quantity.
        # Maximality must use the union rather than the claimed quantity's
        # own spellings alone, or a caller could dodge the check entirely by
        # mis-claiming quantity -- e.g. grounding ``"cm"`` inside
        # ``"cm s^-1"`` while claiming LENGTH (whose vocabulary does not
        # contain the VELOCITY alias ``"cm s^-1"``) would otherwise sail
        # through a maximality check scoped only to LENGTH's own spellings.
        spellings_union: frozenset[str] = frozenset().union(*spellings_by_quantity.values())

        # P1-1 (round-43 review): a registered multi-token spelling's
        # separator is not necessarily a single ASCII space in the SOURCE
        # TEXT -- a PDF/XML extraction can glue tokens together with a
        # double space, an NBSP (U+00A0), a newline, or a tab. Exact
        # ``str.startswith``/slice comparisons (as Layer 3 maximality
        # originally used) treat those as a DIFFERENT string from the
        # registered spelling, so a whitespace variant of a longer
        # registered alias silently fails to trigger maximality and a
        # fragment is wrongly admitted under the wrong quantity (e.g.
        # grounding ``"cm"`` as LENGTH inside ``"cm s^-1"``, a VELOCITY
        # alias). This maps every whitespace-containing registered spelling
        # to a precompiled regex that matches the SAME parts joined by
        # ``\s+`` (one-or-more characters of ANY whitespace, Unicode mode --
        # verified to match ASCII space runs, NBSP, and newline/tab), so
        # maximality can detect a whitespace-glued occurrence of a longer
        # spelling without ever rewriting or normalizing the source text
        # itself. ADMISSION (is ``text[start:end]`` itself a registered
        # spelling) deliberately stays EXACT, not whitespace-equivalent --
        # see :func:`_unit_table_boundary_violation`.
        whitespace_patterns = MappingProxyType(
            {
                spelling: re.compile(r"\s+".join(re.escape(part) for part in spelling.split(" ")))
                for spelling in spellings_union
                if any(ch.isspace() for ch in spelling)
            }
        )

        canonical_json = canonical_json_bytes(table.identity_payload()).decode("utf-8")
        embedded = EmbeddedConversionTable(sha256=table.sha256, canonical_json=canonical_json)
        if embedded.sha256 != table.sha256 or embedded.canonical_json != canonical_json:
            raise DatasetProducerError(
                f"internal invariant violated deriving _ActiveTableBinding from table "
                f"{table.table_id!r}: the embedded table must record exactly this table's own "
                f"sha256 and canonical identity payload, never a different table's"
            )

        return cls(
            table=table,
            spellings_by_quantity=spellings_by_quantity,
            spellings_union=spellings_union,
            whitespace_patterns=whitespace_patterns,
            embedded=embedded,
        )


#: Computed ONCE at import time as an immutable constant rather than lazily
#: behind a mutable module-global -- ``units.TABLE_V1`` is itself a fixed,
#: import-time-constant table, so there is nothing to gain from deferring
#: this computation, and a mutable lazy cache is one more piece of global
#: state that does not need to exist. This is the ONLY place
#: ``units.TABLE_V1`` is named in this module; every other site reads
#: through ``_ACTIVE`` so they cannot independently drift apart.
_ACTIVE: _ActiveTableBinding = _ActiveTableBinding.derive(units.TABLE_V1)


def _unit_table_boundary_violation(
    text: str, start: int, end: int, quantity: QuantityKind, binding: _ActiveTableBinding
) -> str | None:
    """Layer 3 of the UNIT-role boundary gate (D-U2): table-driven admission
    plus maximality against ``binding``'s registered vocabulary.

    Moved HERE from :func:`carmel.services.numeric.unit_boundary_violation`
    (round-43 review, P1-2): that function is an EXPORTED, public API, so a
    vocabulary fed to it via caller-supplied keyword arguments
    (``quantity_spellings=``/``all_spellings=``) was an unvalidated
    policy-injection seam -- any caller could pass a fabricated or empty
    vocabulary and every Layer-3 refusal below would simply never fire, with
    nothing in that function (or anywhere else) to catch it. Rather than add
    validation for caller-supplied policy data (this project's sixth
    instance of exactly that anti-pattern), the fix REMOVES the seam: this
    function is PRIVATE and takes no vocabulary parameter from an arbitrary
    caller.

    ``binding`` IS parameterised (round-N review, dataset-replay admission
    gap): unlike the vocabulary this function used to read directly off the
    module-global :data:`_ACTIVE`, ``binding`` is now an explicit argument so
    that :mod:`carmel.services.dataset_replay` can re-run this exact check
    against a DIFFERENT, registry-bounded binding derived from the envelope's
    OWN recorded ``conversion_table_sha256`` -- never against whatever table
    happens to be current. This is not a new policy-injection seam: the
    caller still cannot supply an arbitrary vocabulary, only select AMONG
    ``_ActiveTableBinding`` objects that are each themselves derived, via the
    same pure :meth:`_ActiveTableBinding.derive`, from a table already
    registered in :data:`carmel.services.units.TABLES_BY_SHA`. This
    module's own production call site below passes :data:`_ACTIVE`
    unconditionally, so producer behavior is unchanged.
    :func:`ground_quote` calls :func:`carmel.services.numeric.unit_boundary_violation`
    (the lexical Layers 1-2) first and only calls this function if that
    returns ``None`` clean, preserving the original layer order.

    Returns the same discriminant strings the old in-``numeric.py``
    implementation returned, unchanged in name and meaning:

    - ``"unit_not_in_vocabulary"``: ``text[start:end]`` is not a member of
      the CLAIMED quantity's registered spellings. Admission is EXACT --
      not whitespace-equivalent -- deliberately: :func:`units.normalize_unit`
      only strips LEADING/TRAILING whitespace, not internal whitespace, so
      admitting a quote like ``"cm  s^-1"`` (double space) here would accept
      a string that the normalizer then rejects. A gate/normalizer
      disagreement is worse than a refusal, so this quote is refused; the
      real fix is teaching ``normalize_unit`` to collapse internal
      whitespace, which is out of scope here.
    - ``"unit_not_maximal_forward"`` / ``"unit_not_maximal_backward"``: a
      longer registered spelling (any quantity, via ``binding``'s
      ``spellings_union``) shares this quote's start (forward) or end
      (backward). Unlike admission, maximality IS whitespace-equivalent
      (``binding``'s ``whitespace_patterns``) for any spelling that contains
      whitespace, so a whitespace-glued occurrence of a longer spelling in
      the SOURCE TEXT is still detected even when its separator is not a
      single ASCII space.
    """
    quote_len = end - start
    quantity_spellings = binding.spellings_by_quantity.get(quantity, frozenset())
    if text[start:end] not in quantity_spellings:
        return "unit_not_in_vocabulary"

    for spelling in binding.spellings_union:
        pattern = binding.whitespace_patterns.get(spelling)
        if pattern is not None:
            match = pattern.match(text, start)
            if match is not None and match.end() > end:
                return "unit_not_maximal_forward"
            continue
        if len(spelling) > quote_len and text.startswith(spelling, start):
            return "unit_not_maximal_forward"

    for spelling in binding.spellings_union:
        pattern = binding.whitespace_patterns.get(spelling)
        if pattern is not None:
            for match in pattern.finditer(text, 0, end):
                if match.end() == end and match.start() < start:
                    return "unit_not_maximal_backward"
            continue
        if len(spelling) <= quote_len:
            continue
        cand_start = end - len(spelling)
        if cand_start < 0 or cand_start >= start:
            continue
        if text[cand_start:end] == spelling:
            return "unit_not_maximal_backward"

    return None


#: Registry-bounded lookup from a table's own sha256 to the
#: :class:`_ActiveTableBinding` derived from it, covering EXACTLY the tables
#: :data:`carmel.services.units.TABLES_BY_SHA` already hand-reviews -- never
#: an arbitrary caller-supplied table. Built once at import time (mirroring
#: :data:`_ACTIVE`'s own rationale: a fixed table registry needs no lazy
#: mutable cache) so :func:`carmel.services.dataset_replay.replay_envelope`
#: can re-run Layer 3 against the binding for an envelope's RECORDED
#: ``conversion_table_sha256`` -- an unknown sha simply has no entry here and
#: must be treated as UNVERIFIABLE, never silently mapped onto ``_ACTIVE``.
_BINDINGS_BY_SHA: Mapping[str, _ActiveTableBinding] = MappingProxyType(
    {sha: _ActiveTableBinding.derive(table) for sha, table in units.TABLES_BY_SHA.items()}
)


def binding_for_known_sha(sha256: str) -> _ActiveTableBinding | None:
    """Look up the :class:`_ActiveTableBinding` for a conversion-table
    ``sha256`` that is a member of the hand-reviewed
    :data:`carmel.services.units.TABLES_BY_SHA` registry.

    Returns ``None`` for a sha256 this registry does not recognise -- there
    is deliberately no fallback to :data:`_ACTIVE` or any other table: a
    caller that cannot resolve a KNOWN binding for the exact sha it holds
    must treat that value as UNVERIFIABLE, never re-check it against a
    different table than the one it actually recorded.
    """
    return _BINDINGS_BY_SHA.get(sha256)


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
                    f"numeral {text[extent[0] : extent[1]]!r} (span "
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
        # not has_clean_token_boundary above: Layer 1 (in
        # carmel.services.numeric.unit_boundary_violation) partitions the
        # character space into delimiters, unit-token characters, and an
        # unclassified bucket that fails closed; Layer 2 (same function)
        # requires each edge to already be maximal over unit-token
        # characters (with a narrow leading-edge exception for a value the
        # unit is genuinely glued to, gated by value_span matching exactly,
        # not merely "some clean numeral"); Layer 3 (private to this module,
        # _unit_table_boundary_violation below -- see its docstring for why
        # it does not live in carmel.services.numeric) requires the quote to
        # be a registered spelling for the claimed quantity and to be
        # maximal against the union of ALL quantities' spellings (so a
        # caller cannot dodge maximality by mis-claiming quantity). Each
        # distinguishable cause gets its own message, mirroring the
        # enclosing_numeric_construct pattern above -- this project has hit
        # masked, indistinguishable refusals of this shape before.
        # Layer 0 above already guarantees quantity is a genuine, non-OTHER
        # QuantityKind for every role=UNIT call that reaches here (it raises
        # first otherwise); this assert only makes that guarantee visible to
        # mypy, which cannot correlate the two separate `role is QuoteRole.UNIT`
        # branches on its own.
        assert isinstance(quantity, QuantityKind)
        # Layers 1-2 (lexical, no vocabulary) first; Layer 3 (table-driven,
        # private to this module -- see _unit_table_boundary_violation) only
        # if the lexical check comes back clean, preserving the original
        # layer order from the single three-layer function this was split
        # out of.
        violation = unit_boundary_violation(text, start, end, value_span=value_span)
        if violation is None:
            violation = _unit_table_boundary_violation(text, start, end, quantity, _ACTIVE)
        # unit_leading_not_maximal and unit_leading_unclassified_char (like
        # their trailing counterparts below) overlap in EFFECT -- both refuse
        # a quote whose edge is not clean -- but differ in DIAGNOSIS: the
        # first fires when the adjacent character IS a unit-token character
        # (a fragment of a larger unit), the second when it is neither
        # whitespace, a delimiter, NOR a unit-token character (something
        # Layer 1 could not classify at all). This is deliberate defence in
        # depth, not redundancy to be merged or deleted: keeping the two
        # discriminants (and the message-pinning tests that keep them
        # separable) distinct lets an operator reading a refusal tell "this
        # looks like a real unit fragment" from "this text has something
        # unexpected right next to the quote" without re-deriving it from
        # the source text by hand.
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
        # unit_trailing_not_maximal / unit_trailing_unclassified_char: the
        # same overlap-in-effect, differ-in-diagnosis relationship as the
        # LEADING pair above -- see that comment. Also note
        # unit_trailing_exponent_or_footnote_ambiguous (checked earlier,
        # inside unit_boundary_violation itself) is carved OUT of this
        # not_maximal bucket for a superscript/subscript-digit neighbour
        # specifically, so that ambiguity gets its own name instead of
        # collapsing into this generic one.
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
        if violation == "unit_trailing_exponent_or_footnote_ambiguous":
            raise QuoteGroundingError(
                f"ground_quote: unit quote {display!r} is immediately followed by a "
                "superscript or subscript digit, which is lexically identical, from a "
                "single-adjacent-character check alone, to either a genuine exponent "
                "continuation of this unit or an unrelated footnote marker -- this "
                "ambiguity is refused outright rather than guessed at; quote the full "
                "unit including its exponent if that is what is meant, or otherwise "
                "disambiguate the source text"
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

    DORMANT -- and, of the six dataset-extraction constructs, the most
    misleading one to read on its own, because everything above describes a
    real interaction: a caller states meaning, ``ground_quote`` computes
    offsets, a ``MeasurementSpec`` gets built. What the paragraphs above do
    not say is that this object has no consumer that can ever finish the
    job. It exists solely as the parameter type for
    :func:`produce_envelope_from_artifact` below, and that function refuses
    EVERY call unconditionally -- see its docstring for the full argument,
    which this note only summarizes: this runtime can only locate a value
    with a ``CharSpanLocator`` into extracted running text, and
    :meth:`~carmel.schemas.datasets.DatasetEnvelope._validate_no_char_span_grounds_a_series_value`
    (V7) rejects a char span as the source of a series VALUE, because a
    series asserts a structured pairing of coordinates to observations that
    running text carries no row structure to prove. So a caller may
    construct a well-formed ``MeasurementSpec`` -- its own validation below
    still runs, and still matters, because a malformed spec must fail
    loudly even though no spec, malformed or not, can ever succeed -- and
    hand it to the producer, and the producer will refuse it every time,
    reporting it back only by count (see :func:`_describe_count`).
    Restoring this interaction needs something that can first emit a
    ``TABLE_CELL`` locator (a table parser) or a ``FIGURE_CROP`` node (a
    figure digitizer); until then this class is schema + refusal apparatus,
    not a live parameter object.
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
        # AxisRole is a StrEnum, so a plain string equal to one of its
        # members' VALUES (e.g. role="coordinate") compares `==` equal to
        # AxisRole.COORDINATE but fails `isinstance`/`is` -- the same trap
        # this codebase already guards against for QuoteRole/QuantityKind
        # elsewhere (see ground_quote's own isinstance checks). A caller
        # that passes a bare string here would silently construct a
        # MeasurementSpec whose `.role` looks right under `==` but is not
        # actually an AxisRole member, so this is checked explicitly rather
        # than trusted from the type annotation alone.
        if not isinstance(self.role, AxisRole):
            raise DatasetProducerError(
                f"MeasurementSpec.role={self.role!r} must be a genuine AxisRole member, "
                f"not {type(self.role).__name__} -- AxisRole is a StrEnum, so a plain "
                "string equal to a member's value would compare `==` equal without "
                "actually being that member"
            )
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


class _ValueQuoteSpec(Protocol):
    """The quote fields :func:`_measured_value` actually reads off a spec.

    Deliberately NARROWER than :class:`MeasurementSpec`: a condition-set scalar
    claim carries no ``axis_id`` and no :class:`AxisRole`, because it is not a
    point on a series -- it is one stated condition. Typing the helper against
    what it READS rather than against one caller's concrete class is what lets
    both producers share it without either borrowing the other's vocabulary.
    The identifying string for refusal messages is passed separately as
    ``where``, so neither producer has to invent a field the other's domain
    has no word for.
    """

    # Declared as read-only properties, not bare attributes: every spec that
    # satisfies this Protocol is a FROZEN dataclass, and a mutable Protocol
    # member cannot be satisfied by a read-only attribute.
    @property
    def quantity_kind(self) -> QuantityKind: ...

    @property
    def value_quote(self) -> str: ...

    @property
    def unit_quote(self) -> str: ...

    @property
    def value_occurrence(self) -> int | None: ...

    @property
    def unit_occurrence(self) -> int | None: ...


def _measured_value(
    text: str,
    spec: _ValueQuoteSpec,
    *,
    where: str,
    document_source_context: SourceContext,
    document_glyph_health: GlyphHealth,
    value_locator: TableCellLocator | None = None,
    unit_locator: TableCellLocator | None = None,
) -> MeasuredValue:
    """Build one grounded :class:`MeasuredValue` from ``spec`` against ``text``.

    ``repairs`` and ``canonical_decimal_value`` are DERIVED from the value quote
    through the same ``normalize_numeric_span`` call the schema's own validator
    re-runs (see ``_CONTEXT_FREE_GLYPH_HEALTH``), never asserted independently --
    this is true wherever the value is LOCATED, because the number is normalized
    from its printed string, not from its position.

    ``value_locator``/``unit_locator`` let a caller supply an ALREADY-BUILT
    locator -- a :class:`TableCellLocator` for a datum that came from a table cell
    rather than running text. When omitted (the default, and the only case the
    dataset producer uses), each is grounded HERE by :func:`ground_quote` as a
    character span, byte-for-byte as before. A supplied locator is trusted: the
    condition-set producer validates a cell citation (the cell exists, its whole
    text equals the whole quote, no cell is two strings) BEFORE calling this, so
    the exact-equality contract is enforced once, centrally, not re-derived here.

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
    if value_locator is None:
        char_value_locator = ground_quote(
            text, spec.value_quote, role=QuoteRole.VALUE, occurrence=spec.value_occurrence
        )
        resolved_value_locator: TableCellLocator | CharSpanLocator = char_value_locator
        # P1: pass the VALUE locator's own span so UNIT's leading-edge digit-glue
        # exception (see carmel.services.numeric.unit_boundary_violation) can
        # require the glued digit run to be THIS measurement's own value, not
        # merely some other clean numeral (e.g. a run id) that happens to sit
        # before the unit quote.
        value_span: tuple[int, int] | None = (char_value_locator.start, char_value_locator.end)
    else:
        # A cell-located value has no character offsets, so UNIT's digit-glue
        # exception cannot fire -- value_span=None makes it fail closed, which is
        # exactly what its docstring says the omitted case does.
        resolved_value_locator = value_locator
        value_span = None
    if unit_locator is None:
        resolved_unit_locator: TableCellLocator | CharSpanLocator = ground_quote(
            text,
            spec.unit_quote,
            role=QuoteRole.UNIT,
            occurrence=spec.unit_occurrence,
            value_span=value_span,
            quantity=spec.quantity_kind,
        )
    else:
        resolved_unit_locator = unit_locator
    canary = normalize_numeric_span(
        spec.value_quote,
        source_context=document_source_context,
        glyph_health=document_glyph_health,
    )
    if isinstance(canary, Unresolvable):
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for {where} is refused under the "
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
            f"value quote {spec.value_quote!r} for {where} is not derivable into a numeral: {normalized.reason}"
        )
    try:
        canonical = canonical_decimal(normalized.text)
    except CanonicalDecimalError as exc:
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for {where} repaired to "
            f"{normalized.text!r}, which is not a valid canonical decimal string: {exc}"
        ) from exc
    try:
        unit_normalized = units.normalize_unit(spec.quantity_kind, spec.unit_quote, table=_ACTIVE.table)
    except units.UnknownUnitError as exc:
        raise DatasetProducerError(
            f"unit quote {spec.unit_quote!r} for {where} is not a known unit or alias "
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
        conversion_table_sha256=_ACTIVE.embedded.sha256,
        value_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=resolved_value_locator),
        unit_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=resolved_unit_locator),
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


def _authenticate_raw_bytes_and_read_source_metadata(
    workspace_root: Path, sha256: str
) -> tuple[str, RootSidecarVerification]:
    """Authenticate ``raw.bin`` and read the source metadata this producer needs.

    Returns ``(content_type, root_sidecar_claim)``: the artifact's declared
    content type, and the honest :class:`RootSidecarVerification` claim to
    record on the produced node.

    This REPLACED ``_load_verified_extracted_text``, which loaded, verified and
    parsed the ROOT ``extracted.json`` sidecar. That helper's three legacy
    carve-outs -- refusing an artifact whose ``meta.extracted_sha256`` or
    ``derivation_binding`` is ``None`` -- were preconditions on evidence this
    producer no longer uses for anything. Since the producer began grounding
    against a genuinely stored extraction record (see
    :func:`produce_envelope_from_artifact`), the root sidecar has not been an
    input to production at all: the grounded text, the ``ExtractionBinding``,
    and every digest carried into the envelope come from the record. What the
    root refusals actually did was block dataset production entirely for every
    artifact stored before those fields existed -- permanently, since
    ``reextract`` writes only under ``extractions/`` and NEVER rewrites a root
    sidecar. That is the whole real corpus: all 8 papers predate both fields.

    So the checks below are exactly the ones whose results this producer can
    honestly assert, and no others:

    1. ``load_artifact_meta`` resolves the store directory from the ``sha256``
       PARAMETER, and does not cross-check it against the ``sha256`` FIELD
       inside the loaded ``meta.json``. That field is cross-checked here, as a
       METADATA sanity check rather than as verification: a ``meta.json``
       disagreeing with the directory it lives in is not a trustworthy source
       of ``content_type`` either, whatever else may be wrong with it.
    2. ``raw.bin`` is re-read and re-hashed against ``sha256``. This is a
       RAW-ONLY check, deliberately not
       :func:`~carmel.services.evidence.verify_artifact` even at
       ``deep=False``: that function ALSO requires ``meta.json`` to load and
       the root sidecar to match its recorded digest when one exists, so its
       ``bool`` result would fold a root-tier fact into a claim this node
       records as raw-tier. (For the legacy corpus specifically, ``deep=False``
       happens to reduce to almost this same check, because
       ``_extracted_sidecar_intact`` returns True unconditionally when
       ``extracted_sha256`` is ``None`` -- but relying on that coincidence would
       make the recorded claim accidentally true rather than true by
       construction, and it would silently become false for a non-legacy
       artifact.)

    ``meta.json`` is read here ONLY for ``content_type``, which is source
    metadata rather than evidence and is not treated as such. The separate
    question of what can honestly be SAID about the root sidecar is decided by
    :func:`_root_sidecar_claim`, which looks -- so that the recorded claim is
    one a reader can refute rather than an assertion about what this producer
    chose to do.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Raw-bytes sha256 of the stored artifact.

    Returns:
        The artifact's ``content_type``, and the
        :class:`RootSidecarVerification` claim -- see
        :func:`_root_sidecar_claim`, which decides it by looking.

    Raises:
        DatasetProducerError: The artifact is absent, its ``meta.json``
            disagrees with the directory it lives in, or ``raw.bin`` is
            missing/unreadable or no longer hashes to ``sha256``.
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
    raw_path = artifact_dir(workspace_root, sha256) / _RAW_NAME
    try:
        raw_artifact_bytes = raw_path.read_bytes()
    except OSError as exc:  # FileNotFoundError is an OSError subclass
        raise DatasetProducerError(f"artifact {sha256!r} has no readable {_RAW_NAME}: {exc}") from exc
    actual_raw_sha256 = hashlib.sha256(raw_artifact_bytes).hexdigest()
    if actual_raw_sha256 != sha256:
        raise DatasetProducerError(
            f"artifact {sha256!r}: {_RAW_NAME} bytes on disk hash to {actual_raw_sha256!r}, not the "
            f"sha256 they are stored under; the evidence store's raw bytes have been tampered with "
            "or corrupted, and no envelope may name them"
        )
    root_sidecar_claim = _root_sidecar_claim(workspace_root, sha256, meta)
    return meta.content_type, root_sidecar_claim


def _root_sidecar_claim(workspace_root: Path, sha256: str, meta: StoredArtifact) -> RootSidecarVerification:
    """Decide, by looking, what can honestly be said about the root sidecar.

    Every value this returns is one a reader can put to the test against the
    store, and that is the entire reason the check runs. The sidecar is NOT an
    input to production, and an earlier revision therefore just recorded
    ``NOT_CHECKED`` for the non-legacy case on the grounds that nothing had read
    it. But ``NOT_CHECKED`` describes a producer CHOICE, not a fact about the
    store, so no consumer could ever contradict it -- and an unfalsifiable claim
    in persisted evidence is indistinguishable from no claim at all while still
    reading like provenance. One hash buys a value replay can refute.

    A mismatch is RECORDED, never raised. Refusing here would re-erect exactly
    the gate this design removed, blocking a dataset whose raw bytes and grounded
    text are both authenticated, over a tier that fed neither.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Raw-bytes sha256 of the artifact (and its store directory).
        meta: The artifact's already-loaded root metadata.

    Returns:
        ``NO_RECORDED_DIGEST`` when the artifact predates ``extracted_sha256``;
        otherwise ``ROOT_SIDECAR_DIGEST_AUTHENTICATED`` or
        ``ROOT_SIDECAR_DIGEST_MISMATCH`` according to what the bytes hash to. An
        unreadable sidecar counts as a mismatch: a recorded digest with nothing
        on disk to match it is a damaged legacy tier, which is what that value
        reports.
    """
    if meta.extracted_sha256 is None:
        return RootSidecarVerification.NO_RECORDED_DIGEST
    sidecar = artifact_dir(workspace_root, sha256) / _ROOT_EXTRACTED_NAME
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError:
        return RootSidecarVerification.ROOT_SIDECAR_DIGEST_MISMATCH
    if hashlib.sha256(sidecar_bytes).hexdigest() == meta.extracted_sha256:
        return RootSidecarVerification.ROOT_SIDECAR_DIGEST_AUTHENTICATED
    return RootSidecarVerification.ROOT_SIDECAR_DIGEST_MISMATCH


@dataclass(frozen=True, slots=True)
class _GroundingContext:
    """Everything a producer needs from ONE authenticated stored artifact.

    Built ONLY by :func:`_prepare_grounding`. This is the shared, fail-closed
    preamble every envelope producer must run before it may grond a single
    quote -- raw-bytes authentication, current-extraction selection, the
    lossy-extraction refusal, and the honest content_type -> node kind
    derivation. It is factored out rather than copied because a SECOND copy of
    a security preamble is a second thing that can drift: a fix applied to one
    producer's authentication path and not the other's is exactly the failure
    this codebase's content-addressed store exists to make impossible.

    Attributes:
        text: The grounded text of the selected CURRENT extraction record.
            Every char offset any producer emits indexes into THIS string.
        graph: The one-node :class:`SourceGraph` whose root honestly states
            what kind of document the artifact is and what was authenticated.
        document_source_context: P1-D canary -- the artifact's REAL source
            context, derived the way ``carmel.services.grounding`` derives it.
        document_glyph_health: P1-D canary -- the artifact's REAL glyph health.
    """

    text: str
    graph: SourceGraph
    document_source_context: SourceContext
    document_glyph_health: GlyphHealth


def _prepare_grounding(
    workspace_root: Path,
    sha256: str,
    *,
    envelope_noun: str,
    envelope_subject: str,
) -> _GroundingContext:
    """Authenticate ``sha256`` and build the grounding context for a producer.

    ``envelope_noun``/``envelope_subject`` appear ONLY in refusal messages, so a
    condition-set refusal does not misname itself a dataset refusal. Both are
    REQUIRED. They used to default to the dataset path's wording, which was
    right while dataset production existed to be byte-identical to the inline
    block this was extracted from -- but the only caller that took those
    defaults was ``produce_envelope_from_artifact``, which now refuses
    unconditionally (P0-c). The condition-set producer passed
    ``envelope_noun`` and NOT ``envelope_subject``, so its no-current-record
    refusal read "refusing to produce a condition set. A dataset must be
    grounded in ..." -- a refusal naming the wrong artifact, unnoticed because
    no test asserted that message on the live path. A default no live caller
    wants is not a convenience; it is a wrong answer waiting for the caller
    that forgets to override it.

    Raises:
        DatasetProducerError: The artifact is missing/legacy/corrupt, has no
            usable current extraction record, was extracted lossily, or has a
            ``content_type`` that maps to no ``SourceNodeKind`` this producer
            may honestly assert.
    """
    content_type, root_sidecar_claim = _authenticate_raw_bytes_and_read_source_metadata(workspace_root, sha256)
    # The text this envelope grounds against must come from a genuinely stored
    # extraction record, never from the root sidecar.
    #
    # This function used to MIRROR the root's already-stored extracted.json into a new
    # record stamped with today's extraction_identity(), purely so the binding below had
    # a resolvable address to name. No re-extraction happened: raw.bin was never
    # re-parsed. While nothing consumed extraction records that was inert. It stopped
    # being inert the moment the corpus gate began PREFERRING an authenticated current
    # record over the root -- because the mirrored record is, by construction, exactly
    # such a record. Running dataset production on a legacy artifact would then have
    # laundered unauthenticated root text into text the corpus reads as
    # EXTRACTION_RECORD_DIGEST_AUTHENTICATED, silently bypassing the legacy-root opt-in
    # that exists precisely to stop that text being read without the operator saying so.
    #
    # A record must therefore be something this producer FINDS, never something it
    # mints. If none exists, the honest answer is that no envelope can be produced until
    # the artifact is genuinely re-extracted (`Carmel.py reextract`).
    selection = select_current_extraction(workspace_root, sha256)
    if selection.kind is not CurrentSelectionKind.SELECTED or selection.selected is None:
        raise DatasetProducerError(
            f"artifact {sha256!r} has no usable current extraction record ({selection.detail}); "
            f"refusing to produce a {envelope_noun}. {envelope_subject} must be grounded in a genuinely "
            "stored extraction, and this producer will not mint a record from the root sidecar's "
            "text to satisfy its own binding -- that would launder unauthenticated text into an "
            "address the corpus gate treats as authenticated. Re-extract the artifact first"
        )
    extraction_sha256 = selection.selected.extraction_id
    extracted = selection.selected.extracted
    if extracted.lossy:
        # `ground_quote` below is a simple substring search, not the full
        # carmel.services.grounding gate -- it has no equivalent of that gate's
        # `unreadable_reason`/ARTIFACT_DEGRADED checks, so a lossy extraction
        # (missing pages, a parse failure, or truncation) could otherwise let a
        # measurement quietly ground against a partial document, or (worse) an
        # empty one that happens to substring-match nothing and fail for the
        # wrong reason. Refuse up front instead of letting a knowingly-partial
        # extraction feed a dataset envelope at all.
        page_note = f" ({len(extracted.page_failures)} page(s) failed to extract)" if extracted.page_failures else ""
        raise DatasetProducerError(
            f"artifact {sha256!r} was extracted lossily (extractor={extracted.extractor!r}){page_note}; "
            f"refusing to produce a {envelope_noun} from a knowingly-partial extraction"
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
    # The binding's identity fields are populated from what the store ACTUALLY holds --
    # the record is loaded back (which self-authenticates its meta.json against the
    # address selected above), never re-asserted from this function's local variables.
    # An ExtractionBinding recomputes its own extraction_sha256 from these fields at
    # construction, so building it from the authenticated record guarantees the binding
    # recomputes to the address on disk; inventing any value here would make the binding
    # schema-invalid, loudly.
    #
    # ExtractionBinding.parent_raw_sha256/extraction_sha256 must name a genuinely
    # RESOLVABLE record, not merely a computed address that resolves to nothing -- a
    # replayer handed an address with no backing record can never do better than report
    # it UNVERIFIABLE forever, which is worse than no replayer at all. That requirement
    # is now met by SELECTING an existing record rather than by minting one, so the only
    # way this lookup can fail is a store that changed underneath the selection.
    record_meta = load_extraction_record(workspace_root, sha256, extraction_sha256)
    if record_meta is None:
        raise DatasetProducerError(
            f"artifact {sha256!r}: extraction record {extraction_sha256!r} authenticated during "
            "selection but no longer resolves; the extraction record store changed underneath this "
            "producer and no binding can honestly be produced"
        )
    binding = ExtractionBinding(
        parent_raw_sha256=record_meta.parent_raw_sha256,
        extraction_sha256=record_meta.extraction_sha256,
        extracted_sha256=record_meta.extracted_sha256,
        extracted_text_sha256=record_meta.extracted_text_sha256,
        extractor=record_meta.extractor,
        extractor_code_sha256=record_meta.extractor_code_sha256,
        identity_payload_version=record_meta.identity_payload_version,
        pypdf_version=(
            record_meta.pypdf_version
            if record_meta.extractor in _PYPDF_DEPENDENT_EXTRACTORS
            # For every other extractor the record's pypdf_version is a
            # diagnostics-only field that is not part of the identity
            # address; the binding states the inapplicability explicitly
            # rather than carrying an identity claim the address does not
            # fold in.
            else Absent(reason=AbsenceReason.NOT_APPLICABLE)
        ),
    )
    root_node = SourceNode(
        node_id=_ROOT_NODE_ID,
        kind=node_kind,
        sha256=sha256,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        # This root is a whole document, never a region cut out of one, so the
        # only reason SourceNode's I7 accepts here is NOT_APPLICABLE.
        crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=binding,
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        # Each tier states what this function ACTUALLY established, and nothing
        # more. The first two are constants here because the alternative to
        # either one is a refusal several lines above, not a weaker envelope:
        # unauthenticated raw bytes and a non-current record both abort
        # production outright. Only the root-sidecar tier varies, and it is the
        # one a reader would otherwise get wrong -- see
        # `_authenticate_raw_bytes_and_read_source_metadata` for why the root
        # sidecar is not an input to production at all.
        verification=SourceVerification(
            raw_artifact=RawArtifactVerification.RAW_SHA256_DIGEST_AUTHENTICATED,
            extracted_text=ExtractedTextVerification.EXTRACTION_RECORD_DIGEST_AUTHENTICATED,
            root_sidecar=root_sidecar_claim,
        ),
    )
    graph = SourceGraph(nodes=(root_node,))
    return _GroundingContext(
        text=text,
        graph=graph,
        document_source_context=document_source_context,
        document_glyph_health=document_glyph_health,
    )


def _describe_count(measurements: object) -> str:
    """Render the spec count for the refusal message without trusting the
    caller to have passed something sized."""
    try:
        return f"{len(measurements)} spec(s)"  # type: ignore[arg-type]
    except TypeError:
        return "an unsized measurements argument"


def produce_envelope_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    series_id: str,
    value_origin: ValueOrigin,
    measurements: tuple[MeasurementSpec, ...],
) -> DatasetEnvelope:
    """REFUSED: this runtime cannot honestly produce a dataset envelope.

    P0-c. This function used to build a fully validated
    :class:`DatasetEnvelope` -- authenticate ``raw.bin``, select the one
    CURRENT extraction, ground every caller-stated quote in its text, and
    assemble one root node, one ``TEXTUAL`` series and one data point. It now
    refuses unconditionally, and the assembly code is deleted rather than
    parked behind the refusal.

    WHY IT REFUSES. A :class:`CharSpanLocator` into extracted running text is
    the only locator kind this runtime can produce (the round-33 ruling), so
    every series this function could emit is ``source_form=TEXTUAL`` -- and
    :meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`
    (V7) now rejects a char span as the source of a series VALUE, and a char
    span is the only thing this producer can emit. Note V7 refuses the
    LOCATOR, not ``source_form``: ``TEXTUAL`` itself remains legal for a
    series whose values are located some other way. A series asserts a structured pairing of coordinates to
    observations, and running text carries no row structure from which that
    pairing can be proven. This was a live fabrication, not a theoretical one:
    pypdf renders a figure's axis furniture into ``text.txt`` as ordinary body
    prose, so grounding a coordinate at the tick ``0.7`` and an observation at
    the tick ``24`` produced a schema-valid envelope that replayed
    ``VERIFIED`` with zero findings. Grounding proves LOCATION, never MEANING.

    WHAT THIS IS NOT. It is not a claim that prose never states a series --
    "At 300, 400 and 500 K the rates were 1.2, 2.4 and 4.8 s-1, respectively"
    is a real one. It is a statement about what THIS RUNTIME can prove. Nor is
    the refusal a furniture detector: it does not inspect the quotes and judge
    them tick-like, which would be a probabilistic guess about meaning inside
    the deterministic S1 lane. It closes the ROUTE.

    WHY THE BODY IS DELETED RATHER THAN KEPT. An unreachable envelope
    assembler reads as an available capability, which is the same trap as
    :class:`~carmel.agents.tools.extract.TextSection`'s documented-but-never-
    emitted ``caption``/``table`` labels. Git holds the previous
    implementation; a future ``TABULAR``/``DIGITIZED`` producer will not want
    its char-span assembly anyway.

    HOW TO RESTORE DATASET PRODUCTION. Something must first be able to emit a
    ``TABLE_CELL`` locator (a table parser) or a ``FIGURE_CROP`` node (a figure
    digitizer). Until then no producer can construct any
    :class:`DatasetEnvelope`, and the dataset slice is schema + replay +
    storage only. For a prose-local SCALAR statement -- the honest thing
    running text CAN support -- use
    :func:`~carmel.services.condition_set_producer.produce_condition_set_from_artifact`.
    Note that a :class:`ConditionSetEnvelope` holds CONDITIONS, not
    observables, so a genuinely prose-stated observable has no home today; a
    survey of the eight-paper corpus found no honest instance of one (every
    observable-shaped prose sentence was a figure caption, axis furniture, a
    method threshold, or an explicit narration of a figure).

    Args:
        workspace_root: Unused; kept so the refusal is reachable from every
            existing call site with a real diagnosis rather than a
            ``TypeError``.
        sha256: Unused, as above.
        series_id: Reported back in the message so a caller with several
            pending series can tell which one was refused.
        value_origin: Reported back, as above.
        measurements: Reported back by count, as above.

    Raises:
        DatasetProducerError: Always.
    """
    raise DatasetProducerError(
        "this producer can only locate values as CHAR_SPAN offsets into running text, which "
        "would make every series it emits source_form=TEXTUAL -- and a char span in running "
        "text cannot ground a series data point (DatasetEnvelope V7). A series asserts a "
        "coordinate/observation pairing that running text carries no structure to prove: a "
        "figure's axis ticks extract as ordinary body prose and ground perfectly (series_id="
        f"{series_id!r}, {len(measurements)} spec(s), value_origin={value_origin.value!r}). "
        "This is a limit of what this runtime can prove, not a claim that prose never states a "
        "series. Series data needs TABULAR (a table parser) or DIGITIZED (a figure digitizer), "
        "neither of which exists yet; a prose-local scalar statement belongs in a "
        "ConditionSetEnvelope via produce_condition_set_from_artifact"
    )
