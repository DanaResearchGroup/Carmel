# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Numeric reconstruction core: turn scoped cell/span text into a trustworthy number.

CRITICAL SCOPING RULE: this module is NOT a "scan the paper for numbers" tool. Every
function here parses a caller-supplied cell or span whose boundaries were already
established by some other component (a table extractor, a JATS cell reader, an
operator's manual transcription, ...). Nothing here searches free-running document
text for numeric substrings; :func:`parse_numeric_span` takes exactly the text some
other component decided is "the cell" and either reconstructs one trustworthy number
(or range) from it, or refuses -- it never widens its own search window.

Pure and I/O-free by design: no network, no LLM calls, no file reads, no third-party
dependencies beyond the standard library. Every function is deterministic given its
arguments.

Why this exists (measured on 8 real papers, corruption shapes reproduced here only as
short synthetic strings for copyright reasons):

- 3 of 8 papers (all Elsevier, Int J Hydrogen Energy) encode an en dash (U+2013) as a
  bare ASCII ``e`` and contain ZERO real en dashes; the other 5 papers have 47-78
  intact en dashes each. A bare ``\\d+e\\d+`` token in a suspect document is therefore
  NOT trustworthy scientific notation -- it may be a corrupted "A-B" range -- and must
  fail closed rather than silently parse. See :class:`GlyphHealth` and the quarantine
  rule in :func:`parse_numeric_span`.
- Some corrupt documents also substitute ``þ`` (U+00FE) for a plus sign and the
  literal sequence ``/C0`` for a minus sign, both ONLY in numeric sign/exponent
  position -- never decoded as a blanket document-wide text rewrite, always as a
  span-local, recorded repair (see the ``repairs`` field on :class:`Scalar` /
  :class:`Range`).
- A numeric-looking token touching letters, ``%``, or other formula fragments (no
  whitespace between them) is a formula fragment, not a value, and is refused.
- Anything that parses to a non-finite float (``inf``, ``-inf``, ``nan``) is never a
  trustworthy :class:`Scalar` -- it is always :class:`Unresolvable`.

Deliberately out of scope for this module (do not extend it to cover these here):
sup/sub-aware reconstruction (e.g. ``3.94 x 10 03`` -> ``3.94e3``), decoding ``¼`` as
``=``, and any blanket/global text rewriting of a whole document.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "NUMERAL_CANDIDATE_RE",
    "NUMERAL_EXTENT_RE",
    "REPAIR_NAMES",
    "GlyphHealth",
    "NormalizedNumeral",
    "NumericResult",
    "QuoteRole",
    "Range",
    "Scalar",
    "SourceContext",
    "Unresolvable",
    "assess_glyph_health",
    "enclosing_numeric_construct",
    "find_numeral_extent",
    "has_clean_token_boundary",
    "label_boundary_violation",
    "normalize_numeric_span",
    "parse_numeric_span",
    "unit_boundary_violation",
]


class SourceContext(StrEnum):
    """Where a cell/span originated. Only :attr:`FLAT_PDF_TEXT` is ever subject to the
    dash-corruption quarantine rule (see :func:`parse_numeric_span`) -- a structured
    cell (:attr:`JATS_CELL`, :attr:`SPREADSHEET_CELL`) never inherits a PDF document's
    quarantine state; it is only ever quarantined if its OWN text independently
    carries the corruption markers, which by construction it cannot signal through
    this enum member alone."""

    FLAT_PDF_TEXT = "flat_pdf_text"
    """Text pulled from a flattened PDF text layer -- the only source known to exhibit
    the en-dash-as-ASCII-``e`` substitution."""

    JATS_CELL = "jats_cell"
    """A cell read from structured JATS/XML markup."""

    SPREADSHEET_CELL = "spreadsheet_cell"
    """A cell read from a spreadsheet (e.g. a supplementary-information workbook)."""

    OPERATOR_RAW = "operator_raw"
    """Text an operator typed or pasted by hand."""


class QuoteRole(StrEnum):
    """What semantic job a quote passed to
    :func:`carmel.services.dataset_producer.ground_quote` is playing in its
    source sentence -- NOT a property of the quote's own characters, but of
    what the caller is claiming it grounds. One boundary rule cannot serve
    all three jobs at once: a unit glued directly after its value
    (``"1023K"`` + quote ``"K"``) MUST stay groundable, while a species/label
    token glued to a following digit (``"CO2"`` + quote ``"CO"``) MUST be
    refused -- and those two quotes look identical (a bare letter run
    immediately before a digit or vice versa) without knowing which job the
    caller means. See :func:`unit_boundary_violation` and
    :func:`label_boundary_violation` for the per-role adjacency rules; the
    :attr:`VALUE` role keeps the pre-existing numeral-grammar path in
    :func:`carmel.services.dataset_producer.ground_quote` unchanged.
    """

    VALUE = "value"
    """A measured/reported numeral (or a multi-token numeric construct like a
    range). Grounded via the numeral grammar (:data:`NUMERAL_CANDIDATE_RE`,
    :func:`find_numeral_extent`, :func:`enclosing_numeric_construct`) when the
    quote fullmatches a numeral candidate, else via the same generic
    token-boundary fallback this role has always used."""

    UNIT = "unit"
    """A unit quote (``"K"``, ``"cm3/mol/s"``, ``"bar"``). Must not abut
    another unit-token character on either edge, EXCEPT that its leading edge
    may abut a preceding digit run if that digit run is itself a clean,
    maximal numeral -- the value the unit is glued to."""

    LABEL = "label"
    """An axis/species/quantity label quote (``"pressure"``, ``"ignition
    delay time"``). The strictest role: must not abut a letter or a digit on
    either edge, with no exception."""


#: Signal for :class:`GlyphHealth`: a bare lowercase ``e`` standing between two digit
#: runs with no decimal point and no explicit sign, e.g. ``2e50`` or ``1000e3000``.
#: Genuine scientific notation in the corrupt corpus always uses uppercase ``E``, so
#: this pattern is deliberately lowercase-only.
_BARE_DASH_CORRUPTION_RE = re.compile(r"\d+e\d+")

#: The ASCII-``6``-for-± shape: ``307 6 10`` meaning ``307 ± 10``. Tight and
#: specific on purpose -- ASCII ``6`` is far too common a digit to repair as a
#: document-wide rule.
_ASCII6_UNCERTAINTY_RE = re.compile(r"\d+\s+6\s+\d+")

#: Direct textual literals that Python's bare ``float()`` would accept but that are
#: never legitimate values for a scoped numeric span.
_DISALLOWED_LITERALS = frozenset(
    {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
)

#: The small hand-rolled grammar for one signed numeric value (mantissa, optional
#: exponent), including the two contextual glyph repairs. Deliberately NOT built as a
#: bag of independent regex alternatives layered by precedence -- range/scalar/sign
#: classification is decided positionally by :func:`_find_range_separator` before this
#: pattern is ever tried, so this pattern only ever has to describe one signed value.
#:
#: - ``lead_sign``: ``/C0`` (+ optional trailing whitespace) repairs to a leading
#:   minus; ASCII ``-``/``+`` are taken as literal signs; U+2212 MINUS SIGN and a
#:   leading U+2013 EN DASH also repair to a leading minus (NFKC does not decompose
#:   either into an ASCII hyphen, so without this they would otherwise never be
#:   recognized as a sign at all). A leading en dash is unambiguous as a sign here:
#:   :func:`_find_range_separator` only ever treats a dash at position ``i > 0`` as a
#:   range separator, so this pattern is only ever handed a span whose leading
#:   character (if any) is genuinely a sign, not a separator.
#: - ``exp_sign``: ``þ`` (+ optional trailing whitespace) repairs to an exponent
#:   plus; ASCII ``-``/``+`` are literal exponent signs.
#: - ``exponent`` intentionally accepts an illegal ``\d+\.\d+`` shape too, purely so
#:   :func:`_parse_single_value` can DETECT and reject it explicitly (Case A: illegal
#:   float literals like ``0.6e1.0`` must never be salvaged into anything).
_CORE_VALUE_RE = re.compile(
    r"(?P<lead_sign>/C0\s*|[-+−]|–)?"
    r"(?P<mantissa>\d+(?:\.\d+)?)"
    r"(?:(?P<emarker>[eE])(?P<exp_sign>þ\s*|[-+])?(?P<exponent>\d+(?:\.\d+)?))?"
)

#: The complete, closed set of repair names this module can ever emit (see the
#: four ``repairs.append(...)`` call sites in :func:`_normalize_single_value`
#: below). Exported so a downstream schema can validate a *recorded* repair
#: list against this module's actual vocabulary instead of accepting free
#: text -- a repair name that is not a member of this set could never have
#: been produced by this module, so a schema can treat that as a hard error
#: rather than trusting whatever string an upstream caller wrote down.
REPAIR_NAMES: frozenset[str] = frozenset(
    {
        "slash_c0_to_minus",
        "unicode_minus_to_ascii",
        "leading_en_dash_to_minus",
        "thorn_to_plus",
    }
)


#: THE single shared GRAMMAR BODY, codebase-wide, for where a numeral "candidate"
#: begins in free-running text -- factored out so two independent trailing-boundary
#: choices (see :data:`NUMERAL_CANDIDATE_RE` and :data:`NUMERAL_EXTENT_RE` below)
#: cannot drift apart on the parts that must stay identical: the leading boundary
#: and the numeral body itself (sign, mantissa, exponent, hyphenated-range
#: alternative). This exists because two modules -- :mod:`carmel.services.grounding`
#: (scanning an evidence window for numbers that corroborate a claim) and
#: :mod:`carmel.services.dataset_producer` (checking that a quoted numeral is the
#: MAXIMAL numeral at its position, not a fragment of a bigger one) -- independently
#: grew their own, weaker, boundary regexes to answer two DIFFERENT questions, and
#: both were demonstrably wrong in different ways (the producer's version, for
#: instance, had no sign alternative at all, so a quote like ``"-3"`` skipped its own
#: maximality check entirely).
#:
#: This body is deliberately WIDER than this module's own strict core
#: (:data:`_CORE_VALUE_RE` / :func:`parse_numeric_span`): the candidate grammar's job
#: is to decide EXTENT -- where a numeral-shaped run of characters starts and stops in
#: surrounding text, so that a fragment of it can never be mistaken for the whole
#: thing -- while the strict core's job is to decide VALUE -- whether that text, once
#: isolated, reconstructs to one trustworthy float (or explicitly refuses to). A span
#: either candidate regex matches is not automatically a valid :class:`Scalar` or
#: :class:`Range`; it still has to survive :func:`parse_numeric_span`. Conversely, a
#: span :func:`parse_numeric_span` accepts is always contained in (or equal to) the
#: candidate span covering its position, by construction: the candidate grammar is a
#: strict superset (it additionally tolerates, e.g., comma-grouped thousands and the
#: ``/c0``/``/C0`` sign repair token, both of which the strict core validates and
#: refuses on its own terms once isolated).
#:
#: Leading boundary (SHARED, identical in both regexes below): a candidate must not
#: be directly PRECEDED by a letter, digit, or dot -- that is what stops a match from
#: starting mid-identifier (``h2o``'s ``2``) or mid-numeral. This is shared because an
#: identifier prefix invalidates a numeral for BOTH questions this module answers --
#: "is this a clean corroborating VALUE" and "where does this numeral's extent END" --
#: alike: ``"2"`` inside ``"H2"`` must stay refused no matter which trailing question
#: is being asked.
#:
#: An optional leading sign (ASCII ``-``/``+``, Unicode minus, a leading en dash
#: before a digit, or the ``/c0``/``/C0`` repair token) is part of the candidate; an
#: optional single ``-``/en-dash-separated second value makes a hyphenated range ONE
#: candidate, never two, so a range's individual endpoint (e.g. ``"1200"`` inside
#: ``"1000-1200"``) is correctly seen as a fragment of the range, not a standalone
#: value; an optional exponent suffix (``e``/``E`` then optional sign then digits)
#: extends the candidate so ``"1023"`` inside ``"1023e5"`` is a fragment of the
#: exponent form, not the whole numeral.
#:
#: Case sensitivity: this body is written with EXPLICIT ``[0-9a-zA-Z...]`` character
#: classes and an explicit ``[eE]``/``/[cC]0`` alternative, NOT ``re.IGNORECASE``, and
#: that is a deliberate choice, not an oversight. :mod:`carmel.services.grounding`
#: runs its regex against text it has ALREADY casefolded (so only the lowercase
#: branches of these classes can ever match there -- adding the uppercase branches is
#: a provable no-op for that caller). But :mod:`carmel.services.dataset_producer` runs
#: its regex against RAW, un-casefolded ``ExtractedText.text``, where an uppercase
#: ``E1023`` exponent marker or a real ``/C0`` sign-repair token must be recognized as
#: part of the numeral. A blanket ``re.IGNORECASE`` flag would also silently case-fold
#: every OTHER character class in this pattern -- in particular it would make the
#: boundary lookaround classes match uppercase letters too, which they already do via
#: the explicit ``A-Z`` here, so that part is harmless -- but relying on the flag
#: instead of writing the classes out would make the pattern's case behavior implicit
#: and easy to get wrong the next time it is edited. Writing every class out
#: explicitly keeps the case-sensitivity of each piece an intentional, visible choice.
_NUMERAL_LEADING_BOUNDARY = r"(?<![0-9a-zA-Z.,])"
#: One signed value: optional leading sign (ASCII/Unicode minus, leading en dash,
#: ``/c0``/``/C0`` repair token) followed by a mantissa and an optional bare
#: ``[eE]``-marked exponent. This is the piece :data:`_NUMERAL_BODY` repeats (once
#: signed, once unsigned) to build the hyphenated-range alternative below, and it is
#: also the shared "NUM" building block :func:`enclosing_numeric_construct` reuses so a
#: fourth, independently-drifting numeral grammar never appears in this module.
_NUMERAL_SINGLE_VALUE = r"(?:/[cC]0\s*|[-+−]|–(?=\d))?\d+(?:\.\d+)?(?:[eE][-+]?\d+(?:\.\d+)?)?"
#: The UNSIGNED tail used for a range's high bound (no lead-sign alternatives: the
#: dash immediately before it already IS the range separator).
_NUMERAL_TAIL_VALUE = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+(?:\.\d+)?)?"
_NUMERAL_BODY = _NUMERAL_SINGLE_VALUE + r"(?:[-–]" + _NUMERAL_TAIL_VALUE + r")?"

#: TRAILING boundary is intentionally NOT shared -- the two callers ask different
#: questions of it, and collapsing them back into one regex is exactly the mistake
#: this split undoes (see commit history: an earlier unification narrowed the
#: trailing boundary to ``(?![0-9,])`` everywhere so that ``"1023"`` would ground
#: inside ``"1023K"``, but that also let :mod:`carmel.services.grounding`'s window
#: scanner start emitting spurious candidates for run/table labels glued to digits,
#: e.g. over ``"run 3a and 4b gave 720k"`` it would find ``['3', '4', '720']`` where
#: the strict boundary correctly finds nothing, and over ``"the temperature was
#: 1023k"`` it would find ``['1023']`` where the strict boundary correctly finds
#: nothing -- a numeric anchor could then look "corroborated" by digits that are
#: really part of a run label or a glued token). DO NOT re-collapse these into one
#: regex.
#:
#: - :data:`NUMERAL_CANDIDATE_RE` (used by :mod:`carmel.services.grounding`'s
#:   window/value scan, :func:`_window_numeric_values`): forbids a following digit,
#:   comma, OR letter. This is the strict "is there a clean corroborating VALUE
#:   here" question -- ``"720k"`` is not a clean value, so this must refuse to see a
#:   candidate there at all.
#: - :data:`NUMERAL_EXTENT_RE` (used by :func:`find_numeral_extent`, which backs
#:   :mod:`carmel.services.dataset_producer`'s ``ground_quote``): forbids only a
#:   following digit or comma, tolerating a following letter. This is the "where does
#:   this numeral END" extent question -- a numeral is routinely glued to a trailing
#:   unit letter in real text (``"1023K"``, ``"5s"``), and a trailing ``K`` is not
#:   part of the numeral's own extent, so ``"1023"`` must still be seen as the FULL
#:   numeral at that position (not a fragment) when a caller is checking maximality.
NUMERAL_CANDIDATE_RE = re.compile(_NUMERAL_LEADING_BOUNDARY + _NUMERAL_BODY + r"(?![0-9a-zA-Z,])")

NUMERAL_EXTENT_RE = re.compile(_NUMERAL_LEADING_BOUNDARY + _NUMERAL_BODY + r"(?![0-9,])")


def find_numeral_extent(text: str, index: int) -> tuple[int, int] | None:
    """Return the ``(start, end)`` span of the numeral candidate covering character
    ``index`` in ``text``, or ``None`` if no candidate covers that position.

    "Covering" means ``start <= index < end``: the character at ``index`` sits
    somewhere inside the matched span, not merely adjacent to it. A caller holding a
    known substring position (e.g. a quote's search-found offset) uses this to answer
    "what is the FULL numeral touching this position?" -- if the returned span is wider
    than the caller's own substring, the caller's substring is a fragment, not the
    whole numeral.

    Implemented by scanning :data:`NUMERAL_EXTENT_RE` left to right via
    ``finditer`` and returning the first (and by construction, since candidates never
    overlap, only) match whose span contains ``index``.
    """
    for match in NUMERAL_EXTENT_RE.finditer(text):
        if match.start() <= index < match.end():
            return match.span()
        if match.start() > index:
            break
    return None


def has_clean_token_boundary(text: str, start: int, end: int) -> bool:
    """Return whether ``text[start:end]`` sits at a clean word/token boundary in
    ``text`` -- i.e. it is not an interior fragment of a larger alphanumeric token.

    This is the NON-numeric counterpart to :func:`find_numeral_extent` /
    :data:`NUMERAL_EXTENT_RE`: those exist to stop a numeral quote from grounding
    to a fragment of a bigger numeral, but a non-numeric quote (a unit like
    ``"K"`` or a label like ``"pressure"``) never matches that grammar at all, so
    it needs its own boundary rule. Only known caller today:
    :func:`carmel.services.dataset_producer.ground_quote`, applied to a quote
    that does NOT fullmatch :data:`NUMERAL_CANDIDATE_RE`.

    The check is PER-EDGE and classified by the character AT that edge of
    ``text[start:end]`` -- deliberately NOT a single uniform "alphanumeric"
    rule, because the two edge classes need different answers:

    - Leading edge (``text[start]``):
      - LETTER: the character immediately before ``start`` (if any) must not
        also be a letter -- otherwise the quote starts mid-word (``"K"`` inside
        ``"Kinetics"``).
      - DIGIT: the character immediately before ``start`` must not be a digit,
        ``.``, or ``,`` -- otherwise the quote starts mid-numeral or mid-thousands
        group.
      - Anything else (punctuation, whitespace, symbol): no constraint at this
        edge.
    - Trailing edge (``text[end - 1]``), symmetric:
      - LETTER: the character immediately after ``end`` (if any) must not also
        be a letter.
      - DIGIT: the character immediately after ``end`` must not be a digit or
        ``,``.
      - Anything else: no constraint at this edge.

    The letter/digit asymmetry is deliberate and load-bearing -- do NOT
    "simplify" it to one uniform alphanumeric rule. It mirrors
    :data:`NUMERAL_EXTENT_RE`'s own asymmetric trailing boundary
    (``(?![0-9,])``, which permits a letter right after a numeral so that
    ``"1023"`` stays groundable inside ``"1023K"``): the symmetric consequence
    is that the unit ``"K"`` glued to that same numeral must ALSO stay
    groundable, so a DIGIT immediately before a LETTER-initial quote must NOT
    trigger a refusal here, even though a LETTER immediately before a
    LETTER-initial quote (or a DIGIT/``.``/``,`` immediately before a
    DIGIT-initial quote) does.

    Note this function does not, by itself, resolve which of several matches
    of a quote in ``text`` to check -- a caller with multiple matches must
    decide (or refuse) ambiguity BEFORE calling this, exactly as
    :func:`carmel.services.dataset_producer.ground_quote` already does for its
    numeral-boundary guard: this function only ever answers the question for
    ONE already-chosen ``(start, end)`` span, never reduces a multi-match set
    to one on its own.
    """
    if start >= end:
        return True
    lead = text[start]
    if lead.isalpha() and start > 0 and text[start - 1].isalpha():
        return False
    if lead.isdigit() and start > 0 and (text[start - 1].isdigit() or text[start - 1] in ".,"):
        return False
    trail = text[end - 1]
    if trail.isalpha() and end < len(text) and text[end].isalpha():
        return False
    return not (trail.isdigit() and end < len(text) and (text[end].isdigit() or text[end] == ","))


#: Characters permitted immediately BEFORE a UNIT-role quote (its leading edge),
#: besides whitespace and start-of-string. Deliberately an ALLOWLIST of permitted
#: separators, not a denylist of forbidden unit-token characters -- a denylist
#: rots (round 40 found the old ``_UNIT_TOKEN_SYMBOLS`` denylist already missing
#: U+2212 MINUS SIGN, U+2013 EN DASH, and superscript minus/one, none of which
#: were "unit-token characters" under the old alnum+symbol test, yet all glue a
#: unit quote to something that is not a clean boundary; it also silently
#: accepted ``"bar"`` inside ``"bar(a)"``). ``(`` is leading-only -- a unit that
#: opens a parenthetical, e.g. the ``K`` in ``"T (K) 1023"``, must accept ``(``
#: immediately before it -- deliberately asymmetric with the trailing allowlist
#: below, which does not include ``(``.
_UNIT_LEADING_ALLOWLIST = frozenset("(=:,")

#: Characters permitted immediately AFTER a UNIT-role quote (its trailing edge),
#: besides whitespace and end-of-string. ``)`` closes a parenthetical the unit
#: sat inside (the same ``"T (K) 1023"`` shape above) and is trailing-only,
#: mirroring ``(`` being leading-only above.
_UNIT_TRAILING_ALLOWLIST = frozenset(",;.)")


def unit_boundary_violation(
    text: str, start: int, end: int, *, value_span: tuple[int, int] | None = None
) -> str | None:
    """Return a discriminant name for why ``text[start:end]`` is not a clean
    UNIT-role quote, or ``None`` if it is clean.

    Mirrors :func:`enclosing_numeric_construct`'s idiom of returning a name
    string (or ``None``) for the caller to map to a message, rather than a
    bare bool, so :func:`carmel.services.dataset_producer.ground_quote` can
    give each refusal reason its own distinct message.

    Fail-closed ALLOWLIST, not a denylist: a unit quote (``"K"``,
    ``"cm3/mol/s"``, ``"bar"``) is refused on an edge unless the character
    immediately outside that edge is on the permitted-neighbour allowlist for
    that edge (:data:`_UNIT_LEADING_ALLOWLIST` / :data:`_UNIT_TRAILING_ALLOWLIST`)
    or whitespace/start-of-string/end-of-string. Anything else -- a letter, a
    digit, or a symbol not on the allowlist -- means the quote is a fragment of
    a larger unit (``"cm3"`` inside ``"cm3/mol/s"``, or ``"C"`` inside
    ``"25°C"``).

    ONE exception, leading edge only, gated by ``value_span``: the unit quote
    MAY abut a preceding digit run if -- and only if -- ``value_span`` is
    supplied AND its end is exactly ``start``, i.e. the caller has already
    independently grounded a VALUE quote and is asserting that THIS digit run
    IS that value, not merely some clean numeral. This is deliberately
    stricter than "is there a clean numeral here": round 40 found that
    checking cleanliness alone (via :func:`find_numeral_extent`) proves "SOME
    clean numeral ends here", not "THIS measurement's value ends here" --
    value and unit are grounded independently, so e.g. the run id ``"1"`` in
    ``"case 1K was the run id. Temperature was 1023 K"`` is itself a clean,
    maximal numeral and would wrongly satisfy a cleanliness-only check. Without
    a ``value_span`` (the default), the exception never fires -- fail closed,
    not fail open -- which is why ``"1023K"`` + quote ``"K"`` with no
    ``value_span`` now refuses exactly like ``"run3K"`` + quote ``"K"`` does;
    only a caller that supplies the matching value span (as
    :func:`carmel.services.dataset_producer._measured_value` does, from the
    VALUE locator it already grounded) gets the glue exception.
    """
    if start >= end:
        return None
    if start > 0:
        lead_prev = text[start - 1]
        if lead_prev.isdigit():
            if value_span is None or value_span[1] != start:
                return "unit_digit_glue"
        elif not (lead_prev.isspace() or lead_prev in _UNIT_LEADING_ALLOWLIST):
            return "unit_leading_adjacency"
    if end < len(text):
        nxt = text[end]
        if not (nxt.isspace() or nxt in _UNIT_TRAILING_ALLOWLIST):
            return "unit_trailing_adjacency"
    return None


#: Characters permitted immediately before/after a LABEL-role quote, besides
#: whitespace and start/end-of-string. Deliberately an ALLOWLIST, mirroring the
#: UNIT allowlists above -- round 40 found the previous letter/digit-only
#: denylist under-refused punctuation-delimited fragments (``"T"`` inside
#: ``"1/T"``, ``"CO"`` inside ``"X_CO"``, ``"H2"`` inside ``"H2/CO"``, ``"S"``
#: inside ``"S_L"``), because ``/`` and ``_`` are neither letters nor digits.
#: Symmetric (unlike the UNIT allowlists): a label never opens or closes a
#: parenthetical the way a unit does, so both edges share one allowlist.
#: ``_`` and ``/`` are deliberately NOT here -- a label glued to either is a
#: fragment of a larger token, never a standalone label on its own.
_LABEL_ALLOWLIST = frozenset("()=:,;.")


def label_boundary_violation(text: str, start: int, end: int) -> str | None:
    """Return a discriminant name for why ``text[start:end]`` is not a clean
    LABEL-role quote, or ``None`` if it is clean.

    LABEL is the strictest role: a label/species/quantity-name quote
    (``"pressure"``, ``"CO"``, ``"H"``) must not abut anything except
    whitespace, start/end-of-string, or a character on :data:`_LABEL_ALLOWLIST`
    on either edge, with NO exception -- unlike :func:`unit_boundary_violation`
    there is no "glued value" shape a label is ever allowed to sit inside. This
    refuses ``"CO"`` inside ``"CO2 mole fraction"`` (trailing digit) and
    ``"NO"`` inside ``"the NO2 profile"`` alike, exactly as the previous
    letter/digit denylist did, but -- being an allowlist -- it ALSO refuses
    ``"T"`` inside ``"1/T"`` and ``"CO"`` inside ``"X_CO"``, which the old
    denylist missed because ``/`` and ``_`` are neither letters nor digits.

    Any character (including a subscript digit like ``'₂'``, or any other
    symbol) that is not whitespace and not on the allowlist triggers a
    refusal -- there is no per-character-class test here at all, unlike the
    old ``isalpha()``/``isdigit()`` denylist.
    """
    if start >= end:
        return None
    if start > 0:
        prev = text[start - 1]
        if not (prev.isspace() or prev in _LABEL_ALLOWLIST):
            return "label_leading_adjacency"
    if end < len(text):
        nxt = text[end]
        if not (nxt.isspace() or nxt in _LABEL_ALLOWLIST):
            return "label_trailing_adjacency"
    return None


#: ``NUM <ws> 6 <ws> NUM`` -- reused verbatim from :func:`assess_glyph_health`'s
#: document-wide fingerprint (:data:`_ASCII6_UNCERTAINTY_RE`) rather than re-encoding
#: the same shape a second time.
_SPACED_RANGE_RE = re.compile(_NUMERAL_SINGLE_VALUE + r"(?:\s+[-–]\s*|[-–]\s+)" + _NUMERAL_TAIL_VALUE)
#: ``NUM <ws>* [x×*] <ws>* 10 <ws>+ NUM`` -- a superscript exponent that PDF text
#: extraction flattened into its own separate token, e.g. ``"3.94 x 10 03"`` for
#: ``3.94 × 10^03``.
_FLATTENED_SCIENTIFIC_RE = re.compile(_NUMERAL_SINGLE_VALUE + r"\s*[x×*]\s*10\s+" + _NUMERAL_TAIL_VALUE)

#: Ordered ``(construct_name, pattern)`` pairs consulted by
#: :func:`enclosing_numeric_construct`. Order only matters in the (currently
#: unobserved) case where more than one pattern's match strictly contains the same
#: span; the first match found wins.
_ENCLOSING_CONSTRUCT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ascii6_uncertainty", _ASCII6_UNCERTAINTY_RE),
    ("spaced_range", _SPACED_RANGE_RE),
    ("flattened_scientific", _FLATTENED_SCIENTIFIC_RE),
)


def enclosing_numeric_construct(text: str, start: int, end: int) -> str | None:
    """Return the NAME of a multi-token numeric construct in ``text`` that STRICTLY
    CONTAINS the span ``text[start:end]`` -- i.e. the span is only a PART of a larger
    construct, never the whole of it -- or ``None`` when the span stands alone.

    This answers a narrower question than either of this module's other two span
    questions, and sits strictly between them in strength:

    - :func:`find_numeral_extent` asks "is this span a WHOLE NUMERAL" -- a
      character-level question about where one numeral's own extent starts and ends.
    - :func:`enclosing_numeric_construct` (this function) asks "is this whole numeral
      only ONE PIECE of a larger multi-token numeric VALUE" -- a token-level question
      that only makes sense to ask once the character-level question above has already
      been answered "yes". A span that is not even a whole numeral is not this
      function's concern.
    - :func:`parse_numeric_span` asks "does this span, taken alone and in isolation,
      PARSE" -- it never looks at the surrounding text at all, so it cannot see a
      multi-token construct this function exists to catch.

    Recognises at least:

    - ``"ascii6_uncertainty"``: ``NUM <ws> 6 <ws> NUM`` -- an ASCII ``6`` standing in
      for a mangled ``±`` (e.g. ``"307 6 10"`` for ``"307 ± 10"``). Shares
      :data:`_ASCII6_UNCERTAINTY_RE` verbatim with :func:`assess_glyph_health`'s
      document-wide fingerprint rather than re-encoding the same shape.
    - ``"spaced_range"``: ``NUM <ws>* [-–] <ws>* NUM`` with whitespace on AT LEAST ONE
      side of the dash (e.g. ``"1000 - 1200"``, ``"1000 – 1200"``). The TIGHT,
      no-whitespace form (``"1000-1200"``) is deliberately NOT reported here -- it is
      already recognised as ONE candidate by :data:`NUMERAL_EXTENT_RE` /
      :data:`NUMERAL_CANDIDATE_RE`, and reporting it here too would double-handle it
      and change existing, already-correct behaviour.
    - ``"flattened_scientific"``: ``NUM <ws>* [x×*] <ws>* 10 <ws>+ NUM`` -- a
      superscript exponent flattened into its own token by PDF text extraction (e.g.
      ``"3.94 x 10 03"`` for ``3.94 × 10^03``).

    The "NUM" piece in every pattern above is built from :data:`_NUMERAL_SINGLE_VALUE`
    / :data:`_NUMERAL_TAIL_VALUE`, the same shared grammar body :data:`_NUMERAL_BODY`
    is built from -- so this function does not introduce a fourth, independently
    drifting numeral grammar into this module.

    Only known caller today: :func:`carmel.services.dataset_producer.ground_quote`,
    which calls this AFTER its own :func:`find_numeral_extent` maximality check
    already passed, to refuse grounding a quote that is a whole numeral in isolation
    but only a fragment of a larger numeric construct in context.

    KNOWN GAP, recorded deliberately rather than forgotten: :mod:`carmel.services.grounding`'s
    evidence-window scanner is a second caller that plausibly needs this same guard --
    it does NOT yet call this function. That gap is an intentional scope boundary of
    the change that introduced this function, not an oversight: the window scanner is
    the load-bearing S1 corroboration gate, and tightening it deserves its own,
    separately reviewed commit rather than riding in on this one.
    """
    for name, pattern in _ENCLOSING_CONSTRUCT_PATTERNS:
        for match in pattern.finditer(text):
            if match.start() <= start and end <= match.end() and (match.start(), match.end()) != (start, end):
                return name
    return None


@dataclass(frozen=True)
class GlyphHealth:
    """A document-level glyph-corruption assessment, produced by
    :func:`assess_glyph_health` and passed into :func:`parse_numeric_span` as CONTEXT.

    GlyphHealth never mutates or rewrites the document text it was computed from --
    it is a read-only signal a caller carries alongside a span, not a rewriting pass.
    """

    suspects_dash_corruption: bool
    """True when the source document has zero U+2013 en dashes AND contains at least
    one bare lowercase ``digit e digit`` token -- the fingerprint of the three
    Elsevier papers that encode en dash as ASCII ``e``."""

    has_thorn_plus_marker: bool
    """Whether ``þ`` (U+00FE, used as a plus-sign substitute) appears anywhere in the
    assessed text."""

    has_equals_ambiguity_marker: bool
    """Whether ``¼`` (U+00BC, sometimes standing in for ``=``) appears anywhere in the
    assessed text. Recorded only -- this module never decodes ``¼``."""

    has_slash_c0_minus_marker: bool
    """Whether the literal sequence ``/C0`` (a minus-sign substitute) appears anywhere
    in the assessed text."""

    has_ascii6_uncertainty_marker: bool
    """Whether the tight ``\\d+ 6 \\d+`` shape (e.g. ``307 6 10`` for ``307 ±
    10``) appears anywhere in the assessed text. Recorded only -- this module never
    decodes ASCII ``6`` into ``±``."""


def assess_glyph_health(document_text: str) -> GlyphHealth:
    """Compute a read-only glyph-health assessment of ``document_text``.

    Pure: never mutates or returns a rewritten copy of ``document_text``. Callers pass
    the resulting :class:`GlyphHealth` alongside individual spans drawn from (or
    related to) that document into :func:`parse_numeric_span`.
    """
    has_en_dash = "–" in document_text
    has_bare_exponent_shape = bool(_BARE_DASH_CORRUPTION_RE.search(document_text))
    return GlyphHealth(
        suspects_dash_corruption=(not has_en_dash) and has_bare_exponent_shape,
        has_thorn_plus_marker="þ" in document_text,
        has_equals_ambiguity_marker="¼" in document_text,
        has_slash_c0_minus_marker="/C0" in document_text,
        has_ascii6_uncertainty_marker=bool(_ASCII6_UNCERTAINTY_RE.search(document_text)),
    )


@dataclass(frozen=True)
class Scalar:
    """One reconstructed value."""

    raw: str
    """The original input span, unmodified, exactly as given to
    :func:`parse_numeric_span`."""
    value: float
    """The reconstructed value. Always finite -- see :class:`Unresolvable` for the
    non-finite case."""
    repairs: tuple[str, ...] = ()
    """Explicit, enumerable names of every glyph repair applied to produce ``value``
    (e.g. ``("slash_c0_to_minus",)``), empty when none were needed. Never free text."""


@dataclass(frozen=True)
class Range:
    """A low-high pair reconstructed from a hyphen/en-dash-separated span."""

    raw: str
    """The original input span, unmodified, exactly as given to
    :func:`parse_numeric_span`."""
    low: float
    """The lower bound."""
    high: float
    """The upper bound."""
    repairs: tuple[str, ...] = ()
    """Explicit, enumerable names of every glyph repair applied to either bound."""


@dataclass(frozen=True)
class Unresolvable:
    """An explicit refusal to reconstruct a value, with the reason why."""

    raw: str
    """The original input span, unmodified, exactly as given to
    :func:`parse_numeric_span`."""
    reason: str
    """Human-readable explanation of why no trustworthy value could be produced."""
    repairs: tuple[str, ...] = ()
    """Always empty in practice (a refusal applies no repair), kept for symmetry with
    :class:`Scalar` / :class:`Range` so callers can treat the union uniformly."""


#: Discriminated union of every possible :func:`parse_numeric_span` outcome.
NumericResult = Scalar | Range | Unresolvable


@dataclass(frozen=True)
class NormalizedNumeral:
    """One span, textually validated and glyph-repaired, but NOT yet evaluated as a
    float. See :func:`normalize_numeric_span`."""

    raw: str
    """The original input span, unmodified, exactly as given to
    :func:`normalize_numeric_span`."""
    text: str
    """A strict ASCII numeral: optional leading ``-``, mantissa digits verbatim,
    optional ``e+NN``/``e-NN`` exponent. Mantissa and exponent digits are preserved
    VERBATIM from the input -- e.g. ``7.000Eþ17`` normalizes to ``"7.000e+17"``, never
    collapsed to ``"7e+17"``, because trailing zeros encode measurement precision, a
    fact a float can't carry and this type must not destroy."""
    repairs: tuple[str, ...] = ()
    """Explicit, enumerable names of every glyph repair applied to produce ``text``
    (e.g. ``("slash_c0_to_minus",)``), empty when none were needed. Never free text;
    every name is a member of :data:`REPAIR_NAMES`."""


def _find_range_separator(text: str) -> int | None:
    """Return the index of the hyphen/en-dash that separates a range's two bounds, or
    None if ``text`` carries no such separator.

    Positional, not regex-alternation-based, per the required grammar: a candidate
    character (ASCII ``-`` or U+2013) is a range separator only when it is NOT the
    first character (a leading ``-`` is a sign, e.g. ``-1.0``) and is NOT immediately
    preceded by ``e``/``E`` (that position is an exponent's own sign, e.g. ``1e-7``).
    """
    for i, ch in enumerate(text):
        if ch in ("-", "–") and i > 0 and text[i - 1] not in ("e", "E"):
            return i
    return None


def _refuse_common(work: str) -> str | None:
    """Apply the refusals that hold for the whole (already-stripped) span, before any
    range-vs-scalar branching -- shared verbatim by :func:`normalize_numeric_span` and
    :func:`parse_numeric_span` so the two can never drift apart on these checks.

    Returns a human-readable refusal reason, or ``None`` if none of these apply.
    """
    if not work:
        return "the span is empty; there is no numeral to reconstruct"

    lowered = work.lower()
    if lowered in _DISALLOWED_LITERALS:
        return f"'{work}' is a disallowed non-finite literal, not a numeral"

    if "_" in work:
        return "digit separators ('_') are not accepted, even though Python's bare float() would allow them"

    if _ASCII6_UNCERTAINTY_RE.fullmatch(work):
        # Rule #8: never silently repaired -- an explicit refusal naming the possible
        # ± interpretation, nothing stronger.
        return "possible ± uncertainty pattern (ASCII '6' standing in for '±') is not decoded by this module"

    return None


def _normalize_single_value(
    text: str,
    *,
    source_context: SourceContext,
    glyph_health: GlyphHealth,
) -> tuple[str, tuple[str, ...]] | str:
    """Validate ``text`` (no range separator) against the strict grammar and apply
    glyph repairs, WITHOUT evaluating it as a float.

    Returns ``(cleaned_text, repairs)`` on success, or a failure reason string.
    Internal helper only -- callers go through either :func:`normalize_numeric_span`
    (textual form, no finiteness check) or :func:`_parse_single_value` (which adds the
    float evaluation and finiteness check on top of this).
    """
    match = _CORE_VALUE_RE.fullmatch(text)
    if match is None:
        # fullmatch requires the ENTIRE (already boundary-trimmed) span to reduce to
        # one clean signed value -- this is what actually enforces rule #4: any
        # letters, '%', or formula fragments left over anywhere in the span (touching
        # the numeric token or not) mean the span was never just a number, so it is
        # refused rather than guessing which substring inside it was "the" value.
        return (
            "the span does not reduce to a single clean numeral: adjacent letters, "
            "'%', or other formula fragments are present"
        )

    mantissa_str = match.group("mantissa")
    exponent_str = match.group("exponent")
    emarker = match.group("emarker")
    lead_sign = match.group("lead_sign")
    exp_sign = match.group("exp_sign")

    if exponent_str is not None and "." in exponent_str:
        # Case A: illegal float shape (e.g. 0.6e1.0, 1e1.5). Always Unresolvable,
        # regardless of GlyphHealth/SourceContext -- never salvaged into some other
        # number.
        return f"illegal float literal: exponent '{exponent_str}' contains a decimal point"

    repairs: list[str] = []
    lead_negative = False
    if lead_sign is not None:
        if lead_sign.startswith("/C0"):
            lead_negative = True
            repairs.append("slash_c0_to_minus")
        elif lead_sign == "-":
            lead_negative = True
        elif lead_sign == "−":
            lead_negative = True
            repairs.append("unicode_minus_to_ascii")
        elif lead_sign == "–":
            lead_negative = True
            repairs.append("leading_en_dash_to_minus")
        # "+" needs no repair and no sign flip.

    exp_negative = False
    exp_has_explicit_sign = exp_sign is not None
    if exp_sign is not None:
        if exp_sign.startswith("þ"):
            repairs.append("thorn_to_plus")
        elif exp_sign == "-":
            exp_negative = True
        # "+" needs no repair and no sign flip.

    if (
        emarker == "e"
        and not exp_has_explicit_sign
        and "." not in mantissa_str
        and source_context == SourceContext.FLAT_PDF_TEXT
        and glyph_health.suspects_dash_corruption
    ):
        # Quarantine rule: a bare lowercase digit-e-digit token, no decimal point, no
        # explicit sign, inside FLAT_PDF_TEXT whose document is suspected of the en
        # dash -> ASCII 'e' substitution. Fail closed even though it would otherwise
        # parse to a perfectly finite float.
        return (
            "quarantined: a bare lowercase 'e' exponent token with no decimal point "
            "and no explicit sign, in FLAT_PDF_TEXT where dash corruption is "
            "suspected, may encode a corrupted en-dash range rather than genuine "
            "scientific notation"
        )

    # Mantissa and exponent digits are carried through VERBATIM -- never re-rendered
    # via float() -- so trailing zeros (measurement precision) survive intact.
    cleaned = mantissa_str
    if emarker is not None:
        cleaned = f"{cleaned}e{'-' if exp_negative else '+'}{exponent_str}"
    if lead_negative:
        cleaned = f"-{cleaned}"

    return cleaned, tuple(repairs)


def _parse_single_value(
    text: str,
    *,
    source_context: SourceContext,
    glyph_health: GlyphHealth,
) -> tuple[float, tuple[str, ...]] | str:
    """Parse ``text`` (no range separator) as one signed numeric value.

    Returns ``(value, repairs)`` on success, or a failure reason string. Internal
    helper only -- callers always go through :func:`parse_numeric_span`, which wraps
    the outcome in the public result types with the ORIGINAL (unstripped) span.

    Built on top of :func:`_normalize_single_value`: this layer adds exactly one
    thing on top of the textual normalization, the float evaluation and its
    finiteness check -- see :func:`normalize_numeric_span` for why that check does
    NOT belong in the textual layer.
    """
    normalized = _normalize_single_value(text, source_context=source_context, glyph_health=glyph_health)
    if isinstance(normalized, str):
        return normalized
    cleaned, repairs = normalized

    value = float(cleaned)  # safe: mantissa/exponent shapes are already validated above
    if not math.isfinite(value):
        return f"'{text}' evaluates to a non-finite value (inf/-inf/nan), which is never a trustworthy value"

    return value, repairs


def normalize_numeric_span(
    span: str,
    *,
    source_context: SourceContext,
    glyph_health: GlyphHealth,
) -> NormalizedNumeral | Unresolvable:
    """Textually validate and glyph-repair an ALREADY-SCOPED ``span`` into a strict
    ASCII numeral, or explicitly refuse -- WITHOUT evaluating it as a float.

    Applies every refusal :func:`parse_numeric_span` applies at the whole-span level
    (empty span, disallowed non-finite literals, digit separators, the ASCII-6
    uncertainty shape) plus the strict per-value grammar (whole-span purity, Case A
    illegal exponents, the FLAT_PDF_TEXT dash-corruption quarantine) applied by
    :func:`_normalize_single_value`. It additionally refuses a range (a hyphen/en-dash
    separator per :func:`_find_range_separator`) outright: a range is two numerals, not
    one, and this function only ever produces a single :class:`NormalizedNumeral`.

    Deliberately does NOT perform the float finiteness check that
    :func:`parse_numeric_span` performs on top of this. ``1E+400`` is a perfectly
    well-formed, exactly-representable decimal numeral -- it normalizes here -- but it
    is not a finite ``float`` (``float("1e+400")`` is ``inf``), so
    :func:`parse_numeric_span` still refuses it. Textual FORM and float EVALUATION are
    different layers, and this module deliberately lets them diverge exactly there:
    normalization is about whether the span reduces to a legitimate numeral written
    down on the page; finiteness is a property of what that numeral evaluates to once
    interpreted as an IEEE double. A caller that only needs the text (e.g. to store a
    significance-preserving decimal string) should use this function directly rather
    than going through the float-evaluating :func:`parse_numeric_span` and converting
    the result back to text, which would silently destroy trailing zeros.
    """
    work = span.strip()
    reason = _refuse_common(work)
    if reason is not None:
        return Unresolvable(raw=span, reason=reason)

    if _find_range_separator(work) is not None:
        return Unresolvable(
            raw=span,
            reason="the span is a range (contains a hyphen/en-dash bound separator), not a single numeral",
        )

    normalized = _normalize_single_value(work, source_context=source_context, glyph_health=glyph_health)
    if isinstance(normalized, str):
        return Unresolvable(raw=span, reason=normalized)
    text, repairs = normalized
    return NormalizedNumeral(raw=span, text=text, repairs=repairs)


def parse_numeric_span(
    span: str,
    *,
    source_context: SourceContext,
    glyph_health: GlyphHealth,
) -> NumericResult:
    """Reconstruct a trustworthy number (or range) from an ALREADY-SCOPED ``span``, or
    explicitly refuse.

    ``span`` must be exactly the cell/text some other, upstream component decided is
    "the value" (see the module scoping rule) -- this function never widens its own
    search window looking for a number inside a larger document. ``glyph_health`` is
    read-only context (see :class:`GlyphHealth`); it is never mutated, and the input
    text is never rewritten wholesale -- every repair is applied, and recorded, at its
    specific position within this one span.
    """
    work = span.strip()
    reason = _refuse_common(work)
    if reason is not None:
        return Unresolvable(raw=span, reason=reason)

    range_separator = _find_range_separator(work)
    if range_separator is not None:
        low_text = work[:range_separator]
        high_text = work[range_separator + 1 :]
        low_result = _parse_single_value(low_text, source_context=source_context, glyph_health=glyph_health)
        high_result = _parse_single_value(high_text, source_context=source_context, glyph_health=glyph_health)
        if isinstance(low_result, str):
            return Unresolvable(raw=span, reason=f"range low bound unresolvable: {low_result}")
        if isinstance(high_result, str):
            return Unresolvable(raw=span, reason=f"range high bound unresolvable: {high_result}")
        low_value, low_repairs = low_result
        high_value, high_repairs = high_result
        if low_value > high_value:
            # Fail closed rather than silently swap: a printed "9-2" never actually
            # states the ordering "2-9", so accepting it (by re-ordering the bounds)
            # would fabricate an ordering the source text does not contain. This
            # matches the module's existing posture toward every other
            # not-quite-a-clean-numeral shape (digit separators, comma-grouped
            # thousands): refuse outright rather than guess what was meant.
            return Unresolvable(
                raw=span,
                reason=(
                    f"range low bound ({low_value}) is greater than its high bound "
                    f"({high_value}); refused rather than silently swapped"
                ),
            )
        return Range(raw=span, low=low_value, high=high_value, repairs=low_repairs + high_repairs)

    single_result = _parse_single_value(work, source_context=source_context, glyph_health=glyph_health)
    if isinstance(single_result, str):
        return Unresolvable(raw=span, reason=single_result)
    value, repairs = single_result
    return Scalar(raw=span, value=value, repairs=repairs)
