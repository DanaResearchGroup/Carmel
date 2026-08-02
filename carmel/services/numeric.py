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
    "REPAIR_NAMES",
    "GlyphHealth",
    "NormalizedNumeral",
    "NumericResult",
    "Range",
    "Scalar",
    "SourceContext",
    "Unresolvable",
    "assess_glyph_health",
    "normalize_numeric_span",
    "parse_numeric_span",
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
