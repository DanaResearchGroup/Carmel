"""Deterministic, non-LLM grounding gate for literature findings.

This module decides whether an LLM-proposed :class:`~carmel.schemas.literature.
FindingPayload` and its claimed ``verbatim_quote`` are corroborated by the actual
bytes fetched for the cited work. It runs *before* a second LLM (the Verifier) ever
sees the finding: asking an LLM to check another LLM's fabrication just rubber-stamps
it, so this gate has to be a plain function of its inputs.

Every function in this module is a **pure function**: no LLM calls, no network, no
file I/O, no wall clock, no randomness. Given the same arguments it always returns
the same result.

**Claim discipline.** This is a *first filter* / defense layer against two specific
failure modes: FABRICATED QUOTES (text the model invented and attributed to a real
paper) and MISATTRIBUTED SOURCES (a real quote attached to the wrong citation). It
does **not**, and cannot, verify that a claim is *true* — a paper can be genuinely
quoted and still be wrong, retracted, or misinterpreted. Never describe this module
as guaranteeing citation integrity; it is one gate in a pipeline, not a proof.

**Offset-mapping honesty.** :func:`find_quote` matches against
:attr:`~carmel.agents.tools.extract.ExtractedText.normalized` (so whitespace
reflow, ligatures, and de-hyphenation don't break matching) but must report offsets
into :attr:`~carmel.agents.tools.extract.ExtractedText.text` (the raw text), because
that is what :class:`~carmel.schemas.literature.EvidenceRef` and downstream review
tooling display. ``extract.py`` explicitly does not guarantee ``normalized`` is
offset-aligned with ``text`` (ligature expansion, hyphen-joining, and whitespace
collapse all change string length). This module recovers raw offsets using
:func:`~carmel.agents.tools.extract.normalize_with_map` — the SAME primitive that
``normalize_for_match`` (and therefore ``extracted.normalized`` itself) is defined
on top of — together with :func:`~carmel.agents.tools.extract.raw_span`. Because
there is only one implementation of the normalization algorithm, there is no
divergence risk and no proportional-scaling fallback: the raw-index map always
describes exactly how ``extracted.normalized`` was produced. Callers should still
treat all raw offsets from this module as best-effort at the *edges* of a matched
span — see :func:`~carmel.agents.tools.extract.raw_span`'s docstring for exactly
how it handles a normalized-space span whose ends fall on a many-to-one collapse
(e.g. a whitespace run) — but the map itself can never silently desynchronize from
``normalize_for_match``.

**Known upstream weakness — references-section detection is fail-open.**
``extract.py``'s heading-based detection recognizes numbered headings and
headings that run into following body text on the same line (not just headings
alone on their own line), closing most of the previous gap. It still cannot
recognize a bibliography whose heading was dropped or garbled entirely by lossy
extraction. ``extract.py`` additionally exposes a *structural* detector,
:func:`~carmel.agents.tools.extract.find_bibliography_like_regions`, that flags
runs of lines dense in citation patterns (author-initial forms, parenthesized
years, "et al.", volume:page ranges, "doi:") as bibliography-like even with no
heading at all. :func:`ground_finding` treats a match inside a *confident*
structural region the same as an explicit references-section match
(``REFERENCES_ONLY``), and a match inside a merely *suggestive* region as an
explicit warning instead. As a last-resort defense, when a quote match falls deep
in the document tail and no ``references`` section was detected at all (labelled
or structural), :func:`ground_finding` also appends an explicit warning rather
than silently treating the match as safely inside the body.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from carmel.agents.tools.extract import (
    ExtractedText,
    TextSection,
    find_bibliography_like_regions,
    normalize_for_match,
    normalize_with_map,
    raw_span,
)
from carmel.schemas.campaign import ReactorType
from carmel.schemas.literature import (
    Citation,
    ExperimentalBenchmarkPayload,
    FindingPayload,
    GroundingStatus,
    GroundingVerdict,
    PriorModelPayload,
    QMCalculationPayload,
    QMProperty,
)
from carmel.services.numeric import (
    GlyphHealth,
    Range,
    Scalar,
    SourceContext,
    Unresolvable,
    assess_glyph_health,
    parse_numeric_span,
)

__all__ = [
    "QuoteMatch",
    "UnsupportedFindingPayloadError",
    "check_evidence_spans",
    "check_identity",
    "QuoteMissReason",
    "find_quote",
    "find_quote_with_reason",
    "ground_finding",
    "required_spans_for",
    "unreadable_reason",
]


#: Candidate numeric spans inside a normalized evidence window, for
#: :func:`_window_numeric_values`. The window is running prose/table text (never an
#: already-scoped cell), so this scanner's ONLY job is to find candidate spans with
#: proper boundaries -- every candidate is then validated by the strict numeric core
#: (:func:`carmel.services.numeric.parse_numeric_span`), which owns the definition of
#: "a trustworthy number". Design points:
#:
#: - Boundaries: a candidate may not touch a letter, digit, or dot on either side, so
#:   digits embedded in identifiers ("h2o", "gri30") are no longer salvaged the way an
#:   unanchored ``findall`` used to salvage them.
#: - The exponent branch deliberately over-captures a decimal point (``1.0`` in
#:   ``0.6e1.0``) so a corrupt token is captured WHOLE and refused by the strict core,
#:   instead of being split into salvageable fragments (``0.6e1`` == 6.0 and ``0``).
#: - An optional second half after a hyphen/en dash captures a printed range
#:   ("1200-1500", "0.5–2.0") as ONE candidate, which the strict core parses as a
#:   Range contributing both bounds -- not as the tokens ``1200`` and ``-1500``.
#: - A trailing sentence period is NOT part of the candidate ("... was 850." yields
#:   ``850``), but a letter directly after the span disqualifies it.
#: - A comma directly touching either end of the span ALSO disqualifies it: a
#:   comma-grouped thousands number ("1,000") is neither a single clean numeral
#:   nor two legitimate standalone values, so the boundary must reject it rather
#:   than let an unanchored digit run either side of the comma slip through as a
#:   spurious candidate ("1" and "000" out of "1,000") -- the strict core (see
#:   ``numeric.py``) already refuses a comma-bearing span outright when it is
#:   handed one whole, but that refusal only helps if the scanner stops carving
#:   the comma-adjacent digits into candidates in the first place.
#:
#: The window text is already casefolded by ``normalize_for_match``, so only a
#: lowercase ``e`` exponent marker can occur.
def _is_percent_adjacent_exponent(window_norm: str, match: re.Match[str]) -> bool:
    """Whether ``match`` is an exponent-form token immediately touching a ``%``.

    Only exponent-form tokens are judged: a plain number beside a percent sign is
    an ordinary percentage ("50%") and must stay readable. See the call site for
    why the exponent form is treated as corruption instead of a value.
    """
    if "e" not in match.group(0).casefold():
        return False
    before = window_norm[match.start() - 1] if match.start() > 0 else ""
    after = window_norm[match.end()] if match.end() < len(window_norm) else ""
    return before == "%" or after == "%"


_WINDOW_NUMBER_CANDIDATE_RE = re.compile(
    r"(?<![0-9a-z.,])"
    r"(?:/c0\s*|[-+−]|–(?=\d))?\d+(?:\.\d+)?(?:e[-+]?\d+(?:\.\d+)?)?"
    r"(?:[-–]\d+(?:\.\d+)?(?:e[-+]?\d+(?:\.\d+)?)?)?"
    r"(?![0-9a-z,])"
)

#: Shapes that mark a required anchor as numeric IN INTENT even when bare ``float()``
#: rejects it (e.g. the corrupt ``0.6e1.0``): an optional sign followed by a digit and
#: then only number-ish characters. Together with :func:`_is_numeric_literal` (which
#: additionally catches ``float()``-accepted forms like ``inf``/``nan``), this decides
#: that an anchor must be VALUE-compared -- a numeric-intent anchor that cannot be
#: strictly resolved is a hard missing-anchor result, never a substring search.
_NUMERIC_INTENT_RE = re.compile(r"[-+]?\d[\d.eE+-]*")

#: Word tokenizer used by :func:`_has_semantic_discrepancy` to compare the fuzzy
#: window and the claimed quote on WORD boundaries (a symmetric difference of word
#: sets), rather than raw ``difflib`` character opcodes -- a single deleted/added
#: word (e.g. "not") is exactly the kind of edit a high character-similarity ratio
#: can hide, since removing 3 characters from a 60-character sentence barely moves
#: the ratio.
_WORD_RE = re.compile(r"[a-zA-Z]+")

#: Negation tokens whose presence in exactly one of {window, needle} (per the word
#: symmetric-difference) flags a semantic discrepancy: a quote with "not"/"no"/etc.
#: deleted (or added) reads as almost the same string to a character-diff but means
#: the opposite thing. NOT calibrated against the 69-paper corpus (no repository of
#: negation-edit near-misses to calibrate against) -- a deliberately conservative,
#: small, unambiguous list.
_NEGATION_TOKENS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "cant",
        "doesnt",
        "dont",
        "didnt",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "without",
        "neither",
        "nor",
    }
)

#: Negating prefixes checked via stem-matching (see ``_has_semantic_discrepancy``):
#: only flagged when one diff word equals ``prefix + other_diff_word`` and both
#: forms are actually present, one on each side of the window/needle diff -- never
#: via a bare ``word.startswith(prefix)``, which would misfire on ordinary words
#: ("increase", "individual", "international", "instrument"). NOT calibrated
#: against the 69-paper corpus.
_NEGATING_PREFIXES = ("un", "in", "im", "il", "ir", "non")

#: Curated antonym pairs checked on the word symmetric difference only (not the
#: full word sets), so a pair that happens to co-occur unchanged in both window and
#: needle is never flagged. Deliberately small and NOT calibrated against the
#: 69-paper corpus -- covers common scientific-claim reversals, not exhaustive.
_ANTONYM_PAIRS = frozenset(
    {
        frozenset({"increase", "decrease"}),
        frozenset({"increases", "decreases"}),
        frozenset({"increased", "decreased"}),
        frozenset({"increasing", "decreasing"}),
        frozenset({"higher", "lower"}),
        frozenset({"more", "less"}),
        frozenset({"above", "below"}),
        frozenset({"positive", "negative"}),
        frozenset({"before", "after"}),
        frozenset({"faster", "slower"}),
        frozenset({"maximum", "minimum"}),
        frozenset({"consistent", "inconsistent"}),
        frozenset({"stable", "unstable"}),
        frozenset({"present", "absent"}),
        frozenset({"agrees", "disagrees"}),
        frozenset({"agreement", "disagreement"}),
    }
)

#: Default sliding-window fuzzy-match acceptance threshold for find_quote.
DEFAULT_FUZZY_THRESHOLD = 0.92
#: Default tight evidence window (~a paragraph / table row), deliberately small so a
#: coincidental nearby number can't satisfy a numeric anchor check.
DEFAULT_EVIDENCE_WINDOW = 300
#: A quote landing at or past this fraction of the raw document, with no labelled
#: (or structurally-detected) references section, triggers the fail-open-references
#: warning. Bibliographies are typically the last 15-40% of a paper's raw text, not
#: merely the last 15% of it -- a fixed 0.85 threshold badly under-covers common
#: cases (e.g. a references section that starts around the two-thirds mark of a
#: short paper with a long author list). 0.65 catches those while still leaving
#: room for a normal citation-heavy discussion section near the end of the body.
_TAIL_FRACTION = 0.65

#: Minimum extracted characters per PDF page below which the PDF is judged to carry no
#: recoverable text layer. Scanned/image-only PDFs (common for pre-1990 combustion
#: literature) extract to ~0 characters per page via pypdf, which has no OCR. Measured
#: on real papers, healthy PDFs yield 1360-5230 characters per page, so 100 is far
#: below any genuine document while still catching an empty text layer. Deliberately
#: applied ONLY to paginated PDFs: a short HTML page or text snippet is legitimately
#: small, and treating it as unreadable would excuse a fabricated quote against it.
_MIN_CHARS_PER_PAGE = 100
#: Space-loss detection. Some PDFs encode fonts without space glyphs, and pypdf does
#: not infer spaces from positional gaps, so text comes back run together
#: ("Mechanismandkineticsoftheisothermal..."). Note pypdf's
#: ``extraction_mode="layout"`` does NOT repair this -- measured, it is worse.
#:
#: Measured per BLOCK rather than over the whole document: damage is typically
#: partial (one real paper's abstract extracted cleanly while its body did not), and a
#: document-wide mean dilutes below any useful threshold precisely on the documents
#: that matter. Calibrated on 69 real papers: healthy documents put at most 0.14 of
#: their blocks over the per-block limit (and all but one put 0.00), while an observed
#: space-lost paper put 0.44 over it -- so 0.25 sits in a wide empty margin.
_SPACE_LOSS_BLOCK_CHARS = 2000
_SPACE_LOSS_BLOCK_MEAN_TOKEN = 12.0
_SPACE_LOSS_BLOCK_FRACTION = 0.25

#: Character-similarity floor for accepting a title as corroborating identity when it
#: does not occur exactly. This exists for the real case the old surname+year fallback
#: was reaching for: a title line that extracts imperfectly (ligatures, a hyphen broken
#: across a line, a dropped subtitle colon, an OCR slip) is still unmistakably the same
#: title.
#:
#: This ratio is NECESSARY BUT NOT SUFFICIENT, and the reason is worth stating plainly
#: because an earlier version of this comment claimed the opposite. It asserted that two
#: DIFFERENT combustion titles "do not reach" 0.85. That was never measured, and it is
#: false (spar round 7 P0). Measured character ratios between titles that differ only in
#: the fuel studied -- the single most common confusion in this literature:
#:
#:     methanol vs methane oxidation ........ 0.974
#:     n-heptane vs n-heptene ............... 0.984
#:     methane vs ethane .................... 0.992
#:     "Erratum to: <title>" vs "<title>" ... 0.905
#:
#: Every one clears 0.85 comfortably. Character similarity is simply the wrong metric
#: for titles: the discriminating word is a few characters inside a long, otherwise
#: identical string, so its contribution to the ratio is negligible -- exactly backwards
#: from its contribution to identity. Raising the threshold does not fix this (it would
#: reject genuine damaged titles long before it rejects 0.974), which is why the fix is
#: a second, orthogonal check -- :func:`_substituted_token` -- rather than a bigger
#: number here.
_TITLE_IDENTITY_FUZZY_THRESHOLD = 0.85

#: Shortest token that carries identity. Below this, tokens are stock connective words
#: and fragments of damaged extraction ("of", "in", "h2", a stray "ame" from a broken
#: "flame"), which are neither discriminating nor reliable enough to reject on.
_TITLE_TOKEN_MIN_LENGTH = 4

#: How similar two tokens must be before one is read as a SUBSTITUTION of the other
#: rather than an unrelated word. Deliberately well below the pairs it must catch
#: ("methane"/"methanol" 0.93, "heptane"/"heptene" 0.86, "ethane"/"methane" 0.92) so
#: the check does not depend on a knife-edge, and well above the similarity of two
#: genuinely different words that happen to share a stem.
_TITLE_TOKEN_SUBSTITUTION_RATIO = 0.7

#: Tokens within a normalized string. Unlike :data:`_WORD_RE` this keeps digits, because
#: a species or condition token is frequently alphanumeric ("co2", "h2o2", "gri30").
_IDENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
#: Everything that is not an identity character; used to collapse a window so a token
#: split across a line break still matches.
_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
#: Whether a single character can be part of an identity token.
_IS_IDENTITY_CHAR = re.compile(r"[a-z0-9]").match

#: Most characters extraction may drop from one token before it stops being a damaged
#: rendering of that token and starts being a different word.
_MAX_DAMAGED_CHARS = 2
#: Shortest token accepted as a damaged rendering, so noise cannot satisfy a title word.
_MIN_DAMAGED_TOKEN_LENGTH = 3

#: Short words that are grammar, not chemistry. Excluded from the must-be-present rule
#: for short tokens, which otherwise exists to pin formulas like "h2" and "co". These
#: carry no identity, and demanding them would fail a title over a dropped preposition.
_TITLE_SHORT_STOPWORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "via", "with"}
)  # fmt: skip

#: Phrases by which a document announces itself as being *about* another paper rather
#: than being it. Each reprints the original's title, and usually its DOI, so neither
#: the title route nor the DOI route can separate them -- only the announcement can.
#:
#: Shared with :mod:`carmel.services.acquisition`, which gates documents ENTERING the
#: evidence store, so that the vocabulary has exactly one definition. Two copies would
#: drift: adding "retraction" to one and not the other would silently reopen the hole
#: in whichever layer was missed.
SECONDARY_DOCUMENT_MARKERS: tuple[str, ...] = (
    "erratum",
    "corrigendum",
    "correction to",
    "comment on",
    "reply to",
    "retraction",
    "editorial expression of concern",
)

#: How far into the front matter to look for an article-type announcement. Journals
#: print the article type in the header; scanning further would match a mere mention of
#: an erratum in the body or a footnote.
#: How far into the document to look for a secondary-document marker.
#:
#: 600 was calibrated on a clean title-first layout and is too small for a real
#: publisher front matter block. An Elsevier ScienceDirect preamble (journal name,
#: volume, page range, ISSN, DOI banner, "Contents lists available at", the running
#: header) measured 816 characters BEFORE the title on a paper in the live corpus, so
#: the marker sat outside the window and an erratum passed as the original.
#:
#: Raised well past that rather than to it: front matter varies by publisher and this
#: window is the only thing standing between an erratum and a confirmed identity. The
#: cost of scanning further is a substring search over a few thousand characters, which
#: is nothing next to the fuzzy scan that follows; the cost of scanning too little is a
#: misattributed finding.
MARKER_SCAN_CHARS = 4000

#: Minimum length of a quote AFTER normalization (whitespace/punctuation collapse).
#: verbatim_quote already has a raw min_length=40 floor (literature_agent.py), but a
#: quote that is mostly whitespace/punctuation can normalize down to almost nothing,
#: which would then trivially exact-match nearly any document and silently bypass
#: the numeric-discrepancy and fuzzy-mismatch defenses built on top of find_quote's
#: result. 20 is meaningfully below the raw 40-char floor (normal whitespace/ligature
#: collapse can shrink a genuine quote somewhat) but well above anything a degenerate,
#: mostly-punctuation quote could normalize down to.
MIN_NORMALIZED_QUOTE_LENGTH = 20


class QuoteMissReason(StrEnum):
    """Why :func:`find_quote_with_reason` could not return a match.

    Every value REJECTS the finding -- this enum changes only the explanation written to
    the decision log, which is what a researcher reads when deciding whether an agent is
    unreliable or merely unlucky.
    """

    TOO_SHORT = "too_short"
    """The normalized quote was under :data:`MIN_NORMALIZED_QUOTE_LENGTH`. This is a
    malformed proposal, NOT evidence of fabrication -- the gate never even searched."""

    NOT_FOUND = "not_found"
    """Nothing resembling the quote appears in the document. The genuine fabrication
    signal: the agent asserted a quote the source does not contain in any form."""

    SEMANTIC_DISCREPANCY = "semantic_discrepancy"
    """A near-identical passage EXISTS, but the difference between it and the claimed
    quote changes the science -- an altered number, a deleted negation, a flipped
    antonym. Strictly more damning than NOT_FOUND and far more actionable: it names the
    specific alteration, so the operator can see exactly what was misrepresented."""


#: Operator-facing explanation per miss reason. These are read by a researcher deciding
#: whether an agent is unreliable, so each says what actually happened rather than
#: repeating one generic fabrication accusation for every case.
_QUOTE_MISS_EXPLANATIONS: dict[QuoteMissReason, str] = {
    QuoteMissReason.TOO_SHORT: (
        f"the claimed quote is shorter than {MIN_NORMALIZED_QUOTE_LENGTH} characters after "
        "normalization, so it was rejected without being searched for. This is a malformed "
        "proposal, NOT evidence that the quote was fabricated."
    ),
    QuoteMissReason.NOT_FOUND: (
        "the claimed verbatim quote was not located in the fetched artifact, exactly or by "
        "fuzzy matching, and no near-identical passage exists either. This is the signature "
        "of a fabricated quote."
    ),
    QuoteMissReason.SEMANTIC_DISCREPANCY: (
        "a near-identical passage EXISTS in the artifact, but it differs from the claimed "
        "quote in a way that changes its meaning -- an altered number, a deleted negation, "
        "or a reversed comparison. The source was read; the quote misrepresents it."
    ),
}

#: Minimum length of an identity term (title/author-surname/DOI/year) AFTER
#: normalization, below which the term is treated as NOT present rather than
#: matched. A 1-3 normalized-character term (e.g. an initials-only surname
#: fragment) can occur in almost any document by chance, spuriously "confirming"
#: identity via check_identity's strict-conjunction rule. 4 sits just below a
#: four-digit publication year (the shortest legitimate identity term this
#: function handles) so real years, surnames, titles, and DOIs are unaffected.
MIN_IDENTITY_TERM_LENGTH = 4


class UnsupportedFindingPayloadError(ValueError):
    """Raised by :func:`required_spans_for` when given a ``FindingPayload`` variant
    it does not have a required-anchor rule for.

    Named (rather than a bare ``ValueError``) so :func:`ground_finding` can catch it
    specifically and fail CLOSED (``SPANS_MISSING``) instead of letting an unrelated
    ``ValueError`` elsewhere in the call chain be misread as "no anchors required."
    """


class QuoteMatch(BaseModel):
    """A located quote, with raw-text offsets and a fuzziness signal.

    Attributes:
        start: Best-effort start offset into ``ExtractedText.text`` (raw text).
        end: Best-effort end offset into ``ExtractedText.text`` (raw text).
        ratio: 1.0 for an exact normalized match; the ``difflib`` similarity ratio
            (in ``[0, 1]``) for a fuzzy fallback match.
        exact: False for a fuzzy-fallback match, so callers can penalize it.
        section_label: The label of the ``TextSection`` containing ``start``
            (e.g. "body", "references"), or "body" if no section covers it.
        page: The page number of that section, if paginated.
    """

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    ratio: float = Field(ge=0, le=1)
    exact: bool
    section_label: str
    page: int | None = None


def _space_loss_fraction(text: str) -> float:
    """Fraction of fixed-size blocks of ``text`` whose mean token length says word
    spacing was lost there.

    Blockwise on purpose: space loss is usually confined to part of a document, so a
    document-wide mean hides it (an observed paper measured 11.1 overall -- under any
    workable limit -- while 44% of its blocks were run together).
    """
    fractions: list[bool] = []
    for start in range(0, len(text), _SPACE_LOSS_BLOCK_CHARS):
        tokens = text[start : start + _SPACE_LOSS_BLOCK_CHARS].split()
        if not tokens:
            continue
        mean_token_len = sum(len(t) for t in tokens) / len(tokens)
        fractions.append(mean_token_len >= _SPACE_LOSS_BLOCK_MEAN_TOKEN)
    if not fractions:
        return 0.0
    return sum(fractions) / len(fractions)


def unreadable_reason(extracted: ExtractedText) -> str | None:
    """Explain why ``extracted`` is too damaged to search for a quote in, or None.

    A failed quote lookup has two very different causes that the gate must not
    conflate: the agent fabricated the quote, or *we* could not read the document.
    Reporting the second as the first accuses an honest agent of fabrication in the
    decision log a researcher then acts on, and hides a fixable ingestion problem.

    Two damage modes are detected, both observed on real papers:

    1. **No searchable text.** Scanned/image-only PDFs extract to essentially nothing
       (pypdf does no OCR). Common for older combustion literature.
    2. **Lost word spacing.** Some PDFs encode fonts without space glyphs; pypdf does
       not infer spaces from positional gaps, so text returns as run-together tokens.
       An honestly-transcribed quote can then never match.

    This is deliberately a *diagnosis*, not a relaxation: it is consulted only after a
    quote has already failed to match, and never turns a rejection into an acceptance.

    Args:
        extracted: The fetched artifact's extracted text.

    Returns:
        A human-readable reason string, or None if the text looks usable.
    """
    if extracted.extractor == "pdf:unavailable":
        return (
            "The artifact is a PDF but no PDF text extractor is installed, so its text was never available to search."
        )

    tokens = extracted.text.split()
    searchable_chars = sum(len(t) for t in tokens)
    if not tokens:
        return "No text at all could be extracted from the artifact, so the quote could not be searched for."

    # Only paginated PDFs get the density check: a short HTML page or text snippet is
    # legitimately small, and calling it unreadable would excuse a fabricated quote.
    if extracted.extractor.startswith("pdf:") and extracted.page_count:
        per_page = searchable_chars / extracted.page_count
        if per_page < _MIN_CHARS_PER_PAGE:
            return (
                f"Only {searchable_chars} characters of text were extracted from a "
                f"{extracted.page_count}-page PDF ({per_page:.0f} per page); the file appears "
                "to have no text layer, which is characteristic of a scanned or image-only "
                "document that would need OCR."
            )

    damaged_fraction = _space_loss_fraction(extracted.text)
    if damaged_fraction >= _SPACE_LOSS_BLOCK_FRACTION:
        return (
            f"Text extraction lost word spacing across {damaged_fraction:.0%} of the "
            "document: its words are run together "
            "('Mechanismandkineticsoftheisothermal...'), so no honestly-transcribed "
            "quote could match it."
        )
    return None


def _section_for(sections: Sequence[TextSection], raw_start: int) -> tuple[str, int | None]:
    """Return the (label, page) of the section covering ``raw_start``, default body."""
    for sec in sections:
        if sec.start <= raw_start < sec.end:
            return sec.label, sec.page
    return "body", None


def _find_all_normalized(haystack: str, needle: str) -> list[int]:
    """All (possibly overlapping) start indices of ``needle`` within ``haystack``."""
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _has_semantic_discrepancy(window: str, needle: str) -> bool:
    """True if ``window`` and ``needle`` differ in a way that can flip the claim's
    meaning even though their character-similarity ratio is high: a changed digit,
    a deleted/added negation token, a negating prefix added/removed, or a curated
    antonym substitution.

    A fuzzy-similarity ratio alone is not a safe acceptance criterion for scientific
    quotes. The digit check is the original guard: changing "1200 K" to "1500 K" is
    a single-character edit with a ratio well above any reasonable threshold, yet is
    exactly the kind of fabrication this gate exists to catch (spar round 3
    hardening note). The negation/prefix/antonym checks close a second hole found
    later: deleting "not" from a 60-character sentence is a 3-character edit that
    barely moves the ratio, but "X does not increase Y" and "X does increase Y"
    are opposite claims. Any fuzzy candidate tripping any of these checks is
    disqualified outright, regardless of its ratio.

    The digit check remains character-opcode-based (unchanged behavior). The
    negation/prefix/antonym checks are word-boundary-based (a symmetric difference
    of lowercased word sets) rather than character-opcode-based, since a single
    word add/delete is exactly what they need to catch, and word-set comparison is
    robust to nearby, unrelated character-level reflow that a fuzzy match already
    tolerates. NOT calibrated against the 69-paper corpus (no repository of
    negation/antonym near-misses to calibrate against) -- deliberately
    conservative, curated lists; see the module-level constants for rationale.
    """
    sm = SequenceMatcher(None, window, needle)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if any(c.isdigit() for c in window[i1:i2]) or any(c.isdigit() for c in needle[j1:j2]):
            return True

    window_words = {w.lower() for w in _WORD_RE.findall(window)}
    needle_words = {w.lower() for w in _WORD_RE.findall(needle)}
    diff_words = window_words ^ needle_words
    if not diff_words:
        return False

    if diff_words & _NEGATION_TOKENS:
        return True

    for word in diff_words:
        for prefix in _NEGATING_PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix):
                stem = word[len(prefix) :]
                if stem in diff_words:
                    return True

    return any(pair <= diff_words for pair in _ANTONYM_PAIRS)


def _scan_best_window(haystack: str, needle: str, threshold: float) -> tuple[int, int, float] | None:
    """The best-scoring window of ``needle``'s length, or None if none clears ``threshold``.

    Shared by :func:`_fuzzy_search` and :func:`_best_fuzzy_window`, which ran
    character-for-character identical scans and differed only in what they did with
    the winner.

    ``difflib``'s ``real_quick_ratio`` and ``quick_ratio`` are documented UPPER BOUNDS
    on ``ratio``, so a window whose bound falls under ``threshold`` cannot reach it and
    is skipped without the quadratic matching pass. This cannot change the answer
    (F9): every window that would have cleared the threshold still does, the global
    maximum is therefore unchanged whenever it clears the threshold, and when it does
    not both the old and new code return None. Ties still resolve to the earliest
    window, because the surviving windows keep their original order.

    One matcher is reused with the needle installed as seq2, so ``difflib``'s
    b2j index is built once for the whole scan instead of once per window.
    """
    n = len(needle)
    h = len(haystack)
    if n == 0 or h == 0 or h < n:
        return None
    last = max(h - n, 0)
    stride = max(1, n // 8)
    positions = list(range(0, last + 1, stride))
    if positions and positions[-1] != last:
        positions.append(last)

    matcher = SequenceMatcher(None)
    matcher.set_seq2(needle)
    best: tuple[int, int, float] | None = None
    for pos in positions:
        matcher.set_seq1(haystack[pos : pos + n])
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if best is None or ratio > best[2]:
            best = (pos, pos + n, ratio)
    if best is None or best[2] < threshold:
        return None
    return best


def _fuzzy_search(haystack: str, needle: str, threshold: float) -> tuple[int, int, float] | None:
    """Sliding-window ``difflib`` search for the best-matching span of ``needle``'s
    length within ``haystack``, accepted only when its ratio clears ``threshold`` AND
    no part of the diff constitutes a semantic discrepancy (see
    :func:`_has_semantic_discrepancy`) — a quote that differs from the source by a
    changed number, a deleted/added negation, or an antonym substitution must never
    be accepted as a fuzzy match, however high its character-similarity ratio.

    The stride is coarsened for long haystacks/needles rather than checking every
    single offset — a best-effort tradeoff appropriate for a fallback path that only
    runs after an exact match has already failed.
    """
    best = _scan_best_window(haystack, needle, threshold)
    if best is None:
        return None

    pos, end, _ratio = best
    if _has_semantic_discrepancy(haystack[pos:end], needle):
        return None
    return best


def _best_fuzzy_window(haystack: str, needle: str, threshold: float) -> tuple[int, int, float] | None:
    """Find the best-matching window IGNORING the semantic-discrepancy veto.

    :func:`_fuzzy_search` deliberately conflates "nothing here resembles the quote" with
    "a passage here is nearly identical but the diff changes the science" -- both return
    None, because both must reject. This re-runs the same scan WITHOUT the veto purely to
    tell those two apart for the decision log. It never grants a match; only
    :func:`_fuzzy_search` can do that.

    Args:
        haystack: Normalized document text.
        needle: Normalized claimed quote.
        threshold: Same ratio floor ``_fuzzy_search`` uses.

    Returns:
        ``(start, end, ratio)`` of the best window clearing ``threshold``, else None.
    """
    return _scan_best_window(haystack, needle, threshold)


def find_quote(
    extracted: ExtractedText, quote: str, *, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> QuoteMatch | None:
    """Locate ``quote`` within ``extracted``, returning raw-text offsets.

    Matching happens in normalized space (so whitespace reflow, ligatures, and
    line-break hyphenation in the source PDF don't defeat an honest quote) but the
    returned offsets are into ``extracted.text``, the raw text — see the module
    docstring for exactly how reliable that back-mapping is.

    Algorithm: try an exact substring match of the normalized quote against
    ``extracted.normalized`` first. Only on failure, fall back to a sliding-window
    ``difflib.SequenceMatcher`` search, accepted only when its ratio is
    ``>= fuzzy_threshold`` AND the diff has no semantic discrepancy (see
    :func:`_has_semantic_discrepancy`) — a quote that changes a number (e.g. "1200 K"
    to "1500 K"), deletes/adds a negation, or substitutes an antonym is a
    near-perfect character match but a scientifically false (or reversed) quote, so
    it is rejected outright rather than fuzzy-accepted. The fallback match, when
    accepted, carries ``exact=False`` so callers (notably :func:`ground_finding`) can
    distinguish and penalize it.

    Args:
        extracted: The fetched artifact's extracted text.
        quote: The claimed verbatim quote to locate.
        fuzzy_threshold: Minimum ``difflib`` ratio to accept a fuzzy fallback match.

    Returns:
        A :class:`QuoteMatch`, or ``None`` if the quote could not be located at all.
    """
    match, _reason = find_quote_with_reason(extracted, quote, fuzzy_threshold=fuzzy_threshold)
    return match


def find_quote_with_reason(
    extracted: ExtractedText, quote: str, *, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> tuple[QuoteMatch | None, QuoteMissReason | None]:
    """Locate ``quote``, also reporting WHY it was not located.

    :func:`find_quote` collapses four genuinely different outcomes into a bare ``None``,
    and :func:`ground_finding` then reports all four to the decision log as "this looks
    like a fabricated quote". For :attr:`QuoteMissReason.TOO_SHORT` that accusation is
    simply false, and for the discrepancy cases it discards the most actionable
    diagnostic the gate produces: "the agent altered a measured value" and "the agent
    invented this wholesale" are different findings about a model's behaviour, and a
    researcher reads this log to tell them apart.

    All four outcomes still REJECT the finding; only the explanation differs.

    Args:
        extracted: The fetched artifact's extracted text.
        quote: The claimed verbatim quote to locate.
        fuzzy_threshold: Minimum ``difflib`` ratio to accept a fuzzy fallback match.

    Returns:
        ``(match, None)`` when located, otherwise ``(None, reason)``.
    """
    normalized_quote = normalize_for_match(quote)
    if len(normalized_quote) < MIN_NORMALIZED_QUOTE_LENGTH:
        return None, QuoteMissReason.TOO_SHORT
    haystack = extracted.normalized
    _, index_map = normalize_with_map(extracted.text)

    idx = haystack.find(normalized_quote)
    if idx != -1:
        norm_start, norm_end = idx, idx + len(normalized_quote)
        raw_start, raw_end = raw_span(index_map, norm_start, norm_end, len(extracted.text))
        label, page = _section_for(extracted.sections, raw_start)
        return (
            QuoteMatch(start=raw_start, end=raw_end, ratio=1.0, exact=True, section_label=label, page=page),
            None,
        )

    fuzzy = _fuzzy_search(haystack, normalized_quote, fuzzy_threshold)
    if fuzzy is None:
        # Distinguish "nothing resembling this text is here" from "something very like
        # it is here, but the diff changes the science". The second is the strongest
        # signal this gate can emit and must not be reported as a generic miss.
        near = _best_fuzzy_window(haystack, normalized_quote, fuzzy_threshold)
        if near is not None:
            return None, QuoteMissReason.SEMANTIC_DISCREPANCY
        return None, QuoteMissReason.NOT_FOUND
    norm_start, norm_end, ratio = fuzzy
    raw_start, raw_end = raw_span(index_map, norm_start, norm_end, len(extracted.text))
    label, page = _section_for(extracted.sections, raw_start)
    return (
        QuoteMatch(start=raw_start, end=raw_end, ratio=ratio, exact=False, section_label=label, page=page),
        None,
    )


def _present_outside_references(extracted: ExtractedText, term_normalized: str) -> bool:
    """True if ``term_normalized`` occurs anywhere in the text outside a references
    section (i.e. at least one occurrence is NOT exclusively inside "references").

    Terms shorter than :data:`MIN_IDENTITY_TERM_LENGTH` after normalization are
    treated as absent (never present): a very short fragment (e.g. an initial or a
    one-to-three-character remainder) can occur in almost any document by chance,
    which would let it spuriously satisfy an identity check it should not confirm.
    This floor is applied here, not only at the surname call site, so it uniformly
    fail-closes every identity term routed through this helper (title, surname,
    DOI, year)."""
    if len(term_normalized) < MIN_IDENTITY_TERM_LENGTH:
        return False
    _, index_map = normalize_with_map(extracted.text)
    for pos in _find_all_normalized(extracted.normalized, term_normalized):
        raw_start, _ = raw_span(index_map, pos, pos + len(term_normalized), len(extracted.text))
        label, _ = _section_for(extracted.sections, raw_start)
        if label != "references":
            return True
    return False


def _all_tokens(text_normalized: str) -> list[str]:
    """Every token of an already-normalized string, in order."""
    return _IDENTITY_TOKEN_RE.findall(text_normalized)


def _identity_tokens(text_normalized: str) -> set[str]:
    """Long, identity-bearing tokens of an already-normalized string."""
    return {t for t in _all_tokens(text_normalized) if len(t) >= _TITLE_TOKEN_MIN_LENGTH}


def _formula_tokens(text_normalized: str) -> set[str]:
    """Short non-stopword tokens: overwhelmingly chemical formulas and their kin.

    "h2", "co", "o2", "no2", "n2o", "ch4". In a combustion title these are the most
    discriminating tokens there are, and they are far too short for
    :func:`_substituted_token`'s near-variant test to see a substitution in them --
    ratio("h2", "d2") is 0.5, well under any usable threshold.
    """
    return {
        t for t in _all_tokens(text_normalized) if len(t) < _TITLE_TOKEN_MIN_LENGTH and t not in _TITLE_SHORT_STOPWORDS
    }


def _is_damaged_form_of(candidate: str, token: str) -> bool:
    """Whether ``candidate`` looks like ``token`` with characters lost in extraction.

    Extraction damage REMOVES characters -- a dropped ligature glyph leaves "tue" for
    "tube", "ame" for "flame" -- so a damaged token is a subsequence of the original. A
    substituted word is not: "benzene" is not a subsequence of "toluene".

    This is the test that character similarity could not do. On similarity alone
    "tube"/"tue" scores 0.86 and "methane"/"methanol" scores 0.93, so any threshold
    admitting the damage also admits the different paper. Subsequence separates them
    cleanly, because "methanol" is not "methane" with letters dropped -- it has an "o"
    that "methane" never had.

    Args:
        candidate: A token found in the document window.
        token: The token from the cited title it might be a damaged rendering of.

    Returns:
        True if ``candidate`` is a subsequence of ``token`` and lost at most
        :data:`_MAX_DAMAGED_CHARS` characters. The length bound stops a short token
        waving through a long one: "in" is a subsequence of "ignition", and without it
        any title word containing common letters in order would be satisfied by noise.
    """
    if len(candidate) < _MIN_DAMAGED_TOKEN_LENGTH or len(token) - len(candidate) > _MAX_DAMAGED_CHARS:
        return False
    if len(candidate) >= len(token):
        return False
    remaining = iter(token)
    return all(character in remaining for character in candidate)


def _expand_to_token_boundaries(haystack: str, start: int, end: int) -> str:
    """Widen a character slice outward until neither edge cuts a token in half.

    The fuzzy scan walks fixed-length windows at a stride, so an edge routinely lands
    mid-token: a title occurrence one character longer than the citation (an inserted
    line break, a ligature expanded to two characters) leaves the window ending at
    "tub" where the document says "tube". Judged as a character slice that reads as a
    missing token and refuses a paper that is in fact the right one -- and at a
    different stride offset the same document confirms, which made the verdict depend
    on where the stride happened to land rather than on what the document says.

    Args:
        haystack: Normalised document text.
        start: Slice start, possibly mid-token.
        end: Slice end, possibly mid-token.

    Returns:
        The widened slice. Never narrower than the input, so this cannot hide a
        discrepancy that the unexpanded window would have caught.
    """
    while start > 0 and _IS_IDENTITY_CHAR(haystack[start - 1]):
        start -= 1
    while end < len(haystack) and _IS_IDENTITY_CHAR(haystack[end]):
        end += 1
    return haystack[start:end]


def _substituted_token(window_normalized: str, title_normalized: str) -> str | None:
    """Return a title token this window fails to carry, or None if it carries them all.

    This is the check that makes fuzzy title matching safe. EVERY identity token of the
    cited title must be present in the window -- long and short alike.

    It previously used a *discrepancy* test for long tokens: contradicted only when a
    token was absent AND the window held a near-variant of it, on the reasoning that a
    merely-absent token is extraction damage while a substituted one is a different
    paper. That rule was calibrated on three near-variant pairs (methane/methanol 0.93,
    heptane/heptene 0.86, ethane/methane 0.92) and does not generalise, because most of
    the class it has to cover -- titles differing in one discriminating word -- differ
    in DISSIMILAR words. Executed against the real code, it confirmed a different paper
    for toluene/benzene, syngas/biogas, kerosene/gasoline, ethanol/ammonia and
    gasoline-surrogate/diesel-surrogate: the discriminating word is not a near-variant,
    nothing contradicted, and the 0.85 character ratio decided alone.

    The premise underneath it was also false in the other direction. Near-variant
    similarity cannot separate damage from substitution, because the two overlap:
    "methane"/"methanol" scores 0.93 and IS a different paper, while a ligature-mangled
    "flame"/"ame" scores 0.75 and is damage. The old rule therefore refused the damage
    it meant to tolerate and admitted the substitution it meant to catch.

    Requiring presence resolves this without needing to tell the two apart. Damage now
    fails closed, which is the correct direction for a gate against misattribution --
    and it costs nothing observed: measured against the 8-paper live corpus, EVERY real
    grounded finding confirms identity through the exact-match path, never through this
    fuzzy fallback. What the fuzzy path still buys is tolerance of word order, spacing
    and punctuation differences, which is real; what it no longer buys is tolerance of a
    missing content word, which was never safe.

    Tokens the window holds but the title does not are IGNORED. The window is a slice of
    document text bracketing the title, so it legitimately overruns into neighbouring
    words; requiring set equality in both directions would refuse on alignment, not on
    identity.
    """
    window_long = _identity_tokens(window_normalized)
    # Separators stripped, so a token split across a PDF line break ("Igni\ntion")
    # still matches. That split is the damage-tolerance case the fuzzy path exists for
    # and it is ubiquitous in real extractions, so whole-token presence alone is too
    # strict. Crucially this still refuses a SUBSTITUTION: "benzene" does not occur
    # anywhere inside a toluene window, and "methane" is not a substring of "methanol".
    window_collapsed = _SEPARATOR_RE.sub("", window_normalized)
    for token in sorted(_identity_tokens(title_normalized)):
        if token in window_long or token in window_collapsed:
            continue
        # Candidates come from ALL window tokens, not just the long ones: damage
        # SHORTENS a token, so the damaged form of a long title word is frequently
        # short enough to fall below the long-token threshold ("tube" -> "tue").
        if any(_is_damaged_form_of(candidate, token) for candidate in _all_tokens(window_normalized)):
            continue
        return token

    # Short formula tokens must match WHOLE. No collapsed fallback for these: "co" is a
    # substring of "combustion", so substring matching would wave through exactly the
    # H2/D2 and CO/CO2 distinctions these tokens exist to enforce.
    window_short = _formula_tokens(window_normalized)
    for token in sorted(_formula_tokens(title_normalized)):
        if token not in window_short:
            return token
    return None


def secondary_document_marker(head: str, requested_title: str) -> str | None:
    """Return the phrase marking ``head`` as a document *about* the requested paper.

    Args:
        head: Lowercased front matter of the document.
        requested_title: Lowercased title being checked against, used to suppress the
            check for papers whose own title contains a marker phrase (a paper
            genuinely titled "Comment on ..." is a legitimate document).

    Returns:
        The matched marker, or None when the document does not announce itself as a
        secondary document.
    """
    # Normalise before matching. The raw text carries the line breaks, hyphenation and
    # runs of whitespace that extraction leaves behind, so "correction to" arrives as
    # "correction\nto" or "correc- tion to" and a literal substring search misses it --
    # on exactly the documents this gate exists to catch, since a marker is a heading
    # and headings are where line breaks land.
    window = _SEPARATOR_RE.sub(" ", head[:MARKER_SCAN_CHARS]).strip()
    for marker in SECONDARY_DOCUMENT_MARKERS:
        # The requested title gets the same normalisation, so the "genuinely titled
        # 'Comment on ...'" exemption still fires when the citation's own spacing is
        # irregular.
        if marker in window and marker not in _SEPARATOR_RE.sub(" ", requested_title):
            return marker
    return None


def _title_confirmed(extracted: ExtractedText, title_normalized: str) -> bool:
    """True if the cited work's title corroborates this document's identity.

    Exact-first, then a bounded fuzzy fallback. The fuzzy pass is NOT a loosening of
    the identity rule -- it is what makes the rule affordable to enforce everywhere.
    The surname+year fallback it replaces existed only because a title can extract
    imperfectly; matching the title fuzzily serves that same case directly, and far
    more specifically than a surname and a four-digit year ever could.

    Every candidate window clearing the threshold is checked, not merely the single
    best one, because the best-scoring occurrence of a title is very often the one in
    a review's *reference list* -- and that occurrence is precisely the one that must
    not count. Stopping at the best window would therefore reject the honest case
    (title on page 1 AND in a bibliography) for the wrong reason.
    """
    if len(title_normalized) < MIN_IDENTITY_TERM_LENGTH:
        return False
    if _present_outside_references(extracted, title_normalized):
        return True

    haystack = extracted.normalized
    n = len(title_normalized)
    if not haystack or n == 0:
        return False
    _, index_map = normalize_with_map(extracted.text)
    last = max(len(haystack) - n, 0)
    stride = max(1, n // 8)
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    # One matcher for the whole scan, with the title installed as seq2 so difflib
    # indexes it once rather than once per window, and the two documented upper bounds
    # on ratio() used as a prefilter. A window whose bound is under the threshold
    # cannot reach it, so this skips the quadratic pass without changing which windows
    # are accepted (F9) -- the same reasoning as _scan_best_window, and the reason the
    # threshold appears in the prefilter rather than some looser proxy.
    matcher = SequenceMatcher(None)
    matcher.set_seq2(title_normalized)
    for pos in positions:
        window = haystack[pos : pos + n]
        matcher.set_seq1(window)
        if matcher.real_quick_ratio() < _TITLE_IDENTITY_FUZZY_THRESHOLD:
            continue
        if matcher.quick_ratio() < _TITLE_IDENTITY_FUZZY_THRESHOLD:
            continue
        if matcher.ratio() < _TITLE_IDENTITY_FUZZY_THRESHOLD:
            continue
        # A high ratio is not enough: see _TITLE_IDENTITY_FUZZY_THRESHOLD. Skip this
        # window rather than rejecting outright, because a document can legitimately
        # contain both a contradicting window (a neighbouring paper named in the body)
        # and an honest one (its own title on page 1).
        # Compare against a TOKEN-ALIGNED widening of this window, not the raw slice:
        # a half-token at either edge is an artefact of where the stride landed, not a
        # statement about the document (F2).
        aligned = _expand_to_token_boundaries(haystack, pos, pos + n)
        if _substituted_token(aligned, title_normalized) is not None:
            continue
        raw_start, raw_end = raw_span(index_map, pos, pos + n, len(extracted.text))
        # Classify by BOTH ends, not just the start (spar round 7). A window is n chars
        # long and the scan is strided, so one can begin just before the references
        # boundary and extend into the first reference entry. Labelling it by raw_start
        # alone would call that window "body" and confirm a title that occurs only in
        # the bibliography -- the precise confusion this section check exists to catch.
        start_label, _ = _section_for(extracted.sections, raw_start)
        end_label, _ = _section_for(extracted.sections, max(raw_start, raw_end - 1))
        if start_label != "references" and end_label != "references":
            return True
    return False


def _surname(author: str) -> str:
    """Best-effort first-author surname extraction from a free-text author string."""
    if "," in author:
        return author.split(",", 1)[0].strip()
    parts = author.split()
    return parts[-1].strip() if parts else author.strip()


def check_identity(extracted: ExtractedText, citation: Citation) -> bool:
    """Confirm the fetched artifact actually IS the cited work.

    A stored artifact's ``sha256`` proves what bytes were stored, never that those
    bytes are the cited paper. This check is a **strict conjunction**, deliberately
    never an OR of weak signals (spar round 3, P1-14): an OR would be satisfied by
    any review article, survey, or reference list that merely *mentions* the cited
    work — exactly the misattribution this function exists to catch.

    Rule:
      - If ``citation.doi`` is set, BOTH the normalized DOI and the title must
        corroborate: ``doi_ok and _title_confirmed(...)``. A DOI mentioned in a
        review article's body text does not, by itself, make that article the cited
        paper (spar round N, P1-2), so DOI presence alone is never sufficient.

        There is deliberately NO surname-based escape from the title requirement any
        more (spar round 5 P0, carried through round 6). The previous rule accepted
        ``doi_ok and (title_ok or (author_ok and doi_year_ok))``, and a review or
        discussion article can carry all three of those weak signals honestly: it
        cites the primary DOI, names the first author while discussing the work, and
        -- being a review -- contains a great many four-digit years, of which the
        citation's is almost certainly one. Such an article would then be confirmed
        as the cited paper, and a quote lifted from the REVIEW's own prose would be
        recorded as fully grounded under the PRIMARY paper's citation. That is the
        exact misattribution this function exists to prevent, and no amount of
        conjoining weak signals fixes it, because each of them is individually
        satisfied by the wrong document for entirely innocent reasons.

        The case the surname fallback was actually reaching for -- a title that
        extracts imperfectly -- is served directly and much more specifically by
        :func:`_title_confirmed`, which accepts a high-similarity fuzzy title match
        outside the references section.
      - Otherwise (no DOI), ALL of the following must hold: the normalized title
        occurs, the first author's surname occurs, and (if ``citation.year`` is
        set) the year occurs. If no authors are given there is nothing to confirm
        authorship against, so identity is not confirmed.

    In every case, a match found only inside a ``references``-labelled section (per
    ``extract.py``'s section typing) does not count: that is a citation mention, not
    corroborated identity.

    Args:
        extracted: The fetched artifact's extracted text.
        citation: The citation the finding claims to come from.

    Returns:
        True only if identity is confirmed by the strict rule above.
    """
    # Gate every acceptance path on the article-type announcement FIRST. An erratum,
    # corrigendum, comment or reply reprints the original's full title by construction
    # ("Erratum to: <title>", measured ratio 0.905) and prints the original's DOI in its
    # own front matter, so BOTH conjuncts of the DOI rule are satisfied honestly and
    # neither identity route can separate the two documents. Only this announcement can.
    #
    # acquisition.check_identity already runs this gate, but it guards documents
    # ENTERING the store, against the request they were dropped for. That does not cover
    # the corpus pass, which is the reachable case: an erratum LEGITIMATELY held on its
    # own merits, whose text the agent then cites as the original paper it concerns. A
    # quote from the erratum's prose would be recorded as fully grounded under the
    # original's citation. Returning False here dominates every accept path below, so
    # the gate holds for any route added in future (spar round 7 P0).
    marker = secondary_document_marker(extracted.text[:MARKER_SCAN_CHARS].lower(), citation.title.lower())
    if marker is not None:
        return False

    title_norm = normalize_for_match(citation.title)
    title_ok = _present_outside_references(extracted, title_norm)

    author_ok = False
    if citation.authors:
        surname_norm = normalize_for_match(_surname(citation.authors[0]))
        author_ok = _present_outside_references(extracted, surname_norm)

    if citation.doi:
        doi_norm = normalize_for_match(citation.doi)
        doi_ok = _present_outside_references(extracted, doi_norm)
        return doi_ok and _title_confirmed(extracted, title_norm)

    if not citation.authors:
        return False

    year_ok = True
    if citation.year is not None:
        year_norm = normalize_for_match(str(citation.year))
        year_ok = _present_outside_references(extracted, year_norm)

    return title_ok and author_ok and year_ok


def _is_numeric_literal(s: str) -> bool:
    try:
        float(s)
    except TypeError, ValueError:
        return False
    return True


def _numeric_anchor_intent(req: str) -> bool:
    """True when required anchor ``req`` is numeric in intent.

    A numeric-intent anchor is ALWAYS value-compared against the strict numeric
    values found in the evidence window; if it cannot itself be strictly resolved to
    a value, it is a hard missing-anchor result. It is never demoted to a substring
    search -- the old behaviour for ``float()``-rejected shapes, which silently
    turned a numeric comparison into a text comparison that corrupt or coincidental
    characters could satisfy.

    Args:
        req: The required anchor string (typically ``str()`` of a typed float/int, or
            a free-text payload field such as a unit or species name).

    Returns:
        True if ``req`` must go through strict numeric resolution and value
        comparison; False if it is an ordinary text anchor.
    """
    if _is_numeric_literal(req):
        # Covers everything bare float() accepts, including the non-finite literals
        # ('inf', 'nan', 'infinity') that must hard-fail rather than text-match.
        return True
    return _NUMERIC_INTENT_RE.fullmatch(req.strip()) is not None


#: Maps :attr:`ExtractedText.extractor` values known to come from a flattened PDF
#: text layer -- the only source known to exhibit the en-dash-as-ASCII-``e``
#: substitution -- to :attr:`~carmel.services.numeric.SourceContext.FLAT_PDF_TEXT`.
#: Every other extractor value (``"html"``, ``"text"``, ``"pdf:unavailable"``,
#: ``"unknown"``, or anything unrecognized) maps to
#: :attr:`~carmel.services.numeric.SourceContext.OPERATOR_RAW`: none of those
#: sources are known to carry that specific corruption, and OPERATOR_RAW is the
#: strict core's "no special quarantine" context. This is a real, documented
#: distinction (``ExtractedText.extractor`` enumerates exactly these string
#: values), not an invented heuristic.
_PDF_EXTRACTOR_PREFIX = "pdf:pypdf"


def _source_context_for(extracted: ExtractedText) -> SourceContext:
    """The strict core's :class:`SourceContext` for text extracted into ``extracted``.

    Derived from ``extracted.extractor`` rather than hardcoded, so only artifacts
    that actually came from a flattened PDF text layer are ever subject to the
    dash-corruption quarantine rule.
    """
    if extracted.extractor == _PDF_EXTRACTOR_PREFIX:
        return SourceContext.FLAT_PDF_TEXT
    return SourceContext.OPERATOR_RAW


def _window_numeric_values(raw_window: str, *, glyph_health: GlyphHealth, source_context: SourceContext) -> list[float]:
    """Every strictly-validated numeric value present in the window text.

    Candidate BOUNDARIES are located by running :data:`_WINDOW_NUMBER_CANDIDATE_RE`
    against the CASEFOLDED window text (which owns the boundary rules for scanning
    running prose -- the strict core never widens its own window), but each matched
    span is then mapped back to the window's ORIGINAL-CASE text (via
    :func:`~carmel.agents.tools.extract.normalize_with_map` and
    :func:`~carmel.agents.tools.extract.raw_span`) before being handed to
    :func:`carmel.services.numeric.parse_numeric_span`. Casefolding is needed to find
    a lowercase-cased repair prefix like ``/c0`` regardless of how the source text
    capitalized it, but casefolding the text handed to the CORE would erase the very
    case distinction the core relies on (e.g. an uppercase ``E`` exponent marker is
    never quarantined as dash-corruption, only a lowercase ``e`` is -- see
    :mod:`carmel.services.numeric`'s dash-corruption quarantine rule). Each candidate
    is validated under the caller-supplied ``source_context`` (see
    :func:`_source_context_for`) with the glyph health of the document being grounded
    against. A candidate the strict core refuses (corrupt shapes like ``0.6e1.0``,
    quarantined bare-exponent tokens in a dash-corrupted PDF document, non-finite
    results) contributes NOTHING -- corrupt text must never corroborate a claim. A
    candidate parsed as a Range contributes both bounds.

    Args:
        raw_window: The evidence window, in its ORIGINAL case (NOT passed through
            ``normalize_for_match``).
        glyph_health: Glyph-corruption context computed from the FULL document text
            (not just the window), via
            :func:`carmel.services.numeric.assess_glyph_health`.
        source_context: Where this window's text originated (see
            :func:`_source_context_for`); only
            :attr:`~carmel.services.numeric.SourceContext.FLAT_PDF_TEXT` is ever
            subject to the dash-corruption quarantine rule.

    Returns:
        The usable numeric values, in window order (Ranges contribute low then high).
    """
    window_norm, index_map = normalize_with_map(raw_window)
    values: list[float] = []
    for m in _WINDOW_NUMBER_CANDIDATE_RE.finditer(window_norm):
        raw_start, raw_end = raw_span(index_map, m.start(), m.end(), len(raw_window))
        candidate = raw_window[raw_start:raw_end]
        if _is_percent_adjacent_exponent(window_norm, m):
            # A percent sign cannot be a blanket boundary-breaker: "50%" is an
            # ordinary percentage and must stay readable. But an EXPONENT-form
            # token touching a '%' is the corrupted-range shape, not a value --
            # "50%H 2e50%CO" is "50% H2 - 50% CO" with the subscript flattened
            # and the en-dash encoded as ASCII 'e'. Emitting 2e50 there invents a
            # magnitude that is nowhere in the paper. Refusing costs only the rare
            # legitimate "1e-3%", and fails closed.
            continue
        result = parse_numeric_span(
            candidate,
            source_context=source_context,
            glyph_health=glyph_health,
        )
        if isinstance(result, Scalar):
            values.append(result.value)
        elif isinstance(result, Range):
            values.append(result.low)
            values.append(result.high)
    return values


def _numeric_isclose(target: float, tok: float) -> bool:
    """Value-equality for a required numeric anchor vs. a window-scanned token.

    Plain ``math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)`` has a sharp edge here:
    the unconditional ``abs_tol=1e-9`` makes ANY value within 1e-9 of zero compare
    equal to zero (``math.isclose(1e-12, 0.0, rel_tol=1e-9, abs_tol=1e-9)`` is
    ``True``), so a claimed value as small as ``1e-12`` would be wrongly
    "corroborated" by a literal ``0`` anywhere in the window. Zero must only ever
    match zero, and two nonzero values are compared by relative tolerance alone (no
    absolute floor at all, so this never re-introduces the same near-zero collapse
    from the other direction).
    """
    if target == 0.0 and tok == 0.0:
        return True
    if target == 0.0 or tok == 0.0:
        return False
    return math.isclose(target, tok, rel_tol=1e-9)


#: A required anchor is either a single literal string, or a tuple of surface-form
#: synonyms of which ANY ONE satisfies the requirement (used for reactor-type terms
#: and "at least one of species/reaction label" groups).
RequiredAnchor = str | tuple[str, ...]

#: Surface forms accepted for each :class:`~carmel.schemas.campaign.ReactorType`
#: value. Matched with word-boundary-aware search (:func:`_term_present_boundary`)
#: rather than plain substring search, because several of these are short
#: abbreviations ("st", "pfr", "rcm") that would otherwise collide with unrelated
#: substrings of ordinary words (e.g. "st" inside "test" or "first").
_REACTOR_TYPE_TERMS: dict[ReactorType, tuple[str, ...]] = {
    ReactorType.SHOCK_TUBE: ("shock tube", "shock-tube", "st"),
    ReactorType.JSR: ("jet-stirred reactor", "jet stirred reactor", "jsr"),
    ReactorType.RCM: ("rapid compression machine", "rcm"),
    ReactorType.PFR: ("plug flow", "plug-flow", "pfr"),
    ReactorType.BATCH: ("batch reactor", "batch"),
    ReactorType.FLAME: ("flame",),
}

#: Surface forms accepted for each :class:`~carmel.schemas.literature.QMProperty`
#: value. ``QMCalculationPayload`` has no raw-text companion field for ``property``
#: (unlike ``observable``/``observable_raw`` on the experimental-benchmark payload),
#: so this table plays the same role as :data:`_REACTOR_TYPE_TERMS`: it maps the
#: controlled-vocabulary enum to the surface forms a paper would actually use.
#: ``QMProperty.OTHER`` is deliberately absent -- it is a catch-all with no verbatim
#: surface form to search for, and requiring the literal word "other" to appear near
#: the quote would spuriously reject genuine findings (spar round 5, P1).
_QM_PROPERTY_TERMS: dict[QMProperty, tuple[str, ...]] = {
    QMProperty.ENTHALPY_OF_FORMATION: ("enthalpy of formation", "heat of formation"),
    QMProperty.ENTROPY: ("entropy",),
    QMProperty.HEAT_CAPACITY: ("heat capacity",),
    QMProperty.RATE_COEFFICIENT: ("rate coefficient", "rate constant"),
    QMProperty.BARRIER_HEIGHT: ("barrier height", "activation barrier", "activation energy"),
    QMProperty.BOND_DISSOCIATION_ENERGY: ("bond dissociation energy", "bde"),
    QMProperty.GEOMETRY: ("geometry",),
    QMProperty.FREQUENCIES: ("frequencies", "vibrational frequencies"),
}

#: Sentence/row/paragraph boundary characters used by :func:`_bounded_window`.
_SENTENCE_BOUNDARY_CHARS = ".!?\n"


def _term_present_boundary(term: str, normalized_window: str) -> bool:
    """True if ``term`` occurs in ``normalized_window`` at a word boundary.

    Plain substring search (as used for numeric/unit anchors elsewhere in this
    module) is unsafe for short reactor-type abbreviations like "st" or "pfr":
    "st" is a substring of "test", "first", "combustion", etc. This instead
    requires that the character immediately before and after the match (if any)
    not be alphanumeric, so "ST" matches "... in a ST facility ..." but not
    "... in the fastest experiment ...".

    Args:
        term: Already-normalized (casefolded) candidate surface form.
        normalized_window: Already-normalized text to search within.

    Returns:
        True if ``term`` occurs in ``normalized_window`` bounded by non-alphanumeric
        characters (or the string edges) on both sides.
    """
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
    return pattern.search(normalized_window) is not None


def _bounded_window(text: str, start: int, end: int, *, fallback: int) -> tuple[int, int]:
    """Return a (window_start, window_end) span around ``[start, end)``.

    A flat character radius (the previous implementation) lets an anchor from a
    *neighbouring* sentence, table row, or paragraph satisfy corroboration for a
    quote it has nothing to do with (spar round N, P1-1). This instead prefers to
    stop the window at the nearest sentence/row/paragraph boundary -- a period,
    exclamation mark, question mark, or newline -- on each side, so the search
    stays confined to the sentence/row/paragraph the quote itself is in.

    A boundary is not always present nearby (e.g. inside one very long unbroken
    sentence, or a table cell with no punctuation), so this falls back to the flat
    ``fallback``-character window on whichever side no boundary is found within
    ``2 * fallback`` characters -- generous enough to reach a real boundary in
    ordinary prose, but bounded so a boundary-free document doesn't degrade to an
    unbounded scan.

    Args:
        text: The raw text to search within.
        start: Start offset of the matched quote (inclusive).
        end: End offset of the matched quote (exclusive).
        fallback: The flat character-radius fallback on each side.

    Returns:
        A ``(window_start, window_end)`` pair of raw-text offsets.
    """
    max_scan = fallback * 2

    lo_limit = max(0, start - max_scan)
    window_start = max(0, start - fallback)
    for idx in range(start - 1, lo_limit - 1, -1):
        if text[idx] in _SENTENCE_BOUNDARY_CHARS:
            window_start = idx + 1
            break

    hi_limit = min(len(text), end + max_scan)
    window_end = min(len(text), end + fallback)
    for idx in range(end, hi_limit):
        if text[idx] in _SENTENCE_BOUNDARY_CHARS:
            window_end = idx + 1
            break

    return window_start, window_end


def check_evidence_spans(
    extracted: ExtractedText,
    match: QuoteMatch,
    required: Sequence[RequiredAnchor],
    *,
    window: int = DEFAULT_EVIDENCE_WINDOW,
) -> list[str]:
    """Return the ``required`` anchors NOT corroborated near the quote.

    The search window is deliberately tight and now sentence/row/paragraph-bounded
    (see :func:`_bounded_window`), not a flat character radius: a flat radius large
    enough to reach a real anchor also reaches unrelated numbers and terms in a
    neighbouring sentence, table row, or paragraph, which would let a fabricated
    combination of true-but-unrelated facts pass as corroborated (spar round N,
    P1-1). The bounded window falls back to the flat ``window``-character radius
    only when no sentence/row/paragraph boundary is found nearby.

    Each item in ``required`` is either a single literal string (must occur
    verbatim/numerically in the window) or a tuple of surface-form synonyms (ANY
    ONE occurring in the window satisfies that requirement) -- used for
    reactor-type terms and "at least one of species/reaction label" groups.
    Tuple-typed anchors are matched with word-boundary-aware search
    (:func:`_term_present_boundary`) rather than plain substring search, to avoid
    short-abbreviation collisions (e.g. "ST" as a substring of "test"). Plain
    string anchors keep the original substring/numeric-literal check.

    Numeric requirements are value-normalized so ``1.0``, ``1``, ``1.00``, and
    ``1e0`` all compare equal (compared with ``math.isclose``), rather than
    requiring an exact string match. BOTH sides of that comparison now go through
    the strict numeric core (:mod:`carmel.services.numeric`): window text is
    tokenized by :func:`_window_numeric_values` (corrupt shapes like ``0.6e1.0``
    contribute nothing, rather than being salvaged as ``6.0``), and a required
    anchor that is numeric in intent (:func:`_numeric_anchor_intent`) but cannot be
    strictly resolved -- ``inf``, ``nan``, corrupt shapes -- is a HARD missing
    anchor with an explanatory suffix in the returned entry, never a fallback
    substring search. Non-numeric string requirements are compared via
    ``normalize_for_match``, which is already case- and whitespace-insensitive.

    Args:
        extracted: The fetched artifact's extracted text.
        match: The located quote, whose raw offsets anchor the window.
        required: The anchors that must be corroborated (see
            :func:`required_spans_for`).
        window: Fallback flat character radius used only when no sentence/row/
            paragraph boundary is found near the quote.

    Returns:
        The subset of ``required`` that could not be corroborated in the window.
        Each tuple-typed entry that fails is reported using its first surface
        form, for a stable, readable ``missing_spans`` value.
    """
    start, end = _bounded_window(extracted.text, match.start, match.end, fallback=window)
    raw_window = extracted.text[start:end]
    window_norm = normalize_for_match(raw_window)
    glyph_health = assess_glyph_health(extracted.text)
    source_context = _source_context_for(extracted)
    window_numbers = _window_numeric_values(raw_window, glyph_health=glyph_health, source_context=source_context)

    missing: list[str] = []
    for req in required:
        if isinstance(req, tuple):
            if not any(_term_present_boundary(normalize_for_match(alt), window_norm) for alt in req):
                missing.append(req[0])
            continue
        if _numeric_anchor_intent(req):
            # The anchor is machine-side text (typically str() of a typed value), not
            # PDF-extracted prose, so it parses under OPERATOR_RAW with its own glyph
            # health -- the document's corruption state must not affect how the
            # REQUIREMENT is read.
            resolved = parse_numeric_span(
                req, source_context=SourceContext.OPERATOR_RAW, glyph_health=assess_glyph_health(req)
            )
            if isinstance(resolved, Unresolvable):
                # Numeric in intent but not strictly resolvable (inf, nan, corrupt
                # shapes): hard missing-anchor result, NEVER a substring search that
                # might coincidentally succeed ('inf' inside 'infinite').
                missing.append(f"{req} (numeric anchor could not be strictly resolved: {resolved.reason})")
                continue
            targets = [resolved.value] if isinstance(resolved, Scalar) else [resolved.low, resolved.high]
            if not all(any(_numeric_isclose(target, tok) for tok in window_numbers) for target in targets):
                missing.append(req)
            continue
        if normalize_for_match(req) not in window_norm:
            missing.append(req)

    return missing


def required_spans_for(payload: FindingPayload) -> list[RequiredAnchor]:
    """The typed values that must be corroborated near the quote, per category.

    Optional payload fields make every quantity trivially omittable: an LLM could
    simply leave out every measured value and otherwise sail through an empty
    requirement list. So each category has a MINIMUM anchor floor (spar round 3,
    P1-15; tightened further in spar round N, P1-1); when the floor is not met
    this returns an **empty list**, and :func:`ground_finding` treats an empty
    return value as a failure (``SPANS_MISSING``) -- never as an automatic pass,
    since there would be nothing left to corroborate.

    **Honesty about the asymmetry.** This check can only demand corroboration for
    fields the payload actually populated. An *empty/unset* optional field (e.g. no
    ``pressure_range_bar`` given at all) is simply not checked -- it neither helps
    nor hurts beyond the category's floor. This means a fabricator who omits a
    field entirely dodges corroboration of that specific field; it does not mean
    the gate has verified the omitted field is unknowable or irrelevant. What this
    function DOES catch is a payload that populates a field with a value
    inconsistent with what the source text actually says near the quote (e.g.
    claiming JSR/1500 K when the source says shock tube/1200 K) -- consistency
    with stated conditions, not proof that every claimed field is true.

    Floors and anchors:
      - ``EXPERIMENTAL_BENCHMARK``: requires at least one ``measured`` Quantity
        AND at least one ``species`` entry; if either is missing the floor is not
        met. When met, EVERY ``measured`` entry's value and unit are required
        anchors (not just the first) -- an earlier version anchored only
        ``measured[0]``, silently letting a fabricator append extra, uncorroborated
        measurements to a finding that otherwise grounds cleanly (spar hardening
        note, P1-4). Likewise, EVERY populated species identifier
        (``species[*].raw_name``) is now its own required anchor, individually --
        an earlier version treated the whole species list as a single
        any-one-suffices anchor, which let a fabricator add extra, uncorroborated
        species alongside one genuine one. Also always requires: the observable as
        printed in the source (``observable_raw`` -- the controlled-vocabulary
        ``observable`` enum is not expected to appear literally in prose); the
        reactor type's surface form (any synonym from :data:`_REACTOR_TYPE_TERMS`,
        since ``reactor_type`` is always set); and, for each of
        ``temperature_range_K``, ``pressure_range_bar``, and
        ``equivalence_ratio_range`` that is populated, BOTH bounds of that range as
        separate anchors. ``measured`` and ``species`` are additionally
        length-capped at the schema level (``max_length=8`` / ``max_length=20``,
        see :class:`~carmel.schemas.literature.ExperimentalBenchmarkPayload`) so
        this per-entry anchoring can't be used to construct a pathologically large
        finding. (Spar round 5, P1) When populated, ``residence_time_s``,
        ``apparatus``, and ``n_data_points`` are ALSO required anchors -- an earlier
        version omitted these three populated fields entirely, which let an LLM
        fabricate a residence time, an apparatus name, or a data-point count freely
        with no corroboration at all.
      - ``QM_CALCULATION``: requires ``level_of_theory`` and its result value and
        unit (all always-set fields), AND at least one of ``species`` or
        ``reaction_label``; if neither is populated the floor is not met. (Spar
        round 5, P1) The ``property`` field is now also anchored via a surface-form
        synonym table (:data:`_QM_PROPERTY_TERMS`), analogous to
        ``_REACTOR_TYPE_TERMS`` -- ``property`` has no raw-text companion field like
        ``observable_raw``, so the controlled-vocabulary enum value is mapped to the
        surface forms a paper would actually use (e.g. ``BOND_DISSOCIATION_ENERGY``
        -> "bond dissociation energy"/"bde"). ``QMProperty.OTHER`` is deliberately
        exempted: it is a catch-all with no verbatim form to search for, and
        demanding the literal word "other" appear near the quote would spuriously
        reject genuine findings. ``software`` is also now a required anchor when
        populated.
      - ``PRIOR_MODEL``: requires ``model_name``, plus at least one of
        ``{n_species, n_reactions, mechanism_url}``; if none of those three is
        populated the floor is not met. (Spar round 5, P1) When populated,
        ``fuel_species`` (each entry individually, like ``species`` above) and
        ``validation_targets`` (each entry individually) are ALSO required anchors
        -- these are factual/identifying claims (which species the model covers,
        which datasets it was validated against), not narrative. ``conditions_note``
        is deliberately NOT anchored: it is free-text commentary rather than a
        discrete, verifiable claim, so requiring it to appear verbatim near the
        quote would reject genuine findings whose note is a paraphrase/summary
        rather than a quotable fact.

    Args:
        payload: The finding's typed payload.

    Returns:
        The required anchors to corroborate. An empty list means the category's
        minimum-anchor floor was not met by this payload.
    """
    if isinstance(payload, ExperimentalBenchmarkPayload):
        if not payload.measured or not payload.species:
            return []
        required: list[RequiredAnchor] = [
            payload.observable_raw,
            tuple(_REACTOR_TYPE_TERMS[payload.reactor_type]),
        ]
        for q in payload.measured:
            required.append(str(q.value))
            required.append(q.unit)
        for s in payload.species:
            required.append(s.raw_name)
        for bound in (
            payload.temperature_range_K,
            payload.pressure_range_bar,
            payload.equivalence_ratio_range,
        ):
            if bound is not None:
                required.append(str(bound[0]))
                required.append(str(bound[1]))
        if payload.residence_time_s is not None:
            required.append(str(payload.residence_time_s))
        if payload.apparatus:
            required.append(payload.apparatus)
        if payload.n_data_points is not None:
            required.append(str(payload.n_data_points))
        return required

    if isinstance(payload, QMCalculationPayload):
        species_or_reaction = tuple(s.raw_name for s in payload.species) + (
            (payload.reaction_label,) if payload.reaction_label else ()
        )
        if not species_or_reaction:
            return []
        required = [
            payload.level_of_theory,
            str(payload.value.value),
            payload.value.unit,
            species_or_reaction,
        ]
        property_terms = _QM_PROPERTY_TERMS.get(payload.property)
        if property_terms is not None:
            required.append(property_terms)
        if payload.software:
            required.append(payload.software)
        return required

    if isinstance(payload, PriorModelPayload):
        extra: list[str] = []
        if payload.n_species is not None:
            extra.append(str(payload.n_species))
        if payload.n_reactions is not None:
            extra.append(str(payload.n_reactions))
        if payload.mechanism_url:
            extra.append(payload.mechanism_url)
        if not extra:
            return []
        required = [payload.model_name, *extra]
        for s in payload.fuel_species:
            required.append(s.raw_name)
        for target in payload.validation_targets:
            required.append(target)
        return required

    raise UnsupportedFindingPayloadError(f"unsupported finding payload type: {type(payload)!r}")


def _bibliography_region_confidence(extracted: ExtractedText, position: int) -> str | None:
    """Classify ``position`` against structurally-detected bibliography-like regions.

    Args:
        extracted: The fetched artifact's extracted text.
        position: A raw-text offset (typically a quote match's start).

    Returns:
        ``"confident"`` if ``position`` falls inside a region
        :func:`~carmel.agents.tools.extract.find_bibliography_like_regions` flagged as
        confident, ``"suggestive"`` if inside a non-confident region, else ``None``.
    """
    for start, end, confident in find_bibliography_like_regions(extracted.text):
        if start <= position < end:
            return "confident" if confident else "suggestive"
    return None


def ground_finding(
    *, payload: FindingPayload, citation: Citation, quote: str, extracted: ExtractedText | None
) -> GroundingVerdict:
    """The grounding gate: decide whether a proposed finding is corroborated.

    Evaluated in this exact order, short-circuiting at the first applicable status:
    ``NO_ARTIFACT`` -> ``ARTIFACT_UNREADABLE`` -> ``ARTIFACT_DEGRADED`` ->
    ``QUOTE_NOT_FOUND`` -> ``REFERENCES_ONLY`` -> ``IDENTITY_MISMATCH`` ->
    ``SPANS_MISSING`` -> ``GROUNDED_FUZZY`` / ``GROUNDED_EXACT``. ``grounded`` on
    the returned verdict is True only for the two ``GROUNDED_*`` statuses.

    This is a *first filter* against fabricated quotes and misattributed sources —
    see the module docstring. It is not, and cannot be, a guarantee that the
    underlying scientific claim is true.

    Args:
        payload: The finding's typed payload.
        citation: The citation the finding claims to come from.
        quote: The claimed verbatim quote.
        extracted: The fetched artifact's extracted text, or ``None`` if no
            artifact was fetched at all.

    Returns:
        A :class:`GroundingVerdict` with a specific, human-readable ``reasons``
        list explaining the outcome (this text lands in a decision log and is how a
        researcher understands a rejection).
    """
    if extracted is None:
        return GroundingVerdict(
            status=GroundingStatus.NO_ARTIFACT,
            grounded=False,
            match_ratio=0.0,
            identity_ok=False,
            missing_spans=[],
            reasons=["No artifact was fetched for this finding, so the claimed quote cannot be checked at all."],
        )

    match, miss_reason = find_quote_with_reason(extracted, quote)
    if match is None:
        # A quote that did not match has two very different causes. Check whether the
        # artifact was readable at all BEFORE attributing the failure to the agent --
        # blaming our own extraction damage on a fabricating model is both wrong and
        # unactionable. Still rejected either way; only the diagnosis differs.
        damage = unreadable_reason(extracted)
        if damage is not None:
            return GroundingVerdict(
                status=GroundingStatus.ARTIFACT_UNREADABLE,
                grounded=False,
                match_ratio=0.0,
                identity_ok=False,
                missing_spans=[],
                reasons=[
                    damage,
                    "The quote therefore could not be checked; this is an extraction failure "
                    "on our side, NOT evidence that the quote was fabricated.",
                ],
            )
        return GroundingVerdict(
            status=GroundingStatus.QUOTE_NOT_FOUND,
            grounded=False,
            match_ratio=0.0,
            identity_ok=False,
            missing_spans=[],
            reasons=[
                _QUOTE_MISS_EXPLANATIONS[miss_reason] if miss_reason is not None else "the quote was not located.",
                "The claimed verbatim quote could not be located in the fetched artifact text, "
                "exactly or via fuzzy matching; this looks like a fabricated quote.",
            ],
        )

    if extracted.lossy and not extracted.sections:
        # extracted.lossy is True both for acceptable truncation (sections are still
        # retained, so structural checks like references-detection keep working) AND
        # for a degraded reload where the sections list itself came back empty (spar
        # hardening note, P1-3). The latter is dangerous, not just lossy: e.g.
        # evidence.load_artifact_text's degraded reload path returns lossy=True,
        # sections=[] when extracted.json is missing, and downstream structural
        # checks (references-section detection in particular) silently behave as if
        # there were no references section at all -- turning what should be a
        # REFERENCES_ONLY rejection into a false GROUNDED_EXACT pass. Checked here,
        # AFTER a quote was actually located (a genuinely unreadable artifact was
        # already ruled out above via unreadable_reason returning None for
        # find_quote to have succeeded at all) and BEFORE the references-section
        # check below, which is exactly the structural check this gap would defeat.
        # Fail CLOSED whenever sections are unavailable on a lossy reload, rather
        # than silently trusting a flat, unstructured text blob for a fail-open
        # structural check.
        return GroundingVerdict(
            status=GroundingStatus.ARTIFACT_DEGRADED,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=False,
            missing_spans=[],
            reasons=[
                "The artifact's extracted text was reloaded in a degraded, structure-free form "
                "(no sections available), so structural checks such as references-section "
                "detection cannot run reliably. Rejected rather than risk a false pass."
            ],
        )

    if match.section_label == "references":
        return GroundingVerdict(
            status=GroundingStatus.REFERENCES_ONLY,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=False,
            missing_spans=[],
            reasons=[
                "The quote text was found only inside the artifact's references/bibliography "
                "section, not its body — this reads as a citation mention, not corroborated content."
            ],
        )

    bib_confidence = _bibliography_region_confidence(extracted, match.start)
    if bib_confidence == "confident":
        return GroundingVerdict(
            status=GroundingStatus.REFERENCES_ONLY,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=False,
            missing_spans=[],
            reasons=[
                "The quote falls inside a run of text that is structurally dense with citation "
                "patterns (author-initial forms, parenthesized years, 'et al.', volume:page ranges, "
                "'doi:') even though no references heading was detected — this reads as an unlabelled "
                "bibliography, not corroborated body content."
            ],
        )

    if not check_identity(extracted, citation):
        if citation.doi:
            reason = (
                f"Citation DOI '{citation.doi}' either does not appear verbatim in the fetched "
                "artifact text, or the citation's title and first author's surname could not be "
                "confirmed outside any references section; a DOI mention alone does not establish "
                "that this artifact is the cited work (e.g. a review article discussing another "
                "paper's DOI in its body text)."
            )
        else:
            reason = (
                "The citation's title, first author's surname, and/or year could not all be "
                "confirmed in the artifact body (outside any references section); this looks like "
                "a misattributed source."
            )
        return GroundingVerdict(
            status=GroundingStatus.IDENTITY_MISMATCH,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=False,
            missing_spans=[],
            reasons=[reason],
        )

    try:
        required = required_spans_for(payload)
    except UnsupportedFindingPayloadError as exc:
        # Fail CLOSED on a payload type required_spans_for has no anchor rule for --
        # never treat "we don't know how to check this" as "there is nothing to
        # check" (which SPANS_MISSING with an empty `required` list would otherwise
        # silently degrade into further down).
        return GroundingVerdict(
            status=GroundingStatus.SPANS_MISSING,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=True,
            missing_spans=[],
            reasons=[
                f"This finding's payload type has no known required-anchor rule ({exc}); "
                "cannot verify corroboration, so it is rejected rather than passed."
            ],
        )
    if not required:
        return GroundingVerdict(
            status=GroundingStatus.SPANS_MISSING,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=True,
            missing_spans=[],
            reasons=[
                "This finding's category requires a minimum set of corroborating anchors "
                "(e.g. a measured value and unit), and none were specified; an "
                "under-specified finding cannot be grounded."
            ],
        )

    missing = check_evidence_spans(extracted, match, required)
    if missing:
        return GroundingVerdict(
            status=GroundingStatus.SPANS_MISSING,
            grounded=False,
            match_ratio=match.ratio,
            identity_ok=True,
            missing_spans=missing,
            reasons=[
                f"The following required anchors were not corroborated within "
                f"{DEFAULT_EVIDENCE_WINDOW} characters of the quote: {', '.join(missing)}."
            ],
        )

    status = GroundingStatus.GROUNDED_EXACT if match.exact else GroundingStatus.GROUNDED_FUZZY
    reasons = [
        "Quote and required anchors were corroborated by the fetched artifact text."
        if match.exact
        else f"Quote matched via fuzzy fallback (ratio={match.ratio:.2f}); required anchors were corroborated."
    ]

    if bib_confidence == "suggestive":
        reasons.append(
            "WARNING: this quote falls near a run of text that is somewhat dense with citation "
            "patterns (author-initial forms, parenthesized years, 'et al.', volume:page ranges, "
            "'doi:'), though not dense enough to be treated as a confirmed unlabelled "
            "bibliography. Manual review of this match's surrounding text is recommended."
        )

    has_references_section = any(sec.label == "references" for sec in extracted.sections)
    doc_len = len(extracted.text)
    if not has_references_section and doc_len > 0 and match.start >= doc_len * _TAIL_FRACTION:
        reasons.append(
            f"WARNING: this quote falls in the last {(1 - _TAIL_FRACTION) * 100:.0f}% of the "
            "document and no labelled references section was detected. extract.py's "
            "references-heading detection is fail-open (it only fires when the heading occupies "
            "its own line), so this document may have an unlabeled reference list and this "
            "match's position cannot be fully trusted as body text."
        )

    return GroundingVerdict(
        status=status,
        grounded=True,
        match_ratio=match.ratio,
        identity_ok=True,
        missing_spans=[],
        reasons=reasons,
    )
