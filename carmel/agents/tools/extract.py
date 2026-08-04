"""Text extraction from fetched documents, with offset-preserving section labels.

``extract_text`` is the single entry point: it dispatches on ``content_type`` to a
PDF extractor (optional ``pypdf`` dependency, lazily imported), a stdlib-only HTML tag
stripper, or a verbatim passthrough for ``text/*``. All offsets recorded on
:class:`TextSection` (and, downstream, on ``QuoteMatch`` in
``carmel.services.grounding``) are indices into :attr:`ExtractedText.text` — the RAW
extracted text — never into :attr:`ExtractedText.normalized`. ``normalize_for_match``
is not offset-preserving (ligature expansion and hyphen-joining change the length), so
any code that locates a match in ``normalized`` must map the match back onto ``text``
before recording an offset.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Iterator
from functools import lru_cache
from html.parser import HTMLParser

from pydantic import BaseModel, ConfigDict

# Compatibility-decomposition already folds these under NFKC, but they are expanded
# explicitly too so the behaviour does not depend on a particular unicodedata version.
_LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# A hyphen at the end of a line, followed by (optional blank) whitespace and a
# continuation word: "combus-\ntion" -> "combustion". Only fires between word
# characters so a real end-of-sentence hyphen followed by a new paragraph is untouched.
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# A references/bibliography heading, tolerant of two lossy-extraction artifacts:
#   - a leading section number ("8. References", "VI REFERENCES", "3) Bibliography"),
#     matched by the optional (?:(?:\d+|[ivxlcdm]+)[.\)]?[ \t]+)? prefix group; and
#   - the heading running directly into the following body/citation text on the same
#     line ("References Smith, J. ...") instead of occupying its own line, because
#     some PDF extractors drop the line break after a heading.
# For the "runs into following text" case we still require *some* signal that we are
# looking at a heading rather than a body sentence that happens to start with one of
# these words ("References to prior work suggest..."): the heading word (+ optional
# colon/whitespace) must be followed by either end-of-line or an uppercase letter,
# which is the overwhelmingly common shape of a citation-list entry start ("Smith,
# J...") and essentially never how a body sentence continues in lowercase prose.
_REFERENCES_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+|[ivxlcdm]+)[.\)]?[ \t]+)?"
    r"(references|bibliography|works cited|literature cited)[ \t]*:?[ \t]*(?=$|[A-Z])"
)
_ABSTRACT_HEADING_RE = re.compile(r"(?im)^[ \t]*abstract[ \t]*:?[ \t]*$")

# Only treat a references/bibliography heading as the trailing section when it starts
# at or past this fraction of the document; a heading that appears earlier is more
# likely a body subsection (e.g. a "prior literature" discussion) than the closing list.
_REFERENCES_MIN_FRACTION = 0.3

# --- Structural (headingless) bibliography-region detection --------------------
#
# Heading-based detection above is fail-open: it does nothing at all when a lossy
# extraction drops or garbles the heading entirely. Bibliography entries have a
# distinctive lexical signature independent of any heading — author-initial patterns
# ("Smith, J."), a parenthesized year, "et al.", volume:page ranges, and "doi:"
# strings — that ordinary prose essentially never reproduces at this density. A run
# of lines dense in these patterns is therefore treated as "bibliography-like" even
# with no heading at all; see :func:`find_bibliography_like_regions`.
_BIB_AUTHOR_INITIAL_RE = re.compile(r"[A-Z][A-Za-z'-]+,\s*[A-Z]\.")
_BIB_YEAR_PAREN_RE = re.compile(r"\(\d{4}[a-z]?\)")
_BIB_ET_AL_RE = re.compile(r"\bet al\.?\b", re.IGNORECASE)
_BIB_VOLUME_PAGE_RE = re.compile(r"\b\d{1,4}\s*[:,]\s*\d{1,5}(?:[-–]\d{1,5})?\b")
_BIB_DOI_RE = re.compile(r"\bdoi\s*:?\s*10\.\d{4,9}/", re.IGNORECASE)
_BIB_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    _BIB_AUTHOR_INITIAL_RE,
    _BIB_YEAR_PAREN_RE,
    _BIB_ET_AL_RE,
    _BIB_VOLUME_PAGE_RE,
    _BIB_DOI_RE,
)

#: Sliding-window size (in non-blank lines) used to find dense citation runs.
_BIB_WINDOW_LINES = 5
#: Minimum fraction of citation-pattern lines within a window to call it "dense".
_BIB_WINDOW_DENSITY = 0.6
#: A dense run at least this many lines long, at this density, is "confident"
#: enough to be treated the same as an explicit references section rather than
#: merely producing a warning.
_BIB_CONFIDENT_LINES = 8
_BIB_CONFIDENT_DENSITY = 0.7
_ABSTRACT_SEARCH_WINDOW = 8000
_ABSTRACT_MAX_LEN = 2000

#: Hard cap on extracted-text length, in characters, applied uniformly across every
#: extractor path (PDF, HTML, plain text) before normalization or section-labeling.
#: Real papers in this corpus run roughly 16k-50k characters; the largest legitimate
#: documents are supplementary-information PDFs, which we've seen run up to the
#: low hundreds of thousands of characters. 500k gives generous headroom over that
#: while staying far below the multi-gigabyte-RSS blowup an attacker-controlled
#: document (e.g. a PDF compression bomb, or simply the largest byte payload the
#: 25 MB fetch cap allows through) would otherwise force through
#: ``normalize_with_map``, which allocates several per-character lists per call and
#: was measured to peak at ~381 MB RSS (76x) for a 5.0 MB input.
MAX_EXTRACTED_TEXT_CHARS = 500_000

#: Hard cap on the number of PDF pages whose ``extract_text()`` we will ever call,
#: independent of :data:`MAX_EXTRACTED_TEXT_CHARS`. The character cap alone is not
#: enough: it only bounds the RETURNED text, and if it were checked only after every
#: page had been materialized, an attacker-controlled PDF with an enormous page count
#: (each page cheap on its own) could still force pypdf to walk the whole document --
#: and, per-page, each ``extract_text()`` call allocates independently -- before the
#: cap ever applied. Real papers run a handful to a few dozen pages; the largest
#: legitimate case seen (supplementary-information PDFs) runs to a few hundred. 2000
#: is generous headroom over that while bounding the worst case.
MAX_PDF_PAGES = 2000


class TextSection(BaseModel):
    """A labeled span of :class:`ExtractedText.text`.

    Attributes:
        label: One of "body", "references", "abstract", "caption", "table".
        start: Start offset into ``ExtractedText.text`` (inclusive).
        end: End offset into ``ExtractedText.text`` (exclusive).
        page: 1-indexed PDF page number, or None for non-paginated sources.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    start: int
    end: int
    page: int | None = None


class PageExtractionFailure(BaseModel):
    """Records that a single PDF page's ``extract_text()`` call raised.

    A page-level failure must never discard the pages around it (see the
    per-page ``try`` in :func:`_extract_pdf`): the surviving pages are kept,
    and this record is the audit trail for the page that was skipped.

    Attributes:
        page: 1-indexed PDF page number that failed.
        error: A short, redacted description of the exception (type name plus
            a path-scrubbed message). Never contains filesystem paths -- the
            input PDF bytes come from content-addressed storage, and a raw
            exception message could otherwise leak a local path into a stored
            artifact.
    """

    model_config = ConfigDict(extra="forbid")

    page: int
    error: str


class ExtractedText(BaseModel):
    """The result of extracting text from a fetched document.

    Attributes:
        text: Raw extracted text. All offsets elsewhere (``TextSection``, and
            downstream ``QuoteMatch``) index into THIS string.
        normalized: ``normalize_for_match(text)``. Not offset-aligned with ``text``:
            ligature expansion and hyphen-joining change string length, so no
            position in ``normalized`` may be assumed to correspond to the same
            position in ``text``. The raw-index map that produced this string
            (see :func:`normalize_with_map`) is deliberately NOT persisted here —
            it would bloat every stored artifact — and must be recomputed on
            demand via ``normalize_with_map(text)`` wherever it's needed.
        sections: Labeled spans covering (not necessarily contiguously) ``text``.
        page_count: Number of PDF pages, or None for non-paginated sources.
        extractor: Which extractor produced this: "pdf:pypdf", "pdf:unavailable",
            "html", "text", or "unknown".
        lossy: True when extraction is known to have dropped or approximated content
            (missing optional dependency, unsupported content type, a parse error,
            or one or more individual PDF pages failing -- see ``page_failures``).
        page_failures: PDF pages whose ``extract_text()`` raised and were skipped.
            Always empty for non-PDF extractors. When every attempted page fails,
            ``text`` and ``sections`` end up empty (the same "total failure" shape
            downstream consumers already fail closed on), but this field still
            carries WHY, rather than the bare empty result -- an already-loud
            refusal is strictly better with a reason attached than without.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    normalized: str
    sections: list[TextSection]
    page_count: int | None = None
    extractor: str
    lossy: bool = False
    page_failures: tuple[PageExtractionFailure, ...] = ()


def normalize_for_match(s: str) -> str:
    """Normalize text for robust substring/fuzzy matching.

    Steps, in order (order matters: de-hyphenation must run before whitespace
    collapse, or the line break that identifies a hyphenated word-split is gone and a
    genuine mid-sentence hyphen becomes indistinguishable from one introduced by line
    wrapping):

    1. NFKC-normalize.
    2. Expand common PDF ligatures (fi, fl, ff, ffi, ffl) to their plain letters.
    3. Join words that were hyphen-split across a line break (``combus-\\ntion`` ->
       ``combustion``).
    4. Collapse every run of whitespace to a single space.
    5. Casefold.
    6. Strip leading/trailing whitespace.

    Args:
        s: Raw text to normalize.

    Returns:
        The normalized string, suitable for exact or fuzzy substring matching.
    """
    return normalize_with_map(s)[0]


def _char_map_transform(
    chars: list[str], idx_map: list[int], transform: Callable[[str], str]
) -> tuple[list[str], list[int]]:
    """Apply a per-character string transform while carrying a raw-index map."""
    out_chars: list[str] = []
    out_map: list[int] = []
    for ch, ridx in zip(chars, idx_map, strict=True):
        for out_ch in transform(ch):
            out_chars.append(out_ch)
            out_map.append(ridx)
    return out_chars, out_map


def _regex_sub_with_map(
    pattern: re.Pattern[str],
    repl_fn: Callable[[re.Match[str], list[int]], tuple[str, list[int]]],
    s: str,
    idx_map: list[int],
) -> tuple[str, list[int]]:
    """Regex-substitute over ``s`` while carrying ``idx_map`` through the rewrite."""
    out_chars: list[str] = []
    out_map: list[int] = []
    pos = 0
    for m in pattern.finditer(s):
        out_chars.append(s[pos : m.start()])
        out_map.extend(idx_map[pos : m.start()])
        rep_str, rep_map = repl_fn(m, idx_map)
        out_chars.append(rep_str)
        out_map.extend(rep_map)
        pos = m.end()
    out_chars.append(s[pos:])
    out_map.extend(idx_map[pos:])
    return "".join(out_chars), out_map


def _hyphen_repl(m: re.Match[str], idx_map: list[int]) -> tuple[str, list[int]]:
    rep = m.group(1) + m.group(2)
    rep_map = [idx_map[m.start(1)], idx_map[m.start(2)]]
    return rep, rep_map


def _whitespace_repl(m: re.Match[str], idx_map: list[int]) -> tuple[str, list[int]]:
    return " ", [idx_map[m.start()]]


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Like :func:`normalize_for_match`, but also returns a raw-index map.

    Threads an index array through every transform step (NFKC per-character,
    ligature replacement, hyphen-linebreak join, whitespace collapse, casefold,
    strip), so the result is the same normalized string ``normalize_for_match``
    produces, plus a same-length ``index_map`` where ``index_map[i]`` is the
    index into the ORIGINAL string ``s`` that produced normalized character
    ``i`` (best-effort for many-to-one collapses: a whitespace run or a
    hyphen-linebreak join maps its single output character to the LEFT/first
    source character it came from).

    This is the single source of truth for the normalization algorithm;
    :func:`normalize_for_match` is defined in terms of this function's first
    return value, so the two can never diverge. Callers needing to translate a
    span of ``normalize_for_match(s)`` back into raw ``s``-space should use this
    function together with :func:`raw_span`.

    Note: this map is deliberately not persisted anywhere (e.g. on
    ``ExtractedText``) — it must be recomputed on demand from the raw text. The
    small cache below does not weaken that: it is keyed on the exact raw text, so
    a cached map can never belong to different text than the caller passed. What
    it removes is only repetition — one ``check_identity`` normalizes the same
    document up to five times, and ``find_quote_with_reason`` again after it, each
    a full pass over the whole document (F10).

    A fresh copy of the index map is returned each call. The map is a mutable list
    and the cache is shared, so handing out the cached object would let one caller's
    edit silently rewrite every later caller's mapping. Copying is O(n) against a
    recomputation that is several passes of O(n) with per-character work.

    Args:
        s: Raw text to normalize.

    Returns:
        A ``(normalized, index_map)`` pair, where ``len(index_map) ==
        len(normalized)``.
    """
    normalized, index_map = _normalize_with_map_cached(s)
    return normalized, list(index_map)


@lru_cache(maxsize=8)
def _normalize_with_map_cached(s: str) -> tuple[str, tuple[int, ...]]:
    """Memoized core of :func:`normalize_with_map`.

    Bounded at 8 entries: the access pattern is repeated calls against the SAME
    document (several per finding), so a tiny cache captures effectively all of the
    reuse while keeping worst-case retention to a handful of documents rather than
    every artifact a long corpus pass touches. Returns an immutable map so a cached
    entry cannot be mutated in place by a caller.
    """
    chars = list(s)
    idx_map = list(range(len(s)))

    chars, idx_map = _char_map_transform(chars, idx_map, lambda ch: unicodedata.normalize("NFKC", ch))
    chars, idx_map = _char_map_transform(chars, idx_map, lambda ch: _LIGATURES.get(ch, ch))

    result = "".join(chars)
    result, idx_map = _regex_sub_with_map(_HYPHEN_LINEBREAK_RE, _hyphen_repl, result, idx_map)
    result, idx_map = _regex_sub_with_map(_WHITESPACE_RUN_RE, _whitespace_repl, result, idx_map)

    chars, idx_map = _char_map_transform(list(result), idx_map, lambda ch: ch.casefold())
    result = "".join(chars)

    lstrip_n = len(result) - len(result.lstrip())
    rstrip_n = len(result) - len(result.rstrip())
    end = len(result) - rstrip_n
    result = result[lstrip_n:end]
    idx_map = idx_map[lstrip_n:end]
    return result, tuple(idx_map)


def raw_span(index_map: list[int], start: int, end: int, raw_len: int) -> tuple[int, int]:
    """Map a ``[start, end)`` span in normalized-space back to raw-space.

    Args:
        index_map: The index map returned by :func:`normalize_with_map` for the
            same raw string whose length is ``raw_len``.
        start: Start offset into the normalized string (inclusive).
        end: End offset into the normalized string (exclusive).
        raw_len: ``len`` of the original raw string that produced ``index_map``
            (needed because ``end`` may equal ``len(index_map)``, past the last
            mapped entry, and the raw end-of-text position isn't otherwise
            recoverable from the map alone).

    Returns:
        The corresponding ``(raw_start, raw_end)`` span into the raw string.

    This is an approximation by nature wherever the normalized span's edges
    fall on a character produced by a many-to-one collapse (a whitespace run
    collapsed to one space, or a hyphen-linebreak join): ``index_map`` only
    records the LEFT/first source character for such a collapsed output
    character. Concretely:

    - ``start``: resolved to ``index_map[start]`` — the left edge of whatever
      raw span produced normalized character ``start``. For a span starting on
      a collapsed whitespace run, this lands at the first character of that
      run, not the exact raw offset "equivalent" to the single collapsed space.
    - ``end``: resolved to ``index_map[end - 1] + 1`` — one past the raw source
      of the LAST normalized character in the span (i.e. the right edge of that
      character's raw span, again the left/first source character's position
      plus one for a collapsed run). If ``end == len(index_map)`` (the span
      runs to the end of the normalized string), ``raw_end`` is ``raw_len``
      instead, since there is no ``index_map[end]`` entry to consult.

    Empty spans (``start == end``) return ``(raw_start, raw_start)`` with
    ``raw_start`` resolved as above (or ``raw_len`` if ``start == len(index_map)``).
    """
    n = len(index_map)
    raw_start = raw_len if start >= n else index_map[max(start, 0)]

    if end <= start:
        return raw_start, raw_start

    raw_end = raw_len if end >= n else index_map[end - 1] + 1

    return raw_start, raw_end


def _find_abstract_region(text: str) -> tuple[int, int] | None:
    """Locate a leading "Abstract" heading and the paragraph that follows it."""
    m = _ABSTRACT_HEADING_RE.search(text[:_ABSTRACT_SEARCH_WINDOW])
    if not m:
        return None
    start = m.end()
    while start < len(text) and text[start] in "\n\r \t":
        start += 1
    window_end = min(start + _ABSTRACT_MAX_LEN, len(text))
    blank = re.search(r"\n\s*\n", text[start:window_end])
    end = start + blank.start() if blank else window_end
    if end <= start:
        return None
    return start, end


def _is_citation_line(line: str) -> bool:
    """True if ``line`` matches at least one citation-pattern regex."""
    return any(pattern.search(line) for pattern in _BIB_LINE_PATTERNS)


def find_bibliography_like_regions(text: str) -> list[tuple[int, int, bool]]:
    """Find runs of ``text`` that look structurally like a bibliography, whether or
    not they sit under a recognized "References" heading.

    Bibliography entries have a distinctive lexical signature independent of any
    heading: author-initial patterns ("Smith, J."), a parenthesized year, "et al.",
    volume:page ranges, and "doi:" strings, all packed at high density line after
    line — a density ordinary prose essentially never reaches. This scans
    non-blank lines with a sliding window (:data:`_BIB_WINDOW_LINES`), flags a line
    as part of a dense run when at least :data:`_BIB_WINDOW_DENSITY` of the lines in
    its window match a citation pattern, and merges contiguous flagged lines into
    regions.

    Args:
        text: Raw document text (same string ``TextSection`` offsets index into).

    Returns:
        A list of ``(start, end, confident)`` tuples (raw offsets into ``text``,
        ``end`` exclusive), one per detected region, in document order.
        ``confident`` is True when the run is long and dense enough
        (:data:`_BIB_CONFIDENT_LINES`, :data:`_BIB_CONFIDENT_DENSITY`) to be treated
        with the same weight as an explicit references heading; otherwise it is
        merely suggestive and callers should attach a warning rather than an outright
        rejection.
    """
    # Build a list of (line_start_offset, line_end_offset, is_citation_line) for
    # every non-blank line, skipping blank lines entirely so they don't dilute the
    # density of an otherwise-dense run that happens to have blank-line spacing.
    lines: list[tuple[int, int, bool]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos
        line_end = pos + len(line.rstrip("\r\n"))
        pos += len(line)
        if not stripped:
            continue
        lines.append((line_start, line_end, _is_citation_line(stripped)))

    n = len(lines)
    if n == 0:
        return []

    is_dense = [False] * n
    for i in range(n):
        lo = max(0, i - _BIB_WINDOW_LINES + 1)
        window = lines[lo : i + 1]
        hits = sum(1 for _, _, is_cite in window if is_cite)
        if hits / len(window) >= _BIB_WINDOW_DENSITY:
            is_dense[i] = True

    regions: list[tuple[int, int, bool]] = []
    i = 0
    while i < n:
        if not is_dense[i]:
            i += 1
            continue
        j = i
        while j < n and is_dense[j]:
            j += 1
        run = lines[i:j]
        run_start = run[0][0]
        run_end = run[-1][1]
        run_hits = sum(1 for _, _, is_cite in run if is_cite)
        run_density = run_hits / len(run)
        confident = len(run) >= _BIB_CONFIDENT_LINES and run_density >= _BIB_CONFIDENT_DENSITY
        regions.append((run_start, run_end, confident))
        i = j
    return regions


def _find_references_region(text: str) -> tuple[int, int] | None:
    """Locate the LAST references/bibliography heading, if it starts late enough in
    the document to plausibly be the closing reference list rather than a body
    subsection."""
    matches = list(_REFERENCES_HEADING_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    if len(text) > 0 and m.start() < len(text) * _REFERENCES_MIN_FRACTION:
        return None
    return m.start(), len(text)


def _overlay_region(
    pieces: list[tuple[int, int, str]], region: tuple[int, int], label: str
) -> list[tuple[int, int, str]]:
    """Relabel the portion of ``pieces`` (currently labeled "body") that overlaps
    ``region`` as ``label``, splitting pieces at the region boundary as needed."""
    region_start, region_end = region
    new_pieces: list[tuple[int, int, str]] = []
    for start, end, existing_label in pieces:
        if existing_label != "body":
            new_pieces.append((start, end, existing_label))
            continue
        overlap_start, overlap_end = max(start, region_start), min(end, region_end)
        if overlap_start >= overlap_end:
            new_pieces.append((start, end, existing_label))
            continue
        if start < overlap_start:
            new_pieces.append((start, overlap_start, existing_label))
        new_pieces.append((overlap_start, overlap_end, label))
        if overlap_end < end:
            new_pieces.append((overlap_end, end, existing_label))
    return new_pieces


def _label_special_sections(text: str, sections: list[TextSection]) -> list[TextSection]:
    """Overlay abstract/references regions onto an initial list of "body" sections.

    Each input section (e.g. one per PDF page) is split as needed so the abstract
    and/or trailing references region is carved out as its own section while the
    remainder keeps its original page number and "body" label.
    """
    abstract_region = _find_abstract_region(text)
    references_region = _find_references_region(text)
    if abstract_region is None and references_region is None:
        return sections

    result: list[TextSection] = []
    for section in sections:
        pieces: list[tuple[int, int, str]] = [(section.start, section.end, "body")]
        if abstract_region is not None:
            pieces = _overlay_region(pieces, abstract_region, "abstract")
        if references_region is not None:
            pieces = _overlay_region(pieces, references_region, "references")
        for start, end, label in pieces:
            if start < end:
                result.append(TextSection(label=label, start=start, end=end, page=section.page))
    return result


def _cap_text(text: str, sections: list[TextSection]) -> tuple[str, list[TextSection], bool]:
    """Truncate ``text`` to :data:`MAX_EXTRACTED_TEXT_CHARS` characters if needed.

    Applied uniformly by every extractor path (PDF, HTML, plain text) before any
    normalization or special-section labeling runs, so no downstream step ever
    processes more than the cap's worth of text and no returned section can point
    past the (possibly truncated) end of ``text``.

    Args:
        text: Raw extracted text, prior to normalization or special-section
            labeling.
        sections: Sections already computed against the untruncated ``text``
            (e.g. one per PDF page).

    Returns:
        A ``(text, sections, truncated)`` triple: ``text`` truncated to the cap
        (identical to the input when already within it), ``sections`` clipped to
        the truncated length (a section entirely past the cap is dropped; a
        section straddling the cap has its ``end`` clipped to it), and
        ``truncated`` is True iff truncation actually occurred.
    """
    if len(text) <= MAX_EXTRACTED_TEXT_CHARS:
        return text, sections, False
    capped = text[:MAX_EXTRACTED_TEXT_CHARS]
    clipped_sections: list[TextSection] = []
    for section in sections:
        if section.start >= MAX_EXTRACTED_TEXT_CHARS:
            continue
        end = min(section.end, MAX_EXTRACTED_TEXT_CHARS)
        if section.start >= end:
            continue
        clipped_sections.append(TextSection(label=section.label, start=section.start, end=end, page=section.page))
    return capped, clipped_sections, True


#: Guards the process-global ``pypdf`` logger level against concurrent extractions.
#: A bare save/restore pair races: two threads extracting at once can interleave so that
#: the first to finish restores the original level while the second is still extracting
#: (unmuting it), or the second's "previous" is the already-muted ERROR and it restores
#: that permanently. Extraction is reachable concurrently through the Flask UI, so this
#: is a live path, not a theoretical one. The depth counter makes the mute reentrant:
#: only the outermost holder restores.
_pypdf_mute_lock = threading.Lock()
_pypdf_mute_depth = 0
_pypdf_mute_previous: int | None = None


@contextlib.contextmanager
def _quiet_pypdf() -> Iterator[None]:
    """Mute ``pypdf``'s per-object repair chatter for the duration of one extraction.

    A malformed-but-recoverable cross-reference table makes ``pypdf`` emit one WARNING
    record per fixed-up object ("Ignoring wrong pointing object 3 0 (offset 641)").
    Real publisher PDFs hit this constantly -- a single 9-page paper produced ~70 such
    lines -- and since the CLI installs no logging configuration they land on the root
    logger's last-resort stderr handler and bury the one line the operator is actually
    reading (ACCEPTED/REJECTED and why).

    They are muted rather than reformatted because they are not actionable: each one
    reports a repair that SUCCEEDED. Whether the extracted text is trustworthy is
    already answered, on our own evidence rather than ``pypdf``'s commentary, by
    ``ExtractedText.lossy`` and the grounding gate's degraded-artifact path -- both of
    which fail closed. ERROR and above still pass through, so a genuine parse failure
    is never hidden.

    Restores the previous level on the way out, including on exception, so this never
    permanently reconfigures logging for a caller that embeds Carmel as a library. The
    save/restore is refcounted under :data:`_pypdf_mute_lock` because the level is
    process-global state and extraction can run concurrently; see that lock's comment for
    the interleavings a bare save/restore pair would corrupt.
    """
    global _pypdf_mute_depth, _pypdf_mute_previous

    pypdf_logger = logging.getLogger("pypdf")
    with _pypdf_mute_lock:
        if _pypdf_mute_depth == 0:
            _pypdf_mute_previous = pypdf_logger.level
            pypdf_logger.setLevel(logging.ERROR)
        _pypdf_mute_depth += 1
    try:
        yield
    finally:
        with _pypdf_mute_lock:
            _pypdf_mute_depth -= 1
            if _pypdf_mute_depth == 0 and _pypdf_mute_previous is not None:
                pypdf_logger.setLevel(_pypdf_mute_previous)
                _pypdf_mute_previous = None


_PATH_LIKE_RE = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][^\s\\/:*?\"<>|]+){2,}")


def _describe_page_error(exc: Exception) -> str:
    """Render a per-page extraction exception as a short, path-redacted string.

    Stored in :class:`PageExtractionFailure.error`, which lands in a
    content-addressed artifact -- so a raw ``str(exc)`` is not safe to keep
    verbatim. ``pypdf`` exceptions do not normally embed local filesystem
    paths (extraction here always reads from an in-memory ``io.BytesIO``,
    never a file path), but nothing in its API contract guarantees that for
    every code path or future version, so any path-shaped substring is
    scrubbed defensively rather than trusted to be absent.
    """
    message = _PATH_LIKE_RE.sub("<redacted-path>", str(exc))
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _extract_pdf(data: bytes) -> ExtractedText:
    """Extract text from a PDF via the optional ``pypdf`` dependency.

    ``pypdf`` is imported lazily so importing this module never fails when the
    optional dependency is absent. When it is absent, or the bytes fail to parse,
    this falls back to an empty, ``lossy=True`` result rather than raising.
    """
    try:
        # `pypdf` ships its own `py.typed` marker, so once the `agents` extra is
        # installed mypy resolves this import directly -- no `type: ignore` needed
        # (and an unconditional one would be a stale, unused-ignore error under that
        # extra, which is exactly what CI's agents-installed lane now checks for).
        import pypdf
    except ImportError:
        return ExtractedText(text="", normalized="", sections=[], extractor="pdf:unavailable", lossy=True)

    try:
        with _quiet_pypdf():
            reader = pypdf.PdfReader(io.BytesIO(data))
            # `len(reader.pages)` is cheap (it reads the page tree, not page content), so
            # checking it before calling any `extract_text()` bounds the page count up
            # front rather than after the fact.
            page_count = len(reader.pages)
    except Exception:
        return ExtractedText(text="", normalized="", sections=[], extractor="pdf:pypdf", lossy=True)

    parts: list[str] = []
    sections: list[TextSection] = []
    page_failures: list[PageExtractionFailure] = []
    cursor = 0
    separator = "\n\n"
    have_prior_page = False
    # `truncated` (-> `lossy=True`) covers BOTH ways a PDF can exceed the bounds we
    # enforce: too many pages, or too many characters. Either one means the returned
    # text is a partial view of the document, and `lossy=True` is load-bearing: the
    # grounding gate fails CLOSED on it rather than silently grounding against a
    # partial document.
    truncated = page_count > MAX_PDF_PAGES
    pages_to_process = min(page_count, MAX_PDF_PAGES)
    with _quiet_pypdf():
        for i in range(pages_to_process):
            # Stop calling `extract_text()` -- the expensive, memory-allocating step
            # -- the moment the running character count would already exceed the cap,
            # rather than materializing every remaining page and trimming only the
            # RETURNED value afterwards. That "trim after the fact" ordering is
            # exactly the gap this fix closes: it let a compression-bomb PDF blow past
            # peak memory before `_cap_text` (below) ever got a chance to run.
            if cursor >= MAX_EXTRACTED_TEXT_CHARS:
                truncated = True
                break
            # Per-page, not per-document: a single damaged page (pypdf's layout mode
            # raises e.g. `KeyError('/Contents')` on a contentless page in real corpus
            # PDFs) must not discard every page around it. Catching around just this
            # call keeps the pages that DO extract cleanly and records the ones that
            # don't, instead of the old behavior of losing an entire multi-page paper
            # to one bad page.
            try:
                page_text = reader.pages[i].extract_text() or ""
            except Exception as exc:
                page_failures.append(PageExtractionFailure(page=i + 1, error=_describe_page_error(exc)))
                continue
            if have_prior_page:
                parts.append(separator)
                cursor += len(separator)
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            sections.append(TextSection(label="body", start=start, end=cursor, page=i + 1))
            have_prior_page = True

    text = "".join(parts)
    text, sections, cap_truncated = _cap_text(text, sections)
    # Any page failure makes this extraction partial, exactly like truncation does:
    # the surviving text is real but incomplete, so `lossy=True` must follow even
    # when only one page out of many failed. When EVERY attempted page fails, `parts`
    # stays empty and `text`/`sections` naturally collapse to the same "total failure"
    # shape the grounding gate and `produce_envelope_from_artifact` already fail
    # closed on -- but `page_failures` still carries why, rather than a bare empty
    # result.
    truncated = truncated or cap_truncated or bool(page_failures)
    sections = _label_special_sections(text, sections)
    return ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=sections,
        page_count=page_count,
        extractor="pdf:pypdf",
        lossy=truncated,
        page_failures=tuple(page_failures),
    )


class _HTMLTextExtractor(HTMLParser):
    """Collects text data, dropping the content of ``<script>``/``<style>`` tags."""

    _STRIPPED_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._STRIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._STRIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def _decode_bytes(data: bytes) -> str:
    """Decode bytes as UTF-8, replacing undecodable bytes rather than raising."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_html(data: bytes) -> ExtractedText:
    """Extract text from HTML using only the stdlib parser; script/style are dropped."""
    parser = _HTMLTextExtractor()
    parser.feed(_decode_bytes(data))
    parser.close()
    text = "".join(parser.parts)
    text, sections, truncated = _cap_text(text, [TextSection(label="body", start=0, end=len(text))])
    sections = _label_special_sections(text, sections)
    return ExtractedText(
        text=text, normalized=normalize_for_match(text), sections=sections, extractor="html", lossy=truncated
    )


def _extract_xml(data: bytes) -> ExtractedText:
    """Extract text from XML (e.g. JATS full text) by tag-stripping, exactly as HTML.

    :class:`_HTMLTextExtractor` handles this without change: dropping markup and
    keeping character data works the same for ``<article-title>`` as for ``<h1>``, and
    JATS carries no ``<script>``/``<style>`` content for the stripping to miss.
    Kept as its own function (and ``extractor`` label) rather than reusing
    :func:`_extract_html` so the two content types remain distinguishable downstream --
    HTML is refused as a primary document by manual acquisition, XML is not.
    """
    parser = _HTMLTextExtractor()
    parser.feed(_decode_bytes(data))
    parser.close()
    text = "".join(parser.parts)
    text, sections, truncated = _cap_text(text, [TextSection(label="body", start=0, end=len(text))])
    sections = _label_special_sections(text, sections)
    return ExtractedText(
        text=text, normalized=normalize_for_match(text), sections=sections, extractor="xml", lossy=truncated
    )


def _extract_plain_text(data: bytes) -> ExtractedText:
    """Take ``text/*`` content verbatim (still section-labeled for references/abstract)."""
    text = _decode_bytes(data)
    text, sections, truncated = _cap_text(text, [TextSection(label="body", start=0, end=len(text))])
    sections = _label_special_sections(text, sections)
    return ExtractedText(
        text=text, normalized=normalize_for_match(text), sections=sections, extractor="text", lossy=truncated
    )


def extract_text(data: bytes, content_type: str) -> ExtractedText:
    """Extract text and section labels from a fetched document's raw bytes.

    Args:
        data: Raw document bytes.
        content_type: Sniffed MIME type, e.g. "application/pdf", "text/html",
            "text/plain".

    Returns:
        An :class:`ExtractedText`. PDF extraction is via the optional ``pypdf``
        dependency (``lossy=True`` and ``extractor="pdf:unavailable"`` when it is not
        installed). HTML and XML are stripped of markup using only the stdlib. Any
        other ``text/*`` type is taken verbatim. An unrecognized content type yields
        empty text with ``lossy=True``.
    """
    if content_type == "application/pdf":
        return _extract_pdf(data)
    if content_type == "text/html":
        return _extract_html(data)
    # Checked BEFORE the generic ``text/*`` prefix: ``application/xml`` does not start
    # with ``text/`` and would otherwise fall through to the empty ``unknown`` result,
    # while ``text/xml`` would be taken verbatim, tags and all.
    if content_type in ("application/xml", "text/xml"):
        return _extract_xml(data)
    if content_type.startswith("text/"):
        return _extract_plain_text(data)
    return ExtractedText(text="", normalized="", sections=[], extractor="unknown", lossy=True)
