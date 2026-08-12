"""Text-show fragments with absolute page geometry, recovered from a PDF.

This is the substrate beneath M1's ``TABLE_CELL`` locators. Validator V7 refuses a
``CharSpanLocator`` as the source of a series VALUE, so a ``DatasetEnvelope`` stays
unconstructible until something can address a value BY CELL -- and a cell is built
out of the fragments this module returns.

It stops deliberately short of grouping. Nothing here decides that two fragments
share a word, a row, a column or a cell. Grouping is a DERIVED structural claim and
the adversarial core of M1; extraction is mechanical. Keeping them in separate
modules is what stops a fabricated pairing from riding in on the back of a
mechanical step, the same way span stitching was separated in the condition-set lane.

Why the engine below looks the way it does -- every one of these was established by
running the alternatives against real publisher PDFs, and each obvious route fails:

* ``extract_text(extraction_mode="layout")`` pads with runs of spaces so that column
  identity is implied by whitespace WIDTH. That is structure inferred from prose,
  which the P0-c ruling outlawed. It also truncates.
* ``visitor_text`` is the obvious API and it is a TRAP: it fires once per LINE and
  reports only that line's STARTING x. Three columns arrive merged into one fragment
  with the per-column x already gone, so a caller is left re-splitting on whitespace
  -- reintroducing the same outlawed fabrication one layer down while believing it
  holds real geometry.
* ``text_show_operations`` returns ``BTGroup``s that are ALSO one-per-line, and it
  left-aligns the whole page by subtracting ``min(tx)`` before returning. Its return
  value therefore carries no absolute coordinates at all.

What does work is the per-text-show ``TextStateParams`` list that pypdf's layout-mode
engine builds internally and then discards. Reaching for it means depending on
private API (see :func:`_engine`), which is guarded rather than assumed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import io
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

__all__ = [
    "FragmentExtraction",
    "FragmentPageFailure",
    "GlyphMapping",
    "TextFragment",
    "extract_fragments",
]

logger = logging.getLogger(__name__)


class GlyphMapping(StrEnum):
    """Whether a fragment's glyphs decoded to real characters.

    Named ``GlyphMapping`` rather than the more obvious ``GlyphHealth`` because
    :class:`carmel.services.numeric.GlyphHealth` already exists and means something
    DIFFERENT: a document-level corruption assessment used to quarantine numerals.
    Two same-named types at different scopes would eventually be passed to each
    other's call sites, and the mistake would typecheck under ``Any``.

    This is a flag, never a repair. See :data:`_UNMAPPED_MARKER_RE`.
    """

    MAPPED = "mapped"
    """Every glyph decoded to a character. Says nothing about whether that character
    is the RIGHT one -- a PDF with a broken ``ToUnicode`` map can decode ``+`` as
    ``þ`` and ``=`` as ``¼`` perfectly "successfully"."""

    UNMAPPED = "unmapped"
    """At least one glyph had no usable mapping and surfaced as a marker rather than
    a character. The fragment's text is returned UNMODIFIED; a consumer that treats
    it as a value is reading a marker as data."""


# Markers a PDF text extractor emits when a glyph has no usable ``ToUnicode`` entry.
#
# This matters far more than it looks. In a real corpus rate-constant table the
# temperature exponent ``n = -1.0`` arrives with its MINUS SIGN as a separate
# fragment whose decoded text is ``/C0`` (an independent extractor renders the same
# glyph ``(cid:3)``). A consumer that ignores the marker reads ``+1.0``: a silent
# SIGN INVERSION inside an otherwise perfectly well-formed number. That is strictly
# worse than a missing value, because nothing downstream looks wrong.
#
# Flagged, never repaired. Mapping ``/C0`` to U+2212 -- or ``þ`` to ``+`` and ``¼``
# to ``=``, which the same PDFs also need -- is a SEMANTIC claim about what the
# document meant, and grounding proves LOCATION, never MEANING. A repair table needs
# its own gate and its own evidence; it does not belong in a mechanical extractor.
# The `/C\d+` arm is DELIBERATELY bounded on both sides. An unanchored `/C\d+`
# substring match is catastrophic in a combustion codebase: it flags `/C2H4`, `C1/C2`,
# `H2/CO`-style species lists, appendix labels and file paths as corrupt. The lookarounds
# require the token to stand alone, so `/C0` matches while `/C2H4` (letter after) and
# `C1/C2` (digit before the slash) do not.
_UNMAPPED_MARKER_RE = re.compile(
    r"""
    \( cid: \d+ \)              # (cid:3)  -- raw character id, no mapping at all
    | (?<![0-9A-Za-z]) /C \d+ (?![0-9A-Za-z])   # /C0 -- a standalone glyph-NAME escape
    | �                    # U+FFFD   -- replacement character
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class TextFragment:
    """One text-show operation, with where on the page it drew.

    A fragment is NOT a word and NOT a cell. Real publisher PDFs emit roughly 2.3
    fragments per word: a single word arrives in several pieces, consecutive
    fragments can OVERLAP in x (kerning), and bare single-space fragments interleave
    with them. Any caller that assumes one fragment is one token is already wrong.
    """

    page: int
    """1-indexed page number, counted AFTER phantom page-tree entries are dropped, so
    that it agrees with the numbering ``carmel.agents.tools.extract`` already
    produces. See :func:`extract_fragments` for why that agreement is not optional."""

    text: str
    """The decoded text, exactly as the document emitted it. Never repaired."""

    x_start: float
    """Absolute page-space x of the first glyph."""

    x_end: float
    """Absolute page-space x after the last glyph. Note ``x_end`` of one fragment may
    exceed ``x_start`` of the next: kerned runs overlap."""

    baseline_y: float
    """Absolute page-space y of the baseline. This is the BASELINE, not a bounding-box
    edge -- comparing it against a bbox-based engine leaves a constant descender
    offset."""

    font_size: float
    rotated: bool
    """True when the text is rotated with respect to the page. Retained rather than
    dropped; see :func:`_page_fragments`."""

    glyph_mapping: GlyphMapping


@dataclass(frozen=True)
class FragmentPageFailure:
    """One page that could not be turned into fragments.

    Recorded rather than merely counted, mirroring
    :class:`carmel.agents.tools.extract.PageExtractionFailure`. A bare ``lossy=True``
    says "something was lost" without saying WHAT, so an operator cannot tell a
    single unreadable page from a document that mostly failed -- and a locator built
    from the pages that DID parse would look complete.
    """

    page: int
    error: str
    """Short, path-redacted description. Built by the text lane's own
    ``_describe_page_error`` so the redaction rules stay in one place."""


@dataclass(frozen=True)
class FragmentExtraction:
    """The result of one whole-document extraction."""

    fragments: tuple[TextFragment, ...] = ()
    lossy: bool = False
    """True when this extraction is known to be incomplete: a page failed, a page
    could not be inspected, or the document was truncated. Mirrors
    ``ExtractedText.lossy``, and like it, fails toward admitting loss."""

    available: bool = True
    """False when pypdf is absent, the capability check refused, or the engine proved
    incompatible partway through. Distinct from ``lossy``: ``available=False`` means
    NOTHING here can be relied on and no claim about this document may be made, while
    ``lossy=True`` means what IS here is real but incomplete. Conflating them is the
    specific error this pair exists to prevent -- an engine-wide incompatibility that
    returned zero fragments while reporting ``available=True`` would read exactly like
    a legitimately empty document."""

    page_failures: tuple[FragmentPageFailure, ...] = ()
    truncated: bool = False
    """True when the document has more pages than ``MAX_PDF_PAGES`` and the tail was
    not processed. The cap is shared with the text lane so the two lanes agree on
    which pages exist; see :func:`extract_fragments`."""

    pypdf_version: str = ""
    """The pypdf version this extraction actually ran against.

    Recorded because the geometry is the evidence, and a pypdf that changed baseline
    semantics, CTM composition, page-rotation normalisation or ``TJ`` displacement
    could keep every attribute name intact -- passing :func:`_engine` -- while
    silently returning DIFFERENT numbers. No capability check can catch that, so the
    version travels with the result and the pin is asserted at runtime."""


def _engine() -> tuple[Any, ...] | None:
    """Resolve pypdf's private layout-mode internals, or refuse.

    Everything this module needs lives under ``pypdf._text_extraction._layout_mode``:
    a leading-underscore package with no API-stability guarantee whatsoever. That is
    a deliberate, guarded trade. The alternative is re-implementing a content-stream
    interpreter -- operator dispatch for ``Tj``/``TJ``/``'``/``"``,
    ``Td``/``TD``/``T*``/``Tm``, ``Tc``/``Tw``/``Tz``/``TL``/``Ts``, ``cm``/``q``/``Q``
    nesting, AND font decoding through ``/Encoding`` and ``ToUnicode`` CMaps -- which
    is a far larger long-term liability than a pinned import, and would be a second
    decoder that could disagree with the one the shipped text lane already uses.

    So the dependency is taken, and the risk is handled where it actually bites: a
    pypdf upgrade that moves or changes these internals must make this module REFUSE
    loudly, never silently return different geometry. Returns ``None`` on any
    mismatch; the caller turns that into ``available=False``.
    """
    try:
        from pypdf._text_extraction._layout_mode._fixed_width_page import (
            recurse_to_target_op,
            resolve_font,
        )
        from pypdf._text_extraction._layout_mode._text_state_manager import (
            TextStateManager,
        )
        from pypdf._text_extraction._layout_mode._text_state_params import (
            TextStateParams,
        )
        from pypdf.generic import ContentStream
    except Exception:  # pragma: no cover - exercised via monkeypatch in tests
        logger.debug("pypdf layout-mode internals unavailable", exc_info=True)
        return None

    # The imports resolving is not enough: the names could survive while the objects
    # behind them change shape. Check every piece actually read below -- including the
    # TextStateParams attributes, which must be checked HERE rather than only at the
    # point of use. A per-page AttributeError is caught as a page failure and degrades
    # to `lossy=True`, so an engine-wide mismatch would otherwise present as "a valid
    # document where every page happened to fail" instead of "the engine is wrong".
    for name in ("set_font", "set_state_param"):
        if not callable(getattr(TextStateManager, name, None)):
            logger.warning("pypdf TextStateManager lacks %s; fragments unavailable", name)
            return None
    # Check FIELDS as well as class attributes. `TextStateParams` is a dataclass, and
    # a field without a default (`font_size` is one) exists only on instances, so a
    # bare `hasattr` on the class reports it missing and would refuse every healthy
    # pypdf. The properties (`tx`, `ty`, `text`, ...) do live on the class, so the
    # available surface is the union of the two.
    available_names = set(dir(TextStateParams))
    with contextlib.suppress(TypeError):  # only if pypdf stops using a dataclass
        available_names |= {field.name for field in dataclasses.fields(TextStateParams)}
    for attr in _REQUIRED_PARAM_ATTRS:
        if attr not in available_names:
            logger.warning("pypdf TextStateParams lacks %s; fragments unavailable", attr)
            return None

    # The geometry is the evidence, and no attribute check can detect a release that
    # keeps every name while changing what the numbers MEAN. pypdf is pinned exactly
    # (`pypdf==6.14.2`) precisely because an extraction's dependency identity has to be
    # provable, and the pin's own comment in pyproject.toml makes bumping a deliberate
    # act with a re-extraction pass attached. Refusing here is the runtime half of that
    # policy: an unpinned pypdf makes this lane UNAVAILABLE rather than silently
    # differently-calibrated.
    try:
        installed = importlib.metadata.version("pypdf")
    except Exception:
        logger.warning("pypdf version is unknown; fragments unavailable")
        return None
    if installed != _PINNED_PYPDF_VERSION:
        logger.warning(
            "pypdf %s is not the pinned %s; fragment geometry is unverified, refusing",
            installed,
            _PINNED_PYPDF_VERSION,
        )
        return None
    return recurse_to_target_op, resolve_font, TextStateManager, ContentStream


_REQUIRED_PARAM_ATTRS = ("text", "tx", "ty", "displaced_tx", "font_size", "rotated")

_PINNED_PYPDF_VERSION = "6.14.2"
"""Must track the ``agents`` extra's exact pin in ``pyproject.toml``."""


class _EngineMismatch(Exception):
    """The pypdf engine is not shaped the way this module requires.

    Raised from page processing but deliberately NOT treated as a page failure: it
    says the ENGINE is wrong, not that one document page is. The distinction is the
    difference between ``available=False`` and a plausible-looking empty result.
    """


def _page_fragments(page: Any, page_number: int, engine: tuple[Any, ...]) -> list[TextFragment]:
    """Recover every text-show operation on one page, with absolute geometry."""
    recurse_to_target_op, resolve_font, text_state_manager, content_stream = engine

    contents = page.get("/Contents")
    if contents is None:
        return []
    content = content_stream(contents.get_object(), page.pdf, "bytes")
    ops: Iterator[tuple[list[Any], bytes]] = iter(content.operations)

    fonts = page._layout_mode_fonts()
    state_mgr = text_state_manager()
    shows: list[Any] = []
    for operands, op in ops:
        if op in (b"BT", b"q"):
            # `strip_rotated=False` is DEFENSIVE, not load-bearing, and the
            # distinction was established by mutating it and watching the tests stay
            # green. In this pypdf the flag is consulted only while assembling the
            # `BTGroup`s -- which this module discards -- while the per-show list it
            # returns alongside them is appended to unconditionally. So today it
            # changes nothing here, and NO TEST CAN CATCH ITS REMOVAL; do not add one
            # that appears to, because it would pass against either value.
            #
            # It is still passed explicitly, because the flag's stated contract is
            # "remove rotated text" and a future pypdf could honour that on this list
            # too. Rotated column headers and rotated axis titles are common, and
            # dropping them would mean a table read as complete with a header row
            # missing -- a silent hole, which is the failure mode this lane exists to
            # refuse. Cheap insurance against a documented behaviour arriving.
            _groups, tjs = recurse_to_target_op(
                ops,
                state_mgr,
                b"ET" if op == b"BT" else b"Q",
                fonts,
                strip_rotated=False,
            )
            shows.extend(tjs)
        elif op == b"Tf":
            state_mgr.set_font(resolve_font(fonts, operands[0]), operands[1])
        else:
            state_mgr.set_state_param(op, operands)

    fragments: list[TextFragment] = []
    for show in shows:
        if any(not hasattr(show, attr) for attr in _REQUIRED_PARAM_ATTRS):
            # Belt-and-braces against a pypdf change that slipped past `_engine`.
            # `_EngineMismatch` rather than a plain error: this must abort the WHOLE
            # extraction as unavailable, not degrade one page to lossy.
            raise _EngineMismatch("pypdf TextStateParams is missing a required attribute")
        text = show.text
        if not text:
            continue
        fragments.append(
            TextFragment(
                page=page_number,
                text=text,
                x_start=float(show.tx),
                x_end=float(show.displaced_tx),
                baseline_y=float(show.ty),
                font_size=float(show.font_size),
                rotated=bool(show.rotated),
                glyph_mapping=(GlyphMapping.UNMAPPED if _UNMAPPED_MARKER_RE.search(text) else GlyphMapping.MAPPED),
            )
        )
    return fragments


def extract_fragments(data: bytes) -> FragmentExtraction:
    """Extract every text-show fragment from ``data``, with absolute page geometry.

    Page numbers come from ``carmel.agents.tools.extract._classify_pdf_page`` rather
    than from ``enumerate(reader.pages)``. That is not a stylistic preference. pypdf's
    ``reader.pages`` walks ``/Kids`` without checking ``/Type``, so a linearized PDF
    can have its LINEARIZATION PARAMETER DICTIONARY counted as a page -- observed on
    real corpus papers -- which shifts every later page index by one. A locator citing
    such an index sends a human to the wrong page while looking perfectly checkable,
    and it is the numbering disagreement, not the crash, that does the damage. The
    shipped text lane already solved this; a second filter that disagreed with it
    would be worse than none, so this reuses that one classifier.

    Never raises for a malformed document: returns a degraded
    :class:`FragmentExtraction` instead, matching how ``_extract_pdf`` degrades.
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return FragmentExtraction(lossy=True, available=False)

    from carmel.agents.tools.extract import (
        MAX_PDF_PAGES,
        _classify_pdf_page,
        _describe_page_error,
        _PageKind,
        _quiet_pypdf,
    )

    engine = _engine()
    if engine is None:
        return FragmentExtraction(lossy=True, available=False)

    version = importlib.metadata.version("pypdf")
    fragments: list[TextFragment] = []
    failures: list[FragmentPageFailure] = []
    lossy = False
    truncated = False
    try:
        with _quiet_pypdf():
            reader = PdfReader(io.BytesIO(data))
            page_number = 0
            for page in reader.pages:
                # Classify BEFORE touching any page attribute: `page.mediabox` raises
                # TypeError on a phantom entry, which has no /MediaBox to resolve.
                kind = _classify_pdf_page(page)
                if kind is _PageKind.PHANTOM:
                    continue
                page_number += 1
                if page_number > MAX_PDF_PAGES:
                    # Same cap, counted the same way, as the text lane. Sharing it is
                    # the point: if one lane stopped at 2000 real pages and the other
                    # walked on, a fragment could carry a page number the text lane
                    # says does not exist, and the two provenance stories would
                    # disagree about the same document.
                    truncated = True
                    lossy = True
                    break
                if kind is _PageKind.UNINSPECTABLE:
                    lossy = True
                try:
                    fragments.extend(_page_fragments(page, page_number, engine))
                except _EngineMismatch:
                    # Not a page failure. The engine is wrong, so nothing extracted
                    # from this document can be relied on.
                    raise
                except Exception as exc:
                    logger.debug("fragment extraction failed on page %d", page_number, exc_info=True)
                    failures.append(FragmentPageFailure(page=page_number, error=_describe_page_error(exc)))
                    lossy = True
    except Exception:
        return FragmentExtraction(lossy=True, available=False, pypdf_version=version)

    return FragmentExtraction(
        fragments=tuple(fragments),
        lossy=lossy,
        available=True,
        page_failures=tuple(failures),
        truncated=truncated,
        pypdf_version=version,
    )
