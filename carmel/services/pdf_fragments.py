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

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

__all__ = [
    "FragmentExtraction",
    "GlyphHealth",
    "TextFragment",
    "extract_fragments",
]

logger = logging.getLogger(__name__)


class GlyphHealth(StrEnum):
    """Whether a fragment's glyphs decoded to real characters.

    This is a flag, never a repair. See :data:`_UNMAPPED_MARKER_RE` for why the
    distinction is load-bearing.
    """

    OK = "ok"
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
_UNMAPPED_MARKER_RE = re.compile(
    r"""
    \(cid: \d+ \)     # (cid:3)  -- raw character id, no mapping at all
    | /C \d+          # /C0      -- a glyph-NAME escape that reached the text layer
    | �          # U+FFFD   -- replacement character
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

    glyph_health: GlyphHealth


@dataclass(frozen=True)
class FragmentExtraction:
    """The result of one whole-document extraction."""

    fragments: tuple[TextFragment, ...] = ()
    lossy: bool = False
    """True when this extraction is known to be incomplete: a page failed, a page
    could not be inspected, or the engine was unavailable. Mirrors
    ``ExtractedText.lossy``, and like it, fails toward admitting loss."""

    available: bool = True
    """False when pypdf is absent or the capability check refused. Distinct from
    ``lossy``: ``available=False`` means NOTHING was extracted and no claim about
    this document can be made, rather than that something partial was."""


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
        from pypdf.generic import ContentStream
    except Exception:  # pragma: no cover - exercised via monkeypatch in tests
        logger.debug("pypdf layout-mode internals unavailable", exc_info=True)
        return None

    # The imports resolving is not enough: the names could survive while the objects
    # behind them change shape. Check the pieces actually read below.
    for name in ("set_font", "set_state_param"):
        if not callable(getattr(TextStateManager, name, None)):
            logger.warning("pypdf TextStateManager lacks %s; fragments unavailable", name)
            return None
    return recurse_to_target_op, resolve_font, TextStateManager, ContentStream


_REQUIRED_PARAM_ATTRS = ("text", "tx", "ty", "displaced_tx", "font_size", "rotated")


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
            # Belt-and-braces against a pypdf change that slipped past `_engine`:
            # refuse the page rather than emit fragments with defaulted geometry.
            raise AttributeError("pypdf TextStateParams is missing a required attribute")
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
                glyph_health=(GlyphHealth.UNMAPPED if _UNMAPPED_MARKER_RE.search(text) else GlyphHealth.OK),
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

    from carmel.agents.tools.extract import _classify_pdf_page, _PageKind, _quiet_pypdf

    engine = _engine()
    if engine is None:
        return FragmentExtraction(lossy=True, available=False)

    import io

    fragments: list[TextFragment] = []
    lossy = False
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
                if kind is _PageKind.UNINSPECTABLE:
                    lossy = True
                try:
                    fragments.extend(_page_fragments(page, page_number, engine))
                except Exception:
                    logger.debug("fragment extraction failed on page %d", page_number, exc_info=True)
                    lossy = True
    except Exception:
        return FragmentExtraction(lossy=True, available=False)

    return FragmentExtraction(fragments=tuple(fragments), lossy=lossy, available=True)
