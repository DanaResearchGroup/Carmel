"""Propose candidate table footprints from a PDF's own geometry, or refuse.

**This is the missing enumerator, and it is deliberately weak where the document is.**
:func:`~carmel.services.pdf_tables.build_inventory` derives a grid inside a box a caller
DREW; nothing in the PDF lane ever proposed that box from the document, so only tables whose
coordinates a human hand-pinned into the source tree were ever reachable. The OOXML and
spreadsheet lanes each carry a byte-only enumerator (``embed_ooxml_tables``,
``embed_xlsx_sheets``, ``unpack_and_embed_member_tables``); this is the PDF one.

**A proposal is a CLAIM, not a fact.** Every footprint this module emits carries
:attr:`~carmel.services.pdf_tables.FootprintGeometryOrigin.DERIVED`, so a reader -- and every
record written from it -- can tell a computed box from a measured one. The positive evidence a
proposal rests on is NOT that a caption regex matched: it is that ``build_inventory``, judging
the derived box by exactly the same rules it applies to a hand-drawn one, returned a COMPLETE
grid of at least two stable columns and at least two rows. This module weakens no guard in
``build_inventory`` and adds no softer door of its own; a candidate that ``build_inventory``
refuses is refused here too, with the reason carried through.

**Refuse rather than guess.** A page whose ``Table N`` band is a prose sentence ("Table 2. The
uncertainty factors were evaluated...") yields no aligned column valley across two rows and is
refused, not proposed. Refusing a page and being right about the pages it proposes is the
intended outcome; proposing a box of unknown quality is a failure even when it "finds a table".

**Two boundaries rest on aligned emptiness, which is weaker than a printed rule.** The
extraction layer does not surface stroked ruled lines, so a proposed column edge is an aligned
whitespace valley and the proposed lower edge (:func:`_body_extent`) is a row-pitch heuristic
with no ruled line under it. Both are marked as the soft signals they are; ``build_inventory``
is the strict judge that follows, and a box drawn too wide or too deep is refused there rather
than shipped.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from enum import StrEnum

from carmel.services.pdf_fragments import FragmentExtraction, TextFragment
from carmel.services.pdf_tables import (
    COLUMN_VALLEY_PT,
    CellInventory,
    ClaimedFootprint,
    FootprintGeometryOrigin,
    InventoryRefusalReason,
    _bands,
    _column_bounds,
    _ink_x_end,
    _looks_like_affix,
    build_inventory,
)

__all__ = [
    "CAPTION_HEADING",
    "ProposalOutcome",
    "ProposalRefusal",
    "ProposalRefusalReason",
    "ProposedTable",
    "propose_tables",
]

#: A band is a CANDIDATE caption when one of its aligned-emptiness blocks, read left to right
#: with whitespace collapsed, opens with ``Table N``.
#:
#: This is only a candidate gate -- an ORACLE, in the same sense
#: :class:`~carmel.services.pdf_tables.InventoryRefusalReason` uses the word: a PDF carrying
#: "Table 99" beside a plot matches it, and a prose sentence that opens "Table 2." matches it
#: too. Nothing is proposed on the strength of this match; it only nominates a band whose
#: column-scoped body ``build_inventory`` is then asked to derive a grid from. The textual form
#: of what FOLLOWS the number (a dash-title versus an "N."-sentence) is a weak secondary signal
#: this module deliberately does not gate on: geometry is the gate.
CAPTION_HEADING = re.compile(r"^(Table|TABLE)\s*\d+")

#: How far below a caption heading a line may sit and still be read as a continuation of the
#: caption paragraph rather than the first table row, as a multiple of the heading's rendered
#: font height.
#:
#: A SOFT proposal heuristic, not a measured typographic property, and named so it is not read
#: as more. A journal caption wraps onto a second line at roughly single leading (~1.0-1.2x the
#: font height); the first table row sits a larger pitch below. The multiple is generous toward
#: absorbing a continuation, because the failure direction is safe: absorbing one line too few
#: leaves a caption line orphaned above the box and ``build_inventory`` refuses under
#: :attr:`~carmel.services.pdf_tables.InventoryRefusalReason.ORPHANED_BAND_ABOVE_THE_BOX`;
#: absorbing one too many folds a table row into the caption and the grid then loses a row or
#: refuses. Neither ships a wrong grid silently.
_CAPTION_CONTINUATION_PITCH_RATIO = 1.6

#: How large a gap between two consecutive in-column body bands must be, as a multiple of the
#: body's own median row pitch, before it is read as the END of the table rather than a wider
#: gap inside it (a header separated from its data by extra leading).
#:
#: A SOFT proposal heuristic. There is no ruled line to rest the lower edge on, so the box's
#: depth is derived from the table's own rhythm: a run of rows at one pitch, ending where a gap
#: departs from it. Generous, because a header-to-data gap can be ~1.8x the data pitch and must
#: not be mistaken for the table's end; a following paragraph typically breaks by more. A box
#: drawn too deep is refused by ``build_inventory``'s edge guards, not shipped.
_BODY_END_PITCH_RATIO = 2.0

#: The least a proposed table must have to be a table at all: two columns and two rows.
_MIN_COLUMNS = 2
_MIN_ROWS = 2


class ProposalRefusalReason(StrEnum):
    """Why a ``Table N`` candidate did not become a proposed table.

    Distinct from :class:`~carmel.services.pdf_tables.InventoryRefusalReason`, which is why a
    grid could not be DERIVED inside a given box. These are why a proposed box was not offered
    at all -- and each one names which of the two tests (the caption extracts as a printed fact;
    the column-scoped body yields >=2 columns and >=2 rows) failed.
    """

    EXTRACTION_UNAVAILABLE = "extraction_unavailable"
    """The fragment lane produced nothing usable for the whole document, so no page can be
    read. A property of the document/toolchain, reported once, not per candidate. This is one
    face of the extraction floor: a document whose every page fails to extract blocks a human
    with a mouse exactly as hard as it blocks this module."""

    CAPTION_COLUMN_UNRESOLVED = "caption_column_unresolved"
    """The candidate's anchor line could not be scoped to a single column, or nothing lies
    below it, so no box could even be drawn. The anchor line must be one contiguous run of
    aligned emptiness (a caption is), and there must be a body beneath it."""

    CAPTION_NOT_PRINTED = "caption_not_printed"
    """``build_inventory`` refused the derived box under
    :attr:`~carmel.services.pdf_tables.InventoryRefusalReason.CAPTION_ANCHOR_ABSENT`: the
    caption test failed. The text this module lifted from the anchor band did not survive
    ``build_inventory``'s own re-check at the derived position -- the caption-extracts test."""

    GRID_NOT_DERIVED = "grid_not_derived"
    """``build_inventory`` refused the derived box for a grid/body reason (anything but the
    caption anchor): the body test failed. The inner reason is carried in ``detail`` so a prose
    sentence refused for lacking column structure is distinguishable from a real table clipped
    by a boundary that cut a fragment."""

    TOO_FEW_COLUMNS = "too_few_columns"
    """``build_inventory`` derived a COMPLETE grid, but of fewer than two columns. A single
    column of values under a caption is not a table this module proposes."""

    TOO_FEW_ROWS = "too_few_rows"
    """``build_inventory`` derived a COMPLETE grid, but of fewer than two rows. One row under a
    caption is a heading, not a table."""


@dataclass(frozen=True, slots=True)
class ProposalRefusal:
    """A ``Table N`` candidate that was refused, and why."""

    reason: ProposalRefusalReason
    detail: str
    page: int
    caption_fragment: str
    """A short quoted fragment of the candidate caption -- never a whole line of the document,
    never a filename or page number alone. It is what lets a reader of the report tell one
    refusal from another."""


@dataclass(frozen=True, slots=True)
class ProposedTable:
    """A derived footprint whose grid ``build_inventory`` confirmed."""

    footprint: ClaimedFootprint
    """Always :attr:`~carmel.services.pdf_tables.FootprintGeometryOrigin.DERIVED`."""

    inventory: CellInventory
    """The complete grid ``build_inventory`` returned for :attr:`footprint`."""

    caption_fragment: str
    """A short quoted fragment of the caption, as on :class:`ProposalRefusal`."""


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    """Every candidate on a document, split into what was proposed and what was refused."""

    proposals: tuple[ProposedTable, ...]
    refusals: tuple[ProposalRefusal, ...]


def _caption_fragment(text: str) -> str:
    """A short, safe quotation of a caption for evidence -- the heading and a few words."""
    collapsed = " ".join(text.split())
    return collapsed[:48]


def _block_text(band: list[TextFragment], block: tuple[float, float]) -> str:
    """The collapsed text of the fragments whose ink falls inside ``block``."""
    left, right = block
    members = sorted(
        (f for f in band if f.x_start >= left - 1e-6 and _ink_x_end(f) <= right + 1e-6),
        key=lambda f: f.x_start,
    )
    return "".join(f.text for f in members)


def _heading_block(band: list[TextFragment]) -> tuple[float, float] | None:
    """The band's block that opens ``Table N``, if any -- the caption's own column.

    Scans every aligned-emptiness block, not just the leftmost: a caption in the RIGHT column of
    a two-column page has the other column's text to its left, so the ``Table N`` block is not
    block zero. Whitespace is collapsed both with and without spaces, because a publisher may or
    may not emit the space between ``Table`` and its number.
    """
    for block in _column_bounds([(0.0, band, [])]):
        text = _block_text(band, block)
        if CAPTION_HEADING.match(text.strip()) or CAPTION_HEADING.match("".join(text.split())):
            return block
    return None


def _band_font_height(band: list[TextFragment]) -> float:
    return max((f.font_height for f in band), default=0.0)


def _in_column(band: list[TextFragment], x_start: float, x_end: float) -> list[TextFragment]:
    return [f for f in band if f.x_start >= x_start and _ink_x_end(f) <= x_end]


def _anchor_index(
    bands: list[tuple[float, list[TextFragment]]],
    heading_index: int,
    heading_x: float,
) -> int:
    """Walk down from a caption heading to its LAST printed line -- the anchor.

    A caption may wrap onto continuation lines, and ``build_inventory`` requires the anchor to
    sit DIRECTLY above the box (a band between the anchor and the box top orphans). So the anchor
    is not the ``Table N`` heading line but the last line of the caption paragraph. A line is a
    continuation when it is contiguous (within :data:`_CAPTION_CONTINUATION_PITCH_RATIO` of the
    heading's font height), left-aligned to the heading, the same font height, and a single
    aligned-emptiness run (prose, not a multi-column table row). The walk stops at the first line
    that breaks any of these -- that line begins the table body.
    """
    height = _band_font_height(bands[heading_index][1])
    anchor = heading_index
    for index in range(heading_index + 1, len(bands)):
        gap = bands[index - 1][0] - bands[index][0]
        band = bands[index][1]
        if height <= 0 or gap > _CAPTION_CONTINUATION_PITCH_RATIO * height:
            break
        blocks = _column_bounds([(0.0, band, [])])
        run = next((b for b in blocks if abs(b[0] - heading_x) <= COLUMN_VALLEY_PT), None)
        if run is None:
            break
        # A continuation is one contiguous run left-aligned to the heading; a table row breaks
        # into more than one block in the caption's column, or is not left-aligned to it.
        if len(_column_bounds([(0.0, _in_column(band, run[0], run[1]), [])])) != 1:
            break
        if abs(_band_font_height(band) - height) > 0.5:
            break
        anchor = index
    return anchor


def _rightmost_ink(bands: list[list[TextFragment]], x_start: float, upper: float) -> list[float]:
    """Ink-end x of every non-rotated, non-blank fragment starting in ``[x_start, upper)``.

    Rotated fragments are excluded because their x-extent is not a horizontal interval -- 23 of
    257 rotated corpus fragments report an ``x_end`` past their own page's mediabox, and one
    such (a sideways "Downloaded from ..." watermark reaching x=1273 on a 595 pt page) would
    otherwise blow the right edge across the whole page. Blank fragments are skipped for the same
    reason ``_straddle_refusal`` skips them: a bare space is not ink that bounds a column.
    """
    return [
        _ink_x_end(f)
        for band in bands
        for f in band
        if not f.rotated and f.text.strip() and x_start - 1e-6 <= f.x_start < upper
    ]


def _column_left(anchor_band: list[TextFragment], heading_x: float) -> tuple[float, float | None] | None:
    """The caption column's left edge and the near edge of the next column, or None.

    Reuses the aligned-emptiness valley on the ANCHOR baseline. The anchor's own block gives the
    left edge; because a caption anchor is one contiguous run it is a single block. The second
    return is the near edge of the first block to the right (the page gutter's far side on a
    two-column page), or ``None`` when the caption's column is the rightmost on the page.
    """
    blocks = _column_bounds([(0.0, anchor_band, [])])
    own = next((b for b in blocks if abs(b[0] - heading_x) <= COLUMN_VALLEY_PT), None)
    if own is None:
        return None
    right_neighbours = [b[0] for b in blocks if b[0] > own[1]]
    return own[0], (min(right_neighbours) if right_neighbours else None)


def _column_right(
    body_bands: list[list[TextFragment]],
    x_start: float,
    adjacent_left: float | None,
) -> float | None:
    """The box's right edge, set inside the PAGE GUTTER rather than on a column's ink.

    Measured over the TABLE BODY only -- the bands the caller has already clipped to the table
    with :func:`_body_extent` -- never the whole lower page, so a full-width footnote below the
    table cannot bridge the gutter and drag the edge across it. On a two-column page the gutter
    runs from the body's own rightmost ink to the neighbour's near edge, and the edge is the
    gutter's MIDPOINT: on the neighbour's edge its fragments jitter a point or two left line to
    line and get cut, so ``build_inventory`` refuses under STRADDLING_FRAGMENT_AT_THE_BOX_EDGE
    (the hand-pinned boxes sit in the gutter, not on it). With no right neighbour the edge is the
    body's own rightmost ink -- there is no column to exclude. Failing closed: too wide lets the
    adjacent column intrude and refuses, too narrow refuses under TRUNCATED_COLUMN_BESIDE_THE_BOX.
    """
    if adjacent_left is None:
        rights = _rightmost_ink(body_bands, x_start, math.inf)
        return max(rights) if rights else None
    # One column cannot begin within a valley-width of the next column's edge, so a fragment
    # starting inside that margin IS the adjacent column, however little its own left edge
    # differs from the anchor baseline's (measured jitter: ~1e-4 pt, enough to slip past a bare
    # `<`). Excluding that margin keeps ``body_right`` on the table's true rightmost ink.
    rights = _rightmost_ink(body_bands, x_start, adjacent_left - COLUMN_VALLEY_PT)
    if not rights:
        return None
    body_right = max(rights)
    return (min(body_right, adjacent_left) + adjacent_left) / 2.0


def _row_band_indices(bands: list[tuple[float, list[TextFragment]]]) -> set[int]:
    """Indices of the bands that are ROWS -- affix-shaped bands excluded.

    See :func:`_row_pitch_gaps` for why raw bands are not rows. Returned as indices rather than
    baselines so both callers -- the pitch statistic and the body walk -- classify each band
    once, by the same rule.
    """
    rows: set[int] = set()
    last_row: list[TextFragment] = []
    for index, (_y, band) in enumerate(bands):
        # Judge against the last band ACCEPTED as a row, not the immediate predecessor: a row
        # trailed by both a subscript and a superscript puts two affix bands in a row, and the
        # second one compared against the first (two small bands) never looks like an affix, so
        # a per-neighbour test leaks the tail of every such run back in. Before the first row is
        # known, the band below is the only reference available.
        reference = last_row or (bands[index + 1][1] if index + 1 < len(bands) else [])
        if _looks_like_affix(band, reference):
            continue
        rows.add(index)
        last_row = band
    return rows


def _row_pitch_gaps(bands: list[tuple[float, list[TextFragment]]]) -> list[float]:
    """Baseline gaps between ROW bands, with affix-shaped bands excluded.

    A subscript or superscript sits on its own baseline, so a raw band is not a row --
    :func:`~carmel.services.pdf_tables._merge_affix_bands` folds such a band into the row it is
    interior to before anything downstream counts rows. Measuring pitch over RAW bands therefore
    mixes two different quantities: the gap between consecutive rows, and the much smaller gap
    between a row and its own subscript. On a formula-heavy table (``H2/CO``, ``x_i``) the affix
    gaps can outnumber the row gaps and drag the median down, and every threshold derived from
    that median shrinks with it -- :func:`_body_extent` then breaks on an ordinary row gap and
    truncates the table, and ``y_bottom`` rests too close to the last baseline.

    Excluding affix-shaped bands from the STATISTIC estimates the rhythm of rows, which is what
    both callers mean by "pitch". It deliberately does not fold or drop them anywhere else: the
    walk still visits every band, and ``build_inventory`` remains the judge of what is a row.
    """
    row_indices = _row_band_indices(bands)
    ys = [y for index, (y, _band) in enumerate(bands) if index in row_indices]
    return [ys[i - 1] - ys[i] for i in range(1, len(ys))]


def _body_extent(
    body_bands: list[tuple[float, list[TextFragment]]],
    x_start: float,
    x_end: float,
) -> float | None:
    """The baseline of the table's LAST row, or None if there is no body in the column.

    Walks the in-column bands below the anchor and stops where the table's own rhythm breaks:
    a gap larger than :data:`_BODY_END_PITCH_RATIO` times the median in-column row pitch, or a
    band that opens a new ``Table N`` caption. A SOFT boundary (see the module docstring) --
    aligned emptiness with no ruled line beneath it.
    """
    in_col_bands = [(by, band) for by, band in body_bands if _in_column(band, x_start, x_end)]
    if not in_col_bands:
        return None
    gaps = _row_pitch_gaps(in_col_bands)
    pitch = statistics.median(gaps) if gaps else 0.0
    row_indices = _row_band_indices(in_col_bands)
    last = in_col_bands[0][0]
    for index in range(1, len(in_col_bands)):
        gap = in_col_bands[index - 1][0] - in_col_bands[index][0]
        band = in_col_bands[index][1]
        if _heading_block(band) is not None:
            break
        if pitch > 0.0 and gap > _BODY_END_PITCH_RATIO * pitch:
            break
        # Only a ROW can be the last row. A table whose final row carries a subscript ends at
        # that row's baseline, not at the subscript's -- otherwise ``y_bottom`` is derived from
        # an affix baseline and the box reaches a half-pitch below the wrong line.
        if index in row_indices:
            last = in_col_bands[index][0]
    return last


def _classify(inventory: CellInventory, page: int, caption: str) -> ProposedTable | ProposalRefusal:
    """Judge one derived inventory against the two tests, by identity of the refusal reason."""
    if inventory.refusals:
        refusal = inventory.refusals[0]
        if refusal.reason is InventoryRefusalReason.CAPTION_ANCHOR_ABSENT:
            return ProposalRefusal(ProposalRefusalReason.CAPTION_NOT_PRINTED, refusal.detail, page, caption)
        return ProposalRefusal(
            ProposalRefusalReason.GRID_NOT_DERIVED,
            f"{refusal.reason.value}: {refusal.detail}",
            page,
            caption,
        )
    if len(inventory.column_bounds) < _MIN_COLUMNS:
        return ProposalRefusal(
            ProposalRefusalReason.TOO_FEW_COLUMNS,
            f"the column-scoped body derived {len(inventory.column_bounds)} column(s), fewer than {_MIN_COLUMNS}",
            page,
            caption,
        )
    if len(inventory.rows) < _MIN_ROWS:
        return ProposalRefusal(
            ProposalRefusalReason.TOO_FEW_ROWS,
            f"the column-scoped body derived {len(inventory.rows)} row(s), fewer than {_MIN_ROWS}",
            page,
            caption,
        )
    return ProposedTable(inventory.footprint, inventory, caption)


def _propose_one(
    extraction: FragmentExtraction,
    page: int,
    page_bands: list[tuple[float, list[TextFragment]]],
    heading_index: int,
    heading_block: tuple[float, float],
) -> ProposedTable | ProposalRefusal:
    """Draw a DERIVED footprint for one ``Table N`` candidate and let ``build_inventory`` judge."""
    heading_x = heading_block[0]
    caption = _caption_fragment(_block_text(page_bands[heading_index][1], heading_block))

    anchor_index = _anchor_index(page_bands, heading_index, heading_x)
    anchor_by, anchor_band = page_bands[anchor_index]
    below = page_bands[anchor_index + 1 :]
    if not below:
        return ProposalRefusal(
            ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED,
            "nothing lies below the caption anchor, so no box can be drawn",
            page,
            caption,
        )

    left = _column_left(anchor_band, heading_x)
    if left is None:
        return ProposalRefusal(
            ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED,
            "the caption anchor did not resolve to a single column, so no box can be drawn",
            page,
            caption,
        )
    x_start, adjacent_left = left

    # Clip the body to the table FIRST, bounded on the right by the adjacent column's edge (or
    # nothing), so the gutter edge below is measured over the table's own rows -- not the whole
    # lower page, where a full-width footnote would bridge the gutter.
    provisional_right = adjacent_left if adjacent_left is not None else math.inf
    last_row_y = _body_extent(below, x_start, provisional_right)
    if last_row_y is None:
        return ProposalRefusal(
            ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED,
            "no body band lies inside the caption's column, so no box can be drawn",
            page,
            caption,
        )
    table_bands = [band for by, band in below if last_row_y - 1e-6 <= by <= anchor_by]
    # The left edge is the leftmost ink of the whole column, caption AND body -- not the
    # caption's alone. A body cell can begin a fraction of a point left of the caption's first
    # glyph (measured ~5e-4 pt), and taking the caption's edge would cut it. There is no
    # neighbour on the left to exclude, so the true left edge is the min over the left-edge
    # cluster (everything within a valley-width of the caption's own left).
    left_cluster = [
        f.x_start
        for band in [anchor_band, *table_bands]
        for f in band
        if not f.rotated and f.text.strip() and abs(f.x_start - x_start) <= COLUMN_VALLEY_PT
    ]
    x_start = min([x_start, *left_cluster])
    # The caption ANCHOR line is part of the column and must fit inside the box -- build_inventory
    # scopes the caption to [x_start, x_end] and refuses if it overflows -- so its own ink counts
    # toward the right edge alongside the body's.
    x_end = _column_right([anchor_band, *table_bands], x_start, adjacent_left)
    if x_end is None or x_end <= x_start:
        return ProposalRefusal(
            ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED,
            "the caption's column has no measurable right edge, so no box can be drawn",
            page,
            caption,
        )

    first_body_y = below[0][0]
    y_top = (anchor_by + first_body_y) / 2.0
    in_col_below = [(by, b) for by, b in below if _in_column(b, x_start, x_end)]
    pitch_gaps = _row_pitch_gaps(in_col_below)
    pitch = statistics.median(pitch_gaps) if pitch_gaps else _band_font_height(anchor_band)
    # Half a median row pitch below the last row, so the box CONTAINS it rather than resting its
    # edge on the baseline. The exact depth is a soft claim; build_inventory is the judge.
    y_bottom = last_row_y - max(pitch, 1.0) / 2.0

    caption_members = _in_column(anchor_band, x_start, x_end)
    if not caption_members:
        return ProposalRefusal(
            ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED,
            "the caption anchor did not fit inside its own column window, so no box can be drawn",
            page,
            caption,
        )
    anchor_text = "".join(f.text for f in sorted(caption_members, key=lambda f: f.x_start))
    caption_x_start = min(f.x_start for f in caption_members)

    footprint = ClaimedFootprint(
        page=page,
        x_start=x_start,
        x_end=x_end,
        y_top=y_top,
        y_bottom=y_bottom,
        caption_text=anchor_text,
        caption_x_start=caption_x_start,
        caption_baseline_y=anchor_by,
        geometry_origin=FootprintGeometryOrigin.DERIVED,
    )
    return _classify(build_inventory(extraction, footprint), page, caption)


def propose_tables(extraction: FragmentExtraction) -> ProposalOutcome:
    """Propose every table footprint the document's geometry supports, and refuse the rest.

    For each ``Table N`` band on each page: derive the caption anchor (its last printed line),
    the caption's page-column x-window, and a body depth from the table's own row pitch; draw a
    :attr:`~carmel.services.pdf_tables.FootprintGeometryOrigin.DERIVED` footprint; and let
    ``build_inventory`` judge it. A complete grid of >=2 columns and >=2 rows is proposed;
    everything else is a typed refusal.
    """
    if not extraction.available:
        return ProposalOutcome(
            proposals=(),
            refusals=(
                ProposalRefusal(
                    ProposalRefusalReason.EXTRACTION_UNAVAILABLE,
                    f"the fragment lane is unavailable for this document ({extraction.status})",
                    page=-1,
                    caption_fragment="",
                ),
            ),
        )

    proposals: list[ProposedTable] = []
    refusals: list[ProposalRefusal] = []
    for page in sorted({f.page for f in extraction.fragments}):
        page_bands = _bands([f for f in extraction.fragments if f.page == page])
        for heading_index, (_by, band) in enumerate(page_bands):
            heading_block = _heading_block(band)
            if heading_block is None:
                continue
            outcome = _propose_one(extraction, page, page_bands, heading_index, heading_block)
            if isinstance(outcome, ProposedTable):
                proposals.append(outcome)
            else:
                refusals.append(outcome)

    return ProposalOutcome(proposals=tuple(proposals), refusals=tuple(refusals))
