"""A claimed-footprint cell inventory: rows, columns and cells derived inside a box.

**This is not a table parser and must never be named one.** Nothing here finds a table.
The caller draws a box and names the caption anchored at its top; this module derives,
inside that box only, what rows and columns the geometry actually supports. Round 108's
ruling stands: a caption gate is an ORACLE, not a safety property, and a PDF carrying
``Table 99`` beside a plot defeats it.

**Why the module exists at all.** A :class:`~carmel.schemas.datasets.TableCellLocator`
carries ``table_key``, ``row`` and ``col`` -- no page, no box, no anchor, no digest. So
nothing in the schema can refute a row ordinal: a footprint whose top edge is one band
too high shifts EVERY ordinal while every cited value stays locally present and every
validator passes. That is the silent-corruption shape this codebase exists to prevent,
and the repair is not a better validator on the locator -- it is to derive the ordinals
from geometry the caller does not control, and to persist enough of that derivation that
an independent replay can recompute it and refuse when it does not reproduce.

**What this module does NOT do, deliberately.**

* It does not approve anything. :func:`~carmel.services.pdf_cells.region_refusals`
  states that an empty refusal tuple is not an approval, not evidence and not a
  verification, and that nothing but refusals may be persisted. The same rule binds
  here: a complete inventory is a DERIVATION, and its positive evidence is that a
  recomputation from ``raw.bin`` reproduces it -- never that no refusal fired.
* It does not model ``rowspan``/``colspan``. A value printed across two columns is
  recorded once, at its own measured position; the neighbouring cell stays EMPTY.
  Duplicating it would fabricate a cell the page never carried, and guessing the
  association is a semantic claim this layer cannot make.
* It does not repair text. Members are concatenated in READING ORDER (x within the
  merged band), so ``H`` + ``2`` + ``/CO`` becomes ``H2/CO`` without a character being
  inserted, substituted or dropped. A ToUnicode map that renders phi as ``f`` is carried
  verbatim.
* It emits no partial inventory. A refused inventory holds no cells and no rows, and
  :class:`CellInventory` makes that ONE combination unconstructible. It does not make
  every malformed inventory unconstructible -- a direct caller can still build one whose
  cells cite columns it does not carry -- and saying otherwise would be the overstatement
  this file elsewhere warns about. What is enforced is stated on
  :meth:`CellInventory.__post_init__`; the rest is `build_inventory`'s to uphold.
* **The row anchor does no work yet.** It is derived and stored, and nothing recomputes
  it, compares it, or ties it to a ``TableCellLocator``. It is the input a future replay
  path needs, not a check that runs today, and until that path exists the ordinal-drift
  defence rests entirely on the footprint refusals above.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum

from carmel.services.pdf_fragments import FragmentExtraction, GlyphMapping, TextFragment

__all__ = [
    "AFFIX_HEIGHT_RATIO",
    "COLUMN_VALLEY_PT",
    "CellInventory",
    "ClaimedFootprint",
    "InventoryCell",
    "InventoryRefusal",
    "InventoryRefusalReason",
    "InventoryRow",
    "build_inventory",
]

#: Width of an aligned fully-empty x-strip that separates two columns, in points.
#:
#: A MEASURED valley, not a tuned threshold. Probe 6 took the widest internal fully-empty
#: strip over windows of consecutive lines across the whole corpus and found **zero
#: windows out of 926 in [4, 8) pt** -- 19.1% below 4 pt (word gaps, which do not align
#: across lines, so the aligned strip collapses toward zero) and 80.9% at 8 pt or more
#: (real gutters and column boundaries). The two populations do not overlap, so any value
#: placed anywhere in that gap classifies identically and the constant is not doing the
#: work. Moving it OUT of the gap is a new claim and needs its own measurement; the test
#: suite asserts it stays inside.
COLUMN_VALLEY_PT = 6.0

#: A band whose fragments are all shorter than this fraction of the band it would join is
#: a candidate affix (subscript/superscript) rather than a row of its own.
#:
#: Probe 49 measured five subscript-only bands out of twelve on the target table: a
#: subscript sits on its OWN baseline, so counting raw baseline bands puts the last row at
#: ordinal 11 instead of 6 while every value stays correctly grounded. The ratio is a
#: shape test, not a size test -- ``font_height`` is the RENDERED height, so it survives
#: the ``Tf /F1 1`` trap that makes ``font_size`` a constant 1.0 across the corpus.
#:
#: Being wrong in the permissive direction here is far worse than being wrong in the
#: strict direction: merging a genuinely short ROW renumbers everything beneath it
#: silently, while failing to merge an affix produces a visibly wrong row count. Hence
#: 0.75 rather than something closer to 1.0, and hence :attr:`InventoryRefusalReason.
#: AMBIGUOUS_AFFIX_BAND` rather than a proximity tie-break.
AFFIX_HEIGHT_RATIO = 0.75

#: How much closer one neighbour must be than the other before the affix band's parent is
#: taken to be determined at all.
#:
#: **This is a decision rule about arbitrariness, NOT a measured property of typography,
#: and the difference matters.** Probe 51 asked whether `nearest / farthest` baseline
#: distance separates into two populations the way probe 6's column valley does, over
#: 1865 affix-shaped bands in the corpus. It does not: there is not one empty bin below
#: 1.0, and the absolute-gap histogram has no floor either. So no threshold here can be
#: called measured, and one that looked measured would be a tuned number wearing
#: borrowed authority.
#:
#: What ships instead is the weaker claim the data does support: when one candidate is at
#: least twice as close, the choice is not arbitrary; otherwise it is, and an arbitrary
#: choice REFUSES. The failure direction is what makes that safe -- a refusal reports a
#: wrong row count, where a silent tie-break would ship one.
AFFIX_PARENT_MARGIN = 0.5

#: How far two baselines may differ and still be the same printed line, in points.
#:
#: A real caption's fragments do NOT share one exact baseline: the target table's seven
#: caption fragments differ by fractions of a point. Anything that asks "is this fragment
#: on that line" must therefore use a tolerance, and it must be the SAME tolerance
#: everywhere -- the caption-anchor test and the orphaned-band test disagreeing by a
#: hundredth of a point is how a correct caption becomes seven orphaned bands, which is
#: exactly what the first run against the real document produced.
_BAND_TOLERANCE_PT = 0.5


class InventoryRefusalReason(StrEnum):
    """Why no inventory can be derived for a claimed footprint.

    Distinct from :class:`~carmel.services.pdf_cells.RegionRefusalReason`, which is
    about ONE value region's neighbourhood, and from
    :class:`~carmel.schemas.datasets.UnextractedReason`, which is about a located
    statement that did not become a claim. These are failures of DERIVATION: the grid
    itself could not be established.
    """

    EXTRACTION_UNAVAILABLE = "extraction_unavailable"
    """The fragment lane produced nothing usable. Which of the four ways lives on
    :class:`~carmel.services.pdf_fragments.FragmentAvailability`; it is a property of
    the document, not of this footprint."""

    PAGE_INCOMPLETE = "page_incomplete"
    """The footprint's page failed, or the document was truncated. A clean grid derived
    from a page that was only partly extracted is an artefact of the loss: the band that
    would have changed the row count may simply never have arrived."""

    FOOTPRINT_INSANE = "footprint_insane"
    """The claimed box is not a box: inverted or zero extent, a non-finite coordinate, or
    a page number no document has. Checked before anything reads the fragments, because
    every derivation below would otherwise silently operate on an empty set and report a
    grid of nothing."""

    CAPTION_ANCHOR_ABSENT = "caption_anchor_absent"
    """No fragment run at the claimed caption position carries the claimed caption text.

    This does not make a WRONG box impossible -- a caller can draw a box that starts one
    band too low and still name the caption correctly. It makes a box that MOVED
    detectable, which is what an independent replay needs in order to refute ordinal
    drift after the document, the engine, or this module's own derivation changes."""

    AMBIGUOUS_AFFIX_BAND = "ambiguous_affix_band"
    """A candidate affix band is horizontally interior to BOTH its neighbours, so the
    band it belongs to cannot be derived. Picking by proximity would change the cell's
    text and the row count at once, and nothing downstream would look wrong."""

    ORPHANED_BAND_ABOVE_THE_BOX = "orphaned_band_above_the_box"
    """A band sits between the caption and the box's top edge, inside its x-window.

    This is the ordinal-drift attack made detectable. A caller who lowers ``y_top`` by one
    band keeps a correct caption, keeps every value correctly grounded, and shifts every
    row ordinal by one -- and no check on the locator, the envelope or the values can see
    it. What CAN see it is that the excluded band is still there, orphaned between the
    caption that anchors the box and the box itself. A table whose first row genuinely
    sits far below its caption is refused too; that is the trade, and it fails toward
    refusal."""

    ORPHANED_BAND_BELOW_THE_BOX = "orphaned_band_below_the_box"
    """A band within one row pitch below ``y_bottom`` has a fragment inside a derived column.

    The bottom-edge twin of the reason above, and it was measured, not imagined: raising
    ``y_bottom`` on the real target deleted the phi row and returned a COMPLETE five-row
    inventory with no refusal. Worse, the target's own pre-registered footprint turned out
    to be truncated this way -- ``T (deg C)`` and ``P (atm)`` are rows of that table, and
    the box stopped 0.9 pt above them.

    The look-below is bounded by the table's OWN median row pitch, so no constant is
    introduced; the bound is derived from the same rows it is protecting. Any single mapped
    fragment CONTAINED in a derived column is enough, rather than a band that occupies
    several: a cut row that lives in one column (an affix-split cell) would otherwise pass,
    and that was one of the measured attacks. The cost is that prose starting at the left
    margin under a table can refuse a complete table -- the same trade the reason above
    makes, in the same direction."""

    TRUNCATED_COLUMN_BESIDE_THE_BOX = "truncated_column_beside_the_box"
    """Fragments excluded by ``x_start``/``x_end`` line up into a column across several rows.

    Shrinking ``x_end`` on the real target deleted an entire fuel mixture and returned a
    COMPLETE seven-row, TWO-column inventory with no refusal: a consumer would read a
    two-mixture study as a one-mixture study.

    Distance cannot catch it. The honest box already has the page's other prose column
    26.2 pt to its right, and the dropped column's fragments sit at 22.6-26.6 pt -- two
    populations that OVERLAP, so any threshold classifies one as the other. What separates
    them is ALIGNMENT: prose is one row's running text at increasing x, while a dropped
    column is many rows sharing one ``x_start``. So the test is a structural count -- an
    x-cluster of excluded fragments that draws from more than one derived row -- and it
    reuses :data:`COLUMN_VALLEY_PT` for the clustering rather than adding a constant.

    A single-row overflow is indistinguishable from prose under this rule and does NOT
    refuse. That is a known residual hole, not a closed one."""

    AMBIGUOUS_ROW_BESIDE_THE_BOX = "ambiguous_row_beside_the_box"
    """A fragment the box excluded is on the baseline of TWO derived rows at once.

    The sibling of :attr:`AMBIGUOUS_AFFIX_BAND`, at the other edge and for the same reason:
    the reason above decides whether an excluded fragment belongs to row *n*, and when two
    rows both claim it, picking one is a row-membership assertion with no evidence behind
    it. That matters because the truncated-column test counts DISTINCT ordinals -- awarding
    an ambiguous fragment to the row that already owns its neighbour collapses a two-row
    cluster to one and the refusal above silently does not fire.

    Rows are separated by more than :data:`_BAND_TOLERANCE_PT`, so this needs two rows
    between one and two tolerances apart with the fragment between them. Measured over the
    eight-paper corpus, 422 of 6886 adjacent band pairs (6.1%) sit inside that window, the
    closest 0.5027 pt apart, and 371 excluded fragments landed on two rows at once
    (probes/m1_band_ambiguity.py). It is ordinary two-column journal typesetting, not an
    exotic shape."""

    STRADDLING_FRAGMENT_AT_THE_BOX_EDGE = "straddling_fragment_at_the_box_edge"
    """A fragment on one of the box's own bands is cut by its left or right edge.

    The sibling of :attr:`~carmel.services.pdf_cells.RegionRefusalReason.STRADDLED`, which
    states the same rule one layer down: something was cut through rather than included or
    excluded, so the claimed boundary is not clean and nothing derived inside it can be
    trusted. The concept is deliberately the single-cell lane's rather than a new one --
    this is the same fault at a different scale.

    It closes a gap where two tests each assumed the other covered the case.
    :func:`_in_footprint` admits a fragment only when it is WHOLLY inside, so a straddler
    is never a member and never reaches banding, columns, cells, the unmapped check or the
    rotation check. :func:`_truncated_column_refusal` reads only fragments wholly BESIDE the
    box, skipping by construction the ones that overlap it. The straddler therefore left no
    trace in either direction. Measured on the real target with its one unmapped page-4
    glyph forced to mapped: the whole-table footprint returned a COMPLETE nine-row,
    three-column inventory with no refusal, and the pressure row read ``1e`` beside ``e8``
    -- the value between them was a single fragment ending 32.5 pt past the box's right
    edge, silently deleted.

    Two boundaries of the rule, both chosen for the direction they fail in:

    * Unmapped fragments are NOT skipped, unlike in :attr:`ORPHANED_BAND_BELOW_THE_BOX`.
      That reason asks whether the band below the box is a ROW, and a marker is not one;
      this one asks whether the edge cut something, and a marker is something. One inside
      the box refuses under :attr:`UNMAPPED_MEMBER`, so a bisected one must not pass.
    * Rotated fragments are skipped HERE, and caught elsewhere. Their ``x_start``/``x_end``
      do not bound them horizontally -- 23 of 257 rotated corpus fragments report an
      ``x_end`` outside their own page's mediabox, one 753.8 pt past a 595.3 pt page -- so
      an edge test on one would be a finding invented from a meaningless number. Asking the
      question without x is what :func:`_rotated_band_sharer_refusal` does, after derivation
      and scoped to the table's printed lines; that function carries the scoping argument,
      the measurement, and the one disclosed difference from ``pdf_cells``. A rotated
      fragment INSIDE the box refuses earlier still, under
      :attr:`ROTATED_OR_INSANE_FRAGMENT`.

    So a tally of THIS reason is a lower bound, short by the rotated fragments an edge cuts
    -- but they are no longer unguarded, and the reason they end up under is the honest one:
    this module cannot read their geometry."""

    COLUMN_STRUCTURE_UNRESOLVED = "column_structure_unresolved"
    """A row's own occupied blocks outnumber the columns derived across all rows.

    Columns come from ALIGNED emptiness -- an x-strip empty in EVERY row -- so a single
    row spanning the table's width erases every boundary beneath it. Measured on a
    fixture: a spanning header collapsed three columns into one, merged three separate
    cells into ``xyz`` at ``col=0``, and reported ``complete``. The inconsistency is
    cheap to see (some row alone resolves more blocks than the union does) and refusing
    on it is the only honest answer, because the column structure genuinely is not
    derivable from aligned emptiness once a row spans."""

    ROTATED_OR_INSANE_FRAGMENT = "rotated_or_insane_fragment"
    """This module cannot read a fragment's geometry, so it declines to read the box.

    Three call sites, one fault -- geometry that cannot be compared -- at three scopes:

    * ANY fragment on the box's page has a non-finite or backwards extent, or a non-finite
      or negative ``font_height``, whether or not the box contains it
      (:func:`_uncomparable_geometry_refusal`);
    * a fragment INSIDE the box is rotated (rotation ONLY -- the bad-extent clauses that
      used to sit here are gone, not merely quiet: the page-scoped gate applies the same
      predicate to a superset of these fragments and so reaches every one of them first);
    * a rotated fragment shares a printed line of the derived table
      (:func:`_rotated_band_sharer_refusal`).

    Rotated text has no meaningful x-interval in this module's model, and a backwards extent
    poisons every column derivation silently. ``pdf_cells`` guards this class explicitly;
    this module must not be the softer door into the same data.

    All three deliberately reuse this member rather than splitting off a sibling for
    "uncomparable OUTSIDER", because the inside/outside partition such a sibling would rest
    on is not knowable for exactly these fragments: a NaN extent is excluded from the box by
    :func:`_in_footprint` only because ``<`` against NaN is quietly False, not because the
    box does not contain it. Splitting a reason along a boundary the fault itself destroys
    would be a distinction the record cannot support."""

    UNATTACHABLE_AFFIX_BAND = "unattachable_affix_band"
    """A candidate affix band is interior to NEITHER neighbour.

    The parent is ABSENT here, where :attr:`AMBIGUOUS_AFFIX_BAND`'s is over-determined --
    a different fault with a different repair, so a different member. The shape that
    produces it is a TRAILING subscript: the ``2`` of ``CO2`` at a cell's right edge
    extends past its parent band's rightmost base glyph, so no interiority test can
    reach it. Letting it fall through to become a row of its own would renumber every
    row beneath it while every cell value stayed correct."""

    UNMAPPED_MEMBER = "unmapped_member"
    """A fragment inside the footprint has unmapped glyphs. Its text is a MARKER, not a
    character: a corpus rate table renders the minus sign of ``-1.0`` as ``/C0``, and a
    cell built from it reads ``+1.0``. Flagged by the fragment lane, refused here."""

    EMPTY = "empty"
    """No fragment lies inside the claimed box below the caption. An empty grid is not a
    table with no rows; it is evidence that the box is wrong."""


@dataclass(frozen=True, slots=True)
class ClaimedFootprint:
    """The box a caller drew, and the caption it says anchors it.

    Every field is caller-controlled. The caption fields are the only ones this module
    can check against the document, and that is precisely why they are required: without
    an anchor, a footprint is an unfalsifiable claim, and every ordinal derived inside it
    inherits that.
    """

    page: int
    x_start: float
    x_end: float
    y_top: float
    """Upper edge, in page space. The caption sits at or above it; body bands below."""
    y_bottom: float
    caption_text: str
    caption_x_start: float
    caption_baseline_y: float


@dataclass(frozen=True, slots=True)
class InventoryCell:
    """One derived cell. ``text`` is its members in reading order, never repaired."""

    row: int
    col: int
    text: str
    x_start: float
    x_end: float
    members: tuple[TextFragment, ...]


@dataclass(frozen=True, slots=True)
class InventoryRow:
    """One derived row: an ordinal, the baselines it was built from, and its anchor."""

    ordinal: int
    baseline_y: float
    merged_baselines: tuple[float, ...]
    """Baselines of the affix bands folded into this row, so a replayer recomputes the
    same merge or refuses. Empty when nothing was merged."""

    anchor_text: str
    """The row's LEFTMOST non-empty cell text.

    The user's rule is "band ordinal + label anchor", and the anchor's job is to pin the
    ordinal against drift. Requiring it to sit in column 0 would refuse a real table: the
    target's four oxidizer continuation rows have no label-column entry at all. So the
    anchor is the leftmost non-empty cell wherever it sits, which pins the row just as
    well and does not eat a legitimate row shape."""

    anchor_x_start: float


@dataclass(frozen=True, slots=True)
class InventoryRefusal:
    reason: InventoryRefusalReason
    detail: str
    """Short and human-readable. Never the document's text beyond the offending token."""


@dataclass(frozen=True, slots=True)
class CellInventory:
    """A derived grid, or a refusal. Never both, and never a partial one."""

    footprint: ClaimedFootprint
    rows: tuple[InventoryRow, ...]
    column_bounds: tuple[tuple[float, float], ...]
    cells: tuple[InventoryCell, ...]
    refusals: tuple[InventoryRefusal, ...]
    pypdf_version: str
    """The engine the geometry came from. Geometry IS the evidence here, and a pypdf that
    changed baseline semantics would keep every attribute name intact while returning
    different numbers, so the version travels with the result."""

    @property
    def complete(self) -> bool:
        """Derived, not stored, so it cannot disagree with :attr:`refusals`."""
        return not self.refusals

    def __post_init__(self) -> None:
        """Enforce the ONE combination whose prose would otherwise be a convention.

        Scope, stated so it is not read as more: this forbids a refusal sitting beside a
        derivation. It does NOT validate that cells cite existing rows and columns, that
        ordinals are dense, that members are non-empty, or that an anchor matches its
        row's leftmost cell. Those hold because :func:`build_inventory` constructs them
        that way, and a direct caller can still violate every one -- which is a real gap,
        recorded here rather than papered over by a docstring that claims otherwise.
        """
        if self.refusals and (self.cells or self.rows or self.column_bounds):
            raise ValueError(
                "a refused inventory carries no derivation: a partial grid is exactly what a "
                "caller would read as a table"
            )


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _footprint_refusal(footprint: ClaimedFootprint) -> InventoryRefusal | None:
    if not _is_finite(
        footprint.x_start,
        footprint.x_end,
        footprint.y_top,
        footprint.y_bottom,
        footprint.caption_x_start,
        footprint.caption_baseline_y,
    ):
        return InventoryRefusal(InventoryRefusalReason.FOOTPRINT_INSANE, "a coordinate is not finite")
    if footprint.page < 1:
        return InventoryRefusal(InventoryRefusalReason.FOOTPRINT_INSANE, f"page {footprint.page} is below 1")
    if footprint.x_end <= footprint.x_start:
        return InventoryRefusal(InventoryRefusalReason.FOOTPRINT_INSANE, "x extent is not positive")
    if footprint.y_top <= footprint.y_bottom:
        return InventoryRefusal(InventoryRefusalReason.FOOTPRINT_INSANE, "y extent is not positive")
    if footprint.y_top >= footprint.caption_baseline_y:
        # The box would contain its own caption, which then becomes row 0 and shifts every
        # ordinal beneath it. Measured on a fixture before this check existed: `y_top`
        # raised above the caption produced three rows whose row 0 was the caption text,
        # with no refusal anywhere. The docstring said the caption sits at or above the
        # top edge; nothing enforced it, and prose the type does not enforce is a
        # convention.
        return InventoryRefusal(
            InventoryRefusalReason.FOOTPRINT_INSANE,
            "the box contains its own caption, which would become row 0",
        )
    if not (footprint.x_start <= footprint.caption_x_start <= footprint.x_end):
        return InventoryRefusal(
            InventoryRefusalReason.FOOTPRINT_INSANE,
            "the claimed caption does not start inside the claimed box",
        )
    return None


def _caption_anchored(extraction: FragmentExtraction, footprint: ClaimedFootprint) -> bool:
    """Whether the claimed caption text is actually printed at the claimed position.

    Compared against the concatenation of the fragments on the caption's own baseline,
    with whitespace collapsed on both sides: a caption arrives in several fragments and a
    publisher may or may not emit the spaces between them, so an exact string match
    against one fragment would refuse every real caption.

    **Scoped to the footprint's x-range, and that is load-bearing.** The first run against
    a real two-column paper refused a correct caption because the OTHER column's prose sat
    0.4 pt from the caption's baseline and concatenated into it. A baseline alone does not
    identify a line on a multi-column page; the pairing of baseline and x-window does. A
    caption that genuinely overflows the claimed box refuses, which is right -- the box is
    then not the one the caption anchors.
    """
    band = [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and math.isclose(f.baseline_y, footprint.caption_baseline_y, abs_tol=_BAND_TOLERANCE_PT)
        and f.x_start >= footprint.x_start
        and f.x_end <= footprint.x_end
    ]
    if not band:
        return False
    # `caption_x_start` is checked against the DOCUMENT here, not merely bounds-checked.
    # `_footprint_refusal` only asks that it fall inside the claimed box -- a test between
    # two caller-supplied numbers the document never sees. Measured consequence: shrinking
    # `x_start` to drop the label column refused, but only because the stale
    # `caption_x_start` now fell outside the new box. A caller who moved both would have
    # passed. That is a coincidence, not a guard.
    ordered = sorted(band, key=lambda f: f.x_start)
    if not math.isclose(ordered[0].x_start, footprint.caption_x_start, abs_tol=_BAND_TOLERANCE_PT):
        return False
    printed = "".join(f.text for f in ordered)
    claimed = footprint.caption_text
    return "".join(printed.split()) == "".join(claimed.split())


def _in_footprint(extraction: FragmentExtraction, footprint: ClaimedFootprint) -> list[TextFragment]:
    return [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and f.x_start >= footprint.x_start
        and f.x_end <= footprint.x_end
        and footprint.y_bottom <= f.baseline_y <= footprint.y_top
    ]


def _has_comparable_geometry(fragment: TextFragment) -> bool:
    """The ONE invariant every table comparison in this module rests on.

    A fragment must carry finite, comparable table geometry -- finite ``x_start``,
    ``x_end``, ``baseline_y`` and ``font_height``; ``x_start <= x_end``; a strictly positive
    ``font_height`` -- before any footprint comparison or row derivation reads it.

    This exists as one predicate because the alternative was tried four times and failed four
    times. Each of ``x_start``/``x_end``, then ``baseline_y``, then ``font_height``'s NaN
    case, then ``font_height``'s ZERO case was found and patched separately, each time by a
    reviewer rather than by the module, and each time the same shape: a value that makes a
    comparison quietly False, a fragment that satisfies no test in either direction, and a
    COMPLETE inventory with a row lost or invented.

    **This closes a FAULT, not a surface, and the difference was itself got wrong here once.**
    An earlier version of this docstring claimed the enumeration was exhaustive -- four
    floats, ``page`` compared only for equality, therefore no fifth patch possible. That
    conflates exhaustiveness over FIELDS with exhaustiveness over ways a number can mislead
    this module, and it is refuted by construction: :func:`_looks_like_affix` is a RATIO with
    no upper bound, so when a header is merely TALLER than the row beneath it, that row reads
    as affix-shaped against the header and is folded INTO it -- the header is the parent and
    the ordinary row beneath is what disappears. ``font_height`` 8.0 gives three rows; 10.7
    gives two, the first anchored ``HeadBbb``; and no value involved is non-finite, negative
    or backwards. :func:`_has_comparable_geometry` cannot reach that and must not be widened
    to try: a magnitude ratio unbounded above is a different fault with a different repair,
    tracked as **I-005** in the campaign ledger. What this predicate covers is comparability,
    completely; what it does not cover is every other way a finite number can be wrong.

    Three boundaries chosen deliberately:

    * **Zero WIDTH stays legal.** ``x_start == x_end`` is a real fragment, not a broken one:
      the corpus's ``/C14`` degree sign is exactly that, at ``x=61.7952-61.7952``. Refusing
      it here would refuse a document over a mark the page really carries.
    * **Zero HEIGHT does not.** It feeds a magnitude comparison rather than bounding
      anything: :func:`_looks_like_affix` guards ``reference <= 0`` for the NEIGHBOUR but not
      for the band, so a band of zero-height fragments is affix-shaped against every
      neighbour and is folded away. Demonstrated on two ordinary rows, the second at
      ``font_height=0.0``: COMPLETE, ONE row, anchor ``AaaBbb`` -- a whole row swallowed by
      the row above it, with every surviving cell correctly grounded.
    * **Rotation is NOT here.** A rotated fragment's numbers are finite and comparable; they
      simply do not mean what this module needs them to mean, which is a different fault
      with a different scope -- see :func:`_rotated_band_sharer_refusal`.
    """
    return (
        _is_finite(fragment.x_start, fragment.x_end, fragment.baseline_y, fragment.font_height)
        and fragment.x_start <= fragment.x_end
        and fragment.font_height > 0.0
    )


def _uncomparable_geometry_refusal(
    extraction: FragmentExtraction, footprint: ClaimedFootprint
) -> InventoryRefusal | None:
    """Refuse when any fragment on the box's page has geometry that cannot be compared.

    Non-finite or backwards extents fail open in the most complete way there is: every
    ``<`` against NaN is quietly False, so such a fragment is not a member (:func:`_in_footprint`
    excludes it), not a straddler (:func:`_straddle_refusal`'s edge tests are both False), not
    an orphan above or below, and not part of any excluded-column cluster. It vanishes exactly
    as the straddler did, one comparison over -- and unlike the straddler it leaves no
    trace anywhere, because there is no comparison it could have satisfied.

    The predicate is :func:`_has_comparable_geometry`, and the reason it is a named
    invariant rather than a list of clauses is written there. ``font_height`` is in it and
    does NOT bound the box: it feeds the ``<=`` in :func:`_looks_like_affix`, where a NaN
    fabricates a row (COMPLETE, THREE rows, a cell reading ``CO`` and a spurious ``2``) and a
    zero loses one (COMPLETE, ONE row, anchor ``AaaBbb``). Both were reproduced against the
    code that shipped without the guard.

    Scoped to the PAGE, not the box's y-window, and that is the whole point: a NaN
    ``baseline_y`` fails ``y_bottom <= baseline_y <= y_top`` too, so a window-scoped check
    would read only the fragments that already survived the test NaN defeats. ``pdf_cells``
    reaches the same conclusion for the same reason at ``UNCOMPARABLE_NEIGHBOUR``.

    Blank fragments are NOT skipped here, and that diverges from BOTH neighbours: this
    lane's own :func:`_straddle_refusal` skips them, and ``pdf_cells`` drops them from
    ``others`` before its ``UNCOMPARABLE_NEIGHBOUR`` test, so its equivalent never sees a
    blank one either. Those two ask whether something was CUT or who the nearest neighbour
    IS, and a bare space text-show operation is neither. This one asks whether the page's
    geometry can be read at all, and an unreadable number is unreadable whether or not a
    glyph was drawn with it. So the conclusion ``pdf_cells`` shares is the PAGE scoping and
    its reason; the blank rule is this lane's own.

    Measured: 0 of 78178 fragments across the eight-paper corpus fail
    :func:`_has_comparable_geometry` -- none has a non-finite or backwards extent, and none
    a non-finite, negative or zero ``font_height`` -- so this costs nothing observed. Like
    ``INVALID_GEOMETRY`` in the single-cell lane, it guards the SHAPE of the input rather
    than an observed instance.
    """
    unreadable = [f for f in extraction.fragments if f.page == footprint.page and not _has_comparable_geometry(f)]
    if not unreadable:
        return None
    return InventoryRefusal(
        InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT,
        f"{len(unreadable)} fragment(s) on the box's page do not carry comparable geometry",
    )


def _straddle_refusal(extraction: FragmentExtraction, footprint: ClaimedFootprint) -> InventoryRefusal | None:
    """Refuse when a side edge falls INSIDE a fragment sitting on one of the box's bands.

    The y-window is :func:`_in_footprint`'s, exactly, and that pairing is the definition:
    a straddler is a fragment that would have been a member but for the x test. Scoping it
    to the page instead would refuse every table with a full-width paragraph beneath it,
    which is a line the box never claimed and never lost.

    ``x_start < edge < x_end`` is strict on both sides, so a cell that begins exactly on
    ``x_start`` or ends exactly on ``x_end`` touches the boundary without being cut by it
    and stays a member -- membership is inclusive of the edge, and this must not narrow it.

    Reaches here only once :func:`_uncomparable_geometry_refusal` has passed, so both edge
    tests are comparisons between real numbers. Rotated fragments are skipped because their
    extent is not one; :func:`_rotated_band_sharer_refusal` asks after them later, without
    reading x.
    """
    cut = [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and f.text.strip()
        and not f.rotated
        and footprint.y_bottom <= f.baseline_y <= footprint.y_top
        and (f.x_start < footprint.x_start < f.x_end or f.x_start < footprint.x_end < f.x_end)
    ]
    if not cut:
        return None
    return InventoryRefusal(
        InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE,
        f"{len(cut)} fragment(s) on the box's own bands are cut by its side edge",
    )


def _bands(fragments: list[TextFragment]) -> list[tuple[float, list[TextFragment]]]:
    """Group by baseline, descending in page space (reading order, top first).

    Grouped by a single-linkage sweep at :data:`_BAND_TOLERANCE_PT` -- the same
    tolerance every other baseline comparison in this module uses, which is the
    module's own stated premise.

    This was ``round(baseline_y, 1)``, and a rounding bin is not a tolerance. Two
    failures, and the second is the one that bites: fragments genuinely more than
    0.1pt apart within a printed line split, AND fragments arbitrarily CLOSE
    split whenever the bin edge happens to fall between them -- 100.04 and 100.06
    are 0.02pt apart and round to different bands. Measured over the eight-paper
    corpus through the production fragment lane, the bin splits 685 printed lines
    that this tolerance holds together, 107 of them at a bin edge, the closest
    pair 1.14e-04 pt apart (probes/m1_band_bins.py).

    A split line is not a cosmetic problem here: a band IS a row ordinal, so it
    renumbers every row beneath it and moves the cell a citation resolves to.
    Single-linkage does not collapse the page -- the same measurement gives 6959
    swept lines against 7679 bins -- because rows sit a pitch apart and an affix
    baseline sits further off than this tolerance, which is what leaves
    :func:`_merge_affix_bands` its work.
    """
    if not fragments:
        return []
    ordered = sorted(fragments, key=lambda f: f.baseline_y)
    clusters: list[list[TextFragment]] = [[ordered[0]]]
    for fragment in ordered[1:]:
        if fragment.baseline_y - clusters[-1][-1].baseline_y <= _BAND_TOLERANCE_PT:
            clusters[-1].append(fragment)
        else:
            clusters.append([fragment])
    # The band's baseline is its topmost, not a rounded stand-in: an exact value taken
    # from a real fragment, so the affix merge's gap arithmetic compares measurements
    # rather than bin labels.
    return [(max(f.baseline_y for f in cluster), cluster) for cluster in reversed(clusters)]


def _is_adjacent_to(band: list[TextFragment], other: list[TextFragment]) -> bool:
    """Whether ``band`` overlaps or abuts ``other``'s horizontal extent.

    OVERLAP, not containment, and the difference was a live silent corruption. A
    trailing subscript -- the ``2`` of ``CO2`` at a cell's right edge -- extends past its
    parent's rightmost base glyph, so a containment test excludes its true parent and
    leaves whichever neighbour happens to be wider. Measured on a fixture: the ``2`` folded
    into an unrelated row 13 pt away and appended its text to that row's anchor, with no
    refusal. Adjacency admits both neighbours as candidates and lets the vertical margin
    decide, which is the only axis that actually separates them.

    Abutment is allowed up to :data:`COLUMN_VALLEY_PT`, the width at which an x-gap stops
    being intra-cell and becomes a column boundary; beyond it the band is in a different
    column and is not this row's affix at all.
    """
    if not other:
        return False
    left = min(f.x_start for f in other) - COLUMN_VALLEY_PT
    right = max(f.x_end for f in other) + COLUMN_VALLEY_PT
    return all(f.x_end >= left and f.x_start <= right for f in band)


def _looks_like_affix(band: list[TextFragment], neighbour: list[TextFragment]) -> bool:
    if not neighbour:
        return False
    reference = max(f.font_height for f in neighbour)
    if reference <= 0:
        return False
    return max(f.font_height for f in band) <= AFFIX_HEIGHT_RATIO * reference


def _merge_affix_bands(
    bands: list[tuple[float, list[TextFragment]]],
) -> tuple[list[tuple[float, list[TextFragment], list[float]]], InventoryRefusal | None]:
    """Fold affix-only bands into the band they are interior to.

    A subscript sits on its own baseline, so a raw band is not a row. The merge is
    recorded (the folded baselines travel on the row) rather than performed silently, and
    a band interior to BOTH neighbours refuses: choosing by proximity would change the
    cell text and the row count together, with nothing downstream looking wrong.
    """
    # Pass 1: decide each band's parent WITHOUT mutating anything. A band is its own row
    # (parent None) unless it is affix-shaped and interior to exactly one neighbour.
    parents: list[int | None] = []
    for index, (y, band) in enumerate(bands):
        above = bands[index - 1][1] if index > 0 else []
        below = bands[index + 1][1] if index + 1 < len(bands) else []
        affix_shaped = _looks_like_affix(band, above) or _looks_like_affix(band, below)
        fits_above = _looks_like_affix(band, above) and _is_adjacent_to(band, above)
        fits_below = _looks_like_affix(band, below) and _is_adjacent_to(band, below)
        if fits_above and fits_below:
            # A table's columns are ALIGNED by construction, so a subscript is adjacent to
            # the row above AND the row below and abuts the same glyph in each. Horizontal
            # geometry cannot separate them; vertical distance can, but only when one
            # candidate is clearly nearer -- see AFFIX_PARENT_MARGIN.
            gap_above = abs(bands[index - 1][0] - y)
            gap_below = abs(y - bands[index + 1][0])
            near, far = min(gap_above, gap_below), max(gap_above, gap_below)
            if far <= 0 or near > AFFIX_PARENT_MARGIN * far:
                return [], InventoryRefusal(
                    InventoryRefusalReason.AMBIGUOUS_AFFIX_BAND,
                    f"a band at y={y} is adjacent to both neighbours and no nearer to either",
                )
            fits_above, fits_below = gap_above < gap_below, gap_below < gap_above
        if affix_shaped and not (fits_above or fits_below):
            return [], InventoryRefusal(
                InventoryRefusalReason.UNATTACHABLE_AFFIX_BAND,
                f"an affix-shaped band at y={y} is adjacent to neither neighbouring band",
            )
        parents.append(index - 1 if fits_above else index + 1 if fits_below else None)

    # A chain of affixes (an affix whose parent is itself an affix) is not a shape this
    # module can resolve into a row, and following the chain would let one ambiguous band
    # renumber a row two positions away. Refuse instead.
    for index, parent in enumerate(parents):
        if parent is not None and parents[parent] is not None:
            return [], InventoryRefusal(
                InventoryRefusalReason.AMBIGUOUS_AFFIX_BAND,
                f"a band at y={bands[index][0]} would fold into another folded band",
            )

    # Pass 2: build the rows in reading order, folding each affix into its parent.
    merged: list[tuple[float, list[TextFragment], list[float]]] = []
    slot_of: dict[int, int] = {}
    for index, (y, band) in enumerate(bands):
        if parents[index] is None:
            slot_of[index] = len(merged)
            merged.append((y, list(band), []))
    for index, (y, band) in enumerate(bands):
        parent = parents[index]
        if parent is None:
            continue
        slot = slot_of[parent]
        parent_y, members, folded = merged[slot]
        merged[slot] = (parent_y, [*members, *band], [*folded, y])
    return merged, None


def _column_bounds(
    rows: list[tuple[float, list[TextFragment], list[float]]],
) -> list[tuple[float, float]]:
    """Column blocks from the aligned-emptiness valley.

    Occupied x-intervals are unioned across ALL rows, and a run of x that no row occupies
    and that is at least :data:`COLUMN_VALLEY_PT` wide separates two columns. Unioning
    across rows is what makes it an ALIGNED emptiness: a gap that exists on one line and
    not the next is a word space, and probe 6 measured those as a population that does
    not reach 4 pt.
    """
    spans = sorted((f.x_start, f.x_end) for _, members, _ in rows for f in members)
    if not spans:
        return []
    blocks: list[list[float]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start - blocks[-1][1] >= COLUMN_VALLEY_PT:
            blocks.append([start, end])
        else:
            blocks[-1][1] = max(blocks[-1][1], end)
    return [(a, b) for a, b in blocks]


def _row_pitch(rows: list[InventoryRow], members: list[TextFragment], footprint: ClaimedFootprint) -> float:
    """How far apart this table's own printed lines are, in points.

    Derived from the table being protected rather than declared as a constant, so the
    look-below in :func:`_orphan_below_refusal` is bounded by the thing it is bounding.
    The MEDIAN, not the mean: a mean is dragged upward by the caption-to-first-row gap,
    widening the window into the prose beneath.

    **The caption's baseline is folded in**, which is what makes a ONE-ROW table defensible.
    Row gaps alone give no pitch at all below two rows, and the fallback that stood here --
    the tallest member's rendered height -- was measured to be far too small: a synthetic
    one-row box failed to notice a row cut off 80 pt beneath it, silently, in the permissive
    direction. The caption is a printed line of the same table at a known position, so the
    gap to the first row is a real line spacing rather than a stand-in. It is systematically
    the LARGEST gap (14.3 pt against a 10.0 pt body pitch on the real target), and including
    it can only widen the window, which is the safe direction. On that table the median is
    10.0 pt either way, so the measured case is unchanged.
    """
    baselines = [footprint.caption_baseline_y, *(row.baseline_y for row in rows)]
    gaps = [above - below for above, below in zip(baselines, baselines[1:], strict=False)]
    if gaps:
        return statistics.median(gaps)
    return max((f.font_height for f in members), default=0.0)


def _orphan_below_refusal(
    extraction: FragmentExtraction,
    footprint: ClaimedFootprint,
    bounds: list[tuple[float, float]],
    pitch: float,
) -> InventoryRefusal | None:
    """Refuse when the box's bottom edge cut a row off, the way the top edge is guarded.

    CONTAINMENT in a derived column is the test, not proximity: prose runs across the
    gutters this table's columns are separated by, so a running-text fragment does not fit
    inside one; a cut table row's cell does, because the column was derived from that very
    alignment. Unmapped fragments are skipped -- they carry markers rather than text, so
    they are not a row (the real target's is a raised zero-width degree sign), and one
    inside the box refuses under :attr:`InventoryRefusalReason.UNMAPPED_MEMBER` anyway.
    """
    if pitch <= 0:
        return None
    cut = [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and f.text.strip()
        and f.glyph_mapping is GlyphMapping.MAPPED
        and footprint.y_bottom - pitch <= f.baseline_y < footprint.y_bottom
        and any(left <= f.x_start and f.x_end <= right for left, right in bounds)
    ]
    if not cut:
        return None
    return InventoryRefusal(
        InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX,
        f"{len(cut)} fragment(s) within {pitch:.1f} pt below the box sit inside a derived column",
    )


def _rotated_band_sharer_refusal(
    extraction: FragmentExtraction,
    footprint: ClaimedFootprint,
    rows_raw: list[tuple[float, list[TextFragment], list[float]]],
) -> InventoryRefusal | None:
    """Refuse when a rotated fragment shares a printed line of the derived table.

    A rotated fragment's ``x_start``/``x_end`` do not bound it horizontally -- measured, 23
    of 257 rotated corpus fragments report an ``x_end`` outside their own page's mediabox,
    one 753.8 pt past a 595.3 pt page, because it is ``x_start`` plus advance widths laid
    along the ROTATED axis and never projected onto x. So it can be neither ruled out as a
    straddler nor recorded as one, and every guard that reads x -- :func:`_straddle_refusal`,
    :func:`_truncated_column_refusal`, :func:`_orphan_below_refusal` -- skips it. That is
    correct individually and, left alone, adds up to a hole: on one of the table's own
    printed lines, a rotated fragment vanishes the way the straddler did.

    So the question is asked where x is not needed: against the table's printed LINES. Every
    baseline a row carries counts, its own and the affix baselines folded into it, because a
    subscript's line is a printed line too and the merge is what would otherwise hide it.

    **The scope is argued structurally AND the corpus discriminates -- measured at THIS
    ordering, which is the only ordering the numbers mean anything at.** Of probe 50's four
    registered footprints, which check each one refuses at decides whether it ever reaches
    this line at all:

    ===========================  ==========  =======================================
    footprint                    y_bottom    refuses at
    ===========================  ==========  =======================================
    ``WHOLE_TABLE`` (p4)              45.0    :func:`_straddle_refusal` -- never reaches
    ``TRUNCATED`` (p4)                65.0    the look-below guard -- REACHES
    ``SHOCK_TUBE_CAPTION_ONLY``      680.0    the orphan-above guard -- never reaches
    ``SHOCK_TUBE_CAPTION_INSIDE``    680.0    :attr:`COLUMN_STRUCTURE_UNRESOLVED` -- REACHES
    ===========================  ==========  =======================================

    So two reach, and band-scoped both pass: the probe reports 4/4. Substituting a
    WINDOW-scoped variant here instead gives **3/4** -- ``SHOCK_TUBE_CAPTION_INSIDE`` flips
    from :attr:`COLUMN_STRUCTURE_UNRESOLVED` to a rotated refusal raised on a page
    watermark, a correct finding replaced by a false one.

    Which of the two the hoist actually added is the sharper point. ``TRUNCATED`` reached
    at both orderings, because the look-below guard has always run after this line.
    ``SHOCK_TUBE_CAPTION_INSIDE`` reached only once this check moved above the
    column-structure test -- and it is precisely the footprint that flips. The hoist did not
    merely change a count; it put the one discriminating footprint in front of this
    function.

    That is also why a window variant measured 4/4 while this check sat below
    column-structure: nothing could be masked there, because the better refusal had already
    fired. **Any measurement quoted in this docstring must be re-taken against the shipped
    order of checks. This paragraph has carried a wrong statement three times -- the masking
    claim, then its retraction, then the identity of the p4 footprint -- and all three were
    the same claim in different clothes: which footprints reach this line, at which ordering.**

    What decides the scope independently of that is what a LINE is in this module. Row
    identity here is
    :data:`_BAND_TOLERANCE_PT` via :func:`_bands` -- the same constant that decides two
    fragments are one row -- so the question "does this rotated fragment share a printed
    line" has exactly one available answer, and it is not the footprint's y-window. The
    window is caller-supplied and 80, 100, 55 and 75 pt tall on the registered footprints;
    it says where the box is, never where a line is. The tolerance is therefore derived
    rather than chosen: widening it would merge distinct rows long before it changed this
    check, which is what makes it a scope and not a knob.

    One disclosed difference from ``pdf_cells``, of tolerance and not of rule: its band is
    ``BASELINE_BAND = 4.0`` pt against this one's 0.5, and each lane uses the constant that
    defines a line in it. The difference is not a corner case -- of 266 non-blank rotated
    corpus fragments, **61 sit within 0.5 pt of a printed baseline and 227 within 4.0 pt**,
    so the two rules disagree about roughly 166 fragments -- the ``Downloaded from
    http://asmedig...`` watermark, 1.92 pt from the nearest printed baseline, is one of many
    rather than a curiosity (probes/i004_bad_geometry_census.py).

    Two denominators appear above and neither is a typo, but they are NOT one filter apart.
    The corpus holds 267 rotated fragments; 266 are non-blank, which is the base for the
    band counts; 257 sit on a page whose mediabox is readable, which is the base for the
    extent measurement, since the other 10 would otherwise be checked against a guessed page
    width and the guess would be what decided whether an extent counts as off-page. The two
    filters are independent and the intersection is 256, so neither figure is reachable from
    the other by subtracting one number.
    """
    printed_baselines = [y for baseline_y, _, folded in rows_raw for y in (baseline_y, *folded)]
    sharers = [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and f.rotated
        and f.text.strip()
        and any(abs(f.baseline_y - y) <= _BAND_TOLERANCE_PT for y in printed_baselines)
    ]
    if not sharers:
        return None
    return InventoryRefusal(
        InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT,
        f"{len(sharers)} rotated fragment(s) share a printed line of the table",
    )


def _truncated_column_refusal(
    extraction: FragmentExtraction,
    footprint: ClaimedFootprint,
    rows: list[InventoryRow],
    rows_raw: list[tuple[float, list[TextFragment], list[float]]],
) -> InventoryRefusal | None:
    """Refuse when fragments the box's sides excluded align into a column across rows.

    Clustered on ``x_start`` at :data:`COLUMN_VALLEY_PT` -- the same width that separates
    two columns INSIDE the box, used here to decide whether two excluded fragments belong
    to the same excluded column. A cluster drawing from more than one derived row is a
    column the box cut off; a cluster from one row cannot be told from that row's own
    overflow, or from the page's other prose column, and does not refuse.

    **Which row an excluded fragment belongs to is decided against the row's whole baseline
    extent, and ambiguity refuses.** Both halves of that replaced a fail-open, and both are
    about the same thing -- this guard counts DISTINCT ordinals, so anything that quietly
    loses an ordinal turns a two-row cluster into a one-row cluster and suppresses the
    refusal:

    - The match used to be against ``row.baseline_y`` alone, but that is
      ``max(member baselines)``, and single linkage only bounds the gap between CONSECUTIVE
      members -- a row can span more than :data:`_BAND_TOLERANCE_PT` in total. An excluded
      fragment on the LOW end of such a row is further than the tolerance from the row's
      representative and matched NOTHING, vanishing from the guard. Measured on the corpus:
      14 of 6959 bands span more than the tolerance, the widest 1.1178 pt, and 1164 excluded
      fragments sat on a row's real membership while matching no representative
      (probes/m1_band_span.py). Comparing against ``[min, max]`` widened by the tolerance
      asks the question the guard actually means -- is this fragment on that printed line.
    - The first match then won, by ``dict`` insertion order, which is descending baseline --
      so an ambiguous fragment silently went to the row ABOVE. See
      :attr:`InventoryRefusalReason.AMBIGUOUS_ROW_BESIDE_THE_BOX` for the measurement.

    ``rows_raw`` is passed rather than read off :class:`InventoryRow` because the extent is
    a property of the members, and putting it on the row would enlarge the stored record for
    a fact every replay recomputes anyway: :func:`verify_inventory_record` re-derives through
    :func:`build_inventory`, so the payload never needs to carry it.
    """
    extents: list[tuple[int, float, float]] = []
    for row in rows:
        # `ordinal` IS the index into `rows_raw` -- `build_inventory` enumerates it there --
        # and a raw band that yielded no cell is simply absent from `rows`, never renumbered.
        baselines = [member.baseline_y for member in rows_raw[row.ordinal][1]]
        extents.append((row.ordinal, min(baselines), max(baselines)))

    excluded: list[tuple[float, int]] = []
    for f in extraction.fragments:
        if f.page != footprint.page or not f.text.strip() or f.rotated:
            continue
        if not (f.x_end <= footprint.x_start or f.x_start >= footprint.x_end):
            continue
        matched = [
            ordinal
            for ordinal, low, high in extents
            if low - _BAND_TOLERANCE_PT <= f.baseline_y <= high + _BAND_TOLERANCE_PT
        ]
        if len(matched) > 1:
            return InventoryRefusal(
                InventoryRefusalReason.AMBIGUOUS_ROW_BESIDE_THE_BOX,
                f"a fragment beside the box at y={f.baseline_y} is on the baseline of rows {matched}",
            )
        if matched:
            excluded.append((f.x_start, matched[0]))

    cluster: list[tuple[float, int]] = []
    for x_start, ordinal in sorted(excluded):
        if cluster and x_start - cluster[-1][0] >= COLUMN_VALLEY_PT:
            if len({o for _, o in cluster}) > 1:
                return InventoryRefusal(
                    InventoryRefusalReason.TRUNCATED_COLUMN_BESIDE_THE_BOX,
                    f"{len(cluster)} fragment(s) beside the box align across "
                    f"{len({o for _, o in cluster})} of its rows",
                )
            cluster = []
        cluster.append((x_start, ordinal))
    if len({o for _, o in cluster}) > 1:
        return InventoryRefusal(
            InventoryRefusalReason.TRUNCATED_COLUMN_BESIDE_THE_BOX,
            f"{len(cluster)} fragment(s) beside the box align across {len({o for _, o in cluster})} of its rows",
        )
    return None


def build_inventory(extraction: FragmentExtraction, footprint: ClaimedFootprint) -> CellInventory:
    """Derive the grid inside ``footprint``, or refuse.

    The order of the checks below is a contract, not a convenience. Document-level
    refusals come first because nothing about the document may be claimed once they
    fire; the footprint's own sanity comes next, because every later step would silently
    operate on an empty fragment set and report a grid of nothing; the caption anchor
    comes before any derivation, because an unanchored box makes every ordinal derived
    inside it unfalsifiable.
    """

    def refused(refusal: InventoryRefusal) -> CellInventory:
        return CellInventory(
            footprint=footprint,
            rows=(),
            column_bounds=(),
            cells=(),
            refusals=(refusal,),
            pypdf_version=extraction.pypdf_version,
        )

    if not extraction.available:
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.EXTRACTION_UNAVAILABLE,
                f"the fragment lane is unavailable for this document ({extraction.status})",
            )
        )
    if extraction.truncated or any(p.page == footprint.page for p in extraction.page_failures):
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.PAGE_INCOMPLETE,
                f"page {footprint.page} was not completely extracted",
            )
        )
    if extraction.lossy:
        # UNLOCATABLE loss: the extraction admits something was lost but names no page and
        # no truncation, so this footprint cannot be excluded from it. `region_refusals`
        # already refuses this shape, and a module that reads the same fragments must not
        # be the softer door -- a grid derived from a document that quietly lost a band is
        # a grid with a wrong row count and no way to know it.
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.PAGE_INCOMPLETE,
                "the extraction is lossy without naming a page, so this box cannot be excluded",
            )
        )
    insane = _footprint_refusal(footprint)
    if insane is not None:
        return refused(insane)
    # Comparable geometry is the precondition of every check below, so it is established
    # before the FIRST of them -- including the caption anchor, which is itself a geometric
    # comparison (`math.isclose` against the claimed baseline). Behind this gate, a caption
    # fragment with a NaN baseline reported CAPTION_ANCHOR_ABSENT: true, in that the anchor
    # was not found, but the reason names the wrong fault and points a reader at the
    # footprint's caption fields when the document's geometry is what is unreadable. That is
    # the same reason-attribution class as the straddle and rotated placements.
    #
    # Page-scoped, for the reason `pdf_cells` gives at UNCOMPARABLE_NEIGHBOUR: a NaN BASELINE
    # never reaches the box's y-window at all, so a window-scoped check would inspect only
    # the fragments that already passed the test NaN defeats.
    uncomparable = _uncomparable_geometry_refusal(extraction, footprint)
    if uncomparable is not None:
        return refused(uncomparable)

    if not _caption_anchored(extraction, footprint):
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.CAPTION_ANCHOR_ABSENT,
                "the claimed caption text is not printed at the claimed baseline",
            )
        )

    orphaned = [
        f
        for f in extraction.fragments
        if f.page == footprint.page
        and footprint.y_top < f.baseline_y < footprint.caption_baseline_y
        # The caption's OWN fragments are not orphans. A real caption's fragments do not
        # share one exact baseline -- the target's seven differ by fractions of a point --
        # so a strict comparison against the claimed baseline classifies most of the
        # caption as a band the caller cut off the top. Same tolerance as
        # `_caption_anchored`, because it must be the same band.
        and not math.isclose(f.baseline_y, footprint.caption_baseline_y, abs_tol=_BAND_TOLERANCE_PT)
        and f.x_end >= footprint.x_start
        and f.x_start <= footprint.x_end
    ]
    if orphaned:
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.ORPHANED_BAND_ABOVE_THE_BOX,
                f"{len(orphaned)} fragment(s) sit between the caption and the box's top edge",
            )
        )

    # The SIDE edges are settled here, before the box's contents are read, because a
    # boundary that cuts through a fragment makes the membership set itself untrustworthy
    # -- the same footing the orphan check above stands on, and unlike the edge guards at
    # the bottom of this function, which genuinely need the derivation they check. Running
    # it after `unmapped` instead would make the reason a caller reads depend on whether
    # some OTHER fragment mapped: on the real target the whole-table footprint carries both
    # faults, and repairing the glyph would have taken the straddler's refusal away again.
    straddling = _straddle_refusal(extraction, footprint)
    if straddling is not None:
        return refused(straddling)

    inside = _in_footprint(extraction, footprint)
    # Rotation ONLY. The non-finite and backwards clauses that used to stand here were dead
    # the moment the page-scoped gate went in above: it applies the same predicate to a
    # superset of these fragments, so nothing could reach here carrying one.
    rotated_members = [f for f in inside if f.rotated]
    if rotated_members:
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT,
                f"{len(rotated_members)} fragment(s) inside the box are rotated",
            )
        )
    if not inside:
        return refused(InventoryRefusal(InventoryRefusalReason.EMPTY, "no fragment lies inside the claimed box"))

    # The ROW derivation is hoisted above the unmapped check, and it is safe to hoist
    # because it reads geometry ONLY: `_bands` reads `baseline_y`, `_merge_affix_bands`
    # reads `baseline_y`, `font_height` and the x-extents, and no glyph is consulted until
    # cell text is built further down -- which is why THAT stays where it is.
    #
    # It is hoisted because the boundary guarantee the straddle check earns is worthless if
    # its rotated twin does not share it. Below the unmapped check, one document with an
    # unmapped member inside the box would report `straddling_fragment_at_the_box_edge` for
    # an upright fragment the edge cuts and `unmapped_member` for a rotated one on a row's
    # own printed line: same document, same class of fault, and an unrelated glyph choosing
    # which reason the caller reads.
    #
    # The affix refusals move with the derivation, since they ARE its outcome. That is a
    # deliberate consequence and the same rule covers it: what the geometry alone decides is
    # settled before anything that needs a glyph to be readable.
    rows_raw, ambiguous = _merge_affix_bands(_bands(inside))
    if ambiguous is not None:
        return refused(ambiguous)
    rotated_sharer = _rotated_band_sharer_refusal(extraction, footprint, rows_raw)
    if rotated_sharer is not None:
        return refused(rotated_sharer)

    unmapped = [f for f in inside if f.glyph_mapping is not GlyphMapping.MAPPED]
    if unmapped:
        return refused(
            InventoryRefusal(
                InventoryRefusalReason.UNMAPPED_MEMBER,
                f"{len(unmapped)} fragment(s) inside the footprint have unmapped glyphs",
            )
        )
    bounds = _column_bounds(rows_raw)
    for baseline_y, members, _ in rows_raw:
        own = _column_bounds([(baseline_y, members, [])])
        if len(own) > len(bounds):
            return refused(
                InventoryRefusal(
                    InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED,
                    f"the row at y={baseline_y} resolves {len(own)} blocks where the table resolves {len(bounds)}",
                )
            )

    rows: list[InventoryRow] = []
    cells: list[InventoryCell] = []
    for ordinal, (baseline_y, members, folded) in enumerate(rows_raw):
        row_cells: list[InventoryCell] = []
        for col, (left, right) in enumerate(bounds):
            in_cell = sorted(
                (f for f in members if f.x_start >= left and f.x_end <= right),
                key=lambda f: f.x_start,
            )
            if not in_cell:
                # An empty cell stays empty. A value printed across two columns is not
                # duplicated into this one -- that would fabricate a cell the page never
                # carried, and the association it implies is a claim, not geometry.
                continue
            row_cells.append(
                InventoryCell(
                    row=ordinal,
                    col=col,
                    text="".join(f.text for f in in_cell),
                    x_start=min(f.x_start for f in in_cell),
                    x_end=max(f.x_end for f in in_cell),
                    members=tuple(in_cell),
                )
            )
        if not row_cells:
            continue
        anchor = row_cells[0]
        rows.append(
            InventoryRow(
                ordinal=ordinal,
                baseline_y=baseline_y,
                merged_baselines=tuple(folded),
                anchor_text=anchor.text,
                anchor_x_start=anchor.x_start,
            )
        )
        cells.extend(row_cells)

    # The EDGE guards run last because they are the only checks that need the derivation
    # they are checking: a dropped column is only visible once the rows exist to align it
    # against, and the look-below is bounded by the pitch of the rows that were kept. Both
    # close the same hole the top edge's orphan check closes, on the sides the box was
    # otherwise free to shrink -- measured: shrinking `x_end` deleted a whole fuel mixture
    # and raising `y_bottom` deleted the phi row, each returning a COMPLETE inventory.
    # Both SKIP rotated fragments, and are only sound because nothing rotated shares a row's
    # printed line -- which `_rotated_band_sharer_refusal` established above, as soon as the
    # bands it is scoped to existed. That reliance used to be implicit; now it is a check
    # that has already run.
    truncated = _truncated_column_refusal(extraction, footprint, rows, rows_raw)
    if truncated is not None:
        return refused(truncated)
    cut_below = _orphan_below_refusal(extraction, footprint, bounds, _row_pitch(rows, inside, footprint))
    if cut_below is not None:
        return refused(cut_below)

    return CellInventory(
        footprint=footprint,
        rows=tuple(rows),
        column_bounds=tuple(bounds),
        cells=tuple(cells),
        refusals=(),
        pypdf_version=extraction.pypdf_version,
    )
