"""Tests for the claimed-footprint cell inventory.

Every fixture here is SYNTHETIC. The geometry mimics what probe 49 measured on a real
corpus table -- a label column, two value columns, and subscript glyphs sitting on their
own baseline bands -- without carrying any of the paper's text into the repository.

The property under test throughout is not "does it parse tables". It is that the ROW and
COLUMN ordinals, which a ``TableCellLocator`` carries with no page, box or digest to
falsify them, are derived from geometry the caller does not control, and that anything
this module cannot derive REFUSES rather than guessing.
"""

from __future__ import annotations

import pytest

from carmel.services.pdf_fragments import (
    FragmentAvailability,
    FragmentExtraction,
    FragmentPageFailure,
    GlyphMapping,
    TextFragment,
)
from carmel.services.pdf_tables import (
    COLUMN_VALLEY_PT,
    CellInventory,
    ClaimedFootprint,
    InventoryCell,
    InventoryRefusal,
    InventoryRefusalReason,
    build_inventory,
)

BODY_HEIGHT = 8.0
AFFIX_HEIGHT = 5.0


def frag(
    text: str,
    x_start: float,
    x_end: float,
    baseline_y: float,
    *,
    page: int = 1,
    font_height: float = BODY_HEIGHT,
    glyph_mapping: GlyphMapping = GlyphMapping.MAPPED,
) -> TextFragment:
    return TextFragment(
        page=page,
        text=text,
        x_start=x_start,
        x_end=x_end,
        baseline_y=baseline_y,
        font_height=font_height,
        rotated=False,
        glyph_mapping=glyph_mapping,
    )


def extraction_of(*fragments: TextFragment, **kwargs: object) -> FragmentExtraction:
    return FragmentExtraction(fragments=fragments, pypdf_version="6.14.2", **kwargs)  # type: ignore[arg-type]


CAPTION = frag("Table 1 - conditions", 53.0, 196.0, 148.8)


def footprint(**overrides: object) -> ClaimedFootprint:
    base = {
        "page": 1,
        "x_start": 50.0,
        "x_end": 290.0,
        "y_top": 145.0,
        "y_bottom": 65.0,
        "caption_text": CAPTION.text,
        "caption_x_start": CAPTION.x_start,
        "caption_baseline_y": CAPTION.baseline_y,
    }
    base.update(overrides)
    return ClaimedFootprint(**base)  # type: ignore[arg-type]


#: Two content bands, three columns, nothing detached. The baseline case every other
#: test perturbs.
def simple_grid() -> tuple[TextFragment, ...]:
    return (
        CAPTION,
        frag("Fuel", 53.0, 70.0, 134.5),
        frag("alpha", 123.0, 181.0, 134.5),
        frag("beta", 223.0, 280.0, 134.5),
        frag("phi", 53.0, 57.0, 71.5),
        frag("0.6", 122.0, 146.0, 71.5),
        frag("0.5", 227.0, 251.0, 71.5),
    )


class TestTheGridComesFromGeometry:
    def test_a_clean_grid_yields_its_rows_and_columns(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint())

        assert inventory.refusals == ()
        assert inventory.complete
        assert len(inventory.rows) == 2
        assert len(inventory.column_bounds) == 3
        assert [(c.row, c.col, c.text) for c in inventory.cells] == [
            (0, 0, "Fuel"),
            (0, 1, "alpha"),
            (0, 2, "beta"),
            (1, 0, "phi"),
            (1, 1, "0.6"),
            (1, 2, "0.5"),
        ]

    def test_rows_are_ordinal_from_the_top(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint())

        assert [r.ordinal for r in inventory.rows] == [0, 1]
        # Ordinal 0 is the HIGHER baseline: page space grows upward, rows read downward.
        assert inventory.rows[0].baseline_y > inventory.rows[1].baseline_y

    def test_every_row_records_an_anchor_a_replayer_can_recompute(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint())

        assert [r.anchor_text for r in inventory.rows] == ["Fuel", "phi"]
        assert [r.anchor_x_start for r in inventory.rows] == [53.0, 53.0]

    def test_a_row_without_a_label_column_entry_is_still_anchored(self) -> None:
        """The user's rule is "band ordinal + label anchor", and a real continuation
        row has no label-column entry at all: probe 49's target has four oxidizer rows
        whose leftmost cell is in column 1. Requiring the anchor to sit in column 0
        would refuse that table outright. The anchor is therefore the row's LEFTMOST
        NON-EMPTY cell, wherever it sits -- which still pins the row against ordinal
        drift, which is the anchor's whole job."""
        fragments = (
            CAPTION,
            frag("Oxidizer", 53.0, 82.0, 121.3),
            frag("Air", 122.0, 132.0, 121.3),
            # No label-column entry on this band at all.
            frag("O2/He", 122.0, 171.0, 101.4),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert [r.anchor_text for r in inventory.rows] == ["Oxidizer", "O2/He"]
        assert [(c.row, c.col, c.text) for c in inventory.cells if c.row == 1] == [(1, 1, "O2/He")]


class TestOneLineIsOneRowWhateverItsBaselineJitter:
    """A band IS a row ordinal, so splitting one printed line into two renumbers
    every row beneath it and moves the cell a citation resolves to.

    Banding was a ``round(baseline_y, 1)`` bucket, which is not a tolerance: it
    splits on where the bin edge happens to fall, not on how far apart two
    fragments are. Measured over the eight-paper corpus, that bucket split 685
    printed lines the module's own 0.5pt tolerance holds together -- 107 of them
    at a bin edge, the closest pair 1.14e-04 pt apart
    (``probes/m1_band_bins.py``). Reported by Copilot on PR #17.
    """

    @pytest.mark.parametrize(
        ("left_y", "right_y", "what"),
        [
            (134.549916, 134.550030, "the corpus's closest real split, 1.14e-04 pt apart"),
            (134.54, 134.56, "0.02 pt apart, straddling a bin edge"),
            (134.5, 134.56, "one fragment exactly ON a bin edge"),
            (134.5, 134.74, "0.24 pt apart, well inside tolerance and across two bins"),
        ],
        ids=["corpus-closest", "straddles-edge", "on-the-edge", "two-bins-apart"],
    )
    def test_baselines_within_tolerance_are_one_row(self, left_y: float, right_y: float, what: str) -> None:
        assert abs(right_y - left_y) < 0.5, f"{what}: the premise is that these are ONE line"
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, left_y),
            frag("alpha", 123.0, 181.0, right_y),
            frag("phi", 53.0, 57.0, 71.5),
            frag("0.6", 122.0, 146.0, 71.5),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert [r.ordinal for r in inventory.rows] == [0, 1], what
        assert [(c.row, c.col, c.text) for c in inventory.cells] == [
            (0, 0, "Fuel"),
            (0, 1, "alpha"),
            (1, 0, "phi"),
            (1, 1, "0.6"),
        ], what

    def test_baselines_beyond_tolerance_remain_separate_rows(self) -> None:
        """The sweep must not swallow real rows. A pitch apart stays two bands,
        which is also what leaves the affix merge something to do.
        """
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("phi", 53.0, 57.0, 124.0),
            frag("0.6", 122.0, 146.0, 124.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.ordinal for r in inventory.rows] == [0, 1]
        assert [r.anchor_text for r in inventory.rows] == ["Fuel", "phi"]

    def test_a_bands_baseline_is_a_measurement_not_a_bin_label(self) -> None:
        """The row's recorded baseline must be a value some fragment really has,
        so a replayer comparing geometry compares measurements. A rounded stand-in
        matches no fragment on the page.
        """
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.549916),
            frag("alpha", 123.0, 181.0, 134.550030),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.rows[0].baseline_y in {134.549916, 134.550030}


class TestCellTextIsReadingOrderAndNeverRepaired:
    def test_a_subscript_joins_its_parent_in_reading_order(self) -> None:
        """`H` + `2` + `/CO` -> `H2/CO`. Ordering merged members by x is READING
        ORDER, not semantic reconstruction: no character is inserted, substituted or
        dropped. The `2` arrives on its own baseline band, which is the geometry
        probe 49 measured."""
        fragments = (
            CAPTION,
            frag("H", 123.0, 130.0, 134.5),
            frag("2", 130.0, 133.0, 133.2, font_height=AFFIX_HEIGHT),
            frag("/CO", 133.0, 147.0, 134.5),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 1
        assert [c.text for c in inventory.cells] == ["H2/CO"]

    def test_a_mangled_glyph_is_carried_verbatim(self) -> None:
        """A ToUnicode map that decodes phi as `f` and an en-dash as `e` is not this
        layer's to repair -- the fragment lane already flags mapping, and grounding
        proves LOCATION, never MEANING."""
        fragments = (CAPTION, frag("f", 53.0, 57.0, 71.5), frag("0.6e1.0", 122.0, 146.0, 71.5))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [c.text for c in inventory.cells] == ["f", "0.6e1.0"]

    def test_an_unmapped_fragment_refuses_the_inventory(self) -> None:
        """An unmapped glyph is a marker, not a character. A cell built from one would
        read a marker as data -- the sign-inversion failure `GlyphMapping` exists to
        prevent."""
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("/C0 1.0", 122.0, 146.0, 134.5, glyph_mapping=GlyphMapping.UNMAPPED),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.UNMAPPED_MEMBER]


class TestTheAffixMergeIsRecordedOrRefused:
    def test_a_merged_band_records_the_baselines_it_joined(self) -> None:
        fragments = (
            CAPTION,
            frag("H", 123.0, 130.0, 134.5),
            frag("2", 130.0, 133.0, 133.2, font_height=AFFIX_HEIGHT),
            frag("/CO", 133.0, 147.0, 134.5),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.rows[0].merged_baselines == (133.2,)

    def test_an_affix_band_attachable_to_neither_neighbour_refuses(self) -> None:
        """A TRAILING subscript -- the `2` of `CO2` at the end of a cell -- extends past
        its parent band's right edge, so it is interior to nothing. Letting it fall
        through and become a row of its own is the silent renumbering this module exists
        to prevent, so an affix-shaped band that cannot be attached refuses instead.

        This is a DIFFERENT fault from AMBIGUOUS_AFFIX_BAND and is named separately:
        there the parent is over-determined, here it is absent, and an operator reading
        one refusal must not be looking at the other."""
        fragments = (
            CAPTION,
            frag("CO", 123.0, 137.0, 134.5),
            frag("2", 137.0, 140.0, 133.2, font_height=AFFIX_HEIGHT),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.UNATTACHABLE_AFFIX_BAND]

    def test_an_affix_band_interior_to_both_neighbours_refuses(self) -> None:
        """Choosing a parent by proximity alone would silently change both the cell's
        text and the row count. Ambiguity refuses."""
        fragments = (
            CAPTION,
            frag("Aaa", 122.0, 150.0, 120.0),
            frag("2", 130.0, 133.0, 115.0, font_height=AFFIX_HEIGHT),
            frag("Bbb", 122.0, 150.0, 110.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.AMBIGUOUS_AFFIX_BAND]

    def test_an_affix_between_aligned_rows_joins_the_one_it_is_far_nearer_to(self) -> None:
        """The real target's geometry, and the case that refuted whole-band interiority.

        A table's columns are ALIGNED, so a subscript is horizontally interior to the row
        above AND the row below and abuts the same glyph in each. Only vertical distance
        separates them: 1.1 pt to its parent against 8.8 pt to the next row."""
        fragments = (
            CAPTION,
            frag("O", 122.0, 127.0, 111.3),
            frag("/N", 130.0, 138.0, 111.3),
            frag("2", 127.0, 130.0, 110.2, font_height=AFFIX_HEIGHT),
            frag("O", 122.0, 127.0, 101.4),
            frag("/N", 130.0, 138.0, 101.4),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 2
        assert inventory.rows[0].merged_baselines == (110.2,)
        assert [c.text for c in inventory.cells] == ["O2/N", "O/N"]

    def test_an_affix_equidistant_between_aligned_rows_still_refuses(self) -> None:
        """The margin is a rule about when the choice is ARBITRARY, not a measured
        typographic fact (probe 51 found no valley). When neither neighbour is clearly
        nearer, it refuses rather than tie-breaking."""
        fragments = (
            CAPTION,
            frag("O", 122.0, 127.0, 115.0),
            frag("/N", 130.0, 138.0, 115.0),
            frag("2", 127.0, 130.0, 110.0, font_height=AFFIX_HEIGHT),
            frag("O", 122.0, 127.0, 105.0),
            frag("/N", 130.0, 138.0, 105.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.AMBIGUOUS_AFFIX_BAND]

    def test_a_full_height_band_is_never_merged_away(self) -> None:
        """A short row is a ROW. Merging one would renumber every row beneath it while
        every cell value stayed correct -- the exact silent failure this module exists
        to make impossible."""
        fragments = (
            CAPTION,
            frag("Aaa", 122.0, 150.0, 120.0),
            frag("2", 130.0, 133.0, 115.0),  # body height, not an affix
            frag("Bbb", 122.0, 150.0, 110.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 3


class TestColumnsComeFromTheMeasuredValley:
    def test_the_valley_constant_sits_inside_the_measured_empty_gap(self) -> None:
        """Probe 6 found ZERO windows out of 926 with an aligned empty strip in
        [4, 8) pt: the two populations do not overlap, so any value in that gap
        classifies identically. This asserts the constant is a measured valley rather
        than a tuned number -- if someone moves it out of the gap, that is a new
        claim needing a new measurement."""
        assert 4.0 <= COLUMN_VALLEY_PT < 8.0

    def test_a_gap_below_the_valley_is_one_column(self) -> None:
        fragments = (CAPTION, frag("aa", 122.0, 140.0, 134.5), frag("bb", 143.0, 160.0, 134.5))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert len(inventory.column_bounds) == 1
        assert [c.text for c in inventory.cells] == ["aabb"]

    def test_a_gap_above_the_valley_is_two_columns(self) -> None:
        fragments = (CAPTION, frag("aa", 122.0, 140.0, 134.5), frag("bb", 150.0, 170.0, 134.5))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert len(inventory.column_bounds) == 2
        assert [(c.col, c.text) for c in inventory.cells] == [(0, "aa"), (1, "bb")]


class TestSpanningAndEmptyCells:
    def test_an_empty_cell_stays_empty_and_no_value_is_duplicated(self) -> None:
        """A value that spans two columns in the printed table appears once, at its own
        measured position. This module does not model colspan and must not invent one:
        duplicating it into the neighbouring column would fabricate a cell the page
        never carried."""
        fragments = (
            CAPTION,
            frag("Oxidizer", 53.0, 82.0, 121.3),
            frag("Air", 122.0, 132.0, 121.3),
            frag("x", 53.0, 60.0, 101.4),
            frag("p", 122.0, 130.0, 101.4),
            frag("q", 227.0, 240.0, 101.4),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [(c.row, c.col, c.text) for c in inventory.cells] == [
            (0, 0, "Oxidizer"),
            (0, 1, "Air"),
            (1, 0, "x"),
            (1, 1, "p"),
            (1, 2, "q"),
        ]
        # Column 2 of row 0 is absent, not "Air" repeated.
        assert not [c for c in inventory.cells if c.row == 0 and c.col == 2]


class TestTheFootprintIsAClaimAndTheAnchorIsWhatPinsIt:
    def test_the_other_text_column_on_the_page_is_outside_the_footprint(self) -> None:
        """Probe 49: the page's right-hand prose column interleaves in y with the
        table's bands. Footprint scoping is what keeps a prose line from fusing into a
        table row."""
        fragments = (*simple_grid(), frag("prose words here", 304.0, 543.0, 134.5))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert all("prose" not in c.text for c in inventory.cells)
        assert len(inventory.rows) == 2

    def test_prose_sharing_the_caption_baseline_does_not_break_the_anchor(self) -> None:
        """A REGRESSION, and it was found by the real document rather than by this suite.

        On a two-column paper the other column's prose sits fractions of a point from the
        caption's baseline, so a baseline-only band collects both and the concatenation
        never matches the claimed caption. A baseline does not identify a line on a
        multi-column page; the pairing of baseline and x-window does."""
        fragments = (*simple_grid(), frag("method along with a", 304.0, 543.0, 148.4))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 2

    def test_a_caption_anchor_that_is_not_where_it_is_claimed_refuses(self) -> None:
        """The anchor does not make a wrong box impossible. It makes a box that MOVED
        detectable, which is what a replayer needs to refute ordinal drift."""
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(caption_baseline_y=999.0))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.CAPTION_ANCHOR_ABSENT]

    def test_a_caption_whose_text_differs_refuses(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(caption_text="Table 9 - something else"))

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.CAPTION_ANCHOR_ABSENT]

    def test_a_footprint_holding_no_fragments_refuses(self) -> None:
        inventory = build_inventory(
            extraction_of(CAPTION), footprint(y_top=60.0, y_bottom=10.0, caption_baseline_y=55.0)
        )

        assert inventory.cells == ()
        assert InventoryRefusalReason.CAPTION_ANCHOR_ABSENT in [r.reason for r in inventory.refusals]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"x_end": 50.0},
            {"y_bottom": 145.0},
            {"x_start": float("nan")},
            {"page": 0},
        ],
    )
    def test_a_degenerate_footprint_refuses(self, overrides: dict[str, object]) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(**overrides))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.FOOTPRINT_INSANE]


class TestTheDocumentLevelRefusals:
    def test_an_unavailable_extraction_refuses(self) -> None:
        inventory = build_inventory(
            FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_ABSENT), footprint()
        )

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.EXTRACTION_UNAVAILABLE]

    def test_a_failed_page_refuses_that_page(self) -> None:
        """The neighbour that would have refused a cell may simply never have been
        extracted, so a clean result on a lossy page is an artefact of the loss."""
        extraction = extraction_of(
            *simple_grid(),
            lossy=True,
            page_failures=(FragmentPageFailure(page=1, error="synthetic"),),
        )

        inventory = build_inventory(extraction, footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.PAGE_INCOMPLETE]

    def test_a_truncated_document_refuses(self) -> None:
        extraction = extraction_of(*simple_grid(), lossy=True, truncated=True)

        inventory = build_inventory(extraction, footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.PAGE_INCOMPLETE]


class TestFailClosedIsStructural:
    def test_a_refused_inventory_carries_no_cells_and_no_rows(self) -> None:
        """Not a convention. A partial inventory is the thing a caller would read as a
        table, so the type makes "refused, and here are some cells anyway"
        unconstructible."""
        with pytest.raises(ValueError, match="refused inventory"):
            CellInventory(
                footprint=footprint(),
                rows=(),
                column_bounds=(),
                cells=(InventoryCell(row=0, col=0, text="x", x_start=1.0, x_end=2.0, members=()),),
                refusals=(InventoryRefusal(reason=InventoryRefusalReason.EMPTY, detail="x"),),
                pypdf_version="6.14.2",
            )

    def test_complete_is_derived_from_the_refusals_not_stored(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid()), footprint())
        assert inventory.complete is True

        refused = build_inventory(extraction_of(*simple_grid()), footprint(x_end=50.0))
        assert refused.complete is False

    def test_every_refusal_reason_is_reachable(self) -> None:
        """A member no code path can emit is a claim about coverage that is not true.
        Each reason must be produced by at least one test above; this census asserts
        the enum has not grown one that nothing produces."""
        produced = {
            InventoryRefusalReason.EXTRACTION_UNAVAILABLE,
            InventoryRefusalReason.PAGE_INCOMPLETE,
            InventoryRefusalReason.FOOTPRINT_INSANE,
            InventoryRefusalReason.CAPTION_ANCHOR_ABSENT,
            InventoryRefusalReason.AMBIGUOUS_AFFIX_BAND,
            InventoryRefusalReason.UNATTACHABLE_AFFIX_BAND,
            InventoryRefusalReason.UNMAPPED_MEMBER,
            InventoryRefusalReason.EMPTY,
        }
        assert produced == set(InventoryRefusalReason)
