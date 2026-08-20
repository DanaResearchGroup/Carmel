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
    rotated: bool = False,
) -> TextFragment:
    return TextFragment(
        page=page,
        text=text,
        x_start=x_start,
        x_end=x_end,
        baseline_y=baseline_y,
        font_height=font_height,
        rotated=rotated,
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


def _EVERY_REFUSAL_SCENARIO() -> list[tuple[FragmentExtraction, ClaimedFootprint]]:
    """One scenario per refusal reason, run for real by the reachability census."""
    wide_row = (
        CAPTION,
        frag("a wide spanning header", 53.0, 280.0, 134.5),
        frag("x", 53.0, 60.0, 121.0),
        frag("y", 122.0, 130.0, 121.0),
        frag("z", 227.0, 240.0, 121.0),
    )
    equidistant_affix = (
        CAPTION,
        frag("O", 122.0, 127.0, 115.0),
        frag("2", 127.0, 130.0, 110.0, font_height=AFFIX_HEIGHT),
        frag("O", 122.0, 127.0, 105.0),
    )
    return [
        (FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_ABSENT), footprint()),
        (extraction_of(*simple_grid(), lossy=True, truncated=True), footprint()),
        (extraction_of(*simple_grid()), footprint(x_end=50.0)),
        (extraction_of(*simple_grid()), footprint(caption_baseline_y=1000.0, y_top=999.0)),
        (extraction_of(*simple_grid()), footprint(y_top=140.0)),
        (extraction_of(*equidistant_affix), footprint()),
        (
            extraction_of(
                CAPTION,
                frag("Aaa", 53.0, 70.0, 120.0),
                frag("2", 250.0, 253.0, 110.0, font_height=AFFIX_HEIGHT),
                frag("Bbb", 53.0, 70.0, 100.0),
            ),
            footprint(),
        ),
        (extraction_of(*simple_grid()), footprint(y_top=130.0)),
        (
            extraction_of(*simple_grid(), frag("sideways", 100.0, 110.0, 100.0, rotated=True)),
            footprint(),
        ),
        (
            extraction_of(CAPTION, frag("/C0 1.0", 122.0, 146.0, 134.5, glyph_mapping=GlyphMapping.UNMAPPED)),
            footprint(),
        ),
        (extraction_of(CAPTION), footprint(y_top=60.0, y_bottom=10.0)),
        (extraction_of(*wide_row), footprint()),
        (extraction_of(*simple_grid(), *cut_off_row()), footprint()),
        (extraction_of(*simple_grid(), *dropped_column()), footprint(x_end=200.0)),
        (
            extraction_of(*two_rows_within_two_tolerances(), frag("?", 320.0, 330.0, 119.65)),
            footprint(),
        ),
    ]


def cut_off_row() -> tuple[TextFragment, ...]:
    """A third row printed below the box's bottom edge, aligned to its derived columns.

    What raising ``y_bottom`` looks like from outside the box -- and what the real target
    turned out to be hiding: ``T (deg C)`` and ``P (atm)`` are rows of that table, and its
    pre-registered footprint stopped just above them.
    """
    return (
        frag("T", 53.0, 57.0, 58.5),
        frag("25", 122.0, 134.0, 58.5),
        frag("25", 227.0, 239.0, 58.5),
    )


def dropped_column() -> tuple[TextFragment, ...]:
    """A fourth column, at one x, on BOTH of ``simple_grid``'s row baselines.

    Alignment across rows is the whole signal: one fragment here would be indistinguishable
    from the page's other prose column, which on the real target sits 26.2 pt from the
    honest box while the dropped column sits 22.6-26.6 pt away.
    """
    return (
        frag("gamma", 320.0, 350.0, 134.5),
        frag("0.4", 320.0, 344.0, 71.5),
    )


def two_rows_within_two_tolerances() -> tuple[TextFragment, ...]:
    """Two rows 0.7 pt apart -- further apart than the band tolerance, closer than twice it.

    That window is what makes a fragment beside the box ambiguous: the rows are separate
    bands, and one baseline between them is within tolerance of BOTH. 422 of the corpus's
    6886 adjacent band pairs sit in it (probes/m1-band-ambiguity.md), so it is ordinary
    typesetting rather than a contrived gap.
    """
    return (
        CAPTION,
        frag("Aaa", 53.0, 70.0, 120.0),
        frag("111", 122.0, 146.0, 120.0),
        frag("Bbb", 53.0, 70.0, 119.3),
        frag("222", 122.0, 146.0, 119.3),
    )


def a_row_wider_than_the_tolerance() -> tuple[TextFragment, ...]:
    """A row whose members CHAIN across 0.8 pt, so ``max()`` does not represent its low end.

    Single linkage bounds the gap between CONSECUTIVE members, not the cluster's total
    span, so 120.0 -> 120.4 -> 120.8 is one row whose representative sits 0.8 pt above its
    lowest member. 14 of the corpus's 6959 bands are this shape, the widest spanning
    1.1178 pt (probes/m1_band_span.py).
    """
    return (
        CAPTION,
        frag("Aaa", 53.0, 70.0, 120.0),
        frag("bbb", 122.0, 146.0, 120.4),
        frag("ccc", 227.0, 250.0, 120.8),
        frag("Ddd", 53.0, 70.0, 100.0),
        frag("eee", 122.0, 146.0, 100.0),
        frag("fff", 227.0, 250.0, 100.0),
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

        assert inventory.refusals == ()
        assert [(r.ordinal, r.anchor_text, r.merged_baselines) for r in inventory.rows] == [(0, "CO2", (133.2,))]

    def test_a_trailing_subscript_is_not_stolen_by_a_wider_row_below(self) -> None:
        """The measured silent corruption that replaced containment with adjacency: with
        a wide row below, containment excluded the true parent, folded the `2` into that
        unrelated row 13 pt away, appended it to that row's anchor -- `wide2` -- and
        refused nothing."""
        fragments = (
            CAPTION,
            frag("CO", 123.0, 137.0, 134.5),
            frag("2", 137.0, 140.0, 133.2, font_height=AFFIX_HEIGHT),
            frag("wide", 53.0, 280.0, 120.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert [(r.ordinal, r.anchor_text) for r in inventory.rows] == [(0, "CO2"), (1, "wide")]

    def test_an_affix_band_adjacent_to_neither_neighbour_refuses(self) -> None:
        """A DIFFERENT fault from AMBIGUOUS_AFFIX_BAND, named separately: there the
        parent is over-determined, here it is absent. A small-font band alone in a column
        belongs to no row, and becoming a row of its own renumbers everything beneath."""
        fragments = (
            CAPTION,
            frag("Aaa", 53.0, 70.0, 120.0),
            frag("2", 250.0, 253.0, 110.0, font_height=AFFIX_HEIGHT),
            frag("Bbb", 53.0, 70.0, 100.0),
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
        inventory = build_inventory(extraction_of(CAPTION), footprint(y_top=60.0, y_bottom=10.0))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.EMPTY]

    def test_the_box_may_not_contain_its_own_caption(self) -> None:
        """Measured before this check existed: raising `y_top` above the caption made the
        caption row 0 and shifted every ordinal beneath it, with no refusal anywhere.
        The docstring said the caption sits above the top edge; nothing enforced it."""
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(y_top=200.0))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.FOOTPRINT_INSANE]

    def test_a_band_orphaned_between_the_caption_and_the_box_refuses(self) -> None:
        """The ordinal-drift attack, made detectable. A caller who lowers `y_top` by one
        band keeps a correct caption and correctly-grounded values while shifting every
        row ordinal by one. Nothing on the locator can see that; the orphaned band can."""
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(y_top=130.0))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ORPHANED_BAND_ABOVE_THE_BOX]

    def test_a_spanning_row_refuses_rather_than_collapsing_the_columns(self) -> None:
        """Columns come from ALIGNED emptiness -- an x-strip empty in EVERY row -- so one
        row spanning the width erases every boundary beneath it. Measured: three columns
        collapsed into one, three cells merged into `xyz` at col=0, `complete` True."""
        fragments = (
            CAPTION,
            frag("a wide spanning header", 53.0, 280.0, 134.5),
            frag("x", 53.0, 60.0, 121.0),
            frag("y", 122.0, 130.0, 121.0),
            frag("z", 227.0, 240.0, 121.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED]

    def test_a_rotated_fragment_inside_the_box_refuses(self) -> None:
        inventory = build_inventory(
            extraction_of(*simple_grid(), frag("sideways", 100.0, 110.0, 100.0, rotated=True)),
            footprint(),
        )

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

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


class TestEveryEdgeOfTheBoxIsFalsifiable:
    """The box may not be shrunk on any side without the shrink being detectable.

    Before these guards existed, exactly ONE of the four edges was watched. Measured on
    the real target: shrinking ``x_end`` deleted an entire fuel mixture and raising
    ``y_bottom`` deleted the phi row, and BOTH returned a complete inventory with no
    refusal and every surviving cell correctly grounded -- the silent-corruption shape
    this module exists to prevent, in the module built to prevent it.
    """

    def test_a_row_cut_off_below_the_box_refuses(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid(), *cut_off_row()), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX]

    def test_a_cut_row_in_a_single_column_still_refuses(self) -> None:
        """One fragment is enough; a cut row need not span the table.

        The permissive alternative -- requiring the band below to occupy two or more
        derived columns -- was measured against the real target and MISSED the attack that
        raised ``y_bottom`` past an affix-split cell living entirely in one column.
        """
        inventory = build_inventory(extraction_of(*simple_grid(), frag("lonely", 122.0, 140.0, 58.5)), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX]

    def test_prose_far_below_the_box_does_not_refuse(self) -> None:
        """The look-below is bounded by the table's OWN median row pitch.

        ``simple_grid`` has one gap of 63 pt, so a band 100 pt below the bottom edge is
        outside the window even though it sits squarely inside a derived column. Without
        the bound the guard would refuse every table with anything printed beneath it.
        """
        inventory = build_inventory(extraction_of(*simple_grid(), frag("body text", 122.0, 140.0, -35.0)), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 2

    def test_an_unmapped_marker_below_the_box_is_not_a_cut_row(self) -> None:
        """It carries markers rather than text, so it cannot be a row.

        The real target's is a raised, zero-width degree sign -- the ``deg`` of
        ``T (deg C)`` -- which did not map. Treating it as a cut row would refuse on the
        wrong reason; one INSIDE the box already refuses under ``UNMAPPED_MEMBER``.
        """
        marker = frag("/C14", 61.8, 61.8, 58.5, glyph_mapping=GlyphMapping.UNMAPPED)
        inventory = build_inventory(extraction_of(*simple_grid(), marker), footprint())

        assert inventory.refusals == ()

    def test_a_column_dropped_by_the_side_of_the_box_refuses(self) -> None:
        inventory = build_inventory(extraction_of(*simple_grid(), *dropped_column()), footprint(x_end=200.0))

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.TRUNCATED_COLUMN_BESIDE_THE_BOX]

    def test_one_row_worth_of_prose_beside_the_box_does_not_refuse(self) -> None:
        """Alignment across rows is the signal, and distance cannot stand in for it.

        On the real target the page's other prose column sits 26.2 pt from the honest box
        while the dropped column sits 22.6-26.6 pt away: the two populations OVERLAP, so a
        threshold classifies one as the other. Prose contributes many fragments on ONE
        baseline; a dropped column contributes few on MANY.
        """
        prose = tuple(
            frag(word, 320.0 + 40.0 * i, 350.0 + 40.0 * i, 134.5)
            for i, word in enumerate(("Many", "studies", "have", "been", "conducted"))
        )
        inventory = build_inventory(extraction_of(*simple_grid(), *prose), footprint())

        assert inventory.refusals == ()
        assert len(inventory.column_bounds) == 3

    def test_a_fragment_beside_the_box_on_two_row_baselines_refuses(self) -> None:
        """Ambiguity is refused, not resolved by whichever row was built first.

        The guard counts DISTINCT ordinals, so awarding an ambiguous fragment to one of
        its two candidate rows is not a tie-break -- it is the difference between a
        two-row cluster (refuse) and a one-row cluster (accept). Nearest-wins would only
        make that guess deterministic; the module already refuses the same shape one edge
        over, at ``AMBIGUOUS_AFFIX_BAND``.
        """
        beside = frag("?", 320.0, 330.0, 119.65)
        inventory = build_inventory(extraction_of(*two_rows_within_two_tolerances(), beside), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.AMBIGUOUS_ROW_BESIDE_THE_BOX]

    def test_a_fragment_beside_the_box_on_one_row_only_still_does_not_refuse(self) -> None:
        """The ambiguity guard must not swallow the ordinary case it sits in front of.

        Same two close rows, but the excluded fragment is squarely on one of them, so
        there is nothing to be ambiguous about and a single-row cluster still reads as
        prose rather than as a dropped column.
        """
        beside = frag("?", 320.0, 330.0, 120.0)
        inventory = build_inventory(extraction_of(*two_rows_within_two_tolerances(), beside), footprint())

        assert inventory.refusals == ()

    def test_a_column_beside_a_row_wider_than_the_tolerance_still_refuses(self) -> None:
        """A row is matched by its whole baseline extent, not by its ``max()`` alone.

        The excluded fragment sits on the LOW end of a row that chains across 0.8 pt, so
        it is 0.75 pt from that row's representative -- outside the tolerance. Matching
        against the representative dropped it, the cluster fell to one ordinal, and the
        dropped column beside the box went unrefused. 1164 corpus fragments are in that
        position across the sampled cuts (probes/m1_band_span.py).
        """
        dropped = (
            frag("gamma", 320.0, 350.0, 120.05),
            frag("0.4", 320.0, 344.0, 100.0),
        )
        inventory = build_inventory(extraction_of(*a_row_wider_than_the_tolerance(), *dropped), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.TRUNCATED_COLUMN_BESIDE_THE_BOX]

    def test_a_wide_row_is_still_one_row(self) -> None:
        """The premise of the test above: the chain really is a single derived row.

        If single linkage had split it, the fragment would match the lower half and the
        refusal would fire for a reason that has nothing to do with the extent.
        """
        inventory = build_inventory(extraction_of(*a_row_wider_than_the_tolerance()), footprint())

        assert inventory.refusals == ()
        assert [r.ordinal for r in inventory.rows] == [0, 1]
        assert inventory.rows[0].baseline_y == 120.8

    def test_a_moved_caption_start_refuses_even_when_the_text_matches(self) -> None:
        """``caption_x_start`` is checked against the document, not against the box.

        It used to be verified only by the footprint's own sanity check -- that it falls
        inside the claimed box -- which compares two caller-supplied numbers the document
        never sees. Measured: shrinking ``x_start`` refused only because the stale
        ``caption_x_start`` fell outside the new box, so a caller who moved both would
        have passed. That is a coincidence, not a guard.
        """
        inventory = build_inventory(extraction_of(*simple_grid()), footprint(caption_x_start=60.0))

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.CAPTION_ANCHOR_ABSENT]


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

        The previous version of this test hard-coded the member list into a set and
        compared it to the enum, so it asserted only that the enum equals itself: it
        would have passed with every code path deleted. A census that cannot fail is
        worse than no census, because it reports coverage nobody has. This one RUNS each
        scenario and collects what the module actually emitted."""
        produced = {
            r.reason for extraction, fp in _EVERY_REFUSAL_SCENARIO() for r in build_inventory(extraction, fp).refusals
        }

        assert produced == set(InventoryRefusalReason), f"unreachable: {set(InventoryRefusalReason) - produced}"
