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
    _bands,
    _column_bounds,
    _merge_affix_bands,
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
    ink_x_end: float | None = None,
    glyph_intervals: tuple[tuple[str, float, float], ...] | None = None,
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
        ink_x_end=ink_x_end,
        glyph_intervals=glyph_intervals,
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
        (extraction_of(*a_row_the_edge_can_cut(), RIGHT_EDGE_STRADDLER), footprint()),
        (
            # A glyph-bridging fragment whose far run lands in a column no other ink
            # supports: splitting there would rest the column on the fragment itself.
            extraction_of(
                CAPTION,
                frag("Fuel", 53.0, 70.0, 134.5),
                frag("aa", 122.0, 146.0, 134.5),
                frag("phi", 53.0, 57.0, 71.5),
                frag(
                    "pq",
                    122.0,
                    206.0,
                    71.5,
                    ink_x_end=206.0,
                    glyph_intervals=(("p", 122.0, 126.0), ("q", 200.0, 206.0)),
                ),
            ),
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


#: A value the box's RIGHT edge falls inside: too wide to be a member, and overlapping the
#: box rather than sitting beside it, so neither the containment test nor the
#: truncated-column guard ever sees it.
#:
#: Mirrors what the real target carries on its pressure row -- one fragment at
#: ``x=130.73-322.47`` where the claimed box ends at 290 -- without carrying the paper's text.
RIGHT_EDGE_STRADDLER = frag("91", 130.7, 322.5, 91.0)

#: The same shape at the other side, so the guard cannot be accidentally one-sided.
LEFT_EDGE_STRADDLER = frag("91", 41.0, 128.0, 91.0)


def a_row_the_edge_can_cut() -> tuple[TextFragment, ...]:
    """``simple_grid`` plus a third band, between its two, for a straddler to sit on.

    Its own three fragments are wholly inside the box, so the band is a clean row on its
    own: whatever the straddler tests then show is the straddler's doing, not this row's.
    """
    return (
        *simple_grid(),
        frag("P", 53.0, 60.0, 91.0),
        frag("1", 122.0, 128.0, 91.0),
        frag("8", 227.0, 233.0, 91.0),
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

    def test_a_repaired_fragment_is_admitted_where_an_unmapped_one_refuses(self) -> None:
        """REPAIRED is an admission by name: the fragment lane concluded the character
        from recorded, document-scoped evidence, and the fragment says so. Its text is
        real characters, so a cell built from it is not a marker read as data -- and
        the member's mapping travels into the stored digest, where the distinction
        from MAPPED stays visible."""
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("\u00b0 1.0", 122.0, 146.0, 134.5, glyph_mapping=GlyphMapping.REPAIRED),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert [c.text for c in inventory.cells] == ["Fuel", "\u00b0 1.0"]


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


class TestContainmentIsJudgedByInkNotByTrailingSpacing:
    """A fragment's ADVANCE extent includes spacing charged after its last glyph, so it
    can reach far past anything drawn -- 92.019 pt past, on the real target table's
    pressure row, where it refused the whole table's containment. Every horizontal
    judgement here reads the INK extent instead, with the advance extent as the exact
    fallback where no ink was measured.
    """

    def test_trailing_spacing_does_not_change_the_inventory(self) -> None:
        """Two extractions with the same ink and wildly different trailing advance must
        derive the same grid -- rows, columns, cells, coordinates -- differing only in
        the member fragments themselves, which carry the differing advance extents."""

        def grid(trailing: float) -> tuple[TextFragment, ...]:
            return (
                CAPTION,
                frag("Fuel", 53.0, 70.0, 134.5),
                frag("alpha", 123.0, 181.0, 134.5),
                frag("beta", 223.0, 280.0, 134.5),
                frag("phi", 53.0, 57.0, 71.5),
                # The motivated case: ink ends at 146, the advance runs `trailing` pt
                # further -- past the box's own right edge in the second variant.
                frag("0.6", 122.0, 146.0 + trailing, 71.5, ink_x_end=146.0),
                frag("0.5", 227.0, 251.0, 71.5),
            )

        tight = build_inventory(extraction_of(*grid(0.0)), footprint())
        trailed = build_inventory(extraction_of(*grid(90.0)), footprint())

        assert tight.refusals == () and trailed.refusals == ()
        assert trailed.rows == tight.rows
        assert trailed.column_bounds == tight.column_bounds
        assert [(c.row, c.col, c.text, c.x_start, c.x_end) for c in trailed.cells] == [
            (c.row, c.col, c.text, c.x_start, c.x_end) for c in tight.cells
        ]

    def test_a_fragment_whose_ink_the_edge_cuts_still_refuses(self) -> None:
        """A genuine straddler -- drawn ink on both sides of the edge -- is exactly as
        refused as before the ink extent existed."""
        fragments = (*simple_grid(), frag("wide", 270.0, 320.0, 71.5, ink_x_end=310.0))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_trailing_spacing_crossing_the_edge_is_not_a_cut(self) -> None:
        """The same advance extent, with the ink inside the box: a member, not a
        straddler, and its cell records the ink extent it actually occupies."""
        fragments = (*simple_grid(), frag("v", 255.0, 320.0, 71.5, ink_x_end=262.0))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        cell = next(c for c in inventory.cells if "v" in c.text)
        assert cell.x_end == pytest.approx(262.0)

    def test_an_unmeasured_ink_extent_falls_back_to_the_advance(self) -> None:
        """`ink_x_end=None` must reproduce the pre-field behaviour exactly: the advance
        extent is never narrower than the ink, so the fallback errs toward refusal."""
        fragments = (*simple_grid(), frag("cut", 270.0, 320.0, 71.5))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_an_uncomparable_ink_extent_refuses_the_page(self) -> None:
        """A NaN or backwards `ink_x_end` fails every comparison quietly -- the exact
        fault class the comparable-geometry gate exists for, arriving via the new
        field."""
        for bad in (float("nan"), 100.0):
            fragments = (*simple_grid(), frag("bad", 122.0, 146.0, 71.5, ink_x_end=bad))

            inventory = build_inventory(extraction_of(*fragments), footprint())

            assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]


class TestOneShowOperatorTwoTableCells:
    """The sub-fragment split: rows first, glyph-ink columns second, split only into
    independently supported columns, refuse otherwise -- in that order, because
    hull-derived bounds let a bridging show erase the very valley needed to detect it.
    """

    @staticmethod
    def _grid_with_bridge(intervals: tuple[tuple[str, float, float], ...], text: str = "xy"):
        """Header row establishing three columns; value row whose second fragment is
        one show bridging the two value columns, glyph evidence attached."""
        return (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("alpha", 122.0, 146.0, 134.5),
            frag("beta", 227.0, 251.0, 134.5),
            frag("phi", 53.0, 57.0, 71.5),
            frag(text, intervals[0][1], intervals[-1][2], 71.5, ink_x_end=intervals[-1][2], glyph_intervals=intervals),
        )

    def test_a_bridging_fragment_splits_into_independently_supported_columns(self) -> None:
        fragments = self._grid_with_bridge((("x", 122.0, 130.0), ("y", 240.0, 251.0)))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert len(inventory.column_bounds) == 3
        value_row = [c for c in inventory.cells if c.row == 1]
        assert [(c.col, c.text) for c in value_row] == [(0, "phi"), (1, "x"), (2, "y")]

    def test_split_members_name_their_parent_and_glyph_range(self) -> None:
        """What replay needs: which glyphs of which fragment grounded which cell."""
        fragments = self._grid_with_bridge((("x", 122.0, 130.0), ("y", 240.0, 251.0)))
        bridge = fragments[-1]

        inventory = build_inventory(extraction_of(*fragments), footprint())

        left = next(c for c in inventory.cells if c.row == 1 and c.col == 1)
        right = next(c for c in inventory.cells if c.row == 1 and c.col == 2)
        assert [m.fragment for m in left.members] == [bridge]
        assert [m.fragment for m in right.members] == [bridge]
        assert (left.members[0].glyph_start, left.members[0].glyph_end) == (0, 1)
        assert (right.members[0].glyph_start, right.members[0].glyph_end) == (1, 2)
        assert left.members[0].split and right.members[0].split
        whole = next(c for c in inventory.cells if c.row == 1 and c.col == 0)
        assert (whole.members[0].glyph_start, whole.members[0].glyph_end) == (None, None)
        assert not whole.members[0].split

    def test_the_split_slices_text_at_glyph_piece_boundaries_not_character_counts(self) -> None:
        """Glyph count and character count are different numbers; a member's text is
        its range's recorded pieces, so multi-character pieces travel whole."""
        fragments = self._grid_with_bridge((("ab", 122.0, 130.0), ("cd", 240.0, 251.0)), text="abcd")

        inventory = build_inventory(extraction_of(*fragments), footprint())

        value_row = [c for c in inventory.cells if c.row == 1]
        assert [(c.col, c.text) for c in value_row] == [(0, "phi"), (1, "ab"), (2, "cd")]

    def test_adjacent_glyphs_in_one_column_stay_one_member(self) -> None:
        """Runs, not per-glyph members: consecutive glyphs sharing a block belong to
        one member with one contiguous range."""
        fragments = self._grid_with_bridge((("a", 122.0, 126.0), ("b", 126.0, 130.0), ("c", 240.0, 251.0)), text="abc")

        inventory = build_inventory(extraction_of(*fragments), footprint())

        left = next(c for c in inventory.cells if c.row == 1 and c.col == 1)
        assert left.text == "ab"
        assert (left.members[0].glyph_start, left.members[0].glyph_end) == (0, 2)

    def test_a_split_into_a_column_only_the_fragment_itself_supports_refuses(self) -> None:
        """Independence is the safety property: a column whose only ink is the
        fragment being split is the fragment's own claim, and splitting on it would
        be self-justifying."""
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("aa", 122.0, 146.0, 134.5),
            frag("phi", 53.0, 57.0, 71.5),
            frag("pq", 122.0, 206.0, 71.5, ink_x_end=206.0, glyph_intervals=(("p", 122.0, 126.0), ("q", 200.0, 206.0))),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.UNSPLITTABLE_SPANNING_FRAGMENT]
        assert "no other fragment's ink supports" in inventory.refusals[0].detail

    def test_a_fragment_weaving_between_columns_refuses(self) -> None:
        fragments = self._grid_with_bridge(
            # The third glyph returns to the first column at a sub-valley gap, so the
            # row's own block structure stays consistent and the WEAVE is what refuses.
            (("x", 122.0, 126.0), ("y", 240.0, 246.0), ("z", 127.0, 131.0)),
            text="xyz",
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.UNSPLITTABLE_SPANNING_FRAGMENT]
        assert "weaves" in inventory.refusals[0].detail

    def test_a_spanning_fragment_without_glyph_evidence_cannot_split(self) -> None:
        """The hull fallback can never expose an internal gap, so a bridging show
        without recorded evidence collapses the valley exactly as before the split
        existed -- and the structure check refuses rather than guessing."""
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("alpha", 122.0, 146.0, 134.5),
            frag("beta", 227.0, 251.0, 134.5),
            frag("phi", 53.0, 57.0, 71.5),
            frag("xy", 122.0, 251.0, 71.5, ink_x_end=251.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED]

    def test_glyph_gaps_below_the_valley_do_not_split_anything(self) -> None:
        """Ordinary per-glyph spacing opens gaps too; only an ALIGNED gap at least
        COLUMN_VALLEY_PT wide is column structure. The refuted magnitude rule is not
        rebuilt here: sub-valley gaps merge into one block and one member."""
        fragments = (
            CAPTION,
            frag("Fuel", 53.0, 70.0, 134.5),
            frag("alpha", 122.0, 146.0, 134.5),
            frag(
                "ab",
                122.0,
                146.0,
                71.5,
                ink_x_end=146.0,
                glyph_intervals=(("a", 122.0, 130.0), ("b", 134.0, 146.0)),
            ),
            frag("phi", 53.0, 57.0, 71.5),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        cell = next(c for c in inventory.cells if c.row == 1 and c.col == 1)
        assert cell.text == "ab"
        assert len(cell.members) == 1
        assert (cell.members[0].glyph_start, cell.members[0].glyph_end) == (0, 2)


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

    def test_a_repaired_fragment_below_the_box_is_a_cut_row(self) -> None:
        """The other half of the admission: a REPAIRED fragment is real text, so where
        an unmapped marker below the box is rightly ignored, a repaired glyph aligned
        with a derived column is evidence a row was cut off."""
        repaired = frag("\u00b0", 125.0, 130.0, 58.5, glyph_mapping=GlyphMapping.REPAIRED)
        inventory = build_inventory(extraction_of(*simple_grid(), repaired), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX]

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

    def test_a_fragment_the_right_edge_cuts_through_refuses(self) -> None:
        """Neither existing side test could see it, and that was the whole gap.

        ``_in_footprint`` admits a fragment only when it is WHOLLY inside, so a straddler
        is not a member; ``_truncated_column_refusal`` reads only fragments wholly BESIDE
        the box, so it skips exactly the ones that overlap it. Between the two, the
        fragment was deleted and nothing recorded that it had been there.
        """
        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), RIGHT_EDGE_STRADDLER), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_a_fragment_the_left_edge_cuts_through_refuses(self) -> None:
        """The same shape at the other side. A guard that watched one edge only would be
        the one-edge state this whole class exists to have ended."""
        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), LEFT_EDGE_STRADDLER), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_the_dropped_straddler_used_to_be_invisible_and_this_pins_that(self) -> None:
        """The corruption the guard replaced, pinned so a regression cannot restore it.

        Before the guard these were the SAME inventory: a document whose middle value the
        box's edge cuts through, and a document that never printed that value at all. Both
        reported COMPLETE, with a row that had quietly lost part of itself and every
        surviving cell correctly grounded. On the real target that turned the pressure row
        into ``1e`` beside ``e8`` and refused nothing.

        The first half of this test is what the module still does with the value ABSENT --
        which is correct, because there is then nothing to notice. The second half is the
        difference the guard makes, and keeping them in one test is deliberate: they are
        the same input to a reader, and a regression that let them agree again would have
        to make this test fail rather than merely leave a stale sibling passing.
        """
        without = build_inventory(extraction_of(*a_row_the_edge_can_cut()), footprint())

        assert without.refusals == ()
        assert [(c.row, c.col, c.text) for c in without.cells if c.row == 1] == [
            (1, 0, "P"),
            (1, 1, "1"),
            (1, 2, "8"),
        ]

        cut = build_inventory(extraction_of(*a_row_the_edge_can_cut(), RIGHT_EDGE_STRADDLER), footprint())

        assert cut.complete is False
        assert [r.reason for r in cut.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_a_fragment_reaching_exactly_to_an_edge_is_still_admitted(self) -> None:
        """Membership is inclusive of the edge, and the guard must not quietly narrow it.

        A cell that starts exactly on ``x_start`` or ends exactly on ``x_end`` touches the
        boundary without being cut by it, so it is a member. That is the case that
        separates "the edge falls INSIDE this fragment" from "the fragment stops at the
        edge", and a ``<`` written as ``<=`` would refuse every table drawn to its own
        content's extent.
        """
        fragments = (
            *simple_grid(),
            frag("P", 50.0, 70.0, 91.0),
            frag("1", 122.0, 128.0, 91.0),
            frag("8", 223.0, 290.0, 91.0),
        )

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert inventory.refusals == ()
        assert [(c.row, c.col, c.text) for c in inventory.cells if c.row == 1] == [
            (1, 0, "P"),
            (1, 1, "1"),
            (1, 2, "8"),
        ]

    def test_a_straddler_on_no_band_of_the_box_does_not_refuse(self) -> None:
        """The guard is scoped to the box's own y-window, exactly as membership is.

        A wide fragment far below the table overlaps the box's x-range on a line the box
        never claimed, and refusing on it would refuse every table with a full-width
        paragraph beneath it. What makes a straddler a straddler is that it would have
        been a member but for the x test.
        """
        below = frag("a wide line of running text", 41.0, 322.5, -35.0)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), below), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 3

    def test_a_rotated_fragment_the_edge_cuts_refuses_on_its_band_not_its_extent(self) -> None:
        """It is not recorded as a straddler, and it does not get away either.

        A rotated fragment's ``x_start``/``x_end`` do not bound it horizontally -- 23 of 257
        rotated corpus fragments report an ``x_end`` outside their own page's mediabox, one
        753.8 pt past a 595.3 pt page -- so the straddle test cannot read it, and neither
        can the truncated-column or look-below guards, which all skip it. What is left is
        its BAND, which needs no x: this one sits on the ``P``/``1``/``8`` row's baseline,
        so it shares a printed line of the table and that is enough.

        The reason is ``ROTATED_OR_INSANE_FRAGMENT`` rather than the straddle reason
        deliberately. The finding is "this module cannot read that fragment", which is true;
        "the edge cut it" would be a claim about a number the module has just declared
        meaningless.
        """
        spun = frag("91", 130.7, 322.5, 91.0, rotated=True)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), spun), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_a_rotated_fragment_inside_the_box_refuses_earlier(self) -> None:
        """Same reason, different gate, and this one runs before any derivation."""
        inside = frag("91", 130.7, 180.0, 91.0, rotated=True)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), inside), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_a_rotated_fragment_on_no_printed_line_does_not_refuse(self) -> None:
        """The measured case that set the scope: a page watermark, not part of any table.

        The block is scoped to the table's printed LINES, not to the box's y-window. The
        four registered corpus footprints are 80, 100, 55 and 75 pt tall, and a window that
        tall is a slab of the page. Exactly one rotated corpus fragment falls in one -- a
        ``Downloaded from http://asmedig...`` watermark on ``10.1115-1.4007737`` p5 at
        ``y=746.99``, ``x_start=588.99`` where the box ends at 555.0 on a 612 pt page, so it
        sits in the right margin 34 pt clear of the box. Measured at the shipped order of
        checks, on THIS branch with the branch expectation set: a window-scoped variant
        substituted at this call site takes probe 50 from 4/4 to 3/4, refusing that footprint
        on the watermark and replacing the ``column_structure_unresolved`` it correctly
        reports. A probe score is scoped to a checkout -- against ``official/main``
        ``499835c`` this file's expectation set cannot run at all, because it names a refusal
        reason that commit does not define.

        The fixture carries that watermark's geometry faithfully, INCLUDING its offset from
        the nearest line: 1.92 pt, the real measured gap, which is outside this module's
        0.5 pt band and inside ``pdf_cells``'s 4.0 pt one. So this is also the fragment on
        which the two lanes' outcomes differ, which is why it is pinned rather than
        described.
        """
        watermark = frag("Downloaded from http://x/", 324.0, 1008.0, 92.92, rotated=True)
        nearest = min(abs(92.92 - y) for y in (134.5, 91.0, 71.5))

        assert nearest == pytest.approx(1.92), "the premise: the real watermark's offset from its nearest line"

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), watermark), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 3

    def test_a_rotated_fragment_on_a_FOLDED_affix_line_refuses(self) -> None:
        """A subscript's baseline is a printed line too, and the merge is what hid it.

        Rows are matched by every baseline they carry -- their own and the affix baselines
        folded into them. Matching the merged representative alone would leave a rotated
        fragment sitting on the ``2`` of ``CO2`` unguarded, on a line the page really
        printed, for no reason other than that this module tidied the band away.
        """
        fragments = (
            CAPTION,
            frag("CO", 122.0, 134.0, 120.0),
            frag("2", 134.0, 137.0, 116.0, font_height=AFFIX_HEIGHT),
            frag("Bbb", 53.0, 70.0, 100.0),
            frag("222", 122.0, 146.0, 100.0),
        )
        merged = build_inventory(extraction_of(*fragments), footprint())

        assert merged.refusals == ()
        assert [r.anchor_text for r in merged.rows] == ["CO2", "Bbb"], "the premise: 116.0 was folded away"

        on_the_folded_line = frag("spun", 320.0, 900.0, 116.0, rotated=True)
        inventory = build_inventory(extraction_of(*fragments, on_the_folded_line), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_an_unmapped_straddler_still_refuses(self) -> None:
        """Deliberately unlike ``ORPHANED_BAND_BELOW_THE_BOX``, which skips unmapped ones.

        That reason asks whether the band below the box is a ROW, and a marker is not one.
        This one asks whether the edge cut something, and a marker is something -- an
        unmapped fragment inside the box refuses under ``UNMAPPED_MEMBER``, so letting the
        same fragment through because the edge happened to bisect it would make the box's
        boundary the softer door into the same data.
        """
        marker = frag("/C14", 130.7, 322.5, 91.0, glyph_mapping=GlyphMapping.UNMAPPED)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), marker), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]

    def test_the_boundary_reason_survives_an_unmapped_member_for_BOTH_twins(self) -> None:
        """One document, one class of fault, two orientations, and no glyph in the answer.

        The upright straddler earned this guarantee by being settled before the box's
        contents are read. Its rotated twin has to be asked AFTER the rows exist, so for a
        while it did not have it: with an unmapped member inside the box, an upright
        fragment the edge cut reported ``straddling_fragment_at_the_box_edge`` while a
        rotated one on a row's own printed line reported ``unmapped_member`` -- the same
        document answering differently depending on a glyph neither fragment involves.

        The repair is that the row derivation reads geometry only, so it can be hoisted
        above the unmapped check; ``test_the_row_derivation_reads_no_glyph`` pins that
        premise separately, because this test is worthless if it is not true.
        """
        marker = frag("/C14", 61.8, 61.8, 91.0, glyph_mapping=GlyphMapping.UNMAPPED)
        upright = RIGHT_EDGE_STRADDLER
        spun = frag("91", 130.7, 322.5, 91.0, rotated=True)

        # The DETAIL is asserted, not only the reason. `ROTATED_OR_INSANE_FRAGMENT` is
        # shared by three checks -- the page-scoped geometry gate, the rotated-member check
        # and the band-sharer -- so the enum alone would pass if the wrong one fired.
        for fragment, expected, detail in (
            (upright, InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE, "cut by its side edge"),
            (spun, InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT, "share a printed line of the table"),
        ):
            with_marker = build_inventory(extraction_of(*a_row_the_edge_can_cut(), fragment, marker), footprint())
            without = build_inventory(extraction_of(*a_row_the_edge_can_cut(), fragment), footprint())

            assert [(r.reason, detail in r.detail) for r in with_marker.refusals] == [(expected, True)]
            assert [(r.reason, detail in r.detail) for r in without.refusals] == [(expected, True)]

    def test_the_row_derivation_reads_no_glyph(self) -> None:
        """The premise the hoist rests on, asserted rather than trusted -- and asserted
        against the DERIVATION HELPERS rather than through ``build_inventory``.

        That indirection is the point. The claim is that ``_bands`` and
        ``_merge_affix_bands`` consult no glyph, so rows may be derived before glyphs are
        known readable. Driving it through ``build_inventory`` cannot show that: the rows ARE
        derived first now -- that is what the hoist did -- but an UNMAPPED member then
        refuses, and a refused inventory carries no rows, so there is nothing left to compare.
        The only corruption that survives the front door is a text change with
        ``glyph_mapping`` left MAPPED, which is a weaker claim than the one the hoist needs.
        Calling the helpers directly is where UNMAPPED is survivable, so that is where the
        real corruption is applied.
        """
        readable = [
            frag("CO", 122.0, 134.0, 120.0),
            frag("2", 134.0, 137.0, 116.0, font_height=AFFIX_HEIGHT),
            frag("Bbb", 53.0, 70.0, 100.0),
            frag("222", 122.0, 146.0, 100.0),
        ]
        corrupted = [
            frag(
                "?" * len(f.text),
                f.x_start,
                f.x_end,
                f.baseline_y,
                font_height=f.font_height,
                glyph_mapping=GlyphMapping.UNMAPPED,
            )
            for f in readable
        ]

        first, clean_ambiguity = _merge_affix_bands(_bands(readable))
        second, corrupt_ambiguity = _merge_affix_bands(_bands(corrupted))

        assert clean_ambiguity is None and corrupt_ambiguity is None
        assert [(y, folded) for y, _, folded in first] == [(y, folded) for y, _, folded in second]
        assert [[(f.x_start, f.x_end, f.baseline_y) for f in members] for _, members, _ in first] == [
            [(f.x_start, f.x_end, f.baseline_y) for f in members] for _, members, _ in second
        ]
        assert _column_bounds(first) == _column_bounds(second)
        assert [[f.text for f in members] for _, members, _ in first] != [
            [f.text for f in members] for _, members, _ in second
        ], "the text and glyph state DID change"

    def test_the_straddle_refusal_does_not_depend_on_the_glyph_state(self) -> None:
        """The reason a caller reads must not turn on whether some OTHER fragment mapped.

        On the real target the whole-table footprint carries both faults at once: an
        unmapped degree sign inside the box, and a value the right edge cuts through. If
        the unmapped check ran first, repairing the glyph -- or meeting a document that
        never had one -- would silently take the straddler's refusal away again. So the
        boundary is settled before the contents are read, and the same input refuses
        identically with the marker present and absent.
        """
        marker = frag("/C14", 61.8, 61.8, 91.0, glyph_mapping=GlyphMapping.UNMAPPED)
        fragments = (*a_row_the_edge_can_cut(), RIGHT_EDGE_STRADDLER)

        with_marker = build_inventory(extraction_of(*fragments, marker), footprint())
        without_marker = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in with_marker.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]
        assert [r.reason for r in without_marker.refusals] == [r.reason for r in with_marker.refusals]

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


class TestGeometryThatCannotBeComparedRefuses:
    """The straddler's defect class, one comparison over.

    A fragment whose extents are ``NaN``, infinite or backwards makes EVERY comparison in
    this module quietly False. It is not a member, because ``_in_footprint``'s test fails;
    not a straddler, because both edge tests fail; not an orphan above or below; and not
    part of any excluded-column cluster. Where the ``'91'`` fragment at least satisfied one
    test in one direction, this one satisfies nothing at all and so leaves no trace
    anywhere. Zero corpus fragments are in this shape (0 of 78178) -- it guards the shape of
    the input, not an observed instance, which is the footing ``INVALID_GEOMETRY`` has
    always stood on in the single-cell lane.
    """

    @pytest.mark.parametrize(
        ("x_start", "x_end", "baseline_y", "what"),
        [
            (300.0, float("nan"), 91.0, "a NaN right extent, beside the box"),
            (float("nan"), 320.0, 91.0, "a NaN left extent, beside the box"),
            (float("inf"), float("inf"), 91.0, "an infinite extent"),
            # A backwards extent satisfies `x_start >= box.x_start and x_end <= box.x_end`
            # by accident more often than not, and those the members-only check caught
            # already. These two fail it, one on each half, and fell straight through the
            # gap before this gate: `40 -> 30` is left of `x_start`, `300 -> 295` runs past
            # `x_end`.
            (40.0, 30.0, 91.0, "a backwards extent left of the box"),
            (300.0, 295.0, 91.0, "a backwards extent past the box's right edge"),
            (300.0, 320.0, float("nan"), "a NaN baseline, which reaches no band at all"),
        ],
        ids=["nan-x-end", "nan-x-start", "infinite", "backwards-left", "backwards-right", "nan-baseline"],
    )
    def test_an_outsider_this_module_cannot_compare_refuses(
        self, x_start: float, x_end: float, baseline_y: float, what: str
    ) -> None:
        unreadable = frag("?", x_start, x_end, baseline_y)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), unreadable), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT], what

    def test_the_nan_baseline_case_is_why_the_check_is_page_scoped(self) -> None:
        """A window-scoped check would read only what survived the test NaN defeats.

        ``y_bottom <= baseline_y <= y_top`` is itself a comparison NaN fails, so a fragment
        with a NaN baseline is dropped one step BEFORE any x test runs -- and a check scoped
        to the box's own bands would therefore never see it. The premise is asserted here
        rather than trusted: the fragment is confirmed absent from the window before the
        refusal that catches it anyway is asserted.
        """
        unreadable = frag("?", 300.0, 320.0, float("nan"))
        fp = footprint()

        assert not (fp.y_bottom <= unreadable.baseline_y <= fp.y_top), "the premise: NaN fails the window test"

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), unreadable), fp)

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_the_unreadable_outsider_used_to_be_invisible_and_this_pins_that(self) -> None:
        """The same pin the straddler has, for the fragment that leaves even less behind.

        Before the guard these were the SAME inventory: a document carrying a fragment no
        comparison in this module can evaluate, and a document that never printed it. Both
        reported COMPLETE with three clean rows. The first half is what the module still
        does with the fragment absent, which is correct; the second is the difference the
        guard makes.
        """
        without = build_inventory(extraction_of(*a_row_the_edge_can_cut()), footprint())

        assert without.refusals == ()
        assert len(without.rows) == 3

        unreadable = build_inventory(
            extraction_of(*a_row_the_edge_can_cut(), frag("?", 300.0, float("nan"), 91.0)), footprint()
        )

        assert unreadable.complete is False
        assert [r.reason for r in unreadable.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_an_unreadable_font_height_fabricates_a_row_and_must_refuse(self) -> None:
        """The same defect class one FIELD over, and this one invents a row rather than
        losing one.

        ``font_height`` never bounds the box, so it looks unrelated to a boundary gate. But
        it feeds the ``<=`` in ``_looks_like_affix``, and a NaN there makes the comparison
        quietly False: the subscript band is not affix-shaped, so it is not folded into its
        parent and becomes a ROW of its own. Every row beneath it renumbers, the cell text
        loses its subscript, and the inventory reports COMPLETE.

        Both halves are asserted, because the fabrication is the point: with a real height
        the band folds to two rows anchored ``CO2``; with NaN the old code returned three
        rows, ``CO``, and a spurious ``2``.
        """
        readable = (
            CAPTION,
            frag("CO", 122.0, 134.0, 120.0),
            frag("2", 134.0, 137.0, 116.0, font_height=AFFIX_HEIGHT),
            frag("Bbb", 53.0, 70.0, 100.0),
            frag("222", 122.0, 146.0, 100.0),
        )
        folded = build_inventory(extraction_of(*readable), footprint())

        assert folded.refusals == ()
        assert [r.anchor_text for r in folded.rows] == ["CO2", "Bbb"]

        unreadable = (*readable[:2], frag("2", 134.0, 137.0, 116.0, font_height=float("nan")), *readable[3:])
        inventory = build_inventory(extraction_of(*unreadable), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_a_zero_font_height_loses_a_whole_row_and_must_refuse(self) -> None:
        """The fourth of these, and the one that finally changed the shape of the fix.

        ``_looks_like_affix`` guards ``reference <= 0`` for the NEIGHBOUR but not for the
        band, so a band of zero-height fragments is affix-shaped against everything and is
        folded away. Two ordinary rows, the second at ``font_height=0.0``: the old code
        returned COMPLETE with ONE row anchored ``AaaBbb`` -- a whole row swallowed by the
        one above it, every surviving cell correctly grounded, no refusal. A row LOST rather
        than fabricated, which is the same silence from the other direction.
        """
        rows = (CAPTION, frag("Aaa", 53.0, 70.0, 120.0), frag("Bbb", 53.0, 70.0, 100.0, font_height=0.0))

        inventory = build_inventory(extraction_of(*rows), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    @pytest.mark.parametrize("field", ["x_start", "x_end", "baseline_y", "font_height"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_every_geometry_float_is_covered_by_the_one_invariant(self, field: str, bad: float) -> None:
        """Every float this module compares, against the one predicate.

        ``TextFragment`` carries exactly four -- ``x_start``, ``x_end``, ``baseline_y``,
        ``font_height`` -- and ``page`` only for equality, so this is exhaustive over the
        FIELDS. It is deliberately not claimed to be exhaustive over the FAULT: an earlier
        version of this docstring said no fifth patch was possible, and
        ``_looks_like_affix``'s unbounded height ratio refutes that with four finite,
        positive, well-ordered numbers. Comparability is what is closed here, completely;
        magnitude is a separate fault, tracked separately.
        """
        geometry = {"x_start": 300.0, "x_end": 320.0, "baseline_y": 91.0, "font_height": BODY_HEIGHT}
        geometry[field] = bad
        broken = frag(
            "?", geometry["x_start"], geometry["x_end"], geometry["baseline_y"], font_height=geometry["font_height"]
        )

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), broken), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_a_negative_font_height_refuses(self) -> None:
        """Kept beside the zero case rather than folded into it.

        The invariant covers it -- ``font_height > 0`` is one comparison -- but a negative
        height is a distinct nonsense value with a distinct effect: measured, a negative band
        reads as affix-shaped against any positive neighbour, so it can always be swallowed,
        while ``reference <= 0`` stops it ever being a parent, so it can never swallow. It is
        a fragment that can only ever disappear. 0 of 78178 corpus fragments carry one, and a
        test that names the value is what stops it silently leaving the predicate.
        """
        marker = frag("2", 134.0, 137.0, 116.0, font_height=-5.0)

        inventory = build_inventory(extraction_of(CAPTION, frag("CO", 122.0, 134.0, 120.0), marker), footprint())

        assert inventory.cells == ()
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_unreadable_geometry_is_named_before_the_caption_anchor_is_blamed(self) -> None:
        """The caption anchor is a geometric comparison too, so it sits behind the gate.

        ``_caption_anchored`` matches the claimed baseline with ``math.isclose``, which a
        NaN defeats. Ahead of the gate, a caption fragment with an unreadable baseline
        reported ``CAPTION_ANCHOR_ABSENT`` -- true in that the anchor was not found, but it
        names the caller's caption fields as the fault when the document's geometry is what
        cannot be read. Same reason-attribution class as the straddle and rotated
        placements, so the same answer.
        """
        broken_caption = frag(CAPTION.text, CAPTION.x_start, 196.0, float("nan"))
        fragments = (broken_caption, *simple_grid()[1:])

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]
        assert "comparable geometry" in inventory.refusals[0].detail

    def test_a_zero_width_fragment_stays_legal(self) -> None:
        """The boundary the invariant deliberately does NOT close.

        ``x_start == x_end`` is a real fragment: the corpus's ``/C14`` degree sign is exactly
        that, at ``x=61.7952-61.7952``. Refusing zero WIDTH would refuse a document over a
        mark the page really carries, which is why the predicate says ``x_start <= x_end``
        and ``font_height > 0`` rather than treating the two axes alike.
        """
        degree_sign = frag("o", 61.8, 61.8, 91.0)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), degree_sign), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 3

    def test_a_blank_unreadable_fragment_still_refuses(self) -> None:
        """Deliberately unlike the straddle guard, which skips blanks.

        That one asks whether the edge cut TEXT, and a bare space text-show operation loses
        nothing when it goes. This one asks whether the page's geometry can be read at all,
        and an unreadable number is unreadable whether or not a glyph was drawn with it.
        """
        blank = frag("   ", 300.0, float("nan"), 91.0)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), blank), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]

    def test_an_unreadable_fragment_on_another_page_does_not_refuse(self) -> None:
        """Page-scoped, not document-scoped. A broken glyph run on page 9 says nothing about
        whether the box on page 1 has a clean boundary, and refusing on it would make one
        malformed figure caption anywhere in a paper condemn every table in it."""
        elsewhere = frag("?", 300.0, float("nan"), 91.0, page=2)

        inventory = build_inventory(extraction_of(*a_row_the_edge_can_cut(), elsewhere), footprint())

        assert inventory.refusals == ()
        assert len(inventory.rows) == 3

    def test_it_is_settled_before_the_straddle_check(self) -> None:
        """Order matters when a page carries both faults, and NaN is the one that decides.

        A straddler's edge test is a comparison, so it is only meaningful once the page's
        geometry is known to be comparable. With both present the reason must be the
        geometry -- the straddle finding is not more specific, it is less well-founded.
        """
        fragments = (*a_row_the_edge_can_cut(), RIGHT_EDGE_STRADDLER, frag("?", 300.0, float("nan"), 91.0))

        inventory = build_inventory(extraction_of(*fragments), footprint())

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.ROTATED_OR_INSANE_FRAGMENT]


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
