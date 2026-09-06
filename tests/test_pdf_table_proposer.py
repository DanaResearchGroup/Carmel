"""The PDF table proposer: derive candidate footprints from geometry, or refuse.

Two kinds of test live here. The UNIT tests build synthetic fragment pages (the corpus is
non-redistributable, so no document bytes are committed) and pin each guard by the identity of
the refusal it raises. The REDISCOVERY tests are the acceptance artifact: they run the proposer
over the two hand-pinned target documents WITHOUT reference to their ``TARGET_TABLE_FOOTPRINT``
constants and show the derived grid is cell-identical to the one the hand-pinned box produces.
Those are corpus-gated exactly like ``test_condition_set_target_acceptance`` -- they SKIP, never
pass, when the document (or pypdf) is absent.
"""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import pytest

from carmel.services.pdf_fragments import FragmentAvailability, FragmentExtraction, GlyphMapping, TextFragment
from carmel.services.pdf_table_proposer import (
    ProposalRefusal,
    ProposalRefusalReason,
    _classify,
    _column_right,
    _row_pitch_gaps,
    propose_tables,
)
from carmel.services.pdf_tables import ClaimedFootprint, FootprintGeometryOrigin, _bands, build_inventory
from tests.pypdf_gate import require_pypdf

BODY_HEIGHT = 8.0


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


def page(*fragments: TextFragment) -> FragmentExtraction:
    return FragmentExtraction(fragments=fragments, pypdf_version="6.14.2")


# A caption over a 2x2 grid in a single column. The caption sits a font-height and a half above
# the first row (so it is not read as a body row), the two rows a clean 15 pt pitch apart, the
# two columns a valley wider than COLUMN_VALLEY_PT apart. The baseline every fixture perturbs.
def _clean_grid() -> tuple[TextFragment, ...]:
    return (
        frag("Table 1 conditions", 50.0, 160.0, 200.0),
        frag("Fuel", 50.0, 70.0, 185.0),
        frag("Air", 120.0, 140.0, 185.0),
        frag("0.5", 50.0, 66.0, 170.0),
        frag("0.6", 120.0, 136.0, 170.0),
    )


class TestItProposesARealGrid:
    def test_a_caption_over_a_two_by_two_grid_is_proposed(self) -> None:
        outcome = propose_tables(page(*_clean_grid()))
        assert len(outcome.proposals) == 1
        assert outcome.refusals == ()
        proposed = outcome.proposals[0]
        assert len(proposed.inventory.rows) == 2
        assert len(proposed.inventory.column_bounds) == 2

    def test_every_proposal_carries_the_derived_geometry_origin(self) -> None:
        proposed = propose_tables(page(*_clean_grid())).proposals[0]
        assert proposed.footprint.geometry_origin is FootprintGeometryOrigin.DERIVED

    def test_a_proposed_footprint_re_derives_the_same_grid(self) -> None:
        # The proposal IS the footprint build_inventory judged; feeding it back must reproduce.
        extraction = page(*_clean_grid())
        proposed = propose_tables(extraction).proposals[0]
        assert build_inventory(extraction, proposed.footprint).cells == proposed.inventory.cells

    def test_the_caption_fragment_is_a_short_quote_not_a_page_number(self) -> None:
        proposed = propose_tables(page(*_clean_grid())).proposals[0]
        assert "Table 1" in proposed.caption_fragment
        assert len(proposed.caption_fragment) <= 48


class TestItRefusesWhatIsNotATable:
    def test_a_prose_mention_opening_table_n_is_refused_not_proposed(self) -> None:
        # "Table 2. The values were tabulated ..." running on as one-column prose. Each line is a
        # single left-aligned run, so the caption walk absorbs the whole paragraph and finds no
        # table body beneath it -- the false-positive shape, refused (this is exactly how the
        # real prose mention on 9c59f1c6 p11 refuses: "no body band inside the caption's column").
        prose = page(
            frag("Table 2. The values were", 50.0, 260.0, 200.0),
            frag("tabulated across the whole", 50.0, 260.0, 190.0),
            frag("range of the study set out", 50.0, 260.0, 180.0),
        )
        outcome = propose_tables(prose)
        assert outcome.proposals == ()
        assert len(outcome.refusals) == 1
        assert outcome.refusals[0].reason is ProposalRefusalReason.CAPTION_COLUMN_UNRESOLVED
        assert "Table 2" in outcome.refusals[0].caption_fragment

    def test_a_single_column_body_is_refused_too_few_columns(self) -> None:
        outcome = propose_tables(
            page(
                frag("Table 1 list", 50.0, 130.0, 200.0),
                frag("first", 50.0, 80.0, 185.0),
                frag("second", 50.0, 82.0, 170.0),
            )
        )
        assert outcome.proposals == ()
        assert outcome.refusals[0].reason is ProposalRefusalReason.TOO_FEW_COLUMNS

    def test_a_one_row_body_is_refused_too_few_rows(self) -> None:
        outcome = propose_tables(
            page(
                frag("Table 1 heads", 50.0, 140.0, 200.0),
                frag("Fuel", 50.0, 70.0, 185.0),
                frag("Air", 120.0, 140.0, 185.0),
            )
        )
        assert outcome.proposals == ()
        assert outcome.refusals[0].reason is ProposalRefusalReason.TOO_FEW_ROWS

    def test_a_grid_body_reason_is_carried_through_as_grid_not_derived(self) -> None:
        # An unmapped glyph inside the derived box: build_inventory refuses UNMAPPED_MEMBER, and
        # the proposer classes it GRID_NOT_DERIVED with the inner reason named in the detail.
        outcome = propose_tables(
            page(
                frag("Table 1 conditions", 50.0, 160.0, 200.0),
                frag("Fuel", 50.0, 70.0, 185.0),
                frag("Air", 120.0, 140.0, 185.0),
                frag("0.5", 50.0, 66.0, 170.0),
                frag("/C0", 120.0, 136.0, 170.0, glyph_mapping=GlyphMapping.UNMAPPED),
            )
        )
        assert outcome.proposals == ()
        assert outcome.refusals[0].reason is ProposalRefusalReason.GRID_NOT_DERIVED
        assert "unmapped_member" in outcome.refusals[0].detail

    def test_a_missing_caption_anchor_is_classed_caption_not_printed(self) -> None:
        # Tested at the _classify seam, not end to end: the proposer derives its footprint FROM a
        # real caption line, so build_inventory's CAPTION_ANCHOR_ABSENT is effectively unreachable
        # through propose_tables. The reason translation is real either way -- a box whose caption
        # baseline names a line the document does not print refuses CAPTION_ANCHOR_ABSENT, which
        # this lane must surface as CAPTION_NOT_PRINTED (distinct from GRID_NOT_DERIVED: the box
        # was never anchored, so no grid was even attempted). Feed a genuine build_inventory
        # refusal of that reason rather than a hand-built inventory, so the mapping is pinned to
        # the reason the extractor actually raises.
        extraction = page(*_clean_grid())
        never_printed = ClaimedFootprint(
            page=1,
            x_start=48.0,
            x_end=145.0,
            y_top=205.0,
            y_bottom=162.0,
            caption_text="Table 1 conditions",
            caption_x_start=50.0,
            caption_baseline_y=999.0,
            geometry_origin=FootprintGeometryOrigin.DERIVED,
        )
        inventory = build_inventory(extraction, never_printed)
        classified = _classify(inventory, page=1, caption="Table 1")
        assert isinstance(classified, ProposalRefusal)
        assert classified.reason is ProposalRefusalReason.CAPTION_NOT_PRINTED

    def test_an_unavailable_extraction_refuses_once_for_the_document(self) -> None:
        outcome = propose_tables(FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_ABSENT))
        assert outcome.proposals == ()
        assert len(outcome.refusals) == 1
        assert outcome.refusals[0].reason is ProposalRefusalReason.EXTRACTION_UNAVAILABLE

    def test_a_band_with_no_table_heading_is_not_a_candidate(self) -> None:
        # A page with a grid but no "Table N" band proposes nothing and refuses nothing: there
        # is no candidate to judge.
        outcome = propose_tables(
            page(
                frag("Results", 50.0, 100.0, 200.0),
                frag("Fuel", 50.0, 70.0, 185.0),
                frag("Air", 120.0, 140.0, 185.0),
                frag("0.5", 50.0, 66.0, 170.0),
                frag("0.6", 120.0, 136.0, 170.0),
            )
        )
        assert outcome.proposals == ()
        assert outcome.refusals == ()


class TestTheCaptionColumnScoping:
    def _two_column_page(self) -> FragmentExtraction:
        # A grid in the LEFT column, article prose in the RIGHT column. The article column runs
        # at its OWN vertical rhythm (baselines offset from the table rows, as real two-column
        # layouts are), so it is not mistaken for a truncated table column. The page gutter runs
        # from x=140 (grid's right ink) to x=320 (the article column). The proposer must set the
        # box's right edge inside that gutter, excluding the article column, or build_inventory
        # refuses on a straddle.
        return page(
            frag("Table 1 conditions", 50.0, 160.0, 200.0),
            frag("an adjacent article column", 320.0, 470.0, 200.0),
            frag("Fuel", 50.0, 70.0, 185.0),
            frag("Air", 120.0, 140.0, 185.0),
            # This body line of the article column starts a couple of points LEFT of the
            # neighbour's caption-baseline edge (320) -- real justified prose jitters like this.
            # The COLUMN_VALLEY_PT margin in _column_right must still read it as the neighbour, or
            # its ink drags the box's right edge across the gutter.
            frag("of running body text here", 316.0, 470.0, 191.0),
            frag("0.5", 50.0, 66.0, 170.0),
            frag("0.6", 120.0, 136.0, 170.0),
            frag("that keeps flowing downward", 320.0, 470.0, 179.0),
        )

    def test_a_two_column_page_proposes_the_grid_and_excludes_the_neighbour(self) -> None:
        outcome = propose_tables(self._two_column_page())
        assert len(outcome.proposals) == 1
        proposed = outcome.proposals[0]
        assert len(proposed.inventory.column_bounds) == 2
        # The right edge sits in the gutter (140 .. 320), never on the article column.
        assert 140.0 < proposed.footprint.x_end < 320.0

    def test_a_caption_in_the_right_column_is_still_found(self) -> None:
        # The "Table N" block is not block zero of its baseline: a left column precedes it.
        outcome = propose_tables(
            page(
                frag("left column running text", 40.0, 190.0, 203.0),
                frag("Table 1 conditions", 320.0, 430.0, 200.0),
                frag("more left column text", 40.0, 190.0, 191.0),
                frag("Fuel", 320.0, 340.0, 185.0),
                frag("Air", 390.0, 410.0, 185.0),
                frag("still more left text", 40.0, 190.0, 179.0),
                frag("0.5", 320.0, 336.0, 170.0),
                frag("0.6", 390.0, 406.0, 170.0),
            )
        )
        assert len(outcome.proposals) == 1
        assert outcome.proposals[0].footprint.x_start >= 320.0


class TestTheTwoLineCaptionAnchor:
    def _two_line_caption(self) -> FragmentExtraction:
        # The B shape: a "Table N" heading line, a caption continuation line, then the grid. The
        # anchor must be the continuation line (build_inventory orphans a band between the
        # anchor and the box top), so the continuation is not read as a body row.
        return page(
            frag("Table 1 flame speed over the", 50.0, 210.0, 200.0),
            frag("range of equivalence ratios", 50.0, 190.0, 190.0),
            frag("phi", 50.0, 66.0, 175.0),
            frag("SL", 120.0, 140.0, 175.0),
            frag("0.5", 50.0, 66.0, 160.0),
            frag("67", 120.0, 136.0, 160.0),
        )

    def test_the_anchor_is_the_last_caption_line_and_the_grid_is_proposed(self) -> None:
        outcome = propose_tables(self._two_line_caption())
        assert len(outcome.proposals) == 1
        proposed = outcome.proposals[0]
        # Anchored on the continuation line, not the heading line.
        assert proposed.footprint.caption_baseline_y == 190.0
        assert "range of equivalence ratios" in proposed.footprint.caption_text
        assert len(proposed.inventory.rows) == 2


class TestRediscoveryOfTheHandPinnedTargets:
    """The acceptance artifact: rediscover A and B from geometry, cell-identical to the hand-pin.

    Corpus-gated: the two papers are non-redistributable, read from the operator's store at
    runtime, and every test SKIPS -- never passes -- when the document or pypdf is absent.
    """

    def _extraction(self, sha: str, roots: tuple[Path, ...], campaign: str) -> FragmentExtraction:
        from carmel.services.pdf_fragments import extract_fragments

        require_pypdf()
        for root in roots:
            raw_path = root / campaign / "evidence" / "literature" / sha / "raw.bin"
            if raw_path.exists():
                raw = raw_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sha:
                    pytest.skip(f"stored raw.bin under {root} is not the measured {sha}")
                return extract_fragments(raw)
        pytest.skip(f"target corpus for {sha} is not present under any known workspace root")

    def _assert_rediscovered(self, extraction: FragmentExtraction, known_footprint: object) -> None:
        from carmel.services.pdf_tables import ClaimedFootprint

        assert isinstance(known_footprint, ClaimedFootprint)
        hand = build_inventory(extraction, known_footprint)
        assert hand.complete, "the hand-pinned footprint no longer derives a grid"
        rediscovered = [
            p
            for p in propose_tables(extraction).proposals
            if p.footprint.page == known_footprint.page
            and {(c.row, c.col): c.text for c in p.inventory.cells} == {(c.row, c.col): c.text for c in hand.cells}
        ]
        assert rediscovered, "no proposed footprint re-derived the hand-pinned grid"
        assert rediscovered[0].footprint.geometry_origin is FootprintGeometryOrigin.DERIVED
        # The proposed box lies within the hand-drawn box on the y-axis and captures every row
        # (the hand-drawn box carries gutter/margin slack the derived one does not claim).
        assert rediscovered[0].footprint.page == known_footprint.page

    def test_condition_set_target_is_rediscovered_from_geometry(self) -> None:
        from carmel.services import condition_set_target as t

        extraction = self._extraction(t.TARGET_DOCUMENT_SHA256, t.TARGET_WORKSPACES_ROOTS, t.TARGET_CAMPAIGN)
        self._assert_rediscovered(extraction, t.TARGET_TABLE_FOOTPRINT)

    def test_tabular_dataset_target_is_rediscovered_from_geometry(self) -> None:
        from carmel.services import tabular_dataset_target as t

        extraction = self._extraction(t.TARGET_DOCUMENT_SHA256, t.TARGET_WORKSPACES_ROOTS, t.TARGET_CAMPAIGN)
        self._assert_rediscovered(extraction, t.TARGET_TABLE_FOOTPRINT)


class TestTheGutterMidpointGuard:
    def test_column_right_places_the_edge_inside_the_gutter(self) -> None:
        # body ink to 140, neighbour at 320: the edge is the gutter midpoint, not the neighbour.
        band = [frag("Fuel", 50.0, 70.0, 185.0), frag("Air", 120.0, 140.0, 185.0)]
        assert _column_right([band], 50.0, 320.0) == (140.0 + 320.0) / 2.0

    def test_column_right_with_no_neighbour_is_the_body_ink(self) -> None:
        band = [frag("Fuel", 50.0, 70.0, 185.0), frag("Air", 120.0, 140.0, 185.0)]
        assert _column_right([band], 50.0, None) == 140.0


class TestAffixBandsDoNotShrinkTheRowPitch:
    """A subscript is not a row, and must not be counted as one when measuring row rhythm.

    Every threshold in the body walk is a multiple of the median in-column pitch, so letting
    affix bands into that median shrinks the threshold and truncates the table. A formula-heavy
    chemistry table is the realistic case: ``H2`` and ``CO2`` put a subscript band under most
    rows, and a row carrying both a sub- and a superscript puts TWO.
    """

    @staticmethod
    def _affix_heavy_page() -> FragmentExtraction:
        # Three rows a clean 40 pt apart, each trailed by two affix bands 3 pt and 6 pt below it.
        # Raw band gaps are then [3, 3, 34] repeating -- median 3, so the end-of-table threshold
        # becomes 6 pt and the first real 34 pt row gap ends the table after row one.
        fragments = [frag("Table 1 conditions", 50.0, 160.0, 260.0)]
        for row_y in (240.0, 200.0, 160.0):
            fragments += [frag("H", 50.0, 60.0, row_y), frag("CO", 120.0, 140.0, row_y)]
            for drop in (3.0, 6.0):
                fragments += [
                    frag("2", 60.0, 64.0, row_y - drop, font_height=5.0),
                    frag("x", 140.0, 144.0, row_y - drop, font_height=5.0),
                ]
        return page(*fragments)

    def test_the_pitch_is_measured_over_rows_not_over_raw_bands(self) -> None:
        bands = _bands(list(self._affix_heavy_page().fragments))
        raw = [bands[i - 1][0] - bands[i][0] for i in range(1, len(bands))]
        assert statistics.median(raw) == 3.0, "fixture no longer reproduces affix-dominated bands"
        # Rows only: the caption-to-row gap plus the two true 40 pt row pitches.
        assert _row_pitch_gaps(bands) == [20.0, 40.0, 40.0]

    def test_a_run_of_two_affix_bands_does_not_leak_its_tail_into_the_pitch(self) -> None:
        # The second affix under a row, compared against the FIRST affix, does not look like an
        # affix -- both are small. Judging against the last accepted ROW is what excludes it.
        bands = _bands(list(self._affix_heavy_page().fragments))
        assert all(gap >= 20.0 for gap in _row_pitch_gaps(bands))

    def test_the_box_reaches_the_last_row_instead_of_stopping_at_the_first(self) -> None:
        # Asserted on the FOOTPRINT, which is what this module derives. How build_inventory then
        # folds affix bands into rows is its own contract (and its own refusals) -- the proposer's
        # job is to hand it a box that contains the whole table rather than one truncated by a
        # threshold that affix gaps shrank.
        outcome = propose_tables(self._affix_heavy_page())
        assert outcome.refusals == ()
        assert len(outcome.proposals) == 1
        footprint = outcome.proposals[0].footprint
        # Every one of the three row baselines lies inside the box; the affix-depressed threshold
        # would have ended it just below 240.
        for row_baseline in (240.0, 200.0, 160.0):
            assert footprint.y_bottom < row_baseline < footprint.y_top

    def test_the_last_row_not_a_trailing_subscript_sets_the_lower_edge(self) -> None:
        # y_bottom is half a row pitch below the LAST ROW (160.0), never below the subscript at
        # 154.0 that trails it -- otherwise the edge is derived from an affix baseline.
        footprint = propose_tables(self._affix_heavy_page()).proposals[0].footprint
        assert footprint.y_bottom == 160.0 - 40.0 / 2.0
