"""The row-partition barrier: a fold may not cross a printed-row boundary.

Two of these tests run against REAL corpus documents (the ``10.1115-1.4007737`` table
whose header subscript folds into the first data row, and ``10.1016-j.ijhydene.2013.10.164``
whose acceptance table and figure legend carry genuine folds that must survive). Corpus
paper text is copyrighted and non-redistributable, so no document is checked in: each such
test SKIPS -- never passes -- when the document is absent or is not byte-for-byte the
measured file, exactly as :mod:`tests.test_target_table_acceptance` does. What is checked in
is the footprint claim (box coordinates and caption anchor, this project's own claim) and the
measured refusal identities and cell values.

The barrier is derived from the RAW bands ``_bands(inside)`` before any fold, so every label
below comes from raw geometry plus the independent partition annotation, never from the
candidate fold itself -- the circular-labelling trap :func:`_row_partitions` exists to avoid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from carmel.paths import default_workspaces_root
from carmel.services.pdf_fragments import (
    FragmentExtraction,
    GlyphMapping,
    TextFragment,
    extract_fragments,
)
from carmel.services.pdf_tables import (
    AFFIX_PARENT_MARGIN,
    ClaimedFootprint,
    InventoryRefusalReason,
    _bands,
    _in_footprint,
    _is_adjacent_to,
    _looks_like_affix,
    _merge_affix_bands,
    _row_partitions,
    build_inventory,
)
from tests.pypdf_gate import require_pypdf

_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")
_INBOX = "live-syngas/literature_requests/inbox"

#: The document whose header-subscript fold is the measured corruption this branch repairs.
_FOLD_DOC = f"{_INBOX}/10.1115-1.4007737.pdf"
_FOLD_SHA256 = "c2be41381e3c55671af2912a46d5ce703c0f56cea9dadfba8789e7417059155a"

#: The document whose p4 acceptance table and p6 figure legend carry genuine folds that must
#: still be performed. Same bytes as :mod:`tests.test_target_table_acceptance` measured.
_FOLD_KEEP_DOC = f"{_INBOX}/10.1016-j.ijhydene.2013.10.164.pdf"
_FOLD_KEEP_SHA256 = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"


def _load(subpath: str, sha256: str) -> FragmentExtraction:
    require_pypdf()
    for root in _WORKSPACES_ROOTS:
        candidate = root / subpath
        if candidate.exists():
            data = candidate.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != sha256:
                pytest.skip(f"document at {candidate} is {actual}, not the measured {sha256}")
            extraction = extract_fragments(data)
            assert extraction.available, f"the fragment lane refused the document: {extraction.status}"
            return extraction
    roots = ", ".join(str(r) for r in _WORKSPACES_ROOTS)
    pytest.skip(f"corpus document {subpath} is not present under any of: {roots}")


#: HAND-DRAWN footprint. These coordinates are a human's box claim over ``10.1115-1.4007737``
#: p4 -- NOT derived by any code under test -- and the caption anchor is measured off the page.
#: The manager confirmed by hand that this exact box returns, on the tree this branch is cut
#: from (the column-guard fix, before the barrier), a COMPLETE 92-cell / 23-row / 4-column
#: grid with ``refusals == ()`` whose phi=0.5 row reads ``L;u67.2`` where the page prints
#: ``67.2`` -- a silent corruption. The barrier turns that into a refusal.
_REPRO_BOX = ClaimedFootprint(
    page=4,
    x_start=305.0,
    x_end=552.5,
    y_top=735.0,
    y_bottom=500.0,
    caption_text="range of equivalence ratios",
    caption_x_start=311.98,
    caption_baseline_y=750.274,
)


def _geometric_parent(bands: list[tuple[float, list[TextFragment]]], index: int) -> int | None:
    """The parent the affix logic would pick from SHAPE, ADJACENCY and proximity -- the
    pre-barrier decision, replaying ``_merge_affix_bands`` pass 1 for one band. Duplicated here
    on purpose: it is the ``before`` state, annotated independently of the partition, so a test
    can show the fold that WOULD happen without asking the barrier. Returns ``None`` when the
    band is its own row or the choice is ambiguous (which the merge itself refuses)."""
    y, band = bands[index]
    above = bands[index - 1][1] if index > 0 else []
    below = bands[index + 1][1] if index + 1 < len(bands) else []
    fits_above = _looks_like_affix(band, above) and _is_adjacent_to(band, above)
    fits_below = _looks_like_affix(band, below) and _is_adjacent_to(band, below)
    if fits_above and fits_below:
        gap_above = abs(bands[index - 1][0] - y)
        gap_below = abs(y - bands[index + 1][0])
        near, far = min(gap_above, gap_below), max(gap_above, gap_below)
        if far <= 0 or near > AFFIX_PARENT_MARGIN * far:
            return None  # ambiguous -- the merge refuses, not a fold
        fits_above, fits_below = gap_above < gap_below, gap_below < gap_above
    return index - 1 if fits_above else index + 1 if fits_below else None


# --------------------------------------------------------------------------------------------
# Verifier clause 1: the corrupting fold refuses, by the SPECIFIC reason.
# --------------------------------------------------------------------------------------------


def test_repro_box_refuses_by_the_named_reason_not_the_corrupted_value() -> None:
    """Clause 1, pinning the REFUSE arm (not the value-reads-67.2 arm).

    The design refuses because the affix band's only geometric parent is a row away: there is
    no same-partition parent to fold into, and folding across is what corrupts the cell. So the
    honest post-fix outcome is a refusal, and this test pins its IDENTITY -- not that the tuple
    is non-empty. Other guards on this page refuse for unrelated reasons
    (``ORPHANED_BAND_ABOVE_THE_BOX``, ``STRADDLING_FRAGMENT_AT_THE_BOX_EDGE``,
    ``COLUMN_STRUCTURE_UNRESOLVED``); asserting the exact single-element tuple proves it is the
    barrier that fired and could not be satisfied by code this branch did not write.

    Before/after, shown together: the ``before`` fold is annotated from raw geometry -- the
    affix band's geometric parent is the data row across a partition boundary -- and the
    ``after`` is that ``build_inventory`` refuses with ``AFFIX_CROSSES_ROW_PARTITION`` and
    inventories no cell reading ``L;u67.2``.
    """
    extraction = _load(_FOLD_DOC, _FOLD_SHA256)

    # `before`, annotated from raw bands: the band at 729.524 is affix-shaped, its geometric
    # parent is the data row at 713.537, and that parent is in a DIFFERENT partition.
    bands = _bands(_in_footprint(extraction, _REPRO_BOX))
    partitions = _row_partitions(bands)
    affix = next(i for i, (y, _b) in enumerate(bands) if round(y, 3) == 729.524)
    data_row = next(i for i, (y, _b) in enumerate(bands) if round(y, 3) == 713.537)
    assert _geometric_parent(bands, affix) == data_row  # the fold that WOULD happen
    assert partitions[affix] != partitions[data_row]  # and it crosses a row boundary

    # `after`: the barrier refuses with the specific reason, and the corruption never reaches
    # a cell.
    inventory = build_inventory(extraction, _REPRO_BOX)
    assert tuple(r.reason for r in inventory.refusals) == (InventoryRefusalReason.AFFIX_CROSSES_ROW_PARTITION,)
    assert all("L;u67.2" not in cell.text for cell in inventory.cells)
    assert inventory.cells == ()  # a refused inventory carries no cells at all


# --------------------------------------------------------------------------------------------
# Verifier clause 2: the barrier does not unattach genuine folds. Boundary + one-step-past,
# the six p4 acceptance folds, and the p6 legend subscripts (negative controls).
# --------------------------------------------------------------------------------------------


def _frag(text: str, x_start: float, x_end: float, baseline_y: float, font_height: float) -> TextFragment:
    return TextFragment(
        page=1,
        text=text,
        x_start=x_start,
        x_end=x_end,
        baseline_y=baseline_y,
        font_height=font_height,
        rotated=False,
        glyph_mapping=GlyphMapping.MAPPED,
        ink_x_end=None,
        glyph_intervals=None,
    )


def _two_band_partition(affix_baseline: float, parent_baseline: float) -> list[int]:
    """Partition labels for an affix band above a full-height parent band. The parent's ink is
    ``[parent_baseline, parent_baseline + 8]``; the affix's is ``[affix_baseline, +5]``."""
    bands = [
        (affix_baseline, [_frag("2", 60.0, 63.0, affix_baseline, 5.0)]),
        (parent_baseline, [_frag("124.4", 53.0, 70.0, parent_baseline, 8.0)]),
    ]
    return _row_partitions(bands)


def test_partition_boundary_is_pinned_at_the_edge_and_one_step_past() -> None:
    """The barrier's boundary, pinned three ways against the ink-overlap edge.

    A parent with baseline 100.0 has ink up to 108.0; an affix at baseline 101.0 (ink from
    101.0) shares its partition while the affix's own ink bottom (101.0) stays below the
    parent's ink top. The overlap ends the instant the parent's ink top drops to the affix
    baseline -- parent baseline 93.0 gives ink top 101.0, exactly touching. ``>`` is strict, so
    touching is already a boundary, and one step further apart is unambiguously across it.
    """
    # Within the partition: parent ink top 108.0 sits well above the affix baseline 101.0.
    assert _two_band_partition(affix_baseline=101.0, parent_baseline=100.0) == [0, 0]
    # AT the boundary: parent ink top == affix baseline (101.0); strict overlap is False.
    assert _two_band_partition(affix_baseline=101.0, parent_baseline=93.0) == [0, 1]
    # One step past: parent a point lower still, ink top 100.0 below the affix baseline.
    assert _two_band_partition(affix_baseline=101.0, parent_baseline=92.0) == [0, 1]


def test_merge_folds_within_a_partition_and_refuses_across_one() -> None:
    """The same two-band shape, taken through ``_merge_affix_bands`` on each side of the edge.

    Negative control (within): the affix folds and the merge returns no refusal, its folded
    baseline recorded on the parent row. Positive control (across): the identical affix, with
    the parent one row-pitch lower, refuses by the named reason instead of folding.
    """
    within = [
        (101.0, [_frag("2", 58.0, 61.0, 101.0, 5.0)]),
        (100.0, [_frag("124.4", 53.0, 70.0, 100.0, 8.0)]),
    ]
    merged, refusal = _merge_affix_bands(within)
    assert refusal is None
    assert [folded for _y, _members, folded in merged] == [[101.0]]  # the affix folded in

    across = [
        (101.0, [_frag("2", 58.0, 61.0, 101.0, 5.0)]),
        (92.0, [_frag("124.4", 53.0, 70.0, 92.0, 8.0)]),
    ]
    _merged, refusal = _merge_affix_bands(across)
    assert refusal is not None
    assert refusal.reason is InventoryRefusalReason.AFFIX_CROSSES_ROW_PARTITION


def test_acceptance_table_keeps_all_six_within_partition_folds() -> None:
    """Clause 2, negative control on real ink: the ``ijhydene`` p4 acceptance table.

    Its six affix folds (five subscript ``2`` bands and one degree-sign superscript) are all
    within-partition, so the barrier is silent and the table stays COMPLETE. This asserts both
    halves: no fold in it crosses a partition (so none is unattached), and the inventory still
    returns its full 9-row / 20-cell grid with no refusal.
    """
    extraction = _load(_FOLD_KEEP_DOC, _FOLD_KEEP_SHA256)
    footprint = ClaimedFootprint(
        page=4,
        x_start=50.0,
        x_end=290.0,
        y_top=145.0,
        y_bottom=45.0,
        caption_text="Table1–Measurementconditions.",
        caption_x_start=53.0,
        caption_baseline_y=148.8,
    )

    bands = _bands(_in_footprint(extraction, footprint))
    partitions = _row_partitions(bands)
    folds = [(i, _geometric_parent(bands, i)) for i in range(len(bands))]
    folds = [(i, p) for i, p in folds if p is not None]
    assert len(folds) == 6  # the six genuine folds
    assert all(partitions[i] == partitions[p] for i, p in folds)  # none crosses -> none unattached

    inventory = build_inventory(extraction, footprint)
    assert inventory.refusals == ()
    assert len(inventory.rows) == 9
    assert len(inventory.cells) == 20


def test_p6_legend_subscripts_still_fold() -> None:
    """Clause 2, second negative control: three figure-legend subscripts on ``ijhydene`` p6.

    The right-hand legend column carries the ``2`` of H₂ and the two ``2``s of O₂:N₂ -- three
    subscript glyphs across two affix bands. They fold into their legend lines, within
    partition, and the merge returns no refusal: the barrier does not touch them.
    """
    extraction = _load(_FOLD_KEEP_DOC, _FOLD_KEEP_SHA256)
    # The right legend column, isolated by x and a y-window over its two legend lines. This is a
    # figure legend, not a table, so it is exercised at the merge (which is where the barrier
    # lives) rather than through a full footprint inventory.
    legend = [
        f
        for f in extraction.fragments
        if f.page == 6 and not f.rotated and f.x_start >= 380.0 and 695.0 <= f.baseline_y <= 725.0
    ]
    bands = _bands(legend)
    partitions = _row_partitions(bands)
    merged, refusal = _merge_affix_bands(bands)
    assert refusal is None

    folds = [(i, _geometric_parent(bands, i)) for i in range(len(bands))]
    folds = [(i, p) for i, p in folds if p is not None]
    assert len(folds) == 2  # the two subscript bands
    assert all(partitions[i] == partitions[p] for i, p in folds)  # both within partition

    subscript_glyphs = sum(len("".join(fr.text for fr in bands[i][1])) for i, _p in folds)
    assert subscript_glyphs == 3  # '2' of H2, plus '2''2' of O2:N2
    assert sum(1 for _y, _members, folded in merged if folded) == 2  # both folds performed


# --------------------------------------------------------------------------------------------
# Verifier clause 3: a real corpus subscript is STILL folded -- asserted positively.
# --------------------------------------------------------------------------------------------


def test_a_real_h2_subscript_is_still_folded_into_its_cell() -> None:
    """Clause 3: the ``2`` of H₂ in the ``ijhydene`` fuel row folds, asserted three ways.

    A barrier that refused everything would satisfy clauses 1 and 2 while destroying the
    function, so this pins a fold that must SURVIVE: ``inventory.refusals == ()``, the exact
    merged band count on the row, and the exact resulting cell text. The subscript sits on its
    own baseline (133.17) and folds up into the fuel-header row (134.48) within one partition.
    """
    extraction = _load(_FOLD_KEEP_DOC, _FOLD_KEEP_SHA256)
    footprint = ClaimedFootprint(
        page=4,
        x_start=50.0,
        x_end=290.0,
        y_top=145.0,
        y_bottom=45.0,
        caption_text="Table1–Measurementconditions.",
        caption_x_start=53.0,
        caption_baseline_y=148.8,
    )
    inventory = build_inventory(extraction, footprint)

    assert inventory.refusals == ()
    fuel_row = inventory.rows[0]
    assert len(fuel_row.merged_baselines) == 1  # exactly the one subscript band folded in
    fuel_cells = sorted(
        ((cell.col, cell.text) for cell in inventory.cells if cell.row == fuel_row.ordinal),
    )
    assert fuel_cells == [
        (0, "Fuel"),
        (1, "H2/CO(50:50%)"),
        (2, "H2/CO(85:15%)"),
    ]
