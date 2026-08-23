"""Acceptance tests against THE real target table: Table 1, page 4 of
``10.1016-j.ijhydene.2013.10.164``.

Every other test module in this repository is synthetic by policy -- corpus paper text is
copyrighted and non-redistributable, so no document may be checked in. This one is the
deliberate, narrow exception in the OTHER direction: the three extraction defects this
branch repairs were measured on one published table, and a suite that never touches that
table can prove each mechanism while silently failing the page they exist for. The
repository still carries no paper: the document is read from the operator's corpus inbox
at runtime, and every test SKIPS -- never passes -- when it is absent or is not
byte-for-byte the measured document. What is checked in is a locator (the caption anchor
and box coordinates, which are this project's own footprint claim, not prose) and the two
cell values the acceptance criterion names.

The expectations here are MEASUREMENTS, not aspirations: each refusal below was observed
by the read-only census that preceded the implementation, and the joint 9x3/20-cell
result was observed under all three fixes together before it was promised.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from carmel.services.pdf_fragments import FragmentExtraction, GlyphMapping, extract_fragments
from carmel.services.pdf_tables import ClaimedFootprint, InventoryRefusalReason, build_inventory
from tests.pypdf_gate import require_pypdf

#: Where the operator's corpus keeps the target document. Runtime-read, never shipped.
_DOCUMENT = (
    Path.home() / "runs/carmel/workspaces/live-syngas/literature_requests/inbox/10.1016-j.ijhydene.2013.10.164.pdf"
)

#: The exact bytes every measurement in this module was taken against. A different file
#: at the same path is a different document, and asserting these expectations against it
#: would report drift in the code when what drifted is the input -- so the gate skips,
#: naming the mismatch, rather than failing or quietly proceeding.
_DOCUMENT_SHA256 = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"

#: The registered whole-table footprint: the acceptance criterion's box. ``x_end`` may
#: sit anywhere in the measured admissible window [280.4, 304.3]; 290 is the registered
#: representative, not a tuned value.
WHOLE_TABLE = ClaimedFootprint(
    page=4,
    x_start=50.0,
    x_end=290.0,
    y_top=145.0,
    y_bottom=45.0,
    caption_text="Table1eMeasurementconditions.",
    caption_x_start=53.0,
    caption_baseline_y=148.8,
)


def _target_extraction() -> FragmentExtraction:
    require_pypdf()
    if not _DOCUMENT.exists():
        pytest.skip(f"target corpus document is not present at {_DOCUMENT}")
    data = _DOCUMENT.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != _DOCUMENT_SHA256:
        pytest.skip(f"document at {_DOCUMENT} is {actual}, not the measured {_DOCUMENT_SHA256}")
    extraction = extract_fragments(data)
    assert extraction.available, f"the fragment lane refused the target document: {extraction.status}"
    return extraction


def _with_the_degree_glyph_forced_mapped(extraction: FragmentExtraction) -> FragmentExtraction:
    """The measurement instrument for isolating the containment fix from the glyph one.

    Table 1's header carries one unmapped glyph (the ``/C14`` degree sign at y=64.06).
    Forcing it MAPPED in memory is not a repair -- the placeholder text stays -- it is
    the knob that lets a test observe what refuses NEXT once the unmapped-member check
    is out of the way, without asserting anything about glyph semantics.
    """
    forced = [
        replace(f, glyph_mapping=GlyphMapping.MAPPED)
        if f.page == WHOLE_TABLE.page and f.text == "/C14" and f.glyph_mapping is GlyphMapping.UNMAPPED
        else f
        for f in extraction.fragments
    ]
    assert any(f.text == "/C14" for f in forced), "the measured /C14 fragment is gone; the premise moved"
    return replace(extraction, fragments=tuple(forced))


class TestInkContainmentOnTheTargetPage:
    """Commit 1's invariants 2 and 3, on the page that motivated them."""

    def test_ink_admits_the_bridging_fragment_and_it_still_bridges_columns(self) -> None:
        """THE load-bearing assertion, and deliberately not a weaker one.

        With the unmapped glyph forced out of the way and the sub-fragment split not in
        play, the box must reach ``column_structure_unresolved`` -- the refusal that can
        only fire if the once-refused ``(91)Tj`` fragment was ADMITTED as a member and
        its span still bridges the value columns. Asserting merely "no longer refuses
        ``straddling_fragment_at_the_box_edge``" would also pass if the containment
        change had silently DROPPED the fragment; measured, that inventory COMPLETES
        with the pressure row reading ``1e`` beside ``e8`` -- so reaching this refusal,
        with this detail, is what proves admission.
        """
        inventory = build_inventory(_with_the_degree_glyph_forced_mapped(_target_extraction()), WHOLE_TABLE)

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED]
        assert "resolves 3 blocks where the table resolves 2" in inventory.refusals[0].detail

    def test_without_the_glyph_repair_the_box_refuses_the_unmapped_member(self) -> None:
        """The refusal chain's middle link, pinned so the order stays observable: ink
        containment alone moves the refusal from the straddler to the unmapped glyph."""
        inventory = build_inventory(_target_extraction(), WHOLE_TABLE)

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.UNMAPPED_MEMBER]

    def test_nine_genuine_ink_straddlers_still_refuse(self) -> None:
        """The negative invariant. Widening ``x_end`` to 324.5 cuts through the page's
        right-hand prose column: nine ordinary body-text fragments with 0.0000 pt of
        trailing overhang, i.e. fragments whose INK the edge genuinely bisects. They
        must refuse exactly as before the ink extent existed."""
        extraction = _target_extraction()

        inventory = build_inventory(extraction, replace(WHOLE_TABLE, x_end=324.5))

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.STRADDLING_FRAGMENT_AT_THE_BOX_EDGE]
        assert "9 fragment(s)" in inventory.refusals[0].detail
        cut = [
            f
            for f in extraction.fragments
            if f.page == WHOLE_TABLE.page
            and f.text.strip()
            and not f.rotated
            and WHOLE_TABLE.y_bottom <= f.baseline_y <= WHOLE_TABLE.y_top
            and f.x_start < 324.5 < (f.x_end if f.ink_x_end is None else f.ink_x_end)
        ]
        assert len(cut) == 9
        for f in cut:
            assert f.ink_x_end is not None
            assert f.x_end - f.ink_x_end == pytest.approx(0.0, abs=1e-9), (
                "a fragment this test counts as a genuine straddler carries trailing "
                "spacing, so it no longer measures what it claims to"
            )
