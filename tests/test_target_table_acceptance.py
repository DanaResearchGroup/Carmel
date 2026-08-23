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

import carmel.services.pdf_fragments as pdf_fragments
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


def _with_the_split_disabled(extraction: FragmentExtraction) -> FragmentExtraction:
    """The "split disabled" knob: strip every fragment's per-glyph evidence.

    The sub-fragment split is not a flag -- it runs exactly where glyph evidence
    exists -- so disabling it faithfully means removing the evidence and nothing
    else. Ink extents stay, so containment still judges by ink; what goes away is the
    per-glyph partition, which sends `build_inventory` down its documented fallback
    (hull-derived columns, no split) -- the same path every pre-split document takes.
    """
    return replace(
        extraction,
        fragments=tuple(replace(f, glyph_intervals=None) for f in extraction.fragments),
    )


def _without_the_glyph_repair(monkeypatch: pytest.MonkeyPatch) -> FragmentExtraction:
    """The target's extraction with the repair registry emptied: the "repair disabled"
    knob the isolation tests below need, applied at the registry rather than by
    editing fragments, so what is measured is the shipped pipeline minus exactly one
    table."""
    monkeypatch.setattr(pdf_fragments, "_GLYPH_REPAIRS", ())
    return _target_extraction()


class TestInkContainmentOnTheTargetPage:
    """Commit 1's invariants 2 and 3, on the page that motivated them."""

    def test_ink_admits_the_bridging_fragment_and_it_still_bridges_columns(self, monkeypatch) -> None:
        """THE load-bearing assertion, and deliberately not a weaker one.

        With the glyph repair DISABLED, the unmapped glyph forced out of the way in
        memory, and the sub-fragment split not in play, the box must reach
        ``column_structure_unresolved`` -- the refusal that can only fire if the
        once-refused ``(91)Tj`` fragment was ADMITTED as a member and its span still
        bridges the value columns. Asserting merely "no longer refuses
        ``straddling_fragment_at_the_box_edge``" would also pass if the containment
        change had silently DROPPED the fragment; measured, that inventory COMPLETES
        with the pressure row reading ``1e`` beside ``e8`` -- so reaching this refusal,
        with this detail, is what proves admission.
        """
        extraction = _with_the_split_disabled(
            _with_the_degree_glyph_forced_mapped(_without_the_glyph_repair(monkeypatch))
        )

        inventory = build_inventory(extraction, WHOLE_TABLE)

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED]
        assert "resolves 3 blocks where the table resolves 2" in inventory.refusals[0].detail

    def test_without_the_glyph_repair_the_box_refuses_the_unmapped_member(self, monkeypatch) -> None:
        """The refusal chain's middle link, pinned so the order stays observable: ink
        containment alone moves the refusal from the straddler to the unmapped glyph."""
        inventory = build_inventory(_without_the_glyph_repair(monkeypatch), WHOLE_TABLE)

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


class TestTheDegreeGlyphRepairOnTheTargetPage:
    """Commit 2's verifier: the header decodes, and the unmapped-member refusal is gone.

    Deliberately NOT verified by table completion -- completion needs the third fix
    too, and would anyway be no justification for a glyph mapping.
    """

    def test_the_temperature_header_glyph_decodes_to_the_degree_sign(self) -> None:
        """The shipped repair, observed end to end on the real bytes: same code, same
        geometry, the placeholder text replaced by the degree sign, and the fragment
        saying REPAIRED -- Carmel's conclusion, never passed off as the document's."""
        extraction = _target_extraction()

        degree = [f for f in extraction.fragments if f.page == WHOLE_TABLE.page and abs(f.baseline_y - 64.063) < 0.01]
        assert len(degree) == 1
        fragment = degree[0]
        assert fragment.text == "\u00b0"
        assert fragment.glyph_mapping is GlyphMapping.REPAIRED
        # The repair is textual only: the measured zero-advance geometry is untouched.
        assert fragment.x_start == pytest.approx(61.7952, abs=1e-4)
        assert fragment.x_end == fragment.x_start

    def test_the_inventory_no_longer_refuses_the_unmapped_member(self) -> None:
        """With the repair live (and the split disabled, so this is not verified by
        table completion) the box's refusal moves PAST unmapped_member to the next
        link of the measured chain, exactly one link."""
        inventory = build_inventory(_with_the_split_disabled(_target_extraction()), WHOLE_TABLE)

        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.COLUMN_STRUCTURE_UNRESOLVED]

    def test_only_the_registered_glyph_is_repaired_anywhere_in_the_document(self) -> None:
        """The scope doing its work on the real input: one document, one font, one
        glyph. The document draws that code in that font FOUR times (three prose uses
        on page 2, the table header on page 4) -- each is the identical conclusion, so
        each repairs -- and nothing else does: the same font's /C0, /C6 and /C20, for
        which no evidence was recorded, stay UNMAPPED and keep refusing with their
        glyph named."""
        extraction = _target_extraction()

        repaired = [f for f in extraction.fragments if f.glyph_mapping is GlyphMapping.REPAIRED]
        assert len(repaired) == 4
        assert all(f.text == "°" for f in repaired)
        assert {f.page for f in repaired} == {2, WHOLE_TABLE.page}

        unmapped = {f.text for f in extraction.fragments if f.glyph_mapping is GlyphMapping.UNMAPPED}
        assert unmapped == {"/C0", "/C6", "/C20"}


class TestJointAcceptance:
    """The pull request's own verifier: under all three fixes together, the table
    completes -- measured before it was promised, asserted here forever after."""

    def test_the_measurement_table_extracts_completely(self) -> None:
        inventory = build_inventory(_target_extraction(), WHOLE_TABLE)

        assert inventory.refusals == ()
        assert len(inventory.rows) == 9
        assert len(inventory.column_bounds) == 3
        assert len(inventory.cells) == 20

    def test_the_pressure_row_reads_1e9_beside_1e8(self) -> None:
        """The row the whole branch exists for: its middle value was drawn by one
        show operator whose `9` closes this cell and whose `1` opens the next."""
        inventory = build_inventory(_target_extraction(), WHOLE_TABLE)

        pressure = [c for c in inventory.cells if any(m.split for m in c.members)]
        assert len(pressure) == 2
        rows = {c.row for c in pressure}
        assert len(rows) == 1
        ordinal = rows.pop()
        by_col = {c.col: c.text for c in inventory.cells if c.row == ordinal}
        assert by_col[1] == "1e9"
        assert by_col[2] == "1e8"

    def test_the_temperature_header_row_carries_the_repaired_degree_sign(self) -> None:
        inventory = build_inventory(_target_extraction(), WHOLE_TABLE)

        header = next(c for c in inventory.cells if "\u00b0" in c.text)
        assert header.text == "T(\u00b0C)"
        assert header.col == 0

    def test_the_complete_inventory_replays_reproduced(self) -> None:
        """The stored form of the same result: a version-2 record whose member
        records (parent digest + glyph range) recompute bit-for-bit from the raw
        bytes -- the record is a claim the document can refute, and does not."""
        import hashlib as _hashlib

        from carmel.services.pdf_table_record import (
            InventoryVerificationStatus,
            inventory_record_payload,
            verify_inventory_record,
        )

        # _target_extraction() owns this module's skip gate (document present, and the
        # measured bytes). Read the document only after it has run: reading first makes
        # this the one test of the ten that raises FileNotFoundError instead of skipping
        # on a machine without the corpus -- which is every CI runner.
        inventory = build_inventory(_target_extraction(), WHOLE_TABLE)
        data = _DOCUMENT.read_bytes()
        payload = inventory_record_payload(inventory, raw_sha256=_hashlib.sha256(data).hexdigest())

        result = verify_inventory_record(payload, data)

        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.identity_moved == ()
