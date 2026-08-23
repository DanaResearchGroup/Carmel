"""The symbol-font-Latin impostor census: does it catch a glyph whose OUTLINE contradicts
the character it decodes to, and does it stay a triage tool rather than a repair engine?

Every fixture here is SYNTHETIC -- a CFF font built glyph by glyph with fontTools, embedded
in a minimal PDF -- so nothing copyrighted is checked in and the test runs anywhere the dev
extra is installed. The impostor is the real fault reduced to its essence: a font whose
`e` slot draws a horizontal bar (an en-dash) and whose `f` slot draws a bowl-with-descender
(a phi), each named and encoded as the Latin letter it is NOT.
"""

from __future__ import annotations

import pytest

fontTools = pytest.importorskip("fontTools", reason="census is a dev-only tool; fontTools is a dev dependency")

from fontTools.fontBuilder import FontBuilder  # noqa: E402
from fontTools.pens.t2CharStringPen import T2CharStringPen  # noqa: E402

from carmel.services.glyph_impostor_census import (  # noqa: E402
    _outline_flags,
    scan_document,
)
from tests.test_pdf_fragments import _pdf  # noqa: E402 - reuse the minimal-PDF assembler


def _cff_program(glyph: str, draw) -> bytes:
    """A one-real-glyph CFF program: ``.notdef`` plus ``glyph``, whose outline ``draw``
    lays down with a :class:`T2CharStringPen`. Returns the raw ``CFF `` table bytes, i.e.
    exactly what a PDF ``/FontFile3`` stream carries."""
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder([".notdef", glyph])
    fb.setupCharacterMap({ord(glyph): glyph})
    notdef = T2CharStringPen(0, None)
    real = T2CharStringPen(750, None)
    draw(real)
    fb.setupCFF("Synth", {"FullName": "Synth"}, {".notdef": notdef.getCharString(), glyph: real.getCharString()}, {})
    fb.setupHorizontalMetrics({".notdef": (0, 0), glyph: (750, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Synth", "styleName": "Reg"})
    fb.setupOS2()
    fb.setupPost()
    return fb.font["CFF "].compile(fb.font)


def _bar(pen) -> None:
    """A wide, short horizontal bar floating on the math axis -- an en-dash."""
    pen.moveTo((50, 274))
    pen.lineTo((700, 274))
    pen.lineTo((700, 326))
    pen.lineTo((50, 326))
    pen.closePath()


def _pdf_drawing(char: str, program: bytes, count: int = 1) -> bytes:
    """A one-page PDF drawing ``char`` ``count`` times from a WinAnsi Type1 font whose
    embedded ``/FontFile3`` is ``program``. No ``/Differences``, no ``/ToUnicode``: the byte
    decodes to the Latin letter by the base encoding alone, as the real impostors do."""
    shows = "".join(f"BT /F1 10 Tf 72 {700 - 12 * i} Td ({char}) Tj ET\n" for i in range(count)).encode("latin-1")
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(shows)).encode() + b" >>\nstream\n" + shows + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Synth /Encoding /WinAnsiEncoding /FontDescriptor 6 0 R >>",
            b"<< /Type /FontDescriptor /FontName /Synth /Flags 32 /FontFile3 7 0 R >>",
            b"<< /Length " + str(len(program)).encode() + b" /Subtype /Type1C >>\nstream\n" + program + b"\nendstream",
        ]
    )


class TestTheOutlineFlags:
    """The pure geometric triage, measured against the numbers the real glyphs carry."""

    def test_a_bar_decoded_to_a_letter_is_flagged(self) -> None:
        assert "HORIZONTAL_BAR" in _outline_flags("e", (50, 274, 700, 326), 1, 1000)

    def test_a_countered_descender_on_a_non_descender_letter_is_flagged(self) -> None:
        # The phi's outline decoded to 'f': three contours, deep descender.
        assert "COUNTERED_DESCENDER" in _outline_flags("f", (44, -189, 564, 660), 3, 1000)

    def test_a_genuine_e_is_not_flagged(self) -> None:
        # A real bowl on the baseline, two contours: nothing trips.
        assert _outline_flags("e", (47, -11, 509, 528), 2, 1000) == ()

    def test_a_genuine_upright_f_is_not_flagged(self) -> None:
        assert _outline_flags("f", (47, 0, 436, 777), 1, 1000) == ()

    def test_a_genuine_italic_f_is_not_flagged(self) -> None:
        # Italic f descends, but as a SINGLE open stroke -- the counter requirement is
        # what keeps this from reading as a phi.
        assert _outline_flags("f", (-121, -210, 440, 789), 1, 1000) == ()


class TestScanDocumentEndToEnd:
    def test_a_symbol_font_bar_in_an_e_slot_is_surfaced_as_a_candidate(self) -> None:
        pytest.importorskip("pypdf")
        pdf = _pdf_drawing("e", _cff_program("e", _bar), count=3)

        census = scan_document(pdf)

        assert len(census.candidates) == 1
        candidate = census.candidates[0]
        assert candidate.decoded_char == "e"
        assert candidate.glyph_name == "e"
        assert "HORIZONTAL_BAR" in candidate.flags
        assert candidate.bbox == (50, 274, 700, 326)
        assert candidate.contours == 1
        assert candidate.drawn_count == 3

    def test_a_genuine_letter_font_yields_no_candidates(self) -> None:
        pytest.importorskip("pypdf")

        def real_e(pen) -> None:
            # A crude two-contour bowl on the baseline: outer ring plus an inner counter.
            pen.moveTo((60, 0))
            pen.lineTo((460, 0))
            pen.lineTo((460, 520))
            pen.lineTo((60, 520))
            pen.closePath()
            pen.moveTo((140, 120))
            pen.lineTo((380, 120))
            pen.lineTo((380, 400))
            pen.lineTo((140, 400))
            pen.closePath()

        pdf = _pdf_drawing("e", _cff_program("e", real_e))

        census = scan_document(pdf)

        assert census.candidates == []

    def test_the_tool_emits_no_replacement_only_evidence(self) -> None:
        """The triage-not-oracle contract, asserted structurally: a candidate carries the
        evidence a human needs and NOTHING that names the character for them. If a future
        edit adds a 'suggested_replacement', this fails and the review that follows is the
        point."""
        pytest.importorskip("pypdf")
        candidate_fields = set(scan_document(_pdf_drawing("e", _cff_program("e", _bar))).candidates[0].__dict__)

        assert "replacement" not in candidate_fields
        assert "suggested_replacement" not in candidate_fields
        assert {"bbox", "contours", "flags", "decoded_char", "font_program_sha256"} <= candidate_fields
