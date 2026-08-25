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
    DocumentCensus,
    ImpostorCandidate,
    _drawn_counts,
    _iter_cff_programs,
    _main,
    _operand_bytes,
    _outline_flags,
    census,
    format_report,
    scan_document,
)
from tests.test_pdf_fragments import _pdf  # noqa: E402 - reuse the minimal-PDF assembler


def _cff_named_program(glyph_name: str, code: int, draw) -> bytes:
    """A one-real-glyph CFF program whose sole non-``.notdef`` glyph is named ``glyph_name``
    and mapped from byte ``code``, with outline ``draw``. Returns the raw ``CFF `` table
    bytes, i.e. exactly what a PDF ``/FontFile3`` stream carries. The name is decoupled from
    the code so a fixture can embed a glyph the charset names non-alphabetically (``period``)
    or names ``e`` while drawing nothing at all."""
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder([".notdef", glyph_name])
    fb.setupCharacterMap({code: glyph_name})
    notdef = T2CharStringPen(0, None)
    real = T2CharStringPen(750, None)
    draw(real)
    fb.setupCFF(
        "Synth", {"FullName": "Synth"}, {".notdef": notdef.getCharString(), glyph_name: real.getCharString()}, {}
    )
    fb.setupHorizontalMetrics({".notdef": (0, 0), glyph_name: (750, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Synth", "styleName": "Reg"})
    fb.setupOS2()
    fb.setupPost()
    return fb.font["CFF "].compile(fb.font)


def _cff_program(glyph: str, draw) -> bytes:
    """A one-real-glyph CFF program whose glyph is both named and encoded as ``glyph``."""
    return _cff_named_program(glyph, ord(glyph), draw)


def _bar(pen) -> None:
    """A wide, short horizontal bar floating on the math axis -- an en-dash."""
    pen.moveTo((50, 274))
    pen.lineTo((700, 274))
    pen.lineTo((700, 326))
    pen.lineTo((50, 326))
    pen.closePath()


def _pdf_drawing(char: str, program: bytes, count: int = 1, *, use_tj: bool = False) -> bytes:
    """A one-page PDF drawing ``char`` ``count`` times from a WinAnsi Type1 font whose
    embedded ``/FontFile3`` is ``program``. No ``/Differences``, no ``/ToUnicode``: the byte
    decodes to the Latin letter by the base encoding alone, as the real impostors do. With
    ``use_tj`` the glyph is shown through the array ``TJ`` operator instead of ``Tj``, which
    exercises the census's other drawn-count path."""
    show = f"[({char})] TJ" if use_tj else f"({char}) Tj"
    shows = "".join(f"BT /F1 10 Tf 72 {700 - 12 * i} Td {show} ET\n" for i in range(count)).encode("latin-1")
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


def _candidate(
    *,
    decoded_char: str = "e",
    glyph_name: str = "e",
    base: str = "Synth",
    flags: tuple[str, ...] = ("HORIZONTAL_BAR",),
    advance_width: int | None = 750,
    drawn_count: int | None = 3,
) -> ImpostorCandidate:
    """A fully-populated candidate for exercising the pure formatting/report code without a PDF."""
    return ImpostorCandidate(
        document_sha256="d" * 64,
        font_program_sha256="p" * 64,
        font_base_name=base,
        decoded_char=decoded_char,
        glyph_name=glyph_name,
        bbox=(50, 274, 700, 326),
        contours=1,
        advance_width=advance_width,
        units_per_em=1000,
        flags=flags,
        drawn_count=drawn_count,
    )


def _catalog(page_body: bytes, *extras: bytes) -> bytes:
    """Catalog + Pages + one page + trailing font/descriptor/program objects (obj 4, 5, ...)."""
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page_body,
            *extras,
        ]
    )


class TestCandidateSummary:
    """``summary()`` is the evidence line a human reads before authoring a repair -- a
    formatting defect here poisons every repair authored downstream, so it is pinned exactly."""

    def test_every_field_is_laid_out(self) -> None:
        line = _candidate(base="AdvPS44", flags=("HORIZONTAL_BAR",)).summary()
        assert "AdvPS44 prog=pppppppppppp " in line
        assert "glyph 'e' decodes to 'e' drawn x3" in line
        assert "bbox=[50,700]x[274,326]" in line
        assert "contours=1" in line
        assert "adv=750" in line
        assert "em=1000" in line
        assert "flags=HORIZONTAL_BAR" in line

    def test_absent_advance_and_drawn_count_render_as_question_marks(self) -> None:
        line = _candidate(advance_width=None, drawn_count=None).summary()
        assert "drawn x?" in line
        assert "adv=?" in line


class TestOperandBytes:
    """The four operand shapes pypdf can hand a show operator, resolved to the raw drawn bytes."""

    def test_prefers_original_bytes_when_present(self) -> None:
        class _Op:
            original_bytes = b"\x65"

        assert _operand_bytes(_Op()) == b"\x65"

    def test_raw_bytes_and_bytearray_pass_through(self) -> None:
        assert _operand_bytes(b"\x65") == b"\x65"
        assert _operand_bytes(bytearray(b"\x65")) == b"\x65"

    def test_str_is_latin1_encoded(self) -> None:
        assert _operand_bytes("e") == b"\x65"

    def test_unknown_operand_is_empty(self) -> None:
        assert _operand_bytes(42) == b""


class TestCensusOverPaths:
    def test_absent_file_yields_an_absent_marker_not_a_clean_scan(self, tmp_path) -> None:
        """A never-scanned document must not read as a clean one: the sentinel is the difference."""
        (result,) = census([tmp_path / "nope.pdf"])

        assert result.document_sha256 == "(absent)"
        assert result.programs_scanned == 0
        assert result.candidates == []

    def test_present_file_is_read_and_scanned(self, tmp_path) -> None:
        pytest.importorskip("pypdf")
        path = tmp_path / "doc.pdf"
        path.write_bytes(_pdf_drawing("e", _cff_program("e", _bar)))

        (result,) = census([path])

        assert result.document_sha256 != "(absent)"
        assert [c.decoded_char for c in result.candidates] == ["e"]


class TestFormatReport:
    def test_clean_document_shows_the_no_candidates_branch_and_a_zero_total(self) -> None:
        report = format_report([("clean.pdf", DocumentCensus("s" * 64, programs_scanned=1, glyphs_scanned=5))])

        assert "### clean.pdf  (1 CFF programs, 5 latin-named glyphs)" in report
        assert "(no candidates)" in report
        assert "TOTAL candidates: 0" in report

    def test_candidates_are_sorted_by_char_then_font_and_counted(self) -> None:
        doc = DocumentCensus(
            "s" * 64,
            programs_scanned=2,
            glyphs_scanned=2,
            candidates=[
                _candidate(decoded_char="f", glyph_name="f", base="Bfont", flags=("COUNTERED_DESCENDER",)),
                _candidate(decoded_char="e", glyph_name="e", base="Afont"),
            ],
        )

        report = format_report([("doc.pdf", doc)])

        assert "TOTAL candidates: 2" in report
        # Sort key (decoded_char, font_base_name): the 'e' line must precede the 'f' line.
        assert report.index("decodes to 'e'") < report.index("decodes to 'f'")


class TestMain:
    def test_no_arguments_prints_usage_and_exits_two(self, capsys) -> None:
        assert _main([]) == 2
        assert "usage:" in capsys.readouterr().out

    def test_happy_path_prints_the_report_and_exits_zero(self, tmp_path, capsys) -> None:
        pytest.importorskip("pypdf")
        path = tmp_path / "doc.pdf"
        path.write_bytes(_pdf_drawing("e", _cff_program("e", _bar)))

        assert _main([str(path)]) == 0

        out = capsys.readouterr().out
        assert "doc.pdf" in out
        assert "TOTAL candidates:" in out


class TestIterCffProgramsSkips:
    """Every structural gap on the path to a ``/FontFile3`` leaves the program unscanned."""

    def test_unreadable_mediabox_page_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        # No /MediaBox: pypdf raises when the mediabox is read, and the page is skipped.
        assert _iter_cff_programs(_catalog(b"<< /Type /Page /Parent 2 0 R >>")) == []

    def test_page_without_resources_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        assert _iter_cff_programs(_catalog(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")) == []

    def test_resources_without_font_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>"
        assert _iter_cff_programs(_catalog(page)) == []

    def test_font_without_descriptor_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X >>"
        assert _iter_cff_programs(_catalog(page, font)) == []

    def test_descriptor_without_fontfile3_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 5 0 R >>"
        descriptor = b"<< /Type /FontDescriptor /FontName /X /Flags 32 >>"
        assert _iter_cff_programs(_catalog(page, font, descriptor)) == []


class TestDrawnCountsSkips:
    """Drawn counts are an enrichment, never a correctness dependency: every malformed shape
    degrades to an empty map rather than sinking the scan."""

    def test_unparseable_document_yields_an_empty_map(self) -> None:
        pytest.importorskip("pypdf")
        assert _drawn_counts(b"not a pdf at all") == {}

    def test_page_without_resources_yields_an_empty_map(self) -> None:
        pytest.importorskip("pypdf")
        assert _drawn_counts(_catalog(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")) == {}

    def test_resources_without_font_survives_the_page(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>"
        assert _drawn_counts(_catalog(page)) == {}

    def test_font_without_descriptor_yields_an_empty_map(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X >>"
        assert _drawn_counts(_catalog(page, font)) == {}

    def test_descriptor_without_fontfile3_yields_an_empty_map(self) -> None:
        pytest.importorskip("pypdf")
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 5 0 R >>"
        descriptor = b"<< /Type /FontDescriptor /FontName /X /Flags 32 >>"
        assert _drawn_counts(_catalog(page, font, descriptor)) == {}

    def test_font_with_program_but_page_without_contents_yields_an_empty_map(self) -> None:
        pytest.importorskip("pypdf")
        program = _cff_program("e", _bar)
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 5 0 R >>"
        descriptor = b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontFile3 6 0 R >>"
        stream = (
            b"<< /Length " + str(len(program)).encode() + b" /Subtype /Type1C >>\nstream\n" + program + b"\nendstream"
        )
        assert _drawn_counts(_catalog(page, font, descriptor, stream)) == {}

    def test_glyph_shown_through_tj_array_is_counted(self) -> None:
        pytest.importorskip("pypdf")
        pdf = _pdf_drawing("e", _cff_program("e", _bar), count=2, use_tj=True)

        census = scan_document(pdf)

        assert census.candidates[0].drawn_count == 2


class TestScanDocumentSkips:
    def test_malformed_cff_program_contributes_no_candidates(self) -> None:
        pytest.importorskip("pypdf")
        pdf = _pdf_drawing("e", b"this is not a CFF program at all")

        census = scan_document(pdf)

        assert census.candidates == []
        assert census.programs_scanned == 0

    def test_non_alphabetic_glyph_name_is_not_judged(self) -> None:
        pytest.importorskip("pypdf")
        # The charset names the bar glyph 'period' -- not a single Latin letter, so the tool
        # has no character to contradict and skips it, even though its outline is a bar.
        pdf = _pdf_drawing("e", _cff_named_program("period", ord("e"), _bar))

        census = scan_document(pdf)

        assert census.candidates == []
        assert census.glyphs_scanned == 0

    def test_glyph_with_empty_outline_is_skipped(self) -> None:
        pytest.importorskip("pypdf")
        # A latin-named glyph that draws nothing: bounds are None, so there is no shape to judge.
        pdf = _pdf_drawing("e", _cff_named_program("e", ord("e"), lambda pen: None))

        census = scan_document(pdf)

        assert census.candidates == []
        assert census.glyphs_scanned == 1
