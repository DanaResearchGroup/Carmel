"""Tests for :mod:`carmel.services.pdf_fragments`.

Every fixture here is SYNTHETIC, built from raw PDF bytes in-process. Corpus paper
text is copyrighted and non-redistributable, so no real document may be checked in --
which is also why the coordinates asserted below are EXACT rather than eyeballed:
the content stream states where each glyph is placed, so ground truth is known.
"""

from __future__ import annotations

import tracemalloc
import zlib

import pytest

import carmel.services.pdf_fragments as pdf_fragments
from carmel.services.pdf_fragments import (
    MAX_PAGE_CONTENT_BYTES,
    FragmentAvailability,
    FragmentExtraction,
    GlyphMapping,
    extract_fragments,
)
from tests.pypdf_gate import require_pypdf


def _pdf(objects: list[bytes], root: int = 1) -> bytes:
    """Assemble numbered objects into a minimal, valid PDF with a real xref table."""
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root " + str(root).encode() + b" 0 R >>\n"
    out += b"startxref\n" + str(xref).encode() + b"\n%%EOF\n"
    return bytes(out)


def _one_page_pdf(stream: str) -> bytes:
    """A single-page PDF whose content stream is exactly ``stream``."""
    body = stream.encode("latin-1")
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _one_page_filtered_pdf(stream: str, *, filters: str, parms: str = "", raw: bytes | None = None) -> bytes:
    """A single-page PDF whose content stream declares ``filters``.

    Every other fixture here writes an UNFILTERED content stream, which reaches only the
    branch of :func:`~carmel.services.pdf_fragments._decoded_content_length` that needs no
    decoding at all. The bounded-decode path and its fail-closed refusals are unreachable
    without this, and the real corpus cannot stand in for it: no paper text may enter the
    repository, and all 161 corpus streams are the one filter that is allowed anyway.

    ``raw`` overrides the compressed bytes, so a test can present a stream whose declared
    filter and actual payload disagree.
    """
    body = zlib.compress(stream.encode("latin-1")) if raw is None else raw
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length "
            + str(len(body)).encode()
            + b" /Filter "
            + filters.encode()
            + (b" /DecodeParms " + parms.encode() if parms else b"")
            + b" >>\nstream\n"
            + body
            + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _two_page_pdf(first: bytes, second: bytes) -> bytes:
    """A two-page PDF whose pages carry ``first`` and ``second`` as content streams."""
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 7 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
            b"<< /Length " + str(len(first)).encode() + b" >>\nstream\n" + first + b"\nendstream",
            b"<< /Length " + str(len(second)).encode() + b" >>\nstream\n" + second + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _fragments(stream: str):
    """Fragments from a clean synthetic page.

    Asserts `lossy is False` as well: without it, a per-page failure that swallowed
    every fragment would leave these tests asserting over an empty tuple, and several
    would pass vacuously.
    """
    result = extract_fragments(_one_page_pdf(stream))
    assert result.available is True
    assert result.lossy is False, f"unexpected loss: {result.page_failures}"
    return result.fragments


def _by_text(fragments, text: str):
    matches = [f for f in fragments if f.text.strip() == text]
    assert len(matches) == 1, f"expected exactly one {text!r}, got {[f.text for f in fragments]}"
    return matches[0]


class TestGeometryIsAbsoluteAndExact:
    """The whole milestone rests on these coordinates being real page coordinates."""

    def test_each_cell_lands_at_its_stated_coordinate(self) -> None:
        require_pypdf()
        # Three columns at known x, one row at known y, placed by explicit Td offsets.
        stream = "BT /F1 10 Tf\n72 700 Td (T) Tj\n148 0 Td (P) Tj\n160 0 Td (IDT) Tj\nET"
        frags = _fragments(stream)
        assert _by_text(frags, "T").x_start == pytest.approx(72.0)
        assert _by_text(frags, "P").x_start == pytest.approx(220.0)
        assert _by_text(frags, "IDT").x_start == pytest.approx(380.0)
        for f in frags:
            assert f.baseline_y == pytest.approx(700.0)

    def test_a_row_does_not_collapse_into_one_fragment(self) -> None:
        """The `visitor_text` trap: it would report one fragment at x=72 for all three.

        If this ever fails by returning a single merged fragment, the per-column x is
        gone and any caller is left re-splitting on whitespace -- the fabrication the
        P0-c ruling closed.
        """
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n72 700 Td (A) Tj\n148 0 Td (B) Tj\nET")
        assert len({f.x_start for f in frags}) == 2

    def test_x_end_is_past_x_start(self) -> None:
        require_pypdf()
        frag = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (Hello) Tj\nET"), "Hello")
        assert frag.x_end > frag.x_start

    def test_tm_sets_an_absolute_position(self) -> None:
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n1 0 0 1 305 512 Tm (X) Tj\nET")
        assert _by_text(frags, "X").x_start == pytest.approx(305.0)
        assert _by_text(frags, "X").baseline_y == pytest.approx(512.0)

    def test_the_recorded_height_is_the_rendered_one_not_the_tf_operand(self) -> None:
        """The trap that made ``font_size`` a constant across the whole real corpus.

        Both streams render 9-unit type. The first says so in the ``Tf`` operand; the
        second sets ``Tf /F1 1`` and carries the 9 in the text matrix instead -- which
        is what real publisher PDFs overwhelmingly do (``Tf`` was 1.0 on 78 169 of
        78 178 corpus fragments). Recording the operand makes these two disagree, and a
        font-relative threshold built on it silently becomes a constant. They must
        agree.
        """
        require_pypdf()
        by_operand = _by_text(_fragments("BT /F1 9 Tf\n1 0 0 1 72 700 Tm (A) Tj\nET"), "A")
        by_matrix = _by_text(_fragments("BT /F1 1 Tf\n9 0 0 9 72 700 Tm (A) Tj\nET"), "A")
        assert by_operand.font_height == pytest.approx(9.0)
        assert by_matrix.font_height == pytest.approx(9.0)

    def test_td_capital_and_t_star_advance_lines(self) -> None:
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n72 700 TD (first) Tj\n0 -14 TD (second) Tj\nT* (third) Tj\nET")
        ys = [_by_text(frags, t).baseline_y for t in ("first", "second", "third")]
        assert ys[0] == pytest.approx(700.0)
        assert ys[1] == pytest.approx(686.0)
        # T* repeats the leading set by the preceding TD, i.e. another -14.
        assert ys[2] == pytest.approx(672.0)

    def test_a_kerned_tj_array_stays_one_fragment_at_its_start(self) -> None:
        """`TJ` with kerning offsets is how real PDFs emit most text."""
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n72 700 Td [(A) -120 (B) -120 (C)] TJ\nET")
        joined = "".join(f.text for f in frags)
        assert "A" in joined and "B" in joined and "C" in joined
        assert min(f.x_start for f in frags) == pytest.approx(72.0)

    def test_cm_translation_shifts_the_reported_position(self) -> None:
        """A `cm` transform nests the text in a shifted space; the fragment must
        report the SHIFTED absolute position, not the pre-transform one."""
        require_pypdf()
        plain = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (Z) Tj\nET"), "Z")
        shifted = _by_text(
            _fragments("q 1 0 0 1 100 50 cm\nBT /F1 10 Tf\n72 700 Td (Z) Tj\nET\nQ"),
            "Z",
        )
        assert shifted.x_start == pytest.approx(plain.x_start + 100.0)
        assert shifted.baseline_y == pytest.approx(plain.baseline_y + 50.0)

    def test_q_restores_the_transform(self) -> None:
        require_pypdf()
        frags = _fragments(
            "q 1 0 0 1 100 0 cm\nBT /F1 10 Tf\n72 700 Td (inside) Tj\nET\nQ\nBT /F1 10 Tf\n72 700 Td (outside) Tj\nET"
        )
        assert _by_text(frags, "inside").x_start == pytest.approx(172.0)
        assert _by_text(frags, "outside").x_start == pytest.approx(72.0)


class TestCharacterSpacingIsChargedPerGlyph:
    """`Tc` advances EVERY glyph, and pypdf's ``displaced_tx`` charges it once.

    The assertions below are deliberately width-independent: each compares the same
    string drawn with and without ``Tc``, so the font's glyph widths cancel and what is
    left is the spacing law alone. Asserting an absolute ``x_end`` would pin Helvetica's
    AFM metrics into the test and would break for a reason that has nothing to do with
    the behaviour under test.

    Measured on the real corpus, this is not a corner: 714 of 72,502 text-show
    operations report an end coordinate wrong by more than half a point, 222 of them
    containing a digit, worst case 149.8 pt, in all eight papers.
    """

    #: Five glyphs, one `Tc`, and the reported width difference tells them apart:
    #: 5 x 4 = 20 pt if spacing is charged per glyph, 4 pt if it is charged per call.
    GLYPHS = "ABCDE"
    SPACING = 4.0

    def _width(self, stream: str) -> float:
        frag = _by_text(_fragments(stream), self.GLYPHS)
        return frag.x_end - frag.x_start

    def test_spacing_widens_the_run_once_per_glyph(self) -> None:
        require_pypdf()
        plain = self._width(f"BT /F1 10 Tf\n72 700 Td ({self.GLYPHS}) Tj\nET")
        spaced = self._width(f"BT /F1 10 Tf\n{self.SPACING} Tc\n72 700 Td ({self.GLYPHS}) Tj\nET")
        assert spaced - plain == pytest.approx(len(self.GLYPHS) * self.SPACING)

    def test_horizontal_scaling_scales_the_spacing_too(self) -> None:
        """`Tz` is a percentage applied to the whole advance, spacing included."""
        require_pypdf()
        plain = self._width(f"BT /F1 10 Tf\n50 Tz\n72 700 Td ({self.GLYPHS}) Tj\nET")
        spaced = self._width(f"BT /F1 10 Tf\n50 Tz\n{self.SPACING} Tc\n72 700 Td ({self.GLYPHS}) Tj\nET")
        assert spaced - plain == pytest.approx(len(self.GLYPHS) * self.SPACING * 0.5)

    def test_a_scaled_text_matrix_scales_the_spacing_too(self) -> None:
        """The real corpus carries its size in the text matrix, not the `Tf` operand.

        So a correction applied in text space and never mapped through the matrix would
        pass the first test here and be wrong on nearly every real page.
        """
        require_pypdf()
        plain = self._width(f"BT /F1 1 Tf\n3 0 0 3 72 700 Tm ({self.GLYPHS}) Tj\nET")
        spaced = self._width(f"BT /F1 1 Tf\n{self.SPACING} Tc\n3 0 0 3 72 700 Tm ({self.GLYPHS}) Tj\nET")
        assert spaced - plain == pytest.approx(len(self.GLYPHS) * self.SPACING * 3.0)

    def test_a_single_glyph_is_charged_once(self) -> None:
        """The boundary the off-by-one lives on: one glyph owes exactly one `Tc`."""
        require_pypdf()
        plain = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (A) Tj\nET"), "A")
        spaced = _by_text(_fragments(f"BT /F1 10 Tf\n{self.SPACING} Tc\n72 700 Td (A) Tj\nET"), "A")
        assert (spaced.x_end - spaced.x_start) - (plain.x_end - plain.x_start) == pytest.approx(self.SPACING)

    def test_the_correction_leaves_unspaced_text_alone(self) -> None:
        """`Tc 0` must produce byte-identical geometry to no `Tc` at all.

        The guard against a fix that quietly re-derives every advance: 98% of corpus
        shows set no character spacing, and their coordinates must not move.
        """
        require_pypdf()
        without = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (Hello) Tj\nET"), "Hello")
        zeroed = _by_text(_fragments("BT /F1 10 Tf\n0 Tc\n72 700 Td (Hello) Tj\nET"), "Hello")
        assert zeroed.x_end == without.x_end
        assert zeroed.x_start == without.x_start

    def test_a_placeholder_glyph_name_is_one_glyph_not_its_spelling(self) -> None:
        """Two bytes that decode to ``"/C20/C21"`` owe TWO spacings, not seven.

        A font whose ``/Differences`` names glyphs the standard list does not know makes
        pypdf keep the raw name, so ONE code becomes a four-character string. Counting
        decoded characters -- or `.text` characters, which are the same eight here --
        would charge seven spacings where two are owed.

        This is the case that decides between the three plausible glyph counts, and it
        is not hypothetical: 152 corpus shows decode to a length their operand does not
        have, and the 10 of those carrying character spacing are the two largest end
        coordinate errors an earlier draft of the census reported (231 pt and 248 pt).
        """
        require_pypdf()

        def width(spacing: str) -> float:
            stream = f"BT /F1 10 Tf\n{spacing}72 700 Td (\x20\x21) Tj\nET".encode("latin-1")
            pdf = _pdf(
                [
                    b"<< /Type /Catalog /Pages 2 0 R >>",
                    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
                    b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
                    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding "
                    b"<< /Type /Encoding /Differences [32 /C20 /C21] >> >>",
                ]
            )
            result = extract_fragments(pdf)
            assert result.available is True and result.lossy is False
            frag = _by_text(result.fragments, "/C20/C21")
            # The premise of the test itself, asserted rather than assumed: if pypdf ever
            # stops expanding the name, the interesting case is gone and this test would
            # otherwise keep passing while measuring nothing.
            assert len(frag.text) == 8
            assert frag.glyph_mapping is GlyphMapping.UNMAPPED
            return frag.x_end - frag.x_start

        assert width(f"{self.SPACING} Tc\n") - width("") == pytest.approx(2 * self.SPACING)

    def test_word_spacing_is_left_where_pypdf_already_has_it_right(self) -> None:
        """`Tw` is charged per space by pypdf already; the fix must not double it."""
        require_pypdf()
        plain = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (A B) Tj\nET"), "A B")
        spaced = _by_text(_fragments("BT /F1 10 Tf\n6 Tw\n72 700 Td (A B) Tj\nET"), "A B")
        assert (spaced.x_end - spaced.x_start) - (plain.x_end - plain.x_start) == pytest.approx(6.0)


class TestPageNumbering:
    def test_a_phantom_page_tree_entry_does_not_shift_page_numbers(self) -> None:
        """pypdf counts a linearization dictionary as a page on real corpus papers.

        If that entry were counted, `second` would report page 3 instead of page 2 --
        a locator that sends a reader to the wrong page while looking checkable.
        """
        require_pypdf()
        first = b"BT /F1 10 Tf 72 700 Td (first) Tj ET"
        second = b"BT /F1 10 Tf 72 700 Td (second) Tj ET"
        pdf = _pdf(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                # The phantom (object 4) sits between two real pages in /Kids.
                b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 8 0 R >> >> /Contents 6 0 R >>",
                # A linearization parameter dictionary: no /Type, no /Contents.
                b"<< /Linearized 1 /L 1000 /O 3 /E 900 /N 2 /T 800 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 8 0 R >> >> /Contents 7 0 R >>",
                b"<< /Length " + str(len(first)).encode() + b" >>\nstream\n" + first + b"\nendstream",
                b"<< /Length " + str(len(second)).encode() + b" >>\nstream\n" + second + b"\nendstream",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            ]
        )
        result = extract_fragments(pdf)
        assert result.available is True
        assert _by_text(result.fragments, "first").page == 1
        assert _by_text(result.fragments, "second").page == 2

    def test_pages_are_one_indexed(self) -> None:
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n72 700 Td (only) Tj\nET")
        assert {f.page for f in frags} == {1}


class TestGlyphMapping:
    @pytest.mark.parametrize("marker", ["(cid:3)", "/C0"])
    def test_an_unmapped_glyph_marker_is_flagged(self, marker: str) -> None:
        """The minus sign of `n = -1.0` arrives exactly like this in a real table."""
        require_pypdf()
        escaped = marker.replace("(", r"\(").replace(")", r"\)")
        frags = _fragments(f"BT /F1 10 Tf\n72 700 Td ({escaped}) Tj\nET")
        flagged = [f for f in frags if f.glyph_mapping is GlyphMapping.UNMAPPED]
        assert flagged, f"{marker!r} should be flagged, got {[f.text for f in frags]}"

    def test_the_replacement_character_is_recognised_as_unmapped(self) -> None:
        """U+FFFD is checked against the classifier rather than end-to-end.

        It cannot be round-tripped through this test's builder: a content stream is
        latin-1 bytes, and U+FFFD has no latin-1 encoding. In a real document it does
        not arrive as a literal either -- it is what a DECODER emits for a byte no
        CMap maps, which is precisely the condition being flagged.
        """
        from carmel.services.pdf_fragments import _UNMAPPED_MARKER_RE

        assert _UNMAPPED_MARKER_RE.search("1.0�") is not None
        assert _UNMAPPED_MARKER_RE.search("1.0") is None

    def test_a_flagged_fragment_keeps_its_text_unmodified(self) -> None:
        """Flag, never repair: rewriting `/C0` to a minus is a claim about MEANING."""
        require_pypdf()
        frags = _fragments(r"BT /F1 10 Tf\n72 700 Td (/C0) Tj\nET".replace(r"\n", "\n"))
        frag = _by_text(frags, "/C0")
        assert frag.text == "/C0"
        assert frag.glyph_mapping is GlyphMapping.UNMAPPED

    def test_ordinary_text_is_not_flagged(self) -> None:
        require_pypdf()
        assert (
            _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (1200) Tj\nET"), "1200").glyph_mapping is GlyphMapping.MAPPED
        )

    def test_mojibake_is_not_flagged_and_not_repaired(self) -> None:
        """`þ` for `+` decodes "successfully" from a PDF's own broken ToUnicode.

        It is left alone deliberately: it is not an unmapped glyph, and repairing it
        would be a semantic claim. This test exists so that a later "helpful" repair
        table cannot be added here without a test going red.
        """
        require_pypdf()
        frag = _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (\xfe) Tj\nET"), "\xfe")
        assert frag.text == "\xfe"
        assert frag.glyph_mapping is GlyphMapping.MAPPED


class TestRotatedText:
    def test_rotated_text_is_retained_and_flagged(self) -> None:
        """Rotated text reaches the caller, marked, rather than vanishing.

        This does NOT guard the `strip_rotated=False` argument in the producer, and
        must not be read as doing so: mutating that argument to True leaves this test
        green, because in this pypdf the flag only filters the `BTGroup`s we discard.
        What this test does prove is the property that matters to a consumer -- that a
        rotated axis title or column header is present and identifiable rather than
        silently absent.
        """
        require_pypdf()
        # A 90-degree rotation matrix, as a rotated axis title or column header.
        frags = _fragments("BT /F1 10 Tf\n0 1 -1 0 300 400 Tm (Uncertainty) Tj\nET")
        assert frags, "rotated text was dropped"
        assert any(f.rotated for f in frags)
        assert "Uncertainty" in "".join(f.text for f in frags)

    def test_unrotated_text_is_not_marked_rotated(self) -> None:
        require_pypdf()
        assert _by_text(_fragments("BT /F1 10 Tf\n72 700 Td (flat) Tj\nET"), "flat").rotated is False


class TestFailsClosed:
    def test_a_missing_engine_internal_makes_the_result_unavailable(self) -> None:
        """A pypdf upgrade that moves these private internals must REFUSE, never
        silently return different geometry."""
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        original = mod._engine
        try:
            mod._engine = lambda: None  # type: ignore[assignment]
            result = extract_fragments(_one_page_pdf("BT /F1 10 Tf\n72 700 Td (x) Tj\nET"))
        finally:
            mod._engine = original  # type: ignore[assignment]
        assert result.available is False
        assert result.lossy is True
        assert result.fragments == ()

    def test_capability_check_rejects_params_missing_a_method_it_calls(self, monkeypatch) -> None:
        require_pypdf()
        from pypdf._text_extraction._layout_mode._text_state_params import TextStateParams

        import carmel.services.pdf_fragments as mod

        monkeypatch.delattr(TextStateParams, "word_tx", raising=True)
        assert mod._engine() is None

    def test_capability_check_rejects_reordered_constructor_fields(self, monkeypatch) -> None:
        """`TextStateParams` is CONSTRUCTED positionally, so its field ORDER is contract.

        A release that swapped two same-typed float fields -- `Tc` and `Tw`, say -- would
        keep every name, pass every `hasattr`, and silently trade character spacing for
        word spacing in every advance computed. Nothing else in the gate can see that.

        Driven from this module's side of the comparison rather than by rewriting the
        dataclass, which cannot be done without rebuilding it: the assertion is that the
        two disagreeing is what refuses, and disagreement has no preferred direction.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        swapped = list(mod._REQUIRED_PARAM_FIELD_ORDER)
        swapped[3], swapped[4] = swapped[4], swapped[3]
        assert swapped != list(mod._REQUIRED_PARAM_FIELD_ORDER)
        monkeypatch.setattr(mod, "_REQUIRED_PARAM_FIELD_ORDER", tuple(swapped))
        assert mod._engine() is None

    def test_garbage_bytes_do_not_raise(self) -> None:
        result = extract_fragments(b"not a pdf at all")
        assert result.available is False
        assert result.lossy is True

    def test_missing_pypdf_degrades_instead_of_raising(self, monkeypatch) -> None:
        """Mirrors how `_extract_pdf` degrades; pypdf is an optional extra."""
        import sys

        monkeypatch.setitem(sys.modules, "pypdf", None)
        result = extract_fragments(_one_page_pdf("BT /F1 10 Tf\n72 700 Td (x) Tj\nET"))
        assert result == FragmentExtraction(fragments=(), lossy=True, status=FragmentAvailability.ENGINE_ABSENT)

    def test_the_module_imports_without_pypdf(self, monkeypatch) -> None:
        import importlib
        import sys

        import carmel.services

        # Restoring `sys.modules` alone is NOT enough to undo a re-import, and getting
        # this wrong silently poisons every later test in the file.
        # `importlib.import_module` also rebinds the attribute on the PARENT package,
        # so `carmel.services.pdf_fragments` keeps pointing at the throwaway module
        # even after monkeypatch puts the original back into `sys.modules`. A later
        # `import carmel.services.pdf_fragments as mod` then resolves through the
        # package attribute and hands back the STALE copy -- so monkeypatching `mod`
        # patches an object the code under test never consults, and the assertions
        # evaluate against unpatched behaviour while looking like they passed.
        # Observed: it broke three later tests in this file.
        monkeypatch.setattr(carmel.services, "pdf_fragments", carmel.services.pdf_fragments)
        monkeypatch.setitem(sys.modules, "pypdf", None)
        monkeypatch.delitem(sys.modules, "carmel.services.pdf_fragments", raising=False)
        module = importlib.import_module("carmel.services.pdf_fragments")
        assert module is not None


class TestFragmentsAreNotWords:
    def test_a_fragment_is_not_promised_to_be_a_token(self) -> None:
        """Real PDFs emit ~2.3 fragments per word, and consecutive kerned fragments
        can overlap in x. Nothing here groups them -- that is the next contract."""
        require_pypdf()
        frags = _fragments("BT /F1 10 Tf\n72 700 Td (Combus) Tj\n(tion) Tj\nET")
        assert len(frags) == 2
        assert "".join(f.text for f in frags) == "Combustion"


class TestUnmappedMarkerDoesNotFireOnChemistry:
    """False positives here are not cosmetic.

    This is a combustion codebase: species labels, mixture ratios and appendix
    numbering are full of slashes followed by C and a digit. An unanchored `/C\\d+`
    substring match flags all of them as corrupt. The flag is advisory today, but it
    is being built to feed a refusal gate, and a gate that refuses `C2H4` rows would
    quietly delete real chemistry.
    """

    @pytest.mark.parametrize(
        "text",
        ["/C2H4", "C1/C2", "H2/CO", "nC7H16/O2", "A/C3", "/C2H2", "Fig/C1a"],
    )
    def test_ordinary_chemistry_is_not_flagged(self, text: str) -> None:
        from carmel.services.pdf_fragments import _UNMAPPED_MARKER_RE

        assert _UNMAPPED_MARKER_RE.search(text) is None, f"{text!r} must not be flagged"

    @pytest.mark.parametrize("text", ["/C0", "-/C0 1.0", "(cid:3)", "x(cid:127)y"])
    def test_real_markers_still_fire(self, text: str) -> None:
        from carmel.services.pdf_fragments import _UNMAPPED_MARKER_RE

        assert _UNMAPPED_MARKER_RE.search(text) is not None, f"{text!r} must be flagged"


class TestAnEngineMismatchIsNotAPageFailure:
    """`available=False` and `lossy=True` must never be conflated.

    An engine-wide incompatibility that degraded to "every page failed" would report
    `available=True` with zero fragments -- indistinguishable from a legitimately
    empty document, which is exactly the conflation this codebase forbids.
    """

    def test_an_engine_mismatch_raised_while_paging_makes_the_result_unavailable(self, monkeypatch) -> None:
        """Same raise site, two exception types, two different classifications.

        An earlier version of this test patched `_REQUIRED_PARAM_ATTRS` and passed for
        the WRONG reason: `_engine` reads that tuple too, so the refusal happened at
        the front door and the propagation path under test never ran. Mutation testing
        exposed it -- demoting the `_EngineMismatch` re-raise to `lossy = True` left
        the suite fully green. Patching the raise directly is what makes the contrast
        with `test_a_failing_page_is_named_not_merely_counted` (a ValueError from the
        very same call, which correctly degrades to lossy) actually load-bearing.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        def _mismatch(page, page_number, engine, budget):
            raise mod._EngineMismatch("pypdf TextStateParams is missing a required attribute")

        monkeypatch.setattr(mod, "_page_fragments", _mismatch)
        result = extract_fragments(_one_page_pdf("BT /F1 10 Tf\n72 700 Td (x) Tj\nET"))
        assert result.available is False
        assert result.fragments == ()
        # Crucially NOT recorded as a page failure: the engine is wrong, not the page.
        assert result.page_failures == ()

    def test_a_wrong_pypdf_version_refuses_rather_than_recalibrating(self, monkeypatch) -> None:
        """No attribute check can catch a release that keeps every name and changes
        what the numbers MEAN, so the exact pin is asserted at runtime."""
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        monkeypatch.setattr(mod, "_PINNED_PYPDF_VERSION", "0.0.0-not-a-real-version")
        assert mod._engine() is None
        result = extract_fragments(_one_page_pdf("BT /F1 10 Tf\n72 700 Td (x) Tj\nET"))
        assert result.available is False


class TestUnavailabilityIsNotOneEvent:
    """Four ways to have no fragments, and the boolean said they were one.

    Every assertion here is about WHICH state, never about whether the region refuses:
    all four refuse identically and always will, so a test that only checked refusal
    would pass under any classification at all -- including the wrong one. That is the
    vacuity this class is written to avoid.
    """

    _PDF = "BT /F1 10 Tf\n72 700 Td (x) Tj\nET"

    def test_an_absent_engine_says_nothing_about_the_document(self, monkeypatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pypdf", None)
        result = extract_fragments(_one_page_pdf(self._PDF))
        assert result.status is FragmentAvailability.ENGINE_ABSENT
        assert result.available is False
        # Nothing ran, so there is no version to record. A version here would claim an
        # extraction happened.
        assert result.pypdf_version == ""

    def test_a_refused_engine_is_not_an_absent_one(self, monkeypatch) -> None:
        """pypdf IS installed and its geometry cannot be trusted -- an alarm, where an
        absent pypdf is a supported configuration. The old boolean read them the same,
        so a broken pin looked exactly like a base-job install."""
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        monkeypatch.setattr(mod, "_engine", lambda: None)
        result = extract_fragments(_one_page_pdf(self._PDF))
        assert result.status is FragmentAvailability.ENGINE_REFUSED
        assert result.pypdf_version == ""

    def test_the_gate_being_contradicted_mid_walk_is_carmels_defect(self, monkeypatch) -> None:
        """`_engine` approved this engine and the engine then broke the same contract.

        Reported as its own state rather than folded into ENGINE_REFUSED because the
        owner differs: ENGINE_REFUSED says repair your install, this says the gate is
        incomplete. Folded together, it would send someone to reinstall a pypdf that
        is installed correctly.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        def _mismatch(page, page_number, engine, budget):
            raise mod._EngineMismatch("pypdf TextStateParams is missing a required attribute")

        monkeypatch.setattr(mod, "_page_fragments", _mismatch)
        result = extract_fragments(_one_page_pdf(self._PDF))
        assert result.status is FragmentAvailability.ENGINE_CONTRADICTED_GATE
        assert result.page_failures == ()

    def test_a_failed_walk_does_not_claim_to_know_whose_fault_it_was(self) -> None:
        """Garbage bytes are the common case, and the name still refuses to say so.

        `DOCUMENT_UNREADABLE` was the first name for this state and it asserts an
        ownership the `except Exception` cannot establish -- the same clause catches
        MemoryError, RecursionError and a bug in this module.

        Needs the engine, and that is the point rather than a fixture detail: with no
        pypdf these same bytes return ENGINE_ABSENT, because the walk is never reached.
        The state means "the walk ran and failed", so a run that cannot walk cannot
        produce it. Without the guard this asserted a property of the ENVIRONMENT.
        """
        require_pypdf()
        result = extract_fragments(b"not a pdf at all")
        assert result.status is FragmentAvailability.READER_WALK_FAILED

    def test_the_engine_clause_must_be_caught_before_the_general_one(self, monkeypatch) -> None:
        """Branch ORDER, pinned as the contract it is.

        `_EngineMismatch` is an `Exception` subclass -- asserted here because that is
        precisely WHY the order matters. Swapping the two clauses compiles, runs, and
        keeps `available` False, silently refiling "the gate is incomplete" as "this
        document is odd". Nothing else in the suite notices.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        assert issubclass(mod._EngineMismatch, Exception)

        def _mismatch(page, page_number, engine, budget):
            raise mod._EngineMismatch("pypdf TextStateParams is missing a required attribute")

        monkeypatch.setattr(mod, "_page_fragments", _mismatch)
        result = extract_fragments(_one_page_pdf(self._PDF))
        assert result.status is not FragmentAvailability.READER_WALK_FAILED

    def test_the_recorded_version_cannot_tell_the_two_engine_faults_apart(self, monkeypatch) -> None:
        """The discriminator that half-existed, pinned as insufficient.

        A previous session's note held that the malformed path could be told apart by
        `pypdf_version` being set. Both of these carry the pin, and they are different
        faults with different owners -- so anything reading the version as the
        discriminator files an incomplete gate as a bad document.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        def _mismatch(page, page_number, engine, budget):
            raise mod._EngineMismatch("pypdf TextStateParams is missing a required attribute")

        monkeypatch.setattr(mod, "_page_fragments", _mismatch)
        contradicted = extract_fragments(_one_page_pdf(self._PDF))
        monkeypatch.undo()
        walk_failed = extract_fragments(b"not a pdf at all")

        assert contradicted.pypdf_version == walk_failed.pypdf_version != ""
        assert contradicted.status is not walk_failed.status

    def test_every_state_is_reachable_from_a_real_trigger(self, monkeypatch) -> None:
        """A census of the enum against the code that produces it.

        A member nothing can return is a category an operator will wait forever to
        see, and a member that quietly stops being produced is worse. This fails in
        both directions.

        What it does NOT prove, stated because an exhaustiveness check reads like a
        correctness check: that any state is the RIGHT one for its trigger. Three of
        the five triggers are monkeypatches, so this is a reachability claim about the
        enum, not a claim about classification. The classification is pinned one test
        at a time above, which is where a wrong answer actually gets caught.
        """
        require_pypdf()
        import sys

        import carmel.services.pdf_fragments as mod

        def _mismatch(page, page_number, engine, budget):
            raise mod._EngineMismatch("pypdf TextStateParams is missing a required attribute")

        produced = {extract_fragments(_one_page_pdf(self._PDF)).status}
        produced.add(extract_fragments(b"not a pdf at all").status)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "pypdf", None)
            produced.add(extract_fragments(_one_page_pdf(self._PDF)).status)
        with monkeypatch.context() as patch:
            patch.setattr(mod, "_engine", lambda: None)
            produced.add(extract_fragments(_one_page_pdf(self._PDF)).status)
        with monkeypatch.context() as patch:
            patch.setattr(mod, "_page_fragments", _mismatch)
            produced.add(extract_fragments(_one_page_pdf(self._PDF)).status)

        assert produced == set(FragmentAvailability)

    def test_an_installed_pypdf_that_will_not_import_is_an_alarm_not_an_absence(self, monkeypatch) -> None:
        """The original defect, one layer down, caught before it shipped.

        `except Exception` around the import called every failure ENGINE_ABSENT -- a
        SUPPORTED configuration -- including an installed pypdf whose own import
        raises. Three shapes, each measured rather than assumed: a missing transitive
        dependency (ModuleNotFoundError naming the DEP), a crash at import time
        (whatever the package raises), and a package that no longer exports
        `PdfReader` (ImportError, not ModuleNotFoundError).
        """
        import sys

        for exc in (
            ModuleNotFoundError("No module named 'some_dep'", name="some_dep"),
            RuntimeError("boom at import time"),
            ImportError("cannot import name 'PdfReader'", name="pypdf"),
        ):

            class _Blocker:
                def __init__(self, error: BaseException) -> None:
                    self.error = error

                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "pypdf":
                        raise self.error
                    return None

            with monkeypatch.context() as patch:
                patch.delitem(sys.modules, "pypdf", raising=False)
                patch.setattr(sys, "meta_path", [_Blocker(exc), *sys.meta_path])
                result = extract_fragments(_one_page_pdf(self._PDF))
            assert result.status is FragmentAvailability.ENGINE_REFUSED, exc

    def test_available_is_derived_and_cannot_disagree(self) -> None:
        """The reason this is one field and not two."""
        for status in FragmentAvailability:
            unavailable = status is not FragmentAvailability.AVAILABLE
            extraction = FragmentExtraction(
                lossy=unavailable,
                status=status,
                pypdf_version="6.14.2" if status in pdf_fragments._ENGINE_RAN else "",
            )
            assert extraction.available is (status is FragmentAvailability.AVAILABLE)

    def test_a_status_that_is_merely_equal_to_a_member_is_refused(self) -> None:
        """`FragmentAvailability` is a StrEnum, so `"engine_absent"` compares EQUAL to
        the member and is not it. A consumer matching with `is` -- which this module
        tells them to do -- would skip a state whose every log line reads correctly."""
        with pytest.raises(TypeError, match="must be a FragmentAvailability"):
            FragmentExtraction(lossy=True, status="engine_absent")  # type: ignore[arg-type]

    def test_an_unavailable_extraction_may_not_carry_evidence(self) -> None:
        """ "Nothing here can be relied on" has to mean nothing is here.

        The suite itself had grown a fixture handing a fragment to an unavailable
        extraction while the docstring said that could not happen -- prose the type did
        not enforce is a convention, not an invariant.
        """
        fragment = pdf_fragments.TextFragment(
            page=1,
            text="1.0",
            x_start=100.0,
            x_end=110.0,
            baseline_y=500.0,
            font_height=10.0,
            rotated=False,
            glyph_mapping=GlyphMapping.MAPPED,
        )
        for kwargs in (
            {"fragments": (fragment,)},
            {"page_failures": (pdf_fragments.FragmentPageFailure(page=1, error="x"),)},
            {"truncated": True},
        ):
            with pytest.raises(ValueError, match="carries evidence it cannot vouch for"):
                FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_ABSENT, **kwargs)

    def test_an_unavailable_extraction_that_claims_completeness_is_a_construction_error(self) -> None:
        for status in FragmentAvailability:
            if status is FragmentAvailability.AVAILABLE:
                continue
            with pytest.raises(ValueError, match="must admit loss"):
                FragmentExtraction(status=status)

    def test_located_loss_must_set_the_document_flag(self) -> None:
        """`page_failures` and `truncated` are what LOCATE loss, so either of them
        beside `lossy=False` is one object contradicting itself."""
        with pytest.raises(ValueError, match="must set lossy"):
            FragmentExtraction(truncated=True, pypdf_version="6.14.2")

    def test_the_version_is_recorded_exactly_when_the_engine_ran(self) -> None:
        """Enforced in both directions, including for AVAILABLE.

        The cost is that a bare `FragmentExtraction()` no longer constructs, and that
        is correct rather than a casualty: an available extraction that never ran an
        engine is a fiction that was only ever convenient as a fixture.
        """
        with pytest.raises(ValueError, match="must record the pypdf version"):
            FragmentExtraction()
        # PRESENCE is not the invariant. The field means "the PINNED engine ran", and
        # `_engine` refuses every other version before a walk can start, so any other
        # value is a construction error and not a record of something that happened.
        with pytest.raises(ValueError, match="pypdf_version must be"):
            FragmentExtraction(pypdf_version="bogus")
        for status in FragmentAvailability:
            wrong = "6.14.2" if status not in pdf_fragments._ENGINE_RAN else ""
            with pytest.raises(ValueError, match="must record the pypdf version"):
                FragmentExtraction(lossy=True, status=status, pypdf_version=wrong)

    def test_an_available_extraction_may_be_complete(self) -> None:
        """The control: no invariant above may fire on the happy path."""
        assert FragmentExtraction(pypdf_version="6.14.2").available is True


class TestLossRecordsWhatWasLost:
    def test_a_failing_page_is_named_not_merely_counted(self, monkeypatch) -> None:
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        real = mod._page_fragments

        def _boom(page, page_number, engine, budget):
            if page_number == 2:
                raise ValueError("synthetic page explosion")
            return real(page, page_number, engine, budget)

        monkeypatch.setattr(mod, "_page_fragments", _boom)
        pdf = _two_page_pdf(
            b"BT /F1 10 Tf 72 700 Td (alpha) Tj ET",
            b"BT /F1 10 Tf 72 700 Td (beta) Tj ET",
        )
        result = extract_fragments(pdf)
        assert result.available is True
        assert result.lossy is True
        assert [f.page for f in result.page_failures] == [2]
        assert "synthetic page explosion" in result.page_failures[0].error
        # The page that DID parse is still returned -- partial, and honest about it.
        assert any(f.text.strip() == "alpha" for f in result.fragments)

    def test_the_page_cap_matches_the_text_lane(self, monkeypatch) -> None:
        """If the two lanes capped differently, a fragment could carry a page number
        the text lane says does not exist."""
        require_pypdf()
        import carmel.agents.tools.extract as extract_mod

        monkeypatch.setattr(extract_mod, "MAX_PDF_PAGES", 1)
        pdf = _two_page_pdf(
            b"BT /F1 10 Tf 72 700 Td (alpha) Tj ET",
            b"BT /F1 10 Tf 72 700 Td (beta) Tj ET",
        )
        result = extract_fragments(pdf)
        assert result.truncated is True
        assert result.lossy is True
        assert {f.page for f in result.fragments} == {1}

    def test_the_pypdf_version_travels_with_the_result(self) -> None:
        require_pypdf()
        result = extract_fragments(_one_page_pdf("BT /F1 10 Tf\n72 700 Td (x) Tj\nET"))
        assert result.pypdf_version

    def test_an_uninspectable_page_is_named_not_merely_counted(self, monkeypatch) -> None:
        """`lossy` is a whole-document flag and cannot say WHICH page is uncertain.

        A per-page gate that asks "is page 2 sound?" reads `page_failures`, so an
        UNINSPECTABLE page recorded only as `lossy=True` would let that gate pass a
        page the text lane records as uncertain -- two lanes disagreeing about one
        page, which is the divergence `_classify_pdf_page` is shared to prevent.
        """
        require_pypdf()
        import carmel.agents.tools.extract as extract_mod

        real_classify = extract_mod._classify_pdf_page

        def _uninspectable_second(page):
            # Keyed on the page's OWN content, not on a call counter and not on
            # `id(page.pdf)`. This test drives BOTH lanes over the same bytes, so each
            # builds its own `PdfReader` and walks the page tree again: a running
            # counter fires on page 2 of the first walk and on nothing in the second,
            # and an `id()` key is reusable after GC. The content is what actually
            # identifies the page.
            if b"beta" in page.get("/Contents").get_object().get_data():
                return extract_mod._PageKind.UNINSPECTABLE
            return real_classify(page)

        monkeypatch.setattr(extract_mod, "_classify_pdf_page", _uninspectable_second)
        pdf = _two_page_pdf(
            b"BT /F1 10 Tf 72 700 Td (alpha) Tj ET",
            b"BT /F1 10 Tf 72 700 Td (beta) Tj ET",
        )
        result = extract_fragments(pdf)
        assert result.available is True
        assert result.lossy is True
        assert [f.page for f in result.page_failures] == [2]
        # The page is KEPT, never dropped: it may well be a real page.
        assert any(f.text.strip() == "beta" for f in result.fragments)

        # And the two lanes must DESCRIBE that page identically. Compared against what
        # the text lane actually emits for the same document, not against a shared
        # constant: the message is duplicated on purpose (hoisting it would perturb
        # `extract_text`'s pinned semantic-dependency sha, under which stored
        # extractions were produced), so behaviour is the only honest thing to pin it
        # to. If either lane's wording drifts, this fails.
        text_result = extract_mod.extract_text(pdf, "application/pdf")
        assert [f.page for f in text_result.page_failures] == [2]
        assert result.page_failures[0].error == text_result.page_failures[0].error

    def test_a_single_page_cannot_exhaust_the_fragment_budget(self, monkeypatch) -> None:
        """The page cap does not bound this lane: one page may carry unboundedly many
        text-show operations, and each one costs a retained Python object."""
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        monkeypatch.setattr(mod, "MAX_PDF_FRAGMENTS", 3)
        shows = " ".join(f"({i}) Tj" for i in range(50))
        result = extract_fragments(_one_page_pdf(f"BT /F1 10 Tf 72 700 Td {shows} ET"))
        assert result.available is True
        assert len(result.fragments) == 3
        assert result.truncated is True
        assert result.lossy is True

    def test_the_budget_bounds_the_show_list_not_only_the_converted_fragments(self, monkeypatch) -> None:
        """One show becomes at most one fragment, so capping only the conversion loop
        lets a hostile page retain millions of `TextStateParams` before the cap fires.

        Measured by counting how many `TextStateParams` are ever CONSTRUCTED, which is
        the object being retained. The walker checks the budget at that construction
        point, inside `TJ` arrays as well as between operators, so unlike the
        group-at-a-time engine this replaced there is no nesting level at which shows
        accumulate past the cap: 3, not 60, and not "3 plus whatever the group held".
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        built = 0
        real_engine = mod._engine()
        assert real_engine is not None
        resolve_font, params_cls, content_stream = real_engine

        def _counting_params(*args, **kwargs):
            nonlocal built
            built += 1
            return params_cls(*args, **kwargs)

        monkeypatch.setattr(mod, "_engine", lambda: (resolve_font, _counting_params, content_stream))
        monkeypatch.setattr(mod, "MAX_PDF_FRAGMENTS", 3)
        groups = " ".join(f"BT /F1 10 Tf 72 {700 - i} Td ({i}) Tj ET" for i in range(60))
        result = extract_fragments(_one_page_pdf(groups))
        assert len(result.fragments) == 3
        assert result.truncated is True
        assert built == 3, f"built {built} shows for a budget of 3"

    def test_a_page_past_the_cap_is_never_parsed_at_all(self, monkeypatch) -> None:
        """The cap is checked BEFORE parsing the next page, not after.

        This guard has NO behavioural signature -- entering with a zero budget produces
        the identical `truncated=True` result, which is why deleting it leaves every
        other test green (verified by mutation). What it actually buys is that the next
        page's content stream is never materialised, so the only honest way to test it
        is to observe the work not being done.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        real = mod._page_fragments
        parsed: list[int] = []

        def _recording(page, page_number, engine, budget):
            parsed.append(page_number)
            return real(page, page_number, engine, budget)

        monkeypatch.setattr(mod, "_page_fragments", _recording)
        monkeypatch.setattr(mod, "MAX_PDF_FRAGMENTS", 1)
        page = b"BT /F1 10 Tf 72 700 Td (a) Tj ET"
        result = extract_fragments(_two_page_pdf(page, page))
        assert result.truncated is True
        assert parsed == [1], f"page 2 was parsed despite the cap already being met: {parsed}"

    def test_the_fragment_budget_spans_pages_rather_than_resetting(self, monkeypatch) -> None:
        """A per-page budget would let an N-page document retain N times the cap."""
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        monkeypatch.setattr(mod, "MAX_PDF_FRAGMENTS", 3)
        page = b"BT /F1 10 Tf 72 700 Td " + b" ".join(f"({i}) Tj".encode() for i in range(2)) + b" ET"
        result = extract_fragments(_two_page_pdf(page, page))
        assert len(result.fragments) == 3
        assert result.truncated is True


def _split_contents_pdf(first: bytes, second: bytes) -> bytes:
    """One page whose `/Contents` is an ARRAY of two streams that concatenate.

    The shape a per-stream cap would miss: the parser sees their concatenation, so only
    their SUM bounds what it allocates.
    """
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 6 0 R >> >> /Contents [4 0 R 5 0 R] >>",
            b"<< /Length " + str(len(first)).encode() + b" >>\nstream\n" + first + b"\nendstream",
            b"<< /Length " + str(len(second)).encode() + b" >>\nstream\n" + second + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


class TestOnePageCannotCostUnboundedMemory:
    """`MAX_PDF_FRAGMENTS` counts fragments, and both expensive things happen before the
    first fragment exists. Capping the page's decompressed content stream is what bounds
    them -- including the one huge `BT` group, which on real papers is up to 99.5% of a
    page and so cannot be interrupted between groups."""

    def test_an_oversized_page_fails_that_page_and_not_the_document(self, monkeypatch) -> None:
        require_pypdf()
        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", 40)
        small = b"BT /F1 9 Tf 1 0 0 1 72 700 Tm (ok) Tj ET"
        big = b"BT /F1 9 Tf 1 0 0 1 72 700 Tm (" + b"x" * 200 + b") Tj ET"
        result = extract_fragments(_two_page_pdf(small, big))

        assert result.available is True
        assert result.lossy is True
        assert [failure.page for failure in result.page_failures] == [2]
        assert "PageContentTooLarge" in result.page_failures[0].error
        assert [fragment.text for fragment in result.fragments] == ["ok"]

    def test_the_recorded_error_states_the_size_and_leaks_no_path(self, monkeypatch) -> None:
        require_pypdf()
        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", 10)
        result = extract_fragments(_one_page_pdf("BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET"))

        error = result.page_failures[0].error
        assert error.startswith("PageContentTooLarge: ")
        assert "10-byte cap" in error
        assert "/home/" not in error

    def test_the_cap_is_checked_before_the_stream_is_parsed(self, monkeypatch) -> None:
        """The whole point. A cap applied after `ContentStream` has built its operation
        list measures the damage instead of preventing it."""
        require_pypdf()
        import pypdf.generic

        def _must_not_run(*args: object, **kwargs: object) -> None:
            raise AssertionError("the content stream was parsed despite exceeding the cap")

        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", 10)
        monkeypatch.setattr(pypdf.generic, "ContentStream", _must_not_run)
        result = extract_fragments(_one_page_pdf("BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET"))

        assert "PageContentTooLarge" in result.page_failures[0].error

    def test_a_page_split_across_streams_is_bounded_by_their_sum(self, monkeypatch) -> None:
        """Otherwise the cap is evaded by splitting one huge stream into many small
        ones, which the parser concatenates back together anyway."""
        require_pypdf()
        first = b"BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET"
        second = b"BT /F1 9 Tf 1 0 0 1 72 680 Tm (B) Tj ET"

        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", len(first) + len(second) + 8)
        allowed = extract_fragments(_split_contents_pdf(first, second))
        assert allowed.page_failures == ()
        assert sorted(fragment.text for fragment in allowed.fragments) == ["A", "B"]

        # Each part alone fits; only their sum does not.
        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", len(first) + 4)
        refused = extract_fragments(_split_contents_pdf(first, second))
        assert "PageContentTooLarge" in refused.page_failures[0].error

    def test_a_page_exactly_on_the_cap_is_kept(self, monkeypatch) -> None:
        """`>` and not `>=`: a cap that refuses the value it names would make every
        headroom figure in its docstring off by one.

        Driven through the REAL measurement rather than a stub returning 100. The boundary
        now lives inside `_decoded_content_length`, so stubbing that function out would
        assert on the caller's arithmetic -- which this change deleted, precisely because
        measure-then-compare is the shape that let the decode run unbounded.
        """
        require_pypdf()
        body = "BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET"
        exact = len(body.encode("latin-1"))

        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", exact)
        assert extract_fragments(_one_page_pdf(body)).page_failures == ()

        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", exact - 1)
        refused = extract_fragments(_one_page_pdf(body))
        assert "PageContentTooLarge" in refused.page_failures[0].error

    def test_a_flate_page_under_the_cap_extracts_normally(self) -> None:
        """The allowed filter, end to end. Every other fixture in this file is unfiltered,
        so without this the bounded-decode branch ships exercised by nothing."""
        require_pypdf()
        result = extract_fragments(
            _one_page_filtered_pdf("BT /F1 9 Tf 1 0 0 1 72 700 Tm (ok) Tj ET", filters="/FlateDecode")
        )
        assert result.page_failures == ()
        assert [fragment.text for fragment in result.fragments] == ["ok"]

    def test_a_compression_bomb_is_refused_without_being_allocated(self, monkeypatch) -> None:
        """The defect this whole change exists for, and the only test that can show it.

        Refusing the page was never the hard part -- the OLD code refused it too, after
        calling `get_data()` and materialising every byte. What has to be demonstrated is
        that the refusal now happens WITHOUT that allocation, so the assertion is on peak
        heap during the call and not on the exception.

        64 MB of zeros compress to about 64 KB. The cap is set to 1 KB, and peak allocation
        is required to stay under 4 MB: comfortably above the compressed input and the
        decode ceiling, and two orders of magnitude below what the old path would have
        allocated. A generous bound rather than a tight one, because the number that
        matters is the ORDER, and a tight one would fail on an unrelated allocation.
        """
        require_pypdf()
        bomb = zlib.compress(b"\0" * (64 * 1024 * 1024))
        assert len(bomb) < 200_000, "the fixture must be small compressed, or it proves nothing"
        monkeypatch.setattr(pdf_fragments, "MAX_PAGE_CONTENT_BYTES", 1024)

        tracemalloc.start()
        try:
            result = extract_fragments(_one_page_filtered_pdf("", filters="/FlateDecode", raw=bomb))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert "PageContentTooLarge" in result.page_failures[0].error
        assert peak < 4 * 1024 * 1024, f"peak allocation was {peak} bytes; the decode was not bounded"

    def test_an_unsupported_filter_fails_the_page_closed(self) -> None:
        """Fail CLOSED, and as its own reason. A guard that bounded Flate and fell through
        to `get_data()` for everything else would read as a bound while being none."""
        require_pypdf()
        result = extract_fragments(
            _one_page_filtered_pdf("BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET", filters="/LZWDecode")
        )
        assert "PageContentUndecodable" in result.page_failures[0].error
        assert result.fragments == ()

    def test_a_filter_chain_containing_flate_is_still_refused(self) -> None:
        """The allowlist compares the whole chain, not membership. `[/ASCII85Decode
        /FlateDecode]` contains the allowed filter and is still undecodable under a bound,
        because the outer stage has already allocated by the time the inner one runs."""
        require_pypdf()
        result = extract_fragments(
            _one_page_filtered_pdf("BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET", filters="[/ASCII85Decode /FlateDecode]")
        )
        assert "PageContentUndecodable" in result.page_failures[0].error

    def test_decode_parms_are_refused_even_with_the_allowed_filter(self) -> None:
        """A predictor changes what a byte bound bounds: the decompressed stream is not
        the decoded content, so a size established before the predictor runs is not the
        size that gets parsed."""
        require_pypdf()
        result = extract_fragments(
            _one_page_filtered_pdf(
                "BT /F1 9 Tf 1 0 0 1 72 700 Tm (A) Tj ET",
                filters="/FlateDecode",
                parms="<< /Predictor 12 /Columns 4 >>",
            )
        )
        assert "PageContentUndecodable" in result.page_failures[0].error

    def test_a_corrupt_flate_stream_fails_the_page_rather_than_the_document(self) -> None:
        """The residual this change accepts, pinned so it stays visible: pypdf's own
        FlateDecode carries recovery machinery for damaged streams and bare zlib does not,
        so a stream pypdf could salvage becomes a recorded page failure here. Per page,
        named, and never a silent difference in the fragments."""
        require_pypdf()
        result = extract_fragments(
            _one_page_filtered_pdf("", filters="/FlateDecode", raw=b"not a deflate stream at all")
        )
        assert "PageContentUndecodable" in result.page_failures[0].error
        assert result.available is True
        assert result.lossy is True

    def test_a_truncated_flate_stream_is_refused_rather_than_measured_short(self) -> None:
        """A valid deflate PREFIX inflates cleanly and reports no error at all.

        This is the silent direction, which is why it gets its own test rather than riding
        along with the corrupt-stream one above. Measured on the exact fixture below:
        dropping the last 20 compressed bytes still yields 3,654 of the 8,200 real bytes,
        with ``unconsumed_tail`` EMPTY -- empty precisely because every compressed byte was
        consumed -- and no ``zlib.error`` raised. A guard checking only that tail therefore
        accepts the prefix, returns 3,654 as the page's size, and the page is parsed as
        though it were whole: fewer operations, no failure recorded, a short page that looks
        complete. ``eof`` is the only witness that separates "consumed all the input" from
        "reached the end of the stream".
        """
        require_pypdf()
        body = ("BT /F1 9 Tf 1 0 0 1 72 700 Tm (ok) Tj ET " * 200).encode("latin-1")
        truncated = zlib.compress(body)[:-20]

        engine = zlib.decompressobj()
        salvaged = engine.decompress(truncated, 10_000_000)
        assert not engine.eof and engine.unconsumed_tail == b""
        assert 0 < len(salvaged) < len(body), "fixture must inflate a SHORT prefix silently"

        result = extract_fragments(_one_page_filtered_pdf("", filters="/FlateDecode", raw=truncated))
        assert "PageContentUndecodable" in result.page_failures[0].error
        assert "truncated prefix" in result.page_failures[0].error
        assert result.lossy is True

    def test_bytes_after_the_deflate_end_are_refused_rather_than_ignored(self) -> None:
        """Trailing bytes inflate to ``eof=True`` with the extras parked on ``unused_data``.

        This function never measured them and ``ContentStream`` may still parse them, so
        admitting the stream returns a number that bounds less than the caller believes.
        """
        require_pypdf()
        body = b"BT /F1 9 Tf 1 0 0 1 72 700 Tm (ok) Tj ET"
        result = extract_fragments(
            _one_page_filtered_pdf("", filters="/FlateDecode", raw=zlib.compress(body) + b"TRAILING GARBAGE")
        )
        assert "PageContentUndecodable" in result.page_failures[0].error
        assert "past the end of its deflate data" in result.page_failures[0].error

    def test_a_trailing_newline_after_the_deflate_end_is_admitted(self) -> None:
        """The counterweight to the test above, and it exists because the guard without it
        was measured to be WRONG on real documents.

        PDF's stream syntax puts an EOL before ``endstream``, and ``/Length`` need not cover
        it. 43 of the 8-paper corpus's 161 content streams carry exactly one trailing
        ``b"\\n"``. Refusing on ``unused_data`` alone therefore failed 43 real pages across 4
        of 8 papers and took the corpus from 78,178 fragments to 34,151 -- a guard costing
        56% of the evidence to catch a byte the format requires.

        Pinned in the admitting direction on purpose: the refusal test above passes whether
        or not whitespace is exempt, so it alone would not notice the exemption being
        "tidied away" by someone reading only the stricter rule.
        """
        require_pypdf()
        body = b"BT /F1 9 Tf 1 0 0 1 72 700 Tm (ok) Tj ET"
        result = extract_fragments(_one_page_filtered_pdf("", filters="/FlateDecode", raw=zlib.compress(body) + b"\n"))
        assert result.page_failures == ()
        assert [fragment.text for fragment in result.fragments] == ["ok"]

    def test_the_two_page_failure_reasons_are_never_conflated(self) -> None:
        """Too big and undecodable are different claims: one is about the document, the
        other about this module's reach. Reporting an unhandled filter as an oversized
        page would blame the PDF for a limit that lives here."""
        assert pdf_fragments.PageContentTooLarge is not pdf_fragments.PageContentUndecodable
        assert not issubclass(pdf_fragments.PageContentUndecodable, pdf_fragments.PageContentTooLarge)
        assert not issubclass(pdf_fragments.PageContentTooLarge, pdf_fragments.PageContentUndecodable)

    def test_the_shipped_cap_admits_the_largest_real_corpus_page(self) -> None:
        """836,591 B is the largest decompressed page in the 8-paper corpus (median
        22,035 B). Pinned so that lowering the cap has to face the measurement."""
        assert MAX_PAGE_CONTENT_BYTES >= 836_591 * 7
