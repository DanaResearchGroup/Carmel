"""Tests for :mod:`carmel.services.pdf_fragments`.

Every fixture here is SYNTHETIC, built from raw PDF bytes in-process. Corpus paper
text is copyrighted and non-redistributable, so no real document may be checked in --
which is also why the coordinates asserted below are EXACT rather than eyeballed:
the content stream states where each glyph is placed, so ground truth is known.
"""

from __future__ import annotations

import pytest

from carmel.services.pdf_fragments import (
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

    def test_capability_check_rejects_a_reshaped_state_manager(self, monkeypatch) -> None:
        require_pypdf()
        from pypdf._text_extraction._layout_mode._text_state_manager import TextStateManager

        import carmel.services.pdf_fragments as mod

        monkeypatch.delattr(TextStateManager, "set_font", raising=True)
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
        assert result == FragmentExtraction(fragments=(), lossy=True, available=False)

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

        Measured through pypdf's own call count: the engine consumes one whole BT group
        per call, so a bounded run must stop calling it once the budget is met. This is
        also the honest granularity of the guard -- a SINGLE group holding a million
        shows is still unbounded, which the docstring says outright.
        """
        require_pypdf()
        import carmel.services.pdf_fragments as mod

        calls = 0
        real_engine = mod._engine()
        assert real_engine is not None
        recurse, resolve_font, manager, content_stream = real_engine

        def _counting_recurse(*args, **kwargs):
            nonlocal calls
            calls += 1
            return recurse(*args, **kwargs)

        monkeypatch.setattr(mod, "_engine", lambda: (_counting_recurse, resolve_font, manager, content_stream))
        monkeypatch.setattr(mod, "MAX_PDF_FRAGMENTS", 3)
        groups = " ".join(f"BT /F1 10 Tf 72 {700 - i} Td ({i}) Tj ET" for i in range(60))
        result = extract_fragments(_one_page_pdf(groups))
        assert len(result.fragments) == 3
        assert result.truncated is True
        # 4, not 60: the loop stops one group after the budget is met.
        assert calls <= 4, f"walked {calls} text groups for a budget of 3"

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
