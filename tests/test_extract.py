"""Tests for carmel.agents.tools.extract and the artifacts binary helpers."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from carmel.agents.tools.extract import (
    MAX_EXTRACTED_TEXT_CHARS,
    MAX_PDF_PAGES,
    ExtractedText,
    TextSection,
    extract_text,
    normalize_for_match,
    normalize_with_map,
    raw_span,
)
from carmel.services.artifacts import read_bytes, write_bytes


class TestNormalizeForMatch:
    """Tests for normalize_for_match."""

    def test_ligature_expansion(self) -> None:
        assert normalize_for_match("eﬃcient") == "efficient"
        assert normalize_for_match("ﬁre") == "fire"
        assert normalize_for_match("ﬂame") == "flame"
        assert normalize_for_match("stiﬀ") == "stiff"
        assert normalize_for_match("waﬄe") == "waffle"

    def test_hyphen_join_across_linebreak(self) -> None:
        assert normalize_for_match("combus-\ntion") == "combustion"

    def test_hyphen_join_with_surrounding_whitespace(self) -> None:
        assert normalize_for_match("combus- \n  tion") == "combustion"

    def test_real_hyphen_not_joined_without_linebreak(self) -> None:
        assert normalize_for_match("well-known") == "well-known"

    def test_whitespace_collapse(self) -> None:
        assert normalize_for_match("a   b\t\tc\n\nd") == "a b c d"

    def test_casefold_and_strip(self) -> None:
        assert normalize_for_match("  HELLO World  ") == "hello world"

    def test_nfkc_normalization(self) -> None:
        # Fullwidth 'A' (U+FF21) NFKC-normalizes to ASCII 'A', then casefolds.
        assert normalize_for_match("ＡＢＣ") == "abc"

    def test_order_of_operations(self) -> None:
        # De-hyphenation must run before whitespace collapse, or the run-together
        # newline can no longer be distinguished as the split point.
        text = "combus-\ntion   of ﬁre"
        assert normalize_for_match(text) == "combustion of fire"


_TRICKY_INPUTS = [
    "ﬁle",
    "eﬃcient",
    "combus-\ntion",
    "combus- \n  tion",
    "a\xa0b",
    "a\tb\tc",
    "  HELLO World  ",
    "éclair",  # 'e' + combining acute accent.
    "",
]


class TestNormalizeWithMap:
    """Tests for normalize_with_map and raw_span."""

    @pytest.mark.parametrize("s", _TRICKY_INPUTS)
    def test_matches_normalize_for_match(self, s: str) -> None:
        assert normalize_for_match(s) == normalize_with_map(s)[0]

    @pytest.mark.parametrize("s", _TRICKY_INPUTS)
    def test_index_map_length_and_bounds(self, s: str) -> None:
        normalized, index_map = normalize_with_map(s)
        assert len(index_map) == len(normalized)
        for idx in index_map:
            assert 0 <= idx < len(s) if s else True

    def test_round_trip_recovers_matching_raw_slice(self) -> None:
        raw = "The combus-\ntion   of ﬁre was studied."
        normalized, index_map = normalize_with_map(raw)
        needle = "combustion of fire"
        start = normalized.index(needle)
        end = start + len(needle)
        raw_start, raw_end = raw_span(index_map, start, end, len(raw))
        raw_slice = raw[raw_start:raw_end]
        assert normalize_for_match(raw_slice) == needle

    def test_multi_char_nfkc_expansion_maps_to_single_source_index(self) -> None:
        # U+00BD VULGAR FRACTION ONE HALF NFKC-expands to the 3-character
        # string "1⁄2" ("1", FRACTION SLASH, "2"). Every one of those three
        # output characters must map back to the single source index.
        raw = "a½b"
        normalized, index_map = normalize_with_map(raw)
        assert normalized == "a1⁄2b"
        half_index = raw.index("½")
        # Output positions for the expansion are index_map entries 1, 2, 3
        # (position 0 is 'a', positions 1-3 are the expansion, position 4 is 'b').
        expansion_map_entries = index_map[1:4]
        assert expansion_map_entries == [half_index, half_index, half_index]

    def test_raw_span_empty_span(self) -> None:
        _, index_map = normalize_with_map("hello world")
        assert raw_span(index_map, 3, 3, len("hello world")) == (3, 3)

    def test_raw_span_end_of_map(self) -> None:
        raw = "hello world"
        normalized, index_map = normalize_with_map(raw)
        raw_start, raw_end = raw_span(index_map, 0, len(normalized), len(raw))
        assert raw_end == len(raw)
        assert raw[raw_start:raw_end] == raw


class TestExtractPlainText:
    """Tests for extract_text on text/* content."""

    def test_plain_text_verbatim(self) -> None:
        data = b"Hello, world!\nSecond line."
        result = extract_text(data, "text/plain")
        assert isinstance(result, ExtractedText)
        assert result.text == "Hello, world!\nSecond line."
        assert result.extractor == "text"
        assert result.lossy is False
        assert result.normalized == normalize_for_match(result.text)
        assert result.sections == [TextSection(label="body", start=0, end=len(result.text))]

    def test_unknown_content_type(self) -> None:
        result = extract_text(b"whatever", "application/octet-stream")
        assert result.text == ""
        assert result.normalized == ""
        assert result.sections == []
        assert result.extractor == "unknown"
        assert result.lossy is True


class TestExtractHtml:
    """Tests for extract_text on text/html content."""

    def test_script_and_style_stripped(self) -> None:
        html = (
            b"<html><head><style>body { color: red; }</style></head>"
            b"<body><script>alert('hi');</script>"
            b"<p>Visible paragraph.</p></body></html>"
        )
        result = extract_text(html, "text/html")
        assert "color" not in result.text
        assert "alert" not in result.text
        assert "Visible paragraph." in result.text
        assert result.extractor == "html"
        assert result.lossy is False

    def test_html_verbatim_text_content_preserved(self) -> None:
        html = b"<div>Alpha</div><div>Beta</div>"
        result = extract_text(html, "text/html")
        assert "Alpha" in result.text
        assert "Beta" in result.text


class TestDecodeBytes:
    """Tests for the internal UTF-8 decode fallback used by HTML/text extraction."""

    def test_invalid_utf8_is_replaced_not_raised(self) -> None:
        data = b"Valid start \xff\xfe invalid bytes end"
        result = extract_text(data, "text/plain")
        assert result.text.startswith("Valid start")
        assert result.lossy is False

    def test_invalid_utf8_in_html(self) -> None:
        data = b"<p>Broken \xff bytes</p>"
        result = extract_text(data, "text/html")
        assert "Broken" in result.text


class TestExtractPdf:
    """Tests for extract_text on application/pdf content, real or simulated-absent."""

    def test_pdf_extraction_via_fake_pypdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Deterministically exercise the "pypdf is installed" success path regardless
        # of whether the real optional dependency happens to be installed here.
        class _FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                self.pages = [_FakePage("Page one content."), _FakePage("Page two content.")]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        result = extract_text(b"%PDF-1.4 irrelevant", "application/pdf")
        assert result.extractor == "pdf:pypdf"
        assert result.lossy is False
        assert result.page_count == 2
        assert "Page one content." in result.text
        assert "Page two content." in result.text
        pages_seen = {sec.page for sec in result.sections if sec.page is not None}
        assert pages_seen == {1, 2}

    def test_pdf_parse_error_via_fake_pypdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Deterministically exercise the "pypdf installed but parsing raises" fallback
        # branch regardless of whether the real optional dependency is installed here.
        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                raise ValueError("not a real pdf")

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        result = extract_text(b"not actually a pdf", "application/pdf")
        assert result.extractor == "pdf:pypdf"
        assert result.lossy is True
        assert result.text == ""

    def test_pdf_huge_page_count_stops_before_extracting_every_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # P1-10 regression: a PDF whose page TREE claims far more pages than
        # MAX_PDF_PAGES must never have extract_text() called on every one of them --
        # that is exactly the "cheap page, huge count" decompression-bomb shape. The
        # fake reader reports a page count 10x the cap up front (mimicking pypdf's
        # cheap `len(reader.pages)`, which reads the page tree, not page content) but
        # would raise if extract_text() were ever called past the cap, so this test
        # fails loudly if the cap is not enforced BEFORE materializing pages.
        call_count = 0

        class _FakePage:
            def extract_text(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count > MAX_PDF_PAGES:
                    raise AssertionError("extract_text() called past MAX_PDF_PAGES")
                return "x"

        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                self.pages = [_FakePage() for _ in range(MAX_PDF_PAGES * 10)]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        result = extract_text(b"%PDF-1.4 irrelevant", "application/pdf")
        assert result.extractor == "pdf:pypdf"
        # Load-bearing: the grounding gate fails closed on lossy=True, so a
        # capped/partial extraction MUST be flagged, never silently returned as if
        # it were the whole document.
        assert result.lossy is True
        # The reported page count is the document's TRUE total (not the truncated
        # processing count), so downstream per-page-density calculations
        # (carmel/services/grounding.py) stay meaningful even when extraction stops
        # early.
        assert result.page_count == MAX_PDF_PAGES * 10
        assert call_count <= MAX_PDF_PAGES

    def test_pdf_huge_per_page_text_stops_before_materializing_every_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # P1-10 regression: a small NUMBER of pages, each individually huge, must
        # also stop early once the running character count reaches the cap -- the
        # other half of the decompression-bomb shape (a Flate-compressed stream that
        # expands 100-1000x per page). The fake page's extract_text() returns text
        # far larger than the cap; the loop must call it only until the cap is
        # crossed, never on every one of the (few) remaining pages.
        call_count = 0
        huge_chunk = "y" * (MAX_EXTRACTED_TEXT_CHARS // 2 + 1)

        class _FakePage:
            def extract_text(self) -> str:
                nonlocal call_count
                call_count += 1
                return huge_chunk

        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                # Only a handful of pages -- well under MAX_PDF_PAGES -- so this
                # exercises the character cap specifically, not the page-count cap.
                self.pages = [_FakePage() for _ in range(10)]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        result = extract_text(b"%PDF-1.4 irrelevant", "application/pdf")
        assert result.extractor == "pdf:pypdf"
        assert result.lossy is True
        assert result.page_count == 10
        # Two ~250,001-char chunks already exceed MAX_EXTRACTED_TEXT_CHARS, so the
        # loop must have stopped well short of materializing all 10 pages.
        assert call_count < 10
        assert len(result.text) <= MAX_EXTRACTED_TEXT_CHARS

    def test_pdf_extraction_or_graceful_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Build a minimal, syntactically valid one-page PDF containing the text "Hello PDF".
        pdf_bytes = _build_tiny_pdf(b"Hello PDF")
        result = extract_text(pdf_bytes, "application/pdf")
        assert isinstance(result, ExtractedText)
        try:
            import pypdf  # noqa: F401

            pypdf_installed = True
        except ImportError:
            pypdf_installed = False

        if pypdf_installed:
            assert result.extractor == "pdf:pypdf"
            assert result.lossy is False
            assert result.page_count == 1
            assert any(sec.page == 1 for sec in result.sections)
            # The original version of this test stopped at the assertions above, which
            # a passthrough/no-op "extraction" (e.g. one that always returns an empty
            # string with the right metadata) would also satisfy. Assert the actual
            # extracted content to prove real text extraction happened, not just that
            # the right shape of result was returned.
            assert "Hello PDF" in result.text
        else:
            assert result.extractor == "pdf:unavailable"
            assert result.lossy is True
            assert result.text == ""

    def test_pdf_unavailable_via_monkeypatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the optional dependency being absent regardless of install state.
        monkeypatch.setitem(__import__("sys").modules, "pypdf", None)
        result = extract_text(b"%PDF-1.4 fake", "application/pdf")
        assert result.extractor == "pdf:unavailable"
        assert result.lossy is True
        assert result.text == ""
        assert result.normalized == ""
        assert result.sections == []

    def test_pdf_parse_error_is_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pypdf")
        result = extract_text(b"not actually a pdf", "application/pdf")
        assert result.extractor == "pdf:pypdf"
        assert result.lossy is True
        assert result.text == ""


class TestReferencesSection:
    """A trailing references/bibliography region must be labeled 'references'."""

    def test_references_section_detected(self) -> None:
        body = "Introduction text about combustion chemistry. " * 20
        refs = "Smith, J. (2020). A paper about kinetics. Journal of Combustion."
        text = body + "\n\nReferences\n\n" + refs
        data = text.encode("utf-8")
        result = extract_text(data, "text/plain")

        ref_sections = [s for s in result.sections if s.label == "references"]
        assert len(ref_sections) == 1
        section = ref_sections[0]
        # The references section must cover the reference text itself.
        ref_offset = result.text.index(refs)
        assert section.start <= ref_offset
        assert section.end == len(result.text)
        assert "Smith" in result.text[section.start : section.end]

    def test_early_references_heading_not_treated_as_trailing(self) -> None:
        # A "References" heading very early in a short document should not swallow
        # the whole thing when it doesn't plausibly sit near the end.
        text = "References\n\n" + ("More discussion follows here. " * 50)
        data = text.encode("utf-8")
        result = extract_text(data, "text/plain")
        assert all(s.label == "body" for s in result.sections)

    def test_bibliography_heading_variant(self) -> None:
        body = "Some body content here that is reasonably long. " * 10
        text = body + "\n\nBibliography\n\nDoe, A. Some Title. 2019."
        result = extract_text(text.encode("utf-8"), "text/plain")
        assert any(s.label == "references" for s in result.sections)

    def test_numbered_references_heading_detected(self) -> None:
        """A leading section number ("8. References") must still be recognized."""
        body = "Some body content here that is reasonably long. " * 10
        text = body + "\n\n8. References\n\nDoe, A. Some Title. 2019."
        result = extract_text(text.encode("utf-8"), "text/plain")
        assert any(s.label == "references" for s in result.sections)

    def test_roman_numeral_references_heading_detected(self) -> None:
        """A roman-numeral leading section number ("VI REFERENCES") is recognized."""
        body = "Some body content here that is reasonably long. " * 10
        text = body + "\n\nVI REFERENCES\n\nDoe, A. Some Title. 2019."
        result = extract_text(text.encode("utf-8"), "text/plain")
        assert any(s.label == "references" for s in result.sections)

    def test_no_references_heading_all_body(self) -> None:
        text = "Just a plain document with no special sections at all."
        result = extract_text(text.encode("utf-8"), "text/plain")
        assert result.sections == [TextSection(label="body", start=0, end=len(text))]


class TestAbstractSection:
    """An 'abstract' section is labeled when detectable near the start."""

    def test_abstract_detected(self) -> None:
        text = "Abstract\n\nThis paper studies combustion kinetics in detail.\n\nIntroduction body text follows."
        result = extract_text(text.encode("utf-8"), "text/plain")
        abstract_sections = [s for s in result.sections if s.label == "abstract"]
        assert len(abstract_sections) == 1
        span = result.text[abstract_sections[0].start : abstract_sections[0].end]
        assert "This paper studies combustion kinetics" in span

    def test_abstract_heading_with_nothing_following_is_ignored(self) -> None:
        # The heading is the very last thing in the document (only trailing
        # whitespace after it), so there is no room for a non-empty abstract span.
        text = "Some short lead-in.\n\nAbstract\n\n"
        result = extract_text(text.encode("utf-8"), "text/plain")
        assert all(s.label == "body" for s in result.sections)

    def test_abstract_and_references_together(self) -> None:
        # Exercises overlaying two separate regions (abstract near the start,
        # references at the end) onto the same initial body section, including the
        # passthrough branches for already-relabeled and non-overlapping pieces.
        body = "Introduction and discussion of combustion chemistry. " * 20
        text = (
            "Abstract\n\nThis paper studies combustion kinetics in detail.\n\n"
            + body
            + "\n\nReferences\n\nSmith, J. (2020). A paper about kinetics."
        )
        result = extract_text(text.encode("utf-8"), "text/plain")
        labels = {s.label for s in result.sections}
        assert "abstract" in labels
        assert "references" in labels
        assert "body" in labels
        # Sections must be contiguous and non-overlapping, covering the whole text.
        ordered = sorted(result.sections, key=lambda s: s.start)
        assert ordered[0].start == 0
        assert ordered[-1].end == len(result.text)
        for prev, nxt in zip(ordered, ordered[1:], strict=False):
            assert prev.end == nxt.start


class TestExtractedTextCap:
    """Tests for the MAX_EXTRACTED_TEXT_CHARS truncation applied by every extractor
    path (Finding 14: unbounded memory/CPU on attacker-influenced document size)."""

    def test_over_cap_plain_text_is_truncated_and_marked_lossy(self) -> None:
        oversized = "a" * (MAX_EXTRACTED_TEXT_CHARS + 1000)
        result = extract_text(oversized.encode("utf-8"), "text/plain")
        assert len(result.text) == MAX_EXTRACTED_TEXT_CHARS
        assert result.lossy is True

    def test_over_cap_html_is_truncated_and_marked_lossy(self) -> None:
        oversized = "<p>" + "b" * (MAX_EXTRACTED_TEXT_CHARS + 1000) + "</p>"
        result = extract_text(oversized.encode("utf-8"), "text/html")
        assert len(result.text) == MAX_EXTRACTED_TEXT_CHARS
        assert result.lossy is True

    def test_over_cap_pdf_is_truncated_and_marked_lossy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                # Two pages, each larger than the cap on their own, so the
                # concatenated document text is well over the cap.
                page_text = "c" * (MAX_EXTRACTED_TEXT_CHARS // 2 + 1000)
                self.pages = [_FakePage(page_text), _FakePage(page_text)]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        result = extract_text(b"%PDF-1.4 irrelevant", "application/pdf")
        assert len(result.text) == MAX_EXTRACTED_TEXT_CHARS
        assert result.lossy is True

    def test_sections_never_exceed_truncated_text_length(self) -> None:
        oversized = "Abstract\n\n" + "a" * (MAX_EXTRACTED_TEXT_CHARS + 1000)
        result = extract_text(oversized.encode("utf-8"), "text/plain")
        assert len(result.text) == MAX_EXTRACTED_TEXT_CHARS
        for section in result.sections:
            assert section.start <= len(result.text)
            assert section.end <= len(result.text)

    def test_normal_size_document_is_unaffected(self) -> None:
        normal = "This is an ordinary short document. " * 50
        assert len(normal) < MAX_EXTRACTED_TEXT_CHARS
        result = extract_text(normal.encode("utf-8"), "text/plain")
        assert result.text == normal
        assert result.lossy is False


class TestWriteReadBytes:
    """Tests for the artifacts write_bytes/read_bytes binary helpers."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "raw.bin"
        payload = b"\x00\x01binary\xffdata"
        write_bytes(path, payload)
        assert read_bytes(path) == payload

    def test_temp_file_does_not_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.bin"
        write_bytes(path, b"hello")
        tmp_candidate = path.with_suffix(path.suffix + ".tmp")
        assert not tmp_candidate.exists()
        assert path.exists()

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_bytes(tmp_path / "does_not_exist.bin")


def _build_tiny_pdf(text: bytes) -> bytes:
    """Build a minimal, syntactically valid single-page PDF containing ``text``.

    Hand-built rather than generated by a PDF library, so this test has no dependency
    on pypdf being installed to CREATE the fixture (only to parse it).
    """
    content_stream = f"BT /F1 24 Tf 72 700 Td ({text.decode('utf-8')}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content_stream)).encode() + b" >>\nstream\n" + content_stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


class TestPypdfNoiseIsMuted:
    """A malformed-but-recoverable PDF must not bury the operator's verdict in repair logs.

    Real publisher PDFs make pypdf emit one WARNING per fixed-up cross-reference entry;
    a single 9-page paper produced ~70 lines of "Ignoring wrong pointing object ...".
    The CLI installs no logging configuration, so those land on the root logger's
    last-resort stderr handler, on top of the one ACCEPTED/REJECTED line being read.
    """

    def test_the_pypdf_logger_is_muted_during_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.agents.tools import extract as extract_mod

        observed: list[int] = []

        class _SpyReader:
            def __init__(self, _stream: object) -> None:
                observed.append(logging.getLogger("pypdf").getEffectiveLevel())
                self.pages: list[object] = []

        monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=_SpyReader))
        extract_mod._extract_pdf(b"%PDF-1.4 whatever")

        assert observed == [logging.ERROR]

    def test_the_previous_level_is_restored_afterwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Carmel is importable as a library; muting a third party's logger permanently
        would be reconfiguring logging behind the embedding application's back."""
        from carmel.agents.tools import extract as extract_mod

        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.DEBUG)
        try:
            monkeypatch.setitem(
                sys.modules,
                "pypdf",
                types.SimpleNamespace(PdfReader=lambda _stream: types.SimpleNamespace(pages=[])),
            )
            extract_mod._extract_pdf(b"%PDF-1.4 whatever")

            assert pypdf_logger.level == logging.DEBUG
        finally:
            pypdf_logger.setLevel(logging.NOTSET)

    def test_the_level_is_restored_even_when_the_pdf_fails_to_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The failure path is the one that matters: a parse error is exactly when a
        naive implementation leaks the muted level and hides every later warning."""
        from carmel.agents.tools import extract as extract_mod

        pypdf_logger = logging.getLogger("pypdf")
        pypdf_logger.setLevel(logging.WARNING)

        def _boom(_stream: object) -> object:
            raise ValueError("corrupt xref")

        try:
            monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=_boom))
            result = extract_mod._extract_pdf(b"%PDF-1.4 broken")

            assert result.lossy is True
            assert pypdf_logger.level == logging.WARNING
        finally:
            pypdf_logger.setLevel(logging.NOTSET)
