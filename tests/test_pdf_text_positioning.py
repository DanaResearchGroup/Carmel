"""ISO 32000-1 conformance for :func:`carmel.services.pdf_fragments.extract_fragments`.

Every other test of this lane checks the extractor against itself or against what a
previous run produced. These check it against the SPECIFICATION, and that difference is
the reason the module they cover exists.

The lane spent 130-odd adversarial review rounds validating pypdf's glyph origins against
pdfminer, and that comparison can only ever report a disagreement -- two implementations
differing says nothing about which one is right. Two live defects survived all of it, and
one afternoon of PDFs whose every operand and every glyph width were chosen so the answer
could be computed by hand found both. The defects are described in
:func:`~carmel.services.pdf_fragments._walk_operations`; the cases below are the fixtures
that found them.

**The widths are declared in the font dictionary and the font name is a subset-style name
no library can recognise.** Both are load-bearing. A fixture named ``/Helvetica`` measures
whose METRICS win rather than whose arithmetic is right: pdfminer resolves a standard-14
``/BaseFont`` against its own AFM tables and ignores ``/Widths`` entirely, and the first
run of the probe these came from had the oracle "disagreeing with the spec" on all seven
cases at exactly Helvetica's 556/1000 digit width. Deliberately all-different widths, so a
formula that indexes the width array wrongly cannot pass by accident on uniform digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from carmel.services.pdf_fragments import extract_fragments
from tests.pypdf_gate import require_pypdf

#: Declared width of each character, per mille.
WIDTHS = {chr(48 + digit): 400 + 20 * digit for digit in range(10)}
WIDTHS[" "] = 300

FIRST_CHAR = 32
LAST_CHAR = 57  # '9'

#: How close a computed coordinate must be to the hand-computed one. Tight enough that a
#: single missing `Tc` (4 pt in the cases below, 20 pt in the tick-row case) cannot hide.
TOLERANCE = 0.001


def _widths_array() -> str:
    return " ".join(str(WIDTHS.get(chr(code), 0)) for code in range(FIRST_CHAR, LAST_CHAR + 1))


def build_pdf(content: str) -> bytes:
    """A one-page PDF whose only resource is a Type1 font with EXPLICIT widths."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /AAAAAA+ProbeTestFont "
        "/Encoding /WinAnsiEncoding "
        f"/FirstChar {FIRST_CHAR} /LastChar {LAST_CHAR} /Widths [{_widths_array()}] >>",
        None,  # the content stream, built below
    ]
    stream = content.encode("latin-1")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        if body is None:
            out += f"{number} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            out += stream + b"\nendstream\nendobj\n"
        else:
            out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("latin-1")
    return bytes(out)


def build_page(
    content: str,
    *,
    resources: str = "/Font << /F1 4 0 R >>",
    extra_objects: list[tuple[str, bytes | None]] | None = None,
) -> bytes:
    """``build_pdf`` with the page's resource dictionary and extra objects under test.

    Objects 1-5 are fixed (catalog, pages, page, the declared-widths font, the content
    stream), so ``extra_objects`` start at 6 and a resource string can reference ``6 0 R``
    without counting. Each extra is ``(dictionary, stream_bytes_or_None)``.
    """
    bodies: list[str | None] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << {resources} >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /AAAAAA+ProbeTestFont "
        "/Encoding /WinAnsiEncoding "
        f"/FirstChar {FIRST_CHAR} /LastChar {LAST_CHAR} /Widths [{_widths_array()}] >>",
        None,
    ]
    streams: dict[int, bytes] = {5: content.encode("latin-1")}
    for offset, (dictionary, data) in enumerate(extra_objects or [], start=6):
        bodies.append(dictionary if data is None else None)
        if data is not None:
            streams[offset] = data
            bodies[-1] = dictionary

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1")
        if number in streams:
            data = streams[number]
            head = body if body is not None else "<<"
            head = (head[:-2] if head.rstrip().endswith(">>") else head).rstrip()
            head = head[2:] if head.startswith("<<") else head
            out += f"<< {head} /Length {len(data)} >>\nstream\n".encode("latin-1")
            out += data + b"\nendstream\n"
        else:
            assert body is not None
            out += body.encode("latin-1") + b"\n"
        out += b"endobj\n"
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset_value in offsets:
        out += f"{offset_value:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("latin-1")
    return bytes(out)


@dataclass(frozen=True)
class Show:
    """One expected fragment: its text, and where the spec says it starts and ends."""

    text: str
    x_start: float
    x_end: float


def spec(
    *items: str | float,
    fs: float,
    tc: float = 0.0,
    tw: float = 0.0,
    tz: float = 100.0,
    scale: float = 1.0,
    start: float = 100.0,
) -> list[Show]:
    """Hand-applied ISO 32000-1 9.4.4, one character at a time.

    ``items`` is the sequence the content stream shows: a string is a run of glyphs, a
    number is a ``TJ`` array adjustment. The spec displaces by ``-k/1000 * Tfs * Th`` for
    the adjustment and charges neither ``Tc`` nor ``Tw`` on it, because it is not a glyph::

        tx = ((w0 - Tj / 1000) * Tfs + Tc + Tw) * Th

    Written as a loop over characters rather than as a closed form on purpose. The closed
    form is where the per-glyph ``Tc`` quietly becomes a per-run one, which is the defect
    these fixtures exist to catch, so the oracle spells out what "per glyph" means.
    """
    pen = start
    shows: list[Show] = []
    for item in items:
        if not isinstance(item, str):
            pen += -float(item) / 1000.0 * fs * (tz / 100.0) * scale
            continue
        x_start = pen
        for char in item:
            advance = (WIDTHS[char] / 1000.0) * fs + tc + (tw if char == " " else 0.0)
            pen += advance * (tz / 100.0) * scale
        shows.append(Show(item, x_start, pen))
    return shows


CASES: list[tuple[str, str, list[Show]]] = [
    (
        "plain Tj",
        "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012345) Tj ET",
        spec("012345", fs=10.0),
    ),
    (
        # The corpus's fused tick row: one show, character spacing set to the tick pitch.
        "Tc is charged per glyph",
        "BT /F1 10 Tf 20 Tc 1 0 0 1 100 700 Tm (012345) Tj ET",
        spec("012345", fs=10.0, tc=20.0),
    ),
    (
        "Tw applies to spaces only",
        "BT /F1 10 Tf 15 Tw 1 0 0 1 100 700 Tm (0 1 2) Tj ET",
        spec("0 1 2", fs=10.0, tw=15.0),
    ),
    (
        "Tz scales every displacement including Tc",
        "BT /F1 10 Tf 5 Tc 50 Tz 1 0 0 1 100 700 Tm (0123) Tj ET",
        spec("0123", fs=10.0, tc=5.0, tz=50.0),
    ),
    (
        "a scaled text matrix scales the advance",
        "BT /F1 10 Tf 3 Tc 2 0 0 2 100 700 Tm (0123) Tj ET",
        spec("0123", fs=10.0, tc=3.0, scale=2.0),
    ),
    (
        "a TJ adjustment displaces without charging Tc",
        "BT /F1 10 Tf 1 0 0 1 100 700 Tm [(01) -500 (23)] TJ ET",
        spec("01", -500.0, "23", fs=10.0),
    ),
    (
        # Defect D2, at three different element lengths. If the deficit were `Tc` charged
        # once per TJ ELEMENT, a longer first element would lose proportionally more: two
        # glyphs lose one `Tc`, three lose two, one loses nothing. That is exactly the
        # pattern pypdf shows, which is how the mechanism was identified rather than
        # merely observed.
        "TJ with Tc, first element two glyphs",
        "BT /F1 10 Tf 4 Tc 1 0 0 1 100 700 Tm [(01) -800 (23)] TJ ET",
        spec("01", -800.0, "23", fs=10.0, tc=4.0),
    ),
    (
        "TJ with Tc, first element three glyphs",
        "BT /F1 10 Tf 4 Tc 1 0 0 1 100 700 Tm [(012) -800 (34)] TJ ET",
        spec("012", -800.0, "34", fs=10.0, tc=4.0),
    ),
    (
        "TJ with Tc, first element one glyph",
        "BT /F1 10 Tf 4 Tc 1 0 0 1 100 700 Tm [(0) -800 (12)] TJ ET",
        spec("0", -800.0, "12", fs=10.0, tc=4.0),
    ),
    (
        # Defect D1. No TJ array anywhere: if the pen only fails to advance inside one,
        # this is clean; if a show operator never advances it at all, it is not.
        "consecutive Tj operators each advance the pen",
        "BT /F1 10 Tf 4 Tc 1 0 0 1 100 700 Tm (01) Tj (23) Tj ET",
        spec("01", "23", fs=10.0, tc=4.0),
    ),
    (
        # The same with `Tc` zero, which separates "a Tc effect" from "unconditional".
        # It is unconditional, which makes D1 a defect in every document with consecutive
        # shows rather than only in Tc-spaced ones.
        "consecutive Tj operators advance with no character spacing",
        "BT /F1 10 Tf 1 0 0 1 100 700 Tm (01) Tj (23) Tj ET",
        spec("01", "23", fs=10.0),
    ),
]


@pytest.mark.parametrize(("name", "content", "expected"), CASES, ids=[c[0] for c in CASES])
def test_show_origins_match_the_specification(name: str, content: str, expected: list[Show]) -> None:
    require_pypdf()
    extraction = extract_fragments(build_pdf(content))

    assert extraction.available, name
    assert not extraction.page_failures, name
    got = [(f.text, f.x_start, f.x_end) for f in extraction.fragments]
    assert [f.text for f in extraction.fragments] == [s.text for s in expected], (
        f"{name}: wrong number or order of shows -- {got}"
    )
    for fragment, want in zip(extraction.fragments, expected, strict=True):
        assert fragment.x_start == pytest.approx(want.x_start, abs=TOLERANCE), (
            f"{name}: {fragment.text!r} starts at {fragment.x_start}, spec says {want.x_start}"
        )
        assert fragment.x_end == pytest.approx(want.x_end, abs=TOLERANCE), (
            f"{name}: {fragment.text!r} ends at {fragment.x_end}, spec says {want.x_end}"
        )


def test_the_pen_advance_survives_a_graphics_state_restore() -> None:
    """``Q`` restores ``Tc`` -- it is graphics state, not a text-object property.

    pypdf's ``TextStateManager`` saves only the font and font size across ``q``, so a
    ``Tc`` set inside a saved block leaked out of it and kept displacing text after the
    ``Q``. Nothing in the eight-paper corpus depends on this, which is precisely why it
    needs a fixture: it is a correctness claim no real document in hand would exercise.
    """
    require_pypdf()
    content = "BT /F1 10 Tf 1 0 0 1 100 700 Tm q 20 Tc (01) Tj Q (23) Tj ET"
    extraction = extract_fragments(build_pdf(content))

    assert extraction.available
    assert not extraction.page_failures
    # `(01)` is drawn with Tc=20; `(23)` after the Q is drawn with Tc back at 0, and
    # starts where the first run's pen left it.
    inner = spec("01", fs=10.0, tc=20.0)[0]
    outer = spec("23", fs=10.0, start=inner.x_end)[0]
    assert [f.text for f in extraction.fragments] == ["01", "23"]
    assert extraction.fragments[0].x_start == pytest.approx(inner.x_start, abs=TOLERANCE)
    assert extraction.fragments[1].x_start == pytest.approx(outer.x_start, abs=TOLERANCE)
    assert extraction.fragments[1].x_end == pytest.approx(outer.x_end, abs=TOLERANCE)


def test_a_form_xobject_fails_its_page_rather_than_dropping_its_text() -> None:
    """Text drawn through ``Do`` is not positioned here, and is not silently lost either.

    pypdf's layout-mode walker has no ``Do`` branch, so a form XObject's text is ABSENT
    from its output -- a page that reads as complete with a column missing. Refusing the
    page converts that into a recorded failure. See
    :func:`~carmel.services.pdf_fragments._refuse_form_xobject` for why recursion is not
    built instead: no document in the corpus contains a single form XObject.
    """
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012345) Tj ET /X1 Do",
            resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
            extra_objects=[
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] >>",
                    b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (99) Tj ET",
                )
            ],
        )
    )

    assert extraction.available
    assert extraction.lossy
    assert [f.page for f in extraction.page_failures] == [1]
    # And nothing from the page is published: a partial page is the failure mode the
    # refusal exists to prevent, so the six glyphs drawn BEFORE the /Do are dropped too.
    assert extraction.fragments == ()


def _refusal(content: str, **kwargs: object) -> str:
    """Extract, assert the page was refused rather than published, return the reason."""
    extraction = extract_fragments(build_page(content, **kwargs))  # type: ignore[arg-type]
    assert extraction.available, "a refused CONSTRUCT must not make the document unavailable"
    assert extraction.lossy
    assert extraction.fragments == ()
    assert [f.page for f in extraction.page_failures] == [1]
    return extraction.page_failures[0].error


def test_invisible_text_is_refused_rather_than_published_as_drawn() -> None:
    """``3 Tr`` paints nothing, and an OCR layer under a scan is drawn exactly that way.

    Publishing its geometry would put an invisible copy of a number in competition with
    the visible one at the same coordinates. Skipping it silently would drop the only
    text such a page has. Refusing is the only option that says what happened.
    """
    require_pypdf()
    assert "paints nothing" in _refusal("BT /F1 10 Tf 3 Tr 1 0 0 1 100 700 Tm (012) Tj ET")
    assert "paints nothing" in _refusal("BT /F1 10 Tf 7 Tr 1 0 0 1 100 700 Tm (012) Tj ET")


def test_a_visible_rendering_mode_is_still_published() -> None:
    """The guard is on modes 3 and 7 alone -- stroke, fill-and-stroke and the clipping
    variants that DO paint stay ordinary text, or the refusal would be a denylist of one
    value pretending to be a scope boundary."""
    require_pypdf()
    for mode in (0, 1, 2, 4, 5, 6):
        extraction = extract_fragments(build_page(f"BT /F1 10 Tf {mode} Tr 1 0 0 1 100 700 Tm (012) Tj ET"))
        assert not extraction.page_failures, f"mode {mode} was refused"
        assert [f.text for f in extraction.fragments] == ["012"], f"mode {mode}"


def test_an_extgstate_that_sets_a_font_is_refused() -> None:
    """A ``gs`` may set the font without a ``Tf``; ignoring it advances the pen with the
    previous font's widths and raises nothing."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
        resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /Font [4 0 R 10] >> >>",
    )
    assert "sets the font without a Tf" in reason


def test_an_ordinary_extgstate_is_not_refused() -> None:
    """2,862 ``gs`` operators on 73 of the corpus's 75 pages carry no ``/Font``. A guard
    on the operator rather than on ``/Font`` would fail almost every page in hand."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca 0.5 >> >>",
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_vertical_font_is_refused_rather_than_advanced_horizontally() -> None:
    """The engine advances in x unconditionally. A vertical CMap advances in y, so every
    glyph after the first would get a fabricated x with nothing raised. "No vertical
    writing modes" is the scope the user set, and a scope that is not enforced is not one.
    """
    require_pypdf()
    reason = _refusal(
        "BT /V1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET",
        resources="/Font << /V1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /Font /Subtype /Type0 /BaseFont /AAAAAA+Probe /Encoding /Identity-V /DescendantFonts [] >>",
                None,
            )
        ],
    )
    assert "horizontal" in reason


def test_a_differences_encoding_is_not_mistaken_for_a_cmap() -> None:
    """The first cut of the vertical-font guard refused any ``/Encoding`` carrying a
    ``/Type``, and a simple encoding dictionary declares ``/Type /Encoding``.

    It passed its own reasoning and was a false positive against the overwhelmingly
    common horizontal case -- every Type1 font with a ``/Differences`` array, which is
    most of the corpus. Kept as a test because the guard is fail-closed by design, and a
    fail-closed guard's failure mode is silently refusing real data.
    """
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /D1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /D1 6 0 R >>",
            extra_objects=[
                (
                    "<< /Type /Font /Subtype /Type1 /BaseFont /AAAAAA+ProbeTestFont "
                    "/Encoding << /Type /Encoding /Differences [48 /zero /one /two] >> "
                    f"/FirstChar {FIRST_CHAR} /LastChar {LAST_CHAR} /Widths [{_widths_array()}] >>",
                    None,
                )
            ],
        )
    )
    assert not extraction.page_failures
    assert extraction.fragments


def test_a_do_on_an_unrecognised_xobject_subtype_is_refused() -> None:
    """An allowlist, not a denylist: the question is whether this can be PROVEN to draw
    no text, and a missing or unfamiliar ``/Subtype`` proves nothing."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET /X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[("<< /Type /XObject /BBox [0 0 10 10] >>", b"")],
    )
    assert "subtype" in reason


def test_a_do_on_an_image_is_not_refused() -> None:
    """All 71 XObjects in the corpus are images, on 37 of 75 pages. The allowlist has to
    admit them or the lane refuses half its own evidence."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET /X1 Do",
            resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
            extra_objects=[
                (
                    "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray "
                    "/BitsPerComponent 8 >>",
                    b"\x00",
                )
            ],
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]
