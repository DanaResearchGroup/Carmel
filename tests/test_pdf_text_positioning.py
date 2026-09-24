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

from carmel.services.pdf_fragments import (
    UnsupportedContentConstruct,
    _num,
    extract_fragments,
)
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
    page_extra: str = "",
) -> bytes:
    """``build_pdf`` with the page's resource dictionary and extra objects under test.

    Objects 1-5 are fixed (catalog, pages, page, the declared-widths font, the content
    stream), so ``extra_objects`` start at 6 and a resource string can reference ``6 0 R``
    without counting. Each extra is ``(dictionary, stream_bytes_or_None)``.

    ``page_extra`` goes into the PAGE dictionary rather than the content stream, which is
    the only way to reach the entries that reframe a page from outside its operators --
    ``/Rotate`` and ``/UserUnit``.
    """
    bodies: list[str | None] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] {page_extra} "
        f"/Resources << {resources} >> /Contents 5 0 R >>",
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


def _form_page(
    page_content: str,
    forms: list[tuple[str, bytes]],
    *,
    page_resources: str = "/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
) -> bytes:
    """A page whose content is ``page_content`` plus one or more form XObjects.

    ``forms`` are ``(dictionary_without_/Length, content_bytes)`` pairs assigned to objects
    6, 7, ... in order, so a page or a form can reference ``6 0 R``, ``7 0 R`` and so on.
    """
    return build_page(page_content, resources=page_resources, extra_objects=list(forms))


def test_a_form_xobject_has_its_text_positioned_in_page_space() -> None:
    """The construct the eight-paper corpus lacked and the 300-paper library is full of.

    pypdf's layout-mode walker has no ``Do`` branch, so a form XObject's text was ABSENT --
    a page that reads as complete with a column missing. The walker now follows the form,
    composing its ``/Matrix`` (here the identity) and clipping to its ``/BBox``, so the
    text it draws is published at its real page position alongside the page's own.
    """
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012345) Tj ET /X1 Do",
            [
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> >>",
                    b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (99) Tj ET",
                )
            ],
        )
    )

    assert extraction.available
    assert not extraction.lossy
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012345", "99"]
    inside = extraction.fragments[1]
    assert inside.x_start == pytest.approx(10.0, abs=TOLERANCE)
    assert inside.baseline_y == pytest.approx(10.0, abs=TOLERANCE)
    assert all(f.glyph_mapping.value == "mapped" for f in extraction.fragments)


def test_a_text_free_form_no_longer_fails_the_page() -> None:
    """The 99% case, and the largest single win: a decorative vector form killed the page.

    In the 300-paper sample almost every form carries no text at all -- a rule, a logo, a
    figure drawn as vector graphics -- and refusing the ``Do`` failed the whole page for it,
    dropping the page's real text with it. Following the form finds no text-show operator,
    so it contributes nothing and, crucially, lets the page complete.
    """
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012345) Tj ET /X1 Do",
            [("<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] >>", b"0 0 50 50 re f")],
        )
    )

    assert not extraction.lossy
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012345"]


def test_a_form_matrix_is_composed_onto_the_ctm() -> None:
    """The form's ``/Matrix`` maps its space into the ``Do``'s space (ISO 32000-1 8.10.1).

    A form drawn at ``(10, 10)`` in its own space, invoked through a ``/Matrix`` that
    translates by ``(300, 400)``, must publish its text at ``(310, 410)`` in page space.
    """
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "/X1 Do",
            [
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Matrix [1 0 0 1 300 400] "
                    "/Resources << /Font << /F1 4 0 R >> >> >>",
                    b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET",
                )
            ],
        )
    )

    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]
    assert extraction.fragments[0].x_start == pytest.approx(310.0, abs=TOLERANCE)
    assert extraction.fragments[0].baseline_y == pytest.approx(410.0, abs=TOLERANCE)


def test_a_form_inherits_page_resources_when_it_declares_none() -> None:
    """A form with no ``/Resources`` draws in the environment it appears in (8.10.1).

    The page's ``/F1`` is in force inside it, so its text is MAPPED rather than falling
    through to an uninterpretable placeholder.
    """
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "/X1 Do",
            [("<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] >>", b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET")],
        )
    )

    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]
    assert extraction.fragments[0].glyph_mapping.value == "mapped"


def test_a_form_inside_a_form_is_followed() -> None:
    """A form may draw another form; the walker follows the chain and composes each matrix."""
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "/X1 Do",
            [
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Matrix [1 0 0 1 100 0] "
                    "/Resources << /XObject << /X2 7 0 R >> >> >>",
                    b"/X2 Do",
                ),
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Matrix [1 0 0 1 0 500] "
                    "/Resources << /Font << /F1 4 0 R >> >> >>",
                    b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET",
                ),
            ],
        )
    )

    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]
    # 10 + 100 (outer matrix) + 0 (inner), 10 + 0 + 500.
    assert extraction.fragments[0].x_start == pytest.approx(110.0, abs=TOLERANCE)
    assert extraction.fragments[0].baseline_y == pytest.approx(510.0, abs=TOLERANCE)


def test_an_image_xobject_still_draws_no_text() -> None:
    """An image ``Do`` is let through unchanged: it draws no text, so the page completes."""
    require_pypdf()
    extraction = extract_fragments(
        _form_page(
            "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET /X1 Do",
            [
                (
                    "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
                    "/ColorSpace /DeviceGray /BitsPerComponent 8 >>",
                    b"\x00",
                )
            ],
        )
    )

    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_non_image_non_form_xobject_is_still_refused() -> None:
    """The allowlist holds: only a provable image or form is admitted, everything else
    refuses. A ``/PS`` (PostScript) XObject could draw anything and is not followed."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET /X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[("<< /Type /XObject /Subtype /PS >>", b"")],
    )
    assert "may draw text this module does not position" in reason


def test_a_form_under_a_rotated_ctm_refuses_its_text() -> None:
    """The neighbouring case the matrix composition must NOT admit.

    Under a rotated CTM the form's ``/BBox`` maps to a parallelogram no rectangle describes,
    so the clip becomes UNKNOWN and the form's text meets the same refusal a rotated
    page-level clip triggers -- rather than being published against a box that is not its
    true extent.
    """
    require_pypdf()
    reason = _refusal(
        "q 0.7 0.7 -0.7 0.7 0 0 cm /X1 Do Q",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Resources << /Font << /F1 4 0 R >> >> >>",
                b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET",
            )
        ],
    )
    assert "cannot reduce to a rectangle" in reason


def test_a_form_whose_bbox_does_not_contain_its_text_refuses() -> None:
    """The ``/BBox`` is a clip: text drawn outside it paints nothing and must not publish."""
    require_pypdf()
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 5 5] /Resources << /Font << /F1 4 0 R >> >> >>",
                b"BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET",
            )
        ],
    )
    assert "does not provably contain" in reason


def test_a_form_that_references_itself_refuses() -> None:
    """A cycle is refused rather than followed until the stack gives out."""
    require_pypdf()
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Resources << /XObject << /X1 6 0 R >> >> >>",
                b"/X1 Do",
            )
        ],
    )
    assert "draws itself" in reason


def test_a_form_nested_past_the_depth_limit_refuses() -> None:
    """A legitimate but pathologically deep acyclic nest is bounded, not stack-crashed."""
    require_pypdf()
    from carmel.services.pdf_fragments import _MAX_FORM_DEPTH

    chain = _MAX_FORM_DEPTH + 2
    forms: list[tuple[str, bytes]] = []
    for level in range(chain):
        obj = 6 + level
        if level < chain - 1:
            forms.append(
                (
                    f"<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] "
                    f"/Resources << /XObject << /X{level + 1} {obj + 1} 0 R >> >> >>",
                    f"/X{level + 1} Do".encode("latin-1"),
                )
            )
        else:
            forms.append(
                (
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Resources << /Font << /F1 4 0 R >> >> >>",
                    b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET",
                )
            )
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=forms,
    )
    assert "nested past the depth" in reason


def test_a_form_with_no_bbox_refuses() -> None:
    """A ``/BBox`` is required for a form; without it its visible extent is undefined."""
    require_pypdf()
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            ("<< /Type /XObject /Subtype /Form /Resources << /Font << /F1 4 0 R >> >> >>", b"BT /F1 10 Tf (012) Tj ET")
        ],
    )
    assert "no /BBox" in reason


def test_a_form_with_a_malformed_matrix_refuses() -> None:
    """A ``/Matrix`` that is not six numbers is malformed and refuses rather than guessed."""
    require_pypdf()
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Matrix [1 0 0 1] "
                "/Resources << /Font << /F1 4 0 R >> >> >>",
                b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (012) Tj ET",
            )
        ],
    )
    assert "/Matrix is not six numbers" in reason


def test_the_shared_form_decode_budget_bounds_a_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """One page cannot inflate unbounded form content: the sum is capped per page.

    With the per-page ceiling dropped low, a form whose content exceeds what the page's own
    ``/Contents`` left fails that page rather than the document -- a recorded, visible
    failure, the fail-closed direction.
    """
    require_pypdf()
    import carmel.services.pdf_fragments as mod

    monkeypatch.setattr(mod, "MAX_PAGE_CONTENT_BYTES", 60)
    reason = _refusal(
        "/X1 Do",
        resources="/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >>",
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 600 792] /Resources << /Font << /F1 4 0 R >> >> >>",
                b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (0123456789) Tj (0123456789) Tj (0123456789) Tj ET",
            )
        ],
    )
    assert "past the" in reason and "cap" in reason


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


def test_a_painting_rendering_mode_is_still_published() -> None:
    """Modes 0, 1 and 2 fill, stroke and do both. All three are ordinary text, or the
    refusal would be a denylist pretending to be a scope boundary."""
    require_pypdf()
    for mode in (0, 1, 2):
        extraction = extract_fragments(build_page(f"BT /F1 10 Tf {mode} Tr 1 0 0 1 100 700 Tm (012) Tj ET"))
        assert not extraction.page_failures, f"mode {mode} was refused"
        assert [f.text for f in extraction.fragments] == ["012"], f"mode {mode}"


def test_a_clipping_rendering_mode_is_refused_even_though_it_paints() -> None:
    """Modes 4-6 paint the glyph AND add it to the clipping path (table 106).

    The glyph itself is visible, so the earlier version of this test asserted all three
    were published. What that missed is the side effect: every LATER glyph on the page is
    confined to the intersection of the clip with these glyph outlines, and this walker
    models neither. Zero corpus population -- ``Tr`` is never set to 4, 5 or 6 in the eight
    papers -- so the refusal costs nothing and closes the channel by construction.
    """
    require_pypdf()
    for mode in (4, 5, 6):
        reason = _refusal(f"BT /F1 10 Tf {mode} Tr 1 0 0 1 100 700 Tm (012) Tj ET")
        assert "adds its glyphs to the clipping path" in reason, f"mode {mode}"


def test_text_a_rectangular_clip_does_not_contain_is_refused() -> None:
    """``re W n`` sets a clip; a glyph outside it paints nothing and must not publish."""
    require_pypdf()
    reason = _refusal("q 0 0 10 10 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_text_inside_a_rectangular_clip_is_published() -> None:
    """The case the boolean version of this guard refused, and the reason it was replaced.

    A plot area is ``x y w h re W n`` and its axis labels are drawn inside it. Refusing
    every show under any clip cost 1,415 fragments -- a whole figure page -- to guard two
    axis labels that were perfectly visible. Modelling the one shape that matters returns
    them, and everything that is not that shape still refuses.
    """
    require_pypdf()
    extraction = extract_fragments(build_page("q 50 650 200 100 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q"))
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_clip_that_cuts_the_end_off_a_run_is_refused() -> None:
    """The reason the test is on the EXTENT and not on the origin.

    The clip starts at x=100 and ends at x=104, so it contains this run's first glyph and
    cuts away the rest. An origin test would publish an ``x_end`` past the clip edge -- a
    coordinate for ink that was never laid down, which is worse than refusing because it
    looks checkable.
    """
    require_pypdf()
    reason = _refusal("q 100 650 4 100 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_a_clip_that_cuts_below_the_baseline_is_refused() -> None:
    """The case that killed the first version of the vertical rule.

    That version anchored the box AT the baseline, on the argument that vertical clipping
    only shaves parts of glyphs. It does not: a comma hangs below the baseline, and a clip
    that takes it turns ``1,234`` into ``1234`` -- a number wrong by three orders of
    magnitude, wearing text that looks clean. Below-baseline ink is evidence, so the clip
    here (bottom edge exactly at the baseline, 700) must refuse.
    """
    require_pypdf()
    reason = _refusal("q 50 700 400 60 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_a_sub_point_overhang_is_not_refused() -> None:
    """The other half of the same rule, and the reason it is a tolerance and not a policy.

    The clip rectangle is exact; the glyph box is a NOMINAL em-square estimate. Demanding
    exact containment of an estimate reports a disagreement between the estimate and
    reality as though it were a fact about the document. A producer setting an axis label
    flush with a plot boundary overhangs by a fraction of a point as a matter of course --
    here 0.1 pt, about a fifth of a pixel at 300 dpi.
    """
    require_pypdf()
    extraction = extract_fragments(build_page("q 50 697.9 400 60 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q"))
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_the_tolerance_cannot_absorb_a_descender() -> None:
    """The tolerance must stay far below the smallest ink that carries meaning, or it
    becomes the policy it replaced. A clip 1 pt above the descender's reach still refuses:
    that is enough to take a comma, and a quarter point is not."""
    require_pypdf()
    reason = _refusal("q 50 699 400 60 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_a_clip_that_cuts_the_top_off_the_glyph_bodies_is_refused() -> None:
    """The band is tested at both edges. A clip whose top sits inside the nominal ascent
    takes the tops off the glyphs, and a digit with its top removed is not a digit anyone
    should read a number from."""
    require_pypdf()
    reason = _refusal("q 50 690 400 15 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_a_non_rectangular_clip_refuses_rather_than_being_approximated() -> None:
    """A clip built from lines is not a rectangle, and its bounding box is not a
    conservative substitute -- a box is LARGER than the region, so publishing against it
    would admit glyphs the real clip cuts away."""
    require_pypdf()
    reason = _refusal("q 0 0 m 500 0 l 250 750 l h W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "cannot reduce to a rectangle" in reason


def test_two_rectangles_in_one_clip_path_are_not_read_as_an_intersection() -> None:
    """Two ``re`` before one ``W`` are one path with two subpaths. Under the nonzero
    winding rule that is a UNION, not an intersection; reading it as either would be a
    guess about which region may be published in."""
    require_pypdf()
    reason = _refusal("q 0 0 600 780 re 50 650 200 100 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "cannot reduce to a rectangle" in reason


def test_a_clip_rectangle_under_a_rotated_ctm_is_not_modelled() -> None:
    """``re`` draws in USER space. Under a sheared or rotated CTM its page-space image is
    a parallelogram, and no ``(x0, y0, x1, y1)`` describes it."""
    require_pypdf()
    reason = _refusal("q 0.7 0.7 -0.7 0.7 0 0 cm 0 0 600 780 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "cannot reduce to a rectangle" in reason


def test_successive_clips_intersect() -> None:
    """Two clips established by two separate ``W n`` sequences DO intersect -- unlike two
    subpaths of one path. Here the second cuts the first down to a band that no longer
    contains the text."""
    require_pypdf()
    reason = _refusal("q 0 0 600 780 re W n 0 0 600 100 re W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q")
    assert "does not provably contain" in reason


def test_a_painted_rectangle_does_not_attach_itself_to_a_later_clip() -> None:
    """The current path is cleared at EVERY path-ending operator, marked or not.

    Without that, the ``re`` painted here would still be in the list when the later ``W``
    arrives, and the clip would be established from a rectangle drawn for something else
    -- a clip this module believes it may publish inside, invented out of stale state.
    """
    require_pypdf()
    reason = _refusal(
        "q 0 0 600 780 re f 0 0 10 10 re 20 20 m 30 30 l W n BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET Q"
    )
    assert "cannot reduce to a rectangle" in reason


def test_a_clip_discarded_by_q_before_any_text_is_not_refused() -> None:
    """The shape that makes the corpus number small, and the reason the guard is on the
    clip being IN FORCE rather than on ``W`` appearing.

    ``q ... re W n ... Q`` clips a figure and restores the state before any text is shown.
    That is 50 of the corpus's 75 pages. Refusing on ``W`` would have failed all of them
    to catch the 1 page where a clip is actually in force over a glyph.
    """
    require_pypdf()
    extraction = extract_fragments(build_page("q 0 0 10 10 re W n Q BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET"))
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_clip_marked_but_not_yet_in_force_still_refuses_a_show() -> None:
    """``W`` with text before the path is ended. Malformed for the state machine, and the
    safe reading of a malformed clip is that a clip is coming. Zero corpus population."""
    require_pypdf()
    reason = _refusal("0 0 10 10 re W BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET")
    assert "between a W and the operator that ends its path" in reason


def test_painting_a_path_without_a_pending_clip_does_not_refuse() -> None:
    """The painting operators left ``_IGNORED_OPERATORS`` and gained a branch. That branch
    must remain a no-op when no ``W`` preceded it, or every ruled table would fail."""
    require_pypdf()
    extraction = extract_fragments(
        build_page("0 0 10 10 re f 20 20 m 30 30 l S BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET")
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


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


def test_an_operator_the_walker_does_not_model_is_refused() -> None:
    """The walker had no final ``else``: anything it did not name was stepped over.

    An inline image is the sharp case. ``BI ... ID <binary> EI`` carries raw bytes that an
    operand parser can read as operators, so a walk that ignores ``BI`` continues from a
    state it cannot vouch for -- and every fragment after it is published anyway.
    """
    require_pypdf()
    reason = _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET\nBI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI")
    assert "does not model" in reason


def test_a_compatibility_section_is_refused_rather_than_obeyed() -> None:
    """``BX`` means "ignore operators you do not recognise", which is the instruction the
    guard exists to disobey. Obeying it would let any unmodelled construct through by
    simply announcing itself first."""
    require_pypdf()
    assert "does not model" in _refusal("BX BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET EX")


def test_the_ignorable_operators_do_not_refuse() -> None:
    """30 distinct unnamed operators appear in the corpus across 236,621 calls -- paths,
    colours, line state, marked content. The allowlist has to cover all of them, or the
    final ``else`` refuses every page in hand rather than the unmodelled ones."""
    require_pypdf()
    noise = (
        "q 0.5 w 1 J 1 j 10 M [3 2] 0 d /RelativeColorimetric ri 0 i "
        "0 0 1 RG 1 0 0 rg 0 g 0 G 0 0 0 1 K 0 0 0 0 k /DeviceGray CS /DeviceGray cs "
        "0 SC 0 sc 0 SCN 0 scn "
        "100 100 m 200 200 l 150 150 100 100 120 120 c 130 130 140 140 v 150 150 160 160 y h "
        "10 10 50 50 re S f f* B B* b b* n W W* "
        "/Span << /Lang (en) >> BDC /MC0 BMC EMC EMC /P MP /P << /X 1 >> DP"
    )
    extraction = extract_fragments(build_page(f"{noise} Q BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET"))
    assert not extraction.page_failures, extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_zero_fill_alpha_is_refused_in_a_filling_mode() -> None:
    """``/ca 0`` paints nothing, exactly as ``3 Tr`` does, arriving through the graphics
    state instead of through a text operator."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
        resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca 0 >> >>",
    )
    assert "fully transparent" in reason


def test_zero_stroke_alpha_does_not_refuse_filled_text() -> None:
    """The measurement that decides the shape of the alpha guard.

    2,552 of the corpus's 2,862 ``gs`` invocations set ``/CA 0`` on 7 pages -- and not one
    corpus page carries a single ``Tr`` operator, so every glyph is drawn in mode 0,
    filled only. A guard reading "any alpha of zero refuses" would have failed 7 real
    pages over a parameter that does not touch their text. Which alpha counts is the
    rendering mode's business.
    """
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /CA 0 >> >>",
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_zero_stroke_alpha_is_refused_when_the_mode_only_strokes() -> None:
    """Mode 1 strokes and does not fill, so ``/CA 0`` is exactly as invisible there as
    ``/ca 0`` is in mode 0. The same state refuses or not depending on the mode, which is
    why the check cannot live at the ``gs``."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf /GS1 gs 1 Tr 1 0 0 1 100 700 Tm (012) Tj ET",
        resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /CA 0 >> >>",
    )
    assert "fully transparent" in reason


def test_a_partially_transparent_glyph_is_still_evidence() -> None:
    """``== 0`` and not a threshold: a glyph at 1% opacity is faint, not absent, and
    picking a visibility cutoff would be this module inventing a perceptual judgement."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca 0.01 >> >>",
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_graphics_state_restore_restores_the_alpha_too() -> None:
    """``/ca`` is graphics state, so ``Q`` undoes it. If the alpha did not ride on the
    saved state, text drawn after the restore would inherit an invisibility that the page
    had already taken back."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "q /GS1 gs Q BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca 0 >> >>",
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_form_inherits_the_callers_alpha() -> None:
    """A ``/Form`` does not reset the graphics state -- ISO 32000-1 8.10.2 paints its
    content under whatever was in force at the ``Do``. ``q /GS1 gs /X1 Do Q`` with ``/ca
    0`` makes the page fully transparent before it draws the form, so the form's text is
    exactly as invisible as text drawn directly on the page under the same state would be.
    A recursive walk that started the form from a fresh, opaque ``_TextState`` would
    publish it as ordinary visible text instead."""
    require_pypdf()
    reason = _refusal(
        "q /GS1 gs /X1 Do Q",
        resources=("/Font << /F1 4 0 R >> /XObject << /X1 6 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca 0 >> >>"),
        extra_objects=[
            (
                "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> >>",
                b"BT /F1 10 Tf 1 0 0 1 10 10 Tm (99) Tj ET",
            )
        ],
    )
    assert "fully transparent" in reason


def test_a_soft_mask_is_refused() -> None:
    """A soft mask can erase what is painted, and evaluating one means owning the mask's
    own content stream. All nine ``/SMask`` entries in the corpus are ``/None``."""
    require_pypdf()
    reason = _refusal(
        "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
        resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /SMask 6 0 R >> >>",
        extra_objects=[("<< /Type /Mask /S /Alpha >>", None)],
    )
    assert "soft mask" in reason


def test_an_smask_of_none_is_not_refused() -> None:
    """``/SMask /None`` is how a page TURNS OFF a soft mask, and it is the only form the
    corpus contains. Refusing it would refuse the disabling of the very thing guarded."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /SMask /None >> >>",
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_a_rotated_page_is_refused_rather_than_published_in_an_unrotated_frame() -> None:
    """``/Rotate`` never appears in the content stream, so the walker cannot see it. Its
    coordinates stay arithmetically consistent while naming a position on an axis the
    reader never sees -- a locator that is checkable and wrong."""
    require_pypdf()
    for angle in (90, 180, 270, -90):
        reason = _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET", page_extra=f"/Rotate {angle}")
        assert "rotated" in reason, angle


def test_a_rotation_of_zero_is_not_refused() -> None:
    """All 75 corpus pages declare ``/Rotate 0`` or nothing at all. A guard on the
    presence of the key rather than on its value would refuse every one of them."""
    require_pypdf()
    for angle in (0, 360, -360):
        extraction = extract_fragments(
            build_page("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET", page_extra=f"/Rotate {angle}")
        )
        assert not extraction.page_failures, angle
        assert [f.text for f in extraction.fragments] == ["012"], angle


def test_a_page_declaring_a_user_unit_is_refused() -> None:
    """Every distance this module publishes, and every threshold compared against one,
    is in default user space. ``/UserUnit`` says that is not the scale."""
    require_pypdf()
    reason = _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET", page_extra="/UserUnit 2.0")
    assert "UserUnit" in reason


def test_a_positioning_operand_that_is_not_a_number_is_refused() -> None:
    """An operand of the wrong TYPE, reached the only way pypdf's parser allows it.

    ``nan``, ``inf`` and ``true`` are not PDF number syntax: the tokenizer returns each as
    an OPERATOR, so they arrive at the walker's final ``else`` instead. Inside a ``TJ``
    array they are parsed as objects, and that is where a non-numeric displacement can
    actually reach :func:`_num`.
    """
    require_pypdf()
    for element in ("true", "null"):
        reason = _refusal(f"BT /F1 10 Tf 1 0 0 1 100 700 Tm [(01) {element} (23)] TJ ET")
        assert "not a number" in reason, element


def test_the_num_guard_refuses_values_its_own_parser_cannot_deliver() -> None:
    """The two branches of :func:`_num` that no fixture PDF can reach.

    pypdf returns ``true`` as an operator and a ``BooleanObject`` (not a ``bool``) inside
    an array, and it refuses a numeric token longer than 64 characters, so neither a
    ``bool`` nor a non-finite float can arrive from a real document today. Both are
    contracts on the function rather than parser guards -- ``float(True)`` is 1.0, and a
    ``nan`` coordinate compares false against the page box -- and a contract with no test
    is a comment. Tested directly, so it is clear that is what they are.
    """
    require_pypdf()
    for value in (True, False, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(UnsupportedContentConstruct):
            _num(value)


def test_a_tj_operand_that_is_not_an_array_is_refused() -> None:
    """``TJ`` takes an array. A bytes operand is iterable too, and iterating it yields
    INTEGERS -- every one of which the loop would apply as a displacement, silently
    turning a string into a run of pen movements."""
    require_pypdf()
    assert "not an array" in _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) TJ ET")


def test_a_name_inside_a_tj_array_is_not_published_as_text() -> None:
    """The sharpest of these guards, because it FABRICATES rather than misplaces.

    ``NameObject`` subclasses ``str``, so the walker's ``isinstance(element, bytes | str)``
    test admitted it and ``[(01) /Nm (23)] TJ`` published a fragment reading ``/Nm`` at
    real page coordinates -- text no glyph drew, wearing checkable geometry. Found by
    trying it, not by review: a name token is not a string, so no amount of looking at
    real papers' strings would ever have shown it.
    """
    require_pypdf()
    reason = _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm [(01) /Nm (23)] TJ ET")
    assert "not a string of bytes" in reason


def test_a_name_operand_to_tj_is_not_published_as_text() -> None:
    """The same hole through the single-string operator: ``/Nm Tj``."""
    require_pypdf()
    assert "not a string of bytes" in _refusal("BT /F1 10 Tf 1 0 0 1 100 700 Tm /Nm Tj ET")


def test_show_operands_are_bytes_only_because_this_module_asks_for_bytes() -> None:
    """Pin the coupling that makes ``_show_operand``'s second branch unreachable.

    ``_page_fragments`` builds its ``ContentStream`` with ``forced_encoding="bytes"``. That
    argument -- not anything about the papers -- is why every show operand this module ever
    sees is a ``ByteStringObject``. Without it pypdf decodes ``(Hello)`` to a
    ``TextStringObject`` whose characters have REPLACED the code bytes that ``/Encoding``
    and ``/Widths`` are indexed by, so every glyph width would then be looked up under the
    wrong key while the fragment still looked well formed.

    Asserted here on both string syntaxes, because they take different parse paths in pypdf
    (``read_string_from_stream`` and ``read_hex_string_from_stream``) and each calls
    ``create_string_object`` separately. A pypdf change that made ``"bytes"`` advisory would
    fail this test rather than quietly re-key the corpus.
    """
    require_pypdf()
    from pypdf.generic import ByteStringObject, ContentStream, DecodedStreamObject, TextStringObject

    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf (Hi) Tj <4869> Tj [(ab) -200 (cd)] TJ ET")

    forced = [operand for operands, _op in ContentStream(stream, None, "bytes").operations for operand in operands]
    assert [type(o).__name__ for o in forced if isinstance(o, bytes)] == [
        "ByteStringObject",
        "ByteStringObject",
    ]
    array = next(operand for operand in forced if isinstance(operand, list))
    assert [type(element).__name__ for element in array if not isinstance(element, int | float)] == [
        "ByteStringObject",
        "ByteStringObject",
    ]
    assert issubclass(ByteStringObject, bytes)

    # And the inverse, which is what the contract protects against: the DEFAULT is decoded.
    default = [operand for operands, _op in ContentStream(stream, None, None).operations for operand in operands]
    assert any(isinstance(operand, TextStringObject) for operand in default)


def test_a_user_unit_of_one_is_not_refused() -> None:
    """On the VALUE, not on the key. ``/UserUnit 1`` is the default and rescales nothing,
    so refusing its presence would fail a page for stating explicitly what every other
    page says by omission -- the same false positive shape as refusing ``/Rotate 0``."""
    require_pypdf()
    extraction = extract_fragments(build_page("BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET", page_extra="/UserUnit 1"))
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_an_alpha_outside_zero_to_one_is_refused() -> None:
    """ISO 32000-1 table 58 makes a constant alpha a number in [0, 1]. Outside it the file
    says something no renderer agrees on, and the visibility test would read ``/ca 2`` as
    "opaque, carry on"."""
    require_pypdf()
    for value in ("2", "-1"):
        reason = _refusal(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources=(f"/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /ca {value} >> >>"),
        )
        assert "outside the [0, 1] range" in reason, value


def test_an_undefined_rendering_mode_is_refused() -> None:
    """Table 106 defines exactly eight modes, as integers. ``3.5 Tr`` is not a shade
    between invisible and clip -- and both visibility tests would have read it as ordinary
    visible text and carried on."""
    require_pypdf()
    for mode in ("3.5", "8", "-1"):
        reason = _refusal(f"BT /F1 10 Tf {mode} Tr 1 0 0 1 100 700 Tm (012) Tj ET")
        assert "not one of the eight defined modes" in reason, mode


def test_an_indirect_smask_of_none_is_not_refused() -> None:
    """``/SMask`` may be an indirect reference, and ``str()`` on one renders
    ``IndirectObject(...)`` -- which is not ``/None``, so a mask being turned OFF through a
    reference would have refused the page."""
    require_pypdf()
    extraction = extract_fragments(
        build_page(
            "BT /F1 10 Tf /GS1 gs 1 0 0 1 100 700 Tm (012) Tj ET",
            resources="/Font << /F1 4 0 R >> /ExtGState << /GS1 << /Type /ExtGState /SMask 6 0 R >> >>",
            extra_objects=[("/None", None)],
        )
    )
    assert not extraction.page_failures
    assert [f.text for f in extraction.fragments] == ["012"]


def test_optional_content_is_refused_rather_than_read_as_evidence() -> None:
    """``/OC ... BDC`` binds its contents to a layer that may be switched off: text in the
    file, positioned exactly as this module would report it, that no reader ever sees.
    Whether the layer is on lives in the catalog and in the viewer's own state, so the
    operator alone cannot settle it."""
    require_pypdf()
    reason = _refusal(
        "/OC /MC0 BDC BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET EMC",
        resources="/Font << /F1 4 0 R >> /Properties << /MC0 6 0 R >>",
        extra_objects=[("<< /Type /OCG /Name (hidden) >>", None)],
    )
    assert "optional-content" in reason


def test_an_ordinary_structure_tag_is_not_refused() -> None:
    """The corpus carries 606 ``BDC`` operators on 62 of 75 pages -- ``/Figure``, ``/P``,
    ``/Caption``, ``/Artifact`` -- and not one ``/OC``. A guard on ``BDC`` itself would
    have failed 62 pages; a guard on the tag costs nothing."""
    require_pypdf()
    for tag in ("/Figure", "/P", "/Artifact", "/Caption"):
        extraction = extract_fragments(
            build_page(f"{tag} << /MCID 0 >> BDC BT /F1 10 Tf 1 0 0 1 100 700 Tm (012) Tj ET EMC")
        )
        assert not extraction.page_failures, tag
        assert [f.text for f in extraction.fragments] == ["012"], tag
