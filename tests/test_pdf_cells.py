"""The refusal layer: every predicate must fire on a real shape, and only on it.

These build :class:`TextFragment` objects directly rather than going through a PDF.
That is not a shortcut -- the layer is pure geometry over the fragment dataclass, so a
PDF would add a pypdf dependency (and a skip in the base CI job) without testing one
extra line. The shapes below are transcribed from the real corpus; the text is not.
"""

from __future__ import annotations

import pytest

from carmel.services.numeric import AffixClass
from carmel.services.pdf_cells import (
    BASELINE_BAND,
    ClaimedRegion,
    RegionRefusalReason,
    refuse_region,
)
from carmel.services.pdf_fragments import (
    FragmentExtraction,
    FragmentPageFailure,
    GlyphMapping,
    TextFragment,
)


def _fragment(
    text: str,
    x_start: float,
    x_end: float,
    *,
    baseline_y: float = 700.0,
    page: int = 1,
    rotated: bool = False,
    mapping: GlyphMapping = GlyphMapping.MAPPED,
) -> TextFragment:
    return TextFragment(
        page=page,
        text=text,
        x_start=x_start,
        x_end=x_end,
        baseline_y=baseline_y,
        font_height=9.0,
        rotated=rotated,
        glyph_mapping=mapping,
    )


def _extraction(*fragments: TextFragment, **kwargs: object) -> FragmentExtraction:
    return FragmentExtraction(fragments=fragments, pypdf_version="6.14.2", **kwargs)  # type: ignore[arg-type]


def _region(*members: TextFragment, page: int = 1) -> ClaimedRegion:
    return ClaimedRegion(
        page=page,
        x_start=min(member.x_start for member in members),
        x_end=max(member.x_end for member in members),
        baseline_y=members[0].baseline_y,
        members=members,
    )


class TestTheMotivatingCase:
    """The detached sign. Everything else in this module exists to support this."""

    def test_a_detached_unmapped_minus_refuses_the_number_beside_it(self) -> None:
        """`/C0` `1.0` -- the shape the real corpus holds 50 times.

        The producer draws its box tightly around `1.0`, so the sign is not a member,
        nothing straddles, and a members-only gate sees a perfect number. Recording it
        would invert the sign silently. It must be refused.
        """
        sign = _fragment("/C0", 96.5, 100.0)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(sign, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE
        assert refusal.affix is AffixClass.SIGN

    def test_a_blank_fragment_does_not_shield_the_sign(self) -> None:
        """`/C0` `' '` `1.0`: 2.76% of corpus proposals have a bare blank as their
        nearest left neighbour, and behind 20 of those sits a real affix. Taking the
        nearest fragment LITERALLY would let a single space defeat the whole layer.
        """
        sign = _fragment("/C0", 90.0, 93.5)
        blank = _fragment(" ", 94.0, 96.0)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(sign, blank, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE

    def test_distance_does_not_rescue_the_sign(self) -> None:
        """No distance cutoff, deliberately: the sign-to-digit gap (3.50 pt) sits BELOW
        the median genuine column gap (3.97 pt), so any cutoff that admits this number
        also merges most real columns. There is nothing to tune, so nothing is tuned.
        """
        sign = _fragment("−", 40.0, 44.0)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(sign, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE


class TestWhatMustNotRefuse:
    """The explicit affix list is a promise in both directions."""

    @pytest.mark.parametrize(
        "neighbour",
        ["[23]", "a", "b", "*", "†", "run", "10", "298", "e", "x", "X", "E"],
    )
    def test_a_legitimate_standalone_cell_does_not_refuse(self, neighbour: str) -> None:
        """Reference brackets, footnote letters and round numbers are real table cells.

        Refusing on "any token that cannot stand alone" was MEASURED and takes the cost
        from 3.10% to 22.5% of proposals. Bare `10` is here too: it is half of every
        `x 10^n` construct, but on its own it is an ordinary value, and the true
        multiplication glyph that makes it an exponent is itself an affix.

        The ASCII operator lookalikes are the sharp ones, and they are here because the
        corpus says so rather than because they seem harmless. As standalone tokens `e`
        occurs 1781 times, `x` 115, `X` 52 and `*` 9 -- word-split fragments out of
        "experimental" and running prose, the mole-fraction LABEL, and
        "*Corresponding author". Classing them as multiplication would refuse a large
        slice of every table for no gain.
        """
        left = _fragment(neighbour, 80.0, 95.0)
        number = _fragment("1.0", 103.5, 115.0)
        assert refuse_region(_extraction(left, number), _region(number)) is None

    def test_a_clean_row_of_numbers_does_not_refuse(self) -> None:
        row = [
            _fragment("298", 100.0, 115.0),
            _fragment("1.0", 150.0, 165.0),
            _fragment("2.5", 200.0, 215.0),
        ]
        for number in row:
            assert refuse_region(_extraction(*row), _region(number)) is None

    def test_an_affix_on_the_line_above_does_not_refuse(self) -> None:
        """The band is what keeps the row above out. A minus sign one line up belongs
        to a different number entirely, and refusing on it would make every table
        unreadable."""
        above = _fragment("−", 96.5, 100.0, baseline_y=700.0 + BASELINE_BAND + 0.5)
        number = _fragment("1.0", 103.5, 115.0, baseline_y=700.0)
        assert refuse_region(_extraction(above, number), _region(number)) is None

    def test_an_affix_on_another_page_does_not_refuse(self) -> None:
        elsewhere = _fragment("−", 96.5, 100.0, page=2)
        number = _fragment("1.0", 103.5, 115.0, page=1)
        assert refuse_region(_extraction(elsewhere, number), _region(number)) is None


class TestEdgeTokenClassification:
    """A fragment is a text-show operation, not a token."""

    def test_a_trailing_affix_inside_a_longer_fragment_refuses(self) -> None:
        """`'Ref. [23] -'` ends in a dangerous affix while the whole string is not one.
        Classifying the whole fragment would miss it."""
        left = _fragment("Ref. [23] -", 60.0, 100.0)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(left, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE

    def test_an_affix_buried_mid_fragment_does_not_refuse(self) -> None:
        """`'a-b'` is one token containing a hyphen, not an affix abutting the number.
        Refusing here is the over-refusal direction of the same mistake."""
        left = _fragment("Fig. 4a-b shows", 40.0, 100.0)
        number = _fragment("1.0", 103.5, 115.0)
        assert refuse_region(_extraction(left, number), _region(number)) is None

    def test_the_right_hand_neighbour_is_checked_too(self) -> None:
        """`1.0` `× 10` -- the exponent construct arrives from the right.

        The TRUE multiplication glyph, not its ASCII lookalike: `×` never appears as a
        standalone corpus token, so guarding it costs nothing, while `x` appears 115
        times and is almost always a word split.
        """
        number = _fragment("1.0", 103.5, 115.0)
        right = _fragment("× 10", 118.0, 135.0)
        refusal = refuse_region(_extraction(number, right), _region(number))
        assert refusal is not None
        assert refusal.affix is AffixClass.EXPONENT


class TestTheRegionItself:
    def test_an_empty_region_refuses(self) -> None:
        number = _fragment("1.0", 103.5, 115.0)
        empty = ClaimedRegion(page=1, x_start=300.0, x_end=320.0, baseline_y=700.0, members=())
        refusal = refuse_region(_extraction(number), empty)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.EMPTY

    def test_an_unmapped_member_refuses(self) -> None:
        number = _fragment("1.�", 103.5, 115.0, mapping=GlyphMapping.UNMAPPED)
        refusal = refuse_region(_extraction(number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.UNMAPPED_MEMBER

    def test_a_rotated_member_refuses(self) -> None:
        number = _fragment("1.0", 103.5, 115.0, rotated=True)
        refusal = refuse_region(_extraction(number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ROTATED_MEMBER

    def test_a_rotated_neighbour_refuses(self) -> None:
        """74 of 5332 corpus proposals have one. Its x-extent is not a horizontal
        extent, so neither including nor excluding it can be justified."""
        rotated = _fragment("axis label", 40.0, 90.0, rotated=True)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(rotated, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ROTATED_NEIGHBOUR

    def test_a_straddling_non_member_refuses(self) -> None:
        """A box drawn around the `3` of `3.14` cuts through the `.14` beside it. The
        boundary is not clean, so the region is not evidence of anything."""
        three = _fragment("3", 103.5, 110.0)
        rest = _fragment(".14", 108.0, 120.0)
        refusal = refuse_region(_extraction(three, rest), _region(three))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.STRADDLED


class TestPartialEvidenceFailsTowardRefusal:
    """The reason `refuse_region` takes the whole extraction, not one page."""

    def test_an_unavailable_extraction_refuses(self) -> None:
        number = _fragment("1.0", 103.5, 115.0)
        extraction = FragmentExtraction(fragments=(number,), available=False, lossy=True)
        refusal = refuse_region(extraction, _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.EXTRACTION_UNAVAILABLE

    def test_a_failure_on_this_page_refuses(self) -> None:
        """The dangerous neighbour that would have refused this region may simply never
        have been extracted. A clean result would be an artefact of the loss."""
        number = _fragment("1.0", 103.5, 115.0)
        extraction = _extraction(
            number,
            lossy=True,
            page_failures=(FragmentPageFailure(page=1, error="unreadable"),),
        )
        refusal = refuse_region(extraction, _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.PAGE_INCOMPLETE

    def test_a_failure_on_a_different_page_does_not_refuse(self) -> None:
        """Page-level loss is recorded per page precisely so this stays answerable. A
        bare `lossy` flag would have to refuse the whole document."""
        number = _fragment("1.0", 103.5, 115.0, page=1)
        extraction = _extraction(
            number,
            lossy=True,
            page_failures=(FragmentPageFailure(page=7, error="unreadable"),),
        )
        assert refuse_region(extraction, _region(number)) is None

    def test_a_truncated_document_refuses(self) -> None:
        number = _fragment("1.0", 103.5, 115.0)
        extraction = _extraction(number, lossy=True, truncated=True)
        refusal = refuse_region(extraction, _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.PAGE_INCOMPLETE


class TestNoApprovalAffordanceExists:
    """`None` will be read as approval no matter what the docstring says, so the
    module must offer nothing that could be persisted as one."""

    def test_the_module_exposes_no_positive_artifact(self) -> None:
        import carmel.services.pdf_cells as module

        forbidden = ("accept", "approve", "clean", "not_refused", "ok", "pass", "valid", "verif")
        offenders = [
            name for name in dir(module) if not name.startswith("_") and any(word in name.lower() for word in forbidden)
        ]
        assert offenders == []

    def test_every_refusal_reason_is_a_refusal(self) -> None:
        """No member may name a non-refusing outcome. A single `NOT_REFUSED` here would
        rebuild the laundering path `RefutationStatus` had to be walled off from."""
        for reason in RegionRefusalReason:
            assert "not_" not in reason.value
            assert reason.value not in {"none", "ok", "clean", "accepted"}
