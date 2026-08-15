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
    HALTS_EVALUATION,
    ClaimedRegion,
    RegionRefusalReason,
    refuse_region,
    region_refusals,
)
from carmel.services.pdf_fragments import (
    _ENGINE_RAN,
    FragmentAvailability,
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

    def test_a_sign_set_as_a_superscript_is_still_seen(self) -> None:
        """`1.0` with its exponent's minus raised ~3 pt. At the old 0.75 pt band this
        was out of scope and returned None -- a factor-of-a-thousand error declared a
        documented limitation. The band reaches the superscript zone precisely so this
        is inspected rather than excused."""
        number = _fragment("1.0", 103.5, 115.0, baseline_y=700.0)
        raised = _fragment("−3", 116.0, 122.0, baseline_y=703.0)
        refusal = refuse_region(_extraction(number, raised), _region(number))
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
        from 3.50% to 22.5% of proposals. Bare `10` is here too: it is half of every
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


class TestTheClaimIsCheckedNotTaken:
    """Every field of `ClaimedRegion` is caller-controlled, so each is a way back in.

    These are the round-104 findings, and three of them are the ORIGINAL exclusion
    attack wearing a different hat: the producer still ends up holding `1.0` with no
    complaint, having moved the band, forged the member, or *included* the sign rather
    than omitted it.
    """

    def test_a_baseline_moved_off_the_members_refuses(self) -> None:
        """Shift the claimed baseline and the band stops containing the sign.

        With the band centred on a number the producer supplies, a producer that wants
        a clean answer just supplies a different number.
        """
        sign = _fragment("/C0", 96.5, 100.0, baseline_y=700.0)
        number = _fragment("1.0", 103.5, 115.0, baseline_y=700.0)
        moved = ClaimedRegion(
            page=1,
            x_start=103.5,
            x_end=115.0,
            baseline_y=700.0 + BASELINE_BAND + 0.5,
            members=(number,),
        )
        refusal = refuse_region(_extraction(sign, number), moved)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.MEMBER_OUTSIDE_REGION

    def test_a_baseline_nudged_within_the_band_still_sees_the_sign(self) -> None:
        """A shift too small to detach the member from the box does not detach the band
        from the sign either. The two checks overlap on purpose: whichever gap the
        producer tries to slip through, the other one still refuses."""
        sign = _fragment("/C0", 96.5, 100.0, baseline_y=700.0)
        number = _fragment("1.0", 103.5, 115.0, baseline_y=700.0)
        nudged = ClaimedRegion(page=1, x_start=103.5, x_end=115.0, baseline_y=700.5, members=(number,))
        refusal = refuse_region(_extraction(sign, number), nudged)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE

    def test_a_member_that_is_not_in_the_extraction_refuses(self) -> None:
        """A forged fragment has no page, no neighbours and no provenance. Without this
        check an empty extraction can be handed fabricated members and come back clean.
        """
        real = _fragment("1.0", 103.5, 115.0)
        forged = _fragment("9.9", 103.5, 115.0)
        refusal = refuse_region(_extraction(real), _region(forged))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.FOREIGN_MEMBER

    def test_naming_the_sign_as_a_member_does_not_silence_the_refusal(self) -> None:
        """The exclusion attack INVERTED, and the sharpest of the round-104 findings.

        The producer claims the box around `1.0` but lists the detached `/C0` as a
        member. That removes it from the outsiders, so the adjacency check never sees
        it -- and the producer still records `1.0`, sign lost, with nothing raised.
        """
        sign = _fragment("/C0", 96.5, 100.0)
        number = _fragment("1.0", 103.5, 115.0)
        smuggled = ClaimedRegion(page=1, x_start=103.5, x_end=115.0, baseline_y=700.0, members=(number, sign))
        refusal = refuse_region(_extraction(sign, number), smuggled)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.MEMBER_OUTSIDE_REGION

    @pytest.mark.parametrize(
        ("x_start", "x_end"),
        [(float("nan"), 115.0), (103.5, float("nan")), (115.0, 103.5), (float("-inf"), 115.0)],
    )
    def test_a_box_that_is_not_a_box_refuses(self, x_start: float, x_end: float) -> None:
        """NaN makes every `<` in this module quietly False, so an unchecked NaN box
        satisfies the straddle test, finds no neighbours, and returns `None` -- the most
        permissive answer -- without one predicate having actually run.
        """
        number = _fragment("1.0", 103.5, 115.0)
        broken = ClaimedRegion(page=1, x_start=x_start, x_end=x_end, baseline_y=700.0, members=(number,))
        refusal = refuse_region(_extraction(number), broken)
        assert refusal is not None
        assert refusal.reason in {
            RegionRefusalReason.INVALID_GEOMETRY,
            RegionRefusalReason.MEMBER_OUTSIDE_REGION,
        }

    def test_a_fragment_whose_extent_runs_backwards_refuses(self) -> None:
        """Mirrored or negative-scale text yields `x_end < x_start` while `rotated` is
        still False, inverting left, right and overlap at once."""
        mirrored = _fragment("1.0", 115.0, 103.5)
        region = ClaimedRegion(page=1, x_start=103.5, x_end=115.0, baseline_y=700.0, members=(mirrored,))
        refusal = refuse_region(_extraction(mirrored), region)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.INVALID_GEOMETRY

    def test_a_zero_width_claimed_box_refuses(self) -> None:
        """A region of zero width contains nothing and can straddle nothing, so it would
        reach the neighbour search and return `None` -- the same permissive non-answer the
        NaN check exists to stop. The refusal text said "non-increasing" while the
        predicate accepted equality, so the gate was looser than its own message.
        """
        number = _fragment("1.0", 103.5, 115.0)
        flat = ClaimedRegion(page=1, x_start=109.0, x_end=109.0, baseline_y=700.0, members=(number,))
        refusal = refuse_region(_extraction(number), flat)
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.INVALID_GEOMETRY

    def test_a_zero_width_member_fragment_does_not_refuse(self) -> None:
        """The other half of that fix, and the half a later reader would "simplify" away.

        A combining diacritic draws over the glyph before it and does not advance the pen,
        so `x_start == x_end` is what a CORRECT extractor reports. On the eight-paper
        corpus 23 such fragments are `MAPPED` -- real accents on real words, in 7 of the 8
        papers. Tightening the fragment predicate to `<` alongside the region one would
        refuse every region containing an accented character.
        """
        accent = _fragment("´", 109.0, 109.0)
        word = _fragment("1.0", 103.5, 115.0)
        region = ClaimedRegion(page=1, x_start=100.0, x_end=120.0, baseline_y=700.0, members=(word, accent))
        refusal = refuse_region(_extraction(word, accent), region)
        assert refusal is None or refusal.reason is not RegionRefusalReason.INVALID_GEOMETRY

    def test_a_non_member_with_a_nan_extent_refuses(self) -> None:
        """The member check was never the whole rule. A NaN extent makes every `<` in the
        straddle and neighbour tests quietly False, so an outsider carrying one is neither
        recorded as a straddler nor found as the nearest neighbour: the region reads clean
        because the checks that would have refused it could not run.

        This one sits INSIDE the band and overlaps the region, so without the guard it is
        a straddler that goes unreported.
        """
        number = _fragment("1.0", 103.5, 115.0)
        broken = _fragment("-", float("nan"), 103.0)
        refusal = refuse_region(_extraction(number, broken), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR

    def test_a_non_member_with_a_nan_baseline_refuses(self) -> None:
        """The other half, and the reason the check is scoped to the PAGE.

        A NaN BASELINE fails the band test itself -- `abs(nan - y) <= BAND` is False -- so
        this fragment never enters the band at all. A band-scoped guard would inspect only
        the fragments that already passed the very comparison NaN defeats, and would
        report this page as clean.
        """
        number = _fragment("1.0", 103.5, 115.0)
        broken = _fragment("-", 96.0, 103.0, baseline_y=float("nan"))
        refusal = refuse_region(_extraction(number, broken), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR

    def test_a_zero_width_non_member_does_not_refuse(self) -> None:
        """The exemption, carried across from the member rule so the two cannot drift.

        A combining diacritic outside the region is still a legitimate zero-width
        fragment, and `_is_sane_extent` admits it deliberately. Refusing here would refuse
        every region on a page that contains an accented character anywhere.
        """
        number = _fragment("1.0", 103.5, 115.0)
        accent = _fragment("´", 96.0, 96.0)
        refusal = refuse_region(_extraction(number, accent), _region(number))
        assert refusal is None or refusal.reason is not RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR

    def test_a_blank_non_member_with_a_nan_extent_does_not_refuse(self) -> None:
        """Blank fragments are excluded from the guard for the same reason the neighbour
        scan already skips them: their geometry is never read, so it cannot mislead. A bare
        space carrying a NaN would otherwise refuse a page that is in no way compromised.
        """
        number = _fragment("1.0", 103.5, 115.0)
        blank = _fragment(" ", float("nan"), float("nan"))
        refusal = refuse_region(_extraction(number, blank), _region(number))
        assert refusal is None or refusal.reason is not RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR

    def test_loss_that_cannot_be_located_refuses(self) -> None:
        """`lossy` with neither carrier is loss this module cannot place. The page-
        specific test stays precise, so this needs its own check."""
        number = _fragment("1.0", 103.5, 115.0)
        extraction = _extraction(number, lossy=True)
        refusal = refuse_region(extraction, _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.PAGE_INCOMPLETE


class TestAffixesAttachedToOtherText:
    """Real typesetting does not leave affixes alone in their own fragment."""

    def test_a_value_split_across_its_decimal_point_refuses(self) -> None:
        """`3` `.14`, touching exactly. Not overlap, so STRADDLED cannot fire, and the
        edge token is `.14` rather than `.`, so an exact-match classifier misses it --
        and `3.14` is silently recorded as `3`.
        """
        three = _fragment("3", 103.5, 110.0)
        rest = _fragment(".14", 110.0, 120.0)
        refusal = refuse_region(_extraction(three, rest), _region(three))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE
        assert refusal.affix is AffixClass.DECIMAL

    @pytest.mark.parametrize("exponent", ["×10", "·10", "×10−3"])
    def test_an_unspaced_exponent_refuses(self, exponent: str) -> None:
        """Ordinary printed scientific notation, not adversarial input. Reading it as a
        neighbouring cell drops a factor of a thousand."""
        number = _fragment("1.0", 103.5, 115.0)
        right = _fragment(exponent, 118.0, 135.0)
        refusal = refuse_region(_extraction(number, right), _region(number))
        assert refusal is not None
        assert refusal.affix is AffixClass.EXPONENT

    @pytest.mark.parametrize("left_text", ["Fig.", "et al.", "conditions.", "sample,"])
    def test_an_abbreviation_period_is_not_a_decimal_point(self, left_text: str) -> None:
        """A decimal point sits BETWEEN digits; a period ending `Fig.` is English.

        Without the digit-adjacency requirement this class alone accounts for 491
        left-hand refusals over the corpus instead of 200, and none of the extra 291 is
        a number split across its point -- the module would be refusing on abbreviation
        style rather than on numeral semantics.
        """
        left = _fragment(left_text, 60.0, 100.0)
        number = _fragment("3", 103.5, 110.0)
        assert refuse_region(_extraction(left, number), _region(number)) is None

    def test_a_digit_before_the_period_refuses_even_when_it_reads_as_prose(self) -> None:
        """`Table 2.` is a caption, but at the abutting edge it is digit-then-point --
        character-identical to the `3.` of a split `3.14`. Geometry cannot separate
        them, so this fails toward refusal rather than guessing which one it is."""
        left = _fragment("Table 2.", 60.0, 100.0)
        number = _fragment("3", 103.5, 110.0)
        refusal = refuse_region(_extraction(left, number), _region(number))
        assert refusal is not None
        assert refusal.affix is AffixClass.DECIMAL

    def test_a_number_split_across_its_point_still_refuses_from_the_left(self) -> None:
        """The mirror of the case above: `3.` then `14` really is one value cut in two,
        and the digit before the point is what distinguishes it from `Fig.`."""
        left = _fragment("3.", 90.0, 103.5)
        number = _fragment("14", 103.5, 115.0)
        refusal = refuse_region(_extraction(left, number), _region(number))
        assert refusal is not None
        assert refusal.affix is AffixClass.DECIMAL

    def test_a_leading_sign_on_the_right_is_that_cells_own_sign(self) -> None:
        """Direction matters. A sign binds to the number on ITS right, so `-2.5` to our
        right is a negative neighbouring cell, not our affix. Refusing here would reject
        every row containing a negative number.
        """
        number = _fragment("1.0", 103.5, 115.0)
        right = _fragment("-2.5", 130.0, 145.0)
        assert refuse_region(_extraction(number, right), _region(number)) is None

    def test_a_trailing_sign_on_the_left_is_ours(self) -> None:
        """The mirror image: a sign at the END of the left neighbour reaches toward us."""
        left = _fragment("k =-", 60.0, 100.0)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(left, number), _region(number))
        assert refusal is not None
        assert refusal.affix is AffixClass.SIGN

    def test_an_unmapped_neighbour_refuses_even_when_no_token_matches(self) -> None:
        """`n=/C0` is UNMAPPED, but its edge token matches no affix. Refusing on the
        FLAG rather than the token stops the check depending on whether the marker
        happened to land alone in its own fragment."""
        left = _fragment("n=/C0", 60.0, 100.0, mapping=GlyphMapping.UNMAPPED)
        number = _fragment("1.0", 103.5, 115.0)
        refusal = refuse_region(_extraction(left, number), _region(number))
        assert refusal is not None
        assert refusal.reason is RegionRefusalReason.ADJACENT_UNREADABLE


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

    def test_every_unavailable_state_refuses_with_the_same_reason(self) -> None:
        """One region-level reason for all four, and the region is refused under each.

        The states differ in WHOSE FAULT the missing evidence is, which is a property
        of the document and identical for every region in it. Splitting the per-region
        enum along a per-document axis would grow a corpus tally without telling it
        anything about a region -- so the distinction stays on the extraction, and this
        test is what stops a later reader from putting it here.
        """
        number = _fragment("1.0", 103.5, 115.0)
        for status in FragmentAvailability:
            if status is FragmentAvailability.AVAILABLE:
                continue
            # The REGION still claims this fragment as a member while the EXTRACTION
            # carries nothing -- which is the interesting shape, not a workaround for
            # the invariant that forbids the other one. It is precisely a producer
            # asserting members that no available extraction backs.
            extraction = FragmentExtraction(
                status=status,
                lossy=True,
                pypdf_version="6.14.2" if status in _ENGINE_RAN else "",
            )
            refusal = refuse_region(extraction, _region(number))
            assert refusal is not None
            assert refusal.reason is RegionRefusalReason.EXTRACTION_UNAVAILABLE
            # Named for a human reading one refusal. NOT a carrier: anything needing
            # the distinction structurally reads `extraction.status`, which it holds.
            assert status.value in refusal.detail

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


class TestTheCensusIsNotBiasedByCheckOrder:
    """`refuse_region` answers a yes/no question and one reason ends it. Counting those
    reasons across a corpus measures the order of the checks instead of the corpus, so
    `region_refusals` reports every reason it could actually evaluate."""

    def test_a_straddler_no_longer_hides_the_detached_sign(self) -> None:
        """The pair that motivated this: both are real findings about the same region,
        and under first-wins the sign -- the dangerous one -- was the one discarded."""
        sign = _fragment("-", 95.0, 98.0)
        number = _fragment("1.0", 100.0, 110.0)
        straddler = _fragment("14", 105.0, 120.0)
        reasons = [refusal.reason for refusal in region_refusals(_extraction(sign, number, straddler), _region(number))]
        assert reasons == [
            RegionRefusalReason.STRADDLED,
            RegionRefusalReason.ADJACENT_UNREADABLE,
        ]

    def test_a_member_level_finding_no_longer_hides_a_band_level_one(self) -> None:
        sign = _fragment("-", 95.0, 98.0)
        number = _fragment("1.0", 100.0, 110.0, mapping=GlyphMapping.UNMAPPED)
        reasons = [refusal.reason for refusal in region_refusals(_extraction(sign, number), _region(number))]
        assert reasons == [
            RegionRefusalReason.UNMAPPED_MEMBER,
            RegionRefusalReason.ADJACENT_UNREADABLE,
        ]

    def test_a_rotated_member_is_recorded_but_not_also_placed(self) -> None:
        """Its extent is not a horizontal one, so declaring it outside the box would
        manufacture a second finding out of the first one's uncertainty."""
        number = _fragment("1.0", 400.0, 410.0, rotated=True)
        region = ClaimedRegion(page=1, x_start=100.0, x_end=110.0, baseline_y=700.0, members=(number,))
        reasons = [refusal.reason for refusal in region_refusals(_extraction(number), region)]
        assert reasons == [RegionRefusalReason.ROTATED_MEMBER]

    def test_a_reason_is_counted_once_per_region_however_many_fragments_earn_it(self) -> None:
        first = _fragment("1", 100.0, 105.0, mapping=GlyphMapping.UNMAPPED)
        second = _fragment("0", 105.0, 110.0, mapping=GlyphMapping.UNMAPPED)
        reasons = [refusal.reason for refusal in region_refusals(_extraction(first, second), _region(first, second))]
        assert reasons == [RegionRefusalReason.UNMAPPED_MEMBER]

    def test_a_halting_reason_truncates_the_census_and_says_so(self) -> None:
        """A forged member and a detached sign. Only the forgery is reported, because
        the members tuple no longer partitions the page and the outsider set every band
        check reads is computed from it."""
        sign = _fragment("-", 95.0, 98.0)
        real = _fragment("1.0", 100.0, 110.0)
        forged = _fragment("1.0", 100.0, 110.0)
        refusals = region_refusals(_extraction(sign, real), _region(forged))
        assert [refusal.reason for refusal in refusals] == [RegionRefusalReason.FOREIGN_MEMBER]
        assert refusals[-1].reason in HALTS_EVALUATION

    def test_a_halting_reason_is_only_ever_the_last_element(self) -> None:
        """The invariant that makes the result readable: everything before the end was
        followed by checks that ran, so its absence from the tail is informative."""
        sign = _fragment("-", 95.0, 98.0)
        number = _fragment("1.0", 100.0, 110.0)
        unmapped = _fragment("1.0", 100.0, 110.0, mapping=GlyphMapping.UNMAPPED)
        rotated_far = _fragment("x", 200.0, 210.0, rotated=True)
        shapes = [
            (_extraction(sign, number), _region(number)),
            (_extraction(sign, unmapped), _region(unmapped)),
            (_extraction(sign, number, _fragment("14", 105.0, 120.0)), _region(number)),
            (_extraction(sign, unmapped, rotated_far), _region(unmapped)),
            (_extraction(number, lossy=True), _region(number)),
            (_extraction(number), _region(number)),
        ]
        for extraction, region in shapes:
            refusals = region_refusals(extraction, region)
            for refusal in refusals[:-1]:
                assert refusal.reason not in HALTS_EVALUATION

    def test_the_yes_no_answer_is_the_first_element_of_the_census(self) -> None:
        """Two entry points, one set of checks -- so a consumer measuring this layer
        measures the layer that runs, not a second implementation that drifts."""
        sign = _fragment("-", 95.0, 98.0)
        number = _fragment("1.0", 100.0, 110.0)
        unmapped = _fragment("1.0", 100.0, 110.0, mapping=GlyphMapping.UNMAPPED)
        for extraction, region in (
            # Multi-reason shapes first: on a one-reason region every choice of index
            # agrees, so a single-reason case asserts nothing about WHICH one is taken.
            (_extraction(sign, number, _fragment("14", 105.0, 120.0)), _region(number)),
            (_extraction(sign, unmapped), _region(unmapped)),
            (_extraction(sign, number), _region(number)),
            (_extraction(number), _region(number)),
            (_extraction(number, lossy=True), _region(number)),
        ):
            refusals = region_refusals(extraction, region)
            refusal = refuse_region(extraction, region)
            assert refusal == (refusals[0] if refusals else None)

    def test_every_reason_is_triaged_as_halting_or_independent(self) -> None:
        """A new reason must be classified deliberately: `HALTS_EVALUATION` is a claim
        about what the check destroyed, and defaulting it either way is a guess."""
        independent = {
            RegionRefusalReason.UNMAPPED_MEMBER,
            RegionRefusalReason.ROTATED_MEMBER,
            RegionRefusalReason.STRADDLED,
            RegionRefusalReason.ADJACENT_UNREADABLE,
        }
        assert HALTS_EVALUATION | independent == set(RegionRefusalReason)
        assert not HALTS_EVALUATION & independent
