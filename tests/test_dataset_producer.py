"""Tests for ``carmel.services.dataset_producer`` -- the M-D2d vertical slice.

One REAL stored evidence artifact (stored through the real
``evidence.store_artifact`` API, never a hand-built directory) is turned into
a validated ``DatasetEnvelope`` whose every ``CharSpanLocator`` was produced
by ``ground_quote`` SEARCHING the artifact's verified extracted text, then
round-tripped through the real content-addressed dataset store, and finally
proven correct by an independent replayer-style check that re-reads and
re-verifies ``extracted.json`` from disk and re-slices every span.

All text here is SYNTHETIC -- invented sentences and numbers, never real
paper text (the project's corpus is closed-access, non-redistributable).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AxisRole,
    ExtractionBinding,
    SourceNodeKind,
    ValueOrigin,
)
from carmel.schemas.literature import StoredArtifact
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_producer import (
    DatasetProducerError,
    MeasurementSpec,
    QuoteGroundingError,
    ground_quote,
    produce_envelope_from_artifact,
)
from carmel.services.dataset_replay import ReplayOutcome, replay_envelope
from carmel.services.dataset_store import compute_dataset_sha
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.numeric import QuoteRole
from carmel.services.units import QuantityKind

MAX_BYTES = 10_000_000

_TEXT = (
    "The reactor was held at a temperature of 1023 K while the measured "
    "mole fraction (-) of the fuel species was 0.0123 at steady state."
)
"""Synthetic source sentence. Every grounded quote below ("temperature",
"1023", "K", "mole fraction", "-", "0.0123") appears exactly once in it, so
`ground_quote`'s uniqueness default applies throughout."""

_MUTATED_TEXT = _TEXT.replace("1023 K", "1024 K")
"""Differs from `_TEXT` by exactly ONE character ('3' -> '4'), at the same
position, INSIDE the grounded value quote "1023"."""

_SPECS = (
    MeasurementSpec(
        axis_id="temperature",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.TEMPERATURE,
        label_quote="temperature",
        value_quote="1023",
        unit_quote="K",
    ),
    MeasurementSpec(
        axis_id="mole_fraction",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.MOLE_FRACTION,
        label_quote="mole fraction",
        value_quote="0.0123",
        unit_quote="-",
    ),
)


def _store_synthetic_artifact(
    workspace_root: Path,
    text: str,
    *,
    content_type: str = "application/pdf",
    extractor: str = "pdf:pypdf",
    lossy: bool = False,
) -> StoredArtifact:
    """Store a synthetic artifact through the REAL evidence.store_artifact API."""
    data = text.encode("utf-8")
    artifact = FetchedArtifact(
        url="https://example.org/synthetic.pdf",
        final_url="https://example.org/synthetic.pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        content_type=content_type,
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    extracted = ExtractedText(
        text=text, normalized=text.casefold(), sections=[], extractor=extractor, lossy=lossy
    )
    return store_artifact(workspace_root, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)




class TestMeasurementSpecOccurrence:
    """P2-E: ``bool`` is a subclass of ``int`` in Python, so an unvalidated
    ``occurrence: int | None`` field would silently accept ``True``/``False``
    as occurrence 1/0 -- a near-certain caller typo, never a real disambiguation
    intent. ``MeasurementSpec`` must reject bool explicitly, not merely
    ``isinstance(x, int)`` (which bool satisfies)."""

    @pytest.mark.parametrize("field", ["label_occurrence", "value_occurrence", "unit_occurrence"])
    @pytest.mark.parametrize("bad_value", [True, False])
    def test_bool_occurrence_rejected(self, field: str, bad_value: bool) -> None:
        kwargs = {
            "axis_id": "temperature",
            "role": AxisRole.COORDINATE,
            "quantity_kind": QuantityKind.TEMPERATURE,
            "label_quote": "temperature",
            "value_quote": "1023",
            "unit_quote": "K",
            field: bad_value,
        }
        with pytest.raises(DatasetProducerError, match=f"{field}={bad_value!r}"):
            MeasurementSpec(**kwargs)

    def test_int_occurrence_still_accepted(self) -> None:
        spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="1023",
            unit_quote="K",
            value_occurrence=0,
        )
        assert spec.value_occurrence == 0

    def test_none_occurrence_still_accepted(self) -> None:
        spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="1023",
            unit_quote="K",
        )
        assert spec.value_occurrence is None


class TestMeasurementSpecRole:
    """Round 43 finding: ``AxisRole`` is a ``StrEnum``, so a plain string
    equal to one of its members' VALUES (e.g. ``role="coordinate"``) compares
    ``==`` equal to ``AxisRole.COORDINATE`` but fails ``isinstance``/``is`` --
    the same trap this codebase already guards against for ``QuoteRole`` and
    ``QuantityKind`` elsewhere (see ``ground_quote``'s own ``isinstance``
    checks). ``__post_init__`` previously validated only the occurrence
    fields, so a bare string silently survived construction here."""

    def test_plain_string_role_rejected_even_though_value_equal(self) -> None:
        assert AxisRole.COORDINATE == "coordinate"  # StrEnum value equality holds
        with pytest.raises(DatasetProducerError, match="role='coordinate'"):
            MeasurementSpec(
                axis_id="temperature",
                role="coordinate",
                quantity_kind=QuantityKind.TEMPERATURE,
                label_quote="temperature",
                value_quote="1023",
                unit_quote="K",
            )

    def test_genuine_axis_role_member_still_accepted(self) -> None:
        spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="1023",
            unit_quote="K",
        )
        assert spec.role is AxisRole.COORDINATE


class TestGroundQuote:
    def test_grounds_unique_quote_by_search(self) -> None:
        locator = ground_quote(_TEXT, "1023", role=QuoteRole.VALUE)
        assert _TEXT[locator.start : locator.end] == "1023"
        # Derived by SEARCH, not asserted: the span really is where "1023"
        # sits in this sentence.
        assert locator.start == _TEXT.index("1023")

    def test_quote_absent_from_text_raises(self) -> None:
        with pytest.raises(QuoteGroundingError, match="not found"):
            ground_quote(_TEXT, "774 K", role=QuoteRole.VALUE)

    def test_ambiguous_quote_raises_and_states_match_count(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote(text, "1023", role=QuoteRole.VALUE)

    def test_empty_quote_raises_with_specific_message(self) -> None:
        with pytest.raises(QuoteGroundingError, match="quote is empty"):
            ground_quote(_TEXT, "", role=QuoteRole.VALUE)

    def test_occurrence_out_of_range_raises_and_states_match_count(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        with pytest.raises(QuoteGroundingError, match=r"2 match\(es\) found"):
            ground_quote(text, "1023", occurrence=2, role=QuoteRole.VALUE)

    def test_explicit_occurrence_selects_that_match(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        first = ground_quote(text, "1023", occurrence=0, role=QuoteRole.VALUE)
        second = ground_quote(text, "1023", occurrence=1, role=QuoteRole.VALUE)
        assert first.start == text.index("1023")
        assert second.start == text.index("1023", first.start + 1)
        assert text[second.start : second.end] == "1023"

    def test_overlapping_matches_count_as_ambiguous(self) -> None:
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote("aaa", "aa", role=QuoteRole.VALUE)


class TestGroundQuoteNumericTokenMaximality:
    """P1-A: a numeral quote must ground to the MAXIMAL numeric token, never
    an interior fragment of a strictly larger one."""

    def test_rejects_fragment_inside_larger_integer(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("T = 11023 K", "1023", role=QuoteRole.VALUE)

    def test_rejects_fragment_before_decimal_point(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("T = 1023.5 K", "1023", role=QuoteRole.VALUE)

    def test_rejects_fragment_after_decimal_point(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("phi 0.51023 was used", "1023", role=QuoteRole.VALUE)

    def test_accepts_whole_number_bounded_by_whitespace(self) -> None:
        locator = ground_quote("T = 1023 K", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 8)

    def test_accepts_number_immediately_followed_by_unit_letter(self) -> None:
        # Deliberate policy: a trailing letter that isn't part of an
        # exponent marker (e/E) does not continue a numeric token, so
        # "1023" IS the maximal numeral in "1023K" -- unlike a trailing
        # digit or decimal point, "K" cannot extend a numeral.
        locator = ground_quote("1023K", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (0, 4)

    def test_non_numeric_quote_is_unaffected_by_maximality_check(self) -> None:
        # "mole fraction" never matches the numeric-token pattern at all, so
        # the maximality check must not even engage for it.
        locator = ground_quote("the mole fraction was measured", "mole fraction", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 17)

    def test_rejects_fragment_of_exponent_form(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("k = 1023e5 1/s", "1023", role=QuoteRole.VALUE)


class TestGroundQuoteNumeralGrammarUnification:
    """P1: `ground_quote`'s numeral-boundary guard is unified onto the shared
    `carmel.services.numeric.NUMERAL_CANDIDATE_RE` / `find_numeral_extent`
    primitive (the same grammar `carmel.services.grounding` uses), closing
    real accepted-but-wrong grounding bugs the old, weaker
    `_NUMERIC_TOKEN_RE` / `_quote_looks_numeric` pair let through."""

    def test_rejects_sign_flip_negative_grounded_as_positive(self) -> None:
        # Old bug: "1.5" alone would ground inside "-1.5", silently dropping
        # the sign and recording +1.5 for a document stating -1.5.
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("T = -1.5 K was used", "1.5", role=QuoteRole.VALUE)

    def test_accepts_signed_value_as_the_maximal_candidate(self) -> None:
        locator = ground_quote("T = -1.5 K was used", "-1.5", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 8)

    def test_rejects_exponent_fragment_lacking_sign_in_old_regex(self) -> None:
        # Old bug: `_quote_looks_numeric("-3")` was False (no sign in the old
        # grammar), so the maximality check was skipped entirely and an
        # exponent fragment grounded as if it were a standalone value.
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("k = 1.0e-3 cm3/mol/s", "-3", role=QuoteRole.VALUE)

    def test_rejects_species_subscript_as_sits_inside_identifier(self) -> None:
        # Old bug: a bare digit subscript of a species name ("H2") grounded
        # as if it were a measured value.
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("Fuel H2 was used", "2", role=QuoteRole.VALUE)

    def test_rejects_unit_digit_as_sits_inside_identifier(self) -> None:
        # "cm3" -- the "3" is a unit power, not a numeral in its own right.
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("k = 1.0e-5 cm3/mol/s", "3", role=QuoteRole.VALUE)

    def test_rejects_thousands_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("T = 1,023 K", "023", role=QuoteRole.VALUE)

    def test_rejects_thousands_leading_digit_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("T = 1,023 K", "1", role=QuoteRole.VALUE)

    def test_rejects_cas_number_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("ethanol 64-17-5 was used", "17", role=QuoteRole.VALUE)

    def test_rejects_range_endpoint_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("over 1000-1200 K", "1200", role=QuoteRole.VALUE)

    def test_accepts_whole_range_as_one_candidate(self) -> None:
        locator = ground_quote("over 1000-1200 K", "1000-1200", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (5, 14)

    def test_accepts_plain_integer_unaffected(self) -> None:
        locator = ground_quote("T = 1023 K", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 8)

    def test_accepts_negative_value_full_precision(self) -> None:
        text = "T = -1.5 K was used"
        locator = ground_quote(text, "-1.5", role=QuoteRole.VALUE)
        assert text[locator.start : locator.end] == "-1.5"

    def test_accepts_number_glued_to_unit_letter(self) -> None:
        # A glued unit must stay groundable: letters are not part of a
        # numeral, so they never extend or block a candidate on their own.
        locator = ground_quote("at 1023K nominal", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (3, 7)

    def test_non_numeral_unit_quote_unaffected(self) -> None:
        locator = ground_quote("T = 1023 K", "K", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (9, 10)

    def test_non_numeral_label_quote_unaffected(self) -> None:
        locator = ground_quote("the mole fraction of CO", "mole fraction", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 17)

    def test_ambiguity_is_checked_before_the_numeral_check_and_still_refuses(self) -> None:
        # The new numeral check must never silently resolve an ambiguous
        # quote: ambiguity is still detected and raised BEFORE the numeral
        # check ever runs, even though "-3" now fullmatches the numeral
        # candidate grammar (unlike under the old, sign-less grammar).
        text = "k1 = 1.0e-3 cm3/mol/s, k2 = 2.0e-3 cm3/mol/s"
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote(text, "-3", role=QuoteRole.VALUE)


class TestGroundQuoteEnclosingConstructGuard:
    """A numeral that is a whole numeral on its own can still be only one PIECE
    of a larger multi-token numeric construct that `find_numeral_extent`'s
    character-level check cannot see (it only asks "is THIS candidate whole",
    never "is a larger sibling construct wrapped around it"). These are real
    accepted-but-wrong grounding bugs this guard closes; each construct must
    refuse with a message an operator can tell apart from the others, not
    merely a distinct exception type."""

    def test_rejects_the_mangled_ascii6_uncertainty_marker_itself(self) -> None:
        with pytest.raises(QuoteGroundingError, match="ascii6_uncertainty"):
            ground_quote("The temperature was 307 6 10 K.", "6", role=QuoteRole.VALUE)

    def test_rejects_the_uncertainty_value_not_the_measurement(self) -> None:
        with pytest.raises(QuoteGroundingError, match="ascii6_uncertainty"):
            ground_quote("The temperature was 307 6 10 K.", "10", role=QuoteRole.VALUE)

    def test_rejects_spaced_range_endpoint_ascii_hyphen(self) -> None:
        with pytest.raises(QuoteGroundingError, match="spaced_range"):
            ground_quote("The range was 1000 - 1200 K.", "1200", role=QuoteRole.VALUE)

    def test_rejects_spaced_range_endpoint_en_dash(self) -> None:
        with pytest.raises(QuoteGroundingError, match="spaced_range"):
            ground_quote("The range was 1000 – 1200 K.", "1200", role=QuoteRole.VALUE)

    def test_rejects_flattened_scientific_notation_exponent(self) -> None:
        with pytest.raises(QuoteGroundingError, match="flattened_scientific"):
            ground_quote("k = 3.94 x 10 03 s-1.", "03", role=QuoteRole.VALUE)

    def test_rejects_flattened_scientific_notation_base(self) -> None:
        with pytest.raises(QuoteGroundingError, match="flattened_scientific"):
            ground_quote("k = 3.94 x 10 03 s-1.", "10", role=QuoteRole.VALUE)

    def test_ascii6_uncertainty_and_spaced_range_messages_are_distinct(self) -> None:
        # Anti-masking-bug regression: the two refusals must be tellable apart
        # by MESSAGE CONTENT, not merely by both being QuoteGroundingError.
        try:
            ground_quote("The temperature was 307 6 10 K.", "6", role=QuoteRole.VALUE)
        except QuoteGroundingError as exc:
            ascii6_message = str(exc)
        try:
            ground_quote("The range was 1000 - 1200 K.", "1200", role=QuoteRole.VALUE)
        except QuoteGroundingError as exc:
            spaced_range_message = str(exc)
        assert ascii6_message != spaced_range_message
        assert "ascii6_uncertainty" in ascii6_message
        assert "spaced_range" in spaced_range_message

    def test_tight_range_endpoint_is_still_rejected_as_a_maximality_fragment(self) -> None:
        # Pre-existing behavior, unaffected by this guard: the tight (no
        # surrounding whitespace) range endpoint is refused earlier, by the
        # find_numeral_extent maximality check.
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("over 1000-1200 K", "1200", role=QuoteRole.VALUE)

    def test_whole_tight_range_still_grounds(self) -> None:
        locator = ground_quote("over 1000-1200 K", "1000-1200", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (5, 14)

    def test_plain_integer_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = 1023 K", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 8)

    def test_negative_value_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = -1.5 K was used", "-1.5", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 8)

    def test_temperature_glued_to_unit_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("at 1023K nominal", "1023", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (3, 7)

    def test_unit_letter_quote_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = 1023 K", "K", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (9, 10)

    def test_label_quote_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("the mole fraction of CO", "mole fraction", role=QuoteRole.VALUE)
        assert (locator.start, locator.end) == (4, 17)

    def test_whole_quote_spaced_range_still_grounds_via_plain_substring_search(self) -> None:
        # The whole-quote spaced range contains internal whitespace, so it
        # never fullmatches NUMERAL_CANDIDATE_RE and the entire numeral-guard
        # block (including this new check) is skipped: it grounds via plain
        # substring search, exactly as before this guard existed. This test
        # pins that capability is NOT lost by the new guard.
        text = "The range was 1000 - 1200 K."
        quote = "1000 - 1200"
        locator = ground_quote(text, quote, role=QuoteRole.VALUE)
        assert text[locator.start : locator.end] == quote


class TestGroundQuoteRoleDispatchIsExhaustive:
    """Round 40, defect 1: `ground_quote`'s role dispatch was an
    if/elif/elif chain with NO final `else` -- a `role` that is not a
    genuine `QuoteRole` member (a plain string, `None`, or any other object)
    matched no branch and fell through to the unconditional assert/return at
    the bottom of the function, grounding via raw, unguarded substring
    search -- a total regression of every boundary guard below. These pin
    the fail-closed guard that now refuses any non-`QuoteRole` role outright,
    before the search for `quote` is even interpreted against a boundary
    rule. The scenario text is the same one `TestGroundQuoteNonNumericTokenBoundary
    .test_rejects_letter_quote_glued_to_a_larger_word` uses for the genuine
    `QuoteRole.UNIT` refusal, to make the contrast explicit: a bad `role`
    must refuse the SAME "K" grounds inside "Kinetics" shape, not silently
    accept it."""

    def test_rejects_a_plain_string_role_that_equals_a_member_value(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ):
            ground_quote("Kinetics pressure 1023", "K", role="value")  # type: ignore[arg-type]

    def test_rejects_a_none_role(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ):
            ground_quote("Kinetics pressure 1023", "K", role=None)  # type: ignore[arg-type]

    def test_rejects_an_arbitrary_object_role(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ):
            ground_quote("Kinetics pressure 1023", "K", role=object())  # type: ignore[arg-type]

    # -- Round 41: the guard must run FIRST, before any later check gets a
    # chance to surface its own message instead. Each test below pairs an
    # invalid role with a quote that WOULD trip a specific later check (an
    # ambiguous quote, a quote absent from the text, a whitespace-padded
    # quote, an out-of-range occurrence) -- previously the role guard ran
    # after all of these, so the invalid role was masked and the LATER
    # check's message surfaced instead. Each asserts the role error message
    # and explicitly asserts the masking check's message text is NOT what
    # was raised.

    def test_rejects_invalid_role_before_reporting_an_ambiguous_quote(self) -> None:
        # "300" appears twice with occurrence=None (the default) -- this
        # would normally raise the "appears 2 times" ambiguity error. The
        # role guard must preempt it.
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ) as excinfo:
            ground_quote("300 K and 300 K again", "300", role=None)  # type: ignore[arg-type]
        assert "appears" not in str(excinfo.value)

    def test_rejects_invalid_role_before_reporting_a_quote_not_found(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ) as excinfo:
            ground_quote("Kinetics pressure 1023", "xyz", role=None)  # type: ignore[arg-type]
        assert "was not found" not in str(excinfo.value)

    def test_rejects_invalid_role_before_reporting_whitespace_padding(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ) as excinfo:
            ground_quote("Kinetics pressure 1023", " K", role=None)  # type: ignore[arg-type]
        assert "leading or trailing whitespace" not in str(excinfo.value)

    def test_rejects_invalid_role_before_reporting_an_out_of_range_occurrence(self) -> None:
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ) as excinfo:
            ground_quote("abc abc", "abc", occurrence=5, role=None)  # type: ignore[arg-type]
        assert "out of range" not in str(excinfo.value)

    def test_rejects_invalid_role_with_an_empty_quote(self) -> None:
        # The role guard is the VERY FIRST statement in the function body,
        # ahead of even the empty-quote check, so an invalid role is refused
        # here too, not with the empty-quote message.
        with pytest.raises(
            QuoteGroundingError, match="not a carmel.services.numeric.QuoteRole member"
        ) as excinfo:
            ground_quote("Kinetics pressure 1023", "", role=None)  # type: ignore[arg-type]
        assert "empty" not in str(excinfo.value)


class TestGroundQuoteNonNumericTokenBoundary:
    """A non-numeric quote (a unit or a label) never fullmatches
    `NUMERAL_CANDIDATE_RE`, so it skipped every boundary guard above entirely
    and grounded via plain, unguarded substring search -- e.g.
    `ground_quote("Kinetics pressure 1023", "K", role=QuoteRole.UNIT)`
    silently accepted the "K" inside "Kinetics" as if it were the unit
    Kelvin. This class pins the ROLE-AWARE boundary guards that close that
    gap: `QuoteRole.UNIT` and `QuoteRole.LABEL` each apply their own
    per-edge ALLOWLIST of permitted neighbouring characters
    (`carmel.services.numeric.unit_boundary_violation` /
    `label_boundary_violation`), distinct from the generic
    `has_clean_token_boundary` fallback that `QuoteRole.VALUE` still uses for
    a non-numeral quote. Every refusal branch below asserts a message
    substring specific to ITS branch and ITS edge (LEADING vs TRAILING),
    never a substring shared across branches -- this project has hit masked,
    indistinguishable refusals of this shape five times before."""

    # -- QuoteRole.UNIT: allowlist adjacency, trailing edge ----------------

    def test_rejects_letter_quote_glued_to_a_larger_word(self) -> None:
        # The original bug report: "K" grounds inside "Kinetics", not as the
        # unit Kelvin. "K" is the first letter of "Kinetics", so this is a
        # TRAILING-edge violation (the "i" right after it).
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote(
                "Kinetics pressure 1023", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
            )

    def test_rejects_quote_that_is_a_prefix_of_a_larger_word(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("5 atm", "at", role=QuoteRole.UNIT, quantity=QuantityKind.PRESSURE)

    def test_rejects_unit_fragment_of_a_compound_unit(self) -> None:
        # "cm3" is a real unit-shaped token on its own, but here it is only
        # the leading piece of the compound unit "cm3/mol/s" -- the trailing
        # "/" is not on the trailing allowlist, so this is a fragment, not
        # the whole unit.
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote(
                "k = 1.0 cm3/mol/s", "cm3", role=QuoteRole.UNIT, quantity=QuantityKind.VOLUME
            )

    def test_rejects_unit_fragment_truncating_an_exponent(self) -> None:
        # "m/s" is a real unit on its own, but here it is truncating "m/s2"
        # -- the trailing "2" is a digit, not on the trailing allowlist, so
        # quoting only "m/s" silently drops the exponent.
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote(
                "a = 9.8 m/s2", "m/s", role=QuoteRole.UNIT, quantity=QuantityKind.VELOCITY
            )

    def test_rejects_unit_quote_followed_by_an_open_paren(self) -> None:
        # "(" is LEADING-only on the unit allowlist -- a unit may OPEN a
        # parenthetical ("T (K) 1023") but must never sit INSIDE one on its
        # trailing edge, so "bar" inside "bar(a)" is refused, not silently
        # accepted as plain "bar" (round 40, defect 2, confirmed-bad case).
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote(
                "the pressure was 1 bar(a)", "bar", role=QuoteRole.UNIT, quantity=QuantityKind.PRESSURE
            )

    def test_rejects_unit_quote_followed_by_a_unicode_minus_sign(self) -> None:
        # U+2212 MINUS SIGN -- not the ASCII hyphen, not on the trailing
        # allowlist, and not whitespace. The old denylist
        # (`_UNIT_TOKEN_SYMBOLS`) missed this character entirely (round 40,
        # defect 2, confirmed-bad case).
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("k = 3 s−1", "s", role=QuoteRole.UNIT, quantity=QuantityKind.TIME)

    def test_rejects_unit_quote_followed_by_an_en_dash(self) -> None:
        # U+2013 EN DASH -- same story as the minus sign above (round 40,
        # defect 2, confirmed-bad case).
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("k = 3 s–1", "s", role=QuoteRole.UNIT, quantity=QuantityKind.TIME)

    def test_rejects_unit_quote_followed_by_a_superscript_minus(self) -> None:
        # U+207B SUPERSCRIPT MINUS -- same story again (round 40, defect 2,
        # confirmed-bad case).
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("k = 3 s⁻¹", "s", role=QuoteRole.UNIT, quantity=QuantityKind.TIME)

    # -- QuoteRole.UNIT: allowlist adjacency, leading edge ------------------

    def test_rejects_unit_quote_with_the_degree_mark_stripped(self) -> None:
        # "C" alone is glued to the preceding degree mark "°" -- not
        # whitespace and not on the leading allowlist -- so quoting bare "C"
        # silently strips the degree mark from "25°C".
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("T = 25°C", "C", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE)

    def test_rejects_unit_quote_immediately_preceded_by_a_letter(self) -> None:
        # Isolates the LEADING-edge, non-digit branch in isolation from the
        # digit-glue exception: "K" in "mK sensor" is preceded by the letter
        # "m", not a digit, so this exercises the plain leading-allowlist
        # check, not `unit_digit_glue` -- round 40 review noted no test
        # isolated this shape.
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("mK sensor", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE)

    # -- QuoteRole.UNIT: digit-glue exception, and its refusal -------------

    def test_rejects_unit_glued_to_a_non_numeral_digit_run(self) -> None:
        # "K" is immediately preceded by a digit ("3"), which is the leading-
        # edge glue exception's shape. A caller supplies a value_span whose
        # end lines up exactly with "K"'s start -- (3, 4) -- but that span is
        # NOT a genuine numeral extent: find_numeral_extent(text, 3) is None
        # because "3" is the tail of the word "run3", not a numeral in its
        # own right (preceded by the letter "n"). Round 41: the FIRST fix
        # only checked value_span[1] == start and would have wrongly let
        # this fire; the current rule independently recomputes
        # find_numeral_extent and refuses because the span names no real
        # numeral at all.
        with pytest.raises(
            QuoteGroundingError, match="not itself a clean, maximal numeral in place"
        ):
            ground_quote(
                "run3K reactor",
                "K",
                role=QuoteRole.UNIT,
                value_span=(3, 4),
                quantity=QuantityKind.TEMPERATURE,
            )

    def test_rejects_a_fabricated_value_span_with_no_numeral_at_its_start(self) -> None:
        # Round 41's exact exploit: a caller fabricates value_span=(0, 4)
        # over "run3K reactor" -- its end (4) lines up with "K"'s start, so
        # the OLD (first-attempt) fix, which only checked that alignment,
        # would have wrongly fired the glue exception. find_numeral_extent
        # at index 0 is None ("r" is not a digit/sign), so the span names no
        # real numeral and must be refused.
        with pytest.raises(
            QuoteGroundingError, match="not itself a clean, maximal numeral in place"
        ):
            ground_quote(
                "run3K reactor",
                "K",
                role=QuoteRole.UNIT,
                value_span=(0, 4),
                quantity=QuantityKind.TEMPERATURE,
            )

    def test_rejects_a_fabricated_value_span_spanning_unrelated_text(self) -> None:
        # A second shape of the same round-41 exploit: value_span=(0, 6)
        # over "case 1K was run" ends exactly at "K"'s start (6) but starts
        # at "c", not a numeral -- find_numeral_extent(text, 0) is None, so
        # the span is refused as not naming a genuine numeral, regardless of
        # where its end happens to land.
        with pytest.raises(
            QuoteGroundingError, match="not itself a clean, maximal numeral in place"
        ):
            ground_quote(
                "case 1K was run",
                "K",
                role=QuoteRole.UNIT,
                value_span=(0, 6),
                quantity=QuantityKind.TEMPERATURE,
            )

    def test_rejects_letter_quote_preceded_by_a_digit_without_a_value_span(self) -> None:
        # Round 40, defect 3: fail-closed by default. "1023" is a clean,
        # maximal numeral ending exactly at "K" -- the OLD (buggy) rule
        # accepted this on cleanliness alone. The new rule requires the
        # caller to additionally supply value_span proving THIS digit run is
        # the measurement's own value; omitting it means the exception never
        # fires, even though the digit run is clean.
        with pytest.raises(QuoteGroundingError, match="no value_span was supplied to confirm"):
            ground_quote("1023K", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE)

    def test_rejects_letter_quote_preceded_by_a_digit_with_a_malformed_value_span(self) -> None:
        # value_span=(10, 12) is out of range for "1023K" (len 5): a
        # malformed/out-of-range span proves nothing about this
        # measurement's value and must be refused defensively, without ever
        # risking an IndexError/TypeError from inside the boundary gate.
        with pytest.raises(QuoteGroundingError, match="malformed or out of range"):
            ground_quote(
                "1023K",
                "K",
                role=QuoteRole.UNIT,
                value_span=(10, 12),
                quantity=QuantityKind.TEMPERATURE,
            )

    def test_rejects_letter_quote_preceded_by_a_digit_with_a_mismatched_value_span(self) -> None:
        # value_span=(0, 3) is well-formed and IS a genuine numeral extent
        # ("102" in "1023K"), but its end (3) does not equal "K"'s start
        # (4) -- it is not a claim about THIS glued digit run, so the
        # exception must not fire.
        with pytest.raises(
            QuoteGroundingError, match="does not end exactly where this quote starts"
        ):
            ground_quote(
                "1023K",
                "K",
                role=QuoteRole.UNIT,
                value_span=(0, 3),
                quantity=QuantityKind.TEMPERATURE,
            )

    def test_accepts_letter_quote_immediately_preceded_by_a_digit(self) -> None:
        # Deliberate asymmetry, load-bearing: NUMERAL_EXTENT_RE's trailing
        # boundary permits a letter right after a numeral (so "1023" stays
        # groundable inside "1023K"). The symmetric consequence is that the
        # unit "K" glued to that same numeral must ALSO stay groundable --
        # this is the ONE exception `unit_boundary_violation` allows, now
        # gated by an explicit value_span proving "1023" really is a clean,
        # maximal numeral ending exactly where "K" starts. This test proves
        # only that a well-formed, clean, correctly-aligned value_span is
        # ACCEPTED -- it does not by itself prove value_span is load-bearing
        # (a stub that ignored value_span entirely would also pass this
        # test); see
        # test_value_span_prevents_grounding_a_run_id_digit_as_the_value for
        # the companion test proving a wrong-but-well-formed value_span is
        # refused, which is what actually demonstrates value_span is checked
        # rather than merely accepted.
        locator = ground_quote(
            "1023K",
            "K",
            role=QuoteRole.UNIT,
            value_span=(0, 4),
            quantity=QuantityKind.TEMPERATURE,
        )
        assert (locator.start, locator.end) == (4, 5)

    def test_value_span_prevents_grounding_a_run_id_digit_as_the_value(self) -> None:
        # Round 40, defect 3's confirmed bug: value and unit are grounded
        # independently, so proving "some clean numeral ends here" is not the
        # same claim as "THIS measurement's value ends here". Occurrence 0 of
        # "K" is glued to the run id "1" in "1K" -- itself a clean, maximal
        # numeral -- but it is NOT this measurement's value ("1023"), so its
        # value_span's end (44) does not match occurrence 0's start (6): a
        # value_span/quote-start MISMATCH, not a malformed or non-numeral
        # span, so it must be refused even though it looks exactly like the
        # accepted shape above. Occurrence 1 is the real unit glued to the
        # real value and must still be accepted.
        text = "case 1K was the run id. Temperature was 1023 K"
        value_locator = ground_quote(text, "1023", role=QuoteRole.VALUE)
        value_span = (value_locator.start, value_locator.end)
        with pytest.raises(
            QuoteGroundingError, match="does not end exactly where this quote starts"
        ):
            ground_quote(
                text,
                "K",
                role=QuoteRole.UNIT,
                occurrence=0,
                value_span=value_span,
                quantity=QuantityKind.TEMPERATURE,
            )
        locator = ground_quote(
            text,
            "K",
            role=QuoteRole.UNIT,
            occurrence=1,
            value_span=value_span,
            quantity=QuantityKind.TEMPERATURE,
        )
        assert text[locator.start : locator.end] == "K"

    def test_accepts_letter_quote_in_parentheses(self) -> None:
        locator = ground_quote(
            "T (K) 1023", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
        )
        assert (locator.start, locator.end) == (3, 4)

    def test_accepts_unit_quote_bounded_by_whitespace(self) -> None:
        locator = ground_quote(
            "the pressure was 1 bar", "bar", role=QuoteRole.UNIT, quantity=QuantityKind.PRESSURE
        )
        assert (locator.start, locator.end) == (19, 22)

    def test_accepts_unit_quote_containing_a_slash(self) -> None:
        locator = ground_quote(
            "flame speed in cm/s", "cm/s", role=QuoteRole.UNIT, quantity=QuantityKind.VELOCITY
        )
        assert (locator.start, locator.end) == (15, 19)

    def test_accepts_unit_quote_containing_the_degree_mark(self) -> None:
        # The full unit "°C", quoted whole, is not a fragment of anything --
        # it is bounded by whitespace on both edges.
        locator = ground_quote(
            "T = 25 °C", "°C", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
        )
        assert (locator.start, locator.end) == (7, 9)

    # -- QuoteRole.LABEL: strict letter/digit adjacency, no exception ------

    def test_rejects_quote_immediately_followed_by_a_letter(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("the temperature rose", "temp", role=QuoteRole.LABEL)

    def test_rejects_quote_immediately_preceded_by_a_letter(self) -> None:
        # Isolates the LEADING-edge, letter-precedes-letter branch: "ics" is
        # the tail of "Kinetics" (preceded by the letter "t"), but its own
        # trailing edge sits cleanly at a space before "pressure" -- so this
        # can only be caught by the leading check, not the trailing one.
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("Kinetics pressure", "ics", role=QuoteRole.LABEL)

    def test_rejects_label_glued_to_a_trailing_digit(self) -> None:
        # "CO" is a real species label on its own, but here it is only the
        # leading piece of "CO2" -- LABEL allows no glue exception, unlike
        # UNIT, so this must refuse even though "CO2" looks like a
        # value-glued-to-a-label shape.
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("CO2 mole fraction = 0.1", "CO", role=QuoteRole.LABEL)

    def test_rejects_label_glued_to_a_trailing_digit_mid_sentence(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("the NO2 profile", "NO", role=QuoteRole.LABEL)

    def test_rejects_single_letter_label_glued_to_a_trailing_digit(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("H2O yield", "H", role=QuoteRole.LABEL)

    def test_rejects_label_glued_to_a_trailing_subscript_digit(self) -> None:
        # '₂' (U+2082, subscript two) is not an ASCII digit, but Python's own
        # str.isdigit() reports True for it -- label_boundary_violation must
        # use that same per-character test, not a narrower ASCII-only one,
        # so a subscript-digit species marker is refused exactly like an
        # ordinary digit would be.
        assert "₂".isdigit()
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("H₂ mole fraction", "H", role=QuoteRole.LABEL)

    def test_rejects_label_quote_immediately_preceded_by_a_digit(self) -> None:
        # Isolates the LEADING-edge, digit-precedes-letter branch: "O" in
        # "H2O yield" sits cleanly at a space before "yield" on its trailing
        # edge, but is immediately preceded by the digit "2" -- so this can
        # only be caught by the leading check, not the trailing one.
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("H2O yield", "O", role=QuoteRole.LABEL)

    # -- QuoteRole.LABEL: round 40, defect 4 -- punctuation delimiters -----
    # "_" and "/" are deliberately NOT on the label allowlist: a label
    # fragment separated from a larger identifier only by an underscore or a
    # slash must refuse just as it would if separated by a bare letter or
    # digit, since species/ratio identifiers like "X_CO" or "H2/CO" are
    # exactly the shape a label-fragment bug would silently mis-ground.

    def test_rejects_label_quote_preceded_by_a_slash(self) -> None:
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("1/T plot", "T", role=QuoteRole.LABEL)

    def test_rejects_label_quote_preceded_by_an_underscore(self) -> None:
        with pytest.raises(QuoteGroundingError, match="LEADING edge"):
            ground_quote("X_CO was reported", "CO", role=QuoteRole.LABEL)

    def test_rejects_label_quote_followed_by_a_slash(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("H2/CO ratio = 1", "H2", role=QuoteRole.LABEL)

    def test_rejects_label_quote_followed_by_an_underscore(self) -> None:
        with pytest.raises(QuoteGroundingError, match="TRAILING edge"):
            ground_quote("S_L measured", "S", role=QuoteRole.LABEL)

    def test_accepts_label_quote_with_an_underscore_quoted_whole(self) -> None:
        locator = ground_quote("X_CO was reported", "X_CO", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 4)

    def test_accepts_label_quote_with_a_slash_quoted_whole(self) -> None:
        locator = ground_quote("1/T plot", "1/T", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 3)

    def test_accepts_multi_species_label_quote_with_a_slash_quoted_whole(self) -> None:
        locator = ground_quote("H2/CO ratio = 1", "H2/CO", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 5)

    def test_accepts_label_quote_with_an_underscore_subscript_quoted_whole(self) -> None:
        locator = ground_quote("S_L measured", "S_L", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 3)

    def test_accepts_label_quote_bounded_by_whitespace(self) -> None:
        locator = ground_quote("Kinetics pressure 1023", "pressure", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (9, 17)

    def test_accepts_multi_word_label_quote(self) -> None:
        locator = ground_quote("ignition delay time", "ignition delay", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 14)

    def test_accepts_label_quote_including_its_own_digit(self) -> None:
        # "H2" quoted WHOLE (not truncated to "H") is a clean, standalone
        # label token bounded by whitespace on both edges -- LABEL refuses a
        # quote that is a FRAGMENT of a larger token, not a quote that simply
        # contains a digit as part of itself.
        locator = ground_quote("H2 mole fraction = 0.1", "H2", role=QuoteRole.LABEL)
        assert (locator.start, locator.end) == (0, 2)

    # -- QuoteRole.VALUE: generic fallback, unchanged ----------------------

    def test_rejects_quote_immediately_preceded_by_a_digit(self) -> None:
        with pytest.raises(QuoteGroundingError, match="token boundary"):
            ground_quote("310 ms elapsed", "10 ms", role=QuoteRole.VALUE)

    def test_rejects_quote_immediately_followed_by_a_digit(self) -> None:
        # Isolates the TRAILING-edge, digit-followed-by-digit branch: "was 10"
        # starts cleanly after a space, but its trailing "0" is immediately
        # followed by another digit ("2" of "102") -- so this can only be
        # caught by the trailing check, not the leading one.
        with pytest.raises(QuoteGroundingError, match="token boundary"):
            ground_quote("value was 102 units", "was 10", role=QuoteRole.VALUE)

    def test_accepts_value_amid_unit_and_label_bearing_text(self) -> None:
        # VALUE-role grounding of a whole numeral is unaffected by the new
        # role-aware UNIT/LABEL branches -- it still goes through the
        # pre-existing numeral-maximality path exactly as before this task.
        locator = ground_quote(
            "T (K) 1023  P (bar) 1  phi 0.5", "1023", role=QuoteRole.VALUE
        )
        assert (locator.start, locator.end) == (6, 10)

    # -- whitespace padding: refused for every role -------------------------

    def test_rejects_unit_quote_padded_with_leading_whitespace(self) -> None:
        with pytest.raises(QuoteGroundingError, match="leading or trailing whitespace"):
            ground_quote(
                "T = 1023 K", " K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
            )

    def test_rejects_label_quote_padded_with_trailing_whitespace(self) -> None:
        with pytest.raises(QuoteGroundingError, match="leading or trailing whitespace"):
            ground_quote("Kinetics pressure 1023", "pressure ", role=QuoteRole.LABEL)

    def test_rejects_value_quote_padded_with_leading_whitespace(self) -> None:
        with pytest.raises(QuoteGroundingError, match="leading or trailing whitespace"):
            ground_quote("T (K) 1023", " 1023", role=QuoteRole.VALUE)

    def test_ambiguity_is_checked_before_the_token_boundary_check_and_still_refuses(self) -> None:
        # Same ordering as `TestGroundQuoteNumeralGrammarUnification
        # .test_ambiguity_is_checked_before_the_numeral_check_and_still_refuses`
        # for the numeral path: `ground_quote` counts every match BEFORE it
        # ever looks at the boundary of the single, already-resolved
        # occurrence. A boundary filter that instead ran first and silently
        # dropped the bad match would turn a genuinely ambiguous quote into a
        # falsely "unique" one -- exactly the ambiguity this function exists
        # to refuse, never resolve. Here "K" appears twice: once glued inside
        # "Kelvin" (which the new guard would refuse on its own) and once as
        # a clean, isolated unit -- ambiguity must still win.
        text = "Kelvin unit and 1023 K measured"
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote(text, "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE)


class TestGroundQuoteUnitAcceptanceMatrix:
    """D-U2's mandatory, two-sided acceptance matrix for role=QuoteRole.UNIT:
    every MUST-REFUSE row from Layer 0 (argument validation) and every
    MUST-ACCEPT row proving a genuinely clean, maximal unit quote for several
    distinct quantities still grounds successfully, all in one place so the
    full boundary is visible at a glance rather than scattered one case per
    class. Layers 1-3's individual refusal branches are already pinned above
    (`TestGroundQuoteNonNumericTokenBoundary`) and are not re-duplicated
    here."""

    # -- MUST-REFUSE: Layer 0 argument validation ---------------------------

    def test_refuses_unit_role_with_quantity_omitted(self) -> None:
        with pytest.raises(QuoteGroundingError, match="requires quantity="):
            ground_quote("T = 1023 K", "K", role=QuoteRole.UNIT)

    def test_refuses_unit_role_with_quantity_none(self) -> None:
        with pytest.raises(QuoteGroundingError, match="requires quantity="):
            ground_quote("T = 1023 K", "K", role=QuoteRole.UNIT, quantity=None)

    def test_refuses_unit_role_with_a_plain_string_quantity(self) -> None:
        # QuantityKind is a StrEnum: a plain string equal to a member's VALUE
        # (e.g. "temperature") compares == to that member but is not a
        # genuine member and fails isinstance() -- the StrEnum equality
        # trap. This must be refused, not silently accepted via ==.
        with pytest.raises(QuoteGroundingError, match="not a genuine"):
            ground_quote("T = 1023 K", "K", role=QuoteRole.UNIT, quantity="temperature")

    def test_refuses_quantity_other(self) -> None:
        with pytest.raises(QuoteGroundingError, match="QuantityKind.OTHER has no unit vocabulary"):
            ground_quote("T = 1023 K", "K", role=QuoteRole.UNIT, quantity=QuantityKind.OTHER)

    def test_refuses_quantity_supplied_for_value_role(self) -> None:
        with pytest.raises(QuoteGroundingError, match="has no meaning for any other role"):
            ground_quote(
                "T = 1023 K", "1023", role=QuoteRole.VALUE, quantity=QuantityKind.TEMPERATURE
            )

    def test_refuses_quantity_supplied_for_label_role(self) -> None:
        with pytest.raises(QuoteGroundingError, match="has no meaning for any other role"):
            ground_quote(
                "pressure was 1 bar",
                "pressure",
                role=QuoteRole.LABEL,
                quantity=QuantityKind.PRESSURE,
            )

    # -- MUST-REFUSE: K^1 exponent-vs-footnote ambiguity ---------------------

    def test_refuses_temperature_unit_followed_by_a_bare_superscript_one(self) -> None:
        # "K¹" is lexically identical, from a single-adjacent-character
        # check alone, whether the "¹" is a genuine exponent continuation or
        # an unrelated footnote marker. This has its own discriminant,
        # `unit_trailing_exponent_or_footnote_ambiguous`, carved out of the
        # generic trailing-not-maximal bucket specifically so this
        # ambiguity is visible to an operator by name (round 43 finding:
        # collapsing it into the generic message hid the coverage gap).
        with pytest.raises(QuoteGroundingError, match="superscript or subscript digit"):
            ground_quote(
                "T = 1023 K¹ (footnote)",
                "K",
                role=QuoteRole.UNIT,
                quantity=QuantityKind.TEMPERATURE,
            )

    # -- MUST-REFUSE: Layer 3 table maximality, both directions -------------

    def test_refuses_unit_that_is_a_prefix_of_a_longer_registered_spelling(self) -> None:
        # "cm" is itself a registered LENGTH unit and sits at a clean Layer
        # 1/2 boundary (bounded by spaces on both sides), but "cm s^-1" is a
        # registered VELOCITY alias starting at the same position and
        # extending past this quote's end -- claiming LENGTH does not dodge
        # the check, because maximality is checked against the UNION over
        # every quantity, not just the claimed one.
        with pytest.raises(QuoteGroundingError, match="unit_not_maximal_forward|LONGER"):
            ground_quote(
                "u = 10 cm s^-1", "cm", role=QuoteRole.UNIT, quantity=QuantityKind.LENGTH
            )

    def test_refuses_unit_that_is_a_suffix_of_a_longer_registered_spelling(self) -> None:
        # "C" is itself a registered TEMPERATURE spelling (the normalized
        # form of the "deg C"/"degC"/"°C" aliases) and sits at a clean Layer
        # 1/2 boundary (preceded by whitespace), but "deg C" is a registered
        # TEMPERATURE alias ending at this same position and starting before
        # this quote -- a suffix fragment of a longer registered spelling.
        with pytest.raises(QuoteGroundingError, match="unit_not_maximal_backward|LONGER"):
            ground_quote(
                "T = 10 deg C", "C", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
            )

    # -- MUST-REFUSE: whitespace-variant forward maximality (P1-1, round 43) -

    def test_refuses_unit_prefix_of_longer_spelling_separated_by_double_space(self) -> None:
        # Same corruption as test_refuses_unit_that_is_a_prefix_of_a_longer_
        # registered_spelling above, but the registered VELOCITY alias "cm
        # s^-1" is separated by TWO ASCII spaces in the source text instead
        # of one. A literal `text.startswith(spelling, start)` check would
        # miss this (the literal spelling contains only a single space), so
        # forward maximality must be whitespace-equivalent -- matching one
        # or more whitespace characters of ANY kind -- rather than an exact
        # string match, or this quote is silently admitted as LENGTH when
        # the source text actually reads the VELOCITY alias.
        with pytest.raises(QuoteGroundingError, match="unit_not_maximal_forward|LONGER"):
            ground_quote(
                "u = 10 cm  s^-1", "cm", role=QuoteRole.UNIT, quantity=QuantityKind.LENGTH
            )

    def test_refuses_unit_prefix_of_longer_spelling_separated_by_nbsp(self) -> None:
        # Same as above, but the separator is U+00A0 NO-BREAK SPACE rather
        # than an ASCII space -- a plausible artefact of PDF text
        # extraction. `str.isspace()` is True for NBSP and `\s+` matches it,
        # so this must refuse identically to the double-space and ASCII-
        # single-space cases.
        with pytest.raises(QuoteGroundingError, match="unit_not_maximal_forward|LONGER"):
            ground_quote(
                "u = 10 cm s^-1", "cm", role=QuoteRole.UNIT, quantity=QuantityKind.LENGTH
            )

    def test_refuses_unit_prefix_of_longer_spelling_separated_by_newline(self) -> None:
        # Same as above, but the separator is a newline -- another
        # plausible PDF-extraction artefact (a line break landing between a
        # unit and its per-time suffix). Must refuse identically.
        with pytest.raises(QuoteGroundingError, match="unit_not_maximal_forward|LONGER"):
            ground_quote(
                "u = 10 cm\ns^-1", "cm", role=QuoteRole.UNIT, quantity=QuantityKind.LENGTH
            )

    def test_refuses_whitespace_variant_alias_quoted_whole_as_not_in_vocabulary(self) -> None:
        # Deliberate, documented coverage gap (round 43, P1-1): unlike
        # maximality, ADMISSION stays EXACT -- `units.normalize_unit` only
        # strips LEADING/TRAILING whitespace, never internal whitespace, so
        # admitting "cm  s^-1" (double space) as a whole quote would let the
        # gate accept a string that `normalize_unit` would then itself
        # reject downstream. Closing this gap by making admission
        # whitespace-equivalent too would be wrong for that reason, so this
        # quote is refused as unrecognised vocabulary, not accepted -- this
        # test pins that refusal so the gap cannot silently regress into an
        # accept without a test failing.
        with pytest.raises(QuoteGroundingError, match="unit_not_in_vocabulary|not a registered spelling"):
            ground_quote(
                "u = 10 cm  s^-1",
                "cm  s^-1",
                role=QuoteRole.UNIT,
                quantity=QuantityKind.VELOCITY,
            )

    # -- MUST-REFUSE: Layer 2 lexer maximality at the trailing edge ---------

    def test_refuses_unit_immediately_followed_by_an_unrelated_unit_token_char(self) -> None:
        # "K" is itself a registered TEMPERATURE spelling and "Kq" is NOT any
        # registered spelling for any quantity, so Layer 3's table-driven
        # forward-maximality check (which only fires for a REGISTERED longer
        # spelling) has nothing to catch here. This case isolates Layer 2's
        # lexer-level rule instead: a bare letter run immediately followed by
        # another unit-token character always looks like a fragment of some
        # larger, possibly-unregistered unit token (cf. "K" inside "K*cm" in
        # the docstring) and must be refused on that basis alone, regardless
        # of what the vocabulary table does or doesn't contain.
        #
        # NOTE: the neighbouring "unclassified trailing character" branch
        # (`unit_trailing_unclassified_char`) ALSO refuses any letter-like
        # `nxt`, since unit-token characters are, by construction, disjoint
        # from the whitespace/delimiter allow-list it checks -- so a generic
        # "TRAILING edge" match would pass even if this specific
        # `unit_trailing_not_maximal` branch were neutered entirely (that
        # branch would then always fall through to the unclassified-char
        # check, which still refuses, just under the wrong discriminant).
        # The match string below is therefore pinned to wording that is
        # UNIQUE to this branch's message, so the test can actually tell the
        # two discriminants apart rather than merely observing "some
        # refusal happened".
        with pytest.raises(QuoteGroundingError, match="not maximal at that edge"):
            ground_quote(
                "the unit is Kq nearby", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
            )

    # -- MUST-ACCEPT: a genuinely clean, maximal unit quote grounds ---------

    def test_accepts_clean_temperature_unit(self) -> None:
        locator = ground_quote(
            "the temperature was 1023 K", "K", role=QuoteRole.UNIT, quantity=QuantityKind.TEMPERATURE
        )
        assert (locator.start, locator.end) == (25, 26)

    def test_accepts_clean_pressure_unit(self) -> None:
        locator = ground_quote(
            "the pressure was 1 bar", "bar", role=QuoteRole.UNIT, quantity=QuantityKind.PRESSURE
        )
        assert (locator.start, locator.end) == (19, 22)

    def test_accepts_clean_time_unit(self) -> None:
        locator = ground_quote("k = 3 s, done", "s", role=QuoteRole.UNIT, quantity=QuantityKind.TIME)
        assert (locator.start, locator.end) == (6, 7)

    def test_accepts_clean_volume_unit(self) -> None:
        locator = ground_quote(
            "k = 1.0 cm3 per mol", "cm3", role=QuoteRole.UNIT, quantity=QuantityKind.VOLUME
        )
        assert (locator.start, locator.end) == (8, 11)

    def test_accepts_clean_velocity_unit(self) -> None:
        locator = ground_quote(
            "a = 9.8 m/s measured", "m/s", role=QuoteRole.UNIT, quantity=QuantityKind.VELOCITY
        )
        assert (locator.start, locator.end) == (8, 11)

    def test_accepts_unit_glued_to_its_own_verified_value(self) -> None:
        # Companion accept-path to the digit-glue MUST-REFUSE rows already
        # pinned in TestGroundQuoteNonNumericTokenBoundary: a well-formed,
        # correctly-aligned value_span naming a genuine, maximal numeral
        # extent lets the unit quote abut it directly.
        text = "T = 1023K exactly"
        value_locator = ground_quote(text, "1023", role=QuoteRole.VALUE)
        value_span = (value_locator.start, value_locator.end)
        locator = ground_quote(
            text,
            "K",
            role=QuoteRole.UNIT,
            value_span=value_span,
            quantity=QuantityKind.TEMPERATURE,
        )
        assert text[locator.start : locator.end] == "K"


class TestProducerEndToEnd:
    def test_produce_store_load_replay(self, tmp_path: Path) -> None:
        """The whole vertical slice: real store_artifact -> producer ->
        DatasetEnvelope -> store_dataset_envelope -> load_dataset_envelope ->
        independent replay via the standalone ``dataset_replay`` service."""
        stored_artifact = _store_synthetic_artifact(tmp_path, _TEXT)
        datasets_root = tmp_path / "datasets"

        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=stored_artifact.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        stored_dataset = store_dataset_envelope(datasets_root, envelope)
        loaded = load_dataset_envelope(datasets_root, stored_dataset.sha256)

        assert compute_dataset_sha(loaded.identity_payload()) == stored_dataset.sha256

        # Replay: the standalone service independently re-reads and
        # re-verifies extracted.json from disk and re-slices every span in
        # the LOADED envelope against it.
        report = replay_envelope(tmp_path, loaded)
        assert report.outcome is ReplayOutcome.VERIFIED, report.findings
        assert report.failures == ()
        assert report.unverifiable == ()
        # 2 axes x (value_ref + unit_ref + label_ref) = 6 char-span refs.
        assert report.checked_char_spans == 6

        # The loaded binding's extracted_text_sha256 must equal a digest
        # recomputed here from an independently re-read extracted.json --
        # deliberately re-read again here, separate from replay_envelope's
        # own internal re-read, so a bug in the service cannot vouch for
        # itself.
        raw_bytes = (artifact_dir(tmp_path, stored_artifact.sha256) / "extracted.json").read_bytes()
        replayed_text = ExtractedText.model_validate(json.loads(raw_bytes)).text
        node = loaded.source_graph.node("paper")
        binding = node.extraction
        assert isinstance(binding, ExtractionBinding), "extraction must be present, not Absent"
        assert binding.extracted_text_sha256 == hashlib.sha256(replayed_text.encode("utf-8")).hexdigest()
        assert binding.extracted_sha256 == stored_artifact.extracted_sha256

    def test_replay_fails_against_single_character_mutation(self, tmp_path: Path) -> None:
        """THE non-vacuousness proof for the replayer service: an envelope
        grounded against the ORIGINAL text must FAIL replay once the
        evidence store's own ``extracted.json`` for that same sha256 is
        tampered with by exactly one character inside a grounded span
        ("1023 K" -> "1024 K"). This proves the service performs a genuine
        on-disk re-read at replay time rather than trusting any cached
        text -- the content-addressed store makes it impossible for two
        different texts to share one sha256, so the only way to exercise
        the mismatch path is to mutate the evidence the envelope's own
        recorded sha256 already points at."""
        original = _store_synthetic_artifact(tmp_path, _TEXT)
        datasets_root = tmp_path / "datasets"

        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=original.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        stored_dataset = store_dataset_envelope(datasets_root, envelope)
        loaded = load_dataset_envelope(datasets_root, stored_dataset.sha256)

        # Sanity: replay passes clean against the untouched evidence...
        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED
        assert clean_report.checked_char_spans == 6

        # ...then tamper with the SAME artifact's extracted.json on disk,
        # in place, so the envelope still names the same sha256 but the
        # evidence underneath it has changed by exactly one character.
        assert len(_TEXT) == len(_MUTATED_TEXT)
        mutated_extracted = ExtractedText(
            text=_MUTATED_TEXT,
            normalized=_MUTATED_TEXT.casefold(),
            sections=[],
            extractor="pdf:pypdf",
            lossy=False,
        )
        extracted_path = artifact_dir(tmp_path, original.sha256) / "extracted.json"
        extracted_path.write_text(json.dumps(mutated_extracted.model_dump(mode="json")), encoding="utf-8")

        mutated_report = replay_envelope(tmp_path, loaded)
        assert mutated_report.outcome is not ReplayOutcome.VERIFIED
        assert any(
            "extracted.json" in finding.reason or "extracted_text_sha256" in finding.reason
            for finding in (mutated_report.failures + mutated_report.unverifiable)
        )


class TestProducerFailClosed:
    def test_refuses_corrupted_extracted_json(self, tmp_path: Path) -> None:
        """Bytes on disk that no longer match StoredArtifact.extracted_sha256
        are refused BEFORE parsing -- the corrupt replacement here is valid
        JSON and would parse fine, so only the digest check can be what
        refuses it. P1-B added ``verify_artifact`` as an earlier gate that
        also checks this same extracted.json digest (among other things), so
        it is now the one that fires first; the still-later, by-hand digest
        check below it (with the older "refusing to parse unverified bytes"
        message) is reached only if verify_artifact's own check is somehow
        bypassed, e.g. by a future refactor -- both are exercised by other
        tests in ``TestProducerFailClosed``. round-36: the shallow
        ``verify_artifact(deep=False)`` integrity check now runs before
        BOTH the legacy carve-outs and the deep=True call, so this is the
        gate that actually fires now -- its "failed integrity verification"
        message, not the deep-check's "failed verify_artifact" message."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        corrupt = ExtractedText(
            text="tampered", normalized="tampered", sections=[], extractor="pdf:pypdf", lossy=False
        )
        extracted_path.write_bytes(corrupt.model_dump_json().encode("utf-8"))

        with pytest.raises(DatasetProducerError, match="failed integrity verification"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_unknown_sha256(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetProducerError, match="no stored artifact"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256="0" * 64,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_constant_role_spec(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        constant_spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.CONSTANT,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="1023",
            unit_quote="K",
        )
        with pytest.raises(DatasetProducerError, match="role=CONSTANT"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=(constant_spec,),
            )

    def test_ungroundable_quote_propagates(self, tmp_path: Path) -> None:
        """A spec whose quote is not in the text fails the whole production --
        the producer never falls back to a hand-supplied or guessed offset."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        bad_spec = MeasurementSpec(
            axis_id="pressure",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.PRESSURE,
            label_quote="pressure",
            value_quote="101325",
            unit_quote="Pa",
        )
        with pytest.raises(QuoteGroundingError, match="not found"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=(bad_spec, *_SPECS),
            )

    def test_refuses_corrupted_raw_bin(self, tmp_path: Path) -> None:
        """P1-B: raw.bin was never verified before this fix -- only
        extracted.json was. Corrupt raw.bin (leave extracted.json/meta.json
        untouched) and confirm production now refuses. round-36: the
        shallow ``verify_artifact(deep=False)`` integrity check now runs
        before the deep=True call, so it is the one that catches raw.bin
        corruption and its "failed integrity verification" message is what
        fires, not the deep-check's "failed verify_artifact" message."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        raw_path.write_bytes(b"not the original bytes at all")

        with pytest.raises(DatasetProducerError, match="failed integrity verification"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_meta_sha256_field_disagreeing_with_directory(self, tmp_path: Path) -> None:
        """P1-B: ``load_artifact_meta`` resolves the directory purely from the
        ``sha256`` PARAMETER and never cross-checks it against the ``sha256``
        FIELD recorded inside meta.json. Hand-edit that field to some other
        (still well-formed) digest, leaving raw.bin/extracted.json under the
        TRUE sha256 directory untouched -- both would otherwise verify fine,
        so only an explicit field-vs-parameter check can catch this."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        meta = StoredArtifact.model_validate_json(meta_path.read_text())
        tampered = meta.model_copy(update={"sha256": "1" * 64})
        meta_path.write_text(tampered.model_dump_json())

        with pytest.raises(DatasetProducerError, match="disagree"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_stale_derivation_binding(self, tmp_path: Path) -> None:
        """Hand-edit ``derivation_binding`` so it no longer recomputes from
        meta.json's own extractor_version/sha256/extracted_sha256, leaving
        extracted_sha256 (and everything on disk) untouched -- raw.bin and
        extracted.json both still verify fine, so only the deep
        derivation_binding re-check (``verify_artifact(deep=True)``) can
        catch this. This is the entire point of the deep=True flip: this
        test MUST fail if that flip is reverted to deep=False."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        meta = StoredArtifact.model_validate_json(meta_path.read_text())
        assert meta.derivation_binding is not None
        tampered = meta.model_copy(update={"derivation_binding": "0" * 64})
        assert tampered.derivation_binding != meta.derivation_binding
        meta_path.write_text(tampered.model_dump_json())

        with pytest.raises(DatasetProducerError, match="failed verify_artifact"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_legacy_artifact_missing_derivation_binding(self, tmp_path: Path) -> None:
        """A legacy artifact stored after extracted_sha256 existed but before
        derivation_binding did has extracted_sha256 set and
        derivation_binding=None -- refused with a message naming that
        specific legacy cause, distinguishable both from the
        extracted_sha256-is-None legacy message and from a stale-binding
        refusal."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        meta = StoredArtifact.model_validate_json(meta_path.read_text())
        assert meta.extracted_sha256 is not None
        tampered = meta.model_copy(update={"derivation_binding": None, "extractor_version": None})
        meta_path.write_text(tampered.model_dump_json())

        with pytest.raises(DatasetProducerError, match="predates derivation_binding"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_refuses_a_legacy_artifact_that_is_also_corrupt_by_naming_the_integrity_failure(
        self, tmp_path: Path
    ) -> None:
        """round-36: an artifact that is BOTH legacy (predates extracted_sha256,
        so it would also hit the legacy carve-out) AND corrupt (raw.bin no
        longer hashes to sha256) must be refused as CORRUPT -- the shallow
        verify_artifact(deep=False) integrity check now runs BEFORE the
        legacy carve-outs, precisely so this case is never misreported as
        routine legacy handling. Before this fix, the legacy carve-out ran
        first and this same scenario raised "predates extracted_sha256"
        instead, hiding the real integrity failure."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        meta = StoredArtifact.model_validate_json(meta_path.read_text())
        assert meta.extracted_sha256 is not None
        legacy = meta.model_copy(update={"extracted_sha256": None, "derivation_binding": None})
        meta_path.write_text(legacy.model_dump_json())
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        raw_path.write_bytes(b"not the original bytes at all")

        with pytest.raises(DatasetProducerError, match="failed integrity verification"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )


class TestProducerNodeKind:
    """P1-C: the root SourceNode's ``kind`` must be derived honestly from the
    artifact's own ``content_type``, never hardcoded, and refused when the
    content_type establishes nothing."""

    def test_pdf_artifact_yields_paper_pdf_node(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _TEXT, content_type="application/pdf")
        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        assert envelope.source_graph.nodes[0].kind == SourceNodeKind.PAPER_PDF

    def test_xml_artifact_yields_jats_xml_node(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(
            tmp_path, _TEXT, content_type="application/xml", extractor="xml"
        )
        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        assert envelope.source_graph.nodes[0].kind == SourceNodeKind.JATS_XML

    def test_unrecognised_content_type_is_refused(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(
            tmp_path, _TEXT, content_type="text/html", extractor="html"
        )
        with pytest.raises(DatasetProducerError, match="does not map to any SourceNodeKind"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )


class TestProducerRefusesLossyExtraction:
    """A ``lossy=True`` extraction (missing pages, a parse failure, or plain
    truncation) must never reach ``ground_quote`` here: unlike the full
    ``carmel.services.grounding`` gate, ``ground_quote`` is a bare substring
    search with no ``unreadable_reason``/ARTIFACT_DEGRADED equivalent of its
    own, so a partial document could otherwise silently ground a measurement
    against text the extractor already knows is incomplete."""

    def test_lossy_extraction_is_refused(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _TEXT, lossy=True)
        with pytest.raises(DatasetProducerError, match="extracted lossily"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=_SPECS,
            )

    def test_non_lossy_extraction_still_succeeds(self, tmp_path: Path) -> None:
        """Regression guard: the new refusal must not fire on ordinary, fully
        successful extractions -- the vast majority of stored artifacts."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT, lossy=False)
        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        assert envelope.source_graph.nodes[0].kind == SourceNodeKind.PAPER_PDF


class TestProducerGlyphHealthQuarantine:
    """P1-D: normalization must not bypass the document-level dash/ligature
    corruption quarantine by hardcoding a healthy glyph-health/OPERATOR_RAW
    context. A document that (a) came from the PDF extractor (so
    SourceContext.FLAT_PDF_TEXT applies) and (b) has no en-dash but does have
    a bare lowercase-``e`` exponent shape (``assess_glyph_health``'s
    ``suspects_dash_corruption`` signal) must have that bare-exponent value
    quote REFUSED -- where, before this fix, the producer's hardcoded
    OPERATOR_RAW/healthy computation would have let it sail through."""

    _SUSPECT_TEXT = (
        "The reactor was held at a temperature of 300 K. The rate constant k "
        "was found to be 1023e5 1/s in the flattened region of the reactor."
    )
    """No en-dash anywhere, and "1023e5" is exactly the bare lowercase-e
    exponent shape `_BARE_DASH_CORRUPTION_RE` (`\\d+e\\d+`) matches -- so
    `assess_glyph_health` flags `suspects_dash_corruption=True` for this
    document. `extractor="pdf:pypdf"` (via `_store_synthetic_artifact`'s
    default) makes `_source_context_for`'s equivalent derive FLAT_PDF_TEXT.
    The temperature sentence exists only to give the envelope a COORDINATE
    axis (schema requires at least one of each); it is unrelated to the
    suspect rate-constant value under test."""

    def test_bare_exponent_value_refused_in_suspect_pdf_document(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, self._SUSPECT_TEXT)
        coordinate_spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="300",
            unit_quote="K",
        )
        spec = MeasurementSpec(
            axis_id="rate_constant",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.STRAIN_RATE,
            label_quote="rate constant",
            value_quote="1023e5",
            unit_quote="1/s",
        )
        with pytest.raises(DatasetProducerError, match="1023e5"):
            produce_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                measurements=(coordinate_spec, spec),
            )


class TestGroundingIsIndependentPerQuote:
    """P1-F (documented, deliberately NOT fixed by this change): value, unit,
    and label quotes for one axis are each grounded INDEPENDENTLY anywhere in
    the document -- nothing checks they were stated together. This is a
    pinning test of that CURRENT, UNSOUND behaviour: it must FAIL the day a
    bounded measurement-context check closes the gap, forcing this test (and
    the docstring paragraph in ``produce_envelope_from_artifact`` describing
    the same gap) to be updated or removed."""

    def test_value_and_unit_from_unrelated_parts_of_document_both_ground(
        self, tmp_path: Path
    ) -> None:
        text = (
            "The reactor was held at a temperature of 1023 K during the run. "
            "In an unrelated paragraph about SI units, Pa was mentioned as the "
            "SI unit of static gauge reading."
        )
        stored = _store_synthetic_artifact(tmp_path, text)
        # "1023" comes from the temperature sentence; "Pa" comes from the
        # unrelated SI-units sentence. Nothing binds them to the same
        # physical statement -- yet this spec grounds and validates cleanly.
        coordinate_spec = MeasurementSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_quote="temperature",
            value_quote="1023",
            unit_quote="K",
        )
        spec = MeasurementSpec(
            axis_id="pressure",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.PRESSURE,
            label_quote="static gauge reading",
            value_quote="1023",
            unit_quote="Pa",
        )
        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=(coordinate_spec, spec),
        )
        # Reachable and schema-valid today: this IS the gap P1-F describes.
        assert envelope.series[0].points[0].observations[0].axis_id == "pressure"


class TestActiveTableBindingTracksItsArgument:
    """``_ActiveTableBinding.derive`` must be a PURE function of the table it
    is given -- every derived artifact must move when the table itself
    differs, never stay pinned to whatever table production happens to use.

    A regression where one derived artifact silently kept reading
    ``units.TABLE_V1`` directly (instead of the ``table`` argument actually
    passed to ``derive``) would be invisible from ``_ACTIVE`` alone, because
    production always derives from ``TABLE_V1`` -- the only way to catch it
    is to derive from a table that is genuinely DIFFERENT and check that the
    difference is visible in every derived field.
    """

    def test_derive_from_a_different_table_moves_every_derived_artifact(self) -> None:
        import dataclasses

        from carmel.services import units
        from carmel.services.dataset_producer import _ActiveTableBinding

        # A table that differs from TABLE_V1 by one EXTRA alias -- a new raw
        # spelling for PRESSURE's base unit "Pa" that TABLE_V1 does not
        # register. This is a genuine difference in the table's ALIASES (and
        # therefore its spelling vocabulary and content-address), not merely
        # a version-string difference that a lazy/broken derive() could pass
        # through unnoticed.
        extra_alias = units.UnitAlias(
            quantity=QuantityKind.PRESSURE, raw="Pa_test_variant", normalized="Pa"
        )
        # A SECOND extra alias whose raw spelling contains whitespace. Without
        # this one, check 4 below is vacuous: `Pa_test_variant` has no
        # whitespace at all, so `active_other.whitespace_patterns` and
        # `active_default.whitespace_patterns` would come out IDENTICAL (both
        # empty of this alias) even if `whitespace_patterns` were silently
        # computed from TABLE_V1's own spellings_union instead of the
        # `table` argument actually passed to `derive` -- the exact
        # regression this test class exists to catch. A whitespace-containing
        # spelling is required to make that regression move a value this
        # test actually reads.
        extra_whitespace_alias = units.UnitAlias(
            quantity=QuantityKind.PRESSURE, raw="Pa test variant", normalized="Pa"
        )
        assert extra_alias not in units.TABLE_V1.aliases
        assert extra_whitespace_alias not in units.TABLE_V1.aliases
        other_table = dataclasses.replace(
            units.TABLE_V1,
            table_id="carmel-unit-conversions-test-variant",
            aliases=units.TABLE_V1.aliases + (extra_alias, extra_whitespace_alias),
        )
        assert other_table.sha256 != units.TABLE_V1.sha256

        active_default = _ActiveTableBinding.derive(units.TABLE_V1)
        active_other = _ActiveTableBinding.derive(other_table)

        # 1. The bound table itself moved.
        assert active_other.table is other_table
        assert active_other.table.sha256 != active_default.table.sha256

        # 2. The new raw spelling is now in PRESSURE's spelling vocabulary,
        #    and was not before.
        assert "Pa_test_variant" not in active_default.spellings_by_quantity[QuantityKind.PRESSURE]
        assert "Pa_test_variant" in active_other.spellings_by_quantity[QuantityKind.PRESSURE]

        # 3. The union vocabulary (used for Layer-3 maximality regardless of
        #    claimed quantity) moved too.
        assert "Pa_test_variant" not in active_default.spellings_union
        assert "Pa_test_variant" in active_other.spellings_union

        # 4. whitespace_patterns is keyed only by whitespace-containing
        #    spellings; confirm each binding's key set is computed from its
        #    OWN (different) spellings_union rather than shared/cached state.
        assert set(active_other.whitespace_patterns) == {
            s for s in active_other.spellings_union if any(ch.isspace() for ch in s)
        }
        assert set(active_default.whitespace_patterns) == {
            s for s in active_default.spellings_union if any(ch.isspace() for ch in s)
        }
        # 4b. The two key sets must actually DIFFER -- otherwise a
        #     `whitespace_patterns` computation silently pinned to
        #     TABLE_V1's own spellings_union (ignoring the `table` argument
        #     `derive` was actually given) would pass 4 above undetected,
        #     since both bindings would independently "match themselves".
        assert "Pa test variant" not in active_default.whitespace_patterns
        assert "Pa test variant" in active_other.whitespace_patterns
        assert set(active_other.whitespace_patterns) != set(active_default.whitespace_patterns)

        # 5. The embedded table's recorded sha256 and canonical_json moved,
        #    and each binding's own embedded table matches its own bound
        #    table -- never the other one's.
        assert active_other.embedded.sha256 == other_table.sha256
        assert active_other.embedded.sha256 != active_default.embedded.sha256
        assert active_default.embedded.sha256 == units.TABLE_V1.sha256
