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
    CharSpanLocator,
    DatasetEnvelope,
    ExtractionBinding,
    SourceNodeKind,
    ValueOrigin,
    iter_measured_values,
    iter_source_refs,
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
from carmel.services.dataset_store import compute_dataset_sha
from carmel.services.evidence import artifact_dir, load_artifact_meta, store_artifact
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
        text=text, normalized=text.casefold(), sections=[], extractor=extractor, lossy=False
    )
    return store_artifact(workspace_root, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)


def _independently_verified_text(workspace_root: Path, raw_sha256: str) -> str:
    """Replayer-side read: re-read extracted.json's BYTES from disk, re-verify
    their digest against StoredArtifact.extracted_sha256, and only then parse.

    Deliberately independent of the producer's own loading code: it re-reads
    the file and recomputes the digest here, so a producer bug cannot vouch
    for itself.
    """
    meta = load_artifact_meta(workspace_root, raw_sha256)
    assert meta is not None, f"no stored artifact under {raw_sha256!r}"
    assert meta.extracted_sha256 is not None
    raw_bytes = (artifact_dir(workspace_root, raw_sha256) / "extracted.json").read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == meta.extracted_sha256, (
        "extracted.json bytes on disk do not match the digest recorded at store time"
    )
    extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    return extracted.text


def _assert_every_char_span_grounds(envelope: DatasetEnvelope, text: str) -> int:
    """The replayer-style check: walk the ENTIRE envelope generically and
    assert every reachable ``CharSpanLocator`` slices ``text`` to exactly the
    verbatim string the envelope claims for it.

    Three claim kinds exist in this schema, each walked generically (never
    "the one span the test knows about"):

    - every ``MeasuredValue``'s ``value_ref`` span must slice to its
      ``raw_text`` (found via ``iter_measured_values``, the schema's own
      shape-agnostic walker);
    - every ``MeasuredValue``'s ``unit_ref`` span must slice to its
      ``unit_raw``;
    - every ``AxisDeclaration``'s ``label_ref`` span must slice to its
      ``label_raw``.

    Coverage is then proven, not assumed: the number of spans checked above
    must equal the TOTAL number of ``CharSpanLocator``-bearing ``SourceRef``s
    reachable via ``iter_source_refs`` -- so a char-span ref hanging anywhere
    this pairing does not know about fails the test loudly instead of
    escaping it.

    Returns the number of spans checked (callers assert it is non-zero).
    """
    checked = 0
    for path, value in iter_measured_values(envelope):
        for which, ref, expected in (
            ("value_ref", value.value_ref, value.raw_text),
            ("unit_ref", value.unit_ref, value.unit_raw),
        ):
            locator = ref.locator
            if isinstance(locator, CharSpanLocator):
                actual = text[locator.start : locator.end]
                assert actual == expected, (
                    f"replay mismatch at {path}.{which}: text[{locator.start}:{locator.end}] == "
                    f"{actual!r}, but the envelope claims {expected!r}"
                )
                checked += 1
    for series in envelope.series:
        for axis in series.axes:
            locator = axis.label_ref.locator
            if isinstance(locator, CharSpanLocator):
                actual = text[locator.start : locator.end]
                assert actual == axis.label_raw, (
                    f"replay mismatch at series {series.series_id!r} axis {axis.axis_id!r} label_ref: "
                    f"text[{locator.start}:{locator.end}] == {actual!r}, but the envelope claims "
                    f"{axis.label_raw!r}"
                )
                checked += 1
    total_char_span_refs = sum(
        1 for _, ref in iter_source_refs(envelope) if isinstance(ref.locator, CharSpanLocator)
    )
    assert checked == total_char_span_refs, (
        f"replayer checked {checked} char-span ref(s) but iter_source_refs finds "
        f"{total_char_span_refs}; some CharSpanLocator is reachable that this pairing never verified"
    )
    return checked


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


class TestGroundQuote:
    def test_grounds_unique_quote_by_search(self) -> None:
        locator = ground_quote(_TEXT, "1023")
        assert _TEXT[locator.start : locator.end] == "1023"
        # Derived by SEARCH, not asserted: the span really is where "1023"
        # sits in this sentence.
        assert locator.start == _TEXT.index("1023")

    def test_quote_absent_from_text_raises(self) -> None:
        with pytest.raises(QuoteGroundingError, match="not found"):
            ground_quote(_TEXT, "774 K")

    def test_ambiguous_quote_raises_and_states_match_count(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote(text, "1023")

    def test_empty_quote_raises_with_specific_message(self) -> None:
        with pytest.raises(QuoteGroundingError, match="quote is empty"):
            ground_quote(_TEXT, "")

    def test_occurrence_out_of_range_raises_and_states_match_count(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        with pytest.raises(QuoteGroundingError, match=r"2 match\(es\) found"):
            ground_quote(text, "1023", occurrence=2)

    def test_explicit_occurrence_selects_that_match(self) -> None:
        text = "measured at 1023 K, then again at 1023 K"
        first = ground_quote(text, "1023", occurrence=0)
        second = ground_quote(text, "1023", occurrence=1)
        assert first.start == text.index("1023")
        assert second.start == text.index("1023", first.start + 1)
        assert text[second.start : second.end] == "1023"

    def test_overlapping_matches_count_as_ambiguous(self) -> None:
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote("aaa", "aa")


class TestGroundQuoteNumericTokenMaximality:
    """P1-A: a numeral quote must ground to the MAXIMAL numeric token, never
    an interior fragment of a strictly larger one."""

    def test_rejects_fragment_inside_larger_integer(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("T = 11023 K", "1023")

    def test_rejects_fragment_before_decimal_point(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("T = 1023.5 K", "1023")

    def test_rejects_fragment_after_decimal_point(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("phi 0.51023 was used", "1023")

    def test_accepts_whole_number_bounded_by_whitespace(self) -> None:
        locator = ground_quote("T = 1023 K", "1023")
        assert (locator.start, locator.end) == (4, 8)

    def test_accepts_number_immediately_followed_by_unit_letter(self) -> None:
        # Deliberate policy: a trailing letter that isn't part of an
        # exponent marker (e/E) does not continue a numeric token, so
        # "1023" IS the maximal numeral in "1023K" -- unlike a trailing
        # digit or decimal point, "K" cannot extend a numeral.
        locator = ground_quote("1023K", "1023")
        assert (locator.start, locator.end) == (0, 4)

    def test_non_numeric_quote_is_unaffected_by_maximality_check(self) -> None:
        # "mole fraction" never matches the numeric-token pattern at all, so
        # the maximality check must not even engage for it.
        locator = ground_quote("the mole fraction was measured", "mole fraction")
        assert (locator.start, locator.end) == (4, 17)

    def test_rejects_fragment_of_exponent_form(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("k = 1023e5 1/s", "1023")


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
            ground_quote("T = -1.5 K was used", "1.5")

    def test_accepts_signed_value_as_the_maximal_candidate(self) -> None:
        locator = ground_quote("T = -1.5 K was used", "-1.5")
        assert (locator.start, locator.end) == (4, 8)

    def test_rejects_exponent_fragment_lacking_sign_in_old_regex(self) -> None:
        # Old bug: `_quote_looks_numeric("-3")` was False (no sign in the old
        # grammar), so the maximality check was skipped entirely and an
        # exponent fragment grounded as if it were a standalone value.
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("k = 1.0e-3 cm3/mol/s", "-3")

    def test_rejects_species_subscript_as_sits_inside_identifier(self) -> None:
        # Old bug: a bare digit subscript of a species name ("H2") grounded
        # as if it were a measured value.
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("Fuel H2 was used", "2")

    def test_rejects_unit_digit_as_sits_inside_identifier(self) -> None:
        # "cm3" -- the "3" is a unit power, not a numeral in its own right.
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("k = 1.0e-5 cm3/mol/s", "3")

    def test_rejects_thousands_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("T = 1,023 K", "023")

    def test_rejects_thousands_leading_digit_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="does not sit at a clean numeral boundary"):
            ground_quote("T = 1,023 K", "1")

    def test_rejects_cas_number_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("ethanol 64-17-5 was used", "17")

    def test_rejects_range_endpoint_fragment(self) -> None:
        with pytest.raises(QuoteGroundingError, match="interior fragment"):
            ground_quote("over 1000-1200 K", "1200")

    def test_accepts_whole_range_as_one_candidate(self) -> None:
        locator = ground_quote("over 1000-1200 K", "1000-1200")
        assert (locator.start, locator.end) == (5, 14)

    def test_accepts_plain_integer_unaffected(self) -> None:
        locator = ground_quote("T = 1023 K", "1023")
        assert (locator.start, locator.end) == (4, 8)

    def test_accepts_negative_value_full_precision(self) -> None:
        text = "T = -1.5 K was used"
        locator = ground_quote(text, "-1.5")
        assert text[locator.start : locator.end] == "-1.5"

    def test_accepts_number_glued_to_unit_letter(self) -> None:
        # A glued unit must stay groundable: letters are not part of a
        # numeral, so they never extend or block a candidate on their own.
        locator = ground_quote("at 1023K nominal", "1023")
        assert (locator.start, locator.end) == (3, 7)

    def test_non_numeral_unit_quote_unaffected(self) -> None:
        locator = ground_quote("T = 1023 K", "K")
        assert (locator.start, locator.end) == (9, 10)

    def test_non_numeral_label_quote_unaffected(self) -> None:
        locator = ground_quote("the mole fraction of CO", "mole fraction")
        assert (locator.start, locator.end) == (4, 17)

    def test_ambiguity_is_checked_before_the_numeral_check_and_still_refuses(self) -> None:
        # The new numeral check must never silently resolve an ambiguous
        # quote: ambiguity is still detected and raised BEFORE the numeral
        # check ever runs, even though "-3" now fullmatches the numeral
        # candidate grammar (unlike under the old, sign-less grammar).
        text = "k1 = 1.0e-3 cm3/mol/s, k2 = 2.0e-3 cm3/mol/s"
        with pytest.raises(QuoteGroundingError, match=r"appears 2 times"):
            ground_quote(text, "-3")


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
            ground_quote("The temperature was 307 6 10 K.", "6")

    def test_rejects_the_uncertainty_value_not_the_measurement(self) -> None:
        with pytest.raises(QuoteGroundingError, match="ascii6_uncertainty"):
            ground_quote("The temperature was 307 6 10 K.", "10")

    def test_rejects_spaced_range_endpoint_ascii_hyphen(self) -> None:
        with pytest.raises(QuoteGroundingError, match="spaced_range"):
            ground_quote("The range was 1000 - 1200 K.", "1200")

    def test_rejects_spaced_range_endpoint_en_dash(self) -> None:
        with pytest.raises(QuoteGroundingError, match="spaced_range"):
            ground_quote("The range was 1000 – 1200 K.", "1200")

    def test_rejects_flattened_scientific_notation_exponent(self) -> None:
        with pytest.raises(QuoteGroundingError, match="flattened_scientific"):
            ground_quote("k = 3.94 x 10 03 s-1.", "03")

    def test_rejects_flattened_scientific_notation_base(self) -> None:
        with pytest.raises(QuoteGroundingError, match="flattened_scientific"):
            ground_quote("k = 3.94 x 10 03 s-1.", "10")

    def test_ascii6_uncertainty_and_spaced_range_messages_are_distinct(self) -> None:
        # Anti-masking-bug regression: the two refusals must be tellable apart
        # by MESSAGE CONTENT, not merely by both being QuoteGroundingError.
        try:
            ground_quote("The temperature was 307 6 10 K.", "6")
        except QuoteGroundingError as exc:
            ascii6_message = str(exc)
        try:
            ground_quote("The range was 1000 - 1200 K.", "1200")
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
            ground_quote("over 1000-1200 K", "1200")

    def test_whole_tight_range_still_grounds(self) -> None:
        locator = ground_quote("over 1000-1200 K", "1000-1200")
        assert (locator.start, locator.end) == (5, 14)

    def test_plain_integer_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = 1023 K", "1023")
        assert (locator.start, locator.end) == (4, 8)

    def test_negative_value_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = -1.5 K was used", "-1.5")
        assert (locator.start, locator.end) == (4, 8)

    def test_temperature_glued_to_unit_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("at 1023K nominal", "1023")
        assert (locator.start, locator.end) == (3, 7)

    def test_unit_letter_quote_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("T = 1023 K", "K")
        assert (locator.start, locator.end) == (9, 10)

    def test_label_quote_unaffected_by_the_new_guard(self) -> None:
        locator = ground_quote("the mole fraction of CO", "mole fraction")
        assert (locator.start, locator.end) == (4, 17)

    def test_whole_quote_spaced_range_still_grounds_via_plain_substring_search(self) -> None:
        # The whole-quote spaced range contains internal whitespace, so it
        # never fullmatches NUMERAL_CANDIDATE_RE and the entire numeral-guard
        # block (including this new check) is skipped: it grounds via plain
        # substring search, exactly as before this guard existed. This test
        # pins that capability is NOT lost by the new guard.
        text = "The range was 1000 - 1200 K."
        quote = "1000 - 1200"
        locator = ground_quote(text, quote)
        assert text[locator.start : locator.end] == quote


class TestProducerEndToEnd:
    def test_produce_store_load_replay(self, tmp_path: Path) -> None:
        """The whole vertical slice: real store_artifact -> producer ->
        DatasetEnvelope -> store_dataset_envelope -> load_dataset_envelope ->
        independent replayer-style verification of every grounded span."""
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

        # Replayer-style check: independently re-read + re-verify + re-parse
        # the stored extraction, then generically verify every span in the
        # LOADED envelope against it.
        replayed_text = _independently_verified_text(tmp_path, stored_artifact.sha256)
        checked = _assert_every_char_span_grounds(loaded, replayed_text)
        # 2 axes x (value_ref + unit_ref + label_ref) = 6 char-span refs.
        assert checked == 6

        # The loaded binding's extracted_text_sha256 must equal a digest
        # recomputed here from the independently re-read, re-verified text.
        node = loaded.source_graph.node("paper")
        binding = node.extraction
        assert isinstance(binding, ExtractionBinding), "extraction must be present, not Absent"
        assert binding.extracted_text_sha256 == hashlib.sha256(replayed_text.encode("utf-8")).hexdigest()
        assert binding.extracted_sha256 == stored_artifact.extracted_sha256

    def test_replay_fails_against_single_character_mutation(self, tmp_path: Path) -> None:
        """THE non-vacuousness proof for the replayer check: an envelope
        grounded against the ORIGINAL text must FAIL the replay when checked
        against a second stored artifact whose text differs by exactly one
        character inside a grounded span ("1023 K" -> "1024 K")."""
        original = _store_synthetic_artifact(tmp_path, _TEXT)
        mutated = _store_synthetic_artifact(tmp_path, _MUTATED_TEXT)
        assert original.sha256 != mutated.sha256
        assert len(_TEXT) == len(_MUTATED_TEXT)

        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=original.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )

        # Sanity: the SAME check passes against the original artifact's text...
        original_text = _independently_verified_text(tmp_path, original.sha256)
        assert _assert_every_char_span_grounds(envelope, original_text) == 6

        # ...and fails against the mutated artifact's (independently
        # verified) text, on the exact claimed-vs-actual slice disagreement.
        mutated_text = _independently_verified_text(tmp_path, mutated.sha256)
        with pytest.raises(AssertionError, match="replay mismatch"):
            _assert_every_char_span_grounds(envelope, mutated_text)


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
        tests in ``TestProducerFailClosed``."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        corrupt = ExtractedText(
            text="tampered", normalized="tampered", sections=[], extractor="pdf:pypdf", lossy=False
        )
        extracted_path.write_bytes(corrupt.model_dump_json().encode("utf-8"))

        with pytest.raises(DatasetProducerError, match="failed verify_artifact"):
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
        untouched) and confirm production now refuses."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        raw_path.write_bytes(b"not the original bytes at all")

        with pytest.raises(DatasetProducerError, match="failed verify_artifact"):
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
