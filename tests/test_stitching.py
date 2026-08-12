"""Tests for the span-stitching refutation gate (P0-d).

The gate REFUTES; it never verifies. Nothing here may assert that a ``None``
return means a claim is true -- only that a specific, named refutation did not
fire. A test written the other way would re-open by assertion the hole the
module closes by construction.
"""

from __future__ import annotations

import pytest

from carmel.services.dataset_producer import _ACTIVE, binding_for_known_sha
from carmel.services.stitching import (
    iter_unit_bearing_numerals,
    refute_stitched_scalar,
    unit_spellings,
)
from carmel.services.units import TABLE_V1, TABLES_BY_SHA, QuantityKind

_SENTENCE = "The initial temperature was 823 K and the pressure was held at 1.2 atm"


def _at(quote: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = _SENTENCE.index(quote, start + 1)
    return (start, start + len(quote))


class TestTheVocabularyIsTheProducersOwn:
    """The gate must read units through the SAME vocabulary the admission gate
    admits them with. A second, separately-maintained list is the drift Codex
    named in round 95: a spelling in one and not the other lets a claim pass
    one check and fail another, with no way to tell which is right."""

    def test_every_admitted_spelling_is_visible_to_the_gate(self) -> None:
        gate = unit_spellings(TABLE_V1)
        missing = {
            spelling
            for spellings in _ACTIVE.spellings_by_quantity.values()
            for spelling in spellings
            if spelling not in gate
        }
        assert not missing

    def test_the_gate_invents_no_spelling_the_producer_rejects(self) -> None:
        gate = set(unit_spellings(TABLE_V1))
        assert not gate - set(_ACTIVE.spellings_union)

    def test_the_agreement_holds_for_every_shipped_table_not_just_the_current_one(
        self,
    ) -> None:
        """Comparing only TABLE_V1 against the CURRENT binding proves the two
        agree today. A future TABLE_V2 is exactly when they could diverge, and
        a replayed claim resolves its own table by sha -- so the property must
        hold for every table the registry ships. Codex round 96."""
        for sha, table in TABLES_BY_SHA.items():
            binding = binding_for_known_sha(sha)
            assert binding is not None
            gate = unit_spellings(table)
            assert set(gate) == set(binding.spellings_union)

    def test_a_spelling_maps_to_every_quantity_it_can_denote(self) -> None:
        """A set, not one kind: real spellings collide, and collapsing that
        would be the module inventing a fact."""
        for quantity, spellings in _ACTIVE.spellings_by_quantity.items():
            for spelling in spellings:
                assert quantity in unit_spellings(TABLE_V1)[spelling]


class TestTheWindowIsDerivedNotChosen:
    def test_the_stitched_claim_is_refuted(self) -> None:
        refutation = refute_stitched_scalar(
            _SENTENCE,
            label_span=_at("pressure"),
            value_span=_at("823"),
            unit_span=_at("atm"),
            quantity_kind=QuantityKind.PRESSURE,
        )
        assert refutation is not None
        assert "823 K" in refutation.found
        assert "1.2 atm" in refutation.found

    def test_a_label_from_another_quantity_is_refuted(self) -> None:
        """The sibling fabrication. Its window holds exactly one PRESSURE pair,
        so a same-kind-only rule would bless it -- which is why the rule counts
        constructs of ANY kind."""
        assert (
            refute_stitched_scalar(
                _SENTENCE,
                label_span=_at("temperature"),
                value_span=_at("1.2"),
                unit_span=_at("atm"),
                quantity_kind=QuantityKind.PRESSURE,
            )
            is not None
        )

    @pytest.mark.parametrize(
        ("label", "value", "unit", "kind"),
        [
            ("pressure", "1.2", "atm", QuantityKind.PRESSURE),
            ("temperature", "823", "K", QuantityKind.TEMPERATURE),
        ],
    )
    def test_both_honest_readings_survive(self, label: str, value: str, unit: str, kind: QuantityKind) -> None:
        """A gate that also killed these would be worthless at any yield. Note
        what this does NOT assert: that either claim is true."""
        assert (
            refute_stitched_scalar(
                _SENTENCE,
                label_span=_at(label),
                value_span=_at(value),
                unit_span=_at(unit),
                quantity_kind=kind,
            )
            is None
        )

    def test_a_declared_kind_the_located_unit_contradicts_is_refuted(self) -> None:
        """Checked against the LOCATED unit, never against the label -- which
        is why no label lexicon is needed, and none exists."""
        refutation = refute_stitched_scalar(
            _SENTENCE,
            label_span=_at("pressure"),
            value_span=_at("1.2"),
            unit_span=_at("atm"),
            quantity_kind=QuantityKind.TEMPERATURE,
        )
        assert refutation is not None
        assert "temperature" in refutation.reason

    def test_offsets_decide_rather_than_the_text_they_spell(self) -> None:
        """Two occurrences of the same numeral are different groundings. A
        string comparison would let one stand in for the other."""
        text = "the pressure was 5 atm, and later the pressure was 5 atm"
        second = text.index("5", text.index("5") + 1)
        refutation = refute_stitched_scalar(
            text,
            label_span=(4, 12),
            value_span=(second, second + 1),
            unit_span=(text.index("atm"), text.index("atm") + 3),
            quantity_kind=QuantityKind.PRESSURE,
        )
        assert refutation is not None


class TestWhatCountsAsAUnitBearingNumeral:
    def test_a_newline_does_not_bind_a_unit_to_a_numeral(self) -> None:
        """Extracted PDF text wraps mid-sentence, so a numeral ending one line
        has no reliable relationship to a token starting the next."""
        text = "the pressure was 5\natm"
        found = list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text)))
        assert found == []

    @pytest.mark.parametrize("gap", ["\u00a0", "\u2009", " ", "\t", "  "])
    def test_intra_line_whitespace_binds_a_unit_to_its_numeral(self, gap: str) -> None:
        """PDF extraction emits NBSP and thin space between a number and its
        unit. Reading those as "no construct" refuses honest claims -- a
        fail-closed direction, but the wrong one: they ARE bound in the source.
        Found by adversarial review (Codex round 96)."""
        text = f"the pressure was 1.2{gap}atm"
        (only,) = list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text)))
        assert (only.value_text, only.unit_text) == ("1.2", "atm")

    @pytest.mark.parametrize("gap", ["\n", "\r", "\u2028", "\u0085"])
    def test_a_line_separator_still_does_not_bind(self, gap: str) -> None:
        """The distinction the widening must preserve: extracted text wraps
        mid-sentence, so a numeral ending one line has no reliable relationship
        to a token beginning the next."""
        text = f"the pressure was 1.2{gap}atm"
        assert list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text))) == []

    def test_a_unit_that_is_only_a_prefix_of_a_longer_word_is_not_a_unit(self) -> None:
        found = list(
            iter_unit_bearing_numerals("held for 5 minutes", window_start=0, window_end=len("held for 5 minutes"))
        )
        assert found == []

    def test_a_slash_composite_yields_no_construct(self) -> None:
        """ "1.0/1.5 atm" states a PAIR. The numeral grammar folds hyphen ranges
        into one match but NOT slash composites, so the second endpoint used to
        look like the window's sole construct and a composite could be stored as
        a scalar. Codex round 96."""
        text = "the pressure was 1.0/1.5 atm"
        assert list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text))) == []

    @pytest.mark.parametrize(("text", "unit"), [("a rate of 40 1/s", "1/s"), ("a speed of 40 cm/s", "cm/s")])
    def test_a_unit_that_legitimately_contains_a_slash_still_binds(self, text: str, unit: str) -> None:
        """The guard must not swallow real units. "1/s" even contains a digit
        before the slash, which is the shape the composite check looks for."""
        (only,) = list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text)))
        assert only.unit_text == unit

    def test_the_longest_spelling_wins(self) -> None:
        text = "a speed of 40 cm/s"
        (only,) = list(iter_unit_bearing_numerals(text, window_start=0, window_end=len(text)))
        assert only.unit_text == "cm/s"

    def test_a_construct_straddling_the_window_edge_is_not_counted(self) -> None:
        """Half a construct is not a construct: counting it would let a window
        borrow meaning from text it does not cover."""
        text = "the pressure was 1.2 atm"
        found = list(iter_unit_bearing_numerals(text, window_start=0, window_end=text.index("atm") + 1))
        assert found == []


class TestAnUncheckableClaimIsNeverAPass:
    def test_a_window_with_no_construct_at_all_is_refuted(self) -> None:
        """Including the unmodelled-unit case: a unit this table does not know
        yields no construct, and that must fail closed. An incomplete table
        caps yield; it must never widen what is accepted."""
        text = "the pressure was 5 furlongs"
        refutation = refute_stitched_scalar(
            text,
            label_span=(4, 12),
            value_span=(text.index("5"), text.index("5") + 1),
            unit_span=(text.index("furlongs"), len(text)),
            quantity_kind=QuantityKind.PRESSURE,
        )
        assert refutation is not None
        assert "none at all" in refutation.reason
