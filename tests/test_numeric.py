# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure, I/O-free numeric reconstruction core.

carmel.services.numeric parses an ALREADY-SCOPED cell/span -- it never scans a whole
document for numbers. The load-bearing properties tested here: illegal float shapes
(``0.6e1.0``) are never salvaged; naively-legal-but-huge floats that overflow to
``inf`` are always refused rather than wrapped as a Scalar; a numeric-looking token
touching letters/``%``/formula fragments is refused; ``/C0`` and ``þ`` are decoded
ONLY in numeric sign/exponent position, and ONLY as an explicit, recorded repair,
never as a blanket strip/rewrite; ASCII ``e`` is NEVER repaired into a dash, even
though it is the exact corruption this module defends against; and the dash-corruption
quarantine rule is gated on FLAT_PDF_TEXT plus a document that actually looks
corrupted -- a healthy document, or a structured cell, must not trigger it.
"""

from __future__ import annotations

import math

from carmel.services.numeric import (
    REPAIR_NAMES,
    NormalizedNumeral,
    Range,
    Scalar,
    SourceContext,
    Unresolvable,
    assess_glyph_health,
    normalize_numeric_span,
    parse_numeric_span,
)

#: A GlyphHealth for a document that shows no sign of dash corruption at all: it
#: contains real en dashes and no bare digit-e-digit tokens.
HEALTHY = assess_glyph_health("intact ranges like 1–2 and 3–4 appear throughout this text.")

#: A GlyphHealth for a document that is a dead ringer for the Elsevier en-dash
#: substitution: zero en dashes, at least one bare lowercase digit-e-digit token.
SUSPECT = assess_glyph_health("no en dashes here, but a range like 0.6e1.0 shows up.")


class TestAssessGlyphHealth:
    def test_a_document_with_en_dashes_is_never_suspected_of_dash_corruption(self) -> None:
        health = assess_glyph_health("the range 1–2 spans two conditions, as does 3–4.")
        assert health.suspects_dash_corruption is False

    def test_a_document_with_no_en_dash_and_a_bare_digit_e_digit_token_is_suspected(self) -> None:
        health = assess_glyph_health("the ignition delay was 1e2 ms across the run.")
        assert health.suspects_dash_corruption is True

    def test_a_document_with_no_en_dash_and_no_bare_digit_e_digit_token_is_not_suspected(self) -> None:
        health = assess_glyph_health("plain prose with a normal number like 12 ms and no exponents.")
        assert health.suspects_dash_corruption is False

    def test_uppercase_e_scientific_notation_alone_does_not_trigger_suspicion(self) -> None:
        """Genuine scientific notation in the corrupt corpus always uses uppercase E;
        a document with no en dash but only an uppercase-E token must not be flagged."""
        health = assess_glyph_health("the rate constant was measured as 3.97E+12 in this run.")
        assert health.suspects_dash_corruption is False

    def test_glyph_health_records_the_thorn_plus_marker(self) -> None:
        health = assess_glyph_health("A value of 7.000Eþ17 was reported.")
        assert health.has_thorn_plus_marker is True

    def test_glyph_health_records_the_slash_c0_minus_marker(self) -> None:
        health = assess_glyph_health("7.000Eþ17 /C0 1.0 describes the fit.")
        assert health.has_slash_c0_minus_marker is True

    def test_glyph_health_records_the_ascii_6_uncertainty_marker(self) -> None:
        health = assess_glyph_health("the particle size was 307 6 10 nm.")
        assert health.has_ascii6_uncertainty_marker is True

    def test_glyph_health_does_not_flag_markers_absent_from_the_document(self) -> None:
        health = assess_glyph_health("nothing unusual here, just plain prose.")
        assert health.has_thorn_plus_marker is False
        assert health.has_slash_c0_minus_marker is False
        assert health.has_ascii6_uncertainty_marker is False

    def test_assess_glyph_health_never_mutates_or_returns_the_input_text(self) -> None:
        """GlyphHealth is a read-only signal, never a rewrite pass."""
        document = "no en dashes here, but a range like 0.6e1.0 shows up."
        before = document
        assess_glyph_health(document)
        assert document == before


class TestIllegalFloatShapesAreNeverSalvaged:
    def test_0_6e1_0_is_unresolvable_not_6_0_not_1e9(self) -> None:
        result = parse_numeric_span("0.6e1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert result.raw == "0.6e1.0"
        assert result.reason  # a real explanation, not a placeholder

    def test_1_6e2_0_is_unresolvable(self) -> None:
        result = parse_numeric_span("1.6e2.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_1e1_5_is_unresolvable(self) -> None:
        result = parse_numeric_span("1e1.5", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_0_1e0_2_is_unresolvable(self) -> None:
        result = parse_numeric_span("0.1e0.2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_illegal_float_shape_is_unresolvable_even_under_a_suspect_flat_pdf_context(self) -> None:
        """Case A is illegal regardless of GlyphHealth/SourceContext -- it must not be
        gated on the quarantine rule, which only concerns bare (dotless) exponents."""
        result = parse_numeric_span("0.6e1.0", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=SUSPECT)
        assert isinstance(result, Unresolvable)


class TestNonFiniteIsNeverAValue:
    def test_1000e3000_is_unresolvable_because_non_finite(self) -> None:
        result = parse_numeric_span("1000e3000", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert "non-finite" in result.reason or "inf" in result.reason.lower()

    def test_230e1000_is_unresolvable_because_non_finite(self) -> None:
        result = parse_numeric_span("230e1000", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_200e360_is_unresolvable_because_non_finite(self) -> None:
        result = parse_numeric_span("200e360", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_a_naive_parse_of_1000e3000_would_be_inf_but_this_module_never_returns_it(self) -> None:
        assert math.isinf(float("1000e3000"))  # pins the premise: Python's own float() overflows silently
        result = parse_numeric_span("1000e3000", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert not isinstance(result, Scalar)


class TestBoundaryAdjacencyEnforcement:
    def test_2e50_extracted_from_50_percent_h_2e50_percent_co_is_unresolvable(self) -> None:
        """The '%' before and the '%CO' immediately after 2e50 touch the numeric
        token with no whitespace -- a formula fragment, not a value."""
        result = parse_numeric_span("50%H 2e50%CO", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_a_clean_2e50_with_no_touching_letters_percent_or_formula_fragments_resolves(self) -> None:
        """With no adjacency contamination and a healthy (non-suspect) document, a
        bare 2e50 is legitimate scientific notation and must resolve to a Scalar --
        it is finite, so only the boundary and quarantine rules could refuse it, and
        neither applies here (clean span, HEALTHY glyph state)."""
        result = parse_numeric_span("2e50", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == 2e50

    def test_2e75_extracted_with_touching_letters_is_unresolvable(self) -> None:
        result = parse_numeric_span("X2e75", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)


class TestSlashC0Minus:
    def test_slash_c0_1_0_is_scalar_negative_one_with_the_repair_recorded(self) -> None:
        result = parse_numeric_span("/C0 1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == -1.0
        assert "slash_c0_to_minus" in result.repairs

    def test_blanket_deletion_of_slash_c0_without_recording_a_sign_flip_is_forbidden(self) -> None:
        """A cleanup pass that just deletes '/C0' (instead of converting it to a
        recorded minus-sign repair) would silently flip the sign of every negative
        exponent it touches: '/C0 1.0' would become '1.0' -> +1.0. Pin that this
        module produces the CORRECT negative value with the repair explicitly named,
        which a blanket-strip implementation could not produce."""
        result = parse_numeric_span("/C0 1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value != 1.0
        assert result.value == -1.0
        assert result.repairs != ()

    def test_slash_c0_applies_only_in_sign_position_bare_text_containing_it_elsewhere_is_unresolvable(
        self,
    ) -> None:
        """'/C0' is decoded only when it sits where a sign belongs; a span where it
        cannot be read as a leading sign for the whole token must not silently drop
        it either -- it fails the single-clean-numeral grammar and is refused."""
        result = parse_numeric_span("1.0/C0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)


class TestThornPlus:
    def test_7_000e_thorn_17_is_scalar_7e17_with_the_repair_recorded(self) -> None:
        result = parse_numeric_span("7.000Eþ17", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == 7.0e17
        assert "thorn_to_plus" in result.repairs

    def test_ascii_e_is_never_repaired_into_a_dash(self) -> None:
        """Case B's dangerous shape: 1000e3000 must never be rewritten into a range
        like 1000-3000. It must be Unresolvable (non-finite), and neither an ASCII
        hyphen nor a real en dash may appear anywhere in the outcome."""
        result = parse_numeric_span("1000e3000", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert not isinstance(result, Range)
        assert "-" not in result.reason.replace("non-finite", "").replace("-inf", "")
        for repair in result.repairs:
            assert "dash" not in repair
            assert "e_to_minus" not in repair
            assert "e_to_dash" not in repair


class TestUnicodeMinusAndLeadingEnDash:
    def test_unicode_minus_sign_1_0_is_scalar_negative_one_with_the_repair_recorded(self) -> None:
        result = parse_numeric_span("−1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == -1.0
        assert "unicode_minus_to_ascii" in result.repairs

    def test_leading_en_dash_1_0_is_scalar_negative_one_with_the_repair_recorded(self) -> None:
        result = parse_numeric_span("–1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == -1.0
        assert "leading_en_dash_to_minus" in result.repairs

    def test_mid_token_en_dash_range_is_still_a_range_not_a_leading_sign(self) -> None:
        """The leading-en-dash-as-sign repair must never disturb the existing
        mid-token en-dash-as-range-separator behavior: '1–2' still splits into the
        range [1.0, 2.0], since the en dash there is consumed by
        '_find_range_separator' before '_parse_single_value' ever sees either
        half."""
        result = parse_numeric_span("1–2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Range)
        assert result.low == 1.0
        assert result.high == 2.0


class TestRangeVsScalarVsExponentGrammar:
    def test_ascii_hyphen_range_1_2(self) -> None:
        result = parse_numeric_span("1-2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Range)
        assert result.low == 1.0
        assert result.high == 2.0

    def test_en_dash_range_1_2(self) -> None:
        result = parse_numeric_span("1–2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Range)
        assert result.low == 1.0
        assert result.high == 2.0

    def test_leading_minus_is_a_sign_not_a_range_separator(self) -> None:
        result = parse_numeric_span("-1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == -1.0

    def test_exponent_sign_is_not_mistaken_for_a_range_separator(self) -> None:
        result = parse_numeric_span("1e-7", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == 1e-7

    def test_slash_c0_1_0_substring_alone_is_a_scalar_not_a_two_value_cell(self) -> None:
        """From '7.000Eþ17 /C0 1.0' (A=7.000E+17, n=-1.0): parsing the '/C0 1.0'
        substring alone must yield Scalar(-1.0); this module does not model the A/n
        pair as a single Range."""
        result = parse_numeric_span("/C0 1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == -1.0

    def test_range_with_low_greater_than_high_is_refused_not_silently_swapped(self) -> None:
        """A hyphen-separated span whose first bound is numerically larger than its
        second ('9-2') is never a legitimate low-high range as printed. This module's
        fail-closed posture (already used for '_' digit separators and comma
        thousands separators) is to REFUSE outright rather than silently swap the
        bounds into a shape that was never actually printed -- swapping would
        fabricate an ordering the source text does not state."""
        result = parse_numeric_span("9-2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_range_with_equal_low_and_high_is_accepted(self) -> None:
        """A degenerate range where both bounds print the same value ('5-5') is not
        an ordering violation -- low <= high still holds -- so it must still resolve
        as an ordinary Range rather than being refused."""
        result = parse_numeric_span("5-5", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Range)
        assert result.low == 5.0
        assert result.high == 5.0


class TestStrictLexerSemantics:
    def test_bare_nan_literal_is_unresolvable(self) -> None:
        result = parse_numeric_span("nan", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_bare_nan_literal_is_unresolvable_case_insensitively(self) -> None:
        result = parse_numeric_span("NaN", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_bare_inf_literal_is_unresolvable(self) -> None:
        result = parse_numeric_span("inf", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_bare_infinity_literal_is_unresolvable(self) -> None:
        result = parse_numeric_span("infinity", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_underscore_digit_separator_is_unresolvable(self) -> None:
        result = parse_numeric_span("1_0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_python_float_would_accept_1_0_underscore_but_this_module_refuses_it(self) -> None:
        assert float("1_0") == 10.0  # pins the premise: Python's bare float() is lenient here
        result = parse_numeric_span("1_0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_thousands_separator_comma_is_unresolvable(self) -> None:
        """'1,000' is refused outright (fail closed) rather than parsed as 1000.0 or
        split into the fragments '1' and '000' -- a comma-bearing span is never a
        single clean numeral, the same posture this module already takes for '_'."""
        result = parse_numeric_span("1,000", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)

    def test_thousands_separator_comma_is_unresolvable_even_with_a_decimal_tail(self) -> None:
        result = parse_numeric_span("12,345.6", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)


class TestDashCorruptionQuarantine:
    def test_a_bare_digit_e_digit_token_is_quarantined_in_flat_pdf_text_under_a_suspect_document(
        self,
    ) -> None:
        result = parse_numeric_span("2e50", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=SUSPECT)
        assert isinstance(result, Unresolvable)
        assert "quarantin" in result.reason.lower()

    def test_a_healthy_document_does_not_quarantine_the_same_bare_token_in_flat_pdf_text(self) -> None:
        """A document containing real en dashes is not suspected of the Elsevier
        en-dash-as-e substitution, so the same bare shape resolves normally."""
        result = parse_numeric_span("2e50", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.value == 2e50

    def test_a_jats_cell_does_not_inherit_a_flat_pdf_documents_quarantine(self) -> None:
        """Quarantine is gated on SourceContext.FLAT_PDF_TEXT specifically. Even a
        GlyphHealth computed from a document independently known to suspect dash
        corruption must not quarantine a JATS_CELL parse of the same bare shape --
        a structured cell never inherits a PDF document's quarantine state."""
        result = parse_numeric_span("2e50", source_context=SourceContext.JATS_CELL, glyph_health=SUSPECT)
        assert isinstance(result, Scalar)
        assert result.value == 2e50

    def test_a_spreadsheet_cell_does_not_inherit_a_flat_pdf_documents_quarantine(self) -> None:
        result = parse_numeric_span("2e50", source_context=SourceContext.SPREADSHEET_CELL, glyph_health=SUSPECT)
        assert isinstance(result, Scalar)
        assert result.value == 2e50

    def test_an_explicit_sign_exponent_is_never_quarantined_even_under_a_suspect_document(self) -> None:
        """1e-7 carries an explicit sign, so it is not the bare shape the quarantine
        rule targets, even in a suspect FLAT_PDF_TEXT document."""
        result = parse_numeric_span("1e-7", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=SUSPECT)
        assert isinstance(result, Scalar)
        assert result.value == 1e-7

    def test_uppercase_e_scientific_notation_is_never_quarantined(self) -> None:
        """Genuine scientific notation in the corrupt corpus always uses uppercase E;
        the quarantine rule targets lowercase 'e' only."""
        result = parse_numeric_span("2E50", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=SUSPECT)
        assert isinstance(result, Scalar)
        assert result.value == 2e50


class TestAsciiSixUncertaintyPattern:
    def test_307_6_10_nm_shape_is_unresolvable_and_names_the_possible_uncertainty_reading(
        self,
    ) -> None:
        result = parse_numeric_span("307 6 10", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert "±" in result.reason or "uncertainty" in result.reason.lower()

    def test_ascii_6_is_never_silently_repaired_into_a_plus_minus_value(self) -> None:
        result = parse_numeric_span("307 6 10", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert not isinstance(result, Scalar)
        assert not isinstance(result, Range)


class TestEveryResultCarriesTheRawSpan:
    def test_scalar_carries_the_original_raw_span_untouched(self) -> None:
        result = parse_numeric_span("  7.000Eþ17  ", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Scalar)
        assert result.raw == "  7.000Eþ17  "

    def test_range_carries_the_original_raw_span_untouched(self) -> None:
        result = parse_numeric_span(" 1-2 ", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Range)
        assert result.raw == " 1-2 "

    def test_unresolvable_carries_the_original_raw_span_untouched(self) -> None:
        result = parse_numeric_span(" nan ", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert result.raw == " nan "


class TestRepairNames:
    def test_repair_names_is_exactly_the_four_repairs_this_module_can_emit(self) -> None:
        # Closed set: every name any repairs tuple can ever contain, no more, no less.
        # A downstream schema validates recorded repairs against this instead of
        # accepting free text.
        assert (
            frozenset(
                {
                    "slash_c0_to_minus",
                    "unicode_minus_to_ascii",
                    "leading_en_dash_to_minus",
                    "thorn_to_plus",
                }
            )
            == REPAIR_NAMES
        )


class TestNormalizeNumericSpan:
    def test_significance_is_preserved_trailing_zeros_survive_unlike_the_float_path(self) -> None:
        # This is the whole point of the split: parse_numeric_span would collapse
        # "7.000Eþ17" to the float 7e17, silently destroying the 4-significant-figure
        # precision the original text actually claims. normalize_numeric_span must not.
        result = normalize_numeric_span("7.000Eþ17", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, NormalizedNumeral)
        assert result.text == "7.000e+17"
        assert result.repairs == ("thorn_to_plus",)
        assert result.raw == "7.000Eþ17"

    def test_slash_c0_repairs_to_ascii_minus_in_the_text(self) -> None:
        result = normalize_numeric_span("/C0 1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, NormalizedNumeral)
        assert result.text == "-1.0"
        assert result.repairs == ("slash_c0_to_minus",)

    def test_unicode_minus_sign_repairs_to_ascii_minus_in_the_text(self) -> None:
        result = normalize_numeric_span("−1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, NormalizedNumeral)
        assert result.text == "-1.0"
        assert result.repairs == ("unicode_minus_to_ascii",)

    def test_leading_en_dash_repairs_to_ascii_minus_in_the_text(self) -> None:
        result = normalize_numeric_span("–1.0", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, NormalizedNumeral)
        assert result.text == "-1.0"
        assert result.repairs == ("leading_en_dash_to_minus",)

    def test_a_range_is_refused_not_normalized_as_if_it_were_a_single_numeral(self) -> None:
        result = normalize_numeric_span("1-2", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(result, Unresolvable)
        assert "range" in result.reason

    def test_the_dash_corruption_quarantine_still_fires_through_the_textual_path(self) -> None:
        result = normalize_numeric_span("1e2", source_context=SourceContext.FLAT_PDF_TEXT, glyph_health=SUSPECT)
        assert isinstance(result, Unresolvable)
        assert "quarantined" in result.reason

    def test_1e_plus_400_normalizes_textually_even_though_it_is_not_a_finite_float(self) -> None:
        # Pin BOTH sides of the deliberate divergence in one place: 1E+400 is a
        # well-formed, exactly-representable decimal numeral -- normalize_numeric_span
        # accepts it -- but float("1e+400") is inf, so parse_numeric_span still refuses
        # it. Textual form and float evaluation are different layers on purpose.
        normalized = normalize_numeric_span("1E+400", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(normalized, NormalizedNumeral)
        assert normalized.text == "1e+400"

        parsed = parse_numeric_span("1E+400", source_context=SourceContext.OPERATOR_RAW, glyph_health=HEALTHY)
        assert isinstance(parsed, Unresolvable)
        assert "non-finite" in parsed.reason
