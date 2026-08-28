# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for carmel.services.units: the unit-conversion table and its rounding policy.

Pins three kinds of fact: (1) TABLE_V1's content address is a deterministic
function of its own content, not of construction order or object identity;
(2) each of ConversionTable's nine construction-time invariants actually
rejects a deliberately malformed table; (3) the rounding policy's two
branches (significant-digits for scale rules, decimal-exponent for affine
rules) produce the exact worked-example values the module's docstring
promises, including a case where ROUND_HALF_EVEN and ROUND_HALF_UP disagree.
"""

from __future__ import annotations

import hashlib
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import pytest

from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.units import (
    TABLE_V1,
    TABLES_BY_SHA,
    AffineRule,
    ConversionRule,
    ConversionTable,
    ConversionTableInvariantError,
    IdentityRule,
    QuantityKind,
    ScaleRule,
    UnitAlias,
    UnitError,
    UnknownConversionTableError,
    UnknownQuantityUnitPairError,
    UnknownUnitError,
    convert,
    normalize_unit,
    table_for_sha,
)


def _clean_base_units() -> tuple[tuple[QuantityKind, str], ...]:
    """A valid base_units tuple: TABLE_V1's own, reused so unit names read naturally in failures."""
    return TABLE_V1.base_units


def _identity_only_rules(*, skip: QuantityKind | None = None) -> tuple[ConversionRule, ...]:
    """One IdentityRule per non-OTHER quantity's base unit -- a fully valid, non-colliding baseline.

    ``skip`` omits the identity rule for one quantity, so a test can add its
    own (possibly malformed) rule for that quantity's base unit without
    colliding with the baseline's own ``(quantity, base_unit)`` rule key --
    a table with two rules at the same key is a *different* invariant
    (duplicate rule key) than the one most of these tests intend to pin.
    """
    return tuple(
        IdentityRule(kind="identity", quantity=quantity, unit=unit)
        for quantity, unit in _clean_base_units()
        if quantity is not skip
    )


def _table_with(
    *,
    base_units: tuple[tuple[QuantityKind, str], ...] | None = None,
    rules: tuple[ConversionRule, ...],
    aliases: tuple[UnitAlias, ...] = (),
) -> ConversionTable:
    return ConversionTable(
        table_id="test-table",
        version=1,
        base_units=base_units if base_units is not None else _clean_base_units(),
        aliases=aliases,
        rules=rules,
    )


class TestTableSha256:
    """TABLE_V1.sha256 is a deterministic function of its own canonical content."""

    def test_matches_independently_computed_sha256(self) -> None:
        expected = hashlib.sha256(canonical_json_bytes(TABLE_V1.identity_payload())).hexdigest()
        assert TABLE_V1.sha256 == expected

    def test_stable_across_two_equal_constructions(self) -> None:
        rebuilt = ConversionTable(
            table_id=TABLE_V1.table_id,
            version=TABLE_V1.version,
            base_units=TABLE_V1.base_units,
            aliases=TABLE_V1.aliases,
            rules=TABLE_V1.rules,
        )
        assert rebuilt.sha256 == TABLE_V1.sha256

    def test_tables_by_sha_is_keyed_by_that_sha256(self) -> None:
        assert TABLES_BY_SHA[TABLE_V1.sha256] is TABLE_V1


class TestIdentityPayload:
    """The canonical projection carries no floats and no fabricated scale/offset keys."""

    def test_round_trips_through_canonical_json_bytes(self) -> None:
        # canonical_json_bytes rejects floats anywhere in the payload; this
        # simply not raising is the assertion that no float leaked in.
        canonical_json_bytes(TABLE_V1.identity_payload())

    def test_identity_rule_projection_has_no_scale_or_offset_key(self) -> None:
        payload = TABLE_V1.identity_payload()
        identity_rules = [rule for rule in payload["rules"] if rule["kind"] == "identity"]
        assert identity_rules
        for rule in identity_rules:
            assert "scale" not in rule
            assert "offset" not in rule

    def test_scale_rule_projection_has_no_offset_key(self) -> None:
        payload = TABLE_V1.identity_payload()
        scale_rules = [rule for rule in payload["rules"] if rule["kind"] == "scale"]
        assert scale_rules
        for rule in scale_rules:
            assert "offset" not in rule
            assert "scale" in rule


class TestFromIdentityPayload:
    """ConversionTable.from_identity_payload inverts identity_payload(), strictly.

    Round-trip must hold exactly (same fields, same sha256). Every rejection
    must fail closed with a marker phrase specific enough to tell which check
    fired -- untrusted JSON must never construct a table whose scale is
    ``None`` or whose ``from_unit`` is a list, because dataclasses do not
    enforce field types themselves.
    """

    def test_round_trips_table_v1_to_an_equal_table_with_the_same_sha256(self) -> None:
        payload = TABLE_V1.identity_payload()
        reconstructed = ConversionTable.from_identity_payload(payload)
        assert reconstructed == TABLE_V1
        assert reconstructed.sha256 == TABLE_V1.sha256

    def test_round_trips_a_smaller_hand_built_table(self) -> None:
        table = _table_with(rules=_identity_only_rules())
        reconstructed = ConversionTable.from_identity_payload(table.identity_payload())
        assert reconstructed == table
        assert reconstructed.sha256 == table.sha256

    def test_payload_that_is_not_a_json_object_is_rejected(self) -> None:
        with pytest.raises(ConversionTableInvariantError, match="JSON object"):
            ConversionTable.from_identity_payload(["not", "an", "object"])

    def test_missing_top_level_key_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        del payload["aliases"]
        with pytest.raises(ConversionTableInvariantError, match="keys must be exactly"):
            ConversionTable.from_identity_payload(payload)

    def test_unexpected_top_level_key_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        payload["checksum"] = "deadbeef"
        with pytest.raises(ConversionTableInvariantError, match="keys must be exactly"):
            ConversionTable.from_identity_payload(payload)

    def test_non_str_table_id_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        payload["table_id"] = 123
        with pytest.raises(ConversionTableInvariantError, match="'table_id' must be a str"):
            ConversionTable.from_identity_payload(payload)

    def test_non_int_version_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        payload["version"] = "1"
        with pytest.raises(ConversionTableInvariantError, match="'version' must be an int"):
            ConversionTable.from_identity_payload(payload)

    def test_base_units_that_is_not_a_list_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        payload["base_units"] = {"length": "m"}
        with pytest.raises(ConversionTableInvariantError, match="'base_units' must be a JSON array"):
            ConversionTable.from_identity_payload(payload)

    def test_bogus_quantity_string_in_base_units_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        base_units = [list(entry) for entry in payload["base_units"]]
        base_units[0][0] = "not-a-real-quantity"
        payload["base_units"] = base_units
        with pytest.raises(ConversionTableInvariantError, match="is not a known QuantityKind"):
            ConversionTable.from_identity_payload(payload)

    def test_bogus_rule_kind_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        rules = [dict(rule) for rule in payload["rules"]]
        rules[0]["kind"] = "logarithmic"
        payload["rules"] = rules
        with pytest.raises(ConversionTableInvariantError, match="'identity'/'scale'/'affine'"):
            ConversionTable.from_identity_payload(payload)

    def test_scale_of_null_on_a_scale_rule_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        rules = [dict(rule) for rule in payload["rules"]]
        scale_rule_index = next(index for index, rule in enumerate(rules) if rule["kind"] == "scale")
        rules[scale_rule_index]["scale"] = None
        payload["rules"] = rules
        with pytest.raises(ConversionTableInvariantError, match=r"'scale' must be a str, got NoneType"):
            ConversionTable.from_identity_payload(payload)

    def test_from_unit_as_a_list_on_a_scale_rule_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        rules = [dict(rule) for rule in payload["rules"]]
        scale_rule_index = next(index for index, rule in enumerate(rules) if rule["kind"] == "scale")
        rules[scale_rule_index]["from_unit"] = ["cm/s"]
        payload["rules"] = rules
        with pytest.raises(ConversionTableInvariantError, match=r"'from_unit' must be a str, got list"):
            ConversionTable.from_identity_payload(payload)

    def test_bool_version_is_rejected_even_though_bool_is_an_int_subclass(self) -> None:
        # isinstance(True, int) is True in Python; a bool must still be refused
        # as a version number, or "version": true would silently become 1.
        payload = dict(TABLE_V1.identity_payload())
        payload["version"] = True
        with pytest.raises(ConversionTableInvariantError, match="'version' must be an int"):
            ConversionTable.from_identity_payload(payload)

    def test_rule_missing_a_required_key_is_rejected(self) -> None:
        payload = dict(TABLE_V1.identity_payload())
        rules = [dict(rule) for rule in payload["rules"]]
        scale_rule_index = next(index for index, rule in enumerate(rules) if rule["kind"] == "scale")
        del rules[scale_rule_index]["scale"]
        payload["rules"] = rules
        with pytest.raises(ConversionTableInvariantError, match="keys must be exactly"):
            ConversionTable.from_identity_payload(payload)


class TestTableInvariants:
    """Each of ConversionTable's nine construction-time invariants rejects a malformed table."""

    def test_invariant_1_missing_quantity_in_base_units(self) -> None:
        base_units = tuple(
            (quantity, "unit0")
            for quantity in QuantityKind
            if quantity not in (QuantityKind.OTHER, QuantityKind.LENGTH)
        )
        with pytest.raises(ConversionTableInvariantError, match="must cover every QuantityKind"):
            ConversionTable(table_id="t", version=1, base_units=base_units, aliases=(), rules=())

    def test_invariant_1_other_included_in_base_units(self) -> None:
        base_units = tuple((quantity, "unit0") for quantity in QuantityKind)
        with pytest.raises(ConversionTableInvariantError, match="must cover every QuantityKind"):
            ConversionTable(table_id="t", version=1, base_units=base_units, aliases=(), rules=())

    def test_invariant_1_duplicate_quantity_in_base_units(self) -> None:
        base_units = tuple((quantity, "unit0") for quantity in QuantityKind if quantity is not QuantityKind.OTHER)
        base_units = base_units + (base_units[0],)
        with pytest.raises(ConversionTableInvariantError, match="duplicate quantity"):
            ConversionTable(table_id="t", version=1, base_units=base_units, aliases=(), rules=())

    def test_invariant_2_scale_rule_to_unit_is_not_the_base_unit(self) -> None:
        bad_rule = ScaleRule(
            kind="scale", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="not-the-base", scale="0.01"
        )
        with pytest.raises(ConversionTableInvariantError, match="base unit"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_3_scale_rule_from_unit_equals_to_unit(self) -> None:
        # Omit VELOCITY's own identity rule so this ScaleRule's key does not
        # collide with it (that would trip invariant 4 instead of this one).
        bad_rule = ScaleRule(kind="scale", quantity=QuantityKind.VELOCITY, from_unit="m/s", to_unit="m/s", scale="1")
        with pytest.raises(ConversionTableInvariantError, match="from_unit == to_unit"):
            _table_with(rules=_identity_only_rules(skip=QuantityKind.VELOCITY) + (bad_rule,))

    def test_invariant_3_identity_rule_for_non_base_unit(self) -> None:
        # Omit LENGTH's own identity rule so this bad one's key ("length", "cm")
        # does not collide with the baseline's ("length", "m") entry.
        bad_rule = IdentityRule(kind="identity", quantity=QuantityKind.LENGTH, unit="cm")
        with pytest.raises(ConversionTableInvariantError, match="identity rules exist only for"):
            _table_with(rules=_identity_only_rules(skip=QuantityKind.LENGTH) + (bad_rule,))

    def test_invariant_4_duplicate_rule_key(self) -> None:
        rules = _identity_only_rules()
        dup = IdentityRule(kind="identity", quantity=QuantityKind.LENGTH, unit="m")
        with pytest.raises(ConversionTableInvariantError, match="duplicate rule key"):
            _table_with(rules=rules + (dup,))

    def test_invariant_4_duplicate_alias_key(self) -> None:
        aliases = (
            UnitAlias(quantity=QuantityKind.LENGTH, raw="centimeter", normalized="m"),
            UnitAlias(quantity=QuantityKind.LENGTH, raw="centimeter", normalized="m"),
        )
        with pytest.raises(ConversionTableInvariantError, match="duplicate alias key"):
            _table_with(rules=_identity_only_rules(), aliases=aliases)

    def test_invariant_5_scale_is_not_canonical_decimal(self) -> None:
        bad_rule = ScaleRule(
            kind="scale", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", scale="+0.01"
        )
        with pytest.raises(ConversionTableInvariantError, match="canonical decimal"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_5_scale_is_not_strictly_positive(self) -> None:
        bad_rule = ScaleRule(kind="scale", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", scale="0")
        with pytest.raises(ConversionTableInvariantError, match="strictly positive"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_5_offset_is_not_canonical_decimal(self) -> None:
        bad_rule = AffineRule(
            kind="affine", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", scale="1", offset="+273.15"
        )
        with pytest.raises(ConversionTableInvariantError, match="canonical decimal"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_6_alias_normalized_is_not_a_known_unit(self) -> None:
        alias = UnitAlias(quantity=QuantityKind.LENGTH, raw="centimeter", normalized="does-not-exist")
        with pytest.raises(ConversionTableInvariantError, match="not a known unit"):
            _table_with(rules=_identity_only_rules(), aliases=(alias,))

    def test_invariant_6_alias_raw_is_already_a_known_unit(self) -> None:
        # Give LENGTH a second known unit ("cm") so the alias's raw form can
        # legitimately collide with something already known.
        rules = _identity_only_rules() + (
            ScaleRule(kind="scale", quantity=QuantityKind.LENGTH, from_unit="cm", to_unit="m", scale="0.01"),
        )
        alias = UnitAlias(quantity=QuantityKind.LENGTH, raw="cm", normalized="m")
        with pytest.raises(ConversionTableInvariantError, match="already a known unit"):
            _table_with(rules=rules, aliases=(alias,))

    def test_invariant_6_alias_raw_equals_normalized(self) -> None:
        alias = UnitAlias(quantity=QuantityKind.LENGTH, raw="m", normalized="m")
        with pytest.raises(ConversionTableInvariantError, match="raw == normalized"):
            _table_with(rules=_identity_only_rules(), aliases=(alias,))

    def test_invariant_7_rule_registered_for_other(self) -> None:
        bad_rule = IdentityRule(kind="identity", quantity=QuantityKind.OTHER, unit="anything")
        with pytest.raises(ConversionTableInvariantError, match="QuantityKind.OTHER"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_7_alias_registered_for_other(self) -> None:
        bad_alias = UnitAlias(quantity=QuantityKind.OTHER, raw="whatever", normalized="anything")
        with pytest.raises(ConversionTableInvariantError, match="QuantityKind.OTHER"):
            _table_with(rules=_identity_only_rules(), aliases=(bad_alias,))

    def test_invariant_8_affine_rule_scale_is_not_one(self) -> None:
        bad_rule = AffineRule(
            kind="affine", quantity=QuantityKind.TEMPERATURE, from_unit="F", to_unit="K", scale="5", offset="255.372"
        )
        with pytest.raises(ConversionTableInvariantError, match="AffineRule's scale must be exactly '1'"):
            _table_with(rules=_identity_only_rules() + (bad_rule,))

    def test_invariant_9_missing_identity_rule_for_base_unit(self) -> None:
        # Omit LENGTH's identity rule entirely and add no replacement rule for
        # it at all, so this trips only the "no IdentityRule for this
        # quantity's base unit" invariant, not any of the per-rule checks.
        with pytest.raises(ConversionTableInvariantError, match="no IdentityRule"):
            _table_with(rules=_identity_only_rules(skip=QuantityKind.LENGTH))


class TestNormalizeUnit:
    def test_micro_sign_spelling_of_microseconds(self) -> None:
        # U+00B5 MICRO SIGN
        assert normalize_unit(QuantityKind.TIME, "µs", table=TABLE_V1) == "us"

    def test_greek_mu_spelling_of_microseconds(self) -> None:
        # U+03BC GREEK SMALL LETTER MU
        assert normalize_unit(QuantityKind.TIME, "μs", table=TABLE_V1) == "us"

    def test_degrees_celsius_alias(self) -> None:
        assert normalize_unit(QuantityKind.TEMPERATURE, "°C", table=TABLE_V1) == "C"

    def test_already_normalized_unit_returns_unchanged(self) -> None:
        assert normalize_unit(QuantityKind.PRESSURE, "Pa", table=TABLE_V1) == "Pa"

    def test_unknown_unit_raises_naming_known_units(self) -> None:
        with pytest.raises(UnknownUnitError, match="known units"):
            normalize_unit(QuantityKind.LENGTH, "furlong", table=TABLE_V1)

    def test_case_is_not_folded_kelvin_normalizes(self) -> None:
        assert normalize_unit(QuantityKind.TEMPERATURE, "K", table=TABLE_V1) == "K"

    def test_case_is_not_folded_lowercase_k_is_unknown(self) -> None:
        # "k" (a kilo- prefix fragment) must never be silently treated as "K" (kelvin).
        with pytest.raises(UnknownUnitError):
            normalize_unit(QuantityKind.TEMPERATURE, "k", table=TABLE_V1)

    def test_other_quantity_passes_text_through_unchanged(self) -> None:
        assert normalize_unit(QuantityKind.OTHER, "anything at all", table=TABLE_V1) == "anything at all"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            normalize_unit(QuantityKind.LENGTH, "", table=TABLE_V1)

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            normalize_unit(QuantityKind.LENGTH, "   ", table=TABLE_V1)

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert normalize_unit(QuantityKind.LENGTH, "  cm  ", table=TABLE_V1) == "cm"

    def test_ppm_is_a_known_mole_fraction_unit(self) -> None:
        assert normalize_unit(QuantityKind.MOLE_FRACTION, "ppm", table=TABLE_V1) == "ppm"

    def test_ppm_is_a_known_mass_fraction_unit(self) -> None:
        assert normalize_unit(QuantityKind.MASS_FRACTION, "ppm", table=TABLE_V1) == "ppm"

    def test_ppmv_aliases_to_ppm_for_mole_fraction(self) -> None:
        assert normalize_unit(QuantityKind.MOLE_FRACTION, "ppmv", table=TABLE_V1) == "ppm"

    def test_ppmv_is_not_a_mass_fraction_alias(self) -> None:
        # ppmv is explicitly volume/mole basis; it must not be usable as a
        # mass-fraction spelling even though bare "ppm" is representable there.
        with pytest.raises(UnknownUnitError):
            normalize_unit(QuantityKind.MASS_FRACTION, "ppmv", table=TABLE_V1)

    def test_equivalence_ratio_admits_only_the_dimensionless_tokens(self) -> None:
        # I058: what the units module accepts for the dimensionless coordinate.
        # Base "1" and the two dimensionless spellings normalize; ANYTHING else is
        # unknown. This is the vocabulary a groundable phi unit must fall within --
        # and none of these tokens is printed near the flame-speed table's phi
        # column, which is why that axis stays OTHER (see
        # tests.test_tabular_dataset_target_acceptance).
        assert normalize_unit(QuantityKind.EQUIVALENCE_RATIO, "1", table=TABLE_V1) == "1"
        assert normalize_unit(QuantityKind.EQUIVALENCE_RATIO, "-", table=TABLE_V1) == "1"
        assert normalize_unit(QuantityKind.EQUIVALENCE_RATIO, "dimensionless", table=TABLE_V1) == "1"

    def test_equivalence_ratio_rejects_the_corrupted_phi_header(self) -> None:
        # I058, the guard against laundering: the symbol-font phi header decodes to
        # "/", which is NOT an equivalence-ratio unit. This must keep raising -- if a
        # later change adds a "/"-to-"1" alias to make the flagship axis look tidy,
        # that is exactly the fabricated unit this project refuses, and this test
        # fails to say so.
        with pytest.raises(UnknownUnitError, match="known units"):
            normalize_unit(QuantityKind.EQUIVALENCE_RATIO, "/", table=TABLE_V1)


class TestConvertTemperatureAffineRounding:
    """25 C -> K: affine rounding respects the source's ABSOLUTE precision (decimal exponent).

    This is the load-bearing demonstration that rounding to a fixed number of
    significant figures would be wrong here: 25 C -> K is exact 298.15, and
    naively rounding *that* to 2 significant figures would claim "300", which
    misrepresents a source value reported to whole-degree precision.
    """

    def test_25_degrees_exact_and_rounded_to_whole_kelvin(self) -> None:
        result = convert("25", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", table=TABLE_V1)
        assert result.exact == "298.15"
        assert result.rounded == "298"
        assert result.rule_kind == "affine"
        assert result.rounding_policy == "decimal_exponent"
        assert result.quantity is QuantityKind.TEMPERATURE
        assert result.from_unit == "C"
        assert result.to_unit == "K"
        assert result.conversion_table_sha256 == TABLE_V1.sha256

    def test_25_point_0_degrees_rounds_to_tenths(self) -> None:
        result = convert("25.0", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", table=TABLE_V1)
        assert result.exact == "298.15"
        assert result.rounded == "298.2"

    def test_25_point_00_degrees_needs_no_rounding(self) -> None:
        result = convert("25.00", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", table=TABLE_V1)
        assert result.exact == "298.15"
        assert result.rounded == "298.15"


class TestConvertPressureScaleRounding:
    """1.23 atm -> Pa: scale rounding respects the source's RELATIVE precision (significant digits)."""

    def test_exact_and_rounded_values(self) -> None:
        result = convert("1.23", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)
        assert result.exact == "124629.75"
        assert result.rounded == "1.25E+5"
        assert result.rule_kind == "scale"
        assert result.rounding_policy == "significant_digits"
        assert result.quantity is QuantityKind.PRESSURE
        assert result.from_unit == "atm"
        assert result.to_unit == "Pa"
        assert result.conversion_table_sha256 == TABLE_V1.sha256

    def test_zero_atm_converts_to_zero_pa_without_special_casing(self) -> None:
        # "0" has one significant digit by the same digit-preserving rule as
        # any other canonical decimal; it is not an edge case this module
        # treats differently. Pinned here because the module docstring cites
        # this exact example as the significance-stance evidence.
        result = convert("0", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)
        assert result.exact == "0"
        assert result.rounded == "0"


class TestConvertMoleFractionPpmScaling:
    """1 ppm -> mole fraction: pins Item 1's new ScaleRule at its documented scale."""

    def test_one_ppm_is_one_millionth(self) -> None:
        result = convert("1", quantity=QuantityKind.MOLE_FRACTION, from_unit="ppm", to_unit="1", table=TABLE_V1)
        assert result.exact == "0.000001"
        assert result.rule_kind == "scale"

    def test_mass_fraction_ppm_is_also_representable(self) -> None:
        result = convert("1", quantity=QuantityKind.MASS_FRACTION, from_unit="ppm", to_unit="1", table=TABLE_V1)
        assert result.exact == "0.000001"


class TestConvertVelocityScaleRounding:
    def test_power_of_ten_factor_survives_exact_and_rounded(self) -> None:
        # cm/s -> m/s is a factor of exactly 0.01 (a power of ten), so shifting
        # the decimal point never manufactures extra coefficient digits --
        # exact and rounded coincide here only because of that, not in general.
        result = convert("1.23", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", table=TABLE_V1)
        assert result.exact == "0.0123"
        assert result.rounded == "0.0123"
        assert result.quantity is QuantityKind.VELOCITY
        assert result.from_unit == "cm/s"
        assert result.to_unit == "m/s"
        assert result.conversion_table_sha256 == TABLE_V1.sha256

    def test_trailing_zero_significance_changes_rounded_result(self) -> None:
        # "1.230" carries 4 significant figures, "1.23" carries 3; both are
        # numerically equal before conversion, but the ROUNDED result must
        # differ afterward because the two inputs claim different precision.
        result_3sf = convert("1.23", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", table=TABLE_V1)
        result_4sf = convert("1.230", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", table=TABLE_V1)
        assert result_3sf.rounded == "0.0123"
        assert result_4sf.rounded == "0.01230"
        assert result_3sf.rounded != result_4sf.rounded


class TestRoundHalfEvenPinnedMode:
    """A case where ROUND_HALF_EVEN and ROUND_HALF_UP disagree, pinning the mode this module uses."""

    def test_half_even_rounds_to_even_digit_not_always_up(self) -> None:
        """A genuine halfway tie, where the two modes give different answers.

        The tie has to be EXACT for the modes to disagree at all. ``0.5`` C is
        one source digit past the point, so rounding targets the tenths place;
        ``0.5 + 273.15 = 273.65`` sits exactly halfway between ``273.6`` and
        ``273.7``. ROUND_HALF_EVEN takes the even digit (``273.6``);
        ROUND_HALF_UP takes ``273.7``.

        Stated because it is easy to get wrong and a near-miss test proves
        nothing: a residue like ``398.15`` rounded to the ones place is NOT a
        tie (it is 0.15 above 398), so both modes return 398 and such a case
        cannot pin the mode. With TABLE_V1's rule set the affine branch is the
        only one that can manufacture an exact tie -- every scale factor here
        is either a power of ten (which never rounds) or 101325 (whose
        products do not land on an exact half at the source's significance).
        """
        result = convert("0.5", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", table=TABLE_V1)
        assert result.exact == "273.65"
        assert result.rounded == "273.6"
        # Demonstrate the disagreement rather than asserting it: the other mode
        # really does answer differently for this input.
        assert Decimal("273.65").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) == Decimal("273.7")
        assert Decimal("273.65").quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN) == Decimal("273.6")


class TestConvertQuantityUnitPairErrors:
    def test_refuses_cross_quantity_pair(self) -> None:
        with pytest.raises(UnknownQuantityUnitPairError):
            convert("1", quantity=QuantityKind.TIME, from_unit="1/s", to_unit="s", table=TABLE_V1)

    def test_refuses_non_canonical_input_value(self) -> None:
        with pytest.raises(UnitError):
            convert("+1.5", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)

    def test_refuses_to_unit_that_is_not_the_quantity_base_unit(self) -> None:
        with pytest.raises(UnknownQuantityUnitPairError):
            convert("1.23", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="atm", table=TABLE_V1)

    def test_other_refuses_non_identity_pair_for_other_quantity(self) -> None:
        with pytest.raises(UnknownQuantityUnitPairError):
            convert("1.23", quantity=QuantityKind.OTHER, from_unit="C", to_unit="F", table=TABLE_V1)

    def test_other_identity_pair_succeeds(self) -> None:
        # QuantityKind.OTHER has no registered rules at all (invariant 7), so
        # the only pair convert() can ever accept for it is from_unit ==
        # to_unit -- a pure passthrough with no fabricated factor.
        result = convert("1.23", quantity=QuantityKind.OTHER, from_unit="anything", to_unit="anything", table=TABLE_V1)
        assert result.exact == "1.23"
        assert result.rounded == "1.23"
        assert result.rule_kind == "identity"
        assert result.rounding_policy == "identity"
        assert result.quantity is QuantityKind.OTHER
        assert result.from_unit == "anything"
        assert result.to_unit == "anything"
        assert result.conversion_table_sha256 == TABLE_V1.sha256


class TestTableForSha:
    def test_round_trips_table_v1_sha256(self) -> None:
        assert table_for_sha(TABLE_V1.sha256) == TABLE_V1

    def test_unknown_sha_raises(self) -> None:
        with pytest.raises(UnknownConversionTableError, match="unknown-sha-value"):
            table_for_sha("unknown-sha-value")


class TestConversionResultsStayRepresentable:
    """convert() must never hand back a string canonical_decimal itself would refuse.

    Exact arithmetic can carry a perfectly legal input OUT of the range the
    store accepts: canonical_decimal bounds both the coefficient digit count
    and the adjusted exponent, and a scale rule grows both. A result that
    escaped that bound would be a value this module called "a canonical
    decimal string" and the store then rejected -- so it is refused here,
    at the operation that created it.
    """

    def test_scale_result_past_the_adjusted_exponent_bound_is_refused(self) -> None:
        # 1E+999 atm is an acceptable canonical decimal; x101325 is not.
        with pytest.raises(UnitError, match="not a representable canonical decimal"):
            convert("1E+999", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)

    def test_a_result_inside_the_bound_is_returned_unchanged(self) -> None:
        # The guard is a pure bounds check: str(Decimal) is already the
        # canonical rendering, so an in-range result keeps its exact characters.
        result = convert("1E+900", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)
        assert result.exact == "1.01325E+905"

    def test_the_refusal_names_the_operands(self) -> None:
        with pytest.raises(UnitError, match="from 'atm' to 'Pa'"):
            convert("1E+999", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)

    def test_the_exact_guard_is_load_bearing_on_its_own(self) -> None:
        """A result whose EXACT form overflows while its ROUNDED form does not.

        canonical_decimal bounds the coefficient digit count at 1000. Rounding
        cuts the coefficient back to the source's significance but leaves the
        adjusted exponent alone, so this is the only shape that separates the
        two guards: 1000 source digits x 101325 gives a 1005-digit exact result
        and a 1000-digit rounded one. Without this case, dropping the `exact`
        guard entirely leaves every other test green -- the `rounded` guard
        covers for it.
        """
        value = "1." + "3" * 999
        with pytest.raises(UnitError, match="exact result"):
            convert(value, quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", table=TABLE_V1)


class TestShippedTablesAreNeverRemoved:
    """Section 6's append-only tripwire.

    The module docstring's doctrine is that a shipped ConversionTable is
    never mutated -- corrections ship as a new version alongside, never as
    a replacement. Nothing before this test enforced the other half of
    that doctrine: that a shipped table's sha256 is never REMOVED from
    TABLES_BY_SHA either. If one ever silently disappeared, every embedded
    dataset that cites it would become unvalidatable (table_for_sha would
    raise UnknownConversionTableError for a sha256 real stored data still
    names), with no fallback to "the current table" by design -- see
    table_for_sha's own docstring.

    This pins an explicit, frozen set of every sha256 TABLES_BY_SHA is
    historically known to have shipped. The set below is a HARDCODED
    STRING LITERAL, deliberately NOT computed from the live TABLE_V1 --
    if it were derived from TABLE_V1.sha256, editing TABLE_V1 in place
    would silently move the "historical" set along with it and this test
    could never catch the very regression (removing/replacing a shipped
    table) it exists to catch. It is a SUBSET check against the live
    registry: adding a new table alongside is fine (the registry only
    grows), but removing one -- or replacing TABLE_V1 in place so its
    sha256 changes -- fails this test.

    When a future commit legitimately adds a new shipped table version,
    APPEND its sha256 to _HISTORICALLY_SHIPPED_SHA256S below as a new
    literal string. NEVER remove or alter an existing entry, and NEVER
    replace this frozenset with one computed from TABLES_BY_SHA/TABLE_V1
    -- doing either defeats the whole point of this tripwire.
    """

    _HISTORICALLY_SHIPPED_SHA256S = frozenset(
        {
            # TABLE_V1, shipped since the module's introduction. This is a
            # literal historical constant -- see class docstring above.
            "1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122",
        }
    )

    def test_every_historically_shipped_sha256_is_still_in_the_registry(self) -> None:
        missing = self._HISTORICALLY_SHIPPED_SHA256S - set(TABLES_BY_SHA)
        assert not missing, (
            f"shipped conversion table(s) {sorted(missing)!r} are no longer in TABLES_BY_SHA -- "
            "removing a shipped table orphans every stored dataset that embeds/cites it "
            "(table_for_sha has no fallback to 'the current table'); a shipped table must never "
            "be removed or replaced in place -- add a replacement table ALONGSIDE it instead, and "
            "keep the old sha256 in TABLES_BY_SHA (and in this test's historically-shipped set)"
        )


class TestTableIsRequiredNotDefaulted:
    """``table`` must be a required keyword-only argument on both functions.

    A default of ``table: ConversionTable = TABLE_V1`` would let a call site
    silently agree with TABLE_V1 by omission rather than by explicit choice,
    which defeats any invariant that depends on every table-consuming site
    naming its table. Pinning the ``TypeError`` here is what keeps a future
    "helpful" default from being reintroduced unnoticed.
    """

    def test_normalize_unit_without_table_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            normalize_unit(QuantityKind.PRESSURE, "Pa")  # type: ignore[call-arg]

    def test_convert_without_table_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            convert(  # type: ignore[call-arg]
                "1.23", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa"
            )
