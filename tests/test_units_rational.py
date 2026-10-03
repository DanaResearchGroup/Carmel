"""Rational conversion arithmetic and shipped-table identity regression checks."""

from dataclasses import replace
from decimal import ROUND_UP, Decimal, localcontext

import pytest

from carmel.services.units import (
    TABLE_V1,
    TABLE_V2,
    TABLE_V3,
    TABLE_V4,
    TABLE_V5,
    ConversionTable,
    ConversionTableInvariantError,
    QuantityKind,
    RationalScaleRule,
    UnitError,
    convert,
    table_for_sha,
)


def test_shipped_table_hashes_before_torr_support_remain_identical() -> None:
    assert {table.version: table.sha256 for table in (TABLE_V1, TABLE_V2, TABLE_V3, TABLE_V4)} == {
        1: "1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122",
        2: "371d93150f0b4d078d91727e3a76cdfb8879ee9e25753a0431bf48a0fdf2e1ec",
        3: "fd718e1d7cf54f0f94bdb24c4f325aef76441a7724605559309a4b2a751395bd",
        4: "0a7504ec2e8fd4e56cc878cc9c9501863060f5aa57a7b9cf0ec48b00c68cae90",
    }


def test_760_torr_is_exactly_one_atmosphere_in_pascal() -> None:
    result = convert("760", quantity=QuantityKind.PRESSURE, from_unit="Torr", to_unit="Pa", table=TABLE_V5)
    assert result.exact == "101325"
    assert result.rounded == "1.01E+5"
    assert result.rule_kind == "rational_scale"
    assert result.rounding_policy == "significant_digits"


def test_recurring_result_uses_working_precision_then_source_significance() -> None:
    with localcontext() as ctx:
        ctx.prec = 4
        ctx.rounding = ROUND_UP
        result = convert("1.000", quantity=QuantityKind.PRESSURE, from_unit="Torr", to_unit="Pa", table=TABLE_V5)
    with localcontext() as ctx:
        ctx.prec = 4096
        expected = Decimal("1.000") * Decimal(101325) / Decimal(760)
    assert result.exact == str(expected)
    assert result.rounded == "133.3"
    assert result.conversion_table_sha256 == TABLE_V5.sha256


def test_rational_table_round_trip_has_stable_sha_and_rule_shape() -> None:
    payload = TABLE_V5.identity_payload()
    rebuilt = ConversionTable.from_identity_payload(payload)
    assert rebuilt.identity_payload() == payload
    assert rebuilt.sha256 == TABLE_V5.sha256 == "1a10cb5949c5330f593f95f1879c09184f0c953e8547f64ded9d66b0f7ce045e"
    assert table_for_sha(rebuilt.sha256) is TABLE_V5
    payload["rules"][-1]["numerator"] = "1"
    assert TABLE_V5.rules[-1].numerator == "101325"


@pytest.mark.parametrize(
    "change",
    [
        {"denominator": "0"},
        {"numerator": "-1"},
        {"denominator": "NaN"},
        {"numerator": "1.0e0"},
        {"denominator": 760},
        {"numerator": None},
        {"to_unit": []},
        {"kind": "mystery"},
        {"extra": "1"},
    ],
)
def test_invalid_rational_payload_is_refused(change: dict[str, object]) -> None:
    payload = TABLE_V5.identity_payload()
    payload["rules"][-1].update(change)
    with pytest.raises(ConversionTableInvariantError):
        ConversionTable.from_identity_payload(payload)


def test_table_rejects_mutable_collections_and_wrong_rule_discriminator() -> None:
    with pytest.raises(ConversionTableInvariantError, match="immutable"):
        replace(TABLE_V5, rules=list(TABLE_V5.rules))
    with pytest.raises(ConversionTableInvariantError, match="rule kind"):
        replace(TABLE_V5, rules=TABLE_V5.rules[:-1] + (replace(TABLE_V5.rules[-1], kind="scale"),))


def test_rational_result_keeps_canonical_exponent_bounds() -> None:
    with pytest.raises(UnitError, match="representable canonical"):
        convert("1E+999", quantity=QuantityKind.PRESSURE, from_unit="Torr", to_unit="Pa", table=TABLE_V5)


def test_half_even_rounding_of_rational_scale() -> None:
    rule = RationalScaleRule(
        kind="rational_scale",
        quantity=QuantityKind.PRESSURE,
        from_unit="third",
        to_unit="Pa",
        numerator="1",
        denominator="3",
    )
    table = replace(TABLE_V5, rules=TABLE_V5.rules + (rule,))
    assert (
        convert("1.000", quantity=QuantityKind.PRESSURE, from_unit="third", to_unit="Pa", table=table).rounded
        == "0.3333"
    )
