"""Format-only NASA-7 fourth-card repair, without Cantera."""

from __future__ import annotations

import hashlib

import pytest

from carmel.benchmarks.cantera_worker import parse_ck2yaml_log
from carmel.benchmarks.chemkin import normalize_for_ck2yaml


def _cards() -> list[bytes]:
    """Build a four-card record with fourteen distinct, known coefficients."""
    header = b"H2".ljust(24) + b"H   2" + b" " * 15 + b"G"
    header += b"   300.000  5000.000  1000.000".ljust(34) + b"1"
    coefficients = [f"{value:15.8E}".encode() for value in range(1, 15)]
    return [
        header,
        b"".join(coefficients[:5]).ljust(79) + b"2",
        b"".join(coefficients[5:10]).ljust(79) + b"3",
        b"".join(coefficients[10:]).ljust(79) + b"4",
    ]


def _source(cards: list[bytes], ending: bytes = b"\n") -> bytes:
    return ending.join([b"THERMO ALL", b"300.0 1000.0 5000.0", *cards, b"END", b""])


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("padding", [0, 3, 19, 20])
def test_restore_missing_fourth_index_only(ending: bytes, padding: int) -> None:
    cards = _cards()
    cards[3] = cards[3][:60] + b" " * padding
    source = _source(cards, ending)

    result = normalize_for_ck2yaml(source, file="thermo.dat")

    assert result.data == _source(_cards(), ending)
    assert result.raw_sha256 == hashlib.sha256(source).hexdigest()
    assert result.normalized_sha256 == hashlib.sha256(result.data).hexdigest()
    assert len(result.rewrites) == 1
    assert result.rewrites[0].file == "thermo.dat"
    assert result.rewrites[0].line == 6
    assert result.rewrites[0].before.encode() == cards[3]
    assert result.rewrites[0].after.encode() == _cards()[3]
    assert normalize_for_ck2yaml(result.data, file="thermo.dat").rewrites == ()


def test_well_formed_and_empty_sources_are_untouched() -> None:
    for source in (b"", _source(_cards())):
        result = normalize_for_ck2yaml(source, file="thermo.dat")
        assert result.data == source
        assert result.rewrites == ()


@pytest.mark.parametrize("card", [1, 2, 3])
def test_malformed_coefficients_remain_unrepaired_and_refused(card: int) -> None:
    cards = _cards()
    cards[3] = cards[3][:60]
    cards[card] = b"    NOT_A_FLOAT" + cards[card][15:]
    source = _source(cards)

    result = normalize_for_ck2yaml(source, file="thermo.dat")

    assert result.data == source
    assert result.rewrites == ()
    log = 'Unparsable lines while reading thermo data in thermo.dat starting on line 3:\n"""\n'
    log += b"\n".join(cards).decode() + '\n"""\nLines could not be parsed as a NASA7 entry.\n'
    assert parse_ck2yaml_log(log)["refusals"]


@pytest.mark.parametrize("suffix", [b" " * 20 + b"5", b" " * 19 + b"44", b" " * 19 + b"5", b" junk"])
def test_nonblank_trailer_is_not_an_absent_index(suffix: bytes) -> None:
    cards = _cards()
    cards[3] = cards[3][:60] + suffix
    source = _source(cards)
    assert normalize_for_ck2yaml(source, file="thermo.dat").data == source


@pytest.mark.parametrize("card", [0, 1, 2])
def test_requires_preceding_card_indices(card: int) -> None:
    cards = _cards()
    cards[3] = cards[3][:60]
    cards[card] = cards[card][:79] + b"9"
    source = _source(cards)
    assert normalize_for_ck2yaml(source, file="thermo.dat").data == source


def test_repair_requires_thermo_section_and_preserves_comments() -> None:
    cards = _cards()
    cards[3] = cards[3][:60] + b"! unchanged comment"
    result = normalize_for_ck2yaml(_source(cards), file="thermo.dat")
    assert result.data == _source([*_cards()[:3], _cards()[3] + b"! unchanged comment"])
    outside = b"\n".join(cards)
    assert normalize_for_ck2yaml(outside, file="mechanism.inp").data == outside


def test_malformed_header_or_interrupted_cards_are_untouched() -> None:
    cards = _cards()
    cards[3] = cards[3][:60]
    cards[0] = cards[0][:45] + b" bad range" + cards[0][55:]
    source = _source(cards)
    assert normalize_for_ck2yaml(source, file="thermo.dat").data == source
    cards = _cards()
    cards[3] = cards[3][:60]
    source = _source([*cards[:3], b"! interrupted", cards[3]])
    assert normalize_for_ck2yaml(source, file="thermo.dat").data == source


def test_lone_column_81_index_is_moved() -> None:
    cards = _cards()
    cards[3] = cards[3][:79] + b" 4"
    result = normalize_for_ck2yaml(_source(cards), file="thermo.dat")
    assert result.data == _source(_cards())
    assert len(result.rewrites) == 1
    assert result.rewrites[0].before == cards[3].decode()
    assert result.rewrites[0].after == _cards()[3].decode()


@pytest.mark.parametrize(
    ("card", "start", "replacement"), [(0, 0, b" " * 18), (0, 44, b"X"), (1, 75, b"X"), (2, 78, b"X")]
)
def test_fourth_card_repair_refuses_invalid_header_or_coefficient_trailer(
    card: int, start: int, replacement: bytes
) -> None:
    cards = _cards()
    cards[3] = cards[3][:60]
    cards[card] = cards[card][:start] + replacement + cards[card][start + len(replacement) :]
    source = _source(cards)
    result = normalize_for_ck2yaml(source, file="thermo.dat")
    assert result.data == source
    assert result.rewrites == ()


def test_fourth_card_repair_accepts_omitted_optional_midpoint() -> None:
    cards = _cards()
    cards[0] = cards[0][:65] + b" " * 10 + cards[0][75:]
    expected = _source(cards)
    cards[3] = cards[3][:60]
    assert normalize_for_ck2yaml(_source(cards), file="thermo.dat").data == expected


_ZHANG_NAMES = (
    b"C8H141-5,3-4",
    b"C8H131-5,3-4,TA",
    b"C8H131-5,3-4,TAO",
    b"C4H6OHOOH1-4-3",
    b"C4H6OHOOH2-2-1",
    b"C4H6OHOOH1-3-4",
    b"C4H6OHOOH1-2-3",
)


@pytest.mark.parametrize("name", _ZHANG_NAMES)
def test_zhang_names_survive_all_identifier_contexts(name: bytes) -> None:
    cards = _cards()
    cards[0] = name.ljust(18) + cards[0][18:]
    source = (
        b"SPECIES\n"
        + name
        + b"\nEND\n"
        + _source(cards)
        + b"REACTIONS\n"
        + name
        + b" + H => "
        + name
        + b" + H 1.25+02 0.0 0.0\nEND\n"
    )
    expected = source.replace(b"1.25+02", b"1.25E+02")
    result = normalize_for_ck2yaml(source, file="mechanism.inp")
    assert result.data == expected
    assert len(result.rewrites) == 1


@pytest.mark.parametrize(
    ("card", "start", "width"),
    [(0, start, 10) for start in (45, 55, 65)]
    + [(card, start, 15) for card in (1, 2, 3) for start in (0, 15, 30, 45, 60)],
)
def test_eless_thermo_fields_retain_their_width(card: int, start: int, width: int) -> None:
    cards = _cards()
    cards[card] = cards[card][:start] + b"1.25+02".rjust(width) + cards[card][start + width :]
    expected = cards.copy()
    expected[card] = cards[card][:start] + b"1.25E+02".rjust(width) + cards[card][start + width :]
    result = normalize_for_ck2yaml(_source(cards), file="thermo.dat")
    assert result.data == _source(expected)
    assert len(result.rewrites) == 1


@pytest.mark.parametrize("keyword", [b"LOW", b"HIGH", b"TROE", b"SRI", b"PLOG", b"REV", b"CHEB"])
def test_auxiliary_numeric_fields_and_comments(keyword: bytes) -> None:
    source = b"REACTIONS\n" + keyword + b"/1.25+02 2.5-03/ ! comment 3.5+02\nEND\n"
    expected = source.replace(b"1.25+02", b"1.25E+02").replace(b"2.5-03", b"2.5E-03")
    assert normalize_for_ck2yaml(source, file="mechanism.inp").data == expected


@pytest.mark.parametrize("position", [0, 1, 2])
def test_reaction_rate_parameter_positions(position: int) -> None:
    parameters = [b"1.0", b"0.0", b"2.0"]
    parameters[position] = b"1.25+02"
    source = b"REACTIONS\nC8H141-5,3-4 = B " + b" ".join(parameters) + b"\nEND\n"
    assert normalize_for_ck2yaml(source, file="mechanism.inp").data == source.replace(b"1.25+02", b"1.25E+02")


def test_unscoped_numbers_and_non_numeric_auxiliary_fields_are_untouched() -> None:
    source = (
        b"1.25+02\nSPECIES 3-4 C8H141-5,3-4\n1.25+02\nEND\n"
        b"REACTIONS\nFORD / C8H141-5,3-4 1.25+02 /\n"
        b"C8H141-5,3-4=>B 1.25+02 invalid 0\nEND\n"
    )
    assert normalize_for_ck2yaml(source, file="mechanism.inp").data == source


def test_header_non_numeric_fields_and_full_numeric_fields_are_untouched() -> None:
    cards = _cards()
    cards[0] = cards[0][:18] + b"1.2+03" + cards[0][24:]
    # A full-width token with no padding cannot acquire E without shifting columns.
    cards[1] = b"123456789012+123" + cards[1][15:]
    source = _source(cards)
    assert normalize_for_ck2yaml(source, file="thermo.dat").data == source
