"""Tests for the member-table inventory record: reading, addressing and byte-replay.

Every member is SYNTHETIC bytes built here; no paper or supplementary text enters the
repository. The property under test is that the record is a claim the MEMBER's bytes can
refute -- replay re-derives the grid from those bytes rather than reading it back -- and
that a corrupted cell is reported as a mismatch that names the cell.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from carmel.services import member_table_record
from carmel.services.member_table_record import (
    MEMBER_INVENTORY_PAYLOAD_KEYS,
    MEMBER_INVENTORY_PAYLOAD_VERSION,
    MemberCellReplayOutcome,
    MemberInventoryVerificationStatus,
    MemberTableTooLarge,
    MemberTableUnreadable,
    cell_text_from_payload,
    compute_member_inventory_sha,
    delimiter_for_sheet,
    member_inventory_record_bytes,
    member_inventory_record_payload,
    read_delimited_member,
    replay_member_cell,
    verify_member_inventory_record,
)

_CSV = b"t_ms,CH4,O2\n0.0,0.21,0.79\n0.5,0.10,0.60\n"


def _payload(data: bytes, *, sheet_name: str = "species.csv") -> dict:
    inventory = read_delimited_member(data, sheet_name=sheet_name)
    return member_inventory_record_payload(inventory, member_sha256=hashlib.sha256(data).hexdigest())


def test_reads_a_csv_grid_including_header_and_blanks() -> None:
    inventory = read_delimited_member(b"a,b,c\n1,,3\n", sheet_name="x.csv")
    assert inventory.row_count == 2
    assert inventory.col_count == 3
    by_pos = {(c.row, c.col): c.text for c in inventory.cells}
    assert by_pos[(0, 0)] == "a"
    assert by_pos[(1, 1)] == ""  # a blank field is a present cell
    assert by_pos[(1, 2)] == "3"


def test_empty_member_has_no_cells() -> None:
    inventory = read_delimited_member(b"", sheet_name="empty.csv")
    assert inventory.row_count == 0
    assert inventory.col_count == 0
    assert inventory.cells == ()


def test_tsv_is_read_with_a_tab_delimiter() -> None:
    assert delimiter_for_sheet("x.tsv") == "\t"
    assert delimiter_for_sheet("x.csv") == ","
    inventory = read_delimited_member(b"a\tb\n1\t2\n", sheet_name="x.tsv")
    assert {(c.row, c.col): c.text for c in inventory.cells}[(1, 1)] == "2"


def test_non_utf8_member_is_unreadable() -> None:
    with pytest.raises(MemberTableUnreadable):
        read_delimited_member(b"a,b\n\xff\xfe,2\n", sheet_name="bad.csv")


def test_member_over_the_cell_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # A member that is trivially legal under every archive byte cap (a handful of bytes)
    # but whose field count crosses the parse's cell cap. The cap is lowered here only so
    # the test need not build eight million cells; the mechanism is the production one.
    monkeypatch.setattr(member_table_record, "MAX_MEMBER_CELL_COUNT", 6)
    with pytest.raises(MemberTableTooLarge) as exc_info:
        read_delimited_member(b"a,b,c\n1,2,3\n4,5,6\n", sheet_name="big.csv")  # 9 cells > 6
    assert "6 cells" in str(exc_info.value)


def test_member_at_the_cell_cap_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exactly at the cap is read (the check refuses the cell that would CROSS it); one
    # more cell is refused. So the change is a bound, not a blanket rejection.
    monkeypatch.setattr(member_table_record, "MAX_MEMBER_CELL_COUNT", 6)
    inventory = read_delimited_member(b"a,b,c\n1,2,3\n", sheet_name="fits.csv")  # 6 cells == cap
    assert len(inventory.cells) == 6
    with pytest.raises(MemberTableTooLarge):
        read_delimited_member(b"a,b,c\n1,2,3\nx\n", sheet_name="over.csv")  # 7 cells > cap


def test_record_addresses_to_its_own_sha() -> None:
    payload = _payload(_CSV)
    assert set(payload) == set(MEMBER_INVENTORY_PAYLOAD_KEYS)
    assert payload["payload_version"] == MEMBER_INVENTORY_PAYLOAD_VERSION
    address = compute_member_inventory_sha(payload)
    assert address == hashlib.sha256(member_inventory_record_bytes(payload)).hexdigest()


def test_record_payload_rejects_a_malformed_member_sha() -> None:
    inventory = read_delimited_member(_CSV, sheet_name="x.csv")
    with pytest.raises(ValueError, match="member_sha256 must be 64"):
        member_inventory_record_payload(inventory, member_sha256="deadbeef")


def test_verify_reproduces_against_the_members_bytes() -> None:
    payload = _payload(_CSV)
    result = verify_member_inventory_record(payload, _CSV)
    assert result.status is MemberInventoryVerificationStatus.REPRODUCED


def test_verify_reports_source_mismatch_for_the_wrong_member() -> None:
    payload = _payload(_CSV)
    result = verify_member_inventory_record(payload, b"different,bytes\n1,2\n")
    assert result.status is MemberInventoryVerificationStatus.SOURCE_MISMATCH


def test_verify_reports_payload_unreadable_for_an_unknown_version() -> None:
    payload = _payload(_CSV)
    payload["payload_version"] = 999
    result = verify_member_inventory_record(payload, _CSV)
    assert result.status is MemberInventoryVerificationStatus.PAYLOAD_UNREADABLE


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p.__setitem__("stray", 1), "unexpected keys"),
        (lambda p: p.__setitem__("sheet_name", ""), "sheet_name"),
        (lambda p: p.__setitem__("member_sha256", "deadbeef"), "member_sha256"),
        (lambda p: p.__setitem__("cells", {"not": "a list"}), "not a list"),
    ],
)
def test_verify_reports_payload_unreadable_for_each_malformation(mutate, needle: str) -> None:
    payload = _payload(_CSV)
    mutate(payload)
    result = verify_member_inventory_record(payload, _CSV)
    assert result.status is MemberInventoryVerificationStatus.PAYLOAD_UNREADABLE
    assert needle in result.detail


def test_verify_mismatches_when_member_bytes_are_not_utf8_for_the_claimed_grid() -> None:
    # A fabricated record naming bytes that are not UTF-8 delimited text: the sha matches
    # (it is these very bytes), but they cannot be re-read as the claimed grid.
    junk = b"\xff\xfe\x00rubbish"
    payload = {
        "cells": [{"col": 0, "row": 0, "text": "x"}],
        "col_count": 1,
        "member_sha256": hashlib.sha256(junk).hexdigest(),
        "payload_version": 1,
        "row_count": 1,
        "sheet_name": "fake.csv",
    }
    result = verify_member_inventory_record(payload, junk)
    assert result.status is MemberInventoryVerificationStatus.MISMATCHED
    assert "cannot be re-read" in result.detail


def test_cell_text_from_payload_returns_none_for_absent_cell() -> None:
    payload = _payload(_CSV)
    assert cell_text_from_payload(payload, row=0, col=0) == "t_ms"
    assert cell_text_from_payload(payload, row=100, col=100) is None


def test_corrupting_a_stored_cell_is_a_mismatch_naming_the_cell() -> None:
    payload = _payload(_CSV)
    for cell in payload["cells"]:
        if cell["row"] == 1 and cell["col"] == 1:
            cell["text"] = "9.99"
    result = verify_member_inventory_record(payload, _CSV)
    assert result.status is MemberInventoryVerificationStatus.MISMATCHED
    assert "row=1, col=1" in result.detail
    assert "9.99" in result.detail


def test_addressed_cell_replays_then_falsifies_then_clears() -> None:
    data = _CSV
    good = _payload(data)
    expected = cell_text_from_payload(good, row=1, col=1)
    assert expected == "0.21"  # row 0 is the header; row 1 is the first data row

    matched = replay_member_cell(good, data, row=1, col=1, expected_text=expected)
    assert matched.outcome is MemberCellReplayOutcome.MATCH

    corrupt = json.loads(member_inventory_record_bytes(good))
    for cell in corrupt["cells"]:
        if cell["row"] == 1 and cell["col"] == 1:
            cell["text"] = "0.10-tampered"
    failed = replay_member_cell(corrupt, data, row=1, col=1, expected_text=expected)
    assert failed.outcome is MemberCellReplayOutcome.FAILED
    assert "row=1, col=1" in failed.detail

    # Restoring the record clears the failure against the very same bytes.
    cleared = replay_member_cell(good, data, row=1, col=1, expected_text=expected)
    assert cleared.outcome is MemberCellReplayOutcome.MATCH


def test_replay_fails_when_expected_text_disagrees() -> None:
    payload = _payload(_CSV)
    result = replay_member_cell(payload, _CSV, row=0, col=0, expected_text="not-t_ms")
    assert result.outcome is MemberCellReplayOutcome.FAILED
    assert "t_ms" in result.detail


def test_replay_fails_for_an_absent_cell() -> None:
    payload = _payload(_CSV)
    result = replay_member_cell(payload, _CSV, row=99, col=99, expected_text="")
    assert result.outcome is MemberCellReplayOutcome.FAILED
    assert "no cell" in result.detail


def test_replay_is_unverifiable_when_the_payload_is_unreadable() -> None:
    payload = _payload(_CSV)
    del payload["cells"]
    result = replay_member_cell(payload, _CSV, row=0, col=0, expected_text="t_ms")
    assert result.outcome is MemberCellReplayOutcome.UNVERIFIABLE
