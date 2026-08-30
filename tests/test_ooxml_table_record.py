"""Tests for the OOXML (`.docx`) table inventory record: reading, addressing, byte-replay,
merged-cell representation, and typed refusals.

Every document under test is SYNTHETIC bytes built by :mod:`tests.ooxml_fixtures`; no paper
or supplementary document enters the repository. The property under test is that the record
is a claim the ``.docx``'s bytes can refute -- replay re-derives the grid from those bytes
rather than reading it back -- and that a merged cell is REPRESENTED, never fabricated into
a misaligned grid.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from carmel.services import ooxml_table_record
from carmel.services.ooxml_table_record import (
    OOXML_INVENTORY_PAYLOAD_KEYS,
    OOXML_INVENTORY_PAYLOAD_VERSION,
    OoxmlCellReplayOutcome,
    OoxmlDocumentUnreadable,
    OoxmlInventoryVerificationStatus,
    OoxmlNestedTable,
    OoxmlNoTable,
    OoxmlTableTooLarge,
    cell_text_from_payload,
    compute_ooxml_inventory_sha,
    count_ooxml_tables,
    ooxml_inventory_record_bytes,
    ooxml_inventory_record_payload,
    read_ooxml_table,
    read_ooxml_tables,
    replay_ooxml_cell,
    verify_ooxml_inventory_record,
)
from tests.ooxml_fixtures import CellSpec, docx_bytes, docx_without_document_part

_SIMPLE = [
    [["phi", "Su (cm/s)"], ["0.40", "9.08"], ["0.45", "13.73"]],
]


def _payload(data: bytes, *, table_index: int = 0) -> dict:
    inventory = read_ooxml_table(data, table_index=table_index)
    source_sha256 = hashlib.sha256(data).hexdigest()
    return ooxml_inventory_record_payload(inventory, source_sha256=source_sha256)


# --- reading a grid -------------------------------------------------------------------


def test_reads_a_table_grid_with_header_and_data() -> None:
    inventory = read_ooxml_table(docx_bytes(_SIMPLE), table_index=0)
    assert inventory.row_count == 3
    assert inventory.col_count == 2
    assert len(inventory.cells) == 6
    header = {(c.row, c.col): c.text for c in inventory.cells if c.row == 0}
    assert header == {(0, 0): "phi", (0, 1): "Su (cm/s)"}


def test_a_document_can_hold_several_tables_in_order() -> None:
    tables = [[["a"]], [["b"], ["c"]], [["d"]]]
    assert count_ooxml_tables(docx_bytes(tables)) == 3
    inventories = read_ooxml_tables(docx_bytes(tables))
    assert [inv.table_index for inv in inventories] == [0, 1, 2]
    assert [inv.row_count for inv in inventories] == [1, 2, 1]


def test_a_blank_cell_is_a_present_cell_with_empty_text() -> None:
    inventory = read_ooxml_table(docx_bytes([[["", "x"]]]), table_index=0)
    texts = {(c.row, c.col): c.text for c in inventory.cells}
    assert texts == {(0, 0): "", (0, 1): "x"}


# --- addressing and byte-replay -------------------------------------------------------


def test_record_addresses_to_its_own_sha() -> None:
    payload = _payload(docx_bytes(_SIMPLE))
    assert compute_ooxml_inventory_sha(payload) == hashlib.sha256(ooxml_inventory_record_bytes(payload)).hexdigest()
    assert set(payload) == set(OOXML_INVENTORY_PAYLOAD_KEYS)
    assert payload["payload_version"] == OOXML_INVENTORY_PAYLOAD_VERSION


def test_record_payload_rejects_a_malformed_source_sha() -> None:
    inventory = read_ooxml_table(docx_bytes(_SIMPLE), table_index=0)
    with pytest.raises(ValueError, match="source_sha256 must be 64 lowercase hex"):
        ooxml_inventory_record_payload(inventory, source_sha256="deadbeef")


def test_verify_reproduces_against_the_documents_bytes() -> None:
    data = docx_bytes(_SIMPLE)
    verification = verify_ooxml_inventory_record(_payload(data), data)
    assert verification.status is OoxmlInventoryVerificationStatus.REPRODUCED


def test_verify_reports_source_mismatch_for_the_wrong_document() -> None:
    payload = _payload(docx_bytes(_SIMPLE))
    other = docx_bytes([[["different"]]])
    verification = verify_ooxml_inventory_record(payload, other)
    assert verification.status is OoxmlInventoryVerificationStatus.SOURCE_MISMATCH


def test_verify_reports_payload_unreadable_for_an_unknown_version() -> None:
    data = docx_bytes(_SIMPLE)
    payload = _payload(data)
    payload["payload_version"] = 999
    verification = verify_ooxml_inventory_record(payload, data)
    assert verification.status is OoxmlInventoryVerificationStatus.PAYLOAD_UNREADABLE
    assert "readable version" in verification.detail


def test_corrupting_a_stored_cell_is_a_mismatch_naming_the_cell() -> None:
    data = docx_bytes(_SIMPLE)
    payload = _payload(data)
    parsed = json.loads(ooxml_inventory_record_bytes(payload).decode("utf-8"))
    # Rewrite the stored text of the (1, 1) cell and re-canonicalise so the record still
    # addresses to a coherent sha -- but no longer to THIS document's grid.
    for cell in parsed["cells"]:
        if cell["row"] == 1 and cell["col"] == 1:
            cell["text"] = "999.9"
    verification = verify_ooxml_inventory_record(parsed, data)
    assert verification.status is OoxmlInventoryVerificationStatus.MISMATCHED
    assert "row=1, col=1" in verification.detail


# --- merged cells ---------------------------------------------------------------------


def test_vertical_merge_is_represented_not_fabricated() -> None:
    # A "Method" column whose value H at the top spans the two data rows below it, exactly
    # as the real corpus file draws it. The origin carries the value; the continuations
    # keep their own blank text AND a flag -- the value is never copied downward.
    tables = [
        [
            ["phi", "Method"],
            ["0.40", CellSpec(text="H", v_merge="restart")],
            ["0.45", CellSpec(text="", v_merge="continue")],
            ["0.50", CellSpec(text="", v_merge="continue")],
        ]
    ]
    inventory = read_ooxml_table(docx_bytes(tables), table_index=0)
    merges = {(c.row, c.col): (c.text, c.row_merge) for c in inventory.cells if c.col == 1 and c.row > 0}
    assert merges == {
        (1, 1): ("H", "start"),
        (2, 1): ("", "continue"),
        (3, 1): ("", "continue"),
    }


def test_horizontal_span_advances_the_grid_column_and_records_its_extent() -> None:
    # A header cell spanning two columns, then a single cell. The spanning cell starts at
    # grid column 0 with col_span=2; the next cell starts at grid column 2, not 1, so the
    # grid is not misaligned. No cell is fabricated for the covered column 1.
    tables = [[[CellSpec(text="Burning velocity", grid_span=2), "Note"], ["9.08", "1.84", "H"]]]
    inventory = read_ooxml_table(docx_bytes(tables), table_index=0)
    row0 = {c.col: (c.text, c.col_span) for c in inventory.cells if c.row == 0}
    assert row0 == {0: ("Burning velocity", 2), 2: ("Note", 1)}
    assert inventory.col_count == 3
    assert not any(c.row == 0 and c.col == 1 for c in inventory.cells)


def test_span_and_merge_survive_replay() -> None:
    tables = [
        [[CellSpec(text="Su", grid_span=2), "M"], ["9.08", "1.84", CellSpec(text="H", v_merge="restart")]],
    ]
    data = docx_bytes(tables)
    verification = verify_ooxml_inventory_record(_payload(data), data)
    assert verification.status is OoxmlInventoryVerificationStatus.REPRODUCED


# --- typed refusals -------------------------------------------------------------------


def test_non_zip_bytes_are_document_unreadable() -> None:
    with pytest.raises(OoxmlDocumentUnreadable, match="not a readable .docx"):
        read_ooxml_table(b"this is plainly not a zip archive", table_index=0)


def test_a_zip_without_document_part_is_unreadable() -> None:
    with pytest.raises(OoxmlDocumentUnreadable, match="no 'word/document.xml'"):
        read_ooxml_table(docx_without_document_part(), table_index=0)


def test_malformed_xml_is_document_unreadable() -> None:
    data = docx_bytes([], document_override="<w:document><w:body><w:tbl></oops>")
    with pytest.raises(OoxmlDocumentUnreadable, match="not well-formed XML"):
        read_ooxml_table(data, table_index=0)


def test_a_doctype_is_refused_before_parsing() -> None:
    hostile = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE lolz [<!ENTITY a "AAAA">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body></w:body></w:document>"
    )
    data = docx_bytes([], document_override=hostile)
    with pytest.raises(OoxmlDocumentUnreadable, match="DOCTYPE"):
        read_ooxml_table(data, table_index=0)


def test_a_document_with_no_table_is_refused_not_empty() -> None:
    data = docx_bytes([])
    assert count_ooxml_tables(data) == 0
    with pytest.raises(OoxmlNoTable, match="none at index"):
        read_ooxml_table(data, table_index=0)


def test_index_past_the_last_table_is_refused() -> None:
    with pytest.raises(OoxmlNoTable, match="index 5"):
        read_ooxml_table(docx_bytes(_SIMPLE), table_index=5)


def test_a_nested_table_is_refused_not_flattened() -> None:
    nested = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>outer</w:t></w:r></w:p>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>inner</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "</w:tc></w:tr></w:tbl>"
        "</w:body></w:document>"
    )
    data = docx_bytes([], document_override=nested)
    with pytest.raises(OoxmlNestedTable, match="nested table"):
        read_ooxml_table(data, table_index=0)


def test_a_malformed_gridspan_is_document_unreadable() -> None:
    tables = [[[CellSpec(text="x", grid_span=0)]]]
    with pytest.raises(OoxmlDocumentUnreadable, match="gridSpan"):
        read_ooxml_table(docx_bytes(tables), table_index=0)


def test_table_over_the_cell_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ooxml_table_record, "MAX_OOXML_TABLE_CELL_COUNT", 3)
    tables = [[["a", "b"], ["c", "d"]]]  # four cells, cap is three
    with pytest.raises(OoxmlTableTooLarge, match="refused unread"):
        read_ooxml_table(docx_bytes(tables), table_index=0)


def test_table_at_the_cell_cap_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ooxml_table_record, "MAX_OOXML_TABLE_CELL_COUNT", 4)
    tables = [[["a", "b"], ["c", "d"]]]
    inventory = read_ooxml_table(docx_bytes(tables), table_index=0)
    assert len(inventory.cells) == 4


# --- cell replay ----------------------------------------------------------------------


def test_replay_matches_the_expected_cell_text() -> None:
    data = docx_bytes(_SIMPLE)
    replay = replay_ooxml_cell(_payload(data), data, row=1, col=1, expected_text="9.08")
    assert replay.outcome is OoxmlCellReplayOutcome.MATCH


def test_replay_fails_for_the_wrong_expected_text() -> None:
    data = docx_bytes(_SIMPLE)
    replay = replay_ooxml_cell(_payload(data), data, row=1, col=1, expected_text="0.0")
    assert replay.outcome is OoxmlCellReplayOutcome.FAILED
    assert "9.08" in replay.detail


def test_replay_is_unverifiable_for_an_unreadable_payload() -> None:
    data = docx_bytes(_SIMPLE)
    payload = _payload(data)
    payload["payload_version"] = 999
    replay = replay_ooxml_cell(payload, data, row=0, col=0, expected_text="phi")
    assert replay.outcome is OoxmlCellReplayOutcome.UNVERIFIABLE


def test_cell_text_from_payload_returns_none_for_absent_cell() -> None:
    payload = _payload(docx_bytes(_SIMPLE))
    assert cell_text_from_payload(payload, row=99, col=99) is None
    assert cell_text_from_payload(payload, row=0, col=0) == "phi"
