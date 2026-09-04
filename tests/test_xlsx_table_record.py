"""Tests for the SpreadsheetML (`.xlsx`) sheet inventory record: reading, addressing, byte-replay,
merged-cell representation, the three spreadsheet-specific traps, and typed refusals.

Every workbook under test is SYNTHETIC bytes built by :mod:`tests.xlsx_fixtures`; no paper or
supplementary workbook enters the repository. The property under test is that the record is a claim
the ``.xlsx``'s bytes can refute -- replay re-derives the grid from those bytes rather than reading
it back -- that a merged cell is REPRESENTED rather than fabricated, and that each of the three
traps (formula-vs-cache, displayed-vs-stored, indirection) is handled as the module documents.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from carmel.services import xlsx_table_record
from carmel.services.xlsx_table_record import (
    XLSX_INVENTORY_PAYLOAD_KEYS,
    XLSX_INVENTORY_PAYLOAD_VERSION,
    XlsxCellReplayOutcome,
    XlsxEmptySheet,
    XlsxInventoryVerificationStatus,
    XlsxNoTable,
    XlsxSheetTooLarge,
    XlsxWorkbookUnreadable,
    cell_text_from_payload,
    compute_xlsx_inventory_sha,
    count_xlsx_sheets,
    read_xlsx_sheet,
    read_xlsx_sheets,
    replay_xlsx_cell,
    verify_xlsx_inventory_record,
    xlsx_inventory_record_bytes,
    xlsx_inventory_record_payload,
)
from tests.xlsx_fixtures import (
    PKG_REL_NS,
    S_NS,
    CellSpec,
    rels_part,
    sheet_data,
    workbook_part,
    worksheet_part,
    xlsx_bytes,
    xlsx_with_parts,
    xlsx_without_workbook_part,
)

# A header row of shared strings over two rows of number cells, exactly the shape of a real
# species/velocity supplement sheet.
_SIMPLE = [[["phi", "Su (cm/s)"], [CellSpec("0.40", kind="number"), CellSpec("9.08", kind="number")]]]

# A 2x2 grid of number cells, for the cell-cap boundary tests.
_FOUR_NUMBERS = [
    [CellSpec("a", kind="number"), CellSpec("b", kind="number")],
    [CellSpec("c", kind="number"), CellSpec("d", kind="number")],
]


def _payload(data: bytes, *, sheet_index: int = 0) -> dict:
    inventory = read_xlsx_sheet(data, sheet_index=sheet_index)
    return xlsx_inventory_record_payload(inventory, source_sha256=hashlib.sha256(data).hexdigest())


# --- reading a grid -------------------------------------------------------------------


def test_reads_a_sheet_grid_with_header_and_data() -> None:
    inventory = read_xlsx_sheet(xlsx_bytes(_SIMPLE), sheet_index=0)
    assert inventory.row_count == 2
    assert inventory.col_count == 2
    assert len(inventory.cells) == 4
    header = {(c.row, c.col): c.value for c in inventory.cells if c.row == 0}
    assert header == {(0, 0): "phi", (0, 1): "Su (cm/s)"}


def test_a_workbook_holds_several_sheets_in_order() -> None:
    sheets = [[[CellSpec("a", kind="number")]], [[CellSpec("b", kind="number")], [CellSpec("c", kind="number")]]]
    data = xlsx_bytes(sheets, names=["First", "Second"])
    assert count_xlsx_sheets(data) == 2
    inventories = read_xlsx_sheets(data)
    assert [inv.sheet_index for inv in inventories] == [0, 1]
    assert [inv.sheet_name for inv in inventories] == ["First", "Second"]
    assert [inv.row_count for inv in inventories] == [1, 2]


def test_sheet_index_follows_workbook_order_not_part_filename() -> None:
    # A workbook whose FIRST <sheet> points at sheet2.xml and whose second points at sheet1.xml.
    # A reader that addressed by filename would return them swapped; addressing by workbook order
    # (through the relationships) must return "Front" (sheet2.xml) at index 0.
    parts = {
        "xl/workbook.xml": workbook_part([("Front", "rId2"), ("Back", "rId1")]),
        "xl/_rels/workbook.xml.rels": rels_part([("rId1", "worksheets/sheet1.xml"), ("rId2", "worksheets/sheet2.xml")]),
        "xl/worksheets/sheet1.xml": worksheet_part(sheet_data('<c r="A1"><v>111</v></c>')),
        "xl/worksheets/sheet2.xml": worksheet_part(sheet_data('<c r="A1"><v>222</v></c>')),
    }
    data = xlsx_with_parts(parts)
    front = read_xlsx_sheet(data, sheet_index=0)
    assert front.sheet_name == "Front"
    assert front.part_name == "xl/worksheets/sheet2.xml"
    assert front.cells[0].value == "222"


def test_a_number_cell_stores_its_raw_v_string() -> None:
    inventory = read_xlsx_sheet(xlsx_bytes([[[CellSpec("0.10", kind="number")]]]), sheet_index=0)
    # "0.10", not a float 0.1 -- the raw <v> text, un-normalised.
    assert inventory.cells[0].value == "0.10"
    assert inventory.cells[0].cell_type == "n"


def test_an_inline_string_is_read() -> None:
    inventory = read_xlsx_sheet(xlsx_bytes([[[CellSpec("inlined", kind="inline")]]]), sheet_index=0)
    assert inventory.cells[0].value == "inlined"
    assert inventory.cells[0].cell_type == "inlineStr"


def test_boolean_and_error_cells_store_their_raw_marker() -> None:
    data = xlsx_bytes([[[CellSpec("1", kind="bool"), CellSpec("#DIV/0!", kind="error")]]])
    cells = {c.col: (c.value, c.cell_type) for c in read_xlsx_sheet(data, sheet_index=0).cells}
    assert cells == {0: ("1", "b"), 1: ("#DIV/0!", "e")}


def test_a_sparse_row_skips_absent_columns() -> None:
    # A value at A and at C, with B absent entirely (no <c>): B must not appear as a cell.
    data = xlsx_bytes([[[CellSpec("x", kind="number"), None, CellSpec("z", kind="number")]]])
    positions = {(c.row, c.col): c.value for c in read_xlsx_sheet(data, sheet_index=0).cells}
    assert positions == {(0, 0): "x", (0, 2): "z"}


# --- addressing and byte-replay -------------------------------------------------------


def test_record_addresses_to_its_own_sha() -> None:
    payload = _payload(xlsx_bytes(_SIMPLE))
    assert compute_xlsx_inventory_sha(payload) == hashlib.sha256(xlsx_inventory_record_bytes(payload)).hexdigest()
    assert set(payload) == set(XLSX_INVENTORY_PAYLOAD_KEYS)
    assert payload["payload_version"] == XLSX_INVENTORY_PAYLOAD_VERSION


def test_record_payload_rejects_a_malformed_source_sha() -> None:
    inventory = read_xlsx_sheet(xlsx_bytes(_SIMPLE), sheet_index=0)
    with pytest.raises(ValueError, match="source_sha256 must be 64 lowercase hex"):
        xlsx_inventory_record_payload(inventory, source_sha256="deadbeef")


def test_verify_reproduces_against_the_workbooks_bytes() -> None:
    data = xlsx_bytes(_SIMPLE)
    verification = verify_xlsx_inventory_record(_payload(data), data)
    assert verification.status is XlsxInventoryVerificationStatus.REPRODUCED


def test_verify_reports_source_mismatch_for_the_wrong_workbook() -> None:
    payload = _payload(xlsx_bytes(_SIMPLE))
    other = xlsx_bytes([[[CellSpec("999", kind="number")]]])
    verification = verify_xlsx_inventory_record(payload, other)
    assert verification.status is XlsxInventoryVerificationStatus.SOURCE_MISMATCH


def test_verify_reports_payload_unreadable_for_an_unknown_version() -> None:
    data = xlsx_bytes(_SIMPLE)
    payload = _payload(data)
    payload["payload_version"] = 999
    verification = verify_xlsx_inventory_record(payload, data)
    assert verification.status is XlsxInventoryVerificationStatus.PAYLOAD_UNREADABLE
    assert "readable version" in verification.detail


def test_corrupting_a_stored_cell_is_a_mismatch_naming_the_cell() -> None:
    data = xlsx_bytes(_SIMPLE)
    payload = _payload(data)
    parsed = json.loads(xlsx_inventory_record_bytes(payload).decode("utf-8"))
    # Rewrite the stored value of the (1, 1) cell and re-canonicalise so the record still addresses
    # to a coherent sha -- but no longer to THIS workbook's grid.
    for cell in parsed["cells"]:
        if cell["row"] == 1 and cell["col"] == 1:
            cell["value"] = "999.9"
    verification = verify_xlsx_inventory_record(parsed, data)
    assert verification.status is XlsxInventoryVerificationStatus.MISMATCHED
    assert "row=1, col=1" in verification.detail


# --- trap 1: formula vs cached value --------------------------------------------------


def test_formula_cell_stores_both_the_formula_and_its_cached_value() -> None:
    # A cell computed by a formula, with a cached result. The record must carry BOTH: the formula
    # marks it computed (not measured), the cached value is what the file last stored.
    data = xlsx_bytes([[[CellSpec("18.16", kind="number", formula="A1*2", cache="18.16")]]])
    cell = read_xlsx_sheet(data, sheet_index=0).cells[0]
    assert cell.formula == "A1*2"
    assert cell.value == "18.16"


def test_a_formula_with_no_cached_value_keeps_the_formula_and_an_empty_value() -> None:
    # An uncalculated formula (no <v>). The cell is neither dropped nor given a fabricated result:
    # formula is preserved, value is empty, so a consumer sees "computed, no cached result".
    data = xlsx_bytes([[[CellSpec("", formula="A1*2", cache=None)]]])
    cell = read_xlsx_sheet(data, sheet_index=0).cells[0]
    assert cell.formula == "A1*2"
    assert cell.value == ""


# --- trap 2: displayed vs stored ------------------------------------------------------


def test_a_rounding_number_format_does_not_change_the_stored_value() -> None:
    # The cell carries a style index (a number format that would DISPLAY 0.12); the reader never
    # reads styles, so the record stores the full 0.123456 the file holds, not the rounded display.
    data = xlsx_bytes([[[CellSpec("0.123456", kind="number", style=7)]]])
    assert read_xlsx_sheet(data, sheet_index=0).cells[0].value == "0.123456"


# --- trap 3: indirection (dates and shared strings) -----------------------------------


def test_a_date_serial_is_stored_not_interpreted() -> None:
    # A date is stored as a serial number under a date number-format. The record stores the serial
    # verbatim; interpreting it as a calendar date is a downstream ruling this lane declines.
    data = xlsx_bytes([[[CellSpec("44269", kind="number", style=3)]]])
    cell = read_xlsx_sheet(data, sheet_index=0).cells[0]
    assert cell.value == "44269"
    assert cell.cell_type == "n"


def test_a_shared_string_resolves_to_its_text_not_its_index() -> None:
    # Two cells sharing one string; the datum is the string, not the table index.
    data = xlsx_bytes([[["Ar", "Ar"]]])
    cells = read_xlsx_sheet(data, sheet_index=0).cells
    assert [c.value for c in cells] == ["Ar", "Ar"]
    assert all(c.cell_type == "s" for c in cells)


def test_a_rich_text_shared_string_is_concatenated() -> None:
    # A shared string whose text is split across runs (<r><t>..</t></r>), as a rich-text cell is;
    # the reader concatenates the runs into the whole string, as the .docx lane concatenates runs.
    sst = f'<sst xmlns="{S_NS}" count="1" uniqueCount="1"><si><r><t>H</t></r><r><t>2</t></r><r><t>O</t></r></si></sst>'
    parts = {
        "xl/workbook.xml": workbook_part([("S", "rId1")]),
        "xl/_rels/workbook.xml.rels": rels_part([("rId1", "worksheets/sheet1.xml")]),
        "xl/sharedStrings.xml": sst,
        "xl/worksheets/sheet1.xml": worksheet_part(sheet_data('<c r="A1" t="s"><v>0</v></c>')),
    }
    assert read_xlsx_sheet(xlsx_with_parts(parts), sheet_index=0).cells[0].value == "H2O"


# --- merged cells ---------------------------------------------------------------------


def test_a_merged_range_is_represented_not_fabricated() -> None:
    # A header "Burning velocity" merged across two columns above two data columns. The origin
    # carries the value and its span (1 row x 2 cols); the covered column has NO cell (not a copy
    # of the header); the next header "IGN" starts at grid column 2, not 1, so the grid is aligned.
    rows = [
        [CellSpec("Burning velocity", kind="shared", merge=(1, 2)), None, "IGN"],
        [CellSpec("9.08", kind="number"), CellSpec("1.84", kind="number"), CellSpec("H", kind="inline")],
    ]
    inventory = read_xlsx_sheet(xlsx_bytes([rows]), sheet_index=0)
    row0 = {c.col: (c.value, c.row_span, c.col_span) for c in inventory.cells if c.row == 0}
    assert row0 == {0: ("Burning velocity", 1, 2), 2: ("IGN", 1, 1)}
    assert not any(c.row == 0 and c.col == 1 for c in inventory.cells)
    assert inventory.col_count == 3


def test_a_merge_survives_replay() -> None:
    rows = [
        [CellSpec("Su", kind="shared", merge=(1, 2)), None, "M"],
        [CellSpec("9.08", kind="number"), CellSpec("1.84", kind="number"), CellSpec("H", kind="inline")],
    ]
    data = xlsx_bytes([rows])
    verification = verify_xlsx_inventory_record(_payload(data), data)
    assert verification.status is XlsxInventoryVerificationStatus.REPRODUCED


def test_row_count_covers_the_rows_a_merge_spans() -> None:
    # A label merged DOWN three rows (row_span=3) is the only tall content. row_count must cover the
    # rows the merge actually spans (3), not just the anchor's own row (1) -- mirroring col_count,
    # which already accounts for col_span. Undercounting would claim fewer rows than the grid holds.
    rows = [[CellSpec("Fuel", kind="shared", merge=(3, 1))]]
    inventory = read_xlsx_sheet(xlsx_bytes([rows]), sheet_index=0)
    assert inventory.cells[0].row_span == 3
    assert inventory.row_count == 3


# --- typed refusals -------------------------------------------------------------------


def test_non_zip_bytes_are_workbook_unreadable() -> None:
    with pytest.raises(XlsxWorkbookUnreadable, match="not a readable .xlsx"):
        read_xlsx_sheet(b"this is plainly not a zip archive", sheet_index=0)


def test_a_zip_without_workbook_part_is_unreadable() -> None:
    with pytest.raises(XlsxWorkbookUnreadable, match="no 'xl/workbook.xml'"):
        read_xlsx_sheet(xlsx_without_workbook_part(), sheet_index=0)


def test_malformed_xml_is_workbook_unreadable() -> None:
    data = xlsx_with_parts({"xl/workbook.xml": "<workbook><sheets></oops>"})
    with pytest.raises(XlsxWorkbookUnreadable, match="not well-formed XML"):
        read_xlsx_sheet(data, sheet_index=0)


def test_a_doctype_is_refused_before_parsing() -> None:
    hostile = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE lolz [<!ENTITY a "AAAA">]>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets/></workbook>'
    )
    with pytest.raises(XlsxWorkbookUnreadable, match="DOCTYPE"):
        read_xlsx_sheet(xlsx_with_parts({"xl/workbook.xml": hostile}), sheet_index=0)


def test_a_sheet_relationship_that_resolves_to_no_part_is_unreadable() -> None:
    workbook = workbook_part([("S", "rIdMissing")])
    rels = f'<Relationships xmlns="{PKG_REL_NS}"></Relationships>'
    data = xlsx_with_parts({"xl/workbook.xml": workbook, "xl/_rels/workbook.xml.rels": rels})
    with pytest.raises(XlsxWorkbookUnreadable, match="resolves to no worksheet part"):
        read_xlsx_sheet(data, sheet_index=0)


def test_a_sheet_pointing_at_a_non_worksheet_relationship_is_unreadable() -> None:
    # A <sheet> whose r:id resolves to a relationship that is NOT a worksheet (here the shared
    # string table). The target is a real part but not a worksheet, so the sheet resolves to no
    # worksheet part and is refused -- rather than being parsed as a worksheet, finding no
    # <sheetData>, and being misclassified as an empty sheet.
    non_worksheet_rel = (
        f'<Relationships xmlns="{PKG_REL_NS}">'
        f'<Relationship Id="rId1" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        f'Target="sharedStrings.xml"/></Relationships>'
    )
    parts = {
        "xl/workbook.xml": workbook_part([("S", "rId1")]),
        "xl/_rels/workbook.xml.rels": non_worksheet_rel,
        "xl/sharedStrings.xml": f'<sst xmlns="{S_NS}" count="0" uniqueCount="0"></sst>',
    }
    with pytest.raises(XlsxWorkbookUnreadable, match="resolves to no worksheet part"):
        read_xlsx_sheet(xlsx_with_parts(parts), sheet_index=0)


def test_a_reversed_merge_range_is_unreadable() -> None:
    # A mergeCell whose end is before its start (C1:A1). Left unchecked it yields a negative column
    # span -- a fabricated merge -- so it is refused as malformed rather than recorded.
    inner = (
        '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>H</t></is></c></row></sheetData>'
        '<mergeCells count="1"><mergeCell ref="C1:A1"/></mergeCells>'
    )
    parts = {
        "xl/workbook.xml": workbook_part([("S", "rId1")]),
        "xl/_rels/workbook.xml.rels": rels_part([("rId1", "worksheets/sheet1.xml")]),
        "xl/worksheets/sheet1.xml": worksheet_part(inner),
    }
    with pytest.raises(XlsxWorkbookUnreadable, match="end before its start"):
        read_xlsx_sheet(xlsx_with_parts(parts), sheet_index=0)


def test_a_malformed_cell_reference_is_unreadable() -> None:
    parts = {
        "xl/workbook.xml": workbook_part([("S", "rId1")]),
        "xl/_rels/workbook.xml.rels": rels_part([("rId1", "worksheets/sheet1.xml")]),
        "xl/worksheets/sheet1.xml": worksheet_part(sheet_data('<c r="1A"><v>1</v></c>')),
    }
    with pytest.raises(XlsxWorkbookUnreadable, match="valid A1 reference"):
        read_xlsx_sheet(xlsx_with_parts(parts), sheet_index=0)


def test_a_shared_string_index_out_of_range_is_unreadable() -> None:
    parts = {
        "xl/workbook.xml": workbook_part([("S", "rId1")]),
        "xl/_rels/workbook.xml.rels": rels_part([("rId1", "worksheets/sheet1.xml")]),
        "xl/sharedStrings.xml": f'<sst xmlns="{S_NS}" count="0" uniqueCount="0"></sst>',
        "xl/worksheets/sheet1.xml": worksheet_part(sheet_data('<c r="A1" t="s"><v>5</v></c>')),
    }
    with pytest.raises(XlsxWorkbookUnreadable, match="out of range"):
        read_xlsx_sheet(xlsx_with_parts(parts), sheet_index=0)


def test_no_sheet_at_index_is_refused() -> None:
    with pytest.raises(XlsxNoTable, match="none at index 5"):
        read_xlsx_sheet(xlsx_bytes(_SIMPLE), sheet_index=5)


def test_an_empty_sheet_is_refused_not_returned_empty() -> None:
    data = xlsx_bytes([[]], names=["Blank"])
    with pytest.raises(XlsxEmptySheet, match="no cell with a value or formula"):
        read_xlsx_sheet(data, sheet_index=0)


def test_sheet_over_the_cell_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xlsx_table_record, "MAX_XLSX_SHEET_CELL_COUNT", 3)
    with pytest.raises(XlsxSheetTooLarge, match="refused unread"):
        read_xlsx_sheet(xlsx_bytes([_FOUR_NUMBERS]), sheet_index=0)


def test_sheet_at_the_cell_cap_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xlsx_table_record, "MAX_XLSX_SHEET_CELL_COUNT", 4)
    assert len(read_xlsx_sheet(xlsx_bytes([_FOUR_NUMBERS]), sheet_index=0).cells) == 4


def test_a_part_over_the_byte_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xlsx_table_record, "MAX_XLSX_PART_BYTES", 16)
    with pytest.raises(XlsxWorkbookUnreadable, match="byte cap|over the"):
        read_xlsx_sheet(xlsx_bytes(_SIMPLE), sheet_index=0)


# --- cell replay ----------------------------------------------------------------------


def test_replay_matches_the_expected_cell_value() -> None:
    data = xlsx_bytes(_SIMPLE)
    replay = replay_xlsx_cell(_payload(data), data, row=1, col=1, expected_text="9.08")
    assert replay.outcome is XlsxCellReplayOutcome.MATCH


def test_replay_fails_for_the_wrong_expected_value() -> None:
    data = xlsx_bytes(_SIMPLE)
    replay = replay_xlsx_cell(_payload(data), data, row=1, col=1, expected_text="0.0")
    assert replay.outcome is XlsxCellReplayOutcome.FAILED
    assert "9.08" in replay.detail


def test_replay_is_unverifiable_for_an_unreadable_payload() -> None:
    data = xlsx_bytes(_SIMPLE)
    payload = _payload(data)
    payload["payload_version"] = 999
    replay = replay_xlsx_cell(payload, data, row=0, col=0, expected_text="phi")
    assert replay.outcome is XlsxCellReplayOutcome.UNVERIFIABLE


def test_cell_text_from_payload_returns_none_for_absent_cell() -> None:
    payload = _payload(xlsx_bytes(_SIMPLE))
    assert cell_text_from_payload(payload, row=99, col=99) is None
    assert cell_text_from_payload(payload, row=0, col=0) == "phi"
