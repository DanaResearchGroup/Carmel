"""Tests for the ``.xlsx`` production path: embedding every worksheet of a workbook as a
byte-replayable inventory, recording per-sheet refusals rather than dropping them, and raising on a
whole-workbook failure.

Every workbook is SYNTHETIC bytes from :mod:`tests.xlsx_fixtures`; no supplementary workbook enters
the repository.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from carmel.services import xlsx_table_record
from carmel.services.xlsx_table_record import (
    XlsxInventoryVerificationStatus,
    XlsxWorkbookUnreadable,
    verify_xlsx_inventory_record,
)
from carmel.services.xlsx_tables import XlsxReadRefusalReason, embed_xlsx_sheets
from tests.xlsx_fixtures import CellSpec, xlsx_bytes


def test_embeds_every_sheet_in_workbook_order() -> None:
    sheets = [
        [[CellSpec("9.08", kind="number")]],
        [[CellSpec("13.7", kind="number")], [CellSpec("15.1", kind="number")]],
    ]
    data = xlsx_bytes(sheets, names=["A", "B"])
    harvest = embed_xlsx_sheets(data)
    assert len(harvest.inventories) == 2
    assert harvest.read_refusals == ()
    source_sha = hashlib.sha256(data).hexdigest()
    for embedded in harvest.inventories:
        assert embedded.source_sha256 == source_sha
        payload = json.loads(embedded.canonical_json)
        # The embedded bytes hash to the advertised address, and reproduce from the workbook's bytes.
        assert hashlib.sha256(embedded.canonical_json.encode("utf-8")).hexdigest() == embedded.inventory_sha256
        assert verify_xlsx_inventory_record(payload, data).status is XlsxInventoryVerificationStatus.REPRODUCED


def test_an_empty_sheet_is_recorded_as_a_refusal_not_dropped() -> None:
    # A workbook with one populated sheet and one blank sheet: the blank is a recorded refusal, and
    # the populated sheet still harvests.
    data = xlsx_bytes([[[CellSpec("9.08", kind="number")]], [[]]], names=["Data", "Blank"])
    harvest = embed_xlsx_sheets(data)
    assert len(harvest.inventories) == 1
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].sheet_index == 1
    assert harvest.read_refusals[0].reason is XlsxReadRefusalReason.EMPTY_SHEET


def test_a_sheet_over_the_cell_cap_is_recorded_as_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xlsx_table_record, "MAX_XLSX_SHEET_CELL_COUNT", 1)
    data = xlsx_bytes([[[CellSpec("1", kind="number"), CellSpec("2", kind="number")]]])
    harvest = embed_xlsx_sheets(data)
    assert harvest.inventories == ()
    assert harvest.read_refusals[0].reason is XlsxReadRefusalReason.TOO_MANY_CELLS


def test_an_unreadable_workbook_raises_rather_than_harvesting_empty() -> None:
    with pytest.raises(XlsxWorkbookUnreadable):
        embed_xlsx_sheets(b"not a workbook at all")
