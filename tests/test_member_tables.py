"""Tests for the supplementary-data production path: unpack an archive, embed each
tabulated member as a byte-replayable inventory.

Synthetic archives only; no supplementary content is committed.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from carmel.schemas.datasets import EmbeddedMemberTableInventory
from carmel.services import member_table_record
from carmel.services.archive_unpack import ArchiveUnpackRefusalReason, unpack_archive
from carmel.services.member_table_record import MemberCellReplayOutcome, replay_member_cell
from carmel.services.member_tables import (
    MemberReadRefusalReason,
    embed_member_table,
    unpack_and_embed_member_tables,
)
from carmel.services.xlsx_tables import embed_xlsx_sheets
from tests.ooxml_fixtures import docx_bytes
from tests.xlsx_fixtures import xlsx_bytes

_CSV = b"t_ms,CH4,O2,CO2\n0.0,0.21,0.79,0.0\n0.5,0.10,0.60,0.11\n1.0,0.02,0.40,0.19\n"


def _zip(*members: tuple[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, data in members:
            archive.writestr(arcname, data)
    return buffer.getvalue()


def test_embed_member_table_yields_a_valid_embedded_inventory() -> None:
    inventory = embed_member_table(_CSV, sheet_name="species.csv")
    assert isinstance(inventory, EmbeddedMemberTableInventory)
    assert inventory.member_sha256 == hashlib.sha256(_CSV).hexdigest()
    assert inventory.has_cell(row=1, col=1)
    assert inventory.cell_text(row=1, col=1) == "0.21"


def test_production_path_unpacks_and_embeds_a_tabulated_member(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(_zip(("supp/species.csv", _CSV)), tmp_path / "extract")

    assert harvest.archive_refusals == ()
    assert harvest.read_refusals == ()
    assert len(harvest.inventories) == 1
    inventory = harvest.inventories[0]
    assert inventory.member_sha256 == hashlib.sha256(_CSV).hexdigest()


def test_embedded_member_cell_replays_against_the_members_stored_bytes(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    harvest = unpack_and_embed_member_tables(_zip(("supp/species.csv", _CSV)), root)
    inventory = harvest.inventories[0]

    member_bytes = (root / "supp" / "species.csv").read_bytes()
    assert hashlib.sha256(member_bytes).hexdigest() == inventory.member_sha256

    payload = json.loads(inventory.canonical_json)
    replay = replay_member_cell(payload, member_bytes, row=2, col=1, expected_text="0.10")
    assert replay.outcome is MemberCellReplayOutcome.MATCH


def test_non_tabulated_member_is_recorded_not_dropped(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(
        _zip(("readme.pdf", b"%PDF-1.7 not a table"), ("data.csv", _CSV)),
        tmp_path / "extract",
    )

    assert [i.member_sha256 for i in harvest.inventories] == [hashlib.sha256(_CSV).hexdigest()]
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.NOT_TABULATED
    assert harvest.read_refusals[0].member_display_path == "readme.pdf"


def test_tabulated_but_non_utf8_member_is_recorded_unreadable(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(
        _zip(("broken.csv", b"a,b\n\xff\xfe,2\n")),
        tmp_path / "extract",
    )

    assert harvest.inventories == ()
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.UNREADABLE


def test_oversize_member_is_recorded_and_does_not_stop_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A member legal under every archive byte cap but over the parse's cell cap is refused
    # with TOO_MANY_CELLS, and the archive keeps processing -- a later member that fits is
    # still read, matching what the total-size cap already documents for itself. The cap is
    # lowered here only so the test need not build eight million cells.
    monkeypatch.setattr(member_table_record, "MAX_MEMBER_CELL_COUNT", 6)
    over = b"a,b,c\n1,2,3\n4,5,6\n"  # 9 cells > cap
    fits = b"x,y\n1,2\n"  # 4 cells <= cap
    harvest = unpack_and_embed_member_tables(
        _zip(("big.csv", over), ("ok.csv", fits)),
        tmp_path / "extract",
    )

    assert [i.member_sha256 for i in harvest.inventories] == [hashlib.sha256(fits).hexdigest()]
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.TOO_MANY_CELLS
    assert harvest.read_refusals[0].member_display_path == "big.csv"


def test_archive_refusals_are_carried_through_the_harvest(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(
        _zip(("../escape.csv", b"x"), ("ok.csv", _CSV)),
        tmp_path / "extract",
    )

    assert len(harvest.inventories) == 1
    assert [r.reason for r in harvest.archive_refusals] == [ArchiveUnpackRefusalReason.PATH_ESCAPE]


# --- routing OOXML members to the sibling readers ------------------------------------------

_XLSX = xlsx_bytes([[["t_ms", "CH4"], ["0.0", "0.21"], ["0.5", "0.10"]]])
_DOCX = docx_bytes([[["t_ms", "CH4"], ["0.0", "0.21"]]])


def test_nested_xlsx_member_becomes_an_embedded_inventory(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(_zip(("supp/data.xlsx", _XLSX)), tmp_path / "extract")

    assert harvest.inventories == ()  # not a delimited-text inventory
    assert harvest.read_refusals == ()
    assert len(harvest.xlsx_inventories) == 1
    # The sheet's inventory is addressed by the MEMBER's own bytes, exactly as a delimited
    # member is addressed by member_sha256 -- the archive is named by re-derivation, not stored.
    assert harvest.xlsx_inventories[0].source_sha256 == hashlib.sha256(_XLSX).hexdigest()


def test_nested_docx_member_becomes_an_embedded_inventory(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(_zip(("supp/table.docx", _DOCX)), tmp_path / "extract")

    assert harvest.inventories == ()
    assert harvest.read_refusals == ()
    assert len(harvest.ooxml_inventories) == 1
    assert harvest.ooxml_inventories[0].source_sha256 == hashlib.sha256(_DOCX).hexdigest()


def test_nested_xlsx_inventory_rederives_from_the_archive_bytes(tmp_path: Path) -> None:
    # The addressing claim, demonstrated: given ONLY the archive's bytes, unpacking finds the
    # one member whose bytes hash to the inventory's source_sha256, and re-embedding those bytes
    # reproduces the identical inventory address and canonical record. Every link in
    # archive -> member -> inventory re-derives; none is merely declared.
    archive = _zip(("supp/data.xlsx", _XLSX))
    inventory = unpack_and_embed_member_tables(archive, tmp_path / "extract").xlsx_inventories[0]

    result = unpack_archive(archive, tmp_path / "rederive")
    member = next(m for m in result.members if m.member_display_path == "supp/data.xlsx")
    member_bytes = member.extracted_path.read_bytes()
    assert hashlib.sha256(member_bytes).hexdigest() == inventory.source_sha256

    reharvest = embed_xlsx_sheets(member_bytes)
    assert reharvest.inventories[0].inventory_sha256 == inventory.inventory_sha256
    assert reharvest.inventories[0].canonical_json == inventory.canonical_json


def test_xlsx_member_that_is_not_a_workbook_is_refused_ooxml_unreadable(tmp_path: Path) -> None:
    # A member whose suffix claims .xlsx but whose bytes are not a readable package fails closed
    # with a typed refusal -- never a silent skip and never a guess.
    harvest = unpack_and_embed_member_tables(
        _zip(("supp/lies.xlsx", b"PK\x03\x04 but not really a workbook")),
        tmp_path / "extract",
    )

    assert harvest.xlsx_inventories == ()
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.OOXML_UNREADABLE
    assert harvest.read_refusals[0].member_display_path == "supp/lies.xlsx"


def test_docx_member_that_is_not_a_document_is_refused_ooxml_unreadable(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(
        _zip(("supp/lies.docx", b"not a zip at all")),
        tmp_path / "extract",
    )

    assert harvest.ooxml_inventories == ()
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.OOXML_UNREADABLE
    assert harvest.read_refusals[0].member_display_path == "supp/lies.docx"


def test_unreadable_ooxml_member_does_not_stop_the_archive(tmp_path: Path) -> None:
    # An unreadable .xlsx is refused individually; a good sibling member still harvests.
    harvest = unpack_and_embed_member_tables(
        _zip(("bad.xlsx", b"corrupt"), ("ok.csv", _CSV), ("good.xlsx", _XLSX)),
        tmp_path / "extract",
    )

    assert [i.member_sha256 for i in harvest.inventories] == [hashlib.sha256(_CSV).hexdigest()]
    assert len(harvest.xlsx_inventories) == 1
    assert [r.reason for r in harvest.read_refusals] == [MemberReadRefusalReason.OOXML_UNREADABLE]


def test_zip_member_stays_not_tabulated_with_no_recursive_descent(tmp_path: Path) -> None:
    # A .zip member is not routed and not descended into: its inner .csv is never extracted or
    # embedded. An archive inside an archive is a refusal, not a feature.
    nested = _zip(("inner.csv", _CSV))
    harvest = unpack_and_embed_member_tables(_zip(("bundle.zip", nested)), tmp_path / "extract")

    assert harvest.inventories == ()
    assert harvest.xlsx_inventories == ()
    assert harvest.ooxml_inventories == ()
    assert [r.reason for r in harvest.read_refusals] == [MemberReadRefusalReason.NOT_TABULATED]
    assert harvest.read_refusals[0].member_display_path == "bundle.zip"


def test_unreadable_sheet_within_a_routed_xlsx_is_recorded_not_dropped(tmp_path: Path) -> None:
    # A workbook whose second sheet is empty: the good sheet harvests, the empty one is recorded
    # as OOXML_TABLE_UNREADABLE rather than silently dropped.
    workbook = xlsx_bytes([[["t_ms", "CH4"], ["0.0", "0.21"]], []])
    harvest = unpack_and_embed_member_tables(_zip(("supp/two.xlsx", workbook)), tmp_path / "extract")

    assert len(harvest.xlsx_inventories) == 1
    assert len(harvest.read_refusals) == 1
    assert harvest.read_refusals[0].reason is MemberReadRefusalReason.OOXML_TABLE_UNREADABLE
    assert harvest.read_refusals[0].member_display_path == "supp/two.xlsx"
