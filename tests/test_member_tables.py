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

from carmel.schemas.datasets import EmbeddedMemberTableInventory
from carmel.services.archive_unpack import ArchiveUnpackRefusalReason
from carmel.services.member_table_record import MemberCellReplayOutcome, replay_member_cell
from carmel.services.member_tables import (
    MemberReadRefusalReason,
    embed_member_table,
    unpack_and_embed_member_tables,
)

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


def test_archive_refusals_are_carried_through_the_harvest(tmp_path: Path) -> None:
    harvest = unpack_and_embed_member_tables(
        _zip(("../escape.csv", b"x"), ("ok.csv", _CSV)),
        tmp_path / "extract",
    )

    assert len(harvest.inventories) == 1
    assert [r.reason for r in harvest.archive_refusals] == [ArchiveUnpackRefusalReason.PATH_ESCAPE]
