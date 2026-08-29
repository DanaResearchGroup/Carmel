"""The supplementary-data lane's production path: unpack a staged archive, read each
tabulated member, and embed it as a byte-replayable table inventory.

This is the middle the schema and acquisition were already built around. Acquisition
stages a received supplementary archive verbatim
(:class:`carmel.schemas.acquisition.SupplementaryFile`); this module unpacks it
fail-closed (:mod:`carmel.services.archive_unpack`), reads each tabulated member into a
member inventory record (:mod:`carmel.services.member_table_record`), and wraps that
record as an :class:`~carmel.schemas.datasets.EmbeddedMemberTableInventory` -- the same
embedded, content-addressed, re-derivable form the PDF table lane produces.

A member's ``sheet_name`` is its display path within the archive. For a delimited-text
member that IS the single table it contains, this is the honest identifier -- there is
no workbook of named sheets to choose from. A true multi-sheet ``.xlsx`` workbook is a
separate, operator-gated delivery (it needs a runtime library this ticket does not add),
at which point one member would yield several inventories, one per real sheet name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from carmel.schemas.datasets import EmbeddedMemberTableInventory
from carmel.services.archive_unpack import ArchiveRefusal, unpack_archive
from carmel.services.member_table_record import (
    MemberTableTooLarge,
    MemberTableUnreadable,
    compute_member_inventory_sha,
    member_inventory_record_bytes,
    member_inventory_record_payload,
    read_delimited_member,
)

__all__ = [
    "MemberReadRefusal",
    "MemberReadRefusalReason",
    "MemberTableHarvest",
    "embed_member_table",
    "unpack_and_embed_member_tables",
]

#: The member extensions this first delivery reads as delimited text. A member with any
#: other extension is not refused as hostile -- it simply is not a tabulated member this
#: lane can read yet, and is recorded as such rather than silently dropped.
_TABULATED_SUFFIXES = (".csv", ".tsv", ".tab", ".txt")


class MemberReadRefusalReason(StrEnum):
    """Why an unpacked member did not become a table inventory. Distinct from an
    :class:`~carmel.services.archive_unpack.ArchiveRefusal`: those are hostile members
    never written to disk; these are members that WERE written safely but could not be
    read as a table."""

    NOT_TABULATED = "not_tabulated"
    """The member's extension is not one this lane reads (not ``.csv``/``.tsv``/``.tab``/
    ``.txt``). It was extracted safely; it is simply not a delimited table."""

    UNREADABLE = "unreadable"
    """The member has a tabulated extension but its bytes are not valid delimited text
    (e.g. not UTF-8)."""

    TOO_MANY_CELLS = "too_many_cells"
    """The member is valid delimited text but its grid exceeds
    :data:`~carmel.services.member_table_record.MAX_MEMBER_CELL_COUNT` fields -- too many
    :class:`~carmel.services.member_table_record.MemberCell` objects to hold in memory. It
    was extracted safely and is legal under every archive byte cap; the cap it crosses is
    the in-memory object count the byte caps do not bound. Refused before the allocation,
    so it is a recorded outcome rather than an OOM."""


@dataclass(frozen=True)
class MemberReadRefusal:
    """One member that unpacked safely but was not turned into an inventory."""

    member_display_path: str
    reason: MemberReadRefusalReason
    detail: str


@dataclass(frozen=True)
class MemberTableHarvest:
    """Everything the production path produced from one archive: the embedded
    inventories, the members that unpacked but could not be read as tables, and the
    archive-level refusals from unpacking."""

    inventories: tuple[EmbeddedMemberTableInventory, ...]
    read_refusals: tuple[MemberReadRefusal, ...]
    archive_refusals: tuple[ArchiveRefusal, ...]


def embed_member_table(member_bytes: bytes, *, sheet_name: str) -> EmbeddedMemberTableInventory:
    """Read one delimited-text member's bytes into an embedded, byte-replayable table
    inventory.

    Raises:
        MemberTableUnreadable: If the bytes are not valid delimited text.
    """
    inventory = read_delimited_member(member_bytes, sheet_name=sheet_name)
    member_sha256 = hashlib.sha256(member_bytes).hexdigest()
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    canonical = member_inventory_record_bytes(payload)
    return EmbeddedMemberTableInventory(
        inventory_sha256=compute_member_inventory_sha(payload),
        member_sha256=member_sha256,
        canonical_json=canonical.decode("utf-8"),
    )


def _is_tabulated(member_display_path: str) -> bool:
    return member_display_path.lower().endswith(_TABULATED_SUFFIXES)


def unpack_and_embed_member_tables(archive_bytes: bytes, extraction_root: Path) -> MemberTableHarvest:
    """Unpack ``archive_bytes`` fail-closed into ``extraction_root`` and embed every
    tabulated member as an :class:`EmbeddedMemberTableInventory`.

    The archive's own refusals (hostile members) are carried through on the harvest;
    members that unpack safely but are not tabulated text are recorded as
    :class:`MemberReadRefusal`, never silently dropped. Each inventory's bytes are read
    back from where unpacking wrote them, so what is embedded is exactly what is on disk.
    """
    result = unpack_archive(archive_bytes, extraction_root)
    inventories: list[EmbeddedMemberTableInventory] = []
    read_refusals: list[MemberReadRefusal] = []
    for member in result.members:
        if not _is_tabulated(member.member_display_path):
            read_refusals.append(
                MemberReadRefusal(
                    member_display_path=member.member_display_path,
                    reason=MemberReadRefusalReason.NOT_TABULATED,
                    detail=f"member {member.member_display_path!r} is not a delimited-text table this lane reads",
                )
            )
            continue
        data = member.extracted_path.read_bytes()
        try:
            inventories.append(embed_member_table(data, sheet_name=member.member_display_path))
        except MemberTableUnreadable as exc:
            read_refusals.append(
                MemberReadRefusal(
                    member_display_path=member.member_display_path,
                    reason=MemberReadRefusalReason.UNREADABLE,
                    detail=str(exc),
                )
            )
        except MemberTableTooLarge as exc:
            read_refusals.append(
                MemberReadRefusal(
                    member_display_path=member.member_display_path,
                    reason=MemberReadRefusalReason.TOO_MANY_CELLS,
                    detail=str(exc),
                )
            )
    return MemberTableHarvest(
        inventories=tuple(inventories),
        read_refusals=tuple(read_refusals),
        archive_refusals=result.refusals,
    )
