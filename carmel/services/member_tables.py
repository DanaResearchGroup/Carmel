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
from carmel.services.archive_unpack import ArchiveRefusal, UnpackedMember, unpack_archive
from carmel.services.member_table_record import (
    MemberTableTooLarge,
    MemberTableUnreadable,
    compute_member_inventory_sha,
    member_inventory_record_bytes,
    member_inventory_record_payload,
    read_delimited_member,
)
from carmel.services.ooxml_tables import (
    EmbeddedOoxmlTable,
    OoxmlDocumentUnreadable,
    embed_ooxml_tables,
)
from carmel.services.xlsx_table_record import XlsxWorkbookUnreadable
from carmel.services.xlsx_tables import EmbeddedXlsxTable, embed_xlsx_sheets

__all__ = [
    "EmbeddedMemberTableInventory",
    "EmbeddedOoxmlTable",
    "EmbeddedXlsxTable",
    "MemberReadRefusal",
    "MemberReadRefusalReason",
    "MemberTableHarvest",
    "embed_member_table",
    "unpack_and_embed_member_tables",
]

#: The member extensions this lane reads as delimited text. A member with any other
#: extension that is not an OOXML suffix below is not refused as hostile -- it simply is not a
#: tabulated member this lane can read, and is recorded as such rather than silently dropped.
_TABULATED_SUFFIXES = (".csv", ".tsv", ".tab", ".txt")

#: The member extension routed to the ``.xlsx`` workbook lane
#: (:func:`carmel.services.xlsx_tables.embed_xlsx_sheets`). A member with this suffix is not a
#: delimited table but an OOXML package whose bytes an existing sibling reader already turns
#: into byte-replayable inventories; this lane only ROUTES those bytes, it does not read them.
_XLSX_SUFFIXES = (".xlsx",)

#: The member extension routed to the WordprocessingML lane
#: (:func:`carmel.services.ooxml_tables.embed_ooxml_tables`). As with ``.xlsx`` above, the read
#: itself belongs to the sibling reader; this lane hands it the member's safely-extracted bytes.
_DOCX_SUFFIXES = (".docx",)


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

    OOXML_UNREADABLE = "ooxml_unreadable"
    """The member's suffix claims ``.xlsx`` or ``.docx`` but its bytes are not a readable
    OOXML package -- not a ZIP, missing the workbook/document part, or a malformed part. It was
    extracted safely; it is simply not the format its suffix declares, so it is refused rather
    than handed on. Fail-closed: a member that is not what its suffix claims is a typed refusal,
    never a silent skip. The sibling reader's own decompression caps mean an OOXML package that
    is a zip bomb also surfaces here rather than expanding."""

    OOXML_TABLE_UNREADABLE = "ooxml_table_unreadable"
    """The member IS a readable ``.xlsx``/``.docx`` package but one of its sheets/tables could
    not be turned into a single grid (empty sheet, nested table, or over the sibling reader's
    cell cap). The rest of the package's sheets/tables still harvest; this records the one that
    did not, carrying the sibling reader's own reason and index in its detail, so it is a
    recorded outcome rather than a silent drop."""


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
    archive-level refusals from unpacking.

    ``inventories`` are the delimited-text members. ``xlsx_inventories`` and
    ``ooxml_inventories`` are the sheets/tables of ``.xlsx``/``.docx`` members routed to the
    sibling readers -- kept as their own tuples because they are a different carrier type
    (:class:`EmbeddedXlsxTable` / :class:`EmbeddedOoxmlTable`), addressed by the member's own
    ``source_sha256`` exactly as a delimited inventory is addressed by ``member_sha256``. Both
    default empty so an archive with no OOXML member, and any pre-existing caller, are unchanged."""

    inventories: tuple[EmbeddedMemberTableInventory, ...]
    read_refusals: tuple[MemberReadRefusal, ...]
    archive_refusals: tuple[ArchiveRefusal, ...]
    xlsx_inventories: tuple[EmbeddedXlsxTable, ...] = ()
    ooxml_inventories: tuple[EmbeddedOoxmlTable, ...] = ()


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


def _embed_delimited_member(
    member: UnpackedMember,
    inventories: list[EmbeddedMemberTableInventory],
    read_refusals: list[MemberReadRefusal],
) -> None:
    """Read one delimited-text member into an inventory, recording a typed refusal instead of
    raising if it is not valid delimited text or its grid is too large to hold."""
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


def _embed_xlsx_member(
    member: UnpackedMember,
    xlsx_inventories: list[EmbeddedXlsxTable],
    read_refusals: list[MemberReadRefusal],
) -> None:
    """Route one ``.xlsx`` member's safely-extracted bytes to the workbook lane.

    This lane reads nothing itself: :func:`~carmel.services.xlsx_tables.embed_xlsx_sheets`
    consumes the member's own bytes (no path, no temp file beyond what unpacking already wrote)
    and addresses each sheet by that member's ``source_sha256``. A member whose suffix says
    ``.xlsx`` but whose bytes are not a readable workbook is a typed ``OOXML_UNREADABLE`` refusal
    -- fail-closed on a member that is not what its suffix claims -- and each sheet the reader
    could not read as one grid is recorded, never dropped.
    """
    data = member.extracted_path.read_bytes()
    try:
        harvest = embed_xlsx_sheets(data)
    except XlsxWorkbookUnreadable as exc:
        read_refusals.append(
            MemberReadRefusal(
                member_display_path=member.member_display_path,
                reason=MemberReadRefusalReason.OOXML_UNREADABLE,
                detail=f"member {member.member_display_path!r} is not a readable .xlsx package: {exc}",
            )
        )
        return
    xlsx_inventories.extend(harvest.inventories)
    for refusal in harvest.read_refusals:
        read_refusals.append(
            MemberReadRefusal(
                member_display_path=member.member_display_path,
                reason=MemberReadRefusalReason.OOXML_TABLE_UNREADABLE,
                detail=f"sheet index {refusal.sheet_index} ({refusal.reason}): {refusal.detail}",
            )
        )


def _embed_docx_member(
    member: UnpackedMember,
    ooxml_inventories: list[EmbeddedOoxmlTable],
    read_refusals: list[MemberReadRefusal],
) -> None:
    """Route one ``.docx`` member's safely-extracted bytes to the WordprocessingML lane.

    The counterpart of :func:`_embed_xlsx_member`:
    :func:`~carmel.services.ooxml_tables.embed_ooxml_tables` consumes the member's own bytes and
    addresses each table by that member's ``source_sha256``. A member whose suffix says ``.docx``
    but whose bytes are not a readable document is a typed ``OOXML_UNREADABLE`` refusal, and each
    table the reader could not read as one grid is recorded rather than dropped.
    """
    data = member.extracted_path.read_bytes()
    try:
        harvest = embed_ooxml_tables(data)
    except OoxmlDocumentUnreadable as exc:
        read_refusals.append(
            MemberReadRefusal(
                member_display_path=member.member_display_path,
                reason=MemberReadRefusalReason.OOXML_UNREADABLE,
                detail=f"member {member.member_display_path!r} is not a readable .docx package: {exc}",
            )
        )
        return
    ooxml_inventories.extend(harvest.inventories)
    for refusal in harvest.read_refusals:
        read_refusals.append(
            MemberReadRefusal(
                member_display_path=member.member_display_path,
                reason=MemberReadRefusalReason.OOXML_TABLE_UNREADABLE,
                detail=f"table index {refusal.table_index} ({refusal.reason}): {refusal.detail}",
            )
        )


def unpack_and_embed_member_tables(archive_bytes: bytes, extraction_root: Path) -> MemberTableHarvest:
    """Unpack ``archive_bytes`` fail-closed into ``extraction_root`` and embed every tabulated
    member as an inventory, routing each member to the lane that reads its format.

    A delimited-text member (``.csv``/``.tsv``/``.tab``/``.txt``) becomes an
    :class:`EmbeddedMemberTableInventory`. A ``.xlsx`` or ``.docx`` member is not a delimited
    table but an OOXML package: its safely-extracted bytes are handed to the existing sibling
    reader (:func:`~carmel.services.xlsx_tables.embed_xlsx_sheets` /
    :func:`~carmel.services.ooxml_tables.embed_ooxml_tables`), and the resulting sheet/table
    inventories land on ``xlsx_inventories`` / ``ooxml_inventories``. This is ROUTING, not a new
    reader: no member type is descended recursively, so a ``.zip`` member is not tabulated and is
    recorded as such rather than unpacked again.

    The archive's own refusals (hostile members) are carried through on the harvest; members that
    unpack safely but cannot be read as a table -- a non-tabulated suffix, invalid delimited text,
    or bytes that are not the OOXML package their suffix claims -- are recorded as
    :class:`MemberReadRefusal`, never silently dropped. Each inventory's bytes are read back from
    where unpacking wrote them, so what is embedded is exactly what is on disk.
    """
    result = unpack_archive(archive_bytes, extraction_root)
    inventories: list[EmbeddedMemberTableInventory] = []
    xlsx_inventories: list[EmbeddedXlsxTable] = []
    ooxml_inventories: list[EmbeddedOoxmlTable] = []
    read_refusals: list[MemberReadRefusal] = []
    for member in result.members:
        lowered = member.member_display_path.lower()
        if lowered.endswith(_TABULATED_SUFFIXES):
            _embed_delimited_member(member, inventories, read_refusals)
        elif lowered.endswith(_XLSX_SUFFIXES):
            _embed_xlsx_member(member, xlsx_inventories, read_refusals)
        elif lowered.endswith(_DOCX_SUFFIXES):
            _embed_docx_member(member, ooxml_inventories, read_refusals)
        else:
            read_refusals.append(
                MemberReadRefusal(
                    member_display_path=member.member_display_path,
                    reason=MemberReadRefusalReason.NOT_TABULATED,
                    detail=(
                        f"member {member.member_display_path!r} is not a format this lane reads "
                        "(delimited text .csv/.tsv/.tab/.txt, or an .xlsx/.docx package)"
                    ),
                )
            )
    return MemberTableHarvest(
        inventories=tuple(inventories),
        read_refusals=tuple(read_refusals),
        archive_refusals=result.refusals,
        xlsx_inventories=tuple(xlsx_inventories),
        ooxml_inventories=tuple(ooxml_inventories),
    )
