"""The ``.xlsx`` supplementary-data lane's production path: read a staged workbook's own bytes and
embed each of its worksheets as a byte-replayable inventory.

This is the spreadsheet counterpart to :mod:`carmel.services.member_tables` (a delimited-text
archive) and to the OOXML ``.docx`` lane. Where the member lane unpacks an archive and embeds each
tabulated member, this one reads a single workbook and embeds each of its worksheets. The container
is different -- an ``.xlsx`` IS the file, not an archive of members, so there is nothing to unpack --
but the shape of the output is the same: an embedded, content-addressed, re-derivable inventory per
sheet, plus a typed refusal for any sheet that could not be read as a table rather than a silent
drop.

The workbook is NOT admitted as an evidence document here, and that is deliberate: the identity gate
(:func:`carmel.services.acquisition._looks_like_full_article`) rejects a supplement by design -- a
supplement is not the paper -- and this lane does not weaken it. The workbook's bytes are read at
runtime from where the operator staged them (``literature_requests/supplementary/<sha256>/<name>``
or the inbox), exactly as the corpus model intends; nothing is committed to the repository. Admitting
a supplement as an evidence document proper is a separate ticket (see the module-level note in
:mod:`carmel.services.acquisition`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from carmel.services.xlsx_table_record import (
    XlsxEmptySheet,
    XlsxSheetTooLarge,
    compute_xlsx_inventory_sha,
    count_xlsx_sheets,
    read_xlsx_sheet,
    xlsx_inventory_record_bytes,
    xlsx_inventory_record_payload,
)

__all__ = [
    "EmbeddedXlsxTable",
    "XlsxReadRefusal",
    "XlsxReadRefusalReason",
    "XlsxTableHarvest",
    "embed_xlsx_sheets",
]


class XlsxReadRefusalReason(StrEnum):
    """Why one of a workbook's worksheets did not become an inventory. Each names a refusal the
    core reader (:mod:`carmel.services.xlsx_table_record`) raised for one sheet, recorded rather
    than raised so the rest of the workbook's sheets still harvest."""

    EMPTY_SHEET = "empty_sheet"
    """The worksheet holds no cell with a value or a formula -- no table-shaped content -- so it is
    recorded as read-but-empty rather than embedded as a grid of nothing."""

    TOO_MANY_CELLS = "too_many_cells"
    """The worksheet's grid exceeds the reader's cell cap and was refused before the allocation
    that would exhaust memory."""


@dataclass(frozen=True)
class XlsxReadRefusal:
    """One worksheet that could not be turned into an inventory, and why."""

    sheet_index: int
    reason: XlsxReadRefusalReason
    detail: str


@dataclass(frozen=True)
class EmbeddedXlsxTable:
    """One worksheet's inventory record, embedded VERBATIM so a consumer holding only these bytes
    can see the grid -- without the source workbook, and without re-deriving.

    The service-level carrier for an ``.xlsx`` sheet inventory, mirroring the delimited and
    ``.docx`` lanes' carriers in shape (an address, the source digest, and the canonical JSON) but
    deliberately NOT a stored-envelope schema: this ticket ends at the byte-replayable inventory
    and adds nothing to the envelope or identity payload. Wiring an ``.xlsx`` inventory into a
    stored dataset is a later ticket.
    """

    inventory_sha256: str
    source_sha256: str
    canonical_json: str


@dataclass(frozen=True)
class XlsxTableHarvest:
    """Everything the production path produced from one workbook: the embedded inventories, in
    workbook order, and the worksheets that could not be read as a single grid."""

    inventories: tuple[EmbeddedXlsxTable, ...]
    read_refusals: tuple[XlsxReadRefusal, ...]


def _embed_one(xlsx_bytes: bytes, *, sheet_index: int, source_sha256: str) -> EmbeddedXlsxTable:
    """Read one worksheet of ``xlsx_bytes`` into an embedded, byte-replayable inventory.

    Raises:
        XlsxEmptySheet, XlsxSheetTooLarge: If the sheet cannot be read as one non-empty grid.
    """
    inventory = read_xlsx_sheet(xlsx_bytes, sheet_index=sheet_index)
    payload = xlsx_inventory_record_payload(inventory, source_sha256=source_sha256)
    canonical = xlsx_inventory_record_bytes(payload)
    return EmbeddedXlsxTable(
        inventory_sha256=compute_xlsx_inventory_sha(payload),
        source_sha256=source_sha256,
        canonical_json=canonical.decode("utf-8"),
    )


def embed_xlsx_sheets(xlsx_bytes: bytes) -> XlsxTableHarvest:
    """Read every worksheet of an ``.xlsx`` and embed each as an :class:`EmbeddedXlsxTable`.

    A worksheet that cannot be read as one non-empty grid (empty, or over the cell cap) is recorded
    as an :class:`XlsxReadRefusal`, never silently dropped, so a workbook with one unreadable sheet
    still yields inventories for the rest. Each inventory's bytes are derived from ``xlsx_bytes``
    and its record names their sha256, so what is embedded is exactly what the workbook holds.

    Raises:
        XlsxWorkbookUnreadable: If the whole workbook is unreadable -- not a ZIP, no
            ``xl/workbook.xml``, malformed XML, a DOCTYPE, or an unresolvable sheet relationship. A
            workbook-level failure is not a per-sheet refusal: there are no sheets to harvest, so it
            is raised rather than swallowed into an empty harvest.
    """
    source_sha256 = hashlib.sha256(xlsx_bytes).hexdigest()
    count = count_xlsx_sheets(xlsx_bytes)  # Raises XlsxWorkbookUnreadable for a bad workbook.
    inventories: list[EmbeddedXlsxTable] = []
    read_refusals: list[XlsxReadRefusal] = []
    for sheet_index in range(count):
        try:
            inventories.append(_embed_one(xlsx_bytes, sheet_index=sheet_index, source_sha256=source_sha256))
        except XlsxEmptySheet as exc:
            read_refusals.append(
                XlsxReadRefusal(
                    sheet_index=sheet_index,
                    reason=XlsxReadRefusalReason.EMPTY_SHEET,
                    detail=str(exc),
                )
            )
        except XlsxSheetTooLarge as exc:
            read_refusals.append(
                XlsxReadRefusal(
                    sheet_index=sheet_index,
                    reason=XlsxReadRefusalReason.TOO_MANY_CELLS,
                    detail=str(exc),
                )
            )
    return XlsxTableHarvest(inventories=tuple(inventories), read_refusals=tuple(read_refusals))
