"""The OOXML supplementary-data lane's production path: read a staged ``.docx``'s own
bytes and embed each of its tables as a byte-replayable inventory.

This is the OOXML counterpart to :mod:`carmel.services.member_tables`. Where that module
unpacks a delimited-text archive and embeds each tabulated member, this one reads a single
WordprocessingML document and embeds each of its top-level tables. The container is
different -- a ``.docx`` IS the file, not an archive of members, so there is nothing to
unpack -- but the shape of the output is the same: an embedded, content-addressed,
re-derivable inventory per table, plus a typed refusal for anything that could not be read
as a table rather than a silent drop.

The document is NOT admitted as an evidence document here, and that is deliberate: the
identity gate (:func:`carmel.services.acquisition._looks_like_full_article`) rejects a
supplement by design -- a supplement is not the paper -- and this lane does not weaken it.
The ``.docx``'s bytes are read at runtime from where the operator staged them
(``literature_requests/supplementary/<sha256>/<name>`` or the inbox), exactly as the corpus
model intends; nothing is committed to the repository. Admitting a supplement as an
evidence document proper is a separate ticket (see the module-level note in
:mod:`carmel.services.acquisition`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from carmel.services.ooxml_table_record import (
    OoxmlDocumentUnreadable,
    OoxmlNestedTable,
    OoxmlTableTooLarge,
    compute_ooxml_inventory_sha,
    count_ooxml_tables,
    ooxml_inventory_record_bytes,
    ooxml_inventory_record_payload,
    read_ooxml_table,
)

# ``OoxmlDocumentUnreadable`` is re-exported, not merely imported for typing: it is the
# document-level failure ``embed_ooxml_tables`` propagates (its docstring says so), so a
# caller must be able to catch it THROUGH this production-path module without reaching into
# the core reader -- exactly as member_tables surfaces its own unreadable exception.
__all__ = [
    "EmbeddedOoxmlTable",
    "OoxmlDocumentUnreadable",
    "OoxmlReadRefusal",
    "OoxmlReadRefusalReason",
    "OoxmlTableHarvest",
    "embed_ooxml_tables",
]


class OoxmlReadRefusalReason(StrEnum):
    """Why one of a document's tables did not become an inventory. Each names a refusal the
    core reader (:mod:`carmel.services.ooxml_table_record`) raised for one table, recorded
    rather than raised so the rest of the document's tables still harvest."""

    NESTED_TABLE = "nested_table"
    """The table nests another table inside a cell and cannot be honestly flattened into a
    single grid, so it is refused rather than misaligned."""

    TOO_MANY_CELLS = "too_many_cells"
    """The table's grid exceeds the reader's cell cap and was refused before the allocation
    that would exhaust memory."""


@dataclass(frozen=True)
class OoxmlReadRefusal:
    """One table that could not be turned into an inventory, and why."""

    table_index: int
    reason: OoxmlReadRefusalReason
    detail: str


@dataclass(frozen=True)
class EmbeddedOoxmlTable:
    """One ``.docx`` table's inventory record, embedded VERBATIM so a consumer holding only
    these bytes can see the grid -- without the source document, and without re-deriving.

    The service-level carrier for an OOXML table inventory, mirroring
    :class:`carmel.schemas.datasets.EmbeddedMemberTableInventory` in shape (an address, the
    source digest, and the canonical JSON) but deliberately NOT a stored-envelope schema:
    this ticket ends at the byte-replayable inventory and adds nothing to the envelope or
    identity payload. Wiring an OOXML inventory into a stored dataset is a later ticket.
    """

    inventory_sha256: str
    source_sha256: str
    canonical_json: str


@dataclass(frozen=True)
class OoxmlTableHarvest:
    """Everything the production path produced from one ``.docx``: the embedded inventories,
    in document order, and the tables that could not be read as a single grid."""

    inventories: tuple[EmbeddedOoxmlTable, ...]
    read_refusals: tuple[OoxmlReadRefusal, ...]


def _embed_one(docx_bytes: bytes, *, table_index: int, source_sha256: str) -> EmbeddedOoxmlTable:
    """Read one table of ``docx_bytes`` into an embedded, byte-replayable inventory.

    Raises:
        OoxmlNestedTable, OoxmlTableTooLarge: If the table cannot be read as one grid.
    """
    inventory = read_ooxml_table(docx_bytes, table_index=table_index)
    payload = ooxml_inventory_record_payload(inventory, source_sha256=source_sha256)
    canonical = ooxml_inventory_record_bytes(payload)
    return EmbeddedOoxmlTable(
        inventory_sha256=compute_ooxml_inventory_sha(payload),
        source_sha256=source_sha256,
        canonical_json=canonical.decode("utf-8"),
    )


def embed_ooxml_tables(docx_bytes: bytes) -> OoxmlTableHarvest:
    """Read every top-level table of a ``.docx`` and embed each as an
    :class:`EmbeddedOoxmlTable`.

    A table that cannot be read as one grid (nested, or over the cell cap) is recorded as
    an :class:`OoxmlReadRefusal`, never silently dropped, so a document with one unreadable
    table still yields inventories for the rest. Each inventory's bytes are derived from
    ``docx_bytes`` and its record names their sha256, so what is embedded is exactly what
    the document holds.

    Raises:
        OoxmlDocumentUnreadable: If the whole document is unreadable -- not a ZIP, no
            ``word/document.xml``, malformed XML, a DOCTYPE, or a malformed span. A
            document-level failure is not a per-table refusal: there are no tables to
            harvest, so it is raised rather than swallowed into an empty harvest.
    """
    source_sha256 = hashlib.sha256(docx_bytes).hexdigest()
    count = count_ooxml_tables(docx_bytes)  # Raises OoxmlDocumentUnreadable for a bad document.
    inventories: list[EmbeddedOoxmlTable] = []
    read_refusals: list[OoxmlReadRefusal] = []
    for table_index in range(count):
        try:
            inventories.append(_embed_one(docx_bytes, table_index=table_index, source_sha256=source_sha256))
        except OoxmlNestedTable as exc:
            read_refusals.append(
                OoxmlReadRefusal(
                    table_index=table_index,
                    reason=OoxmlReadRefusalReason.NESTED_TABLE,
                    detail=str(exc),
                )
            )
        except OoxmlTableTooLarge as exc:
            read_refusals.append(
                OoxmlReadRefusal(
                    table_index=table_index,
                    reason=OoxmlReadRefusalReason.TOO_MANY_CELLS,
                    detail=str(exc),
                )
            )
    return OoxmlTableHarvest(inventories=tuple(inventories), read_refusals=tuple(read_refusals))
