"""A byte-replayable table inventory for an OOXML (WordprocessingML) ``.docx`` table.

This is the OOXML counterpart to :mod:`carmel.services.member_table_record`. Where that
module derives a grid from a delimited member's own bytes (CSV/TSV) and re-derives it to
verify, this one derives a grid from a ``.docx``'s own bytes -- the ``word/document.xml``
part inside the OOXML ZIP -- and re-derives it the same way. The shape of the two is
deliberately identical: a canonical-JSON record that hashes to its own address, that
names the document it came from, and that a verifier can REPRODUCE from that document's
raw bytes without any store, so a ``.docx`` table cell is as byte-replayable as a CSV
member cell or a PDF table cell.

It is a SEPARATE record type, not a reuse of the delimited one, and that is not an
oversight. Two structural facts a ``.docx`` table states that a delimited member cannot,
and that the delimited record therefore has no field for:

* ONE ``.docx`` holds MANY tables, in document order, so the address needs a
  ``table_index`` the delimited record (one member, one grid) never had.
* An OOXML cell can span columns (``w:gridSpan``) or rows (``w:vMerge``). A delimited
  grid is strictly rectangular; a ``.docx`` grid is not. This record therefore carries,
  per cell, the GRID column it starts at, how many columns it spans, and whether it is a
  vertical-merge origin or a continuation of one -- so a merged cell is REPRESENTED
  rather than either dropped (a misaligned grid) or its value fabricated downward into
  cells the document leaves blank (a fabricated join). See :class:`OoxmlCell`.

Forcing a ``.docx`` table through the delimited record would mean either losing those
facts or bumping ``MEMBER_INVENTORY_PAYLOAD_VERSION`` -- a stored-data-semantics change
that is the operator's call, not this lane's. A parallel record type is the same choice
the member lane already made against the PDF record for the same kind of reason.

Scope of this first delivery: ONE ``.docx`` (a single WordprocessingML package), tables
that are direct children of the document body. A table that nests another table is
refused rather than flattened (:class:`OoxmlNestedTable`); ``.doc``, ``.xlsx`` and
archives-of-docx are out of scope entirely and are not read here.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from xml.etree import ElementTree

from carmel.services.dataset_store import canonical_json_bytes

__all__ = [
    "MAX_OOXML_DOCUMENT_XML_BYTES",
    "MAX_OOXML_TABLE_CELL_COUNT",
    "OOXML_INVENTORY_PAYLOAD_KEYS",
    "OOXML_INVENTORY_PAYLOAD_VERSION",
    "OoxmlCell",
    "OoxmlCellReplay",
    "OoxmlCellReplayOutcome",
    "OoxmlDocumentUnreadable",
    "OoxmlInventoryVerification",
    "OoxmlInventoryVerificationStatus",
    "OoxmlNestedTable",
    "OoxmlNoTable",
    "OoxmlTableInventory",
    "OoxmlTableTooLarge",
    "cell_text_from_payload",
    "compute_ooxml_inventory_sha",
    "count_ooxml_tables",
    "ooxml_inventory_record_bytes",
    "ooxml_inventory_record_payload",
    "read_ooxml_table",
    "read_ooxml_tables",
    "replay_ooxml_cell",
    "verify_ooxml_inventory_record",
]

#: The on-disk shape of an OOXML table inventory record. Bumped whenever a field is added
#: or changed in a way that is not simply optional-with-a-default. A reader that does not
#: know a version must not guess at its shape -- see :func:`verify_ooxml_inventory_record`,
#: which returns ``PAYLOAD_UNREADABLE`` rather than read an unknown one.
OOXML_INVENTORY_PAYLOAD_VERSION = 1

#: Exactly the top-level keys a version-``OOXML_INVENTORY_PAYLOAD_VERSION`` record has --
#: no more, no fewer. EXACT rather than "at least these", for the same reason as
#: :data:`carmel.services.member_table_record.MEMBER_INVENTORY_PAYLOAD_KEYS`: the address
#: is over the canonical bytes and the verifier compares them against a freshly built
#: payload, so a record carrying a stray key can never reproduce.
OOXML_INVENTORY_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "cells",
        "col_count",
        "part_name",
        "payload_version",
        "row_count",
        "source_sha256",
        "table_index",
    }
)

#: The WordprocessingML namespace every element and attribute in ``word/document.xml``
#: lives under. ElementTree renders a namespaced tag as ``{uri}local``.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: The ZIP member that holds a ``.docx``'s main document body. A package without it is
#: not a WordprocessingML document this lane can read.
_DOCUMENT_PART = "word/document.xml"

#: The most cells a single ``.docx`` table's grid may hold before it is refused, unread.
#: Bounded for the same reason as
#: :data:`carmel.services.member_table_record.MAX_MEMBER_CELL_COUNT`: the object graph the
#: parse builds (one frozen :class:`OoxmlCell` per cell) is the axis the byte caps below
#: do not bound. Same ceiling as the delimited lane; no ``.docx`` table this lane reads
#: comes within orders of magnitude of it.
MAX_OOXML_TABLE_CELL_COUNT = 4_000_000

#: The most uncompressed bytes ``word/document.xml`` may expand to before it is refused
#: unread. A ``.docx`` is a ZIP, so a few kilobytes on disk can declare gigabytes of
#: document body -- a decompression bomb the file's own size cap (imposed when it was
#: staged) does not catch, because that cap is on the COMPRESSED archive. Read with an
#: explicit ceiling so a hostile package is refused rather than exhausting memory. Set to
#: the delimited lane's per-member ceiling (256 MiB): far above any real supplementary
#: document, far below what would take the process down.
MAX_OOXML_DOCUMENT_XML_BYTES = 256 * 1024 * 1024


class OoxmlDocumentUnreadable(ValueError):
    """The ``.docx``'s bytes are not a readable WordprocessingML document: not a ZIP, no
    ``word/document.xml``, that part is not well-formed XML, it declares a DOCTYPE (which a
    WordprocessingML document never does and which is the vector for an entity-expansion
    attack), a cell carries a malformed ``w:gridSpan``, or the part exceeds
    :data:`MAX_OOXML_DOCUMENT_XML_BYTES`."""


class OoxmlNoTable(ValueError):
    """The document is readable but has no table at the requested ``table_index`` -- it
    holds fewer top-level tables than that, or none at all. Distinct from an unreadable
    document: the bytes parsed, they simply do not contain the addressed grid."""


class OoxmlNestedTable(ValueError):
    """The addressed table contains another table nested inside one of its cells. A nested
    table cannot be honestly flattened into its parent's grid -- doing so would misalign
    every row below it -- so it is refused rather than guessed at, the same fabricated-join
    class this project refuses elsewhere."""


class OoxmlTableTooLarge(ValueError):
    """The addressed table's grid exceeds :data:`MAX_OOXML_TABLE_CELL_COUNT` cells -- too
    large to hold in memory as one :class:`OoxmlCell` per cell. Refused while accumulating,
    before the crossing allocation, so the process is never taken down by the allocator."""


@dataclass(frozen=True)
class OoxmlCell:
    """One cell of a ``.docx`` table's grid.

    ``row`` is the 0-indexed row ordinal; ``col`` is the 0-indexed GRID column the cell
    STARTS at, which is the running sum of the ``col_span`` of the cells before it in the
    same row -- so a cell after a two-column span begins at grid column 2, not 1.

    ``col_span`` is how many grid columns the cell occupies (``w:gridSpan``, default 1).
    The columns a span covers beyond its start hold no separate cell: the record leaves
    them empty rather than duplicating the spanning cell across them, because duplicating
    would fabricate cells the document does not print.

    ``row_merge`` records a vertical merge (``w:vMerge``): ``"start"`` for the origin that
    carries the value, ``"continue"`` for a cell the document draws as a continuation of
    the origin above it, ``None`` for an ordinary cell. A ``"continue"`` cell keeps its own
    verbatim ``text`` (blank in every real case) and is NOT given the origin's value: the
    merge is represented, never fabricated downward. A consumer that ignores the flag sees
    the blank the cell physically prints; one that reads it knows the value belongs to the
    origin. Deciding whether the origin's value propagates is a downstream, semantic ruling
    this lane deliberately does not make.

    ``text`` is the verbatim concatenation of the cell's own ``w:t`` runs.
    """

    row: int
    col: int
    text: str
    col_span: int
    row_merge: str | None


@dataclass(frozen=True)
class OoxmlTableInventory:
    """One ``.docx`` table's grid, derived from its own bytes.

    ``part_name`` is the ZIP member the table was read from (always
    :data:`_DOCUMENT_PART` in this delivery); ``table_index`` is its 0-based ordinal among
    the document body's top-level tables. Together they are the address a verifier
    re-derives from: the same part, the same ordinal, the same bytes yield the same grid.
    """

    part_name: str
    table_index: int
    row_count: int
    col_count: int
    cells: tuple[OoxmlCell, ...]


def _document_xml_bytes(docx_bytes: bytes) -> bytes:
    """The ``word/document.xml`` part of ``docx_bytes``, read with an explicit
    uncompressed-size ceiling so a decompression bomb is refused rather than expanded.

    Raises:
        OoxmlDocumentUnreadable: If the bytes are not a ZIP, carry no
            ``word/document.xml``, or that part exceeds
            :data:`MAX_OOXML_DOCUMENT_XML_BYTES`.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(docx_bytes))
    except zipfile.BadZipFile as exc:
        raise OoxmlDocumentUnreadable(f"not a readable .docx (ZIP) package: {exc}") from exc
    # Hold the archive in a context manager so its handle is closed on EVERY path out --
    # the refusals below and the return alike -- rather than leaked to the collector. Same
    # pattern archive_unpack.unpack_archive already follows: construct inside the try that
    # catches a bad ZIP, then `with archive:` over the body.
    with archive:
        try:
            info = archive.getinfo(_DOCUMENT_PART)
        except KeyError as exc:
            raise OoxmlDocumentUnreadable(
                f"package has no {_DOCUMENT_PART!r} part, so it is not a WordprocessingML document"
            ) from exc
        # The declared size is a hint, not a guarantee -- a crafted entry can under-declare
        # it, so the bounded read below is what actually holds memory. Refusing an
        # over-declared entry first just avoids reading one whose header already admits it
        # is too large.
        if info.file_size > MAX_OOXML_DOCUMENT_XML_BYTES:
            raise OoxmlDocumentUnreadable(
                f"{_DOCUMENT_PART} declares {info.file_size} uncompressed bytes, over the "
                f"{MAX_OOXML_DOCUMENT_XML_BYTES} cap; refused unread"
            )
        try:
            with archive.open(info) as part:
                data = part.read(MAX_OOXML_DOCUMENT_XML_BYTES + 1)
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            raise OoxmlDocumentUnreadable(f"{_DOCUMENT_PART} could not be decompressed: {exc}") from exc
        if len(data) > MAX_OOXML_DOCUMENT_XML_BYTES:
            raise OoxmlDocumentUnreadable(
                f"{_DOCUMENT_PART} expands past the {MAX_OOXML_DOCUMENT_XML_BYTES} byte cap; refused unread"
            )
        return data


def _parse_document_body(docx_bytes: bytes) -> ElementTree.Element:
    """Parse ``docx_bytes`` and return the ``w:body`` element of its main document.

    A DOCTYPE declaration is refused before parsing: WordprocessingML never carries one,
    and it is the vector for a billion-laughs entity-expansion attack, so its presence is
    treated as a malformed document rather than parsed. External entities are not a concern
    -- :mod:`xml.etree.ElementTree`'s parser does not resolve them -- but internal entity
    expansion is, and rejecting the DOCTYPE that would declare any entity closes that off.

    Raises:
        OoxmlDocumentUnreadable: If the bytes are unreadable (see
            :func:`_document_xml_bytes`), the part is not well-formed XML, it declares a
            DOCTYPE, or it has no ``w:body``.
    """
    data = _document_xml_bytes(docx_bytes)
    # A cheap scan for a DOCTYPE before handing the bytes to the parser. It can only appear
    # in the XML prolog, before the root element; a `<` in element content is escaped, so a
    # literal "<!DOCTYPE" in the byte stream is a real declaration, not payload.
    if b"<!DOCTYPE" in data:
        raise OoxmlDocumentUnreadable(
            f"{_DOCUMENT_PART} declares a DOCTYPE, which a WordprocessingML document never does; refused"
        )
    try:
        root = ElementTree.fromstring(data)  # noqa: S314 - DOCTYPE refused above; no entity expansion is possible
    except ElementTree.ParseError as exc:
        raise OoxmlDocumentUnreadable(f"{_DOCUMENT_PART} is not well-formed XML: {exc}") from exc
    body = root.find(f"{_W}body")
    if body is None:
        raise OoxmlDocumentUnreadable(f"{_DOCUMENT_PART} has no w:body element")
    return body


def _top_level_tables(body: ElementTree.Element) -> list[ElementTree.Element]:
    """The document body's top-level tables, in document order. Direct ``w:tbl`` children
    of ``w:body`` only: a table nested inside a cell is not a top-level table and is not
    enumerated here (its enclosing table is refused when read; see
    :func:`read_ooxml_table`)."""
    return body.findall(f"{_W}tbl")


def _grid_span(tc: ElementTree.Element) -> int:
    """The number of grid columns a cell occupies (``w:gridSpan``), defaulting to 1.

    Raises:
        OoxmlDocumentUnreadable: If the span attribute is present but not a positive
            integer -- a malformed document, refused rather than coerced to a guess.
    """
    grid_span = tc.find(f"{_W}tcPr/{_W}gridSpan")
    if grid_span is None:
        return 1
    raw = grid_span.get(f"{_W}val")
    if raw is None or not raw.isdigit() or int(raw) < 1:
        raise OoxmlDocumentUnreadable(f"w:gridSpan carries a non-positive-integer w:val {raw!r}")
    return int(raw)


def _row_merge(tc: ElementTree.Element) -> str | None:
    """The vertical-merge state of a cell: ``"start"`` for a ``w:vMerge`` origin
    (``w:val="restart"``), ``"continue"`` for a continuation (any other or absent
    ``w:val`` on a present ``w:vMerge``), ``None`` when there is no ``w:vMerge`` at all."""
    v_merge = tc.find(f"{_W}tcPr/{_W}vMerge")
    if v_merge is None:
        return None
    return "start" if v_merge.get(f"{_W}val") == "restart" else "continue"


def _cell_text(tc: ElementTree.Element) -> str:
    """The verbatim concatenation of a cell's own ``w:t`` runs. Only called for cells in a
    table with no nested tables (see :func:`read_ooxml_table`), so no descendant ``w:t``
    can belong to a nested grid -- there is nothing to bleed in from one."""
    return "".join(node.text or "" for node in tc.iter(f"{_W}t"))


def count_ooxml_tables(docx_bytes: bytes) -> int:
    """How many top-level tables the document body holds, in document order.

    Raises:
        OoxmlDocumentUnreadable: If the bytes are not a readable WordprocessingML document.
    """
    return len(_top_level_tables(_parse_document_body(docx_bytes)))


def read_ooxml_table(docx_bytes: bytes, *, table_index: int) -> OoxmlTableInventory:
    """Read the ``table_index``-th top-level table of a ``.docx`` into an
    :class:`OoxmlTableInventory`.

    The grid is bounded at :data:`MAX_OOXML_TABLE_CELL_COUNT` cells, checked before the
    crossing cell is built, so a table whose cell count would exhaust memory is refused
    rather than allowed to take the process down by allocator.

    Raises:
        OoxmlDocumentUnreadable: If the bytes are not a readable WordprocessingML document,
            or a cell carries a malformed ``w:gridSpan``.
        OoxmlNoTable: If the document has no table at ``table_index``.
        OoxmlNestedTable: If the addressed table nests another table inside a cell.
        OoxmlTableTooLarge: If the grid exceeds :data:`MAX_OOXML_TABLE_CELL_COUNT` cells.
    """
    body = _parse_document_body(docx_bytes)
    tables = _top_level_tables(body)
    if table_index < 0 or table_index >= len(tables):
        raise OoxmlNoTable(f"document has {len(tables)} top-level table(s); there is none at index {table_index}")
    table = tables[table_index]
    # A nested table anywhere under a cell of this one makes its grid ambiguous -- refuse
    # the whole table rather than flatten. Checked before any cell is read so the refusal
    # is total. `.//w:tc//w:tbl` catches a nested table at any depth inside any cell.
    if table.find(f".//{_W}tc//{_W}tbl") is not None:
        raise OoxmlNestedTable(
            f"table {table_index} contains a nested table; refused rather than flattened into a misaligned grid"
        )

    cells: list[OoxmlCell] = []
    col_count = 0
    row_count = 0
    for row_index, tr in enumerate(table.findall(f"{_W}tr")):
        row_count = row_index + 1
        grid_col = 0
        for tc in tr.findall(f"{_W}tc"):
            if len(cells) >= MAX_OOXML_TABLE_CELL_COUNT:
                raise OoxmlTableTooLarge(
                    f"table {table_index} exceeds {MAX_OOXML_TABLE_CELL_COUNT} cells; refused unread"
                )
            span = _grid_span(tc)
            cells.append(
                OoxmlCell(
                    row=row_index,
                    col=grid_col,
                    text=_cell_text(tc),
                    col_span=span,
                    row_merge=_row_merge(tc),
                )
            )
            grid_col += span
        col_count = max(col_count, grid_col)
    return OoxmlTableInventory(
        part_name=_DOCUMENT_PART,
        table_index=table_index,
        row_count=row_count,
        col_count=col_count,
        cells=tuple(cells),
    )


def read_ooxml_tables(docx_bytes: bytes) -> tuple[OoxmlTableInventory, ...]:
    """Read every top-level table of a ``.docx``, in document order.

    A read error on any one table propagates: this reads a document already known to be
    well formed. The production path that wants per-table refusals recorded rather than
    raised calls :func:`read_ooxml_table` per index and catches.

    Raises:
        OoxmlDocumentUnreadable, OoxmlNestedTable, OoxmlTableTooLarge: As
            :func:`read_ooxml_table`.
    """
    count = count_ooxml_tables(docx_bytes)
    return tuple(read_ooxml_table(docx_bytes, table_index=index) for index in range(count))


def _cell_payload(cell: OoxmlCell) -> dict[str, Any]:
    """The stored form of one cell. ``row_merge`` is emitted as ``null`` for an ordinary
    cell so the record's shape is fixed rather than depending on whether a merge is
    present."""
    return {
        "col": cell.col,
        "col_span": cell.col_span,
        "row": cell.row,
        "row_merge": cell.row_merge,
        "text": cell.text,
    }


def ooxml_inventory_record_payload(inventory: OoxmlTableInventory, *, source_sha256: str) -> dict[str, Any]:
    """The full stored form of one table inventory, ready for :func:`canonical_json_bytes`.

    Cells are sorted by ``(row, col)`` so the record's bytes -- and therefore its address
    -- do not depend on iteration order.

    Raises:
        ValueError: If ``source_sha256`` is not a well-formed digest. It is the record's
            only link to the ``.docx``, and nothing downstream re-derives it, so a
            malformed one would mint a record that reports ``SOURCE_MISMATCH`` forever.
    """
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError(f"source_sha256 must be 64 lowercase hex characters, got {source_sha256!r}")
    return {
        "cells": [_cell_payload(cell) for cell in sorted(inventory.cells, key=lambda c: (c.row, c.col))],
        "col_count": inventory.col_count,
        "part_name": inventory.part_name,
        "payload_version": OOXML_INVENTORY_PAYLOAD_VERSION,
        "row_count": inventory.row_count,
        "source_sha256": source_sha256,
        "table_index": inventory.table_index,
    }


def ooxml_inventory_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of this record: what its address is over. One definition, shared
    with :func:`compute_ooxml_inventory_sha`, so the two cannot drift."""
    return canonical_json_bytes(dict(payload))


def compute_ooxml_inventory_sha(payload: Mapping[str, Any]) -> str:
    """This record's content address: the sha256 of its canonical JSON bytes."""
    return hashlib.sha256(ooxml_inventory_record_bytes(payload)).hexdigest()


class OoxmlInventoryVerificationStatus(StrEnum):
    """The outcome of checking a stored OOXML inventory record against a ``.docx``'s bytes.
    Mirrors
    :class:`~carmel.services.member_table_record.MemberInventoryVerificationStatus`."""

    REPRODUCED = "reproduced"
    """The re-derivation ran and its canonical bytes are identical to the stored ones. The
    only positive result: the stored grid follows from these bytes under this reader."""

    MISMATCHED = "mismatched"
    """The re-derivation ran and produced a different grid. The record is not evidence."""

    SOURCE_MISMATCH = "source_mismatch"
    """``data`` is not the ``.docx`` this record is about; its sha256 differs from the
    stored one. Nothing was re-derived -- reporting it as "could not verify" would let a
    caller retry forever against the wrong file."""

    PAYLOAD_UNREADABLE = "payload_unreadable"
    """The stored payload is malformed or of an unknown version. A verifier that guessed at
    an unknown shape would be inventing the fields it is meant to check."""


@dataclass(frozen=True)
class OoxmlInventoryVerification:
    """What checking a record against a ``.docx``'s bytes established."""

    status: OoxmlInventoryVerificationStatus
    detail: str = ""


def _payload_unreadable_reason(payload: Mapping[str, Any]) -> str | None:
    """Why this payload cannot be read back, or ``None`` if it can. One definition of
    "readable", relied on by both the verifier and any schema that embeds the record."""
    version = payload.get("payload_version")
    if version != OOXML_INVENTORY_PAYLOAD_VERSION:
        return f"payload_version {version!r} is not the readable version {OOXML_INVENTORY_PAYLOAD_VERSION!r}"
    keys = set(payload)
    if keys != set(OOXML_INVENTORY_PAYLOAD_KEYS):
        unexpected = sorted(keys - OOXML_INVENTORY_PAYLOAD_KEYS)
        missing = sorted(OOXML_INVENTORY_PAYLOAD_KEYS - keys)
        return (
            f"record is not the shape of a version-{OOXML_INVENTORY_PAYLOAD_VERSION} inventory "
            f"(unexpected keys {unexpected!r}, missing keys {missing!r})"
        )
    part_name = payload.get("part_name")
    if not isinstance(part_name, str) or not part_name:
        return f"part_name {part_name!r} is not a non-empty string"
    table_index = payload.get("table_index")
    if not isinstance(table_index, int) or isinstance(table_index, bool) or table_index < 0:
        return f"table_index {table_index!r} is not a non-negative integer"
    source_sha256 = payload.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_sha256)
    ):
        return f"source_sha256 {source_sha256!r} is not 64 lowercase hex characters"
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return f"cells is {type(cells).__name__}, not a list"
    return None


def verify_ooxml_inventory_record(payload: Mapping[str, Any], data: bytes) -> OoxmlInventoryVerification:
    """Check a stored OOXML inventory record against a ``.docx``'s raw bytes.

    Re-derives the addressed table from ``data`` under the same reader that built the
    record and compares canonical bytes. A difference is reported ``MISMATCHED`` and its
    detail NAMES the first cell that disagrees, so a falsification test can point at what
    changed. The store is never consulted: ``data`` is the source of truth.
    """
    reason = _payload_unreadable_reason(payload)
    if reason is not None:
        return OoxmlInventoryVerification(status=OoxmlInventoryVerificationStatus.PAYLOAD_UNREADABLE, detail=reason)

    source_sha256 = str(payload["source_sha256"])
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != source_sha256:
        return OoxmlInventoryVerification(
            status=OoxmlInventoryVerificationStatus.SOURCE_MISMATCH,
            detail=f"these bytes hash to {actual_sha!r}, but the record is about document {source_sha256!r}",
        )

    table_index = int(payload["table_index"])
    try:
        inventory = read_ooxml_table(data, table_index=table_index)
    except (OoxmlDocumentUnreadable, OoxmlNoTable, OoxmlNestedTable, OoxmlTableTooLarge) as exc:
        return OoxmlInventoryVerification(
            status=OoxmlInventoryVerificationStatus.MISMATCHED,
            detail=f"the document bytes cannot be re-read as the claimed grid: {exc}",
        )
    rebuilt = ooxml_inventory_record_payload(inventory, source_sha256=actual_sha)
    if canonical_json_bytes(rebuilt) == ooxml_inventory_record_bytes(payload):
        return OoxmlInventoryVerification(status=OoxmlInventoryVerificationStatus.REPRODUCED)

    difference = _first_cell_difference(payload, rebuilt)
    return OoxmlInventoryVerification(status=OoxmlInventoryVerificationStatus.MISMATCHED, detail=difference)


def _cell_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Map ``(row, col) -> cell dict`` for a payload whose ``cells`` has passed
    :func:`_payload_unreadable_reason`. Non-conforming cell entries are skipped, so a
    hand-mangled cell surfaces as a difference rather than a crash."""
    mapping: dict[tuple[int, int], dict[str, Any]] = {}
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return mapping
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = cell.get("row")
        col = cell.get("col")
        if isinstance(row, bool) or isinstance(col, bool):
            continue
        if isinstance(row, int) and isinstance(col, int):
            mapping[(row, col)] = cell
    return mapping


def _first_cell_difference(stored: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> str:
    """A message naming the first ``(row, col)`` where ``stored`` and ``rebuilt`` disagree
    on any cell field. Falls back to a whole-record message if the difference is not at
    cell granularity (e.g. a differing ``row_count`` or ``col_count``)."""
    stored_cells = _cell_map(stored)
    rebuilt_cells = _cell_map(rebuilt)
    for key in sorted(set(stored_cells) | set(rebuilt_cells)):
        stored_cell = stored_cells.get(key)
        rebuilt_cell = rebuilt_cells.get(key)
        if stored_cell != rebuilt_cell:
            return (
                f"cell (row={key[0]}, col={key[1]}) is stored as {stored_cell!r} but the document's bytes "
                f"yield {rebuilt_cell!r}"
            )
    return "the re-derived grid differs from the stored record (shape or counts), though every named cell agrees"


def cell_text_from_payload(payload: Mapping[str, Any], *, row: int, col: int) -> str | None:
    """This record's own text for ``(row, col)``, or ``None`` if the grid has no such cell,
    or the cell carries no string ``text``. Answers from the payload's cells; it proves
    nothing about whether that text is what the document printed -- only
    :func:`verify_ooxml_inventory_record` does."""
    cell = _cell_map(payload).get((row, col))
    if cell is None:
        return None
    text = cell.get("text")
    return text if isinstance(text, str) else None


class OoxmlCellReplayOutcome(StrEnum):
    """The outcome of replaying one addressed OOXML cell against ``.docx`` bytes."""

    MATCH = "match"
    """The record reproduced from the bytes AND the addressed cell's stored text equals the
    expected value."""

    FAILED = "failed"
    """A comparison ran and disagreed: the grid did not reproduce, the bytes are the wrong
    document, or the addressed cell's text is not the expected value."""

    UNVERIFIABLE = "unverifiable"
    """No comparison could run -- the stored payload is unreadable -- so the citation is
    neither confirmed nor refuted."""


@dataclass(frozen=True)
class OoxmlCellReplay:
    """What replaying one addressed OOXML cell established."""

    outcome: OoxmlCellReplayOutcome
    detail: str = ""


def replay_ooxml_cell(
    payload: Mapping[str, Any],
    data: bytes,
    *,
    row: int,
    col: int,
    expected_text: str,
) -> OoxmlCellReplay:
    """Replay the value ``expected_text``, claimed to sit at ``(row, col)`` of the table
    the record addresses in the ``.docx`` whose bytes are ``data``, against those bytes.

    Two things must hold for a ``MATCH``: the whole record must REPRODUCE from the
    document's bytes (so the grid is real), and the addressed cell's stored text must equal
    ``expected_text`` (the exact-equality contract the other lanes also enforce). Any
    disagreement is ``FAILED`` with a detail naming the cause; an unreadable payload is
    ``UNVERIFIABLE``.
    """
    verification = verify_ooxml_inventory_record(payload, data)
    if verification.status is OoxmlInventoryVerificationStatus.PAYLOAD_UNREADABLE:
        return OoxmlCellReplay(outcome=OoxmlCellReplayOutcome.UNVERIFIABLE, detail=verification.detail)
    if verification.status is not OoxmlInventoryVerificationStatus.REPRODUCED:
        return OoxmlCellReplay(
            outcome=OoxmlCellReplayOutcome.FAILED,
            detail=f"OOXML inventory did not reproduce ({verification.status.value}): {verification.detail}",
        )
    actual = cell_text_from_payload(payload, row=row, col=col)
    if actual is None:
        return OoxmlCellReplay(
            outcome=OoxmlCellReplayOutcome.FAILED,
            detail=f"the grid has no cell at (row={row}, col={col})",
        )
    if actual != expected_text:
        return OoxmlCellReplay(
            outcome=OoxmlCellReplayOutcome.FAILED,
            detail=f"cell (row={row}, col={col}) holds {actual!r}, not the expected {expected_text!r}",
        )
    return OoxmlCellReplay(outcome=OoxmlCellReplayOutcome.MATCH)
