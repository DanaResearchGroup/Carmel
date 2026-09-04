"""A byte-replayable table inventory for one worksheet of a SpreadsheetML (``.xlsx``) workbook.

This is the spreadsheet counterpart to :mod:`carmel.services.member_table_record` (delimited
CSV/TSV members) and to the OOXML ``.docx`` lane built alongside it. Each derives a grid from a
file's own bytes and re-derives it to verify; each produces a canonical-JSON record that hashes
to its own address, names the document it came from, and that a verifier can REPRODUCE from that
document's raw bytes without any store. A ``.xlsx`` cell is thereby made as byte-replayable as a
CSV member cell, a ``.docx`` table cell, or a PDF table cell.

An ``.xlsx`` IS an OOXML ZIP, exactly like a ``.docx``, so this lane reads it the same way the
``.docx`` lane reads its container: :mod:`zipfile` plus :mod:`xml.etree.ElementTree`, with no
third-party dependency. The premise that reading a workbook "needs a runtime library the project
does not depend on" does not hold -- the raw stored value of every cell sits in a ``<v>`` element
of a sheet part, next to its type marker, and is read directly. A library such as ``openpyxl``
would instead hand back TYPED Python values -- a date serial as a ``datetime``, a number as a
``float``, a formula's text OR its cached value but never both -- which is the same type inference
that makes ``pandas`` the wrong tool for a provenance lane. Reading the bytes directly is what
keeps the three spreadsheet-specific facts below faithful.

It is a SEPARATE record type, not a reuse of the delimited or ``.docx`` one, for reasons a
spreadsheet states that neither of those can:

* ONE workbook holds MANY worksheets, in an order set by ``xl/workbook.xml`` (NOT by sheet-part
  filename), so the address needs a ``sheet_index`` and carries the sheet's ``sheet_name``.
* A cell can span a merged range (``<mergeCell>``), stored as a sheet-level range whose top-left
  cell holds the value and whose other cells are blank. This record carries, per cell, its
  ``row_span`` and ``col_span`` -- so a merged header is REPRESENTED rather than fabricated
  downward into cells the workbook leaves blank. See :class:`XlsxCell`.
* Three facts a ``.docx`` table never had to state, each decided here as a position (see the
  module tests, which pin each):

  - **Formula vs cached value.** A formula cell holds both a formula (``<f>``) and a cached
    result (``<v>``). This record stores BOTH and marks the cell a formula cell (``formula`` is
    non-``None``): a computed number is not a measured one, so a consumer must be able to see it
    was computed. When the cache is absent -- a real case -- ``value`` is empty and ``formula``
    still carries the source, so the cell is neither dropped nor given a fabricated result.
  - **Displayed vs stored.** ``styles.xml`` is never read: the value stored is the raw ``<v>``
    string the file holds (``0.123456``), never the number-format rendering a screen would show
    (``0.12``). Rendering is a display concern this lane does not perform.
  - **Indirection.** A shared string (``t="s"``) stores an index into ``xl/sharedStrings.xml``;
    the datum is the string it points at, so the resolved text is stored (rich-text runs
    concatenated), not the meaningless index. A date stores a serial number under a date
    number-format; the serial is the stored datum, so the serial is stored -- interpreting it as
    a calendar date is a downstream semantic ruling (it needs the workbook's 1900/1904 epoch)
    this lane declines, exactly as it declines every other downstream ruling.

Scope of this first delivery: ONE ``.xlsx`` package, its worksheets' cell grids. ``.xls`` (legacy),
a ``.csv`` inside an ``.xlsx``, and archives of workbooks are out of scope entirely and are not
read here. Charts, pivot tables, styles and drawings are not read: only ``sheetData``, the shared
string table, and merged ranges.
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
    "MAX_XLSX_PART_BYTES",
    "MAX_XLSX_SHEET_CELL_COUNT",
    "XLSX_INVENTORY_PAYLOAD_KEYS",
    "XLSX_INVENTORY_PAYLOAD_VERSION",
    "XlsxCell",
    "XlsxCellReplay",
    "XlsxCellReplayOutcome",
    "XlsxEmptySheet",
    "XlsxInventoryVerification",
    "XlsxInventoryVerificationStatus",
    "XlsxNoTable",
    "XlsxSheetInventory",
    "XlsxSheetTooLarge",
    "XlsxWorkbookUnreadable",
    "cell_text_from_payload",
    "compute_xlsx_inventory_sha",
    "count_xlsx_sheets",
    "read_xlsx_sheet",
    "read_xlsx_sheets",
    "replay_xlsx_cell",
    "verify_xlsx_inventory_record",
    "xlsx_inventory_record_bytes",
    "xlsx_inventory_record_payload",
]

#: The on-disk shape of an ``.xlsx`` sheet inventory record. Bumped whenever a field is added or
#: changed in a way that is not simply optional-with-a-default. A reader that does not know a
#: version must not guess at its shape -- see :func:`verify_xlsx_inventory_record`, which returns
#: ``PAYLOAD_UNREADABLE`` rather than read an unknown one.
XLSX_INVENTORY_PAYLOAD_VERSION = 1

#: Exactly the top-level keys a version-``XLSX_INVENTORY_PAYLOAD_VERSION`` record has -- no more,
#: no fewer. EXACT rather than "at least these": the address is over the canonical bytes and the
#: verifier compares them against a freshly built payload, so a record carrying a stray key can
#: never reproduce. Mirrors :data:`carmel.services.member_table_record.MEMBER_INVENTORY_PAYLOAD_KEYS`.
XLSX_INVENTORY_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "cells",
        "col_count",
        "part_name",
        "payload_version",
        "row_count",
        "sheet_index",
        "sheet_name",
        "source_sha256",
    }
)

#: The SpreadsheetML namespace every element in a sheet part, the workbook part, and the shared
#: string table lives under. ElementTree renders a namespaced tag as ``{uri}local``.
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: The relationship-id namespace ``xl/workbook.xml`` uses to point a ``<sheet>`` at its part.
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

#: The namespace of the package relationships file that resolves those ids to part names.
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

#: The relationship ``Type`` a ``<sheet>``'s ``r:id`` MUST resolve to. ``workbook.xml.rels`` carries
#: relationships to many part kinds (styles, the shared string table, a theme ...); only one is a
#: worksheet. A ``<sheet>`` pointing at any other Type is structurally malformed, not an empty sheet,
#: so resolution is filtered to this Type -- see :func:`_worksheet_parts`.
_WORKSHEET_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"

#: The workbook part that names the worksheets and, crucially, their ORDER.
_WORKBOOK_PART = "xl/workbook.xml"

#: The relationships part that maps a ``<sheet>``'s ``r:id`` to its worksheet part.
_WORKBOOK_RELS_PART = "xl/_rels/workbook.xml.rels"

#: The shared string table. Optional: a workbook that uses only inline strings and numbers has none.
_SHARED_STRINGS_PART = "xl/sharedStrings.xml"

#: The most cells a single worksheet's grid may hold before it is refused, unread. Bounded for the
#: same reason as the delimited and ``.docx`` lanes: the object graph the parse builds (one frozen
#: :class:`XlsxCell` per cell) is the axis the byte caps below do not bound. Same ceiling as the
#: sibling lanes; no real supplement comes within orders of magnitude of it.
MAX_XLSX_SHEET_CELL_COUNT = 4_000_000

#: The most uncompressed bytes any single ZIP part this lane reads (the workbook, the shared
#: string table, or one sheet) may expand to before it is refused unread. An ``.xlsx`` is a ZIP,
#: so a few kilobytes on disk can declare gigabytes of part -- a decompression bomb the file's own
#: (compressed) size cap does not catch. Set to the sibling lanes' 256 MiB ceiling: far above any
#: real supplement, far below what would take the process down.
MAX_XLSX_PART_BYTES = 256 * 1024 * 1024


class XlsxWorkbookUnreadable(ValueError):
    """The bytes are not a readable SpreadsheetML workbook: not a ZIP; no ``xl/workbook.xml``; a
    part that is not well-formed XML; a part that declares a DOCTYPE (which SpreadsheetML never
    does and which is the vector for an entity-expansion attack); a part that exceeds
    :data:`MAX_XLSX_PART_BYTES`; a ``<sheet>`` whose relationship id resolves to no part; or a
    cell whose reference or shared-string index is malformed."""


class XlsxNoTable(ValueError):
    """The workbook is readable but has no worksheet at the requested ``sheet_index`` -- it holds
    fewer worksheets than that, or none at all. Distinct from an unreadable workbook: the bytes
    parsed, they simply do not contain the addressed sheet."""


class XlsxEmptySheet(ValueError):
    """The addressed worksheet holds no cell with a value or a formula -- no table-shaped content.
    Refused rather than returned as an empty inventory presented as success, so a blank sheet is a
    recorded outcome, not a grid of nothing."""


class XlsxSheetTooLarge(ValueError):
    """The addressed worksheet's grid exceeds :data:`MAX_XLSX_SHEET_CELL_COUNT` cells -- too large
    to hold in memory as one :class:`XlsxCell` per cell. Refused while accumulating, before the
    crossing allocation, so the process is never taken down by the allocator."""


@dataclass(frozen=True)
class XlsxCell:
    """One cell of a worksheet's grid, carrying its RAW stored value.

    ``row`` and ``col`` are 0-indexed, parsed from the cell's ``r`` reference (``"AD25"``). A cell
    without a parseable reference is refused, not positionally guessed at.

    ``value`` is the raw stored value's text: the literal ``<v>`` string for a number, boolean or
    error; the resolved text of a shared or inline string; the cached RESULT text for a formula
    cell (empty when the file cached none). It is never a number-format rendering -- the file's
    stored bytes, not what a screen would show.

    ``cell_type`` is the cell's raw ``t`` marker (``"n"`` number [the default], ``"s"`` shared
    string, ``"str"`` formula string result, ``"b"`` boolean, ``"e"`` error, ``"inlineStr"``
    inline string). It is a stored fact, not an inference, and it is what tells a number ``42``
    apart from the string ``"42"``.

    ``formula`` is the cell's formula source text when it is a formula cell, else ``None``. A
    formula cell therefore always has ``formula is not None`` (``""`` for a shared-formula
    continuation, whose source lives on the master and is NOT reconstructed here -- reconstructing
    it would be formula evaluation, which this lane declines). This is the flag that keeps a
    computed number from being mistaken for a measured one.

    ``row_span`` and ``col_span`` record a merged range whose top-left this cell is (``1``/``1``
    for an ordinary cell). The cells a merge covers beyond its top-left are blank in the file and
    are not emitted: the merge is REPRESENTED by the span on its origin, never fabricated by
    copying the value into the covered cells.
    """

    row: int
    col: int
    value: str
    cell_type: str
    formula: str | None
    row_span: int
    col_span: int


@dataclass(frozen=True)
class XlsxSheetInventory:
    """One worksheet's grid, derived from its own bytes.

    ``part_name`` is the ZIP member the sheet was read from (e.g. ``xl/worksheets/sheet1.xml``);
    ``sheet_index`` is its 0-based ordinal in WORKBOOK order (``xl/workbook.xml``'s ``<sheet>``
    sequence, which need not match the part filename); ``sheet_name`` is its title. Together they
    are the address a verifier re-derives from: the same workbook order, the same ordinal, the
    same bytes yield the same grid.
    """

    part_name: str
    sheet_index: int
    sheet_name: str
    row_count: int
    col_count: int
    cells: tuple[XlsxCell, ...]


def _bounded_part_bytes(archive: zipfile.ZipFile, name: str) -> bytes:
    """The bytes of ZIP member ``name``, read with an explicit uncompressed-size ceiling so a
    decompression bomb is refused rather than expanded.

    Raises:
        XlsxWorkbookUnreadable: If the member is absent or exceeds :data:`MAX_XLSX_PART_BYTES`.
    """
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise XlsxWorkbookUnreadable(f"package has no {name!r} part") from exc
    # The declared size is a hint, not a guarantee -- a crafted entry can under-declare it, so the
    # bounded read below is what actually holds memory. Refusing an over-declared entry first just
    # avoids reading one whose header already admits it is too large.
    if info.file_size > MAX_XLSX_PART_BYTES:
        raise XlsxWorkbookUnreadable(
            f"{name} declares {info.file_size} uncompressed bytes, over the {MAX_XLSX_PART_BYTES} cap; refused unread"
        )
    try:
        with archive.open(info) as part:
            data = part.read(MAX_XLSX_PART_BYTES + 1)
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise XlsxWorkbookUnreadable(f"{name} could not be decompressed: {exc}") from exc
    if len(data) > MAX_XLSX_PART_BYTES:
        raise XlsxWorkbookUnreadable(f"{name} expands past the {MAX_XLSX_PART_BYTES} byte cap; refused unread")
    return data


def _parse_part(archive: zipfile.ZipFile, name: str) -> ElementTree.Element:
    """Parse ZIP member ``name`` and return its root element.

    A DOCTYPE declaration is refused before parsing: SpreadsheetML never carries one, and it is
    the vector for a billion-laughs entity-expansion attack, so its presence is treated as a
    malformed part rather than parsed. External entities are not a concern --
    :mod:`xml.etree.ElementTree`'s parser does not resolve them -- but internal entity expansion
    is, and rejecting the DOCTYPE that would declare any entity closes that off.

    Raises:
        XlsxWorkbookUnreadable: If the member is absent, over the byte cap, declares a DOCTYPE, or
            is not well-formed XML.
    """
    data = _bounded_part_bytes(archive, name)
    # A cheap scan for a DOCTYPE before handing the bytes to the parser. It can only appear in the
    # XML prolog, before the root element; a `<` in element content is escaped, so a literal
    # "<!DOCTYPE" in the byte stream is a real declaration, not payload.
    if b"<!DOCTYPE" in data:
        raise XlsxWorkbookUnreadable(f"{name} declares a DOCTYPE, which a SpreadsheetML part never does; refused")
    try:
        return ElementTree.fromstring(data)  # noqa: S314 - DOCTYPE refused above; no entity expansion is possible
    except ElementTree.ParseError as exc:
        raise XlsxWorkbookUnreadable(f"{name} is not well-formed XML: {exc}") from exc


def _open_archive(xlsx_bytes: bytes) -> zipfile.ZipFile:
    """The ``.xlsx`` bytes as a ZIP archive.

    Raises:
        XlsxWorkbookUnreadable: If the bytes are not a ZIP.
    """
    try:
        return zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    except zipfile.BadZipFile as exc:
        raise XlsxWorkbookUnreadable(f"not a readable .xlsx (ZIP) package: {exc}") from exc


def _worksheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """The workbook's worksheets as ``(part_name, sheet_name)`` in WORKBOOK order.

    The order is ``xl/workbook.xml``'s ``<sheet>`` sequence; each ``<sheet>``'s ``r:id`` is
    resolved to a part through ``xl/_rels/workbook.xml.rels``. The part filename (``sheet1.xml``,
    ``sheet3.xml`` ...) need NOT match visual order, so resolving through the relationships is the
    only correct way to address the n-th sheet.

    Raises:
        XlsxWorkbookUnreadable: If the workbook or its relationships are unreadable, or a
            ``<sheet>``'s relationship id resolves to no part.
    """
    workbook = _parse_part(archive, _WORKBOOK_PART)
    sheets_el = workbook.find(f"{_S}sheets")
    if sheets_el is None:
        raise XlsxWorkbookUnreadable(f"{_WORKBOOK_PART} has no <sheets> element")

    relationships = _parse_part(archive, _WORKBOOK_RELS_PART)
    targets: dict[str, str] = {}
    for rel in relationships.findall(f"{_PKG_REL}Relationship"):
        rid = rel.get("Id")
        target = rel.get("Target")
        # Only WORKSHEET relationships may back a <sheet>. A rel of any other Type (styles, the
        # shared string table, a theme) is not a worksheet, so it is not admitted as a resolution
        # target -- a <sheet> pointing at one then resolves to no worksheet part and is refused,
        # rather than being parsed as a worksheet and misread as an empty sheet.
        if rid is not None and target is not None and rel.get("Type") == _WORKSHEET_REL_TYPE:
            # Targets are relative to the workbook part's directory (``xl/``); an absolute target
            # ("/xl/worksheets/sheet1.xml") is anchored at the package root instead.
            targets[rid] = target[1:] if target.startswith("/") else f"xl/{target}"

    parts: list[tuple[str, str]] = []
    for sheet in sheets_el.findall(f"{_S}sheet"):
        rid = sheet.get(f"{_R}id")
        name = sheet.get("name", "")
        if rid is None or rid not in targets:
            raise XlsxWorkbookUnreadable(
                f"<sheet name={name!r}> has relationship id {rid!r} that resolves to no worksheet part"
            )
        parts.append((targets[rid], name))
    return parts


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """The shared string table as a list indexed by position, or ``[]`` when the workbook has
    none. Each ``<si>`` is the concatenation of its ``<t>`` runs (a rich-text string spreads its
    text across ``<r><t>`` runs, exactly as a ``.docx`` cell spreads text across ``w:t`` runs)."""
    if _SHARED_STRINGS_PART not in archive.namelist():
        return []
    sst = _parse_part(archive, _SHARED_STRINGS_PART)
    return ["".join(node.text or "" for node in si.iter(f"{_S}t")) for si in sst.findall(f"{_S}si")]


def _ref_to_row_col(ref: str) -> tuple[int, int]:
    """The 0-indexed ``(row, col)`` of an A1-style cell reference (``"AD25"`` -> ``(24, 29)``).

    Raises:
        XlsxWorkbookUnreadable: If the reference is not letters-then-digits -- a malformed cell,
            refused rather than positionally guessed at.
    """
    letters = ref.rstrip("0123456789")
    digits = ref[len(letters) :]
    if not letters or not digits or not letters.isalpha() or not digits.isdigit():
        raise XlsxWorkbookUnreadable(f"cell reference {ref!r} is not a valid A1 reference")
    col = 0
    for char in letters.upper():
        col = col * 26 + (ord(char) - ord("A") + 1)
    return int(digits) - 1, col - 1


def _cell_value(cell: ElementTree.Element, cell_type: str, shared: list[str]) -> str:
    """The raw stored value text of a cell of type ``cell_type``.

    Raises:
        XlsxWorkbookUnreadable: If a shared-string cell carries an out-of-range or malformed index.
    """
    if cell_type == "inlineStr":
        is_el = cell.find(f"{_S}is")
        return "".join(node.text or "" for node in is_el.iter(f"{_S}t")) if is_el is not None else ""
    v_el = cell.find(f"{_S}v")
    if cell_type == "s":
        raw = v_el.text if v_el is not None else None
        if raw is None or not raw.strip().isdigit():
            raise XlsxWorkbookUnreadable(f"shared-string cell has a non-integer index {raw!r}")
        index = int(raw)
        if index < 0 or index >= len(shared):
            raise XlsxWorkbookUnreadable(f"shared-string index {index} is out of range (table has {len(shared)})")
        return shared[index]
    return v_el.text if v_el is not None and v_el.text is not None else ""


def _merged_anchors(sheet: ElementTree.Element) -> dict[tuple[int, int], tuple[int, int]]:
    """Map ``(row, col) -> (row_span, col_span)`` for the top-left cell of every merged range.

    Raises:
        XlsxWorkbookUnreadable: If a ``<mergeCell>`` reference is malformed.
    """
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    merge_container = sheet.find(f"{_S}mergeCells")
    if merge_container is None:
        return anchors
    for merge in merge_container.findall(f"{_S}mergeCell"):
        ref = merge.get("ref", "")
        if ":" not in ref:
            raise XlsxWorkbookUnreadable(f"mergeCell ref {ref!r} is not a range")
        start, end = ref.split(":", 1)
        (r0, c0), (r1, c1) = _ref_to_row_col(start), _ref_to_row_col(end)
        # A well-formed range names its top-left first and bottom-right second. A reversed range
        # (end above or left of start) would yield a zero or negative span -- a fabricated,
        # meaningless merge -- so it is refused rather than recorded as a degenerate anchor.
        if r1 < r0 or c1 < c0:
            raise XlsxWorkbookUnreadable(f"mergeCell ref {ref!r} has its end before its start")
        anchors[(r0, c0)] = (r1 - r0 + 1, c1 - c0 + 1)
    return anchors


def count_xlsx_sheets(xlsx_bytes: bytes) -> int:
    """How many worksheets the workbook holds, in workbook order.

    Raises:
        XlsxWorkbookUnreadable: If the bytes are not a readable SpreadsheetML workbook.
    """
    return len(_worksheet_parts(_open_archive(xlsx_bytes)))


def read_xlsx_sheet(xlsx_bytes: bytes, *, sheet_index: int) -> XlsxSheetInventory:
    """Read the ``sheet_index``-th worksheet (in workbook order) into an :class:`XlsxSheetInventory`.

    A cell is emitted for every ``<c>`` that carries a value or a formula, and for every merged
    range's top-left even if it is blank (so the merge is represented). A purely styled-but-empty
    cell carries no stored value and is not emitted: the grid is inherently sparse, and inventing a
    cell for an empty position would fabricate data the sheet does not hold. ``row_count`` and
    ``col_count`` are derived from the emitted cells, not from the sheet's ``<dimension>`` hint.

    The grid is bounded at :data:`MAX_XLSX_SHEET_CELL_COUNT` cells, checked before the crossing
    cell is built, so a sheet whose cell count would exhaust memory is refused rather than allowed
    to take the process down.

    Raises:
        XlsxWorkbookUnreadable: If the bytes are not a readable workbook, or a cell reference,
            shared-string index or merge range is malformed.
        XlsxNoTable: If the workbook has no worksheet at ``sheet_index``.
        XlsxEmptySheet: If the addressed worksheet holds no cell with a value or a formula.
        XlsxSheetTooLarge: If the grid exceeds :data:`MAX_XLSX_SHEET_CELL_COUNT` cells.
    """
    archive = _open_archive(xlsx_bytes)
    parts = _worksheet_parts(archive)
    if sheet_index < 0 or sheet_index >= len(parts):
        raise XlsxNoTable(f"workbook has {len(parts)} worksheet(s); there is none at index {sheet_index}")
    part_name, sheet_name = parts[sheet_index]
    shared = _shared_strings(archive)
    sheet = _parse_part(archive, part_name)
    anchors = _merged_anchors(sheet)

    sheet_data = sheet.find(f"{_S}sheetData")
    rows = [] if sheet_data is None else sheet_data.findall(f"{_S}row")
    cells: list[XlsxCell] = []
    row_count = 0
    col_count = 0
    for tr in rows:
        for tc in tr.findall(f"{_S}c"):
            ref = tc.get("r")
            if ref is None:
                raise XlsxWorkbookUnreadable(
                    "cell has no r reference; a positional grid is refused rather than guessed"
                )
            row, col = _ref_to_row_col(ref)
            f_el = tc.find(f"{_S}f")
            has_value = tc.find(f"{_S}v") is not None or tc.find(f"{_S}is") is not None
            is_anchor = (row, col) in anchors
            if f_el is None and not has_value and not is_anchor:
                continue  # styled-but-empty cell: no stored value, nothing to represent
            if len(cells) >= MAX_XLSX_SHEET_CELL_COUNT:
                raise XlsxSheetTooLarge(
                    f"sheet {sheet_index} exceeds {MAX_XLSX_SHEET_CELL_COUNT} cells; refused unread"
                )
            cell_type = tc.get("t") or "n"
            row_span, col_span = anchors.get((row, col), (1, 1))
            cells.append(
                XlsxCell(
                    row=row,
                    col=col,
                    value=_cell_value(tc, cell_type, shared),
                    cell_type=cell_type,
                    formula=(f_el.text or "") if f_el is not None else None,
                    row_span=row_span,
                    col_span=col_span,
                )
            )
            row_count = max(row_count, row + row_span)
            col_count = max(col_count, col + col_span)
    if not cells:
        raise XlsxEmptySheet(f"worksheet {sheet_index} ({sheet_name!r}) holds no cell with a value or formula")
    return XlsxSheetInventory(
        part_name=part_name,
        sheet_index=sheet_index,
        sheet_name=sheet_name,
        row_count=row_count,
        col_count=col_count,
        cells=tuple(cells),
    )


def read_xlsx_sheets(xlsx_bytes: bytes) -> tuple[XlsxSheetInventory, ...]:
    """Read every worksheet of an ``.xlsx``, in workbook order.

    A read error on any one sheet propagates: this reads a workbook already known to be well
    formed. The production path that wants per-sheet refusals recorded rather than raised calls
    :func:`read_xlsx_sheet` per index and catches (see :mod:`carmel.services.xlsx_tables`).

    Raises:
        XlsxWorkbookUnreadable, XlsxEmptySheet, XlsxSheetTooLarge: As :func:`read_xlsx_sheet`.
    """
    count = count_xlsx_sheets(xlsx_bytes)
    return tuple(read_xlsx_sheet(xlsx_bytes, sheet_index=index) for index in range(count))


def _cell_payload(cell: XlsxCell) -> dict[str, Any]:
    """The stored form of one cell. ``formula`` is emitted as ``null`` for a non-formula cell so
    the record's shape is fixed rather than depending on whether a formula is present."""
    return {
        "cell_type": cell.cell_type,
        "col": cell.col,
        "col_span": cell.col_span,
        "formula": cell.formula,
        "row": cell.row,
        "row_span": cell.row_span,
        "value": cell.value,
    }


def xlsx_inventory_record_payload(inventory: XlsxSheetInventory, *, source_sha256: str) -> dict[str, Any]:
    """The full stored form of one sheet inventory, ready for :func:`canonical_json_bytes`.

    Cells are sorted by ``(row, col)`` so the record's bytes -- and therefore its address -- do not
    depend on iteration order.

    Raises:
        ValueError: If ``source_sha256`` is not a well-formed digest. It is the record's only link
            to the ``.xlsx``, and nothing downstream re-derives it, so a malformed one would mint a
            record that reports ``SOURCE_MISMATCH`` forever.
    """
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError(f"source_sha256 must be 64 lowercase hex characters, got {source_sha256!r}")
    return {
        "cells": [_cell_payload(cell) for cell in sorted(inventory.cells, key=lambda c: (c.row, c.col))],
        "col_count": inventory.col_count,
        "part_name": inventory.part_name,
        "payload_version": XLSX_INVENTORY_PAYLOAD_VERSION,
        "row_count": inventory.row_count,
        "sheet_index": inventory.sheet_index,
        "sheet_name": inventory.sheet_name,
        "source_sha256": source_sha256,
    }


def xlsx_inventory_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of this record: what its address is over. One definition, shared with
    :func:`compute_xlsx_inventory_sha`, so the two cannot drift."""
    return canonical_json_bytes(dict(payload))


def compute_xlsx_inventory_sha(payload: Mapping[str, Any]) -> str:
    """This record's content address: the sha256 of its canonical JSON bytes."""
    return hashlib.sha256(xlsx_inventory_record_bytes(payload)).hexdigest()


class XlsxInventoryVerificationStatus(StrEnum):
    """The outcome of checking a stored ``.xlsx`` inventory record against a workbook's bytes.
    Mirrors :class:`carmel.services.member_table_record.MemberInventoryVerificationStatus`."""

    REPRODUCED = "reproduced"
    """The re-derivation ran and its canonical bytes are identical to the stored ones. The only
    positive result: the stored grid follows from these bytes under this reader."""

    MISMATCHED = "mismatched"
    """The re-derivation ran and produced a different grid. The record is not evidence."""

    SOURCE_MISMATCH = "source_mismatch"
    """``data`` is not the ``.xlsx`` this record is about; its sha256 differs from the stored one.
    Nothing was re-derived -- reporting it as "could not verify" would let a caller retry forever
    against the wrong file."""

    PAYLOAD_UNREADABLE = "payload_unreadable"
    """The stored payload is malformed or of an unknown version. A verifier that guessed at an
    unknown shape would be inventing the fields it is meant to check."""


@dataclass(frozen=True)
class XlsxInventoryVerification:
    """What checking a record against a workbook's bytes established."""

    status: XlsxInventoryVerificationStatus
    detail: str = ""


def _payload_unreadable_reason(payload: Mapping[str, Any]) -> str | None:
    """Why this payload cannot be read back, or ``None`` if it can. One definition of "readable",
    relied on by both the verifier and any schema that embeds the record."""
    version = payload.get("payload_version")
    if version != XLSX_INVENTORY_PAYLOAD_VERSION:
        return f"payload_version {version!r} is not the readable version {XLSX_INVENTORY_PAYLOAD_VERSION!r}"
    keys = set(payload)
    if keys != set(XLSX_INVENTORY_PAYLOAD_KEYS):
        unexpected = sorted(keys - XLSX_INVENTORY_PAYLOAD_KEYS)
        missing = sorted(XLSX_INVENTORY_PAYLOAD_KEYS - keys)
        return (
            f"record is not the shape of a version-{XLSX_INVENTORY_PAYLOAD_VERSION} inventory "
            f"(unexpected keys {unexpected!r}, missing keys {missing!r})"
        )
    part_name = payload.get("part_name")
    if not isinstance(part_name, str) or not part_name:
        return f"part_name {part_name!r} is not a non-empty string"
    sheet_name = payload.get("sheet_name")
    if not isinstance(sheet_name, str):
        return f"sheet_name {sheet_name!r} is not a string"
    sheet_index = payload.get("sheet_index")
    if not isinstance(sheet_index, int) or isinstance(sheet_index, bool) or sheet_index < 0:
        return f"sheet_index {sheet_index!r} is not a non-negative integer"
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


def verify_xlsx_inventory_record(payload: Mapping[str, Any], data: bytes) -> XlsxInventoryVerification:
    """Check a stored ``.xlsx`` inventory record against a workbook's raw bytes.

    Re-derives the addressed worksheet from ``data`` under the same reader that built the record
    and compares canonical bytes. A difference is reported ``MISMATCHED`` and its detail NAMES the
    first cell that disagrees, so a falsification test can point at what changed. The store is
    never consulted: ``data`` is the source of truth.
    """
    reason = _payload_unreadable_reason(payload)
    if reason is not None:
        return XlsxInventoryVerification(status=XlsxInventoryVerificationStatus.PAYLOAD_UNREADABLE, detail=reason)

    source_sha256 = str(payload["source_sha256"])
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != source_sha256:
        return XlsxInventoryVerification(
            status=XlsxInventoryVerificationStatus.SOURCE_MISMATCH,
            detail=f"these bytes hash to {actual_sha!r}, but the record is about workbook {source_sha256!r}",
        )

    sheet_index = int(payload["sheet_index"])
    try:
        inventory = read_xlsx_sheet(data, sheet_index=sheet_index)
    except (XlsxWorkbookUnreadable, XlsxNoTable, XlsxEmptySheet, XlsxSheetTooLarge) as exc:
        return XlsxInventoryVerification(
            status=XlsxInventoryVerificationStatus.MISMATCHED,
            detail=f"the workbook bytes cannot be re-read as the claimed grid: {exc}",
        )
    rebuilt = xlsx_inventory_record_payload(inventory, source_sha256=actual_sha)
    if canonical_json_bytes(rebuilt) == xlsx_inventory_record_bytes(payload):
        return XlsxInventoryVerification(status=XlsxInventoryVerificationStatus.REPRODUCED)

    difference = _first_cell_difference(payload, rebuilt)
    return XlsxInventoryVerification(status=XlsxInventoryVerificationStatus.MISMATCHED, detail=difference)


def _cell_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Map ``(row, col) -> cell dict`` for a payload whose ``cells`` has passed
    :func:`_payload_unreadable_reason`. Non-conforming cell entries are skipped, so a hand-mangled
    cell surfaces as a difference rather than a crash."""
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
    """A message naming the first ``(row, col)`` where ``stored`` and ``rebuilt`` disagree on any
    cell field. Falls back to a whole-record message if the difference is not at cell granularity
    (e.g. a differing ``row_count`` or ``sheet_name``)."""
    stored_cells = _cell_map(stored)
    rebuilt_cells = _cell_map(rebuilt)
    for key in sorted(set(stored_cells) | set(rebuilt_cells)):
        stored_cell = stored_cells.get(key)
        rebuilt_cell = rebuilt_cells.get(key)
        if stored_cell != rebuilt_cell:
            return (
                f"cell (row={key[0]}, col={key[1]}) is stored as {stored_cell!r} but the workbook's bytes "
                f"yield {rebuilt_cell!r}"
            )
    return "the re-derived grid differs from the stored record (shape or counts), though every named cell agrees"


def cell_text_from_payload(payload: Mapping[str, Any], *, row: int, col: int) -> str | None:
    """This record's own stored value for ``(row, col)``, or ``None`` if the grid has no such cell,
    or the cell carries no string ``value``. Answers from the payload's cells; it proves nothing
    about whether that value is what the workbook holds -- only :func:`verify_xlsx_inventory_record`
    does."""
    cell = _cell_map(payload).get((row, col))
    if cell is None:
        return None
    value = cell.get("value")
    return value if isinstance(value, str) else None


class XlsxCellReplayOutcome(StrEnum):
    """The outcome of replaying one addressed ``.xlsx`` cell against workbook bytes."""

    MATCH = "match"
    """The record reproduced from the bytes AND the addressed cell's stored value equals the
    expected value."""

    FAILED = "failed"
    """A comparison ran and disagreed: the grid did not reproduce, the bytes are the wrong
    workbook, or the addressed cell's value is not the expected one."""

    UNVERIFIABLE = "unverifiable"
    """No comparison could run -- the stored payload is unreadable -- so the citation is neither
    confirmed nor refuted."""


@dataclass(frozen=True)
class XlsxCellReplay:
    """What replaying one addressed ``.xlsx`` cell established."""

    outcome: XlsxCellReplayOutcome
    detail: str = ""


def replay_xlsx_cell(
    payload: Mapping[str, Any],
    data: bytes,
    *,
    row: int,
    col: int,
    expected_text: str,
) -> XlsxCellReplay:
    """Replay the value ``expected_text``, claimed to sit at ``(row, col)`` of the worksheet the
    record addresses in the ``.xlsx`` whose bytes are ``data``, against those bytes.

    Two things must hold for a ``MATCH``: the whole record must REPRODUCE from the workbook's bytes
    (so the grid is real), and the addressed cell's stored value must equal ``expected_text`` (the
    exact-equality contract the other lanes also enforce). Any disagreement is ``FAILED`` with a
    detail naming the cause; an unreadable payload is ``UNVERIFIABLE``.
    """
    verification = verify_xlsx_inventory_record(payload, data)
    if verification.status is XlsxInventoryVerificationStatus.PAYLOAD_UNREADABLE:
        return XlsxCellReplay(outcome=XlsxCellReplayOutcome.UNVERIFIABLE, detail=verification.detail)
    if verification.status is not XlsxInventoryVerificationStatus.REPRODUCED:
        return XlsxCellReplay(
            outcome=XlsxCellReplayOutcome.FAILED,
            detail=f"xlsx inventory did not reproduce ({verification.status.value}): {verification.detail}",
        )
    actual = cell_text_from_payload(payload, row=row, col=col)
    if actual is None:
        return XlsxCellReplay(
            outcome=XlsxCellReplayOutcome.FAILED,
            detail=f"the grid has no cell at (row={row}, col={col})",
        )
    if actual != expected_text:
        return XlsxCellReplay(
            outcome=XlsxCellReplayOutcome.FAILED,
            detail=f"cell (row={row}, col={col}) holds {actual!r}, not the expected {expected_text!r}",
        )
    return XlsxCellReplay(outcome=XlsxCellReplayOutcome.MATCH)
