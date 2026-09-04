"""Synthetic SpreadsheetML (`.xlsx`) builders for the xlsx table-lane tests.

Every `.xlsx` under test is built HERE from bytes, so no real supplementary workbook enters the
repository. A cell is either a plain string (stored as a shared string, the way a real workbook
stores its header text) or a :class:`CellSpec` carrying an explicit stored value, type, optional
formula and cached result, an optional style index (to prove the reader ignores number formats),
and an optional merged-range span (to prove a merge is represented, not fabricated). A test can
therefore construct exactly the number / formula / date-serial / merged-header shapes the real
corpus uses without shipping a corpus file.

:func:`xlsx_with_parts` is the low-level escape hatch: it assembles a ZIP from an explicit
``{part_name: xml}`` mapping, for the structural refusal tests (a missing workbook part, a DOCTYPE,
a scrambled sheet order, an unresolvable relationship).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

_S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

#: Public aliases the structural tests reference when they hand-build a raw part.
S_NS = _S_NS
PKG_REL_NS = _PKG_REL_NS


@dataclass(frozen=True)
class CellSpec:
    """One worksheet cell.

    ``stored`` is the raw stored value text (a number's digits, a date's serial, a string's text).
    ``kind`` selects how it is stored: ``"shared"`` (shared-string table, the default for plain
    strings), ``"inline"`` (an inline string), ``"number"`` (a bare ``<v>``), ``"bool"``,
    ``"error"``. When ``formula`` is set the cell is a formula cell: ``formula`` is its source,
    ``cache`` its cached result (``None`` -> no ``<v>`` at all), ``result_type`` the ``t`` marker of
    a non-numeric result (e.g. ``"str"``). ``style`` sets an ``s`` attribute the reader must ignore.
    ``merge`` sets ``(row_span, col_span)`` for a cell that is a merged range's top-left.
    """

    stored: str = ""
    kind: str = "shared"
    formula: str | None = None
    cache: str | None = None
    result_type: str | None = None
    style: int | None = None
    merge: tuple[int, int] | None = None


#: One cell as a test spells it: a bare string (stored as a shared string), an explicit
#: :class:`CellSpec`, or ``None`` for an absent cell in a sparse row.
type CellValue = str | CellSpec | None


def _col_letter(col: int) -> str:
    """The A1 column letters for a 0-based column index (``0 -> "A"``, ``26 -> "AA"``)."""
    letters = ""
    n = col + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _spec(cell: CellValue) -> CellSpec:
    return cell if isinstance(cell, CellSpec) else CellSpec(stored=cell, kind="shared")


def _cell_xml(ref: str, spec: CellSpec, shared_index: dict[str, int]) -> str:
    style_attr = f" s={quoteattr(str(spec.style))}" if spec.style is not None else ""
    if spec.formula is not None:
        t_attr = f" t={quoteattr(spec.result_type)}" if spec.result_type else ""
        v = f"<v>{escape(spec.cache)}</v>" if spec.cache is not None else ""
        return f"<c r={quoteattr(ref)}{style_attr}{t_attr}><f>{escape(spec.formula)}</f>{v}</c>"
    if spec.kind == "inline":
        return f'<c r={quoteattr(ref)}{style_attr} t="inlineStr"><is><t>{escape(spec.stored)}</t></is></c>'
    if spec.kind == "shared":
        index = shared_index.setdefault(spec.stored, len(shared_index))
        return f'<c r={quoteattr(ref)}{style_attr} t="s"><v>{index}</v></c>'
    if spec.kind in ("bool", "error"):
        marker = "b" if spec.kind == "bool" else "e"
        return f"<c r={quoteattr(ref)}{style_attr} t={quoteattr(marker)}><v>{escape(spec.stored)}</v></c>"
    # number (t="n" is the default, so it is left implicit exactly as a real file leaves it)
    return f"<c r={quoteattr(ref)}{style_attr}><v>{escape(spec.stored)}</v></c>"


def worksheet_xml(rows: Sequence[Sequence[CellValue]], shared_index: dict[str, int]) -> str:
    """The ``xl/worksheets/sheetN.xml`` string for ``rows``, collecting shared strings into
    ``shared_index``. Merged ranges are emitted from any cell carrying a ``merge`` span."""
    row_xml: list[str] = []
    merges: list[str] = []
    for r, row in enumerate(rows):
        cells_xml: list[str] = []
        for c, cell in enumerate(row):
            if cell is None:
                continue  # an absent cell: advances the column but emits no <c>, as a sparse grid does
            spec = _spec(cell)
            ref = f"{_col_letter(c)}{r + 1}"
            cells_xml.append(_cell_xml(ref, spec, shared_index))
            if spec.merge is not None:
                row_span, col_span = spec.merge
                end = f"{_col_letter(c + col_span - 1)}{r + row_span}"
                merges.append(f"<mergeCell ref={quoteattr(f'{ref}:{end}')}/>")
        row_xml.append(f'<row r="{r + 1}">{"".join(cells_xml)}</row>')
    merge_xml = f'<mergeCells count="{len(merges)}">{"".join(merges)}</mergeCells>' if merges else ""
    return f'<worksheet xmlns="{_S_NS}"><sheetData>{"".join(row_xml)}</sheetData>{merge_xml}</worksheet>'


def _shared_strings_xml(shared_index: Mapping[str, int]) -> str:
    ordered = sorted(shared_index, key=lambda text: shared_index[text])
    items = [f"<si><t>{escape(text)}</t></si>" for text in ordered]
    return f'<sst xmlns="{_S_NS}" count="{len(items)}" uniqueCount="{len(items)}">{"".join(items)}</sst>'


def xlsx_bytes(sheets: Sequence[Sequence[Sequence[CellValue]]], *, names: Sequence[str] | None = None) -> bytes:
    """A minimal but valid ``.xlsx`` holding ``sheets`` (each a list of rows, each row a list of
    cells), in workbook order, part filenames ``sheet1.xml`` .. matching that order. ``names`` sets
    the sheet titles (defaults to ``Sheet1`` ..). Shared strings are collected across all sheets
    into one table, as a real workbook does."""
    if names is None:
        names = [f"Sheet{i + 1}" for i in range(len(sheets))]
    shared_index: dict[str, int] = {}
    sheet_parts = {
        f"xl/worksheets/sheet{i + 1}.xml": worksheet_xml(rows, shared_index) for i, rows in enumerate(sheets)
    }

    sheet_entries = "".join(
        f'<sheet name={quoteattr(names[i])} sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i in range(len(sheets))
    )
    workbook = f'<workbook xmlns="{_S_NS}" xmlns:r="{_R_NS}"><sheets>{sheet_entries}</sheets></workbook>'

    rels = [
        f'<Relationship Id="rId{i + 1}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i + 1}.xml"/>'
        for i in range(len(sheets))
    ]
    rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        f'Target="sharedStrings.xml"/>'
    )
    workbook_rels = f'<Relationships xmlns="{_PKG_REL_NS}">{"".join(rels)}</Relationships>'

    parts: dict[str, str] = {
        "xl/workbook.xml": workbook,
        "xl/_rels/workbook.xml.rels": workbook_rels,
        "xl/sharedStrings.xml": _shared_strings_xml(shared_index),
        **sheet_parts,
    }
    return xlsx_with_parts(parts)


def xlsx_with_parts(parts: Mapping[str, str]) -> bytes:
    """A ZIP holding exactly ``parts`` ({part_name: xml}) plus the fixed ``[Content_Types].xml`` and
    root ``_rels/.rels`` a package needs. The escape hatch for structural tests: omit
    ``xl/workbook.xml`` to test a non-workbook ZIP, put a DOCTYPE or junk in a part, scramble the
    sheet order, or point a relationship at a missing part."""
    content_types = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        "</Types>"
    )
    root_rels = (
        f'<Relationships xmlns="{_PKG_REL_NS}">'
        f'<Relationship Id="rIdRoot" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        f'Target="xl/workbook.xml"/></Relationships>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        for name, xml in parts.items():
            archive.writestr(name, xml)
    return buffer.getvalue()


def xlsx_without_workbook_part() -> bytes:
    """A valid ZIP that is not a SpreadsheetML package: it has no ``xl/workbook.xml``."""
    return xlsx_with_parts({"docProps/core.xml": "<coreProperties/>"})


# --- lower-level builders for the structural tests ------------------------------------
#
# The refusal / ordering tests need to hand-shape a workbook (a scrambled sheet order, an
# unresolvable relationship, a raw sheet part). These keep those tests short and free of repeated
# namespace literals.

_WORKSHEET_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"


def worksheet_part(inner: str) -> str:
    """A worksheet part wrapping raw ``inner`` XML (a ``<sheetData>`` block and any ``<mergeCells>``)."""
    return f'<worksheet xmlns="{_S_NS}">{inner}</worksheet>'


def sheet_data(cells_xml: str) -> str:
    """A one-row ``<sheetData>`` holding raw ``cells_xml`` (one or more ``<c>`` elements)."""
    return f'<sheetData><row r="1">{cells_xml}</row></sheetData>'


def workbook_part(sheets: Sequence[tuple[str, str]]) -> str:
    """``xl/workbook.xml`` for ``sheets`` = ``[(sheet_name, r_id)]`` in workbook order."""
    entries = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{i + 1}" r:id={quoteattr(rid)}/>'
        for i, (name, rid) in enumerate(sheets)
    )
    return f'<workbook xmlns="{_S_NS}" xmlns:r="{_R_NS}"><sheets>{entries}</sheets></workbook>'


def rels_part(rels: Sequence[tuple[str, str]]) -> str:
    """``xl/_rels/workbook.xml.rels`` for ``rels`` = ``[(r_id, target)]``, each a worksheet target."""
    items = "".join(
        f"<Relationship Id={quoteattr(rid)} Type={quoteattr(_WORKSHEET_REL)} Target={quoteattr(target)}/>"
        for rid, target in rels
    )
    return f'<Relationships xmlns="{_PKG_REL_NS}">{items}</Relationships>'
