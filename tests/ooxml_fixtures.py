"""Synthetic WordprocessingML (`.docx`) builders for the OOXML table-lane tests.

Every `.docx` under test is built HERE from bytes, so no real supplementary document
enters the repository. A cell is either a plain string (its text) or a
:class:`CellSpec` carrying a horizontal span (`grid_span`) and/or a vertical-merge state
(`v_merge`), so a test can construct exactly the merged-cell shapes the real corpus uses
without shipping a corpus file.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

CellValue = "str | CellSpec"


@dataclass(frozen=True)
class CellSpec:
    """One table cell: its verbatim text, an optional ``w:gridSpan`` (columns spanned),
    and an optional ``w:vMerge`` state (``"restart"`` for the origin, ``"continue"`` for a
    continuation)."""

    text: str = ""
    grid_span: int | None = None
    v_merge: str | None = None


def _cell_xml(cell: CellValue) -> str:
    spec = cell if isinstance(cell, CellSpec) else CellSpec(text=cell)
    props = []
    if spec.grid_span is not None:
        props.append(f"<w:gridSpan w:val={quoteattr(str(spec.grid_span))}/>")
    if spec.v_merge is not None:
        # "continue" is expressed by a bare <w:vMerge/> in the wild; carry it as an
        # explicit val too so both spellings are exercised somewhere.
        props.append(f"<w:vMerge w:val={quoteattr(spec.v_merge)}/>")
    tc_pr = f"<w:tcPr>{''.join(props)}</w:tcPr>" if props else ""
    return f"<w:tc>{tc_pr}<w:p><w:r><w:t>{escape(spec.text)}</w:t></w:r></w:p></w:tc>"


def _table_xml(rows: list[list[CellValue]]) -> str:
    trs = []
    for row in rows:
        tcs = "".join(_cell_xml(cell) for cell in row)
        trs.append(f"<w:tr>{tcs}</w:tr>")
    return f"<w:tbl>{''.join(trs)}</w:tbl>"


def document_xml(tables: list[list[list[CellValue]]]) -> str:
    """The ``word/document.xml`` string for a document holding ``tables`` (each a list of
    rows, each row a list of cells), in order, as direct children of ``w:body``."""
    body = "".join(_table_xml(rows) for rows in tables)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>{body}</w:body></w:document>'
    )


def docx_bytes(tables: list[list[list[CellValue]]], *, document_override: str | None = None) -> bytes:
    """A minimal but valid ``.docx`` (ZIP) holding ``tables``. ``document_override`` puts
    arbitrary bytes in ``word/document.xml`` instead, for malformed-document tests."""
    document = document_override if document_override is not None else document_xml(tables)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            "</Types>",
        )
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def docx_without_document_part() -> bytes:
    """A valid ZIP that is not a WordprocessingML package: it has no ``word/document.xml``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("hello.txt", "not a word document")
    return buffer.getvalue()
