"""Acceptance test against the real split-header table this branch (i032) exists for:
Table 1, page 4 of ``10.1115-1.4007737`` -- a shock-tube paper whose flame-speed and
uncertainty columns are labelled ``S`` ... ``(cm/s)`` and ``U`` ... ``(cm/s)``, each
symbol separated from its unit by a gap wider than a column valley.

Like :mod:`tests.test_target_table_acceptance`, this is the deliberate corpus exception:
the document is copyrighted and non-redistributable, so nothing here is a transcription of
its prose. What is checked in is this project's OWN footprint claim (the caption anchor and
the four box edges) and the STRUCTURAL shape of the grid the fix now extracts -- a count of
rows, columns and cells, never the values themselves. The document is read from the
operator's corpus inbox at runtime, and this module SKIPS -- never passes -- when it is
absent or is not byte-for-byte the measured document.

The defect: the header row resolves one more block than the data rows do, because the data
bridges the ``S`` / ``(cm/s)`` gap with a single value while the header does not. The old
``column_structure_unresolved`` guard read that as a spanning row and refused the whole
table. It is not a spanning row: removing the header reveals no hidden column, and removing
any one data row leaves the rest still filling the gap. The refined guard (leave-one-out)
admits it, and this is the page the measurement was taken on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from carmel.paths import default_workspaces_root
from carmel.services.pdf_fragments import FragmentExtraction, extract_fragments
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory
from tests.pypdf_gate import require_pypdf

#: Runtime-read, never shipped.
_DOCUMENT_SUBPATH = "live-syngas/literature_requests/inbox/10.1115-1.4007737.pdf"

#: Same resolver order as the sibling acceptance module, for the same reason: the corpus
#: lives under the pinned workspaces root on some machines and under ``~/runs`` on others.
_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")

#: The exact bytes the measurement was taken against; a different file at this path is a
#: different document and the gate SKIPS naming the mismatch rather than asserting on it.
_DOCUMENT_SHA256 = "c2be41381e3c55671af2912a46d5ce703c0f56cea9dadfba8789e7417059155a"

#: The registered footprint for Table 1. The table sits in the page's RIGHT column
#: (x in [312, 552]); ``x`` in [305, 555] bounds it and excludes the left column's Fig. 5/6
#: prose. The caption is two printed lines; the box is anchored on its LOWER line
#: ("range of equivalence ratios", baseline 750.274), the line nearest the table, so the
#: upper line sits above the box rather than orphaned inside it. ``y_top`` = 745 clears the
#: header band at 734.7; ``y_bottom`` = 520 clears the last data row at 524.5. These are the
#: honest edges of the table, not values tuned to slip a guard.
_TABLE_1 = ClaimedFootprint(
    page=4,
    x_start=305.0,
    x_end=555.0,
    y_top=745.0,
    y_bottom=520.0,
    caption_text="rangeofequivalenceratios",
    caption_x_start=311.981,
    caption_baseline_y=750.274,
)


def _locate_document() -> Path | None:
    for root in _WORKSPACES_ROOTS:
        candidate = root / _DOCUMENT_SUBPATH
        if candidate.exists():
            return candidate
    return None


def _target_extraction() -> FragmentExtraction:
    require_pypdf()
    document = _locate_document()
    if document is None:
        roots = ", ".join(str(r) for r in _WORKSPACES_ROOTS)
        pytest.skip(f"split-header corpus document is not present under any of: {roots}")
    data = document.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != _DOCUMENT_SHA256:
        pytest.skip(f"document at {document} is {actual}, not the measured {_DOCUMENT_SHA256}")
    extraction = extract_fragments(data)
    assert extraction.available, f"the fragment lane refused the target document: {extraction.status}"
    return extraction


def test_the_split_header_table_extracts_a_full_grid() -> None:
    """The whole point of the branch, on the page it was measured on: the table that the
    old guard refused now extracts a complete 23x4 grid (92 cells), and refuses nothing.
    A caption line becoming a row or a value column collapsing would change these counts,
    so the shape is asserted, not merely non-emptiness."""
    inventory = build_inventory(_target_extraction(), _TABLE_1)

    assert inventory.refusals == (), [(r.reason, r.detail) for r in inventory.refusals]
    assert len(inventory.column_bounds) == 4
    assert len(inventory.rows) == 23
    assert len(inventory.cells) == 92
    assert inventory.complete
