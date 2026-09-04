"""Tests for the OOXML supplementary-data lane's production path, and the acceptance test
that proves it on the real staged ``.docx`` -- read at runtime, never committed.

The synthetic tests build their own ``.docx`` bytes. The acceptance test reads the operator's
staged file if (and only if) it is present with the expected sha256, and is SKIPPED otherwise,
so CI without the corpus still passes while a machine holding it gets the real proof. Replay in
the acceptance test runs in a SEPARATE PROCESS that re-reads the file from disk, so a match
cannot be an artifact of the object graph the inventory was built from.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from carmel.paths import default_workspaces_root
from carmel.services.ooxml_table_record import OoxmlDocumentUnreadable, OoxmlInventoryVerificationStatus
from carmel.services.ooxml_tables import OoxmlReadRefusalReason, embed_ooxml_tables
from tests.ooxml_fixtures import docx_bytes

# --- the production path over synthetic documents -------------------------------------


def test_harvest_embeds_every_table_in_order() -> None:
    harvest = embed_ooxml_tables(docx_bytes([[["a"]], [["b"], ["c"]]]))
    assert len(harvest.inventories) == 2
    assert harvest.read_refusals == ()
    # Each embedded record's address is the sha of its own canonical bytes, and all name
    # the one source document.
    source_sha = hashlib.sha256(docx_bytes([[["a"]], [["b"], ["c"]]])).hexdigest()
    for embedded in harvest.inventories:
        assert hashlib.sha256(embedded.canonical_json.encode("utf-8")).hexdigest() == embedded.inventory_sha256
        assert embedded.source_sha256 == source_sha


def test_harvest_records_a_nested_table_as_a_refusal_and_keeps_the_rest() -> None:
    nested = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>outer</w:t></w:r></w:p>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>inner</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "</w:tc></w:tr></w:tbl>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>ok</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "</w:body></w:document>"
    )
    harvest = embed_ooxml_tables(docx_bytes([], document_override=nested))
    assert len(harvest.inventories) == 1  # the second, un-nested table still harvests
    assert [(r.table_index, r.reason) for r in harvest.read_refusals] == [(0, OoxmlReadRefusalReason.NESTED_TABLE)]


def test_harvest_raises_for_a_wholly_unreadable_document() -> None:
    with pytest.raises(OoxmlDocumentUnreadable):
        embed_ooxml_tables(b"not a zip at all")


def test_document_unreadable_is_catchable_through_the_production_module() -> None:
    # embed_ooxml_tables documents that it raises OoxmlDocumentUnreadable on a document-level
    # failure. A caller must be able to catch that THROUGH this production-path module --
    # importing it from here, not reaching into the core reader -- and it must be the very
    # same class the reader raises, so the catch actually composes.
    import carmel.services.ooxml_tables as ooxml_tables

    assert "OoxmlDocumentUnreadable" in ooxml_tables.__all__
    assert ooxml_tables.OoxmlDocumentUnreadable is OoxmlDocumentUnreadable
    with pytest.raises(ooxml_tables.OoxmlDocumentUnreadable):
        ooxml_tables.embed_ooxml_tables(b"not a zip at all")


def test_harvest_of_a_tableless_document_is_empty_without_refusals() -> None:
    harvest = embed_ooxml_tables(docx_bytes([]))
    assert harvest.inventories == ()
    assert harvest.read_refusals == ()


# --- acceptance: the real staged supplementary .docx ----------------------------------

_STAGED_SHA256 = "32fc25d717a1f63e9fc8d074a1ea04e67bed15a87e9d8df0d7c0640d0d4abf77"
_STAGED_RELPATH = "live-syngas/literature_requests/inbox/10.1016-j.ijhydene.2014.11.056-si1.docx"
_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")


def _staged_docx() -> Path:
    for root in _WORKSPACES_ROOTS:
        candidate = root / _STAGED_RELPATH
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == _STAGED_SHA256:
            return candidate
    pytest.skip(f"staged supplementary .docx (sha {_STAGED_SHA256[:12]}...) is not present in this environment")


# The replay half of the acceptance test, run as a SEPARATE PROCESS. It is handed only the
# file PATH and the stored records' canonical JSON; it re-reads the bytes from disk and
# reports, per table, the verification status and per-cell replay outcomes. Nothing from the
# building process's memory reaches it -- the match is genuinely from disk.
_REPLAY_CHILD = r"""
import hashlib, json, sys
from carmel.services.ooxml_table_record import (
    verify_ooxml_inventory_record, replay_ooxml_cell, cell_text_from_payload,
    OoxmlInventoryVerificationStatus, OoxmlCellReplayOutcome,
)
docx_path, records_path = sys.argv[1], sys.argv[2]
data = open(docx_path, "rb").read()
records = json.load(open(records_path))
report = []
for rec in records:
    payload = json.loads(rec)
    v = verify_ooxml_inventory_record(payload, data)
    cells = json.loads(rec)["cells"]
    outcomes = {}
    for cell in cells:
        expected = cell_text_from_payload(payload, row=cell["row"], col=cell["col"])
        r = replay_ooxml_cell(payload, data, row=cell["row"], col=cell["col"], expected_text=expected)
        outcomes[r.outcome.value] = outcomes.get(r.outcome.value, 0) + 1
    report.append({"status": v.status.value, "cell_count": len(cells), "outcomes": outcomes})
json.dump(report, sys.stdout)
"""


def test_the_real_staged_docx_reads_and_replays_from_disk_in_a_separate_process(tmp_path: Path) -> None:
    docx_path = _staged_docx()
    data = docx_path.read_bytes()

    harvest = embed_ooxml_tables(data)
    assert harvest.read_refusals == ()
    assert len(harvest.inventories) == 3, "the file holds exactly three tables"

    payloads = [json.loads(inv.canonical_json) for inv in harvest.inventories]
    total_rows = sum(p["row_count"] for p in payloads)
    total_cells = sum(len(p["cells"]) for p in payloads)
    assert total_rows == 116, f"expected 116 rows across the three tables, got {total_rows}"
    assert total_cells == 648, f"expected 648 cells across the three tables, got {total_cells}"

    # Replay in a separate process that re-reads the file from disk.
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps([inv.canonical_json for inv in harvest.inventories]))
    result = subprocess.run(
        [sys.executable, "-c", _REPLAY_CHILD, str(docx_path), str(records_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert len(report) == 3
    for table in report:
        # The child emits the enum's string value; REPRODUCED is the only passing status.
        assert table["status"] == OoxmlInventoryVerificationStatus.REPRODUCED.value
        # Every cell of every table replays to a MATCH -- no FAILED, no UNVERIFIABLE.
        assert set(table["outcomes"]) == {"match"}, table["outcomes"]
    assert sum(t["cell_count"] for t in report) == 648
