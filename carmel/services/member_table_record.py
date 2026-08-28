"""A byte-replayable table inventory for a tabulated supplementary member.

This is the member-lane counterpart to :mod:`carmel.services.pdf_table_record`. Where
that module derives a grid from a PDF's text fragments (geometry, glyph ink, a pypdf
engine version) and re-derives it to verify, this one derives a grid from a delimited
member's own bytes (CSV/TSV) and re-derives it the same way. The shape of the two is
deliberately identical -- a canonical-JSON record that hashes to its own address, that
names the document it came from, and that a verifier can REPRODUCE from that document's
raw bytes without any store -- so an SI member's cell is as byte-replayable as a PDF
table cell.

It is a SEPARATE record type, not a reuse of the PDF one, and that is not an oversight:
:class:`~carmel.schemas.datasets.EmbeddedTableInventory`'s payload is version-4 and
PDF-shaped (``footprint`` page geometry, ``column_bounds``, ``pypdf_version``), and its
schema refuses any payload whose footprint is not a readable PDF box -- a workbook sheet
has no page geometry at all. Forcing a sheet through it would mean weakening exactly the
validators that make the PDF citation honest.

Scope of this first delivery: DELIMITED TEXT only. The delimiter is a pure function of
the sheet name's extension (``.tsv``/``.tab`` -> tab, else comma), so a verifier
re-derives with the same delimiter the record was built under without the record having
to store it. Real multi-sheet workbooks (``.xlsx``) need a runtime library and are a
separate, operator-gated delivery.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from carmel.services.dataset_store import canonical_json_bytes

__all__ = [
    "MEMBER_INVENTORY_PAYLOAD_KEYS",
    "MEMBER_INVENTORY_PAYLOAD_VERSION",
    "MemberCell",
    "MemberCellInventory",
    "MemberCellReplay",
    "MemberCellReplayOutcome",
    "MemberInventoryVerification",
    "MemberInventoryVerificationStatus",
    "MemberTableUnreadable",
    "cell_text_from_payload",
    "compute_member_inventory_sha",
    "delimiter_for_sheet",
    "member_inventory_record_bytes",
    "member_inventory_record_payload",
    "read_delimited_member",
    "replay_member_cell",
    "verify_member_inventory_record",
]

#: The on-disk shape of a member-table inventory record. Bumped whenever a field is
#: added or changed in a way that is not simply optional-with-a-default. A reader that
#: does not know a version must not guess at its shape -- see
#: :func:`verify_member_inventory_record`, which returns ``PAYLOAD_UNREADABLE`` rather
#: than read an unknown one.
MEMBER_INVENTORY_PAYLOAD_VERSION = 1

#: Exactly the top-level keys a version-``MEMBER_INVENTORY_PAYLOAD_VERSION`` record has
#: -- no more, no fewer. EXACT rather than "at least these", for the same reason as
#: :data:`carmel.services.pdf_table_record.INVENTORY_PAYLOAD_KEYS`: the address is over
#: the canonical bytes, and the verifier compares them against a freshly built payload,
#: so a record carrying a stray key can never reproduce.
MEMBER_INVENTORY_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "cells",
        "col_count",
        "member_sha256",
        "payload_version",
        "row_count",
        "sheet_name",
    }
)

_SHEET_TAB_SUFFIXES = (".tsv", ".tab")


class MemberTableUnreadable(ValueError):
    """The member's bytes could not be read as delimited text (e.g. not UTF-8)."""


@dataclass(frozen=True)
class MemberCell:
    """One cell of a member's grid: its 0-indexed position and verbatim text."""

    row: int
    col: int
    text: str


@dataclass(frozen=True)
class MemberCellInventory:
    """A tabulated member's grid, derived from its own bytes.

    ``cells`` is every field of every row, so an empty field is a present cell with
    empty text -- addressing must be able to distinguish "this cell is blank" from
    "this row was too short to have this column".
    """

    sheet_name: str
    row_count: int
    col_count: int
    cells: tuple[MemberCell, ...]


def delimiter_for_sheet(sheet_name: str) -> str:
    """The delimiter a member with this sheet name is read with.

    A pure function of the name's extension so that :func:`read_delimited_member` and
    :func:`verify_member_inventory_record` cannot disagree about it, and so the record
    need not store it: a stored delimiter would be a second place the truth could live.
    """
    lowered = sheet_name.lower()
    if lowered.endswith(_SHEET_TAB_SUFFIXES):
        return "\t"
    return ","


def read_delimited_member(data: bytes, *, sheet_name: str) -> MemberCellInventory:
    """Read a delimited-text member's bytes into a :class:`MemberCellInventory`.

    Decoding is UTF-8 (BOM-tolerant) and STRICT: a member that is not valid UTF-8 is
    refused rather than silently repaired, because replay must reproduce byte-for-byte
    and a lossy decode would make two different files read the same. Parsing uses the
    stdlib :mod:`csv` reader under the excel dialect, so quoting and embedded newlines
    are honoured deterministically.

    Raises:
        MemberTableUnreadable: If the bytes are not valid UTF-8.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MemberTableUnreadable(f"member {sheet_name!r} is not valid UTF-8 delimited text: {exc}") from exc
    delimiter = delimiter_for_sheet(sheet_name)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    # The reader is consumed as a STREAM rather than materialised with `list(reader)`:
    # the parsed rows and `cells` hold the same field strings, so keeping both alive at
    # once adds an intermediate for no purpose. Measured on a 15 MB delimited member the
    # saving is about 9% of peak -- real but modest, because `cells` (one frozen
    # dataclass per field) dominates either way. `row_count` is tracked here because
    # there is no `len()` on a stream.
    cells: list[MemberCell] = []
    col_count = 0
    row_count = 0
    for row_index, row in enumerate(reader):
        row_count = row_index + 1
        col_count = max(col_count, len(row))
        for col_index, field in enumerate(row):
            cells.append(MemberCell(row=row_index, col=col_index, text=field))
    return MemberCellInventory(
        sheet_name=sheet_name,
        row_count=row_count,
        col_count=col_count,
        cells=tuple(cells),
    )


def member_inventory_record_payload(inventory: MemberCellInventory, *, member_sha256: str) -> dict[str, Any]:
    """The full stored form of one member inventory, ready for
    :func:`canonical_json_bytes`.

    Cells are sorted by ``(row, col)`` so the record's bytes -- and therefore its
    address -- do not depend on iteration order.

    Raises:
        ValueError: If ``member_sha256`` is not a well-formed digest. It is the
            record's only link to the member, and nothing downstream re-derives it, so
            a malformed one would mint a record that reports ``SOURCE_MISMATCH`` forever.
    """
    if len(member_sha256) != 64 or any(c not in "0123456789abcdef" for c in member_sha256):
        raise ValueError(f"member_sha256 must be 64 lowercase hex characters, got {member_sha256!r}")
    return {
        "cells": [
            {"col": cell.col, "row": cell.row, "text": cell.text}
            for cell in sorted(inventory.cells, key=lambda c: (c.row, c.col))
        ],
        "col_count": inventory.col_count,
        "member_sha256": member_sha256,
        "payload_version": MEMBER_INVENTORY_PAYLOAD_VERSION,
        "row_count": inventory.row_count,
        "sheet_name": inventory.sheet_name,
    }


def member_inventory_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of this record: what its address is over. One definition,
    shared with :func:`compute_member_inventory_sha`, so the two cannot drift."""
    return canonical_json_bytes(dict(payload))


def compute_member_inventory_sha(payload: Mapping[str, Any]) -> str:
    """This record's content address: the sha256 of its canonical JSON bytes."""
    return hashlib.sha256(member_inventory_record_bytes(payload)).hexdigest()


class MemberInventoryVerificationStatus(StrEnum):
    """The outcome of checking a stored member inventory record against a member's
    bytes. Mirrors :class:`~carmel.services.pdf_table_record.InventoryVerificationStatus`
    minus the PDF-engine-specific states, which have no member-lane analogue."""

    REPRODUCED = "reproduced"
    """The re-derivation ran and its canonical bytes are identical to the stored ones.
    The only positive result: the stored grid follows from these bytes under this
    reader."""

    MISMATCHED = "mismatched"
    """The re-derivation ran and produced a different grid. The record is not
    evidence."""

    SOURCE_MISMATCH = "source_mismatch"
    """``data`` is not the member this record is about; its sha256 differs from the
    stored one. Nothing was re-derived -- reporting it as "could not verify" would let
    a caller retry forever against the wrong file."""

    PAYLOAD_UNREADABLE = "payload_unreadable"
    """The stored payload is malformed or of an unknown version. A verifier that
    guessed at an unknown shape would be inventing the fields it is meant to check."""


@dataclass(frozen=True)
class MemberInventoryVerification:
    """What checking a record against a member's bytes established."""

    status: MemberInventoryVerificationStatus
    detail: str = ""


def _payload_unreadable_reason(payload: Mapping[str, Any]) -> str | None:
    """Why this payload cannot be read back, or ``None`` if it can. Performs the same
    shape checks the schema and verifier both rely on, so there is one definition of
    "readable"."""
    version = payload.get("payload_version")
    if version != MEMBER_INVENTORY_PAYLOAD_VERSION:
        return f"payload_version {version!r} is not the readable version {MEMBER_INVENTORY_PAYLOAD_VERSION!r}"
    keys = set(payload)
    if keys != set(MEMBER_INVENTORY_PAYLOAD_KEYS):
        unexpected = sorted(keys - MEMBER_INVENTORY_PAYLOAD_KEYS)
        missing = sorted(MEMBER_INVENTORY_PAYLOAD_KEYS - keys)
        return (
            f"record is not the shape of a version-{MEMBER_INVENTORY_PAYLOAD_VERSION} inventory "
            f"(unexpected keys {unexpected!r}, missing keys {missing!r})"
        )
    sheet_name = payload.get("sheet_name")
    if not isinstance(sheet_name, str) or not sheet_name:
        return f"sheet_name {sheet_name!r} is not a non-empty string"
    member_sha256 = payload.get("member_sha256")
    if (
        not isinstance(member_sha256, str)
        or len(member_sha256) != 64
        or any(c not in "0123456789abcdef" for c in member_sha256)
    ):
        return f"member_sha256 {member_sha256!r} is not 64 lowercase hex characters"
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return f"cells is {type(cells).__name__}, not a list"
    return None


def verify_member_inventory_record(payload: Mapping[str, Any], data: bytes) -> MemberInventoryVerification:
    """Check a stored member inventory record against a member's raw bytes.

    Re-derives the grid from ``data`` under the same reader that built the record and
    compares canonical bytes. A difference is reported ``MISMATCHED`` and its detail
    NAMES the first cell that disagrees, so a falsification test can point at what
    changed. The store is never consulted: ``data`` is the source of truth.
    """
    reason = _payload_unreadable_reason(payload)
    if reason is not None:
        return MemberInventoryVerification(status=MemberInventoryVerificationStatus.PAYLOAD_UNREADABLE, detail=reason)

    member_sha256 = str(payload["member_sha256"])
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != member_sha256:
        return MemberInventoryVerification(
            status=MemberInventoryVerificationStatus.SOURCE_MISMATCH,
            detail=f"these bytes hash to {actual_sha!r}, but the record is about member {member_sha256!r}",
        )

    sheet_name = str(payload["sheet_name"])
    try:
        inventory = read_delimited_member(data, sheet_name=sheet_name)
    except MemberTableUnreadable as exc:
        return MemberInventoryVerification(
            status=MemberInventoryVerificationStatus.MISMATCHED,
            detail=f"the member bytes cannot be re-read as the claimed grid: {exc}",
        )
    rebuilt = member_inventory_record_payload(inventory, member_sha256=actual_sha)
    if canonical_json_bytes(rebuilt) == member_inventory_record_bytes(payload):
        return MemberInventoryVerification(status=MemberInventoryVerificationStatus.REPRODUCED)

    difference = _first_cell_difference(payload, rebuilt)
    return MemberInventoryVerification(status=MemberInventoryVerificationStatus.MISMATCHED, detail=difference)


def _cell_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    """Map ``(row, col) -> text`` for a payload whose ``cells`` has passed
    :func:`_payload_unreadable_reason`. Non-conforming cell entries are skipped, so a
    hand-mangled cell surfaces as a difference rather than a crash."""
    mapping: dict[tuple[int, int], str] = {}
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return mapping
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = cell.get("row")
        col = cell.get("col")
        text = cell.get("text")
        if isinstance(row, bool) or isinstance(col, bool):
            continue
        if isinstance(row, int) and isinstance(col, int) and isinstance(text, str):
            mapping[(row, col)] = text
    return mapping


def _first_cell_difference(stored: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> str:
    """A message naming the first ``(row, col)`` where ``stored`` and ``rebuilt``
    disagree. Falls back to a whole-record message if the difference is not at cell
    granularity (e.g. a differing ``row_count``)."""
    stored_cells = _cell_map(stored)
    rebuilt_cells = _cell_map(rebuilt)
    for key in sorted(set(stored_cells) | set(rebuilt_cells)):
        stored_text = stored_cells.get(key)
        rebuilt_text = rebuilt_cells.get(key)
        if stored_text != rebuilt_text:
            return (
                f"cell (row={key[0]}, col={key[1]}) is stored as {stored_text!r} but the member's bytes "
                f"yield {rebuilt_text!r}"
            )
    return "the re-derived grid differs from the stored record (shape or counts), though every named cell agrees"


def cell_text_from_payload(payload: Mapping[str, Any], *, row: int, col: int) -> str | None:
    """This record's own text for ``(row, col)``, or ``None`` if the grid has no such
    cell. Answers from the payload's cells; it proves nothing about whether that text
    is what the member printed -- only :func:`verify_member_inventory_record` does."""
    return _cell_map(payload).get((row, col))


class MemberCellReplayOutcome(StrEnum):
    """The outcome of replaying one addressed member cell against member bytes."""

    MATCH = "match"
    """The record reproduced from the bytes AND the addressed cell's stored text equals
    the expected value."""

    FAILED = "failed"
    """A comparison ran and disagreed: the grid did not reproduce, the bytes are the
    wrong member, or the addressed cell's text is not the expected value."""

    UNVERIFIABLE = "unverifiable"
    """No comparison could run -- the stored payload is unreadable -- so the citation is
    neither confirmed nor refuted."""


@dataclass(frozen=True)
class MemberCellReplay:
    """What replaying one addressed member cell established."""

    outcome: MemberCellReplayOutcome
    detail: str = ""


def replay_member_cell(
    payload: Mapping[str, Any],
    data: bytes,
    *,
    row: int,
    col: int,
    expected_text: str,
) -> MemberCellReplay:
    """Replay the value ``expected_text``, claimed to sit at ``(row, col)`` of the
    member whose bytes are ``data``, against those bytes.

    Two things must hold for a ``MATCH``: the whole record must REPRODUCE from the
    member's bytes (so the grid is real), and the addressed cell's stored text must
    equal ``expected_text`` (the exact-equality contract the PDF lane also enforces).
    Any disagreement is ``FAILED`` with a detail naming the cause; an unreadable
    payload is ``UNVERIFIABLE``.
    """
    verification = verify_member_inventory_record(payload, data)
    if verification.status is MemberInventoryVerificationStatus.PAYLOAD_UNREADABLE:
        return MemberCellReplay(outcome=MemberCellReplayOutcome.UNVERIFIABLE, detail=verification.detail)
    if verification.status is not MemberInventoryVerificationStatus.REPRODUCED:
        return MemberCellReplay(
            outcome=MemberCellReplayOutcome.FAILED,
            detail=f"member inventory did not reproduce ({verification.status.value}): {verification.detail}",
        )
    actual = cell_text_from_payload(payload, row=row, col=col)
    if actual is None:
        return MemberCellReplay(
            outcome=MemberCellReplayOutcome.FAILED,
            detail=f"the grid has no cell at (row={row}, col={col})",
        )
    if actual != expected_text:
        return MemberCellReplay(
            outcome=MemberCellReplayOutcome.FAILED,
            detail=f"cell (row={row}, col={col}) holds {actual!r}, not the expected {expected_text!r}",
        )
    return MemberCellReplay(outcome=MemberCellReplayOutcome.MATCH)
