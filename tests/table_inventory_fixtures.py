"""Shared fixtures for building a valid PDF-table-cell citation in tests.

Every envelope fixture that locates a table cell in a ``PAPER_PDF`` node now has
to satisfy V8/T4/T5 (see :mod:`carmel.schemas.datasets`), which means minting a
real :class:`~carmel.schemas.datasets.EmbeddedTableInventory`. Doing that once,
here, keeps seven test modules from each growing their own near-copy.

**The payload is hand-built, NOT derived by running the extractor**, for two
reasons. It keeps every consumer of these helpers runnable in the pypdf-free
suite (deriving a real grid needs ``pdf_fragments``, which needs pypdf), and it
matches what the schema actually validates: ``EmbeddedTableInventory`` checks
canonical self-coherence, self-addressing and replay-READABILITY, never that the
grid describes a real table. A fixture that pretended otherwise would be
claiming a guarantee the schema explicitly disclaims -- see that class's "SCOPE
OF WHAT VALIDATION HERE PROVES".

Hand-built no longer means shape-free. T1 requires exactly the top-level keys of
a real record, so these payloads carry all ten; what stays fake is every VALUE.
That split is the point: a fixture may lie about what a table says, but it may
not stand in for a record that could never be replayed at all, because the
schema now refuses those and a fixture exempt from that refusal would be testing
a schema nobody ships.

The one thing these helpers must never become is a way to assert that a grid is
CORRECT. That is ``verify_inventory_record``'s job, against real bytes, and
:mod:`tests.test_pdf_table_record` is where it is tested.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from carmel.schemas.datasets import EmbeddedTableInventory, SourceRef, iter_source_refs
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.pdf_table_record import INVENTORY_PAYLOAD_VERSION

__all__ = [
    "GRID_CELLS",
    "cited_inventories",
    "cover_for",
    "embed",
    "inventory_for",
    "inventory_payload",
    "make_embedded_inventory",
    "sha_of_payload",
]

#: A grid covering every ``row``/``col`` existing fixtures actually cite -- including the
#: one that reaches row 90 / col 90 as an arbitrary ordinal rather than as a claim about a
#: table -- so no existing fixture has to change what it measures.
#:
#: SPARSE rather than a dense 100x100 block, and deliberately so: once a fixture payload
#: has to carry a real record's per-cell shape (digests, text, an x-extent), a dense block
#: costs ~1.7 MB of canonical JSON and breaches
#: ``_MAX_EMBEDDED_CANONICAL_JSON_LENGTH``. This is ~27 KB. Nothing checks that a grid is
#: contiguous, because nothing here checks that a grid is REAL -- see this module's
#: docstring. Fixtures that care about a SPECIFIC grid build their own; these exist so a
#: test about composition or identity does not have to model a table to keep citing one.
_GRID_ROWS: tuple[int, ...] = (*range(32), 90, 91)
_GRID_COLS: tuple[int, ...] = (0, 1, 2, 3, 90)
GRID_CELLS: tuple[tuple[int, int], ...] = tuple((row, col) for row in _GRID_ROWS for col in _GRID_COLS)

#: Stands in for a sha-shaped field whose VALUE nothing under test reads. Distinct from any
#: real digest by construction, so a test that started depending on one would fail loudly
#: rather than pass on a coincidence.
_PLACEHOLDER_SHA = "f" * 64


def inventory_payload(
    *,
    raw_sha256: str,
    cells: tuple[tuple[int, int], ...] = ((0, 0), (0, 1)),
    refusals: list[dict[str, str]] | None = None,
    marker: str = "",
) -> dict[str, Any]:
    """A structurally complete inventory record payload.

    Carries EXACTLY the top-level keys of a real version-1 record --
    ``INVENTORY_PAYLOAD_KEYS``, which T1 now requires -- because a payload
    missing ``footprint`` could never be read by ``verify_inventory_record``
    and one with a stray key could never reproduce. A fixture standing in for
    a record has to be replay-readable or it is standing in for something the
    schema no longer accepts.

    The VALUES are still hand-built and mean nothing: the coordinates are
    arbitrary, the identity shas are placeholders, and no PDF was parsed. That
    is the honest position, because T1 checks shape and self-coherence and
    explicitly does not check that the grid is real.

    ``marker`` varies the footprint's ``caption_text`` -- a real field of a
    real record -- so two otherwise identical payloads get different content
    addresses, for tests about duplicate or decorative embedding.
    """
    return {
        "cells": [
            {
                "col": col,
                "member_digests": [_PLACEHOLDER_SHA],
                "row": row,
                "text": f"r{row}c{col}",
                "x_end": float(10 * col + 9).hex(),
                "x_start": float(10 * col).hex(),
            }
            for row, col in cells
        ],
        "column_bounds": [[float(0).hex(), float(500).hex()]],
        "footprint": {
            "caption_baseline_y": float(700).hex(),
            "caption_text": marker or "Table 1. A fixture, not a table.",
            "caption_x_start": float(72).hex(),
            "page": 0,
            "x_end": float(500).hex(),
            "x_start": float(72).hex(),
            "y_bottom": float(200).hex(),
            "y_top": float(690).hex(),
        },
        "fragment_geometry_sha256": _PLACEHOLDER_SHA,
        "inventory_code_sha256": _PLACEHOLDER_SHA,
        "payload_version": INVENTORY_PAYLOAD_VERSION,
        "pypdf_version": "0.0.0-fixture",
        "raw_sha256": raw_sha256,
        "refusals": refusals if refusals is not None else [],
        "rows": [
            {
                "anchor_text": f"row {row}",
                "anchor_x_start": float(72).hex(),
                "baseline_y": float(600 - row).hex(),
                "merged_baselines": [],
                "ordinal": row,
            }
            for row in sorted({row for row, _ in cells})
        ],
    }


def embed(payload: dict[str, Any], *, raw_sha256: str) -> EmbeddedTableInventory:
    """Construct an ``EmbeddedTableInventory`` over ``payload`` at its TRUE address.

    For negative tests. The address and the canonical bytes are always made
    self-consistent, so whatever the caller perturbed in ``payload`` is the only
    thing left for validation to object to -- a test that hand-rolled the digest
    could fail on the address check while believing it had proved something
    about the perturbation.
    """
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=raw_sha256,
        canonical_json=canonical,
    )


def sha_of_payload(payload: dict[str, Any]) -> str:
    """The address a payload lives at: sha256 over its canonical bytes, which
    is exactly how ``compute_inventory_sha`` defines it."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def make_embedded_inventory(
    *,
    raw_sha256: str,
    cells: tuple[tuple[int, int], ...] = ((0, 0), (0, 1)),
    refusals: list[dict[str, str]] | None = None,
    marker: str = "",
) -> EmbeddedTableInventory:
    """An ``EmbeddedTableInventory`` that validates, over ``raw_sha256``."""
    payload = inventory_payload(raw_sha256=raw_sha256, cells=cells, refusals=refusals, marker=marker)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    inventory = EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=raw_sha256,
        canonical_json=canonical,
    )
    # Registered so cover_for can resolve it: a fixture that builds its own narrow
    # grid (a pinned-bytes test cannot embed the shared wide one and stay readable)
    # must still be coverable.
    _BY_INVENTORY_SHA[inventory.inventory_sha256] = inventory
    return inventory


def inventory_for(raw_sha256: str) -> EmbeddedTableInventory:
    """The standard :data:`GRID_CELLS` inventory over ``raw_sha256``.

    Memoized, so two refs into the same document cite ONE address -- which is
    what T4's exact cover and the duplicate guard expect, and what a real
    producer would emit.
    """
    cached = _BY_RAW_SHA.get(raw_sha256)
    if cached is None:
        cached = make_embedded_inventory(raw_sha256=raw_sha256, cells=GRID_CELLS)
        _BY_RAW_SHA[raw_sha256] = cached
        _BY_INVENTORY_SHA[cached.inventory_sha256] = cached
    return cached


def cited_inventories(*raw_sha256s: str) -> tuple[EmbeddedTableInventory, ...]:
    """The embedded collection for an envelope citing table cells in exactly
    these documents -- deduplicated and sorted, as T5 requires."""
    unique = {inventory_for(sha).inventory_sha256: inventory_for(sha) for sha in raw_sha256s}
    return tuple(sorted(unique.values(), key=lambda inventory: inventory.inventory_sha256))


def cover_for(*refs: Any) -> tuple[EmbeddedTableInventory, ...]:
    """The EXACT embedded collection for an envelope holding these refs.

    Reads each ref's own citation rather than being TOLD which documents are
    involved, so a fixture cannot drift out of T4's exact cover by changing a
    ref and forgetting a separate list. A ref that cites nothing (a bbox, an
    XML table cell) contributes nothing -- which is what "exact" means here.

    Only inventories this module minted are resolvable: a fixture citing a
    hand-built address gets a ``KeyError`` rather than a silently short cover.
    """
    cited: dict[str, EmbeddedTableInventory] = {}
    for item in refs:
        # An argument may be a SourceRef itself, or any container that holds them
        # (a Composition, a tuple of Series). iter_source_refs is the SAME
        # shape-agnostic walker the envelope's own validators use, so a fixture and
        # the validator that judges it can never disagree about which refs exist.
        candidates = [item] if isinstance(item, SourceRef) else [ref for _, ref in iter_source_refs(item)]
        for ref in candidates:
            sha = getattr(ref.locator, "pdf_table_inventory_sha256", None)
            if isinstance(sha, str):
                cited[sha] = _BY_INVENTORY_SHA[sha]
    return tuple(sorted(cited.values(), key=lambda inventory: inventory.inventory_sha256))


_BY_RAW_SHA: dict[str, EmbeddedTableInventory] = {}
_BY_INVENTORY_SHA: dict[str, EmbeddedTableInventory] = {}


def canonical_json_of(payload: dict[str, Any]) -> str:
    """``payload``'s canonical rendering as a str, for tests that need to
    perturb the bytes rather than the object."""
    return canonical_json_bytes(payload).decode("utf-8")


def json_roundtrip(canonical: str) -> dict[str, Any]:
    """Parse a canonical rendering back, for assertions about its content."""
    parsed: dict[str, Any] = json.loads(canonical)
    return parsed
