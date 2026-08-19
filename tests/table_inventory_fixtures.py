"""Shared fixtures for building a valid PDF-table-cell citation in tests.

Every envelope fixture that locates a table cell in a ``PAPER_PDF`` node now has
to satisfy V8/T4/T5 (see :mod:`carmel.schemas.datasets`), which means minting a
real :class:`~carmel.schemas.datasets.EmbeddedTableInventory`. Doing that once,
here, keeps seven test modules from each growing their own near-copy.

**The payload is hand-built, NOT derived by running the extractor**, for two
reasons. It keeps every consumer of these helpers runnable in the pypdf-free
suite (deriving a real grid needs ``pdf_fragments``, which needs pypdf), and it
matches what the schema actually validates: ``EmbeddedTableInventory`` checks
canonical self-coherence and self-addressing, never that the grid describes a
real table. A fixture that pretended otherwise would be claiming a guarantee the
schema explicitly disclaims -- see that class's "SCOPE OF WHAT VALIDATION HERE
PROVES".

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
    "inventory_for",
    "inventory_payload",
    "make_embedded_inventory",
    "sha_of_payload",
]

#: A grid generous enough for every existing fixture's ``row``/``col`` -- some reach
#: row 90 / col 90 as arbitrary ordinals rather than as claims about a table. Widened
#: to cover them so no existing fixture has to change what it measures; ~198 KB of
#: canonical JSON, well inside _MAX_EMBEDDED_CANONICAL_JSON_LENGTH. Fixtures that
#: care about a SPECIFIC grid build their own; these exist so a test about composition
#: or identity does not have to model a table to keep citing one.
GRID_CELLS: tuple[tuple[int, int], ...] = tuple((row, col) for row in range(100) for col in range(100))


def inventory_payload(
    *,
    raw_sha256: str,
    cells: tuple[tuple[int, int], ...] = ((0, 0), (0, 1)),
    refusals: list[dict[str, str]] | None = None,
    marker: str = "",
) -> dict[str, Any]:
    """A minimal but structurally honest inventory record payload.

    Only the fields ``EmbeddedTableInventory``'s T1 actually reads are
    populated: a payload carrying more would imply those extra fields were
    checked here, and they are not.

    ``marker`` exists so two otherwise identical payloads can be given
    different content addresses, for tests about duplicate or decorative
    embedding.
    """
    payload: dict[str, Any] = {
        "cells": [{"row": row, "col": col} for row, col in cells],
        "payload_version": INVENTORY_PAYLOAD_VERSION,
        "raw_sha256": raw_sha256,
        "refusals": refusals if refusals is not None else [],
    }
    if marker:
        payload["caption_text"] = marker
    return payload


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
