# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Pin what replay does when a stored artifact cites a SUPERSEDED geometry composite.

The fragment-geometry lane in :mod:`carmel.services.semantic_deps` has been
superseded thirteen times, and a durable dataset artifact that records the
composite it was produced under now exists on disk. From here a geometry change
can move the numbers under an artifact that is already stored and already cited,
so what replay does in that case is a contract, not a hypothetical.

These tests fix that contract against a REAL superseded registry row (never a
fabricated sha), at both layers replay runs at:

* :func:`~carmel.services.pdf_table_record.verify_inventory_record` -- one
  record against the raw bytes. Already pinned for a fabricated composite by
  ``tests.test_pdf_table_record``; pinned here for a genuine historical row.
* :func:`carmel.services.dataset_replay._verify_embedded_inventories` -- the
  layer a stored DATASET envelope actually replays through, which nothing else
  exercises for a moved geometry composite.

They also demonstrate that the composite's two halves localize the move, run
against the real registry rows: every superseded row resolves to the ``own``
half, with ``borrowed`` unmoved -- which is the whole reason the identity is a
composite rather than an opaque sha.
"""

from __future__ import annotations

import hashlib

import pytest

from carmel.schemas.datasets import EmbeddedTableInventory
from carmel.services import semantic_deps as sd
from carmel.services.dataset_replay import ReplayOutcome, _verify_embedded_inventories
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import (
    InventoryVerificationStatus,
    inventory_record_bytes,
    inventory_record_payload,
    verify_inventory_record,
)
from carmel.services.pdf_tables import build_inventory
from tests.pypdf_gate import require_pypdf
from tests.test_pdf_table_record import FOOTPRINT, GRID

#: The first coordinate-moving supersession (V4 replaced pypdf's text-show walk),
#: chosen because it is the first row an artifact stored under it would carry
#: DIFFERENT numbers for the same page -- exactly the "geometry moved under a
#: stored artifact" case this module exists to pin. It is a real, non-current
#: registry row, asserted so below rather than assumed.
_SUPERSEDED_COMPOSITE = "4ae9d68f0bcbf55bfbcaef1f7c7a2dda02b64ef4bc6bdf7cc504672d59810545"


@pytest.fixture(autouse=True)
def _needs_pypdf() -> None:
    """Every test re-derives a real grid, so every one needs the engine.

    Mirrors ``tests.test_pdf_table_record``: the derivation goes through
    ``pdf_fragments``, which imports ``pypdf`` lazily, and CI's base job installs
    without the ``agents`` extra.
    """
    require_pypdf()


def _record() -> dict:
    require_pypdf()
    inventory = build_inventory(extract_fragments(GRID), FOOTPRINT)
    assert inventory.refusals == (), f"fixture does not derive cleanly: {inventory.refusals}"
    return inventory_record_payload(inventory, raw_sha256=hashlib.sha256(GRID).hexdigest())


def _embed(payload: dict) -> EmbeddedTableInventory:
    """Wrap a record payload as an envelope-embedded inventory, self-addressed.

    The schema's T1 checks self-coherence, self-addressing and replay-readability
    -- never that the record cites the CURRENT composite -- so a record naming a
    superseded one embeds cleanly, which is what lets replay meet it downstream.
    """
    canonical = inventory_record_bytes(payload)
    return EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical).hexdigest(),
        raw_sha256=payload["raw_sha256"],
        canonical_json=canonical.decode("utf-8"),
    )


class TestTheSupersededCompositeIsARealHistoricalRow:
    def test_the_composite_under_test_is_registered_and_not_current(self) -> None:
        """A fabricated sha would not exercise the registry's localization at all.

        The whole point is a composite the registry KNOWS and has retired, so the
        halves it recorded for that row are available to say which one moved.
        """
        definition = sd.dependency_for_sha(_SUPERSEDED_COMPOSITE)

        assert definition.dependency_id == sd.FRAGMENT_GEOMETRY_DEPENDENCY_ID
        assert definition.is_current is False
        assert sd.current_sha_for(sd.FRAGMENT_GEOMETRY_DEPENDENCY_ID) != _SUPERSEDED_COMPOSITE


class TestTheHalvesLocalizeTheMove:
    """Q3: given an artifact citing an old composite, the registry says which half moved."""

    @staticmethod
    def _moved_halves(stored: str, current: str) -> tuple[str, ...]:
        stored_c = sd._fragment_geometry_components(stored)
        current_c = sd._fragment_geometry_components(current)
        moved: list[str] = []
        if stored_c.own_sha256 != current_c.own_sha256:
            moved.append("own")
        if stored_c.borrowed_sha256 != current_c.borrowed_sha256:
            moved.append("borrowed")
        return tuple(moved)

    def test_the_first_coordinate_move_localizes_to_the_own_half(self) -> None:
        current = sd.current_sha_for(sd.FRAGMENT_GEOMETRY_DEPENDENCY_ID)

        assert self._moved_halves(_SUPERSEDED_COMPOSITE, current) == ("own",)

    def test_every_superseded_row_localizes_and_the_borrowed_half_never_moved(self) -> None:
        """The composite is worth its cost only if it can attribute the move.

        All thirteen supersessions are own-half changes in ``pdf_fragments`` --
        no borrowed name's behaviour in ``extract.py`` ever moved -- and the
        registry can say so for each, without recomputing anything.
        """
        current = sd.current_sha_for(sd.FRAGMENT_GEOMETRY_DEPENDENCY_ID)
        superseded = [
            sha
            for sha, definition in sd.DEPENDENCIES_BY_SHA.items()
            if definition.dependency_id == sd.FRAGMENT_GEOMETRY_DEPENDENCY_ID and not definition.is_current
        ]

        assert len(superseded) == 13
        assert all(self._moved_halves(sha, current) == ("own",) for sha in superseded)


class TestReplayMeetsASupersededComposite:
    """Q2: what replay does when the cited composite differs from the current one."""

    def test_the_record_layer_reproduces_and_names_the_moved_geometry(self) -> None:
        """A supersession that moved no coordinate must still recompute and say so.

        ``fragment_geometry_sha256`` is excluded from the compared bytes, so a
        record built under an old geometry that still yields this grid comes back
        REPRODUCED with the drift named -- never an indistinguishable MISMATCHED.
        """
        stale = {**_record(), "fragment_geometry_sha256": _SUPERSEDED_COMPOSITE}

        result = verify_inventory_record(stale, GRID)

        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.identity_moved == ("fragment_geometry",)

    def test_a_reproduced_supersession_passes_at_the_dataset_layer(self) -> None:
        """A stored envelope whose grid survives a geometry supersession still replays.

        ``_verify_embedded_inventories`` returns REPRODUCED to no finding: the
        artifact genuinely re-derives, and a finding would wrongly downgrade the
        report's outcome. The record-layer ``identity_moved`` is NOT re-surfaced
        here -- the composite the artifact was built under lives in the artifact's
        own bytes, not in the replay report. This is pinned so a future
        supersession cannot quietly turn a passing artifact into a finding.
        """
        stale = {**_record(), "fragment_geometry_sha256": _SUPERSEDED_COMPOSITE}
        raw_sha = stale["raw_sha256"]

        findings = _verify_embedded_inventories((_embed(stale),), {raw_sha: GRID})

        assert findings == []

    def test_a_supersession_that_moves_the_grid_fails_and_names_the_move(self) -> None:
        """The dangerous case: geometry moved AND the numbers changed under the artifact.

        Replay must not pass this silently, and it must not be indistinguishable
        from a document change -- a document change is ``SOURCE_MISMATCH`` (the
        bytes are not the ones the record names), a different status entirely.
        Here the bytes match, the grid does not re-derive, and the finding names
        the moved geometry identity in its reason so a reader can tell a
        superseded dependency from a record that simply never replayed.
        """
        rec = _record()
        moved_grid = {
            **rec,
            "fragment_geometry_sha256": _SUPERSEDED_COMPOSITE,
            "rows": [{**rec["rows"][0], "ordinal": 7}, *rec["rows"][1:]],
        }
        raw_sha = moved_grid["raw_sha256"]

        findings = _verify_embedded_inventories((_embed(moved_grid),), {raw_sha: GRID})

        assert len(findings) == 1
        (finding,) = findings
        assert finding.category is ReplayOutcome.FAILED
        assert "recorded identities moved: fragment_geometry" in finding.reason
