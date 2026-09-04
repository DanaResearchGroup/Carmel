# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Render helpers shared by the typed target slices.

Both :mod:`carmel.services.tabular_dataset_target` and
:mod:`carmel.services.condition_set_target` render a stored envelope to
human-readable text, and both need the SAME fact off the envelope first: the
table label and source page the envelope itself cites, so the render shows the
envelope's OWN facts rather than whatever the module constants happen to say
today. That helper was hand-copied into both slices; the copies drifted (one
grew a guard the other lacked), which is why it now lives here once.

It is a free function over a structural :class:`_TableCitingEnvelope`, not a base
class or a method: the two envelope types must stay free to differ (see
:class:`carmel.schemas.datasets.ConditionSetEnvelope`'s docstring for why
inheritance was rejected), so this follows the same shape as
``carmel.schemas.datasets``'s SHARED-PROVENANCE-VALIDATORS -- a module-level
function reading only the structural surface both envelopes share.

Private (leading underscore) because it is an implementation detail of the two
target slices, not a public rendering API.
"""

from __future__ import annotations

import json
from typing import Protocol

from carmel.schemas.datasets import (
    CaptionLabelKey,
    EmbeddedTableInventory,
    TableCellLocator,
    iter_source_refs,
)

__all__ = ["stored_table_reference"]


class _TableCitingEnvelope(Protocol):
    """The only surface :func:`stored_table_reference` reads off an envelope.

    Both :class:`~carmel.schemas.datasets.DatasetEnvelope` and
    :class:`~carmel.schemas.datasets.ConditionSetEnvelope` satisfy it
    structurally; the walker :func:`~carmel.schemas.datasets.iter_source_refs`
    takes ``object``, so passing the whole envelope to it is fine either way.
    Mirrors :class:`carmel.schemas.datasets._SourceGraphEnvelope`, which lets the
    same helper serve both envelope classes without inheritance.
    """

    @property
    def table_inventories(self) -> tuple[EmbeddedTableInventory, ...]: ...


def stored_table_reference(envelope: _TableCitingEnvelope) -> tuple[str, str]:
    """Read the table label and source page the envelope itself cites.

    Walks every :class:`~carmel.schemas.datasets.SourceRef` reachable in the
    envelope (the same choke point V1 validates every ref through) for the
    first ``TableCellLocator`` naming a ``CaptionLabelKey``, then reads the
    page from the embedded inventory its ``pdf_table_inventory_sha256`` names.
    Both facts are read from the STORED envelope, never a module constant, so
    a previously stored envelope renders with ITS OWN label and page even if
    ``TARGET_TABLE_KEY``/``TARGET_TABLE_FOOTPRINT`` are edited later -- the
    drift the render docstrings promise is impossible.

    Falls back to ``"unknown"`` for both, exactly like the ``raw_sha256``
    fallback in the renders, for an envelope with no PDF table-cell citation to
    read either fact from. The PAGE alone degrades to ``"unknown"`` -- the label
    stays the one actually cited -- when a cited inventory's ``canonical_json``
    cannot be read for a page. ``EmbeddedTableInventory``'s T1 validator makes
    that impossible for any inventory that passed construction (it rejects
    unparseable, ``footprint``-less, or wrong-shaped canonical JSON outright), so
    it can only arise from a validation-bypassed or partially constructed
    envelope; the render path stays total rather than raising an untyped
    traceback into a caller that does not expect one.
    """
    for _, ref in iter_source_refs(envelope):
        locator = ref.locator
        if not (isinstance(locator, TableCellLocator) and isinstance(locator.table_key, CaptionLabelKey)):
            continue
        label = locator.table_key.label
        page: object = "unknown"
        if isinstance(locator.pdf_table_inventory_sha256, str):
            for inventory in envelope.table_inventories:
                if inventory.inventory_sha256 == locator.pdf_table_inventory_sha256:
                    try:
                        page = json.loads(inventory.canonical_json)["footprint"]["page"]
                    except json.JSONDecodeError, KeyError, TypeError:
                        # Unreachable for a validated inventory -- T1 above rejects
                        # every malformation this expression could trip on -- so this
                        # only fires for a validation-bypassed envelope. Degrade the
                        # page to "unknown" (the same honest fallback as an
                        # unresolvable citation) rather than crash the render path,
                        # which the project's fail-closed contract forbids.
                        page = "unknown"
                    break
        return label, str(page)
    return "unknown", "unknown"
