# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Typed bridge between :class:`~carmel.schemas.datasets.DatasetEnvelope` and
the content-addressed store in :mod:`carmel.services.dataset_store`.

``dataset_store`` is deliberately schema-blind: it hashes, stores, and loads
whatever plain ``dict`` a caller hands it, and explicitly does NOT know about
(or import) :mod:`carmel.schemas.datasets` -- see
:func:`carmel.services.dataset_store.compute_dataset_sha`'s docstring for why
(an unrelated change to a model's field shape/order/defaults must never be
able to silently rename the address of every already-stored dataset). These
two wrappers -- :func:`store_dataset_envelope` and
:func:`load_dataset_envelope` -- live in their own module, separate from
``dataset_store.py``, specifically so that contract is never at risk of
erosion: nothing in ``dataset_store.py`` needs to change, or ever import this
module's schema-aware code, for a typed caller to get a typed round trip.
This module is the one place allowed to know about both sides at once.
"""

from __future__ import annotations

from pathlib import Path

from carmel.schemas.datasets import DatasetEnvelope
from carmel.services.dataset_store import StoredDataset, load_dataset, store_dataset

__all__ = [
    "load_dataset_envelope",
    "store_dataset_envelope",
]


def store_dataset_envelope(root: Path, envelope: DatasetEnvelope) -> StoredDataset:
    """Store ``envelope`` under ``root``, addressed by its own identity.

    Thin wrapper over :func:`carmel.services.dataset_store.store_dataset`:
    projects ``envelope`` to the plain dict that defines its identity via
    :meth:`DatasetEnvelope.identity_payload` and hands that dict, unmodified,
    to the schema-blind store. No part of the store's contract (canonical
    JSON, content addressing, atomic write-once semantics) is bypassed or
    weakened here -- this function adds nothing but the projection step.
    """
    return store_dataset(root, envelope.identity_payload())


def load_dataset_envelope(root: Path, sha256: str) -> DatasetEnvelope:
    """Load and reconstruct the :class:`DatasetEnvelope` stored under ``sha256``.

    Thin wrapper over :func:`carmel.services.dataset_store.load_dataset`:
    loads (and, per that function's own contract, verifies) the plain dict
    addressed by ``sha256``, then reconstructs it into a ``DatasetEnvelope``
    via :meth:`DatasetEnvelope.from_identity_payload` -- which itself proves
    the reconstruction round-trips byte-for-byte before returning.
    """
    payload = load_dataset(root, sha256)
    return DatasetEnvelope.from_identity_payload(payload)
