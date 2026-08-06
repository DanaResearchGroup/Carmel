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

from carmel.schemas.datasets import DatasetEnvelope, DatasetEnvelopeParseError
from carmel.services.dataset_store import StoredDataset, load_dataset, store_dataset

__all__ = [
    "UnstorableDatasetEnvelopeError",
    "load_dataset_envelope",
    "store_dataset_envelope",
]


class UnstorableDatasetEnvelopeError(Exception):
    """Raised when an envelope would be written that could never be read back.

    Deliberately NOT a :class:`~carmel.schemas.datasets.DatasetEnvelopeParseError`:
    that one means "the bytes on disk are bad", a fact about the store's past.
    This one means "the bytes were never written", a fact about a refused
    future. Conflating them would let a caller handling read corruption
    silently swallow a rejected write.
    """


def store_dataset_envelope(root: Path, envelope: DatasetEnvelope) -> StoredDataset:
    """Store ``envelope`` under ``root``, addressed by its own identity.

    Thin wrapper over :func:`carmel.services.dataset_store.store_dataset`:
    projects ``envelope`` to the plain dict that defines its identity via
    :meth:`DatasetEnvelope.identity_payload` and hands that dict, unmodified,
    to the schema-blind store. No part of the store's contract (canonical
    JSON, content addressing, atomic write-once semantics) is bypassed or
    weakened here -- this function adds the projection step and one refusal.

    The refusal: the payload is rehydrated through
    :meth:`DatasetEnvelope.from_identity_payload` -- the EXACT call
    :func:`load_dataset_envelope` will later make -- and nothing is written
    if that fails. A typed ``DatasetEnvelope`` in hand is NOT proof it was
    ever validated, because ``model_construct`` builds one with every
    validator skipped, and this store is write-once and IMMUTABLE. Without
    this check an unvalidated envelope stores happily and can then never be
    read back by its only reader: ``series=()`` writes fine and load refuses
    it forever with "Tuple should have at least 1 item". The address is
    burned and, the store being immutable, stays burned.

    So the check runs at the one moment it is worth anything -- before the
    bytes become permanent -- and it checks the thing that actually matters:
    not "is this object well-formed" in the abstract, but "will the reader
    that has to live with these bytes accept them". Validating the payload
    rather than the object is deliberate; the payload is what gets written.
    """
    payload = envelope.identity_payload()
    try:
        DatasetEnvelope.from_identity_payload(payload)
    except DatasetEnvelopeParseError as exc:
        raise UnstorableDatasetEnvelopeError(
            "refusing to store a DatasetEnvelope that load_dataset_envelope could never "
            f"read back -- the store is write-once and immutable, so writing it would burn "
            f"an address on unreadable bytes: {exc}"
        ) from exc
    return store_dataset(root, payload)


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
