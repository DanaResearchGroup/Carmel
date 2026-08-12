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

The store/refuse/load mechanics themselves live in
:mod:`carmel.services._envelope_bridge`, shared with
:mod:`carmel.services.condition_set_bridge`; that module's docstring explains
why the shared form is private and why these two entry points stay concrete.
What this module owns is the SPEC below -- the one place that says which class,
which discriminator, and which directory a dataset uses -- and the fact that
the same spec feeds both the pre-write refusal and the load.
"""

from __future__ import annotations

from pathlib import Path

from carmel.schemas.datasets import _DATASET_ENVELOPE_TYPE, DatasetEnvelope
from carmel.services._envelope_bridge import (
    TypedEnvelopeSpec,
    load_typed_envelope,
    store_typed_envelope,
)
from carmel.services.dataset_store import DATASET_STORE_DIR, StoredDataset

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

    Also deliberately unrelated by inheritance to
    :class:`~carmel.services.condition_set_bridge.UnstorableConditionSetEnvelopeError`:
    if either subclassed the other, an ``except`` written for one type's
    rejected write would silently swallow the other's -- the exact type
    confusion the two separate bridges exist to prevent.
    """


_DATASET_SPEC: TypedEnvelopeSpec[DatasetEnvelope] = TypedEnvelopeSpec(
    envelope_cls=DatasetEnvelope,
    envelope_type=_DATASET_ENVELOPE_TYPE,
    store_dir=DATASET_STORE_DIR,
    unstorable_error=UnstorableDatasetEnvelopeError,
    store_function_name="store_dataset_envelope",
)


def store_dataset_envelope(root: Path, envelope: DatasetEnvelope) -> StoredDataset:
    """Store ``envelope`` under ``root``, addressed by its own identity.

    Refuses, before writing anything, a payload that does not declare itself a
    dataset, and a payload that :func:`load_dataset_envelope` could not later
    read back. See :func:`carmel.services._envelope_bridge.store_typed_envelope`
    for why each refusal is there and why both are checked against the payload
    rather than the object.
    """
    return store_typed_envelope(_DATASET_SPEC, root, envelope)


def load_dataset_envelope(root: Path, sha256: str) -> DatasetEnvelope:
    """Load and reconstruct the :class:`DatasetEnvelope` stored under ``sha256``.

    Loads (and, per :func:`carmel.services.dataset_store.load_dataset`'s own
    contract, verifies) the plain dict addressed by ``sha256`` from the dataset
    directory, then reconstructs it via
    :meth:`DatasetEnvelope.from_identity_payload` -- which proves the
    reconstruction round-trips byte-for-byte, and refuses a payload declaring
    any other envelope type, before returning.
    """
    return load_typed_envelope(_DATASET_SPEC, root, sha256)
