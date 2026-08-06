# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Typed bridge between :class:`~carmel.schemas.datasets.ConditionSetEnvelope`
and the content-addressed store in :mod:`carmel.services.dataset_store`.

The sibling of :mod:`carmel.services.dataset_bridge`, and separate from it for
the same reason ``ConditionSetEnvelope`` is not a subclass of
``DatasetEnvelope``: the two are different payload shapes with different
hand-written projections, and any construct that lets one be mistaken for the
other in a write-once store is a permanent corruption, not a recoverable bug.

Condition sets are stored in their OWN directory
(:data:`~carmel.services.dataset_store.CONDITION_SET_STORE_DIR`), so the
store's "query by scanning files" model keeps working: enumerating datasets
never turns up a condition set that every dataset consumer would then have to
recognise and skip.

The store/refuse/load mechanics are shared with the dataset bridge via
:mod:`carmel.services._envelope_bridge`; what this module owns is the SPEC
below, and the fact that the same spec feeds both the pre-write refusal and the
load.
"""

from __future__ import annotations

from pathlib import Path

from carmel.schemas.datasets import _CONDITION_SET_ENVELOPE_TYPE, ConditionSetEnvelope
from carmel.services._envelope_bridge import (
    TypedEnvelopeSpec,
    load_typed_envelope,
    store_typed_envelope,
)
from carmel.services.dataset_store import CONDITION_SET_STORE_DIR, StoredDataset

__all__ = [
    "UnstorableConditionSetEnvelopeError",
    "load_condition_set_envelope",
    "store_condition_set_envelope",
]


class UnstorableConditionSetEnvelopeError(Exception):
    """Raised when a condition set would be written that could never be read back.

    Deliberately NOT a :class:`~carmel.schemas.datasets.DatasetEnvelopeParseError`
    (that means "the bytes on disk are bad", a fact about the store's past;
    this means "the bytes were never written", a fact about a refused future),
    and deliberately unrelated by inheritance to
    :class:`~carmel.services.dataset_bridge.UnstorableDatasetEnvelopeError` --
    if either subclassed the other, an ``except`` written for one type's
    rejected write would silently swallow the other's.
    """


_CONDITION_SET_SPEC: TypedEnvelopeSpec[ConditionSetEnvelope] = TypedEnvelopeSpec(
    envelope_cls=ConditionSetEnvelope,
    envelope_type=_CONDITION_SET_ENVELOPE_TYPE,
    store_dir=CONDITION_SET_STORE_DIR,
    unstorable_error=UnstorableConditionSetEnvelopeError,
    store_function_name="store_condition_set_envelope",
)


def store_condition_set_envelope(root: Path, envelope: ConditionSetEnvelope) -> StoredDataset:
    """Store ``envelope`` under ``root``, addressed by its own identity.

    Refuses, before writing anything, a payload that does not declare itself a
    condition set, and a payload that :func:`load_condition_set_envelope` could
    not later read back. See
    :func:`carmel.services._envelope_bridge.store_typed_envelope` for why each
    refusal is there.
    """
    return store_typed_envelope(_CONDITION_SET_SPEC, root, envelope)


def load_condition_set_envelope(root: Path, sha256: str) -> ConditionSetEnvelope:
    """Load and reconstruct the :class:`ConditionSetEnvelope` under ``sha256``.

    Reads from the condition-set directory only, and rehydrates through
    :meth:`ConditionSetEnvelope.from_identity_payload` -- which proves the
    round trip byte-for-byte and refuses a payload declaring any other envelope
    type, so a dataset payload that reached this directory by some other route
    is refused rather than reinterpreted.
    """
    return load_typed_envelope(_CONDITION_SET_SPEC, root, sha256)
