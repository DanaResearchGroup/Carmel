"""Recorded secondary-score revisions and frozen document integrity."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from carmel.benchmarks.contract import FrozenModel


class Revision(FrozenModel):
    number: Literal[2]
    previous_contract_sha256: str
    reason: str


class IDTTarget(FrozenModel):
    temperature_min_k: Literal[600]
    temperature_max_k: Literal[1000]
    inclusive: Literal[True]
    frozen_points: Literal[124]
    metric: Literal["mean_abs_log10_delay_ratio"]


class JSRWindowTarget(FrozenModel):
    """A target-chemistry sub-score over a frozen temperature window of a JSR holdout.

    Every holdout row whose temperature lies in the inclusive window is scored, over every
    measured species, with the same species-balanced JSR metric as the aggregate score. The
    row and scalar counts are frozen so a reader or archive change that moves rows in or out of
    the window is refused instead of silently changing the sub-score's population.
    """

    temperature_min_k: Literal[600]
    temperature_max_k: Literal[900]
    inclusive: Literal[True]
    frozen_rows: Literal[31]
    frozen_scalars: Literal[434]
    metric: Literal["existing_species_balanced_jsr_metric"]


class EtherTarget(FrozenModel):
    species: dict[str, str]
    metric: Literal["existing_species_balanced_jsr_metric"]


# Canonical full-document digests: changing a contract requires a reviewed revision
# record AND an explicit update here. File formatting is not scientific content.
FROZEN_REVISIONS = {
    "n-heptane-low-t": "f9389f365e688a52815abe594d059c44bdb3438df7dbf98e387d7c1bd968520c",
    "n-pentane-low-t": "e83ea400fe20335c4d3661d03191c51212f5ad3adb139e8640ce13ed99848605",
    "iso-pentane-low-t": "6cebf1232e9b33f1d65ae3f0bc4232415aa52742c76a368312cb6b6158ebf6ea",
}


def verify_revision(data: bytes | str) -> None:
    document: dict[str, Any] = json.loads(data)
    digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if FROZEN_REVISIONS.get(document["case"]) != digest:
        raise ValueError("UNRECORDED CONTRACT CHANGE: a reviewed versioned revision is required")
