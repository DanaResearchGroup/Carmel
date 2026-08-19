"""Persist a cell inventory so an independent run can recompute it and say what differs.

:mod:`carmel.services.pdf_tables` derives a grid inside a caller-drawn box. That derivation
lives for the length of one function call, and its own module docstring says the repair for
an unfalsifiable :class:`~carmel.schemas.datasets.TableCellLocator` is to derive ordinals
from geometry the caller does not control **and to persist enough of that derivation that an
independent replay can recompute it and refuse when it does not reproduce**. This module is
that second half.

**What makes the record worth anything is the box, not the bytes.** Replay re-derives from
the STORED footprint, so it can only ever confirm that this box over this document still
yields this grid. That is circular unless the box itself is falsifiable -- which is why the
edge guards in ``pdf_tables`` came first, and why this module was deliberately not written
before them. It is still not a check that the box is the RIGHT one; nothing here can be.

**Reproduction is compared as canonical bytes, not field by field.** One comparison cannot
silently omit a field the way a hand-written comparison can, and every field that enters the
address enters the comparison by construction. Field-level differences are reported only as
human-readable detail, never as the test.

**Geometry is serialized with :meth:`float.hex`, and this is load-bearing.**
:func:`~carmel.services.dataset_store.canonical_json_bytes` rejects floats outright and
directs callers to :func:`~carmel.services.dataset_store.canonical_decimal`. That function is
the wrong tool here: it routes numeric text through
:func:`~carmel.services.numeric.normalize_numeric_span`, the numeric-EXTRACTION grammar built
for values a paper reported, so a change to what that grammar accepts would stale every
stored coordinate for reasons having nothing to do with geometry. ``float.hex()`` is exact,
total over finite floats, round-trips through :meth:`float.fromhex` bit-for-bit, and couples
to nothing. The cost is that a coordinate is not human-readable in the stored JSON; the cell
TEXT beside it is, and that is what an auditor actually reads.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from carmel.services import pdf_tables
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.pdf_fragments import TextFragment, extract_fragments
from carmel.services.pdf_tables import (
    CellInventory,
    ClaimedFootprint,
    InventoryRefusalReason,
    build_inventory,
)
from carmel.services.semantic_deps import compute_dependency_sha, fragment_geometry_identity

__all__ = [
    "INVENTORY_PAYLOAD_VERSION",
    "InventoryVerification",
    "InventoryVerificationStatus",
    "compute_inventory_sha",
    "inventory_code_sha256",
    "inventory_record_bytes",
    "inventory_record_payload",
    "refusal_reasons_of",
    "verify_inventory_record",
]

#: Bumped whenever the payload's SHAPE changes, independently of the code that fills it.
#:
#: A reader that does not know a shape must not guess at it, and a shape change alters every
#: address even when no coordinate moved -- so the version is what lets a verifier say "I
#: cannot read this" instead of "this does not reproduce". Those are different facts.
INVENTORY_PAYLOAD_VERSION = 1

#: The names whose within-module closure defines "the derivation this record came from".
#:
#: Entry points rather than the whole module: :func:`compute_dependency_sha` walks the
#: closure of these names, so a docstring fix elsewhere in ``pdf_tables`` does not stale
#: every stored record, while any change reachable from the derivation does. The measured
#: constants are listed explicitly because they are read as globals -- moving
#: ``COLUMN_VALLEY_PT`` changes every column boundary and must change this sha.
_INVENTORY_ENTRY_POINTS: tuple[str, ...] = (
    "build_inventory",
    "COLUMN_VALLEY_PT",
    "AFFIX_HEIGHT_RATIO",
    "AFFIX_PARENT_MARGIN",
    "InventoryRefusalReason",
)


#: Fields that identify the CODE and ENGINE a record was written by, rather than what the
#: derivation produced. Compared exhaustively by :func:`_identity_drift` and excluded from the
#: byte comparison, so that a record whose grid still reproduces under changed code reports
#: ``REPRODUCED`` with the drift named instead of an indistinguishable ``MISMATCHED``.
#: They remain part of the content ADDRESS: a record written under different code is a
#: different record, even when it says the same thing.
_IDENTITY_FIELDS: frozenset[str] = frozenset({"inventory_code_sha256", "fragment_geometry_sha256", "pypdf_version"})


class InventoryIdentityUnavailable(RuntimeError):
    """Raised when this machine cannot compute the identity of its own derivation code.

    A ``RuntimeError`` subclass rather than a new hierarchy, so a caller that does not care
    still catches it with the handlers it already has.
    """


def _derivation_only(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The payload minus the identity fields: what the derivation actually produced."""
    return {key: value for key, value in payload.items() if key not in _IDENTITY_FIELDS}


class InventoryVerificationStatus(StrEnum):
    """The outcome of checking a stored inventory record against a document.

    Deliberately NOT a three-state enum with one broad ``UNVERIFIABLE`` bucket. Wrong bytes,
    an unreadable payload, an absent engine and a genuine difference are not operationally
    equivalent, and collapsing them is how a consumer learns to read "unverifiable" as
    "fine". Each way of failing to reach a verdict is therefore its own status.
    """

    REPRODUCED = "reproduced"
    """The recomputation ran and its canonical bytes are identical to the stored ones.

    This is the ONLY positive result in this module, and it is positive because something was
    recomputed -- not because no refusal fired. It says the stored grid follows from the
    stored footprint over these bytes. It does NOT say the footprint is the right box, and
    no status here can."""

    MISMATCHED = "mismatched"
    """The recomputation ran and produced something different. The record is not evidence."""

    SOURCE_MISMATCH = "source_mismatch"
    """``data`` is not the document this record is about; its sha256 differs from the stored
    one. An input or tampering fault, not an inability to verify -- nothing was even
    attempted, and reporting it as "could not verify" would let a caller retry forever
    against the wrong file."""

    PAYLOAD_UNREADABLE = "payload_unreadable"
    """The stored payload is malformed, or carries a ``payload_version`` this code does not
    know. A verifier that guessed at an unknown shape would be inventing the very fields it
    is meant to be checking."""

    ENGINE_UNAVAILABLE = "engine_unavailable"
    """The fragment lane could not read the document at all, so no recomputation happened.
    A property of this machine, not of the record."""

    IDENTITY_UNAVAILABLE = "identity_unavailable"
    """This machine cannot say what its own derivation code IS, so no comparison is honest.

    Reachable on a real deployment: a ``.pyc``-only or zipped install has no source for
    :func:`inspect.getsource` to read. Its own status rather than a sentinel identity,
    because a sentinel would make two records written under two DIFFERENT unknown versions
    compare equal -- the conflation the identity exists to prevent."""


@dataclass(frozen=True)
class InventoryVerification:
    """What checking a record against a document established.

    ``identity_moved`` is reported ALONGSIDE the status rather than short-circuiting it. An
    earlier design refused to recompute at all once any code identity had moved, which is
    fail-closed but too coarse: several fragment-lane supersessions explicitly moved no
    coordinate, and staling every stored record on each of them is the kind of treadmill that
    gets bypassed under pressure. So the identities are compared and reported, and the
    recomputation runs anyway. ``REPRODUCED`` with a moved identity is a stronger result than
    a refusal to look -- it says the output survived the change -- and ``MISMATCHED`` with a
    moved identity names the likely cause instead of leaving it to be guessed.
    """

    status: InventoryVerificationStatus
    identity_moved: tuple[str, ...] = ()
    """Which recorded identities differ from this machine's: any of ``inventory_code``,
    ``fragment_geometry``, ``pypdf_version``. Empty means every one of them matches."""

    detail: str = ""

    @property
    def reproduced(self) -> bool:
        """True only for :attr:`InventoryVerificationStatus.REPRODUCED`.

        Derived rather than stored so it cannot disagree with ``status``. Note that it stays
        True when ``identity_moved`` is non-empty: the grid did reproduce, and a caller that
        cares whether it reproduced under the SAME code must read ``identity_moved`` too.
        """
        return self.status is InventoryVerificationStatus.REPRODUCED


def inventory_code_sha256() -> str:
    """The identity of the derivation as it exists on this machine, computed from source.

    Computed live rather than registered in :mod:`carmel.services.semantic_deps` with a
    pinned constant. The registry's supersession machinery exists to carry a HISTORY of what
    each past identity meant; this record needs only "is the code the same as when this was
    written", which the live computation answers exactly and which no one can forget to
    update. If a history of inventory derivations is ever needed, that is the moment to
    register it, not before.

    Raises:
        InventoryIdentityUnavailable: If the derivation's source cannot be read. That is a
            real deployment (a ``.pyc``-only or zipped install, a source-stripped package),
            and it must NOT degrade to a sentinel: two records written under two different
            unknown versions would then compare equal, which is the silent conflation this
            identity exists to prevent. It raises, and the verifier reports it as its own
            status rather than as a difference in the grid.
    """
    try:
        source = inspect.getsource(pdf_tables)
    except (OSError, TypeError) as exc:
        raise InventoryIdentityUnavailable(
            f"cannot read the source of {pdf_tables.__name__} to identify the derivation: {exc}"
        ) from exc
    return compute_dependency_sha(source, _INVENTORY_ENTRY_POINTS)


def _pt(value: float) -> str:
    """Serialize one coordinate exactly.

    Non-finite values are refused rather than serialized. ``float("nan").hex()`` is the
    perfectly valid string ``'nan'``, so without this a ``CellInventory`` built directly with
    a ``nan`` coordinate would round-trip into the store looking like a measurement.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"refusing to record a non-finite coordinate: {value!r}")
    return value.hex()


def _fragment_digest(fragment: TextFragment) -> str:
    """A fragment's identity: everything the derivation reads off it, and nothing else.

    Cell text and an x-extent alone would not pin the members -- two different fragment sets
    can concatenate to the same string across the same span. The digest is what makes the
    cell's COMPOSITION checkable rather than just its rendering.
    """
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "baseline_y": _pt(fragment.baseline_y),
                "font_height": _pt(fragment.font_height),
                "glyph_mapping": fragment.glyph_mapping.value,
                "page": fragment.page,
                "rotated": fragment.rotated,
                "text": fragment.text,
                "x_end": _pt(fragment.x_end),
                "x_start": _pt(fragment.x_start),
            }
        )
    ).hexdigest()


def inventory_record_payload(inventory: CellInventory, *, raw_sha256: str) -> dict[str, Any]:
    """The full stored form of one inventory, ready for :func:`canonical_json_bytes`.

    Every field here is identity: there is no diagnostics half. A timestamp or a hostname
    would have to be excluded from the address and would then be a field the verifier does
    not check, which is a place for a wrong value to live unnoticed. Nothing needed one, so
    nothing was added.

    ``pypdf_version`` is identity rather than diagnostics because the geometry IS the
    evidence: an engine that changed baseline semantics or CTM composition could keep every
    attribute name intact while returning different numbers.

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed sha256 digest. It is the record's
            only link to the document and nothing downstream re-derives it, so a typo or a
            truncated digest would mint a record that can never match any bytes -- and would
            report ``SOURCE_MISMATCH`` forever, blaming the caller's file for the record's
            own defect.
    """
    if len(raw_sha256) != 64 or any(c not in "0123456789abcdef" for c in raw_sha256):
        raise ValueError(f"raw_sha256 must be 64 lowercase hex characters, got {raw_sha256!r}")
    footprint = inventory.footprint
    return {
        "column_bounds": [[_pt(left), _pt(right)] for left, right in inventory.column_bounds],
        "cells": [
            {
                "col": cell.col,
                "member_digests": [_fragment_digest(f) for f in cell.members],
                "row": cell.row,
                "text": cell.text,
                "x_end": _pt(cell.x_end),
                "x_start": _pt(cell.x_start),
            }
            for cell in inventory.cells
        ],
        "footprint": {
            "caption_baseline_y": _pt(footprint.caption_baseline_y),
            "caption_text": footprint.caption_text,
            "caption_x_start": _pt(footprint.caption_x_start),
            "page": footprint.page,
            "x_end": _pt(footprint.x_end),
            "x_start": _pt(footprint.x_start),
            "y_bottom": _pt(footprint.y_bottom),
            "y_top": _pt(footprint.y_top),
        },
        "fragment_geometry_sha256": fragment_geometry_identity().composite_sha256,
        "inventory_code_sha256": inventory_code_sha256(),
        "payload_version": INVENTORY_PAYLOAD_VERSION,
        "pypdf_version": inventory.pypdf_version,
        "raw_sha256": raw_sha256,
        "refusals": [{"detail": r.detail, "reason": r.reason.value} for r in inventory.refusals],
        "rows": [
            {
                "anchor_text": row.anchor_text,
                "anchor_x_start": _pt(row.anchor_x_start),
                "baseline_y": _pt(row.baseline_y),
                "merged_baselines": [_pt(b) for b in row.merged_baselines],
                "ordinal": row.ordinal,
            }
            for row in inventory.rows
        ],
    }


def inventory_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of this record: what a store writes, and what its address is over.

    Exists so those two are one definition rather than two that happen to agree.
    :mod:`carmel.services.pdf_table_store` needs the bytes AND the address; had it
    canonicalized on its own, a later change to what "canonical" means here would
    move the stored bytes while :func:`compute_inventory_sha` kept computing the
    old address, and the divergence would surface as every record failing to hash
    to its own name -- corruption's signature, from a refactor.
    """
    return canonical_json_bytes(dict(payload))


def compute_inventory_sha(payload: Mapping[str, Any]) -> str:
    """This record's content address: the sha256 of its canonical JSON bytes."""
    return hashlib.sha256(inventory_record_bytes(payload)).hexdigest()


def _footprint_from(stored: Mapping[str, Any]) -> ClaimedFootprint:
    return ClaimedFootprint(
        page=int(stored["page"]),
        x_start=float.fromhex(stored["x_start"]),
        x_end=float.fromhex(stored["x_end"]),
        y_top=float.fromhex(stored["y_top"]),
        y_bottom=float.fromhex(stored["y_bottom"]),
        caption_text=str(stored["caption_text"]),
        caption_x_start=float.fromhex(stored["caption_x_start"]),
        caption_baseline_y=float.fromhex(stored["caption_baseline_y"]),
    )


def _identity_drift(payload: Mapping[str, Any], pypdf_version: str) -> tuple[str, ...]:
    moved: list[str] = []
    if payload.get("inventory_code_sha256") != inventory_code_sha256():
        moved.append("inventory_code")
    if payload.get("fragment_geometry_sha256") != fragment_geometry_identity().composite_sha256:
        moved.append("fragment_geometry")
    if payload.get("pypdf_version") != pypdf_version:
        moved.append("pypdf_version")
    return tuple(moved)


def _first_difference(stored: Mapping[str, Any], recomputed: Mapping[str, Any]) -> str:
    """A human-readable pointer at what differs. Never the test -- the bytes are the test."""
    for key in sorted(set(stored) | set(recomputed)):
        if stored.get(key) != recomputed.get(key):
            if key in {"rows", "cells", "column_bounds", "refusals"}:
                was, now = len(stored.get(key) or []), len(recomputed.get(key) or [])
                if was != now:
                    return f"{key}: stored {was}, recomputed {now}"
            return f"{key} differs"
    return "canonical bytes differ with no differing top-level key"


def verify_inventory_record(payload: Mapping[str, Any], data: bytes) -> InventoryVerification:
    """Re-derive the inventory from ``data`` and the STORED footprint, and compare.

    Reads nothing from the payload except the footprint and the identities: the rows,
    columns and cells are recomputed, never trusted. That is what makes the record a claim
    the document can refute rather than a note the record makes about itself.
    """
    try:
        stored_footprint = payload["footprint"]
        version = payload["payload_version"]
        footprint = _footprint_from(stored_footprint)
    except (KeyError, TypeError, ValueError) as exc:
        return InventoryVerification(InventoryVerificationStatus.PAYLOAD_UNREADABLE, detail=f"malformed payload: {exc}")
    if version != INVENTORY_PAYLOAD_VERSION:
        return InventoryVerification(
            InventoryVerificationStatus.PAYLOAD_UNREADABLE,
            detail=f"payload_version {version!r}, this code reads {INVENTORY_PAYLOAD_VERSION}",
        )

    actual_sha = hashlib.sha256(data).hexdigest()
    if payload.get("raw_sha256") != actual_sha:
        return InventoryVerification(
            InventoryVerificationStatus.SOURCE_MISMATCH,
            detail=f"these bytes are {actual_sha}, the record is about {payload.get('raw_sha256')}",
        )

    extraction = extract_fragments(data)
    try:
        moved = _identity_drift(payload, extraction.pypdf_version)
    except InventoryIdentityUnavailable as exc:
        return InventoryVerification(InventoryVerificationStatus.IDENTITY_UNAVAILABLE, detail=str(exc))
    if not extraction.available:
        return InventoryVerification(
            InventoryVerificationStatus.ENGINE_UNAVAILABLE,
            identity_moved=moved,
            detail=f"fragment lane reported {extraction.status.value}",
        )

    try:
        recomputed = inventory_record_payload(build_inventory(extraction, footprint), raw_sha256=actual_sha)
        stored_bytes = canonical_json_bytes(_derivation_only(payload))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return InventoryVerification(
            InventoryVerificationStatus.PAYLOAD_UNREADABLE,
            identity_moved=moved,
            detail=f"payload is not comparable: {exc}",
        )

    # The IDENTITY fields are excluded from this comparison, and the exclusion is the point
    # rather than a loophole: `_identity_drift` compares them exhaustively and its result is
    # reported on every outcome. Leaving them in made REPRODUCED-with-a-moved-identity
    # UNREACHABLE -- any drift changed the bytes, so the answer was always MISMATCHED, and
    # this module's docstring promised a result its code could not produce. Caught by review,
    # not by a test: the test had codified the contradiction instead of exposing it.
    if canonical_json_bytes(_derivation_only(recomputed)) == stored_bytes:
        return InventoryVerification(InventoryVerificationStatus.REPRODUCED, identity_moved=moved)
    return InventoryVerification(
        InventoryVerificationStatus.MISMATCHED,
        identity_moved=moved,
        detail=_first_difference(_derivation_only(payload), _derivation_only(recomputed)),
    )


def refusal_reasons_of(payload: Mapping[str, Any]) -> tuple[InventoryRefusalReason, ...]:
    """The refusals a stored record carries, as enum members.

    A record of a REFUSED inventory is worth storing and is not a failure to store: it is the
    document saying, reproducibly, that this box yields nothing. Callers must not read an
    empty tuple here as an approval -- it means the derivation succeeded, and the positive
    evidence for it is :attr:`InventoryVerificationStatus.REPRODUCED`, never this.
    """
    reasons: Sequence[Any] = payload.get("refusals") or ()
    return tuple(InventoryRefusalReason(entry["reason"]) for entry in reasons)
