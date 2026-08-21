# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Private generic kernel shared by the typed envelope bridges.

There are two content-addressed envelope types --
:class:`~carmel.schemas.datasets.DatasetEnvelope` and
:class:`~carmel.schemas.datasets.ConditionSetEnvelope` -- and both need the same
store/load treatment: project to the identity payload, refuse anything the
matching loader could never read back, write through the schema-blind store,
and rehydrate through the SAME class on the way out.

Three shapes were available and two of them are traps:

1. **One public generic function taking the envelope class as an argument.**
   Rejected. A public ``cls=`` is precisely the hole that lets a caller hand in
   the wrong class and get a silently valid parse of the wrong type -- the same
   hole that kept ``ConditionSetEnvelope`` from inheriting from
   ``DatasetEnvelope`` in the first place (see that class's docstring: two
   hand-written projections, and a subclass silently inheriting the parent's
   would address two DIFFERENT payloads identically in a write-once store).
2. **Two fully duplicated concrete bridges.** Rejected. The pre-write refusal
   and the load path have to agree exactly -- the class used to refuse a write
   must be the class used to read it back -- and two copies of that agreement
   drift silently. Silent drift here burns addresses permanently.
3. **Public concrete bridges over a PRIVATE generic kernel** -- this module.
   The class, the expected ``envelope_type``, and the store directory all come
   from one :class:`TypedEnvelopeSpec` object, and the SAME spec object feeds
   both the pre-write refusal and the load. A caller cannot pass a class
   because a caller never touches a spec; a spec cannot disagree with itself
   because there is only one of it.

This module is private (leading underscore, per ``carmel.adapters._launcher``)
because nothing outside ``carmel.services.dataset_bridge`` and
``carmel.services.condition_set_bridge`` may reach the generic form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

from carmel.schemas.datasets import _ENVELOPE_TYPE_KEY, DatasetEnvelopeParseError
from carmel.services.dataset_store import (
    StoredDataset,
    canonical_json_bytes,
    load_dataset,
    store_dataset,
)

__all__ = [
    "TypedEnvelopeSpec",
    "load_typed_envelope",
    "store_typed_envelope",
]


class AddressableEnvelope(Protocol):
    """What the kernel needs of an envelope: a projection and its inverse.

    Deliberately structural rather than a base class, for the same reason the
    two envelopes share their provenance validators by CALL and not by
    inheritance: ``identity_payload`` is a hand-written projection per type, and
    anything that could let one type inherit another's projection is a route to
    addressing two different payloads identically.
    """

    def identity_payload(self) -> dict[str, Any]:
        """Project to the plain dict whose canonical bytes define identity."""
        ...

    @classmethod
    def from_identity_payload(cls, payload: dict[str, Any]) -> Self:
        """Rehydrate from a payload, proving the round trip before returning."""
        ...


@dataclass(frozen=True)
class TypedEnvelopeSpec[E: AddressableEnvelope]:
    """Everything that distinguishes one envelope type's bridge from another's.

    Frozen because a mutable spec would reintroduce, at runtime, exactly the
    caller-supplied-class hole the design exists to close.
    """

    envelope_cls: type[E]
    """The ONE class used to refuse a write and to rehydrate a read."""

    envelope_type: str
    """The discriminator this type's payloads must declare. Sourced from the
    schema module's own constants, never a literal written here -- a literal
    could drift from the projection and the drift would be silent."""

    store_dir: str
    """Which store directory this type is enumerated in."""

    unstorable_error: type[Exception]
    """Raised when a write is refused. Distinct per type, so an ``except``
    written for one type's rejected write can never swallow the other's."""

    store_function_name: str
    """The public entry point a caller should have used, named in refusals so a
    caller who reached the wrong door is told which one is right."""


def store_typed_envelope[E: AddressableEnvelope](spec: TypedEnvelopeSpec[E], root: Path, envelope: E) -> StoredDataset:
    """Project, refuse if unreadable, then persist through the schema-blind store.

    Two refusals run here, in this order, and they are NOT redundant:

    1. **The declared type must match the spec's.** A payload declaring
       ``envelope_type="condition_set"`` reaching the dataset bridge is a
       caller routing error, and it is caught HERE, by design of the bridge --
       not left to the parser, which would also refuse it but only as a side
       effect of validation, and whose message names a parser rather than the
       function the caller should have called instead.
    2. **The payload must rehydrate through this spec's class.** A typed
       envelope in hand is not proof it was ever validated:
       ``model_construct`` builds one with every validator skipped, and this
       store is write-once and IMMUTABLE, so an unreadable write is forever --
       the address is burned and stays burned. The check runs at the only
       moment it is worth anything (before the bytes become permanent) and
       checks the thing that matters: not "is this object well-formed" in the
       abstract, but "will the reader that has to live with these bytes accept
       them". It validates the PAYLOAD, not the object, because the payload is
       what gets written.

       Which bounds what it can see: a VIOLATION this check would have caught
       is only caught if the projection EMITS the field that violates it. One
       field makes the boundary concrete. ``SourceNode.crop_region`` is
       projected CONDITIONALLY (see
       ``carmel.schemas.datasets._CONDITIONALLY_PROJECTED_FIELDS``), and the
       condition is read off the field's VALUE rather than off the node's
       ``kind`` (``_addresses_a_crop_region``) -- which is what keeps this
       refusal working for the case. A ``PAPER_PDF`` carrying a real ``BBox``
       is illegal under I7, so it is reachable only through
       ``model_construct`` or ``object.__setattr__``: precisely the
       validator-skipped object named above. It still projects that region,
       so the payload still fails to parse and the write is still refused.
       Keyed on ``kind`` instead, the projection would drop the stray region,
       the tampered node would address byte-identically to a clean one, and
       this check would pass -- nothing untrue in the store (those bytes
       faithfully describe a legal envelope), but a producer bug gone
       unremarked.

       Only the half of that tamper which puts a REGION where none belongs,
       though: the same ``PAPER_PDF`` tampered to hold
       ``Absent(reason=NOT_EXTRACTED_YET)`` -- equally illegal under I7 --
       projects nothing either way, parses back as
       ``Absent(NOT_APPLICABLE)``, and stores at the clean envelope's own
       address, silently. Safe only because I7 admits exactly one reason
       there today, so the value the parse invents is the only one a legal
       node could have held. A field that ever becomes conditional on
       something other than its own value narrows this check by that much,
       silently, and belongs in this paragraph.

    No part of the store's contract (canonical JSON, content addressing, atomic
    write-once semantics) is bypassed or weakened here.
    """
    payload = envelope.identity_payload()

    declared = payload.get(_ENVELOPE_TYPE_KEY)
    if declared != spec.envelope_type:
        raise spec.unstorable_error(
            f"refusing to store a payload declaring {_ENVELOPE_TYPE_KEY}={declared!r} through "
            f"{spec.store_function_name}, which stores {spec.envelope_type!r} payloads only -- "
            "one envelope type must never be written through another's door, because the store "
            "is write-once and the wrong door also means the wrong directory and the wrong "
            "reader"
        )

    try:
        parsed = spec.envelope_cls.from_identity_payload(payload)
    except DatasetEnvelopeParseError as exc:
        raise spec.unstorable_error(
            f"refusing to store a {spec.envelope_cls.__name__} that its own loader could never "
            "read back -- the store is write-once and immutable, so writing it would burn an "
            f"address on unreadable bytes: {exc}"
        ) from exc

    # Third refusal: the rehydrated envelope must project back to the SAME
    # bytes. Both envelope types check this inside `from_identity_payload`
    # already, so this looks redundant -- and it is, for exactly the two
    # classes that exist today. It is here because `AddressableEnvelope` is a
    # `Protocol`, which constrains nothing at runtime: a future third envelope
    # type that satisfies it structurally while ACCEPTING-AND-NORMALIZING a
    # payload (dropping an unknown key, coercing a value) would round-trip to
    # different bytes, and the payload written is the one that was projected
    # BEFORE the parse. The stored address would then belong to bytes no
    # subsequent read could reproduce, permanently, in a write-once store.
    # A static check nobody runs in production is not a defence for that.
    reprojected = canonical_json_bytes(parsed.identity_payload())
    if reprojected != canonical_json_bytes(payload):
        raise spec.unstorable_error(
            f"refusing to store a {spec.envelope_cls.__name__} whose payload does not survive its "
            "own round trip: rehydrating it and projecting it again produced DIFFERENT canonical "
            "bytes, so the address computed here would not be the address a reader reproduces -- "
            "in a write-once store that is an unrecoverable mismatch, not a warning"
        )

    return store_dataset(root, payload, store_dir=spec.store_dir)


def load_typed_envelope[E: AddressableEnvelope](spec: TypedEnvelopeSpec[E], root: Path, sha256: str) -> E:
    """Load from this spec's directory and rehydrate through this spec's class.

    ``load_dataset`` verifies the bytes against the address; the class's own
    ``from_identity_payload`` then refuses any payload whose declared
    ``envelope_type`` is not its own, so a payload that reached this directory
    by some route other than :func:`store_typed_envelope` (a hand-copied file,
    a restored backup, a direct call to the schema-blind store) is refused
    rather than reinterpreted as the wrong type.
    """
    payload = load_dataset(root, sha256, store_dir=spec.store_dir)
    return spec.envelope_cls.from_identity_payload(payload)
