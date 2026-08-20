# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""On-disk, content-addressed storage for table-cell inventory records.

:mod:`carmel.services.pdf_table_record` makes an inventory SERIALIZABLE and
VERIFIABLE: it builds the payload, computes its address, and re-derives rows,
columns and cells from raw PDF bytes to decide whether the payload reproduces.
It never touches the filesystem, so until this module existed nothing could hand
it a payload that came from anywhere but the same process that built it -- a
verifier whose only possible input was its own output.

This module is the missing half. Layout::

    <workspace_root>/evidence/literature/<raw_sha>/table_inventories/<inventory_sha>.json

The record is nested UNDER the document it is about rather than living in a flat
top-level store, and that is a fail-closed decision, not tidiness. A flat store
cannot localize a corrupt file: an unparseable record whose own ``raw_sha256``
cannot be read belongs to an unknown document, so a fail-closed scan would have
to poison every document's "no inventory stored here" answer. Nesting confines
that damage to the one document's own directory.

``<inventory_sha>`` is :func:`~carmel.services.pdf_table_record.compute_inventory_sha`
-- the sha256 of the payload's own canonical JSON bytes, which are exactly the
bytes on disk. That makes the address self-authenticating in the narrow sense
that matters here: a file whose bytes do not hash to the name it sits under is
not the record that address names, full stop. It does NOT make the record TRUE.
A payload can be perfectly addressed and still claim a grid the document does
not contain; the only thing that refutes that is
:func:`~carmel.services.pdf_table_record.verify_inventory_record` re-deriving the
inventory from ``raw.bin``. Address authenticates the FILE; verification
authenticates the CLAIM. :func:`verify_stored_inventory` is the one call that
does both.

Deliberately NOT built here, each for a stated reason:

- **No reuse of :func:`carmel.services.dataset_store.store_dataset`**, despite it
  being a general content-addressed JSON store with a ``store_dir`` allowlist and
  every invariant this module re-implements. It injects the reserved
  ``_carmel_decimal_repr_version`` marker into the payload BEFORE hashing, which
  would bind an inventory's content address to ``canonical_decimal``'s rendering
  rules. Inventory geometry is ``float.hex()`` precisely so it does not depend on
  that grammar (see :mod:`carmel.services.pdf_table_record`'s docstring), and a
  bump of that marker -- which ``dataset_store`` documents as intended behaviour
  -- would re-address every stored inventory for a dependency it does not have.
  It would also mint a second address for one payload, competing with the
  ``compute_inventory_sha`` the verifier is written against.
- **No ``meta.json``, and no ``stored_at``.** A timestamp cannot be part of the
  address (two identical stores would collide), so it would be a field the
  verifier does not check -- a place for a wrong value to live unnoticed. Its only
  use would be ordering "which record is newest for this document", and newest is
  not a selection rule here: two inventories under one document usually describe
  two DIFFERENT tables, so a later one does not supersede an earlier one. Storing
  the field would invite exactly the wrong query.
- **No "the inventory for this document" accessor.** There is no such thing, for
  the reason above. :func:`scan_inventory_records` enumerates and reports what it
  could not read; choosing among the results is the caller's claim to make.

Trust model: same as :mod:`carmel.services.extraction_record` -- a trusted,
single-user workspace. Symlinks are refused outright, at the record, at
``raw.bin``, and at both directories above them; containment is checked behind
that. Refusing a symlink for BEING one rather than for where it points is the
part that took a second pass: a link that never leaves the store still lets one
file answer for two addresses, and a directory link still lets one document's
bytes answer for another's, neither of which a containment test can see. Reads
are bounded by reading one byte past the cap rather than by trusting a preceding
``stat``. A hostile same-privilege process racing between a check and the
operation that follows it (TOCTOU) remains out of scope.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from carmel.paths import normalize_path
from carmel.services.dataset_store import (
    _MAX_JSON_DEPTH,
    CanonicalJsonError,
    _raw_bytes_nest_too_deeply,
)
from carmel.services.evidence import artifact_dir
from carmel.services.pdf_table_record import (
    InventoryVerification,
    InventoryVerificationStatus,
    inventory_record_bytes,
    inventory_record_payload,
    refusal_reasons_of,
    verify_inventory_record,
)
from carmel.services.pdf_tables import CellInventory, InventoryRefusalReason

__all__ = [
    "TABLE_INVENTORIES_SUBDIR",
    "InventoryScan",
    "InventoryScanProblem",
    "InventoryStoreError",
    "StoredInventoryOutcome",
    "StoredInventoryVerification",
    "inventory_record_path",
    "load_inventory_record",
    "scan_inventory_records",
    "store_inventory_record",
    "verify_stored_inventory",
]

#: Subdirectory of ``evidence/literature/<raw_sha>/`` that holds inventory records.
TABLE_INVENTORIES_SUBDIR = "table_inventories"

_RAW_NAME = "raw.bin"

#: Mirrors the same five-character regex in :mod:`carmel.services.evidence`,
#: :mod:`carmel.services.dataset_store` and :mod:`carmel.services.extraction_record`
#: -- deliberately duplicated rather than imported, for the reason
#: :mod:`carmel.services.extraction_record` states: four stores share a digest
#: SHAPE and share no layout, and coupling four modules' internals to save five
#: characters is the worse trade.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Hard cap on the size of one ``inventory.json``-shaped record file, checked via
#: ``stat()`` before the bytes are read into memory. A DECISION, not a
#: measurement, but a bounded one: the payload holds one table, and the widest
#: table this lane can produce -- a few thousand cells, each carrying its text
#: plus a member digest per fragment -- lands under a megabyte. 8 MiB leaves an
#: order of magnitude of headroom while still refusing a hand-placed file whose
#: only purpose is to be read into memory.
_MAX_RECORD_BYTES = 8 * 1024 * 1024


class InventoryStoreError(ValueError):
    """A stored record is present but is not what its address says it is.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers still
    catch it, mirroring :class:`carmel.services.extraction_record.ExtractionRecordError`.

    Deliberately never raised for ABSENCE. :func:`load_inventory_record` returns
    ``None`` when nothing is stored at an address and raises this when something
    is -- because for THIS store the address is the sha256 of the file's own
    complete bytes, so a mismatch has exactly one meaning and there is no
    legitimate record that could fail to authenticate. That is a narrower stance
    than :func:`carmel.services.extraction_record._load_meta`'s, which folds
    "does not authenticate" into ``None`` on purpose: there the address is
    computed over a SUBSET of the record's fields, so a record written under a
    different identity shape can legitimately fail to authenticate without being
    corrupt.
    """


def inventory_record_path(workspace_root: Path, raw_sha256: str, inventory_sha256: str) -> Path:
    """Compute (but never create or validate) the path of one inventory record.

    Pure path helper, mirroring :func:`carmel.services.evidence.artifact_dir` and
    :func:`carmel.services.extraction_record.extraction_record_dir`. Callers
    holding a digest from outside this module must go through
    :func:`load_inventory_record` or :func:`verify_stored_inventory`, which
    validate shape and containment before touching disk.
    """
    return artifact_dir(workspace_root, raw_sha256) / TABLE_INVENTORIES_SUBDIR / f"{inventory_sha256}.json"


def _validate_sha(value: str, *, label: str) -> None:
    """Validate ``value`` is a well-formed 64-lowercase-hex sha256 digest.

    ``fullmatch``, never ``match``: ``$`` also matches just before a trailing
    newline, so ``"a" * 64 + "\\n"`` would pass a ``match``-based check and go on
    to be used as a filesystem path component.
    """
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r} (expected 64 lowercase hex characters)")


def _assert_contained(workspace_root: Path, path: Path) -> Path:
    """Resolve ``path`` and confirm it stays under the resolved ``workspace_root``."""
    resolved_root = normalize_path(workspace_root)
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"refusing to touch a path outside workspace root: {resolved_path} not under {resolved_root}")
    return resolved_path


def _records_dir(workspace_root: Path, raw_sha256: str) -> Path:
    """Validate ``raw_sha256`` and resolve the containment-checked ``table_inventories/``."""
    _validate_sha(raw_sha256, label="raw_sha256")
    root = normalize_path(workspace_root)
    return _unaliased_dir(root, artifact_dir(root, raw_sha256) / TABLE_INVENTORIES_SUBDIR, label="table_inventories")


def _unaliased_dir(root: Path, literal: Path, *, label: str) -> Path:
    """Containment-check ``literal``, and require it to be its own resolved self.

    Containment alone is not enough, and the gap is a document-ALIASING one rather
    than a traversal one. If ``evidence/literature/<rawA>`` -- or its
    ``table_inventories/`` -- is a symlink to document B's directory somewhere else
    inside the workspace, the resolved path is perfectly contained, so
    :func:`_assert_contained` accepts it and every store and scan for A then
    silently operates in B's directory. One document would answer for another with
    no check anywhere able to see it.

    Requiring ``resolve()`` to be a fixed point closes that: it holds exactly when
    no component below ``root`` is a symlink. It is also total on an absent path,
    since a non-strict ``resolve`` of one that does not exist returns itself -- so
    "nothing is stored here yet" still reads as absence rather than as tampering.
    """
    resolved = _assert_contained(root, literal)
    if resolved != literal:
        raise InventoryStoreError(
            f"{label} for this document is a symlink to {resolved}, not its own directory {literal}: "
            "one document's store must never resolve into another's, even inside the workspace"
        )
    return resolved


def store_inventory_record(workspace_root: Path, inventory: CellInventory, *, raw_sha256: str) -> str:
    """Content-address and durably persist one inventory, complete or REFUSED.

    A refused inventory is stored exactly like a complete one, and that is the
    point: today no table in the corpus yields a complete inventory, so a store
    that only accepted complete ones would be a store with nothing in it. What
    must never happen is a refusal READING as evidence -- see
    :attr:`StoredInventoryVerification.usable_as_table_evidence`, which is the
    only place the "reproduced AND unrefused" conjunction is written down.

    Publishes via a private temp file plus an exclusive ``os.link``, mirroring
    :func:`carmel.services.dataset_store.store_dataset`: ``os.link`` fails
    atomically with ``FileExistsError`` rather than replacing, so a concurrent
    writer targeting the same address cannot silently overwrite an append-only
    record. A writer that loses the race compares the on-disk bytes with the ones
    it was about to publish:

    - Equal -> no rewrite, the existing address is returned (idempotent re-store).
    - Not equal -> raises. Two payloads that canonicalize differently but hash
      the same would be a genuine sha256 collision; far more likely, this is
      on-disk corruption. Either way, overwriting would destroy the auditability
      this store exists to provide.

    Args:
        workspace_root: Root of the campaign workspace.
        inventory: The derived inventory to persist.
        raw_sha256: sha256 of the PDF bytes the inventory was derived from. Not
            re-derived from anything on disk and deliberately not required to
            already have a stored artifact -- the two stores stay decoupled, as
            they are in :func:`carmel.services.extraction_record.store_extraction_record`.

    Returns:
        The record's content address (also its filename stem).

    Raises:
        ValueError: If ``raw_sha256`` is malformed, or the destination resolves
            outside the workspace root.
        InventoryStoreError: If a different file already occupies the address.
        CanonicalJsonError: If the payload is not representable as canonical JSON.
        OSError: If publishing fails for any reason other than an existing
            occupant (disk full, permission denied).
    """
    payload = inventory_record_payload(inventory, raw_sha256=raw_sha256)
    # Both the bytes and the address come from `inventory_record_bytes`, so the file
    # written and the name it is written under cannot drift apart under a refactor.
    canonical_bytes = inventory_record_bytes(payload)
    # The cap belongs on BOTH sides or it is worse than useless. Enforced only on
    # read, it would let this function publish a record that `load_inventory_record`
    # then refuses forever -- and since records are append-only, that is unreachable
    # garbage at a permanent address rather than a failure someone can clean up.
    if len(canonical_bytes) > _MAX_RECORD_BYTES:
        raise InventoryStoreError(
            f"refusing to store a {len(canonical_bytes)}-byte inventory record, over the "
            f"{_MAX_RECORD_BYTES}-byte cap: it would be written to an append-only address that could "
            "never be read back"
        )
    inventory_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    records_dir = _records_dir(workspace_root, raw_sha256)
    # Contained by construction, for the reason `load_inventory_record` spells out:
    # `records_dir` is checked, and a fresh sha256 hex digest cannot traverse.
    path = records_dir / f"{inventory_sha256}.json"
    records_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=records_dir, prefix=f".{inventory_sha256}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            # `path.read_bytes()` alone would FOLLOW a symlink sitting at the
            # destination, so a link aimed at any file holding these bytes would make
            # this call report an idempotent success while no regular record was ever
            # published -- and the address would stay occupied by the link. The
            # occupant has to survive the same check a reader applies before its bytes
            # are worth comparing.
            occupant = _resolved_regular_file(path, within=records_dir, label=f"inventory record {inventory_sha256}")
            if occupant is not None and occupant.read_bytes() == canonical_bytes:
                return inventory_sha256
            raise InventoryStoreError(
                f"inventory record collision or corruption at {path}: the on-disk bytes are not the "
                f"canonical bytes for inventory_sha256={inventory_sha256!r}; refusing to overwrite an "
                "append-only record"
            ) from None
    finally:
        tmp_path.unlink(missing_ok=True)

    dir_fd = os.open(records_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return inventory_sha256


def _resolved_regular_file(path: Path, *, within: Path, label: str) -> Path | None:
    """Refuse a symlink outright, then require a regular file at ``path``.

    ``None`` means the file is simply not there. Everything else raises.

    A symlink is refused because it IS one, never because of where it points, and
    that is the correction round 157 forced. Refusing only an ESCAPE leaves the
    aliasing case wide open: ``A.json -> B.json`` inside the very same directory
    never leaves ``within``, so a containment test cannot see it, yet it lets one
    file answer for two addresses -- and B's bytes hash to B, so an attacker only
    has to name the link A for the digest check to be irrelevant. The same link on
    ``raw.bin`` would let one document's bytes stand in for another's. There is no
    legitimate reason for a record or a raw artifact to be a link, so the honest
    rule is the blunt one, and it is the rule this module's trust model already
    claimed.

    The containment test is kept behind it for a symlink higher in the chain, and
    ``OSError`` is converted rather than allowed to escape: a symlink loop or an
    unreadable directory is a corrupt store, and a store state must reach a verdict
    instead of raising through one.
    """
    if path.is_symlink():
        raise InventoryStoreError(
            f"{label} is a symlink, not a file at its own address: a link lets one file answer for two "
            "addresses even when it never leaves the store"
        )
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError, NotADirectoryError:
        return None
    except OSError as exc:
        raise InventoryStoreError(f"{label} could not be resolved: {exc}") from exc
    if not resolved.is_relative_to(within):
        raise InventoryStoreError(
            f"{label} resolves outside its own directory (symlink escape): {path} resolves to {resolved}, "
            f"which is not under {within}"
        )
    if not resolved.is_file():
        raise InventoryStoreError(f"{label} does not resolve to a regular file: {resolved}")
    return resolved


def _bounded_read(path: Path, *, cap: int, label: str) -> bytes:
    """Read at most ``cap`` bytes, and refuse rather than return a truncated file.

    Reads ``cap + 1`` and checks the length, instead of ``stat()``-then-read: the
    size a ``stat`` reports is not the size the following read returns, so a
    stat-based cap states a memory bound it does not actually hold. This one holds
    by construction, and the extra byte is what distinguishes "exactly at the cap"
    from "over it" without a second syscall.
    """
    with path.open("rb") as handle:
        data = handle.read(cap + 1)
    if len(data) > cap:
        raise InventoryStoreError(f"{label} is larger than the {cap}-byte cap; refusing to read it into memory")
    return data


def load_inventory_record(workspace_root: Path, raw_sha256: str, inventory_sha256: str) -> dict[str, Any] | None:
    """Load the payload stored at one address, or ``None`` if nothing is stored there.

    Every check below is on the FILE, never on the claim it makes. In order: the
    resolved path stays inside the record directory and is a regular file; its
    size is under :data:`_MAX_RECORD_BYTES`; its bytes hash to the address they
    sit under; the bytes do not nest JSON containers too deeply to parse safely
    (a pre-parse byte scan, borrowed from :mod:`carmel.services.dataset_store`,
    because a hand-placed file can blow the interpreter's stack inside
    ``json.loads`` before any validator runs); the parse is a JSON object; and
    re-canonicalizing that object reproduces the bytes exactly, so a file that
    satisfies the digest without being the encoding this store guarantees is
    refused rather than returned as merely self-consistent.

    One check is on the claim, and it is the one the layout makes possible: the
    payload's own ``raw_sha256`` must equal the directory it was found under. A
    record that says it is about document B while sitting under document A has
    been moved or forged, and returning ``None`` for it would let a corpus pass
    believe A has no stored record at all.

    Returns:
        The payload dict, or ``None`` when no file exists at that address.

    Raises:
        ValueError: If either digest is malformed, or the path resolves outside
            the workspace root.
        InventoryStoreError: If a file exists at the address but is not the
            record that address names, by any of the checks above.
    """
    _validate_sha(inventory_sha256, label="inventory_sha256")
    records_dir = _records_dir(workspace_root, raw_sha256)
    # The leaf is deliberately NOT run through `_assert_contained`. Both digests are
    # already fullmatch-validated, so `records_dir / <64-hex>.json` cannot traverse
    # anywhere by construction, and `records_dir` itself was contained above. What
    # `_assert_contained` WOULD add here is a resolve() that follows a symlink at the
    # leaf and then reports it as a workspace escape -- a bare ValueError out of
    # `verify_stored_inventory`, for what is unambiguously a corrupt store rather than
    # a caller error. Leaving the leaf to `_resolved_regular_file` gives every symlink
    # one answer, whether it points outside the workspace or merely outside this
    # document's own directory.
    path = records_dir / f"{inventory_sha256}.json"

    resolved = _resolved_regular_file(path, within=records_dir, label=f"inventory record {inventory_sha256}")
    if resolved is None:
        return None

    raw_bytes = _bounded_read(resolved, cap=_MAX_RECORD_BYTES, label=f"inventory record at {path}")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != inventory_sha256:
        raise InventoryStoreError(
            f"inventory record at {path} does not hash to its own address: recomputed {digest!r} != "
            f"{inventory_sha256!r}"
        )
    if _raw_bytes_nest_too_deeply(raw_bytes):
        raise InventoryStoreError(
            f"inventory record at {path} nests JSON containers more than {_MAX_JSON_DEPTH} deep; rejected "
            "before parsing rather than risking a bare RecursionError inside json.loads"
        )
    try:
        parsed: Any = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InventoryStoreError(f"inventory record at {path} hashes correctly but is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise InventoryStoreError(
            f"inventory record at {path} hashes correctly but parses as a {type(parsed).__name__}, not a JSON object"
        )
    try:
        recanonical = inventory_record_bytes(parsed)
    except CanonicalJsonError as exc:
        raise InventoryStoreError(
            f"inventory record at {path} hashes correctly but cannot be re-canonicalized: {exc}"
        ) from exc
    if recanonical != raw_bytes:
        raise InventoryStoreError(
            f"inventory record at {path} hashes correctly but is not in canonical form; refusing to return "
            "content that is merely self-consistent"
        )
    stored_raw = parsed.get("raw_sha256")
    if stored_raw != raw_sha256:
        raise InventoryStoreError(
            f"inventory record at {path} is about raw_sha256={stored_raw!r} but is stored under "
            f"{raw_sha256!r}: a moved or forged record, never an absent one"
        )
    return parsed


@dataclass(frozen=True)
class InventoryScanProblem:
    """One thing :func:`scan_inventory_records` could not read.

    ``address`` is ``None`` when the problem is the ``table_inventories/``
    directory itself rather than any record inside it -- a typed distinction
    rather than an empty-string convention, because "this document's whole store
    is unreadable" and "one record in it is corrupt" support different decisions.
    """

    address: str | None
    detail: str


@dataclass(frozen=True)
class InventoryScan:
    """What is stored for one document, and what could not be read.

    ``addresses`` alone is a lossy answer -- it cannot distinguish "nothing is
    stored" from "something is stored and unreadable", and that collapse is
    exactly how an unverifiable record gets reported as an absent one. Callers
    that need a fail-closed answer must check
    :attr:`every_entry_could_be_a_record` first -- and must still put each
    address through :func:`verify_stored_inventory`, because being listed here
    means only that a name and a file type were plausible.
    """

    addresses: tuple[str, ...]
    problems: tuple[InventoryScanProblem, ...]

    @property
    def every_entry_could_be_a_record(self) -> bool:
        """Whether every entry is shaped like a record. NOT that any of them can be read.

        Named for what it actually checks, after round 157 caught the previous
        name -- ``complete`` -- promising far more. A scan reads no content, so a
        corrupt file at a perfectly sha-shaped name is listed as an ``address``
        and leaves this True. A producer reading ``complete`` as "every record
        here is readable" would have been wrong in exactly the case that matters,
        and the name is the only place that misreading can be stopped: making the
        scan verify what it lists is the whole-corpus read this deliberately is
        not.
        """
        return not self.problems


def scan_inventory_records(workspace_root: Path, raw_sha256: str) -> InventoryScan:
    """Enumerate the record addresses stored for one document, reporting what did not read.

    Names beginning with ``.`` are skipped as this module's own temp files: a
    ``.``-prefixed name can never be a 64-hex address, so skipping them cannot
    hide a record. Anything else that is not ``<64-hex>.json`` is reported as a
    problem rather than skipped -- an unexplained file in a store directory is a
    fact about the store, not debris to swallow.

    Entries are judged by NAME and by TYPE, never by content. The type check is
    what stops a sha-shaped directory, FIFO or symlink being reported as an
    address that a later load would have to reject; it costs one ``lstat`` per
    entry and it does not make this a verification. Nothing here is opened or
    hashed, so every address it returns still has to survive
    :func:`load_inventory_record` and :func:`verify_stored_inventory`.

    Raises:
        ValueError: If ``raw_sha256`` is malformed or the directory resolves
            outside the workspace root. A malformed digest is a fact about the
            CALLER -- there is no document to refuse on behalf of -- so it stays
            an exception rather than becoming a per-document problem entry.
    """
    records_dir = _records_dir(workspace_root, raw_sha256)
    addresses: list[str] = []
    problems: list[InventoryScanProblem] = []
    try:
        entries = sorted(records_dir.iterdir())
    except FileNotFoundError:
        return InventoryScan(addresses=(), problems=())
    except OSError as exc:
        return InventoryScan(
            addresses=(),
            problems=(InventoryScanProblem(address=None, detail=f"{records_dir} could not be listed: {exc}"),),
        )
    for entry in entries:
        name = entry.name
        if name.startswith("."):
            continue
        stem = name[: -len(".json")] if name.endswith(".json") else ""
        if not _SHA256_RE.fullmatch(stem):
            problems.append(
                InventoryScanProblem(address=None, detail=f"{name} is not a <64-hex>.json inventory record")
            )
            continue
        if entry.is_symlink() or not entry.is_file():
            problems.append(
                InventoryScanProblem(address=stem, detail=f"{name} is not a regular file at its own address")
            )
            continue
        addresses.append(stem)
    return InventoryScan(addresses=tuple(addresses), problems=tuple(problems))


class StoredInventoryOutcome(StrEnum):
    """How far :func:`verify_stored_inventory` got before it had an answer.

    These are STORE states -- facts about the filesystem -- and they are a
    separate vocabulary from
    :class:`carmel.services.pdf_table_record.InventoryVerificationStatus`, which
    describes only what happened once the payload and the document bytes were
    both in hand. Folding a seventh member like ``SOURCE_UNAVAILABLE`` into that
    enum would have put a filesystem state into a pure function that already has
    its bytes and could never produce it.
    """

    RECORD_ABSENT = "record_absent"
    """No record is stored at that address. Not a failure; nothing was claimed."""

    RECORD_CORRUPT = "record_corrupt"
    """A file is there and is not the record that address names. See ``detail``."""

    SOURCE_UNAVAILABLE = "source_unavailable"
    """``raw.bin`` is missing, oversized, not a regular file, or escapes its directory.

    Distinct from ``RECORD_CORRUPT``: the record may be perfectly fine and simply
    unverifiable in THIS workspace, which is a statement about the workspace.
    """

    SOURCE_CORRUPT = "source_corrupt"
    """``raw.bin`` does not hash to its own directory name.

    Distinct from ``SOURCE_UNAVAILABLE``: bytes are present and are provably not
    the document they are filed as, so no verdict about the record can be drawn
    from them and none is attempted.
    """

    DERIVED = "derived"
    """The record and the document bytes were both reached; read ``verification``."""


@dataclass(frozen=True)
class StoredInventoryVerification:
    """The store's verdict on one addressed record.

    Attributes:
        outcome: How far the store got. Only ``DERIVED`` carries a
            ``verification``.
        verification: The pure verifier's result, or ``None`` when the outcome is
            anything but ``DERIVED``.
        reproduced_refusals: The refusals the record carries, populated ONLY when
            the derivation reproduced -- at which point the stored refusals are
            the recomputed ones, because the comparison that returned
            ``REPRODUCED`` was over bytes that include them. Named for the
            condition rather than as a bare ``refusals`` so an empty tuple on a
            record that did NOT reproduce cannot be read as "this record has no
            refusals".
        detail: Human-readable pointer at what went wrong. Never the test.
    """

    outcome: StoredInventoryOutcome
    verification: InventoryVerification | None = None
    reproduced_refusals: tuple[InventoryRefusalReason, ...] = ()
    detail: str = ""

    @property
    def usable_as_table_evidence(self) -> bool:
        """The ONE place the "reproduced AND unrefused" conjunction is written down.

        Both halves are load-bearing and each is useless alone.
        ``REPRODUCED`` is true of a REFUSED record too -- reproducing a refusal
        proves the document really does refuse, which is evidence of a refusal
        and never evidence of a table. An empty ``refusals`` list in a payload
        proves nothing at all on its own, since it is a claim the record makes
        about itself and any file can make it. Only together do they say: this
        box yields a grid, and the document agrees.

        ``identity_moved`` is deliberately NOT a third condition. What this
        answers is "does TODAY's code derive this grid from these bytes", and a
        record that still reproduces after the derivation code or ``pypdf``
        moved underneath it is stronger evidence than one written this morning,
        not weaker -- the grid survived a change that could have moved it.
        Disqualifying drift would discard exactly the records that proved
        themselves most. The drift is reported on
        :attr:`~carmel.services.pdf_table_record.InventoryVerification.identity_moved`
        for a caller who wants to know the record is old; it is not a defect.
        """
        return (
            self.outcome is StoredInventoryOutcome.DERIVED
            and self.verification is not None
            and self.verification.status is InventoryVerificationStatus.REPRODUCED
            and not self.reproduced_refusals
        )


def _read_source_bytes(workspace_root: Path, raw_sha256: str, *, max_bytes: int) -> bytes | StoredInventoryVerification:
    """Read and content-check ``raw.bin``, or return the outcome that stops verification.

    The union return is deliberate over a ``(bytes | None, outcome | None)`` pair:
    the pair has two states that cannot occur and a caller has to assert its way
    past them, while the union makes "bytes, or the reason there are none"
    exhaustive by construction.

    Mirrors :mod:`carmel.services.reextraction`'s discipline: refuse a symlink,
    require a regular file, bound the read, then re-verify the content address from
    the bytes themselves. No sidecar is consulted -- ``verify_artifact(..., deep=True)`` is deliberately
    not used, because it returns False for every one of the real legacy artifacts
    in the live corpus and would make this path unreachable for the corpus it
    exists to serve.
    """
    root = normalize_path(workspace_root)
    try:
        # `raw_sha256` is validated before this is ever called, so an escape or an
        # alias here can only come from a symlink placed on disk -- a statement about
        # the workspace, not about the caller. It refuses like every other unreadable
        # source instead of raising through the verdict. `_unaliased_dir` rather than
        # bare containment: an artifact directory symlinked to ANOTHER document's,
        # still inside the workspace, would silently serve B's bytes as A's.
        # Unreachable from `verify_stored_inventory` as things stand, because
        # `table_inventories/` lives INSIDE the artifact directory and the same link
        # makes `_records_dir` refuse first. Kept regardless, so this function does not
        # rest on its caller having checked: a security check that holds only because of
        # call order stops holding the first time someone reorders the calls.
        artifact = _unaliased_dir(root, artifact_dir(root, raw_sha256), label=f"artifact directory {raw_sha256}")
        raw_path = artifact / _RAW_NAME
        resolved = _resolved_regular_file(raw_path, within=artifact, label=f"raw.bin for {raw_sha256}")
    except ValueError as exc:
        return StoredInventoryVerification(StoredInventoryOutcome.SOURCE_UNAVAILABLE, detail=str(exc))
    if resolved is None:
        return StoredInventoryVerification(
            StoredInventoryOutcome.SOURCE_UNAVAILABLE, detail=f"raw.bin does not exist for {raw_sha256}: {raw_path}"
        )
    try:
        data = _bounded_read(resolved, cap=max_bytes, label=f"raw.bin for {raw_sha256}")
    except InventoryStoreError as exc:
        return StoredInventoryVerification(StoredInventoryOutcome.SOURCE_UNAVAILABLE, detail=str(exc))
    digest = hashlib.sha256(data).hexdigest()
    if digest != raw_sha256:
        return StoredInventoryVerification(
            StoredInventoryOutcome.SOURCE_CORRUPT,
            detail=f"raw.bin does not hash to its own directory name: recomputed {digest!r} != {raw_sha256!r}",
        )
    return data


def verify_stored_inventory(
    workspace_root: Path, raw_sha256: str, inventory_sha256: str, *, max_bytes: int
) -> StoredInventoryVerification:
    """Load one stored record and put it to the document it claims to be about.

    This is the whole point of the store existing: until a payload can come off
    disk, the verifier's only possible input was the same process's own output,
    and a layer that reproduces its own output has proved nothing. Here the
    payload is read from a file, the bytes are read from ``raw.bin``, and the
    rows, columns and cells are RE-DERIVED -- nothing but the footprint and the
    identities is taken from the payload.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: The document the record is filed under.
        inventory_sha256: The record's content address.
        max_bytes: Hard cap on the on-disk ``raw.bin`` size, checked via
            ``stat()`` before any read. Required, with no default: the cap is a
            campaign budget (``config.budget.max_artifact_bytes``), and a default
            here would be this module quietly inventing a policy it does not own.

    Raises:
        ValueError: If either digest is malformed or a path escapes the workspace.
    """
    try:
        payload = load_inventory_record(workspace_root, raw_sha256, inventory_sha256)
    except InventoryStoreError as exc:
        return StoredInventoryVerification(StoredInventoryOutcome.RECORD_CORRUPT, detail=str(exc))
    if payload is None:
        return StoredInventoryVerification(StoredInventoryOutcome.RECORD_ABSENT)

    source = _read_source_bytes(workspace_root, raw_sha256, max_bytes=max_bytes)
    if isinstance(source, StoredInventoryVerification):
        return source

    verification = verify_inventory_record(payload, source)
    refusals: tuple[InventoryRefusalReason, ...] = ()
    if verification.status is InventoryVerificationStatus.REPRODUCED:
        try:
            refusals = refusal_reasons_of(payload)
        except (KeyError, TypeError, ValueError) as exc:
            # Unreachable via a REPRODUCED payload -- the derivation that reproduced
            # built these very entries -- but a reader must never be handed
            # `reproduced_refusals=()` for a payload whose refusals could not be
            # read, because that empty tuple is what `usable_as_table_evidence`
            # tests. Fail closed onto RECORD_CORRUPT instead.
            return StoredInventoryVerification(
                StoredInventoryOutcome.RECORD_CORRUPT,
                detail=f"record reproduced but its refusals could not be read: {exc}",
            )
    return StoredInventoryVerification(
        StoredInventoryOutcome.DERIVED,
        verification=verification,
        reproduced_refusals=refusals,
        detail=verification.detail,
    )
