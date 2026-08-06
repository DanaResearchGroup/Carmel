# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed, append-only storage for extraction records.

Today ``evidence/literature/<raw_sha>/`` (:mod:`carmel.services.evidence`) holds
``raw.bin``, ``text.txt``, ``extracted.json``, and ``meta.json`` together, in one
directory addressed ONLY by the raw bytes' sha256. That conflates two things with
very different lifetimes: ``raw.bin`` is the immutable source (fetched once, never
changes), while ``extracted.json``/``text.txt`` are a DERIVED, VERSIONED,
REPLACEABLE rendering of it -- a function of the raw bytes, Carmel's own extraction
code, and (for PDF-derived extractions) the installed ``pypdf`` version. Because
:func:`carmel.services.evidence.store_artifact` returns the intact existing
artifact on re-store, re-running extraction against an unchanged raw artifact
silently no-ops and a stale extraction persists forever, even after the
extraction code or ``pypdf`` itself has moved on.

This module adds the second, correct address: an extraction record is stored
UNDER its parent raw artifact's directory, at its own content address, layout::

    <workspace_root>/evidence/literature/<raw_sha>/extractions/<extraction_sha>/extracted.json
    <workspace_root>/evidence/literature/<raw_sha>/extractions/<extraction_sha>/text.txt
    <workspace_root>/evidence/literature/<raw_sha>/extractions/<extraction_sha>/meta.json

``<extraction_sha>`` is the sha256 of the CANONICAL JSON bytes (see
:func:`carmel.services.dataset_store.canonical_json_bytes`) of a deliberately
constructed, NAMED-FIELD identity payload -- never delimiter-joined string
concatenation, which under-specifies the boundary between fields and would let two
different (extractor, version) pairs collide onto the same joined string. See
:func:`compute_extraction_sha` for the exact field set. The address is
SELF-AUTHENTICATING: :func:`load_extraction_record` and
:func:`verify_extraction_record` both recompute this sha from a loaded record's OWN
recorded identity fields and require it to equal the directory the record was
loaded from, so a ``meta.json`` that was forged, corrupted, or moved to a
different address's directory does not silently authenticate under that address.

This store is READ, by two consumers, and what each does when a record is missing
differs on purpose:

- The corpus pass (:func:`carmel.services.literature._load_corpus`) PREFERS a
  digest-authenticated current record over the root sidecar, uniformly. Only an
  artifact for which no record was ever stored reaches the root tiers at all; every
  other route to "no usable record" refuses. See :func:`select_current_extraction`.
- Dataset production (:func:`carmel.services.dataset_producer.produce_envelope_from_artifact`)
  REQUIRES one, and refuses outright without it. It used to mirror the root sidecar's
  bytes into a record to satisfy its own binding; that became a laundering path the
  moment the corpus pass started preferring records, because a mirrored record is
  indistinguishable on disk from a genuinely re-extracted one.

The root ``evidence/literature/<raw_sha>/extracted.json`` layout is UNCHANGED and is
never rewritten or migrated by this module. Other readers of it (grounding, the CLI)
keep working exactly as before.

Multiple records may accumulate under the same ``<raw_sha>/extractions/`` for one
raw artifact over time (a ``pypdf`` upgrade, an extraction code change). Nothing on
disk marks any one of them "current" -- see :func:`current_extraction_records` for
why a stored index/symlink/flag was deliberately rejected in favour of computing
currentness fresh, every time, from today's code.

Trust model and known limitations (deliberately out of scope to close here):

- This module assumes a trusted, single-user workspace: every path it writes to or
  reads from is validated for containment under ``workspace_root`` (see
  :func:`_assert_contained`), which stops accidental path traversal from a
  caller-supplied sha-shaped string. It does NOT defend against a hostile,
  co-resident, same-privilege process racing this module between a containment
  check and the filesystem operation that follows it (a TOCTOU window) -- doing so
  would require ``openat``/``O_NOFOLLOW``-based fd-relative filesystem access
  throughout, which this module deliberately does not build. A workspace shared
  with an untrusted process is out of scope for this store's guarantees.
- ``os.rename`` publishing a completed record directory onto its final address is
  followed by an ``fsync`` of the containing ``extractions/`` directory's fd, so
  the *record's own publish* is durable. The FIRST creation of the parent
  ``extractions/`` directory (``records_dir.mkdir(parents=True, exist_ok=True)``)
  is not itself fsynced here -- an interrupted process could, in principle, lose
  that directory-entry creation on some filesystems even though the record it
  contains was durably published moments later. This mirrors the same
  not-fully-durable directory-creation gap already present in
  :mod:`carmel.services.evidence` and :mod:`carmel.services.dataset_store`.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from carmel.agents.tools.extract import ExtractedText
from carmel.logger import get_logger
from carmel.paths import normalize_path

# ROOT_EXTRACTION_ID lives beside the corpus-coverage key that consumes it
# (carmel.schemas.CoveredDocument.extraction_id); imported here rather than
# redefined so there is exactly one definition of the sentinel. This import
# direction (services -> schemas) is the one this codebase allows.
from carmel.schemas.literature import ROOT_EXTRACTION_ID
from carmel.services.artifacts import read_bytes, read_json, write_bytes, write_json, write_text
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import artifact_dir, load_artifact_text
from carmel.services.semantic_deps import (
    _PYPDF_VERSION_UNKNOWN,
    ExtractionIdentity,
    extraction_identity,
)

__all__ = [
    "EXTRACTIONS_SUBDIR",
    "ExtractionPreference",
    "ExtractionRecordError",
    "ExtractionRecordMeta",
    "ExtractionSelectionError",
    "SelectedExtraction",
    "UnknownPypdfVersionError",
    "compute_extraction_sha",
    "current_extraction_records",
    "extraction_record_dir",
    "list_extraction_records",
    "load_extraction_record",
    "select_extraction",
    "store_extraction_record",
    "stored_extraction_sha256",
    "verify_extraction_record",
]

logger = get_logger("services.extraction_record")

#: Subdirectory of ``evidence/literature/<raw_sha>/`` that holds extraction records.
EXTRACTIONS_SUBDIR = "extractions"

_EXTRACTED_NAME = "extracted.json"
_TEXT_NAME = "text.txt"
_META_NAME = "meta.json"

#: Mirrors :mod:`carmel.services.evidence`'s and :mod:`carmel.services.dataset_store`'s
#: own private ``_SHA256_RE`` -- intentionally NOT imported from either. Both this
#: module's addresses (``raw_sha``, ``extraction_sha``) are the same 64-lowercase-hex
#: shape, but the three stores have three unrelated layouts; duplicating a
#: five-character regex is cheaper than coupling three modules' internals together
#: for it.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Version tag for the SHAPE of the identity payload this module hashes -- not a
#: version of any single field's content. Bumped whenever a future change adds,
#: removes, or renames one of the named identity fields; existing extraction shas
#: are then understood to have been computed under an OLD shape, and are never
#: silently reinterpreted as though computed under the new one. Bumped from "1" to
#: "2" for the fix that stopped folding ``pypdf_version`` into the identity of
#: extractors that do not depend on ``pypdf`` at all (see
#: :data:`_PYPDF_DEPENDENT_EXTRACTORS`).
_IDENTITY_PAYLOAD_VERSION = "2"

#: The only extractor strings (see :class:`carmel.agents.tools.extract.ExtractedText`
#: and its docstring for the authoritative vocabulary: ``"pdf:pypdf"``,
#: ``"pdf:unavailable"``, ``"html"``, ``"xml"``, ``"text"``, ``"unknown"``) whose
#: identity actually depends on the installed ``pypdf`` version. Deliberately NOT a
#: ``"pdf:"``-prefix test: ``"pdf:unavailable"`` also starts with ``"pdf:"`` but is
#: emitted precisely when ``pypdf`` is NOT importable (see
#: ``carmel/agents/tools/extract.py`` where ``pdf:unavailable`` is produced on
#: ``ImportError``), so a prefix test would wrongly demand a known ``pypdf`` version
#: exactly where there provably is none. Only ``"pdf:pypdf"`` -- the extractor string
#: that means "pypdf actually ran" -- belongs in this set.
_PYPDF_DEPENDENT_EXTRACTORS: frozenset[str] = frozenset({"pdf:pypdf"})

#: The extractor string emitted precisely when ``pypdf`` was NOT importable. It is
#: deliberately absent from :data:`_PYPDF_DEPENDENT_EXTRACTORS` above -- demanding a
#: known pypdf version to STORE it would be wrong, since there provably is none -- but
#: it is still pypdf-dependent for the purpose of deciding whether a stored record is
#: CURRENT: see :func:`_is_current`.
_PYPDF_UNAVAILABLE_EXTRACTOR = "pdf:unavailable"

#: The base identity fields every payload must carry, regardless of extractor.
_BASE_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "identity_payload_version",
        "parent_raw_sha256",
        "extractor",
        "extractor_code_sha256",
        "extracted_sha256",
        "extracted_text_sha256",
    }
)

#: Identity fields whose value must itself be a well-formed sha256 hex digest.
_SHA_SHAPED_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"parent_raw_sha256", "extractor_code_sha256", "extracted_sha256", "extracted_text_sha256"}
)


class ExtractionRecordError(ValueError):
    """Base class for domain errors raised by this module.

    A subclass of ``ValueError`` (not a bare new hierarchy) so existing
    ``except ValueError`` handlers elsewhere in the codebase still catch these;
    the named subclasses exist so a caller that specifically cares can catch
    narrowly instead.
    """


class UnknownPypdfVersionError(ExtractionRecordError):
    """Raised when addressing/storing a ``pdf:pypdf`` extraction whose ``pypdf_version`` is unknown.

    :func:`carmel.services.semantic_deps._pypdf_version` deliberately falls back to
    the sentinel ``"unknown"`` rather than raising, because version discovery for an
    optional/lazily-imported third-party package must never crash a caller that only
    wants an identity for logging/comparison. That is fine for diagnostics. It is NOT
    fine for provenance: an extraction record whose PDF dependency identity is
    unknown cannot later prove what actually produced its text, so this module
    refuses to compute an address for one at all rather than silently minting one
    that looks like every other record.
    """


@dataclass(frozen=True)
class ExtractionRecordMeta:
    """Persisted identity and provenance record for one stored extraction.

    Deliberately a plain frozen dataclass (mirrors
    :class:`carmel.services.dataset_store.StoredDataset`'s minimalism) rather than a
    pydantic model in :mod:`carmel.schemas` -- this increment is scoped to the store
    alone; growing a shared schema module is a separate decision for whichever
    caller eventually consumes this store.

    Attributes:
        extraction_sha256: This record's own content address (also its directory
            name under ``<raw_sha>/extractions/``).
        parent_raw_sha256: The sha256 of the raw artifact this extraction was
            derived from (also the parent directory's name).
        extractor: The extractor string recorded on the underlying
            ``ExtractedText`` (e.g. ``"pdf:pypdf"``, ``"html"``, ``"text"``).
        extractor_code_sha256: Carmel's own extraction/normalization code identity
            -- see :func:`carmel.services.semantic_deps.current_sha_for` for
            ``EXTRACT_TEXT_DEPENDENCY_ID``.
        pypdf_version: The installed ``pypdf`` version string at extraction time,
            or :data:`carmel.services.semantic_deps._PYPDF_VERSION_UNKNOWN`. Always
            recorded, purely for diagnostics, even for extractors that do not
            depend on ``pypdf`` -- but only folded into the identity ADDRESS (see
            :func:`compute_extraction_sha`) when ``extractor`` is one of
            :data:`_PYPDF_DEPENDENT_EXTRACTORS`.
        extracted_sha256: sha256 of the stored ``extracted.json`` bytes. This is
            the ONE digest :func:`verify_extraction_record` checks; see that
            function's docstring for why ``text.txt`` is not checked.
        extracted_text_sha256: sha256 of the stored ``text.txt`` bytes, folded into
            the identity address itself (see :func:`compute_extraction_sha`) so
            that two extractions producing different text can never collide onto
            one address even if every other identity field matches.
        identity_payload_version: The identity-payload SHAPE version this record's
            address was computed under -- see the module-level
            ``_IDENTITY_PAYLOAD_VERSION`` comment.
        stored_at: ISO-8601 timestamp of when this record was first written.
    """

    extraction_sha256: str
    parent_raw_sha256: str
    extractor: str
    extractor_code_sha256: str
    pypdf_version: str
    extracted_sha256: str
    extracted_text_sha256: str
    identity_payload_version: str
    stored_at: str


def extraction_record_dir(workspace_root: Path, raw_sha256: str, extraction_sha256: str) -> Path:
    """Compute (but never create) the directory for one extraction record.

    Pure path helper: no filesystem access, no validation, no side effects --
    mirrors :func:`carmel.services.evidence.artifact_dir`'s contract exactly.
    Callers that accept ``raw_sha256``/``extraction_sha256`` from outside this
    module (i.e. everything except the fresh digest just computed by
    :func:`store_extraction_record`) MUST go through :func:`load_extraction_record`,
    :func:`verify_extraction_record`, or :func:`list_extraction_records` instead,
    which validate shape and containment before touching disk.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.
        extraction_sha256: sha256 content address of this extraction record.

    Returns:
        ``<workspace_root>/evidence/literature/<raw_sha256>/extractions/<extraction_sha256>``.
    """
    return artifact_dir(workspace_root, raw_sha256) / EXTRACTIONS_SUBDIR / extraction_sha256


def _assert_contained(workspace_root: Path, path: Path) -> Path:
    """Resolve ``path`` and ``workspace_root`` and confirm containment.

    Duplicated from :mod:`carmel.services.evidence`'s private helper of the same
    name (which duplicated it, in turn, from nothing -- this is the third copy in
    the "evidence/dataset/extraction record" family; see this module's docstring
    on why these small helpers are duplicated rather than imported cross-module).

    Args:
        workspace_root: Root of the campaign workspace.
        path: Candidate destination path.

    Returns:
        The resolved path.

    Raises:
        ValueError: If the resolved path is not inside the resolved workspace root.
    """
    resolved_root = normalize_path(workspace_root)
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"refusing to write outside workspace root: {resolved_path} not under {resolved_root}")
    return resolved_path


def _validate_sha(value: str, *, label: str) -> None:
    """Validate ``value`` is a well-formed 64-lowercase-hex sha256 digest.

    Read paths receive both ``raw_sha256`` and ``extraction_sha256`` directly from
    the caller and interpolate them into a filesystem path. Without this,
    something like ``"../../etc/passwd"`` would walk straight into a path. Mirrors
    :func:`carmel.services.evidence._validate_sha256`'s check, generalized to a
    caller-supplied label so one helper serves both sha-shaped parameters this
    module accepts.

    Args:
        value: Caller-supplied hex digest.
        label: Name of the parameter, folded into the error message.

    Raises:
        ValueError: If ``value`` is not exactly 64 lowercase hex characters.
    """
    # Matched with `fullmatch`, never `match`. `re.match` only anchors at the
    # start of the string, so `$` also matches just before a trailing
    # newline -- `"a" * 64 + "\n"` would pass a `match`-based check and go on
    # to be used as a filesystem path component and as an identity field.
    # `fullmatch` requires the whole string to be the 64 lowercase-hex
    # characters, with nothing before or after.
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r} (expected 64 lowercase hex characters)")


def _validated_records_dir(workspace_root: Path, raw_sha256: str) -> Path:
    """Validate ``raw_sha256`` and resolve the containment-checked ``extractions/`` directory."""
    _validate_sha(raw_sha256, label="raw_sha256")
    root = normalize_path(workspace_root)
    return _assert_contained(root, artifact_dir(root, raw_sha256) / EXTRACTIONS_SUBDIR)


def _validated_record_dir(workspace_root: Path, raw_sha256: str, extraction_sha256: str) -> Path:
    """Validate both shas and resolve the containment-checked record directory."""
    _validate_sha(extraction_sha256, label="extraction_sha256")
    records_dir = _validated_records_dir(workspace_root, raw_sha256)
    root = normalize_path(workspace_root)
    return _assert_contained(root, records_dir / extraction_sha256)


def _refuse_unknown_pypdf_version(identity_payload: dict[str, str]) -> None:
    """Raise :exc:`UnknownPypdfVersionError` iff a ``pypdf``-dependent extractor's version is unknown.

    Args:
        identity_payload: The (not yet hashed) identity payload; only its
            ``"extractor"`` and ``"pypdf_version"`` fields are inspected.

    Raises:
        UnknownPypdfVersionError: If ``identity_payload["extractor"]`` is one of
            :data:`_PYPDF_DEPENDENT_EXTRACTORS` and
            ``identity_payload["pypdf_version"]`` equals
            :data:`carmel.services.semantic_deps._PYPDF_VERSION_UNKNOWN`.
    """
    extractor = identity_payload.get("extractor", "")
    pypdf_version = identity_payload.get("pypdf_version", "")
    if extractor in _PYPDF_DEPENDENT_EXTRACTORS and pypdf_version == _PYPDF_VERSION_UNKNOWN:
        raise UnknownPypdfVersionError(
            f"refusing to address a {extractor!r} extraction record whose pypdf_version is "
            f"{_PYPDF_VERSION_UNKNOWN!r}: this extraction's dependency identity cannot be proven, so "
            "it must never be given a stored, addressed record"
        )


def _build_identity_payload(
    *,
    identity_payload_version: str = _IDENTITY_PAYLOAD_VERSION,
    raw_sha256: str,
    extractor: str,
    extractor_code_sha256: str,
    pypdf_version: str,
    extracted_sha256: str,
    extracted_text_sha256: str,
) -> dict[str, str]:
    """Build the named-field identity payload shared by store, load, and resolve paths.

    A single place for this shape so :func:`store_extraction_record`,
    :func:`_load_meta`, and :func:`current_extraction_records` can never
    independently drift on which fields go into the address -- see
    :func:`compute_extraction_sha` for what happens to the result.

    ``pypdf_version`` is folded into the returned payload ONLY when ``extractor``
    is one of :data:`_PYPDF_DEPENDENT_EXTRACTORS` -- see that constant's docstring.
    For every other extractor (including ``"pdf:unavailable"``) the payload omits
    the key entirely, so a ``pypdf`` upgrade/downgrade can never change that
    record's address.

    Args:
        identity_payload_version: The identity-payload SHAPE version STRING to
            stamp into the payload. Defaults to the module's current
            :data:`_IDENTITY_PAYLOAD_VERSION`; callers re-authenticating an
            EXISTING record pass that record's own recorded value instead (see
            :func:`_load_meta`). Note this only reproduces the version STRING
            that was stamped into the original payload -- it does NOT
            reproduce the KEY-SET rule (which fields, e.g. ``pypdf_version``,
            get included) as it stood when that version was current. The
            key-set rule below (``if extractor in _PYPDF_DEPENDENT_EXTRACTORS``)
            always applies TODAY's logic, regardless of which
            ``identity_payload_version`` is passed in. If that rule ever
            changes, a record stamped with an older
            ``identity_payload_version`` will have its payload rebuilt under
            the NEW rule, not reinterpreted under the rule that was current
            when it was stored.
    """
    payload = {
        "identity_payload_version": identity_payload_version,
        "parent_raw_sha256": raw_sha256,
        "extractor": extractor,
        "extractor_code_sha256": extractor_code_sha256,
        "extracted_sha256": extracted_sha256,
        "extracted_text_sha256": extracted_text_sha256,
    }
    if extractor in _PYPDF_DEPENDENT_EXTRACTORS:
        payload["pypdf_version"] = pypdf_version
    return payload


def _identity_payload_from_meta(meta: ExtractionRecordMeta) -> dict[str, str]:
    """Rebuild the identity payload a stored record's OWN fields claim to hash to.

    Used only by :func:`_load_meta` to re-derive ``meta.extraction_sha256`` from
    ``meta``'s other recorded fields and confirm the two agree -- the
    self-authenticating property the whole address scheme depends on (see the
    module docstring). Deliberately uses ``meta.identity_payload_version`` (the
    version THIS record claims), not today's module-level constant, for the
    version STRING stamped into the rebuilt payload.

    This does NOT guarantee the rebuild reproduces the exact payload shape
    that was hashed when the record was stored: :func:`_build_identity_payload`
    decides which keys to include (e.g. whether ``pypdf_version`` is present)
    using TODAY's :data:`_PYPDF_DEPENDENT_EXTRACTORS`/key-set rule, not a rule
    keyed off ``identity_payload_version``. For a record whose
    ``identity_payload_version`` differs from the module's current
    :data:`_IDENTITY_PAYLOAD_VERSION`, the rebuild still applies today's
    key-set rule -- so such a record fails to authenticate (its recomputed
    address will not match ``meta.extraction_sha256``) rather than being
    correctly reinterpreted under the rule that was current for its own
    version.
    """
    return _build_identity_payload(
        identity_payload_version=meta.identity_payload_version,
        raw_sha256=meta.parent_raw_sha256,
        extractor=meta.extractor,
        extractor_code_sha256=meta.extractor_code_sha256,
        pypdf_version=meta.pypdf_version,
        extracted_sha256=meta.extracted_sha256,
        extracted_text_sha256=meta.extracted_text_sha256,
    )


def compute_extraction_sha(identity_payload: dict[str, str]) -> str:
    """Compute the content address of ``identity_payload``.

    Mirrors :func:`carmel.services.dataset_store.compute_dataset_sha`'s contract:
    ``identity_payload`` must already be the plain, deliberately-constructed dict
    that defines this record's identity (see :func:`_build_identity_payload`) --
    never a delimiter-joined string, and never a model object passed in for this
    function to unwrap. sha256 of :func:`~carmel.services.dataset_store.
    canonical_json_bytes` of the payload is stable regardless of the dict's key
    insertion order (canonical JSON sorts keys) and identical across platforms and
    Python versions (no float/dict-order/rounding nondeterminism reaches it, since
    every field here is a plain string).

    ``identity_payload`` is validated before hashing: it must contain EXACTLY the
    fields :func:`_build_identity_payload` would produce for its own
    ``"extractor"`` value (the base fields always; ``"pypdf_version"`` additionally
    iff that extractor is one of :data:`_PYPDF_DEPENDENT_EXTRACTORS`) -- no missing
    field, no extra field. Every value must be a non-empty ``str``, and every
    sha-shaped field (see :data:`_SHA_SHAPED_IDENTITY_FIELDS`) must be a
    well-formed 64-lowercase-hex digest. This makes this module's docstring claim
    that the payload is a "deliberately constructed identity payload" an enforced
    property, not just a naming convention a caller could quietly violate.

    Args:
        identity_payload: The plain dict whose canonical JSON form defines this
            extraction record's content address.

    Returns:
        The sha256 hex digest of ``identity_payload``'s canonical JSON bytes.

    Raises:
        ValueError: If ``identity_payload`` does not contain exactly the expected
            field set for its own ``"extractor"`` value, or any field's value is
            not a non-empty ``str`` of the expected shape.
        UnknownPypdfVersionError: If ``identity_payload`` describes a
            ``pypdf``-dependent extraction whose ``pypdf_version`` is the "could
            not determine" sentinel -- see :func:`_refuse_unknown_pypdf_version`.
        carmel.services.dataset_store.CanonicalJsonError: If ``identity_payload``
            is not representable as canonical JSON.
    """
    _validate_identity_payload(identity_payload)
    _refuse_unknown_pypdf_version(identity_payload)
    return hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()


def _validate_identity_payload(identity_payload: dict[str, str]) -> None:
    """Validate ``identity_payload`` has exactly the expected field set and value shapes.

    See :func:`compute_extraction_sha` for the full contract this enforces.
    """
    if not isinstance(identity_payload, dict):
        raise ValueError(f"identity_payload must be a dict, got {type(identity_payload).__name__}")

    extractor = identity_payload.get("extractor")
    expected_fields = set(_BASE_IDENTITY_FIELDS)
    if extractor in _PYPDF_DEPENDENT_EXTRACTORS:
        expected_fields.add("pypdf_version")

    actual_fields = set(identity_payload)
    if actual_fields != expected_fields:
        raise ValueError(
            f"malformed extraction identity payload: expected exactly {sorted(expected_fields)}, "
            f"got {sorted(actual_fields)}"
        )

    for key, value in identity_payload.items():
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"malformed extraction identity payload field {key!r}: expected a non-empty str, got {value!r}"
            )

    for key in _SHA_SHAPED_IDENTITY_FIELDS:
        _validate_sha(identity_payload[key], label=key)


def _require_str(raw: dict[str, Any], key: str) -> str:
    """Return ``raw[key]`` as a ``str``, or raise if absent/wrong-typed.

    Raises:
        KeyError: If ``key`` is absent from ``raw``.
        TypeError: If ``raw[key]`` is present but not a ``str``.
    """
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be a str, got {type(value).__name__}")
    return value


def _load_meta(
    meta_path: Path, *, expected_raw_sha256: str, expected_extraction_sha256: str
) -> ExtractionRecordMeta | None:
    """Load, type-validate, and self-authenticate one ``meta.json``.

    Three outcomes, deliberately distinguished (see this module's F8 test class
    and the module docstring's self-authenticating-address paragraph):

    - Genuinely no record at this path (``meta.json`` does not exist): returns
      ``None``. This is the ordinary "nothing stored here" case every caller
      already treats as unremarkable.
    - ``meta.json`` exists but is unreadable (permission error), not valid JSON,
      not a JSON object, missing an expected field, or has a wrong-typed field
      (including a non-string ``stored_at`` -- caught HERE, at load time, rather
      than later crashing :func:`list_extraction_records`' ``sorted()`` call with
      an opaque ``TypeError``): raises :exc:`ExtractionRecordError`. This is
      corruption, not absence, and must never be silently swallowed into the same
      ``None`` a caller would see for "nothing here".
    - ``meta.json`` parses and type-checks cleanly, but does not AUTHENTICATE to
      the address it was loaded from -- its own ``parent_raw_sha256`` or
      ``extraction_sha256`` disagrees with what the caller asked for, or
      recomputing :func:`compute_extraction_sha` from its own recorded identity
      fields (see :func:`_identity_payload_from_meta`) does not reproduce
      ``expected_extraction_sha256``: returns ``None``. A forged or moved
      ``meta.json`` must never verify under an address it does not actually
      belong to, but from a caller's perspective this is indistinguishable from
      "no valid record at this address" -- not a distinct corruption signal.

    Args:
        meta_path: Path to the candidate ``meta.json``.
        expected_raw_sha256: The ``raw_sha256`` the caller is asking about (i.e.
            the parent directory two levels up from ``meta_path``).
        expected_extraction_sha256: The directory name ``meta_path`` was loaded
            from (i.e. what this record's address is claimed to be).

    Returns:
        The authenticated :class:`ExtractionRecordMeta`, or ``None`` if no record
        exists or the loaded record does not authenticate to this address.

    Raises:
        ExtractionRecordError: If ``meta.json`` exists but is unreadable, not
            valid JSON, not a JSON object, or has a missing/wrong-typed/malformed
            field.
    """
    try:
        raw = read_json(meta_path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ExtractionRecordError(
            f"extraction record meta.json at {meta_path} is unreadable or not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ExtractionRecordError(f"extraction record meta.json at {meta_path} is not a JSON object")

    try:
        meta = ExtractionRecordMeta(
            extraction_sha256=_require_str(raw, "extraction_sha256"),
            parent_raw_sha256=_require_str(raw, "parent_raw_sha256"),
            extractor=_require_str(raw, "extractor"),
            extractor_code_sha256=_require_str(raw, "extractor_code_sha256"),
            pypdf_version=_require_str(raw, "pypdf_version"),
            extracted_sha256=_require_str(raw, "extracted_sha256"),
            extracted_text_sha256=_require_str(raw, "extracted_text_sha256"),
            identity_payload_version=_require_str(raw, "identity_payload_version"),
            stored_at=_require_str(raw, "stored_at"),
        )
    except (KeyError, TypeError) as exc:
        raise ExtractionRecordError(
            f"extraction record meta.json at {meta_path} has a missing or wrong-typed field: {exc}"
        ) from exc

    for label in (
        "extraction_sha256",
        "parent_raw_sha256",
        "extractor_code_sha256",
        "extracted_sha256",
        "extracted_text_sha256",
    ):
        try:
            _validate_sha(getattr(meta, label), label=label)
        except ValueError as exc:
            raise ExtractionRecordError(f"extraction record meta.json at {meta_path}: {exc}") from exc

    if meta.parent_raw_sha256 != expected_raw_sha256:
        return None
    if meta.extraction_sha256 != expected_extraction_sha256:
        return None
    try:
        recomputed = compute_extraction_sha(_identity_payload_from_meta(meta))
    except (ExtractionRecordError, ValueError):
        return None
    if recomputed != expected_extraction_sha256:
        return None

    return meta


def _records_identical(candidate_dir: Path, existing_dir: Path) -> bool:
    """Whether two candidate extraction-record directories describe the same record.

    ``extracted.json`` and ``text.txt`` must be byte-identical: those are the
    content this record exists to preserve, so any difference there is a genuine
    collision. ``meta.json`` is compared field-by-field EXCLUDING ``stored_at``:
    that field records when a directory was first written, which necessarily
    differs between two otherwise-identical calls to :func:`store_extraction_record`
    made at two different times -- comparing it byte-for-byte would make a
    perfectly idempotent re-store look like a collision on every single re-store.

    Used only to decide whether a re-store at an already-occupied address is the
    idempotent no-op it is supposed to be, or a genuine collision that must raise
    rather than silently overwrite an existing, append-only record.

    ``pypdf_version`` is excluded from the ``meta.json`` comparison alongside
    ``stored_at``: for a non-``pypdf``-dependent extractor (see
    :data:`_PYPDF_DEPENDENT_EXTRACTORS`), ``pypdf_version`` is recorded purely for
    diagnostics and is NOT folded into the address, so two calls to
    :func:`store_extraction_record` made under two different installed ``pypdf``
    versions can legitimately compute the SAME address while disagreeing on this
    one diagnostic-only field -- that must read as the idempotent no-op it is, not
    as a collision.
    """
    for name in (_EXTRACTED_NAME, _TEXT_NAME):
        candidate_bytes = (candidate_dir / name).read_bytes()
        try:
            existing_bytes = (existing_dir / name).read_bytes()
        except FileNotFoundError:
            return False
        if candidate_bytes != existing_bytes:
            return False

    try:
        candidate_meta = read_json(candidate_dir / _META_NAME)
        existing_meta = read_json(existing_dir / _META_NAME)
    except (FileNotFoundError, ValueError):
        return False
    ignored = {"stored_at", "pypdf_version"}
    candidate_identity = {k: v for k, v in candidate_meta.items() if k not in ignored}
    existing_identity = {k: v for k, v in existing_meta.items() if k not in ignored}
    return candidate_identity == existing_identity


def store_extraction_record(
    workspace_root: Path,
    *,
    raw_sha256: str,
    extractor: str,
    extractor_code_sha256: str,
    pypdf_version: str,
    extracted_json_bytes: bytes,
) -> str:
    """Parse, content-address, and durably persist one extraction record.

    ``extracted_json_bytes`` is parsed as :class:`carmel.agents.tools.extract.
    ExtractedText` (the same parse :mod:`carmel.services.dataset_producer` already
    performs on the root ``extracted.json`` layout) BEFORE anything is hashed or
    written. This is deliberate: the caller's ``extractor`` argument is an
    assertion about what produced these bytes, and the ONLY way to know it is true
    is to parse the bytes and read their own ``extractor`` field back out. The
    text persisted to ``text.txt`` (and folded into the identity address via
    ``extracted_text_sha256``) is the parsed ``ExtractedText.text`` -- there is no
    separate ``text`` parameter for a caller to accidentally pass bytes that
    describe one string while asserting another; that entire mismatch class is
    deleted by construction rather than validated around.

    Writes ``extracted.json``, ``text.txt``, and ``meta.json`` under
    ``<workspace_root>/evidence/literature/<raw_sha256>/extractions/<extraction_sha>/``,
    where ``<extraction_sha>`` is computed by :func:`compute_extraction_sha` from a
    named-field identity payload built from this call's own arguments (see
    :func:`_build_identity_payload`).

    Append-only and race-safe against a concurrent writer targeting the same
    address, mirroring :func:`carmel.services.dataset_store.store_dataset`'s
    collision idiom: all three files are first written into a private temp
    directory (each write itself atomic and fsynced -- see
    :mod:`carmel.services.artifacts`), then the temp directory is published via
    ``os.rename`` onto the final address. ``os.rename`` onto an existing
    non-empty directory fails atomically at the kernel level rather than merging
    or silently replacing, so at most one concurrent writer's directory ever
    becomes visible at that address. A writer that loses the race re-reads the
    now-visible directory and compares it, file for file, against what it was
    about to publish:

    - Byte-identical -> no rewrite; the existing address is returned unchanged
      (idempotent re-store).
    - Any difference -> raises. Two identity payloads that hash to the same
      sha256 but disagree on-disk would be a genuine sha256 collision; far more
      likely in practice, this indicates on-disk corruption. Either way this is
      always a hard failure -- an extraction record is append-only, so silently
      overwriting one would destroy exactly the auditability this store exists
      to provide.

    Any OTHER failure to publish (e.g. out of disk space, permission denied,
    missing parent) is NOT treated as a collision -- it is re-raised as the
    original ``OSError`` rather than being misreported as an on-disk conflict.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact this extraction was derived
            from. Must already be a well-formed 64-character lowercase hex digest;
            this function does NOT require ``raw_sha256`` to already have a stored
            artifact under ``evidence/literature/`` -- the two stores are
            deliberately decoupled.
        extractor: The extractor string this extraction is asserted to have been
            produced with (e.g. ``"pdf:pypdf"``, ``"html"``, ``"text"``). Must
            equal the ``extractor`` field the parsed ``extracted_json_bytes``
            itself records, or this call raises.
        extractor_code_sha256: Carmel's own extraction/normalization code identity
            at the time this extraction ran -- see
            :func:`carmel.services.semantic_deps.current_sha_for`.
        pypdf_version: The installed ``pypdf`` version string at extraction time,
            or :data:`carmel.services.semantic_deps._PYPDF_VERSION_UNKNOWN`. Always
            recorded for diagnostics; only folded into the identity ADDRESS when
            ``extractor`` is one of :data:`_PYPDF_DEPENDENT_EXTRACTORS` -- see
            :exc:`UnknownPypdfVersionError`.
        extracted_json_bytes: The exact bytes to persist as ``extracted.json``.
            Must parse as :class:`~carmel.agents.tools.extract.ExtractedText` whose
            own ``extractor`` field equals ``extractor``. Never empty.

    Returns:
        The sha256 hex digest of this record's identity payload (also its
        directory name).

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase
            hex digest, if ``extractor``/``extractor_code_sha256``/``pypdf_version``
            is empty, if ``extracted_json_bytes`` is empty, or if the resolved
            destination would fall outside the resolved workspace root.
        ExtractionRecordError: If ``extracted_json_bytes`` does not parse as JSON
            or as ``ExtractedText``, if the parsed ``ExtractedText.extractor``
            disagrees with the ``extractor`` argument, or if an existing record
            already occupies the computed address and its on-disk bytes disagree
            with the bytes about to be stored.
        UnknownPypdfVersionError: See :func:`compute_extraction_sha`.
        OSError: If publishing the record fails for a reason other than an
            existing occupant at the destination (e.g. disk full, permission
            denied).
    """
    _validate_sha(raw_sha256, label="raw_sha256")
    if not extractor:
        raise ValueError("extractor must not be empty")
    if not extractor_code_sha256:
        raise ValueError("extractor_code_sha256 must not be empty")
    if not pypdf_version:
        raise ValueError("pypdf_version must not be empty")
    if not extracted_json_bytes:
        raise ValueError(
            "refusing to store an empty extracted.json: an extraction that yields no bytes is a "
            "failed extraction, never a record"
        )

    try:
        parsed = json.loads(extracted_json_bytes)
    except ValueError as exc:
        raise ExtractionRecordError(f"extracted_json_bytes is not valid JSON: {exc}") from exc
    try:
        extracted = ExtractedText.model_validate(parsed)
    except ValueError as exc:
        raise ExtractionRecordError(f"extracted_json_bytes does not parse as ExtractedText: {exc}") from exc
    if extracted.extractor != extractor:
        raise ExtractionRecordError(
            f"extractor mismatch: caller asserted extractor={extractor!r}, but the parsed "
            f"ExtractedText records extractor={extracted.extractor!r}; refusing to store a record "
            "whose claimed identity does not match its own parsed bytes"
        )
    text = extracted.text

    extracted_sha256 = hashlib.sha256(extracted_json_bytes).hexdigest()
    extracted_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity_payload = _build_identity_payload(
        raw_sha256=raw_sha256,
        extractor=extractor,
        extractor_code_sha256=extractor_code_sha256,
        pypdf_version=pypdf_version,
        extracted_sha256=extracted_sha256,
        extracted_text_sha256=extracted_text_sha256,
    )
    extraction_sha256 = compute_extraction_sha(identity_payload)

    root = normalize_path(workspace_root)
    dest_dir = _assert_contained(root, extraction_record_dir(root, raw_sha256, extraction_sha256))
    records_dir = dest_dir.parent
    records_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {
        "extraction_sha256": extraction_sha256,
        "parent_raw_sha256": raw_sha256,
        "extractor": extractor,
        "extractor_code_sha256": extractor_code_sha256,
        "pypdf_version": pypdf_version,
        "extracted_sha256": extracted_sha256,
        "extracted_text_sha256": extracted_text_sha256,
        "identity_payload_version": _IDENTITY_PAYLOAD_VERSION,
        "stored_at": datetime.now(UTC).isoformat(),
    }

    tmp_dir = Path(tempfile.mkdtemp(dir=records_dir, prefix=f".{extraction_sha256}.tmp-"))
    try:
        write_bytes(tmp_dir / _EXTRACTED_NAME, extracted_json_bytes)
        write_text(tmp_dir / _TEXT_NAME, text)
        write_json(tmp_dir / _META_NAME, meta_payload)
        try:
            os.rename(tmp_dir, dest_dir)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if not _records_identical(tmp_dir, dest_dir):
                raise ExtractionRecordError(
                    f"extraction record collision at {dest_dir}: an existing on-disk record does "
                    f"not match the bytes about to be stored for extraction_sha256="
                    f"{extraction_sha256!r}; refusing to overwrite an append-only record"
                ) from None
            return extraction_sha256
        dir_fd = os.open(records_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return extraction_sha256
    finally:
        # No-op once `os.rename` above has succeeded (nothing left at `tmp_dir` to
        # remove); only actually cleans up on the collision/raise paths.
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_extraction_record(
    workspace_root: Path, raw_sha256: str, extraction_sha256: str
) -> ExtractionRecordMeta | None:
    """Load one extraction record's metadata, by (``raw_sha256``, ``extraction_sha256``).

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.
        extraction_sha256: sha256 content address of the extraction record.

    Returns:
        The persisted, self-authenticated :class:`ExtractionRecordMeta`, or
        ``None`` when no record is stored at that address, or a stored
        ``meta.json`` does not authenticate to it (wrong ``parent_raw_sha256``,
        wrong ``extraction_sha256``, or a recomputed address that disagrees with
        the directory it was loaded from -- see :func:`_load_meta`).

    Raises:
        ValueError: If either digest is not a well-formed 64-character lowercase
            hex digest, or if the resolved record directory would fall outside the
            resolved workspace root.
        ExtractionRecordError: If a ``meta.json`` exists at that address but is
            unreadable, not valid JSON, or has a missing/wrong-typed field -- see
            :func:`_load_meta`.
    """
    dest_dir = _validated_record_dir(workspace_root, raw_sha256, extraction_sha256)
    return _load_meta(
        dest_dir / _META_NAME,
        expected_raw_sha256=raw_sha256,
        expected_extraction_sha256=extraction_sha256,
    )


def verify_extraction_record(workspace_root: Path, raw_sha256: str, extraction_sha256: str) -> bool:
    """Authenticate the address, then confirm the stored ``extracted.json`` still matches its recorded digest.

    Only ``extracted.json`` is digest-checked here, deliberately mirroring
    :mod:`carmel.services.evidence`'s own documented stance on its ``text.txt``
    sidecar: ``text.txt`` is written for human/tooling convenience, never claimed
    to be the integrity-checked artifact, so checking it here would assert a
    guarantee this store never made.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.
        extraction_sha256: sha256 content address of the extraction record.

    Returns:
        True iff a record is stored at that address, its ``meta.json`` authenticates
        to it (see :func:`_load_meta`), and the on-disk ``extracted.json`` bytes
        hash to the ``extracted_sha256`` recorded in ``meta.json``. False
        otherwise -- including when ``meta.json`` is present but does not
        authenticate to this address (forged or moved).

    Raises:
        ValueError: If either digest is not a well-formed 64-character lowercase
            hex digest, or if the resolved record directory would fall outside the
            resolved workspace root.
        ExtractionRecordError: If a ``meta.json`` exists at that address but is
            unreadable, not valid JSON, or has a missing/wrong-typed field -- see
            :func:`_load_meta`.
    """
    dest_dir = _validated_record_dir(workspace_root, raw_sha256, extraction_sha256)
    meta = _load_meta(
        dest_dir / _META_NAME,
        expected_raw_sha256=raw_sha256,
        expected_extraction_sha256=extraction_sha256,
    )
    if meta is None:
        return False
    try:
        data = read_bytes(dest_dir / _EXTRACTED_NAME)
    except FileNotFoundError:
        return False
    return hashlib.sha256(data).hexdigest() == meta.extracted_sha256


class RecordsDirState(StrEnum):
    """What the ``extractions/`` directory itself turned out to be.

    The distinction this exists to draw is ABSENT vs UNLISTABLE. Both used to present
    to a caller as an empty record list, and they license opposite decisions: an
    artifact that never had a record may legitimately be judged by its root sidecar,
    while a store that cannot be read says nothing about the artifact at all and must
    never be read as though it had answered.
    """

    ABSENT = "absent"
    """No ``extractions/`` directory exists. The artifact predates the record store."""

    LISTED = "listed"
    """The directory was read successfully. It may still have contained nothing."""

    UNLISTABLE = "unlistable"
    """The directory exists but could not be enumerated (permissions, EIO, ...)."""


@dataclass(frozen=True)
class RecordScanProblem:
    """One sha-shaped entry that could not be turned into a usable record.

    Deliberately reported rather than only logged. A ``logger.warning`` is invisible to
    the decision the caller is about to make, and the caller's decision -- whether to
    serve text -- depends on whether anything was skipped: a skipped entry is a
    CANDIDATE record, not noise, so its absence from the list silently changes a count
    the caller treats as meaningful.
    """

    entry_name: str
    """The ``extractions/`` entry name, i.e. the claimed extraction sha256."""

    reason: str
    """Operator-facing phrase naming what was wrong with this entry."""


@dataclass(frozen=True)
class ExtractionRecordScan:
    """Everything one pass over ``extractions/`` observed, including what it could not use.

    Returned instead of a bare list because a bare list cannot distinguish "there is
    nothing here" from "there is something here I could not read", and those two facts
    license opposite decisions at every call site that matters.
    """

    records: tuple[ExtractionRecordMeta, ...]
    """Readable, authenticated records, sorted by (``stored_at``, ``extraction_sha256``)."""

    problems: tuple[RecordScanProblem, ...]
    """Sha-shaped entries that could NOT be turned into a record. Empty is the good case."""

    dir_state: RecordsDirState
    """What the directory itself was; see :class:`RecordsDirState`."""

    unlistable_reason: str | None = None
    """Why enumeration failed, when ``dir_state`` is :attr:`RecordsDirState.UNLISTABLE`."""


def scan_extraction_records(workspace_root: Path, raw_sha256: str) -> ExtractionRecordScan:
    """One pass over ``extractions/``, reporting what was unusable as well as what was not.

    The richer sibling of :func:`list_extraction_records`, and the one any caller that
    is about to DECIDE something on the strength of the result should use. See
    :class:`ExtractionRecordScan` for why a bare list is not enough.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.

    Returns:
        An :class:`ExtractionRecordScan`. Never raises for a store-content problem;
        those are reported in ``problems``/``dir_state`` instead.

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved directory would fall outside the
            resolved workspace root.
    """
    records_dir = _validated_records_dir(workspace_root, raw_sha256)
    if not records_dir.exists():
        return ExtractionRecordScan(records=(), problems=(), dir_state=RecordsDirState.ABSENT)
    try:
        entries = sorted(records_dir.iterdir())
    except OSError as exc:
        logger.warning(
            "extraction record store: %s/extractions could not be listed (%s); the store has NOT "
            "reported that this artifact has no records -- it has reported nothing at all",
            raw_sha256,
            exc,
        )
        return ExtractionRecordScan(
            records=(),
            problems=(),
            dir_state=RecordsDirState.UNLISTABLE,
            unlistable_reason=str(exc),
        )

    records: list[ExtractionRecordMeta] = []
    problems: list[RecordScanProblem] = []

    def _problem(entry_name: str, reason: str) -> None:
        logger.warning(
            "extraction record store: skipping %s/extractions/%s (%s)", raw_sha256, entry_name, reason
        )
        problems.append(RecordScanProblem(entry_name=entry_name, reason=reason))

    for entry in entries:
        if not _SHA256_RE.fullmatch(entry.name):
            # Not sha-shaped: never a record this store minted, so not a candidate.
            # Editor swap files and the like are genuine noise and stay noise.
            continue
        try:
            resolved_entry = entry.resolve()
        except OSError as exc:
            _problem(entry.name, f"could not be resolved: {exc}")
            continue
        if not resolved_entry.is_relative_to(records_dir):
            _problem(entry.name, "resolves outside the extractions root")
            continue
        if not resolved_entry.is_dir():
            _problem(entry.name, "is sha-named but not a directory")
            continue
        try:
            meta = _load_meta(
                resolved_entry / _META_NAME, expected_raw_sha256=raw_sha256, expected_extraction_sha256=entry.name
            )
        except ExtractionRecordError:
            _problem(entry.name, "unreadable or malformed meta.json")
            continue
        if meta is None:
            _problem(entry.name, "no authenticating meta.json")
            continue
        records.append(meta)
    records.sort(key=lambda m: (m.stored_at, m.extraction_sha256))
    return ExtractionRecordScan(
        records=tuple(records), problems=tuple(problems), dir_state=RecordsDirState.LISTED
    )


def list_extraction_records(workspace_root: Path, raw_sha256: str) -> list[ExtractionRecordMeta]:
    """Every extraction record currently held for one raw artifact.

    Returned sorted by (``stored_at``, ``extraction_sha256``) so repeated calls
    against an unchanged store present the same records in the same order --
    mirrors :func:`carmel.services.evidence.list_artifacts_with_unreadable`'s
    ordering rationale.

    Each candidate entry's name must be sha256-shaped AND (after
    :meth:`~pathlib.Path.resolve`) still fall inside the containment-checked
    ``extractions/`` directory before its ``meta.json`` is even opened -- a
    64-hex-named SYMLINK under ``extractions/`` pointing outside the workspace is
    skipped, never followed and parsed as though it were a record. Directories
    that do not parse as a record, or whose ``meta.json`` fails to authenticate
    (see :func:`_load_meta`), are also skipped, in every case with a
    ``logger.warning`` rather than raising -- an append-only store that let one
    corrupted or hostile entry make the whole raw artifact's extraction history
    unreadable would be worse than the corruption itself.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.

    Returns:
        Sorted list of every readable, authenticated :class:`ExtractionRecordMeta`
        under ``<raw_sha256>/extractions/``. Empty (never raises) when the raw
        artifact has no ``extractions/`` directory at all.

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved directory would fall outside the
            resolved workspace root.

    Warning:
        This LOSES the distinction between "nothing is stored", "something is stored
        that I could not read", and "the directory could not be listed at all" -- all
        three return an empty list. Any caller deciding whether to trust or serve
        something must call :func:`scan_extraction_records` instead; this remains for
        callers that genuinely only want the readable records (inventory, display).
    """
    return list(scan_extraction_records(workspace_root, raw_sha256).records)


def stored_extraction_sha256(
    workspace_root: Path,
    raw_sha256: str,
    *,
    extractor: str,
    extracted_sha256: str,
    extracted_text_sha256: str,
) -> str | None:
    """Whether a FRESHLY-COMPUTED extraction result is already stored, and at what address.

    This is a lookup, not a currentness query: it answers "if I just re-ran
    extraction and got these exact digests, is that already on disk?" -- which
    requires the CALLER to have already re-run the extractor and reproduced the
    exact serialized ``extracted.json``/``text.txt`` bytes before calling this.
    (Previously named ``current_extraction_sha256``, which claimed to answer "is
    this raw artifact's extraction current" -- it never did; see
    :func:`current_extraction_records` for the query that actually answers that,
    without requiring re-extraction.)

    Recomputes the address :func:`store_extraction_record` would compute for these
    exact caller-supplied digests, combined with TODAY's code/``pypdf`` identity
    (via :func:`carmel.services.semantic_deps.extraction_identity`), and checks
    whether a record already exists at that address.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.
        extractor: The extractor string a fresh extraction of this raw artifact
            just produced (e.g. ``"pdf:pypdf"``).
        extracted_sha256: sha256 of the ``extracted.json`` bytes a fresh extraction
            just produced.
        extracted_text_sha256: sha256 of the ``text.txt`` bytes a fresh extraction
            just produced.

    Returns:
        The extraction sha of the matching on-disk record, or None when no record
        exists at that address (including: no records exist for this raw sha at
        all).

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved record directory would fall outside the
            resolved workspace root.
        UnknownPypdfVersionError: If ``extractor`` is one of
            :data:`_PYPDF_DEPENDENT_EXTRACTORS` and the CURRENTLY installed
            ``pypdf`` cannot be introspected -- see :func:`compute_extraction_sha`.
    """
    identity = extraction_identity()
    identity_payload = _build_identity_payload(
        raw_sha256=raw_sha256,
        extractor=extractor,
        extractor_code_sha256=identity.code_sha256,
        pypdf_version=identity.pypdf_version,
        extracted_sha256=extracted_sha256,
        extracted_text_sha256=extracted_text_sha256,
    )
    candidate_sha256 = compute_extraction_sha(identity_payload)
    dest_dir = _validated_record_dir(workspace_root, raw_sha256, candidate_sha256)
    meta = _load_meta(
        dest_dir / _META_NAME,
        expected_raw_sha256=raw_sha256,
        expected_extraction_sha256=candidate_sha256,
    )
    if meta is None:
        return None
    return candidate_sha256


def current_extraction_records(workspace_root: Path, raw_sha256: str) -> list[ExtractionRecordMeta]:
    """Every stored extraction record for ``raw_sha256`` that TODAY's code would still produce.

    This is the real currentness query, requiring no re-extraction: a stored
    record already records its own ``extractor_code_sha256`` and
    ``pypdf_version``, so "current" is simply every record whose recorded
    extraction-code identity equals what
    :func:`carmel.services.semantic_deps.extraction_identity` reports RIGHT NOW.
    Note the extracted digests (``extracted_sha256``, ``extracted_text_sha256``)
    are OUTPUTS of the extractor, not part of its identity -- unlike
    :func:`stored_extraction_sha256`, this function never requires the caller to
    supply them, and never requires re-running extraction at all.

    Currentness is DERIVED here, never stored: there is no ``current`` symlink, no
    ``superseded`` flag, no index file recording it. A mutable index would be
    exactly the kind of ad hoc, easily-desynced state this project already rejected
    when it rejected SQLite for the rest of its stores -- it would need to be kept
    in lockstep with every store, by hand, forever, and nothing enforces that.
    Every record NOT returned here is "superseded" purely by virtue of not
    matching; nothing on disk ever annotates it as such.

    A raw artifact may have zero, one, or several current records at once: e.g.
    one stored as ``"pdf:unavailable"`` before ``pypdf`` was installed and one
    later stored as ``"pdf:pypdf"`` after, both produced by the same (unchanged)
    Carmel extraction code, are BOTH current -- ``pypdf_version`` only
    disqualifies a record when the record's own extractor is one of
    :data:`_PYPDF_DEPENDENT_EXTRACTORS`.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.

    Returns:
        The subset of :func:`list_extraction_records`'s result whose
        ``extractor_code_sha256`` matches today's, and (for ``pypdf``-dependent
        extractors) whose ``pypdf_version`` also matches today's. Empty when no
        record for this raw sha is current (including: no records exist at all).

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved directory would fall outside the
            resolved workspace root.
    """
    identity = extraction_identity()
    return [r for r in list_extraction_records(workspace_root, raw_sha256) if _is_current(r, identity)]


def _is_current(record: ExtractionRecordMeta, identity: ExtractionIdentity) -> bool:
    """Would today's extractor still produce a record with this one's identity?

    Three cases, not two. The version comparison alone is not sufficient, because it is
    conditional on the record's OWN extractor being pypdf-dependent, which leaves
    ``pdf:unavailable`` -- the degraded placeholder written when pypdf could not be
    imported -- matching unconditionally. That placeholder then stays "current" after
    pypdf is installed, and would be selected and served as authenticated text even
    though today's extraction would produce ``pdf:pypdf`` instead. Currentness is a
    claim about what today would produce, so the availability of pypdf has to be
    compared in BOTH directions.
    """
    if record.extractor_code_sha256 != identity.code_sha256:
        return False
    pypdf_is_identifiable = identity.pypdf_version != _PYPDF_VERSION_UNKNOWN
    if record.extractor in _PYPDF_DEPENDENT_EXTRACTORS:
        # "pypdf actually ran." Current only if pypdf still runs, at the same version.
        return pypdf_is_identifiable and record.pypdf_version == identity.pypdf_version
    if record.extractor == _PYPDF_UNAVAILABLE_EXTRACTOR:
        # "pypdf could not be imported." Current only while that is still true.
        return not pypdf_is_identifiable
    # html/xml/text: nothing about them depends on pypdf either way.
    return True


class ExtractionSelectionError(ExtractionRecordError):
    """No single extraction could be resolved under the requested policy.

    Raised by :func:`select_extraction` in place of ANY silent fallback.
    Every raise site names the artifact (``raw_sha256``, and where relevant
    ``extraction_sha256``) and what was actually found on disk, so a caller
    reading the message alone can tell why the request was refused.
    """


class ExtractionPreference(StrEnum):
    """The caller-stated policy :func:`select_extraction` resolves against.

    There is deliberately no "prefer current, else root" or "newest wins"
    member: an ambiguous or unsatisfiable request is a refusal
    (:exc:`ExtractionSelectionError`), never a tie-break decided for the
    caller.
    """

    #: Read the root ``extracted.json`` sidecar
    #: (``evidence/literature/<raw_sha256>/extracted.json``). Extraction
    #: records under ``extractions/`` are not consulted at all.
    ROOT = "root"

    #: Read the ONE extraction record named by the caller's
    #: ``extraction_sha256``. Never falls back to the root sidecar or to
    #: any other record.
    EXACT = "exact"

    #: Read the single CURRENT extraction record -- and only if there is
    #: exactly one. "Current" is a weak, mutable, derived property (see
    #: :func:`current_extraction_records`'s docstring): zero, one, or
    #: several records can be current at once, and "current" is never used
    #: to pick a winner among several.
    CURRENT = "current"


@dataclass(frozen=True)
class SelectedExtraction:
    """The one extraction :func:`select_extraction` resolved to, and its text."""

    #: :data:`ROOT_EXTRACTION_ID` for the root sidecar, else the winning
    #: record's 64-lowercase-hex ``extraction_sha256``.
    extraction_id: str

    #: The text that was actually selected.
    extracted: ExtractedText


def _load_authenticated_record_text(
    workspace_root: Path, raw_sha256: str, extraction_sha256: str, meta: ExtractionRecordMeta
) -> ExtractedText:
    """Read and authenticate one extraction record's stored ``extracted.json``.

    Reuses the digest-then-parse shape already established by
    :func:`carmel.services.dataset_replay`'s node-verification path (read
    bytes, hash-compare against a trusted anchor, ``json.loads``, then
    :meth:`ExtractedText.model_validate`) rather than inventing a second way
    to read a record's ``extracted.json``. There is no envelope in scope
    here, so the anchor is ``meta.extracted_sha256`` from the record's own
    self-authenticated ``meta.json`` -- the same anchor
    :func:`verify_extraction_record` checks against.

    Raises:
        ExtractionSelectionError: the record's ``extracted.json`` is
            missing/unreadable, its bytes disagree with the digest recorded
            in ``meta.json``, the bytes do not parse as
            :class:`ExtractedText`, or the parsed text disagrees with
            ``meta.extracted_text_sha256``.
    """
    dest_dir = extraction_record_dir(workspace_root, raw_sha256, extraction_sha256)
    try:
        raw_bytes = read_bytes(dest_dir / _EXTRACTED_NAME)
    except OSError as exc:
        # OSError, not FileNotFoundError: a record whose extracted.json exists but
        # cannot be read (permissions, EIO, a dangling symlink) is exactly as unusable
        # as one that is absent, and the narrow catch let that case escape as an
        # unhandled exception which aborts the caller's entire pass over every OTHER
        # artifact -- turning one unreadable file into a campaign-wide crash.
        raise ExtractionSelectionError(
            f"extraction record raw_sha256={raw_sha256!r} extraction_sha256={extraction_sha256!r} "
            f"has a meta.json but no readable extracted.json: {exc}"
        ) from exc
    actual_extracted_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_extracted_sha256 != meta.extracted_sha256:
        raise ExtractionSelectionError(
            f"extraction record raw_sha256={raw_sha256!r} extraction_sha256={extraction_sha256!r} does not "
            f"authenticate: extracted.json bytes on disk hash to {actual_extracted_sha256!r}, but meta.json "
            f"records extracted_sha256={meta.extracted_sha256!r}"
        )
    try:
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    except (ValueError, RecursionError) as exc:
        # PEP 758 (Python 3.14) permits an unparenthesized multi-exception
        # `except A, B:`, but only without an `as` binding -- with `as`,
        # the grammar still requires parentheses around the tuple, which is
        # what this clause uses. RecursionError is caught explicitly
        # (never a bare `except Exception`, which would also swallow
        # KeyboardInterrupt/SystemExit-adjacent BaseException-only
        # escapes it shouldn't): a digest-verified extracted.json can
        # still be pathologically deep enough to blow the interpreter's
        # recursion limit during json.loads or pydantic validation.
        raise ExtractionSelectionError(
            f"extraction record raw_sha256={raw_sha256!r} extraction_sha256={extraction_sha256!r} has "
            f"digest-authentic but unparseable extracted.json: {exc}"
        ) from exc
    actual_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    if actual_text_sha256 != meta.extracted_text_sha256:
        raise ExtractionSelectionError(
            f"extraction record raw_sha256={raw_sha256!r} extraction_sha256={extraction_sha256!r} does not "
            f"authenticate: parsed extracted.json text hashes to {actual_text_sha256!r}, but meta.json "
            f"records extracted_text_sha256={meta.extracted_text_sha256!r}"
        )
    return extracted


def select_extraction(
    workspace_root: Path,
    raw_sha256: str,
    *,
    prefer: ExtractionPreference,
    extraction_sha256: str | None = None,
) -> SelectedExtraction:
    """Resolve exactly the extraction the caller asked for, or raise.

    This is the ONLY entry point in this module that decides which of an
    artifact's several possible extractions (the root sidecar, plus zero or
    more content-addressed records under ``extractions/``) a consumer reads.
    It is deliberately explicit and caller-stated: every path either returns
    the extraction the caller asked for, or raises
    :exc:`ExtractionSelectionError` (or :exc:`ValueError` for a malformed
    ``raw_sha256``/``extraction_sha256`` argument). There is no path that
    silently substitutes a different extraction than the one requested --
    not the root sidecar, not "the newest record", not "the first current
    record". See :class:`ExtractionPreference` for the three policies.

    Args:
        workspace_root: Root campaign workspace.
        raw_sha256: sha256 of the parent artifact.
        prefer: which policy to resolve under.
        extraction_sha256: the extraction record's own content address.
            REQUIRED for :attr:`ExtractionPreference.EXACT`; FORBIDDEN
            (raises -- the request is ambiguous) for
            :attr:`ExtractionPreference.ROOT` and
            :attr:`ExtractionPreference.CURRENT`.

    Returns:
        The one :class:`SelectedExtraction` the caller asked for.

    Raises:
        ValueError: ``raw_sha256`` (or, for ``EXACT``, ``extraction_sha256``)
            is not a well-formed 64-lowercase-hex digest.
        ExtractionSelectionError: no single extraction can be resolved under
            the requested policy -- see :class:`ExtractionPreference` and
            this module's docstring for the exact per-policy rules.
    """
    _validate_sha(raw_sha256, label="raw_sha256")

    if prefer is ExtractionPreference.ROOT:
        if extraction_sha256 is not None:
            raise ExtractionSelectionError(
                f"ambiguous request: prefer=ROOT for raw_sha256={raw_sha256!r} was given "
                f"extraction_sha256={extraction_sha256!r}, but ROOT never takes an extraction_sha256"
            )
        root_extracted = load_artifact_text(workspace_root, raw_sha256)
        if root_extracted is None:
            raise ExtractionSelectionError(
                f"prefer=ROOT for raw_sha256={raw_sha256!r}: no readable root extracted.json sidecar "
                "is stored for this artifact"
            )
        return SelectedExtraction(extraction_id=ROOT_EXTRACTION_ID, extracted=root_extracted)

    if prefer is ExtractionPreference.EXACT:
        if extraction_sha256 is None:
            raise ExtractionSelectionError(
                f"prefer=EXACT for raw_sha256={raw_sha256!r} requires an extraction_sha256, but None "
                "was given"
            )
        _validate_sha(extraction_sha256, label="extraction_sha256")
        meta = load_extraction_record(workspace_root, raw_sha256, extraction_sha256)
        if meta is None:
            raise ExtractionSelectionError(
                f"prefer=EXACT for raw_sha256={raw_sha256!r} extraction_sha256={extraction_sha256!r}: no "
                "extraction record is stored at that address (or it does not authenticate to it); EXACT "
                "never falls back to the root sidecar or to any other record"
            )
        extracted = _load_authenticated_record_text(workspace_root, raw_sha256, extraction_sha256, meta)
        return SelectedExtraction(extraction_id=extraction_sha256, extracted=extracted)

    if prefer is ExtractionPreference.CURRENT:
        if extraction_sha256 is not None:
            raise ExtractionSelectionError(
                f"ambiguous request: prefer=CURRENT for raw_sha256={raw_sha256!r} was given "
                f"extraction_sha256={extraction_sha256!r}, but CURRENT never takes an extraction_sha256"
            )
        candidates = current_extraction_records(workspace_root, raw_sha256)
        if not candidates:
            raise ExtractionSelectionError(
                f"prefer=CURRENT for raw_sha256={raw_sha256!r}: no extraction record is current for "
                "today's extractor identity (either no records exist, or none of the ones that do match "
                "today's extractor_code_sha256/pypdf_version); CURRENT never falls back to the root sidecar"
            )
        if len(candidates) > 1:
            addresses = ", ".join(sorted(record.extraction_sha256 for record in candidates))
            raise ExtractionSelectionError(
                f"prefer=CURRENT for raw_sha256={raw_sha256!r}: {len(candidates)} extraction records are "
                f"current at once ({addresses}); 'current' is never used to pick a winner among several -- "
                "this is a refusal, not a ranking problem"
            )
        (winner,) = candidates
        extracted = _load_authenticated_record_text(
            workspace_root, raw_sha256, winner.extraction_sha256, winner
        )
        return SelectedExtraction(extraction_id=winner.extraction_sha256, extracted=extracted)

    raise ExtractionSelectionError(f"unknown ExtractionPreference: {prefer!r}")


class CurrentSelectionKind(StrEnum):
    """Which of the mutually exclusive facts one selector scan established.

    The point of naming them is that exactly ONE of them --
    :attr:`NO_RECORDS_STORED` -- licenses a caller to go on and judge the artifact by
    its root sidecar. Every other member is a refusal. Collapsing them (to a count, to
    an empty list, to a caught exception) is what let five different situations all
    present as "nothing to prefer here" and quietly serve unauthenticated text.
    """

    SELECTED = "selected"
    """Exactly one authenticated current record, whose body also authenticated."""

    NO_RECORDS_STORED = "no_records_stored"
    """No ``extractions/`` directory at all. The ONLY member that may fall through."""

    NO_CURRENT_RECORD = "no_current_record"
    """Records exist; none matches today's extractor identity. Re-extraction is the fix."""

    MULTIPLE_CURRENT_RECORDS = "multiple_current_records"
    """Several records are current at once: the store is ambiguous, not broken."""

    UNUSABLE_RECORD_PRESENT = "unusable_record_present"
    """At least one sha-shaped entry could not be read. See the warning below."""

    STORE_UNREADABLE = "store_unreadable"
    """``extractions/`` exists but could not be enumerated at all."""

    EXTRACTOR_IDENTITY_UNAVAILABLE = "extractor_identity_unavailable"
    """Today's extractor identity is unknowable, so currentness cannot be decided."""

    RECORD_AUTHENTICATION_FAILED = "record_authentication_failed"
    """One current record was found and it failed to authenticate."""


@dataclass(frozen=True)
class CurrentSelection:
    """The typed outcome of :func:`select_current_extraction`.

    Carries the decision AND the evidence for it, so a caller never has to re-derive
    one from a count or an exception message. ``selected`` is populated if and only if
    ``kind`` is :attr:`CurrentSelectionKind.SELECTED`.
    """

    kind: CurrentSelectionKind
    detail: str
    """Operator-facing phrase saying what was observed and what would change it."""

    selected: SelectedExtraction | None = None


def select_current_extraction(workspace_root: Path, raw_sha256: str) -> CurrentSelection:
    """Decide, in ONE scan, whether this artifact has a usable current extraction record.

    Replaces the count-then-select pair a caller previously had to write by hand. That
    shape had two defects beyond the obvious TOCTOU window between the two scans:

    - It read a decision out of an exception. ``ExtractionSelectionError`` means both
      "not exactly one current record" (ordinary) and "the record failed to
      authenticate" (a refusal), so telling them apart meant matching message prose.
    - It could reach a WRONG count. A sha-shaped entry whose ``meta.json`` is corrupt
      was skipped silently, so two current records became one, and an ambiguity that
      must refuse instead SELECTED the survivor. Corrupting a single file therefore
      PROMOTED a read -- the opposite direction from the downgrade the caller was
      guarding, and reachable by anyone who can delete a byte.

    Hence: any unusable sha-shaped candidate blocks selection outright, rather than
    merely not counting. The result reflects the store as observed during this one
    scan; a record appearing afterwards is not excluded (see the module docstring on
    why full closure needs lock discipline this store does not have).

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: sha256 of the parent raw artifact.

    Returns:
        A :class:`CurrentSelection`. Never raises for a store-content problem.

    Raises:
        ValueError: If ``raw_sha256`` is not a well-formed 64-character lowercase hex
            digest, or if the resolved directory would fall outside the workspace root.
    """
    scan = scan_extraction_records(workspace_root, raw_sha256)

    if scan.dir_state is RecordsDirState.UNLISTABLE:
        return CurrentSelection(
            kind=CurrentSelectionKind.STORE_UNREADABLE,
            detail=(
                f"the extractions directory for {raw_sha256} exists but could not be listed "
                f"({scan.unlistable_reason}); the store has reported nothing about this artifact, "
                "which is not the same as reporting that it has no records"
            ),
        )
    if scan.problems:
        described = ", ".join(f"{p.entry_name[:12]} ({p.reason})" for p in scan.problems)
        return CurrentSelection(
            kind=CurrentSelectionKind.UNUSABLE_RECORD_PRESENT,
            detail=(
                f"{len(scan.problems)} extraction record(s) for {raw_sha256} could not be read: "
                f"{described}. Each is a CANDIDATE record, so none of them may be silently "
                "dropped from the count -- doing so could turn an ambiguous store into an "
                "apparently unambiguous one and unlock a read"
            ),
        )
    if scan.dir_state is RecordsDirState.ABSENT or not scan.records:
        return CurrentSelection(
            kind=CurrentSelectionKind.NO_RECORDS_STORED,
            detail=f"no extraction record has ever been stored for {raw_sha256}",
        )

    identity = extraction_identity()
    if identity.pypdf_version == _PYPDF_VERSION_UNKNOWN and any(
        record.extractor in _PYPDF_DEPENDENT_EXTRACTORS for record in scan.records
    ):
        # Distinguished from NO_CURRENT_RECORD deliberately. Both present as "nothing is
        # current", but this one is a fact about the ENVIRONMENT, not the documents:
        # every pypdf-extracted record in the campaign stops being current at once, and
        # telling the operator their documents are stale would send them to re-extract
        # when what they need is to fix their pypdf install.
        return CurrentSelection(
            kind=CurrentSelectionKind.EXTRACTOR_IDENTITY_UNAVAILABLE,
            detail=(
                f"{raw_sha256} has pypdf-extracted records, but the installed pypdf version could "
                "not be determined, so whether any of them is current is unknowable. This is an "
                "environment problem, not a stale-document problem: fix the pypdf install"
            ),
        )

    current = [record for record in scan.records if _is_current(record, identity)]
    if not current:
        return CurrentSelection(
            kind=CurrentSelectionKind.NO_CURRENT_RECORD,
            detail=(
                f"{raw_sha256} has {len(scan.records)} extraction record(s), none matching today's "
                "extractor identity; re-extract it to produce one that does"
            ),
        )
    if len(current) > 1:
        addresses = ", ".join(sorted(record.extraction_sha256[:12] for record in current))
        return CurrentSelection(
            kind=CurrentSelectionKind.MULTIPLE_CURRENT_RECORDS,
            detail=(
                f"{raw_sha256} has {len(current)} records current at once ({addresses}); which one "
                "speaks for this document is ambiguous, and ambiguity is never resolved by ranking"
            ),
        )

    (winner,) = current
    try:
        extracted = _load_authenticated_record_text(
            workspace_root, raw_sha256, winner.extraction_sha256, winner
        )
    except ExtractionSelectionError as exc:
        return CurrentSelection(
            kind=CurrentSelectionKind.RECORD_AUTHENTICATION_FAILED,
            detail=(
                f"{raw_sha256}'s one current record ({winner.extraction_sha256[:12]}) failed to "
                f"authenticate: {exc}"
            ),
        )
    return CurrentSelection(
        kind=CurrentSelectionKind.SELECTED,
        detail=f"{raw_sha256} was read from record {winner.extraction_sha256[:12]}",
        selected=SelectedExtraction(extraction_id=winner.extraction_sha256, extracted=extracted),
    )
