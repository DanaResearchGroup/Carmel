# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Canonical serialization and content-addressed storage for literature dataset payloads.

Carmel stores experimental kinetics datasets extracted from papers as
content-addressed JSON under ``evidence/datasets/``. The cardinal project rule is
that every load-bearing number must be grounded against stored bytes and
auditable -- which means the on-disk representation of a dataset must be a pure,
deterministic function of its content, never of incidental things like dict
insertion order, float rounding, or platform-specific ``repr`` behaviour.

This module builds ONLY the foundation:

1. :func:`canonical_json_bytes` -- the single canonical JSON serializer every
   dataset write and every address computation goes through.
2. :func:`canonical_decimal` -- canonicalizes a numeric string via
   :class:`decimal.Decimal` so numeric facts travel as exact decimal strings
   (never ``float``) and their significance (trailing zeros) is preserved.
3. An addressed store over ``evidence/datasets/<sha256>.json`` --
   :func:`compute_dataset_sha`, :func:`store_dataset`, :func:`load_dataset`,
   :func:`verify_dataset`, :func:`list_datasets`.

Deliberately NOT built here: any dataset schema (pydantic model, field list,
required-vs-optional structure). That is a later milestone; this module only
knows how to hash and store an arbitrary ``dict`` payload the caller has already
constructed.

Why floats are rejected outright: a float that entered a payload as e.g. ``1.23``
in one Python build and printed as ``1.2300000000000001`` in another (or on a
different platform, or after a numpy/pandas round-trip) would silently change a
dataset's content address -- the exact failure mode this store exists to
prevent. Callers must convert every numeric fact to a canonical decimal string
via :func:`canonical_decimal` *before* it ever reaches :func:`canonical_json_bytes`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from typing import Any

from carmel.paths import normalize_path
from carmel.services.artifacts import _atomic_write_bytes  # noqa: PLC2701 - same-package reuse

__all__ = [
    "DATASET_STORE_DIR",
    "StoredDataset",
    "canonical_decimal",
    "canonical_json_bytes",
    "compute_dataset_sha",
    "dataset_path",
    "list_datasets",
    "load_dataset",
    "store_dataset",
    "verify_dataset",
]

DATASET_STORE_DIR = "evidence/datasets"

# Mirrors `evidence.py`'s `_SHA256_RE`, but is intentionally NOT imported from there:
# that pattern is a module-private detail of the `evidence/literature/<sha>/` layout,
# and this store has its own layout (`evidence/datasets/<sha>.json`). Duplicating a
# five-character regex is cheaper than coupling two unrelated storage layouts.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalJsonError(ValueError):
    """Raised when a payload cannot be serialized to canonical JSON.

    A subclass of ``ValueError`` (not a bare ``ValueError``) so callers can catch
    it specifically while still satisfying any code that catches ``ValueError``
    broadly.
    """


def _validate_json_value(value: Any, *, path: str) -> None:
    """Recursively validate ``value`` is safe for canonical JSON serialization.

    Walks the structure BEFORE handing it to ``json.dumps`` so that violations
    are caught explicitly, with a precise location, rather than relying on
    ``json.dumps`` to either silently coerce them (non-string dict keys are
    stringified silently by ``json.dumps`` otherwise) or to accept a float that
    would churn the resulting hash across platforms/Python versions.

    Args:
        value: The (sub-)value being validated.
        path: A human-readable location string (e.g. ``"root['a'][0]"``) used in
            error messages so a violation deep in a large payload is easy to find.

    Raises:
        CanonicalJsonError: If ``value`` (or anything nested in it) is a float,
            a non-string dict key, or any type ``json.dumps`` cannot represent
            losslessly without a ``default=`` fallback.
    """
    # `float` is checked BEFORE the `(str, bool, int)` tuple even though `float` is
    # not a subclass of `int` in Python (unlike `bool`, which is), so this ordering
    # is not load-bearing for correctness -- it just keeps the float rejection as
    # the first, most visible check in this function.
    if isinstance(value, float):
        raise CanonicalJsonError(
            f"float not allowed at {path}: {value!r} -- convert to a canonical decimal "
            "string via canonical_decimal() before serializing"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"non-string dict key not allowed at {path}: {key!r} ({type(key).__name__})")
            _validate_json_value(sub_value, path=f"{path}[{key!r}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise CanonicalJsonError(f"unsupported type at {path}: {type(value).__name__} ({value!r})")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` to the single canonical JSON byte form.

    Rules (all load-bearing -- this is the one function every dataset address
    and every dataset write goes through, so any deviation here silently changes
    every dataset's content address):

    - UTF-8 encoded, ``sort_keys=True``, compact separators (``","``/``":"``),
      ``ensure_ascii=False`` (non-ASCII text is kept literal, never ``\\uXXXX``
      escaped), plus exactly one trailing ``"\\n"``.
    - Deterministic: independent of input key order, and identical across runs.
    - Floats are rejected outright, anywhere in the structure (top-level, nested
      in a list, nested in a dict value) -- never coerced, rounded, or
      stringified silently. Numeric facts must arrive as canonical decimal
      strings (see :func:`canonical_decimal`) chosen by the caller before they
      ever reach this function.
    - Non-string dict keys are rejected outright, rather than silently
      stringified by ``json.dumps`` (e.g. an int key ``1`` silently becoming the
      string key ``"1"``).
    - Anything ``json`` cannot represent losslessly (arbitrary objects) is
      rejected -- this function never passes a ``default=`` fallback to
      ``json.dumps``.

    Args:
        payload: The value to serialize. Typically a ``dict`` at the top level,
            but any JSON-safe value is accepted.

    Returns:
        The canonical UTF-8-encoded JSON bytes, ending in a single ``"\\n"``.

    Raises:
        CanonicalJsonError: If ``payload`` (or anything nested in it) contains a
            float, a non-string dict key, or a type ``json`` cannot represent
            losslessly.
    """
    _validate_json_value(payload, path="root")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


class CanonicalDecimalError(ValueError):
    """Raised when a numeric string cannot be canonicalized.

    A subclass of ``ValueError`` (not a bare ``ValueError``) so callers can catch
    it specifically while still satisfying any code that catches ``ValueError``
    broadly. Raised rather than returning a sentinel (e.g. ``"nan"`` or
    ``None``) for invalid input, so a caller can never mistake a rejected value
    for a successfully canonicalized one.
    """


# Bound on a value's order of magnitude (``Decimal.adjusted()``: the exponent
# of its most significant digit), enforced by canonical_decimal below.
#
# This was added after an empirical probe found that plain fixed-point
# rendering (the previous design) was unsound for exponent-form input:
# `format(Decimal("1E+1000000"), "f")` materializes a MILLION-character digit
# string. PDF-extracted numeric text is exactly the kind of input that can
# carry a corrupted/garbage exponent (OCR turning "1.0e3" into "1.0e3000", for
# example) -- a store that will happily expand that into a multi-megabyte
# string is a memory-exhaustion vector sitting on an ingestion path, not a
# theoretical concern. 1000 orders of magnitude is enormously generous for
# real experimental kinetics data (Avogadro's number is ~1e24, Boltzmann's
# constant in SI is ~1e-23 -- both comfortably inside +/-1000) while still
# rejecting obviously-corrupted exponents loudly instead of silently building
# an enormous string.
_MAX_ADJUSTED_EXPONENT_MAGNITUDE = 1000


def canonical_decimal(text: str) -> str:
    """Canonicalize a numeric string via :class:`decimal.Decimal`.

    Design choices, both deliberate and both load-bearing:

    - **Trailing zeros are preserved, never collapsed.** ``.normalize()`` is
      deliberately NEVER called on the parsed ``Decimal``: ``.normalize()``
      strips trailing zeros, which would silently collapse ``"1.2300"`` (5
      significant figures) and ``"1.23"`` (3 significant figures) into the same
      output. In experimental kinetics data, trailing zeros in a reported value
      encode the precision the original measurement actually claimed -- that is
      a genuinely different fact from a value reported with fewer digits, and
      the two must hash differently. A future reader must NOT "optimize" this
      into a ``.normalize()`` call; doing so would silently merge distinct
      measurements onto the same address.
    - **Canonical form is Python's own ``str(Decimal)`` rendering.** An earlier
      version of this function rendered via plain fixed-point expansion
      (``format(d, "f")``) to stay human-readable without exponents. That was
      wrong on three counts, all found by direct probing of the
      implementation rather than by inspection: (1) it silently collapsed
      exponent-form input onto the same output as an equal-value integer with
      MORE apparent digits -- ``"1E+3"`` (1 significant figure) and ``"1000"``
      (up to 4 significant figures as literally written) both rendered as
      ``"1000"``, destroying exactly the kind of significance distinction this
      function exists to preserve; (2) ``format(Decimal("1E+1000000"), "f")``
      materializes a million-character string -- unbounded memory blowup
      driven entirely by the exponent, independent of how few digits the
      input actually had; (3) it destroyed significant-figure information for
      ordinary large magnitude values written in scientific notation, e.g. a
      combustion A-factor of ``"7.000E+17"`` (4 significant figures) expanded
      to the 18-digit integer ``"700000000000000000"``, which reads as having
      far more precision than was actually reported.

      A follow-up attempt fixed all three by hand-encoding the raw ``(sign,
      digits, exponent)`` triple from ``Decimal.as_tuple()`` as
      ``[-]<digits>E<+/-exponent>``. That was CORRECT (verified: significance
      preserved, bounded, idempotent) but gratuitous -- it reinvented exactly
      what ``str(Decimal)`` already does. ``str(Decimal("1E+3"))`` ==
      ``"1E+3"``, ``str(Decimal("1.0E+3"))`` == ``"1.0E+3"``,
      ``str(Decimal("1000"))`` == ``"1000"`` -- all three stay distinct
      because ``str()`` also round-trips ``as_tuple()`` exactly, with no
      normalization step. It is equally bounded: ``str(Decimal("1E+1000000"))``
      is ``"1E+1000000"``, ~10 characters, never expanded -- ``str()`` never
      switches TO plain notation for a large positive exponent, only ever
      switches OUT of scientific notation for small-magnitude values within a
      fixed adjusted-exponent window, so the "expand a huge exponent into a
      huge string" failure mode the hand-rolled encoding was built to dodge
      does not exist on this path either. And unlike the hand-rolled digit
      concatenation, ``str(Decimal)`` keeps a value like ``"7.000E+17"``
      rendered AS ``"7.000E+17"`` -- exactly as printed in the source paper,
      not re-encoded into an unfamiliar ``"7000E+14"`` form a reviewer would
      have to decode. Since the whole reason this store uses per-dataset JSON
      text rather than a binary format is that the stored bytes stay directly
      human-auditable, preserving the paper's own notation is a real
      advantage the hand-rolled triple encoding gave up for nothing.
    - **Exponent magnitude is still bounded**, independently of the rendering
      choice above. Values whose order of magnitude (``Decimal.adjusted()``)
      exceeds ``_MAX_ADJUSTED_EXPONENT_MAGNITUDE`` are rejected outright --
      see that constant's comment for why 1000 orders of magnitude is both
      generous for real data and a defensible line against corrupted/garbage
      exponents. This check stands on its own merits (loud rejection of
      obviously-corrupted OCR/extraction input) even though, per the note
      above, ``str(Decimal)`` itself no longer has an unbounded-output failure
      mode for it to guard against.
    - **Equal value + equal significance -> identical output.** A redundant
      leading ``"+"`` sign is not itself precision information, so
      ``canonical_decimal("+1.50") == canonical_decimal("1.50")``. This function
      never changes a value's magnitude or its number of significant figures --
      only its sign presentation is normalized (``Decimal`` parsing itself
      already drops a redundant leading ``"+"``, so no extra step is needed
      here).
    - **Idempotent**: ``canonical_decimal(canonical_decimal(x)) ==
      canonical_decimal(x)`` for any valid ``x``, because ``str(Decimal(...))``
      round-trips through the ``Decimal`` constructor to the exact same
      ``(sign, digits, exponent)`` tuple, so re-canonicalizing it is a no-op.

    Args:
        text: The raw numeric string to canonicalize.

    Returns:
        The canonical decimal string, as rendered by ``str(Decimal(text))``.

    Raises:
        CanonicalDecimalError: If ``text`` is empty, whitespace-only, one of the
            non-finite spellings ``"nan"``, ``"inf"``, ``"-inf"`` (case-insensitive,
            matching what :class:`decimal.Decimal` itself accepts), anything else
            :class:`decimal.Decimal` cannot parse, or a value whose order of
            magnitude exceeds ``_MAX_ADJUSTED_EXPONENT_MAGNITUDE``.
    """
    if not text or not text.strip():
        raise CanonicalDecimalError(f"cannot canonicalize empty/whitespace-only string: {text!r}")
    try:
        d = Decimal(text)
    except (InvalidOperation, DecimalException) as exc:
        raise CanonicalDecimalError(f"cannot parse {text!r} as a decimal: {exc}") from exc
    if not d.is_finite():
        raise CanonicalDecimalError(f"non-finite decimal not allowed: {text!r}")
    if abs(d.adjusted()) > _MAX_ADJUSTED_EXPONENT_MAGNITUDE:
        raise CanonicalDecimalError(
            f"magnitude out of range (|adjusted exponent| > {_MAX_ADJUSTED_EXPONENT_MAGNITUDE}): {text!r}; "
            "this is far beyond any real experimental kinetics value and is treated as corrupted input "
            "(e.g. OCR-mangled exponent) rather than canonicalized"
        )
    return str(d)


@dataclass(frozen=True)
class StoredDataset:
    """Minimal record of a dataset persisted by :func:`store_dataset`.

    Deliberately minimal: the real dataset schema (pydantic model, field
    validation) is a later milestone. This record only needs to tell a caller
    where the bytes it just stored (or found already stored) live.
    """

    sha256: str
    """The content address: sha256 hex digest of the canonical JSON bytes."""

    path: Path
    """The resolved on-disk path of the stored ``<sha256>.json`` file."""


def dataset_path(root: Path, dataset_sha: str) -> Path:
    """Compute the content-addressed path for ``dataset_sha`` under ``root``.

    Validates ``dataset_sha`` is exactly 64 lowercase hex characters before ever
    interpolating it into a filesystem path -- without that check, a caller
    could pass something like ``"../../etc/passwd"`` and have it walked straight
    into a path. ``root`` is resolved via :func:`carmel.paths.normalize_path` and
    the resulting path's containment under the resolved root is asserted as
    defence in depth (mirrors ``carmel.services.evidence``'s
    ``_validate_sha256``/``_assert_contained`` pair, which validates the
    ``evidence/literature/<sha>/`` layout -- this validates the sibling
    ``evidence/datasets/<sha>.json`` layout instead).

    Args:
        root: Root of the campaign workspace.
        dataset_sha: Hex digest identifying the dataset.

    Returns:
        The resolved ``<root>/evidence/datasets/<dataset_sha>.json`` path.

    Raises:
        ValueError: If ``dataset_sha`` is not exactly 64 lowercase hex
            characters, or if the resolved path would fall outside the resolved
            workspace root.
    """
    if not _SHA256_RE.match(dataset_sha):
        raise ValueError(f"invalid dataset sha256: {dataset_sha!r} (expected 64 lowercase hex characters)")
    resolved_root = normalize_path(root)
    candidate = resolved_root / DATASET_STORE_DIR / f"{dataset_sha}.json"
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"refusing to use path outside workspace root: {resolved} not under {resolved_root}")
    return resolved


def compute_dataset_sha(identity_payload: dict[str, Any]) -> str:
    """Compute the content address of ``identity_payload``.

    ``identity_payload`` must be a plain, deliberately-constructed ``dict`` --
    this function does NOT accept, detect, or special-case a pydantic model
    (e.g. by calling ``.model_dump()`` internally). That restriction is
    deliberate: if this function silently unwrapped a model, an unrelated
    change to that model's field shape/order/defaults could silently rename
    the address of every already-stored dataset, with no caller ever having
    asked for that. Callers own translating whatever schema they use into the
    plain dict that defines identity.

    Args:
        identity_payload: The plain dict whose canonical JSON form defines this
            dataset's content address.

    Returns:
        The sha256 hex digest of ``identity_payload``'s canonical JSON bytes.

    Raises:
        CanonicalJsonError: If ``identity_payload`` is not representable as
            canonical JSON (see :func:`canonical_json_bytes`).
    """
    return hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()


def store_dataset(root: Path, identity_payload: dict[str, Any]) -> StoredDataset:
    """Content-address and durably persist ``identity_payload``.

    Computes the sha256 of the canonical JSON bytes and writes them atomically
    to ``<root>/evidence/datasets/<sha256>.json``.

    Idempotent: if the target file already exists, its on-disk bytes are read
    and compared for EXACT equality with the canonical bytes about to be
    written.

    - Equal -> no rewrite; the existing record is returned unchanged (matches
      ``carmel.services.evidence.store_artifact``'s re-store behaviour).
    - Not equal -> raises. Two payloads that canonicalize to different bytes
      but hash to the same sha256 would be a genuine sha256 collision; more
      likely in practice, this indicates on-disk corruption (bit rot, a partial
      write, tampering). Either way, silently overwriting would destroy the
      auditability this store exists to provide, so this is always a hard
      failure rather than a silent repair.

    Args:
        root: Root of the campaign workspace.
        identity_payload: The plain dict to store. See :func:`compute_dataset_sha`
            for why this must be a plain dict, never a pydantic model.

    Returns:
        The :class:`StoredDataset` record (freshly written, or pre-existing and
        verified byte-identical).

    Raises:
        CanonicalJsonError: If ``identity_payload`` is not representable as
            canonical JSON.
        ValueError: If the on-disk bytes at the target path exist and differ
            from the canonical bytes about to be written (hash collision or
            on-disk corruption).
    """
    canonical_bytes = canonical_json_bytes(identity_payload)
    sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    path = dataset_path(root, sha256)

    if path.exists():
        on_disk = path.read_bytes()
        if on_disk == canonical_bytes:
            return StoredDataset(sha256=sha256, path=path)
        raise ValueError(
            f"dataset store collision or corruption at {path}: on-disk bytes do not match "
            f"the canonical bytes for sha256 {sha256!r}; refusing to overwrite"
        )

    _atomic_write_bytes(path, canonical_bytes)
    return StoredDataset(sha256=sha256, path=path)


def load_dataset(root: Path, sha256: str) -> dict[str, Any]:
    """Load and parse the dataset stored under ``sha256``.

    Args:
        root: Root of the campaign workspace.
        sha256: Hex digest identifying the dataset.

    Returns:
        The parsed dataset payload.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved path would fall outside the resolved
            workspace root.
        FileNotFoundError: If no dataset is stored under ``sha256``.
    """
    path = dataset_path(root, sha256)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def verify_dataset(root: Path, sha256: str) -> bool:
    """Re-read the stored bytes and confirm they still hash to ``sha256``.

    Because the canonical JSON form IS the on-disk form for this store (there
    is no pydantic model or other layer in between), this recomputes the digest
    straight from the on-disk bytes with NO re-serialization step -- there is no
    round-trip through a model or re-canonicalization that could itself mask or
    introduce drift. That is the whole point of keeping this store schema-free:
    the bytes on disk are exactly what gets hashed, both at store time and at
    verify time.

    Args:
        root: Root of the campaign workspace.
        sha256: Hex digest identifying the dataset (and its filename).

    Returns:
        True if the file exists and its bytes hash to ``sha256``; False if the
        file is absent or its bytes have been altered/corrupted.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved path would fall outside the resolved
            workspace root.
    """
    path = dataset_path(root, sha256)
    if not path.exists():
        return False
    on_disk = path.read_bytes()
    return hashlib.sha256(on_disk).hexdigest() == sha256


def list_datasets(root: Path) -> list[str]:
    """List every dataset sha256 currently held in this workspace's dataset store.

    Filenames that don't match the ``<64-hex>.json`` shape are skipped rather
    than raising -- the store is content-addressed and append-only, but a
    directory can end up holding stray files (mirrors
    ``carmel.services.evidence.list_artifacts_with_unreadable``'s "skip what
    doesn't parse" posture; this is simpler since there is no metadata to parse,
    just a filename shape to check).

    Args:
        root: Root of the campaign workspace.

    Returns:
        Sorted list of sha256 hex digests for every validly named dataset file
        found. Empty list if the dataset directory does not exist.
    """
    resolved_root = normalize_path(root)
    datasets_dir = resolved_root / DATASET_STORE_DIR
    try:
        entries = list(datasets_dir.iterdir())
    except OSError:
        return []

    shas: list[str] = []
    for entry in entries:
        if not entry.is_file() or entry.suffix != ".json":
            continue
        stem = entry.stem
        if _SHA256_RE.match(stem):
            shas.append(stem)
    return sorted(shas)
