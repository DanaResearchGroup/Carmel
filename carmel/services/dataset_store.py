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
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from typing import Any

from carmel.paths import normalize_path
from carmel.services.numeric import GlyphHealth, SourceContext, Unresolvable, normalize_numeric_span

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


# Depth bound for _validate_json_value's recursion. Load-bearing for two distinct
# reasons, both of which must hold at once:
#
# 1. This validator itself recurses once per nesting level, in pure Python, against
#    the interpreter's own call-stack limit (default 1000). Left unbounded, a
#    deeply-nested-but-otherwise-valid payload raises a bare ``RecursionError``
#    instead of the ``CanonicalJsonError`` this module's contract promises for every
#    other rejection -- and building that nesting is a cheap way to make ANY caller
#    of this validator crash, not just fail cleanly.
# 2. Even a payload that clears this validator still has to survive ``json.dumps``,
#    which recurses too. 500 was chosen (and is verified by a test) to leave enough
#    of the interpreter's 1000-frame budget free -- above whatever frames the
#    validator's own call chain and the caller's existing stack have already used --
#    that ``json.dumps`` cannot then blow the recursion limit on a payload that
#    passed validation at this depth.
_MAX_JSON_DEPTH = 500

# Digit-count bound for int values. See the comment on `isinstance(value, int)`
# below for why int gets a magnitude bound rather than the outright ban `float`
# gets -- the failure modes are not the same and the fix is not the same.
_MAX_INT_DIGITS = 1000


def _validate_json_value(value: Any, *, path: str, depth: int = 0) -> None:
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
        depth: Current nesting depth, counted from the root (depth 0). Callers
            outside this module never pass this; it is incremented on each
            recursive call into a dict value or list item.

    Raises:
        CanonicalJsonError: If ``value`` (or anything nested in it) is a float,
            a non-string dict key, an int whose magnitude exceeds
            ``_MAX_INT_DIGITS`` digits, any type ``json.dumps`` cannot represent
            losslessly without a ``default=`` fallback, or if ``value`` nests
            deeper than ``_MAX_JSON_DEPTH`` levels.
    """
    if depth > _MAX_JSON_DEPTH:
        raise CanonicalJsonError(
            f"value nested too deeply at {path} (> {_MAX_JSON_DEPTH} levels); this is treated as "
            "corrupted or adversarial input rather than a real dataset shape, and is rejected here "
            "rather than being allowed to blow this validator's or json.dumps's recursion limit"
        )
    # `float` is checked BEFORE the `(str, bool, int)` tuple even though `float` is
    # not a subclass of `int` in Python (unlike `bool`, which is), so this ordering
    # is not load-bearing for correctness -- it just keeps the float rejection as
    # the first, most visible check in this function.
    if isinstance(value, float):
        raise CanonicalJsonError(
            f"float not allowed at {path}: {value!r} -- convert to a canonical decimal "
            "string via canonical_decimal() before serializing"
        )
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        # Must be checked before `int` below: `bool` is a subclass of `int` in
        # Python, but a bool has no "magnitude" in the sense the int digit-count
        # bound below is guarding against, so it is exempt from that check.
        return
    if isinstance(value, int):
        # Unlike `float`, `int` is bounded here, NOT banned -- these are different
        # failure modes with different fixes. The float ban exists because float
        # *repr* is unstable: the same value can print differently across
        # platforms/interpreter versions, which would churn this store's content
        # addresses. `int` has none of that problem -- Python's int-to-str
        # conversion is exact and stable, so `1` always serializes as `"1"`,
        # everywhere, forever; there is no repr instability to guard against, and
        # banning int would also break real schema fields that legitimately
        # serialize to int (e.g. page rotation, table cell row/col indices in
        # carmel.schemas.datasets). What IS real for int, and IS worth guarding,
        # is unbounded magnitude: Python ints are arbitrary-precision, so a
        # payload can carry a million-digit integer -- the same memory-blowup
        # class canonical_decimal's exponent-magnitude bound already guards
        # against for decimals, just via a different mechanism (digit count
        # instead of adjusted exponent, since int has no exponent).
        if len(str(abs(value))) > _MAX_INT_DIGITS:
            raise CanonicalJsonError(
                f"int magnitude out of range (> {_MAX_INT_DIGITS} digits) at {path}: "
                f"{value!r}; treated as corrupted/adversarial input rather than a real dataset value"
            )
        return
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"non-string dict key not allowed at {path}: {key!r} ({type(key).__name__})")
            _validate_json_value(sub_value, path=f"{path}[{key!r}]", depth=depth + 1)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
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
    - Ints are bounded (rejected past ``_MAX_INT_DIGITS`` digits) but not banned
      like floats -- see the comment in :func:`_validate_json_value` for why the
      two are not the same failure mode.
    - Nesting is bounded (rejected past ``_MAX_JSON_DEPTH`` levels) so a
      pathologically deep payload raises ``CanonicalJsonError`` instead of a bare
      ``RecursionError``.

    Args:
        payload: The value to serialize. Typically a ``dict`` at the top level,
            but any JSON-safe value is accepted.

    Returns:
        The canonical UTF-8-encoded JSON bytes, ending in a single ``"\\n"``.

    Raises:
        CanonicalJsonError: If ``payload`` (or anything nested in it) contains a
            float, a non-string dict key, an out-of-range int, a type ``json``
            cannot represent losslessly, or nests deeper than ``_MAX_JSON_DEPTH``.
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

# canonical_decimal has no document to assess for glyph corruption -- it receives a
# bare numeric string with no surrounding text -- so it cannot call
# assess_glyph_health on a real document. Constructed directly from the dataclass
# fields (all False/clean) rather than fabricating a document to feed through
# assess_glyph_health: the quarantine rule guarded by suspects_dash_corruption is a
# document-level EXTRACTION-time gate (is this PDF's text suspected of an en-dash ->
# ASCII 'e' substitution?), not a property that a standalone canonical decimal string
# could ever carry. SourceContext.OPERATOR_RAW is the matching choice on the other
# side of the call: it is the one context that never inherits or implies a PDF
# document's quarantine state (see SourceContext's docstring), which is exactly
# correct here since there is no document at all.
_HEALTHY_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)


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

    - **Routed through the strict numeric core first.** Before ``text`` ever reaches
      :class:`decimal.Decimal`, it is passed through
      :func:`carmel.services.numeric.normalize_numeric_span`, the same strict textual
      grammar the paper-extraction path uses. Bare ``Decimal(text)`` is, by design,
      far looser than that grammar: it silently accepts digit-separator input like
      ``"1_000"`` (-> ``1000``), a leading-dot form like ``".5"`` (-> ``0.5``), and a
      trailing-dot form like ``"1."`` (-> ``1``) that the strict core refuses
      outright. Routing through the core closes that gap so a value that would never
      survive paper extraction cannot sneak into a canonical dataset via some other
      caller that types a numeric string by hand.
    - **A repaired input is refused, never silently repaired here.** If
      ``normalize_numeric_span`` had to apply a glyph repair (e.g. U+2212 MINUS SIGN
      -> ``"-"``) to make ``text`` parse, this function raises rather than accepting
      the repaired form. A canonical decimal string is the END of the numeric
      pipeline -- by the time a value reaches here, any repair it needed must already
      have happened upstream, at extraction time, where it can be recorded alongside
      its provenance. Silently repairing it again here would produce a canonical
      value with no record of the repair that produced it.
    - **``normalize_numeric_span`` is called with**
      ``source_context=SourceContext.OPERATOR_RAW`` **and a healthy, hand-constructed**
      ``GlyphHealth``. This function receives a bare numeric string with no
      surrounding document, so it has nothing to run
      :func:`~carmel.services.numeric.assess_glyph_health` on. The dash-corruption
      quarantine that assessment feeds is a document-level, EXTRACTION-time gate (is
      this PDF's flattened text suspected of an en-dash -> ASCII ``e`` substitution?)
      -- not a property a standalone decimal string can carry -- and
      ``OPERATOR_RAW`` is the one :class:`~carmel.services.numeric.SourceContext` that
      never inherits or implies that quarantine state, which matches: there is no
      document here at all.

    Args:
        text: The raw numeric string to canonicalize.

    Returns:
        The canonical decimal string, as rendered by ``str(Decimal(text))``.

    Raises:
        CanonicalDecimalError: If ``text`` fails the strict numeric core (empty,
            whitespace-only, a disallowed non-finite literal, a digit separator, an
            ASCII-6 uncertainty shape, a range, or any other shape
            :func:`~carmel.services.numeric.normalize_numeric_span` refuses); if it
            required a glyph repair to normalize; if the normalized text is one
            :class:`decimal.Decimal` itself cannot parse or evaluates to a non-finite
            decimal; or if its order of magnitude exceeds
            ``_MAX_ADJUSTED_EXPONENT_MAGNITUDE``.
    """
    normalized = normalize_numeric_span(
        text,
        source_context=SourceContext.OPERATOR_RAW,
        glyph_health=_HEALTHY_GLYPH_HEALTH,
    )
    if isinstance(normalized, Unresolvable):
        raise CanonicalDecimalError(f"cannot canonicalize {text!r}: {normalized.reason}")
    if normalized.repairs:
        raise CanonicalDecimalError(
            f"cannot canonicalize {text!r}: it required glyph repair(s) {normalized.repairs} to parse; "
            "a canonical decimal string is never produced by a silent repair -- the repair must happen "
            "upstream, at extraction time, where it can be recorded alongside its provenance"
        )
    try:
        d = Decimal(normalized.text)
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


# Reserved key namespace, owned exclusively by this store -- never by callers.
#
# `store_dataset`/`compute_dataset_sha` inject a marker recording which version
# of `canonical_decimal`'s rendering rules produced a payload's canonical bytes,
# under a key in this namespace, BEFORE canonicalizing -- so the injected form
# is what actually gets hashed AND written, and "canonical form IS the on-disk
# form" still holds exactly (see `_inject_decimal_repr_version`). This is what
# turns the standing six-month-rot risk into a migration instead: if
# `canonical_decimal`'s rendering ever changes, every already-stored dataset
# keeps proof of which rules produced it, rather than silently re-addressing
# with no way to tell old bytes from new.
#
# A payload that already contains any key in this namespace is rejected
# outright -- a caller can neither forge nor accidentally omit the marker this
# store alone is responsible for.
_RESERVED_KEY_PREFIX = "_carmel_"
_DECIMAL_REPR_VERSION_KEY = f"{_RESERVED_KEY_PREFIX}decimal_repr_version"
_DECIMAL_REPR_VERSION = 1
_RECOGNIZED_DECIMAL_REPR_VERSIONS = frozenset({_DECIMAL_REPR_VERSION})


def _inject_decimal_repr_version(identity_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``identity_payload`` with the reserved version marker added.

    Args:
        identity_payload: The caller-supplied payload.

    Returns:
        A new dict equal to ``identity_payload`` plus
        ``{_DECIMAL_REPR_VERSION_KEY: _DECIMAL_REPR_VERSION}``.

    Raises:
        ValueError: If ``identity_payload`` already contains a top-level key in
            the ``_RESERVED_KEY_PREFIX`` namespace -- that namespace is owned by
            this store, never by callers.
    """
    for key in identity_payload:
        if isinstance(key, str) and key.startswith(_RESERVED_KEY_PREFIX):
            raise ValueError(
                f"payload key {key!r} is in the {_RESERVED_KEY_PREFIX!r} namespace reserved for "
                "the dataset store itself (e.g. the decimal-representation version marker); "
                "callers must not set it directly"
            )
    return {**identity_payload, _DECIMAL_REPR_VERSION_KEY: _DECIMAL_REPR_VERSION}


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

    The reserved decimal-repr version marker is injected (see the
    ``_RESERVED_KEY_PREFIX`` module comment) before the address is computed, so
    the address itself is a function of both the payload AND the rendering
    rules that produced it.

    Args:
        identity_payload: The plain dict whose canonical JSON form defines this
            dataset's content address.

    Returns:
        The sha256 hex digest of ``identity_payload``'s canonical JSON bytes
        (with the version marker injected).

    Raises:
        ValueError: If ``identity_payload`` already contains a reserved-namespace
            key.
        CanonicalJsonError: If ``identity_payload`` is not representable as
            canonical JSON (see :func:`canonical_json_bytes`).
    """
    return hashlib.sha256(canonical_json_bytes(_inject_decimal_repr_version(identity_payload))).hexdigest()


def store_dataset(root: Path, identity_payload: dict[str, Any]) -> StoredDataset:
    """Content-address and durably persist ``identity_payload``.

    Injects the reserved decimal-repr version marker (see the module-level
    ``_RESERVED_KEY_PREFIX`` comment), computes the sha256 of the resulting
    canonical JSON bytes, and durably publishes them to
    ``<root>/evidence/datasets/<sha256>.json``.

    Idempotent, and race-safe against a concurrent writer targeting the same
    path: the canonical bytes are first written and fsynced to a private temp
    file, then published via an exclusive hard link (``os.link``), which
    atomically fails with ``FileExistsError`` if the target already exists
    rather than silently replacing it. This is deliberately NOT
    ``path.exists()`` followed by a write -- that check-then-act pair leaves a
    window in which a different or corrupt file can appear at ``path`` between
    the check and the write and be silently overwritten instead of raising as
    promised. ``os.link`` closes that window because "does the target already
    exist" and "create it" are a single atomic kernel operation. If the link
    fails because the target already exists, the on-disk bytes are read and
    compared for EXACT equality with the canonical bytes that were about to be
    published:

    - Equal -> no rewrite; the existing record is returned unchanged (matches
      ``carmel.services.evidence.store_artifact``'s re-store behaviour).
    - Not equal -> raises. Two payloads that canonicalize to different bytes
      but hash to the same sha256 would be a genuine sha256 collision; more
      likely in practice, this indicates on-disk corruption (bit rot, a partial
      write, tampering), or a second writer's file landing at ``path`` between
      this call's own checks. Either way, silently overwriting would destroy
      the auditability this store exists to provide, so this is always a hard
      failure rather than a silent repair.

    Args:
        root: Root of the campaign workspace.
        identity_payload: The plain dict to store. See :func:`compute_dataset_sha`
            for why this must be a plain dict, never a pydantic model.

    Returns:
        The :class:`StoredDataset` record (freshly written, or pre-existing and
        verified byte-identical).

    Raises:
        ValueError: If ``identity_payload`` already contains a reserved-namespace
            key, or if the on-disk bytes at the target path exist and differ
            from the canonical bytes about to be written (hash collision,
            on-disk corruption, or a lost race with a different writer).
        CanonicalJsonError: If ``identity_payload`` is not representable as
            canonical JSON.
    """
    canonical_bytes = canonical_json_bytes(_inject_decimal_repr_version(identity_payload))
    sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    path = dataset_path(root, sha256)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_bytes)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            on_disk = path.read_bytes()
            if on_disk == canonical_bytes:
                return StoredDataset(sha256=sha256, path=path)
            raise ValueError(
                f"dataset store collision or corruption at {path}: on-disk bytes do not match "
                f"the canonical bytes for sha256 {sha256!r}; refusing to overwrite"
            ) from None
    finally:
        tmp_path.unlink(missing_ok=True)

    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return StoredDataset(sha256=sha256, path=path)


def load_dataset(root: Path, sha256: str) -> dict[str, Any]:
    """Load, verify, and parse the dataset stored under ``sha256``.

    Recomputes the digest from the on-disk bytes and confirms they both hash to
    ``sha256`` and are already in canonical form (the same two checks
    :func:`verify_dataset` performs) before returning anything. A file that
    merely happens to be valid, tampered-but-still-parseable JSON would
    otherwise load "successfully" with content the filename never actually
    vouches for -- exactly the failure this store exists to prevent, and a
    caller should never have to remember to call :func:`verify_dataset` first
    to be safe. There is deliberately no ``verify=False`` escape hatch: an
    opt-out on a content-addressed load is a footgun, and a caller who
    genuinely wants raw, unverified bytes can read the file directly instead
    of going through this function.

    The reserved decimal-repr version marker (see the module-level
    ``_RESERVED_KEY_PREFIX`` comment) is stripped from the returned dict -- it
    is this store's own bookkeeping, not part of the payload the caller handed
    to :func:`store_dataset`.

    Args:
        root: Root of the campaign workspace.
        sha256: Hex digest identifying the dataset.

    Returns:
        The parsed dataset payload, with the reserved version marker removed.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase
            hex digest, if the resolved path would fall outside the resolved
            workspace root, if the on-disk bytes do not hash to ``sha256``, if
            they are not already the canonical encoding of their own parsed
            content, or if the decimal-repr version marker is missing or
            unrecognized.
        FileNotFoundError: If no dataset is stored under ``sha256``.
    """
    path = dataset_path(root, sha256)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"dataset at {path} is corrupted or has been tampered with: on-disk bytes hash to "
            f"{digest!r}, not the requested {sha256!r}; refusing to return unverified content"
        )
    parsed: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
    if canonical_json_bytes(parsed) != raw_bytes:
        raise ValueError(
            f"dataset at {path} hashes correctly but is not in canonical form: a hand-placed or "
            "hand-edited file can satisfy the sha256 without being the exact encoding this store "
            "guarantees -- refusing to return content that is merely self-consistent"
        )
    version = parsed.pop(_DECIMAL_REPR_VERSION_KEY, None)
    if version not in _RECOGNIZED_DECIMAL_REPR_VERSIONS:
        raise ValueError(
            f"dataset at {path} has a missing or unrecognized decimal-repr version marker "
            f"({version!r}); recognized versions are {sorted(_RECOGNIZED_DECIMAL_REPR_VERSIONS)!r}"
        )
    return parsed


def verify_dataset(root: Path, sha256: str) -> bool:
    """Re-read the stored bytes and confirm they are a genuine, canonical dataset.

    Verifies BOTH that the on-disk bytes hash to ``sha256`` AND that they are
    already the canonical encoding of their own parsed content, i.e.
    ``canonical_json_bytes(json.loads(raw_bytes)) == raw_bytes``, carrying a
    recognized decimal-repr version marker. The hash check alone is not
    enough: a hand-placed file named by the sha256 of some OTHER, non-canonical
    JSON bytes (different key order, different whitespace, a stale or missing
    version marker, ...) would satisfy a hash-only check while violating this
    module's "canonical form IS the on-disk form" contract -- this store never
    wrote that file, no matter how internally self-consistent it looks. Because
    there is no pydantic model or other layer between the canonical bytes and
    the file, this recomputes both the digest and the canonical re-encoding
    straight from the on-disk bytes, with no round-trip that could itself mask
    drift.

    Args:
        root: Root of the campaign workspace.
        sha256: Hex digest identifying the dataset (and its filename).

    Returns:
        True if the file exists, its bytes hash to ``sha256``, those bytes are
        already in canonical form, and they carry a recognized decimal-repr
        version marker. False otherwise -- including if the bytes are absent,
        altered/corrupted, not valid UTF-8 JSON, or not a JSON object at all.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved path would fall outside the resolved
            workspace root.
    """
    path = dataset_path(root, sha256)
    if not path.exists():
        return False
    raw_bytes = path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != sha256:
        return False
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError, json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    try:
        if canonical_json_bytes(parsed) != raw_bytes:
            return False
    except CanonicalJsonError:
        return False
    return parsed.get(_DECIMAL_REPR_VERSION_KEY) in _RECOGNIZED_DECIMAL_REPR_VERSIONS


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
