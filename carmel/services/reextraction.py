"""Re-extraction: append a fresh, independently re-parsed extraction record.

``store_artifact`` (``carmel/services/evidence.py``) has a short-circuit: when an
artifact is already intact on disk, it returns the *existing* metadata and silently
DISCARDS whatever freshly-extracted ``ExtractedText`` the caller just handed it
(``evidence.py:215-218``). That is correct behaviour for its own contract -- root
sidecars are meant to be written once, at first-fetch time, and never rewritten --
but it also means there is currently no way to record a genuine re-extraction (a
newer ``pypdf``, a bug fix in ``extract_text``, ...) against an artifact that is
already stored.

This module is that path. ``reextract_artifact`` re-reads an artifact's ``raw.bin``,
verifies those bytes against the content address (the directory name) itself --
never against a stored digest -- genuinely re-parses them with today's
``extract_text``, and appends the result as a new, separately-addressed extraction
record under ``evidence/literature/<raw_sha>/extractions/<extraction_sha>/`` via
``store_extraction_record()``. It is append-only: it never opens any of the four
ROOT files (``raw.bin``, ``text.txt``, ``extracted.json``, ``meta.json``) for
writing. ``raw.bin`` and ``meta.json`` are read-only inputs; ``text.txt`` and root
``extracted.json`` are never even read.

The load-bearing distinction from a naive implementation: this module MUST re-parse
``raw.bin`` with ``extract_text`` on every call. Mirroring the already-stored root
``extracted.json`` (as ``dataset_producer.produce_envelope_from_artifact`` does,
by its own admission) is NOT re-extraction -- it would silently persist the same
stale extraction under a new address and would pass a naive "did it append
something" test.

Root sidecars are DELIBERATELY left describing the OLD extraction after a call to
``reextract_artifact``. Nothing in this module reconciles ``meta.json``,
``text.txt``, or root ``extracted.json`` with the new record, and nothing ever
will: the correct replay anchor for anything built on top of a re-extraction is the
extraction record's own address (``extraction_sha256``), never the root sidecar.
Treat the resulting divergence between root and nested record as intended, not as
a bug to "fix" by writing the new extraction back over the root files.

On the serializer and "currentness": ``_canonical_extracted_json_bytes`` centralizes
the one place that replicates ``carmel.services.artifacts.write_json``'s encoding.
What that buys is narrow and mechanical -- the bytes this module persists cannot
accidentally drift from what a root ``write_json`` call would have produced for the
same ``ExtractedText``, because there is only one implementation of the encoding to
drift. It does NOT make a stored record's "currentness" well-defined, and does not
make serializer changes visible anywhere: ``current_extraction_records``
(``carmel/services/extraction_record.py``) compares only a record's
``extractor_code_sha256`` and ``pypdf_version`` against what ``extraction_identity()``
reports right now. Serializer output is not part of the record's identity payload or
its content address, and is not one of the fields ``current_extraction_records``
compares -- so a serializer-only change (e.g. different key order, different float
formatting) would mint a different ``extracted_sha256`` on the next re-extraction
while every already-stored record, produced by the OLD serializer, keeps reporting as
current. Centralizing the serializer prevents accidental byte drift between this path
and root writes; it says nothing about whether a given serializer's output is still
the one today's code would produce.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText, extract_text
from carmel.paths import normalize_path
from carmel.services.evidence import artifact_dir, load_artifact_meta
from carmel.services.extraction_record import (
    ExtractionRecordError,
    _build_identity_payload,  # noqa: PLC2701 -- see preview_reextraction()'s docstring
    compute_extraction_sha,
    extraction_record_dir,
    store_extraction_record,
    verify_extraction_record,
)
from carmel.services.semantic_deps import extraction_identity

__all__ = ["ReextractionError", "preview_reextraction", "reextract_artifact"]

_RAW_NAME = "raw.bin"
_META_NAME = "meta.json"
_PDF_MAGIC = b"%PDF-"
_SHA256_HEX_LEN = 64


class ReextractionError(RuntimeError):
    """Raised when an artifact cannot be safely re-extracted.

    Every raise site names the specific invariant that failed (malformed digest,
    missing/oversized/tampered ``raw.bin``, non-PDF content, a root ``meta.json``
    that disagrees with the sniffed content type, or a re-extraction that failed
    the cleanliness predicate) so a caller never has to guess why re-extraction was
    refused.
    """


def _looks_like_pdf(data: bytes) -> bool:
    """Sniff ``data`` for a PDF header, independent of any stored content-type claim."""
    return data.startswith(_PDF_MAGIC)


def _canonical_extracted_json_bytes(extracted: ExtractedText) -> bytes:
    """Serialize ``extracted`` exactly as ``carmel.services.artifacts.write_json`` would.

    ``write_json`` writes ``json.dumps(data.model_dump(mode="json"), indent=2,
    default=str)`` encoded as UTF-8. This helper is the single place that
    replicates that contract, so the two serializations cannot drift apart.

    What this buys: the bytes this module persists cannot accidentally diverge
    from what a root ``write_json`` call would have produced for the same
    ``ExtractedText``, because there is only one implementation of the encoding.
    What it does NOT buy: serializer identity is not part of a record's identity
    payload or its content address, and it is not one of the fields
    ``current_extraction_records`` (``carmel/services/extraction_record.py``)
    compares -- that function looks only at ``extractor_code_sha256`` and
    ``pypdf_version``. A serializer-only change here would mint a different
    ``extracted_sha256`` on the next re-extraction while every already-stored
    record, produced by the old serializer, keeps reporting as current.
    """
    payload = extracted.model_dump(mode="json")
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def _prepare_reextraction(
    workspace_root: Path, *, raw_sha256: str, max_bytes: int
) -> tuple[Path, ExtractedText, str, str, bytes]:
    """Do every step of re-extraction that reads and parses, up to (not including) the write.

    Shared by :func:`reextract_artifact` (which takes this result and writes it)
    and :func:`preview_reextraction` (which takes the same result and computes
    what WOULD be written, without writing). Both callers get the identical
    real read-raw.bin / re-verify-digest / sniff / cross-check-meta /
    re-parse / cleanliness-predicate work -- there is exactly one
    implementation of "was this artifact genuinely re-extracted", so a dry run
    can never silently diverge from what ``--apply`` would actually do.

    Returns:
        A tuple of ``(root, extracted, extractor_code_sha256, pypdf_version,
        extracted_json_bytes)``, where ``root`` is the normalized workspace root
        (so callers do not redo that resolution), ``extracted`` is the freshly
        re-parsed ``ExtractedText``, and the remaining fields are exactly the
        arguments ``store_extraction_record`` needs beyond ``extractor`` (which
        is ``extracted.extractor``) and ``extracted_json_bytes`` itself.

    Raises:
        ReextractionError: See :func:`reextract_artifact`.
    """
    # Step 1: raw_sha256 must be a well-formed 64-char lowercase hex digest. This is
    # also the path-traversal guard: artifact_dir() joins this string directly onto
    # the workspace root, so a malformed value must be refused before it is ever
    # used to build a path.
    if len(raw_sha256) != _SHA256_HEX_LEN or any(c not in "0123456789abcdef" for c in raw_sha256):
        raise ReextractionError(f"raw_sha256 is not a well-formed 64-character lowercase hex digest: {raw_sha256!r}")

    # Step 2: resolve the artifact directory with the existing helpers, reusing the
    # same containment discipline store_artifact() applies -- never reimplemented.
    root = normalize_path(workspace_root)
    dest_dir = artifact_dir(root, raw_sha256)
    resolved_dest_dir = dest_dir.resolve()
    if not resolved_dest_dir.is_relative_to(root):
        raise ReextractionError(f"refusing to read outside workspace root: {resolved_dest_dir} not under {root}")
    raw_path = dest_dir / _RAW_NAME

    # Step 3: raw.bin must exist AND -- resolved -- remain inside the resolved
    # artifact directory, and must resolve to a regular file. This closes a
    # symlink escape: the artifact directory name is attacker-controlled input (it
    # is just a sha the caller supplies), so if raw.bin were a symlink, an attacker
    # could point it at any readable file outside the workspace, compute THAT
    # file's sha256, and name the directory that hash -- making the content-address
    # check in Step 5 pass while pulling outside content into the evidence store.
    # Resolving and containing here, before Step 4's size check and before any
    # read, closes that: every later step operates on the resolved, contained path.
    try:
        resolved_raw_path = raw_path.resolve(strict=True)
    except FileNotFoundError, NotADirectoryError, OSError:
        raise ReextractionError(f"raw.bin does not exist for {raw_sha256}: {raw_path}") from None
    if not resolved_raw_path.is_relative_to(resolved_dest_dir):
        raise ReextractionError(
            f"raw.bin for {raw_sha256} resolves outside its own artifact directory "
            f"(symlink escape): {raw_path} resolves to {resolved_raw_path}, "
            f"which is not under {resolved_dest_dir}"
        )
    if not resolved_raw_path.is_file():
        raise ReextractionError(f"raw.bin for {raw_sha256} does not resolve to a regular file: {resolved_raw_path}")

    # Step 4: refuse an oversized artifact via stat() BEFORE reading it into memory.
    # stat() and the read below both go through the RESOLVED path, never raw_path.
    size = resolved_raw_path.stat().st_size
    if size > max_bytes:
        raise ReextractionError(f"raw.bin size {size} exceeds max_bytes cap {max_bytes} for {raw_sha256}")

    # Step 5: read the bytes and re-verify the content address ourselves. This is the
    # ONLY trust gate on input -- no sidecar is consulted, and verify_artifact(...,
    # deep=True) is deliberately NOT used here (it returns False for every one of the
    # 8 real legacy artifacts in the live corpus, which would make this path
    # unreachable for the entire corpus it exists to serve).
    data = resolved_raw_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != raw_sha256:
        raise ReextractionError(
            f"raw.bin content does not hash to its own directory name: recomputed {digest!r} != {raw_sha256!r}"
        )

    # Step 6: sniff content type from the bytes themselves. PDF-only path.
    if not _looks_like_pdf(data):
        raise ReextractionError(f"raw.bin for {raw_sha256} does not sniff as a PDF (missing %PDF- header)")

    # Step 7: cross-check root meta.json, if it is readable. ABSENT meta is not
    # fatal -- the sniff above already established the content type. But
    # ``load_artifact_meta`` collapses absent, malformed, invalid, and unreadable
    # meta all down to None alike, so None alone cannot tell us whether meta.json
    # was simply never written or whether it EXISTS but is corrupt -- and a
    # corrupt-but-present meta.json must not silently bypass this check the way an
    # absent one legitimately does. So check existence of the file itself,
    # separately from whether it parsed: present-but-unparseable is fatal on its
    # own, distinct from (and checked before) the content-type disagreement below.
    # Use os.path.lexists() rather than Path.exists(): the latter FOLLOWS symlinks,
    # so a meta.json that is a symlink to a missing target would report as absent
    # and take the permissive "no meta, proceed" path below -- when a dangling
    # symlink is exactly the present-but-unreadable case that must refuse.
    meta_path = dest_dir / _META_NAME
    meta = load_artifact_meta(root, raw_sha256)
    if meta is None and os.path.lexists(meta_path):
        raise ReextractionError(
            f"root meta.json for {raw_sha256} exists but is corrupt or unreadable; "
            "refusing to re-extract past unauthenticated root bookkeeping"
        )
    if meta is not None and meta.content_type != "application/pdf":
        raise ReextractionError(
            f"root meta.json for {raw_sha256} claims content_type {meta.content_type!r}, "
            "which disagrees with the sniffed application/pdf bytes"
        )

    # Step 8: genuinely re-parse the bytes. This is the load-bearing step: never
    # substitute the stored root extracted.json here.
    extracted = extract_text(data, "application/pdf")

    # Step 9: cleanliness predicate -- refuse to persist a lossy or incomplete
    # extraction as permanent evidence, naming which condition failed.
    if extracted.extractor != "pdf:pypdf":
        raise ReextractionError(
            f"fresh extraction of {raw_sha256} used extractor {extracted.extractor!r}, not the expected 'pdf:pypdf'"
        )
    if extracted.lossy:
        raise ReextractionError(f"fresh extraction of {raw_sha256} is lossy; refusing to persist it as evidence")
    if extracted.page_failures:
        raise ReextractionError(
            f"fresh extraction of {raw_sha256} recorded {len(extracted.page_failures)} page failure(s); "
            "refusing to persist an incomplete extraction as evidence"
        )
    if not extracted.text.strip():
        raise ReextractionError(f"fresh extraction of {raw_sha256} produced empty text; refusing to persist it")

    # Step 10: identity -- Carmel's own code sha plus the installed pypdf version,
    # since pypdf is an unpinned third-party dependency whose behaviour can drift.
    identity = extraction_identity()

    # Step 11: serialize exactly as carmel.services.artifacts.write_json would, so
    # the persisted extraction record bytes are byte-identical to what a root
    # extracted.json write would have produced for the same ExtractedText.
    extracted_json_bytes = _canonical_extracted_json_bytes(extracted)

    return root, extracted, identity.code_sha256, identity.pypdf_version, extracted_json_bytes


def reextract_artifact(workspace_root: Path, *, raw_sha256: str, max_bytes: int) -> str:
    """Re-parse a stored artifact's ``raw.bin`` and append a new extraction record.

    Never opens any root file (``raw.bin``, ``text.txt``, ``extracted.json``,
    ``meta.json``) for writing, never calls ``store_artifact``, and never
    reconciles the new record with the root sidecars -- see the module docstring.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: Content address of the already-stored artifact to re-extract.
        max_bytes: Hard cap on the on-disk ``raw.bin`` size, checked via ``stat()``
            before any bytes are read into memory. Callers pass their own budget
            (e.g. ``config.budget.max_artifact_bytes``); this module has no default.

    Returns:
        The ``extraction_sha256`` of the appended (or, if byte-identical bytes were
        already re-extracted before, pre-existing) extraction record.

    Raises:
        ReextractionError: If ``raw_sha256`` is malformed, ``raw.bin`` is missing,
            oversized, or fails to hash to ``raw_sha256``, the bytes do not sniff as
            a PDF, a readable root ``meta.json`` claims a different content type, or
            the fresh extraction fails the cleanliness predicate (wrong extractor,
            lossy, page failures, or empty text).
    """
    root, extracted, extractor_code_sha256, pypdf_version, extracted_json_bytes = _prepare_reextraction(
        workspace_root, raw_sha256=raw_sha256, max_bytes=max_bytes
    )
    # Step 12: append the new extraction record. store_extraction_record() is
    # itself idempotent (byte-identical re-extraction returns the same address
    # unchanged) and append-only (mkdtemp + os.rename + EEXIST-compare); it is
    # called directly and is the only write this module performs.
    return store_extraction_record(
        root,
        raw_sha256=raw_sha256,
        extractor=extracted.extractor,
        extractor_code_sha256=extractor_code_sha256,
        pypdf_version=pypdf_version,
        extracted_json_bytes=extracted_json_bytes,
    )


def preview_reextraction(workspace_root: Path, *, raw_sha256: str, max_bytes: int) -> tuple[str, bool]:
    """Do everything ``reextract_artifact`` does, except the write: report what it would do.

    Runs the identical real read-raw.bin / re-verify-digest / sniff /
    cross-check-meta / re-parse / cleanliness-predicate pipeline as
    :func:`reextract_artifact` (via the shared :func:`_prepare_reextraction`
    helper), then computes the ``extraction_sha256`` the write WOULD produce and
    checks whether a record already lives at that address -- all without calling
    :func:`~carmel.services.extraction_record.store_extraction_record` and
    therefore without writing anything.

    "Already present" is decided by
    :func:`~carmel.services.extraction_record.verify_extraction_record`, never by
    a directory's mere existence: a directory at the computed address that is
    empty, half-written, or holds forged/corrupt contents is NOT an authenticated
    record, and must not be reported as one. Such a directory occupying the
    address without authenticating to it is itself a distinct, fatal collision --
    :func:`~carmel.services.extraction_record.store_extraction_record` refuses to
    overwrite it (append-only), so ``--apply`` can neither report success nor
    silently clobber it; this raises :exc:`ReextractionError` instead, surfacing
    the collision as an explicit refusal.

    This deliberately reaches into ``carmel.services.extraction_record``'s
    private ``_build_identity_payload`` rather than re-deriving the identity
    payload's field set locally: that field set (which fields are included, and
    when ``pypdf_version`` is folded in at all) is exactly what
    ``store_extraction_record`` uses to compute the address it writes to, and it
    is not part of that module's public contract (not in ``__all__``). Reusing it
    here -- instead of guessing the same shape a second time -- is what makes this
    preview's ``extraction_sha256`` provably the SAME address ``--apply`` would
    write to, rather than a second, independently-written guess that could drift.

    Args:
        workspace_root: Root of the campaign workspace.
        raw_sha256: Content address of the already-stored artifact to preview
            re-extraction for.
        max_bytes: Hard cap on the on-disk ``raw.bin`` size -- see
            :func:`reextract_artifact`.

    Returns:
        A ``(extraction_sha256, already_present)`` pair: the address a call to
        ``reextract_artifact`` with the same arguments would return, and whether a
        record already lives there (``True`` means ``--apply`` would be a no-op;
        ``False`` means ``--apply`` would append a new record).

    Raises:
        ReextractionError: See :func:`reextract_artifact`. Also raised when a
            directory already occupies the computed address but does not
            authenticate as a stored extraction record (present-but-not-authentic
            collision) -- see the "Already present" paragraph above.
    """
    root, extracted, extractor_code_sha256, pypdf_version, extracted_json_bytes = _prepare_reextraction(
        workspace_root, raw_sha256=raw_sha256, max_bytes=max_bytes
    )
    extracted_sha256 = hashlib.sha256(extracted_json_bytes).hexdigest()
    extracted_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    identity_payload = _build_identity_payload(
        raw_sha256=raw_sha256,
        extractor=extracted.extractor,
        extractor_code_sha256=extractor_code_sha256,
        pypdf_version=pypdf_version,
        extracted_sha256=extracted_sha256,
        extracted_text_sha256=extracted_text_sha256,
    )
    extraction_sha256 = compute_extraction_sha(identity_payload)
    record_dir = extraction_record_dir(root, raw_sha256, extraction_sha256)
    try:
        authentic = verify_extraction_record(root, raw_sha256, extraction_sha256)
    except ExtractionRecordError as exc:
        raise ReextractionError(
            f"extraction record directory for {raw_sha256} at {record_dir} exists but its "
            f"meta.json does not authenticate: {exc}; refusing to report already-present or to "
            "overwrite an append-only record"
        ) from exc
    if authentic:
        return extraction_sha256, True
    # Use os.path.lexists() rather than Path.exists(): the latter FOLLOWS
    # symlinks, so a dangling symlink at the computed record directory would
    # report as absent and let this preview say "would be written" -- when a
    # dangling symlink is exactly the present-but-unreadable case that must
    # refuse, the same defect already fixed for the root meta.json in
    # ``reextract_artifact``.
    if os.path.lexists(record_dir):
        raise ReextractionError(
            f"extraction record directory for {raw_sha256} at {record_dir} exists but does not "
            "authenticate as a stored extraction record; refusing to report already-present or "
            "to overwrite an append-only record"
        )
    return extraction_sha256, False
