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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText, extract_text
from carmel.paths import normalize_path
from carmel.services.evidence import artifact_dir, load_artifact_meta
from carmel.services.extraction_record import store_extraction_record
from carmel.services.semantic_deps import extraction_identity

__all__ = ["ReextractionError", "reextract_artifact"]

_RAW_NAME = "raw.bin"
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
    """
    payload = extracted.model_dump(mode="json")
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


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

    # Step 3: raw.bin must exist.
    if not raw_path.exists():
        raise ReextractionError(f"raw.bin does not exist for {raw_sha256}: {raw_path}")

    # Step 4: refuse an oversized artifact via stat() BEFORE reading it into memory.
    size = raw_path.stat().st_size
    if size > max_bytes:
        raise ReextractionError(f"raw.bin size {size} exceeds max_bytes cap {max_bytes} for {raw_sha256}")

    # Step 5: read the bytes and re-verify the content address ourselves. This is the
    # ONLY trust gate on input -- no sidecar is consulted, and verify_artifact(...,
    # deep=True) is deliberately NOT used here (it returns False for every one of the
    # 8 real legacy artifacts in the live corpus, which would make this path
    # unreachable for the entire corpus it exists to serve).
    data = raw_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != raw_sha256:
        raise ReextractionError(
            f"raw.bin content does not hash to its own directory name: recomputed {digest!r} != {raw_sha256!r}"
        )

    # Step 6: sniff content type from the bytes themselves. PDF-only path.
    if not _looks_like_pdf(data):
        raise ReextractionError(f"raw.bin for {raw_sha256} does not sniff as a PDF (missing %PDF- header)")

    # Step 7: cross-check root meta.json, if it is readable. Unreadable/missing meta
    # is not fatal -- the sniff above already established the content type -- but a
    # readable meta.json that disagrees IS fatal.
    meta = load_artifact_meta(root, raw_sha256)
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

    # Step 12: append the new extraction record. store_extraction_record() is
    # itself idempotent (byte-identical re-extraction returns the same address
    # unchanged) and append-only (mkdtemp + os.rename + EEXIST-compare); it is
    # called directly and is the only write this module performs.
    return store_extraction_record(
        root,
        raw_sha256=raw_sha256,
        extractor=extracted.extractor,
        extractor_code_sha256=identity.code_sha256,
        pypdf_version=identity.pypdf_version,
        extracted_json_bytes=extracted_json_bytes,
    )
