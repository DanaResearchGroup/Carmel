"""Content-addressed storage for fetched literature artifacts.

Every fetched document is kept as raw bytes inside the user's campaign
workspace — never inside the code repo — so evidence for a citation remains
replayable years later, even after the source URL has rotted. The store is
content-addressed by the sha256 of the exact bytes: identical content is
stored exactly once, and a citation can be pinned to precise, verifiable
content rather than to a URL that may later point at something else.

Layout, exactly::

    <workspace_root>/evidence/literature/<sha256>/raw.bin         # exact fetched bytes
    <workspace_root>/evidence/literature/<sha256>/text.txt        # extracted text (humans)
    <workspace_root>/evidence/literature/<sha256>/extracted.json  # full ExtractedText (source of truth)
    <workspace_root>/evidence/literature/<sha256>/meta.json       # StoredArtifact

The directory name is ALWAYS the sha256 hex digest of the bytes, recomputed
here rather than trusted from the caller — never a URL-derived name, which
would both collide and open a path-traversal vector (e.g. a source URL of
``http://x/../../../etc/passwd``).

``extracted.json`` persists the FULL :class:`~carmel.agents.tools.extract.ExtractedText`
(section labels, page count, extractor, lossy flag) — not just the raw text. The
grounding gate (:func:`carmel.services.grounding.ground_finding`) depends on section
labels to reject a quote found only in a reference list, so a reload that dropped
sections would silently lose that protection. ``text.txt`` is kept alongside purely
as a human-readable convenience; ``extracted.json`` is the source of truth for
:func:`load_artifact_text`. An artifact stored by the OLD layout (no ``extracted.json``
on disk) falls back to reconstructing from ``text.txt`` with empty sections, and that
reconstruction is marked degraded (``lossy=True``) so a caller can never mistake it
for a fully section-labelled extraction.

The directory name doubles as an integrity check: on every store, and again on every
idempotent re-store, ``raw.bin``'s digest, ``meta.json``'s recorded sha256/byte-count,
and the presence of ``text.txt``/``extracted.json`` are all verified against the
directory name. Any mismatch is treated as on-disk corruption (bit rot, partial
write, tampering) and repaired from the caller's bytes-in-hand rather than trusted.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.logger import get_logger
from carmel.paths import normalize_path
from carmel.schemas.literature import ArtifactProvenance, StoredArtifact
from carmel.services.artifacts import read_bytes, read_json, write_bytes, write_json, write_text

__all__ = [
    "EVIDENCE_LITERATURE_DIR",
    "artifact_dir",
    "list_artifacts",
    "load_artifact_text",
    "store_artifact",
    "verify_artifact",
]

logger = get_logger("services.evidence")

EVIDENCE_LITERATURE_DIR = "evidence/literature"

_RAW_NAME = "raw.bin"
_TEXT_NAME = "text.txt"
_META_NAME = "meta.json"
_EXTRACTED_NAME = "extracted.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def artifact_dir(workspace_root: Path, sha256: str) -> Path:
    """Compute (but never create) the content-addressed directory for ``sha256``.

    Pure path helper: no filesystem access, no side effects.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Hex digest identifying the artifact.

    Returns:
        ``<workspace_root>/evidence/literature/<sha256>``.
    """
    return Path(workspace_root) / EVIDENCE_LITERATURE_DIR / sha256


def _assert_contained(workspace_root: Path, path: Path) -> Path:
    """Resolve ``path`` and ``workspace_root`` and confirm containment.

    Defence in depth: the sha256-derived name is already safe, but
    ``workspace_root`` itself arrives from persisted config that could in
    principle be tampered with, so containment is asserted independently
    before any write.

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


def store_artifact(
    workspace_root: Path,
    *,
    data: bytes,
    artifact: FetchedArtifact,
    extracted: ExtractedText,
    license_note: str | None = None,
    provenance: ArtifactProvenance = ArtifactProvenance.FETCHED,
    max_bytes: int,
) -> StoredArtifact:
    """Content-address and persist a fetched artifact inside the workspace.

    Writes ``raw.bin``, ``text.txt``, ``extracted.json``, and ``meta.json`` under
    ``<workspace_root>/evidence/literature/<sha256>/``, where ``<sha256>`` is
    recomputed from ``data`` (never trusted from ``artifact.sha256``).
    Idempotent: re-storing identical bytes is a no-op that returns the existing
    metadata, PROVIDED the existing on-disk artifact is verified intact first
    (``raw.bin`` re-hashed and compared against both the directory name and
    ``meta.json``'s recorded sha256/byte-count, and ``text.txt`` confirmed present).
    If ``meta.json`` is missing/unreadable, or that verification fails for any
    reason (corruption, partial write, tampering), the artifact is repaired by
    rewriting all four files from the ``data``/``artifact``/``extracted`` the caller
    has in hand (never silently returning metadata that describes bytes different
    from what is actually on disk).

    Args:
        workspace_root: Root of the campaign workspace.
        data: The exact fetched bytes.
        artifact: Fetch metadata (URL, claimed sha256, content type, etc.).
        extracted: The extracted text to persist alongside the raw bytes.
        license_note: Optional free-text license/usage note.
        provenance: How these bytes reached the workspace. Defaults to ``FETCHED``;
            manual acquisition passes ``MANUAL`` so a reader of the evidence chain can
            tell a machine-verified retrieval from a human-supplied file.
        max_bytes: Hard cap on artifact size.

    Returns:
        The persisted (or pre-existing) :class:`StoredArtifact` metadata.

    Raises:
        ValueError: If ``data`` is empty, if ``len(data)`` exceeds ``max_bytes``, if
            the recomputed sha256 disagrees with ``artifact.sha256``, or if the
            resolved destination would fall outside the resolved workspace root.
    """
    if not data:
        raise ValueError(
            "refusing to store a zero-byte artifact: a fetch that yields no bytes is a "
            "failed acquisition, never evidence"
        )
    if len(data) > max_bytes:
        raise ValueError(f"artifact size {len(data)} exceeds max_bytes cap {max_bytes}")

    digest = hashlib.sha256(data).hexdigest()
    if digest != artifact.sha256:
        raise ValueError(
            f"sha256 mismatch: recomputed {digest!r} disagrees with claimed {artifact.sha256!r}; "
            "bytes and metadata have diverged"
        )

    root = normalize_path(workspace_root)
    dest_dir = _assert_contained(root, artifact_dir(root, digest))
    raw_path = dest_dir / _RAW_NAME
    text_path = dest_dir / _TEXT_NAME
    meta_path = dest_dir / _META_NAME

    extracted_path = dest_dir / _EXTRACTED_NAME

    if raw_path.exists():
        existing = _load_meta(meta_path)
        if existing is not None and _artifact_intact(raw_path, text_path, extracted_path, digest, existing):
            return existing
        logger.warning(
            "repairing corrupted or incomplete artifact %s (missing/unreadable meta.json, "
            "raw.bin digest mismatch, meta/byte-count disagreement, missing text.txt, or "
            "missing/unparseable extracted.json)",
            digest,
        )
        return _write_all(
            dest_dir,
            raw_path,
            text_path,
            meta_path,
            extracted_path,
            data,
            artifact,
            extracted,
            digest,
            license_note,
            provenance,
        )

    return _write_all(
        dest_dir,
        raw_path,
        text_path,
        meta_path,
        extracted_path,
        data,
        artifact,
        extracted,
        digest,
        license_note,
        provenance,
    )


def _artifact_intact(raw_path: Path, text_path: Path, extracted_path: Path, digest: str, meta: StoredArtifact) -> bool:
    """Verify that an existing on-disk artifact still agrees with the directory name.

    Re-reads ``raw.bin`` from disk (never trusts that a prior write succeeded) and
    checks its digest and byte count against both the directory name and the
    previously persisted ``meta.json``, confirms ``text.txt`` is present, and confirms
    ``extracted.json`` is present AND parses as a valid :class:`ExtractedText` (spar
    round 5, Finding 2). ``extracted.json`` is the source of truth the grounding gate
    depends on for section labels; a missing or truncated/corrupt sidecar must be
    treated exactly like a missing ``text.txt`` -- repaired from the caller's bytes in
    hand -- rather than silently trusted forever because ``raw.bin`` and a parseable
    ``meta.json`` both happen to exist. This is what prevents a corrupted or
    partially-written artifact from being trusted forever.

    Args:
        raw_path: Path to the stored raw bytes.
        text_path: Path to the stored extracted text.
        extracted_path: Path to the stored full ``ExtractedText`` sidecar.
        digest: The directory name (expected sha256 of the raw bytes).
        meta: The previously persisted metadata for this artifact.

    Returns:
        True only if the on-disk bytes, their digest, and ``meta.json`` all agree with
        ``digest`` and with each other, ``text.txt`` exists, and ``extracted.json``
        exists and parses as a valid ``ExtractedText``.
    """
    if meta.sha256 != digest:
        return False
    if not text_path.exists():
        return False
    try:
        raw_extracted = read_json(extracted_path)
        ExtractedText.model_validate(raw_extracted)
    except FileNotFoundError, ValueError, OSError:
        return False
    try:
        on_disk = read_bytes(raw_path)
    except FileNotFoundError, OSError:
        return False
    if len(on_disk) != meta.n_bytes:
        return False
    return hashlib.sha256(on_disk).hexdigest() == digest


def _load_meta(meta_path: Path) -> StoredArtifact | None:
    """Best-effort load of an existing ``meta.json``; None if absent/unreadable."""
    try:
        raw = read_json(meta_path)
    except FileNotFoundError, ValueError, OSError:
        return None
    try:
        return StoredArtifact.model_validate(raw)
    except Exception:  # noqa: BLE001 - any validation failure means "unreadable"
        return None


def _write_all(
    dest_dir: Path,
    raw_path: Path,
    text_path: Path,
    meta_path: Path,
    extracted_path: Path,
    data: bytes,
    artifact: FetchedArtifact,
    extracted: ExtractedText,
    digest: str,
    license_note: str | None,
    provenance: ArtifactProvenance,
) -> StoredArtifact:
    """Write raw.bin, text.txt, extracted.json, and meta.json for a (new or repaired) artifact."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_bytes(raw_path, data)
    write_text(text_path, extracted.text)
    write_json(extracted_path, extracted)
    stored = StoredArtifact(
        sha256=digest,
        source_url=artifact.url,
        final_url=artifact.final_url,
        content_type=artifact.content_type,
        n_bytes=len(data),
        stored_at=datetime.now(UTC),
        extractor=extracted.extractor,
        lossy=extracted.lossy,
        license_note=license_note,
        provenance=provenance,
    )
    write_json(meta_path, stored)
    return stored


def _validate_sha256(workspace_root: Path, sha256: str) -> Path:
    """Validate ``sha256`` is a well-formed digest and resolve its containment-checked directory.

    Unlike the write path (which recomputes the digest from bytes in hand and
    therefore cannot be spoofed), read paths receive ``sha256`` directly from the
    caller and interpolate it into a filesystem path. Without validation, a caller
    could pass something like ``"../../etc/passwd"`` or a truncated/garbage string
    and have it walked straight into a path. Requiring exactly 64 lowercase hex
    characters closes that off, and containment is re-asserted as defence in depth.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Caller-supplied hex digest.

    Returns:
        The resolved, containment-checked artifact directory for ``sha256``.

    Raises:
        ValueError: If ``sha256`` is not exactly 64 lowercase hex characters, or if
            the resolved artifact directory would fall outside the resolved
            workspace root.
    """
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"invalid sha256 digest: {sha256!r} (expected 64 lowercase hex characters)")
    root = normalize_path(workspace_root)
    return _assert_contained(root, artifact_dir(root, sha256))


def list_artifacts(workspace_root: Path) -> list[StoredArtifact]:
    """Every artifact currently held in this workspace's evidence store.

    Returned in a stable order (``stored_at``, then ``sha256``) so a corpus pass
    presents the same corpus in the same order on every run: the whole point of
    reading the store rather than the web is that the input is reproducible, and an
    order that varied with directory iteration would undermine that for no benefit.

    Directories that do not parse as artifacts are skipped rather than raising. The
    store is content-addressed and append-only, but a crashed write or a partially
    copied workspace can leave a directory without usable ``meta.json``; that is a
    reason to ignore one artifact, never to make the whole corpus unreadable.
    """
    root = Path(workspace_root) / EVIDENCE_LITERATURE_DIR
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []

    artifacts: list[StoredArtifact] = []
    for entry in entries:
        if not entry.is_dir() or not _SHA256_RE.match(entry.name):
            continue
        meta = _load_meta(entry / _META_NAME)
        if meta is None:
            logger.warning("evidence store: skipping %s (no readable meta.json)", entry.name)
            continue
        artifacts.append(meta)
    artifacts.sort(key=lambda a: (a.stored_at, a.sha256))
    return artifacts


def load_artifact_text(workspace_root: Path, sha256: str) -> ExtractedText | None:
    """Load the stored extracted text for a previously stored artifact.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Hex digest identifying the artifact.

    Returns:
        None when no artifact is stored under ``sha256``. Otherwise the persisted
        :class:`ExtractedText`. When ``extracted.json`` is present (current layout),
        it is loaded verbatim and is fully section-labelled — safe for the grounding
        gate to rely on. When only the OLD layout is present (``text.txt`` +
        ``meta.json``, no ``extracted.json``), the text is reconstructed with
        ``sections=[]`` and ``lossy`` is forced to True regardless of the originally
        stored value — this is the ONLY signal available on this model that marks the
        reconstruction as degraded, and callers (in particular the grounding gate)
        must never treat a ``lossy=True`` reload as equivalent to a fully labelled
        extraction.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase hex
            digest, or if the resolved artifact directory would fall outside the
            resolved workspace root.
    """
    from carmel.agents.tools.extract import normalize_for_match

    dest_dir = _validate_sha256(workspace_root, sha256)
    text_path = dest_dir / _TEXT_NAME
    meta_path = dest_dir / _META_NAME
    extracted_path = dest_dir / _EXTRACTED_NAME
    if not text_path.exists() or not meta_path.exists():
        return None

    meta = _load_meta(meta_path)
    if meta is None:
        return None

    if extracted_path.exists():
        try:
            raw = read_json(extracted_path)
            return ExtractedText.model_validate(raw)
        except FileNotFoundError, ValueError, OSError:
            pass  # noqa: S110 - fall through to degraded reconstruction below

    text = text_path.read_text(encoding="utf-8")
    return ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=[],
        extractor=meta.extractor,
        lossy=True,
    )


def verify_artifact(workspace_root: Path, sha256: str) -> bool:
    """Re-read ``raw.bin`` and confirm its digest still equals the directory name.

    This is what makes the stored provenance auditable rather than merely
    claimed: a caller can, at any later time, prove that the bytes under
    ``sha256`` have not been altered on disk.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Hex digest identifying the artifact (and its directory name).

    Returns:
        True if ``raw.bin`` exists and its sha256 matches ``sha256``; False
        if the artifact is absent or the bytes have been corrupted/tampered.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase hex
            digest, or if the resolved artifact directory would fall outside the
            resolved workspace root.
    """
    dest_dir = _validate_sha256(workspace_root, sha256)
    raw_path = dest_dir / _RAW_NAME
    try:
        data = read_bytes(raw_path)
    except FileNotFoundError:
        return False
    return hashlib.sha256(data).hexdigest() == sha256
