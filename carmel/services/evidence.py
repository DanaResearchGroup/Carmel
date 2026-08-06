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

Neither of the checks above proves that ``extracted.json`` was actually DERIVED FROM
``raw.bin`` -- only that each file, taken alone, matches its own recorded digest. A
stale or swapped extraction (produced from different bytes, or by a different/older
extractor) that also carries a self-consistent ``extracted_sha256`` would pass both
checks. Whether re-running the extractor on ``raw.bin`` is even guaranteed to
reproduce ``extracted.json`` byte-for-byte was investigated directly:

- :mod:`carmel.agents.tools.extract` embeds no timestamps or UUIDs anywhere in
  :class:`~carmel.agents.tools.extract.ExtractedText` (its fields are ``text``,
  ``normalized``, ``sections``, ``page_count``, ``extractor``, ``lossy`` -- see that
  module's docstring), and its extraction paths (PDF page loop, HTML tag stripping,
  plain-text passthrough) run single-threaded with no parallelism, so nothing there
  varies run to run for the SAME library versions.
- But the PDF path (the common case) delegates the actual text extraction to the
  third-party ``pypdf`` library (``import pypdf`` in
  ``carmel/agents/tools/extract.py``, ``_extract_pdf``'s ``reader.pages[i]
  .extract_text()``), and that dependency is unpinned in this project
  (``pyproject.toml``: ``pypdf>=5.0``). Different installed versions of the SAME
  library are not guaranteed to extract byte-identical text from identical PDF
  bytes -- ``pypdf``'s extraction algorithm has changed release to release -- and
  the persisted ``extractor`` string ("pdf:pypdf") does not even record which
  version produced a given ``extracted.json``.
- Conclusion: extraction is **not proven deterministic** across time/library
  upgrades. Re-running the extractor and diffing the result would therefore be
  unsound as a verification step -- a legitimate ``pypdf`` upgrade between store
  time and verify time would make an untouched, perfectly good artifact fail a
  byte-diff check. So :func:`verify_artifact`'s ``deep=True`` path does NOT
  re-extract; instead it checks a *derivation binding* recorded at store time (see
  :data:`StoredArtifact.derivation_binding` for exactly what that does and does not
  prove).
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
    "EvidenceStoreUnreadableError",
    "artifact_dir",
    "list_artifacts",
    "list_artifacts_with_unreadable",
    "load_artifact_meta",
    "load_artifact_text",
    "store_artifact",
    "verify_artifact",
]

logger = get_logger("services.evidence")

EVIDENCE_LITERATURE_DIR = "evidence/literature"


class EvidenceStoreUnreadableError(OSError):
    """The evidence store exists but could not be enumerated.

    Deliberately an exception rather than an empty result. Every other "I could not read
    this" in this module degrades to skipping ONE artifact, which the caller folds into
    its reported coverage; there is no equivalent move for the store as a whole, because
    a caller handed an empty list cannot tell "nothing is stored" from "I could not
    look", and reports full coverage of nothing either way.

    A subclass of ``OSError`` so a caller that already handles I/O failure around store
    access keeps working, while one that wants to name this case specifically can.
    """


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
    if not _extracted_sidecar_intact(extracted_path, meta):
        return False
    try:
        on_disk = read_bytes(raw_path)
    except FileNotFoundError, OSError:
        return False
    if len(on_disk) != meta.n_bytes:
        return False
    return hashlib.sha256(on_disk).hexdigest() == digest


def _extracted_sidecar_intact(extracted_path: Path, meta: StoredArtifact) -> bool:
    """Check ``extracted.json`` against the digest recorded in ``meta.json``.

    Validating that the sidecar PARSES is not enough. A write interrupted at a buffer
    boundary, or a file truncated by a full disk, can leave JSON that still parses and
    still validates as an :class:`ExtractedText` while holding only part of the
    document -- at which point the grounding gate happily confirms quotes against a
    fragment and reports them as grounded in the stored bytes.

    Args:
        extracted_path: Path of the stored ``extracted.json``.
        meta: Metadata for the artifact, carrying the expected digest.

    Returns:
        True when the sidecar matches its recorded digest, or when no digest was
        recorded. ``None`` means the artifact predates this field, NOT that it is
        corrupt: refusing those would make every artifact stored before this change
        unreadable, discarding good evidence to enforce a check that could not have
        run when it was written. They keep the guarantees they were stored with.
    """
    if meta.extracted_sha256 is None:
        return True
    try:
        return hashlib.sha256(extracted_path.read_bytes()).hexdigest() == meta.extracted_sha256
    except FileNotFoundError, OSError:
        return False


def _extractor_identity(extractor: str) -> str:
    """Best-effort extractor identity+version string for the derivation binding.

    For the ``"pdf:pypdf"`` extractor this records the ACTUAL installed ``pypdf``
    version (e.g. ``"pdf:pypdf==5.1.0"``), because that dependency is unpinned
    (``pypdf>=5.0`` in ``pyproject.toml``) and different installed versions of the
    same library are not guaranteed to extract byte-identical text from identical
    PDF bytes -- see the module docstring's determinism analysis. For every other
    extractor (``"html"``, ``"text"``, ``"pdf:unavailable"``, ``"unknown"``) there
    is no third-party version to record -- those paths are stdlib-only -- so the
    extractor string itself is the whole identity.

    Args:
        extracted.extractor: The extractor string recorded on the ``ExtractedText``.

    Returns:
        The identity+version string to fold into :func:`_derivation_binding`.
    """
    if extractor == "pdf:pypdf":
        try:
            import pypdf

            return f"{extractor}=={pypdf.__version__}"
        except Exception:  # noqa: BLE001 - version discovery is best-effort, never fatal
            return extractor
    return extractor


def _derivation_binding(raw_sha256: str, extracted_sha256: str, extractor_identity: str) -> str:
    """Bind extractor identity, raw digest, and sidecar digest into one digest.

    Read exactly what the result proves -- documented in full on
    :data:`~carmel.schemas.literature.StoredArtifact.derivation_binding`, which this
    function's docstring must not repeat and drift from. In short: internal
    consistency of a single ``meta.json`` record, never literal re-derivation.

    Args:
        raw_sha256: Digest of the stored ``raw.bin`` bytes.
        extracted_sha256: Digest of the stored ``extracted.json`` bytes.
        extractor_identity: Result of :func:`_extractor_identity`.

    Returns:
        sha256 hex digest of the three inputs, joined by ``"|"``.
    """
    return hashlib.sha256(f"{extractor_identity}|{raw_sha256}|{extracted_sha256}".encode()).hexdigest()


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
    # Hash the file AFTER writing rather than re-serialising the model: the digest has
    # to describe the bytes actually on disk, and re-deriving them here would silently
    # drift the moment `write_json`'s formatting changed.
    extracted_digest = hashlib.sha256(extracted_path.read_bytes()).hexdigest()
    extractor_identity = _extractor_identity(extracted.extractor)
    stored = StoredArtifact(
        sha256=digest,
        source_url=artifact.url,
        final_url=artifact.final_url,
        content_type=artifact.content_type,
        n_bytes=len(data),
        stored_at=datetime.now(UTC),
        extractor=extracted.extractor,
        extracted_sha256=extracted_digest,
        lossy=extracted.lossy,
        license_note=license_note,
        provenance=provenance,
        extractor_version=extractor_identity,
        derivation_binding=_derivation_binding(digest, extracted_digest, extractor_identity),
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
    # Matched with fullmatch, never match: Python's `$` also matches just BEFORE a
    # trailing newline, so match would let "a" * 64 + "\n" through.
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"invalid sha256 digest: {sha256!r} (expected 64 lowercase hex characters)")
    root = normalize_path(workspace_root)
    return _assert_contained(root, artifact_dir(root, sha256))


def list_artifacts_with_unreadable(workspace_root: Path) -> tuple[list[StoredArtifact], list[str]]:
    """Every held artifact, AND the sha-named directories that would not parse as one.

    Returned in a stable order (``stored_at``, then ``sha256``) so a corpus pass
    presents the same corpus in the same order on every run: the whole point of
    reading the store rather than the web is that the input is reproducible, and an
    order that varied with directory iteration would undermine that for no benefit.

    Directories that do not parse as artifacts are skipped rather than raising. The
    store is content-addressed and append-only, but a crashed write or a partially
    copied workspace can leave a directory without usable ``meta.json``; that is a
    reason to ignore one artifact, never to make the whole corpus unreadable.

    Skipping them is right; skipping them SILENTLY is not. A ``logger.warning`` does
    not reach the operator where it matters: a corpus pass reports "N held artifact(s)
    not covered by this pass" from the shas its loader skipped, and an artifact dropped
    here never reaches that loader — so it was absent from the corpus and absent from
    the count of what the pass missed. That reads as full coverage of a smaller store,
    which is the one reading that makes a barren pass look conclusive. Returning the
    names lets the caller fold them into the coverage it reports (F11).
    """
    root = Path(workspace_root) / EVIDENCE_LITERATURE_DIR
    if not root.exists():
        # A workspace that has never held an artifact. Genuinely empty, and saying so is
        # honest -- this is the ONLY route by which this function reports an empty store.
        return [], []
    try:
        entries = sorted(root.iterdir())
    except OSError as exc:
        # NOT `return [], []`. That is the very failure this function's docstring argues
        # against, applied to the whole store rather than to one artifact: "the store
        # cannot be read" would present as "the store is empty", and a pass over an empty
        # store reports full coverage of nothing. A barren pass would look conclusive,
        # which is strictly worse than failing -- nobody re-runs a pass that said it was
        # done.
        raise EvidenceStoreUnreadableError(
            f"the evidence store at {root} exists but could not be listed: {exc}. Refusing to "
            "report it as empty -- an unreadable store establishes nothing about what it holds"
        ) from exc

    artifacts: list[StoredArtifact] = []
    unreadable: list[str] = []
    for entry in entries:
        if not entry.is_dir() or not _SHA256_RE.fullmatch(entry.name):
            continue
        meta = _load_meta(entry / _META_NAME)
        if meta is None:
            logger.warning("evidence store: skipping %s (no readable meta.json)", entry.name)
            unreadable.append(entry.name)
            continue
        artifacts.append(meta)
    artifacts.sort(key=lambda a: (a.stored_at, a.sha256))
    return artifacts, sorted(unreadable)


def list_artifacts(workspace_root: Path) -> list[StoredArtifact]:
    """Every artifact currently held in this workspace's evidence store.

    Thin view over :func:`list_artifacts_with_unreadable` for callers that only
    want the readable artifacts. Anything reporting COVERAGE to an operator should
    use that function instead and account for what it could not read.
    """
    return list_artifacts_with_unreadable(workspace_root)[0]


def load_artifact_meta(workspace_root: Path, sha256: str) -> StoredArtifact | None:
    """Load the :class:`StoredArtifact` metadata for one artifact, by sha256.

    Direct single-artifact lookup, as opposed to :func:`list_artifacts` (which
    scans and parses every ``meta.json`` in the store and silently drops
    unreadable ones -- fine for corpus enumeration, wrong for "resolve THIS
    artifact"). Goes through the same sha-shape and containment validation as
    every other read path in this module.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Hex digest identifying the artifact.

    Returns:
        The persisted metadata, or None when no artifact is stored under
        ``sha256`` or its ``meta.json`` is missing/unreadable/invalid.

    Raises:
        ValueError: If ``sha256`` is not a well-formed 64-character lowercase
            hex digest, or if the resolved artifact directory would fall
            outside the resolved workspace root.
    """
    dest_dir = _validate_sha256(workspace_root, sha256)
    return _load_meta(dest_dir / _META_NAME)


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


def verify_artifact(workspace_root: Path, sha256: str, *, deep: bool = False) -> bool:
    """Re-read the stored bytes and confirm they still match their recorded digests.

    This is what makes the stored provenance auditable rather than merely
    claimed: a caller can, at any later time, prove that the bytes under
    ``sha256`` have not been altered on disk.

    BOTH ``raw.bin`` and ``extracted.json`` are checked. Verifying only ``raw.bin``
    would leave the guarantee pointing at the wrong file: the grounding gate never
    reads ``raw.bin``, it reads ``extracted.json``, so an intact-but-unverified
    sidecar is the one that can actually corrupt a result. Artifacts stored before
    ``extracted_sha256`` existed carry no sidecar digest and are judged on
    ``raw.bin`` alone, exactly as they were when written.

    Neither check above proves ``extracted.json`` was actually DERIVED FROM
    ``raw.bin`` -- only that each file, taken alone, matches its own recorded
    digest. A stale or swapped extraction whose ``extracted_sha256`` was updated to
    match its (wrong) new bytes would pass both checks. ``deep=True`` opts into an
    additional derivation-binding check for that gap -- see
    :data:`~carmel.schemas.literature.StoredArtifact.derivation_binding` for
    precisely what it does and does not prove (in short: internal consistency of
    the stored metadata record, never literal re-derivation from ``raw.bin`` -- see
    this module's docstring for why true re-derivation is unsound to check at all).
    It defaults to False so the cheap, backward-compatible check remains the
    default for existing callers.

    Args:
        workspace_root: Root of the campaign workspace.
        sha256: Hex digest identifying the artifact (and its directory name).
        deep: When True, additionally require the derivation binding recorded in
            ``meta.json`` to match one freshly recomputed from ``meta.json``'s own
            ``extractor_version``, ``sha256``, and ``extracted_sha256`` fields.
            Artifacts stored before this field existed (``derivation_binding is
            None``) fail this check -- deep verification is a strictly stronger,
            opt-in standard that legacy artifacts never had the chance to meet; the
            default (non-deep) check above is unaffected and keeps working for them.

    Returns:
        True if ``raw.bin`` exists, its sha256 matches ``sha256``, the
        ``extracted.json`` sidecar matches its recorded digest (when one exists),
        and -- when ``deep`` is True -- the recorded derivation binding is
        internally consistent; False if the artifact is absent, any of those bytes
        have been corrupted/tampered, or (``deep=True``) the binding is missing or
        inconsistent.

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
    if hashlib.sha256(data).hexdigest() != sha256:
        return False
    meta = _load_meta(dest_dir / _META_NAME)
    if meta is None:
        # Fail CLOSED. Without metadata there is no recorded `extracted_sha256`, so
        # whether this artifact ever had a verifiable sidecar is unknown -- and
        # returning True would report "verified" for a check that could not run. That
        # is the assertion-for-observation conflation this codebase has already fixed
        # twice on the acquisition side (PAYWALLED vs NO_OPEN_ACCESS_COPY, then
        # NO_OPEN_ACCESS_COPY vs OA_LOOKUP_INCOMPLETE); the grounding gate quotes from
        # `extracted.json`, so "could not check" must not read as "intact".
        #
        # Legacy artifacts are unaffected: they have READABLE meta with
        # `extracted_sha256=None`, which _extracted_sidecar_intact still accepts.
        logger.warning("evidence store: %s has no readable meta.json; cannot verify", sha256)
        return False
    if not _extracted_sidecar_intact(dest_dir / _EXTRACTED_NAME, meta):
        return False
    if not deep:
        return True
    return _derivation_binding_intact(meta)


def _derivation_binding_intact(meta: StoredArtifact) -> bool:
    """Recompute the derivation binding from ``meta``'s own fields and compare.

    Fail-closed, same pattern as the rest of this module: a legacy artifact with no
    recorded ``derivation_binding``/``extractor_version`` (predates the field) or
    with no recorded ``extracted_sha256`` (predates THAT field, so there is nothing
    to bind) cannot satisfy a check it never had the data for, and is treated as
    failing deep verification rather than as "can't tell so allow it". This is a
    deliberate, documented exception to "fail closed by default" ONLY in the sense
    that the DEFAULT (non-deep) check above is unaffected -- ``deep=True`` is opt-in,
    and legacy artifacts keep passing the default check exactly as before.

    Args:
        meta: Metadata already loaded and known non-None.

    Returns:
        True iff ``meta.derivation_binding`` equals a fresh
        :func:`_derivation_binding` of ``meta``'s own ``extractor_version``,
        ``sha256``, and ``extracted_sha256``.
    """
    if meta.derivation_binding is None or meta.extractor_version is None or meta.extracted_sha256 is None:
        return False
    recomputed = _derivation_binding(meta.sha256, meta.extracted_sha256, meta.extractor_version)
    return recomputed == meta.derivation_binding
