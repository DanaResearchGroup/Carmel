"""The manual paper-acquisition queue: request, drop, verify, admit.

Most papers a combustion campaign needs cannot be fetched (a live probe put the
end-to-end success rate at 3.3%), so this is a first-class workflow rather than a
fallback. The loop is:

1. Automated retrieval fails -> :func:`record_request` files the paper in
   ``<workspace>/literature_requests/manifest.json`` and names the file the operator
   should produce.
2. The operator downloads it through their institutional subscription and drops it in
   ``<workspace>/literature_requests/inbox/<slug>.pdf``.
3. :func:`collect_inbox` matches each dropped file to its request, VERIFIES the file is
   really that paper (:func:`check_identity`), and only then admits it to the
   content-addressed evidence store with ``ArtifactProvenance.MANUAL``.

Step 3's verification is the load-bearing part. Filename-to-request matching alone is
an unchecked human assertion: a mis-drop would silently attach one paper's bytes to
another paper's citation, and every downstream grounding check would then pass against
the wrong document -- producing a finding that is fully "grounded" and entirely false.
The quote-grounding gate cannot catch this, because it only ever asks whether the quote
appears in the supplied bytes, never whether those bytes are the right paper.

:func:`pending_requests`, :func:`drop_path_for` and :func:`admit_file` exist to shrink
the operator's loop for step 2: today it costs opening the manifest by hand, reading a
slug, naming the dropped file exactly right, and then re-running an entire literature
pass just to learn whether the drop was accepted. ``admit_file`` is a thin front door
onto :func:`collect_inbox` -- it copies one file into the inbox under the right name and
runs the identity check immediately, so the operator learns accepted-or-rejected in one
step instead of after the next full run.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from carmel.agents.tools.extract import ExtractedText, extract_text
from carmel.logger import get_logger
from carmel.paths import normalize_path
from carmel.schemas.acquisition import (
    AcquisitionManifest,
    AcquisitionReason,
    AcquisitionRequest,
    AcquisitionStatus,
)
from carmel.schemas.literature import ArtifactProvenance, StoredArtifact
from carmel.services.artifacts import read_json, write_json, write_text
from carmel.services.evidence import store_artifact

logger = get_logger("services.acquisition")

REQUESTS_DIR = "literature_requests"
INBOX_DIR = "inbox"
MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"

#: Fraction of a title's significant words that must appear in the document for a
#: title-only identity match. Deliberately high: title matching is the WEAKER of the two
#: checks (it is the fallback when no DOI is available), so it is not the place to be
#: generous. Cross-checked against real papers, whose front matter repeats the title
#: verbatim; the failure mode this guards against is a plausible-but-different paper on
#: the same topic, which shares topic words but not the full title.
TITLE_MATCH_THRESHOLD = 0.8

#: Fraction of a title's significant words that must appear when a DOI has ALSO matched.
#: Lower than :data:`TITLE_MATCH_THRESHOLD` because it corroborates an already-strong
#: signal instead of carrying the decision alone. NOT calibrated against a corpus --
#: chosen as half the standalone bar, on the reasoning that a genuine paper reprints its
#: own title verbatim in the front matter while the documents this rejects (landing
#: pages, cover sheets) carry mostly chrome.
DOI_CORROBORATION_THRESHOLD = 0.4

#: Front-matter phrases that mark a document as being *about* the requested paper rather
#: than being it. Each of these reprints the original's DOI and usually its full title,
#: so neither the DOI route nor the title route can separate them -- only the announcement
#: can. Matched against the first :data:`_MARKER_SCAN_CHARS` characters, where a journal
#: prints the article type, and suppressed when the requested title itself contains the
#: word (a paper genuinely titled "Comment on ..." is a legitimate request).
_SECONDARY_DOCUMENT_MARKERS: tuple[str, ...] = (
    "erratum",
    "corrigendum",
    "correction to",
    "comment on",
    "reply to",
    "retraction",
    "editorial expression of concern",
)

#: How far into the front matter to look for an article-type announcement. Much shorter
#: than :data:`IDENTITY_SEARCH_CHARS`: journals print the article type in the header, and
#: scanning further would match a mere mention of an erratum in the body or a footnote.
_MARKER_SCAN_CHARS = 600

#: Words too generic to distinguish one combustion paper from another.
_TITLE_STOPWORDS = frozenset(
    {
        "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to",
        "with", "using", "via", "study", "studies", "analysis", "effect", "effects",
        "new", "novel", "high", "low",
    }
)  # fmt: skip

#: How much of the document to search for identity evidence. The title and DOI live in
#: the front matter; scanning the whole document would also match a mere citation of the
#: paper inside some OTHER paper's reference list, which is exactly the confusion this
#: check exists to prevent.
IDENTITY_SEARCH_CHARS = 6000

_SLUG_SAFE_RE = re.compile(r"[^a-z0-9._-]+")
_VALID_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,180}$")
_WORD_RE = re.compile(r"[a-z0-9]+")


def requests_dir(workspace_root: Path) -> Path:
    """Return the acquisition-queue directory for a workspace."""
    return normalize_path(workspace_root) / REQUESTS_DIR


def inbox_dir(workspace_root: Path) -> Path:
    """Return the directory the operator drops obtained papers into."""
    return requests_dir(workspace_root) / INBOX_DIR


def manifest_path(workspace_root: Path) -> Path:
    """Return the path of the acquisition manifest."""
    return requests_dir(workspace_root) / MANIFEST_NAME


def drop_path_for(workspace_root: Path, slug: str, suffix: str = ".pdf") -> Path:
    """Return the exact path the operator should copy their downloaded file to.

    A caller (CLI, UI) prints this so the operator never has to reconstruct
    ``inbox/<slug>.pdf`` by hand -- getting that name exactly right is the whole reason
    this subsystem needs a human to type carefully in the first place.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    inbox = inbox_dir(workspace_root)
    candidate = inbox / f"{slug}{suffix}"
    # Defence in depth, on top of the schema's ``pattern`` constraint on ``slug``: this
    # function is also reachable with a raw string that never passed through
    # :class:`carmel.schemas.acquisition.AcquisitionRequest` validation (e.g. a CLI
    # ``--slug`` flag), so a ``../`` traversal attempt must be caught here too, not only
    # at the schema boundary.
    resolved_inbox = inbox.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_inbox):
        raise ValueError(
            f"slug {slug!r} would place the drop path outside the inbox directory "
            f"({resolved_candidate} is not inside {resolved_inbox})"
        )
    return candidate


def slug_for(doi: str | None, title: str) -> str:
    """Derive a filesystem-safe slug identifying one requested paper.

    Prefers the DOI, which is globally unique and stable; falls back to a truncated
    title plus a short digest so two same-titled papers cannot collide.

    Args:
        doi: Bare DOI (``10.xxxx/yyy``) if known.
        title: Paper title; used only when there is no DOI.

    Returns:
        A slug matching ``^[a-z0-9][a-z0-9._-]{0,180}$``, safe to use as a single path
        component. Never contains a path separator, so it cannot escape the inbox.
    """
    if doi:
        candidate = _SLUG_SAFE_RE.sub("-", doi.strip().lower()).strip("-")
    else:
        digest = hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()[:8]
        stem = _SLUG_SAFE_RE.sub("-", title.strip().lower()).strip("-")[:80].strip("-")
        candidate = f"{stem}-{digest}" if stem else f"paper-{digest}"

    candidate = candidate[:180].strip("-.") or "paper"
    if not _VALID_SLUG_RE.match(candidate):
        # Fall back to a pure digest rather than emitting anything path-unsafe.
        candidate = "paper-" + hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    return candidate


def load_manifest(workspace_root: Path) -> AcquisitionManifest:
    """Load the acquisition manifest, returning an empty one when absent or corrupt.

    Args:
        workspace_root: Root of the campaign workspace.

    Returns:
        The manifest; empty (rather than raising) if unreadable, so a damaged manifest
        never blocks a run -- it only loses queue history, which the run re-files.
    """
    path = manifest_path(workspace_root)
    if not path.exists():
        return AcquisitionManifest()
    try:
        return AcquisitionManifest.model_validate(read_json(path))
    # Deliberately NOT a bare ``except Exception``: a catch-all here silently converts a
    # programming error in this module into the indistinguishable message "your manifest
    # is corrupt" and loses the whole queue. (It already did exactly that once, hiding a
    # wrong-arity call to ``read_json``.) Only genuine on-disk damage is tolerated.
    except OSError, ValueError, ValidationError:
        logger.warning("acquisition manifest at %s is unreadable; starting a fresh one", path)
        return AcquisitionManifest()


def save_manifest(workspace_root: Path, manifest: AcquisitionManifest) -> None:
    """Persist the acquisition manifest and refresh the operator instructions."""
    directory = requests_dir(workspace_root)
    directory.mkdir(parents=True, exist_ok=True)
    inbox_dir(workspace_root).mkdir(parents=True, exist_ok=True)
    write_json(manifest_path(workspace_root), manifest)
    write_text(directory / README_NAME, _readme_text(manifest))


_PENDING_STATUSES = (AcquisitionStatus.REQUESTED, AcquisitionStatus.REJECTED)
"""Statuses a request can be admitted (or re-admitted) against.

REJECTED is included deliberately, not just REQUESTED: a rejection means the last drop
was the wrong file, not that the paper is no longer needed. Excluding it here would
drop the request from the operator's worklist at exactly the moment they most need to
see it again -- immediately after being told their first attempt was wrong."""


def pending_requests(workspace_root: Path) -> list[AcquisitionRequest]:
    """Return the requests still awaiting an admitted file (REQUESTED or REJECTED).

    See :data:`_PENDING_STATUSES` for why REJECTED counts as pending rather than final.
    """
    manifest = load_manifest(workspace_root)
    return [r for r in manifest.requests if r.status in _PENDING_STATUSES]


def record_request(
    workspace_root: Path,
    *,
    title: str,
    doi: str | None,
    landing_url: str,
    reason: AcquisitionReason,
    detail: str = "",
) -> AcquisitionRequest:
    """File a request for a paper Carmel could not obtain. Idempotent per slug.

    Args:
        workspace_root: Root of the campaign workspace.
        title: Paper title.
        doi: Bare DOI, if known.
        landing_url: Where a human should go to obtain it.
        reason: Why automated retrieval failed.
        detail: Optional human-readable specifics.

    Returns:
        The new request, or the existing one when this paper was already queued (a
        second failed attempt at the same paper must not enqueue it twice).
    """
    manifest = load_manifest(workspace_root)
    slug = slug_for(doi, title)
    for existing in manifest.requests:
        if existing.slug == slug:
            return existing

    request = AcquisitionRequest(
        slug=slug,
        title=title,
        doi=doi,
        landing_url=landing_url,
        reason=reason,
        detail=detail,
        requested_at=datetime.now(UTC),
    )
    manifest.requests.append(request)
    save_manifest(workspace_root, manifest)
    logger.info("queued paper for manual acquisition: %s (%s)", slug, reason.value)
    return request


def _secondary_document_marker(head: str, requested_title: str) -> str | None:
    """Return the phrase marking ``head`` as a document *about* the requested paper.

    Args:
        head: Lowercased front matter of the dropped document.
        requested_title: Lowercased title of the request, used to suppress the check for
            papers whose own title contains one of the marker phrases.

    Returns:
        The matched marker, or ``None`` when the document does not announce itself as a
        secondary document.
    """
    window = head[:_MARKER_SCAN_CHARS]
    for marker in _SECONDARY_DOCUMENT_MARKERS:
        if marker in window and marker not in requested_title:
            return marker
    return None


def _gate_on_secondary_document_marker(ok: bool, note: str, *, marker: str | None) -> tuple[bool, str]:
    """The single exit point every acceptance path in :func:`check_identity` must route
    through before returning ``True``.

    An erratum, corrigendum, comment, or reply has its OWN DOI, different from the
    original paper's -- so when a human drops one of these by mistake, the requested
    (original) DOI is simply absent from the text. That fails ``doi_found`` and sends
    the check down the title-only fallback, and a secondary document reprints the
    original's full title by construction ("Erratum to: <title>"), so the title-only
    ratio passes too. Neither identity route can separate an erratum from the paper it
    concerns; only the marker scan can. If the marker check is nested under just the
    DOI-matched branch, this exact substitution -- an erratum dropped in place of the
    original -- sails through the title-only branch untouched. Routing every ``True``
    through this single gate makes that structurally impossible: there is no accept
    path left that does not pass the marker check, including any added in the future.
    """
    if ok and marker:
        return False, (
            f"the document announces itself as a '{marker}' for the requested paper "
            f"rather than being that paper. Secondary documents like this commonly "
            f"reprint the original's DOI and its full title, so neither the DOI route "
            f"nor the title route can separate them from the original -- drop the "
            f"article itself"
        )
    return ok, note


def check_identity(extracted: ExtractedText, request: AcquisitionRequest) -> tuple[bool, str]:
    """Verify that a dropped document really is the requested paper.

    Two routes, strongest first:

    1. **DOI present, corroborated by the title.** A DOI is unique to the work and papers
       print their own DOI in the front matter, but a DOI alone is NOT accepted: an
       erratum, a comment, a reply, or a publisher landing page all reprint the DOI of
       the paper they concern. So a DOI match additionally requires
       :data:`DOI_CORROBORATION_THRESHOLD` of the title's significant words, and is
       refused outright when the front matter announces the document as a secondary one
       (see :data:`_SECONDARY_DOCUMENT_MARKERS`).
    2. **Title overlap alone.** Fallback when no DOI is known or the DOI is not printed.
       Requires the stricter :data:`TITLE_MATCH_THRESHOLD`, since nothing corroborates it.

    This mirrors the conjunction that :func:`carmel.services.grounding.check_identity`
    documents as load-bearing (``doi_ok and (title_ok or author_ok)``). The two functions
    answer the same question and must not disagree: this one gates the MANUAL acquisition
    path, and once a document is admitted the stricter rule never re-runs -- the
    quote-grounding gate only ever asks whether a quote appears in the supplied bytes,
    never whether those bytes are the right paper. The author half of the conjunction is
    unavailable here because :class:`AcquisitionRequest` carries no author list.

    Only the first :data:`IDENTITY_SEARCH_CHARS` characters are searched, so that a mere
    *citation* of the requested paper inside a different paper's reference list cannot
    masquerade as the paper itself.

    Args:
        extracted: Text extracted from the dropped file.
        request: The request the file was dropped against.

    Returns:
        ``(ok, note)``. ``note`` always explains the decision, including on failure, so
        the operator can tell "wrong paper" from "unreadable scan".
    """
    if not extracted.text.strip():
        return False, (
            "the dropped file yielded no extractable text (an image-only scan?); "
            "identity cannot be confirmed and no quote could ever be grounded against it"
        )

    head = extracted.text[:IDENTITY_SEARCH_CHARS].lower()
    collapsed = re.sub(r"\s+", "", head)

    title_words = [
        word for word in _WORD_RE.findall(request.title.lower()) if len(word) > 2 and word not in _TITLE_STOPWORDS
    ]
    present = sum(1 for word in set(title_words) if word in head) if title_words else 0
    ratio = present / len(set(title_words)) if title_words else 0.0

    doi_found = False
    if request.doi:
        doi = request.doi.lower()
        doi_found = doi in head or re.sub(r"\s+", "", doi) in collapsed

    # Computed ONCE, ahead of both accept branches below, and both branches' ``True``
    # returns are routed through :func:`_gate_on_secondary_document_marker` -- see that
    # function's docstring for why the marker check must never be nested under just the
    # DOI-matched branch.
    marker = _secondary_document_marker(head, request.title.lower())

    if doi_found:
        # A DOI in the front matter is strong but NOT self-sufficient, and this is the
        # deliberate difference from an earlier version of this function that returned
        # True here. The documents that print another work's DOI in their own front
        # matter are exactly the ones a human mis-drop produces: an erratum or comment
        # on the requested paper, a preprint cover page listing the published version's
        # DOI, or a publisher landing page saved as PDF. Each of those carries the right
        # DOI and is the wrong document, and nothing downstream re-asks the question --
        # once admitted, the quote-grounding gate only checks whether the quote appears
        # in these bytes, never whether these bytes are the right paper.
        #
        # So require corroboration, matching the conjunction that
        # :func:`carmel.services.grounding.check_identity` documents as load-bearing
        # (``doi_ok and (title_ok or author_ok)``). The threshold is lower than
        # TITLE_MATCH_THRESHOLD because it is corroborating an already-strong signal
        # rather than standing alone: a genuine paper reprints its own title verbatim in
        # its front matter, while a publisher landing page saved as PDF carries mostly
        # navigation chrome. NOT calibrated against a corpus; chosen as half of the
        # standalone bar. AcquisitionRequest carries no author list, so the author route
        # that grounding.check_identity can take is not available here.
        if ratio >= DOI_CORROBORATION_THRESHOLD:
            return _gate_on_secondary_document_marker(
                True,
                f"DOI {request.doi} found in the document's front matter, corroborated by "
                f"{ratio:.0%} of the title's significant words",
                marker=marker,
            )
        return False, (
            f"DOI {request.doi} appears in the front matter but nothing corroborates it: only "
            f"{ratio:.0%} of the title's significant words are present. This is what a publisher "
            f"landing page or a cover sheet for the requested paper looks like -- the right DOI "
            f"on the wrong document"
        )

    if not title_words:
        return False, "the request has no DOI and no distinctive title words to match on"

    if ratio >= TITLE_MATCH_THRESHOLD:
        # Title overlap alone does NOT separate a paper from its own erratum, which
        # reprints the full title by construction ("Erratum to: <title>") and has no DOI
        # of the original paper's in it at all -- that is exactly why this branch is
        # reached instead of the DOI branch above. The marker gate below is what catches
        # it; do not remove this call believing the DOI branch already covers erratum
        # documents, it only covers the ones that also print the original's DOI.
        return _gate_on_secondary_document_marker(
            True,
            f"title matched at {ratio:.0%} of significant words (no DOI in text)",
            marker=marker,
        )

    doi_note = " and its DOI was not found" if request.doi else ""
    return False, (
        f"the document does not look like this paper: only {ratio:.0%} of the title's "
        f"significant words appear in its first {IDENTITY_SEARCH_CHARS} characters"
        f"{doi_note}"
    )


def collect_inbox(workspace_root: Path, *, max_bytes: int) -> list[AcquisitionRequest]:
    """Admit verified dropped papers into the evidence store.

    Each file in the inbox is matched to a request by filename stem, extracted, and
    identity-checked. Only files that pass are stored (with
    :attr:`ArtifactProvenance.MANUAL`); a file that fails marks its request
    ``REJECTED`` with the reason, and its bytes are NOT admitted.

    Args:
        workspace_root: Root of the campaign workspace.
        max_bytes: Hard cap on a single artifact's size.

    Returns:
        The requests whose state changed during this sweep.
    """
    inbox = inbox_dir(workspace_root)
    if not inbox.is_dir():
        return []

    manifest = load_manifest(workspace_root)
    by_slug = {request.slug: request for request in manifest.requests}
    changed: list[AcquisitionRequest] = []

    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        request = by_slug.get(path.stem.lower())
        if request is None:
            logger.warning("dropped file %s matches no queued request; leaving it untouched", path.name)
            continue
        if request.status == AcquisitionStatus.FULFILLED:
            continue

        stored = _admit_one(workspace_root, path, request, max_bytes=max_bytes)
        if stored is not None:
            request.status = AcquisitionStatus.FULFILLED
            request.fulfilled_sha256 = stored.sha256
        changed.append(request)

    if changed:
        save_manifest(workspace_root, manifest)
    return changed


def _sniff_content_type(data: bytes) -> str:
    """Determine a dropped file's type from its bytes, never from its filename.

    The filename is chosen by the operator to name the REQUEST, so its extension says
    nothing reliable about the contents. Classifying everything non-PDF as opaque bytes
    would extract no text from a perfectly good HTML or plain-text copy of a paper and
    then misreport it as an image-only scan.

    Args:
        data: The dropped file's bytes.

    Returns:
        A MIME type understood by :func:`carmel.agents.tools.extract.extract_text`.
    """
    if data[:5].startswith(b"%PDF"):
        return "application/pdf"
    try:
        head = data[:4096].decode("utf-8").lstrip().lower()
    except UnicodeDecodeError:
        return "application/octet-stream"
    if head.startswith(("<!doctype html", "<html", "<?xml")):
        return "text/html"
    return "text/plain"


def _admit_one(
    workspace_root: Path, path: Path, request: AcquisitionRequest, *, max_bytes: int
) -> StoredArtifact | None:
    """Extract, identity-check and store one dropped file. Mutates ``request``'s notes.

    Returns:
        The stored artifact, or ``None`` when the file was rejected for any reason.
    """
    from carmel.agents.tools.fetch import FetchedArtifact

    # Check the size via stat BEFORE reading: a huge dropped file must be rejected
    # without ever being pulled fully into memory, or the cap below defeats its own
    # purpose. The stat check is cheap and reads nothing; it is not, on its own,
    # sufficient (the file could grow between this stat and the read below), so the
    # length check after the read stays in place too -- this is a belt-and-suspenders
    # pair, not a replacement.
    try:
        st_size = path.stat().st_size
    except OSError as exc:
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"could not read the dropped file: {exc}"
        return None
    if st_size > max_bytes:
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"dropped file is {st_size} bytes, over the {max_bytes} cap"
        return None

    try:
        data = path.read_bytes()
    except OSError as exc:
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"could not read the dropped file: {exc}"
        return None

    # TOCTOU: the file can grow between the stat above and this read completing, so the
    # cap is enforced a second time here, against what was actually read.
    if len(data) > max_bytes:
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"dropped file is {len(data)} bytes, over the {max_bytes} cap"
        return None

    content_type = _sniff_content_type(data)
    try:
        extracted = extract_text(data, content_type)
    except Exception as exc:  # noqa: BLE001 - a bad drop must not kill the run
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"could not extract text from the dropped file: {exc}"
        return None

    ok, note = check_identity(extracted, request)
    request.identity_note = note
    if not ok:
        request.status = AcquisitionStatus.REJECTED
        logger.warning("rejected dropped file for %s: %s", request.slug, note)
        return None

    digest = hashlib.sha256(data).hexdigest()
    artifact = FetchedArtifact(
        # The landing URL records where the paper CAME from; the bytes did not travel
        # that path in this process, which is exactly what ``provenance=MANUAL`` says.
        url=request.landing_url,
        final_url=request.landing_url,
        sha256=digest,
        content_type=content_type,
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    try:
        return store_artifact(
            workspace_root,
            data=data,
            artifact=artifact,
            extracted=extracted,
            license_note="manually acquired by the operator; not redistributable",
            provenance=ArtifactProvenance.MANUAL,
            max_bytes=max_bytes,
        )
    except ValueError as exc:
        request.status = AcquisitionStatus.REJECTED
        request.identity_note = f"storing the dropped file failed: {exc}"
        return None


def _infer_slug(source: Path, pending: list[AcquisitionRequest], *, max_bytes: int) -> str:
    """Work out which pending request ``source`` is meant to fulfil, when the caller
    did not say. NEVER guesses between two plausible papers: a wrong guess attaches
    one paper's bytes to another paper's citation, exactly the failure this whole
    subsystem exists to prevent.

    Args:
        source: The operator's file, not yet copied anywhere.
        pending: Candidate requests (see :func:`pending_requests`).
        max_bytes: Hard cap on the file read for inference -- the same cap
            :func:`_admit_one` enforces on the file actually admitted, applied here too
            since this reads the same untrusted operator-dropped bytes.

    Returns:
        The one slug this file can be confidently matched to.

    Raises:
        ValueError: no requests are pending, the file is over ``max_bytes``, or the
            file's extracted text does not let :func:`check_identity` settle on exactly
            one candidate.
    """
    if not pending:
        raise ValueError("no acquisition requests are pending; there is nothing to admit this file against")
    if len(pending) == 1:
        return pending[0].slug

    candidates = ", ".join(sorted(r.slug for r in pending))

    # Stat before reading, same reasoning as in `_admit_one`: a huge file must be
    # rejected without being pulled fully into memory first.
    try:
        st_size = source.stat().st_size
    except OSError as exc:
        raise ValueError(
            f"could not read {source} well enough to infer which request it is for: {exc}. "
            f"Pass slug= explicitly. Candidates: {candidates}"
        ) from exc
    if st_size > max_bytes:
        raise ValueError(
            f"{source} is {st_size} bytes, over the {max_bytes} cap, so it cannot be read to "
            f"infer which request it is for. Pass slug= explicitly. Candidates: {candidates}"
        )

    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"could not read {source} well enough to infer which request it is for: {exc}. "
            f"Pass slug= explicitly. Candidates: {candidates}"
        ) from exc

    # TOCTOU: enforce the cap again against what was actually read.
    if len(data) > max_bytes:
        raise ValueError(
            f"{source} is {len(data)} bytes, over the {max_bytes} cap, so it cannot be read to "
            f"infer which request it is for. Pass slug= explicitly. Candidates: {candidates}"
        )

    try:
        extracted = extract_text(data, _sniff_content_type(data))
    except Exception as exc:  # noqa: BLE001 - inference failing must not crash, just refuse to guess
        raise ValueError(
            f"could not read {source} well enough to infer which request it is for: {exc}. "
            f"Pass slug= explicitly. Candidates: {candidates}"
        ) from exc

    matches = [r.slug for r in pending if check_identity(extracted, r)[0]]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"cannot tell which pending request this file is for ({len(matches)} matched); "
        f"pass slug= explicitly. Candidates: {candidates}"
    )


def admit_file(workspace_root: Path, source: Path, *, slug: str | None = None, max_bytes: int) -> AcquisitionRequest:
    """Copy ``source`` into the inbox under the right name and identity-check it now.

    This is the fast front door onto :func:`collect_inbox`: instead of the operator
    hand-editing the manifest, hand-naming the file, and re-running an entire
    literature pass just to learn whether the drop was accepted, this does the copy
    and the check in one call and returns the outcome immediately.

    Args:
        workspace_root: Root of the campaign workspace.
        source: The operator's downloaded file. Read-only: never mutated or deleted,
            and never the file actually stored (a copy is made).
        slug: Which request this file is for. If omitted: the sole pending request if
            there is exactly one, otherwise inferred by running :func:`check_identity`
            against every pending request and requiring exactly one match (see
            :func:`_infer_slug`).
        max_bytes: Hard cap on the admitted artifact's size, enforced by
            :func:`_admit_one` exactly as :func:`collect_inbox` enforces it -- an
            operator-dropped file is the same untrusted-input path as a fetched one.

    Returns:
        The request after the attempt: ``FULFILLED`` with ``fulfilled_sha256`` set on
        success, or ``REJECTED`` with ``identity_note`` explaining why on failure. The
        caller can report this straight back to the operator without a fresh run.

    Raises:
        ValueError: ``source`` does not exist, is a directory, cannot be copied, no
            slug could be confidently resolved, or the given/resolved slug matches no
            queued request.
    """
    if not source.exists():
        raise ValueError(f"source file does not exist: {source}")
    if source.is_dir():
        raise ValueError(f"source is a directory, not a file: {source}")

    manifest = load_manifest(workspace_root)
    by_slug = {r.slug: r for r in manifest.requests}

    if slug is None:
        pending = [r for r in manifest.requests if r.status in _PENDING_STATUSES]
        slug = _infer_slug(source, pending, max_bytes=max_bytes)

    request = by_slug.get(slug)
    if request is None:
        known = ", ".join(sorted(by_slug)) or "(none queued)"
        raise ValueError(f"no acquisition request queued under slug {slug!r}; known slugs: {known}")

    drop_path = drop_path_for(workspace_root, slug, suffix=source.suffix or ".pdf")
    inbox_dir(workspace_root).mkdir(parents=True, exist_ok=True)
    try:
        # Overwrite, deliberately, rather than refuse: the whole point of this function
        # is to shorten the retry loop after a wrong drop, and the file is re-verified
        # immediately below, so a stale inbox copy is never left standing as the thing
        # that determines the request's fate -- this call's ``source`` always is.
        shutil.copyfile(source, drop_path)
    except OSError as exc:
        raise ValueError(f"could not copy {source} into the inbox: {exc}") from exc

    stored = _admit_one(workspace_root, drop_path, request, max_bytes=max_bytes)
    if stored is not None:
        request.status = AcquisitionStatus.FULFILLED
        request.fulfilled_sha256 = stored.sha256
    save_manifest(workspace_root, manifest)
    return request


def _readme_text(manifest: AcquisitionManifest) -> str:
    """Render operator instructions listing every outstanding request."""
    pending = [r for r in manifest.requests if r.status == AcquisitionStatus.REQUESTED]
    rejected = [r for r in manifest.requests if r.status == AcquisitionStatus.REJECTED]

    lines = [
        "# Papers Carmel needs you to obtain",
        "",
        "Carmel could not retrieve these papers automatically -- most are behind a",
        "publisher paywall. Please download each one (institutional subscription,",
        "interlibrary loan, or by asking the authors) and save it into the `inbox/`",
        "directory next to this file, using EXACTLY the filename shown below.",
        "",
        "Carmel checks each dropped file really is the paper requested -- by finding the",
        "DOI or the title inside the document -- before it will use it as evidence. A file",
        "that does not match is reported back here rather than being used.",
        "",
    ]

    if pending:
        lines.append(f"## Outstanding ({len(pending)})")
        lines.append("")
        for request in pending:
            lines.append(f"### {request.title}")
            lines.append("")
            lines.append(f"- Save as: `inbox/{request.slug}.pdf`")
            if request.doi:
                lines.append(f"- DOI: `{request.doi}`")
            lines.append(f"- Obtain from: {request.landing_url}")
            detail = f" ({request.detail})" if request.detail else ""
            lines.append(f"- Why manual: {request.reason.value}{detail}")
            lines.append("")
    else:
        lines.append("## Outstanding (0)")
        lines.append("")
        lines.append("Nothing is waiting on you right now.")
        lines.append("")

    if rejected:
        lines.append(f"## Needs attention ({len(rejected)})")
        lines.append("")
        lines.append("A file was dropped for these, but it did not pass the identity check:")
        lines.append("")
        for request in rejected:
            lines.append(f"- `{request.slug}.pdf` -- {request.identity_note}")
        lines.append("")

    return "\n".join(lines)
