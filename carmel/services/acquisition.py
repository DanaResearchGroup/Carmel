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
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from carmel.agents.tools.extract import ExtractedText, extract_text
from carmel.logger import get_logger
from carmel.paths import normalize_path
from carmel.schemas.acquisition import (
    CURRENT_ACQUISITION_MANIFEST_VERSION,
    AcquisitionManifest,
    AcquisitionReason,
    AcquisitionRequest,
    AcquisitionStatus,
)
from carmel.schemas.literature import ArtifactProvenance, StoredArtifact
from carmel.services.artifacts import read_json, write_json, write_text
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.grounding import (
    secondary_document_marker,
)

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
#: than being it. Defined in :mod:`carmel.services.grounding` and re-exported here under
#: the names this module already used, so the two layers that gate on secondary documents
#: -- this one at store entry, grounding at attribution time -- share ONE vocabulary.
#: Adding a marker to only one of them would silently reopen the hole in the other.
#:
#: Matched against the first :data:`MARKER_SCAN_CHARS` characters, where a journal prints
#: the article type -- much shorter than :data:`IDENTITY_SEARCH_CHARS`, since scanning
#: further would match a mere mention of an erratum in the body or a footnote. Suppressed
#: when the requested title itself contains the word (a paper genuinely titled "Comment
#: on ..." is a legitimate request).
#: Words too generic to distinguish one combustion paper from another.
#: Hosts whose documents may enter the evidence store AUTOMATICALLY.
#:
#: The identity gate asks whether a document contains the cited title and DOI outside
#: its reference list. A document that merely PRINTS another paper's title and DOI in
#: its body satisfies that too, so its own prose can be recorded as grounded under the
#: impersonated citation (provenance survives -- ``EvidenceRef`` and ``source_url``
#: still name the true artifact -- but attribution does not).
#:
#: Measured on the 8-paper live corpus, this does not happen by accident: every DOI
#: appearing outside a references section was the document's OWN front-matter DOI, and
#: each testable citation was confirmed only by its own document. It needs a document
#: that prints a foreign title and DOI in its body -- which an attacker-controlled page
#: reached through a poisoned search result can do deliberately.
#:
#: Tightening the gate instead would mean a "front matter only" window, and the same
#: corpus says a paper's own DOI sits 2.6k-5.1k characters in (5.8%-15.1%) and is
#: labelled ``body``, never ``abstract``. A threshold with that little headroom,
#: calibrated on eight documents, is the shape that produced the F1 identity bug. So
#: the control lives here instead, where it costs nothing to be strict.
#:
#: Matching is on the registrable suffix: an entry matches the host itself and any
#: subdomain of it. A non-matching host is not a failure -- the paper is queued for
#: MANUAL acquisition, which runs its own identity check on admission.
DEFAULT_ADMISSIBLE_HOSTS: frozenset[str] = frozenset(
    {
        # Resolvers and registries
        "doi.org",
        "crossref.org",
        "openalex.org",
        "unpaywall.org",
        "semanticscholar.org",
        "core.ac.uk",
        # Publishers
        "sciencedirect.com",
        "elsevier.com",
        "els-cdn.com",
        "springer.com",
        "springernature.com",
        "wiley.com",
        "tandfonline.com",
        "acs.org",
        "rsc.org",
        "aip.org",
        "iop.org",
        "nature.com",
        "science.org",
        "pnas.org",
        "cambridge.org",
        "oup.com",
        "sagepub.com",
        "mdpi.com",
        "frontiersin.org",
        "plos.org",
        "hindawi.com",
        "degruyter.com",
        "aiaa.org",
        "asme.org",
        # Preprints, repositories, government
        "arxiv.org",
        "chemrxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "osf.io",
        "zenodo.org",
        "figshare.com",
        "osti.gov",
        "nih.gov",
        "nasa.gov",
    }
)


def host_is_admissible(url: str, additional_hosts: Iterable[str] = ()) -> bool:
    """Whether a document from ``url`` may enter the evidence store automatically.

    Fail-closed by construction: anything that does not parse, carries no host, or
    does not match an allowed suffix is refused. ``additional_hosts`` is the operator's
    extension point (an institutional proxy, a lab mirror) -- there is deliberately no
    switch that disables the check, because "allow everything" is the configuration
    this exists to prevent, and adding the one host you need is never harder.

    Matching is on the registrable suffix, never a substring: ``sciencedirect.com``
    admits ``www.sciencedirect.com`` but NOT ``sciencedirect.com.evil.net``, which a
    substring test would wave through.

    Args:
        url: The candidate URL.
        additional_hosts: Extra admissible hosts from configuration.

    Returns:
        True if the URL's host is, or is a subdomain of, an admissible host.
    """
    try:
        host = (urlsplit(url).hostname or "").lower().strip(".")
    except ValueError:
        return False
    if not host:
        return False
    allowed = {h.lower().strip(".") for h in (*DEFAULT_ADMISSIBLE_HOSTS, *additional_hosts) if h}
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


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


class ManifestUnreadable(ValueError):
    """The on-disk acquisition manifest exists but could not be loaded.

    A distinct type -- rather than letting the underlying ``OSError``/``ValueError``
    propagate bare -- so every caller sees the SAME thing regardless of which of the
    four ways a manifest can fail actually happened, and gets a specific,
    human-actionable reason rather than a generic traceback. Deliberately a
    :class:`ValueError` subclass, matching :class:`AlreadyAcquired` and
    :class:`~carmel.services.literature.ReportSchemaTooNewError`: existing callers that
    catch ``ValueError`` broadly around acquisition calls still catch this.

    This is never raised for an ABSENT manifest -- that is legitimate first-run
    behaviour and :func:`load_manifest` still returns an empty manifest for it. It is
    raised for every other failure precisely because the manifest is the operator's
    outstanding download queue, not disposable scratch: silently starting fresh over a
    damaged file used to destroy that queue (and, via :func:`save_manifest`, the
    ``README.md`` telling the operator what they still needed to go get) the moment the
    next request was recorded.

    Attributes:
        path: The manifest file that could not be loaded.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"acquisition manifest at {path} is unreadable: {reason}")
        self.path = path


class _ManifestSchemaTooNew(ValueError):
    """Internal signal: ``migrate_manifest_payload`` saw a version newer than this
    build understands. Caught and rewrapped as :class:`ManifestUnreadable` by
    :func:`load_manifest`, which is the only place that knows the on-disk path -- this
    type exists purely so :func:`migrate_manifest_payload` (which, like
    :func:`carmel.services.literature.migrate_report_payload`, only ever sees the raw
    dict, never the path) does not have to fabricate one."""


def migrate_manifest_payload(payload: object) -> object:
    """Bring a persisted manifest payload up to the current schema version.

    Mirrors :func:`carmel.services.literature.migrate_report_payload`: a chain applied
    step by step from the payload's own version, so a bump two versions from now does
    not have to re-derive what an intermediate version looked like. There is exactly
    one version so far, so the chain is currently empty -- the next bump adds a
    ``_migrate_v1_to_v2`` step here, matching that module's structure.

    Raises:
        _ManifestSchemaTooNew: If ``payload``'s ``version`` is newer than
            :data:`~carmel.schemas.acquisition.CURRENT_ACQUISITION_MANIFEST_VERSION`.
            Rewrapped as :class:`ManifestUnreadable` by :func:`load_manifest`.
    """
    if not isinstance(payload, dict):
        return payload
    version = int(payload.get("version", 1))
    if version > CURRENT_ACQUISITION_MANIFEST_VERSION:
        # Fail closed on a manifest from the FUTURE, exactly as
        # ``migrate_report_payload`` does: passing it through unmigrated hands it to a
        # validator with ``extra="forbid"``, which would reject an unrecognised field
        # with a schema error naming something the operator has never heard of -- or,
        # worse, silently validate and then have this older Carmel write back a
        # truncated version of a newer manifest the next time a request is recorded.
        raise _ManifestSchemaTooNew(
            f"written by a newer Carmel (manifest schema version {version}, this build "
            f"understands at most {CURRENT_ACQUISITION_MANIFEST_VERSION}); upgrade "
            f"Carmel rather than letting an older version rewrite (and silently "
            f"truncate) a newer manifest"
        )
    if version == CURRENT_ACQUISITION_MANIFEST_VERSION:
        return payload

    migrated = dict(payload)
    # No migration steps exist yet -- add them here as ``if version < N:`` blocks,
    # matching ``migrate_report_payload``, the first time the schema changes.
    migrated["version"] = CURRENT_ACQUISITION_MANIFEST_VERSION
    return migrated


def load_manifest(workspace_root: Path) -> AcquisitionManifest:
    """Load the acquisition manifest, or an empty one if none has been written yet.

    Args:
        workspace_root: Root of the campaign workspace.

    Returns:
        The manifest -- empty ONLY when no manifest file exists yet (legitimate
        first-run behaviour).

    Raises:
        ManifestUnreadable: The manifest file exists but could not be loaded: it is
            unreadable (permissions, I/O error), not valid JSON, valid JSON that fails
            schema validation even after migration, or was written by a newer Carmel
            than this build understands. In every case the on-disk file is left
            untouched -- this function never writes -- so the operator's queue survives
            for manual recovery.
    """
    path = manifest_path(workspace_root)
    if not path.exists():
        return AcquisitionManifest()
    try:
        payload = read_json(path)
    except OSError as exc:
        raise ManifestUnreadable(path, f"could not be read ({exc})") from exc
    except ValueError as exc:
        raise ManifestUnreadable(path, f"is not valid JSON ({exc})") from exc
    try:
        migrated = migrate_manifest_payload(payload)
    except _ManifestSchemaTooNew as exc:
        raise ManifestUnreadable(path, str(exc)) from exc
    try:
        return AcquisitionManifest.model_validate(migrated)
    except ValidationError as exc:
        raise ManifestUnreadable(path, f"fails schema validation ({exc})") from exc


def save_manifest(workspace_root: Path, manifest: AcquisitionManifest) -> None:
    """Persist the acquisition manifest and refresh the operator instructions.

    Raises:
        ValueError: ``manifest`` carries zero requests while an on-disk manifest with
            at least one request already exists. This is a belt-and-braces invariant
            independent of :func:`load_manifest`'s own fail-closed behaviour: whatever
            upstream bug or corrupt read produced an empty manifest in memory, this
            function refuses to be the thing that overwrites the operator's actual
            outstanding queue (and regenerates ``README.md`` to say nothing is
            waiting on them) with it. An empty manifest is only ever a legitimate
            write when there is nothing on disk to lose.
    """
    if not manifest.requests:
        existing_path = manifest_path(workspace_root)
        if existing_path.exists():
            try:
                on_disk = AcquisitionManifest.model_validate(migrate_manifest_payload(read_json(existing_path)))
            except OSError, ValueError, ValidationError:
                # Unreadable is treated as "unknown, possibly populated" -- the same
                # fail-closed instinct as everywhere else in this module -- rather than
                # assumed empty just because it could not be parsed.
                on_disk = None
            if on_disk is None or on_disk.requests:
                held = (
                    "could not be read, so it may still hold requests"
                    if on_disk is None
                    else f"holds {len(on_disk.requests)} request(s)"
                )
                raise ValueError(
                    f"refusing to write an empty acquisition manifest over the existing one "
                    f"at {existing_path}, which {held}; this would silently destroy the "
                    f"operator's outstanding download queue"
                )
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


#: Alias of :func:`carmel.services.grounding.secondary_document_marker`, kept under this
#: module's original private name so the call sites below read unchanged. The detector
#: itself now lives beside the marker vocabulary it consults.
_secondary_document_marker = secondary_document_marker


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


class AlreadyAcquired(ValueError):
    """A dropped file is a paper the evidence store already holds.

    Deliberately a :class:`ValueError` subclass: every existing caller catches
    ``ValueError`` around :func:`admit_file`, and this must not become a crash for one
    that has not been taught the distinction. Callers that HAVE been taught it catch this
    first and report a skip, because re-offering a paper Carmel already has is a no-op,
    not an error -- reporting it as a rejection tells the operator their download was bad
    when it was in fact accepted on an earlier pass.

    Attributes:
        slug: The request this file was already acquired for.
    """

    def __init__(self, slug: str, detail: str) -> None:
        super().__init__(f"already acquired as {slug} ({detail}); nothing to do")
        self.slug = slug


def _doi_in_front_matter(extracted: ExtractedText, request: AcquisitionRequest) -> bool:
    """Whether ``request``'s DOI is printed in the document's own front matter.

    Split out of :func:`check_identity` so that the *strength* of a match can be asked
    about separately from whether it passed, with exactly one implementation of the
    whitespace-tolerant comparison. A DOI can be broken across lines by PDF extraction
    ("10.1016/j.ijhy\\ndene.2012.10.075"), so the collapsed form is compared too.

    This answers a strictly narrower question than :func:`check_identity` and must never
    be used in its place: on its own a DOI hit does NOT establish identity (an erratum or
    a landing page prints the DOI of the paper it concerns). It is only ever used to rank
    candidates that have ALREADY passed the full check.
    """
    if not request.doi:
        return False
    head = extracted.text[:IDENTITY_SEARCH_CHARS].lower()
    doi = request.doi.lower()
    return doi in head or re.sub(r"\s+", "", doi) in re.sub(r"\s+", "", head)


def check_identity(extracted: ExtractedText, request: AcquisitionRequest) -> tuple[bool, str]:
    """Verify that a dropped document really is the requested paper.

    Two routes, strongest first:

    1. **DOI present, corroborated by the title.** A DOI is unique to the work and papers
       print their own DOI in the front matter, but a DOI alone is NOT accepted: an
       erratum, a comment, a reply, or a publisher landing page all reprint the DOI of
       the paper they concern. So a DOI match additionally requires
       :data:`DOI_CORROBORATION_THRESHOLD` of the title's significant words, and is
       refused outright when the front matter announces the document as a secondary one
       (see :data:`SECONDARY_DOCUMENT_MARKERS`).
    2. **Title overlap alone.** Fallback when no DOI is known or the DOI is not printed.
       Requires the stricter :data:`TITLE_MATCH_THRESHOLD`, since nothing corroborates it.

    This mirrors the conjunction that :func:`carmel.services.grounding.check_identity`
    documents as load-bearing. That rule is now ``doi_ok and title_ok``: the
    surname-based escape from the title requirement was removed, because a review or
    discussion article can carry every weak signal it rested on honestly. The two
    functions answer the same question and must not disagree: this one gates the MANUAL
    acquisition path, and once a document is admitted the stricter rule never re-runs --
    the quote-grounding gate only ever asks whether a quote appears in the supplied
    bytes, never whether those bytes are the right paper.

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

    title_words = [
        word for word in _WORD_RE.findall(request.title.lower()) if len(word) > 2 and word not in _TITLE_STOPWORDS
    ]
    present = sum(1 for word in set(title_words) if word in head) if title_words else 0
    ratio = present / len(set(title_words)) if title_words else 0.0

    doi_found = _doi_in_front_matter(extracted, request)

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
        # (``doi_ok and title_ok``). The threshold is lower than
        # TITLE_MATCH_THRESHOLD because it is corroborating an already-strong signal
        # rather than standing alone: a genuine paper reprints its own title verbatim in
        # its front matter, while a publisher landing page saved as PDF carries mostly
        # navigation chrome. NOT calibrated against a corpus; chosen as half of the
        # standalone bar.
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


def _already_stored_slug(
    workspace_root: Path, source: Path, acquired: Sequence[AcquisitionRequest], *, max_bytes: int
) -> tuple[str, bool] | None:
    """Match ``source``'s bytes against the artifacts fulfilled requests claim to hold.

    The manifest alone is NOT taken as proof that the paper is present.
    ``fulfilled_sha256`` is a record written when the paper was admitted; the evidence
    store is what actually holds the bytes, and the two can diverge -- a pruned evidence
    directory, a workspace copied without its artifacts, a manifest restored from backup.
    Reporting "already acquired" off the record alone would tell the operator Carmel has
    a paper it no longer has, and because the report is a silent skip they would have no
    way to find out. So presence in the store is checked separately and returned, rather
    than folded into the match.

    Args:
        workspace_root: Root of the campaign workspace, for locating the evidence store.
        source: The operator's file, not yet copied anywhere.
        acquired: Fulfilled requests, each carrying its artifact's ``fulfilled_sha256``.
        max_bytes: Same cap the admission path enforces. A file over the cap is not
            hashed -- it cannot be admitted anyway, and the point of the cap is to avoid
            reading oversized operator input into memory at all.

    Returns:
        ``(slug, present_in_store)`` when the file's bytes are exactly the artifact that
        request recorded, or ``None`` when the file is over the cap, unreadable, or not
        one of them. ``present_in_store=False`` means the record is there but the bytes
        are gone, which the caller repairs by re-admitting. Never raises: this is a
        shortcut on the way to the real checks, so any difficulty must fall through to
        them rather than becoming the operator's error message.
    """
    by_sha = {r.fulfilled_sha256: r.slug for r in acquired if r.fulfilled_sha256}
    if not by_sha:
        return None
    try:
        if source.stat().st_size > max_bytes:
            return None
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        return None
    slug = by_sha.get(digest)
    if slug is None:
        return None
    try:
        present = artifact_dir(workspace_root, digest).is_dir()
    except OSError:
        present = False
    return slug, present


def _infer_slug(
    source: Path,
    pending: list[AcquisitionRequest],
    *,
    max_bytes: int,
    acquired: Sequence[AcquisitionRequest] = (),
) -> tuple[str, bool]:
    """Work out which pending request ``source`` is meant to fulfil, when the caller
    did not say. NEVER guesses between two plausible papers: a wrong guess attaches
    one paper's bytes to another paper's citation, exactly the failure this whole
    subsystem exists to prevent.

    When exactly one request is outstanding and nothing matched, the file is still
    attributed to it -- but the second element of the return value says so, so the
    caller can label the resulting rejection honestly. That distinction is the fix for
    an observed batch-ingest defect: attributing on "it is the only one left" alone let
    four unrelated files each flip one innocent request to ``REJECTED``, leaving a note
    accusing it of not looking like itself, and it survived only because the file that
    genuinely matched it happened to sort last. The attribution is kept because it is
    what produces the precise diagnostic an operator needs ("this is an erratum", "the
    file is over the cap") instead of a bare "matched nothing"; what changes is that the
    note no longer claims the operator offered this file for that paper.

    Args:
        source: The operator's file, not yet copied anywhere.
        pending: Candidate requests (see :func:`pending_requests`).
        max_bytes: Hard cap on the file read for inference -- the same cap
            :func:`_admit_one` enforces on the file actually admitted, applied here too
            since this reads the same untrusted operator-dropped bytes.
        acquired: Already-fulfilled requests, consulted ONLY when no pending request
            matches, so that re-offering a paper Carmel already holds is reported as
            such instead of as an unresolvable file.

    Returns:
        ``(slug, matched_on_evidence)``. ``matched_on_evidence`` is ``True`` when the
        document's own text identified the request, and ``False`` when it was attributed
        to the sole outstanding request without matching it.

    Raises:
        AlreadyAcquired: the file is a paper already in the evidence store.
        ValueError: no requests are pending, or -- with more than one outstanding -- the
            file could not be read or its text does not settle on exactly one candidate.
    """
    if not pending:
        raise ValueError("no acquisition requests are pending; there is nothing to admit this file against")

    candidates = ", ".join(sorted(r.slug for r in pending))
    # With one request outstanding there is no ambiguity to resolve, so a file we cannot
    # read is still attributed to it and rejected with a real reason ("over the cap",
    # "no extractable text"). With several, an unreadable file is genuinely unresolvable
    # and must raise rather than pick one.
    sole = pending[0].slug if len(pending) == 1 else None

    def _unresolvable(detail: str) -> ValueError:
        return ValueError(
            f"could not read {source} well enough to infer which request it is for: {detail}. "
            f"Pass slug= explicitly. Candidates: {candidates}"
        )

    # Stat before reading, same reasoning as in `_admit_one`: a huge file must be
    # rejected without being pulled fully into memory first.
    try:
        st_size = source.stat().st_size
    except OSError as exc:
        if sole is not None:
            return sole, False
        raise _unresolvable(str(exc)) from exc
    if st_size > max_bytes:
        if sole is not None:
            return sole, False
        raise ValueError(
            f"{source} is {st_size} bytes, over the {max_bytes} cap, so it cannot be read to "
            f"infer which request it is for. Pass slug= explicitly. Candidates: {candidates}"
        )

    try:
        data = source.read_bytes()
    except OSError as exc:
        if sole is not None:
            return sole, False
        raise _unresolvable(str(exc)) from exc

    # TOCTOU: enforce the cap again against what was actually read.
    if len(data) > max_bytes:
        if sole is not None:
            return sole, False
        raise ValueError(
            f"{source} is {len(data)} bytes, over the {max_bytes} cap, so it cannot be read to "
            f"infer which request it is for. Pass slug= explicitly. Candidates: {candidates}"
        )

    try:
        extracted = extract_text(data, _sniff_content_type(data))
    except Exception as exc:  # noqa: BLE001 - inference failing must not crash, just refuse to guess
        if sole is not None:
            return sole, False
        raise _unresolvable(str(exc)) from exc

    matches = [r for r in pending if check_identity(extracted, r)[0]]

    if len(matches) > 1:
        # Break the tie by preferring the STRONGER signal, never a weaker one: a
        # candidate whose DOI is printed in this document's own front matter beats
        # candidates that matched on title words alone. Papers in one focused campaign
        # share a subfield vocabulary heavily enough that title-only overlap collides by
        # construction ("laminar flame speeds of ... syngas ... mixtures" describes
        # several of them), while a DOI is unique to the work.
        #
        # This only ever narrows a set that ALREADY passed the full check, so it cannot
        # admit anything :func:`check_identity` refused. That direction is the whole
        # discipline: the fix for an ambiguous match is to demand a stronger signal, and
        # NEVER to lower a threshold or blend the checks into a fuzzy similarity score --
        # an erratum scores ~0.95 against its own paper's title and would sail through
        # any such score, reopening precisely the defect the secondary-document marker
        # gate exists to close.
        doi_matches = [r for r in matches if _doi_in_front_matter(extracted, r)]
        if len(doi_matches) == 1:
            matches = doi_matches

    if len(matches) == 1:
        return matches[0].slug, True

    if not matches:
        # Nothing pending matched. Before treating this as a mismatch, ask whether it is a
        # paper already in the store -- the common, entirely benign case of re-running an
        # ingest over a download folder whose earlier files were accepted on a previous
        # pass. Reporting that as a failure trains the operator to distrust correct
        # verdicts. Checked ahead of the sole-request fallback below so that an
        # already-held paper is never re-blamed on whatever is still outstanding.
        for request in acquired:
            if check_identity(extracted, request)[0]:
                raise AlreadyAcquired(request.slug, "its content matches a request already fulfilled")
        if sole is not None:
            return sole, False

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

    # An explicitly passed slug IS the operator's assertion that this file is for that
    # paper, so a mismatch is theirs to hear about plainly; only inference can attribute
    # a file to a request the operator never named.
    matched_on_evidence = True

    if slug is None:
        acquired = [r for r in manifest.requests if r.status == AcquisitionStatus.FULFILLED]

        # Exact bytes first: if this file IS an artifact already in the store, that is
        # decisive and costs one hash, with no text extraction and no thresholds. Only
        # when the bytes differ (the same paper re-downloaded from another source, so a
        # different PDF of the same work) does the content-based check below get a say.
        already = _already_stored_slug(workspace_root, source, acquired, max_bytes=max_bytes)
        if already is not None:
            stored_slug, present = already
            if present:
                raise AlreadyAcquired(stored_slug, "byte-for-byte identical to the stored artifact")
            # The manifest records this paper as held but the evidence store no longer
            # has it. Re-admit under the same request to repair the gap rather than
            # falling through: the request is FULFILLED, so it is not pending, and
            # inference would report "nothing is pending" about a file whose identity is
            # in fact established beyond doubt -- these are the exact bytes that request
            # recorded. Identical bytes are the strongest evidence available, so this is
            # a match, not a fallback.
            slug, matched_on_evidence = stored_slug, True
        else:
            pending = [r for r in manifest.requests if r.status in _PENDING_STATUSES]
            slug, matched_on_evidence = _infer_slug(source, pending, max_bytes=max_bytes, acquired=acquired)

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
    elif not matched_on_evidence:
        # The operator never said this file was for this paper -- it was the only one
        # outstanding. Say that in the note, so the record does not read as "the operator
        # offered this paper and it was wrong" when it is really "nothing else was left to
        # check it against". Without this the manifest's account of a batch ingest depends
        # on the alphabetical order of the operator's filenames.
        request.identity_note = (
            f"this file matched no outstanding request and was checked against "
            f"{request.slug} only because it was the sole one left: {request.identity_note}"
        )
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
