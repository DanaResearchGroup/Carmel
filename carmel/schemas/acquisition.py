"""Schemas for the manual paper-acquisition queue.

A live probe of 60 combustion-kinetics works found only 2 (3.3%) whose full text
Carmel could fetch and read on its own. Manual acquisition is therefore the ORDINARY
path for this field, not an error branch: most papers a campaign needs must be
obtained by a human through an institutional subscription and handed to Carmel.

The queue is a durable, workspace-local record of "these are the papers I could not
get; please drop them here", plus the outcome of matching each dropped file back to
the paper it was supposed to be.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Schemes an operator can safely be shown as a clickable link. Rejecting everything
#: else (in particular ``javascript:`` and ``data:``) at the schema boundary means no
#: downstream renderer -- the operator dashboard's ``<a href=...>``, or anything else
#: that ever trusts this field -- has to remember to re-check it: an LLM-authored or
#: otherwise attacker-influenced ``landing_url`` can never carry a scheme that would
#: execute script or smuggle a payload if a human simply clicks it.
ALLOWED_LANDING_URL_SCHEMES = frozenset({"http", "https"})


class AcquisitionReason(StrEnum):
    """Why automated retrieval did not yield readable text."""

    PAYWALLED = "paywalled"
    """A host actually REFUSED an attempted fetch (HTTP 401/402/403). This is an
    observation about a real request, never a model's assertion; when no full-text URL
    was known to try at all, the reason is :attr:`NO_OPEN_ACCESS_COPY` instead."""
    NO_OPEN_ACCESS_COPY = "no_open_access_copy"
    """Automated open-access resolution RAN TO COMPLETION and produced no fetchable
    candidate URL -- either the OA indexes advertise none for this DOI, or resolution
    could not run at all (no DOI, no resolver configured, consent withheld);
    ``detail`` says which. Unlike :attr:`PAYWALLED` this asserts nothing about how
    any host responded, because no host was ever asked.

    Requires that every enabled provider actually answered. If resolution was cut
    short, the reason is :attr:`OA_LOOKUP_INCOMPLETE` instead."""
    OA_LOOKUP_INCOMPLETE = "oa_lookup_incomplete"
    """Open-access resolution was CUT SHORT, so nothing is established about whether a
    copy exists: the per-paper lookup cap was reached, or a provider's lookup failed in
    transit -- a real request could not be completed at all (a live run saw an arXiv
    read timeout in this campaign). ``detail`` carries the resolver's per-provider note
    saying which.

    Exists because :attr:`NO_OPEN_ACCESS_COPY` was being used for this case too, which
    overstated what had actually been established -- the same asserted-vs-observed
    defect already fixed once for :attr:`PAYWALLED`, one level down. A provider merely
    declining for a missing optional API key does NOT count as incomplete; that is a
    stable configuration fact reported in ``detail``, not an unknown.

    A provider that was reached and answered normally with "no record for this
    identifier" (HTTP 404, e.g. a live run saw Semantic Scholar 404 on
    ``/paper/DOI:10.1115/1.4007737``, observed 2026-07-30 and 2026-07-31) does NOT
    count as incomplete either: that provider's own lookup completed, it simply
    contributed nothing, so it counts toward :attr:`NO_OPEN_ACCESS_COPY` instead --
    the mirror image of the same overstatement this reason was introduced to fix. See
    :class:`carmel.agents.tools.search.SearchNotFound`."""
    HOST_NOT_ADMISSIBLE = "host_not_admissible"
    """The URL's host is not on the list of sources whose documents may enter the
    evidence store automatically.

    Not a failure of the source -- a refusal to auto-admit it. The identity gate
    confirms a document IS the cited work by finding the title and DOI outside its
    reference list, which a document that merely PRINTS another paper's title and DOI
    also satisfies. That gate cannot separate "is" from "mentions" without a
    threshold calibrated on far more documents than exist to calibrate against, so the
    cheaper control is upstream: keep documents of unknown provenance out of the store.

    The paper is queued for manual acquisition rather than dropped, so the human-gated
    path -- which runs its own identity check on admission -- still gets it."""
    NOT_A_DOCUMENT = "not_a_document"
    """A full-text URL was advertised but served something other than a document --
    in the probe, an HTML landing page five times out of eleven."""
    EMPTY_DOCUMENT = "empty_document"
    """A full-text URL was advertised, the content type was a document type, and the
    fetch itself succeeded -- but the response was zero bytes, or yielded no
    non-whitespace extractable text. Distinct from :attr:`NO_OPEN_ACCESS_COPY`: a copy
    WAS found and fetched, it was simply unusable. A live campaign stored a zero-byte
    ``text/plain`` response from a figshare landing page as a successful acquisition
    and never queued the paper for a human; this reason exists so that never happens
    silently again."""
    UNREADABLE = "unreadable"
    """The bytes were fetched but yielded no usable text (scanned/image-only pages, or
    a font encoding that loses spaces). See
    :func:`carmel.services.grounding.unreadable_reason`."""
    FETCH_FAILED = "fetch_failed"
    """Network-level failure, redirect refusal, or SSRF-guard rejection."""


class AcquisitionStatus(StrEnum):
    """Where a request sits in its lifecycle."""

    REQUESTED = "requested"
    FULFILLED = "fulfilled"
    REJECTED = "rejected"
    """A file was dropped for this request but did not pass the identity check, so it
    was NOT admitted to the evidence store."""


class SupplementaryFile(BaseModel):
    """One supplementary-information file received for a request: held, not ingested.

    A file dropped as ``<slug>.si.<ext>`` (or ``<slug>.si.<n>.<ext>``) is bound to its
    parent request and staged verbatim, but it is deliberately NOT admitted to the
    evidence store: no text is extracted, no identity check runs, and nothing here may
    be cited as evidence. Carmel cannot process these formats yet -- this record exists
    so a received file is visible (to the operator and to a future ingestion pass)
    instead of being silently ignored, which is the defect it replaces.
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str
    """Digest of the received bytes; also names the staging directory the raw file is
    held in (``literature_requests/supplementary/<sha256>/<original_filename>``)."""
    original_filename: str
    """The name the operator dropped the file under, kept verbatim -- it carries the
    role marker (``.si.``) and the extension, which is all the format information a
    future ingestion pass will have."""
    parent_slug: str
    """Slug of the request this file supplements."""
    content_type: str
    """MIME type sniffed from the bytes (never from the filename)."""
    received_at: datetime


class AcquisitionRequest(BaseModel):
    """One paper Carmel needs a human to obtain."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]{0,180}$")
    """Filesystem-safe identifier derived from the DOI (or title). This is the name the
    human gives the dropped file, and the only link between a file on disk and the
    paper it is meant to be.

    The pattern matches :data:`carmel.services.acquisition._VALID_SLUG_RE` -- the same
    charset :func:`carmel.services.acquisition.slug_for` is guaranteed to emit -- and
    exists so a slug read back from a hand-edited or otherwise tampered manifest can
    never carry a path separator or a ``..`` traversal segment into
    :func:`carmel.services.acquisition.drop_path_for`, which builds a filesystem path
    directly from this field."""
    title: str = Field(min_length=1)
    doi: str | None = None
    landing_url: str = Field(min_length=1)
    """Where a human should go to obtain the paper."""

    reason: AcquisitionReason
    detail: str = ""
    """Human-readable specifics (e.g. ``"HTTP 403"``), for the operator's benefit."""
    requested_at: datetime
    status: AcquisitionStatus = AcquisitionStatus.REQUESTED
    fulfilled_sha256: str | None = None
    """Digest of the admitted artifact, once a dropped file passed the identity check."""
    identity_note: str = ""
    """Why the identity check passed or failed. Kept for rejected requests too, so a
    puzzled operator can see what the check was looking for."""
    supplementary: list[SupplementaryFile] = Field(default_factory=list)
    """Supplementary-information files received for this paper. Received-and-held
    only: see :class:`SupplementaryFile` -- nothing in this list is usable as
    evidence."""

    @field_validator("landing_url")
    @classmethod
    def _reject_unsafe_url_scheme(cls, value: str) -> str:
        """Reject any scheme but http/https (P1-11: this field reaches an operator
        dashboard ``href`` unsanitized; ``javascript:``/``data:``/``file:`` etc. must
        never survive construction of this model)."""
        scheme = urlsplit(value).scheme.lower()
        if scheme not in ALLOWED_LANDING_URL_SCHEMES:
            raise ValueError(f"landing_url scheme {scheme!r} is not allowed; must be http or https")
        return value


#: The current on-disk shape of :class:`AcquisitionManifest`. Bumped whenever a field is
#: added or changed in a way that is not simply optional-with-a-default; every bump must
#: be paired with a migration step in
#: :func:`carmel.services.acquisition.migrate_manifest_payload`, mirroring
#: :data:`carmel.schemas.literature.CURRENT_REPORT_SCHEMA_VERSION`.
CURRENT_ACQUISITION_MANIFEST_VERSION = 2


class AcquisitionManifest(BaseModel):
    """The full acquisition queue for one campaign workspace."""

    model_config = ConfigDict(extra="forbid")

    version: int = CURRENT_ACQUISITION_MANIFEST_VERSION
    requests: list[AcquisitionRequest] = Field(default_factory=list)
