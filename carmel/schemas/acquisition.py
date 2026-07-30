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
    transit (a live run saw an arXiv read timeout and a Semantic Scholar HTTP 404 in the
    same campaign). ``detail`` carries the resolver's per-provider note saying which.

    Exists because :attr:`NO_OPEN_ACCESS_COPY` was being used for this case too, which
    overstated what had actually been established -- the same asserted-vs-observed
    defect already fixed once for :attr:`PAYWALLED`, one level down. A provider merely
    declining for a missing optional API key does NOT count as incomplete; that is a
    stable configuration fact reported in ``detail``, not an unknown."""
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


class AcquisitionManifest(BaseModel):
    """The full acquisition queue for one campaign workspace."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    requests: list[AcquisitionRequest] = Field(default_factory=list)
