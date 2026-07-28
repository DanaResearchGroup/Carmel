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

from pydantic import BaseModel, ConfigDict, Field


class AcquisitionReason(StrEnum):
    """Why automated retrieval did not yield readable text."""

    PAYWALLED = "paywalled"
    """The host refused the request (typically HTTP 401/403), or no full-text URL
    was advertised at all."""
    NOT_A_DOCUMENT = "not_a_document"
    """A full-text URL was advertised but served something other than a document --
    in the probe, an HTML landing page five times out of eleven."""
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

    slug: str = Field(min_length=1)
    """Filesystem-safe identifier derived from the DOI (or title). This is the name the
    human gives the dropped file, and the only link between a file on disk and the
    paper it is meant to be."""
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


class AcquisitionManifest(BaseModel):
    """The full acquisition queue for one campaign workspace."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    requests: list[AcquisitionRequest] = Field(default_factory=list)
