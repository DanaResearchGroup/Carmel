# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Persist a figure digitization so its PARTIALNESS is stated, not inferred from a count.

A curve recovered from a plot arrives as a :class:`~carmel.schemas.datasets.Series`, and the
only thing a ``Series`` says about its own completeness is ``len(points)``. A series that lost
a marker -- straddling an axis boundary, occluded where two curves cross, unplaceable against
the axes -- is byte-for-byte indistinguishable from one that lost nothing. That is the gap this
module closes, and it closes it by making the record STATE what happened rather than leaving a
reader to work it out from a number that looks the same either way.

**Two facts, not one.** The record answers two questions that are routinely confused and are
genuinely independent:

- **Coverage** -- *is anything missing?* Answered by :attr:`FigureDigitization.coverage` and by
  an itemised ledger, :attr:`FigureDigitization.omissions`, that names each marker which did
  not become a point and why.
- **Auditability** -- *could the instrument have told?* Answered by
  :attr:`FigureDigitization.census`, which either gives a marker total for the plot region or
  says, with a typed reason, that no total exists.

Collapsing them is the failure this module exists to prevent. "The ledger is empty" means
"nothing was dropped" only when a census exists to have counted; with no census it means
"nothing was WRITTEN DOWN", which is not the same claim and must never read as one. So the
three states a reader gets are :attr:`FigureCoverage.COMPLETE`, :attr:`FigureCoverage.PARTIAL`
and :attr:`FigureCoverage.UNCHECKABLE`, and ``UNCHECKABLE`` is reached from an absent census
*whether or not the ledger is empty* -- a run that named three omissions and then lost the
ability to enumerate knows three things are missing and still cannot say that only three are.

**Illegal combinations are refused at construction, never merely documented.** See
:meth:`FigureDigitization.__post_init__`: D7-D9 make a ``COMPLETE`` record carrying an omission
unconstructible, and the census balance (``detected == recovered + len(omissions)``) makes a
``COMPLETE`` record that silently shed counted markers unconstructible too. A record cannot be
brought into existence claiming more coverage than its own ledger and census support.

SCOPE OF WHAT THIS PROVES -- read before trusting a record, and read it as strictly as the
equivalent paragraph in :class:`~carmel.schemas.datasets.EmbeddedTableInventory`. There is no
verifier here and no ``code_sha256``, because there is nothing to re-derive: a cell inventory
is DERIVED from a PDF and can therefore be put to the document again, whereas a digitization
census is a STATED observation about an image and no function in this repository produces one.
So what a valid record establishes is INTERNAL HONESTY -- it addresses itself, it round-trips,
and its coverage claim cannot contradict its own ledger or its own census. What it does NOT
establish is that any detector ran, that ``detected`` is the true marker count, or that the
ledger names every marker that was dropped. A hand-written record asserting a plausible census
over a figure that has none satisfies every check in this module. Closing that gap requires a
digitizer to re-derive against the crop's bytes, which does not exist yet; until it does, a
valid record is a well-formed CLAIM about partialness, not evidence that the claim is true.

WHAT COUNTS AS A CANDIDATE, AND WHERE A NON-CANDIDATE GOES. The ledger is a LOSS ledger: every
entry is a marker that BELONGED in this series and is not in it. Candidacy is a containment
question, and this record settles it on one rule -- **a marker's centre inside the plot region,
boundary included, makes it a candidate; a centre outside makes it not one.** The centre is the
only point of a marker this record stores, and it decides both directions:

- Centre inside, extent crossing the boundary: a real candidate whose coordinate cannot be read.
  That is :attr:`MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE`, it is a LOSS, and it is ledgered.
- Centre outside: a mark that is mostly not in the plot -- a legend key, an inset, an axis label,
  an annotation abutting the frame. It was never a candidate for this series.

**A non-candidate mark is not counted in ``detected`` and is not ledgered.** Both halves matter.
Not counted, because :class:`MarkerCensus` counts candidates *inside the region*, so including it
would unbalance D9 against a series that lost nothing. Not ledgered, because coverage keys off
ledger-emptiness, so an entry for it would report :attr:`FigureCoverage.PARTIAL` for a
digitization that recovered every marker it ever had -- a record UNDERSTATING its completeness,
which is as untrue as one overstating it. There is deliberately no reason code for it: an earlier
draft carried ``OUTSIDE_PLOT_REGION`` and it was removed for exactly this, so the absence is the
design and not an oversight. D6b enforces the rule rather than trusting a producer to have read
it -- an omission whose centre falls outside the region is REFUSED, which is what stops the
nearest available workaround (label it ``OCCLUDED``, record it at out-of-frame coordinates) from
being silently accepted.

What a producer that meets such a mark should do is therefore: nothing. Do not count it, do not
ledger it, and if a record of "we saw this and ruled it out" is wanted, that is a DETECTION-log
fact about the crop -- a different subject, needing a different structure, which this record does
not carry and this module does not define.

TWO ADDRESSES, TWO QUESTIONS. This module computes two content addresses over one digitization,
and confusing them is the sharpest available mistake:

- :func:`compute_digitization_sha` -- THE CLAIM ADDRESS -- names the COVERAGE CLAIM and nothing
  else. Its payload holds a series id, a crop identity, a plot region, a coverage claim, a census
  total, a recovered COUNT and an omission ledger -- and not one recovered coordinate. So two
  genuinely different digitizations of the same figure that agree on all of that (different curve
  fitting, different axis calibration, a different operator, therefore different points) produce
  IDENTICAL bytes and collide on one address. That collision is deliberate and UNCHANGED by this
  module's identity address: :func:`compute_digitization_sha` repeats it at the point of use, and
  ``TestTheAddressNamesTheClaimNotTheDigitization`` pins it as a tested fact. Use the claim
  address to deduplicate and to CITE claims -- two records at one address mean two producers said
  the same thing about coverage -- never as evidence the two recovered the same data.

- :func:`compute_digitization_identity` -- THE IDENTITY ADDRESS -- names the DIGITIZATION. It
  hashes the claim address together with the recovered points, the pixel-to-data calibration and
  the producer, so two attestations of one figure that agree on coverage but differ in a single
  recovered coordinate, in the calibration used, or in who produced them get DIFFERENT identity
  addresses -- while two byte-identical attestations get the SAME one, because nothing incidental
  (no timestamp, no object identity) is folded in. This is the address to ask "is this the same
  digitization?" of, and the one that detects a re-attestation as a change;
  ``TestTheIdentityAddressNamesTheDigitization`` pins that it separates exactly what the claim
  address collides. Figure values in this project are operator ATTESTATIONS rather than re-derived
  measurements, so the producer is part of a digitization's identity: an attestation is who made
  it as much as what it says.

Why a second address and not a widened one: the claim address is already the citation and dedup
key wired through :class:`~carmel.schemas.datasets.Series`,
:class:`~carmel.schemas.datasets.EmbeddedFigureDigitization` and the envelope validators, and its
collision-on-points is a tested contract those depend on. Folding points into it would change what
every stored citation addresses and silently invalidate every value already filed at one of those
addresses. So the identity is additive: the claim address keeps its exact meaning and
:data:`DIGITIZATION_PAYLOAD_VERSION` stays 1; the identity sits beside it under its own
:data:`DIGITIZATION_IDENTITY_VERSION`. The identity is NOT yet a citation surface -- nothing
embeds or resolves it -- because wiring figure verification into replay is a separate, later
ticket; today it is the address a caller computes to compare two digitizations.

**Canonicalization is the table lane's, not a second one.**
:func:`~carmel.services.dataset_store.canonical_json_bytes` produces the bytes and the address
is their sha256, exactly as :func:`~carmel.services.pdf_table_record.compute_inventory_sha`
defines it -- so a figure record and a table record are addressed by one rule. Geometry is
serialized with :meth:`float.hex` for the reasons :mod:`carmel.services.pdf_table_record`'s
docstring sets out (``canonical_json_bytes`` rejects floats outright, and ``canonical_decimal``
would bind every stored coordinate to the numeric-EXTRACTION grammar, which geometry has no
business depending on).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from carmel.services.dataset_store import canonical_json_bytes

__all__ = [
    "DIGITIZATION_IDENTITY_KEYS",
    "DIGITIZATION_IDENTITY_VERSION",
    "DIGITIZATION_PAYLOAD_KEYS",
    "DIGITIZATION_PAYLOAD_VERSION",
    "AxisCalibration",
    "AxisScale",
    "CensusUnavailable",
    "CensusUnavailableReason",
    "DigitizationIdentity",
    "DigitizedPoint",
    "FigureCoverage",
    "FigureDigitization",
    "MarkerCensus",
    "MarkerCount",
    "MarkerOmission",
    "MarkerOmissionReason",
    "PlotRegion",
    "UNREADABLE_PAYLOAD",
    "UnknownPayloadVersion",
    "census_of",
    "compute_digitization_identity",
    "compute_digitization_sha",
    "coverage_of",
    "digitization_identity_bytes",
    "digitization_identity_payload",
    "digitization_record_bytes",
    "digitization_record_payload",
    "is_auditable",
    "omission_reasons_of",
    "payload_unreadable_reason",
]

#: Bumped whenever the payload's SHAPE changes, independently of what fills it.
#:
#: Same rule and same reason as
#: :data:`~carmel.services.pdf_table_record.INVENTORY_PAYLOAD_VERSION`: a reader that does not
#: know a shape must not guess at it, so the version is what lets a reader say "I cannot read
#: this" rather than "this record is wrong". Those are different facts.
DIGITIZATION_PAYLOAD_VERSION = 1

#: Exactly the top-level keys a version-``DIGITIZATION_PAYLOAD_VERSION`` record has.
#:
#: EXACT rather than "at least these", for the reason
#: :data:`~carmel.services.pdf_table_record.INVENTORY_PAYLOAD_KEYS` states: the address is over
#: the canonical bytes, so a record carrying a stray key is a record whose address covers a
#: field nothing reads. Pinned against a real built payload by the tests, so the two cannot
#: drift.
DIGITIZATION_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "census",
        "coverage",
        "figure_crop_node_id",
        "figure_crop_sha256",
        "omissions",
        "payload_version",
        "plot_region",
        "raw_sha256",
        "recovered",
        "series_id",
    }
)

#: Bumped whenever the IDENTITY payload's SHAPE changes, independently of the CLAIM payload's
#: :data:`DIGITIZATION_PAYLOAD_VERSION`. The two are versioned apart on purpose: the claim address
#: and the identity address answer different questions (see the module docstring), so a shape
#: change to one must never force a reader of the other to say it cannot read a record it can.
DIGITIZATION_IDENTITY_VERSION = 1

#: Exactly the top-level keys a version-``DIGITIZATION_IDENTITY_VERSION`` identity payload has.
#:
#: EXACT rather than "at least these", for the same reason :data:`DIGITIZATION_PAYLOAD_KEYS` is:
#: the identity address is over the canonical bytes, so a stray key is a field the address covers
#: and nothing reads. Pinned against a real built payload by the tests.
DIGITIZATION_IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "calibration",
        "claim_sha256",
        "identity_version",
        "producer",
        "recovered_points",
    }
)

#: Mirrors :data:`carmel.schemas.datasets._IDENTIFIER_PATTERN`, deliberately duplicated rather
#: than imported: :mod:`carmel.schemas.datasets` imports FROM this package, so importing back
#: would be a cycle. The shape is pinned against the schema's own by the tests.
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Mirrors the same regex in :mod:`carmel.services.pdf_table_store`,
#: :mod:`carmel.services.evidence` and :mod:`carmel.services.dataset_store`, for the reason
#: :mod:`carmel.services.extraction_record` states: several stores share a digest SHAPE and
#: share no layout, and coupling their internals to save five characters is the worse trade.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: How reading a stored payload can fail. Named once because two callers depend on the SAME
#: answer -- :func:`payload_unreadable_reason`, and
#: :class:`carmel.schemas.datasets.EmbeddedFigureDigitization`, which uses it to refuse a record
#: before it is ever cited. A tuple written out twice is two tuples that agree until one is
#: edited. Public for that second caller: it lives in another package.
#:
#: ``KeyError`` and ``TypeError`` are in it because :meth:`FigureDigitization.from_payload`
#: indexes and calls into an untrusted mapping -- a missing nested key raises the first and
#: ``float.fromhex(None)`` the second -- and either escaping as itself would crash a caller that
#: correctly catches only ``ValueError``.
#:
#: ``OverflowError`` is in it because ``float.fromhex`` has a THIRD failure mode, and it is not a
#: ``ValueError``: ``float.fromhex("0x1p+99999")`` names a finite number too large to be one, and
#: raises ``OverflowError`` rather than refusing the string. A stored coordinate is exactly where
#: that input arrives from, so it escaped every caller here -- including, once the read path
#: started re-running the whole reconstruction, every accessor of
#: :class:`carmel.schemas.datasets.EmbeddedFigureDigitization`. Named specifically rather than as
#: its ``ArithmeticError`` base: a ``ZeroDivisionError`` raised in here would be a bug in this
#: module, and catching it as "unreadable payload" would report that bug as bad data. Keeping the
#: tuple narrow is safe only because a test sweeps this surface for escapes rather than trusting
#: the list to be complete -- see
#: ``tests/test_embedded_figure_digitization.py::TestTheRefusalSurfaceHasNoUndocumentedEscapes``,
#: which states what that sweep does and does not reach.
UNREADABLE_PAYLOAD: tuple[type[Exception], ...] = (KeyError, OverflowError, TypeError, ValueError)


class UnknownPayloadVersion(ValueError):
    """A payload declares a ``payload_version`` this reader does not know how to read.

    A :class:`ValueError` subclass ON PURPOSE, so it stays inside :data:`UNREADABLE_PAYLOAD`:
    every existing caller that refuses an unreadable payload -- :func:`payload_unreadable_reason`,
    and :class:`carmel.schemas.datasets.EmbeddedFigureDigitization`'s refusal of a record before it
    is ever embedded or cited -- must keep refusing an unknown-version record exactly as before,
    because refusing to guess at a shape it does not know is the whole point of the version. What
    the subclass ADDS is a fact ONE caller can act on separately: an unknown version is the reader
    admitting "I cannot read this shape", never a charge that the record is WRONG. Those are
    different facts (see :data:`DIGITIZATION_PAYLOAD_VERSION`), and
    :mod:`carmel.services.dataset_replay` reads the distinction to report a version it cannot read
    as UNVERIFIABLE rather than FAILED -- so a record that may merely be newer is never accused of
    a falsification no reader here earned.

    Carries the offending ``version`` and the ``readable_version`` as attributes so a caller can
    name the version in its own report without re-parsing the message.
    """

    def __init__(self, version: object, readable_version: int) -> None:
        self.version = version
        self.readable_version = readable_version
        super().__init__(
            f"payload_version {version!r} is not the readable version {readable_version!r} "
            "-- a reader that does not know a shape must not guess at it"
        )


#: Discriminator values for the two arms of :data:`MarkerCount` in the stored payload.
#:
#: A tagged object rather than two differently-shaped values under one key: a reader deciding
#: which arm it holds by probing for the presence of ``detected`` would answer "unavailable"
#: for a counted census whose key was dropped, which is the auditability axis silently
#: reporting the safe-looking direction on a malformed record.
_CENSUS_COUNTED = "counted"
_CENSUS_UNAVAILABLE = "unavailable"


def _require_coordinate(value: Any, *, where: str) -> float:
    """Require a finite ``float``, refusing everything a type hint alone would let through.

    A ``float`` annotation is documentation, not a guard: nothing stops ``PlotRegion(x_start=0)``
    or ``MarkerOmission(x=Decimal("1.5"))``. Both construct happily and then raise
    ``AttributeError`` inside :func:`_pt`, which has no ``.hex()`` to call -- a record that can
    be BUILT but not SERIALIZED, discovered at the moment it was about to be stored. ``int`` is
    refused rather than widened to ``float``: a silent widening is a second, undeclared idea of
    what a coordinate is, and the caller that passed an ``int`` believed something about this
    API that was not true.

    ``bool`` never reaches the ``float`` branch (``isinstance(True, float)`` is False), so it is
    refused by the type check along with everything else.

    Finiteness is checked at CONSTRUCTION by every type that holds a coordinate, not merely at
    serialization: by the time a record is serialized it has been constructed, compared and
    passed around, so a guard in :func:`_pt` alone would fire several steps after it mattered.
    ``float("nan").hex()`` is the perfectly valid string ``'nan'`` and ``float("inf").hex()`` is
    ``'inf'``, so without this a record built with either would round-trip through the store
    looking like a measurement.
    """
    if not isinstance(value, float):
        raise ValueError(
            f"{where}: a coordinate must be a float, got {type(value).__name__} {value!r} -- "
            "int is refused rather than widened, so this API cannot mean two things at once"
        )
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{where}: refusing to record a non-finite coordinate: {value!r}")
    return value


def _require_count(value: Any, *, where: str) -> int:
    """Require a non-negative ``int`` that is not a ``bool``.

    ``bool`` is excluded explicitly because ``isinstance(True, int)``, so without this
    ``MarkerCensus(detected=True)`` constructs, arithmetics as ``1`` through the balance check,
    and serializes as the JSON literal ``true`` -- which :meth:`FigureDigitization.from_payload`
    then refuses. A record its own reader cannot read back is one that reaches the store and is
    discovered broken by whoever needed it, which is the worst available moment.

    Shared by the dataclasses' ``__post_init__`` and by the payload readers, deliberately: two
    definitions of "a count" would agree until one was edited, and the entire point of the guard
    is that what can be BUILT is exactly what can be READ BACK.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where}: a count must be an int, got {type(value).__name__} {value!r}")
    if value < 0:
        raise ValueError(f"{where}: a count cannot be negative, got {value!r}")
    return value


def _pt(value: float) -> str:
    """Serialize one coordinate exactly.

    ``float.hex()``, the same encoding and the same non-finite guard as
    :mod:`carmel.services.pdf_table_record` uses, for the reasons its module docstring gives.
    Written out here rather than imported across module boundaries -- that module's copy is
    private -- and pinned equal to it by a test, so the two cannot drift into two different
    ideas of how a stored coordinate is spelled.

    The guard is kept even though every type that reaches this function has already run
    :func:`_require_coordinate` at construction: this is the last thing between a coordinate and
    the stored bytes, and a serializer that trusts its callers to have checked stops being safe
    the first time a new caller appears.
    """
    return _require_coordinate(value, where="_pt").hex()


class FigureCoverage(StrEnum):
    """Whether this digitization recovered everything the figure offered.

    Three members, not two, because "nothing is missing" and "there is no way to know" are
    different facts and a two-state field would have to report one of them as the other.
    """

    COMPLETE = "complete"
    """A census exists and the omission ledger is empty: every marker the instrument counted
    became a point. The strongest claim in this module, and still only a CLAIM -- see the
    module docstring on what a valid record does not establish."""

    PARTIAL = "partial"
    """A census exists and the ledger is not empty: named markers did not become points, and
    the ledger says which and why. This is the state that used to be indistinguishable from
    ``COMPLETE``, because both rendered as a bare point count."""

    UNCHECKABLE = "uncheckable"
    """No census exists, so no completeness claim can be made either way.

    Reached from an absent census WHETHER OR NOT the ledger is empty, and that is the point
    rather than an oversight. A run that recorded three omissions and then lost the ability to
    enumerate knows three markers are missing and cannot say only three are; reporting it as
    ``PARTIAL`` would assert a bound on what is missing that nothing established."""


class MarkerOmissionReason(StrEnum):
    """Why one marker that WAS a candidate for this series did not become a data point.

    THE LEDGER'S CONTRACT: every entry is a LOSS. Each member below names a marker that belonged
    in this series and is not in it, so a non-empty ledger and "something is missing" are the
    same fact, and :attr:`FigureCoverage.PARTIAL` follows from ledger-emptiness alone.

    That contract is load-bearing and was chosen over the alternative. An earlier draft carried
    an ``OUTSIDE_PLOT_REGION`` member for a mark ruled out as a legend key or an annotation --
    whose own description conceded that nothing was lost. Because coverage keys off
    ledger-emptiness, a digitization that recovered every one of its markers and merely noted a
    legend key would have reported PARTIAL: a record UNDERSTATING its completeness, which is
    exactly as untrue as one overstating it and worse in practice, since a field that cries wolf
    is a field readers learn to discount. The two ways out were to keep the member and make
    coverage something other than a function of the ledger, or to keep the ledger meaning one
    thing. The second was taken: a reader who must first partition reasons into lossy and
    non-lossy before knowing whether anything is missing is back to inferring partialness, which
    is the habit this whole record exists to end.

    A mark that was never a candidate -- a legend key, an inset, an annotation -- has no member
    here and needs none: it is not counted in :class:`MarkerCensus` and not ledgered, and D6b
    REFUSES an omission whose centre lies outside the plot region rather than leaving a producer
    to find that out. See "WHAT COUNTS AS A CANDIDATE, AND WHERE A NON-CANDIDATE GOES" in the
    module docstring for the containment rule and for what to do with such a mark instead.

    Typed rather than free text because a consumer has to be able to act on the distinction --
    an occlusion may be recoverable by re-cropping, an axis-boundary straddle is not -- and a
    free-text ledger is one no query can ever ask a question of. ``detail`` carries the
    specifics alongside.
    """

    AXIS_BOUNDARY_STRADDLE = "axis_boundary_straddle"
    """The marker's centre is inside the plot region but its extent crosses a boundary, so its
    coordinate cannot be read off the axis.

    The centre being INSIDE is what makes it a candidate and therefore a loss; a mark centred
    outside the frame whose extent merely reaches in was never a candidate and is refused by D6b
    (see the module docstring). Recording the centre inclusive of the boundary is deliberate: a
    marker centred exactly on the axis is the paradigm straddle.

    Such a marker stays EXCLUDED. Admitting it would mean choosing a coordinate the figure does
    not determine, and the whole reason this ledger exists is that a chosen coordinate and a
    read one are indistinguishable once they are both just points. The record's job is to say
    the marker was there and was dropped, never to rescue it."""

    OCCLUDED = "occluded"
    """Another mark overlaps this one enough that its extent, and therefore its centre, cannot
    be recovered. Typically a curve crossing."""

    SERIES_AMBIGUOUS = "series_ambiguous"
    """The marker is readable but cannot be attributed to THIS series rather than another on
    the same axes. A point assigned to the wrong curve is worse than a point dropped, which is
    why the ambiguity resolves to an omission."""

    COORDINATE_UNRESOLVED = "coordinate_unresolved"
    """The marker's position could not be converted to axis coordinates -- a broken or
    unreadable axis calibration, a log scale whose decades could not be pinned."""


class CensusUnavailableReason(StrEnum):
    """Why no marker total exists for this plot region.

    Each member is a different thing having gone wrong, and none of them is "the figure had no
    markers" -- that is a census of zero, which is available and is a different record. Keeping
    these apart is what stops "we could not count" from being stored as "we counted none".
    """

    DETECTOR_UNAVAILABLE = "detector_unavailable"
    """No marker detector ran at all: the crop was digitized by some other route, or the
    detector is not installed on the machine that produced this record."""

    PLOT_REGION_UNBOUNDED = "plot_region_unbounded"
    """The axis frame could not be established, so "every marker inside it" has no referent to
    count over. A total would be a count of an undefined set."""

    ENUMERATION_INCOMPLETE = "enumeration_incomplete"
    """The detector started and stopped before covering the region -- a budget, a timeout, an
    error partway. Anything it did find may be recorded as an omission; what it did not reach
    is why there is no total."""

    SERIES_INSEPARABLE = "series_inseparable"
    """Markers were counted for the region as a whole but could not be attributed per series,
    so no PER-SERIES total exists -- and this record is about one series. A region-wide total
    cannot stand in: it would make every single-curve digitization of a three-curve plot look
    like it dropped two thirds of its markers."""


@dataclass(frozen=True)
class PlotRegion:
    """The axis frame the digitization worked inside, in the crop's own coordinate space.

    Carried because without it the ledger's coordinates are uninterpretable numbers: an
    ``AXIS_BOUNDARY_STRADDLE`` at ``x=413.0`` says nothing unless a reader can see where the
    boundary was. This is the figure lane's counterpart to
    :class:`~carmel.services.pdf_tables.ClaimedFootprint`, and it carries the same limitation:
    it says which box was worked in, never that it was the RIGHT box.

    ``page`` is the page of the parent document the crop was taken from, recorded because a
    crop's own node identity does not carry one (see :meth:`FigureDigitization.__post_init__`).
    """

    page: int
    x_start: float
    x_end: float
    y_bottom: float
    y_top: float

    def __post_init__(self) -> None:
        # Types FIRST. The annotations above are documentation and stop nothing: `page=True`
        # satisfies `int` and serializes as the JSON literal `true`, and `x_start=0` satisfies
        # nobody but raises `AttributeError` deep inside `_pt`, which has no `.hex()` to call on
        # an int. Either way the region can be BUILT and not STORED, and the discovery happens
        # at the moment of writing rather than the moment of the mistake.
        _require_count(self.page, where="PlotRegion.page")
        for name, value in (
            ("x_start", self.x_start),
            ("x_end", self.x_end),
            ("y_bottom", self.y_bottom),
            ("y_top", self.y_top),
        ):
            _require_coordinate(value, where=f"PlotRegion.{name}")
        if self.page < 1:
            raise ValueError(f"PlotRegion.page must be a 1-based page number, got {self.page!r}")
        # Finiteness is already settled by `_require_coordinate` above, and it has to be settled
        # BEFORE the ordering checks below: every comparison against a `nan` is False, so
        # `not nan < x` is True and these checks would refuse a `nan` on their own -- as "spans
        # no height", which sends a reader looking for a box that was never the problem. A guard
        # that catches the right input for the wrong stated reason is a guard whose message
        # cannot be trusted the next time it fires.
        #
        # A degenerate or inverted box gives every omission an unfalsifiable frame: with
        # x_start >= x_end, no coordinate is inside it and "straddling its boundary" is true of
        # nothing and of everything at once. Refused here rather than left to a reader, because
        # a reader holding only the record cannot tell a degenerate frame from a real one
        # without knowing what the crop looked like.
        if not self.x_start < self.x_end:
            raise ValueError(
                f"PlotRegion spans no width: x_start={self.x_start!r} is not less than x_end={self.x_end!r}, "
                "so no marker can be inside it and no boundary can be straddled"
            )
        if not self.y_bottom < self.y_top:
            raise ValueError(
                f"PlotRegion spans no height: y_bottom={self.y_bottom!r} is not less than y_top={self.y_top!r}, "
                "so no marker can be inside it and no boundary can be straddled"
            )


@dataclass(frozen=True)
class MarkerOmission:
    """One marker the instrument saw and this series did not get.

    ``marker_id`` is the instrument's own handle for the marker, and it is what makes the
    ledger an itemised statement rather than a count: two records that each dropped "one
    marker" are still telling a reader different things when the markers differ.
    """

    marker_id: str
    reason: MarkerOmissionReason
    x: float
    y: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.marker_id, str):
            raise ValueError(f"MarkerOmission.marker_id must be a string, got {type(self.marker_id).__name__}")
        if not self.marker_id or self.marker_id != self.marker_id.strip():
            raise ValueError(
                f"MarkerOmission.marker_id must be non-empty and free of surrounding whitespace, got {self.marker_id!r}"
            )
        if not isinstance(self.reason, MarkerOmissionReason):
            raise ValueError(
                f"MarkerOmission({self.marker_id!r}).reason must be a MarkerOmissionReason, got "
                f"{type(self.reason).__name__} {self.reason!r}"
            )
        if not isinstance(self.detail, str):
            raise ValueError(
                f"MarkerOmission({self.marker_id!r}).detail must be a string, got {type(self.detail).__name__}"
            )
        # Guarded HERE rather than only at serialization, where ``_pt`` would catch it. By the
        # time a record is serialized it has already been constructed, compared and passed
        # around; neither a `nan` nor an `int` may inhabit an omission at all.
        _require_coordinate(self.x, where=f"MarkerOmission({self.marker_id!r}).x")
        _require_coordinate(self.y, where=f"MarkerOmission({self.marker_id!r}).y")


@dataclass(frozen=True)
class MarkerCensus:
    """A marker total for the plot region: the instrument COULD tell, and this is what it saw.

    ``detected`` counts candidate markers attributable to this series inside the region,
    including the ones that went on to be omitted -- which is what makes the balance check in
    :meth:`FigureDigitization.__post_init__` (D9) mean anything. A census that counted only the
    survivors would balance trivially and prove nothing.

    ``detected == 0`` is legal and is a real answer -- "this frame contains no markers of this
    series" -- and is emphatically not the same as :class:`CensusUnavailable`. It is, though,
    unreachable in a valid record today, because a record must have recovered at least one
    point to be about a series at all (D5); a zero census would have to balance against zero
    recovered points, which D5 refuses first.
    """

    detected: int

    def __post_init__(self) -> None:
        # `bool` is the one that matters: `isinstance(True, int)`, so without this
        # `MarkerCensus(detected=True)` balances as 1 against D9 and then serializes as the JSON
        # literal `true`, which `from_payload` refuses. A census that can be built and not read
        # back is one that reaches the store looking fine.
        _require_count(self.detected, where="MarkerCensus.detected")


@dataclass(frozen=True)
class CensusUnavailable:
    """No marker total exists for this plot region, and this is why.

    Deliberately a TYPE rather than a sentinel value on :class:`MarkerCensus` (``detected =
    -1``, ``detected = None``): the arithmetic in D9 is defined for a census and undefined for
    a non-census, and a sentinel is exactly the shape that lets undefined arithmetic run
    anyway. Mirrors :class:`carmel.schemas.datasets.Absent`, which makes the same argument for
    the same reason one layer up, and carries a typed reason for the same reason it does --
    an unexplained absence is one a reader has to guess at.
    """

    reason: CensusUnavailableReason
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.reason, CensusUnavailableReason):
            raise ValueError(
                "CensusUnavailable.reason must be a CensusUnavailableReason, got "
                f"{type(self.reason).__name__} {self.reason!r} -- an unexplained absence is one a "
                "reader has to guess at, and a reason it cannot interpret is unexplained"
            )
        if not isinstance(self.detail, str):
            raise ValueError(f"CensusUnavailable.detail must be a string, got {type(self.detail).__name__}")


#: A census, or the reason there is none. The auditability axis, in one type.
#:
#: Named so that a signature can say "auditability" in one word, and so that adding a third
#: arm later is a change to one alias rather than to every annotation.
MarkerCount = MarkerCensus | CensusUnavailable


def _require_sha256(value: str, *, field_name: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"FigureDigitization.{field_name} must be 64 lowercase hex characters, got {value!r}")
    return value


@dataclass(frozen=True)
class FigureDigitization:
    """One series recovered from one figure crop, and an honest account of what it cost.

    Every invariant below is enforced in :meth:`__post_init__`, so an instance of this class is
    a record whose coverage claim cannot contradict its own evidence. See the module docstring
    for what that does and does not amount to.
    """

    series_id: str
    """The :class:`~carmel.schemas.datasets.Series` this digitization produced. Same identifier
    grammar as ``Series.series_id`` so the two can be joined without a translation step."""

    raw_sha256: str
    """The document the figure crop came from -- the join to
    :attr:`carmel.schemas.datasets.SourceNode.sha256` of the crop's parent, and the directory a
    stored record is filed under, exactly as in the table lane."""

    figure_crop_node_id: str
    """The ``node_id`` of the ``FIGURE_CROP`` :class:`~carmel.schemas.datasets.SourceNode` this
    series was read off."""

    figure_crop_sha256: str
    """That node's ``sha256`` -- the crop's own bytes.

    Both halves of the node's existing identity are carried, and NOTHING is invented about the
    crop beyond them: ``node_id`` alone is only unique within one envelope's source graph,
    while ``sha256`` alone cannot say which node of an envelope was meant when two nodes hold
    identical bytes."""

    plot_region: PlotRegion
    coverage: FigureCoverage
    """The coverage claim, STATED rather than derived.

    It is redundant against ``census`` and ``omissions``, and D7-D8 pin it equal to them --
    the same shape, and the same argument, as
    :attr:`carmel.schemas.datasets.EmbeddedTableInventory.raw_sha256`, which is likewise
    declared redundantly and likewise pinned. Stating it is the entire point of this ticket:
    partialness has to be something the stored evidence SAYS. A derived property is one every
    reader re-implements -- including readers in other languages holding nothing but the JSON --
    and a reader that gets the inference subtly wrong gets it wrong silently."""

    census: MarkerCount
    """The auditability axis: a total for the region, or the typed reason there is none."""

    recovered: int
    """How many markers became points in the resulting series."""

    omissions: tuple[MarkerOmission, ...]
    """The coverage ledger: every marker that did not become a point, and why.

    Sorted by ``marker_id`` with duplicates refused (D6), for the reason
    :meth:`carmel.schemas.datasets.Series._validate_points_sorted_and_nonempty` gives for
    ``points``: the record is content-addressed, so an unordered collection would give one
    ledger as many addresses as it has permutations.

    Every entry's centre must lie inside :attr:`plot_region`, boundary included (D6b): the ledger
    holds LOSSES, a loss is a marker that was a candidate, and candidacy is containment. A mark
    centred outside the frame is not counted and not ledgered -- see the module docstring."""

    def __post_init__(self) -> None:
        """Refuse every record whose coverage claim outruns its own evidence.

        Each check carries its own marker phrase so a test can tell which one fired.
        """
        # D0: types. The annotations on the fields above stop nothing at all -- `recovered=True`
        # satisfies `int`, arithmetics as 1 through D9, and serializes as the JSON literal
        # `true`, which this class's OWN `from_payload` then refuses. Every such value produces a
        # record that can be built and not read back, so it survives to the store and is
        # discovered broken by whoever needed it. Checked before D1 because a wrong-typed field
        # makes the checks below either crash or, worse, quietly pass.
        _require_count(self.recovered, where=f"FigureDigitization(series_id={self.series_id!r}).recovered")
        if not isinstance(self.plot_region, PlotRegion):
            raise ValueError(
                f"FigureDigitization.plot_region must be a PlotRegion, got {type(self.plot_region).__name__}"
            )
        if not isinstance(self.coverage, FigureCoverage):
            raise ValueError(
                f"FigureDigitization.coverage must be a FigureCoverage, got "
                f"{type(self.coverage).__name__} {self.coverage!r}"
            )
        # The census's type is what D7-D9 branch on, so an object that is neither arm would slip
        # through the unauditable branch's early return -- which is how any object at all could
        # inhabit an UNCHECKABLE record and only blow up, or silently misreport, at
        # serialization. Constrained here so `census` is one of exactly two things everywhere
        # below, and `isinstance` narrowing means what it says.
        if not isinstance(self.census, MarkerCensus | CensusUnavailable):
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): census must be a MarkerCensus or a "
                f"CensusUnavailable, got {type(self.census).__name__} {self.census!r} -- the auditability "
                "axis has exactly two states and an object that is neither answers no question"
            )
        if not isinstance(self.omissions, tuple):
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): omissions must be a tuple, got "
                f"{type(self.omissions).__name__} -- a list is mutable and this record is content-addressed"
            )
        for position, entry in enumerate(self.omissions):
            if not isinstance(entry, MarkerOmission):
                raise ValueError(
                    f"FigureDigitization(series_id={self.series_id!r}): omissions[{position}] must be a "
                    f"MarkerOmission, got {type(entry).__name__} {entry!r}"
                )

        # D1-D3: identity. Checked before anything else that reads them, because a record that
        # cannot be joined to a series, a document and a crop is not a record of anything.
        for name, value in (
            ("series_id", self.series_id),
            ("raw_sha256", self.raw_sha256),
            ("figure_crop_node_id", self.figure_crop_node_id),
            ("figure_crop_sha256", self.figure_crop_sha256),
        ):
            if not isinstance(value, str):
                raise ValueError(f"FigureDigitization.{name} must be a string, got {type(value).__name__}")
        if not _IDENTIFIER_RE.fullmatch(self.series_id):
            raise ValueError(
                f"FigureDigitization.series_id {self.series_id!r} is not a lowercase identifier "
                "(^[a-z][a-z0-9_]*$), so it cannot name a Series"
            )
        _require_sha256(self.raw_sha256, field_name="raw_sha256")
        _require_sha256(self.figure_crop_sha256, field_name="figure_crop_sha256")
        if not self.figure_crop_node_id or self.figure_crop_node_id != self.figure_crop_node_id.strip():
            raise ValueError(
                "FigureDigitization.figure_crop_node_id must be non-empty and free of surrounding "
                f"whitespace, got {self.figure_crop_node_id!r}"
            )

        # D4 is not here. Geometry is guarded by the types that HOLD it -- `PlotRegion` and
        # `MarkerOmission` each run `_require_coordinate` in their own `__post_init__` -- and D0
        # above has just established that this record's are those types, so neither a non-finite
        # nor a non-float coordinate can reach this point in a constructed instance. A second
        # sweep here would be a check that passes because an earlier one already fired, which is
        # the kind of guard that quietly stops testing anything.

        # D5: this record is ABOUT a series, and a Series requires at least one point (S7). A
        # digitization that recovered nothing produced no series to be the subject of, so the
        # honest record of it is a figure-level refusal about the CROP -- a different subject,
        # which this type does not model. Refusing here also forecloses the emptiest possible
        # completeness claim: detected=0, recovered=0, omissions=(), vacuously COMPLETE.
        if self.recovered < 1:
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): recovered={self.recovered!r}, but a "
                "record names the Series it produced and a Series has at least one point -- a digitization "
                "that recovered nothing is a refusal about the crop, not a record about a series"
            )

        # D6: an itemised ledger has to be itemised. A repeated marker_id makes "which omission
        # does this id refer to" ill-defined and lets one dropped marker be counted twice
        # against the census balance in D9; unsorted entries give one ledger several addresses.
        seen: set[str] = set()
        for omission in self.omissions:
            if omission.marker_id in seen:
                raise ValueError(
                    f"FigureDigitization(series_id={self.series_id!r}): duplicate marker_id "
                    f"{omission.marker_id!r} in omissions -- one dropped marker, counted twice, would "
                    "balance a census that does not balance"
                )
            seen.add(omission.marker_id)
        if list(self.omissions) != sorted(self.omissions, key=lambda entry: entry.marker_id):
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): omissions must be sorted ascending by "
                "marker_id, or one ledger has as many content addresses as it has permutations"
            )

        # D6b: every ledger entry's centre lies inside the plot region, boundary included.
        #
        # Candidacy is a containment question -- MarkerCensus counts candidates INSIDE THE REGION
        # -- and the centre is the only point of a marker this record stores, so the centre is
        # what decides it. Enforced rather than documented because removing OUTSIDE_PLOT_REGION
        # took away a producer's reason code without taking away the mark it met: faced with a
        # legend key, the nearest available action is to call it OCCLUDED and record it at
        # out-of-frame coordinates, which is silently accepted without this and is the exact
        # misclassification the removal was meant to prevent. A non-candidate is not counted and
        # not ledgered; see the module docstring for what to do with one instead.
        region = self.plot_region
        for omission in self.omissions:
            inside_x = region.x_start <= omission.x <= region.x_end
            inside_y = region.y_bottom <= omission.y <= region.y_top
            if not (inside_x and inside_y):
                raise ValueError(
                    f"FigureDigitization(series_id={self.series_id!r}): omission {omission.marker_id!r} is "
                    f"centred at ({omission.x!r}, {omission.y!r}), outside the plot region "
                    f"x=[{region.x_start!r}, {region.x_end!r}] y=[{region.y_bottom!r}, {region.y_top!r}] -- "
                    "a mark centred outside the frame was never a candidate for this series, so it is "
                    "neither counted in the census nor ledgered as a loss; ledgering it would report "
                    "PARTIAL for a digitization that lost nothing"
                )

        # D7: the auditability axis and the coverage claim, pinned in BOTH directions.
        #
        # This is the join that stops the two collapsing. Forward: no census, no completeness
        # claim of any kind -- neither COMPLETE nor PARTIAL, because PARTIAL asserts a BOUND on
        # what is missing ("these and no others") that an absent census never established.
        # Backward: a census present with an UNCHECKABLE claim is a record declining to read
        # evidence it holds, which would let any producer opt out of the coverage question
        # simply by not answering it.
        # Narrowed with `isinstance` in the branch rather than through a `bool` computed above,
        # so the census's arm is a fact the type checker holds too and D9 below cannot reach for
        # `detected` on an arm that has none.
        census = self.census
        if not isinstance(census, MarkerCensus):
            if self.coverage is not FigureCoverage.UNCHECKABLE:
                raise ValueError(
                    f"FigureDigitization(series_id={self.series_id!r}): coverage={self.coverage.value!r} "
                    "claims the completeness of this series, but census is unavailable "
                    f"({census.reason.value!r} -- see CensusUnavailableReason), so nothing ever counted "
                    "the markers this claim would have to be measured against; the only honest coverage "
                    "here is UNCHECKABLE"
                )
            # Nothing below applies: with no total there is no balance to strike, and the
            # ledger may be empty or not. THAT is the orthogonality -- a run that named three
            # omissions and then lost the ability to enumerate is UNCHECKABLE with a non-empty
            # ledger, and it is telling a reader strictly more than an empty one would.
            return
        if self.coverage is FigureCoverage.UNCHECKABLE:
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): coverage=UNCHECKABLE, but this record "
                "carries a census -- the completeness question IS answerable here, and a record may not "
                "decline to answer a question its own evidence settles"
            )

        # D8: with a census in hand, the ledger decides the claim, both ways.
        #
        # The first branch is the one this ticket exists for: a record claiming a complete
        # series while carrying an omission is refused HERE, at construction, so no such
        # record can be stored, embedded, cited or read.
        if self.coverage is FigureCoverage.COMPLETE and self.omissions:
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): coverage=COMPLETE, but the record "
                f"carries {len(self.omissions)} omission(s) "
                f"({sorted(entry.marker_id for entry in self.omissions)!r}) -- a series that lost a marker "
                "is PARTIAL, and a complete claim over a non-empty ledger is the exact confusion this "
                "record exists to make unconstructible"
            )
        if self.coverage is FigureCoverage.PARTIAL and not self.omissions:
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): coverage=PARTIAL, but the omission "
                "ledger is empty -- partialness a reader cannot itemise is a warning, not a record, and "
                "the census here can settle the question either way"
            )

        # D9: the census must balance. Every marker counted either became a point or is named
        # in the ledger; there is no third destination.
        #
        # This is what makes D8 more than a consistency check between two fields the same
        # producer wrote. Without it, a producer that detected 12 markers, emitted 9 points and
        # wrote no ledger entries would pass D8 and claim COMPLETE -- the exact failure this
        # ticket describes, three markers lost with the record still reading as whole.
        expected = self.recovered + len(self.omissions)
        if census.detected != expected:
            raise ValueError(
                f"FigureDigitization(series_id={self.series_id!r}): the census does not balance -- "
                f"detected={census.detected}, but recovered={self.recovered} plus "
                f"{len(self.omissions)} omission(s) accounts for {expected}. Every counted marker became "
                "a point or is named in the ledger; a difference is a marker that vanished unrecorded"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FigureDigitization:
        """Rebuild a record from its stored payload, re-running every invariant above.

        This is the reconstruction :class:`carmel.schemas.datasets.EmbeddedTableInventory`
        documents itself as NOT having, and its absence there is why that class's T1 is
        materially weaker than :class:`carmel.schemas.datasets.EmbeddedConversionTable`'s. A
        cell inventory cannot be rebuilt from a payload because the only thing that establishes
        a grid is deriving it from the PDF again. A digitization record is not derived from
        anything -- it is a stated ledger -- so every rule that governs it is a rule this
        function can re-apply to bytes that arrived from disk. A payload that reaches this
        function from an untrusted file is held to exactly the invariants an in-process
        construction is held to, D1 through D9.

        Raises:
            UnknownPayloadVersion: Only if ``payload_version`` is GENUINELY DECLARED -- present,
                an ``int`` (not a ``bool``), and different from :data:`DIGITIZATION_PAYLOAD_VERSION`.
                That record may be NEWER rather than wrong, so this is the reader's admission "I
                cannot read this shape". A ``ValueError`` subclass, so a caller that treats every
                unreadable payload alike is unaffected -- but a caller that wants to tell "I cannot
                read this shape" apart from "this record is wrong" can, because the two are
                different facts and only this one is the reader's admission. A version key that is
                absent, or present but of a type a version cannot be, is NOT this: it is a malformed
                record, refused below as an ordinary ``ValueError``.
            ValueError: If the record the payload describes violates any construction invariant, if
                the payload is not a version-:data:`DIGITIZATION_PAYLOAD_VERSION` object at all, or
                if it declares a ``payload_version`` of the wrong type. These mean the same thing to
                a caller: these bytes are not a record.
            TypeError: If a field holds a type the reconstruction cannot use at all.
            OverflowError: If a stored coordinate is a hex float naming a magnitude no ``float``
                can hold -- ``float.fromhex`` raises this instead of refusing the string.
            KeyError: If a nested object is missing a key this reconstruction indexes.

        Callers that treat any unreadable payload alike should use
        :func:`payload_unreadable_reason`, which catches all four; the tuple it catches is
        :data:`UNREADABLE_PAYLOAD`, and a test sweeps this function for types missing from it.
        """
        if not isinstance(payload, Mapping):
            # The annotation says Mapping and an untrusted payload does not read annotations: a
            # decoded JSON list reaching here raised AttributeError off `payload.get`, which is
            # outside UNREADABLE_PAYLOAD and so crashed the caller that asked this function
            # whether the payload was readable. Refused the same way a non-object `plot_region`
            # is, and for the same reason.
            raise ValueError(f"a digitization record payload is {type(payload).__name__}, not an object")
        version = payload.get("payload_version")
        if isinstance(version, int) and not isinstance(version, bool):
            # A GENUINELY DECLARED version: present, and of the type a version is. Only a
            # different one of those is an unknown SHAPE -- the reader admitting "I cannot read
            # this, it may be newer than me", reported UNVERIFIABLE downstream, never a charge that
            # the record is wrong. `bool` is excluded because `isinstance(True, int)` and
            # `True == 1`: without excluding it a `payload_version` of `true` would sail through
            # this gate AS version 1, reading a mistyped record as a valid one. Same reasoning, and
            # the same spelling, as :func:`_require_count`.
            if version != DIGITIZATION_PAYLOAD_VERSION:
                raise UnknownPayloadVersion(version, DIGITIZATION_PAYLOAD_VERSION)
        elif "payload_version" in payload:
            # PRESENT but not the type a version is (a string "1", a float, a bool, an explicit
            # null). This is a MALFORMED record, not a newer one: a version bump writes a larger
            # integer, it never respells the one field that tells a reader how to read the rest. A
            # reader that cannot trust that field's type has bytes it cannot read as a record at
            # all, so it says the record is WRONG (FAILED, via `UNREADABLE_PAYLOAD`), never that it
            # might merely be newer. Refused HERE rather than left to the shape check below, which
            # would PASS an otherwise-legal record whose only defect is a mistyped version, letting
            # a broken record reconstruct clean.
            raise ValueError(
                f"'payload_version' is {type(version).__name__} {version!r}, not the integer a "
                "version is -- a mistyped discriminator is a malformed record, not a newer one"
            )
        # ABSENT entirely: not declared, so there is nothing to call an unknown VERSION. Left to
        # the shape check below, which refuses it as a reconstruction failure naming the missing
        # 'payload_version' key -- FAILED, and a reason that never claims the record declared a
        # version it did not (the overclaim the earlier `payload.get(...) != VERSION` short-circuit
        # made, reporting a keyless payload as "declares payload_version None").
        keys = set(payload)
        if keys != set(DIGITIZATION_PAYLOAD_KEYS):
            unexpected = sorted(keys - DIGITIZATION_PAYLOAD_KEYS)
            missing = sorted(DIGITIZATION_PAYLOAD_KEYS - keys)
            raise ValueError(
                f"not the shape of a version-{DIGITIZATION_PAYLOAD_VERSION} digitization record "
                f"(unexpected keys {unexpected!r}, missing keys {missing!r})"
            )
        region = payload["plot_region"]
        if not isinstance(region, Mapping):
            raise ValueError(f"'plot_region' is {type(region).__name__}, not an object")
        stored_omissions = payload["omissions"]
        if not isinstance(stored_omissions, list):
            raise ValueError(f"'omissions' is {type(stored_omissions).__name__}, not a list, so it itemises nothing")
        return cls(
            series_id=_require_str(payload["series_id"], field_name="series_id"),
            raw_sha256=_require_str(payload["raw_sha256"], field_name="raw_sha256"),
            figure_crop_node_id=_require_str(payload["figure_crop_node_id"], field_name="figure_crop_node_id"),
            figure_crop_sha256=_require_str(payload["figure_crop_sha256"], field_name="figure_crop_sha256"),
            plot_region=PlotRegion(
                page=_require_count(region["page"], where="plot_region.page"),
                x_start=float.fromhex(_require_str(region["x_start"], field_name="plot_region.x_start")),
                x_end=float.fromhex(_require_str(region["x_end"], field_name="plot_region.x_end")),
                y_bottom=float.fromhex(_require_str(region["y_bottom"], field_name="plot_region.y_bottom")),
                y_top=float.fromhex(_require_str(region["y_top"], field_name="plot_region.y_top")),
            ),
            coverage=FigureCoverage(payload["coverage"]),
            census=_census_from(payload["census"]),
            recovered=_require_count(payload["recovered"], where="recovered"),
            omissions=tuple(_omission_from(entry, position) for position, entry in enumerate(stored_omissions)),
        )


def _require_str(value: Any, *, field_name: str) -> str:
    """Require a JSON string, refusing the types ``json`` will happily hand back instead.

    ``str(value)`` would be the shorter spelling and is the wrong one: it turns ``None`` into
    ``"None"`` and ``123`` into ``"123"``, so a malformed payload would reconstruct into a
    perfectly well-formed record of something nobody wrote.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is {type(value).__name__}, not a string")
    return value


def _census_from(stored: Any) -> MarkerCount:
    """Rebuild the auditability axis from its tagged stored form."""
    if not isinstance(stored, Mapping):
        raise ValueError(f"'census' is {type(stored).__name__}, not an object, so it states no auditability")
    kind = stored.get("kind")
    if kind == _CENSUS_COUNTED:
        return MarkerCensus(detected=_require_count(stored["detected"], where="census.detected"))
    if kind == _CENSUS_UNAVAILABLE:
        return CensusUnavailable(
            reason=CensusUnavailableReason(stored["reason"]),
            detail=_require_str(stored["detail"], field_name="census.detail"),
        )
    raise ValueError(
        f"'census' has kind {kind!r}, which is neither {_CENSUS_COUNTED!r} nor {_CENSUS_UNAVAILABLE!r} -- "
        "an untagged census cannot be read as either a total or the absence of one, and guessing which "
        "would be the auditability axis answering from a coin toss"
    )


def _omission_from(stored: Any, position: int) -> MarkerOmission:
    if not isinstance(stored, Mapping):
        raise ValueError(f"omissions[{position}] is {type(stored).__name__}, not an object")
    return MarkerOmission(
        marker_id=_require_str(stored["marker_id"], field_name=f"omissions[{position}].marker_id"),
        reason=MarkerOmissionReason(stored["reason"]),
        x=float.fromhex(_require_str(stored["x"], field_name=f"omissions[{position}].x")),
        y=float.fromhex(_require_str(stored["y"], field_name=f"omissions[{position}].y")),
        detail=_require_str(stored["detail"], field_name=f"omissions[{position}].detail"),
    )


def _census_payload(census: MarkerCount) -> dict[str, Any]:
    if isinstance(census, MarkerCensus):
        return {"detected": census.detected, "kind": _CENSUS_COUNTED}
    return {"detail": census.detail, "kind": _CENSUS_UNAVAILABLE, "reason": census.reason.value}


def digitization_record_payload(record: FigureDigitization) -> dict[str, Any]:
    """The full stored form of one digitization, ready for :func:`canonical_json_bytes`.

    Every field here is identity; there is no diagnostics half. Same rule and same reason as
    :func:`carmel.services.pdf_table_record.inventory_record_payload`: a timestamp or a
    hostname would have to be excluded from the address and would then be a field nothing
    checks, which is a place for a wrong value to live unnoticed.

    ``coverage`` is emitted even though ``census`` and ``omissions`` determine it, so a reader
    holding only these bytes reads the claim instead of re-deriving it -- which is the whole
    reason this record exists rather than a point count.
    """
    return {
        "census": _census_payload(record.census),
        "coverage": record.coverage.value,
        "figure_crop_node_id": record.figure_crop_node_id,
        "figure_crop_sha256": record.figure_crop_sha256,
        "omissions": [
            {
                "detail": omission.detail,
                "marker_id": omission.marker_id,
                "reason": omission.reason.value,
                "x": _pt(omission.x),
                "y": _pt(omission.y),
            }
            for omission in record.omissions
        ],
        "payload_version": DIGITIZATION_PAYLOAD_VERSION,
        "plot_region": {
            "page": record.plot_region.page,
            "x_end": _pt(record.plot_region.x_end),
            "x_start": _pt(record.plot_region.x_start),
            "y_bottom": _pt(record.plot_region.y_bottom),
            "y_top": _pt(record.plot_region.y_top),
        },
        "raw_sha256": record.raw_sha256,
        "recovered": record.recovered,
        "series_id": record.series_id,
    }


def digitization_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of this record: what a store writes, and what its address is over.

    One definition rather than two that happen to agree, for the reason
    :func:`carmel.services.pdf_table_record.inventory_record_bytes` gives: had a store
    canonicalized on its own, a later change to what canonical means here would move the
    stored bytes while :func:`compute_digitization_sha` kept computing the old address, and the
    divergence would surface as every record failing to hash to its own name.
    """
    return canonical_json_bytes(dict(payload))


def compute_digitization_sha(payload: Mapping[str, Any]) -> str:
    """This record's content address: the sha256 of its canonical JSON bytes.

    The same rule as :func:`carmel.services.pdf_table_record.compute_inventory_sha`, so a figure
    record and a table record are addressed by one definition and not two.

    WHAT THIS ADDRESSES IS THE CLAIM, NOT THE DIGITIZATION. The payload carries a recovered
    COUNT and no recovered coordinate, so two different digitizations of one figure that agree
    on series id, crop, region, coverage, census and ledger hash to the SAME value while holding
    different points. Use this to deduplicate and to cite coverage claims; never as evidence
    that two producers recovered the same data, and never as a change detector for a
    re-digitization. To ask "is this the SAME digitization?" -- which the recovered points, the
    calibration and the producer settle and this address cannot -- use
    :func:`compute_digitization_identity` instead; see the module docstring for the two-addresses
    split and why this one keeps its meaning unchanged.
    """
    return hashlib.sha256(digitization_record_bytes(payload)).hexdigest()


def coverage_of(payload: Mapping[str, Any]) -> FigureCoverage:
    """The coverage claim these bytes state: COMPLETE, PARTIAL or UNCHECKABLE.

    Reads the STORED claim rather than re-deriving it from the census and the ledger, which is
    the point of storing it. The two cannot disagree in a record that was constructed at all
    (D7-D8), so re-deriving would be a second implementation of an invariant that already
    holds -- and the one place the two implementations differed would be a silent wrong answer.

    Answers the coverage question ALONE. It says nothing about whether the instrument could
    have told, which is :func:`is_auditable`'s question, and reading UNCHECKABLE as a kind of
    coverage verdict is the collapse this pair of functions exists to prevent.

    Raises:
        ValueError: If ``payload`` is not readable as a version-
            :data:`DIGITIZATION_PAYLOAD_VERSION` record. Never returns a default: a reader
            handed an unreadable record must not be told the series is fine.
    """
    return FigureDigitization.from_payload(payload).coverage


def census_of(payload: Mapping[str, Any]) -> MarkerCount:
    """The marker total these bytes carry, or the typed reason there is none.

    Raises:
        ValueError: If ``payload`` is not a readable record.
    """
    return FigureDigitization.from_payload(payload).census


def is_auditable(payload: Mapping[str, Any]) -> bool:
    """Whether the instrument could have told that something was missing.

    The auditability question ALONE, deliberately separate from :func:`coverage_of`. True means
    a marker total exists to measure a completeness claim against; it says NOTHING about
    whether anything was actually missing, which is coverage's question. The two are
    independent: a record can be auditable and partial, auditable and complete, or
    unauditable while still naming omissions it happens to know about.

    Raises:
        ValueError: If ``payload`` is not a readable record. A bare ``False`` for an unreadable
            payload would be indistinguishable from an honest "no census", and those are
            different facts -- one is a broken record, the other a working one reporting a real
            limitation.
    """
    return isinstance(census_of(payload), MarkerCensus)


def omission_reasons_of(payload: Mapping[str, Any]) -> tuple[MarkerOmissionReason, ...]:
    """The reasons the ledger records, in ledger order, as enum members.

    An empty tuple is NOT an approval. It means the ledger names nothing, which is a
    completeness claim only when :func:`is_auditable` is True -- the conjunction a caller must
    write out, and the reason this function does not answer "was anything missing".

    Raises:
        ValueError: If ``payload`` is not a readable record.
    """
    return tuple(omission.reason for omission in FigureDigitization.from_payload(payload).omissions)


def payload_unreadable_reason(payload: Mapping[str, Any]) -> str | None:
    """Why this payload cannot be read as a record, or ``None`` if it can.

    Performs the SAME read the readers above perform -- it calls
    :meth:`FigureDigitization.from_payload`, not a second description of what a record must
    contain -- so a caller can find out before storing or citing a record whether anything will
    be able to say anything about it at all. Mirrors
    :func:`carmel.services.pdf_table_record.footprint_unreadable_reason`, which exists for the
    same reason: a record that cannot be read is not one that fails a check, it is one that can
    never be put to the question, and discovering that after it has been embedded in an
    envelope as a citation is far too late.
    """
    try:
        FigureDigitization.from_payload(payload)
    except UNREADABLE_PAYLOAD as exc:
        return repr(exc)
    return None


# --------------------------------------------------------------------------------------------
# The identity address: what the claim address deliberately cannot answer.
#
# Everything above addresses a CLAIM about coverage. Everything below addresses the DIGITIZATION
# itself -- the recovered points, the calibration that mapped pixels to data coordinates, and the
# producer who attested them. The two are kept apart, not merged: see the module docstring's
# "TWO ADDRESSES, TWO QUESTIONS" for why the claim address must not gain these fields.
# --------------------------------------------------------------------------------------------


class AxisScale(StrEnum):
    """How one axis's pixel positions map to data values.

    Carried so the identity distinguishes two digitizations that read the same pixels off the
    same crop under different axis interpretations -- a log axis mistaken for linear recovers
    entirely different data from identical marker positions, and the identity must not call those
    the same digitization. This record does not EVALUATE the mapping (it renders no pixel to a
    value); it carries which mapping the operator attested, so a change to it changes the address.
    """

    LINEAR = "linear"
    LOG10 = "log10"


@dataclass(frozen=True)
class DigitizedPoint:
    """One recovered coordinate, in the figure's own DATA space.

    This is the thing the claim address has no field for and this identity exists to fold in: two
    attestations of one figure that agree on coverage but disagree here are different digitizations,
    and only an address that hashes these can tell them apart. Coordinates are guarded exactly as
    :class:`MarkerOmission`'s are -- a non-finite or non-``float`` value is refused at construction,
    not discovered at serialization -- because a point that can be BUILT and not hashed is one that
    reaches an identity computation and crashes it.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        _require_coordinate(self.x, where="DigitizedPoint.x")
        _require_coordinate(self.y, where="DigitizedPoint.y")


@dataclass(frozen=True)
class AxisCalibration:
    """One axis's attested pixel-to-data mapping, pinned by two reference anchors.

    ``pixel_low``/``value_low`` and ``pixel_high``/``value_high`` are two points on the axis whose
    pixel positions and data values the operator read off; together with :attr:`scale` they define
    the mapping the digitization used. This record CARRIES the calibration for identity; it does
    not interpret it, so it enforces only that the anchors are finite and distinct enough to name
    two points rather than one -- validating a log axis's sign or an anchor's plausibility would be
    a second measurement this attestation-only record does not make.

    The two anchors must not coincide on either coordinate: a pixel range collapsed to one point,
    or a data range collapsed to one value, is not a calibration, and admitting it would let a
    degenerate mapping stand in the identity as if it were a real one.
    """

    axis_id: str
    scale: AxisScale
    pixel_low: float
    pixel_high: float
    value_low: float
    value_high: float

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, str) or not _IDENTIFIER_RE.fullmatch(self.axis_id):
            raise ValueError(
                f"AxisCalibration.axis_id must be a lowercase identifier (^[a-z][a-z0-9_]*$), got {self.axis_id!r}"
            )
        if not isinstance(self.scale, AxisScale):
            raise ValueError(
                f"AxisCalibration({self.axis_id!r}).scale must be an AxisScale, got "
                f"{type(self.scale).__name__} {self.scale!r}"
            )
        for name, value in (
            ("pixel_low", self.pixel_low),
            ("pixel_high", self.pixel_high),
            ("value_low", self.value_low),
            ("value_high", self.value_high),
        ):
            _require_coordinate(value, where=f"AxisCalibration({self.axis_id!r}).{name}")
        if self.pixel_low == self.pixel_high:
            raise ValueError(
                f"AxisCalibration({self.axis_id!r}): pixel_low and pixel_high are both "
                f"{self.pixel_low!r} -- two anchors at one pixel name no mapping"
            )
        if self.value_low == self.value_high:
            raise ValueError(
                f"AxisCalibration({self.axis_id!r}): value_low and value_high are both "
                f"{self.value_low!r} -- a pixel range mapped to one value is not a calibration"
            )


@dataclass(frozen=True)
class DigitizationIdentity:
    """A digitization's own identity: the claim, plus the data it stands for and who made it.

    The claim address (:func:`compute_digitization_sha`) answers "did two producers say the same
    thing about coverage?". This answers "is this the SAME digitization?" -- and it must, because
    figure values in this project are operator ATTESTATIONS, and two operators attesting one figure
    are precisely who will agree on coverage and differ on the points. An address that could not
    tell them apart would make "which digitization produced this value" unanswerable and a
    re-attestation undetectable.

    So the identity binds three things the claim address omits, on top of the claim itself:

    - :attr:`recovered_points` -- the actual data coordinates. Exactly :attr:`FigureDigitization.recovered`
      of them (D-I1): the identity must account for the points the claim SAYS were recovered, no
      more and no fewer, or it is an identity for a different digitization than the claim describes.
    - :attr:`calibration` -- the pixel-to-data mapping, one entry per axis, so a re-reading under a
      different calibration is a different identity even when it happens to land the same points.
    - :attr:`producer` -- who or what attested them. An attestation is who made it as much as what
      it says, so a different producer is a different digitization.

    The claim is bound by its ADDRESS rather than re-embedded, so the identity differs whenever the
    claim differs (series id, crop, region, coverage, census, ledger) OR any of the three above
    differ. Nothing incidental is folded in -- no timestamp, no object identity -- so two
    byte-identical attestations share one identity address, which is the property that makes the
    address a stable name rather than a per-construction nonce.

    NOT a citation surface. Nothing embeds or resolves this yet: wiring figure verification into
    replay is a separate, later ticket. Today it is the structure a caller builds to COMPARE two
    digitizations, via :func:`compute_digitization_identity`.
    """

    record: FigureDigitization
    recovered_points: tuple[DigitizedPoint, ...]
    calibration: tuple[AxisCalibration, ...]
    producer: str

    def __post_init__(self) -> None:
        if not isinstance(self.record, FigureDigitization):
            raise ValueError(
                f"DigitizationIdentity.record must be a FigureDigitization, got {type(self.record).__name__}"
            )
        if not isinstance(self.recovered_points, tuple):
            raise ValueError(
                f"DigitizationIdentity.recovered_points must be a tuple, got "
                f"{type(self.recovered_points).__name__} -- a list is mutable and this identity is an address"
            )
        for position, point in enumerate(self.recovered_points):
            if not isinstance(point, DigitizedPoint):
                raise ValueError(
                    f"DigitizationIdentity.recovered_points[{position}] must be a DigitizedPoint, got "
                    f"{type(point).__name__} {point!r}"
                )
        # D-I1: the identity accounts for exactly the points the claim says were recovered. A
        # claim that recovered ten points and an identity carrying nine is an identity for a
        # different digitization than its own claim describes, and the address would silently name
        # that mismatch instead of refusing it.
        if len(self.recovered_points) != self.record.recovered:
            raise ValueError(
                f"DigitizationIdentity: the claim says recovered={self.record.recovered} but the identity "
                f"carries {len(self.recovered_points)} recovered point(s) -- an identity must account for "
                "exactly the points its claim describes"
            )
        if not isinstance(self.calibration, tuple):
            raise ValueError(
                f"DigitizationIdentity.calibration must be a tuple, got {type(self.calibration).__name__} -- "
                "a list is mutable and this identity is an address"
            )
        for position, axis in enumerate(self.calibration):
            if not isinstance(axis, AxisCalibration):
                raise ValueError(
                    f"DigitizationIdentity.calibration[{position}] must be an AxisCalibration, got "
                    f"{type(axis).__name__} {axis!r}"
                )
        # D-I2: a digitization recovered data coordinates, and a coordinate has no meaning without
        # an axis mapping. An empty calibration is an identity that cannot say what its points mean.
        if not self.calibration:
            raise ValueError(
                "DigitizationIdentity.calibration is empty -- a recovered coordinate without an axis "
                "calibration names no data value, so an identity for one cannot omit it"
            )
        seen: set[str] = set()
        for axis in self.calibration:
            if axis.axis_id in seen:
                raise ValueError(
                    f"DigitizationIdentity: duplicate axis_id {axis.axis_id!r} in calibration -- one axis, "
                    "calibrated twice, makes 'which mapping does this axis use' ill-defined"
                )
            seen.add(axis.axis_id)
        # Sorted for the same reason the omission ledger is (D6): the identity is an address, and an
        # unordered collection would give one calibration as many addresses as it has permutations.
        if list(self.calibration) != sorted(self.calibration, key=lambda axis: axis.axis_id):
            raise ValueError(
                "DigitizationIdentity.calibration must be sorted ascending by axis_id, or one calibration "
                "has as many content addresses as it has permutations"
            )
        if not isinstance(self.producer, str):
            raise ValueError(f"DigitizationIdentity.producer must be a string, got {type(self.producer).__name__}")
        if not self.producer or self.producer != self.producer.strip():
            raise ValueError(
                "DigitizationIdentity.producer must be non-empty and free of surrounding whitespace, got "
                f"{self.producer!r} -- an unnamed producer is not who attested this"
            )


def _calibration_payload(axis: AxisCalibration) -> dict[str, Any]:
    return {
        "axis_id": axis.axis_id,
        "pixel_high": _pt(axis.pixel_high),
        "pixel_low": _pt(axis.pixel_low),
        "scale": axis.scale.value,
        "value_high": _pt(axis.value_high),
        "value_low": _pt(axis.value_low),
    }


def digitization_identity_payload(identity: DigitizationIdentity) -> dict[str, Any]:
    """The full stored form of one digitization's identity, ready for :func:`canonical_json_bytes`.

    The claim is bound by its ADDRESS (``claim_sha256``), not re-embedded: the identity differs
    whenever the claim differs, without duplicating the claim's fields, and the claim address keeps
    its own separate meaning. The recovered points are hashed in the order given -- order is the
    attested sequence, part of what the operator recorded, not incidental -- and the calibration is
    already sorted by ``axis_id`` at construction. Coordinates are spelled with :func:`_pt`, the
    one coordinate spelling this module and the table lane share.
    """
    return {
        "calibration": [_calibration_payload(axis) for axis in identity.calibration],
        "claim_sha256": compute_digitization_sha(digitization_record_payload(identity.record)),
        "identity_version": DIGITIZATION_IDENTITY_VERSION,
        "producer": identity.producer,
        "recovered_points": [{"x": _pt(point.x), "y": _pt(point.y)} for point in identity.recovered_points],
    }


def digitization_identity_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact byte form of one identity: what its address is over.

    One definition, canonicalized by :func:`canonical_json_bytes` exactly as the claim payload is,
    so a claim address and an identity address are addressed by one rule and not two.
    """
    return canonical_json_bytes(dict(payload))


def compute_digitization_identity(payload: Mapping[str, Any]) -> str:
    """This digitization's identity address: the sha256 of its canonical identity bytes.

    WHAT THIS ADDRESSES IS THE DIGITIZATION, NOT ONLY THE CLAIM. Unlike
    :func:`compute_digitization_sha`, this folds in the recovered points, the calibration and the
    producer, so two attestations of one figure that agree on coverage but differ in a single
    recovered coordinate, in the calibration used, or in who produced them get DIFFERENT addresses
    -- while two byte-identical attestations get the SAME one, because nothing incidental is folded
    in. Use this to ask "is this the same digitization?" and to detect a re-attestation as a
    change; use :func:`compute_digitization_sha` to deduplicate and cite coverage CLAIMS. See the
    module docstring's "TWO ADDRESSES, TWO QUESTIONS" for the split.
    """
    return hashlib.sha256(digitization_identity_bytes(payload)).hexdigest()
