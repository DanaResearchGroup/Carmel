"""A general entry point for the geometric PDF table lane.

The lane -- :func:`~carmel.services.pdf_table_proposer.propose_tables`,
:func:`~carmel.services.pdf_tables.build_inventory` (called inside the proposer),
:func:`~carmel.services.table_data_discriminator.classify_table`,
:func:`~carmel.services.pdf_table_store.store_inventory_record` and
:func:`~carmel.services.pdf_table_store.verify_stored_inventory` -- was built and
tested over months, and until this module NOTHING in production called any of it. The
only paths that reached a stored artifact
(:mod:`carmel.services.tabular_dataset_target`,
:mod:`carmel.services.condition_set_target`) refuse every document but one hard-coded
``sha256`` and pass a HAND-DRAWN footprint constant, so the pipeline could process
exactly two documents chosen at compile time.

This module connects the lane to an ARBITRARY document, addressed only by its
``raw_sha256`` argument -- no sha constant, no footprint constant, no page number, no
caption string lives on this path. It carries the document as far as the existing
pieces honestly allow:

1. read the bytes from the evidence store by ``raw_sha256`` and AUTHENTICATE them
   (re-hash; the stored bytes must be the document they are filed under);
2. extract fragments, carrying the extraction's ``lossy``/``status`` honestly rather
   than discarding it;
3. propose table regions (the proposer draws its own ``DERIVED`` footprints);
4. build an inventory per proposal (the proposer does this inside ``propose_tables``);
5. classify each proposed grid;
6. for a grid the classifier calls MEASURED over a NON-lossy extraction, store the
   inventory record and prove it REPLAYS off disk before reporting it stored.

**Every judgement here is already made by a function that exists.** This module adds no
heuristic, no ruled-line detection, no quantity-name matching. It is wiring.

**The tabular dataset SERIES step is OPTIONAL, gated on a supplied ``series_agent``.**
The final envelope producer
(:func:`~carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`,
and the bridge :func:`~carmel.services.proposal_intake.tabular_series_from_proposal`)
requires the caller to ASSERT, per column, an axis ``role`` (independent coordinate vs
dependent observation), a ``quantity_kind``, a grounded ``unit_quote`` and the series'
``value_origin``. :func:`classify_table` supplies none of these: its verdict is
whole-grid (MEASURED / NOT_MEASURED / UNDECIDED), and while its grounds name which
columns are numeric or sweep, they bind no quantity, no unit and no axis role to a
column -- by design, since quantity-name matching is a thing the classifier refuses on
principle. So those per-column semantics come from the extraction agent, not from wiring:

* With ``series_agent=None`` (the default), this path stops at the stored, replayable
  GRID and records the series step DEFERRED (:data:`_SERIES_DEFERRED`), pending a source
  of per-column axis semantics. That deferral is honest and is NOT a refusal to store:
  the grid IS stored.
* With a ``series_agent`` supplied, each MEASURED-and-stored grid is additionally carried
  through :func:`~carmel.services.tabular_series_intake.series_from_classified_grid` --
  which prompts the agent for a ``TabularSeriesProposal``, resolves it against the grid,
  and stores the series only after it replays -- becoming
  :attr:`CandidateStatus.SERIES_STORED` or, on any per-candidate refusal,
  :attr:`CandidateStatus.SERIES_REFUSED`. Never a raise: one grid's series refusal never
  aborts the document.

Failure is closed. A document whose bytes are absent or do not authenticate raises
:class:`GeneralTableReportError` and processes no candidate. Everything else is
per-candidate: a geometric refusal (including every grid on a lossy extraction, which
``build_inventory`` refuses page-incomplete), a not-measured or undecided grid, or a
measured grid whose staged record does not replay off disk (refused, never stored) is
reported with a typed reason that names the step and the document, and the OTHER
candidates are still carried. A document with four tables, one of which is measurable,
yields that one and reports the rest -- and if a fifth's record failed to replay, that
one is refused beside them, not a reason to abort the document.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from carmel.agents.bridge import AgentBridgeError, CarmelAgent
from carmel.schemas.datasets import CaptionLabelKey, EmbeddedTableInventory
from carmel.services.dataset_bridge import UnstorableDatasetEnvelopeError
from carmel.services.dataset_producer import DatasetProducerError, QuoteGroundingError
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import artifact_dir
from carmel.services.pdf_fragments import (
    FragmentAvailability,
    FragmentExtraction,
    extract_fragments,
)
from carmel.services.pdf_table_proposer import (
    CAPTION_HEADING,
    ProposalRefusalReason,
    ProposedTable,
    propose_tables,
)
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_table_store import (
    store_inventory_record,
    verify_stored_inventory,
)
from carmel.services.proposal_intake import ProposalIntakeError, current_extraction_text
from carmel.services.table_data_discriminator import (
    DataVerdict,
    Ground,
    TableClassification,
    classify_table,
    table_view_from_pdf_payload,
)
from carmel.services.tabular_series_intake import TabularSeriesIntakeError, series_from_classified_grid
from carmel.services.tabular_series_resolver import TabularSeriesResolutionError

__all__ = [
    "CandidateOutcome",
    "CandidateStatus",
    "DocumentTableReport",
    "GeneralTableReportError",
    "report_document_tables",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_NAME = "raw.bin"

#: Deferral note recorded on every MEASURED candidate. NOT a refusal -- the grid is
#: stored -- but the record of WHY the lane stops at the grid rather than a series.
_SERIES_DEFERRED = (
    "grid classified MEASURED and its inventory stored; a tabular dataset SERIES is NOT "
    "produced because it needs per-column axis semantics (role, quantity_kind, unit, "
    "value_origin) that classify_table does not supply and this wiring may not invent "
    "(a non-goal). Pending a source of axis semantics."
)


class GeneralTableReportError(RuntimeError):
    """A whole-document failure that stops before any candidate can be considered.

    Raised only for the conditions under which no honest per-candidate work is possible
    at all: the requested ``raw_sha256`` is malformed, the document's ``raw.bin`` is
    absent under it, the stored bytes do not hash to it, or the store promoted a record
    to an address that differs from the one just verified (a store-integrity failure
    that impugns the whole store, not one candidate). Everything a single table
    candidate can do wrong is reported per candidate on :class:`CandidateOutcome`, never
    raised, so one bad candidate never aborts the rest -- a candidate whose staged record
    does not replay off disk is a :attr:`CandidateStatus.REPLAY_REFUSED` outcome, NOT a
    raise.
    """


class CandidateStatus(StrEnum):
    """What became of one table candidate."""

    STORED = "stored"
    """A MEASURED grid over a non-lossy extraction: its inventory record was stored and
    PROVED to replay off disk. The only status that writes an artifact."""

    PROPOSAL_REFUSED = "proposal_refused"
    """The proposer declined to offer this candidate at all. The geometric reason is
    carried in :attr:`CandidateOutcome.proposal_refusal_reason`."""

    NOT_MEASURED = "not_measured"
    """A proposed grid the classifier ruled carries no measured data. Not stored."""

    UNDECIDED = "undecided"
    """A proposed grid the classifier could neither confirm nor deny. Not stored;
    :attr:`CandidateOutcome.needed` names what a human would look for."""

    REPLAY_REFUSED = "replay_refused"
    """A MEASURED grid whose STAGED inventory record did not replay off disk, so it was
    refused rather than promoted to the store. A per-candidate refusal, never a
    document-level raise: the record only ever existed in the torn-down staging
    workspace, so the real store is untouched and the document still yields its other
    candidates. The typed reason (including the verifier's outcome) is in
    :attr:`CandidateOutcome.detail`."""

    SERIES_STORED = "series_stored"
    """A STORED grid additionally carried through :func:`series_from_classified_grid`
    (only attempted when a ``series_agent`` is supplied): the agent's proposal resolved,
    the resulting dataset envelope replayed cleanly off disk, and it was stored. The
    dataset's address is in :attr:`CandidateOutcome.series_dataset_sha256`; the grid's
    own inventory record is still recorded on :attr:`CandidateOutcome.stored_inventory_sha256`."""

    SERIES_REFUSED = "series_refused"
    """A STORED grid was offered to a supplied ``series_agent`` but the series step was
    refused -- typed and per-candidate, never a raise, so this candidate's underlying
    grid stays stored (:attr:`CandidateOutcome.stored_inventory_sha256` is still set) and
    the rest of the document is unaffected. The typed reason is in
    :attr:`CandidateOutcome.detail`."""


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """The outcome for one table candidate, success or refusal.

    Exactly one of two shapes is populated: a geometric refusal (``status`` is
    :attr:`CandidateStatus.PROPOSAL_REFUSED`, ``proposal_refusal_reason`` set,
    ``verdict`` is ``None``) or a proposed-and-classified grid (``verdict`` set). A
    STORED outcome additionally carries ``stored_inventory_sha256``. The MEASURED-grid
    series step then depends on whether a ``series_agent`` was supplied: without one,
    ``series_deferred`` explains why the lane stops at the grid; with one, the grid is
    carried on and the outcome becomes ``SERIES_STORED`` (with ``series_dataset_sha256``
    set) or ``SERIES_REFUSED`` (with the typed reason in ``detail``).
    """

    candidate: str
    page: int
    caption_fragment: str
    status: CandidateStatus
    detail: str = ""
    proposal_refusal_reason: ProposalRefusalReason | None = None
    verdict: DataVerdict | None = None
    grounds: tuple[Ground, ...] = ()
    needed: tuple[str, ...] = ()
    stored_inventory_sha256: str | None = None
    series_deferred: str | None = None
    series_dataset_sha256: str | None = None

    @property
    def stored(self) -> bool:
        return self.status in (CandidateStatus.STORED, CandidateStatus.SERIES_STORED)


@dataclass(frozen=True, slots=True)
class DocumentTableReport:
    """Everything the pipeline could make of one document, per candidate.

    ``extraction_lossy`` and ``extraction_status`` are carried straight off the
    :class:`FragmentExtraction` so a reader sees the extraction floor this document sat
    on. ``extraction_unavailable`` is the proposer's document-wide
    :attr:`ProposalRefusalReason.EXTRACTION_UNAVAILABLE` -- a property of the
    document/toolchain, reported once here rather than smeared across candidates.
    """

    raw_sha256: str
    extraction_lossy: bool
    extraction_status: FragmentAvailability
    extraction_unavailable: bool
    extraction_unavailable_detail: str
    outcomes: tuple[CandidateOutcome, ...] = field(default_factory=tuple)

    @property
    def stored(self) -> tuple[CandidateOutcome, ...]:
        return tuple(o for o in self.outcomes if o.stored)

    @property
    def refused(self) -> tuple[CandidateOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.stored)


def _authenticated_raw(workspace_root: Path, raw_sha256: str) -> bytes:
    """Read ``raw.bin`` for ``raw_sha256`` and prove it is the document it is filed as.

    Mirrors the discipline of
    :func:`carmel.services.tabular_dataset_target.read_target_raw`, but with the sha as
    an ARGUMENT rather than a compile-time constant: the bytes are re-hashed and the
    digest must equal ``raw_sha256``, so a corrupt or mislabelled store cannot feed a
    document the caller did not ask for into the lane.
    """
    if _SHA256_RE.match(raw_sha256) is None:
        raise GeneralTableReportError(f"{raw_sha256!r} is not a lowercase hex sha256 digest; refusing to resolve it")
    raw_path = artifact_dir(workspace_root, raw_sha256) / _RAW_NAME
    if not raw_path.is_file():
        raise GeneralTableReportError(
            f"no stored raw.bin for {raw_sha256} in this workspace's evidence store (looked at {raw_path})"
        )
    try:
        raw = raw_path.read_bytes()
    except OSError as exc:
        raise GeneralTableReportError(f"cannot read the stored raw.bin at {raw_path}: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != raw_sha256:
        raise GeneralTableReportError(
            f"stored raw.bin at {raw_path} hashes to {actual}, not the requested {raw_sha256}; "
            "this evidence directory is not trustworthy, refusing to use it"
        )
    return raw


def _label(page: int, caption_fragment: str) -> str:
    return f"page {page}, caption {caption_fragment!r}"


#: The series step must never abort the document (see the module docstring's "closed
#: failure" promise, extended per-candidate to the series step). Every exception the
#: series bridge and the agent call itself can raise for an honest, typed reason on ONE
#: candidate is caught here and turned into :attr:`CandidateStatus.SERIES_REFUSED`;
#: nothing else is caught, so a genuine bug still surfaces.
_SERIES_REFUSAL_EXCEPTIONS = (
    TabularSeriesIntakeError,
    AgentBridgeError,
    ProposalIntakeError,
    TabularSeriesResolutionError,
    DatasetProducerError,
    QuoteGroundingError,
    # The intake owns the single dataset store; a store that refuses the envelope
    # (UnstorableDatasetEnvelopeError) is this candidate's refusal, not a document abort.
    UnstorableDatasetEnvelopeError,
)


def _caption_label_key(caption_fragment: str) -> CaptionLabelKey | None:
    """Derive the table label the series agent must echo, from the proposer's caption.

    Only the ``"Table N"`` heading text is used -- never fabricated. A candidate whose
    caption fragment carries no such heading (the table_label friction) has no usable
    key, and the caller must report that as a typed refusal rather than inventing one.

    Args:
        caption_fragment: :attr:`CandidateOutcome.caption_fragment` for the candidate.

    Returns:
        The :class:`CaptionLabelKey` naming the table, or ``None`` if the caption
        carries no ``"Table N"`` heading.
    """
    match = CAPTION_HEADING.match(caption_fragment)
    if match is None:
        return None
    label = " ".join(match.group(0).split())
    if not label:
        return None
    return CaptionLabelKey(label=label)


def _classify_and_maybe_store(
    workspace_root: Path,
    staging_root: Path,
    raw_sha256: str,
    proposal: ProposedTable,
    *,
    max_bytes: int,
    series_agent: CarmelAgent | None = None,
) -> CandidateOutcome:
    """Classify one proposed grid and, only if honestly warranted, store it.

    The store is gated two ways, each a guard this module owns:

    * the grid must classify :attr:`DataVerdict.MEASURED` -- NOT_MEASURED and UNDECIDED
      grids are reported, never stored;
    * the record must REPLAY off disk
      (``StoredInventoryVerification.usable_as_table_evidence``) BEFORE it reaches the
      real store, or THIS candidate is refused
      (:attr:`CandidateStatus.REPLAY_REFUSED`) -- because a stored value that does not
      replay is the one failure this whole project exists to prevent. The refusal is
      per-candidate, never a raise: the real store is never touched (promotion is gated
      on the replay proof), so the document still yields its other candidates.

    That "before" is the whole repair. The record store is append-only and cannot
    retract (:func:`carmel.services.pdf_table_store.store_inventory_record` has no
    inverse), and :func:`verify_stored_inventory` proves replay by reading the record
    back OFF DISK -- so a record verified at its permanent content address could not be
    withdrawn when the check fails, leaving a non-replaying record in the store forever.
    Instead the record is published to a private ``staging_root`` workspace, verified
    off disk THERE with the real (un-weakened) verifier against a byte-identical copy of
    ``raw.bin``, and only then promoted to ``workspace_root`` -- a content-addressed,
    idempotent write of the very bytes just proven to replay. On any replay failure the
    real store is never touched.

    Lossiness needs no gate here: ``build_inventory`` refuses every footprint on a lossy
    extraction (``InventoryRefusalReason.PAGE_INCOMPLETE``), so a lossy document never
    produces a proposal to reach this function -- its grids surface as
    :attr:`CandidateStatus.PROPOSAL_REFUSED` upstream, and the document-level
    ``extraction_lossy`` flag carries the fact.
    """
    page = proposal.footprint.page
    label = _label(page, proposal.caption_fragment)

    payload = inventory_record_payload(proposal.inventory, raw_sha256=raw_sha256)
    view = table_view_from_pdf_payload(payload)
    classification = classify_table(view)
    verdict = classification.verdict

    if verdict is DataVerdict.NOT_MEASURED:
        return CandidateOutcome(
            candidate=label,
            page=page,
            caption_fragment=proposal.caption_fragment,
            status=CandidateStatus.NOT_MEASURED,
            detail="classifier ruled this grid carries no measured data",
            verdict=verdict,
            grounds=classification.grounds,
        )
    if verdict is DataVerdict.UNDECIDED:
        return CandidateOutcome(
            candidate=label,
            page=page,
            caption_fragment=proposal.caption_fragment,
            status=CandidateStatus.UNDECIDED,
            detail="classifier could neither confirm nor deny measured data",
            verdict=verdict,
            grounds=classification.grounds,
            needed=classification.needed,
        )

    # verdict is MEASURED from here. Prove replay off disk BEFORE the real store sees the
    # record: publish to staging, verify there, promote only on success (see docstring).
    staged_sha256 = store_inventory_record(staging_root, proposal.inventory, raw_sha256=raw_sha256)
    stored = verify_stored_inventory(staging_root, raw_sha256, staged_sha256, max_bytes=max_bytes)
    if not stored.usable_as_table_evidence:
        # Refuse THIS candidate, do not abort the document. The record lived only in the
        # torn-down staging workspace and was never promoted below, so the real store is
        # untouched and the other candidates are still carried (partial yield).
        return CandidateOutcome(
            candidate=label,
            page=page,
            caption_fragment=proposal.caption_fragment,
            status=CandidateStatus.REPLAY_REFUSED,
            detail=(
                f"staged inventory {staged_sha256} does not replay off disk "
                f"({stored.outcome.value}: {stored.detail}); refusing to publish a stored "
                "value that does not reproduce"
            ),
            verdict=verdict,
            grounds=classification.grounds,
        )
    inventory_sha256 = store_inventory_record(workspace_root, proposal.inventory, raw_sha256=raw_sha256)
    if inventory_sha256 != staged_sha256:
        # Content-addressed from identical inputs, so this is unreachable barring a bug;
        # fail closed rather than report a promoted address that is not the verified one.
        raise GeneralTableReportError(
            f"promoted inventory {inventory_sha256} for {raw_sha256} is not the staged-and-verified "
            f"address {staged_sha256}; refusing to report a value whose replay was not proven"
        )
    if series_agent is None:
        return CandidateOutcome(
            candidate=label,
            page=page,
            caption_fragment=proposal.caption_fragment,
            status=CandidateStatus.STORED,
            detail="inventory record stored and proved to replay off disk",
            verdict=verdict,
            grounds=classification.grounds,
            stored_inventory_sha256=inventory_sha256,
            series_deferred=_SERIES_DEFERRED,
        )

    series_status, series_detail, series_dataset_sha256 = _attempt_series_step(
        workspace_root,
        raw_sha256=raw_sha256,
        proposal=proposal,
        payload=payload,
        classification=classification,
        series_agent=series_agent,
    )
    return CandidateOutcome(
        candidate=label,
        page=page,
        caption_fragment=proposal.caption_fragment,
        status=series_status,
        detail=series_detail,
        verdict=verdict,
        grounds=classification.grounds,
        stored_inventory_sha256=inventory_sha256,
        series_dataset_sha256=series_dataset_sha256,
    )


def _attempt_series_step(
    workspace_root: Path,
    *,
    raw_sha256: str,
    proposal: ProposedTable,
    payload: object,
    classification: TableClassification,
    series_agent: CarmelAgent,
) -> tuple[CandidateStatus, str, str | None]:
    """Drive one STORED candidate's grid through :func:`series_from_classified_grid`.

    Never raises: every honest, typed refusal from the series bridge or the agent call
    itself is caught and reported as :attr:`CandidateStatus.SERIES_REFUSED` on THIS
    candidate alone, so one grid's series refusal can never abort the document (the same
    partial-yield discipline the module docstring promises for every other step).

    Args:
        workspace_root: The real workspace to store the resulting dataset envelope in.
        raw_sha256: The document's authenticated sha256.
        proposal: The candidate's proposed table, already classified MEASURED.
        payload: The inventory record payload already computed for this candidate
            (``inventory_record_payload(proposal.inventory, raw_sha256=raw_sha256)``).
        classification: The candidate's :class:`TableClassification`.
        series_agent: The extraction agent driving the series proposal.

    Returns:
        A ``(status, detail, series_dataset_sha256)`` triple: ``SERIES_STORED`` with the
        dataset's sha256, or ``SERIES_REFUSED`` with a typed detail and no sha256.
    """
    table_key = _caption_label_key(proposal.caption_fragment)
    if table_key is None:
        return (
            CandidateStatus.SERIES_REFUSED,
            f"caption {proposal.caption_fragment!r} carries no 'Table N' heading; refusing to "
            "fabricate a table label for the series agent",
            None,
        )
    canonical = canonical_json_bytes(payload).decode("utf-8")
    embedded_inventory = EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=raw_sha256,
        canonical_json=canonical,
    )
    try:
        document_text = current_extraction_text(workspace_root, raw_sha256)
        stored = series_from_classified_grid(
            workspace_root,
            agent=series_agent,
            expected_sha256=raw_sha256,
            table_key=table_key,
            inventory=embedded_inventory,
            classification=classification,
            document_text=document_text,
        )
    except _SERIES_REFUSAL_EXCEPTIONS as exc:
        return (CandidateStatus.SERIES_REFUSED, f"series step refused: {exc}", None)
    return (
        CandidateStatus.SERIES_STORED,
        "inventory record stored, and the agent-proposed series stored and proved to replay off disk",
        stored.sha256,
    )


def report_document_tables(
    workspace_root: Path,
    raw_sha256: str,
    *,
    max_bytes: int | None = None,
    series_agent: CarmelAgent | None = None,
) -> DocumentTableReport:
    """Carry one arbitrary document through the geometric table lane.

    Args:
        workspace_root: The workspace whose ``evidence/literature/<raw_sha256>/``
            directory holds the document, and under which any stored inventory record
            is written.
        raw_sha256: The document to process. The ONLY document identity on this path;
            no value is hard-coded anywhere.
        max_bytes: Cap on the ``raw.bin`` size for the off-disk replay check. Defaults
            to the exact size of the authenticated bytes -- the honest bound, since the
            replay verifies against the very file just read.
        series_agent: Optional extraction agent. When ``None`` (the default), behaviour
            is byte-unchanged from before this parameter existed: every STORED candidate
            keeps :attr:`CandidateOutcome.series_deferred` set and no series step runs.
            When supplied, every MEASURED-and-stored candidate is additionally carried
            through :func:`~carmel.services.tabular_series_intake.series_from_classified_grid`,
            becoming :attr:`CandidateStatus.SERIES_STORED` or
            :attr:`CandidateStatus.SERIES_REFUSED` -- never a raise, so one candidate's
            series refusal can never abort the document.

    Returns:
        A :class:`DocumentTableReport`: one :class:`CandidateOutcome` per proposed table
        and per geometric refusal, plus the document-level extraction facts.

    Raises:
        GeneralTableReportError: The document's ``raw.bin`` is absent under
            ``raw_sha256`` or does not authenticate, or the store promoted a record to an
            address differing from the one just verified. Never raised for anything one
            candidate can do wrong -- a candidate whose staged record does not replay off
            disk is reported as :attr:`CandidateStatus.REPLAY_REFUSED`, not raised.
    """
    raw = _authenticated_raw(workspace_root, raw_sha256)
    replay_cap = len(raw) if max_bytes is None else max_bytes

    extraction: FragmentExtraction = extract_fragments(raw)
    outcome = propose_tables(extraction)

    extraction_unavailable = ""
    outcomes: list[CandidateOutcome] = []

    for refusal in outcome.refusals:
        if refusal.reason is ProposalRefusalReason.EXTRACTION_UNAVAILABLE:
            # A property of the document/toolchain, reported once at document level.
            extraction_unavailable = refusal.detail
            continue
        outcomes.append(
            CandidateOutcome(
                candidate=_label(refusal.page, refusal.caption_fragment),
                page=refusal.page,
                caption_fragment=refusal.caption_fragment,
                status=CandidateStatus.PROPOSAL_REFUSED,
                detail=refusal.detail,
                proposal_refusal_reason=refusal.reason,
            )
        )

    if outcome.proposals:
        # A private staging workspace holds each MEASURED record for its off-disk replay
        # proof before it is promoted to the real store. Torn down unconditionally: it is
        # scratch, never evidence, and a byte-identical copy of raw.bin is enough for the
        # verifier's re-derivation (raw.bin is content-addressed by its own hash).
        staging_root = Path(tempfile.mkdtemp(prefix="carmel-gtr-stage-"))
        try:
            staged_raw = artifact_dir(staging_root, raw_sha256) / _RAW_NAME
            staged_raw.parent.mkdir(parents=True, exist_ok=True)
            staged_raw.write_bytes(raw)
            for proposal in outcome.proposals:
                outcomes.append(
                    _classify_and_maybe_store(
                        workspace_root,
                        staging_root,
                        raw_sha256,
                        proposal,
                        max_bytes=replay_cap,
                        series_agent=series_agent,
                    )
                )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    return DocumentTableReport(
        raw_sha256=raw_sha256,
        extraction_lossy=extraction.lossy,
        extraction_status=extraction.status,
        extraction_unavailable=bool(extraction_unavailable),
        extraction_unavailable_detail=extraction_unavailable,
        outcomes=tuple(outcomes),
    )
