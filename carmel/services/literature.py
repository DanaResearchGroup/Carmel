"""Literature-research run orchestration.

Implements the fixed sequence from the agentic-layer interface doc (section 8):

1. Acquire an exclusive run lock (``evidence/literature/.run.lock``) and a session run
   slot; a concurrent second run raises :class:`LiteratureRunLockedError`.
2. Run the Literature Agent in one bounded loop (wall-clock checked every iteration;
   self-stop and no-new-information both terminate it).
3. For each proposed finding: ``fetch -> extract_text -> store_artifact ->
   ground_finding``. An ungrounded finding goes straight to ``report.rejected`` and
   NEVER reaches the Verifier — the entire point of this design is that a second LLM
   must never be able to launder the first LLM's fabrication.
4. Only grounded findings are scored by the Verifier, which is shown ONLY the payload,
   citation, quote, a bounded window of EXTRACTED text around the located quote, and
   the grounding verdict — never the author agent's raw ``source_url`` or unquoted
   assertions.
5. Species are canonicalized deterministically; failed canonicalization flags the
   finding and caps credence, as does a fuzzy (non-exact) grounding match.
6. The report is always returned — budget exhaustion and bridge errors become typed
   ``StopReason`` values on a PARTIAL report, never a crash.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from carmel.agents.bridge import AgentBridgeError, ModelProtocol
from carmel.agents.budget import (
    BudgetExceededError,
    BudgetLedger,
    session_budget,
)
from carmel.agents.literature_agent import (
    CorpusProposal,
    LiteratureProposal,
    ProposedFinding,
    RequestedPaper,
    VerifierAssessment,
    build_corpus_agent,
    build_literature_agent,
    build_verifier_agent,
)
from carmel.agents.models import build_model
from carmel.agents.tools.academic import (
    CrossrefSearchTool,
    OaLookupCoverage,
    OpenAccessResolver,
    OpenAccessResolverProtocol,
    OpenAlexSearchTool,
    normalize_doi,
)
from carmel.agents.tools.extract import ExtractedText, extract_text, normalize_for_match
from carmel.agents.tools.fetch import (
    FetchError,
    FetchToolProtocol,
    HttpFetchTool,
    MockFetchTool,
)
from carmel.agents.tools.search import (
    HttpSearchTool,
    MockSearchTool,
    SearchError,
    SearchResult,
    SearchToolProtocol,
)
from carmel.config import (
    KEYLESS_SEARCH_PROVIDERS,
    AgentConfig,
    AgentProvider,
    SearchProvider,
)
from carmel.logger import get_logger
from carmel.schemas.acquisition import AcquisitionReason
from carmel.schemas.campaign import Campaign
from carmel.schemas.literature import (
    CURRENT_REPORT_SCHEMA_VERSION,
    ROOT_EXTRACTION_ID,
    STOP_REASON_FOR_DIMENSION,
    CorpusReadOutcome,
    CoveredDocument,
    CredenceVerdict,
    EvidenceRef,
    GroundingStatus,
    GroundingVerdict,
    LiteratureFinding,
    LiteraturePassMode,
    LiteratureReport,
    PassRecord,
    QueryRecord,
    RejectedFinding,
    SpeciesRef,
    StopReason,
    StoredArtifact,
)
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.services import chem
from carmel.services.acquisition import collect_inbox, host_is_admissible, record_request
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_typed_event
from carmel.services.evidence import (
    list_artifacts_with_unreadable,
    load_artifact_text,
    store_artifact,
    verify_artifact,
)
from carmel.services.extraction_record import (
    CurrentSelectionKind,
    select_current_extraction,
)
from carmel.services.grounding import find_quote, ground_finding
from carmel.services.plan_progress import (
    DEFAULT_LOCK_GRACE_S,
    LITERATURE_RUN_LOCK_DIR,
    lock_is_live,
    publish_lock_info,
    read_lock_info,
)
from carmel.services.provenance import record_agent_provenance

logger = get_logger("services.literature")

#: How each refusing :class:`CurrentSelectionKind` is reported to the operator.
#:
#: Exhaustive over every kind EXCEPT the two that are not refusals:
#: :attr:`~CurrentSelectionKind.SELECTED` (the document was read) and
#: :attr:`~CurrentSelectionKind.NO_RECORDS_STORED` (the one route that legitimately
#: falls through to the root tiers). A mapping rather than a chain of ``if``s so that
#: adding a kind without deciding how to report it fails with a ``KeyError`` at the
#: point of use, instead of silently taking some default branch -- which, in a function
#: whose default branch serves unauthenticated text, is precisely the failure mode this
#: whole selector exists to remove.
_RECORD_REFUSAL_OUTCOMES: dict[CurrentSelectionKind, CorpusReadOutcome] = {
    CurrentSelectionKind.NO_CURRENT_RECORD: CorpusReadOutcome.NO_CURRENT_EXTRACTION_RECORD,
    CurrentSelectionKind.MULTIPLE_CURRENT_RECORDS: (CorpusReadOutcome.MULTIPLE_CURRENT_EXTRACTION_RECORDS),
    CurrentSelectionKind.UNUSABLE_RECORD_PRESENT: (CorpusReadOutcome.UNUSABLE_EXTRACTION_RECORD_PRESENT),
    CurrentSelectionKind.STORE_UNREADABLE: CorpusReadOutcome.EXTRACTION_RECORD_STORE_UNREADABLE,
    CurrentSelectionKind.RECORD_STORE_ESCAPES_WORKSPACE: (CorpusReadOutcome.EXTRACTION_RECORD_STORE_ESCAPES_WORKSPACE),
    CurrentSelectionKind.EMPTY_RECORD_STORE_PRESENT: (CorpusReadOutcome.EMPTY_EXTRACTION_RECORD_STORE_PRESENT),
    CurrentSelectionKind.RECORD_STORE_LINK_DANGLING: (CorpusReadOutcome.EXTRACTION_RECORD_STORE_LINK_DANGLING),
    CurrentSelectionKind.EXTRACTOR_IDENTITY_UNAVAILABLE: (CorpusReadOutcome.EXTRACTOR_IDENTITY_UNAVAILABLE),
    CurrentSelectionKind.RECORD_AUTHENTICATION_FAILED: (CorpusReadOutcome.EXTRACTION_RECORD_AUTHENTICATION_FAILED),
}

LITERATURE_REPORT_NAME = "literature_report.json"
RUN_LOCK_DIR_NAME = ".run.lock"
LOCK_INFO_NAME = "info.json"

#: Chars of extracted text shown to the Verifier on each side of the located quote.
VERIFIER_EVIDENCE_WINDOW = 600

#: Credence ceiling applied when any species failed canonicalization.
NON_CANONICAL_CREDENCE_CAP = 0.7

#: Why a corpus finding naming a document Carmel does not hold is refused. Stated
#: once so the report's ``reason`` text and the grounding verdict cannot drift apart.
_UNHELD_ARTIFACT_REASON = "the agent named a document that is not in this workspace's evidence store"

#: Max queries actually executed (and recorded in ``report.queries``/provenance) per
#: Literature Agent round. Anything beyond this is dropped -- loudly, via a warning
#: and a decision-log event -- rather than recorded as if it had run (Finding 6).
MAX_QUERIES_PER_ROUND = 5

#: Search results requested per executed query.
SEARCH_RESULTS_PER_QUERY = 5

#: Max open-access candidate URLs actually fetched per wanted paper. OA resolution can
#: advertise many repository mirrors for one DOI; without a cap a single paper could
#: burn the run's whole fetch budget on copies of bytes it already failed to get.
MAX_OA_FETCH_ATTEMPTS_PER_PAPER = 5

#: Grace period during which a run-lock directory with missing/unparseable
#: ``info.json`` is still treated as LIVE -- a peer may be between ``mkdir()`` and
#: its metadata write. Kept as an alias so existing callers/tests do not need to
#: import two names for the same constant; the actual liveness decision is made
#: exclusively by :func:`carmel.services.plan_progress.lock_is_live`, which takes
#: this same default.
LOCK_GRACE_S = DEFAULT_LOCK_GRACE_S

#: Stop reasons that mean "a budget ceiling ended the run".
_BUDGET_STOP_REASONS = frozenset(STOP_REASON_FOR_DIMENSION.values())

__all__ = [
    "LITERATURE_REPORT_NAME",
    "LiteratureDeps",
    "LiteratureRunLockedError",
    "build_deps",
    "load_literature_report",
    "migrate_report_payload",
    "run_corpus_pass",
    "run_literature_research",
    "run_record_for",
    "save_literature_report",
]


def save_literature_report(workspace_root: Path, report: LiteratureReport) -> None:
    """Durably persist ``report`` as the workspace's literature report artifact.

    Uses :func:`carmel.services.artifacts.write_json` (mkstemp + fsync + ``os.replace`` +
    parent-dir fsync) rather than a naive ``Path.write_text()``, so a crash mid-write can
    never leave a torn/truncated file masquerading as a completed report -- every sibling
    workspace-artifact service already writes this way (Finding 5).

    Args:
        workspace_root: Root directory of the campaign workspace.
        report: The report to persist.
    """
    write_json(workspace_root / LITERATURE_REPORT_NAME, report)


def load_literature_report(workspace_root: Path) -> LiteratureReport:
    """Load the workspace's persisted literature report artifact.

    Args:
        workspace_root: Root directory of the campaign workspace.

    Returns:
        The parsed :class:`LiteratureReport`.

    Raises:
        FileNotFoundError: If no report has been saved for this workspace yet.
    """
    return LiteratureReport.model_validate(migrate_report_payload(read_json(workspace_root / LITERATURE_REPORT_NAME)))


class ReportSchemaTooNewError(ValueError):
    """A persisted report was written by a NEWER Carmel than this one.

    A distinct type so the dispatcher can turn it into a clean, actionable refusal
    ("upgrade Carmel") instead of a generic handler crash, while genuine corruption --
    which also raises ValueError -- keeps failing loudly (spar round 8, P2).
    """


def migrate_report_payload(payload: object) -> object:
    """Bring a persisted report payload up to the current schema version.

    v1 held exactly one run, with that run's identifiers and outcome at the top
    level. v2 accumulates passes, so the migration lifts those top-level fields into
    a single :class:`PassRecord` and stamps every finding, rejection and query with
    that run's identity -- the attribution was always true of a v1 report, it simply
    had nowhere to be written down.

    v3 adds ``covered_sha256`` to each pass, so a corpus pass can skip documents an
    earlier pass already mined. Nothing can be reconstructed for an older report --
    what a v1/v2 pass covered was never written down -- so migrated passes carry an
    empty list, which reads as "not recorded" and makes the first v3 pass re-read
    the corpus once. That is the conservative direction: re-reading costs tokens,
    while inventing coverage would silently skip documents nobody has mined.

    v4 re-keys coverage from a bare raw sha256 to the PAIR (raw sha256, extraction
    identity), because a single stored document can hold more than one extraction
    (the root sidecar, plus whatever re-extraction has since produced) and coverage
    of one must not be read as coverage of another. The v3->v4 step maps each raw
    sha in ``covered_sha256`` to a pair with ``extraction_id`` set to
    :data:`ROOT_EXTRACTION_ID`. That mapping is not a guess: for as long as v3 was the
    current schema, `_load_corpus` loaded text solely via `load_artifact_text`, which
    reads the ROOT sidecar, so root text is the only thing any v3 pass could possibly
    have mined. It can now also serve an authenticated extraction record, which is
    exactly why this mapping is stated as a fact about PAST passes and must never be
    re-derived from what the loader does today.

    This is a CHAIN, applied step by step from the payload's own version. It used to
    be a single branch that ran the v1 lift for ANY version below current, which was
    correct only while current was 2: at v3 a v2 payload would have been fed through
    the v1 lift, whose `pop("run_id")` finds nothing and whose rebuilt `passes` would
    have overwritten the real ones (the review flagged this as latent).

    A live campaign already holds a v1 report on disk (the first real run), and it
    contains the only existing evidence that the grounding gate refuses ungrounded
    claims. Migrating rather than discarding is therefore not a courtesy to old
    files; it preserves the demonstration.
    """
    if not isinstance(payload, dict):
        return payload
    version = int(payload.get("schema_version", 1))
    if version > CURRENT_REPORT_SCHEMA_VERSION:
        # Fail closed on a report from the FUTURE (spar round 7, P2). Passing it
        # through unmigrated hands it to a validator that does not know the fields it
        # carries, and `extra="forbid"` would reject it with a schema error naming a
        # field the operator has never heard of. Worse, a future version that only
        # ADDED optional fields would validate cleanly and silently drop them on the
        # next write -- a newer Carmel's report quietly downgraded by an older one.
        raise ReportSchemaTooNewError(
            f"literature report is schema version {version}, but this Carmel understands "
            f"at most {CURRENT_REPORT_SCHEMA_VERSION}. Upgrade Carmel rather than letting "
            f"an older version rewrite (and silently truncate) a newer report."
        )
    if version == CURRENT_REPORT_SCHEMA_VERSION:
        return payload

    migrated = dict(payload)
    if version < 2:
        migrated = _migrate_v1_to_v2(migrated)
    if version < 3:
        migrated = _migrate_v2_to_v3(migrated)
    if version < 4:
        migrated = _migrate_v3_to_v4(migrated)
    if version < 5:
        migrated = _migrate_v4_to_v5(migrated)
    if version < 6:
        migrated = _migrate_v5_to_v6(migrated)
    # Not a literal. This produces whatever the CURRENT schema is, so hardcoding the
    # number means the next version bump silently stamps migrated reports with a
    # stale version -- and the `version == CURRENT` early return above then treats
    # them as already migrated.
    migrated["schema_version"] = CURRENT_REPORT_SCHEMA_VERSION
    return migrated


def _migrate_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Lift v1's single flat run into a one-element ``passes`` list."""
    migrated = dict(payload)
    run_id = str(migrated.pop("run_id", "") or "")
    action_id = str(migrated.pop("action_id", "") or "")
    created_at = migrated.get("created_at")
    pass_record: dict[str, Any] = {
        "run_id": run_id,
        "action_id": action_id,
        "created_at": created_at,
        "mode": LiteraturePassMode.SEARCH.value,
        "model_name": migrated.pop("model_name", ""),
        "stop_reason": migrated.pop("stop_reason", StopReason.ERROR.value),
        "usage": migrated.pop("usage", None),
        "warnings": migrated.pop("warnings", []),
    }
    migrated["passes"] = [pass_record]
    migrated["queries"] = [
        {"text": q, "run_id": run_id, "action_id": action_id} for q in migrated.get("queries", []) if q
    ]
    for key in ("findings", "rejected"):
        items = migrated.get(key) or []
        migrated[key] = [
            {**item, "run_id": item.get("run_id") or run_id, "action_id": item.get("action_id") or action_id}
            if isinstance(item, dict)
            else item
            for item in items
        ]
    return migrated


def _migrate_v2_to_v3(payload: dict[str, Any]) -> dict[str, Any]:
    """Give every pass an explicit empty ``covered_sha256``.

    The field defaults to empty anyway, so this changes no value. It is written
    out because a migration that silently relies on a default is indistinguishable
    from a migration step somebody forgot to write, and the next bump has to be able
    to tell those apart.
    """
    migrated = dict(payload)
    passes = migrated.get("passes")
    if isinstance(passes, list):
        migrated["passes"] = [
            {**p, "covered_sha256": p.get("covered_sha256", [])} if isinstance(p, dict) else p for p in passes
        ]
    return migrated


def _migrate_v3_to_v4(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-key each pass's ``covered_sha256`` to ``covered``, a list of (raw sha256,
    extraction identity) pairs.

    Every raw sha in the old list is mapped to ``ROOT_EXTRACTION_ID``. This is not a
    guess: while v3 was current, `_load_corpus` loaded document text solely via
    `load_artifact_text`, which reads the ROOT `extracted.json` sidecar, so root text
    is the only thing any v3 pass could possibly have mined. The loader can now also
    serve an authenticated extraction record; this mapping describes what OLD passes
    did and is not affected by that. The old
    ``covered_sha256`` key is removed, not left alongside the new one -- `PassRecord`
    sets ``extra="forbid"``, so a payload still carrying it would be rejected.
    """
    migrated = dict(payload)
    passes = migrated.get("passes")
    if isinstance(passes, list):
        new_passes = []
        for p in passes:
            if not isinstance(p, dict):
                new_passes.append(p)
                continue
            new_p = dict(p)
            old_covered = new_p.pop("covered_sha256", [])
            new_p["covered"] = [{"raw_sha256": sha, "extraction_id": ROOT_EXTRACTION_ID} for sha in old_covered]
            new_passes.append(new_p)
        migrated["passes"] = new_passes
    return migrated


def _migrate_v4_to_v5(payload: dict[str, Any]) -> dict[str, Any]:
    """Set ``extraction_id`` to :data:`ROOT_EXTRACTION_ID` on every finding's
    ``evidence`` mapping.

    This is not a guess: at the time pre-v5 findings were written,
    `_ground_and_record`'s only two callers obtained the text a finding was grounded
    against either from `_fetch_and_store` (which extracts freshly and stores the
    result as the root sidecar) or from `_load_corpus` (which then loaded text solely
    via `load_artifact_text`, i.e. the root sidecar). So root text is the only thing
    any pre-v5 finding could possibly have been grounded against. `_load_corpus` can
    now also serve an authenticated extraction record, which changes nothing about
    findings already on disk -- and is why this reasoning is pinned to when those
    findings were written rather than to today's loader. ``quote_start``/``quote_end`` and every other field are left
    untouched -- this migration only adds the identity that was previously implicit.
    """
    migrated = dict(payload)
    findings = migrated.get("findings")
    if isinstance(findings, list):
        new_findings = []
        for f in findings:
            if not isinstance(f, dict):
                new_findings.append(f)
                continue
            evidence = f.get("evidence")
            if isinstance(evidence, dict):
                new_f = dict(f)
                new_evidence = dict(evidence)
                # Assigned unconditionally, NOT via setdefault. By the argument above, a
                # pre-v5 payload cannot legitimately carry this key at all, and
                # ``EvidenceRef`` sets ``extra="forbid"``, so one that did used to be
                # REJECTED outright. Honouring such a value would quietly turn that
                # refusal into acceptance of a claim -- "these offsets index record
                # <sha>" -- that no v4 writer could have had grounds to make, and that
                # nothing downstream can check once the offsets are frozen.
                new_evidence["extraction_id"] = ROOT_EXTRACTION_ID
                new_f["evidence"] = new_evidence
                new_findings.append(new_f)
            else:
                new_findings.append(f)
        migrated["findings"] = new_findings
    return migrated


def _migrate_v5_to_v6(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp ``"unrecorded"`` onto every pass's ``covered`` entries'
    ``verification_standard``.

    ``"unrecorded"`` is a fact, not a guess: no pre-v6 writer ever considered which
    :class:`~carmel.schemas.literature.CorpusReadOutcome` a document was read under,
    because the field did not exist, so the honest value for every existing record is
    "this was never recorded" rather than a reconstruction from the store's CURRENT
    state (which may since have changed, and in any case cannot speak for what the
    pass actually checked at the time).

    Assigned unconditionally, NOT via setdefault. A pre-v6 payload cannot legitimately
    carry this key at all -- ``CoveredDocument`` sets ``extra="forbid"``, so one that
    did was REJECTED outright -- and honouring such a value would quietly turn that
    refusal into acceptance of a claim ("this was read to standard <x>") that no v5
    writer could have had grounds to make. A mutation audit previously caught exactly
    this ``setdefault`` substitution in ``_migrate_v4_to_v5``; the same guard applies
    here.
    """
    migrated = dict(payload)
    passes = migrated.get("passes")
    if isinstance(passes, list):
        new_passes = []
        for p in passes:
            if not isinstance(p, dict):
                new_passes.append(p)
                continue
            covered = p.get("covered")
            if isinstance(covered, list):
                new_p = dict(p)
                new_p["covered"] = [
                    {**c, "verification_standard": "unrecorded"} if isinstance(c, dict) else c for c in covered
                ]
                new_passes.append(new_p)
            else:
                new_passes.append(p)
        migrated["passes"] = new_passes
    return migrated


def _load_previous_report(workspace_root: Path) -> LiteratureReport | None:
    """The report this run will append to, or None on the first pass.

    Deliberately called BEFORE the run does any work. If an existing report cannot
    be read, the correct outcome is to refuse to start -- discovering it afterwards
    would leave a finished run holding results it can only persist by overwriting
    the record it failed to parse, which is how accumulated evidence gets destroyed
    by an error handler.
    """
    try:
        return load_literature_report(workspace_root)
    except FileNotFoundError:
        return None


class LiteratureRunLockedError(RuntimeError):
    """A literature run is already in progress for this workspace."""


@dataclass
class LiteratureDeps:
    """Injected dependencies for one literature-research run.

    ``model`` backs the proposing Literature Agent; ``verifier_model`` backs the
    independent Verifier. These MUST be two distinct objects, never the same one:
    the Verifier's independence from the Literature Agent is the entire point of
    having it, and a model implementation that carries state across ``complete()``
    calls (e.g. a conversational wrapper that accumulates history) would leak the
    proposing agent's context into the "independent" assessment if the two personas
    shared an instance. See :class:`~carmel.agents.bridge.ModelProtocol` for the
    statelessness requirement this reuse-as-default would violate.

    ``build_deps`` builds ``verifier_model`` as a FRESH instance from a second call to
    :func:`~carmel.agents.models.build_model` — never a reference to ``model``. A
    caller with a custom model factory (e.g. a test wanting to prove independence
    with a deliberately stateful fake) may construct :class:`LiteratureDeps` directly
    and pass any two distinct ``ModelProtocol`` objects.

    ``verifier_model`` is REQUIRED, not optional: a caller who omitted it used to get
    a silent ``MockModel`` fallback that pays for a full agent loop plus fetches
    before failing on the first grounded finding with a PARTIAL report whose only
    symptom is a warning string (Finding 22). Construction now fails loudly instead.
    """

    config: AgentConfig
    model: ModelProtocol
    search: SearchToolProtocol
    fetch: FetchToolProtocol
    ledger: BudgetLedger
    verifier_model: ModelProtocol
    """The Verifier's own, independent model. MUST be a distinct object from
    ``model`` -- see :meth:`__post_init__`. Real production wiring should prefer
    :func:`build_deps`, which always supplies a genuinely fresh model built the same
    way as ``model``."""
    oa_resolver: OpenAccessResolverProtocol | None = None
    """Deterministic DOI -> open-access-PDF resolution for the ``wanted`` channel.

    ``None`` (the mock tier, and any caller that does not wire one) means resolution
    simply cannot run; a wanted paper is then queued with
    :attr:`~carmel.schemas.acquisition.AcquisitionReason.NO_OPEN_ACCESS_COPY` and a
    detail saying so -- NEVER with ``PAYWALLED``, which would assert an observation
    nobody made."""

    def __post_init__(self) -> None:
        """Guarantee ``verifier_model`` is never the same object as ``model``."""
        if self.verifier_model is self.model:
            raise ValueError(
                "LiteratureDeps.verifier_model must not be the same object as model: "
                "the Verifier's independence from the proposing Literature Agent requires "
                "two distinct model instances"
            )


def build_deps(config: AgentConfig, *, daily_ledger_path: Path | None = None) -> LiteratureDeps:
    """Construct real or mock dependencies per the config tier, fail-closed.

    ``AgentProvider.MOCK`` gets mock tools (no network, empty canned responses unless a
    caller replaces them). Any real provider gets the real, SSRF-guarded,
    budget-reserved tools — and, exactly like :func:`carmel.agents.models.build_model`,
    a missing search endpoint or API key raises instead of silently downgrading to a
    mock.

    ``deps.model`` and ``deps.verifier_model`` are always two SEPARATE instances built
    by two separate calls to :func:`build_model` — never the same object — so the
    Verifier can never share state (conversation history, caches, ...) with the
    Literature Agent it is meant to independently assess. This holds for
    ``AgentProvider.MOCK`` too: a mock config still gets two distinct
    :class:`~carmel.agents.models.MockModel` instances (each with its own canned
    response queue and call log).

    Args:
        config: The agent configuration.
        daily_ledger_path: Optional path of the persisted daily cost ledger.

    Returns:
        Fully-wired dependencies.

    Raises:
        AgentBridgeError: If a real provider is configured without consent/keys
            (from :func:`build_model`), or without a usable search configuration.
    """
    ledger = BudgetLedger(config.budget, daily_ledger_path=daily_ledger_path)

    if config.provider == AgentProvider.MOCK:
        return LiteratureDeps(
            config=config,
            model=build_model(config),
            verifier_model=build_model(config),
            search=MockSearchTool({}),
            fetch=MockFetchTool({}),
            ledger=ledger,
        )

    # Built BEFORE the models: this is pure local-config validation, so a campaign
    # misconfigured for search must fail on that, not on whichever error a later
    # model construction happens to raise first.
    search = _build_search_tool(config, ledger)

    return LiteratureDeps(
        config=config,
        model=build_model(config),
        verifier_model=build_model(config),
        search=search,
        fetch=HttpFetchTool(
            ledger=ledger,
            max_artifact_bytes=config.budget.max_artifact_bytes,
            external_provider_consent=config.external_provider_consent,
        ),
        ledger=ledger,
        oa_resolver=OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=config.external_provider_consent,
            contact_email=config.search_contact_email,
            unpaywall_email=config.resolved_unpaywall_email(),
            core_api_key=config.resolved_core_api_key(),
            semantic_scholar_api_key=config.resolved_semantic_scholar_api_key(),
            elsevier_api_key=config.resolved_elsevier_api_key(),
            elsevier_insttoken=config.resolved_elsevier_insttoken(),
        ),
    )


def _build_search_tool(config: AgentConfig, ledger: BudgetLedger) -> SearchToolProtocol:
    """Build the configured search backend, failing closed on missing credentials.

    Only ``SearchProvider.HTTP_JSON`` needs an operator-supplied endpoint and key; the
    scholarly indices are keyless by design and requiring credentials for them made
    them impossible to configure at all.

    Args:
        config: The agent configuration.
        ledger: Budget ledger shared with the rest of the run's tools.

    Returns:
        A search tool satisfying :class:`SearchToolProtocol`.

    Raises:
        AgentBridgeError: If ``HTTP_JSON`` is selected without an endpoint, without a
            key env var name, or with that env var unset/empty.
    """
    if config.search_provider == SearchProvider.OPENALEX:
        return OpenAlexSearchTool(
            ledger=ledger,
            contact_email=config.search_contact_email,
            external_provider_consent=config.external_provider_consent,
        )
    if config.search_provider == SearchProvider.CROSSREF:
        return CrossrefSearchTool(
            ledger=ledger,
            contact_email=config.search_contact_email,
            external_provider_consent=config.external_provider_consent,
        )
    if config.search_provider in KEYLESS_SEARCH_PROVIDERS:  # pragma: no cover - defensive
        raise AgentBridgeError(
            f"search provider {config.search_provider!r} is declared keyless but has no "
            "backend wired here; add it above rather than falling through to the "
            "keyed HTTP path, which would demand credentials it does not use"
        )

    if not config.search_endpoint:
        raise AgentBridgeError(f"provider {config.provider!r} requires search_endpoint to be set")
    env_var = config.search_api_key_env
    if not env_var:
        raise AgentBridgeError(f"provider {config.provider!r} requires search_api_key_env to be set")
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise AgentBridgeError(f"environment variable {env_var!r} (search_api_key_env) is not set or empty")
    return HttpSearchTool(
        endpoint=config.search_endpoint,
        api_key=api_key,
        ledger=ledger,
        external_provider_consent=config.external_provider_consent,
    )


# --------------------------- run lock -----------------------------------------
#
# This used to be a 131-line hand-rolled mkdir lock with no literature-specific
# content at all -- and it got all three hard parts of a breakable mkdir lock
# wrong relative to the lock `carmel.services.dispatcher` implements for the
# structurally identical `.dispatch.lock` (both are leases that outlive a short
# critical section, guarded by pid+pid_start liveness, breakable when stale):
#
#   - stealing a stale lock via `rmtree` then `mkdir` is NOT atomic: two racers
#     that both judge the lock stale can both `rmtree` + `mkdir` and both
#     believe they hold it, because each racer's `rmtree` unconditionally
#     removes whatever is there -- including a peer's brand-new live lock.
#   - hand-writing `info.json` with `write_text` (no `pid_start`) leaves the
#     pid-reuse guard in `plan_progress.lock_is_live` permanently dead for this
#     lock: a crash whose pid later gets reused makes the lock unbreakable.
#   - releasing via a bare, unconditional `rmtree(lock_dir)` deletes whatever is
#     at that path when this frame returns, including a SUCCESSOR's live lock if
#     this frame's own lock was ever broken as stale (e.g. cross-host, or via
#     the missing-pid_start bug above).
#
# `plan_progress` already solves all three (atomic rename-to-a-uuid-suffixed
# stale target, publish_lock_info with pid_start, and -- via the pattern this
# module mirrors from `dispatcher._DispatchLease` -- an ownership-checked
# release). The steal/acquire loop below deliberately mirrors
# `dispatcher._acquire_dispatch_lock` rather than reusing it directly: that
# function and its `_DispatchLease` companion are private to `dispatcher.py`
# (owned by a different part of this change), so the acquire/steal/release
# *loop* is necessarily written twice for now. What is NOT duplicated is the
# actual liveness/staleness JUDGMENT (`lock_is_live`) or the metadata
# read/write (`read_lock_info`/`publish_lock_info`) -- those are the shared,
# previously-buggy parts, and both call sites now defer to the same
# `plan_progress` implementation. The right long-term fix is to lift the
# lease type itself (acquire loop + ownership-checked release) into
# `plan_progress` so neither module carries its own copy of the loop; that is
# a cross-cutting change outside this module's ownership boundary and is
# flagged in the review response rather than done here.


def _describe_stale_reason(lock_dir: Path, info: dict[str, Any], *, stale_after_s: float) -> str:
    """Human-readable explanation for a lock ``lock_is_live`` already judged not-live.

    Purely descriptive, for the ``literature.lock_broken`` decision-log event --
    the stale/live DECISION itself is made exclusively by
    :func:`carmel.services.plan_progress.lock_is_live` before this is ever
    called, so there is only one place that judgment can be gotten wrong.
    """
    pid = info.get("pid")
    hostname = info.get("hostname")
    if hostname == socket.gethostname() and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return f"pid {pid} on this host is dead"
        else:
            # Alive but judged not-live: `lock_is_live` only reaches this state
            # for a live pid when the recorded `pid_start` no longer matches --
            # i.e. the pid number was reused after the original holder died.
            return f"pid {pid} on this host was reused since the lock was taken (original holder is dead)"

    started_at_raw = info.get("started_at")
    if isinstance(started_at_raw, str):
        try:
            started_at = datetime.fromisoformat(started_at_raw)
        except ValueError:
            pass  # Unparseable: fall through to the mtime-based explanation below.
        else:
            # `publish_lock_info` writes an aware timestamp, but this value comes off
            # disk and may have been written by an older Carmel, hand-edited, or
            # truncated. Subtracting a naive datetime from an aware one raises
            # TypeError, which here would escape as an unhandled crash from a function
            # whose whole job is to EXPLAIN a lock -- so normalize instead of trusting.
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            age_s = (datetime.now(UTC) - started_at).total_seconds()
            return f"lock is {age_s:.0f}s old (> stale_after_s={stale_after_s:.0f}s)"

    if lock_dir.exists():
        lock_age_s = max(0.0, datetime.now(UTC).timestamp() - lock_dir.stat().st_mtime)
    else:
        lock_age_s = float("inf")
    return (
        f"lock metadata missing/unparseable and lock dir is {lock_age_s:.0f}s old "
        f"(> lock_grace_s={DEFAULT_LOCK_GRACE_S:.0f}s)"
    )


class _LiteratureLockLease:
    """Ownership token for a held literature run lock (P1-7).

    Carries the pid + ``/proc`` start time this frame itself published, so
    :meth:`release` can refuse to remove a lock dir whose ``info.json`` no
    longer names this process -- mirroring
    ``carmel.services.dispatcher._DispatchLease`` exactly, for the same
    reason: a frame whose lock was broken as stale (by a peer, cross-host, or
    formerly via the missing-``pid_start`` bug) must never delete its
    SUCCESSOR's live lock just because it still holds the bare ``Path``.
    """

    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = lock_dir
        info = read_lock_info(lock_dir)
        self._pid = info.get("pid")
        self._pid_start = info.get("pid_start")

    def release(self) -> None:
        """Remove the lock dir iff its ``info.json`` still names THIS process."""
        info = read_lock_info(self.lock_dir)
        if info.get("pid") == self._pid and info.get("pid_start") == self._pid_start:
            shutil.rmtree(self.lock_dir, ignore_errors=True)


def _acquire_run_lock(
    workspace_root: Path,
    *,
    action_id: str,
    run_id: str,
    stale_after_s: float,
    log_path: Path,
) -> _LiteratureLockLease:
    """Atomically acquire ``evidence/literature/.run.lock`` as a lease.

    Stealing a stale lock is an ATOMIC RENAME to a process-unique
    ``.stale.<uuid>`` path, never ``rmtree`` + ``mkdir`` (P1-5): renaming a
    given inode can only ever succeed for exactly one racer, so a peer that
    already renamed (or replaced) this same stale lock makes our rename raise
    ``FileNotFoundError`` -- which MUST be treated as losing the race, looping
    back to re-evaluate the lock from scratch, rather than assuming we
    performed the steal.

    Raises:
        LiteratureRunLockedError: If a live (non-stale) lock is already held.
    """
    lock_dir = workspace_root / LITERATURE_RUN_LOCK_DIR
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if lock_is_live(lock_dir, stale_after_s=stale_after_s):
                info = read_lock_info(lock_dir)
                raise LiteratureRunLockedError(
                    f"a literature run already holds {lock_dir} "
                    f"(pid={info.get('pid')}, hostname={info.get('hostname')})"
                ) from None
            info = read_lock_info(lock_dir)
            reason = _describe_stale_reason(lock_dir, info, stale_after_s=stale_after_s)
            stale_target = lock_dir.with_name(f"{lock_dir.name}.stale.{uuid.uuid4().hex}")
            try:
                lock_dir.rename(stale_target)
            except FileNotFoundError:
                # Lost the steal race: a peer already renamed this exact stale lock
                # away (or has since replaced it with a fresh one) between our
                # liveness check and this rename landing. We broke nothing and own
                # nothing -- retry from the top instead of proceeding as if we
                # performed the steal (P1-5).
                continue
            logger.warning("breaking stale literature run lock %s: %s", lock_dir, reason)
            append_typed_event(
                log_path,
                event="literature.lock_broken",
                action_id=action_id,
                run_id=run_id,
                payload={
                    "level": "warning",
                    "reason": reason,
                    "lock_pid": info.get("pid"),
                    "lock_hostname": info.get("hostname"),
                    "lock_started_at": info.get("started_at"),
                },
            )
            # The rename gave this frame exclusive ownership of the renamed-aside
            # inode (no other racer can have a reference to it), so removing it
            # here is race-free.
            shutil.rmtree(stale_target, ignore_errors=True)
            continue
        else:
            publish_lock_info(lock_dir, extra={"action_id": action_id})
            return _LiteratureLockLease(lock_dir)


# --------------------------- helpers ------------------------------------------


def _finding_key(finding: ProposedFinding) -> tuple[str, str]:
    """Dedupe key: normalized DOI (or source URL) plus normalized quote."""
    source = finding.citation.doi or finding.source_url.strip().lower()
    return (source, normalize_for_match(finding.verbatim_quote))


def _campaign_context(campaign: Campaign) -> str:
    """Render the campaign's chemistry context for the Literature Agent."""
    ci = campaign.input
    species = ", ".join(c.species for c in ci.initial_mixture.components)
    observables = ", ".join(o.name for o in ci.target_observables)
    reactors = ", ".join(
        f"{r.reactor_type.value} (T={r.temperature_range_K[0]}-{r.temperature_range_K[1]} K)"
        for r in ci.target_reactor_systems
    )
    return f"Campaign mixture species: {species}\nTarget observables: {observables}\nTarget reactor systems: {reactors}"


def _literature_prompt(
    campaign: Campaign,
    *,
    round_index: int,
    search_results: dict[str, list[SearchResult]],
    reported_keys: list[str],
) -> str:
    """Build the Literature Agent's user prompt for one round."""
    parts = [_campaign_context(campaign), f"Round: {round_index}"]
    if search_results:
        lines = [
            "Search results from your previous queries. FULL TEXT says whether a "
            "readable copy is available: when it says no, you cannot quote the paper, so "
            "put it under `wanted` if it is relevant rather than inventing a quote."
        ]
        for query, results in search_results.items():
            lines.append(f"- query: {query}")
            for r in results:
                availability = "FULL TEXT: yes" if r.pdf_url else "FULL TEXT: no"
                identifier = f"doi:{r.doi}" if r.doi else r.url
                lines.append(f"  * {r.title} | {identifier} | {availability}")
                if r.snippet:
                    lines.append(f"    {r.snippet}")
        parts.append("\n".join(lines))
    if reported_keys:
        parts.append(
            "Already-reported findings (do NOT re-report these sources/quotes):\n"
            + "\n".join(f"- {k}" for k in reported_keys)
        )
    parts.append(
        "Return your structured proposal: new queries to run, new findings with exact "
        "verbatim quotes, and done=true if the search is complete."
    )
    return "\n\n".join(parts)


def _verifier_prompt(
    finding: ProposedFinding,
    *,
    evidence_window: str,
    grounding_dict: dict[str, Any],
) -> str:
    """Build the Verifier's prompt from sanitized evidence ONLY.

    Deliberately excluded: the author agent's raw ``source_url`` and the citation's
    agent-supplied ``url`` — the Verifier must judge from the extracted evidence, not
    from where the author CLAIMED it came from.
    """
    evidence = {
        "payload": finding.payload.model_dump(mode="json"),
        "citation": finding.citation.model_dump(mode="json", exclude={"url"}),
        "verbatim_quote": finding.verbatim_quote,
        "extracted_text_window": evidence_window,
        "grounding_verdict": grounding_dict,
    }
    return "Assess the following grounded literature finding using ONLY this evidence:\n" + json.dumps(
        evidence, indent=2, default=str
    )


def _canonicalize_payload(finding: ProposedFinding) -> tuple[ProposedFinding, bool]:
    """Canonicalize all species refs in a finding's payload via RDKit.

    Returns:
        The (possibly updated) finding and whether ALL species canonicalized. A
        payload with no species refs counts as fully canonicalized.
    """
    payload = finding.payload
    field = "fuel_species" if hasattr(payload, "fuel_species") else "species"
    refs: list[SpeciesRef] = list(getattr(payload, field, []))
    if not refs:
        return finding, True

    all_ok = True
    new_refs: list[SpeciesRef] = []
    for ref in refs:
        smiles = chem.canonical_smiles(ref.canonical_smiles or ref.raw_name)
        if smiles is None:
            all_ok = False
            new_refs.append(ref.model_copy(update={"canonicalized": False}))
        else:
            new_refs.append(
                ref.model_copy(
                    update={
                        "canonical_smiles": smiles,
                        "inchikey": chem.inchikey(smiles),
                        "canonicalized": True,
                    }
                )
            )
    updated = finding.model_copy(update={"payload": payload.model_copy(update={field: new_refs})})
    return updated, all_ok


@dataclass
class _RunState:
    """Mutable accumulator threaded through one run."""

    queries: list[str]
    artifacts: list[StoredArtifact]
    findings: list[LiteratureFinding]
    rejected: list[RejectedFinding]
    warnings: list[str]
    stop_reason: StopReason = StopReason.SELF_TERMINATED
    fetch_failures: dict[str, tuple[AcquisitionReason, str]] = field(default_factory=dict)
    """URL -> (why acquisition failed, detail), so :func:`_process_finding` can queue the
    paper for manual acquisition with an accurate reason instead of re-deriving one by
    parsing warning text."""
    acquisition_slugs: list[str] = field(default_factory=list)
    """Slugs queued for manual acquisition during this run, for the run summary."""
    covered: list[CoveredDocument] = field(default_factory=list)
    """(artifact sha, extraction identity) pairs this pass actually READ, whether or
    not they yielded a finding.

    Keyed by the pair, not the raw sha alone, because a single stored document can
    have more than one extraction on disk, and re-reading a paper under a NEW
    extraction is real work, not a repeat.

    Appended AFTER the model call returns and BEFORE its proposal is processed -- see
    the ordering comment at the append site in ``_corpus_loop`` for why both halves of
    that ordering matter."""

    def query_records(self, pass_record: PassRecord) -> list[QueryRecord]:
        """This run's executed queries, attributed to the pass that ran them."""
        return [
            QueryRecord(text=text, run_id=pass_record.run_id, action_id=pass_record.action_id) for text in self.queries
        ]


def _process_finding(
    workspace_root: Path,
    proposed: ProposedFinding,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    artifact_cache: dict[str, tuple[StoredArtifact, ExtractedText] | None],
    log_path: Path,
    action_id: str,
    run_id: str,
) -> None:
    """Ground one proposed finding; verify and record it only if it survives.

    Order is the design: fetch -> extract -> store -> ground, and ONLY a grounded
    finding is ever shown to the Verifier.
    """
    url = proposed.source_url

    if url not in artifact_cache:
        artifact_cache[url] = _fetch_and_store(workspace_root, url, deps, config=config, state=state)
    cached = artifact_cache[url]
    stored, extracted = cached if cached is not None else (None, None)
    _ground_and_record(
        workspace_root,
        proposed,
        deps,
        config=config,
        state=state,
        stored=stored,
        extracted=extracted,
        log_path=log_path,
        action_id=action_id,
        run_id=run_id,
        queue_acquisition=True,
        # This path fetches and extracts in one breath (`_fetch_and_store` above) and
        # stores the result as the root sidecar, so root is a fact about this path,
        # not a default.
        extraction_id=ROOT_EXTRACTION_ID,
    )


def _ground_and_record(
    workspace_root: Path,
    proposed: ProposedFinding,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    stored: StoredArtifact | None,
    extracted: ExtractedText | None,
    log_path: Path,
    action_id: str,
    run_id: str,
    queue_acquisition: bool,
    extraction_id: str,
) -> None:
    """Ground one proposed finding against resolved bytes; verify and record it only
    if it survives.

    Shared by both passes, and deliberately so: a corpus finding must clear exactly
    the same gate as a fetched one. The only difference between the two callers is
    how the artifact was resolved -- fetched from a URL, or looked up in the store --
    and that difference is settled before this function is entered.

    ``queue_acquisition`` is False for a corpus pass. There, a failed grounding means
    the quote is not in a document Carmel already holds, so asking a human to go and
    obtain that same document would be nonsense.

    ``extraction_id`` is required, not defaulted, for the same reason the
    ``EvidenceRef`` field it feeds is required: a caller that forgets to state which
    text ``extracted`` actually is must fail loudly, not silently claim the root.
    """
    finding_id = uuid.uuid4().hex
    if stored is not None and all(a.sha256 != stored.sha256 for a in state.artifacts):
        state.artifacts.append(stored)

    verdict = ground_finding(
        payload=proposed.payload,
        citation=proposed.citation,
        quote=proposed.verbatim_quote,
        extracted=extracted,
    )
    if not verdict.grounded or stored is None or extracted is None:
        reason = "; ".join(verdict.reasons) or f"grounding failed with status {verdict.status.value}"
        if queue_acquisition:
            _maybe_queue_acquisition(workspace_root, proposed, verdict=verdict, state=state)
        state.rejected.append(
            RejectedFinding(
                finding_id=finding_id,
                run_id=run_id,
                action_id=action_id,
                category=proposed.payload.category,
                citation_title=proposed.citation.title,
                grounding=verdict,
                reason=reason,
            )
        )
        append_typed_event(
            log_path,
            event="literature.finding_rejected",
            action_id=action_id,
            run_id=run_id,
            payload={"finding_id": finding_id, "status": verdict.status.value, "reason": reason},
        )
        return

    match = find_quote(extracted, proposed.verbatim_quote)
    window_start = max(0, (match.start if match else 0) - VERIFIER_EVIDENCE_WINDOW)
    window_end = min(len(extracted.text), (match.end if match else 0) + VERIFIER_EVIDENCE_WINDOW)
    evidence_window = extracted.text[window_start:window_end]

    proposed, canonical_ok = _canonicalize_payload(proposed)

    verifier = build_verifier_agent(model=deps.verifier_model, ledger=deps.ledger)
    verifier_prompt = _verifier_prompt(
        proposed,
        evidence_window=evidence_window,
        grounding_dict=verdict.model_dump(mode="json"),
    )
    result = verifier.run(
        verifier_prompt,
        estimated_tokens=estimated_tokens_for(verifier_prompt),
    )
    assessment = VerifierAssessment.model_validate(result.output)

    credence_value = assessment.credence
    flags = list(assessment.flags)
    if verdict.status == GroundingStatus.GROUNDED_FUZZY:
        credence_value *= verdict.match_ratio
        flags.append("fuzzy_grounding_match")
    if not canonical_ok:
        credence_value = min(credence_value, NON_CANONICAL_CREDENCE_CAP)
        flags.append("species_not_canonicalized")

    # ``VerifierAssessment`` subclasses ``CredenceVerdict`` with an identical field set, so
    # copy it down to the base type rather than hand-transcribing every field (Finding 19).
    credence = CredenceVerdict(
        **assessment.model_dump(mode="python"),
    ).model_copy(update={"credence": credence_value, "flags": flags})

    state.findings.append(
        LiteratureFinding(
            finding_id=finding_id,
            run_id=run_id,
            action_id=action_id,
            payload=proposed.payload,
            citation=proposed.citation,
            verbatim_quote=proposed.verbatim_quote,
            evidence=EvidenceRef(
                artifact_sha256=stored.sha256,
                extraction_id=extraction_id,
                quote_start=match.start if match else None,
                quote_end=match.end if match else None,
                page=match.page if match else None,
                section_label=match.section_label if match else None,
            ),
            grounding=verdict,
            credence=credence,
        )
    )
    append_typed_event(
        log_path,
        event="literature.finding_recorded",
        action_id=action_id,
        run_id=run_id,
        payload={"finding_id": finding_id, "sha256": stored.sha256, "status": verdict.status.value},
    )
    append_typed_event(
        log_path,
        event="literature.credence_assigned",
        action_id=action_id,
        run_id=run_id,
        payload={"finding_id": finding_id, "credence": credence.credence, "flags": flags},
    )


def _validate_document(content_type: str, data: bytes) -> tuple[ExtractedText | None, AcquisitionReason | None]:
    """Gate one fetched payload before it may become a stored evidence artifact.

    This is the ONE place both storage paths in this module must route through --
    :func:`_attempt_oa_fetch` (open-access candidate URLs) and :func:`_fetch_and_store`
    (the LLM-proposed ``source_url`` on a grounding finding). It used to be
    duplicated in the former and simply absent in the latter: live campaign
    ``5b766b4b-bf72-4db9-bb28-4229b037bf07`` (workspace ``live-syngas``) stored all
    3 artifacts fetched via the LLM-proposed-URL path as ``text/html`` with no gate
    at all. One (sha256 ``fdaceab39e73...``) was an Elsevier redirect stub -- 2710
    bytes of HTML whose entire extracted text was the word "Redirecting" (83
    whitespace-padded chars, so it passes a naive non-empty check). The other two
    were repository landing pages carrying nav chrome ("Skip to main content",
    "Log In", "Communities & Collections") rather than the paper. A landing page
    still carries the paper's title/abstract, so a quote lifted from an abstract
    could pass :func:`~carmel.services.grounding.ground_finding` against that
    artifact and read as backed by the full paper.

    HTML is rejected even when it holds real full text -- one artifact from that
    same run was a legitimate ~39.7K-char Frontiers full-text HTML page -- because
    that paper should route to the human PDF queue instead of being trusted as a
    document: consistency and fail-closed behaviour beat retrieval breadth here.

    Args:
        content_type: The sniffed MIME type of the fetched bytes.
        data: The fetched bytes.

    Returns:
        ``(extracted, None)`` on success, else ``(None, reason)`` where ``reason``
        is the OBSERVED :class:`AcquisitionReason` (``NOT_A_DOCUMENT`` or
        ``EMPTY_DOCUMENT``).
    """
    if content_type not in ("application/pdf", "text/plain"):
        return None, AcquisitionReason.NOT_A_DOCUMENT
    if not data:
        return None, AcquisitionReason.EMPTY_DOCUMENT
    extracted = extract_text(data, content_type)
    if not extracted.text.strip():
        return None, AcquisitionReason.EMPTY_DOCUMENT
    return extracted, None


def _attempt_oa_fetch(
    workspace_root: Path,
    url: str,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
) -> tuple[StoredArtifact | None, AcquisitionReason | None, str]:
    """Try one open-access candidate URL through the ordinary guarded fetch tool.

    Every byte moves through ``deps.fetch`` -- the same SSRF guard, consent gate,
    budget reservation and size caps as any agent-proposed fetch; this function opens
    no sockets of its own.

    Success requires the bytes to actually BE a document (sniffed
    ``application/pdf``/``text/plain``): a live probe found 5 of 11 advertised OA
    "PDF" URLs served an HTML landing page, and storing one of those as "the paper"
    would silently retire the acquisition request while leaving no readable text.
    It also requires the bytes to be non-empty and to yield non-whitespace extractable
    text: a live campaign fetched a figshare landing page's zero-byte response and
    stored it as a successful acquisition, silently suppressing the manual queue --
    see :attr:`AcquisitionReason.EMPTY_DOCUMENT`.

    Args:
        workspace_root: Root of the campaign workspace.
        url: The OA candidate URL to try.
        deps: Injected dependencies (the fetch tool in particular).
        config: The agent configuration (artifact size cap).

    Returns:
        ``(stored, None, "")`` on success, else ``(None, reason, outcome)`` where
        ``reason`` is the OBSERVED :class:`AcquisitionReason` and ``outcome`` a
        human-readable account naming the host (e.g. ``"HTTP 403 from
        pubs.rsc.org"``).

    Raises:
        BudgetExceededError: Propagated from the fetch tool's own reservation, like
            every other fetch in this module -- a budget ceiling is a run-level stop,
            not a per-candidate outcome.
    """
    host = urlsplit(url).hostname or url
    if not host_is_admissible(url, config.additional_admissible_hosts):
        # An OA index advertised this URL, which is not the same as vouching for it.
        return (
            None,
            AcquisitionReason.HOST_NOT_ADMISSIBLE,
            f"{host} is not on the admissible-source list",
        )
    try:
        artifact, data = deps.fetch.fetch(url)
    except FetchError as exc:
        if exc.status in (401, 402, 403):
            return None, AcquisitionReason.PAYWALLED, f"HTTP {exc.status} from {host}"
        outcome = f"HTTP {exc.status} from {host}" if exc.status else f"fetch from {host} failed: {exc}"
        return None, AcquisitionReason.FETCH_FAILED, outcome
    extracted, reason = _validate_document(artifact.content_type, data)
    if reason is AcquisitionReason.NOT_A_DOCUMENT:
        return None, reason, f"{artifact.content_type} served from {host}"
    if reason is AcquisitionReason.EMPTY_DOCUMENT:
        detail = "was empty (0 bytes)" if not data else "yielded no extractable text"
        return None, reason, f"{artifact.content_type} served from {host} {detail}"
    assert extracted is not None  # only remaining case: _validate_document succeeded
    try:
        stored = store_artifact(
            workspace_root,
            data=data,
            artifact=artifact,
            extracted=extracted,
            max_bytes=config.budget.max_artifact_bytes,
        )
    except ValueError as exc:
        return None, AcquisitionReason.FETCH_FAILED, f"storing artifact from {host} failed: {exc}"
    return stored, None, ""


def _resolve_oa_candidates(
    paper_doi: str | None, paper_title: str | None, deps: LiteratureDeps
) -> tuple[list[str], str, OaLookupCoverage]:
    """Deterministically resolve a wanted paper's OA candidates, with an honest note.

    Args:
        paper_doi: The paper's normalized DOI, if it has one.
        paper_title: The paper's title, passed through to the resolver's
            title-matched providers (ChemRxiv, arXiv); those skip themselves when it
            is unavailable.
        deps: Injected dependencies (the resolver in particular).

    Returns:
        ``(candidates, note, coverage)``: candidate URLs capped at
        :data:`MAX_OA_FETCH_ATTEMPTS_PER_PAPER`, a note describing what resolution did
        (or why it could not run) for the operator-facing ``detail``, and how much of
        resolution actually ran -- see :class:`~carmel.agents.tools.academic.OaLookupCoverage`.
        Anything short of :attr:`~carmel.agents.tools.academic.OaLookupCoverage.COMPLETE`
        means an empty ``candidates`` establishes nothing.

        The two early returns below synthesise
        :attr:`~carmel.agents.tools.academic.OaLookupCoverage.NOT_ATTEMPTED` directly:
        no resolution object exists in either case, because no provider ever ran.
    """
    if paper_doi is None:
        return (
            [],
            "paper has no DOI, so automated open-access resolution was not attempted",
            OaLookupCoverage.NOT_ATTEMPTED,
        )
    if deps.oa_resolver is None:
        return [], "no open-access resolver is configured for this run", OaLookupCoverage.NOT_ATTEMPTED
    resolution = deps.oa_resolver.resolve(paper_doi, title=paper_title)
    candidates = list(resolution.candidates)
    note = resolution.note
    if len(candidates) > MAX_OA_FETCH_ATTEMPTS_PER_PAPER:
        note = f"{note}; trying the first {MAX_OA_FETCH_ATTEMPTS_PER_PAPER} of {len(candidates)} candidates"
        candidates = candidates[:MAX_OA_FETCH_ATTEMPTS_PER_PAPER]
    return candidates, note, resolution.coverage


def _queue_wanted_paper(
    workspace_root: Path,
    paper: RequestedPaper,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    log_path: Path,
    action_id: str,
    run_id: str,
) -> None:
    """Acquire a wanted paper's open-access copy if one exists; queue it otherwise.

    The agent's claim that this paper is relevant is still NOT checked -- nothing in
    the queue is evidence, and the evidentiary gates apply unchanged to whatever is
    eventually obtained. What IS no longer taken from the agent is the paper's access
    status: a real run queued 12 papers as ``paywalled`` on the model's say-so, of
    which 5 were open access and never once fetched. So, before queuing:

    1. Resolve OA candidates for the DOI (and, for the title-matched preprint
       indexes, the title) deterministically via the pluggable OA provider list --
       never model judgement.
    2. Fetch them in order through the ordinary guarded fetch tool. A success is
       stored as FETCHED evidence and the paper is NOT queued.
    3. Only then queue, with the OBSERVED reason: 401/402/403 -> ``PAYWALLED``
       (preferred over other failures when both were seen, because a paywall is the
       one outcome the operator can act on with a subscription), other failures ->
       ``FETCH_FAILED``/``NOT_A_DOCUMENT``, and no candidate at all ->
       ``NO_OPEN_ACCESS_COPY``. The detail records every URL tried and what it
       returned; the model's relevance prose is kept, clearly labelled, and is never
       the sole content.
    """
    doi = normalize_doi(paper.doi)
    landing_url = paper.landing_url or (f"https://doi.org/{doi}" if doi else None)
    if landing_url is None:
        state.warnings.append(f"ignored a requested paper with neither a DOI nor a URL: {paper.title!r}")
        return

    candidates, resolution_note, resolution_coverage = _resolve_oa_candidates(doi, paper.title or None, deps)

    attempts: list[tuple[str, str]] = []
    observed_reason: AcquisitionReason | None = None
    for url in candidates:
        stored, failure_reason, outcome = _attempt_oa_fetch(workspace_root, url, deps, config=config)
        if stored is not None:
            if all(a.sha256 != stored.sha256 for a in state.artifacts):
                state.artifacts.append(stored)
            state.warnings.append(
                f"fetched an open-access copy of {paper.title!r} from {url}; not queued for manual acquisition"
            )
            append_typed_event(
                log_path,
                event="literature.oa_copy_acquired",
                action_id=action_id,
                run_id=run_id,
                payload={"doi": doi, "title": paper.title, "url": url, "sha256": stored.sha256},
            )
            return
        attempts.append((url, outcome))
        if failure_reason == AcquisitionReason.PAYWALLED or observed_reason is None:
            observed_reason = failure_reason

    if observed_reason is None:
        # No candidate was even attempted. Only claim "no open-access copy exists" when
        # resolution actually finished; if it was cut short (per-paper lookup cap, or a
        # provider failing in transit) nothing has been established, so say exactly
        # that; and if no provider ever ran (no DOI, no resolver, consent withheld) say
        # that instead of either. Total map, no else and no default: a coverage value
        # this dict does not know about must fail loudly, not silently fall through.
        reason = {
            OaLookupCoverage.NOT_ATTEMPTED: AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED,
            OaLookupCoverage.PARTIAL: AcquisitionReason.OA_LOOKUP_INCOMPLETE,
            OaLookupCoverage.COMPLETE: AcquisitionReason.NO_OPEN_ACCESS_COPY,
        }[resolution_coverage]
        observed = resolution_note
    else:
        reason = observed_reason
        observed = "open-access fetch failed: " + "; ".join(f"{url} -> {outcome}" for url, outcome in attempts)
    detail = observed if not paper.relevance else f"{observed} | agent's stated relevance: {paper.relevance}"

    request = record_request(
        workspace_root,
        title=paper.title,
        doi=doi,
        landing_url=landing_url,
        reason=reason,
        detail=detail,
    )
    if request.slug in state.acquisition_slugs:
        return
    state.acquisition_slugs.append(request.slug)
    append_typed_event(
        log_path,
        event="literature.paper_requested",
        action_id=action_id,
        run_id=run_id,
        payload={
            "slug": request.slug,
            "doi": doi,
            "title": paper.title,
            "reason": reason.value,
            "oa_attempts": [url for url, _ in attempts],
        },
    )


def _maybe_queue_acquisition(
    workspace_root: Path,
    proposed: ProposedFinding,
    *,
    verdict: GroundingVerdict,
    state: _RunState,
) -> None:
    """Queue a paper for manual acquisition when — and only when — the EVIDENCE was
    unobtainable.

    The distinction this function draws is the important one. A finding can fail
    grounding for two very different reasons:

    - Carmel could not obtain or read the document (paywall, dead link, image-only
      scan). The claim is untested, and a human with a subscription could settle it.
      That is what the acquisition queue is for.
    - Carmel read the document fine and the claimed quote was not in it
      (``QUOTE_NOT_FOUND``). That is the fabrication signal the whole grounding gate
      exists to raise.

    Only the first is queued. Queueing the second would convert "this agent may have
    invented a quote" into "we are waiting on a human", quietly retiring the strongest
    rejection the system can produce -- so ``QUOTE_NOT_FOUND`` deliberately falls
    through to a plain rejection here and is never given a second chance.

    ``ARTIFACT_DEGRADED`` is a THIRD case and is queued for neither. It means the stored
    bytes are fine but their ``extracted.json`` sidecar is missing, so the gate refused
    to judge against an unlabelled reload. Asking a human to obtain a paper Carmel
    already holds would be nonsense; the remedy is mechanical re-extraction. Since a
    bare rejection in the decision log reads like a verdict on the CLAIM rather than on
    our storage, it is surfaced as a run warning naming the real remedy.
    """
    url = proposed.source_url
    failure = state.fetch_failures.get(url)

    if failure is not None:
        acquisition_reason, detail = failure
    elif verdict.status == GroundingStatus.ARTIFACT_UNREADABLE:
        acquisition_reason = AcquisitionReason.UNREADABLE
        detail = "; ".join(verdict.reasons)
    elif verdict.status == GroundingStatus.ARTIFACT_DEGRADED:
        state.warnings.append(
            f"finding for {proposed.citation.title!r} was rejected because its stored artifact "
            f"has no extraction sidecar, not because the claim failed: re-extract the artifact "
            f"and re-run to get a real verdict"
        )
        return
    else:
        return

    citation = proposed.citation
    landing_url = f"https://doi.org/{citation.doi}" if citation.doi else (citation.url or url)
    request = record_request(
        workspace_root,
        title=citation.title,
        doi=citation.doi,
        landing_url=landing_url,
        reason=acquisition_reason,
        detail=detail,
    )
    if request.slug not in state.acquisition_slugs:
        state.acquisition_slugs.append(request.slug)
        state.warnings.append(
            f"queued for manual acquisition ({acquisition_reason.value}): "
            f"{citation.title!r} -> literature_requests/inbox/{request.slug}.pdf"
        )


def _fetch_and_store(
    workspace_root: Path,
    url: str,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
) -> tuple[StoredArtifact, ExtractedText] | None:
    """Fetch, extract and store one URL; None on any non-budget failure.

    A :class:`BudgetExceededError` raised by the fetch tool's own reservation is NOT
    swallowed here — it propagates to the top-level handler and becomes the run's
    stop reason.

    ``url`` here is the agent-PROPOSED ``source_url`` on a grounding finding, not an
    open-access candidate Carmel resolved itself -- so it goes through the exact same
    :func:`_validate_document` gate as :func:`_attempt_oa_fetch`. Before this gate
    existed, this path stored whatever the LLM's URL returned unvalidated; live
    campaign ``5b766b4b-bf72-4db9-bb28-4229b037bf07`` shows what that produces (see
    :func:`_validate_document`'s docstring for the observed artifacts).
    """
    if not host_is_admissible(url, config.additional_admissible_hosts):
        # Refused BEFORE the fetch: an inadmissible host is never contacted, so a
        # poisoned search result cannot even be told which campaign is reading it.
        host = urlsplit(url).hostname or url
        state.warnings.append(
            f"not auto-admitted from {host}: not a recognised publisher, repository or "
            f"resolver. Queued for manual acquisition instead."
        )
        state.fetch_failures[url] = (
            AcquisitionReason.HOST_NOT_ADMISSIBLE,
            f"host {host!r} is not on the admissible-source list",
        )
        return None
    try:
        artifact, data = deps.fetch.fetch(url)
    except FetchError as exc:
        state.warnings.append(f"fetch failed for {url}: {exc}")
        # Every fetch failure is worth a manual request, because the failure describes
        # this URL, not the paper: a repository link that 404s, a host that will not
        # resolve, and a publisher refusing an unsubscribed client all leave a real,
        # citable paper that a human with a subscription can still obtain. The status
        # only decides which REASON the operator is shown -- 401/402/403 is a paywall
        # they can act on directly, anything else is a broken retrieval path.
        paywalled = exc.status in (401, 402, 403)
        state.fetch_failures[url] = (
            AcquisitionReason.PAYWALLED if paywalled else AcquisitionReason.FETCH_FAILED,
            f"HTTP {exc.status}" if exc.status else str(exc),
        )
        return None
    host = urlsplit(url).hostname or url
    extracted, reason = _validate_document(artifact.content_type, data)
    if reason is not None:
        if reason is AcquisitionReason.NOT_A_DOCUMENT:
            detail = f"{artifact.content_type} served from {host}"
        else:
            detail = f"{artifact.content_type} served from {host} " + (
                "was empty (0 bytes)" if not data else "yielded no extractable text"
            )
        # Same reasoning as the FetchError branch above: this failure describes the
        # URL the LLM proposed, not the paper itself, so it is worth queuing for
        # manual acquisition rather than silently dropping -- see
        # ``_maybe_queue_acquisition``, which consults ``state.fetch_failures`` before
        # it looks at the grounding verdict at all.
        logger.warning("literature: rejected fetch for %s: %s (%s)", url, reason.value, detail)
        state.warnings.append(f"rejected fetch for {url}: {detail}")
        state.fetch_failures[url] = (reason, detail)
        return None
    assert extracted is not None  # only remaining case: _validate_document succeeded
    try:
        stored = store_artifact(
            workspace_root,
            data=data,
            artifact=artifact,
            extracted=extracted,
            max_bytes=config.budget.max_artifact_bytes,
        )
    except ValueError as exc:
        state.warnings.append(f"storing artifact for {url} failed: {exc}")
        return None
    return stored, extracted


def _research_loop(
    workspace_root: Path,
    campaign: Campaign,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    log_path: Path,
    action_id: str,
    run_id: str,
    already_covered: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    """The bounded propose->ground->verify loop. Mutates ``state`` in place."""
    # A search pass discovers documents rather than re-reading held ones, so prior
    # corpus coverage says nothing about what it should fetch.
    del already_covered
    # No live search tool is handed to the agent: the deterministic round-trip below is the
    # *only* path that ever calls ``deps.search.search``. Handing the agent its own live
    # search tool as well would let it re-query out-of-band, double-billing a real provider
    # and bypassing the auditable, truncated query list recorded into ``state.queries``
    # (Finding 23).
    literature = build_literature_agent(
        model=deps.model,
        ledger=deps.ledger,
    )
    seen_keys: set[tuple[str, str]] = set()
    artifact_cache: dict[str, tuple[StoredArtifact, ExtractedText] | None] = {}
    search_results: dict[str, list[SearchResult]] = {}
    round_index = 0

    while True:
        deps.ledger.check_wall_clock()
        round_index += 1
        prompt = _literature_prompt(
            campaign,
            round_index=round_index,
            search_results=search_results,
            reported_keys=[f"{src} :: {quote[:80]}" for src, quote in sorted(seen_keys)],
        )
        result = literature.run(prompt, estimated_tokens=estimated_tokens_for(prompt))
        proposal = LiteratureProposal.model_validate(result.output)

        # Truncate FIRST: only queries we actually execute get recorded into
        # ``state.queries``/provenance. Recording the full proposed list (before truncation)
        # would overstate what the run actually did (Finding 6).
        executed_queries = proposal.queries[:MAX_QUERIES_PER_ROUND]
        dropped_queries = proposal.queries[MAX_QUERIES_PER_ROUND:]
        # Queries never run before. These are what make ANOTHER round worth paying for:
        # their results have not been shown to the agent yet, so the round after this one
        # can still produce something this one could not.
        fresh_queries = [q for q in executed_queries if q not in state.queries]
        for query in executed_queries:
            if query not in state.queries:
                state.queries.append(query)
        search_results = {q: deps.search.search(q, limit=SEARCH_RESULTS_PER_QUERY) for q in executed_queries}
        if dropped_queries:
            state.warnings.append(
                f"round {round_index}: dropped {len(dropped_queries)} quer"
                f"{'y' if len(dropped_queries) == 1 else 'ies'} beyond the "
                f"{MAX_QUERIES_PER_ROUND}-query-per-round cap: {dropped_queries!r}"
            )
            append_typed_event(
                log_path,
                event="literature.queries_truncated",
                action_id=action_id,
                run_id=run_id,
                payload={
                    "round": round_index,
                    "executed": executed_queries,
                    "dropped": dropped_queries,
                },
            )

        for paper in proposal.wanted:
            _queue_wanted_paper(
                workspace_root,
                paper,
                deps,
                config=config,
                state=state,
                log_path=log_path,
                action_id=action_id,
                run_id=run_id,
            )

        new_findings = []
        for proposed in proposal.findings:
            key = _finding_key(proposed)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_findings.append(proposed)

        for proposed in new_findings:
            _process_finding(
                workspace_root,
                proposed,
                deps,
                config=config,
                state=state,
                artifact_cache=artifact_cache,
                log_path=log_path,
                action_id=action_id,
                run_id=run_id,
            )

        if proposal.done:
            state.stop_reason = StopReason.SELF_TERMINATED
            return
        # Stop only when nothing could change next round: no new findings AND no query
        # whose results the agent has not already seen.
        #
        # Requiring new findings alone ended EVERY run after round 1. The agent cannot
        # produce a finding in round 1 by construction -- a finding needs a verbatim
        # quote, quotes come from fetched documents, and documents come from search
        # results, which are not fed back until the NEXT round. So round 1 always
        # returned zero findings, the loop always stopped, and the search results it had
        # just paid for were never shown to anyone. The propose -> search -> read ->
        # propose round-trip that this loop exists to perform had never once run. Tests
        # missed it because a scripted mock returns findings in round 1, which no real
        # model can do.
        if not new_findings and not fresh_queries:
            state.stop_reason = StopReason.NO_NEW_INFORMATION
            return


def _permission_from(parameters: Mapping[str, Any], name: str, state: _RunState) -> bool:
    """Read one permission out of an action's untyped parameter bag, strictly.

    ``PlannedAction.parameters`` is a plain ``dict[str, Any]`` reloaded from
    ``plan.json``, so what arrives here is whatever JSON happened to hold -- and
    ``bool()`` was the wrong reader for it. ``bool("false")`` is True. So is
    ``bool("no")``, ``bool(0.1)``, and ``bool([])``'s inverse for any non-empty list.
    A parameter bag carrying the STRING ``"false"`` therefore GRANTED the permission
    it was plainly trying to decline, which is a fail-open in the one place that
    decides whether unauthenticated text may be read at all.

    Only the literal booleans are honoured. Anything else is refused AND surfaced as
    a warning on the pass, rather than quietly treated as either answer: a value that
    is neither True nor False is a corrupt plan, and an operator who wrote one needs
    to see that their intent did not take effect. Refusing silently would leave them
    believing a permission was granted (or withheld) on evidence that never existed --
    the same assertion-for-observation conflation this module refuses elsewhere.

    Args:
        parameters: The action's parameter bag, exactly as reloaded from disk.
        name: The permission to read.
        state: Run state, for recording a warning when the value is malformed.

    Returns:
        True only when ``name`` maps to the literal ``True``; False when it is absent
        or maps to the literal ``False``; False, with a warning, for anything else.
    """
    if name not in parameters:
        return False
    value = parameters[name]
    # `is` rather than `==`, deliberately: `1 == True` and `0 == False` in Python, so
    # `==` would let an int back in through the door this function exists to close.
    if value is True:
        return True
    if value is False:
        return False
    state.warnings.append(
        f"action parameter {name!r} is not a boolean (got {type(value).__name__} {value!r}); "
        "the permission was NOT granted"
    )
    logger.warning("action parameter %r is not a boolean (%r); permission refused", name, value)
    return False


def _load_corpus(
    workspace_root: Path,
    *,
    allow_unauthenticated_legacy_roots: bool = False,
) -> tuple[list[tuple[StoredArtifact, ExtractedText, str]], dict[str, CorpusReadOutcome]]:
    """Every held artifact paired with its extracted text and the extraction it was
    read from, plus the typed outcome EVERY held artifact was actually classified
    under -- read or not.

    The extraction identity is the address of whatever was actually served: a nested
    extraction record's ``extraction_sha256`` when one was read, and
    :data:`ROOT_EXTRACTION_ID` when the root ``extracted.json`` sidecar was. A record is
    preferred whenever exactly one is current for today's extractor identity and it
    authenticates.

    That preference is UNIFORM, not a fallback for a root that fails to authenticate.
    Were it a fallback, deleting ``extracted_sha256`` from a modern root would silently
    switch which text is served, and nothing on disk distinguishes "legacy root" from
    "field just deleted" -- so the deletion would PROMOTE. Preferring the record
    unconditionally makes deletion incapable of changing which path is taken.

    This function reports what it actually read, not what it could have read.

    The outcomes mapping is RETURNED rather than a bare skipped-shas list dropped
    (spar round 7, P2, now generalised). A pass that silently reads 6 of 8 held
    papers and reports nothing looks exactly like a pass that read all 8 and found
    nothing, and the two call for opposite responses -- fix the corrupt sidecar, or
    accept that the corpus is exhausted. Coverage the operator cannot see is coverage
    they will assume. Every artifact this function considered, read or not, gets
    exactly one :class:`CorpusReadOutcome` entry: one mechanism, total coverage,
    rather than a "skipped" list that only ever named the failures.

    Every artifact is VERIFIED against its digest before it is read (spar round 8). The
    search pass fetches and extracts in the same breath, so its text is necessarily
    fresh; a corpus pass re-reads sidecars of arbitrary age, which is the one place
    where "the gate runs against content-addressed bytes" can quietly stop being true.
    :func:`verify_artifact` re-hashes ``raw.bin`` and refuses the artifact if the stored
    bytes no longer match the directory naming them.

    Two tiers of verification are distinguished rather than one boolean, because
    ``deep=True`` verification requires a ``derivation_binding`` that no artifact
    stored before that field existed carries -- which describes every document in a
    long-lived real corpus, not a corrupted one. ``SELF_CONSISTENT_METADATA`` passed the strict
    check; ``SIDECAR_DIGEST_ONLY`` only ever had its raw bytes and (if present) its
    sidecar digest checked, with no binding tying the two together;
    ``UNAUTHENTICATED_LEGACY_ROOT`` never had even its sidecar digest recorded, so
    nothing at all authenticates the text it would serve. Only the first two are read
    by default; the third is read only when ``allow_unauthenticated_legacy_roots=True``.
    ``INTEGRITY_FAILED`` -- the default check itself failing -- is never read,
    regardless of that flag: the opt-in exists for artifacts that are merely
    unauthenticated, not for ones whose bytes are actually damaged.

    What deep verification does NOT prove, stated plainly rather than left implied:
    ``extracted.json`` is a DERIVED cache, and verifying ``raw.bin`` does not
    establish that the cache was derived from those bytes. Someone able to rewrite the
    sidecar in place could still present text that the raw bytes do not contain.
    Closing that would mean re-extracting every document on every pass, and it buys
    little here -- anyone who can write into the evidence store can equally rewrite
    the report, so this is not a privilege boundary. The realistic failure this DOES
    catch is the non-adversarial one: bytes truncated by a full disk or an
    interrupted write.

    Args:
        workspace_root: Root workspace.
        allow_unauthenticated_legacy_roots: Opt-in to read artifacts classified
            ``UNAUTHENTICATED_LEGACY_ROOT``. Defaults to False (fail-closed): an
            unauthenticated root sidecar is refused unless the operator explicitly
            says so.
    """
    corpus: list[tuple[StoredArtifact, ExtractedText, str]] = []
    outcomes: dict[str, CorpusReadOutcome] = {}
    # An artifact directory with no readable meta.json never becomes a StoredArtifact,
    # so it cannot be classified by the loop below -- it would vanish from the corpus
    # AND from the coverage this function reports. Seed the outcomes with those (F11).
    artifacts, unreadable = list_artifacts_with_unreadable(workspace_root)
    for sha256 in unreadable:
        outcomes[sha256] = CorpusReadOutcome.UNREADABLE_META
    for artifact in artifacts:
        try:
            shallow_intact = verify_artifact(workspace_root, artifact.sha256, deep=False)
        except ValueError:
            shallow_intact = False
        if not shallow_intact:
            logger.warning("evidence store: %s failed digest verification and was not read", artifact.sha256)
            outcomes[artifact.sha256] = CorpusReadOutcome.INTEGRITY_FAILED
            continue

        # Prefer a digest-authenticated extraction record over the root sidecar, for
        # EVERY artifact -- deliberately not as a fallback for a root that fails to
        # authenticate. A fallback keyed on root failure would mean deleting
        # `extracted_sha256` from a modern root silently switches which text is served,
        # and nothing on disk distinguishes "legacy root" from "field just deleted", so
        # the deletion would PROMOTE. Preferring the record unconditionally makes
        # deletion incapable of changing which path is taken.
        #
        # This sits AFTER the shallow check on purpose: an artifact whose raw.bin no
        # longer hashes to its own name is not evidence, whatever records point at it.
        # A record must never launder a corrupt artifact.
        #
        # ONE scan decides this, and it returns a typed result rather than a count or an
        # exception. Both of those shapes previously hid a decision: a count could be
        # WRONG (a corrupt meta.json made a candidate record vanish, turning an ambiguous
        # store into an apparently unambiguous one and PROMOTING a read), and a single
        # exception type meant "not exactly one current record" and "the record failed to
        # authenticate" -- which license opposite decisions -- were distinguishable only
        # by matching message prose.
        #
        # Of everything this scan can establish, exactly ONE (no record was ever stored)
        # may fall through to the root tiers below. Every other route to "nothing to
        # prefer here" is a downgrade wearing a different face, and refuses.
        selection = select_current_extraction(workspace_root, artifact.sha256)
        if selection.kind is CurrentSelectionKind.SELECTED:
            selected = selection.selected
            if selected is None:  # pragma: no cover - CurrentSelection's own invariant
                raise AssertionError("SELECTED selection carried no extraction")
            outcomes[artifact.sha256] = CorpusReadOutcome.EXTRACTION_RECORD_DIGEST_AUTHENTICATED
            corpus.append((artifact, selected.extracted, selected.extraction_id))
            continue
        if selection.kind is not CurrentSelectionKind.NO_RECORDS_STORED:
            logger.warning(
                "evidence store: %s was NOT read from an extraction record (%s). The root sidecar "
                "is deliberately NOT served instead -- every route to 'no usable record' except a "
                "store that never held one is a downgrade, not a licence",
                artifact.sha256,
                selection.detail,
            )
            outcomes[artifact.sha256] = _RECORD_REFUSAL_OUTCOMES[selection.kind]
            continue

        try:
            deep_intact = verify_artifact(workspace_root, artifact.sha256, deep=True)
        except ValueError:
            deep_intact = False
        if deep_intact:
            outcome = CorpusReadOutcome.SELF_CONSISTENT_METADATA
        elif artifact.extracted_sha256 is not None:
            outcome = CorpusReadOutcome.SIDECAR_DIGEST_ONLY
        else:
            outcome = CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

        if outcome == CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT and not allow_unauthenticated_legacy_roots:
            logger.warning(
                "evidence store: %s has no derivation binding AND no extracted_sha256, "
                "so it cannot be authenticated at all -- was NOT read. Re-extract it into "
                "an authenticated record, or re-run this pass with the explicit opt-in "
                "to read unauthenticated legacy roots",
                artifact.sha256,
            )
            outcomes[artifact.sha256] = outcome
            continue

        extracted = load_artifact_text(workspace_root, artifact.sha256)
        if extracted is None:
            outcomes[artifact.sha256] = CorpusReadOutcome.MISSING_TEXT
        else:
            outcomes[artifact.sha256] = outcome
            corpus.append((artifact, extracted, ROOT_EXTRACTION_ID))
    return corpus, outcomes


#: Characters per token, used to size a budget reservation from a prompt.
#:
#: Deliberately conservative (real English is ~4 chars/token, and the technical text and
#: chemical formulae in these prompts tokenise WORSE than English, not better). Under-
#: reserving is the dangerous direction: the ledger checks the reservation BEFORE the
#: call, so a reservation smaller than the true cost lets a run sail past the operator's
#: ceiling and only discover it when settling, after the money is spent.
_CHARS_PER_TOKEN = 3
#: Headroom for the model's own response, which the prompt length cannot predict.
_RESPONSE_TOKEN_ALLOWANCE = 4000


def estimated_tokens_for(prompt: str) -> int:
    """Size a budget reservation from the prompt actually being sent.

    The bridge defaults to a flat 8000-token reservation, which was fine for a short
    search prompt and badly wrong for a corpus prompt that embeds an entire paper: a
    single document ran ~25k tokens on the live corpus, so the reservation understated
    the real call by roughly three times. That matters more now that tokens are the
    operator's authorisation unit, because the reservation IS the enforcement point --
    an understated one means the cap does not bind until after the call that breached
    it.

    Args:
        prompt: The prompt about to be sent.

    Returns:
        A conservative worst-case token estimate, never below the bridge default.
    """
    return max(8000, len(prompt) // _CHARS_PER_TOKEN + _RESPONSE_TOKEN_ALLOWANCE)


def _corpus_prompt(campaign: Campaign, artifact: StoredArtifact, extracted: ExtractedText) -> str:
    """Present ONE held document to the agent, whole.

    Takes a single document rather than a sequence, and the signature is the point.
    It previously accepted the whole corpus, which invited exactly the call this
    design exists to avoid: handing all 8 papers of the live syngas campaign over in
    one 116k-token prompt produced ZERO proposed findings, where the same papers one
    at a time produced findings. A ``Sequence`` parameter with one caller passing a
    one-element list documents an option that measurement has already refuted.

    The document is included in full rather than summarized or chunked. A finding
    requires a quote copied character-for-character out of the source, so anything
    the agent is not shown verbatim, it cannot ground a finding in -- and a summary
    would actively invite paraphrase, which the gate then rejects.
    """
    corpus = [(artifact, extracted)]
    blocks = []
    for artifact, extracted in corpus:
        blocks.append(
            f"--- DOCUMENT {artifact.sha256} "
            f"(provenance: {artifact.provenance.value}, source: {artifact.source_url}) ---\n"
            f"{extracted.text}\n--- END DOCUMENT {artifact.sha256} ---"
        )
    documents = "\n\n".join(blocks)
    return (
        f"{_campaign_context(campaign)}\n\n"
        f"You hold {len(corpus)} document(s). Extract every finding relevant to this "
        f"campaign that is supported by a verbatim quote from one of them, and set "
        f"`artifact_sha256` to the digest in that document's header.\n\n"
        f"{documents}"
    )


def _corpus_loop(
    workspace_root: Path,
    campaign: Campaign,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    log_path: Path,
    action_id: str,
    run_id: str,
    already_covered: frozenset[tuple[str, str]] = frozenset(),
    allow_unauthenticated_legacy_roots: bool = False,
) -> None:
    """The corpus-only pass: read what is held, propose, ground, verify.

    Documents in ``already_covered`` are SKIPPED. Without that, every pass re-read
    the whole store: five passes over a corpus growing 8->40 papers is 120 document
    reads to mine 40 papers, and the prompt for a given document is byte-identical
    between passes (it carries no prior findings and no pass number), so the repeat
    asks the same question and pays again for the same answer.

    Cost is the smaller half. Because the corpus is presented in a deliberately
    stable order and a pass stops when the token budget runs out, an unscoped pass
    always re-read the same prefix and stopped -- so under a fixed budget, papers
    acquired later were never reached at all, and the operator saw a completed pass
    rather than an unread tail (F14).

    ONE MODEL CALL PER DOCUMENT, not one call holding the whole corpus. This is not a
    stylistic choice -- it was measured. Handing the model all 8 papers of the live
    syngas campaign in a single 116k-token prompt produced ZERO proposed findings;
    handing it one of those same papers alone produced two, from the same prompt and
    the same model. A corpus large enough to be worth re-reading is large enough to
    bury the instruction to quote from it.

    Per-document calls also make partial work survivable: exhausting the budget on
    document six keeps the findings from documents one to five, where a single call
    would have lost everything.

    No repeated rounds per document. The search loop iterates because search results
    are only fed back on the NEXT round, so a finding is impossible in round one;
    here the document is in front of the agent immediately, and a second round would
    re-read identical input at full cost.
    """
    corpus, outcomes = _load_corpus(
        workspace_root, allow_unauthenticated_legacy_roots=allow_unauthenticated_legacy_roots
    )
    read_shas = {artifact.sha256 for artifact, _, _ in corpus}
    skipped = [sha256 for sha256 in outcomes if sha256 not in read_shas]
    # Grouped by REASON, not merely counted. Flattening every unread class back into
    # one list here would reproduce, one layer up, the exact conflation the typed
    # outcomes exist to remove -- and this is the only version a human ever sees. The
    # classes call for opposite responses: an unauthenticated legacy root wants
    # re-extraction or the explicit opt-in, damaged bytes want the paper re-acquired,
    # and an unreadable meta.json wants the directory itself looked at.
    unread_by_reason: dict[str, list[str]] = {}
    for sha256 in skipped:
        unread_by_reason.setdefault(outcomes[sha256].value, []).append(sha256)
    if already_covered:
        before = len(corpus)
        corpus = [(a, e, x) for a, e, x in corpus if (a.sha256, x) not in already_covered]
        n_skipped = before - len(corpus)
        if n_skipped:
            state.warnings.append(
                f"{n_skipped} document(s) already mined by an earlier pass were not re-read; "
                f"{len(corpus)} new document(s) in this pass"
            )
            append_typed_event(
                log_path,
                event="literature.corpus_already_covered",
                action_id=action_id,
                run_id=run_id,
                payload={"n_skipped": n_skipped, "n_new": len(corpus)},
            )
    if skipped:
        state.warnings.append(
            f"{len(skipped)} held artifact(s) were NOT covered by this pass -- "
            + "; ".join(
                f"{reason}: " + ", ".join(sha[:12] for sha in shas) for reason, shas in sorted(unread_by_reason.items())
            )
        )
        append_typed_event(
            log_path,
            event="literature.corpus_artifacts_unreadable",
            action_id=action_id,
            run_id=run_id,
            # `sha256` is kept as it was so nothing reading this event breaks, and the
            # per-reason grouping is ADDED beside it rather than replacing it.
            payload={"sha256": skipped, "n_skipped": len(skipped), "unread_by_reason": unread_by_reason},
        )
    # Which text was served is not a detail: a document read from a record is a
    # DIFFERENT document from the same raw bytes read through the root sidecar, and it
    # is quoted under a different extraction id. Saying so here means the operator sees
    # it while the pass runs, instead of only by reading the stored report afterwards.
    from_records = sorted(
        (artifact.sha256, extraction_id) for artifact, _, extraction_id in corpus if extraction_id != ROOT_EXTRACTION_ID
    )
    if from_records:
        state.warnings.append(
            f"{len(from_records)} document(s) were read from an authenticated extraction record "
            "rather than the root sidecar -- "
            + ", ".join(f"{sha[:12]} via {extraction_id[:12]}" for sha, extraction_id in from_records)
        )
        append_typed_event(
            log_path,
            event="literature.corpus_read_from_extraction_record",
            action_id=action_id,
            run_id=run_id,
            payload={
                "n_from_records": len(from_records),
                "read_from_records": [
                    {"sha256": sha, "extraction_id": extraction_id} for sha, extraction_id in from_records
                ],
            },
        )
    if not corpus:
        state.stop_reason = StopReason.NO_NEW_INFORMATION
        # Distinguish the two, because they call for opposite responses: acquire
        # papers, versus stop paying for passes over a corpus already mined.
        state.warnings.append(
            "every held document has already been mined by an earlier pass; nothing new to read"
            if already_covered
            else "the evidence store holds no readable artifacts, so there was nothing to read"
        )
        return

    agent = build_corpus_agent(model=deps.model, ledger=deps.ledger)
    for artifact, extracted, extraction_id in corpus:
        deps.ledger.check_wall_clock()
        # Only the document actually shown is resolvable. The agent is looking at one
        # paper, so a digest naming any other is a mistake worth surfacing, even when
        # that other paper happens to be in the store.
        by_sha = {artifact.sha256: (artifact, extracted, extraction_id)}
        corpus_prompt = _corpus_prompt(campaign, artifact, extracted)
        result = agent.run(corpus_prompt, estimated_tokens=estimated_tokens_for(corpus_prompt))
        # Recorded once the call has RETURNED, which is the moment the tokens are
        # definitely spent -- not before it is attempted. Both halves matter:
        #
        #   * After the call, so a document whose reservation the ledger REFUSED is
        #     not marked covered. Recording it first meant a budget-truncated pass
        #     silently dropped the document it stopped on: nothing was paid for it,
        #     it was never read, and every later pass skipped it while reporting
        #     "every held document has already been mined". Observed live
        #     2026.08.01 -- 8 documents recorded as covered by 7 model calls.
        #   * Before the proposal is processed, so a document that WAS paid for and
        #     then failed downstream (grounding, validation) is not re-bought. That
        #     is the case the original ordering was reaching for, and it is kept.
        state.covered.append(
            CoveredDocument(
                raw_sha256=artifact.sha256,
                extraction_id=extraction_id,
                verification_standard=outcomes[artifact.sha256].value,
            )
        )
        proposal = CorpusProposal.model_validate(result.output)
        append_typed_event(
            log_path,
            event="literature.corpus_pass_proposed",
            action_id=action_id,
            run_id=run_id,
            payload={"sha256": artifact.sha256, "n_proposed": len(proposal.findings)},
        )
        _process_corpus_proposal(
            workspace_root,
            proposal,
            deps,
            config=config,
            state=state,
            by_sha=by_sha,
            log_path=log_path,
            action_id=action_id,
            run_id=run_id,
        )


def _process_corpus_proposal(
    workspace_root: Path,
    proposal: CorpusProposal,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    state: _RunState,
    by_sha: dict[str, tuple[StoredArtifact, ExtractedText, str]],
    log_path: Path,
    action_id: str,
    run_id: str,
) -> None:
    """Ground every finding one document's proposal claimed."""
    seen: set[tuple[str, str]] = set()
    for proposed in proposal.findings:
        resolved = by_sha.get(proposed.artifact_sha256)
        if resolved is None:
            # The agent named a document that is not in the corpus it was shown. The
            # sha256 pattern already rejects a malformed handle, so this is a
            # well-formed digest for something Carmel does not hold -- recorded as a
            # rejection rather than dropped, because a reader of the report must be
            # able to see that the agent claimed evidence which does not exist.
            state.rejected.append(
                RejectedFinding(
                    finding_id=uuid.uuid4().hex,
                    run_id=run_id,
                    action_id=action_id,
                    category=proposed.payload.category,
                    citation_title=proposed.citation.title,
                    grounding=GroundingVerdict(
                        status=GroundingStatus.NO_ARTIFACT,
                        grounded=False,
                        match_ratio=0.0,
                        reasons=[_UNHELD_ARTIFACT_REASON],
                    ),
                    reason=f"{_UNHELD_ARTIFACT_REASON}: {proposed.artifact_sha256}",
                )
            )
            continue

        artifact, extracted, extraction_id = resolved
        key = (proposed.artifact_sha256, proposed.verbatim_quote.strip()[:200])
        if key in seen:
            continue
        seen.add(key)

        _ground_and_record(
            workspace_root,
            ProposedFinding(
                payload=proposed.payload,
                citation=proposed.citation,
                verbatim_quote=proposed.verbatim_quote,
                source_url=artifact.source_url,
            ),
            deps,
            config=config,
            state=state,
            stored=artifact,
            extracted=extracted,
            log_path=log_path,
            action_id=action_id,
            run_id=run_id,
            queue_acquisition=False,
            extraction_id=extraction_id,
        )


def _collect_manual_acquisitions(
    workspace_root: Path,
    *,
    config: AgentConfig,
    state: _RunState,
    log_path: Path,
    action_id: str,
    run_id: str,
) -> None:
    """Admit any papers a human dropped in the inbox since the last run.

    Runs BEFORE the research loop and inside the run lock, so the manifest cannot be
    mutated by two runs at once and the report reflects the workspace as it is at the
    moment the run starts. Every admission is identity-checked in
    :func:`~carmel.services.acquisition.collect_inbox`: a file whose text does not match
    the request it was dropped against is REJECTED rather than stored, because a
    mis-filed PDF would otherwise attach one paper's bytes to another paper's citation
    and quietly corrupt the evidence chain.

    Admission makes the document available in the evidence store with
    :attr:`ArtifactProvenance.MANUAL`. It does not, by itself, extract any finding
    from that document: grounding runs against the corpus in a separate pass
    (:func:`run_corpus_pass`), which an operator appends deliberately. The
    operator-visible warning below therefore says only that the paper was admitted,
    which is all that has happened at this point.

    A failure here must never take down the run: an unreadable inbox is an operator
    problem to be reported, not a reason to lose a literature search.
    """
    try:
        admitted = collect_inbox(workspace_root, max_bytes=config.budget.max_artifact_bytes)
    except (OSError, ValueError, ValidationError) as exc:
        state.warnings.append(f"could not read the manual-acquisition inbox: {exc}")
        logger.warning("literature run %s could not collect the acquisition inbox: %s", run_id, exc)
        return

    if not admitted:
        return

    slugs = [request.slug for request in admitted]
    state.warnings.append(
        f"admitted {len(slugs)} manually-supplied paper(s) into the evidence store: {', '.join(slugs)}"
    )
    append_typed_event(
        log_path,
        event="literature.manual_acquisitions_admitted",
        action_id=action_id,
        run_id=run_id,
        payload={"slugs": slugs},
    )


def _report_with_pass(
    previous: LiteratureReport | None,
    *,
    pass_record: PassRecord,
    state: _RunState,
    report_id: str,
    campaign_id: str,
    created_at: datetime,
) -> LiteratureReport:
    """Fold one finished pass into the campaign's accumulated literature report.

    Artifacts are deduplicated by ``sha256`` because the evidence store is
    content-addressed: the same paper re-encountered in a later pass is the same
    bytes, and listing it twice would overstate how much evidence the campaign
    holds. Findings and rejections are NOT deduplicated -- two passes reaching the
    same conclusion independently is a real signal, and each carries its own
    ``run_id`` so a reader can see that is what happened.
    """
    if previous is None:
        return LiteratureReport(
            report_id=report_id,
            campaign_id=campaign_id,
            created_at=created_at,
            passes=[pass_record],
            queries=state.query_records(pass_record),
            artifacts=list(state.artifacts),
            findings=list(state.findings),
            rejected=list(state.rejected),
        )

    seen = {artifact.sha256 for artifact in previous.artifacts}
    artifacts = list(previous.artifacts) + [a for a in state.artifacts if a.sha256 not in seen]
    return previous.model_copy(
        update={
            "passes": [*previous.passes, pass_record],
            "queries": [*previous.queries, *state.query_records(pass_record)],
            "artifacts": artifacts,
            "findings": [*previous.findings, *state.findings],
            "rejected": [*previous.rejected, *state.rejected],
        }
    )


def run_literature_research(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
) -> LiteratureReport:
    """Run one complete literature-research action. ALWAYS returns a report.

    Budget exhaustion anywhere is mapped through
    :data:`~carmel.schemas.literature.STOP_REASON_FOR_DIMENSION` onto the report's
    ``stop_reason`` (a partial report, never a crash); bridge/model errors map to
    :attr:`StopReason.ERROR`. A run that grounds nothing is a valid report with zero
    findings.

    Raises:
        LiteratureRunLockedError: If another literature run holds the workspace lock.
    """
    return _run_pass(workspace_root, campaign, action, deps, config=config, mode=LiteraturePassMode.SEARCH)


def run_corpus_pass(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
) -> LiteratureReport:
    """Re-read the papers this workspace already holds and ground findings in them.

    The second pass. Carmel could previously acquire a paper and never use it: manual
    admission stored an identity-checked document and stopped, and the state machine
    has no edge back into a literature run, so a completed action could not re-run.
    This closes that gap.

    Reads the evidence store ONLY. No search, no fetching, no acquisition requests.
    That is a deliberate restriction rather than a missing feature:

    - **Reproducibility.** The input is a fixed set of sha256-addressed files, so a
      reviewer re-running a benchmark gets the same input every time. A live search
      cannot offer that.
    - **Isolating what is under test.** This pass is the first time the deterministic
      grounding gate runs against real downloaded bytes. Adding search would inject
      fresh cost and nondeterminism into the very step being validated.

    It cannot discover a finding that needs a paper not yet acquired. That is
    accepted: discovery is what the search pass and the acquisition queue already do.

    Raises:
        LiteratureRunLockedError: If another literature run holds the workspace lock.
    """
    return _run_pass(workspace_root, campaign, action, deps, config=config, mode=LiteraturePassMode.CORPUS)


def _run_pass(
    workspace_root: Path,
    campaign: Campaign,
    action: PlannedAction,
    deps: LiteratureDeps,
    *,
    config: AgentConfig,
    mode: LiteraturePassMode,
) -> LiteratureReport:
    """Shared scaffolding for one pass of either mode.

    The run lock, the budget-to-stop-reason mapping, the decision-log events and the
    fold into the accumulated report are identical for both, and must stay identical:
    a corpus pass that could bypass the lock, or that mapped budget exhaustion to a
    crash instead of a partial report, would be a second and weaker safety envelope.
    Only the loop in the middle differs.
    """
    workspace_root = Path(workspace_root)
    run_id = uuid.uuid4().hex
    report_id = uuid.uuid4().hex
    created_at = datetime.now(UTC)
    log_path = workspace_root / "decision_log.jsonl"

    lease = _acquire_run_lock(
        workspace_root,
        action_id=action.action_id,
        run_id=run_id,
        stale_after_s=2 * config.budget.max_wall_clock_s,
        log_path=log_path,
    )
    try:
        previous = _load_previous_report(workspace_root)
        state = _RunState(queries=[], artifacts=[], findings=[], rejected=[], warnings=[])
        append_typed_event(
            log_path,
            event="literature.search_started",
            action_id=action.action_id,
            run_id=run_id,
            payload={"campaign_id": campaign.campaign_id, "model_name": deps.model.name, "mode": mode.value},
        )
        _collect_manual_acquisitions(
            workspace_root, config=config, state=state, log_path=log_path, action_id=action.action_id, run_id=run_id
        )

        # What earlier passes already mined. `reread_all` is the operator's escape
        # hatch: coverage is recorded per document, so re-reading is otherwise never
        # automatic, and a changed prompt or a newer model is a real reason to want it.
        reread_all = _permission_from(action.parameters, "reread_all", state)
        # Opt-in to reading a held artifact whose stored text carries no digest
        # binding it to anything (`CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT`).
        # Meaningless outside the corpus path -- a search pass has no held corpus to
        # gate -- so it is read here but only ever threaded through on that branch
        # below, rather than added to `_research_loop`'s signature where it would
        # silently do nothing.
        allow_unauthenticated_legacy_roots = _permission_from(
            action.parameters, "allow_unauthenticated_legacy_roots", state
        )
        already_covered: frozenset[tuple[str, str]] = frozenset()
        if mode == LiteraturePassMode.CORPUS and previous is not None and not reread_all:
            already_covered = frozenset(
                (cd.raw_sha256, cd.extraction_id) for record in previous.passes for cd in record.covered
            )

        slot_acquired = False
        try:
            session_budget().acquire_run_slot(config.budget.max_concurrent_runs)
            slot_acquired = True
            if mode == LiteraturePassMode.CORPUS:
                _corpus_loop(
                    workspace_root,
                    campaign,
                    deps,
                    config=config,
                    state=state,
                    log_path=log_path,
                    action_id=action.action_id,
                    run_id=run_id,
                    already_covered=already_covered,
                    allow_unauthenticated_legacy_roots=allow_unauthenticated_legacy_roots,
                )
            else:
                _research_loop(
                    workspace_root,
                    campaign,
                    deps,
                    config=config,
                    state=state,
                    log_path=log_path,
                    action_id=action.action_id,
                    run_id=run_id,
                    already_covered=already_covered,
                )
        except BudgetExceededError as exc:
            state.stop_reason = STOP_REASON_FOR_DIMENSION[exc.dimension]
            state.warnings.append(f"budget exceeded: {exc}")
            logger.warning("literature run %s stopped on budget: %s", run_id, exc)
        except AgentBridgeError as exc:
            state.stop_reason = StopReason.ERROR
            state.warnings.append(f"agent error: {exc}")
            logger.warning("literature run %s stopped on agent error: %s", run_id, exc)
        except SearchError as exc:
            # P1-12: a search backend transport failure (503, DNS blip, timeout)
            # describes this one round's queries, not the campaign -- everything
            # ``state`` already accumulated in EARLIER completed rounds (findings,
            # artifacts, queries, acquisitions) is real, paid-for work and must still
            # be reported. Mirrors the ``AgentBridgeError`` handling immediately
            # above: stop the run with a typed reason rather than losing the whole
            # report to an exception that used to propagate past every handler here.
            state.stop_reason = StopReason.ERROR
            state.warnings.append(f"search failed: {exc}")
            logger.warning("literature run %s stopped on search error: %s", run_id, exc)
        finally:
            if slot_acquired:
                session_budget().release_run_slot()

        report = _report_with_pass(
            previous,
            pass_record=PassRecord(
                run_id=run_id,
                action_id=action.action_id,
                created_at=created_at,
                mode=mode,
                model_name=deps.model.name,
                stop_reason=state.stop_reason,
                usage=deps.ledger.usage(),
                warnings=state.warnings,
                covered=list(state.covered),
            ),
            state=state,
            report_id=report_id,
            campaign_id=campaign.campaign_id,
            created_at=created_at,
        )
        save_literature_report(workspace_root, report)
        append_typed_event(
            log_path,
            event="literature.search_finished",
            action_id=action.action_id,
            run_id=run_id,
            payload={
                "stop_reason": state.stop_reason.value,
                "n_findings": len(state.findings),
                "n_rejected": len(state.rejected),
            },
        )
        # Counts describe THIS run, not the campaign's accumulated total. The report
        # now spans every pass, so reading them off ``report`` would silently restate
        # earlier passes' work as this one's each time a new pass is appended.
        grounding_summary: dict[str, int] = {}
        for status in [f.grounding.status.value for f in state.findings] + [
            r.grounding.status.value for r in state.rejected
        ]:
            grounding_summary[status] = grounding_summary.get(status, 0) + 1
        record_agent_provenance(
            workspace_root,
            "literature_run",
            {
                "action_id": action.action_id,
                "run_id": run_id,
                "campaign_id": campaign.campaign_id,
                "report_id": report_id,
                "model_name": deps.model.name,
                "provider": config.provider.value,
                "tier": config.tier.value,
                "queries": list(state.queries),
                "artifacts": [a.sha256 for a in state.artifacts],
                "usage": deps.ledger.usage().model_dump(mode="json"),
                "stop_reason": state.stop_reason.value,
                "n_findings": len(state.findings),
                "n_rejected": len(state.rejected),
                "grounding_summary": grounding_summary,
                "created_at": created_at.isoformat(),
            },
        )
        return report
    finally:
        # Ownership-checked release (P1-7): only removes the lock dir if it still
        # names THIS process's pid/pid_start, never a successor's lock this frame
        # does not actually hold.
        lease.release()


def run_record_for(report: LiteratureReport, action: PlannedAction) -> RunRecord:
    """Build the :class:`RunRecord` for a finished literature run.

    Budget stop reasons map to ``FAILED``/``BUDGET_EXCEEDED``; ``ERROR`` maps to
    ``FAILED``/``AGENT_ERROR``; everything else is ``SUCCEEDED``/``NONE``. The
    dispatcher persists the record.
    """
    if report.stop_reason in _BUDGET_STOP_REASONS:
        status, failure_code = RunStatus.FAILED, FailureCode.BUDGET_EXCEEDED
    elif report.stop_reason == StopReason.ERROR:
        status, failure_code = RunStatus.FAILED, FailureCode.AGENT_ERROR
    else:
        status, failure_code = RunStatus.SUCCEEDED, FailureCode.NONE
    return RunRecord(
        run_id=report.run_id,
        action_id=action.action_id,
        tool_name="literature_agent",
        status=status,
        failure_code=failure_code,
        started_at=report.created_at,
        ended_at=datetime.now(UTC),
        submission_mode=SubmissionMode.LOCAL,
    )
