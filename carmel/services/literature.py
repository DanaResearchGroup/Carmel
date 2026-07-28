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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carmel.agents.bridge import AgentBridgeError, ModelProtocol
from carmel.agents.budget import (
    BudgetExceededError,
    BudgetLedger,
    session_budget,
)
from carmel.agents.literature_agent import (
    LiteratureProposal,
    ProposedFinding,
    RequestedPaper,
    VerifierAssessment,
    build_literature_agent,
    build_verifier_agent,
)
from carmel.agents.models import build_model
from carmel.agents.tools.academic import CrossrefSearchTool, OpenAlexSearchTool, normalize_doi
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
    STOP_REASON_FOR_DIMENSION,
    CredenceVerdict,
    EvidenceRef,
    GroundingStatus,
    GroundingVerdict,
    LiteratureFinding,
    LiteratureReport,
    RejectedFinding,
    SpeciesRef,
    StopReason,
    StoredArtifact,
)
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import FailureCode, RunRecord, RunStatus, SubmissionMode
from carmel.services import chem
from carmel.services.acquisition import record_request
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_typed_event
from carmel.services.evidence import EVIDENCE_LITERATURE_DIR, store_artifact
from carmel.services.grounding import find_quote, ground_finding
from carmel.services.provenance import record_agent_provenance

logger = get_logger("services.literature")

LITERATURE_REPORT_NAME = "literature_report.json"
RUN_LOCK_DIR_NAME = ".run.lock"
LOCK_INFO_NAME = "info.json"

#: Chars of extracted text shown to the Verifier on each side of the located quote.
VERIFIER_EVIDENCE_WINDOW = 600

#: Credence ceiling applied when any species failed canonicalization.
NON_CANONICAL_CREDENCE_CAP = 0.7

#: Max queries actually executed (and recorded in ``report.queries``/provenance) per
#: Literature Agent round. Anything beyond this is dropped -- loudly, via a warning
#: and a decision-log event -- rather than recorded as if it had run (Finding 6).
MAX_QUERIES_PER_ROUND = 5

#: Search results requested per executed query.
SEARCH_RESULTS_PER_QUERY = 5

#: Grace period during which a run-lock directory with missing/unparseable
#: ``info.json`` is still treated as LIVE -- a peer may be between ``mkdir()`` and
#: its metadata write. Mirrors ``carmel.services.plan_progress.DEFAULT_LOCK_GRACE_S``.
LOCK_GRACE_S = 30.0

#: Stop reasons that mean "a budget ceiling ended the run".
_BUDGET_STOP_REASONS = frozenset(STOP_REASON_FOR_DIMENSION.values())

__all__ = [
    "LITERATURE_REPORT_NAME",
    "LiteratureDeps",
    "LiteratureRunLockedError",
    "build_deps",
    "load_literature_report",
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
    return LiteratureReport.model_validate(read_json(workspace_root / LITERATURE_REPORT_NAME))


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
        fetch=HttpFetchTool(ledger=ledger, max_artifact_bytes=config.budget.max_artifact_bytes),
        ledger=ledger,
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
        return OpenAlexSearchTool(ledger=ledger, contact_email=config.search_contact_email)
    if config.search_provider == SearchProvider.CROSSREF:
        return CrossrefSearchTool(ledger=ledger, contact_email=config.search_contact_email)
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
    return HttpSearchTool(endpoint=config.search_endpoint, api_key=api_key, ledger=ledger)


# --------------------------- run lock -----------------------------------------


def _read_lock_info(lock_dir: Path) -> dict[str, Any]:
    """Best-effort read of the lock's ``info.json``; empty dict when unreadable."""
    try:
        parsed = json.loads((lock_dir / LOCK_INFO_NAME).read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lock_dir_age_s(lock_dir: Path) -> float:
    """Seconds since the lock directory's mtime; +inf when it vanished."""
    try:
        mtime = lock_dir.stat().st_mtime
    except OSError:
        return float("inf")
    return max(0.0, datetime.now(UTC).timestamp() - mtime)


def _lock_is_stale(
    lock_dir: Path, info: dict[str, Any], *, stale_after_s: float, lock_grace_s: float = LOCK_GRACE_S
) -> tuple[bool, str]:
    """Decide whether an existing lock may be broken.

    A lock is stale when (a) it was taken on THIS host and its pid is dead, (b) it is
    older than ``stale_after_s``, or (c) its ``info.json`` is missing/unparseable AND
    the lock directory itself is older than ``lock_grace_s``. A lock whose pid is
    verifiably ALIVE on this host is NEVER stale, regardless of age.

    Case (c) matters because ``mkdir()`` (claiming the lock) and the ``info.json``
    write (publishing who holds it) are two separate steps: a crash in between used
    to leave a lock directory with no metadata that was PERMANENTLY non-stale, since
    neither the pid branch nor the ``started_at`` branch had anything to check
    (Finding 4) -- only ``rm -rf`` could recover. Within ``lock_grace_s`` of the lock
    dir's own mtime, missing metadata still fails CLOSED (not stale), since a peer may
    simply be mid-publication; only past the grace period is it treated as abandoned.
    """
    pid = info.get("pid")
    hostname = info.get("hostname")
    if hostname == socket.gethostname() and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True, f"pid {pid} on this host is dead"
        except PermissionError:
            return False, ""  # pid exists (owned by another user): live lock.
        else:
            return False, ""  # live pid on this host: never break.

    started_at: datetime | None = None
    started_at_raw = info.get("started_at")
    if isinstance(started_at_raw, str):
        try:
            started_at = datetime.fromisoformat(started_at_raw)
        except ValueError:
            started_at = None
    if started_at is not None:
        age_s = (datetime.now(UTC) - started_at).total_seconds()
        if age_s > stale_after_s:
            return True, f"lock is {age_s:.0f}s old (> stale_after_s={stale_after_s:.0f}s)"
        return False, ""

    lock_age_s = _lock_dir_age_s(lock_dir)
    if lock_age_s > lock_grace_s:
        return True, (
            f"lock metadata missing/unparseable and lock dir is {lock_age_s:.0f}s old "
            f"(> lock_grace_s={lock_grace_s:.0f}s)"
        )
    return False, ""


def _acquire_run_lock(
    workspace_root: Path,
    *,
    action_id: str,
    run_id: str,
    stale_after_s: float,
    log_path: Path,
) -> Path:
    """Atomically acquire ``evidence/literature/.run.lock``.

    Raises:
        LiteratureRunLockedError: If a live (non-stale) lock is already held.
    """
    lock_dir = workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        info = _read_lock_info(lock_dir)
        stale, reason = _lock_is_stale(lock_dir, info, stale_after_s=stale_after_s)
        if not stale:
            raise LiteratureRunLockedError(
                f"a literature run already holds {lock_dir} (pid={info.get('pid')}, hostname={info.get('hostname')})"
            ) from None
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
        shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            raise LiteratureRunLockedError(
                f"lost the race re-acquiring {lock_dir} after breaking a stale lock"
            ) from None

    (lock_dir / LOCK_INFO_NAME).write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": datetime.now(UTC).isoformat(),
                "action_id": action_id,
            }
        ),
        encoding="utf-8",
    )
    return lock_dir


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
    finding_id = uuid.uuid4().hex
    url = proposed.source_url

    if url not in artifact_cache:
        artifact_cache[url] = _fetch_and_store(workspace_root, url, deps, config=config, state=state)
    cached = artifact_cache[url]
    stored, extracted = cached if cached is not None else (None, None)
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
        _maybe_queue_acquisition(workspace_root, proposed, verdict=verdict, state=state)
        state.rejected.append(
            RejectedFinding(
                finding_id=finding_id,
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
    result = verifier.run(
        _verifier_prompt(
            proposed,
            evidence_window=evidence_window,
            grounding_dict=verdict.model_dump(mode="json"),
        )
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
            payload=proposed.payload,
            citation=proposed.citation,
            verbatim_quote=proposed.verbatim_quote,
            evidence=EvidenceRef(
                artifact_sha256=stored.sha256,
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


def _queue_wanted_paper(
    workspace_root: Path,
    paper: RequestedPaper,
    *,
    state: _RunState,
    log_path: Path,
    action_id: str,
    run_id: str,
) -> None:
    """Queue a paper the agent asked a human to obtain.

    Note what is NOT checked here: the agent's claim that this paper is relevant, or even
    that it exists. Nothing in the queue is evidence -- a queued paper makes no assertion
    about combustion chemistry, it only asks a person to look something up, and that
    person can see the title, DOI and stated reason before spending any effort. The
    evidentiary checks all still apply later, unchanged, to whatever they drop in.
    """
    doi = normalize_doi(paper.doi)
    landing_url = paper.landing_url or (f"https://doi.org/{doi}" if doi else None)
    if landing_url is None:
        state.warnings.append(f"ignored a requested paper with neither a DOI nor a URL: {paper.title!r}")
        return

    request = record_request(
        workspace_root,
        title=paper.title,
        doi=doi,
        landing_url=landing_url,
        reason=AcquisitionReason.PAYWALLED,
        detail=paper.relevance,
    )
    if request.slug in state.acquisition_slugs:
        return
    state.acquisition_slugs.append(request.slug)
    append_typed_event(
        log_path,
        event="literature.paper_requested",
        action_id=action_id,
        run_id=run_id,
        payload={"slug": request.slug, "doi": doi, "title": paper.title},
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
    """
    url = proposed.source_url
    failure = state.fetch_failures.get(url)

    if failure is not None:
        acquisition_reason, detail = failure
    elif verdict.status == GroundingStatus.ARTIFACT_UNREADABLE:
        acquisition_reason = AcquisitionReason.UNREADABLE
        detail = "; ".join(verdict.reasons)
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
    """
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
    extracted = extract_text(data, artifact.content_type)
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
) -> None:
    """The bounded propose->ground->verify loop. Mutates ``state`` in place."""
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
        result = literature.run(prompt)
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
                workspace_root, paper, state=state, log_path=log_path, action_id=action_id, run_id=run_id
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
    workspace_root = Path(workspace_root)
    run_id = uuid.uuid4().hex
    report_id = uuid.uuid4().hex
    created_at = datetime.now(UTC)
    log_path = workspace_root / "decision_log.jsonl"

    lock_dir = _acquire_run_lock(
        workspace_root,
        action_id=action.action_id,
        run_id=run_id,
        stale_after_s=2 * config.budget.max_wall_clock_s,
        log_path=log_path,
    )
    try:
        state = _RunState(queries=[], artifacts=[], findings=[], rejected=[], warnings=[])
        append_typed_event(
            log_path,
            event="literature.search_started",
            action_id=action.action_id,
            run_id=run_id,
            payload={"campaign_id": campaign.campaign_id, "model_name": deps.model.name},
        )
        slot_acquired = False
        try:
            session_budget().acquire_run_slot(config.budget.max_concurrent_runs)
            slot_acquired = True
            _research_loop(
                workspace_root,
                campaign,
                deps,
                config=config,
                state=state,
                log_path=log_path,
                action_id=action.action_id,
                run_id=run_id,
            )
        except BudgetExceededError as exc:
            state.stop_reason = STOP_REASON_FOR_DIMENSION[exc.dimension]
            state.warnings.append(f"budget exceeded: {exc}")
            logger.warning("literature run %s stopped on budget: %s", run_id, exc)
        except AgentBridgeError as exc:
            state.stop_reason = StopReason.ERROR
            state.warnings.append(f"agent error: {exc}")
            logger.warning("literature run %s stopped on agent error: %s", run_id, exc)
        finally:
            if slot_acquired:
                session_budget().release_run_slot()

        report = LiteratureReport(
            report_id=report_id,
            campaign_id=campaign.campaign_id,
            action_id=action.action_id,
            run_id=run_id,
            created_at=created_at,
            queries=state.queries,
            artifacts=state.artifacts,
            findings=state.findings,
            rejected=state.rejected,
            stop_reason=state.stop_reason,
            model_name=deps.model.name,
            usage=deps.ledger.usage(),
            warnings=state.warnings,
        )
        save_literature_report(workspace_root, report)
        append_typed_event(
            log_path,
            event="literature.search_finished",
            action_id=action.action_id,
            run_id=run_id,
            payload={
                "stop_reason": report.stop_reason.value,
                "n_findings": len(report.findings),
                "n_rejected": len(report.rejected),
            },
        )
        grounding_summary: dict[str, int] = {}
        for status in [f.grounding.status.value for f in report.findings] + [
            r.grounding.status.value for r in report.rejected
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
                "queries": report.queries,
                "artifacts": [a.sha256 for a in report.artifacts],
                "usage": report.usage.model_dump(mode="json"),
                "stop_reason": report.stop_reason.value,
                "n_findings": len(report.findings),
                "n_rejected": len(report.rejected),
                "grounding_summary": grounding_summary,
                "created_at": created_at.isoformat(),
            },
        )
        return report
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


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
