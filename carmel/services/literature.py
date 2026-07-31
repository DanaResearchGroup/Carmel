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
from urllib.parse import urlsplit

from pydantic import ValidationError

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
from carmel.agents.tools.academic import (
    CrossrefSearchTool,
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
    STOP_REASON_FOR_DIMENSION,
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
from carmel.services.acquisition import collect_inbox, record_request
from carmel.services.artifacts import read_json, write_json
from carmel.services.decision_log import append_typed_event
from carmel.services.evidence import store_artifact
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


def migrate_report_payload(payload: object) -> object:
    """Bring a persisted report payload up to the current schema version.

    v1 held exactly one run, with that run's identifiers and outcome at the top
    level. v2 accumulates passes, so the migration lifts those top-level fields into
    a single :class:`PassRecord` and stamps every finding, rejection and query with
    that run's identity -- the attribution was always true of a v1 report, it simply
    had nowhere to be written down.

    A live campaign already holds a v1 report on disk (the first real run), and it
    contains the only existing evidence that the grounding gate refuses ungrounded
    claims. Migrating rather than discarding is therefore not a courtesy to old
    files; it preserves the demonstration.
    """
    if not isinstance(payload, dict):
        return payload
    if int(payload.get("schema_version", 1)) >= 2:
        return payload

    migrated = dict(payload)
    run_id = str(migrated.pop("run_id", "") or "")
    action_id = str(migrated.pop("action_id", "") or "")
    created_at = migrated.get("created_at")
    pass_record: dict[str, object] = {
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
    migrated["schema_version"] = 2
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
            run_id=run_id,
            action_id=action_id,
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
) -> tuple[list[str], str, bool]:
    """Deterministically resolve a wanted paper's OA candidates, with an honest note.

    Args:
        paper_doi: The paper's normalized DOI, if it has one.
        paper_title: The paper's title, passed through to the resolver's
            title-matched providers (ChemRxiv, arXiv); those skip themselves when it
            is unavailable.
        deps: Injected dependencies (the resolver in particular).

    Returns:
        ``(candidates, note, complete)``: candidate URLs capped at
        :data:`MAX_OA_FETCH_ATTEMPTS_PER_PAPER`, a note describing what resolution did
        (or why it could not run) for the operator-facing ``detail``, and whether every
        enabled provider actually answered. ``complete=False`` means an empty
        ``candidates`` establishes nothing -- see
        :attr:`~carmel.agents.tools.academic.OaResolution.complete`.

        The two early returns below are ``complete=True`` deliberately: "this paper has
        no DOI" and "no resolver is configured" are fully-established facts about why
        resolution did not run, not truncated attempts.
    """
    if paper_doi is None:
        return [], "paper has no DOI, so automated open-access resolution was not attempted", True
    if deps.oa_resolver is None:
        return [], "no open-access resolver is configured for this run", True
    resolution = deps.oa_resolver.resolve(paper_doi, title=paper_title)
    candidates = list(resolution.candidates)
    note = resolution.note
    if len(candidates) > MAX_OA_FETCH_ATTEMPTS_PER_PAPER:
        note = f"{note}; trying the first {MAX_OA_FETCH_ATTEMPTS_PER_PAPER} of {len(candidates)} candidates"
        candidates = candidates[:MAX_OA_FETCH_ATTEMPTS_PER_PAPER]
    return candidates, note, resolution.complete


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

    candidates, resolution_note, resolution_complete = _resolve_oa_candidates(doi, paper.title or None, deps)

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
        # provider failing in transit) nothing has been established, so say exactly that.
        reason = (
            AcquisitionReason.NO_OPEN_ACCESS_COPY if resolution_complete else AcquisitionReason.OA_LOOKUP_INCOMPLETE
        )
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

    NOTE: admission makes the document available in the evidence store with
    :attr:`ArtifactProvenance.MANUAL`; it does not yet re-run grounding to extract
    findings from it. That further step is not wired, and the operator-visible warning
    below deliberately says only what actually happened.

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
            payload={"campaign_id": campaign.campaign_id, "model_name": deps.model.name},
        )
        _collect_manual_acquisitions(
            workspace_root, config=config, state=state, log_path=log_path, action_id=action.action_id, run_id=run_id
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
                mode=LiteraturePassMode.SEARCH,
                model_name=deps.model.name,
                stop_reason=state.stop_reason,
                usage=deps.ledger.usage(),
                warnings=state.warnings,
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
