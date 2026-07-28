"""Integration tests for the Literature Agent + Verifier orchestration."""

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Iterator
from collections.abc import Sequence as AbcSequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import carmel.services.literature
from carmel.agents.bridge import AgentTool, ModelResponse
from carmel.agents.budget import BudgetLedger, session_budget
from carmel.agents.literature_agent import (
    LITERATURE_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    LiteratureProposal,
    VerifierAssessment,
)
from carmel.agents.models import AgentBridgeError, MockModel
from carmel.agents.tools.fetch import FetchError, MockFetchTool
from carmel.agents.tools.search import MockSearchTool, SearchResult
from carmel.config import AgentBudgetConfig, AgentConfig, AgentProvider, ModelTier
from carmel.schemas import (
    ActionKind,
    ApprovalRequirement,
    Budgets,
    CampaignInput,
    FailureCode,
    InitialMixture,
    MixtureComponent,
    PlannedAction,
    ReactorSystem,
    ReactorType,
    RunStatus,
    TargetObservable,
)
from carmel.schemas.acquisition import AcquisitionReason, AcquisitionStatus
from carmel.schemas.campaign import Campaign
from carmel.schemas.literature import GroundingStatus, LiteratureReport, StopReason
from carmel.services import chem
from carmel.services.acquisition import load_manifest
from carmel.services.campaigns import create_campaign
from carmel.services.decision_log import read_events
from carmel.services.evidence import EVIDENCE_LITERATURE_DIR
from carmel.services.literature import (
    LITERATURE_REPORT_NAME,
    LOCK_GRACE_S,
    MAX_QUERIES_PER_ROUND,
    RUN_LOCK_DIR_NAME,
    LiteratureDeps,
    LiteratureRunLockedError,
    build_deps,
    load_literature_report,
    run_literature_research,
    run_record_for,
)

DOI = "10.1000/test.doi"
SOURCE_URL = "https://example.com/papers/secret-paper-url"
QUOTE = "The ignition delay time of O2 was measured to be 1.25 ms at 1100 K in the shock tube."
DOC = (
    "A shock tube study of oxygen ignition\n"
    "J. Smith and A. Jones (2020)\n"
    f"doi: {DOI}\n\n"
    "Abstract text here.\n\n"
    f"{QUOTE}\n"
    "Further discussion of the measurements follows here.\n"
)


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    session_budget().reset()
    yield
    session_budget().reset()


@pytest.fixture()
def campaign(tmp_path: Path) -> Campaign:
    campaign_input = CampaignInput(
        workspace_name="lit-test",
        initial_mixture=InitialMixture(components=[MixtureComponent(species="O2", mole_fraction=1.0)]),
        target_observables=[TargetObservable(name="ignition_delay")],
        target_reactor_systems=[
            ReactorSystem(
                reactor_type=ReactorType.SHOCK_TUBE,
                temperature_range_K=(800.0, 1200.0),
                pressure_range_bar=(1.0, 5.0),
            )
        ],
        budgets=Budgets(cpu_hours=20.0, experiment_budget=0.0),
    )
    return create_campaign(tmp_path / "ws", campaign_input)


def _action() -> PlannedAction:
    return PlannedAction(
        action_id="lit-a1",
        kind=ActionKind.LITERATURE_SEARCH,
        description="literature search",
        estimated_cpu_hours=0.0,
        rationale="testing",
        approval_requirement=ApprovalRequirement.AUTO_APPROVED,
    )


def _finding_dict(*, quote: str = QUOTE, source_url: str = SOURCE_URL) -> dict[str, Any]:
    return {
        "payload": {
            "category": "experimental_benchmark",
            "reactor_type": "shock_tube",
            "observable": "ignition_delay_time",
            "observable_raw": "ignition delay time",
            "species": [{"raw_name": "O2"}],
            "measured": [{"value": 1.25, "unit": "ms"}],
        },
        "citation": {
            "title": "A shock tube study of oxygen ignition",
            "authors": ["J. Smith"],
            "year": 2020,
            "doi": DOI,
        },
        "verbatim_quote": quote,
        "source_url": source_url,
    }


def _proposal(
    *, findings: list[dict[str, Any]] | None = None, done: bool = True, queries: list[str] | None = None
) -> dict[str, Any]:
    return {
        "queries": queries if queries is not None else ["oxygen ignition delay shock tube"],
        "findings": findings if findings is not None else [],
        "done": done,
    }


def _assessment(credence: float = 0.9) -> dict[str, Any]:
    return {
        "credence": credence,
        "provenance_score": 0.9,
        "quality_score": 0.8,
        "consistency_score": 0.85,
        "rationale": "well supported by the supplied evidence window",
        "flags": [],
    }


def _split_responses(responses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one flat response list into (literature-agent, verifier) queues, in order.

    Existing tests author responses as a single interleaved list in call order, which
    made sense back when the Literature Agent and Verifier shared one MockModel
    (the Defect 5 bug this suite now guards against). Now that they are two distinct
    model instances (see :class:`~carmel.services.literature.LiteratureDeps`), each
    response is routed to whichever model will actually receive it: a
    ``VerifierAssessment``-shaped dict (has ``"credence"``) goes to the verifier
    queue, everything else (a ``LiteratureProposal``-shaped dict) goes to the
    literature-agent queue. Relative order within each queue is preserved.
    """
    literature_responses = [r for r in responses if "credence" not in r]
    verifier_responses = [r for r in responses if "credence" in r]
    return literature_responses, verifier_responses


def _make_deps(
    responses: list[dict[str, Any]],
    *,
    fetch: dict[str, tuple[bytes, str]] | None = None,
    budget: AgentBudgetConfig | None = None,
    search: dict[str, list[SearchResult]] | None = None,
) -> tuple[LiteratureDeps, MockModel, AgentConfig]:
    config = AgentConfig(budget=budget) if budget is not None else AgentConfig()
    literature_responses, verifier_responses = _split_responses(responses)
    model = MockModel(literature_responses)
    verifier_model = MockModel(verifier_responses, name="mock-verifier")
    deps = LiteratureDeps(
        config=config,
        model=model,
        verifier_model=verifier_model,
        search=MockSearchTool(search or {}),
        fetch=MockFetchTool(fetch if fetch is not None else {SOURCE_URL: (DOC.encode(), "text/plain")}),
        ledger=BudgetLedger(config.budget),
    )
    return deps, model, config


def _patch_chem_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chem, "canonical_smiles", lambda raw: "O=O")
    monkeypatch.setattr(chem, "inchikey", lambda raw: "MYMOFIZGZYHOMD-UHFFFAOYSA-N")


def _patch_chem_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chem, "canonical_smiles", lambda raw: None)


class _StatefulLeakyModel:
    """A deliberately STATEFUL fake ``ModelProtocol`` implementation.

    Unlike :class:`~carmel.agents.models.MockModel`, this fake accumulates every
    prompt it has ever been given into a running ``transcript`` that persists for the
    lifetime of the instance -- simulating a conversational model wrapper that would
    carry context across calls. This is exactly the shape of bug DEFECT 5 guards
    against: if such an object were reused as both the proposing Literature Agent's
    model and the Verifier's model, the Verifier's "independent" assessment would be
    contaminated by whatever the Literature Agent said. It exists only to prove, by
    construction, that giving the Verifier its own separate model instance keeps this
    fake's accumulated state scoped to whichever persona actually holds it.
    """

    def __init__(self, responses: list[dict[str, Any]], *, name: str = "stateful-leaky") -> None:
        self.name = name
        self._responses = list(responses)
        self.transcript: list[str] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: AbcSequence[AgentTool],
    ) -> ModelResponse:
        self.transcript.append(f"{system_prompt}\n{user_prompt}")
        if not self._responses:
            raise AgentBridgeError(f"stateful fake {self.name!r} exhausted: no canned responses remain")
        output = self._responses.pop(0)
        return ModelResponse(output=output, model_name=self.name)


class TestHappyPath:
    def test_grounded_finding_lands_in_report(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, model, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.grounding.status == GroundingStatus.GROUNDED_EXACT
        assert finding.credence is not None
        assert finding.credence.credence == pytest.approx(0.9)
        assert finding.payload.species[0].canonicalized is True  # type: ignore[union-attr]
        assert report.rejected == []
        assert report.queries == ["oxygen ignition delay shock tube"]
        assert report.model_name == model.name

    def test_report_persisted_and_artifact_stored(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        report_path = campaign.workspace_root / LITERATURE_REPORT_NAME
        assert report_path.exists()
        reloaded = LiteratureReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        assert reloaded.report_id == report.report_id
        assert len(reloaded.findings) == 1

        # Finding 5: the report is round-trippable through the dedicated loader too.
        via_loader = load_literature_report(campaign.workspace_root)
        assert via_loader.report_id == report.report_id
        assert len(via_loader.findings) == 1

        sha = hashlib.sha256(DOC.encode()).hexdigest()
        artifact_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / sha
        assert (artifact_dir / "raw.bin").read_bytes() == DOC.encode()
        assert report.findings[0].evidence.artifact_sha256 == sha

    def test_save_literature_report_writes_atomically(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Finding 5: persistence must go through the atomic write_json helper (mkstemp +
        # fsync + os.replace), not a naive Path.write_text(), so a crash mid-write can never
        # leave a torn file masquerading as a completed report.
        calls: list[Path] = []
        real_write_json = carmel.services.literature.write_json

        def _spying_write_json(path: Path, data: Any) -> None:
            calls.append(path)
            real_write_json(path, data)

        monkeypatch.setattr(carmel.services.literature, "write_json", _spying_write_json)
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert calls == [campaign.workspace_root / LITERATURE_REPORT_NAME]

    def test_decision_log_events_written(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        events = [e["event"] for e in read_events(campaign.workspace_root / "decision_log.jsonl")]
        assert "literature.search_started" in events
        assert "literature.finding_recorded" in events
        assert "literature.credence_assigned" in events
        assert "literature.search_finished" in events

    def test_lock_released_after_normal_run(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert not (campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME).exists()


class TestGroundingGate:
    def test_fabricated_quote_rejected_and_verifier_never_called(self, campaign: Campaign) -> None:
        fabricated = _finding_dict(quote="This sentence appears nowhere in the fetched document at all.")
        deps, model, config = _make_deps([_proposal(findings=[fabricated])])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.QUOTE_NOT_FOUND
        # The Verifier model was NEVER called: exactly one (literature) model call.
        assert len(model.calls) == 1
        assert model.calls[0]["system_prompt"] == LITERATURE_SYSTEM_PROMPT
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        assert any(e["event"] == "literature.finding_rejected" for e in events)
        assert report.stop_reason == StopReason.SELF_TERMINATED

    def test_unfetchable_source_is_rejected_not_fatal(self, campaign: Campaign) -> None:
        deps, model, config = _make_deps([_proposal(findings=[_finding_dict()])], fetch={})

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.NO_ARTIFACT
        assert len(model.calls) == 1
        assert report.warnings  # the fetch failure is surfaced, not swallowed

    def test_verifier_prompt_never_contains_source_url(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chem_success(monkeypatch)
        deps, model, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(model.calls) == 1
        assert deps.verifier_model is not None
        assert len(deps.verifier_model.calls) == 1
        verifier_call = deps.verifier_model.calls[0]
        assert verifier_call["system_prompt"] == VERIFIER_SYSTEM_PROMPT
        assert verifier_call["output_schema"] is VerifierAssessment
        # The author agent's raw claims must not reach the Verifier.
        assert SOURCE_URL not in verifier_call["user_prompt"]
        assert SOURCE_URL not in verifier_call["system_prompt"]
        assert "secret-paper-url" not in verifier_call["user_prompt"]
        # It sees only sanitized evidence: quote, extracted-text window, grounding.
        assert QUOTE in verifier_call["user_prompt"]
        assert "grounded_exact" in verifier_call["user_prompt"]
        assert "Further discussion of the measurements" in verifier_call["user_prompt"]
        # And it is offered no tools whatsoever.
        assert verifier_call["tool_names"] == []


class TestLoopTermination:
    def test_done_on_first_round_stops_immediately(self, campaign: Campaign) -> None:
        deps, model, config = _make_deps([_proposal(findings=[], done=True)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        assert len(model.calls) == 1  # exactly one literature model call, no second round

    def test_repeat_round_with_nothing_new_stops(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, model, config = _make_deps(
            [
                _proposal(findings=[_finding_dict()], done=False),
                _assessment(),
                _proposal(findings=[_finding_dict()], done=False),  # same finding again
            ]
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.NO_NEW_INFORMATION
        assert len(model.calls) == 2  # two literature rounds
        assert deps.verifier_model is not None
        assert len(deps.verifier_model.calls) == 1  # one verifier call, on a distinct model instance
        assert len(report.findings) == 1  # the duplicate was deduped, not re-verified

    def test_search_results_fed_back_to_next_round(self, campaign: Campaign) -> None:
        hit = SearchResult(title="Shock tube data", url="https://example.org/hit", snippet="tau_ign data")
        deps, model, config = _make_deps(
            [
                _proposal(
                    findings=[_finding_dict(quote="this exact sentence does not appear in the document")],
                    done=False,
                    queries=["q1"],
                ),
                _proposal(findings=[], done=True, queries=[]),
            ],
            search={"q1": [hit]},
        )

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert "https://example.org/hit" in model.calls[1]["user_prompt"]

    def test_queries_beyond_the_round_cap_are_dropped_not_recorded(self, campaign: Campaign) -> None:
        # Finding 6: the agent may propose more queries than we execute. Only the executed
        # subset may land in report.queries/provenance -- recording the full proposed list
        # would overstate what actually ran.
        many_queries = [f"q{i}" for i in range(MAX_QUERIES_PER_ROUND + 3)]
        deps, _, config = _make_deps([_proposal(queries=many_queries, done=True)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.queries == many_queries[:MAX_QUERIES_PER_ROUND]
        assert any("dropped" in w and str(len(many_queries) - MAX_QUERIES_PER_ROUND) in w for w in report.warnings)
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        truncated = [e for e in events if e["event"] == "literature.queries_truncated"]
        assert len(truncated) == 1
        assert truncated[0]["executed"] == many_queries[:MAX_QUERIES_PER_ROUND]
        assert truncated[0]["dropped"] == many_queries[MAX_QUERIES_PER_ROUND:]

    def test_literature_agent_is_never_given_a_live_search_tool(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Finding 23: the deterministic round-trip in _research_loop is the ONLY path
        # that may call deps.search.search. Handing the agent its own live search tool
        # as well would let it re-query out-of-band, double-billing a real provider.
        captured: dict[str, Any] = {}
        real_build = carmel.services.literature.build_literature_agent

        def _spying_build(**kwargs: Any) -> Any:
            captured["tools"] = kwargs.get("tools", ())
            return real_build(**kwargs)

        monkeypatch.setattr(carmel.services.literature, "build_literature_agent", _spying_build)
        deps, _, config = _make_deps([_proposal()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert captured["tools"] == ()


class TestBudgets:
    def test_budget_ceiling_yields_partial_report(self, campaign: Campaign) -> None:
        budget = AgentBudgetConfig(max_model_calls=1)
        deps, model, config = _make_deps([_proposal(findings=[_finding_dict()], done=False)], fetch={}, budget=budget)

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.MAX_MODEL_CALLS
        assert len(model.calls) == 1
        assert len(report.rejected) == 1  # first round's work is preserved in the partial report
        assert any("budget exceeded" in w for w in report.warnings)

    def test_verifier_budget_stop_is_partial_not_fatal(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chem_success(monkeypatch)
        # One model call allowed: the literature round succeeds, the verifier call trips.
        budget = AgentBudgetConfig(max_model_calls=1)
        deps, model, config = _make_deps([_proposal(findings=[_finding_dict()])], budget=budget)

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.MAX_MODEL_CALLS
        assert report.findings == []  # never credence-scored, never recorded
        assert len(model.calls) == 1

    def test_agent_bridge_error_maps_to_error_stop(self, campaign: Campaign) -> None:
        deps, _, config = _make_deps([])  # exhausted MockModel raises AgentBridgeError

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.ERROR
        assert any("agent error" in w for w in report.warnings)


class TestRunLock:
    def test_double_trigger_raises_and_live_pid_lock_never_broken(self, campaign: Campaign) -> None:
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        # Even an ANCIENT started_at must not allow breaking a live-pid lock.
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": (datetime.now(UTC) - timedelta(days=365)).isoformat(),
                    "action_id": "other",
                }
            ),
            encoding="utf-8",
        )
        deps, model, config = _make_deps([_proposal()])

        with pytest.raises(LiteratureRunLockedError):
            run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        assert model.calls == []  # the second run never started
        assert lock_dir.exists()  # and the live lock is intact

    def test_stale_dead_pid_lock_is_broken_and_logged(self, campaign: Campaign) -> None:
        proc = subprocess.Popen(["true"])
        proc.wait()  # reaped: os.kill(pid, 0) now raises ProcessLookupError
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                    "action_id": "crashed",
                }
            ),
            encoding="utf-8",
        )
        deps, _, config = _make_deps([_proposal()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        broken = [e for e in events if e["event"] == "literature.lock_broken"]
        assert len(broken) == 1
        assert broken[0]["level"] == "warning"
        assert not lock_dir.exists()  # released after the run

    def test_lock_released_after_unexpected_exception(self, campaign: Campaign) -> None:
        class _BoomModel:
            name = "boom"

            def complete(
                self,
                *,
                system_prompt: str,
                user_prompt: str,
                output_schema: type[BaseModel],
                tools: AbcSequence[Any],
            ) -> Any:
                raise ValueError("catastrophic model failure")

        config = AgentConfig()
        deps = LiteratureDeps(
            config=config,
            model=_BoomModel(),
            verifier_model=MockModel([], name="mock-verifier"),
            search=MockSearchTool({}),
            fetch=MockFetchTool({}),
            ledger=BudgetLedger(config.budget),
        )

        with pytest.raises(ValueError, match="catastrophic"):
            run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert not (campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME).exists()

    def test_lock_with_missing_metadata_within_grace_period_is_not_broken(self, campaign: Campaign) -> None:
        # A lock dir that exists but has no info.json yet (peer is between mkdir() and its
        # metadata write) must fail CLOSED while it is fresh.
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        deps, model, config = _make_deps([_proposal()])

        with pytest.raises(LiteratureRunLockedError):
            run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        assert model.calls == []
        assert lock_dir.exists()

    def test_lock_with_missing_metadata_past_grace_period_is_broken_and_logged(self, campaign: Campaign) -> None:
        # Finding 4: a crash between lock_dir.mkdir() and the info.json write used to leave
        # a PERMANENTLY non-stale lock (neither the pid-branch nor the started_at-branch had
        # anything to check), recoverable only via `rm -rf`. Past LOCK_GRACE_S, it must now
        # be treated as abandoned and broken like any other stale lock.
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        old_mtime = (datetime.now(UTC) - timedelta(seconds=LOCK_GRACE_S + 60)).timestamp()
        os.utime(lock_dir, (old_mtime, old_mtime))
        deps, _, config = _make_deps([_proposal()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        broken = [e for e in events if e["event"] == "literature.lock_broken"]
        assert len(broken) == 1
        assert "missing/unparseable" in broken[0]["reason"]
        assert not lock_dir.exists()


class TestCredencePenalties:
    def test_failed_canonicalization_caps_credence_and_flags(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chem_failure(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment(credence=0.95)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1
        credence = report.findings[0].credence
        assert credence is not None
        assert credence.credence == pytest.approx(0.7)
        assert "species_not_canonicalized" in credence.flags
        assert report.findings[0].payload.species[0].canonicalized is False  # type: ignore[union-attr]

    def test_low_credence_not_raised_by_cap(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_failure(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment(credence=0.4)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        credence = report.findings[0].credence
        assert credence is not None
        assert credence.credence == pytest.approx(0.4)  # a cap, not a floor


class TestProvenance:
    def test_provenance_contains_no_prompts_or_secrets(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        prov_dir = campaign.workspace_root / "provenance"
        lit_records = [p for p in prov_dir.glob("*_literature_run.json")]
        assert len(lit_records) == 1
        for path in prov_dir.iterdir():
            raw = path.read_text(encoding="utf-8")
            # No system-prompt text, no user-prompt text, no key-shaped strings.
            assert "VERBATIM QUOTES ONLY" not in raw
            assert "Campaign mixture species" not in raw
            assert QUOTE not in raw
            assert "sk-" not in raw
        record = json.loads(lit_records[0].read_text(encoding="utf-8"))
        assert record["stop_reason"] == "self_terminated"
        assert record["n_findings"] == 1
        assert record["grounding_summary"] == {"grounded_exact": 1}


class TestRunRecord:
    def test_success_maps_to_succeeded_none(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])
        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        record = run_record_for(report, _action())

        assert record.status == RunStatus.SUCCEEDED
        assert record.failure_code == FailureCode.NONE
        assert record.tool_name == "literature_agent"
        assert record.run_id == report.run_id
        assert record.submission_mode.value == "local"

    def test_budget_stop_maps_to_budget_exceeded(self, campaign: Campaign) -> None:
        budget = AgentBudgetConfig(max_model_calls=1)
        deps, _, config = _make_deps([_proposal(findings=[], done=False)], budget=budget)
        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        # A round that issues fresh queries but no findings now continues to a second
        # round (its results have not been shown yet), so the one-call ceiling is what
        # actually stops the run. This used to report NO_NEW_INFORMATION instead: the
        # loop always ended after round 1, so no budget ceiling could ever be reached
        # and this test had to fabricate a budget stop to exercise the mapping.
        assert report.stop_reason == StopReason.MAX_MODEL_CALLS

        record = run_record_for(report, _action())

        assert record.status == RunStatus.FAILED
        assert record.failure_code == FailureCode.BUDGET_EXCEEDED

    def test_error_stop_maps_to_agent_error(self, campaign: Campaign) -> None:
        deps, _, config = _make_deps([])
        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        assert report.stop_reason == StopReason.ERROR

        record = run_record_for(report, _action())

        assert record.status == RunStatus.FAILED
        assert record.failure_code == FailureCode.AGENT_ERROR


class TestBuildDeps:
    def test_mock_tier_gets_mock_everything(self) -> None:
        deps = build_deps(AgentConfig())

        assert isinstance(deps.model, MockModel)
        assert isinstance(deps.search, MockSearchTool)
        assert isinstance(deps.fetch, MockFetchTool)

    def test_verifier_model_is_a_distinct_object_from_the_literature_agent_model(self) -> None:
        """DEFECT 5 regression: the Verifier's independence from the proposing
        Literature Agent depends on it never sharing a model instance -- a shared,
        stateful model wrapper would leak the proposing agent's context into the
        supposedly independent assessment. `build_deps` must hand back two distinct
        objects, not the same one twice."""
        deps = build_deps(AgentConfig())

        assert deps.verifier_model is not None
        assert isinstance(deps.verifier_model, MockModel)
        assert deps.verifier_model is not deps.model

    def test_real_provider_without_search_endpoint_fails_closed(self) -> None:
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="CARMEL_TEST_FAKE_KEY",
            external_provider_consent=True,
        )
        with pytest.raises(AgentBridgeError, match="search_endpoint"):
            build_deps(config)

    def test_real_provider_without_search_key_env_fails_closed(self) -> None:
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="CARMEL_TEST_FAKE_KEY",
            external_provider_consent=True,
            search_endpoint="https://search.example.com/api",
        )
        with pytest.raises(AgentBridgeError, match="search_api_key_env"):
            build_deps(config)

    def test_real_provider_with_unset_search_key_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARMEL_TEST_SEARCH_KEY", raising=False)
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="CARMEL_TEST_FAKE_KEY",
            external_provider_consent=True,
            search_endpoint="https://search.example.com/api",
            search_api_key_env="CARMEL_TEST_SEARCH_KEY",
        )
        with pytest.raises(AgentBridgeError, match="CARMEL_TEST_SEARCH_KEY"):
            build_deps(config)


class TestStatefulModelIndependence:
    """DEFECT 5 regression (f): a deliberately stateful fake model must not carry the
    proposing Literature Agent's context into the Verifier call."""

    def test_stateful_model_reused_only_by_literature_agent_never_leaks_into_verifier(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run a full literature-research pass with a deliberately stateful fake as
        the Literature Agent's model and a plain, separate ``MockModel`` as the
        Verifier's model. The stateful fake's accumulated transcript must contain
        ONLY the Literature Agent's own prompt (never the Verifier's system prompt),
        and the Verifier's own model must receive its call independently -- proving
        the two personas never shared state through a common object.
        """
        _patch_chem_success(monkeypatch)
        config = AgentConfig()
        stateful_model = _StatefulLeakyModel([_proposal(findings=[_finding_dict()])])
        verifier_model = MockModel([_assessment()], name="mock-verifier")
        deps = LiteratureDeps(
            config=config,
            model=stateful_model,
            verifier_model=verifier_model,
            search=MockSearchTool({}),
            fetch=MockFetchTool({SOURCE_URL: (DOC.encode(), "text/plain")}),
            ledger=BudgetLedger(config.budget),
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        assert len(report.findings) == 1

        # The stateful fake accumulated exactly one call, and it was the Literature
        # Agent's -- never the Verifier's persona.
        assert len(stateful_model.transcript) == 1
        assert LITERATURE_SYSTEM_PROMPT in stateful_model.transcript[0]
        assert VERIFIER_SYSTEM_PROMPT not in stateful_model.transcript[0]

        # The Verifier's own, distinct model instance received its call independently,
        # with no trace of the stateful fake ever being involved.
        assert len(verifier_model.calls) == 1
        assert verifier_model.calls[0]["system_prompt"] == VERIFIER_SYSTEM_PROMPT

    def test_constructing_deps_with_same_model_instance_for_both_personas_raises(self) -> None:
        """The independence guarantee is enforced, not merely documented: passing the
        exact same object as both ``model`` and ``verifier_model`` must fail loudly at
        construction time rather than silently letting one (potentially stateful)
        instance back both personas."""
        config = AgentConfig()
        shared = MockModel([], name="shared")

        with pytest.raises(ValueError, match="two distinct model instances"):
            LiteratureDeps(
                config=config,
                model=shared,
                verifier_model=shared,
                search=MockSearchTool({}),
                fetch=MockFetchTool({}),
                ledger=BudgetLedger(config.budget),
            )


class TestSchemas:
    def test_proposal_and_assessment_round_trip(self) -> None:
        proposal = LiteratureProposal.model_validate(_proposal(findings=[_finding_dict()]))
        assert proposal.done is True
        assert proposal.findings[0].citation.doi == DOI
        assessment = VerifierAssessment.model_validate(_assessment())
        assert assessment.credence == pytest.approx(0.9)


class _StatusFetchTool:
    """Fetch tool that fails every URL with a chosen HTTP status.

    ``MockFetchTool`` raises a status-less :class:`FetchError`, which cannot express the
    difference between "the publisher wants a subscription" and "the link is broken" --
    the very distinction the acquisition triage turns on.
    """

    def __init__(self, status: int | None) -> None:
        self._status = status

    def fetch(self, url: str) -> tuple[Any, bytes]:
        raise FetchError(f"simulated failure for {url!r}", status=self._status)


class TestAcquisitionTriage:
    """A finding that fails grounding is queued for a human ONLY when the EVIDENCE was
    unobtainable -- never when the document was read and the quote simply was not in it."""

    def test_a_paywalled_paper_is_queued_for_manual_acquisition(self, campaign: Campaign) -> None:
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])
        deps.fetch = _StatusFetchTool(403)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        manifest = load_manifest(campaign.workspace_root)
        assert len(manifest.requests) == 1
        request = manifest.requests[0]
        assert request.reason == AcquisitionReason.PAYWALLED
        assert request.doi == DOI
        assert request.status == AcquisitionStatus.REQUESTED

    def test_a_broken_link_is_queued_but_distinguished_from_a_paywall(self, campaign: Campaign) -> None:
        """The paper is still real and still obtainable by a human; only the REASON
        shown to the operator differs."""
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])
        deps.fetch = _StatusFetchTool(404)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.FETCH_FAILED]

    def test_a_fabricated_quote_is_never_queued_for_acquisition(self, campaign: Campaign) -> None:
        """THE load-bearing case. The document was fetched and read perfectly; the quote
        simply is not in it. That is the fabrication signal the grounding gate exists to
        raise. Queueing it would convert "this agent may have invented a quote" into
        "we are waiting on a human", quietly retiring the strongest rejection the system
        can produce -- and a later drop of the genuine paper would then hand that same
        fabricated claim a second chance at being accepted."""
        fabricated = _finding_dict(
            quote="This sentence appears nowhere in the fetched document whatsoever.",
        )
        deps, _, config = _make_deps([_proposal(findings=[fabricated]), _assessment()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.QUOTE_NOT_FOUND
        assert load_manifest(campaign.workspace_root).requests == []

    def test_the_same_unreachable_paper_proposed_twice_is_queued_once(self, campaign: Campaign) -> None:
        deps, _, config = _make_deps(
            [
                _proposal(findings=[_finding_dict()], done=False),
                _proposal(findings=[_finding_dict(quote=QUOTE + " Repeated.")], done=True),
                _assessment(),
                _assessment(),
            ]
        )
        deps.fetch = _StatusFetchTool(403)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(load_manifest(campaign.workspace_root).requests) == 1

    def test_a_paper_the_agent_cannot_read_is_queued_via_the_wanted_channel(self, campaign: Campaign) -> None:
        """The structural case: a paywalled paper can never be a finding (a finding needs
        a verbatim quote, which needs the document), so without this channel the papers
        most worth having would vanish silently."""
        proposal = _proposal(findings=[])
        proposal["wanted"] = [
            {
                "title": "High pressure shock tube ignition delay of oxy-methane",
                "doi": "10.1115/1.4036254",
                "relevance": "direct IDT benchmark at the campaign's conditions",
            }
        ]
        deps, _, config = _make_deps([proposal])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert len(requests) == 1
        assert requests[0].doi == "10.1115/1.4036254"
        assert requests[0].landing_url == "https://doi.org/10.1115/1.4036254"
        assert "direct IDT benchmark" in requests[0].detail

    def test_a_wanted_paper_with_no_identifier_is_dropped_with_a_warning(self, campaign: Campaign) -> None:
        """Without a DOI or URL there is nothing a human could act on, so it must not
        become a request that cannot be fulfilled."""
        proposal = _proposal(findings=[])
        proposal["wanted"] = [{"title": "Some paper with no identifier at all"}]
        deps, _, config = _make_deps([proposal])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert load_manifest(campaign.workspace_root).requests == []
        assert any("neither a DOI nor a URL" in w for w in report.warnings)


class TestResearchLoopReachesASecondRound:
    """The loop's whole purpose is propose -> search -> read -> propose. It could not
    do that: round 1 cannot contain findings (a finding needs a quote, quotes come from
    fetched documents, documents come from search results, and results are only fed back
    the NEXT round), so the "no new findings" stop fired every time and the search
    results the run had just paid for were never shown to anyone."""

    def test_search_results_are_fed_back_to_a_second_round(self, campaign: Campaign) -> None:
        results = {"q1": [SearchResult(title="A paper", url=SOURCE_URL, snippet="abstract")]}
        deps, model, config = _make_deps(
            [
                _proposal(findings=[], queries=["q1"], done=False),
                _proposal(findings=[_finding_dict()], queries=["q1"], done=True),
                _assessment(),
            ],
            search=results,
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(model.calls) >= 2, "the agent was never asked a second time"
        assert "A paper" in model.calls[1]["user_prompt"], "round 2 did not include the search results"
        assert report.stop_reason == StopReason.SELF_TERMINATED

    def test_a_round_repeating_only_stale_queries_still_stops(self, campaign: Campaign) -> None:
        """The stop condition must not become unreachable: an agent that keeps
        re-issuing queries it has already run learns nothing new and must terminate."""
        deps, _, config = _make_deps(
            [
                _proposal(findings=[], queries=["q1"], done=False),
                _proposal(findings=[], queries=["q1"], done=False),
            ],
            search={"q1": []},
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.NO_NEW_INFORMATION
