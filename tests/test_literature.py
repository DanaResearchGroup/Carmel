"""Integration tests for the Literature Agent + Verifier orchestration."""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import types
import urllib.error
from collections.abc import Iterator
from collections.abc import Sequence as AbcSequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import carmel.services.literature
from carmel.agents.bridge import AgentTool, ModelResponse
from carmel.agents.budget import BudgetExceededError, BudgetLedger, BudgetUsage, session_budget
from carmel.agents.literature_agent import (
    LITERATURE_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    LiteratureProposal,
    VerifierAssessment,
)
from carmel.agents.models import AgentBridgeError, MockModel
from carmel.agents.tools.academic import OaResolution, OpenAccessResolver
from carmel.agents.tools.extract import extract_text
from carmel.agents.tools.fetch import FetchedArtifact, FetchError, HttpFetchTool, MockFetchTool
from carmel.agents.tools.search import MockSearchTool, SearchError, SearchResult
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
from carmel.schemas.literature import (
    ArtifactProvenance,
    GroundingStatus,
    LiteraturePassMode,
    LiteratureReport,
    StopReason,
)
from carmel.services import chem
from carmel.services import literature as literature_module
from carmel.services.acquisition import inbox_dir, load_manifest, record_request
from carmel.services.campaigns import create_campaign
from carmel.services.decision_log import read_events
from carmel.services.evidence import EVIDENCE_LITERATURE_DIR, store_artifact
from carmel.services.literature import (
    LITERATURE_REPORT_NAME,
    LOCK_GRACE_S,
    MAX_QUERIES_PER_ROUND,
    RUN_LOCK_DIR_NAME,
    LiteratureDeps,
    LiteratureRunLockedError,
    build_deps,
    load_literature_report,
    run_corpus_pass,
    run_literature_research,
    run_record_for,
)
from carmel.services.plan_progress import publish_lock_info

DOI = "10.1000/test.doi"
SOURCE_URL = "https://example.com/papers/secret-paper-url"

#: Title used for manual-acquisition drops; long enough that the identity check has
#: real signal to match on rather than a couple of common words.
TITLE_FOR_DROP = "Ignition delay times in methane oxygen argon mixtures behind reflected shock waves"
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
    oa_resolver: Any = None,
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
        oa_resolver=oa_resolver,
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
        assert [q.text for q in report.queries] == ["oxygen ignition delay shock tube"]
        assert [q.run_id for q in report.queries] == [report.run_id]
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

    def test_html_source_url_is_not_stored_as_the_paper(self, campaign: Campaign) -> None:
        """Live campaign 5b766b4b-bf72-4db9-bb28-4229b037bf07 (workspace ``live-syngas``)
        stored all 3 artifacts fetched via this exact path -- the LLM-proposed
        ``source_url`` -- as ``text/html``. This path had NO content-type gate at all,
        unlike ``_attempt_oa_fetch``'s OA-candidate path. A landing page still carries
        the paper's title/abstract, so a quote lifted from an abstract could pass
        ``ground_finding`` against it and read as backed by the full paper."""
        deps, model, config = _make_deps(
            [_proposal(findings=[_finding_dict()])],
            fetch={SOURCE_URL: (b"<html><body>Please sign in to view this article</body></html>", "text/html")},
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert report.artifacts == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.NO_ARTIFACT
        assert any("rejected fetch" in w for w in report.warnings)
        # It must still be queued for a human, not silently dropped.
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NOT_A_DOCUMENT]
        assert "text/html" in requests[0].detail

    def test_elsevier_redirect_stub_is_rejected_not_grounded_against(self, campaign: Campaign) -> None:
        """Regression test for the exact observed defect: artifact sha256
        ``fdaceab39e73...`` from the live campaign above was a 2710-byte Elsevier
        redirect stub whose entire extracted text was the single word "Redirecting"
        (83 whitespace-padded chars) -- non-empty, so a check that only looked at
        extracted-text length would have let it through. The content-type gate must
        reject it before extracted-text length is even considered."""
        redirect_stub = (
            b'<html><head><meta HTTP-EQUIV="REFRESH" '
            b'content="0; URL=https://linkinghub.elsevier.com/retrieve/pii/S0000000000000000">'
            b"</head><body>Redirecting</body></html>"
        )
        deps, model, config = _make_deps(
            [_proposal(findings=[_finding_dict()])],
            fetch={SOURCE_URL: (redirect_stub, "text/html")},
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.artifacts == []
        assert report.findings == []
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NOT_A_DOCUMENT]

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


class TestValidateDocument:
    """Unit tests for the shared gate both storage paths route through.

    ``_attempt_oa_fetch`` (open-access candidates) and ``_fetch_and_store``
    (the LLM-proposed ``source_url``) used to duplicate this validation in one
    and omit it entirely in the other -- see ``_validate_document``'s docstring
    for the live-campaign evidence that omission produced.
    """

    def test_html_content_type_is_rejected(self) -> None:
        extracted, reason = literature_module._validate_document(
            "text/html", b"<html><body>Full text of the paper here.</body></html>"
        )

        assert extracted is None
        assert reason == AcquisitionReason.NOT_A_DOCUMENT

    def test_zero_bytes_is_rejected(self) -> None:
        extracted, reason = literature_module._validate_document("text/plain", b"")

        assert extracted is None
        assert reason == AcquisitionReason.EMPTY_DOCUMENT

    def test_whitespace_only_text_is_rejected(self) -> None:
        extracted, reason = literature_module._validate_document("text/plain", b"   \n\t  ")

        assert extracted is None
        assert reason == AcquisitionReason.EMPTY_DOCUMENT

    def test_text_plain_with_real_content_is_accepted(self) -> None:
        extracted, reason = literature_module._validate_document("text/plain", DOC.encode())

        assert reason is None
        assert extracted is not None
        assert QUOTE in extracted.text

    def test_application_pdf_with_real_content_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakePage:
            def extract_text(self) -> str:
                return QUOTE

        class _FakeReader:
            def __init__(self, _stream: object) -> None:
                self.pages = [_FakePage()]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

        extracted, reason = literature_module._validate_document("application/pdf", b"%PDF-1.4 irrelevant")

        assert reason is None
        assert extracted is not None
        assert QUOTE in extracted.text


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

        assert [q.text for q in report.queries] == many_queries[:MAX_QUERIES_PER_ROUND]
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

    def test_losing_the_steal_race_reacquires_from_scratch(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1-5: two racers must not both acquire a lock they each judge stale.

        The old implementation stole a stale lock via a non-atomic
        ``rmtree`` then ``mkdir``, so two racers could both observe the
        lock gone and both believe they had acquired it. The rewrite
        reuses ``plan_progress``'s atomic-rename-to-a-uuid-suffixed-target
        steal, mirroring ``dispatcher._acquire_dispatch_lock``'s own
        race-safety test. Here we simulate a peer winning the SAME steal
        race: this frame's ``Path.rename`` is intercepted so that, on its
        first call against the literature lock dir, a peer moves the
        stale lock dir aside itself and publishes a fresh, live lock
        before this frame's rename lands. Our rename must then observe
        ``FileNotFoundError`` and loop back to re-evaluate from scratch,
        seeing the peer's fresh lock as live and refusing with
        ``LiteratureRunLockedError`` instead of also believing it holds
        the lease.
        """
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": 2**22 - 1,  # extremely unlikely to be alive: looks stale
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                    "pid_start": 0,
                }
            )
        )

        original_rename = Path.rename
        state = {"intercepted": False}

        def racy_rename(self_path: Path, target: Path, *args: object, **kwargs: object) -> object:
            if self_path == lock_dir and not state["intercepted"]:
                state["intercepted"] = True
                # A peer wins the steal race first: it moves the same stale
                # lock dir aside under its OWN process-unique name and
                # immediately publishes a fresh, live lock in its place —
                # all before our rename call (below) can land.
                peer_target = lock_dir.with_name(f"{lock_dir.name}.stale.peer")
                original_rename(self_path, peer_target)
                lock_dir.mkdir()
                publish_lock_info(lock_dir)
                shutil.rmtree(peer_target, ignore_errors=True)
                raise FileNotFoundError("lock dir already moved by a peer")
            return original_rename(self_path, target, *args, **kwargs)

        monkeypatch.setattr(Path, "rename", racy_rename)

        deps, model, config = _make_deps([_proposal()])
        with pytest.raises(LiteratureRunLockedError):
            run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        assert model.calls == []
        # Exactly one lock dir exists afterwards (the peer's) — no stray
        # renamed-aside directory left over from either racer.
        assert lock_dir.exists()
        assert not lock_dir.with_name(f"{lock_dir.name}.stale.peer").exists()
        leftovers = list((campaign.workspace_root / EVIDENCE_LITERATURE_DIR).glob(f"{lock_dir.name}.stale.*"))
        assert leftovers == []

    def test_live_pid_with_mismatched_pid_start_is_reused_pid_and_broken(self, campaign: Campaign) -> None:
        """P1-6: a lock recorded against a live pid whose ``pid_start`` differs
        from the process currently holding that pid is a stale lock from a
        crashed/recycled holder, not a live one, and must be breakable.

        The old hand-rolled ``info.json`` never recorded ``pid_start`` at
        all, so this guard was permanently disabled: any lock whose pid
        happened to be reused by an unrelated process would look live
        forever and wedge the workspace. We use this very test process's
        own (guaranteed-alive) pid but an impossible ``pid_start`` so the
        real ``/proc/<pid>/stat`` start time can never match it.
        """
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                    "pid_start": -1,  # cannot match the real /proc/<pid>/stat start time
                    "action_id": "crashed",
                }
            )
        )
        deps, _, config = _make_deps([_proposal()])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        broken = [e for e in events if e["event"] == "literature.lock_broken"]
        assert len(broken) == 1
        assert "reused" in broken[0]["reason"]
        assert not lock_dir.exists()

    def test_release_does_not_delete_a_successors_lock(self, campaign: Campaign) -> None:
        """P1-7: releasing a lease this frame no longer actually owns (because
        it was stolen out from under it) must NOT delete the successor's live
        lock.

        The old implementation released via an unconditional
        ``shutil.rmtree(lock_dir, ignore_errors=True)`` with no ownership
        check at all, so a frame that woke up after its lock had been
        stolen as stale (e.g. after a long GC pause or a slow unwind of an
        unrelated exception) could delete a successor's in-progress lock.
        The new lease records the pid/pid_start it observed at acquire
        time and only removes the lock dir if those still match at
        release time.
        """
        lock_dir = campaign.workspace_root / EVIDENCE_LITERATURE_DIR / RUN_LOCK_DIR_NAME
        log_path = campaign.workspace_root / "decision_log.jsonl"
        lease = literature_module._acquire_run_lock(
            campaign.workspace_root,
            action_id="a",
            run_id="r1",
            stale_after_s=1.0,
            log_path=log_path,
        )
        assert lease.lock_dir == lock_dir

        # Simulate a successor having stolen this (now-stale) lock: the
        # original holder's lease token no longer describes what is on disk.
        # A different pid/pid_start than this test process's own (which the
        # lease captured at acquire time) stands in for a genuinely different
        # holder -- a real successor would never share our pid.
        shutil.rmtree(lock_dir, ignore_errors=True)
        lock_dir.mkdir(parents=True)
        (lock_dir / "info.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                    "pid_start": -1,
                    "action_id": "successor",
                }
            )
        )

        lease.release()

        assert lock_dir.exists()
        info = json.loads((lock_dir / "info.json").read_text(encoding="utf-8"))
        assert info["action_id"] == "successor"


class _FailingAfterNSearchTool:
    """A search backend that succeeds ``n_ok`` times, then raises ``SearchError``.

    Simulates a transport failure (503, DNS blip, timeout) arriving mid-run: earlier,
    already-completed rounds must still make it into the final report (P1-12).
    """

    def __init__(self, n_ok: int, results: dict[str, list[SearchResult]] | None = None) -> None:
        self._remaining_ok = n_ok
        self._results = results or {}
        self.calls = 0

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        self.calls += 1
        if self._remaining_ok <= 0:
            raise SearchError("simulated transport failure (503)")
        self._remaining_ok -= 1
        return self._results.get(query, [])[:limit]


class TestSearchTransportFailure:
    """P1-12: ``HttpSearchTool``/``budgeted_get_json`` used to let a bare transport
    exception propagate straight out of ``HttpSearchTool.search``, past every handler in
    ``run_literature_research`` -- crashing the whole run and discarding every finding
    already paid for in earlier rounds. A search failure must instead surface as a typed
    ``SearchError``, caught alongside ``BudgetExceededError``/``AgentBridgeError``, so the
    run stops with ``StopReason.ERROR`` and still reports whatever prior rounds produced.
    """

    def test_earlier_rounds_findings_survive_a_later_search_failure(self, campaign: Campaign) -> None:
        # Round 1: search succeeds, a finding is proposed and processed (verified via the
        # single queued VerifierAssessment). ``done=False`` so the loop proceeds to round 2
        # -- proving this is about SURVIVING a later failure, not merely returning early.
        # Round 2: the literature agent proposes another query, but the search backend
        # itself has now failed; this must happen strictly BEFORE round 2's (nonexistent)
        # findings would be processed, per ``_research_loop``'s query-then-findings order.
        deps, model, config = _make_deps(
            [
                _proposal(findings=[_finding_dict()], queries=["q1"], done=False),
                _proposal(findings=[], queries=["q2"], done=False),
                _assessment(),
            ],
        )
        deps.search = _FailingAfterNSearchTool(n_ok=1)

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.search.calls == 2, "the fake was not called for both rounds"
        assert report.stop_reason == StopReason.ERROR
        assert len(report.findings) == 1, "round 1's finding must survive round 2's search failure"
        assert any("search failed" in warning for warning in report.warnings)
        assert "q1" in [q.text for q in report.queries]


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


class TestManualAcquisitionsAreCollectedByARun:
    """A paper a human obtained and dropped in the inbox must be admitted by the next
    run.

    Without this the manual-acquisition loop is open-ended: Carmel asks for a paper,
    the operator supplies it, and nothing in the product ever picks it up. `collect_inbox`
    was fully unit-tested but had NO caller, so the feature was complete everywhere
    except at the point where it would have been used.
    """

    @staticmethod
    def _queue_and_drop(workspace_root: Path, body: str) -> str:
        """Queue a request the way a run would, then drop a matching file for it."""
        request = record_request(
            workspace_root,
            title=TITLE_FOR_DROP,
            doi=DOI,
            landing_url=f"https://doi.org/{DOI}",
            reason=AcquisitionReason.PAYWALLED,
        )
        (inbox_dir(workspace_root) / f"{request.slug}.txt").write_bytes(body.encode("utf-8"))
        return request.slug

    def test_a_dropped_paper_is_admitted_at_the_start_of_the_next_run(self, campaign: Campaign) -> None:
        slug = self._queue_and_drop(
            campaign.workspace_root, f"{TITLE_FOR_DROP}\nDOI: {DOI}\nAbstract: measurements follow."
        )
        deps, _, config = _make_deps([_proposal(done=True)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        request = next(r for r in load_manifest(campaign.workspace_root).requests if r.slug == slug)
        assert request.status == AcquisitionStatus.FULFILLED
        assert request.fulfilled_sha256
        # The operator has to be told, or an admission is invisible until someone
        # goes looking in the manifest.
        assert any(slug in warning for warning in report.warnings)

    def test_a_wrong_paper_is_rejected_and_never_admitted(self, campaign: Campaign) -> None:
        # The negative case is the one that matters: a mis-filed PDF would otherwise
        # attach one paper's bytes to another paper's citation and silently corrupt the
        # evidence chain -- a human handing Carmel a file is not automatically trustworthy.
        slug = self._queue_and_drop(
            campaign.workspace_root, "Laminar flame speeds of hydrogen air mixtures at elevated pressure."
        )
        deps, _, config = _make_deps([_proposal(done=True)])

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        request = next(r for r in load_manifest(campaign.workspace_root).requests if r.slug == slug)
        assert request.status == AcquisitionStatus.REJECTED
        assert not request.fulfilled_sha256

    def test_an_unreadable_inbox_never_takes_down_the_run(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(workspace_root: Path, *, max_bytes: int) -> list[Any]:
            raise OSError("inbox is on a dead mount")

        monkeypatch.setattr(literature_module, "collect_inbox", _boom)
        deps, _, config = _make_deps([_proposal(done=True)])

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        # Reported, not fatal: losing a literature search over an inbox problem would be
        # a far worse trade than running without the drop.
        assert report.stop_reason != StopReason.ERROR
        assert any("inbox" in warning for warning in report.warnings)


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


WANTED_DOI = "10.1039/c9re00429g"
WANTED_TITLE = "Continuous flow synthesis under catalytic conditions in a packed bed reactor"
OA_PDF_URL = "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g"
RELEVANCE = "direct benchmark at the campaign's exact conditions"


class _FakeOaResolver:
    """Canned DOI -> OaResolution, recording every resolution request."""

    def __init__(self, resolutions: dict[str, OaResolution] | None = None) -> None:
        self._resolutions = resolutions or {}
        self.calls: list[str] = []
        self.titles: list[str | None] = []

    def resolve(self, doi: str, *, title: str | None = None) -> OaResolution:
        self.calls.append(doi)
        self.titles.append(title)
        return self._resolutions.get(
            doi, OaResolution(candidates=(), note="no open-access copy advertised by any open-access index")
        )


class _RoutedFetchTool:
    """Per-URL outcomes: canned bytes for some URLs, a chosen HTTP status for others."""

    def __init__(self, ok: dict[str, tuple[bytes, str]], statuses: dict[str, int]) -> None:
        self._ok = MockFetchTool(ok)
        self._statuses = statuses
        self.fetched: list[str] = []

    def fetch(self, url: str) -> tuple[Any, bytes]:
        self.fetched.append(url)
        if url in self._statuses:
            raise FetchError(f"simulated failure for {url!r}", status=self._statuses[url])
        return self._ok.fetch(url)


def _wanted_proposal() -> dict[str, Any]:
    proposal = _proposal(findings=[])
    proposal["wanted"] = [{"title": WANTED_TITLE, "doi": WANTED_DOI, "relevance": RELEVANCE}]
    return proposal


class TestWantedPaperOpenAccessResolution:
    """A wanted paper's access status must be OBSERVED, never asserted by the model.

    The defect this class pins down: a real campaign queued 12 papers as ``paywalled``
    on nothing but the proposing agent's say-so -- 5 of the 12 were genuinely open
    access and Carmel never attempted a single fetch.
    """

    def test_an_open_access_copy_is_fetched_and_stored_instead_of_queued(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver(
            {WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="OpenAlex: 1 OA PDF candidate")}
        )
        deps, _, config = _make_deps(
            [_wanted_proposal()],
            fetch={OA_PDF_URL: (DOC.encode(), "text/plain")},
            oa_resolver=resolver,
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert resolver.calls == [WANTED_DOI]
        assert resolver.titles == [WANTED_TITLE], "the paper's title must reach the title-matched OA providers"
        assert load_manifest(campaign.workspace_root).requests == []
        sha = hashlib.sha256(DOC.encode()).hexdigest()
        assert [a.sha256 for a in report.artifacts] == [sha]
        assert (campaign.workspace_root / EVIDENCE_LITERATURE_DIR / sha / "raw.bin").exists()
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        acquired = [e for e in events if e["event"] == "literature.oa_copy_acquired"]
        assert len(acquired) == 1
        assert acquired[0]["url"] == OA_PDF_URL
        assert acquired[0]["sha256"] == sha
        assert not any(e["event"] == "literature.paper_requested" for e in events)

    def test_candidates_are_tried_in_order_until_one_succeeds(self, campaign: Campaign) -> None:
        second = "https://repo.example/green.pdf"
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL, second), note="2 candidates")})
        fetch = _RoutedFetchTool(ok={second: (DOC.encode(), "text/plain")}, statuses={OA_PDF_URL: 404})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = fetch

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert fetch.fetched == [OA_PDF_URL, second]
        assert load_manifest(campaign.workspace_root).requests == []
        assert [a.sha256 for a in report.artifacts] == [hashlib.sha256(DOC.encode()).hexdigest()]

    def test_a_403_on_the_oa_copy_is_queued_as_an_observed_paywall(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="1 candidate")})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = _StatusFetchTool(403)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert len(requests) == 1
        request = requests[0]
        assert request.reason == AcquisitionReason.PAYWALLED
        assert "HTTP 403" in request.detail
        assert "pubs.rsc.org" in request.detail
        assert RELEVANCE in request.detail
        assert request.landing_url == f"https://doi.org/{WANTED_DOI}"
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        requested = [e for e in events if e["event"] == "literature.paper_requested"]
        assert requested[0]["oa_attempts"] == [OA_PDF_URL]
        assert requested[0]["reason"] == "paywalled"

    def test_a_404_on_the_oa_copy_is_queued_as_fetch_failed_not_paywalled(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="1 candidate")})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = _StatusFetchTool(404)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.FETCH_FAILED]
        assert "HTTP 404" in requests[0].detail

    def test_an_observed_paywall_wins_over_an_earlier_broken_link(self, campaign: Campaign) -> None:
        """403 is the reason the operator can act on (a subscription): if ANY candidate
        observed one, that is the request's reason, not whichever failure came first."""
        second = "https://pubs.example/vor.pdf"
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL, second), note="2 candidates")})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = _RoutedFetchTool(ok={}, statuses={OA_PDF_URL: 404, second: 403})

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.PAYWALLED]
        assert "HTTP 404" in requests[0].detail and "HTTP 403" in requests[0].detail

    def test_an_html_landing_page_is_not_passed_off_as_the_paper(self, campaign: Campaign) -> None:
        """The live probe found 5 of 11 advertised PDF URLs served an HTML landing page;
        storing one as 'the paper' would silently retire the acquisition request while
        leaving Carmel with no readable full text."""
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="1 candidate")})
        deps, _, config = _make_deps(
            [_wanted_proposal()],
            fetch={OA_PDF_URL: (b"<html><body>Please sign in</body></html>", "text/html")},
            oa_resolver=resolver,
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NOT_A_DOCUMENT]
        assert "text/html" in requests[0].detail
        assert report.artifacts == []

    def test_a_zero_byte_response_is_not_stored_and_the_paper_is_queued_as_empty(self, campaign: Campaign) -> None:
        """The defect this pins down: a live campaign fetched a figshare landing page,
        received zero bytes, and stored it as a successful acquisition -- suppressing
        the manual queue that is the whole fallback. A zero-byte fetch must be treated
        as a failed acquisition, not evidence."""
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="1 candidate")})
        deps, _, config = _make_deps(
            [_wanted_proposal()],
            fetch={OA_PDF_URL: (b"", "text/plain")},
            oa_resolver=resolver,
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.artifacts == []
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.EMPTY_DOCUMENT]
        assert "0 bytes" in requests[0].detail
        events = read_events(campaign.workspace_root / "decision_log.jsonl")
        assert not any(e["event"] == "literature.oa_copy_acquired" for e in events)
        requested = [e for e in events if e["event"] == "literature.paper_requested"]
        assert requested[0]["reason"] == "empty_document"

    def test_a_response_with_bytes_but_no_extractable_text_is_queued_as_empty(self, campaign: Campaign) -> None:
        """Non-empty bytes that extract to whitespace-only text are just as unusable
        as zero bytes and must be rejected the same way."""
        resolver = _FakeOaResolver({WANTED_DOI: OaResolution(candidates=(OA_PDF_URL,), note="1 candidate")})
        deps, _, config = _make_deps(
            [_wanted_proposal()],
            fetch={OA_PDF_URL: (b"   \n\t  ", "text/plain")},
            oa_resolver=resolver,
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.artifacts == []
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.EMPTY_DOCUMENT]
        assert "no extractable text" in requests[0].detail

    def test_no_oa_candidate_is_queued_with_the_truthful_reason(self, campaign: Campaign) -> None:
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=_FakeOaResolver())

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert len(requests) == 1
        assert requests[0].reason == AcquisitionReason.NO_OPEN_ACCESS_COPY
        assert "no open-access copy advertised" in requests[0].detail
        assert RELEVANCE in requests[0].detail

    def test_a_truncated_resolution_is_queued_as_incomplete_not_as_no_open_access_copy(
        self, campaign: Campaign
    ) -> None:
        """A resolution cut short establishes nothing, so it must not claim nothing exists.

        Guards the asserted-vs-observed distinction: ``no_open_access_copy`` is a
        finding, ``oa_lookup_incomplete`` is an admission. A live run hit this via an
        arXiv read timeout and a Semantic Scholar 404 in the same campaign.
        """
        truncated = OaResolution(
            candidates=(),
            note="OpenAlex: 0 OA PDF candidates; arXiv: lookup failed (read operation timed out)",
            complete=False,
        )
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=_FakeOaResolver({WANTED_DOI: truncated}))

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.OA_LOOKUP_INCOMPLETE]
        assert "lookup failed" in requests[0].detail

    def test_a_completed_resolution_that_found_nothing_is_still_no_open_access_copy(self, campaign: Campaign) -> None:
        """The complement of the test above -- the honest negative must survive."""
        exhausted = OaResolution(
            candidates=(), note="OpenAlex: 0 OA PDF candidates; Unpaywall: 0 OA PDF candidates", complete=True
        )
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=_FakeOaResolver({WANTED_DOI: exhausted}))

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]

    def test_without_a_resolver_the_paper_is_still_queued_without_asserting_a_paywall(self, campaign: Campaign) -> None:
        """Even with no resolver wired (mock tier), 'paywalled' must not be recorded:
        nothing observed a paywall."""
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=None)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]
        assert "resolver" in requests[0].detail

    def test_a_wanted_paper_without_a_doi_skips_resolution(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver()
        proposal = _proposal(findings=[])
        proposal["wanted"] = [{"title": WANTED_TITLE, "landing_url": "https://example.org/paper"}]
        deps, _, config = _make_deps([proposal], oa_resolver=resolver)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert resolver.calls == []
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]
        assert "no DOI" in requests[0].detail

    def test_consent_withheld_makes_zero_network_calls_end_to_end(self, campaign: Campaign) -> None:
        """A run with consent withheld, wired with the REAL resolver and REAL fetch
        tool, must never open a socket: the booby-trapped opener proves it."""

        def _boom(url: str, **kwargs: Any) -> Any:
            raise AssertionError(f"network call attempted without consent: {url}")

        deps, _, config = _make_deps([_wanted_proposal()])
        deps.oa_resolver = OpenAccessResolver(
            ledger=deps.ledger,
            external_provider_consent=False,
            unpaywall_email="ops@example.org",
            opener=_boom,
        )
        deps.fetch = HttpFetchTool(
            ledger=deps.ledger,
            external_provider_consent=False,
            opener=_boom,
            resolver=lambda hostname: ["93.184.216.34"],
        )

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]
        assert "consent" in requests[0].detail

    def test_every_provider_answering_404_is_no_open_access_copy_not_incomplete(self, campaign: Campaign) -> None:
        """Ties Fix 1 (HTTP 404 as a definitive "no record", not a cut-short lookup)
        to the operator-facing outcome end to end, through the REAL resolver.

        A live run saw Semantic Scholar answer a DOI lookup with a plain HTTP 404
        (observed 2026-07-30 and again 2026-07-31); before ``SearchNotFound`` existed,
        that 404 was indistinguishable from a transport failure and got reported to
        the operator as ``oa_lookup_incomplete`` -- "resolution was cut short" -- when
        every provider had, in fact, answered. If every enabled provider answers 404,
        the resolution completed and found nothing, so the queued reason must be the
        honest negative ``NO_OPEN_ACCESS_COPY``, never ``OA_LOOKUP_INCOMPLETE``.
        """

        def _always_404(url: str, **kwargs: Any) -> Any:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        deps, _, config = _make_deps([_wanted_proposal()])
        deps.oa_resolver = OpenAccessResolver(
            ledger=deps.ledger,
            external_provider_consent=True,
            unpaywall_email="ops@example.org",
            opener=_always_404,
        )

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]
        assert requests[0].reason != AcquisitionReason.OA_LOOKUP_INCOMPLETE


class TestReportSchemaVersionGate:
    """Spar round 7, P2. The migration accepted any ``schema_version >= 2`` unchanged.

    A report from a FUTURE Carmel would then be handed to a validator that does not
    know its fields. Either it fails with a schema error naming a field the operator
    has never heard of, or -- worse, if the newer version only added optional fields --
    it validates cleanly and the next write silently drops them, downgrading a newer
    report in place. Refuse it instead and say what to do.
    """

    def test_a_future_schema_version_is_refused(self) -> None:
        from carmel.services.literature import migrate_report_payload

        with pytest.raises(ValueError, match="schema version 3"):
            migrate_report_payload({"schema_version": 3, "report_id": "r1"})

    def test_the_current_version_passes_through_untouched(self) -> None:
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION
        from carmel.services.literature import migrate_report_payload

        payload = {"schema_version": CURRENT_REPORT_SCHEMA_VERSION, "report_id": "r1"}

        assert migrate_report_payload(payload) is payload

    def test_a_v1_report_is_still_migrated(self) -> None:
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION
        from carmel.services.literature import migrate_report_payload

        migrated = migrate_report_payload({"schema_version": 1, "run_id": "r", "action_id": "a", "queries": []})

        assert isinstance(migrated, dict)
        assert migrated["schema_version"] == CURRENT_REPORT_SCHEMA_VERSION
        assert len(migrated["passes"]) == 1


class TestReportAccumulatesAcrossPasses:
    """Decision 0004/D1: one report per campaign, appended to, with every finding and
    query carrying the pass that produced it."""

    def test_a_v1_report_migrates_into_a_single_search_pass(self, campaign: Campaign) -> None:
        """A v1 report predates the notion of a pass, but its attribution was always
        true -- it simply had nowhere to be written down. Migration must recover it
        rather than discard the record, because the existing live report holds the
        only evidence that the grounding gate refuses ungrounded claims."""
        v1 = {
            "schema_version": 1,
            "report_id": "rep1",
            "campaign_id": campaign.campaign_id,
            "action_id": "act-1",
            "run_id": "run-1",
            "created_at": datetime.now(UTC).isoformat(),
            "queries": ["syngas ignition delay"],
            "artifacts": [],
            "findings": [],
            "rejected": [],
            "stop_reason": StopReason.SELF_TERMINATED.value,
            "model_name": "mock",
            "usage": BudgetUsage(
                model_calls=2, tokens=12238, cost_usd=0.048843, fetches=6, fetch_bytes=772351, elapsed_s=127.5
            ).model_dump(mode="json"),
            "warnings": ["a warning worth keeping"],
        }
        (campaign.workspace_root / "literature_report.json").write_text(json.dumps(v1))

        report = load_literature_report(campaign.workspace_root)

        assert report.schema_version == 2
        assert len(report.passes) == 1
        assert report.passes[0].mode == LiteraturePassMode.SEARCH
        assert report.run_id == "run-1"
        assert report.action_id == "act-1"
        assert report.stop_reason == StopReason.SELF_TERMINATED
        assert report.warnings == ["a warning worth keeping"]
        assert report.usage.cost_usd == pytest.approx(0.048843)
        assert [(q.text, q.run_id, q.action_id) for q in report.queries] == [
            ("syngas ignition delay", "run-1", "act-1")
        ]

    def test_a_second_pass_appends_and_never_destroys_the_first(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason overwriting is forbidden: the first pass's rejections are the
        record of the safety design working. A second pass must not erase them."""
        deps, _, config = _make_deps([_proposal(queries=["first query"], done=True)])
        first = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)
        first_run_id = first.run_id
        n_first_rejected = len(first.rejected)

        deps2, _, config2 = _make_deps([_proposal(queries=["second query"], done=True)])
        second_action = _action().model_copy(update={"action_id": "lit-a2"})
        second = run_literature_research(campaign.workspace_root, campaign, second_action, deps2, config=config2)

        assert len(second.passes) == 2
        assert [p.run_id for p in second.passes] == [first_run_id, second.run_id]
        assert second.run_id != first_run_id
        assert len(second.rejected) >= n_first_rejected, "the first pass's rejections must survive"
        assert [q.text for q in second.queries] == ["first query", "second query"]
        assert [q.run_id for q in second.queries] == [first_run_id, second.run_id]

        reloaded = load_literature_report(campaign.workspace_root)
        assert len(reloaded.passes) == 2

    def test_findings_carry_the_pass_that_produced_them(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attribution lives on the finding, not on its position in a list, so it
        survives any later reordering, filtering or merge."""
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])
        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        produced = report.findings + report.rejected
        assert produced, "the fixture must produce something to attribute"
        assert all(item.run_id == report.run_id for item in produced)
        assert all(item.action_id == "lit-a1" for item in produced)
        assert report.findings_for(report.run_id) == report.findings
        assert report.findings_for("a-run-that-never-happened") == []


SECOND_QUOTE = "The measured ignition delay time was 3.40 ms at 900 K in the same shock tube."
SECOND_DOI = "10.1000/second.doi"
SECOND_DOC = (
    "A second shock tube study of argon-diluted ignition\n"
    "R. Jones and K. Lee (2021)\n"
    f"doi: {SECOND_DOI}\n\n"
    "Results\n\n"
    f"{SECOND_QUOTE}\n"
    "Further discussion here.\n"
)


def _store(workspace_root: Path, *, text: str, url: str) -> str:
    """Put one document into the workspace's evidence store, as a real artifact."""
    data = text.encode()
    stored = store_artifact(
        workspace_root,
        data=data,
        artifact=FetchedArtifact(
            url=url,
            final_url=url,
            sha256=hashlib.sha256(data).hexdigest(),
            content_type="text/plain",
            n_bytes=len(data),
            fetched_at=datetime.now(UTC),
        ),
        extracted=extract_text(data, "text/plain"),
        provenance=ArtifactProvenance.MANUAL,
        max_bytes=10_000_000,
    )
    return stored.sha256


def _corpus_finding(
    *, sha256: str, quote: str = QUOTE, doi: str = DOI, title: str = "A shock tube study of oxygen ignition"
) -> dict[str, Any]:
    return {
        "payload": {
            "category": "experimental_benchmark",
            "reactor_type": "shock_tube",
            "observable": "ignition_delay_time",
            "observable_raw": "ignition delay time",
            "species": [{"raw_name": "O2"}],
            "measured": [{"value": 1.25, "unit": "ms"}],
        },
        "citation": {"title": title, "authors": ["J. Smith"], "year": 2020, "doi": doi},
        "verbatim_quote": quote,
        "artifact_sha256": sha256,
    }


def _corpus_proposal(findings: list[dict[str, Any]] | None = None, *, done: bool = True) -> dict[str, Any]:
    return {"findings": findings or [], "done": done}


class TestCorpusPass:
    """The second pass: re-read what the workspace already holds.

    Closes the gap where Carmel could acquire a paper and never use it, and is the
    first time the deterministic grounding gate runs against real stored bytes.
    """

    def test_a_grounded_corpus_finding_lands_in_the_report(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.latest.mode == LiteraturePassMode.CORPUS
        assert len(report.findings) == 1, f"expected one grounded finding, got rejections: {report.rejected}"
        finding = report.findings[0]
        assert finding.evidence.artifact_sha256 == sha
        assert finding.grounding.grounded is True
        assert finding.run_id == report.run_id

    def test_the_pass_never_searches_and_never_fetches(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproducibility is the whole point: the input must be exactly the stored
        bytes, so a corpus pass that quietly reached the network would defeat it."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])

        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a corpus pass reached the network")

        monkeypatch.setattr(deps.search, "search", _forbidden)
        monkeypatch.setattr(deps.fetch, "fetch", _forbidden)

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1, "the pass must still work with the network forbidden"
        assert deps.ledger.usage().fetches == 0
        assert deps.ledger.usage().index_lookups == 0
        assert report.queries == []

    def test_a_quote_absent_from_the_named_document_is_rejected(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate running against real stored bytes, which is the thing this whole
        increment exists to exercise."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        fabricated = "The measured ignition delay time was 9.99 ms at 2500 K under these conditions."
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha, quote=fabricated)])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.grounded is False

    def test_quoting_one_document_while_naming_another_is_rejected(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worst error available in corpus mode: real quote, real document, wrong
        pairing. It would attach one paper's evidence to another paper's citation,
        producing a finding that looks fully grounded and is entirely false."""
        _patch_chem_success(monkeypatch)
        sha_one = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        sha_two = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        assert sha_one != sha_two
        # A verbatim quote from document TWO, attributed to document ONE. One
        # proposal per document, since the pass makes one model call per document.
        deps, _, config = _make_deps(
            [
                _corpus_proposal([_corpus_finding(sha256=sha_one, quote=SECOND_QUOTE)]),
                _corpus_proposal([]),
                _assessment(),
            ]
        )

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == [], "a quote from a different document must never ground"
        assert len(report.rejected) == 1

    def test_a_finding_naming_an_unheld_document_is_rejected_not_dropped(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader of the report must be able to see that the agent claimed evidence
        which does not exist. Silently discarding it would hide that."""
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        absent = "0" * 64
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=absent)])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.NO_ARTIFACT
        assert absent in report.rejected[0].reason

    def test_an_unreadable_held_document_does_not_queue_an_acquisition(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one rejection reason that DOES queue an acquisition in a search pass --
        an artifact Carmel could not read -- must not do so here.

        In a search pass, "unreadable" means the fetched copy was bad and a human with
        a subscription might obtain a better one. In a corpus pass the document is
        already in the workspace, so asking a human to go and get it is nonsense: the
        remedy is mechanical re-extraction, not acquisition. Without this, a corpus
        pass would refill the operator's queue with papers they already supplied.
        """
        _patch_chem_success(monkeypatch)
        # Extraction that lost word spacing: real bytes, genuinely unreadable text.
        run_together = "Mechanismandkineticsoftheisothermaloxidationofoxygeninshocktubesatelevatedpressure" * 40
        sha = _store(campaign.workspace_root, text=run_together, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert len(report.rejected) == 1
        assert report.rejected[0].grounding.status == GroundingStatus.ARTIFACT_UNREADABLE
        assert load_manifest(campaign.workspace_root).requests == [], (
            "a corpus pass must never queue acquisition for a paper the workspace already holds"
        )

    def test_a_held_artifact_that_cannot_be_read_is_named_in_the_warnings(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spar round 7, P2. An artifact whose text will not load is correctly skipped,
        but skipping it silently makes a partial pass indistinguishable from a complete
        one that found nothing -- and the two call for opposite responses.

        Coverage the operator cannot see is coverage they will assume.
        """
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        broken = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/broken")
        # Remove the text sidecars but KEEP meta.json: list_artifacts still yields the
        # artifact (it is genuinely held), while load_artifact_text returns None. That
        # is exactly the shape of a corrupt or half-written extraction, and the shape
        # the pass must not paper over. Deleting meta.json instead would make the
        # artifact vanish from the listing entirely, which tests nothing.
        for name in ("extracted.json", "text.txt"):
            path = campaign.workspace_root / "evidence" / "literature" / broken / name
            if path.exists():
                path.unlink()
        # No proposed findings, so the run needs no verifier response: this test is
        # about coverage reporting, not about grounding.
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert any(broken[:12] in w for w in report.latest.warnings), (
            f"the unreadable artifact was skipped without saying so: {report.latest.warnings}"
        )
        assert any("NOT covered" in w for w in report.latest.warnings)

    def test_an_artifact_whose_bytes_no_longer_match_its_digest_is_not_read(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spar round 8, P0. The search pass fetches and extracts in one breath, so its
        text is necessarily fresh. A corpus pass re-reads sidecars of arbitrary age --
        the one place where "the gate runs against content-addressed bytes" can quietly
        stop being true.

        Truncated raw bytes (a full disk, an interrupted write) no longer hash to the
        directory naming them, and such an artifact must be refused and reported, not
        silently grounded against.
        """
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        tampered = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/tampered")
        raw = campaign.workspace_root / "evidence" / "literature" / tampered / "raw.bin"
        raw.write_bytes(b"these are not the bytes this digest names")
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert any(tampered[:12] in w for w in report.latest.warnings), (
            f"the artifact failed verification but was not reported: {report.latest.warnings}"
        )
        assert all(f.evidence.artifact_sha256 != tampered for f in report.findings), (
            "a finding was grounded against bytes that do not match their digest"
        )

    def test_an_empty_corpus_stops_without_calling_the_model(self, campaign: Campaign) -> None:
        """Nothing to read is an honest outcome, and must not cost a model call."""
        deps, model, config = _make_deps([])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.latest.stop_reason == StopReason.NO_NEW_INFORMATION
        assert report.findings == []
        assert deps.ledger.usage().model_calls == 0
        assert any("nothing to read" in w for w in report.latest.warnings)

    def test_a_corpus_pass_appends_to_an_existing_search_report(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decision 0004/D1 end to end: the second pass adds to the record rather
        than replacing it, and the two passes are distinguishable by mode."""
        _patch_chem_success(monkeypatch)
        search_deps, _, search_config = _make_deps([_proposal(queries=["first query"], done=True)])
        first = run_literature_research(campaign.workspace_root, campaign, _action(), search_deps, config=search_config)

        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])
        second_action = _action().model_copy(update={"action_id": "lit-a2"})

        report = run_corpus_pass(campaign.workspace_root, campaign, second_action, deps, config=config)

        assert [p.mode for p in report.passes] == [LiteraturePassMode.SEARCH, LiteraturePassMode.CORPUS]
        assert [p.run_id for p in report.passes] == [first.run_id, report.run_id]
        assert [q.text for q in report.queries] == ["first query"], "a corpus pass contributes no queries"
        assert report.findings_for(report.run_id) == report.findings


class TestTheDailyCapIsActuallyWired:
    """The daily cost cap was configurable but unenforced in production.

    ``build_deps`` defaults ``daily_ledger_path`` to ``None`` and ``BudgetLedger``
    reads ``None`` as "no daily cap", skipping the check outright. The dispatcher --
    the only production caller -- passed nothing, so every literature action ran with
    ``daily_max_cost_usd`` silently inert. The bug was invisible precisely because
    ``_apply_action_budget`` looks like it protects the daily ceiling: it carries the
    path over when rebuilding the ledger, but it was carrying over ``None``.

    This asserts the path reaches the ledger, which is the thing that was missing.
    Asserting only that ``default_daily_ledger_path()`` returns a path would pass
    against the broken code, since the resolver was never the problem.
    """

    def test_a_dispatched_literature_run_gets_a_daily_ledger_path(
        self, monkeypatch: pytest.MonkeyPatch, campaign: Campaign
    ) -> None:
        """The load-bearing assertion: the path reaches ``build_deps`` on the real
        dispatch path. Asserting only that the resolver returns a path would pass
        against the broken code, because the resolver was never what was missing."""
        from carmel.config import AgentConfig
        from carmel.schemas.approval import ActionKind
        from carmel.services import literature as literature_mod
        from carmel.services.dispatcher import make_literature_handler

        seen: dict[str, object] = {}

        class _Stop(Exception):
            pass

        def _spy(config: object, *, daily_ledger_path: object = None) -> object:
            seen["daily_ledger_path"] = daily_ledger_path
            raise _Stop  # the wiring is all this test needs; do not build a real stack

        monkeypatch.setattr(literature_mod, "build_deps", _spy)

        handler = make_literature_handler(agent_config=AgentConfig(), literature_deps=None)
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS, "estimated_tokens": 5_000})

        with pytest.raises(_Stop):
            handler(campaign.workspace_root, campaign, action)

        assert seen["daily_ledger_path"] is not None, (
            "the dispatcher built the ledger without a daily path, so daily_max_cost_usd "
            "is configurable but never enforced"
        )

    def test_the_resolver_never_returns_none(self) -> None:
        """A resolver that could return ``None`` would make the cap quietly optional,
        which is exactly how it came to be unenforced while looking configured."""
        from carmel.paths import default_daily_ledger_path

        assert default_daily_ledger_path() is not None

    def test_the_env_var_overrides_the_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from carmel.paths import DAILY_LEDGER_ENV_VAR, default_daily_ledger_path

        target = tmp_path / "elsewhere" / "ledger.json"
        monkeypatch.setenv(DAILY_LEDGER_ENV_VAR, str(target))

        assert default_daily_ledger_path() == target

    def test_resolving_the_path_creates_nothing_on_disk(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A read-only query must have no side effect; the ledger creates the file on
        first write."""
        from carmel.paths import DAILY_LEDGER_ENV_VAR, default_daily_ledger_path

        target = tmp_path / "untouched" / "ledger.json"
        monkeypatch.setenv(DAILY_LEDGER_ENV_VAR, str(target))

        default_daily_ledger_path()

        assert not target.exists()
        assert not target.parent.exists()


class TestOperatorBudgetBinds:
    """Decision 0004/D2: the budget named when appending the action must bound the
    run, not merely be recorded on it."""

    def test_the_actions_budget_replaces_the_config_ceiling(self, campaign: Campaign) -> None:
        """Without this the number shows up in the plan as if it were a ceiling while
        the run spends up to whatever the config file allows. A safety number that
        does not bind is worse than none, because it is believed."""
        from carmel.schemas.approval import ActionKind
        from carmel.services.dispatcher import _apply_action_budget

        deps, _, config = _make_deps([])
        assert deps.config.budget.max_tokens != 5_000
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS, "estimated_tokens": 5_000})

        bound = _apply_action_budget(deps, action)

        assert bound.config.budget.max_tokens == 5_000
        # The ledger must actually refuse above the operator's number, not merely
        # carry it: this is the assertion that would fail if the budget were wired
        # into the config and not into the thing that does the gating.
        with pytest.raises(BudgetExceededError):
            bound.ledger.reserve_model_call(estimated_tokens=6_000, estimated_cost_usd=0.01)
        # Session and daily ceilings are machine-wide protections; one operator
        # authorising one action does not authorise breaching them.
        assert bound.config.budget.session_max_cost_usd == deps.config.budget.session_max_cost_usd
        assert bound.config.budget.daily_max_cost_usd == deps.config.budget.daily_max_cost_usd

    def test_an_absent_budget_is_refused_rather_than_defaulted(self, campaign: Campaign) -> None:
        """Spar round 7, P1. This used to fall back to the config file's ceiling.

        For a corpus pass the operator's ``--budget-tokens`` IS the authorisation, so an
        action that reached the handler without one did not come from
        ``append_corpus_pass_action`` (which refuses it) -- it came from a hand-edited
        or tampered plan. Spending up to a ceiling nobody named is the wrong direction
        to fail for a control whose entire purpose is to bound spend.
        """
        from carmel.schemas.approval import ActionKind
        from carmel.services.dispatcher import _apply_action_budget

        deps, _, _ = _make_deps([])
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS, "estimated_tokens": 0})

        with pytest.raises(ValueError, match="no positive budget"):
            _apply_action_budget(deps, action)

    def test_the_rebuilt_ledger_keeps_the_daily_and_session_ceilings(self, campaign: Campaign) -> None:
        """Spar round 7, P1. Binding the operator's per-action ceiling must not switch
        the aggregate ones off.

        ``BudgetLedger(budget)`` built bare leaves ``daily_ledger_path=None``, silently
        disabling the file-backed daily cap for exactly the runs an operator has just
        authorised extra money for -- while the function's docstring promised the
        opposite. A ceiling believed to hold is worse than one known to be absent.
        """
        import dataclasses

        from carmel.agents.budget import BudgetLedger
        from carmel.schemas.approval import ActionKind
        from carmel.services.dispatcher import _apply_action_budget

        deps, _, _ = _make_deps([])
        daily = campaign.workspace_root / "daily_ledger.json"
        deps = dataclasses.replace(deps, ledger=BudgetLedger(deps.config.budget, daily_ledger_path=daily))
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS, "estimated_tokens": 250_000})

        bound = _apply_action_budget(deps, action)

        assert bound.config.budget.max_tokens == 250_000, "the operator ceiling did not bind"
        assert bound.ledger.daily_ledger_path == daily, "the daily cap was silently dropped"
        assert bound.ledger.session is deps.ledger.session, "the session cap was silently dropped"


class TestOutcomeReflectsThisPassOnly:
    """Spar round 7, P1. ``report.findings`` accumulates across every pass, so judging
    the outcome on it lets one old finding make every later barren pass look SUCCEEDED.

    A barren pass is exactly the signal an operator needs -- it says the corpus is
    exhausted or the prompt is wrong -- so it must not be masked by history.
    """

    def _report(self, *, old: int, new: int) -> LiteratureReport:
        from carmel.schemas.literature import (
            GroundingStatus,
            LiteratureFinding,
            LiteraturePassMode,
            PassRecord,
            QueryRecord,
        )

        now = datetime.now(UTC)

        def _pass(run_id: str, mode: LiteraturePassMode) -> PassRecord:
            return PassRecord(
                run_id=run_id,
                action_id=f"act-{run_id}",
                created_at=now,
                mode=mode,
                model_name="mock",
                stop_reason=StopReason.SELF_TERMINATED,
                usage=BudgetUsage(model_calls=0, tokens=0, cost_usd=0.0, fetches=0, fetch_bytes=0, elapsed_s=0.0),
            )

        def _f(run_id: str, n: int) -> LiteratureFinding:
            return LiteratureFinding.model_validate(
                {
                    **{k: v for k, v in _finding_dict().items() if k != "source_url"},
                    "finding_id": f"{run_id}-{n}",
                    "run_id": run_id,
                    "action_id": f"act-{run_id}",
                    "evidence": {
                        "artifact_sha256": "a" * 64,
                        "quote_start": 0,
                        "quote_end": len(QUOTE),
                    },
                    "grounding": {
                        "status": GroundingStatus.GROUNDED_EXACT,
                        "grounded": True,
                        "match_ratio": 1.0,
                        "identity_ok": True,
                    },
                }
            )

        findings = [_f("run-old", i) for i in range(old)] + [_f("run-new", i) for i in range(new)]
        return LiteratureReport(
            report_id="r1",
            campaign_id="c1",
            created_at=now,
            passes=[
                _pass("run-old", LiteraturePassMode.SEARCH),
                _pass("run-new", LiteraturePassMode.CORPUS),
            ],
            queries=[QueryRecord(text="q", run_id="run-old", action_id="act-run-old")],
            artifacts=[],
            findings=findings,
            rejected=[],
        )

    def test_a_barren_pass_after_a_productive_one_reports_no_grounded_findings(self) -> None:
        from carmel.schemas.action_state import ActionOutcome
        from carmel.schemas.approval import ActionKind
        from carmel.services.dispatcher import _literature_outcome

        report = self._report(old=1, new=0)
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS})

        assert report.findings, "fixture is wrong: the accumulated report must not be empty"
        assert _literature_outcome(report, action) == ActionOutcome.NO_GROUNDED_FINDINGS

    def test_a_pass_that_grounds_something_still_reports_success(self) -> None:
        from carmel.schemas.action_state import ActionOutcome
        from carmel.schemas.approval import ActionKind
        from carmel.services.dispatcher import _literature_outcome

        report = self._report(old=1, new=1)
        action = _action().model_copy(update={"kind": ActionKind.LITERATURE_CORPUS_PASS})

        assert _literature_outcome(report, action) == ActionOutcome.SUCCEEDED


class TestCorpusPassReadsOneDocumentPerCall:
    """Measured, not stylistic.

    Handing the model all 8 papers of the live syngas campaign in one 116k-token
    prompt produced ZERO proposed findings. The same model, same prompt, one of those
    papers alone: two proposals. A corpus big enough to be worth re-reading is big
    enough to bury the instruction to quote from it.
    """

    def test_each_document_gets_its_own_model_call(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        deps, model, config = _make_deps([_corpus_proposal([]), _corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 2, "one call per document, not one call for the corpus"
        assert report.latest.stop_reason != StopReason.ERROR

    def test_a_prompt_shows_exactly_one_document(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        """The isolation is the point: if both documents appeared in every prompt,
        per-document calls would cost twice as much and change nothing."""
        _patch_chem_success(monkeypatch)
        sha_one = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        sha_two = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        deps, _, config = _make_deps([_corpus_proposal([]), _corpus_proposal([])])

        prompts: list[str] = []
        original = deps.model.complete

        def _spy(**kwargs: Any) -> Any:
            prompts.append(kwargs["user_prompt"])
            return original(**kwargs)

        monkeypatch.setattr(deps.model, "complete", _spy)
        run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(prompts) == 2
        for prompt in prompts:
            shown = [s for s in (sha_one, sha_two) if s in prompt]
            assert len(shown) == 1, "each call must show exactly one document"

    def test_findings_from_earlier_documents_survive_a_later_failure(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other reason for per-document calls: exhausting the budget on document
        six keeps documents one to five. A single call would lose everything."""
        _patch_chem_success(monkeypatch)
        sha_one = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        # Only ONE proposal queued: the second document's call exhausts the mock and
        # raises, standing in for a budget or provider failure partway through.
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha_one)]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1, "the first document's grounded finding must survive"
        assert report.latest.stop_reason == StopReason.ERROR
