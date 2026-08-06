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
from carmel.agents.tools.academic import OaLookupCoverage, OaResolution, OpenAccessResolver
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
    CURRENT_REPORT_SCHEMA_VERSION,
    ROOT_EXTRACTION_ID,
    ArtifactProvenance,
    CorpusReadOutcome,
    CoveredDocument,
    GroundingStatus,
    LiteraturePassMode,
    LiteratureReport,
    PassRecord,
    StopReason,
)
from carmel.services import chem
from carmel.services import literature as literature_module
from carmel.services.acquisition import (
    _REASON_PHRASES,
    README_NAME,
    REQUESTS_DIR,
    inbox_dir,
    load_manifest,
    record_request,
)
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
from tests.test_acquisition import _matching_body, _patch_text_sniff_to_pdf

DOI = "10.1000/test.doi"
# An admissible host: production refuses to auto-admit documents from hosts
# that are not recognised publishers/repositories/resolvers, so a fixture on
# example.com would exercise the refusal rather than the path under test.
SOURCE_URL = "https://arxiv.org/pdf/2401.00001v1"

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
            def get(self, key: str, default: object = None) -> object:
                return "/Page" if key == "/Type" else default

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

    @pytest.mark.parametrize(
        "bad_extraction_id",
        [
            pytest.param("root2", id="short_string"),
            pytest.param("a" * 63 + "g", id="non_hex_char"),
            pytest.param("A" * 64, id="uppercase_hex"),
            # Python's ``$`` matches just before a trailing newline, so a validator
            # written with ``re.match`` accepts this and carries a value that is not a
            # sha256 around as though it were one. ``fullmatch`` is what refuses it.
            pytest.param("a" * 64 + "\n", id="trailing_newline"),
        ],
    )
    def test_t9_covered_document_rejects_a_bad_extraction_id(self, bad_extraction_id: str) -> None:
        """T9. ``extraction_id`` must be either the ``"root"`` sentinel or 64
        lowercase-hex characters -- anything else is refused, not coerced."""
        with pytest.raises(ValueError):
            CoveredDocument(raw_sha256="a" * 64, extraction_id=bad_extraction_id)

    @pytest.mark.parametrize(
        "bad_raw_sha256",
        [
            pytest.param("deadbeef", id="short_string"),
            pytest.param("a" * 63 + "g", id="non_hex_char"),
            pytest.param("A" * 64, id="uppercase_hex"),
            pytest.param("a" * 64 + "\n", id="trailing_newline"),
        ],
    )
    def test_t10_covered_document_rejects_a_bad_raw_sha256(self, bad_raw_sha256: str) -> None:
        """T10. ``raw_sha256`` must be 64 lowercase-hex characters."""
        with pytest.raises(ValueError):
            CoveredDocument(raw_sha256=bad_raw_sha256, extraction_id=ROOT_EXTRACTION_ID)

    def test_t10b_covered_document_refuses_the_retired_verified_deep_value(self) -> None:
        """``CoveredDocument`` must refuse a ``verification_standard`` of the OLD,
        retired value ``"verified_deep"``. That vocabulary was retired precisely
        because it overclaimed (it named a conclusion, not the check actually
        performed), so a permanent record must not silently accept it -- an operator
        reading `verified_deep` off a report years later would believe something the
        rename exists to stop anyone from claiming."""
        with pytest.raises(ValueError, match="invalid verification_standard"):
            CoveredDocument(
                raw_sha256="a" * 64,
                extraction_id=ROOT_EXTRACTION_ID,
                verification_standard="verified_deep",
            )

    def test_t10c_covered_document_refuses_a_standard_naming_a_document_never_read(self) -> None:
        """Every ``CorpusReadOutcome`` that names a REFUSAL must be rejected here.

        A coverage entry asserts a document WAS read; a standard like
        ``extraction_record_authentication_failed`` or
        ``multiple_current_extraction_records`` says it was not. Accepting one would let
        a report permanently claim coverage of text nobody ever served -- and coverage
        drives what a later pass treats as already done, so the document would never be
        looked at again.

        Loops over the refusal members rather than spot-checking one, so a member added
        later is caught here instead of discovered in a report years on.
        """
        refusals = [
            CorpusReadOutcome.EXTRACTION_RECORD_AUTHENTICATION_FAILED,
            CorpusReadOutcome.MULTIPLE_CURRENT_EXTRACTION_RECORDS,
            CorpusReadOutcome.INTEGRITY_FAILED,
            CorpusReadOutcome.MISSING_TEXT,
            CorpusReadOutcome.UNREADABLE_META,
        ]
        for refusal in refusals:
            with pytest.raises(ValueError, match="invalid verification_standard"):
                CoveredDocument(
                    raw_sha256="a" * 64,
                    extraction_id=ROOT_EXTRACTION_ID,
                    verification_standard=refusal.value,
                )

    def test_t11_pass_record_rejects_the_old_covered_sha256_key(self) -> None:
        """T11. ``PassRecord`` sets ``extra="forbid"``, so a payload still carrying
        the retired ``covered_sha256`` key proves the rename is real, not merely
        additive."""
        with pytest.raises(ValueError):
            PassRecord.model_validate(
                {
                    "run_id": "r",
                    "action_id": "a",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered_sha256": ["a" * 64],
                }
            )

    def test_t12_a_report_round_trips_several_covered_pairs(self, campaign: Campaign) -> None:
        """T12. A report written with a pass carrying several covered pairs reads
        back with exactly those pairs -- schema round trip, not just construction."""
        pairs = [
            CoveredDocument(
                raw_sha256="a" * 64,
                extraction_id=ROOT_EXTRACTION_ID,
                verification_standard=CorpusReadOutcome.SELF_CONSISTENT_METADATA.value,
            ),
            CoveredDocument(
                raw_sha256="b" * 64,
                extraction_id="c" * 64,
                verification_standard=CorpusReadOutcome.SELF_CONSISTENT_METADATA.value,
            ),
        ]
        pass_record = PassRecord(
            run_id="r",
            action_id="a",
            created_at=datetime.now(UTC),
            mode=LiteraturePassMode.CORPUS,
            model_name="mock",
            stop_reason=StopReason.SELF_TERMINATED,
            usage=BudgetUsage(model_calls=1, tokens=100, cost_usd=0.01, fetches=0, fetch_bytes=0, elapsed_s=1.0),
            warnings=[],
            covered=pairs,
        )
        report = LiteratureReport(
            report_id="rep-1",
            campaign_id=campaign.campaign_id,
            created_at=datetime.now(UTC),
            passes=[pass_record],
        )
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(report.model_dump_json())

        reloaded = load_literature_report(campaign.workspace_root)

        assert reloaded.passes[0].covered == pairs


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

    @pytest.fixture(autouse=True)
    def _pdf_sniff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `_queue_and_drop` writes a plain-text body to stand in for "the paper the
        # operator dropped" -- this class is about the inbox-collection wiring at the
        # start of a run, not format gating, which is exercised on its own in
        # tests/test_acquisition.py::TestPlainTextLandingPageRefused.
        _patch_text_sniff_to_pdf(monkeypatch)

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
            campaign.workspace_root,
            _matching_body("Abstract: measurements follow.\n", title=TITLE_FOR_DROP, doi=DOI),
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


def _fully_resolved(candidates: tuple[str, ...], note: str) -> OaResolution:
    """An :class:`OaResolution` from a lookup that fully ran -- the common fixture shape.

    ``coverage`` is deliberately required with no default, so every construction site
    must state what it observed. Most fixtures below exercise the FETCH path and only
    need a resolution that ran; they say so once through this helper. The tests that are
    about coverage itself build :class:`OaResolution` directly, so the value they turn on
    stays visible in the test body.
    """
    return OaResolution(candidates=candidates, note=note, coverage=OaLookupCoverage.COMPLETE)


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
            doi, _fully_resolved((), "no open-access copy advertised by any open-access index")
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
            {WANTED_DOI: _fully_resolved((OA_PDF_URL,), "OpenAlex: 1 OA PDF candidate")}
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
        second = "https://zenodo.org/records/1/files/green.pdf"
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL, second), "2 candidates")})
        fetch = _RoutedFetchTool(ok={second: (DOC.encode(), "text/plain")}, statuses={OA_PDF_URL: 404})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = fetch

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert fetch.fetched == [OA_PDF_URL, second]
        assert load_manifest(campaign.workspace_root).requests == []
        assert [a.sha256 for a in report.artifacts] == [hashlib.sha256(DOC.encode()).hexdigest()]

    def test_a_403_on_the_oa_copy_is_queued_as_an_observed_paywall(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL,), "1 candidate")})
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

    def test_an_oa_candidate_on_an_unrecognised_host_is_never_fetched(self, campaign: Campaign) -> None:
        """F18. An OA index advertising a URL is not the same as vouching for it.

        The identity gate confirms a document IS the cited work by finding the title
        and DOI outside its reference list -- which a document that merely PRINTS
        another paper's title and DOI also satisfies. Rather than tighten that gate on
        a threshold calibrated against eight documents, keep documents of unknown
        provenance out of the store.
        """
        hostile = "https://evil.example.net/looks-like-a-paper.pdf"
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((hostile,), "1 candidate")})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        fetched: list[str] = []

        class _RecordingFetch:
            def fetch(self, url: str) -> tuple[object, bytes]:
                fetched.append(url)
                raise AssertionError(f"an inadmissible host was contacted: {url}")

        deps.fetch = _RecordingFetch()  # type: ignore[assignment]

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert fetched == [], "the host was contacted before being refused"
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.HOST_NOT_ADMISSIBLE]
        # Queued, not dropped: the human-gated path runs its own identity check.
        assert requests[0].status == AcquisitionStatus.REQUESTED

    def test_an_operator_can_admit_an_extra_host(self, campaign: Campaign) -> None:
        """The extension point an institutional proxy or lab mirror needs."""
        from carmel.services.acquisition import host_is_admissible

        url = "https://proxy.my-university.edu/paper.pdf"
        assert not host_is_admissible(url)
        assert host_is_admissible(url, ["proxy.my-university.edu"])
        # Subdomains of an admitted host are admitted; lookalike suffixes are not.
        assert host_is_admissible("https://a.b.proxy.my-university.edu/p.pdf", ["proxy.my-university.edu"])
        assert not host_is_admissible("https://proxy.my-university.edu.evil.net/p.pdf", ["proxy.my-university.edu"])

    def test_a_404_on_the_oa_copy_is_queued_as_fetch_failed_not_paywalled(self, campaign: Campaign) -> None:
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL,), "1 candidate")})
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=resolver)
        deps.fetch = _StatusFetchTool(404)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.FETCH_FAILED]
        assert "HTTP 404" in requests[0].detail

    def test_an_observed_paywall_wins_over_an_earlier_broken_link(self, campaign: Campaign) -> None:
        """403 is the reason the operator can act on (a subscription): if ANY candidate
        observed one, that is the request's reason, not whichever failure came first."""
        second = "https://arxiv.org/pdf/2401.00002v1"
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL, second), "2 candidates")})
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
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL,), "1 candidate")})
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
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL,), "1 candidate")})
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
        resolver = _FakeOaResolver({WANTED_DOI: _fully_resolved((OA_PDF_URL,), "1 candidate")})
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
            coverage=OaLookupCoverage.PARTIAL,
        )
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=_FakeOaResolver({WANTED_DOI: truncated}))

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.OA_LOOKUP_INCOMPLETE]
        assert "lookup failed" in requests[0].detail

    def test_a_completed_resolution_that_found_nothing_is_still_no_open_access_copy(self, campaign: Campaign) -> None:
        """The complement of the test above -- the honest negative must survive."""
        exhausted = OaResolution(
            candidates=(),
            note="OpenAlex: 0 OA PDF candidates; Unpaywall: 0 OA PDF candidates",
            coverage=OaLookupCoverage.COMPLETE,
        )
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=_FakeOaResolver({WANTED_DOI: exhausted}))

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.NO_OPEN_ACCESS_COPY]

    def test_without_a_resolver_the_paper_is_still_queued_without_asserting_a_paywall(self, campaign: Campaign) -> None:
        """Even with no resolver wired (mock tier), neither 'paywalled' nor 'no open-access
        copy was found' must be recorded: nothing observed a paywall, and no lookup ran at
        all, so the honest state is OA_LOOKUP_NOT_ATTEMPTED, not the honest-negative
        NO_OPEN_ACCESS_COPY -- that reason is reserved for a lookup that actually ran and
        came back empty.
        """
        deps, _, config = _make_deps([_wanted_proposal()], oa_resolver=None)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED]
        assert requests[0].reason != AcquisitionReason.NO_OPEN_ACCESS_COPY
        assert requests[0].reason != AcquisitionReason.OA_LOOKUP_INCOMPLETE
        assert "resolver" in requests[0].detail

        readme = (campaign.workspace_root / REQUESTS_DIR / README_NAME).read_text(encoding="utf-8")
        assert "no open-access copy was found" not in readme

    def test_a_wanted_paper_without_a_doi_skips_resolution(self, campaign: Campaign) -> None:
        """A paper with no DOI is not a special case that earns the honest-negative
        NO_OPEN_ACCESS_COPY: the title-matched providers (arXiv, ChemRxiv) exist
        precisely so a paper can be found without a DOI, so no lookup ran and the honest
        state is OA_LOOKUP_NOT_ATTEMPTED.
        """
        resolver = _FakeOaResolver()
        proposal = _proposal(findings=[])
        proposal["wanted"] = [{"title": WANTED_TITLE, "landing_url": "https://example.org/paper"}]
        deps, _, config = _make_deps([proposal], oa_resolver=resolver)

        run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert resolver.calls == []
        requests = load_manifest(campaign.workspace_root).requests
        assert [r.reason for r in requests] == [AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED]
        assert requests[0].reason != AcquisitionReason.NO_OPEN_ACCESS_COPY
        assert requests[0].reason != AcquisitionReason.OA_LOOKUP_INCOMPLETE
        assert "no DOI" in requests[0].detail

    def test_consent_withheld_makes_zero_network_calls_end_to_end(self, campaign: Campaign) -> None:
        """A run with consent withheld, wired with the REAL resolver and REAL fetch
        tool, must never open a socket: the booby-trapped opener proves it.

        The consent-withheld branch inside ``OpenAccessResolver.resolve`` is
        unreachable in production today -- ``build_deps`` returns early with no
        resolver at all for the mock provider (the default), and for any real
        provider ``build_model`` raises ``AgentBridgeError`` before this resolver is
        ever constructed when consent is withheld. This test constructs the resolver
        directly, the same way ``_make_deps`` wires it below, so it still reaches
        that branch; no provider ran, so the honest state is OA_LOOKUP_NOT_ATTEMPTED,
        not the honest-negative NO_OPEN_ACCESS_COPY.
        """

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
        assert [r.reason for r in requests] == [AcquisitionReason.OA_LOOKUP_NOT_ATTEMPTED]
        assert requests[0].reason != AcquisitionReason.NO_OPEN_ACCESS_COPY
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

    def test_every_acquisition_reason_has_an_operator_facing_phrase(self) -> None:
        """Total mapping check: `_REASON_PHRASES` must cover every member of
        `AcquisitionReason`, `OA_LOOKUP_NOT_ATTEMPTED` included. The dict has no
        `else`/default branch by design -- an unmapped reason must fail loudly
        (`KeyError`) at render time rather than silently leaking `reason.value`, so
        this test is the only thing that catches a newly added reason before an
        operator does.
        """
        assert set(_REASON_PHRASES) == set(AcquisitionReason)


class TestReportSchemaVersionGate:
    """Spar round 7, P2. The migration accepted any ``schema_version >= 2`` unchanged.

    A report from a FUTURE Carmel would then be handed to a validator that does not
    know its fields. Either it fails with a schema error naming a field the operator
    has never heard of, or -- worse, if the newer version only added optional fields --
    it validates cleanly and the next write silently drops them, downgrading a newer
    report in place. Refuse it instead and say what to do.
    """

    def test_a_future_schema_version_is_refused(self) -> None:
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION
        from carmel.services.literature import migrate_report_payload

        # Relative to the current version, so a bump does not turn this into a test
        # that the CURRENT version is refused.
        future = CURRENT_REPORT_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match=f"schema version {future}"):
            migrate_report_payload({"schema_version": future, "report_id": "r1"})

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

    def _v3_pass(self, *, covered_sha256: list[str]) -> dict[str, Any]:
        return {
            "run_id": "r",
            "action_id": "a",
            "created_at": datetime.now(UTC).isoformat(),
            "mode": LiteraturePassMode.CORPUS.value,
            "model_name": "mock",
            "stop_reason": StopReason.SELF_TERMINATED.value,
            "usage": {
                "model_calls": 1,
                "tokens": 100,
                "cost_usd": 0.01,
                "fetches": 0,
                "fetch_bytes": 0,
                "elapsed_s": 1.0,
            },
            "warnings": [],
            "covered_sha256": covered_sha256,
        }

    def test_t5_a_v3_payloads_covered_sha256_migrates_to_covered_pairs_with_root(self) -> None:
        """T5. A v3 payload's ``covered_sha256: [a, b]`` migrates to a v4 ``covered``
        equal to those two shas each paired with ``ROOT_EXTRACTION_ID``, and the
        payload's ``schema_version`` is the current one -- which, having also passed
        through the v5->v6 step, stamps each pair's ``verification_standard`` as
        ``"unrecorded"``."""
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION
        from carmel.services.literature import migrate_report_payload

        sha_a, sha_b = "a" * 64, "b" * 64
        payload = {"schema_version": 3, "passes": [self._v3_pass(covered_sha256=[sha_a, sha_b])]}

        migrated = migrate_report_payload(payload)

        assert isinstance(migrated, dict)
        assert migrated["schema_version"] == CURRENT_REPORT_SCHEMA_VERSION == 6
        assert migrated["passes"][0]["covered"] == [
            {"raw_sha256": sha_a, "extraction_id": ROOT_EXTRACTION_ID, "verification_standard": "unrecorded"},
            {"raw_sha256": sha_b, "extraction_id": ROOT_EXTRACTION_ID, "verification_standard": "unrecorded"},
        ]

    def test_t6_the_old_covered_sha256_key_is_gone_after_migration(self) -> None:
        """T6. ``covered_sha256`` must be removed, not left alongside ``covered`` --
        ``PassRecord`` forbids extra fields, so leaving it would break validation."""
        from carmel.services.literature import migrate_report_payload

        payload = {"schema_version": 3, "passes": [self._v3_pass(covered_sha256=["a" * 64])]}

        migrated = migrate_report_payload(payload)

        assert isinstance(migrated, dict)
        assert "covered_sha256" not in migrated["passes"][0]

    def test_t7_v1_and_v2_reports_still_migrate_to_an_empty_covered_list(self) -> None:
        """T7. What a v1/v2 pass covered was never written down, so migration must
        not invent it -- an empty ``covered`` list stays the honest answer."""
        from carmel.services.literature import migrate_report_payload

        v1_migrated = migrate_report_payload(
            {"schema_version": 1, "run_id": "r", "action_id": "a", "queries": []}
        )
        assert isinstance(v1_migrated, dict)
        assert v1_migrated["passes"][0]["covered"] == []

        v2_payload = {
            "schema_version": 2,
            "passes": [
                {
                    "run_id": "r",
                    "action_id": "a",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.SEARCH.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                }
            ],
        }
        v2_migrated = migrate_report_payload(v2_payload)
        assert isinstance(v2_migrated, dict)
        assert v2_migrated["passes"][0]["covered"] == []

    def test_t8_a_payload_already_at_the_current_version_passes_through_unchanged(self) -> None:
        """T8. A payload already at the current version (6) is idempotent -- returned
        as-is, not re-migrated."""
        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION
        from carmel.services.literature import migrate_report_payload

        assert CURRENT_REPORT_SCHEMA_VERSION == 6
        payload = {
            "schema_version": 6,
            "passes": [
                {
                    **self._v3_pass(covered_sha256=[]),
                    "covered": [
                        {
                            "raw_sha256": "a" * 64,
                            "extraction_id": ROOT_EXTRACTION_ID,
                            "verification_standard": "unrecorded",
                        }
                    ],
                }
            ],
        }
        del payload["passes"][0]["covered_sha256"]

        assert migrate_report_payload(payload) is payload


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

        from carmel.schemas.literature import CURRENT_REPORT_SCHEMA_VERSION

        assert report.schema_version == CURRENT_REPORT_SCHEMA_VERSION
        assert len(report.passes) == 1
        assert report.passes[0].mode == LiteraturePassMode.SEARCH
        # v1 recorded no coverage, so the migration must not invent any -- an empty
        # list reads as "not recorded" and makes the next corpus pass re-read once,
        # which is the conservative direction.
        assert report.passes[0].covered == []
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
        # Pin the DEFENCE, not merely the refusal. Asserting a rejection alone also
        # passes via NO_ARTIFACT -- a different mechanism entirely (the named sha is
        # not held) -- so a regression that stopped checking the quote against the
        # named document's bytes would still have shown green here.
        assert report.rejected[0].grounding.status == GroundingStatus.QUOTE_NOT_FOUND

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
        # the pass must not paper over. Destroying meta.json is a DIFFERENT skip path
        # (the artifact never enters the listing at all); the test below covers it.
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

    def test_an_artifact_with_no_readable_meta_is_reported_as_uncovered_too(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F11. The quietest way for a held paper to disappear.

        A directory whose ``meta.json`` will not parse never becomes a
        ``StoredArtifact``, so it cannot be skipped by the corpus loader -- it was
        missing from the corpus AND from the count of what the pass could not read.
        That reads to an operator as complete coverage of a smaller store, which is
        exactly the reading that makes a barren pass look conclusive.
        """
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        broken = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/nometa")
        meta_path = campaign.workspace_root / "evidence" / "literature" / broken / "meta.json"
        meta_path.write_text("{not json at all", encoding="utf-8")
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert any(broken[:12] in w for w in report.latest.warnings), (
            f"an artifact with unreadable meta.json vanished silently: {report.latest.warnings}"
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

    def test_a_pass_records_every_document_it_read(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        """Coverage is per DOCUMENT, not per finding.

        A document read and found barren produces nothing to attribute, and is
        exactly the document a later pass must not pay to re-read -- so findings
        cannot serve as the record of what was covered.
        """
        _patch_chem_success(monkeypatch)
        first = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        second = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        # Both proposals empty: neither document yields a finding.
        deps, _, config = _make_deps([_corpus_proposal([]), _corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert report.findings == []
        assert sorted(cd.raw_sha256 for cd in report.latest.covered) == sorted([first, second])

    def test_a_second_pass_does_not_re_read_what_the_first_already_mined(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F14. The prompt for a document is byte-identical between passes, so a
        re-read asks the same question and pays for the same answer twice."""
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        first_deps, _, first_config = _make_deps([_corpus_proposal([])])
        first = run_corpus_pass(campaign.workspace_root, campaign, _action(), first_deps, config=first_config)
        assert first_deps.ledger.usage().model_calls == 1
        assert len(first.latest.covered) == 1

        # A second pass with NO new documents must make no model call at all.
        second_deps, _, second_config = _make_deps([_corpus_proposal([])])
        second_action = _action().model_copy(update={"action_id": "lit-a2"})
        second = run_corpus_pass(campaign.workspace_root, campaign, second_action, second_deps, config=second_config)

        assert second_deps.ledger.usage().model_calls == 0, "the second pass re-read an already-mined document"
        assert second.latest.stop_reason == StopReason.NO_NEW_INFORMATION
        assert any("already been mined" in w for w in second.latest.warnings)

    def test_a_second_pass_reads_only_the_newly_acquired_document(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The starvation case: under a fixed budget an unscoped pass always re-read
        the same stable-ordered prefix, so later papers were never reached."""
        _patch_chem_success(monkeypatch)
        old = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        first_deps, _, first_config = _make_deps([_corpus_proposal([])])
        run_corpus_pass(campaign.workspace_root, campaign, _action(), first_deps, config=first_config)

        new = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        second_deps, _, second_config = _make_deps([_corpus_proposal([])])
        second_action = _action().model_copy(update={"action_id": "lit-a2"})
        report = run_corpus_pass(campaign.workspace_root, campaign, second_action, second_deps, config=second_config)

        assert second_deps.ledger.usage().model_calls == 1
        assert [cd.raw_sha256 for cd in report.latest.covered] == [new]
        assert old not in [cd.raw_sha256 for cd in report.latest.covered]

    def test_a_document_the_budget_refused_is_not_recorded_as_covered(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The document a truncated pass stops ON must stay readable.

        Coverage used to be recorded BEFORE the model call, so a pass that ran out
        of budget marked the document it never reached as covered: nothing was paid
        for it, it was never read, and every later pass skipped it while reporting
        that the whole corpus had been mined. Observed live 2026.08.01 -- a pass
        recorded 8 documents covered using 7 model calls, and the next pass then
        made zero calls and declared there was nothing new to read.

        This is the F14 skip turned against itself: the optimisation that stops a
        re-read is only safe if 'covered' means 'actually mined'.
        """
        _patch_chem_success(monkeypatch)
        _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        # One call is affordable; the second document's reservation is refused.
        deps, _, config = _make_deps(
            [_corpus_proposal([]), _corpus_proposal([])],
            budget=AgentBudgetConfig(max_model_calls=1),
        )

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 1
        assert report.latest.stop_reason == StopReason.MAX_MODEL_CALLS
        # The heart of it: one call read one document, so exactly one is covered.
        assert len(report.latest.covered) == 1, (
            f"a refused reservation was recorded as covered: {report.latest.covered}"
        )

        # And the refused document is still reachable -- not silently skipped forever.
        second_deps, _, second_config = _make_deps([_corpus_proposal([])])
        second_action = _action().model_copy(update={"action_id": "lit-a2"})
        second = run_corpus_pass(campaign.workspace_root, campaign, second_action, second_deps, config=second_config)

        assert second_deps.ledger.usage().model_calls == 1, "the unread document was skipped by the next pass"
        assert len(second.latest.covered) == 1

    def test_reread_all_overrides_the_scoping(self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch) -> None:
        """The operator's escape hatch: a changed model or prompt is a real reason."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        first_deps, _, first_config = _make_deps([_corpus_proposal([])])
        run_corpus_pass(campaign.workspace_root, campaign, _action(), first_deps, config=first_config)

        second_deps, _, second_config = _make_deps([_corpus_proposal([])])
        forced = _action().model_copy(update={"action_id": "lit-a2", "parameters": {"reread_all": True}})
        report = run_corpus_pass(campaign.workspace_root, campaign, forced, second_deps, config=second_config)

        assert second_deps.ledger.usage().model_calls == 1
        assert [cd.raw_sha256 for cd in report.latest.covered] == [sha]

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


class TestCoverageIsKeyedByExtractionIdentity:
    """Coverage must be keyed by (raw_sha256, extraction_id), not raw_sha256 alone.

    A single stored document can have more than one extraction (the root
    ``extracted.json`` sidecar today, plus per-extraction sidecars re-extraction can
    produce later). Recording coverage as a bare raw sha256 makes "this raw document
    was covered" indistinguishable from "this SPECIFIC extraction of it was covered"
    -- so a document mined under one extraction identity would be silently skipped
    forever, even once a materially different extraction of the same document became
    available to read.
    """

    def test_t1_red_a_document_covered_under_a_different_extraction_identity_is_not_skipped(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T1, and the RED test required before any non-test file was touched.

        Writes a previous report directly (rather than through a real pass) because
        this increment deliberately keeps ``_load_corpus`` root-sidecar-only (see
        Non-goals) -- it cannot itself discover a non-root extraction to read, so the
        only way to exercise "covered under a DIFFERENT extraction identity" is to
        assert it into the previous report's ``covered`` records directly.

        Against the schema this branch starts from, a bare raw sha256 is the entire
        coverage key: once ``sha`` is covered, EVERY reading of that raw document is
        skipped, with no way to say "but not that extraction." That is exactly the
        bug this test is written to prove.
        """
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        other_extraction_id = "b" * 64
        previous_payload = {
            "schema_version": 4,
            "report_id": "rep-prev",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [
                {
                    "run_id": "run-prev",
                    "action_id": "act-prev",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered": [{"raw_sha256": sha, "extraction_id": other_extraction_id}],
                }
            ],
            "queries": [],
            "artifacts": [],
            "findings": [],
            "rejected": [],
        }
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(previous_payload))
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 1, (
            "a document covered under a DIFFERENT extraction identity was skipped as "
            "already mined; raw sha256 alone is not a valid coverage key -- "
            f"warnings={report.latest.warnings}"
        )

    def test_t2_a_document_covered_under_the_same_extraction_identity_is_skipped(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T2. The mirror of T1: same raw sha256 AND same extraction identity ->
        the document is skipped, exactly as bare raw-sha coverage always was."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        previous_payload = {
            "schema_version": 4,
            "report_id": "rep-prev",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [
                {
                    "run_id": "run-prev",
                    "action_id": "act-prev",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered": [{"raw_sha256": sha, "extraction_id": ROOT_EXTRACTION_ID}],
                }
            ],
            "queries": [],
            "artifacts": [],
            "findings": [],
            "rejected": [],
        }
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(previous_payload))
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 0, (
            "a document covered under the SAME (raw sha256, extraction identity) pair "
            f"was re-read; warnings={report.latest.warnings}"
        )
        assert report.latest.stop_reason == StopReason.NO_NEW_INFORMATION

    def test_t3_only_the_uncovered_document_of_two_is_read(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T3. A corpus of two documents where exactly one (raw sha256, extraction
        identity) pair is covered -> only the uncovered document is read."""
        _patch_chem_success(monkeypatch)
        covered_sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        uncovered_sha = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.com/papers/second")
        previous_payload = {
            "schema_version": 4,
            "report_id": "rep-prev",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [
                {
                    "run_id": "run-prev",
                    "action_id": "act-prev",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered": [{"raw_sha256": covered_sha, "extraction_id": ROOT_EXTRACTION_ID}],
                }
            ],
            "queries": [],
            "artifacts": [],
            "findings": [],
            "rejected": [],
        }
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(previous_payload))
        deps, _, config = _make_deps([_corpus_proposal([])])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 1, (
            f"expected exactly one new document to be read; warnings={report.latest.warnings}"
        )
        assert [cd.raw_sha256 for cd in report.latest.covered] == [uncovered_sha]

    def test_t4_when_every_held_document_is_covered_the_pass_still_stops_as_already_mined(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T4. When every held document's (raw sha256, extraction identity) pair is
        covered, the pass still stops with the existing "already mined" stop reason
        and warning -- re-keying coverage must not disturb that outcome."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        previous_payload = {
            "schema_version": 4,
            "report_id": "rep-prev",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [
                {
                    "run_id": "run-prev",
                    "action_id": "act-prev",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered": [{"raw_sha256": sha, "extraction_id": ROOT_EXTRACTION_ID}],
                }
            ],
            "queries": [],
            "artifacts": [],
            "findings": [],
            "rejected": [],
        }
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(previous_payload))
        deps, _, config = _make_deps([])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert deps.ledger.usage().model_calls == 0
        assert report.latest.stop_reason == StopReason.NO_NEW_INFORMATION
        assert any("already" in w and "mined" in w for w in report.latest.warnings), (
            f"expected an 'already mined' warning; warnings={report.latest.warnings}"
        )


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
                        "extraction_id": ROOT_EXTRACTION_ID,
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


class TestTheReservationMatchesThePromptSize:
    """The bridge reserves a flat 8000 tokens by default. That was sized for a short
    search prompt and is badly wrong for a corpus prompt that embeds a whole paper --
    a single document ran ~25k tokens on the live corpus.

    This matters more since tokens became the operator's authorisation unit: the
    reservation IS the enforcement point, checked BEFORE the call. An understated
    reservation means the ceiling does not bind until after the call that breached it,
    which is the one moment it needed to.
    """

    def test_a_whole_document_prompt_reserves_far_more_than_the_flat_default(self) -> None:
        from carmel.services.literature import estimated_tokens_for

        # ~100k characters, the scale of one extracted paper.
        assert estimated_tokens_for("x" * 100_000) > 8000 * 3

    def test_a_short_prompt_never_reserves_below_the_bridge_default(self) -> None:
        """Sizing DOWN would be a regression: the default is also a floor covering the
        response, which the prompt length cannot predict."""
        from carmel.services.literature import estimated_tokens_for

        assert estimated_tokens_for("short") == 8000

    def test_the_estimate_grows_with_the_prompt(self) -> None:
        from carmel.services.literature import estimated_tokens_for

        assert estimated_tokens_for("x" * 200_000) > estimated_tokens_for("x" * 100_000)


class TestAFindingRecordsWhichExtractionItsOffsetsIndex:
    """``EvidenceRef`` must name the extraction its offsets were computed against.

    An ``EvidenceRef`` stores ``quote_start``/``quote_end`` as offsets into "the
    artifact's extracted text". That was unambiguous while an artifact had exactly one
    text. It stopped being so when the nested extraction-record store landed: one raw
    sha256 can now carry a root ``extracted.json`` sidecar AND any number of
    authenticated re-extraction records, each with its own text and therefore its own
    offsets for the same quote.

    Nothing reads these offsets today -- there is no literature replayer, only the
    dataset one -- so this is not a live mis-resolution. It is a capture defect, and
    that is what makes it urgent rather than deferrable: the report is append-only and
    stored character offsets are NEVER migrated, so a finding accepted without this
    identity can never afterwards be told which text it indexed. The only moment the
    information exists to be captured is the moment of acceptance.
    """

    #: A stand-in for an authenticated re-extraction record's address. The tests below
    #: patch ``_load_corpus`` rather than mint a real record, because teaching
    #: ``_load_corpus`` to SELECT a record is the next increment and is deliberately
    #: out of scope here -- what must be proven now is that whatever it read gets
    #: recorded on the finding.
    RECORD_ID = hashlib.sha256(b"a-re-extraction-record").hexdigest()

    #: The same quote, in a text where it sits at a DIFFERENT offset than in ``DOC``.
    #: The leading matter is what shifts it; without a shift the two texts would agree
    #: on the offsets and the ambiguity would be invisible.
    RECORD_DOC = (
        "A shock tube study of oxygen ignition\n"
        "J. Smith and A. Jones (2020)\n"
        f"doi: {DOI}\n\n"
        "Abstract text here.\n\n"
        "A paragraph recovered only by the newer extractor, which the root sidecar\n"
        "dropped entirely, and which therefore shifts everything after it.\n\n"
        f"{QUOTE}\n"
        "Further discussion of the measurements follows here.\n"
    )

    def _patch_corpus_to_read_a_record(
        self, monkeypatch: pytest.MonkeyPatch, *, extraction_id: str, text: str
    ) -> None:
        """Make ``_load_corpus`` report that it read a NON-root extraction.

        It calls the real loader and substitutes the text and identity, so the
        StoredArtifact, the digest verification and the skipped-sha bookkeeping are
        all still the production ones.
        """
        real = literature_module._load_corpus

        def _patched(workspace_root: Path, **kwargs: Any) -> Any:
            corpus, outcomes = real(workspace_root, **kwargs)
            return (
                [(artifact, extract_text(text.encode(), "text/plain"), extraction_id) for artifact, _, _ in corpus],
                outcomes,
            )

        monkeypatch.setattr(literature_module, "_load_corpus", _patched)

    def test_p1_red_offsets_from_a_non_root_extraction_are_recorded_against_that_extraction(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1, RED. THE ambiguity test.

        A pass reads a non-root extraction and accepts a finding from it. The offsets
        index the RECORD's text. Nothing on the finding says so, so a later reader
        holding only ``artifact_sha256`` would resolve them against the root sidecar
        and slice out the wrong span -- which the final assertion demonstrates is not
        a hypothetical: the same offsets against the root text do not yield the quote.
        """
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        self._patch_corpus_to_read_a_record(monkeypatch, extraction_id=self.RECORD_ID, text=self.RECORD_DOC)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1, f"expected one grounded finding, got rejections: {report.rejected}"
        evidence = report.findings[0].evidence
        assert evidence.artifact_sha256 == sha
        assert evidence.extraction_id == self.RECORD_ID, (
            "the finding was mined from a re-extraction record, so its offsets index that "
            "record's text -- the finding must say so"
        )
        start, end = evidence.quote_start, evidence.quote_end
        assert start is not None and end is not None
        assert self.RECORD_DOC[start:end] == QUOTE, "the recorded offsets must index the text actually read"
        assert DOC[start:end] != QUOTE, (
            "if the same offsets also resolved correctly against the root text there would be "
            "no ambiguity to close, and this test would prove nothing"
        )

    def test_p2_a_corpus_finding_from_the_root_sidecar_is_recorded_as_root(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P2. The honest case, and the guard against a fix that stamps every finding
        with a record id."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1, f"expected one grounded finding, got rejections: {report.rejected}"
        evidence = report.findings[0].evidence
        assert evidence.extraction_id == ROOT_EXTRACTION_ID
        start, end = evidence.quote_start, evidence.quote_end
        assert start is not None and end is not None
        assert DOC[start:end] == QUOTE

    def test_p3_a_search_pass_finding_is_recorded_as_root(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P3. The OTHER call site. A search pass fetches and extracts in one breath and
        stores the result as the root sidecar, so root is a fact about that path, not a
        default -- but it still has to be stated, because the field has no default."""
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps(
            [
                _proposal(findings=[_finding_dict()], done=True),
                _assessment(),
            ],
            search={"oxygen ignition delay shock tube": [SearchResult(title="t", url=SOURCE_URL, snippet="s")]},
        )

        report = run_literature_research(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert len(report.findings) == 1, f"expected one grounded finding, got rejections: {report.rejected}"
        assert report.findings[0].evidence.extraction_id == ROOT_EXTRACTION_ID

    def test_p4_an_evidence_ref_cannot_be_built_without_stating_an_extraction(self) -> None:
        """P4. No default. A default would let a producer that never considered the
        question silently claim the root -- which is the exact ambiguity being closed.
        ``CharSpanLocator.text_space`` makes the same argument for the dataset lane."""
        from pydantic import ValidationError

        from carmel.schemas.literature import EvidenceRef

        with pytest.raises(ValidationError, match="extraction_id"):
            EvidenceRef(artifact_sha256="a" * 64, quote_start=0, quote_end=5)  # type: ignore[call-arg]

    def test_p5_a_v4_report_migrates_every_evidence_ref_to_root(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P5. Every ref written before this field existed was necessarily root-derived:
        ``_ground_and_record``'s only two callers read text via ``_fetch_and_store``
        (fresh extraction, stored as the root sidecar) and ``_load_corpus`` (root
        sidecar only). Root here is a fact about what could have been read, not a
        default standing in for an unstated intent."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        start = DOC.index(QUOTE)
        legacy = self._legacy_v4_report(campaign, sha=sha, start=start)
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(legacy))

        report = load_literature_report(campaign.workspace_root)

        assert report is not None
        assert len(report.findings) == 1
        evidence = report.findings[0].evidence
        assert evidence.extraction_id == ROOT_EXTRACTION_ID
        assert evidence.quote_start == start, "a migration must never move a stored character offset"
        assert evidence.quote_end == start + len(QUOTE)

    @staticmethod
    def _legacy_v4_report(campaign: Campaign, *, sha: str, start: int) -> dict[str, Any]:
        """A v4 report holding one grounded finding whose evidence has no
        ``extraction_id`` -- i.e. exactly what was on disk before this field existed."""
        return {
            "schema_version": 4,
            "report_id": "rep-legacy",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [
                {
                    "run_id": "run-legacy",
                    "action_id": "act-legacy",
                    "created_at": datetime.now(UTC).isoformat(),
                    "mode": LiteraturePassMode.CORPUS.value,
                    "model_name": "mock",
                    "stop_reason": StopReason.SELF_TERMINATED.value,
                    "usage": {
                        "model_calls": 1,
                        "tokens": 100,
                        "cost_usd": 0.01,
                        "fetches": 0,
                        "fetch_bytes": 0,
                        "elapsed_s": 1.0,
                    },
                    "warnings": [],
                    "covered": [{"raw_sha256": sha, "extraction_id": ROOT_EXTRACTION_ID}],
                }
            ],
            "queries": [],
            "rejected": [],
            "findings": [
                {
                    "finding_id": "f-legacy",
                    "run_id": "run-legacy",
                    "action_id": "act-legacy",
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
                    "verbatim_quote": QUOTE,
                    "evidence": {
                        "artifact_sha256": sha,
                        "quote_start": start,
                        "quote_end": start + len(QUOTE),
                    },
                    "grounding": {
                        "status": GroundingStatus.GROUNDED_EXACT.value,
                        "grounded": True,
                        "match_ratio": 1.0,
                        "identity_ok": True,
                    },
                }
            ],
        }

    @pytest.mark.parametrize(
        "bogus",
        ["ROOT", "Root", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n", "", "  "],
    )
    def test_p7_an_extraction_id_that_is_neither_root_nor_a_sha256_is_refused(self, bogus: str) -> None:
        """P7. The field's SHAPE, not merely its presence.

        Found by a mutation audit: with the shape check neutered, the whole literature
        suite still passed, so nothing was certifying that this value even looks like an
        extraction identity. The trailing-newline case is the specific reason the check
        uses ``fullmatch`` rather than ``match`` -- Python's ``$`` also matches just
        BEFORE a trailing newline, so ``"a" * 64 + "\\n"`` would otherwise be accepted
        and then used as a filesystem path component.
        """
        from pydantic import ValidationError

        from carmel.schemas.literature import EvidenceRef

        with pytest.raises(ValidationError, match="extraction_id"):
            EvidenceRef(artifact_sha256="a" * 64, extraction_id=bogus)

    def test_p8_a_v4_payload_claiming_a_record_is_normalised_to_root_not_honoured(
        self, campaign: Campaign
    ) -> None:
        """P8. The migration OVERWRITES; it does not merely fill in a blank.

        No v4 writer could have had grounds to say "these offsets index record <sha>" --
        that impossibility IS the argument for stamping root -- and ``EvidenceRef``'s
        ``extra="forbid"`` meant a payload carrying the key was previously rejected
        outright. A migration that only defaulted the key would silently promote that
        impossible claim into an accepted one, attached to offsets nothing can re-check.
        Also found by a mutation audit: nothing distinguished ``setdefault`` from a
        plain assignment.
        """
        start = DOC.index(QUOTE)
        payload = self._legacy_v4_report(campaign, sha="c" * 64, start=start)
        payload["findings"][0]["evidence"]["extraction_id"] = "d" * 64
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(payload))

        report = load_literature_report(campaign.workspace_root)

        assert report is not None
        assert report.findings[0].evidence.extraction_id == ROOT_EXTRACTION_ID

    def test_p6_a_report_from_a_future_schema_version_still_fails_closed(self, campaign: Campaign) -> None:
        """P6. The version bump must not cost the fail-closed guard on future reports.
        A bump that forgets to move the ceiling turns 'refuse a report I cannot
        understand' into 'accept it and silently drop its unknown fields on the next
        write'."""
        from carmel.services.literature import ReportSchemaTooNewError

        future = {
            "schema_version": CURRENT_REPORT_SCHEMA_VERSION + 1,
            "report_id": "rep-future",
            "campaign_id": campaign.campaign_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passes": [],
            "queries": [],
            "findings": [],
            "rejected": [],
        }
        (campaign.workspace_root / LITERATURE_REPORT_NAME).write_text(json.dumps(future))

        with pytest.raises(ReportSchemaTooNewError, match="Upgrade Carmel"):
            load_literature_report(campaign.workspace_root)


def _store_legacy(workspace_root: Path, *, text: str, url: str) -> str:
    """Store one document in the shape the operator's REAL corpus is actually in.

    Every one of the 8 manually-acquired papers in the live workspace was written
    before ``extracted_sha256``, ``extractor_version`` and ``derivation_binding``
    existed, so its ``meta.json`` does not carry those keys AT ALL -- they are absent,
    not null. Nothing in this suite constructed that shape before, and that absence is
    exactly why the whole suite stayed green while every real document was unreadable.

    Root sidecars are never rewritten to upgrade a legacy artifact, so this helper
    reproduces the shape by dropping the keys after a normal store rather than by
    pretending a legacy writer could have produced them.
    """
    sha = _store(workspace_root, text=text, url=url)
    meta_path = workspace_root / "evidence" / "literature" / sha / "meta.json"
    meta = json.loads(meta_path.read_text())
    for absent_before_it_existed in ("extracted_sha256", "extractor_version", "derivation_binding"):
        meta.pop(absent_before_it_existed, None)
    meta_path.write_text(json.dumps(meta))
    return sha


def _corrupt_raw_bytes(workspace_root: Path, sha256: str) -> None:
    """Make ``raw.bin`` stop matching the directory that names it."""
    (workspace_root / "evidence" / "literature" / sha256 / "raw.bin").write_bytes(b"not the stored bytes")


class TestALegacyRootIsUnauthenticatedNotCorrupt:
    """A corpus pass must not report "your bytes are broken" about intact bytes.

    ``_load_corpus`` gates every held artifact on ``verify_artifact(deep=True)``, which
    requires a ``derivation_binding`` no legacy artifact carries. So all 8 real papers
    are skipped on every pass -- and dropped into the SAME undifferentiated ``skipped``
    list as an artifact whose ``raw.bin`` genuinely no longer hashes to its own name,
    logged with the same "failed digest verification" line. Their digests are fine.

    That is the UNAUTHENTICATED/INTEGRITY_FAILED conflation this project forbids, and
    the two call for opposite operator responses: one is "re-extract, or opt in"; the
    other is "these bytes are damaged, re-acquire the paper."
    """

    def test_g1_red_a_legacy_root_is_reported_differently_from_corrupt_bytes(self, campaign: Campaign) -> None:
        """G1, and the RED test required before any non-test file is touched.

        Deliberately asserts only that the two are DISTINGUISHABLE, not what either
        is called. A test that pinned the spelling would pass for a rename alone; the
        defect is that the caller has no way to tell the two apart at all.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        corrupt_sha = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.org/second")
        _corrupt_raw_bytes(campaign.workspace_root, corrupt_sha)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert [artifact.sha256 for artifact, _, _ in corpus] == [], (
            "neither document should be READ here: the legacy one is unauthenticated and the "
            "corrupt one is broken -- this test is about how they are REPORTED"
        )
        assert outcomes[legacy_sha] != outcomes[corrupt_sha], (
            "an intact legacy root and genuinely corrupt bytes must not be reported as the "
            "same thing -- one is 're-extract or opt in', the other is 're-acquire the paper'"
        )

    def test_g2_a_legacy_root_is_read_when_the_operator_opts_in(self, campaign: Campaign) -> None:
        """G2. ``allow_unauthenticated_legacy_roots=True`` actually reads a legacy root.

        The opt-in exists precisely so an operator who has decided the risk is
        acceptable can get their 8 real papers read at all -- if the flag flipped the
        outcome label without ever admitting the artifact into ``corpus``, the opt-in
        would be theatre.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)

        corpus, outcomes = literature_module._load_corpus(
            campaign.workspace_root, allow_unauthenticated_legacy_roots=True
        )

        assert [artifact.sha256 for artifact, _, _ in corpus] == [legacy_sha]
        assert outcomes[legacy_sha] == CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_g3_a_legacy_root_is_not_read_by_default(self, campaign: Campaign) -> None:
        """G3. The opt-in defaults to False -- fail closed unless the operator says
        otherwise. Calling ``_load_corpus`` with no keyword at all must behave exactly
        like the explicit ``allow_unauthenticated_legacy_roots=False`` case."""
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert [artifact.sha256 for artifact, _, _ in corpus] == []
        assert outcomes[legacy_sha] == CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_g4_corrupt_bytes_stay_unread_even_with_the_opt_in(self, campaign: Campaign) -> None:
        """G4. The opt-in is for artifacts that are merely unauthenticated, not for
        ones whose bytes are actually damaged -- ``allow_unauthenticated_legacy_roots=True``
        must not launder a corrupt artifact into being read."""
        corrupt_sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _corrupt_raw_bytes(campaign.workspace_root, corrupt_sha)

        corpus, outcomes = literature_module._load_corpus(
            campaign.workspace_root, allow_unauthenticated_legacy_roots=True
        )

        assert [artifact.sha256 for artifact, _, _ in corpus] == []
        assert outcomes[corrupt_sha] == CorpusReadOutcome.INTEGRITY_FAILED

    def test_g5_a_fully_bound_artifact_is_self_consistent_metadata_and_read_with_no_opt_in(
        self, campaign: Campaign
    ) -> None:
        """G5. A modern artifact with an intact ``derivation_binding`` needs no opt-in
        at all -- it is the strict, no-compromise case the whole tiering exists to
        keep distinct from the legacy one."""
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert [artifact.sha256 for artifact, _, _ in corpus] == [sha]
        assert outcomes[sha] == CorpusReadOutcome.SELF_CONSISTENT_METADATA

    def test_g6_a_binding_dropped_artifact_is_sidecar_digest_only_and_read_with_no_opt_in(
        self, campaign: Campaign
    ) -> None:
        """G6. An artifact that still carries ``extracted_sha256`` but has lost its
        ``derivation_binding`` is a real, distinct middle tier: not the strict
        ``SELF_CONSISTENT_METADATA`` case, but also not the wholly-unauthenticated legacy case --
        so it is read WITHOUT the opt-in, unlike a legacy root."""
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        meta_path = campaign.workspace_root / "evidence" / "literature" / sha / "meta.json"
        meta = json.loads(meta_path.read_text())
        del meta["derivation_binding"]
        meta_path.write_text(json.dumps(meta))

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert [artifact.sha256 for artifact, _, _ in corpus] == [sha]
        assert outcomes[sha] == CorpusReadOutcome.SIDECAR_DIGEST_ONLY

    def test_g7_a_document_read_under_the_opt_in_is_covered_at_the_legacy_standard(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """G7. When ``_corpus_loop`` reads a document that ``_load_corpus`` classified
        ``UNAUTHENTICATED_LEGACY_ROOT``, the ``CoveredDocument`` it appends must carry
        THAT standard, not a stronger one it never actually met. This is the single
        write site in ``_corpus_loop`` that maps ``outcomes[sha256]`` onto
        ``CoveredDocument.verification_standard`` -- forcing the outcome here isolates
        that mapping from the gating logic G2/G3 already cover."""
        _patch_chem_success(monkeypatch)
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        real_load_corpus = literature_module._load_corpus

        def _forced_legacy(workspace_root: Path, **kwargs: Any) -> Any:
            corpus, _ = real_load_corpus(workspace_root, **kwargs)
            return corpus, {artifact.sha256: CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT for artifact, _, _ in corpus}

        monkeypatch.setattr(literature_module, "_load_corpus", _forced_legacy)
        deps, _, config = _make_deps([_corpus_proposal([_corpus_finding(sha256=sha)]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        covered = report.passes[0].covered
        assert [c.raw_sha256 for c in covered] == [sha]
        assert covered[0].verification_standard == CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT.value

    def test_g8_the_v5_to_v6_migration_stamps_unrecorded_on_an_existing_covered_record(
        self,
    ) -> None:
        """G8. A v5 payload's ``covered`` entries (which never had
        ``verification_standard`` at all) gain the value ``"unrecorded"`` once migrated
        -- the honest admission that no pre-v6 writer ever recorded a standard."""
        from carmel.services.literature import migrate_report_payload

        payload = {
            "schema_version": 5,
            "passes": [
                {
                    **self._v3_pass_like(),
                    "covered": [{"raw_sha256": "a" * 64, "extraction_id": ROOT_EXTRACTION_ID}],
                }
            ],
        }

        migrated = migrate_report_payload(payload)

        assert isinstance(migrated, dict)
        assert migrated["passes"][0]["covered"] == [
            {"raw_sha256": "a" * 64, "extraction_id": ROOT_EXTRACTION_ID, "verification_standard": "unrecorded"}
        ]

    def test_g9_the_v5_to_v6_migration_overwrites_rather_than_honours_a_present_value(
        self,
    ) -> None:
        """G9. Pins the setdefault-vs-assign mutation directly: a v5 payload cannot
        legitimately carry ``verification_standard`` at all (``CoveredDocument`` was
        ``extra="forbid"`` and the field did not exist), so if one is present anyway
        the migration must overwrite it with ``"unrecorded"``, not honour it. A
        ``setdefault`` substitution here would silently accept a claim no v5 writer
        could have had grounds to make -- the exact mutation a prior audit caught in
        ``_migrate_v4_to_v5``."""
        from carmel.services.literature import migrate_report_payload

        payload = {
            "schema_version": 5,
            "passes": [
                {
                    **self._v3_pass_like(),
                    "covered": [
                        {
                            "raw_sha256": "a" * 64,
                            "extraction_id": ROOT_EXTRACTION_ID,
                            "verification_standard": "self_consistent_metadata",
                        }
                    ],
                }
            ],
        }

        migrated = migrate_report_payload(payload)

        assert isinstance(migrated, dict)
        assert migrated["passes"][0]["covered"][0]["verification_standard"] == "unrecorded", (
            "a pre-existing value must be OVERWRITTEN, not honoured -- no v5 writer could "
            "have legitimately set this field at all"
        )

    def test_g10_covered_document_refuses_an_unreadable_standard(self) -> None:
        """G10. ``verification_standard`` accepts only the outcomes that mean "this was
        actually read" (plus the migration sentinel ``"unrecorded"``) -- the other
        ``CorpusReadOutcome`` members name ways a document was NOT read, so nothing
        could have been read under them, and pure nonsense must be refused too."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="verification_standard"):
            CoveredDocument(
                raw_sha256="a" * 64,
                extraction_id=ROOT_EXTRACTION_ID,
                verification_standard=CorpusReadOutcome.INTEGRITY_FAILED.value,
            )
        with pytest.raises(ValidationError, match="verification_standard"):
            CoveredDocument(
                raw_sha256="a" * 64,
                extraction_id=ROOT_EXTRACTION_ID,
                verification_standard="nonsense",
            )

    def test_g11_the_legacy_root_log_line_does_not_claim_failed_verification(
        self, campaign: Campaign, caplog: pytest.LogCaptureFixture
    ) -> None:
        """G11. The log line for a legacy root must not reuse the "failed digest
        verification" phrasing that describes genuinely corrupt bytes -- an operator
        grepping logs for that phrase to find damaged papers must not also catch every
        intact-but-unauthenticated legacy paper."""
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)

        with caplog.at_level("WARNING", logger="carmel.services.literature"):
            literature_module._load_corpus(campaign.workspace_root)

        messages = [r.getMessage() for r in caplog.records]
        assert not any("failed digest verification" in m and legacy_sha in m for m in messages), (
            "the legacy root must not be described with the corrupt-bytes phrasing"
        )
        assert any("cannot be authenticated" in m and legacy_sha in m for m in messages), (
            "the legacy root must be described with its own, distinct phrasing"
        )

    def _v3_pass_like(self) -> dict[str, Any]:
        return {
            "run_id": "r",
            "action_id": "a",
            "created_at": datetime.now(UTC).isoformat(),
            "mode": LiteraturePassMode.CORPUS.value,
            "model_name": "mock",
            "stop_reason": StopReason.SELF_TERMINATED.value,
            "usage": {
                "model_calls": 1,
                "tokens": 100,
                "cost_usd": 0.01,
                "fetches": 0,
                "fetch_bytes": 0,
                "elapsed_s": 1.0,
            },
            "warnings": [],
        }

    def test_g12_the_operator_facing_warning_says_why_each_document_went_unread(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """G12. The typed outcome must survive to the operator, not die inside the loader.

        `_corpus_loop` flattened every unread class back into one bare sha list for the
        warning and the `literature.corpus_artifacts_unreadable` event, which reproduces
        the exact conflation this whole increment exists to remove -- one layer up, where
        it is the only version a human ever sees. An operator told "2 artifacts could not
        be read" cannot act: one of these needs re-extraction or an opt-in, the other
        needs the paper re-acquired.
        """
        _patch_chem_success(monkeypatch)
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        corrupt_sha = _store(campaign.workspace_root, text=SECOND_DOC, url="https://example.org/second")
        _corrupt_raw_bytes(campaign.workspace_root, corrupt_sha)
        deps, _, config = _make_deps([])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        unreadable = [w for w in report.passes[0].warnings if legacy_sha[:12] in w]
        assert unreadable, "the operator must be told about the artifact that went unread"
        warning = unreadable[0]
        assert CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT.value in warning, (
            "the warning must name WHY this document went unread, not merely that it did"
        )
        assert CorpusReadOutcome.INTEGRITY_FAILED.value in warning, (
            "and it must name the different reason for the genuinely damaged one"
        )
        # Each sha must sit under its OWN reason. Asserting only that both reasons
        # appear somewhere in the string would pass for one undifferentiated blob that
        # lists every reason and every sha together -- which tells the operator nothing
        # about which document is which, the very thing this test exists to pin.
        segments = {seg.split(":")[0].strip(): seg for seg in warning.split("--", 1)[1].split(";")}
        legacy_segment = segments[CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT.value]
        corrupt_segment = segments[CorpusReadOutcome.INTEGRITY_FAILED.value]
        assert legacy_sha[:12] in legacy_segment and corrupt_sha[:12] not in legacy_segment
        assert corrupt_sha[:12] in corrupt_segment and legacy_sha[:12] not in corrupt_segment

    def test_g13_a_legacy_artifact_is_not_covered_without_the_action_parameter(
        self, campaign: Campaign
    ) -> None:
        """G13. End to end through the queued action, not just ``_load_corpus``
        directly: an action carrying no ``allow_unauthenticated_legacy_roots`` key at
        all leaves the sole held legacy artifact unread and uncovered."""
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        assert [c.raw_sha256 for c in report.latest.covered] == []
        assert deps.ledger.usage().model_calls == 0
        assert legacy_sha  # the artifact exists; it is simply not among the covered

    def test_g14_a_legacy_artifact_is_covered_when_the_action_carries_the_parameter(
        self, campaign: Campaign
    ) -> None:
        """G14. The parameter must actually reach ``_load_corpus`` from the queued
        action's ``parameters``, not merely be recorded and dropped -- the only test
        in this suite that proves that end-to-end wiring, as opposed to the gating
        logic (G2/G3) or the recorded standard (G7), which are covered separately."""
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([_corpus_proposal([])])
        action = _action().model_copy(update={"parameters": {"allow_unauthenticated_legacy_roots": True}})

        report = run_corpus_pass(campaign.workspace_root, campaign, action, deps, config=config)

        assert [c.raw_sha256 for c in report.latest.covered] == [legacy_sha]

    def test_g15_the_parameter_is_not_passed_to_a_search_mode_pass(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """G15. A SEARCH-mode pass has no held corpus to gate, so
        ``allow_unauthenticated_legacy_roots`` must not reach ``_research_loop`` at all
        -- and a search pass whose action happens to carry the key regardless must
        still run normally rather than being refused by an unexpected keyword."""
        _patch_chem_success(monkeypatch)
        deps, _, config = _make_deps([_proposal(findings=[_finding_dict()]), _assessment()])
        action = _action().model_copy(update={"parameters": {"allow_unauthenticated_legacy_roots": True}})

        report = run_literature_research(campaign.workspace_root, campaign, action, deps, config=config)

        assert report.stop_reason == StopReason.SELF_TERMINATED
        assert len(report.findings) == 1


class TestAPermissionIsReadStrictlyFromThePlan:
    """`bool()` was the wrong reader for an untyped parameter bag.

    `PlannedAction.parameters` is a plain dict reloaded from `plan.json`, so what
    arrives is whatever JSON held. `bool("false")` is True -- so a plan carrying the
    STRING "false" GRANTED the permission it was plainly trying to decline, in the one
    place that decides whether unauthenticated text may be read at all.

    Found by spar round 64, on code shipped the same session. Review did not catch it
    and neither did 20 new tests, because every test built its parameters in Python
    where a bool stays a bool -- the defect only exists on the round trip through disk.
    """

    def _state(self) -> Any:
        return literature_module._RunState(queries=[], artifacts=[], findings=[], rejected=[], warnings=[])

    def test_d1_the_string_false_does_not_grant_a_permission(self) -> None:
        """D1. The exact fail-open: truthy string, falsy intent."""
        state = self._state()
        name = "allow_unauthenticated_legacy_roots"
        assert literature_module._permission_from({name: "false"}, name, state) is False

    def test_d2_a_malformed_value_is_surfaced_not_silently_swallowed(self) -> None:
        """D2. Refusing quietly would leave the operator believing their plan took effect."""
        state = self._state()
        literature_module._permission_from({"reread_all": "false"}, "reread_all", state)
        assert any("reread_all" in w and "NOT granted" in w for w in state.warnings), (
            "an operator who wrote a non-boolean must be told their intent did not take effect"
        )

    def test_d3_the_literal_true_still_grants(self) -> None:
        """D3. Guards against a fix that refuses everything."""
        state = self._state()
        assert literature_module._permission_from({"reread_all": True}, "reread_all", state) is True
        assert state.warnings == []

    def test_d4_an_int_one_does_not_grant(self) -> None:
        """D4. `1 == True` in Python, so an `==` comparison would let this back in."""
        state = self._state()
        assert literature_module._permission_from({"reread_all": 1}, "reread_all", state) is False

    def test_d5_an_absent_permission_is_a_clean_refusal_with_no_warning(self) -> None:
        """D5. Absence is the normal case and must not become noise."""
        state = self._state()
        assert literature_module._permission_from({}, "reread_all", state) is False
        assert state.warnings == []

    def test_d6_the_literal_false_is_a_clean_refusal_with_no_warning(self) -> None:
        """D6. An explicitly-declined permission is well-formed, not malformed."""
        state = self._state()
        assert literature_module._permission_from({"reread_all": False}, "reread_all", state) is False
        assert state.warnings == []

    def test_d7_end_to_end_a_plan_carrying_the_string_false_does_not_read_a_legacy_root(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D7. The only test that proves the strict read is on the PATH, not just callable.

        Goes through a real corpus pass with the parameter as it would survive a JSON
        round trip, rather than asserting on the helper in isolation.
        """
        _patch_chem_success(monkeypatch)
        _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        deps, _, config = _make_deps([])
        action = _action()
        action.parameters["allow_unauthenticated_legacy_roots"] = "false"

        report = run_corpus_pass(campaign.workspace_root, campaign, action, deps, config=config)

        assert report.passes[0].covered == [], "a string must never grant the permission to read unauthenticated text"


def _store_current_record(workspace_root: Path, raw_sha256: str, *, text: str) -> str:
    """Append ONE extraction record that is current for today's extractor identity.

    Stamped with the real `extraction_identity()` rather than a fixed string, because
    `current_extraction_records` compares against what today's code reports: a
    hardcoded sha would make the record permanently non-current and the test would
    pass for the wrong reason -- silently exercising the root path it exists to avoid.
    """
    from carmel.agents.tools.extract import ExtractedText, normalize_for_match
    from carmel.services.extraction_record import store_extraction_record
    from carmel.services.semantic_deps import extraction_identity

    identity = extraction_identity()
    extracted = ExtractedText(
        text=text, normalized=normalize_for_match(text), sections=[], extractor="pdf:pypdf", lossy=False
    )
    payload = json.dumps(extracted.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
    return store_extraction_record(
        workspace_root,
        raw_sha256=raw_sha256,
        extractor="pdf:pypdf",
        extractor_code_sha256=identity.code_sha256,
        pypdf_version=identity.pypdf_version,
        extracted_json_bytes=payload,
    )


def _store_record(
    workspace_root: Path,
    raw_sha256: str,
    *,
    text: str,
    extractor: str = "pdf:pypdf",
    code_sha256: str | None = None,
    pypdf_version: str | None = None,
) -> str:
    """Append ONE extraction record with the identity fields spelled out.

    The generalisation of :func:`_store_current_record` for tests that need a record
    which is deliberately NOT current, or not a ``pdf:pypdf`` one. Defaults still come
    from the real `extraction_identity()` for the same reason that helper does it.
    """
    from carmel.agents.tools.extract import ExtractedText, normalize_for_match
    from carmel.services.extraction_record import store_extraction_record
    from carmel.services.semantic_deps import extraction_identity

    identity = extraction_identity()
    extracted = ExtractedText(
        text=text, normalized=normalize_for_match(text), sections=[], extractor=extractor, lossy=False
    )
    payload = json.dumps(extracted.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
    return store_extraction_record(
        workspace_root,
        raw_sha256=raw_sha256,
        extractor=extractor,
        extractor_code_sha256=code_sha256 if code_sha256 is not None else identity.code_sha256,
        pypdf_version=pypdf_version if pypdf_version is not None else identity.pypdf_version,
        extracted_json_bytes=payload,
    )


def _records_dir(workspace_root: Path, raw_sha256: str) -> Path:
    return workspace_root / "evidence" / "literature" / raw_sha256 / "extractions"


RECORD_TEXT = "Text served from the authenticated extraction record, not the root sidecar."


class TestTheCorpusPrefersAnAuthenticatedExtractionRecord:
    """A record that authenticates is read in preference to the root sidecar.

    The root `extracted.json` of an older artifact is checked against nothing at all,
    so the gate refuses it. Re-extraction was supposed to be the way out, but it writes
    only under `extractions/` while the gate read only root fields -- so re-extracting
    changed nothing about whether a document could be read.

    The preference is UNIFORM, not a fallback triggered by the root failing to
    authenticate. A fallback keyed on root failure would mean deleting
    `extracted_sha256` from a modern root silently switches which text is served, and
    the store cannot tell "old legacy root" from "field just deleted" -- deletion would
    PROMOTE. A uniform rule makes deletion incapable of changing which path is taken.
    """

    def test_a_legacy_root_with_one_authenticated_record_is_read_with_no_opt_in_flag(
        self, campaign: Campaign
    ) -> None:
        """The operator's 8 papers, after re-extraction, without the opt-in flag.

        The record's text is deliberately DIFFERENT from the root's, so serving the
        root would fail this test rather than passing by coincidence -- the difference
        is the whole point. Using the record's mere existence to bless the ROOT text
        would authenticate one artifact by pointing at a different one.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        record_sha = _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert [artifact.sha256 for artifact, _, _ in corpus] == [legacy_sha], (
            "a legacy root with one authenticated current record must be READ without the opt-in"
        )
        _, extracted, extraction_id = corpus[0]
        assert extracted.text == RECORD_TEXT, "the RECORD's text must be served, never the root sidecar's"
        assert extracted.text != DOC
        assert extraction_id == record_sha, "coverage must name the record that was actually read"
        assert extraction_id != ROOT_EXTRACTION_ID
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_a_record_whose_text_fails_its_digest_is_not_read_and_not_downgraded_to_the_root(
        self, campaign: Campaign
    ) -> None:
        """A tampered record must not buy a read of the unauthenticated root text.

        Falling back here would hand an attacker exactly the downgrade the operator
        never authorised: break the record, get the root served instead.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        record_sha = _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        record_json = (
            campaign.workspace_root
            / "evidence"
            / "literature"
            / legacy_sha
            / "extractions"
            / record_sha
            / "extracted.json"
        )
        tampered = json.loads(record_json.read_text(encoding="utf-8"))
        tampered["text"] = "swapped after the digest was recorded"
        record_json.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "a record that fails its own digest must not be read"
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "and must NOT silently downgrade to the root path"
        )

    def test_a_corrupt_raw_bin_is_not_read_even_with_a_valid_current_record(self, campaign: Campaign) -> None:
        """A record must never launder a corrupt artifact.

        The shallow integrity check is absolute and runs first: whatever records exist,
        bytes that no longer hash to their own directory name are not evidence.
        """
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_current_record(campaign.workspace_root, sha, text=RECORD_TEXT)
        _corrupt_raw_bytes(campaign.workspace_root, sha)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == []
        assert outcomes[sha] == CorpusReadOutcome.INTEGRITY_FAILED

    def test_deleting_extracted_sha256_from_a_modern_root_does_not_change_the_served_text(
        self, campaign: Campaign
    ) -> None:
        """Deletion cannot promote -- asserted directly rather than argued.

        If the record path were a FALLBACK for a root that fails to authenticate, then
        deleting one root field would flip which text is served. Because the preference
        is uniform, deletion changes nothing: the record already won.
        """
        sha = _store(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        record_sha = _store_current_record(campaign.workspace_root, sha, text=RECORD_TEXT)

        before, _ = literature_module._load_corpus(campaign.workspace_root)

        meta_path = campaign.workspace_root / "evidence" / "literature" / sha / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["extracted_sha256"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        after, _ = literature_module._load_corpus(campaign.workspace_root)

        assert [(a.sha256, e.text, x) for a, e, x in before] == [(sha, RECORD_TEXT, record_sha)]
        assert [(a.sha256, e.text, x) for a, e, x in after] == [(sha, RECORD_TEXT, record_sha)], (
            "deleting a root field must not change which text is served, in either direction"
        )

    def test_the_operator_is_told_when_a_document_was_read_from_a_record(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which text was served is not a detail the operator should have to dig for.

        A document read from a record is a different document from the same raw bytes
        read through the root sidecar, and it is quoted under a different extraction id.
        Pinned here because the previous round's audit showed that a fix nothing asserts
        on is free to regress silently.
        """
        _patch_chem_success(monkeypatch)
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        record_sha = _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        deps, _, config = _make_deps([_corpus_proposal([]), _assessment()])

        report = run_corpus_pass(campaign.workspace_root, campaign, _action(), deps, config=config)

        warnings = "\n".join(report.passes[0].warnings)
        assert "authenticated extraction record" in warnings
        assert legacy_sha[:12] in warnings and record_sha[:12] in warnings
        covered = report.passes[0].covered
        assert [c.extraction_id for c in covered] == [record_sha]
        assert [c.verification_standard for c in covered] == [
            CorpusReadOutcome.EXTRACTION_RECORD_DIGEST_AUTHENTICATED.value
        ], "the permanent record must name the standard the document was ACTUALLY read under"

    def test_two_records_current_at_once_is_ambiguous_and_does_not_fall_back_to_the_root(
        self, campaign: Campaign
    ) -> None:
        """Ambiguity among records is not a licence to serve unchecked text.

        Every record here may be perfectly intact -- it is the STORE that cannot say
        which one speaks for this document. That is a different fact from a broken
        record, so it gets its own outcome rather than sending the operator hunting for
        a corrupt file that does not exist.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        _store_current_record(campaign.workspace_root, legacy_sha, text="a second, differently-worded extraction")

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "with two current records the document must not be read at all"
        assert outcomes[legacy_sha] == CorpusReadOutcome.MULTIPLE_CURRENT_EXTRACTION_RECORDS
        assert outcomes[legacy_sha] != CorpusReadOutcome.EXTRACTION_RECORD_AUTHENTICATION_FAILED, (
            "nothing failed to authenticate -- saying so would send the operator after a "
            "corrupt file that does not exist"
        )
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "and it must not quietly fall through to the root sidecar"
        )


class TestEveryRouteToNoUsableRecordRefuses:
    """Only a genuinely absent ``extractions/`` may fall through to the root tiers.

    `4c0aa23` guarded exactly one route -- a record that is FOUND and fails to
    authenticate. Every other way of arriving at "zero current records" still served
    unauthenticated root text, and two of those routes are worse than the downgrade that
    commit fixed: one of them SELECTS rather than falls through, and one is reachable by
    tampering with a single file.
    """

    def test_corrupting_one_meta_json_must_not_turn_a_refusal_into_a_read(
        self, campaign: Campaign
    ) -> None:
        """Tamper-to-PROMOTE: the attacker gains a read rather than losing one.

        Two current records are ambiguous and refuse. Corrupting ONE record's meta.json
        makes `list_extraction_records` skip it, so the count falls to one and the
        survivor is served as authenticated corpus text. That is strictly worse than the
        downgrade the record-preference was introduced to prevent, because deleting
        evidence must never be able to unlock a read.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        doomed = _store_record(
            campaign.workspace_root, legacy_sha, text="a second, differently-worded extraction"
        )
        (_records_dir(campaign.workspace_root, legacy_sha) / doomed / "meta.json").write_text(
            "{ this is not json", encoding="utf-8"
        )

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], (
            "a sha-shaped record that cannot be read is a candidate, not noise -- it must "
            "block the read, not silently reduce the count to one"
        )
        assert outcomes[legacy_sha] != CorpusReadOutcome.EXTRACTION_RECORD_DIGEST_AUTHENTICATED, (
            "corrupting one file must never PROMOTE the survivor to an authenticated read"
        )
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_a_pdf_unavailable_record_is_not_current_once_pypdf_is_installed(
        self, campaign: Campaign
    ) -> None:
        """False currentness: today's extractor would never produce this record.

        ``pdf:unavailable`` is the degraded placeholder written when pypdf could not be
        imported. It is deliberately excluded from the pypdf-version comparison -- which
        is right at STORE time, where demanding a version there provably is none would be
        wrong, but wrong at QUERY time: with pypdf installed, today's extraction would
        produce ``pdf:pypdf``, so the placeholder is stale and must not be served.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_record(
            campaign.workspace_root,
            legacy_sha,
            text="degraded placeholder written when pypdf was missing",
            extractor="pdf:unavailable",
            pypdf_version="unknown",
        )

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "a degraded placeholder must not be served as a current record"
        assert outcomes[legacy_sha] != CorpusReadOutcome.EXTRACTION_RECORD_DIGEST_AUTHENTICATED
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_records_that_no_longer_match_todays_code_identity_refuse(
        self, campaign: Campaign
    ) -> None:
        """Records exist but none is current: a refusal, not a fall-through.

        "No record was ever stored" and "records exist, none matches today's extractor"
        are different facts about the document, and only the first one licenses reading
        text that is checked against nothing.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_record(
            campaign.workspace_root, legacy_sha, text=RECORD_TEXT, code_sha256="0" * 64
        )

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "a stale record is not a licence to serve the root sidecar"
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_an_undiscoverable_pypdf_version_refuses_and_says_so(
        self, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken pypdf install must not silently revert the whole campaign to root text.

        `_pypdf_version()` collapses every failure to ``"unknown"``, which matches no
        stored version, so EVERY pypdf-extracted record in the campaign stops being
        current at once. The operator must be told their environment cannot identify the
        extractor dependency -- not that their documents are stale.
        """
        from carmel.services import semantic_deps

        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        monkeypatch.setattr(semantic_deps, "_pypdf_version", lambda: "unknown")

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "an unidentifiable extractor dependency must refuse, not downgrade"
        assert outcomes[legacy_sha] == CorpusReadOutcome.EXTRACTOR_IDENTITY_UNAVAILABLE
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a chmod-000 directory")
    def test_an_unlistable_extractions_directory_refuses(self, campaign: Campaign) -> None:
        """"The store cannot be read" must not present as "the store is empty"."""
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _store_current_record(campaign.workspace_root, legacy_sha, text=RECORD_TEXT)
        records_dir = _records_dir(campaign.workspace_root, legacy_sha)
        records_dir.chmod(0o000)
        try:
            corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)
        finally:
            records_dir.chmod(0o700)

        assert corpus == [], "an unreadable record store is not an absent one"
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT

    def test_a_genuinely_absent_extractions_directory_is_the_only_fall_through(
        self, campaign: Campaign
    ) -> None:
        """The one route that legitimately reaches the root tiers.

        This is the counterweight to every refusal above: if the refusals swallowed this
        case too, the legacy corpus would be unreadable rather than opt-in, and the
        refusals would be indistinguishable from a blanket ban.
        """
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        assert not _records_dir(campaign.workspace_root, legacy_sha).exists()

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == [], "still gated behind the legacy-root opt-in"
        assert outcomes[legacy_sha] == CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "an artifact that never had a record must still reach the root tiers"
        )


class TestOnePoisonedEntryDoesNotKillTheWholePass:
    """A containment failure must cost ONE document, not the whole corpus.

    `f5ea5bb` made every route to "no usable extraction record" refuse rather than serve
    unauthenticated root text. It left one route that does not refuse -- it CRASHES.
    ``_validated_records_dir`` raises ``ValueError`` when an artifact's ``extractions/``
    resolves outside the workspace, and nothing between it and the per-artifact loop in
    ``_load_corpus`` catches it, so a single poisoned entry aborts the pass for every
    OTHER document too.

    The root tier immediately below already degrades on its own ``ValueError`` rather
    than propagating it, so the record tier is the odd one out -- and it is the tier
    where the blast radius is the whole campaign instead of one artifact.
    """

    def test_a_symlinked_extractions_dir_refuses_only_its_own_document(
        self, campaign: Campaign
    ) -> None:
        """Two healthy documents must survive a third one's escaping symlink.

        The failure this pins is availability, not authenticity: before the fix the
        assertion below could not even be reached, because the call raised out of the
        loop and the two healthy artifacts were never classified at all.
        """
        healthy = [
            _store_legacy(
                campaign.workspace_root,
                text=f"Healthy synthetic document {index}. Measured 1.{index} ms.\n",
                url=f"https://example.invalid/healthy-{index}",
            )
            for index in range(2)
        ]
        for sha in healthy:
            _store_current_record(campaign.workspace_root, sha, text=RECORD_TEXT)
        poisoned = _store_legacy(
            campaign.workspace_root,
            text="Poisoned synthetic document. Measured 9.9 ms.\n",
            url="https://example.invalid/poisoned",
        )
        outside = campaign.workspace_root.parent / "outside-the-workspace"
        outside.mkdir(parents=True, exist_ok=True)
        _records_dir(campaign.workspace_root, poisoned).symlink_to(outside, target_is_directory=True)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert sorted(artifact.sha256 for artifact, _, _ in corpus) == sorted(healthy), (
            "one artifact whose record store escapes the workspace must not take the "
            "healthy documents down with it"
        )
        assert outcomes[poisoned] == CorpusReadOutcome.EXTRACTION_RECORD_STORE_ESCAPES_WORKSPACE
        assert outcomes[poisoned] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "escaping the workspace is never a licence to fall through to the root sidecar"
        )

    def test_the_escape_is_not_reported_as_a_merely_unreadable_store(
        self, campaign: Campaign
    ) -> None:
        """"Points outside the workspace" and "could not be listed" are different facts.

        The first is a containment breach -- the store is being asked to follow a path
        out of the workspace it is supposed to be sealed inside. The second is an IO
        error. Folding them into one outcome is the exact collapse this whole arc exists
        to undo, and it would tell an operator to check permissions when what they have
        is a planted or restored symlink.
        """
        poisoned = _store_legacy(
            campaign.workspace_root, text=DOC, url="https://example.invalid/escapes"
        )
        outside = campaign.workspace_root.parent / "outside-for-distinctness"
        outside.mkdir(parents=True, exist_ok=True)
        _records_dir(campaign.workspace_root, poisoned).symlink_to(outside, target_is_directory=True)

        _, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert outcomes[poisoned] != CorpusReadOutcome.EXTRACTION_RECORD_STORE_UNREADABLE, (
            "a containment breach must not be reported as a permissions problem"
        )


class TestAnIncompleteRecordStoreRefusesInTheCorpusPass:
    """The two forged/accidental routes to the root fall-through, end to end.

    Asserts the OUTCOME, not merely that the read refused: an operator facing an
    interrupted write is told to re-extract, and one facing a broken link is told their
    store has a pointer to nothing. Collapsing them would send the first person hunting
    for a symlink that does not exist.
    """

    def test_an_interrupted_write_is_reported_as_an_empty_store_not_an_absent_one(
        self, campaign: Campaign
    ) -> None:
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        _records_dir(campaign.workspace_root, legacy_sha).mkdir(parents=True)

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == []
        assert outcomes[legacy_sha] == CorpusReadOutcome.EMPTY_EXTRACTION_RECORD_STORE_PRESENT
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "a write that was begun and did not finish must not license the root sidecar"
        )

    def test_a_dangling_store_link_is_reported_as_such(self, campaign: Campaign) -> None:
        legacy_sha = _store_legacy(campaign.workspace_root, text=DOC, url=SOURCE_URL)
        artifact_root = campaign.workspace_root / "evidence" / "literature" / legacy_sha
        (artifact_root / "extractions").symlink_to(
            artifact_root / "never-created", target_is_directory=True
        )

        corpus, outcomes = literature_module._load_corpus(campaign.workspace_root)

        assert corpus == []
        assert outcomes[legacy_sha] == CorpusReadOutcome.EXTRACTION_RECORD_STORE_LINK_DANGLING
        assert outcomes[legacy_sha] != CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT, (
            "planting one broken symlink must not forge the store's most permissive answer"
        )
