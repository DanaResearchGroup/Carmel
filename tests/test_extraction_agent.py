"""The Extraction Agent persona: schema, builder, and the mandatory budget gate.

Every fixture here is SYNTHETIC. No text from any real paper appears in this
repository. The persona reads a stored document and PROPOSES conditions; nothing it
emits is believed until the condition-set producer grounds it -- that grounding is
exercised in ``tests.test_proposal_intake``. This file pins the persona's shape and
the one invariant the bridge enforces for every persona: budget is reserved BEFORE the
model is ever called.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from carmel.agents.budget import BudgetExceededError, BudgetLedger, session_budget
from carmel.agents.extraction_agent import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionProposal,
    ProposedDeviceClass,
    ProposedScalarCondition,
    ProposedUnresolvedSubject,
    build_extraction_agent,
)
from carmel.agents.models import MockModel
from carmel.config import AgentBudgetConfig
from carmel.schemas.datasets import SubjectRefusalReason
from carmel.services.units import QuantityKind

_SHA = "a" * 64


def _ledger(**limits: object) -> BudgetLedger:
    return BudgetLedger(AgentBudgetConfig(**limits))  # type: ignore[arg-type]


def _proposal_dict() -> dict[str, object]:
    """A minimal well-formed proposal payload, as a model would emit it."""
    return {
        "artifact_sha256": _SHA,
        "attribution": "own_experiment",
        "attribution_quote": "Measurements were carried out",
        "subject": {"kind": "device_class", "label_quote": "jet-stirred reactor"},
        "scalars": [
            {
                "claim_id": "t",
                "label_quote": "initial temperature",
                "quantity_kind": "temperature",
                "value_quote": "823",
                "unit_quote": "K",
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset_session_budget() -> object:
    session_budget().reset()
    yield
    session_budget().reset()


class TestBuildExtractionAgent:
    def test_builds_an_agent_that_produces_the_extraction_proposal_schema(self) -> None:
        agent = build_extraction_agent(model=MockModel(), ledger=_ledger())
        assert agent.name == "extraction"
        assert agent.output_schema is ExtractionProposal
        assert agent.system_prompt is EXTRACTION_SYSTEM_PROMPT

    def test_takes_no_tools_by_default(self) -> None:
        """Like the corpus and verifier personas, the extraction agent is handed no
        live tools -- the document text is placed in its prompt deterministically."""
        agent = build_extraction_agent(model=MockModel(), ledger=_ledger())
        assert agent.tools == []

    def test_a_canned_proposal_round_trips_through_the_bridge(self) -> None:
        """The production path with the mock-model seam: the model returns a payload,
        the bridge validates it against ExtractionProposal, and the caller gets a
        proposal it can re-parse."""
        model = MockModel(responses=[_proposal_dict()])
        agent = build_extraction_agent(model=model, ledger=_ledger())
        result = agent.run("prompt with the document text embedded")
        proposal = ExtractionProposal.model_validate(result.output)
        assert proposal.artifact_sha256 == _SHA
        assert proposal.scalars[0].value_quote == "823"
        assert len(model.calls) == 1


class TestBudgetIsReservedBeforeTheModelCall:
    """The bridge invariant, asserted for THIS persona: no headroom means the model is
    never invoked at all -- the reservation happens first and fails first."""

    def test_no_headroom_refuses_before_calling_the_model(self) -> None:
        model = MockModel(responses=[_proposal_dict()])
        ledger = _ledger(max_model_calls=1)
        # Consume the single allowed call slot directly on the ledger, so the agent's
        # own reservation has nothing left.
        ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=0.01)
        agent = build_extraction_agent(model=model, ledger=ledger)

        with pytest.raises(BudgetExceededError):
            agent.run("prompt")
        # The reservation is checked BEFORE the call: the mock was never popped.
        assert len(model.calls) == 0


class TestExtractionProposalSchema:
    def test_a_scalar_value_shorter_than_forty_chars_is_accepted(self) -> None:
        """A ProposedFinding needs a 40-char quote; a scalar condition value is a bare
        numeral. The persona must NOT inherit that floor or it would reject every
        honest scalar."""
        proposed = ProposedScalarCondition(
            claim_id="t",
            label_quote="T",
            quantity_kind=QuantityKind.TEMPERATURE,
            value_quote="823",
            unit_quote="K",
        )
        assert proposed.value_quote == "823"

    def test_subject_discriminates_on_kind(self) -> None:
        device = ExtractionProposal.model_validate(_proposal_dict()).subject
        assert isinstance(device, ProposedDeviceClass)

        payload = _proposal_dict()
        payload["subject"] = {
            "kind": "unresolved_subject",
            "reason": "device_unnamed",
            "reason_quote": "conditions were held constant throughout",
        }
        refused = ExtractionProposal.model_validate(payload).subject
        assert isinstance(refused, ProposedUnresolvedSubject)
        assert refused.reason is SubjectRefusalReason.DEVICE_UNNAMED

    def test_unknown_field_is_forbidden(self) -> None:
        payload = _proposal_dict()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            ExtractionProposal.model_validate(payload)

    def test_a_hallucinated_sha_handle_fails_validation_immediately(self) -> None:
        payload = _proposal_dict()
        payload["artifact_sha256"] = "not-a-real-digest"
        with pytest.raises(ValidationError):
            ExtractionProposal.model_validate(payload)

    def test_observables_and_done_default_sensibly(self) -> None:
        proposal = ExtractionProposal.model_validate(_proposal_dict())
        assert proposal.observables == []
        assert proposal.done is True
