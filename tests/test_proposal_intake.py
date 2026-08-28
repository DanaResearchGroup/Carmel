"""Carrying an ExtractionProposal to a stored, replayable condition set.

This is the demonstration the ticket asks for: a proposal the Extraction Agent could
have produced is carried, through the UNCHANGED condition-set producer, into a stored
envelope that replays VERIFIED -- and a proposal whose quote is not in the document is
REFUSED rather than stored. Every fixture is SYNTHETIC; no real paper text appears here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from carmel.agents.budget import BudgetLedger, session_budget
from carmel.agents.extraction_agent import ExtractionProposal, build_extraction_agent
from carmel.agents.models import MockModel
from carmel.config import AgentBudgetConfig
from carmel.services.condition_set_bridge import (
    load_condition_set_envelope,
    store_condition_set_envelope,
)
from carmel.services.dataset_producer import QuoteGroundingError
from carmel.services.dataset_replay import ReplayOutcome, replay_condition_set
from carmel.services.proposal_intake import (
    ProposalIntakeError,
    build_extraction_prompt,
    condition_set_from_proposal,
    current_extraction_text,
)
from tests.test_dataset_producer import _store_synthetic_artifact

#: A synthetic methods paragraph, invented wholesale. It states conditions the way
#: running prose does: a named apparatus, resolvable scalars, a token, and one
#: condition that genuinely cannot be reduced to a single number.
_METHODS_TEXT = (
    "2. Experimental methods\n"
    "Measurements were carried out in a jet-stirred reactor of fused silica.\n"
    "The initial temperature was 823 K and the pressure was held at 1.2 atm.\n"
    "The fuel was methane in all cases.\n"
    "The equivalence ratio was varied from 0.6 to 1.4 across the campaign.\n"
)


@pytest.fixture(autouse=True)
def _reset_session_budget() -> object:
    session_budget().reset()
    yield
    session_budget().reset()


def _ledger(**limits: object) -> BudgetLedger:
    return BudgetLedger(AgentBudgetConfig(**limits))  # type: ignore[arg-type]


def _proposal_payload(sha256: str) -> dict[str, object]:
    """A proposal whose every quote occurs verbatim in ``_METHODS_TEXT``."""
    return {
        "artifact_sha256": sha256,
        "attribution": "own_experiment",
        "attribution_quote": "Measurements were carried out",
        "subject": {"kind": "device_class", "label_quote": "jet-stirred reactor"},
        "scalars": [
            {
                "claim_id": "initial_temperature",
                "label_quote": "initial temperature",
                "quantity_kind": "temperature",
                "value_quote": "823",
                "unit_quote": "K",
            },
            {
                "claim_id": "pressure",
                "label_quote": "pressure",
                "quantity_kind": "pressure",
                "value_quote": "1.2",
                "unit_quote": "atm",
            },
        ],
        "categoricals": [{"claim_id": "fuel", "label_quote": "fuel", "token_quote": "methane"}],
        "unextracted": [
            {
                "statement_id": "equivalence_ratio",
                "label_quote": "equivalence ratio",
                "statement_quote": "varied from 0.6 to 1.4",
                "reason": "multi_valued_sweep",
                "quantity_kind": "equivalence_ratio",
            }
        ],
    }


class TestAProposalBecomesAStoredReplayableConditionSet:
    def test_proposal_to_stored_envelope_replays_verified(self, tmp_path: Path) -> None:
        """The whole point of the ticket: a proposal goes in, a stored envelope comes
        out, and re-read from the store it replays VERIFIED -- every grounded quote
        re-sliced from the document's own bytes."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        proposal = ExtractionProposal.model_validate(_proposal_payload(stored.sha256))

        envelope = condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)
        saved = store_condition_set_envelope(tmp_path, envelope)
        reloaded = load_condition_set_envelope(tmp_path, saved.sha256)
        assert reloaded == envelope

        report = replay_condition_set(tmp_path, reloaded)
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.unchecked_char_spans == 0
        # 2 scalars (label+value+unit) + 1 categorical (label+token)
        # + 1 unextracted (label+statement) + subject label + attribution.
        assert report.total_char_spans >= 2 * 3 + 2

    def test_full_path_through_the_agent_and_mock_model(self, tmp_path: Path) -> None:
        """End to end through the production seam: the mock model returns the payload,
        the bridge validates it, and the carrier stores a VERIFIED envelope -- no live
        model call anywhere."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        model = MockModel(responses=[_proposal_payload(stored.sha256)])
        agent = build_extraction_agent(model=model, ledger=_ledger())

        prompt = build_extraction_prompt(
            objective="jet-stirred reactor conditions for methane oxidation",
            artifact_sha256=stored.sha256,
            text=current_extraction_text(tmp_path, stored.sha256),
        )
        result = agent.run(prompt)
        proposal = ExtractionProposal.model_validate(result.output)

        envelope = condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)
        assert replay_condition_set(tmp_path, envelope).evidence_outcome is ReplayOutcome.VERIFIED

    def test_a_refusal_only_proposal_still_produces(self, tmp_path: Path) -> None:
        """A proposal that resolved nothing but recorded WHY is a legitimate result --
        the narrow honest slice blesses refusals; it does not require yield."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        payload = _proposal_payload(stored.sha256)
        payload["scalars"] = []
        payload["categoricals"] = []
        proposal = ExtractionProposal.model_validate(payload)

        envelope = condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)
        assert envelope.scalar_claims == ()
        assert len(envelope.unextracted) == 1
        assert replay_condition_set(tmp_path, envelope).evidence_outcome is ReplayOutcome.VERIFIED


class TestTheGrounderRefusesRatherThanStores:
    def test_a_quote_absent_from_the_document_is_refused(self, tmp_path: Path) -> None:
        """The assertion that matters: an extraction agent that cannot be refused is a
        hallucination pipeline. A proposed value that does not occur in the document
        surfaces as the producer's grounding refusal, and nothing is stored."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        payload = _proposal_payload(stored.sha256)
        # 9999 K never appears in the document; the value cannot be grounded.
        payload["scalars"] = [
            {
                "claim_id": "t",
                "label_quote": "initial temperature",
                "quantity_kind": "temperature",
                "value_quote": "9999",
                "unit_quote": "K",
            }
        ]
        proposal = ExtractionProposal.model_validate(payload)

        with pytest.raises(QuoteGroundingError):
            condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)

    def test_an_observable_proposal_is_refused_not_dropped(self, tmp_path: Path) -> None:
        """Observables have no assembly path today; the carrier refuses them loudly
        rather than silently dropping the observation."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        payload = _proposal_payload(stored.sha256)
        payload["observables"] = [{"observable_id": "idt", "label_quote": "ignition delay time"}]
        proposal = ExtractionProposal.model_validate(payload)

        with pytest.raises(ProposalIntakeError, match="observable"):
            condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)


class TestTheDeterministicTextTools:
    def test_current_extraction_text_returns_the_stored_text(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        assert current_extraction_text(tmp_path, stored.sha256) == _METHODS_TEXT

    def test_current_extraction_text_refuses_when_no_record(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        for record in (tmp_path / "evidence" / "literature" / stored.sha256 / "extractions").glob("*"):
            shutil.rmtree(record)
        with pytest.raises(ProposalIntakeError, match="no usable current extraction record"):
            current_extraction_text(tmp_path, stored.sha256)

    def test_build_extraction_prompt_embeds_the_document_and_sha(self, tmp_path: Path) -> None:
        prompt = build_extraction_prompt(
            objective="find the reactor conditions",
            artifact_sha256="b" * 64,
            text=_METHODS_TEXT,
        )
        assert "b" * 64 in prompt
        assert _METHODS_TEXT in prompt
        assert "find the reactor conditions" in prompt
        assert "<<<DOCUMENT>>>" in prompt and "<<<END DOCUMENT>>>" in prompt


#: Two DIFFERENT synthetic documents that share every quote the proposal below uses, so
#: the proposal grounds cleanly against EITHER. This is what makes the sha-selector test
#: honest: the refusal must fire because the wrong document was selected, not because the
#: quotes happen to be absent from it.
_DOC_PROMPTED = "Measurements were carried out in a jet-stirred reactor.\nThe fuel was methane.\n-- run on rig alpha.\n"
_DOC_SUBSTITUTED = (
    "Measurements were carried out in a jet-stirred reactor.\nThe fuel was methane.\n-- run on rig beta.\n"
)


def _grounding_payload(sha256: str) -> dict[str, object]:
    """A proposal whose every quote occurs in BOTH _DOC_PROMPTED and _DOC_SUBSTITUTED."""
    return {
        "artifact_sha256": sha256,
        "attribution": "own_experiment",
        "attribution_quote": "Measurements were carried out",
        "subject": {"kind": "device_class", "label_quote": "jet-stirred reactor"},
        "categoricals": [{"claim_id": "fuel", "label_quote": "fuel", "token_quote": "methane"}],
    }


class TestTheDocumentSelectorIsTheCallersAuthority:
    """artifact_sha256 is a model-filled selector, not a checkable claim: a wrong sha
    grounds every quote against the wrong paper. The caller's own sha is the authority."""

    def test_a_substituted_sha_is_refused_even_though_it_would_ground(self, tmp_path: Path) -> None:
        """The assertion the ticket names: a proposal whose artifact_sha256 differs from
        the sha the caller prompted with is refused, naming BOTH shas -- and the refusal
        fires even though the quotes WOULD ground against the substituted document (the
        control below proves they do), so it is catching the mis-selection itself, not a
        grounding failure."""
        prompted = _store_synthetic_artifact(tmp_path, _DOC_PROMPTED)
        substituted = _store_synthetic_artifact(tmp_path, _DOC_SUBSTITUTED)
        assert prompted.sha256 != substituted.sha256

        # The model echoes the WRONG document's sha; the caller prompted with `prompted`.
        proposal = ExtractionProposal.model_validate(_grounding_payload(substituted.sha256))
        with pytest.raises(ProposalIntakeError) as excinfo:
            condition_set_from_proposal(tmp_path, proposal, expected_sha256=prompted.sha256)
        message = str(excinfo.value)
        assert prompted.sha256 in message and substituted.sha256 in message

    def test_control_the_same_quotes_really_do_ground_against_the_substituted_doc(self, tmp_path: Path) -> None:
        """Control for the test above: pointed at the substituted document as its OWN
        authority, the identical proposal grounds and produces -- proving the refusal
        was about mis-selection, not absent quotes."""
        substituted = _store_synthetic_artifact(tmp_path, _DOC_SUBSTITUTED)
        proposal = ExtractionProposal.model_validate(_grounding_payload(substituted.sha256))
        envelope = condition_set_from_proposal(tmp_path, proposal, expected_sha256=substituted.sha256)
        assert replay_condition_set(tmp_path, envelope).evidence_outcome is ReplayOutcome.VERIFIED


#: A document where the subject label "reactor" occurs exactly three times, so a 1-based
#: occurrence selector is visibly off-by-one if the two ends disagree.
_REPEATS_TEXT = (
    "First the reactor A ran.\nThen the reactor B ran.\nThen the reactor C ran.\n"
    "The equivalence ratio was swept widely.\n"
)


def _repeats_payload(sha256: str, *, subject_occurrence: int) -> dict[str, object]:
    return {
        "artifact_sha256": sha256,
        "attribution": "own_experiment",
        "attribution_quote": "First",
        "subject": {
            "kind": "device_class",
            "label_quote": "reactor",
            "label_occurrence": subject_occurrence,
        },
        "unextracted": [
            {
                "statement_id": "phi",
                "label_quote": "equivalence ratio",
                "statement_quote": "swept widely",
                "reason": "multi_valued_sweep",
            }
        ],
    }


class TestTheOccurrenceSelectorAgreesEndToEnd:
    """A proposal naming 1-based occurrence N grounds to the span a reader counting the
    same way would pick -- asserted against a document where the quote genuinely repeats,
    so an off-by-one between the prompt's 1-based and the grounder's 0-based is visible."""

    def _grounded_reactor_index(self, tmp_path: Path, *, subject_occurrence: int) -> int:
        stored = _store_synthetic_artifact(tmp_path, _REPEATS_TEXT)
        proposal = ExtractionProposal.model_validate(
            _repeats_payload(stored.sha256, subject_occurrence=subject_occurrence)
        )
        envelope = condition_set_from_proposal(tmp_path, proposal, expected_sha256=stored.sha256)
        locator = envelope.subject.label_ref.locator
        assert _REPEATS_TEXT[locator.start : locator.end] == "reactor"
        # How many "reactor" occurrences precede this span -> its 1-based position.
        return _REPEATS_TEXT[: locator.start].count("reactor") + 1

    def test_occurrence_two_grounds_the_second_reactor(self, tmp_path: Path) -> None:
        """The heart of the fix: '2' means the SECOND occurrence, not the third."""
        assert self._grounded_reactor_index(tmp_path, subject_occurrence=2) == 2

    def test_occurrence_one_and_three_bracket_it(self, tmp_path: Path) -> None:
        """The endpoints agree too: 1 -> first, 3 -> third, no drift at the edges."""
        assert self._grounded_reactor_index(tmp_path, subject_occurrence=1) == 1
        assert self._grounded_reactor_index(tmp_path, subject_occurrence=3) == 3

    def test_zeroth_occurrence_is_refused_at_validation(self, tmp_path: Path) -> None:
        """There is no 1-based 'zeroth' occurrence: the schema floor refuses it before
        it can be silently converted to a negative grounder index."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            ExtractionProposal.model_validate(_repeats_payload("a" * 64, subject_occurrence=0))

    def test_a_boolean_occurrence_is_refused_rather_than_read_as_one(self) -> None:
        """A JSON ``true`` must not become "the first occurrence".

        Pydantic's lax mode coerces ``True`` into the integer ``1`` -- and into a
        genuine ``int``, not a ``bool``. So the downstream anti-bool guard
        (``_require_int_occurrences``, which tests ``isinstance(value, bool)``) is
        blind to it: by the time the spec dataclass sees the value, the boolean is
        gone and an ordinary ``1`` has taken its place. This boundary is the last
        point at which the difference between "the model said true" and "the model
        said 1" still exists, so it has to be refused here or nowhere.
        """
        for wrong in (True, False):
            with pytest.raises(ValueError, match="Input should be a valid integer"):
                ExtractionProposal.model_validate(_repeats_payload("a" * 64, subject_occurrence=wrong))

    def test_the_document_block_adds_no_character_the_text_does_not_carry(self) -> None:
        """Everything between the markers is the stored text, byte for byte.

        The agent is told to quote character for character from inside the markers,
        so any character the prompt adds there is a character it may faithfully copy
        into a quote that then cannot be grounded -- a refusal whose visible words
        match the document perfectly.
        """
        for text in ("ends without a newline", "ends with one\n", "blank line after\n\n"):
            prompt = build_extraction_prompt(objective="o", artifact_sha256="a" * 64, text=text)
            body = prompt.split("<<<DOCUMENT>>>\n", 1)[1].rsplit("<<<END DOCUMENT>>>", 1)[0]
            assert body == text or body == text + "\n"
            assert body.startswith(text)
            # The only permitted addition is a single newline to put the end marker on
            # its own line, and only when the text does not already end one.
            assert len(body) - len(text) == (0 if text.endswith("\n") else 1)
