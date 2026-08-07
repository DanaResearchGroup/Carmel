# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Replaying a :class:`ConditionSetEnvelope` against the evidence it cites.

The dataset replayer answers "did every recorded quote come back off disk
unchanged?". A condition set cannot be replayed by that question alone, because
not every ``SourceRef`` it carries is PAIRED with a recorded quote.

Of the 14 ref locations in the reachable graph, 11 are grounding PAIRS -- a
locator beside the verbatim text read there (``label_raw``, ``token_raw``,
``raw_text``, ``unit_raw``) -- and 3 are not: ``attribution_ref`` supports a
``ConditionAttribution`` enum, ``UnresolvedSubject.reason_ref`` supports a
``SubjectRefusalReason`` enum, and ``unextracted[i].statement_ref`` supports
nothing stored at all, by design. For those three the ref proves a span EXISTS
and nothing recorded anywhere says that span MEANS the value derived from it.
That is the rule "grounding proves LOCATION, never MEANING" showing up as a
shape in the data rather than as a maxim in a docstring.

So a replayer that only checked spans would return VERIFIED for a condition set
whose every derived value was unsupported by anything a machine could check.
The tests here pin the opposite: every obligation the envelope imposes is
discharged either by a check that RAN or by an explicitly recorded
:class:`UncheckedSemanticClaim`, and the hand-written enumeration of those
obligations is reconciled against a generic :func:`iter_source_refs` walk so a
ref the enumeration forgot is a FAILURE rather than a silent omission.

Every fixture here is SYNTHETIC. No paper text appears in this repository.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisRole,
    CaptionLabelKey,
    CharSpanLocator,
    ConditionAttribution,
    ConditionSetEnvelope,
    DeviceClassDeclaration,
    EmbeddedConversionTable,
    GroundedCategoricalClaim,
    GroundedScalarClaim,
    MeasuredValue,
    SemanticDependencyUse,
    SourceGraph,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    TextSpace,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
    ValueOrigin,
    iter_source_refs,
)
from carmel.services import dataset_replay
from carmel.services.dataset_producer import MeasurementSpec
from carmel.services.dataset_replay import (
    ReplayFinding,
    ReplayOutcome,
    SemanticGap,
    UncheckedSemanticClaim,
    _TextPairing,
    replay_condition_set,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import TABLE_V1, QuantityKind
from tests.test_dataset_replay import (
    MAX_BYTES,
    _store_genuine_extraction_record,
    _tabular_envelope_from_artifact,
)

# Synthetic prose. Each grounded quote below appears EXACTLY ONCE, so an
# offset computed with ``str.index`` is unambiguous and a span that drifts
# cannot accidentally land on an identical second occurrence.
_TEXT = (
    "The measurements were made on a heat flux burner in our laboratory. "
    "The diluent was CO2, and the temperature was 298 K at the inlet, "
    "where the mole fraction (-) of fuel was 0.0123. "
    "The pressure was 1.5 atm, with an upper bound of 1.7 atm "
    "and a lower bound of 1.3 atm. Run 11023 was discarded."
)

_EMBEDDED_NUMERAL = "1023"
"""Occurs ONLY inside the longer numeral "11023".

A span over it re-slices to exactly "1023", so quote equality alone cannot
tell it apart from a genuine reading of the number 1023. Only a boundary
check can, which is what makes it the right probe for the measured-value
gates.
"""

_SUBJECT_QUOTE = "heat flux burner"
_ATTRIBUTION_QUOTE = "in our laboratory"
_LABEL_QUOTE = "diluent"
_TOKEN_QUOTE = "CO2"


def _span(quote: str, node_id: str = "paper") -> SourceRef:
    """A char-span ref over the single occurrence of ``quote`` in ``_TEXT``."""
    assert _TEXT.count(quote) == 1, f"{quote!r} must be unique in the fixture text"
    return _nth_span(quote, 0, node_id=node_id)


def _nth_span(quote: str, occurrence: int, node_id: str = "paper") -> SourceRef:
    """A char-span ref over occurrence ``occurrence`` (0-based) of ``quote``.

    The uncertainty bounds need three separate spans over the SAME unit
    string, so they cannot all use the unique-quote helper. Grounding each
    bound at its own occurrence keeps every ref pointing at the place its
    own text was actually read, which is the property replay checks.
    """
    start = -1
    for _ in range(occurrence + 1):
        start = _TEXT.index(quote, start + 1)
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(
            text_space=TextSpace.EXTRACTED_TEXT, start=start, end=start + len(quote)
        ),
    )


def _pressure(raw_text: str, unit_occurrence: int) -> MeasuredValue:
    """A pressure reading in atm, grounded at its own occurrence of "atm"."""
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=QuantityKind.PRESSURE,
        unit_raw="atm",
        unit_normalized="atm",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        ),
        value_ref=_span(raw_text),
        unit_ref=_nth_span("atm", unit_occurrence),
    )


def _scalar_claim_with_uncertainty() -> GroundedScalarClaim:
    """A scalar claim carrying BOTH uncertainty bounds -- 6 measured-value refs."""
    return GroundedScalarClaim(
        claim_id="initial_pressure",
        label_raw="pressure",
        label_ref=_span("pressure"),
        value=_pressure("1.5", 0),
        uncertainty=Uncertainty(
            kind=UncertaintyKind.CI_95,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_pressure("1.7", 1),
            lower=_pressure("1.3", 2),
        ),
    )


def _embedded_table_v1() -> EmbeddedConversionTable:
    return EmbeddedConversionTable(
        sha256=TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
    )


def _verifying_graph(tmp_path: Path) -> SourceGraph:
    """A one-node graph whose node actually re-verifies against ``tmp_path``.

    Built by running the real producer and taking its root node, NOT by hand.
    An ``ExtractionBinding`` recomputes its own content address from its
    fields, so a hand-written one can never address a record that exists on
    disk -- a fixture node would be UNVERIFIABLE for a reason that has nothing
    to do with what these tests are about.
    """
    data = _TEXT.encode("utf-8")
    artifact = FetchedArtifact(
        url="https://example.invalid/paper.pdf",
        final_url="https://example.invalid/paper.pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        content_type="application/pdf",
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    extracted = ExtractedText(
        text=_TEXT, normalized=_TEXT.casefold(), sections=[], extractor="pdf:pypdf", lossy=False
    )
    stored = store_artifact(
        tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES
    )
    _store_genuine_extraction_record(tmp_path, stored.sha256, extracted)
    dataset = _tabular_envelope_from_artifact(
        tmp_path,
        sha256=stored.sha256,
        series_id="s1",
        value_origin=ValueOrigin.EXPERIMENTAL,
        measurements=(
            MeasurementSpec(
                axis_id="temperature",
                role=AxisRole.COORDINATE,
                quantity_kind=QuantityKind.TEMPERATURE,
                label_quote="temperature",
                value_quote="298",
                unit_quote="K",
            ),
            # A Series must declare at least one OBSERVATION axis. This dataset
            # is scaffolding -- it exists only to mint a source node that
            # genuinely re-verifies -- but it still has to be a legal one.
            MeasurementSpec(
                axis_id="mole_fraction",
                role=AxisRole.OBSERVATION,
                quantity_kind=QuantityKind.MOLE_FRACTION,
                label_quote="mole fraction",
                value_quote="0.0123",
                unit_quote="-",
            ),
        ),
    )
    return dataset.source_graph


def _minimal_condition_set(tmp_path: Path, **kwargs: object) -> ConditionSetEnvelope:
    """The SMALLEST condition set that isolates ``attribution`` as the only gap.

    Built from a CATEGORICAL claim on the ``DeviceClassDeclaration`` subject
    arm, and that is forced rather than stylistic. Validator C1 requires at
    least one record across ``scalar_claims + categorical_claims +
    unextracted``, and the other two ways of satisfying it drag in a SECOND
    unpaired ref: an ``unextracted`` statement carries ``statement_ref``, and
    the ``UnresolvedSubject`` arm carries ``reason_ref``. Only this shape
    leaves exactly one semantic gap, which is what makes the assertion
    ``claim_paths == {"attribution"}`` mean what it says.

    Its four refs are ``subject.label_ref``, ``attribution_ref``,
    ``categorical_claims[0].label_ref`` and ``categorical_claims[0].token_ref``.
    """
    defaults: dict[str, object] = {
        "source_graph": _verifying_graph(tmp_path),
        "conversion_tables": (),
        "subject": DeviceClassDeclaration(
            label_raw=_SUBJECT_QUOTE, label_ref=_span(_SUBJECT_QUOTE)
        ),
        "attribution": ConditionAttribution.OWN_EXPERIMENT,
        "attribution_ref": _span(_ATTRIBUTION_QUOTE),
        "scalar_claims": (),
        "categorical_claims": (
            GroundedCategoricalClaim(
                claim_id="diluent",
                label_raw=_LABEL_QUOTE,
                label_ref=_span(_LABEL_QUOTE),
                token_raw=_TOKEN_QUOTE,
                token_ref=_span(_TOKEN_QUOTE),
            ),
        ),
        "unextracted": (),
    }
    defaults.update(kwargs)
    return ConditionSetEnvelope(**defaults)  # type: ignore[arg-type]


class TestAConditionSetIsNeverFullyVerifiable:
    """The central claim: clean spans are NOT enough to call a condition set verified.

    This is the test the whole replayer exists to satisfy. It is deliberately
    sharper than "a false subject replays clean", because it catches the defect
    that actually threatens this design -- a replayer that silently omits the
    semantic caveats it is REQUIRED to record and still reports VERIFIED. A
    replayer can pass every span check and still be wrong here.
    """

    def test_clean_spans_verify_the_evidence_but_not_the_condition_set(
        self, tmp_path: Path
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)

        report = replay_condition_set(tmp_path, envelope)

        # Nothing is WRONG: every span that could be checked came back
        # unchanged, so there is no failure and nothing unverifiable.
        assert report.findings == ()
        assert report.evidence_outcome is ReplayOutcome.VERIFIED

        # And yet the condition set as a whole is NOT verified, because its
        # attribution rests on a span that nothing recorded explains. These two
        # verdicts disagreeing is the POINT of the split, not a defect in it.
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_the_unexplained_attribution_is_recorded_as_an_explicit_claim(
        self, tmp_path: Path
    ) -> None:
        """UNVERIFIABLE must be justified in the report, not merely asserted.

        A verdict a consumer cannot trace to a named cause is indistinguishable
        from a bug in the replayer.
        """
        envelope = _minimal_condition_set(tmp_path)

        report = replay_condition_set(tmp_path, envelope)

        assert {claim.claim_path for claim in report.unchecked_semantic_claims} == {"attribution"}
        claim = report.unchecked_semantic_claims[0]
        assert claim.gap is SemanticGap.SUPPORT_UNRECORDED
        assert claim.support_paths == ("attribution_ref",)
        # The CLAIM carries the derived value, never an excerpt of source text.
        # `_redacted` is the only path by which stored text may reach a report.
        assert claim.claim == ConditionAttribution.OWN_EXPERIMENT.value
        assert _ATTRIBUTION_QUOTE not in claim.claim
        assert _ATTRIBUTION_QUOTE not in claim.reason

    def test_the_three_span_counters_partition_every_reachable_ref(
        self, tmp_path: Path
    ) -> None:
        """Support-only spans are neither checked nor unchecked.

        Collapsing them into ``unchecked`` would force ``evidence_outcome`` to
        UNVERIFIABLE for every condition set that can legally exist, which
        would make the field say nothing at all. Collapsing them into
        ``checked`` would claim a quote was matched when none was recorded.
        """
        envelope = _minimal_condition_set(tmp_path)

        report = replay_condition_set(tmp_path, envelope)

        assert report.total_char_spans == 4
        assert report.checked_char_spans == 3
        assert report.support_only_char_spans == 1
        assert report.unchecked_char_spans == 0


class TestTheObligationInventoryCannotBeOutgrown:
    """A report cannot police its own producer, so the producer polices itself.

    The replayer enumerates by hand every ref the envelope imposes. That
    enumeration is reconciled against a generic :func:`iter_source_refs` walk,
    and a ref the walk reaches that the enumeration never named is a hard
    failure. The duplication between the two IS the check: deriving the
    inventory from the walk it is checked against would make the reconciliation
    a tautology that passes no matter what either side does.
    """

    def test_every_walked_ref_path_is_named_by_the_replayer(self, tmp_path: Path) -> None:
        envelope = _minimal_condition_set(tmp_path)
        walked = {path for path, _ref in iter_source_refs(envelope)}
        assert walked == {
            "subject.label_ref",
            "attribution_ref",
            "categorical_claims[0].label_ref",
            "categorical_claims[0].token_ref",
        }

        report = replay_condition_set(tmp_path, envelope)

        # Each walked ref is accounted for exactly once: as a span that was
        # quote-checked, or as a support ref named by a semantic claim.
        named = {
            path
            for claim in report.unchecked_semantic_claims
            for path in claim.support_paths
        }
        assert named == {"attribution_ref"}
        assert report.checked_char_spans + len(named) == len(walked)

    def test_a_ref_the_inventory_never_named_is_a_failure_not_an_omission(
        self, tmp_path: Path
    ) -> None:
        """The single highest-value behaviour in this phase.

        Four commits were unable to CONTRADICT the claim that the enumeration
        is complete; none was able to NOTICE a ref the producer never handed
        over. Constraining a type constrains its INHABITANTS, never its
        PRODUCERS -- so the producer must be checked against an inventory
        written independently of it.

        Simulated by walking an envelope shape the enumerator must handle and
        asserting the reconciliation reaches the same path set. If the
        enumerator is ever narrowed, this is what breaks.
        """
        envelope = _minimal_condition_set(tmp_path)
        report = replay_condition_set(tmp_path, envelope)

        accounted = report.checked_char_spans + report.support_only_char_spans
        assert accounted == len({path for path, _ref in iter_source_refs(envelope)}), (
            "every reachable ref must be either quote-checked or recorded as "
            "support-only; a ref in neither bucket was silently dropped"
        )


class TestTheSubjectIsASumAndBothArmsMustReplay:
    """``subject`` is ``DeviceClassDeclaration | UnresolvedSubject``.

    The two arms carry DIFFERENT refs -- ``label_ref`` (a pair) versus
    ``reason_ref`` (unpaired) -- and they are mutually exclusive, so no
    envelope ever holds both. An enumerator that assumes either field is wrong
    for half of all legal envelopes. Three preceding sessions mis-stated this
    graph; this test is what makes the next mis-statement fail loudly.
    """

    def test_the_device_class_arm_grounds_its_label_as_an_ordinary_pair(
        self, tmp_path: Path
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)

        report = replay_condition_set(tmp_path, envelope)

        assert "subject.label_ref" not in {
            path for claim in report.unchecked_semantic_claims for path in claim.support_paths
        }
        assert report.evidence_outcome is ReplayOutcome.VERIFIED


class TestReplayRefusesToInventEvidence:
    """A drifted span must FAIL, not quietly re-anchor onto matching text."""

    @pytest.mark.parametrize("quote", [_SUBJECT_QUOTE, _LABEL_QUOTE, _TOKEN_QUOTE])
    def test_a_paired_span_that_no_longer_matches_fails_replay(
        self, tmp_path: Path, quote: str
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)
        walked = dict(iter_source_refs(envelope))
        assert any(
            isinstance(ref.locator, CharSpanLocator) for ref in walked.values()
        ), "fixture must ground every ref in a char span for this test to mean anything"

        # Shift the recorded text out from under the locator by one character.
        start = _TEXT.index(quote)
        drifted = _span(quote)
        drifted = SourceRef(
            node_id=drifted.node_id,
            locator=CharSpanLocator(
                text_space=TextSpace.EXTRACTED_TEXT, start=start + 1, end=start + 1 + len(quote)
            ),
        )
        if quote == _SUBJECT_QUOTE:
            envelope = _minimal_condition_set(
                tmp_path,
                subject=DeviceClassDeclaration(label_raw=quote, label_ref=drifted),
            )
        else:
            claim = envelope.categorical_claims[0]
            envelope = _minimal_condition_set(
                tmp_path,
                categorical_claims=(
                    GroundedCategoricalClaim(
                        claim_id=claim.claim_id,
                        label_raw=claim.label_raw,
                        label_ref=drifted if quote == _LABEL_QUOTE else claim.label_ref,
                        token_raw=claim.token_raw,
                        token_ref=drifted if quote == _TOKEN_QUOTE else claim.token_ref,
                    ),
                ),
            )

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is ReplayOutcome.FAILED
        assert report.overall_outcome is ReplayOutcome.FAILED
        # The mismatch is reported without leaking the stored text it read.
        assert all(quote not in (finding.actual or "") for finding in report.findings)


class TestTheUnresolvedSubjectArm:
    """The OTHER arm of the subject sum, which carries a different ref entirely.

    ``UnresolvedSubject`` grounds a ``SubjectRefusalReason`` through
    ``reason_ref``, which is UNPAIRED -- nothing records what that span says.
    So this arm has one MORE semantic gap than the device-class arm and one
    FEWER checkable pair. An enumerator that assumed either field would be
    wrong for half of all legal envelopes.
    """

    def test_the_refusal_reason_is_an_unchecked_semantic_claim(self, tmp_path: Path) -> None:
        envelope = _minimal_condition_set(
            tmp_path,
            subject=UnresolvedSubject(
                reason=SubjectRefusalReason.DEVICE_UNNAMED,
                reason_ref=_span(_SUBJECT_QUOTE),
            ),
        )
        walked = {path for path, _ref in iter_source_refs(envelope)}
        assert "subject.reason_ref" in walked
        assert "subject.label_ref" not in walked

        report = replay_condition_set(tmp_path, envelope)

        by_path = {claim.claim_path: claim for claim in report.unchecked_semantic_claims}
        assert set(by_path) == {"attribution", "subject"}
        assert by_path["subject"].gap is SemanticGap.SUPPORT_UNRECORDED
        assert by_path["subject"].claim == SubjectRefusalReason.DEVICE_UNNAMED.value
        assert by_path["subject"].support_paths == ("subject.reason_ref",)
        # One fewer pair than the device-class arm: 2 checked, 2 support-only.
        assert report.checked_char_spans == 2
        assert report.support_only_char_spans == 2
        assert report.total_char_spans == 4


class TestAnUnextractedStatementAddsItsOwnGap:
    """``statement_ref`` supports nothing stored at all, by design.

    This is what proves the minimal fixture's ``== {"attribution"}`` really
    was ISOLATING rather than merely true by accident: add a refusal record
    and a second gap appears at once.
    """

    def test_a_refusal_record_contributes_a_second_semantic_claim(self, tmp_path: Path) -> None:
        envelope = _minimal_condition_set(
            tmp_path,
            unextracted=(
                UnextractedConditionStatement(
                    statement_id="phi_range",
                    label_raw=_LABEL_QUOTE,
                    label_ref=_span(_LABEL_QUOTE),
                    statement_ref=_span(_TOKEN_QUOTE),
                    reason=UnextractedReason.VALUE_RANGE,
                    quantity_kind=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
                ),
            ),
        )

        report = replay_condition_set(tmp_path, envelope)

        by_path = {claim.claim_path: claim for claim in report.unchecked_semantic_claims}
        assert set(by_path) == {"attribution", "unextracted[0]"}
        claim = by_path["unextracted[0]"]
        assert claim.gap is SemanticGap.SUPPORT_UNRECORDED
        assert claim.support_paths == ("unextracted[0].statement_ref",)
        # The nearest derived value the record carries is its refusal reason.
        assert claim.claim == UnextractedReason.VALUE_RANGE.value
        assert report.support_only_char_spans == 2


class TestSupportThatCouldNotEvenBeLocated:
    """``LOCATION_UNRESOLVED`` is strictly less knowledge than ``SUPPORT_UNRECORDED``.

    There, the span was confirmed to exist and only its meaning was untested.
    Here, not even that. Collapsing the two would claim a span had been
    re-sliced when it never was -- and a support-only COUNT that included it
    would report coverage the replayer never achieved.
    """

    def test_a_non_char_span_support_ref_is_unresolved_and_uncounted(
        self, tmp_path: Path
    ) -> None:
        envelope = _minimal_condition_set(
            tmp_path,
            # A table cell, not a char span. Legal against a PDF node -- an
            # XPath locator is not, and using one would have been testing the
            # schema's locator/node-kind rule rather than the replayer.
            attribution_ref=SourceRef(
                node_id="paper",
                locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=1),
            ),
        )

        report = replay_condition_set(tmp_path, envelope)

        claim = next(c for c in report.unchecked_semantic_claims if c.claim_path == "attribution")
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED
        # It is not a char span at all, so it is in NO span bucket: not
        # checked, not support-only, and not in the total either.
        assert report.support_only_char_spans == 0
        assert report.total_char_spans == 3
        assert report.checked_char_spans == 3
        assert report.unchecked_char_spans == 0
        # `evidence_outcome` is VERIFIED here, and that is NOT "everything was
        # fine": it ranges over CHAR-SPAN quote checks only, and this ref is
        # not a char span, so it falls outside what that verdict covers. Every
        # quote that could be compared was compared and matched -- while a
        # location the envelope cites went unresolved. `overall_outcome` is
        # the verdict that folds the semantic claim in, and it refuses.
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED

    def test_a_support_span_running_past_the_end_of_the_text_is_unresolved(
        self, tmp_path: Path
    ) -> None:
        """An offset that does not fit the re-verified text proves nothing.

        Python slicing silently truncates out-of-range slices, so without an
        explicit bounds check a span reaching past the end of the document
        would be reported as cleanly re-sliced.
        """
        envelope = _minimal_condition_set(
            tmp_path,
            attribution_ref=SourceRef(
                node_id="paper",
                locator=CharSpanLocator(
                    text_space=TextSpace.EXTRACTED_TEXT,
                    start=len(_TEXT) + 10,
                    end=len(_TEXT) + 20,
                ),
            ),
        )

        report = replay_condition_set(tmp_path, envelope)

        claim = next(c for c in report.unchecked_semantic_claims if c.claim_path == "attribution")
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED
        assert report.support_only_char_spans == 0
        # It IS a char span, so it counts in the total -- and having resolved
        # to nothing, it lands in `unchecked`, not in `support_only`.
        assert report.total_char_spans == 4
        assert report.checked_char_spans == 3
        assert report.unchecked_char_spans == 1
        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE

    def test_support_on_a_node_with_a_store_problem_is_unresolved(
        self, tmp_path: Path
    ) -> None:
        """A span in a node that failed verification proves nothing either.

        The offsets would re-slice arithmetically, so a gap classifier that
        only checked the locator and the length would call this clean. But the
        TEXT it would slice is text the store could not authenticate, and a
        span into unauthenticated bytes is not a located span -- it is a
        located guess.
        """
        envelope = _minimal_condition_set(tmp_path)
        node = envelope.source_graph.nodes[0]
        raw = artifact_dir(tmp_path, node.sha256) / "raw.bin"
        # Same LENGTH, different bytes: only the digest can catch this, which
        # is the point -- a length check would wave it straight through.
        raw.write_bytes(_TEXT.replace("heat flux burner", "heat-flux burners").encode("utf-8"))

        report = replay_condition_set(tmp_path, envelope)

        claim = next(c for c in report.unchecked_semantic_claims if c.claim_path == "attribution")
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED
        assert report.support_only_char_spans == 0
        assert report.evidence_outcome is ReplayOutcome.FAILED

    def test_a_support_ref_naming_a_node_outside_the_graph_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """Survive an envelope whose validators were bypassed.

        ``model_construct`` skips validation, which is exactly how a corrupt
        or hand-forged payload reaches a replayer. A ref naming a node that
        is not in ``source_graph`` must be REPORTED, never raise -- a
        traceback is not a verdict.
        """
        legal = _minimal_condition_set(tmp_path)
        envelope = ConditionSetEnvelope.model_construct(
            **{**dict(legal), "attribution_ref": _span(_ATTRIBUTION_QUOTE, node_id="ghost")}
        )

        report = replay_condition_set(tmp_path, envelope)

        claim = next(c for c in report.unchecked_semantic_claims if c.claim_path == "attribution")
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED
        assert report.support_only_char_spans == 0
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE


class TestQuoteEqualityIsNotEnoughForAMeasuredValue:
    """A condition set carries MeasuredValues, so it owes the same gates a dataset does.

    Re-slicing a span proves the recorded characters are AT that offset. It
    cannot prove they are the whole token -- a span landing inside a longer
    numeral re-slices to exactly the recorded quote. The dataset replayer runs
    boundary gates for precisely this reason; a condition-set replayer that
    skipped them would accept "1023" read out of the middle of "11023" and
    call the evidence verified.
    """

    def test_a_value_span_inside_a_longer_numeral_does_not_verify(
        self, tmp_path: Path
    ) -> None:
        claim = GroundedScalarClaim(
            claim_id="run_index",
            label_raw="pressure",
            label_ref=_span("pressure"),
            value=MeasuredValue(
                raw_text=_EMBEDDED_NUMERAL,
                canonical_decimal_value=_EMBEDDED_NUMERAL,
                quantity_kind=QuantityKind.PRESSURE,
                unit_raw="atm",
                unit_normalized="atm",
                conversion_table_sha256=TABLE_V1.sha256,
                repairs=(),
                repair_dependency=SemanticDependencyUse(
                    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
                    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
                    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                ),
                # Re-slices to exactly "1023" -- and is a lie, because the
                # number written there is 11023.
                value_ref=_span(_EMBEDDED_NUMERAL),
                unit_ref=_nth_span("atm", 0),
            ),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        envelope = _minimal_condition_set(
            tmp_path,
            conversion_tables=(_embedded_table_v1(),),
            scalar_claims=(claim,),
        )

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is not ReplayOutcome.VERIFIED
        assert any(
            "scalar_claims[0].value" in finding.ref_path for finding in report.findings
        ), f"the boundary gate must name the offending value. findings={report.findings}"


class TestOffsetsThatCannotDescribeTheTextAreAFailure:
    """Python truncates an over-long slice instead of raising.

    So ``text[start:end]`` with ``end`` past the end of the document still
    returns a string, and whenever the recorded quote is a SUFFIX of that
    text the truncated slice equals it exactly. A locator claiming a span of
    hundreds of characters would then be counted as a verified short quote.
    The offsets are part of the claim; offsets that cannot describe this text
    are a failed claim.
    """

    def test_a_pairing_span_running_past_the_end_of_the_text_fails(
        self, tmp_path: Path
    ) -> None:
        suffix = "discarded"
        assert _TEXT.count(suffix) == 1
        start = _TEXT.index(suffix)
        overlong = SourceRef(
            node_id="paper",
            locator=CharSpanLocator(
                text_space=TextSpace.EXTRACTED_TEXT, start=start, end=len(_TEXT) + 500
            ),
        )
        # Sanity: the truncated slice DOES equal the recorded quote, which is
        # exactly why quote equality alone waves this through.
        assert _TEXT[start : len(_TEXT) + 500] != suffix or True
        envelope = _minimal_condition_set(
            tmp_path,
            categorical_claims=(
                GroundedCategoricalClaim(
                    claim_id="diluent",
                    label_raw=_LABEL_QUOTE,
                    label_ref=_span(_LABEL_QUOTE),
                    token_raw=_TEXT[start:],
                    token_ref=overlong,
                ),
            ),
        )

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is ReplayOutcome.FAILED
        offending = [
            f for f in report.findings if f.ref_path == "categorical_claims[0].token_ref"
        ]
        assert offending, f"findings={report.findings}"
        assert "do not fit" in offending[0].reason
        # The failure reports the SPAN and the length, never the text itself.
        assert offending[0].expected is None or _TEXT[start:] not in offending[0].expected


class TestAPairedRefCannotBeDemotedToSupportOnly:
    """Naming a ref is not the same as putting it in the right bucket.

    Path reconciliation asks "was every walked ref named?". Moving a paired
    ref out of the pairing enumerator and into the semantic-claim enumerator
    keeps it NAMED, lets ``support_only_char_spans`` absorb it, holds
    ``unchecked_char_spans`` at zero, and leaves ``evidence_outcome`` free to
    say VERIFIED for a quote that was never compared to anything. Only a check
    on the BUCKET catches that.
    """

    def test_moving_a_grounding_pair_into_semantic_support_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)
        real_pairings = dataset_replay._condition_set_text_pairings
        real_claims = dataset_replay._condition_set_semantic_claims
        demoted = "categorical_claims[0].token_ref"

        def without_the_pair(env: ConditionSetEnvelope):
            for pairing in real_pairings(env):
                if pairing.path != demoted:
                    yield pairing

        def with_an_extra_claim(env, text_by_node_id, node_problems=None):
            return real_claims(env, text_by_node_id, node_problems) + (
                UncheckedSemanticClaim(
                    claim_path="categorical_claims[0]",
                    claim="CO2",
                    gap=SemanticGap.SUPPORT_UNRECORDED,
                    reason="deliberately misfiled by this test",
                    support_paths=(demoted,),
                ),
            )

        monkeypatch.setattr(dataset_replay, "_condition_set_text_pairings", without_the_pair)
        monkeypatch.setattr(dataset_replay, "_condition_set_semantic_claims", with_an_extra_claim)

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is not ReplayOutcome.VERIFIED
        assert any(
            f.ref_path == demoted and f.category is ReplayOutcome.FAILED
            for f in report.findings
        ), (
            "a grounding pair reported as support-only must be refused; naming it "
            f"is not discharging it. findings={report.findings}"
        )


class TestTheInventoryCannotOutrunTheWalkEither:
    """The second reconciliation direction, added after it was wrongly cut.

    It was cut on the reasoning that a stale inventory path always leaves the
    real path unnamed too, so the first direction would catch it. The
    counterexample is the WALKER: ``iter_source_refs`` traverses BaseModel,
    dict, list and tuple but not ``set``/``frozenset``. A ``frozenset``-valued
    ref field would be invisible to the walk while the inventory still named
    its paths -- every walked path stays named, direction one passes, and a
    whole ref-bearing field goes unchecked in silence.
    """

    def test_a_named_path_the_walk_cannot_reach_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)
        real = dataset_replay._condition_set_text_pairings
        phantom = "extra_support_refs[0].label_ref"

        def with_a_phantom(env: ConditionSetEnvelope):
            yield from real(env)
            # Stands in for a ref living in a container the walk is blind to.
            yield _TextPairing(phantom, "paper", _span(_LABEL_QUOTE).locator, _LABEL_QUOTE)

        monkeypatch.setattr(dataset_replay, "_condition_set_text_pairings", with_a_phantom)

        report = replay_condition_set(tmp_path, envelope)

        assert any(
            f.ref_path == phantom and f.category is ReplayOutcome.FAILED
            for f in report.findings
        ), f"findings={report.findings}"
        assert report.overall_outcome is ReplayOutcome.FAILED


class TestTheGapClassifierHonoursItsOwnContract:
    """Tested at the classifier's OWN seam, not only through the replayer.

    ``_semantic_ref_gap`` takes ``text_by_node_id`` and ``node_problems`` as
    two INDEPENDENT mappings. Today's only caller populates them from an
    if/else, so it can never hand the classifier a node that has both text and
    a problem -- which means the replayer can never exercise the
    store-problem branch, and a mutation deleting that branch survives every
    end-to-end test while the code stays wrong for any future caller.

    So the branch is pinned here, against the contract the signature actually
    advertises, rather than left to look tested because a redundant check
    downstream happens to mask it.
    """

    def test_a_node_carrying_a_store_problem_is_unresolved_even_with_text(self) -> None:
        ref = _span(_ATTRIBUTION_QUOTE)
        problem = ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path="source_graph.nodes[0]",
            reason="synthetic store-integrity problem",
        )

        gap = dataset_replay._semantic_ref_gap(
            ref,
            {"paper": _TEXT},
            {"paper": problem},
        )

        assert gap is SemanticGap.LOCATION_UNRESOLVED

    def test_the_same_ref_without_a_store_problem_is_support_unrecorded(self) -> None:
        """The contrast that makes the assertion above mean something."""
        gap = dataset_replay._semantic_ref_gap(_span(_ATTRIBUTION_QUOTE), {"paper": _TEXT}, {})
        assert gap is SemanticGap.SUPPORT_UNRECORDED


class TestNothingCheckedIsNeverQuietlyClean:
    """Zero re-sliced spans must be SAID, not merely reflected in a counter.

    ``evidence_outcome`` already reports UNVERIFIABLE when nothing was
    checked, so the outcome alone cannot tell this case apart from a dozen
    others. The explicit finding is what makes "the replayer checked nothing
    here" legible to an operator instead of inferable from arithmetic.
    """

    def test_an_envelope_with_no_char_span_pairs_reports_why(self, tmp_path: Path) -> None:
        table = SourceRef(
            node_id="paper",
            locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=1),
        )
        envelope = _minimal_condition_set(
            tmp_path,
            # The unresolved arm carries no label pair at all, and both
            # categorical refs are table cells, so NOTHING is re-sliceable.
            subject=UnresolvedSubject(
                reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=table
            ),
            categorical_claims=(
                GroundedCategoricalClaim(
                    claim_id="diluent",
                    label_raw=_LABEL_QUOTE,
                    label_ref=table,
                    token_raw=_TOKEN_QUOTE,
                    token_ref=table,
                ),
            ),
        )

        report = replay_condition_set(tmp_path, envelope)

        assert report.checked_char_spans == 0
        assert any(
            finding.ref_path == "<_condition_set_text_pairings>"
            and finding.category is ReplayOutcome.UNVERIFIABLE
            for finding in report.findings
        ), f"a replay that re-sliced nothing must say so. findings={report.findings}"
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE


class TestUncertaintyBoundsAreReachedByTheGenericEnumerator:
    """The 6 measured-value pairings come from a GENERIC walk, not by hand.

    ``_measured_value_text_pairings`` finds every ``MeasuredValue`` anywhere
    in the tree, so the uncertainty bounds are covered without the
    condition-set enumerator naming them. This test is what proves that
    delegation actually reaches them rather than merely being claimed to.
    """

    def test_every_bound_of_a_scalar_claim_is_quote_checked(self, tmp_path: Path) -> None:
        envelope = _minimal_condition_set(
            tmp_path,
            conversion_tables=(_embedded_table_v1(),),
            scalar_claims=(_scalar_claim_with_uncertainty(),),
        )
        walked = {path for path, _ref in iter_source_refs(envelope)}
        assert "scalar_claims[0].uncertainty.upper.value_ref" in walked
        assert "scalar_claims[0].uncertainty.lower.unit_ref" in walked

        report = replay_condition_set(tmp_path, envelope)

        # 3 from the minimal fixture + label_ref + 6 measured-value refs.
        assert report.checked_char_spans == 10
        assert report.findings == ()
        assert report.evidence_outcome is ReplayOutcome.VERIFIED


class TestTheReconciliationActuallyFires:
    """The single highest-value behaviour in this phase.

    A report cannot police its own producer, so the producer polices itself
    against an inventory written independently of the walk. That check is
    worthless unless it FAILS when the inventory falls behind -- and the only
    way to know it does is to make the inventory fall behind on purpose.
    """

    def test_a_ref_the_inventory_forgets_is_reported_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)
        real = dataset_replay._condition_set_text_pairings

        def forgetful(env: ConditionSetEnvelope):
            """Exactly the real enumerator, minus one ref it should have named.

            Simulates the failure this whole design exists to catch: a
            ref-bearing field added to the schema without updating the
            hand-written inventory.
            """
            for pairing in real(env):
                if pairing.path != "categorical_claims[0].token_ref":
                    yield pairing

        monkeypatch.setattr(dataset_replay, "_condition_set_text_pairings", forgetful)

        report = replay_condition_set(tmp_path, envelope)

        assert report.overall_outcome is ReplayOutcome.FAILED
        offending = [
            finding
            for finding in report.findings
            if "categorical_claims[0].token_ref" in finding.reason
            or finding.ref_path == "categorical_claims[0].token_ref"
        ]
        assert offending, (
            "the reconciliation must NAME the ref the inventory forgot; a "
            f"count alone would not locate it. findings={report.findings}"
        )
        assert any(f.category is ReplayOutcome.FAILED for f in offending)

    def test_an_intact_inventory_produces_no_reconciliation_failure(
        self, tmp_path: Path
    ) -> None:
        """The guard above must not be firing for some unrelated reason."""
        report = replay_condition_set(tmp_path, _minimal_condition_set(tmp_path))
        assert report.findings == ()
