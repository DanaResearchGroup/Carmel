"""The condition-set producer: the first path that MAKES what the replayer reads.

Until this module existed, every ``ConditionSetEnvelope`` in the suite was
hand-built in test code, so ``replay_condition_set`` had never been handed real
producer output. The round-trip class below is the point of the whole file: it
is the first test in this repo where a PRODUCED condition set is stored, read
back, and independently re-verified.

Every fixture here is SYNTHETIC. No text from any real paper appears in this
repository.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ConditionAttribution,
    DeviceClassDeclaration,
    SubjectRefusalReason,
    UnextractedReason,
    UnresolvedSubject,
)
from carmel.services import units
from carmel.services.condition_set_bridge import (
    load_condition_set_envelope,
    store_condition_set_envelope,
)
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    ConditionSetProducerError,
    DeviceClassSpec,
    ScalarConditionSpec,
    UnextractedConditionSpec,
    UnresolvedSubjectSpec,
    produce_condition_set_from_artifact,
)
from carmel.services.dataset_producer import DatasetProducerError, QuoteGroundingError
from carmel.services.dataset_replay import ReplayOutcome, replay_condition_set
from tests.test_dataset_producer import _store_synthetic_artifact

#: A synthetic methods paragraph. Invented wholesale -- it is not, and must never
#: be, an excerpt of any real paper. It states conditions the way running prose
#: actually states them: a named apparatus, some resolvable scalars, a token, and
#: one condition that genuinely cannot be reduced to a single number.
_METHODS_TEXT = (
    "2. Experimental methods\n"
    "Measurements were carried out in a jet-stirred reactor of fused silica.\n"
    "The initial temperature was 823 K and the pressure was held at 1.2 atm.\n"
    "The fuel was methane in all cases.\n"
    "The equivalence ratio was varied from 0.6 to 1.4 across the campaign.\n"
)


def _subject() -> DeviceClassSpec:
    return DeviceClassSpec(label_quote="jet-stirred reactor")


def _temperature() -> ScalarConditionSpec:
    return ScalarConditionSpec(
        claim_id="initial_temperature",
        label_quote="initial temperature",
        quantity_kind=units.QuantityKind.TEMPERATURE,
        value_quote="823",
        unit_quote="K",
    )


def _pressure() -> ScalarConditionSpec:
    return ScalarConditionSpec(
        claim_id="pressure",
        label_quote="pressure",
        quantity_kind=units.QuantityKind.PRESSURE,
        value_quote="1.2",
        unit_quote="atm",
    )


def _fuel() -> CategoricalConditionSpec:
    return CategoricalConditionSpec(
        claim_id="fuel",
        label_quote="fuel",
        token_quote="methane",
    )


def _swept_phi() -> UnextractedConditionSpec:
    return UnextractedConditionSpec(
        statement_id="equivalence_ratio",
        label_quote="equivalence ratio",
        statement_quote="varied from 0.6 to 1.4",
        reason=UnextractedReason.MULTI_VALUED_SWEEP,
        quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
    )


def _produce(tmp_path: Path, **overrides: object) -> object:
    stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
    kwargs: dict[str, object] = {
        "sha256": stored.sha256,
        "attribution": ConditionAttribution.OWN_EXPERIMENT,
        "attribution_quote": "Measurements were carried out",
        "subject": _subject(),
        "scalars": (_temperature(), _pressure()),
        "categoricals": (_fuel(),),
        "unextracted": (_swept_phi(),),
    }
    kwargs.update(overrides)
    return produce_condition_set_from_artifact(tmp_path, **kwargs)  # type: ignore[arg-type]


class TestAProducedConditionSetSurvivesItsOwnReplay:
    """The gap this module was written to close.

    A hand-built envelope proves the replayer can read a shape the TEST author
    constructed. Only a PRODUCED envelope proves the two agree about the shape
    that the runtime actually emits -- which is the only agreement that matters
    once real extraction runs.
    """

    def test_produce_store_load_replay_reports_verified_evidence(self, tmp_path: Path) -> None:
        """The full loop, with the envelope re-read from the store rather than
        replayed from the in-memory object the producer returned."""
        envelope = _produce(tmp_path)
        stored = store_condition_set_envelope(tmp_path, envelope)
        reloaded = load_condition_set_envelope(tmp_path, stored.sha256)
        assert reloaded == envelope

        report = replay_condition_set(tmp_path, reloaded)
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.findings == ()
        # Not vacuous: spans were actually re-sliced, and every one was either
        # CHECKED or accounted for as support-only. A condition set legitimately
        # carries spans that no pass/fail check can consume -- the attribution,
        # a subject refusal, and a refused statement locate text without
        # asserting a value. They are SUPPORT, not unchecked work, and the two
        # must never be conflated: `unchecked_char_spans` must still be zero.
        assert report.total_char_spans > 0
        assert report.unchecked_char_spans == 0
        assert report.checked_char_spans + report.support_only_char_spans == (report.total_char_spans)
        assert report.support_only_char_spans > 0

    def test_the_replay_is_not_silently_skipping_the_produced_claims(self, tmp_path: Path) -> None:
        """Guards the way this test could rot into a tautology: if the producer
        emitted an envelope the replayer walked no part of, the assertions above
        would still pass. Pin that every produced claim is reached."""
        envelope = _produce(tmp_path)
        report = replay_condition_set(tmp_path, envelope)
        # 2 scalars (label+value+unit) + 1 categorical (label+token)
        # + 1 unextracted (label+statement) + subject label + attribution.
        assert report.total_char_spans >= 2 * 3 + 2


class TestTheProducerRefusesRatherThanFabricates:
    def test_an_empty_condition_set_is_refused(self, tmp_path: Path) -> None:
        """An envelope with no claims and no refusals asserts that the paper
        stated no conditions -- which grounding cannot establish. Emitting it
        would produce a vacuously VERIFIED envelope, the exact overclaim shape
        this system exists to prevent."""
        with pytest.raises(ConditionSetProducerError, match="no scalar claims"):
            _produce(tmp_path, scalars=(), categoricals=(), unextracted=())

    def test_a_refusal_alone_is_enough_to_produce(self, tmp_path: Path) -> None:
        """The mirror of the above, and the more important half: a condition set
        that resolved NOTHING but recorded WHY is a legitimate result. The
        narrow honest slice blesses refusals; it does not require yield."""
        envelope = _produce(tmp_path, scalars=(), categoricals=())
        assert envelope.scalar_claims == ()
        assert len(envelope.unextracted) == 1
        assert replay_condition_set(tmp_path, envelope).evidence_outcome is ReplayOutcome.VERIFIED

    def test_duplicate_claim_ids_are_refused(self, tmp_path: Path) -> None:
        """Two claims sharing an id makes every per-claim finding ambiguous
        about which claim it means."""
        with pytest.raises(ConditionSetProducerError, match="duplicate id"):
            _produce(tmp_path, scalars=(_temperature(), _temperature()))

    def test_a_claim_id_may_not_collide_across_the_two_claim_kinds(self, tmp_path: Path) -> None:
        """Scalar and categorical claims share one id namespace, because a
        report path names a claim without naming its kind."""
        clashing = CategoricalConditionSpec(claim_id="pressure", label_quote="fuel", token_quote="methane")
        with pytest.raises(ConditionSetProducerError, match="duplicate id"):
            _produce(tmp_path, categoricals=(clashing,))

    def test_duplicate_statement_ids_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConditionSetProducerError, match="duplicate id"):
            _produce(tmp_path, unextracted=(_swept_phi(), _swept_phi()))

    def test_a_claim_id_may_not_collide_with_a_statement_id(self, tmp_path: Path) -> None:
        """``claim_id`` and ``statement_id`` share ONE namespace -- that is what
        ``ConditionSetEnvelope`` itself validates. Checking the two kinds
        separately let this collision through the producer and surface as a
        pydantic ValidationError from inside construction: a late refusal that
        does not name the caller's actual mistake."""
        colliding = UnextractedConditionSpec(
            statement_id="pressure",
            label_quote="equivalence ratio",
            statement_quote="varied from 0.6 to 1.4",
            reason=UnextractedReason.MULTI_VALUED_SWEEP,
        )
        with pytest.raises(ConditionSetProducerError, match="duplicate id"):
            _produce(tmp_path, unextracted=(colliding,))

    def test_a_quote_absent_from_the_document_cannot_be_grounded(self, tmp_path: Path) -> None:
        """The producer never invents a location for a quote it cannot find."""
        absent = ScalarConditionSpec(
            claim_id="t",
            label_quote="initial temperature",
            quantity_kind=units.QuantityKind.TEMPERATURE,
            value_quote="9999",
            unit_quote="K",
        )
        with pytest.raises(QuoteGroundingError):
            _produce(tmp_path, scalars=(absent,))

    def test_a_missing_current_record_refusal_never_calls_itself_a_dataset(self, tmp_path: Path) -> None:
        """The preamble's OTHER refusal, and the one that was actually wrong.

        ``_prepare_grounding`` took its refusal wording from two keyword
        arguments that both defaulted to the DATASET path's phrasing. This
        producer overrode only one of them, so this message read "refusing to
        produce a condition set. A dataset must be grounded in a genuinely
        stored extraction" -- a refusal that names the wrong artifact in its
        own second sentence. Nothing caught it because the only test covering
        the shared preamble from this side matched on the noun that WAS
        overridden. Both arguments are now required.
        """
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        # Delete the stored extraction record, then call the producer DIRECTLY:
        # the shared `_produce` helper re-stores the artifact (and so re-creates
        # the record) on its way in, which would undo the setup.
        for record in (tmp_path / "evidence" / "literature" / stored.sha256 / "extractions").glob("*"):
            shutil.rmtree(record)
        with pytest.raises(DatasetProducerError) as excinfo:
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=_subject(),
                scalars=(_temperature(),),
            )
        message = str(excinfo.value)
        assert "condition set" in message
        assert "A dataset" not in message, message

    def test_a_lossy_extraction_is_refused_and_names_a_condition_set(self, tmp_path: Path) -> None:
        """The shared preamble's refusal, reached through this producer. The
        message must name a CONDITION SET -- a refusal that misnames what it
        refused sends a reader to the wrong producer."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT, lossy=True)
        with pytest.raises(DatasetProducerError, match="condition set"):
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=_subject(),
                scalars=(_temperature(),),
            )


class TestTheStrEnumTrapIsClosedOnEverySpec:
    """``StrEnum`` members compare ``==`` equal to their own string values, so a
    caller passing a bare string builds a spec that looks right under ``==`` and
    is not actually the member. Every enum-typed spec field checks explicitly."""

    def test_a_bare_string_attribution_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConditionSetProducerError, match="ConditionAttribution"):
            _produce(tmp_path, attribution="own_experiment")

    def test_a_bare_string_quantity_kind_is_refused(self) -> None:
        with pytest.raises(ConditionSetProducerError, match="QuantityKind"):
            ScalarConditionSpec(
                claim_id="t",
                label_quote="initial temperature",
                quantity_kind="temperature",  # type: ignore[arg-type]
                value_quote="823",
                unit_quote="K",
            )

    def test_a_bare_string_unextracted_reason_is_refused(self) -> None:
        with pytest.raises(ConditionSetProducerError, match="UnextractedReason"):
            UnextractedConditionSpec(
                statement_id="phi",
                label_quote="equivalence ratio",
                statement_quote="varied from 0.6 to 1.4",
                reason="multi_valued_sweep",  # type: ignore[arg-type]
            )

    def test_a_bare_string_refused_quantity_kind_is_refused(self) -> None:
        """The one spec field where this check was missing. Left open, a bare
        string is coerced downstream by pydantic and the refusal surfaces as a
        schema error from inside construction instead of naming the field."""
        with pytest.raises(ConditionSetProducerError, match="QuantityKind"):
            UnextractedConditionSpec(
                statement_id="phi",
                label_quote="equivalence ratio",
                statement_quote="varied from 0.6 to 1.4",
                reason=UnextractedReason.MULTI_VALUED_SWEEP,
                quantity_kind="equivalence_ratio",  # type: ignore[arg-type]
            )

    def test_a_bare_string_subject_refusal_reason_is_refused(self) -> None:
        with pytest.raises(ConditionSetProducerError, match="SubjectRefusalReason"):
            UnresolvedSubjectSpec(
                reason="device_unnamed",  # type: ignore[arg-type]
                reason_quote="Measurements were carried out",
            )

    @pytest.mark.parametrize("bad", [True, False])
    def test_a_bool_occurrence_is_refused_because_bool_is_an_int(self, bad: bool) -> None:
        """``isinstance(True, int)`` is True in Python, so a stray boolean flag
        would silently mean occurrence 1/0."""
        with pytest.raises(ConditionSetProducerError, match="bool is a subclass of int"):
            CategoricalConditionSpec(
                claim_id="fuel",
                label_quote="fuel",
                token_quote="methane",
                token_occurrence=bad,  # type: ignore[arg-type]
            )


class TestTheSubjectIsEitherNamedOrExplicitlyRefused:
    def test_a_named_device_class_is_grounded(self, tmp_path: Path) -> None:
        envelope = _produce(tmp_path)
        assert isinstance(envelope.subject, DeviceClassDeclaration)
        assert envelope.subject.label_raw == "jet-stirred reactor"

    def test_an_unresolved_subject_still_grounds_its_reason(self, tmp_path: Path) -> None:
        """A refusal that points at the text motivating it can be checked. A
        refusal with no span has to be taken on trust, which is not evidence."""
        envelope = _produce(
            tmp_path,
            subject=UnresolvedSubjectSpec(
                # Deliberately NOT the most obvious member: a mutation audit
                # showed that testing with DEVICE_UNNAMED let a producer that
                # hardcoded DEVICE_UNNAMED survive. A test whose expected value
                # equals the value a bug would guess discriminates nothing.
                reason=SubjectRefusalReason.ASSIGNMENT_DEPENDS_ON_RESULT,
                reason_quote="Experimental methods",
            ),
        )
        assert isinstance(envelope.subject, UnresolvedSubject)
        assert envelope.subject.reason is SubjectRefusalReason.ASSIGNMENT_DEPENDS_ON_RESULT
        assert replay_condition_set(tmp_path, envelope).evidence_outcome is ReplayOutcome.VERIFIED


class TestARefusedConditionKeepsItsReasonAndItsSpan:
    def test_the_sweep_is_recorded_as_a_refusal_not_as_a_number(self, tmp_path: Path) -> None:
        """'varied from 0.6 to 1.4' must never become the scalar 0.6. A sweep
        squeezed into one number is a fabricated condition."""
        envelope = _produce(tmp_path)
        assert {c.claim_id for c in envelope.scalar_claims} == {"initial_temperature", "pressure"}
        statement = envelope.unextracted[0]
        assert statement.reason is UnextractedReason.MULTI_VALUED_SWEEP
        assert statement.quantity_kind is units.QuantityKind.EQUIVALENCE_RATIO

    def test_an_unknown_quantity_kind_is_absent_not_guessed(self, tmp_path: Path) -> None:
        """A refused statement whose quantity is not known records ABSENCE. It
        does not pick a plausible kind, and absence is not the same as a kind."""
        spec = UnextractedConditionSpec(
            statement_id="phi",
            label_quote="equivalence ratio",
            statement_quote="varied from 0.6 to 1.4",
            reason=UnextractedReason.MULTI_VALUED_SWEEP,
        )
        envelope = _produce(tmp_path, unextracted=(spec,))
        assert isinstance(envelope.unextracted[0].quantity_kind, Absent)


class TestTheProducedValuesAreDerivedNotAsserted:
    def test_the_scalar_carries_its_raw_quote_and_normalized_unit(self, tmp_path: Path) -> None:
        envelope = _produce(tmp_path)
        temperature = next(c for c in envelope.scalar_claims if c.claim_id == "initial_temperature")
        assert temperature.value.raw_text == "823"
        assert temperature.value.unit_raw == "K"

    def test_an_unknown_unit_for_the_quantity_kind_is_refused(self, tmp_path: Path) -> None:
        """A unit that is not a known unit or alias of the stated quantity kind
        cannot be normalized, so no value may be produced from it."""
        wrong = ScalarConditionSpec(
            claim_id="t",
            label_quote="initial temperature",
            quantity_kind=units.QuantityKind.TEMPERATURE,
            value_quote="823",
            unit_quote="atm",
        )
        with pytest.raises(QuoteGroundingError, match="not a registered spelling"):
            _produce(tmp_path, scalars=(wrong,))

    def test_the_refusal_names_the_claim_not_an_axis(self, tmp_path: Path) -> None:
        """A condition set has no axes. A refusal borrowing the dataset
        producer's vocabulary would send a reader looking for a series that
        does not exist."""
        wrong = ScalarConditionSpec(
            claim_id="initial_temperature",
            label_quote="initial temperature",
            quantity_kind=units.QuantityKind.TEMPERATURE,
            value_quote="fused",
            unit_quote="K",
        )
        with pytest.raises(DatasetProducerError, match="claim 'initial_temperature'"):
            _produce(tmp_path, scalars=(wrong,))


class TestAbsenceReasonsAreNotInterchangeable:
    """``NOT_EXTRACTED_YET`` and ``NOT_REPORTED_HERE`` are different claims about
    the world, and a producer must not trade one for the other.

    This producer reads no uncertainty from the document, so the honest reason is
    NOT_EXTRACTED_YET -- "we have not looked". ``NOT_REPORTED_HERE`` would assert
    something stronger and false: that the PAPER states no uncertainty. A reader
    deciding whether to go back to the source is misled by exactly that swap, and
    a mutation swapping the two survived the whole suite until this test existed.
    """

    def test_a_produced_scalar_records_not_extracted_yet_not_not_reported(self, tmp_path: Path) -> None:
        envelope = _produce(tmp_path)
        for claim in envelope.scalar_claims:
            assert isinstance(claim.uncertainty, Absent)
            assert claim.uncertainty.reason is AbsenceReason.NOT_EXTRACTED_YET

    def test_a_refused_statement_of_unknown_kind_says_not_extracted_yet(self, tmp_path: Path) -> None:
        """Same rule on the refusal path: an unknown quantity kind is one we did
        not extract, not one the paper declined to state."""
        spec = UnextractedConditionSpec(
            statement_id="phi",
            label_quote="equivalence ratio",
            statement_quote="varied from 0.6 to 1.4",
            reason=UnextractedReason.MULTI_VALUED_SWEEP,
        )
        envelope = _produce(tmp_path, unextracted=(spec,))
        kind = envelope.unextracted[0].quantity_kind
        assert isinstance(kind, Absent)
        assert kind.reason is AbsenceReason.NOT_EXTRACTED_YET


class TestSpanStitchingIsRefused:
    """The P0-d refusal: a scalar assembled from spans that do not cohere.

    This class REPLACED a characterization of the same shape. Until this rule
    landed, a caller could stitch the label "pressure" to the value "823" and
    the unit "atm" -- a pressure of 823 atm the paper never states -- and every
    span grounded, and replay reported VERIFIED with zero findings. The old
    class asserted that behaviour, with a docstring saying a green run meant
    the hole was still open. Found by adversarial review (Codex round 92), not
    by the suite.

    The defence is NOT co-location: the false triple is drawn from ONE
    sentence, so any bounded measurement window would still bless it. It is a
    rule about which quotes the paper predicates of which, enforced as a
    REFUTATION over the span covering the claim's three grounds -- see
    :mod:`carmel.services.stitching`.

    Passing this gate is not verification, and no test here should ever be
    written to assert that it is.
    """

    def _stitched(self) -> ScalarConditionSpec:
        return ScalarConditionSpec(
            claim_id="fabricated_pressure",
            label_quote="pressure",
            quantity_kind=units.QuantityKind.PRESSURE,
            value_quote="823",
            unit_quote="atm",
        )

    def _mislabeled(self) -> ScalarConditionSpec:
        """The sibling fabrication: an HONEST pressure value under a label that
        belongs to the temperature. Its window holds one PRESSURE pair, so a
        rule counting only same-kind pairs would bless it."""
        return ScalarConditionSpec(
            claim_id="mislabeled_pressure",
            label_quote="temperature",
            quantity_kind=units.QuantityKind.PRESSURE,
            value_quote="1.2",
            unit_quote="atm",
        )

    def test_the_producer_refuses_a_value_stolen_from_another_quantity(self, tmp_path: Path) -> None:
        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, scalars=(self._stitched(),), categoricals=(), unextracted=())
        assert "fabricated_pressure" in str(excinfo.value)

    def test_the_refusal_names_what_it_found_rather_than_only_that_it_refused(self, tmp_path: Path) -> None:
        """A refusal a reader cannot check is a refusal they must take on
        faith. The message names the window and both constructs in it."""
        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, scalars=(self._stitched(),), categoricals=(), unextracted=())
        message = str(excinfo.value)
        assert "823 K" in message
        assert "1.2 atm" in message

    def test_a_label_belonging_to_another_quantity_is_also_refused(self, tmp_path: Path) -> None:
        """Closing the stitch while leaving this open would be theatre: the two
        are the same defect reached from opposite ends."""
        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, scalars=(self._mislabeled(),), categoricals=(), unextracted=())
        assert "mislabeled_pressure" in str(excinfo.value)

    def test_the_honest_claims_the_same_sentence_states_still_produce(self, tmp_path: Path) -> None:
        """The rule must discriminate, not merely refuse. Both true readings of
        the very sentence the fabrication was drawn from survive it -- a gate
        that also killed these would be worthless at any yield."""
        envelope = _produce(tmp_path, categoricals=(), unextracted=())
        claims = {claim.claim_id: claim for claim in envelope.scalar_claims}  # type: ignore[attr-defined]
        assert set(claims) == {"initial_temperature", "pressure"}
        assert claims["initial_temperature"].value.raw_text == "823"
        assert claims["pressure"].value.raw_text == "1.2"

    def test_the_gate_can_fail_the_way_the_defect_fails(self, tmp_path: Path) -> None:
        """A verifier that cannot fail is not a verifier. With uniqueness
        disabled the fabrication is accepted again, which is what proves the
        uniqueness test -- not some neighbouring check -- is load-bearing."""
        from unittest.mock import patch

        with patch("carmel.services.stitching.refute_stitched_scalar", return_value=None):
            envelope = _produce(tmp_path, scalars=(self._stitched(),), categoricals=(), unextracted=())
        assert envelope.scalar_claims[0].value.raw_text == "823"  # type: ignore[attr-defined]
