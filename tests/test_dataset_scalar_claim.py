"""Tests for ``GroundedScalarClaim``: one scalar fact a paper states about an
experiment ("the initial pressure was 1 atm"), grounded label and all.

Kept in its own module rather than folded into test_dataset_schemas.py or
test_dataset_series.py: this type deliberately belongs to NEITHER of those
layers. It is not a series element (it has no axis, no point, and no
siblings), and it is not one of the M-D2a primitives (it composes several of
them). It exists because conditions -- reactor pressure, initial temperature,
shock-tube bore -- have no legal home in the schema today: ``Series`` requires
at least one COORDINATE axis, at least one OBSERVATION axis and at least one
point (S3/S4/S7), and ``DatasetEnvelope.series`` carries ``MinLen(1)``, so a
constants-only series and a series-free envelope are BOTH unrepresentable.

Fixtures here are self-contained by the same convention the other dataset test
modules follow: each builds its own valid ``MeasuredValue``/``SourceRef``
rather than importing another test module's private helpers.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    CoordinateFrame,
    GroundedScalarClaim,
    Maybe,
    MeasuredValue,
    QuantityKind,
    SemanticDependencyUse,
    SourceRef,
    TableCellLocator,
    TextSpace,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    iter_measured_values,
    iter_source_refs,
)
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import TABLE_V1

_CURRENT_REPAIR_DEPENDENCY = SemanticDependencyUse(
    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
)

_NO_UNCERTAINTY: Maybe[Uncertainty] = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)


def _frame() -> CoordinateFrame:
    return CoordinateFrame(
        render_fingerprint="fp-1",
        cropbox=("0", "0", "612", "792"),
        mediabox=("0", "0", "612", "792"),
        rotation=0,
        units="pt",
        dpi=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        render_settings=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
    )


def _bbox_ref(node_id: str = "n1") -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=BBoxLocator(bbox=BBox(frame=_frame(), x0="10", y0="20", x1="30", y1="40")),
    )


def _table_ref(node_id: str = "n1", row: int = 0, col: int = 1) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=row, col=col),
    )


def _char_span_ref(node_id: str = "n1", start: int = 0, end: int = 20) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end),
    )


def _measured_value(
    raw_text: str = "1.0",
    canonical_decimal_value: str = "1.0",
    quantity_kind: QuantityKind = QuantityKind.PRESSURE,
    unit_raw: str = "atm",
    unit_normalized: str = "atm",
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=canonical_decimal_value,
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=_bbox_ref(),
        unit_ref=_table_ref(),
    )


def _claim(**kwargs: object) -> GroundedScalarClaim:
    defaults: dict[str, object] = {
        "claim_id": "initial_pressure",
        "label_raw": "initial pressure",
        "label_ref": _char_span_ref(),
        "value": _measured_value(),
        "uncertainty": _NO_UNCERTAINTY,
    }
    defaults.update(kwargs)
    return GroundedScalarClaim(**defaults)  # type: ignore[arg-type]


class TestGroundedScalarClaimHoldsOneGroundedScalar:
    def test_a_well_formed_claim_keeps_every_field_it_was_given(self) -> None:
        claim = _claim()

        assert claim.claim_id == "initial_pressure"
        assert claim.label_raw == "initial pressure"
        assert claim.label_ref == _char_span_ref()
        assert claim.value.canonical_decimal_value == "1.0"
        assert claim.value.quantity_kind is QuantityKind.PRESSURE
        assert isinstance(claim.uncertainty, Absent)

    def test_the_claim_is_frozen_and_forbids_extra_fields(self) -> None:
        claim = _claim()

        with pytest.raises(ValidationError):
            claim.claim_id = "other"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            _claim(source_form="textual")

    def test_the_quantity_kind_is_read_off_the_value_not_declared_twice(self) -> None:
        """Unlike ``AxisDeclaration``, a scalar claim does NOT carry its own
        ``quantity_kind``: the declaration and the value co-locate here, so a
        second copy would be one fact stored twice in a content-addressed
        payload -- and every way the two could disagree is an error class that
        only the duplication creates."""
        assert "quantity_kind" not in GroundedScalarClaim.model_fields


class TestAClaimWithoutAGroundedLabelCannotBeConstructed:
    """The label is the load-bearing field, not decoration.

    Proving a number is an exact located substring of the source can never
    prove what the number MEANS -- that is this project's central finding, and
    for a standalone scalar the meaning lives entirely in the surrounding
    prose ("the initial pressure was 1 atm"). So the label carries its own
    ``SourceRef``, independent of the value's, exactly as ``MeasuredValue``
    splits ``value_ref`` from ``unit_ref`` and for the same reason.
    """

    def test_a_missing_label_ref_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            GroundedScalarClaim(  # type: ignore[call-arg]
                claim_id="initial_pressure",
                label_raw="initial pressure",
                value=_measured_value(),
                uncertainty=_NO_UNCERTAINTY,
            )

        assert "label_ref" in str(excinfo.value)

    def test_an_empty_label_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _claim(label_raw="")

    def test_an_absent_label_ref_is_not_even_a_representable_state(self) -> None:
        """``label_ref`` is ``SourceRef``, never ``Maybe[SourceRef]``: an
        ungrounded label is not a weaker claim to be recorded honestly, it is
        no claim at all."""
        with pytest.raises(ValidationError):
            _claim(label_ref=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))

    def test_an_absent_value_is_refused(self) -> None:
        """Mirrors ``Coordinate`` rather than ``Observation``: an observation
        may honestly be absent (the paper plotted a point it never tabulated),
        but a condition the paper never stated is not a condition with a
        missing number -- it is simply not a claim, and must not occupy a
        claim_id."""
        with pytest.raises(ValidationError):
            _claim(value=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))


class TestClaimIdIsARestrictedIdentifier:
    """Same rule, and the same reason, as ``axis_id``/``series_id``/``point_id``:
    a ``claim_id`` is interpolated into the dotted diagnostic paths
    :func:`iter_source_refs` builds, so ``.``, ``[``, ``]`` and non-ASCII would
    poison a path that is parsed positionally."""

    @pytest.mark.parametrize(
        "claim_id",
        [
            "Initial_Pressure",
            "1st_pressure",
            "initial.pressure",
            "initial[0]",
            "initial pressure",
            "phi_φ",
            "",
            "_leading",
        ],
    )
    def test_a_non_identifier_claim_id_is_refused(self, claim_id: str) -> None:
        with pytest.raises(ValidationError):
            _claim(claim_id=claim_id)

    @pytest.mark.parametrize("claim_id", ["p", "initial_pressure", "t5", "phi"])
    def test_a_plain_lowercase_identifier_is_accepted(self, claim_id: str) -> None:
        assert _claim(claim_id=claim_id).claim_id == claim_id


class TestUncertaintyBoundsAgreeWithTheValueTheyBound:
    """S13, reused rather than reimplemented: an ABSOLUTE bound is a magnitude
    in the value's own physical quantity, a RELATIVE bound is a fraction and
    must be ``RELATIVE_UNCERTAINTY``."""

    def _uncertainty(self, basis: UncertaintyBasis, bound: MeasuredValue) -> Uncertainty:
        return Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=basis,
            scale=UncertaintyScale.LINEAR,
            upper=bound,
            lower=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )

    def test_an_absolute_bound_in_the_values_own_quantity_is_accepted(self) -> None:
        claim = _claim(
            uncertainty=self._uncertainty(
                UncertaintyBasis.ABSOLUTE,
                _measured_value(raw_text="0.05", canonical_decimal_value="0.05"),
            )
        )

        assert not isinstance(claim.uncertainty, Absent)

    def test_an_absolute_bound_in_a_different_quantity_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _claim(
                uncertainty=self._uncertainty(
                    UncertaintyBasis.ABSOLUTE,
                    _measured_value(
                        raw_text="0.05",
                        canonical_decimal_value="0.05",
                        quantity_kind=QuantityKind.VELOCITY,
                        unit_raw="cm/s",
                        unit_normalized="cm/s",
                    ),
                )
            )

    def test_the_refusal_names_the_claim_that_failed(self) -> None:
        """The ``where=`` diagnostic is the ONLY thing this validator adds over
        the shared S13 helper, so it is the only part worth its own test -- and
        it earns one: a set holds many claims, and an error that says a bound
        disagreed without saying WHICH claim it belonged to is not actionable.
        A copy-pasted ``where=`` naming some other model would otherwise be
        invisible, since validation would still (correctly) raise."""
        with pytest.raises(ValidationError) as excinfo:
            _claim(
                claim_id="chamber_pressure",
                uncertainty=self._uncertainty(
                    UncertaintyBasis.ABSOLUTE,
                    _measured_value(
                        raw_text="0.05",
                        canonical_decimal_value="0.05",
                        quantity_kind=QuantityKind.VELOCITY,
                        unit_raw="cm/s",
                        unit_normalized="cm/s",
                    ),
                ),
            )

        assert "GroundedScalarClaim(claim_id='chamber_pressure')" in str(excinfo.value)

    def test_a_relative_bound_must_be_a_relative_uncertainty(self) -> None:
        with pytest.raises(ValidationError):
            _claim(
                uncertainty=self._uncertainty(
                    UncertaintyBasis.RELATIVE,
                    _measured_value(raw_text="5", canonical_decimal_value="5"),
                )
            )

    def test_a_relative_uncertainty_bound_is_accepted(self) -> None:
        claim = _claim(
            uncertainty=self._uncertainty(
                UncertaintyBasis.RELATIVE,
                _measured_value(
                    raw_text="5",
                    canonical_decimal_value="5",
                    quantity_kind=QuantityKind.RELATIVE_UNCERTAINTY,
                    unit_raw="%",
                    unit_normalized="%",
                ),
            )
        )

        assert not isinstance(claim.uncertainty, Absent)


class TestTheGenericWalkersReachAClaimWithNoChangesToThem:
    """The property that makes steps 2 and 3 cheap, pinned here rather than
    assumed: ``iter_source_refs`` and ``iter_measured_values`` are generic over
    payload shape, so every "does this cite something real" check and the
    conversion-table coverage check (T2) already cover a claim nested anywhere,
    with zero edits to either walker. If a future refactor makes either walker
    hand-list its fields, this test is what notices.
    """

    class _Holder(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        claims: tuple[GroundedScalarClaim, ...]

    def test_every_ref_a_claim_carries_is_reachable_by_path(self) -> None:
        holder = self._Holder(claims=(_claim(),))

        paths = {path for path, _ in iter_source_refs(holder)}

        assert paths == {
            "claims[0].label_ref",
            "claims[0].value.value_ref",
            "claims[0].value.unit_ref",
        }

    def test_the_claims_measured_value_is_reachable_by_path(self) -> None:
        holder = self._Holder(claims=(_claim(),))

        paths = {path for path, _ in iter_measured_values(holder)}

        assert paths == {"claims[0].value"}

    def test_an_uncertainty_bound_is_walked_too(self) -> None:
        holder = self._Holder(
            claims=(
                _claim(
                    uncertainty=Uncertainty(
                        kind=UncertaintyKind.STD_DEV,
                        basis=UncertaintyBasis.ABSOLUTE,
                        scale=UncertaintyScale.LINEAR,
                        upper=_measured_value(raw_text="0.05", canonical_decimal_value="0.05"),
                        lower=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
                    )
                ),
            )
        )

        assert {path for path, _ in iter_measured_values(holder)} == {
            "claims[0].value",
            "claims[0].uncertainty.upper",
        }


class TestWhatThisTypeDeliberatelyDoesNotProve:
    """Counterweight tests: these pass BY DESIGN, and say so out loud.

    A ``GroundedScalarClaim`` proves that a label and a number were each
    located in a source. It does NOT prove the label describes the number --
    that is a semantic relation between two spans, and no element model
    holding no document can check it. Recording these as tests keeps the type
    from being sold later as a safety property it was never able to be; the
    check that actually constrains label/value co-location is the prose-local
    scalar rule, which lives in the extraction gate, not here.
    """

    def test_a_label_that_describes_a_different_quantity_still_constructs(self) -> None:
        claim = _claim(
            label_raw="laminar burning velocity",
            value=_measured_value(),  # PRESSURE, atm
        )

        assert claim.label_raw == "laminar burning velocity"
        assert claim.value.quantity_kind is QuantityKind.PRESSURE

    def test_a_label_and_a_value_from_unrelated_places_still_construct(self) -> None:
        """Nothing here requires the label span to sit anywhere near the value
        span, or even in the same document -- ``label_ref`` names node ``n2``
        while the value's refs name ``n1``. Whether the two nodes exist at all
        is a SourceGraph question (V1/V2, envelope level); whether they are
        near each other is the prose-local rule's question. Neither is
        answerable from a lone element, and neither is faked here."""
        claim = _claim(label_ref=_char_span_ref(node_id="n2", start=9_000, end=9_020))

        assert claim.label_ref.node_id == "n2"
        assert claim.value.value_ref.node_id == "n1"
