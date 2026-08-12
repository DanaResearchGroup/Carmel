# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""``UnextractedConditionStatement``: a record that a condition statement WAS
located in a source, and was deliberately NOT turned into a claim.

Structural model for this file: ``tests/test_dataset_scalar_claim.py``. Kept
self-contained, exactly as that file is, by building its own minimal valid
``SourceRef`` helpers rather than importing fixtures from elsewhere.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    BBox,
    BBoxLocator,
    CharSpanLocator,
    CoordinateFrame,
    QuantityKind,
    SourceRef,
    TextSpace,
    UnextractedConditionStatement,
    UnextractedReason,
    iter_measured_values,
    iter_source_refs,
)


def _frame() -> CoordinateFrame:
    return CoordinateFrame(
        render_fingerprint="fp-1",
        cropbox=("0", "0", "612", "792"),
        mediabox=("0", "0", "612", "792"),
        rotation=0,
        dpi=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        render_settings=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
    )


def _bbox_ref(node_id: str = "n1") -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=BBoxLocator(bbox=BBox(frame=_frame(), x0="10", y0="20", x1="30", y1="40")),
    )


def _char_span_ref(node_id: str = "n1", start: int = 0, end: int = 20) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end),
    )


def _statement(**kwargs: object) -> UnextractedConditionStatement:
    defaults: dict[str, object] = {
        "statement_id": "phi_sweep",
        "label_raw": "equivalence ratio",
        "label_ref": _char_span_ref(),
        "statement_ref": _char_span_ref(start=20, end=40),
        "reason": UnextractedReason.VALUE_RANGE,
        "quantity_kind": QuantityKind.EQUIVALENCE_RATIO,
    }
    defaults.update(kwargs)
    return UnextractedConditionStatement(**defaults)  # type: ignore[arg-type]


class TestUnextractedConditionStatementHoldsALocatedRefusal:
    """A well-formed record keeps every field it was given."""

    def test_a_well_formed_record_keeps_every_field(self) -> None:
        statement = _statement()

        assert statement.statement_id == "phi_sweep"
        assert statement.label_raw == "equivalence ratio"
        assert statement.label_ref == _char_span_ref()
        assert statement.statement_ref == _char_span_ref(start=20, end=40)
        assert statement.reason is UnextractedReason.VALUE_RANGE
        assert statement.quantity_kind is QuantityKind.EQUIVALENCE_RATIO

    def test_the_record_is_frozen_and_forbids_extra_fields(self) -> None:
        statement = _statement()

        with pytest.raises(ValidationError):
            statement.statement_id = "other"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            _statement(source_form="textual")


class TestStatementIdIsARestrictedIdentifier:
    """Same rule, and the same reason, as ``claim_id``/``axis_id``/
    ``series_id``/``point_id``: a ``statement_id`` is interpolated into the
    dotted diagnostic paths :func:`iter_source_refs` builds, so ``.``,
    ``[``, ``]`` and non-ASCII would poison a path that is parsed
    positionally."""

    @pytest.mark.parametrize(
        "statement_id",
        [
            "Phi_Sweep",
            "1st_sweep",
            "phi.sweep",
            "phi[0]",
            "phi sweep",
            "phi_φ",
            "",
            "_leading",
        ],
    )
    def test_a_non_identifier_statement_id_is_refused(self, statement_id: str) -> None:
        with pytest.raises(ValidationError):
            _statement(statement_id=statement_id)

    @pytest.mark.parametrize("statement_id", ["p", "phi_sweep", "t5", "phi"])
    def test_a_plain_lowercase_identifier_is_accepted(self, statement_id: str) -> None:
        assert _statement(statement_id=statement_id).statement_id == statement_id


class TestLabelRawMustBeNonEmpty:
    def test_an_empty_label_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _statement(label_raw="")


class TestTheTwoRefsAreRequiredAndNeverAbsent:
    """``label_ref`` and ``statement_ref`` are each ``SourceRef``, never
    ``Maybe[SourceRef]``: an ungrounded label or an ungrounded statement is
    not a weaker record to keep honestly, it is no record at all --
    ``statement_ref`` in particular is what makes this type auditable, since
    a refusal that doesn't say WHERE is indistinguishable from a guess."""

    def test_a_missing_label_ref_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            UnextractedConditionStatement(  # type: ignore[call-arg]
                statement_id="phi_sweep",
                label_raw="equivalence ratio",
                statement_ref=_char_span_ref(start=20, end=40),
                reason=UnextractedReason.VALUE_RANGE,
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            )

        assert "label_ref" in str(excinfo.value)

    def test_a_missing_statement_ref_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            UnextractedConditionStatement(  # type: ignore[call-arg]
                statement_id="phi_sweep",
                label_raw="equivalence ratio",
                label_ref=_char_span_ref(),
                reason=UnextractedReason.VALUE_RANGE,
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            )

        assert "statement_ref" in str(excinfo.value)

    def test_an_absent_label_ref_is_not_even_a_representable_state(self) -> None:
        with pytest.raises(ValidationError):
            _statement(label_ref=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))

    def test_an_absent_statement_ref_is_not_even_a_representable_state(self) -> None:
        with pytest.raises(ValidationError):
            _statement(statement_ref=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))


class TestQuantityKindHonestlyAcceptsAbsence:
    """Unlike ``label_ref``/``statement_ref``, ``quantity_kind`` DOES accept
    ``Absent(...)``: a statement whose quantity could not even be determined
    must still be recordable, or the honest-coverage purpose this type exists
    for fails exactly where coverage is worst -- the qualitative-only
    statements ("atmospheric pressure") that name no clean quantity token at
    all."""

    def test_an_absent_quantity_kind_is_accepted(self) -> None:
        statement = _statement(
            reason=UnextractedReason.QUALITATIVE_ONLY,
            quantity_kind=Absent(reason=AbsenceReason.UNKNOWN),
        )

        assert isinstance(statement.quantity_kind, Absent)
        assert statement.quantity_kind.reason is AbsenceReason.UNKNOWN


class TestEveryUnextractedReasonIsConstructible:
    """A member added later to ``UnextractedReason`` without a matching
    construction test would fail nothing here silently -- this parametrizes
    over every member that exists TODAY so a future addition is visibly
    exercised the moment it is added to this list, not merely assumed
    covered because the enum itself validates."""

    @pytest.mark.parametrize("reason", list(UnextractedReason))
    def test_every_member_constructs_on_a_record(self, reason: UnextractedReason) -> None:
        statement = _statement(reason=reason)

        assert statement.reason is reason


class TestTheGenericWalkersReachAStatementWithNoChangesToThem:
    """Mirrors ``TestTheGenericWalkersReachAClaimWithNoChangesToThem`` in
    ``test_dataset_scalar_claim.py``: ``iter_source_refs`` and
    ``iter_measured_values`` are generic over payload shape, so both refs on
    this type are already covered by every "does this cite something real"
    check with zero edits to either walker."""

    class _Holder(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        statements: tuple[UnextractedConditionStatement, ...]

    def test_both_refs_are_reachable_by_path(self) -> None:
        holder = self._Holder(statements=(_statement(),))

        paths = {path for path, _ in iter_source_refs(holder)}

        assert paths == {
            "statements[0].label_ref",
            "statements[0].statement_ref",
        }

    def test_the_walker_yields_no_measured_value_for_this_type(self) -> None:
        """Pinned deliberately: this type carries no measured value at all --
        that is the entire point of it -- so ``iter_measured_values`` must
        yield nothing for it, not merely "nothing today by accident"."""
        holder = self._Holder(statements=(_statement(),))

        assert list(iter_measured_values(holder)) == []


class TestWhatThisTypeDeliberatelyDoesNotProve:
    """Counterweight tests: these pass BY DESIGN, and say so out loud.

    An ``UnextractedConditionStatement`` proves that a label and a statement
    were each located in a source, and records a classification decision the
    extractor already made. It does NOT prove that decision is correct --
    that would require re-reading the document, which no element model here
    holds. The real check for "is ``reason`` the right classification of
    this statement" lives upstream, in whatever process assigned the reason
    and the refs in the first place -- not in this model.
    """

    def test_a_reason_that_does_not_match_the_real_statement_still_constructs(self) -> None:
        """``reason=VALUE_RANGE`` is recorded even though nothing here checks
        that the statement at ``statement_ref`` is actually a range rather
        than, say, a single value -- checking that needs the document."""
        statement = _statement(reason=UnextractedReason.VALUE_RANGE)

        assert statement.reason is UnextractedReason.VALUE_RANGE

    def test_a_label_and_statement_from_unrelated_places_still_construct(self) -> None:
        """Nothing here requires the label span to sit anywhere near the
        statement span, or even name the same node -- ``label_ref`` names
        node ``n1`` while ``statement_ref`` names node ``n2``. Whether the
        two nodes exist at all is a ``SourceGraph`` question (envelope
        level); whether they are near each other is a document-level
        question. Neither is answerable from a lone element, and neither is
        faked here."""
        statement = _statement(statement_ref=_char_span_ref(node_id="n2", start=9_000, end=9_020))

        assert statement.label_ref.node_id == "n1"
        assert statement.statement_ref.node_id == "n2"
