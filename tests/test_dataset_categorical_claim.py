"""Tests for ``GroundedCategoricalClaim``: one condition a source states as a
NAME rather than a number ("the diluent is CO2", "the bath gas is argon").

Kept in its own module, mirroring ``test_dataset_scalar_claim.py``'s
rationale: this type is a sibling to ``GroundedScalarClaim``, not a series
element and not one of the M-D2a primitives, and it exists because named
conditions -- diluent identity, apparatus/method choices -- have no legal
home in ``Series`` or ``GroundedScalarClaim`` (which is numeric-only).

Fixtures here are self-contained by the same convention the other dataset
test modules follow: this file builds its own ``SourceRef``s rather than
importing another test module's private helpers.
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
    GroundedCategoricalClaim,
    SourceRef,
    TableCellLocator,
    TextSpace,
    iter_measured_values,
    iter_source_refs,
)


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


def _claim(**kwargs: object) -> GroundedCategoricalClaim:
    defaults: dict[str, object] = {
        "claim_id": "diluent",
        "label_raw": "diluent",
        "label_ref": _char_span_ref(),
        "token_raw": "CO2",
        "token_ref": _table_ref(),
    }
    defaults.update(kwargs)
    return GroundedCategoricalClaim(**defaults)  # type: ignore[arg-type]


class TestGroundedCategoricalClaimHoldsOneNamedCondition:
    def test_a_well_formed_claim_keeps_every_field_it_was_given(self) -> None:
        claim = _claim()

        assert claim.claim_id == "diluent"
        assert claim.label_raw == "diluent"
        assert claim.label_ref == _char_span_ref()
        assert claim.token_raw == "CO2"
        assert claim.token_ref == _table_ref()

    def test_the_claim_is_frozen_and_forbids_extra_fields(self) -> None:
        claim = _claim()

        with pytest.raises(ValidationError):
            claim.claim_id = "other"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            _claim(quantity_kind="pressure")


class TestClaimIdIsARestrictedIdentifier:
    """Same rule, and the same reason, as ``GroundedScalarClaim.claim_id``: a
    ``claim_id`` is interpolated into the dotted diagnostic paths
    :func:`iter_source_refs` builds, so ``.``, ``[``, ``]`` and non-ASCII
    would poison a path that is parsed positionally."""

    @pytest.mark.parametrize(
        "claim_id",
        [
            "CO2",
            "1st_diluent",
            "a.b",
            "a[0]",
            "has space",
            "phi_φ",
            "",
            "_leading",
        ],
    )
    def test_a_non_identifier_claim_id_is_refused(self, claim_id: str) -> None:
        with pytest.raises(ValidationError):
            _claim(claim_id=claim_id)

    @pytest.mark.parametrize("claim_id", ["diluent", "bath_gas", "t5", "phi"])
    def test_a_plain_lowercase_identifier_is_accepted(self, claim_id: str) -> None:
        assert _claim(claim_id=claim_id).claim_id == claim_id


class TestLabelAndTokenMustBeNonEmpty:
    def test_an_empty_label_raw_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _claim(label_raw="")

    def test_an_empty_token_raw_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _claim(token_raw="")


class TestALabelOrTokenWithoutAGroundedRefCannotBeConstructed:
    """The label and the token are the load-bearing fields, not decoration --
    each carries its own independent :class:`SourceRef`, exactly the split
    ``MeasuredValue`` makes between ``value_ref`` and ``unit_ref``, for the
    same reason: proving the string "CO2" sits at an offset can never prove
    it is the diluent rather than a product species or a cylinder label.
    """

    def test_a_missing_label_ref_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            GroundedCategoricalClaim(  # type: ignore[call-arg]
                claim_id="diluent",
                label_raw="diluent",
                token_raw="CO2",
                token_ref=_table_ref(),
            )

        assert "label_ref" in str(excinfo.value)

    def test_a_missing_token_ref_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            GroundedCategoricalClaim(  # type: ignore[call-arg]
                claim_id="diluent",
                label_raw="diluent",
                label_ref=_char_span_ref(),
                token_raw="CO2",
            )

        assert "token_ref" in str(excinfo.value)

    def test_an_absent_label_ref_is_not_even_a_representable_state(self) -> None:
        """``label_ref`` is ``SourceRef``, never ``Maybe[SourceRef]``: an
        ungrounded label is not a weaker claim to be recorded honestly, it is
        no claim at all."""
        with pytest.raises(ValidationError):
            _claim(label_ref=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))

    def test_an_absent_token_ref_is_not_even_a_representable_state(self) -> None:
        """Same rule as ``label_ref`` above, applied to ``token_ref``."""
        with pytest.raises(ValidationError):
            _claim(token_ref=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))


class TestTheGenericWalkersReachAClaimWithNoChangesToThem:
    """The property that makes steps 2 and 3 cheap, pinned here rather than
    assumed: ``iter_source_refs`` and ``iter_measured_values`` are generic
    over payload shape, so every "does this cite something real" check
    already covers a categorical claim nested anywhere, with zero edits to
    either walker. If a future refactor makes either walker hand-list its
    fields, this test is what notices.
    """

    class _Holder(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        claims: tuple[GroundedCategoricalClaim, ...]

    def test_every_ref_a_claim_carries_is_reachable_by_path(self) -> None:
        holder = self._Holder(claims=(_claim(),))

        paths = {path for path, _ in iter_source_refs(holder)}

        assert paths == {
            "claims[0].label_ref",
            "claims[0].token_ref",
        }

    def test_a_categorical_claim_yields_no_measured_value(self) -> None:
        """A categorical claim carries no measured value at all -- pinned
        here rather than left implicit, since a future field addition (e.g.
        a numeric fraction alongside the token) could silently change this."""
        holder = self._Holder(claims=(_claim(),))

        assert list(iter_measured_values(holder)) == []


class TestWhatThisTypeDeliberatelyDoesNotProve:
    """Counterweight tests: these pass BY DESIGN, and say so out loud.

    A ``GroundedCategoricalClaim`` proves that a label and a token were each
    located in a source. It does NOT prove the token names a real species or
    apparatus, that the label describes the token, or that either belongs to
    THIS experiment rather than one the source merely cites. Those are
    semantic relations between spans (or between a span and the world), and
    no element model -- holding no document -- can check them. The real
    check lives in the extraction gate, which has the document; recording
    these as tests keeps this type from being sold later as a safety
    property it was never able to be.
    """

    def test_a_nonsense_token_still_constructs(self) -> None:
        claim = _claim(token_raw="xyzzy-not-a-real-species")

        assert claim.token_raw == "xyzzy-not-a-real-species"

    def test_a_label_and_a_token_from_unrelated_places_still_construct(self) -> None:
        """Nothing here requires the label span to sit anywhere near the
        token span, or even in the same document -- ``label_ref`` names node
        ``n2`` while ``token_ref`` names node ``n1``. Whether the two nodes
        exist at all is a ``SourceGraph`` question (envelope level); whether
        they are near each other is the prose-local rule's question. Neither
        is answerable from a lone element, and neither is faked here."""
        claim = _claim(label_ref=_char_span_ref(node_id="n2", start=9_000, end=9_020))

        assert claim.label_ref.node_id == "n2"
        assert claim.token_ref.node_id == "n1"
