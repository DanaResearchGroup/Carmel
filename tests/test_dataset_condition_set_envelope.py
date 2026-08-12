"""Behavioural tests for ``ConditionSetEnvelope`` -- the container that gives
the three condition atoms (``GroundedScalarClaim``,
``GroundedCategoricalClaim``, ``UnextractedConditionStatement``) a home, a
SUBJECT, and provenance validation.

Kept in its own module, sibling to ``test_dataset_scalar_claim.py`` /
``test_dataset_categorical_claim.py`` / ``test_dataset_unextracted_statement.py``:
those cover the atoms in isolation; this file covers the container's own
invariants (C1--C4), the wiring of the seven shared provenance validators
through this class, and the subject sum. Identity-projection meta-tests live
separately in ``test_dataset_condition_set_identity.py``.

TDD NOTE: written before ``ConditionSetEnvelope`` existed; ImportErrors are
the expected first RED, not a defect here.

Fixtures are self-contained (the atom-module convention) except for the
second-conversion-table helpers, imported from
``tests.test_dataset_graph_and_envelope`` exactly as that module built them:
a second, genuinely-cited table is the only way to exercise T3's sort order
without tripping T2, and rebuilding that machinery here would drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    ConditionAttribution,
    ConditionSetEnvelope,
    CoordinateFrame,
    DeviceClassDeclaration,
    EmbeddedConversionTable,
    GroundedCategoricalClaim,
    GroundedScalarClaim,
    MeasuredValue,
    QuantityKind,
    SemanticDependencyUse,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    TextSpace,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
    XPathLocator,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import TABLE_V1
from tests.test_dataset_graph_and_envelope import (
    _embedded_second_table,
    _registered_second_table,
    _second_conversion_table,
)

SHA_A = "a" * 64
SHA_B = "b" * 64

_CURRENT_REPAIR_DEPENDENCY = SemanticDependencyUse(
    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
)

_NO_EXTRACTION = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_GLYPH_HEALTH = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_UNCERTAINTY = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
_NO_QUANTITY_KIND = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)


def _node(node_id: str = "paper", parent_node_id: str | None = None) -> SourceNode:
    is_root = parent_node_id is None
    return SourceNode(
        node_id=node_id,
        kind=SourceNodeKind.PAPER_PDF if is_root else SourceNodeKind.SI_MEMBER,
        sha256=SHA_A if is_root else SHA_B,
        parent_node_id=parent_node_id,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE)
        if is_root
        else ArchiveOrigin(archive_sha256=SHA_B, member_display_path=None),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
    )


def _graph(*nodes: SourceNode) -> SourceGraph:
    return SourceGraph(nodes=nodes or (_node(),))


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


def _bbox_ref(node_id: str = "paper") -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=BBoxLocator(bbox=BBox(frame=_frame(), x0="10", y0="20", x1="30", y1="40")),
    )


def _table_ref(node_id: str = "paper", row: int = 0, col: int = 1) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=row, col=col),
    )


def _char_span_ref(node_id: str = "paper", start: int = 0, end: int = 20) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end),
    )


def _xpath_ref(node_id: str = "paper") -> SourceRef:
    return SourceRef(node_id=node_id, locator=XPathLocator(xpath="//table/row[1]/cell[1]"))


def _measured_value(
    raw_text: str = "1.0",
    quantity_kind: QuantityKind = QuantityKind.PRESSURE,
    unit_raw: str = "atm",
    unit_normalized: str = "atm",
    value_ref: SourceRef | None = None,
    unit_ref: SourceRef | None = None,
    conversion_table_sha256: str | None = None,
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=conversion_table_sha256 or TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref or _table_ref(),
        unit_ref=unit_ref or _table_ref(col=2),
    )


def _scalar_claim(**kwargs: object) -> GroundedScalarClaim:
    defaults: dict[str, object] = {
        "claim_id": "initial_pressure",
        "label_raw": "initial pressure",
        "label_ref": _table_ref(),
        "value": _measured_value(),
        "uncertainty": _NO_UNCERTAINTY,
    }
    defaults.update(kwargs)
    return GroundedScalarClaim(**defaults)  # type: ignore[arg-type]


def _categorical_claim(**kwargs: object) -> GroundedCategoricalClaim:
    defaults: dict[str, object] = {
        "claim_id": "diluent",
        "label_raw": "diluent",
        "label_ref": _table_ref(),
        "token_raw": "CO2",
        "token_ref": _table_ref(col=3),
    }
    defaults.update(kwargs)
    return GroundedCategoricalClaim(**defaults)  # type: ignore[arg-type]


def _refusal(**kwargs: object) -> UnextractedConditionStatement:
    defaults: dict[str, object] = {
        "statement_id": "phi_range",
        "label_raw": "equivalence ratio",
        "label_ref": _table_ref(),
        "statement_ref": _bbox_ref(),
        "reason": UnextractedReason.VALUE_RANGE,
        "quantity_kind": _NO_QUANTITY_KIND,
    }
    defaults.update(kwargs)
    return UnextractedConditionStatement(**defaults)  # type: ignore[arg-type]


def _subject(**kwargs: object) -> DeviceClassDeclaration:
    defaults: dict[str, object] = {"label_raw": "heat flux burner", "label_ref": _table_ref()}
    defaults.update(kwargs)
    return DeviceClassDeclaration(**defaults)  # type: ignore[arg-type]


def _unresolved(**kwargs: object) -> UnresolvedSubject:
    defaults: dict[str, object] = {
        "reason": SubjectRefusalReason.MULTIPLE_INDISTINGUISHABLE_DEVICES,
        "reason_ref": _bbox_ref(),
    }
    defaults.update(kwargs)
    return UnresolvedSubject(**defaults)  # type: ignore[arg-type]


def _embedded_table_v1() -> EmbeddedConversionTable:
    return EmbeddedConversionTable(
        sha256=TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
    )


def _envelope(**kwargs: object) -> ConditionSetEnvelope:
    defaults: dict[str, object] = {
        "source_graph": _graph(),
        "conversion_tables": (_embedded_table_v1(),),
        "subject": _subject(),
        "attribution": ConditionAttribution.OWN_EXPERIMENT,
        "attribution_ref": _table_ref(),
        "scalar_claims": (_scalar_claim(),),
        "categorical_claims": (_categorical_claim(),),
        "unextracted": (_refusal(),),
    }
    defaults.update(kwargs)
    return ConditionSetEnvelope(**defaults)  # type: ignore[arg-type]


def _refusals_only_envelope(**kwargs: object) -> ConditionSetEnvelope:
    defaults: dict[str, object] = {
        "scalar_claims": (),
        "categorical_claims": (),
        "conversion_tables": (),
    }
    defaults.update(kwargs)
    return _envelope(**defaults)


class TestConditionSetEnvelopeHoldsGroundedConditionsForOneSubject:
    def test_a_well_formed_envelope_keeps_every_field_it_was_given(self) -> None:
        env = _envelope()

        assert isinstance(env.subject, DeviceClassDeclaration)
        assert env.subject.label_raw == "heat flux burner"
        assert env.attribution is ConditionAttribution.OWN_EXPERIMENT
        assert env.scalar_claims[0].claim_id == "initial_pressure"
        assert env.categorical_claims[0].claim_id == "diluent"
        assert env.unextracted[0].statement_id == "phi_range"

    def test_the_envelope_is_frozen_and_forbids_extra_fields(self) -> None:
        env = _envelope()
        with pytest.raises(ValidationError):
            env.attribution = ConditionAttribution.SIMULATION  # type: ignore[misc]
        with pytest.raises(ValidationError):
            _envelope(scope_id="vessel-1")

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(
                lambda: DeviceClassDeclaration(label_raw="bomb", label_ref=_table_ref(), extra="x"),
                id="device-class",
            ),
            pytest.param(
                lambda: UnresolvedSubject(
                    reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=_table_ref(), extra="x"
                ),
                id="unresolved-subject",
            ),
        ],
    )
    def test_the_subject_models_forbid_extra_fields(self, build: object) -> None:
        with pytest.raises(ValidationError):
            build()  # type: ignore[operator]

    def test_the_subject_models_are_frozen(self) -> None:
        subject = _subject()
        with pytest.raises(ValidationError):
            subject.label_raw = "bomb"  # type: ignore[misc]
        unresolved = _unresolved()
        with pytest.raises(ValidationError):
            unresolved.reason = SubjectRefusalReason.DEVICE_UNNAMED  # type: ignore[misc]

    def test_a_device_class_label_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            _subject(label_raw="")


class TestSubjectIsARequiredSum:
    """The subject slot must hold exactly one of the two variants -- a
    grounded device-CLASS declaration or a grounded refusal to resolve the
    subject. There is no third state: no None, no bare string, no envelope
    without a subject at all. A condition set with no subject is how a bomb
    and a shock tube from one paper silently merge into one record."""

    def test_a_device_class_declaration_is_accepted(self) -> None:
        env = _envelope(subject=_subject())
        assert isinstance(env.subject, DeviceClassDeclaration)

    def test_an_unresolved_subject_is_accepted(self) -> None:
        env = _envelope(subject=_unresolved())
        assert isinstance(env.subject, UnresolvedSubject)
        assert env.subject.reason is SubjectRefusalReason.MULTIPLE_INDISTINGUISHABLE_DEVICES

    def test_every_subject_refusal_reason_is_constructible(self) -> None:
        for reason in SubjectRefusalReason:
            env = _envelope(subject=_unresolved(reason=reason))
            assert isinstance(env.subject, UnresolvedSubject)

    def test_an_envelope_without_a_subject_is_refused(self) -> None:
        kwargs = {
            "source_graph": _graph(),
            "conversion_tables": (_embedded_table_v1(),),
            "attribution": ConditionAttribution.OWN_EXPERIMENT,
            "attribution_ref": _table_ref(),
            "scalar_claims": (_scalar_claim(),),
            "categorical_claims": (),
            "unextracted": (),
        }
        with pytest.raises(ValidationError, match="subject"):
            ConditionSetEnvelope(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_subject", [None, "spherical vessel"], ids=["none", "bare-string"])
    def test_a_subject_that_is_not_one_of_the_two_variants_is_refused(self, bad_subject: object) -> None:
        with pytest.raises(ValidationError):
            _envelope(subject=bad_subject)

    def test_the_two_subject_projections_are_tag_distinct(self) -> None:
        """Tagged-projection distinctness: the two variants can never produce
        the same identity payload, because each carries its own tag value."""
        device_payload = _envelope(subject=_subject()).identity_payload()["subject"]
        unresolved_payload = _envelope(subject=_unresolved()).identity_payload()["subject"]

        assert device_payload["subject_kind"] == "device_class"
        assert unresolved_payload["subject_kind"] == "unresolved"
        assert device_payload != unresolved_payload


class TestAtLeastOneRecordAcrossTheThreeCollections:
    """C1: an envelope with no scalar claim, no categorical claim, and no
    refusal is an audit-shaped artifact -- a validated source graph and a
    grounded subject wrapped around nothing -- and must be refused. Refusals
    COUNT as records: coverage honesty is the whole point of the
    unextracted collection."""

    def test_an_all_empty_envelope_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="ConditionSetEnvelope"):
            _envelope(scalar_claims=(), categorical_claims=(), unextracted=(), conversion_tables=())

    def test_an_envelope_holding_only_refusals_is_accepted_by_design(self) -> None:
        """DELIBERATE: a paper stating only "pressures from 1 to 10 atm"
        yields zero claims and one VALUE_RANGE refusal, and that envelope is
        exactly the coverage honesty this design exists for -- it must stay
        legal, and this test's name is meant to be read as that decision."""
        env = _refusals_only_envelope()
        assert env.scalar_claims == ()
        assert env.categorical_claims == ()
        assert len(env.unextracted) == 1

    def test_an_envelope_holding_only_claims_is_accepted(self) -> None:
        env = _envelope(unextracted=())
        assert env.unextracted == ()

    def test_an_envelope_holding_only_a_categorical_claim_is_accepted(self) -> None:
        env = _envelope(scalar_claims=(), unextracted=(), conversion_tables=())
        assert len(env.categorical_claims) == 1


class TestOneIdNamespaceAcrossAllThreeCollections:
    """C2: no id may repeat anywhere in the envelope -- across
    ``scalar_claims``, ``categorical_claims`` and ``unextracted`` JOINTLY,
    not merely within each. The concrete failure this prevents: a coverage
    map keyed by logical condition id where scalar_claims["pressure"] and
    unextracted["pressure"] coexist and one silently overwrites the other,
    turning "refused a range" into "extracted a scalar"."""

    def test_a_duplicate_id_within_one_collection_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="initial_pressure"):
            _envelope(scalar_claims=(_scalar_claim(), _scalar_claim()))

    def test_a_duplicate_id_across_scalar_and_unextracted_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="initial_pressure"):
            _envelope(unextracted=(_refusal(statement_id="initial_pressure"),))

    def test_a_duplicate_id_across_scalar_and_categorical_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="diluent"):
            _envelope(scalar_claims=(_scalar_claim(claim_id="diluent"),))

    def test_a_duplicate_id_across_categorical_and_unextracted_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="diluent"):
            _envelope(unextracted=(_refusal(statement_id="diluent"),))

    def test_distinct_ids_across_all_three_collections_are_accepted(self) -> None:
        env = _envelope()
        ids = (
            [c.claim_id for c in env.scalar_claims]
            + [c.claim_id for c in env.categorical_claims]
            + [s.statement_id for s in env.unextracted]
        )
        assert len(ids) == len(set(ids))


class TestEachCollectionIsSortedByItsOwnId:
    """C3: one canonical order per collection, so exactly one byte
    representation exists for one logical condition set -- the same
    address-uniqueness idiom as series/table sorting on DatasetEnvelope."""

    def test_unsorted_scalar_claims_are_refused(self) -> None:
        first = _scalar_claim(claim_id="pressure_b")
        second = _scalar_claim(claim_id="pressure_a")
        with pytest.raises(ValidationError, match="scalar_claims"):
            _envelope(scalar_claims=(first, second))

    def test_unsorted_categorical_claims_are_refused(self) -> None:
        first = _categorical_claim(claim_id="diluent_b")
        second = _categorical_claim(claim_id="diluent_a")
        with pytest.raises(ValidationError, match="categorical_claims"):
            _envelope(categorical_claims=(first, second))

    def test_unsorted_refusals_are_refused(self) -> None:
        first = _refusal(statement_id="range_b")
        second = _refusal(statement_id="range_a")
        with pytest.raises(ValidationError, match="unextracted"):
            _envelope(unextracted=(first, second))

    def test_sorted_collections_are_accepted(self) -> None:
        env = _envelope(
            scalar_claims=(_scalar_claim(claim_id="pressure_a"), _scalar_claim(claim_id="pressure_b")),
        )
        assert [c.claim_id for c in env.scalar_claims] == ["pressure_a", "pressure_b"]


class TestTheWholeEnvelopeIsGroundedUnderOneRootArtifact:
    """C4: every SourceRef in the WHOLE envelope must resolve under one
    parentless root artifact. Deliberately stronger than DatasetEnvelope's
    per-series V5: a condition set is one subject's conditions from one
    source, so a label span in paper A and a value span in paper B is two
    papers stitched into one record, not one record."""

    def test_refs_under_two_root_artifacts_are_refused(self) -> None:
        paper_a = _node("paper_a")
        paper_b = SourceNode(
            node_id="paper_b",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=SHA_B,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=_NO_EXTRACTION,
            glyph_health=_NO_GLYPH_HEALTH,
            verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        claim = _scalar_claim(
            label_ref=_table_ref("paper_a"),
            value=_measured_value(value_ref=_table_ref("paper_a"), unit_ref=_table_ref("paper_b")),
        )
        with pytest.raises(ValidationError, match="root artifact"):
            _envelope(
                source_graph=_graph(paper_a, paper_b),
                subject=_subject(label_ref=_table_ref("paper_a")),
                attribution_ref=_table_ref("paper_a"),
                scalar_claims=(claim,),
                categorical_claims=(
                    _categorical_claim(label_ref=_table_ref("paper_a"), token_ref=_table_ref("paper_a", col=3)),
                ),
                unextracted=(_refusal(label_ref=_table_ref("paper_a"), statement_ref=_bbox_ref("paper_a")),),
            )

    def test_a_refusal_reaching_into_a_second_paper_is_refused(self) -> None:
        """The cross-root ref lives ONLY on an unextracted statement here --
        every claim stays in one paper.

        This is not a redundant angle on the test above, it is the likeliest
        one in practice. Refusals are precisely where another lab's restated
        conditions land: a sentence like "Smith et al. measured this in a
        shock tube at 1 atm" is a located statement that cannot become a
        claim, so it goes to `unextracted` -- and if its span is allowed to
        point into the OTHER paper, the set has quietly annexed a second
        source while every claim it holds still looks clean. A C4 that walks
        only the claim collections passes this envelope.
        """
        paper_a = _node("paper_a")
        paper_b = SourceNode(
            node_id="paper_b",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=SHA_B,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=_NO_EXTRACTION,
            glyph_health=_NO_GLYPH_HEALTH,
            verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        with pytest.raises(ValidationError, match="root artifact"):
            _envelope(
                source_graph=_graph(paper_a, paper_b),
                subject=_subject(label_ref=_table_ref("paper_a")),
                attribution_ref=_table_ref("paper_a"),
                scalar_claims=(
                    _scalar_claim(
                        label_ref=_table_ref("paper_a"),
                        value=_measured_value(
                            value_ref=_table_ref("paper_a"),
                            unit_ref=_table_ref("paper_a", col=2),
                        ),
                    ),
                ),
                categorical_claims=(
                    _categorical_claim(label_ref=_table_ref("paper_a"), token_ref=_table_ref("paper_a", col=3)),
                ),
                unextracted=(_refusal(label_ref=_table_ref("paper_a"), statement_ref=_bbox_ref("paper_b")),),
            )

    def test_a_subject_label_reaching_into_a_second_paper_is_refused(self) -> None:
        """The cross-root ref lives ONLY on the subject's ``label_ref``
        here -- every claim, the refusal, and ``attribution_ref`` all stay
        in one paper.

        A C4 that walks claims and refusals but skips the subject would
        pass this envelope: the subject is not a claim collection, so a
        walk keyed off ``scalar_claims``/``categorical_claims``/
        ``unextracted`` alone never visits it. That would let a device-class
        label read in one paper ground conditions actually extracted from a
        different paper -- the subject looking grounded while addressing
        the wrong source entirely.
        """
        paper_a = _node("paper_a")
        paper_b = SourceNode(
            node_id="paper_b",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=SHA_B,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=_NO_EXTRACTION,
            glyph_health=_NO_GLYPH_HEALTH,
            verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        with pytest.raises(ValidationError, match="root artifact"):
            _envelope(
                source_graph=_graph(paper_a, paper_b),
                subject=_subject(label_ref=_table_ref("paper_b")),
                attribution_ref=_table_ref("paper_a"),
                scalar_claims=(
                    _scalar_claim(
                        label_ref=_table_ref("paper_a"),
                        value=_measured_value(
                            value_ref=_table_ref("paper_a"),
                            unit_ref=_table_ref("paper_a", col=2),
                        ),
                    ),
                ),
                categorical_claims=(
                    _categorical_claim(label_ref=_table_ref("paper_a"), token_ref=_table_ref("paper_a", col=3)),
                ),
                unextracted=(_refusal(label_ref=_table_ref("paper_a"), statement_ref=_bbox_ref("paper_a")),),
            )

    def test_an_attribution_ref_reaching_into_a_second_paper_is_refused(self) -> None:
        """The cross-root ref lives ONLY on ``attribution_ref`` here -- the
        subject, every claim, and the refusal all stay in one paper.

        ``attribution_ref`` is a top-level scalar field on the envelope,
        not a member of any of the three claim collections, so a C4 that
        walks only ``scalar_claims``/``categorical_claims``/``unextracted``
        (and even the subject) would still pass this envelope -- letting an
        attribution assertion ("own_experiment") cite evidence from a paper
        that is not even the one the conditions were extracted from.
        """
        paper_a = _node("paper_a")
        paper_b = SourceNode(
            node_id="paper_b",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=SHA_B,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=_NO_EXTRACTION,
            glyph_health=_NO_GLYPH_HEALTH,
            verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        with pytest.raises(ValidationError, match="root artifact"):
            _envelope(
                source_graph=_graph(paper_a, paper_b),
                subject=_subject(label_ref=_table_ref("paper_a")),
                attribution_ref=_table_ref("paper_b"),
                scalar_claims=(
                    _scalar_claim(
                        label_ref=_table_ref("paper_a"),
                        value=_measured_value(
                            value_ref=_table_ref("paper_a"),
                            unit_ref=_table_ref("paper_a", col=2),
                        ),
                    ),
                ),
                categorical_claims=(
                    _categorical_claim(label_ref=_table_ref("paper_a"), token_ref=_table_ref("paper_a", col=3)),
                ),
                unextracted=(_refusal(label_ref=_table_ref("paper_a"), statement_ref=_bbox_ref("paper_a")),),
            )

    def test_refs_to_two_nodes_under_one_root_are_accepted(self) -> None:
        paper = _node("paper")
        si = _node("si", parent_node_id="paper")
        env = _envelope(
            source_graph=_graph(paper, si),
            scalar_claims=(
                _scalar_claim(
                    label_ref=_table_ref("paper"),
                    value=_measured_value(value_ref=_table_ref("si"), unit_ref=_table_ref("si", col=2)),
                ),
            ),
        )
        roots = {ref.node_id for ref in (env.scalar_claims[0].label_ref, env.scalar_claims[0].value.value_ref)}
        assert roots == {"paper", "si"}


class TestSharedProvenanceValidatorsAreWiredThroughThisClass:
    """Each of the seven shared provenance helpers must actually RUN when a
    ConditionSetEnvelope validates. The helpers' own behaviour is already
    covered against DatasetEnvelope; the defect these tests close is a
    forgotten wrapper -- an envelope class that simply never calls one of
    them and silently accepts what DatasetEnvelope would refuse."""

    def test_t2_a_cited_but_unembedded_table_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is missing table\\(s\\)"):
            _envelope(conversion_tables=())

    def test_t2_a_decorative_embedded_table_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="embeds decorative table\\(s\\)"):
            _refusals_only_envelope(conversion_tables=(_embedded_table_v1(),))

    def test_a_duplicate_embedded_table_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="embeds duplicate sha256"):
            _envelope(conversion_tables=(_embedded_table_v1(), _embedded_table_v1()))

    def test_t3_conversion_tables_out_of_sha256_order_are_refused(self) -> None:
        with _registered_second_table():
            second = _second_conversion_table()
            second_claim = _scalar_claim(
                claim_id="dilution_fraction",
                label_raw="dilution fraction",
                value=_measured_value(
                    raw_text="0.04",
                    quantity_kind=QuantityKind.MOLE_FRACTION,
                    unit_raw="1",
                    unit_normalized="1",
                    conversion_table_sha256=second.sha256,
                ),
            )
            claims = tuple(sorted((_scalar_claim(), second_claim), key=lambda c: c.claim_id))
            ascending = tuple(sorted((_embedded_table_v1(), _embedded_second_table()), key=lambda t: t.sha256))
            descending = tuple(reversed(ascending))
            assert descending != ascending, "test setup bug: the two tables must sort differently"
            with pytest.raises(ValidationError, match="must be sorted ascending by sha256"):
                _envelope(scalar_claims=claims, conversion_tables=descending)
            env = _envelope(scalar_claims=claims, conversion_tables=ascending)
            assert env.conversion_tables == ascending

    def test_v1_a_ref_naming_an_unknown_node_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not present in"):
            _envelope(attribution_ref=_table_ref("ghost"))

    def test_v2_a_decorative_node_is_refused(self) -> None:
        orphan = SourceNode(
            node_id="orphan",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=SHA_B,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=_NO_EXTRACTION,
            glyph_health=_NO_GLYPH_HEALTH,
            verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        with pytest.raises(ValidationError, match="decorative provenance"):
            _envelope(source_graph=_graph(_node(), orphan))

    def test_v3_an_incompatible_locator_kind_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="may only target nodes of kind"):
            _envelope(attribution_ref=_xpath_ref("paper"))

    def test_v6_a_char_span_into_an_unextracted_node_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="never extracted"):
            _envelope(attribution_ref=_char_span_ref("paper"))


class TestWhatThisEnvelopeDeliberatelyDoesNotProve:
    """Counterweight tests: these pass BY DESIGN, and say so out loud.

    A ``ConditionSetEnvelope`` proves that a subject declaration (or a
    refusal to make one), an attribution assertion, and a set of condition
    records were each LOCATED in one source, under one root artifact, with
    one id namespace. It does NOT prove any semantic relation between them:
    not that the claims describe the declared device class, not that the
    attribution is true, not that the conditions belong to this paper's own
    experiment rather than a cited one. Those are document-level questions
    no element model holding no document can answer, and recording them as
    passing tests keeps this type from later being sold as a safety
    property it never was."""

    def test_claims_need_not_describe_the_declared_device_class(self) -> None:
        """A shock-tube bore claim under a 'heat flux burner' subject still
        constructs: nothing here can check that relation, and pretending to
        would be fabricated authority."""
        env = _envelope(
            subject=_subject(label_raw="heat flux burner"),
            scalar_claims=(_scalar_claim(claim_id="bore", label_raw="shock-tube bore"),),
        )
        assert env.subject.label_raw == "heat flux burner"  # type: ignore[union-attr]
        assert env.scalar_claims[0].label_raw == "shock-tube bore"

    def test_attribution_is_an_assertion_its_ref_is_evidence_not_a_gate(self) -> None:
        """The same located span constructs under OWN_EXPERIMENT and under
        SIMULATION alike: ``attribution_ref`` is evidence FOR the assertion,
        never a check OF it -- a real ref can be paired with a false
        OWN_EXPERIMENT, and only a reader of the document could tell. The
        probe that motivated ATTRIBUTION_UNCLEAR found CHEMKIN-run
        conditions worded exactly like experimental ones."""
        ref = _table_ref()
        own = _envelope(attribution=ConditionAttribution.OWN_EXPERIMENT, attribution_ref=ref)
        sim = _envelope(attribution=ConditionAttribution.SIMULATION, attribution_ref=ref)
        assert own.attribution_ref == sim.attribution_ref

    def test_conditions_may_in_truth_belong_to_a_cited_experiment(self) -> None:
        env = _envelope(attribution=ConditionAttribution.CITED_THIRD_PARTY)
        assert env.attribution is ConditionAttribution.CITED_THIRD_PARTY

    def test_a_device_class_never_identifies_one_physical_apparatus(self) -> None:
        """Two envelopes may declare the identical device class while holding
        different conditions from different vessels of that class -- the
        declaration is a CLASS, and the type refuses to pretend otherwise
        (that is why there is no scope_id). In a survey of eight
        combustion-kinetics papers, one used two physically different
        vessels its text never named apart; an extractor-invented vessel id
        would be a fabricated identity wearing an identifier."""
        first = _envelope(scalar_claims=(_scalar_claim(value=_measured_value("1.0")),))
        second = _envelope(scalar_claims=(_scalar_claim(value=_measured_value("10.0")),))
        assert first.subject == second.subject
        assert canonical_json_bytes(first.identity_payload()) != canonical_json_bytes(second.identity_payload())
