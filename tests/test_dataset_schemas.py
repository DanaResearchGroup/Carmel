"""Tests for carmel.schemas.datasets: the M-D2a schema primitives (absence
states, coordinate frames, the source graph, measured values, uncertainty,
and composition) for literature-extracted experimental kinetics datasets."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveMemberLocator,
    BBox,
    BBoxLocator,
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    CoordinateFrame,
    Maybe,
    MeasuredValue,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    XPathLocator,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _frame(**kwargs: object) -> CoordinateFrame:
    defaults: dict[str, object] = {
        "render_fingerprint": "fp-1",
        "cropbox": (0.0, 0.0, 612.0, 792.0),
        "mediabox": (0.0, 0.0, 612.0, 792.0),
        "rotation": 0,
        "units": "pt",
    }
    defaults.update(kwargs)
    return CoordinateFrame(**defaults)  # type: ignore[arg-type]


def _bbox(**kwargs: object) -> BBox:
    defaults: dict[str, object] = {"frame": _frame(), "x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0}
    defaults.update(kwargs)
    return BBox(**defaults)  # type: ignore[arg-type]


def _node(node_id: str = "n1", kind: SourceNodeKind = SourceNodeKind.PAPER_PDF, sha256: str = SHA_A) -> SourceNode:
    return SourceNode(node_id=node_id, kind=kind, sha256=sha256)


def _bbox_ref(node_id: str = "n1") -> SourceRef:
    return SourceRef(node_id=node_id, locator=BBoxLocator(bbox=_bbox()))


def _table_ref(node_id: str = "n1", row: int = 0, col: int = 1) -> SourceRef:
    return SourceRef(node_id=node_id, locator=TableCellLocator(row=row, col=col))


def _measured_value(
    raw_text: str = "1.20",
    canonical_decimal_value: str = "1.20",
    unit_raw: str = "cm/s",
    unit_canonical: str = "m/s",
    conversion_factor: str = "0.01",
    conversion_table_version: str = "v1",
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=canonical_decimal_value,
        unit_raw=unit_raw,
        unit_canonical=unit_canonical,
        conversion_factor=conversion_factor,
        conversion_table_version=conversion_table_version,
        value_ref=_bbox_ref(),
        unit_ref=_table_ref(),
    )


class TestAbsenceStates:
    @pytest.mark.parametrize(
        "reason",
        [
            AbsenceReason.NOT_APPLICABLE,
            AbsenceReason.NOT_REPORTED_HERE,
            AbsenceReason.NOT_EXTRACTED_YET,
            AbsenceReason.CONFLICTING_SOURCES,
            AbsenceReason.UNKNOWN,
            AbsenceReason.SAME_AS_DATASET,
        ],
    )
    def test_each_reason_is_representable(self, reason: AbsenceReason) -> None:
        absent = Absent(reason=reason)
        assert absent.reason == reason

    def test_all_six_reasons_are_distinct(self) -> None:
        reasons = {
            AbsenceReason.NOT_APPLICABLE,
            AbsenceReason.NOT_REPORTED_HERE,
            AbsenceReason.NOT_EXTRACTED_YET,
            AbsenceReason.CONFLICTING_SOURCES,
            AbsenceReason.UNKNOWN,
            AbsenceReason.SAME_AS_DATASET,
        }
        assert len(reasons) == 6

    def test_not_reported_here_and_not_extracted_yet_are_not_collapsible(self) -> None:
        """These encode different facts (property of the paper vs. our own
        extraction gap) and must never compare equal or be interchangeable."""
        assert AbsenceReason.NOT_REPORTED_HERE != AbsenceReason.NOT_EXTRACTED_YET
        assert Absent(reason=AbsenceReason.NOT_REPORTED_HERE) != Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)

    def test_absent_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Absent(reason=AbsenceReason.UNKNOWN, surprise="y")  # type: ignore[call-arg]

    def test_none_cannot_substitute_for_explicit_absence(self) -> None:
        """A field typed Maybe[T] must reject a plain None outright -- None
        carries no reason and must never be usable as a stand-in for an
        explicit Absent."""

        class _Holder(BaseModel):
            model_config = ConfigDict(extra="forbid")
            field: Maybe[str]

        with pytest.raises(ValidationError):
            _Holder(field=None)  # type: ignore[arg-type]

    def test_maybe_field_accepts_present_value(self) -> None:
        class _Holder(BaseModel):
            model_config = ConfigDict(extra="forbid")
            field: Maybe[str]

        holder = _Holder(field="present")
        assert holder.field == "present"

    def test_maybe_field_accepts_explicit_absent(self) -> None:
        class _Holder(BaseModel):
            model_config = ConfigDict(extra="forbid")
            field: Maybe[str]

        holder = _Holder(field=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET))
        assert isinstance(holder.field, Absent)
        assert holder.field.reason == AbsenceReason.NOT_EXTRACTED_YET


class TestCoordinateFrameAndBBox:
    def test_bbox_requires_frame(self) -> None:
        with pytest.raises(ValidationError):
            BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)  # type: ignore[call-arg]

    def test_bbox_with_frame_is_constructible(self) -> None:
        bbox = _bbox()
        assert bbox.frame.render_fingerprint == "fp-1"

    def test_frame_rejects_non_multiple_of_90_rotation(self) -> None:
        with pytest.raises(ValidationError):
            _frame(rotation=45)

    def test_frame_accepts_multiple_of_90_rotation(self) -> None:
        frame = _frame(rotation=180)
        assert frame.rotation == 180

    def test_frame_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _frame(surprise="y")

    def test_frame_has_no_page_number_field(self) -> None:
        """Page NUMBER must never be usable as a provenance key (pdfminer/pypdf
        were measured to silently drop pages in 3 of 8 corpus documents), so
        CoordinateFrame must not expose one at all."""
        assert "page_number" not in CoordinateFrame.model_fields
        assert "page" not in CoordinateFrame.model_fields


class TestSourceGraph:
    def test_source_node_round_trips(self) -> None:
        node = _node()
        assert node.kind == SourceNodeKind.PAPER_PDF
        assert node.sha256 == SHA_A

    def test_source_node_rejects_bad_sha(self) -> None:
        with pytest.raises(ValidationError):
            SourceNode(node_id="n1", kind=SourceNodeKind.PAPER_PDF, sha256="not-a-sha")

    def test_si_member_can_link_to_parent_paper(self) -> None:
        parent = _node(node_id="paper", kind=SourceNodeKind.PAPER_PDF, sha256=SHA_A)
        member = SourceNode(node_id="si-1", kind=SourceNodeKind.SI_MEMBER, sha256=SHA_B, parent_node_id=parent.node_id)
        assert member.parent_node_id == "paper"

    def test_bbox_locator_ref_round_trips(self) -> None:
        ref = _bbox_ref()
        assert isinstance(ref.locator, BBoxLocator)

    def test_table_cell_locator_ref_round_trips(self) -> None:
        ref = _table_ref(row=2, col=3)
        assert isinstance(ref.locator, TableCellLocator)
        assert ref.locator.row == 2
        assert ref.locator.col == 3

    def test_xpath_locator_ref_round_trips(self) -> None:
        ref = SourceRef(node_id="n1", locator=XPathLocator(xpath="//table/row[1]/cell[2]"))
        assert isinstance(ref.locator, XPathLocator)

    def test_archive_member_locator_uses_sha_as_identity(self) -> None:
        ref = SourceRef(node_id="n1", locator=ArchiveMemberLocator(member_sha256=SHA_B, display_path="./SI/data.xlsx"))
        assert isinstance(ref.locator, ArchiveMemberLocator)
        assert ref.locator.member_sha256 == SHA_B

    def test_archive_member_locator_rejects_bad_sha(self) -> None:
        with pytest.raises(ValidationError):
            ArchiveMemberLocator(member_sha256="not-a-sha")

    def test_archive_member_locator_display_path_is_optional(self) -> None:
        ref = ArchiveMemberLocator(member_sha256=SHA_B)
        assert ref.display_path is None

    def test_two_different_display_paths_can_share_identity(self) -> None:
        """Member SHA is identity; the path is display-only, so two locators
        with the same sha but differently-normalized paths are the same
        reference in every way that matters (sha256 identity)."""
        a = ArchiveMemberLocator(member_sha256=SHA_B, display_path="./a/b.csv")
        b = ArchiveMemberLocator(member_sha256=SHA_B, display_path="a/b.csv")
        assert a.member_sha256 == b.member_sha256

    def test_source_ref_rejects_unknown_locator_kind(self) -> None:
        with pytest.raises(ValidationError):
            SourceRef(node_id="n1", locator={"kind": "not_a_real_kind"})  # type: ignore[arg-type]


class TestMeasuredValue:
    def test_valid_measured_value_round_trips(self) -> None:
        mv = _measured_value()
        assert mv.canonical_decimal_value == "1.20"
        assert mv.unit_raw == "cm/s"
        assert mv.unit_canonical == "m/s"

    def test_cannot_construct_without_value_ref(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                unit_ref=_table_ref(),
            )  # type: ignore[call-arg]

    def test_cannot_construct_without_unit_ref(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
            )  # type: ignore[call-arg]

    def test_value_and_unit_refs_are_independent_source_refs(self) -> None:
        """The unit-binding fix: a value ref pointing at narrative prose and a
        unit ref pointing at a different table cell must both be
        representable independently -- this is the whole point of splitting
        them, since units were measured to be inconsistent WITHIN a single
        paper (narrative cm/s vs. a table column in m/s)."""
        mv = _measured_value()
        assert mv.value_ref.locator != mv.unit_ref.locator

    def test_raw_text_canonical_decimal_disagreement_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.30",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_non_canonical_decimal_rendering_rejected(self) -> None:
        """canonical_decimal_value must be canonical_decimal(raw_text) exactly
        -- a value that parses to the same number but was never actually run
        through canonical_decimal() (e.g. it strips a trailing zero) must
        still be rejected, not silently accepted because it's "close"."""
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.2",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_float_raw_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text=1.20,  # type: ignore[arg-type]
                canonical_decimal_value="1.20",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_float_canonical_decimal_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value=1.20,  # type: ignore[arg-type]
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_unparseable_raw_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="not-a-number",
                canonical_decimal_value="1.20",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_1e3_and_1000_remain_distinct(self) -> None:
        a = MeasuredValue(
            raw_text="1E+3",
            canonical_decimal_value="1E+3",
            unit_raw="cm/s",
            unit_canonical="m/s",
            conversion_factor="1",
            conversion_table_version="v1",
            value_ref=_bbox_ref(),
            unit_ref=_table_ref(),
        )
        b = MeasuredValue(
            raw_text="1000",
            canonical_decimal_value="1000",
            unit_raw="cm/s",
            unit_canonical="m/s",
            conversion_factor="1",
            conversion_table_version="v1",
            value_ref=_bbox_ref(),
            unit_ref=_table_ref(),
        )
        assert a.canonical_decimal_value != b.canonical_decimal_value
        assert a.canonical_decimal_value == "1E+3"
        assert b.canonical_decimal_value == "1000"

    def test_no_conversion_still_requires_explicit_factor(self) -> None:
        mv = _measured_value(unit_raw="m/s", unit_canonical="m/s", conversion_factor="1")
        assert mv.conversion_factor == "1"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                unit_raw="cm/s",
                unit_canonical="m/s",
                conversion_factor="0.01",
                conversion_table_version="v1",
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
                surprise="y",
            )  # type: ignore[call-arg]


def _uncertainty_measured_value(
    raw_text: str,
    unit_raw: str = "%",
    unit_canonical: str = "%",
    conversion_factor: str = "1",
    node_id: str = "n1",
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        unit_raw=unit_raw,
        unit_canonical=unit_canonical,
        conversion_factor=conversion_factor,
        conversion_table_version="v1",
        value_ref=_bbox_ref(node_id=node_id),
        unit_ref=_table_ref(node_id=node_id),
    )


class TestUncertainty:
    def test_genuinely_unknown_uncertainty_is_constructible(self) -> None:
        """The real corpus case: a paper states only a bare magnitude with no
        stated basis or scale at all. This must NOT require inventing a
        basis or scale -- that would be exactly the fabrication this schema
        exists to make structurally impossible."""
        unc = Uncertainty(
            kind=UncertaintyKind.UNKNOWN,
            basis=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            scale=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        assert unc.kind == UncertaintyKind.UNKNOWN
        assert isinstance(unc.basis, Absent)
        assert isinstance(unc.scale, Absent)

    def test_genuinely_unknown_uncertainty_round_trips(self) -> None:
        unc = Uncertainty(
            kind=UncertaintyKind.UNKNOWN,
            basis=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            scale=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        restored = Uncertainty.model_validate(unc.model_dump(mode="json"))
        assert restored.kind == UncertaintyKind.UNKNOWN
        assert isinstance(restored.basis, Absent)
        assert isinstance(restored.scale, Absent)
        assert isinstance(restored.upper, MeasuredValue)
        assert restored.upper.canonical_decimal_value == "5"

    def test_bounds_can_also_be_entirely_absent(self) -> None:
        """A bare "reported without a bound at all" case must also be
        representable -- kind/basis/scale/upper/lower are independently
        Maybe, not a package deal."""
        unc = Uncertainty(
            kind=UncertaintyKind.UNKNOWN,
            basis=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            scale=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            upper=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            lower=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        assert isinstance(unc.upper, Absent)
        assert isinstance(unc.lower, Absent)

    def test_unknown_kind_blocks_statistical_interpretation(self) -> None:
        unc = Uncertainty(
            kind=UncertaintyKind.UNKNOWN,
            basis=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            scale=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        assert unc.blocks_statistical_interpretation is True

    def test_stated_kind_does_not_block_statistical_interpretation(self) -> None:
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value("0.05", unit_raw="m/s", unit_canonical="m/s"),
            lower=_uncertainty_measured_value("0.05", unit_raw="m/s", unit_canonical="m/s"),
        )
        assert unc.blocks_statistical_interpretation is False

    def test_unknown_kind_is_not_flagged_as_lower_quality(self) -> None:
        """There must be no separate "quality" signal on Uncertainty that
        distinguishes an unknown kind from a stated one -- the model only
        exposes `blocks_statistical_interpretation`, which is orthogonal."""
        assert not hasattr(Uncertainty, "quality")
        assert "quality" not in Uncertainty.model_fields

    def test_bound_magnitude_carries_unit_and_source_ref(self) -> None:
        """This is the unit-binding requirement: a bound is not a bare
        number, it is a MeasuredValue with its own unit and provenance."""
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value("0.05", unit_raw="m/s", unit_canonical="m/s"),
            lower=_uncertainty_measured_value("0.05", unit_raw="m/s", unit_canonical="m/s"),
        )
        assert isinstance(unc.upper, MeasuredValue)
        assert unc.upper.unit_canonical == "m/s"
        assert unc.upper.value_ref is not None

    def test_bare_string_bound_rejected(self) -> None:
        """A bare string must be rejected where a MeasuredValue is required
        -- this is the exact shape of the reported defect."""
        with pytest.raises(ValidationError):
            Uncertainty(
                kind=UncertaintyKind.STD_DEV,
                basis=UncertaintyBasis.ABSOLUTE,
                scale=UncertaintyScale.LINEAR,
                upper="0.05",  # type: ignore[arg-type]
                lower="0.05",  # type: ignore[arg-type]
            )

    def test_asymmetric_bounds_with_different_units_round_trip(self) -> None:
        """Asymmetric bounds are independent MeasuredValues, so they may even
        carry different (but comparably-canonical) units and different
        source refs, and must still round-trip faithfully."""
        unc = Uncertainty(
            kind=UncertaintyKind.CI_95,
            basis=UncertaintyBasis.RELATIVE,
            scale=UncertaintyScale.LOG,
            upper=_uncertainty_measured_value("10", unit_raw="%", unit_canonical="%", node_id="n-upper"),
            lower=_uncertainty_measured_value("4", unit_raw="%", unit_canonical="%", node_id="n-lower"),
        )
        restored = Uncertainty.model_validate(unc.model_dump(mode="json"))
        assert isinstance(restored.upper, MeasuredValue)
        assert isinstance(restored.lower, MeasuredValue)
        assert restored.upper.canonical_decimal_value == "10"
        assert restored.lower.canonical_decimal_value == "4"
        assert restored.upper.value_ref.node_id == "n-upper"
        assert restored.lower.value_ref.node_id == "n-lower"
        assert restored.upper.canonical_decimal_value != restored.lower.canonical_decimal_value

    def test_none_rejected_for_basis(self) -> None:
        with pytest.raises(ValidationError):
            Uncertainty(
                kind=UncertaintyKind.STD_DEV,
                basis=None,  # type: ignore[arg-type]
                scale=UncertaintyScale.LINEAR,
                upper=_uncertainty_measured_value("0.05"),
                lower=_uncertainty_measured_value("0.05"),
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Uncertainty(
                kind=UncertaintyKind.STD_DEV,
                basis=UncertaintyBasis.ABSOLUTE,
                scale=UncertaintyScale.LINEAR,
                upper=_uncertainty_measured_value("0.05"),
                lower=_uncertainty_measured_value("0.05"),
                surprise="y",
            )  # type: ignore[call-arg]


class TestComposition:
    def test_air_is_representable_with_no_components(self) -> None:
        air = Composition(
            raw_name="air",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        assert air.components == []

    def test_air_round_trips_without_gaining_components(self) -> None:
        air = Composition(
            raw_name="air",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        dumped = air.model_dump(mode="json")
        restored = Composition.model_validate(dumped)
        assert restored.components == []
        assert restored.raw_name == "air"

    def test_unresolved_mixture_cannot_be_given_components(self) -> None:
        with pytest.raises(ValidationError):
            Composition(
                raw_name="air",
                resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
                basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                equivalence_ratio=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
                components=[
                    CompositionComponent(
                        species_raw_name="O2",
                        amount=_measured_value(),
                        role=ComponentRole.OXIDIZER,
                    )
                ],
            )

    def test_resolved_composition_with_components(self) -> None:
        mixture = Composition(
            raw_name="4% H2 in N2",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            components=[
                CompositionComponent(species_raw_name="H2", amount=_measured_value(), role=ComponentRole.FUEL),
                CompositionComponent(species_raw_name="N2", amount=_measured_value(), role=ComponentRole.BALANCE),
            ],
        )
        assert len(mixture.components) == 2
        assert mixture.components[0].role == ComponentRole.FUEL

    def test_equivalence_ratio_absence_reason_distinguishable(self) -> None:
        mixture = Composition(
            raw_name="4% H2 in N2",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.UNKNOWN),
        )
        assert isinstance(mixture.equivalence_ratio, Absent)
        assert mixture.equivalence_ratio.reason == AbsenceReason.UNKNOWN

    def test_equivalence_ratio_can_be_a_measured_value(self) -> None:
        mixture = Composition(
            raw_name="stoichiometric CH4/air",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            equivalence_ratio=_measured_value(
                raw_text="1.0",
                canonical_decimal_value="1.0",
                unit_raw="-",
                unit_canonical="dimensionless",
            ),
        )
        assert isinstance(mixture.equivalence_ratio, MeasuredValue)

    def test_basis_none_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Composition(
                raw_name="air",
                resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
                basis=None,  # type: ignore[arg-type]
                equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Composition(
                raw_name="air",
                resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
                basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                surprise="y",
            )  # type: ignore[call-arg]
