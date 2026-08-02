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
    QuantityKind,
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
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.units import TABLE_V1

SHA_A = "a" * 64
SHA_B = "b" * 64


def _frame(**kwargs: object) -> CoordinateFrame:
    defaults: dict[str, object] = {
        "render_fingerprint": "fp-1",
        "cropbox": ("0", "0", "612", "792"),
        "mediabox": ("0", "0", "612", "792"),
        "rotation": 0,
        "units": "pt",
        "dpi": Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        "render_settings": Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
    }
    defaults.update(kwargs)
    return CoordinateFrame(**defaults)  # type: ignore[arg-type]


def _bbox(**kwargs: object) -> BBox:
    defaults: dict[str, object] = {"frame": _frame(), "x0": "10", "y0": "20", "x1": "30", "y1": "40"}
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
    quantity_kind: QuantityKind = QuantityKind.VELOCITY,
    unit_raw: str = "cm/s",
    unit_normalized: str = "cm/s",
    conversion_table_sha256: str = TABLE_V1.sha256,
    repairs: tuple[str, ...] = (),
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=canonical_decimal_value,
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=conversion_table_sha256,
        repairs=repairs,
        value_ref=_bbox_ref(),
        unit_ref=_table_ref(),
    )


class TestBBoxCrossesStoreBoundary:
    """The load-bearing regression test for the confirmed defect: the store's
    ``canonical_json_bytes`` rejects a Python float ANYWHERE in a payload
    (floats churn content-address hashes across platforms/interpreters), but
    this schema used to declare bbox/coordinate-frame fields as float --
    meaning no payload containing a bbox could ever be stored. A test that
    only constructs a BBox in isolation cannot catch that: it must actually
    cross the seam into ``canonical_json_bytes``, built the way a real caller
    would (``model_dump(mode="json")``), or it validates nothing about the
    defect this module exists to fix."""

    def test_bbox_bearing_payload_survives_canonical_json_bytes(self) -> None:
        bbox = _bbox()
        payload = {"bbox": bbox.model_dump(mode="json")}
        encoded = canonical_json_bytes(payload)
        assert isinstance(encoded, bytes)
        assert b"10" in encoded
        assert encoded.endswith(b"\n")


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
            BBox(x0="0", y0="0", x1="1", y1="1")  # type: ignore[call-arg]

    def test_bbox_with_frame_is_constructible(self) -> None:
        bbox = _bbox()
        assert bbox.frame.render_fingerprint == "fp-1"

    def test_frame_rejects_non_multiple_of_90_rotation(self) -> None:
        with pytest.raises(ValidationError):
            _frame(rotation=45)

    def test_frame_accepts_multiple_of_90_rotation(self) -> None:
        frame = _frame(rotation=180)
        assert frame.rotation == 180

    def test_frame_rejects_negative_rotation_even_though_multiple_of_90(self) -> None:
        """-90 and 270 describe the same physical page but serialize to
        different bytes, so admitting both would give one physical fact two
        different content addresses. rotation is constrained to exactly
        {0, 90, 180, 270}, not merely "a multiple of 90"."""
        with pytest.raises(ValidationError):
            _frame(rotation=-90)

    def test_frame_accepts_270_rotation(self) -> None:
        frame = _frame(rotation=270)
        assert frame.rotation == 270

    def test_frame_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _frame(surprise="y")

    def test_frame_has_no_page_number_field(self) -> None:
        """Page NUMBER must never be usable as a provenance key (pdfminer/pypdf
        were measured to silently drop pages in 3 of 8 corpus documents), so
        CoordinateFrame must not expose one at all."""
        assert "page_number" not in CoordinateFrame.model_fields
        assert "page" not in CoordinateFrame.model_fields

    def test_frame_rejects_degenerate_cropbox(self) -> None:
        """A cropbox with x0 >= x1 describes no actual page area."""
        with pytest.raises(ValidationError):
            _frame(cropbox=("0", "0", "0", "792"))

    def test_frame_rejects_inverted_mediabox(self) -> None:
        with pytest.raises(ValidationError):
            _frame(mediabox=("612", "0", "0", "792"))

    def test_frame_dpi_can_be_present(self) -> None:
        frame = _frame(dpi="300")
        assert frame.dpi == "300"

    def test_frame_dpi_can_be_explicitly_absent(self) -> None:
        frame = _frame(dpi=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
        assert isinstance(frame.dpi, Absent)

    def test_frame_none_rejected_for_dpi(self) -> None:
        """A bare None must never stand in for an explicit Absent -- see
        module docstring on Maybe."""
        with pytest.raises(ValidationError):
            _frame(dpi=None)

    def test_frame_rejects_zero_dpi(self) -> None:
        with pytest.raises(ValidationError):
            _frame(dpi="0")

    def test_frame_rejects_negative_dpi(self) -> None:
        with pytest.raises(ValidationError):
            _frame(dpi="-300")

    def test_frame_render_settings_can_be_present(self) -> None:
        frame = _frame(render_settings="pdftoppm 24.08, 300dpi")
        assert frame.render_settings == "pdftoppm 24.08, 300dpi"

    def test_frame_render_settings_can_be_explicitly_absent(self) -> None:
        frame = _frame(render_settings=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET))
        assert isinstance(frame.render_settings, Absent)

    def test_frame_none_rejected_for_render_settings(self) -> None:
        with pytest.raises(ValidationError):
            _frame(render_settings=None)

    def test_bbox_rejects_degenerate_box(self) -> None:
        """x0 == x1 locates no actual point/region."""
        with pytest.raises(ValidationError):
            _bbox(x0="10", x1="10")

    def test_bbox_rejects_inverted_box(self) -> None:
        with pytest.raises(ValidationError):
            _bbox(y0="40", y1="20")

    def test_bbox_rejects_non_canonical_coordinate(self) -> None:
        """Coordinates go through the same canonical-decimal machinery as
        every other numeric fact in this schema -- a hand-formatted string
        that isn't already canonical (e.g. a spurious leading zero) must be
        rejected, not silently accepted."""
        with pytest.raises(ValidationError):
            _bbox(x0="010")


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

    def test_table_cell_locator_rejects_negative_row(self) -> None:
        """A negative row locates no real table cell."""
        with pytest.raises(ValidationError):
            TableCellLocator(row=-1, col=0)

    def test_table_cell_locator_rejects_negative_col(self) -> None:
        with pytest.raises(ValidationError):
            TableCellLocator(row=0, col=-1)

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
        assert mv.unit_normalized == "cm/s"

    def test_cannot_construct_without_value_ref(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                unit_ref=_table_ref(),
            )  # type: ignore[call-arg]

    def test_cannot_construct_without_unit_ref(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
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
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
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
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_float_raw_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text=1.20,  # type: ignore[arg-type]
                canonical_decimal_value="1.20",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_float_canonical_decimal_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value=1.20,  # type: ignore[arg-type]
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_unparseable_raw_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="not-a-number",
                canonical_decimal_value="1.20",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
            )

    def test_1e3_and_1000_remain_distinct(self) -> None:
        a = MeasuredValue(
            raw_text="1E+3",
            canonical_decimal_value="1E+3",
            quantity_kind=QuantityKind.VELOCITY,
            unit_raw="cm/s",
            unit_normalized="cm/s",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=_bbox_ref(),
            unit_ref=_table_ref(),
        )
        b = MeasuredValue(
            raw_text="1000",
            canonical_decimal_value="1000",
            quantity_kind=QuantityKind.VELOCITY,
            unit_raw="cm/s",
            unit_normalized="cm/s",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=_bbox_ref(),
            unit_ref=_table_ref(),
        )
        assert a.canonical_decimal_value != b.canonical_decimal_value
        assert a.canonical_decimal_value == "1E+3"
        assert b.canonical_decimal_value == "1000"

    def test_no_conversion_needed_still_resolves_explicitly(self) -> None:
        """Reinterpretation of the old ``conversion_factor``-era test (that
        field is deleted -- see the migration report): a value whose
        ``unit_raw`` already equals the quantity's base unit ("no conversion
        needed") is still resolved EXPLICITLY via the table at call time, as
        an ``identity`` rule -- never implicitly skipped."""
        mv = _measured_value(unit_raw="m/s", unit_normalized="m/s")
        converted = mv.converted_to_base()
        assert converted.rule_kind == "identity"
        assert converted.exact == mv.canonical_decimal_value

    def test_alias_raw_unit_normalizes_to_recorded_spelling(self) -> None:
        """A raw unit spelled with a degree glyph ("°C") still constructs
        successfully when ``unit_normalized`` is the table's own spelling
        normalization ("C", still Celsius, not a conversion to Kelvin)."""
        mv = _measured_value(
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="°C",
            unit_normalized="C",
        )
        assert mv.unit_raw == "°C"
        assert mv.unit_normalized == "C"

    def test_wrong_unit_normalized_rejected(self) -> None:
        """The load-bearing case: ``unit_normalized`` must be the table's own
        spelling normalization of ``unit_raw``, never a unit CONVERSION
        smuggled in as if it were a normalization -- "°C" normalizes to "C"
        (still Celsius), never to "K" (a different fact, Kelvin)."""
        with pytest.raises(ValidationError):
            _measured_value(
                quantity_kind=QuantityKind.TEMPERATURE,
                unit_raw="°C",
                unit_normalized="K",
            )

    def test_unknown_conversion_table_sha_rejected(self) -> None:
        """A well-formed but unrecognized sha256 is refused -- a MeasuredValue
        is validated against the table it RECORDS, never silently
        re-interpreted against whatever table happens to be live today."""
        with pytest.raises(ValidationError, match="does not name any known conversion table"):
            _measured_value(conversion_table_sha256="0" * 64)

    def test_malformed_conversion_table_sha_rejected(self) -> None:
        """A ``conversion_table_sha256`` that is not 64 lowercase hex
        characters is rejected by the field validator before it ever reaches
        the table-lookup validator."""
        with pytest.raises(ValidationError):
            _measured_value(conversion_table_sha256="not-a-sha")

    def test_unit_not_known_for_declared_quantity_rejected(self) -> None:
        """A unit that is real but not known for the DECLARED quantity kind
        is rejected -- "atm" is a pressure unit, not a time unit."""
        with pytest.raises(ValidationError):
            _measured_value(
                quantity_kind=QuantityKind.TIME,
                unit_raw="atm",
                unit_normalized="atm",
            )

    def test_same_unit_string_differs_in_validity_across_quantity_kinds(self) -> None:
        """The same raw unit string can be valid under one quantity_kind and
        invalid under another. The spec's own example (``"%"`` resolving to
        different normalized spellings under MOLE_FRACTION vs
        RELATIVE_UNCERTAINTY) does not actually hold against the real
        carmel.services.units table -- both quantity kinds normalize and
        convert ``"%"`` identically. This test pins a real, substitutable
        distinction instead: ``"%"`` is a known unit for MOLE_FRACTION (and
        RELATIVE_UNCERTAINTY) but is NOT a known unit for EQUIVALENCE_RATIO,
        whose only known unit is ``"1"``."""
        mv = _measured_value(
            quantity_kind=QuantityKind.MOLE_FRACTION,
            unit_raw="%",
            unit_normalized="%",
        )
        assert mv.unit_normalized == "%"
        with pytest.raises(ValidationError):
            _measured_value(
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
                unit_raw="%",
                unit_normalized="%",
            )

    def test_converted_to_base_produces_expected_exact_and_rounded_values(self) -> None:
        """``converted_to_base`` recomputes the conversion on demand from
        recorded facts rather than trusting a stored value -- pin the actual
        numbers for two real quantity kinds."""
        velocity = _measured_value(
            raw_text="1.23",
            canonical_decimal_value="1.23",
            quantity_kind=QuantityKind.VELOCITY,
            unit_raw="cm/s",
            unit_normalized="cm/s",
        )
        converted = velocity.converted_to_base()
        assert converted.exact == "0.0123"
        assert converted.rounded == "0.0123"

        temperature = _measured_value(
            raw_text="25",
            canonical_decimal_value="25",
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="C",
            unit_normalized="C",
        )
        converted = temperature.converted_to_base()
        assert converted.exact == "298.15"
        assert converted.rounded == "298"

    def test_converted_to_base_under_other_quantity_kind_is_identity(self) -> None:
        """``QuantityKind.OTHER`` has no base unit, so ``converted_to_base``
        performs an identity conversion to ``unit_normalized`` itself rather
        than raising or inventing a target unit."""
        mv = _measured_value(
            raw_text="42",
            canonical_decimal_value="42",
            quantity_kind=QuantityKind.OTHER,
            unit_raw="widgets",
            unit_normalized="widgets",
        )
        converted = mv.converted_to_base()
        assert converted.rule_kind == "identity"
        assert converted.exact == "42"
        assert converted.rounded == "42"

    def test_slash_c0_repair_recovers_negative_value_with_evidence_preserved(self) -> None:
        """The load-bearing case, drawn from real corpus data: ``/C0`` is the
        minus sign in 7 of 8 real papers, so most real negative numbers in
        this corpus can only be represented via a recorded repair -- and
        raw_text must still read back exactly as printed, evidence intact."""
        mv = _measured_value(
            raw_text="/C0 1.0",
            canonical_decimal_value="-1.0",
            repairs=("slash_c0_to_minus",),
        )
        assert mv.raw_text == "/C0 1.0"
        assert mv.canonical_decimal_value == "-1.0"

    def test_thorn_repair_preserves_significant_figures(self) -> None:
        """``þ`` (U+00FE) stands in for ``+`` in a corrupted exponent. A float
        round-trip would collapse ``7.000E+17`` (4 significant figures) to
        ``7E+17`` (1 significant figure) -- assert the coefficient survives
        with all 4 digits, since canonical_decimal never re-renders via
        float()."""
        mv = _measured_value(
            raw_text="7.000Eþ17",
            canonical_decimal_value="7.000E+17",
            repairs=("thorn_to_plus",),
        )
        assert mv.canonical_decimal_value == "7.000E+17"
        mantissa = mv.canonical_decimal_value.split("E")[0]
        assert len(mantissa.replace(".", "")) == 4

    def test_unicode_minus_repair(self) -> None:
        mv = _measured_value(
            raw_text="−1.5",
            canonical_decimal_value="-1.5",
            repairs=("unicode_minus_to_ascii",),
        )
        assert mv.canonical_decimal_value == "-1.5"

    def test_underclaimed_repair_rejected(self) -> None:
        """A repair the text actually needs (``/C0`` -> minus) must be
        recorded -- silently accepting it without a claimed repair would hide
        the fact that raw_text was corrupted at all."""
        with pytest.raises(ValidationError):
            _measured_value(raw_text="/C0 1.0", canonical_decimal_value="-1.0", repairs=())

    def test_fabricated_repair_rejected(self) -> None:
        """A recorded repair the text never needed must be rejected -- the
        repairs list is a claim about the evidence and must be exactly true,
        not merely a superset of what happened."""
        with pytest.raises(ValidationError):
            _measured_value(raw_text="1.0", canonical_decimal_value="1.0", repairs=("slash_c0_to_minus",))

    def test_unknown_repair_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _measured_value(raw_text="1.0", canonical_decimal_value="1.0", repairs=("made_up_repair",))

    def test_silent_sign_loss_rejected(self) -> None:
        """The exact corpus failure mode: a subsetted font deleting a minus
        glyph produced a silent sign flip on every value in one real table.
        Claiming the /C0 repair happened is not enough on its own --
        canonical_decimal_value must actually reflect the repaired (negative)
        value, not the unrepaired positive one."""
        with pytest.raises(ValidationError):
            _measured_value(
                raw_text="/C0 1.0",
                canonical_decimal_value="1.0",
                repairs=("slash_c0_to_minus",),
            )

    def test_range_is_not_a_measured_value(self) -> None:
        with pytest.raises(ValidationError):
            _measured_value(raw_text="0.6–1.0", canonical_decimal_value="0.6")

    def test_clean_no_repair_path_still_works(self) -> None:
        mv = _measured_value(raw_text="1.23", canonical_decimal_value="1.23", repairs=())
        assert mv.canonical_decimal_value == "1.23"
        assert mv.repairs == ()

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            MeasuredValue(
                raw_text="1.20",
                canonical_decimal_value="1.20",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                value_ref=_bbox_ref(),
                unit_ref=_table_ref(),
                surprise="y",
            )  # type: ignore[call-arg]

    def test_non_finite_canonical_value_rejected(self) -> None:
        """The normalize/parse finiteness asymmetry in carmel.services.numeric
        (normalize_numeric_span accepts "1E+400" as a well-formed decimal
        while parse_numeric_span refuses it as non-finite) must not leak
        through MeasuredValue: a stored measured quantity is never 1E+400, so
        canonical_decimal_value must evaluate to a finite float even though
        canonical_decimal() itself stays permissive (it also canonicalizes
        bbox coordinates and conversion factors, which must not gain a
        finiteness opinion -- see _require_finite_as_float's docstring)."""
        with pytest.raises(ValidationError):
            _measured_value(raw_text="1E+400", canonical_decimal_value="1E+400", repairs=())
        # The corresponding positive case: a large-but-finite exponent must
        # still work -- only the finiteness-as-float boundary is enforced,
        # not an arbitrary tightening of magnitude.
        mv = _measured_value(raw_text="1E+300", canonical_decimal_value="1E+300", repairs=())
        assert mv.canonical_decimal_value == "1E+300"

    # test_zero_or_negative_conversion_factor_rejected: DELETED, not
    # preservable. Its entire premise was a user-supplied `conversion_factor`
    # field that could be zero/negative/corrupted; that field is gone by
    # design (see MeasuredValue's docstring) -- conversion coefficients now
    # live only inside the vetted, code-reviewed carmel.services.units
    # ConversionTable, never as per-value untrusted input, so there is no
    # analogous sign-corruption surface left in this schema to test. Flagged
    # in the migration report rather than silently dropped.


def _uncertainty_measured_value(
    raw_text: str,
    quantity_kind: QuantityKind = QuantityKind.RELATIVE_UNCERTAINTY,
    unit_raw: str = "%",
    unit_normalized: str = "%",
    conversion_table_sha256: str = TABLE_V1.sha256,
    node_id: str = "n1",
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=conversion_table_sha256,
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

    def test_known_kind_with_no_usable_magnitude_blocks_statistical_interpretation(self) -> None:
        """The real corpus case this fix prevents: a paper stating a bare
        "+-5%, method unstated" -- a known kind but basis/scale/upper/lower
        all Absent -- must NOT be readable as a fully quantified standard
        deviation; a known kind alone is not sufficient usability."""
        unc = Uncertainty(
            kind=UncertaintyKind.UNSPECIFIED_PERCENTAGE,
            basis=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            scale=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            upper=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
            lower=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        assert unc.blocks_statistical_interpretation is True

    def test_fully_specified_uncertainty_does_not_block_statistical_interpretation(self) -> None:
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value(
                "0.05", quantity_kind=QuantityKind.VELOCITY, unit_raw="m/s", unit_normalized="m/s"
            ),
            lower=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        assert unc.blocks_statistical_interpretation is False

    def test_stated_kind_does_not_block_statistical_interpretation(self) -> None:
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value(
                "0.05", quantity_kind=QuantityKind.VELOCITY, unit_raw="m/s", unit_normalized="m/s"
            ),
            lower=_uncertainty_measured_value(
                "0.05", quantity_kind=QuantityKind.VELOCITY, unit_raw="m/s", unit_normalized="m/s"
            ),
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
            upper=_uncertainty_measured_value(
                "0.05", quantity_kind=QuantityKind.VELOCITY, unit_raw="m/s", unit_normalized="m/s"
            ),
            lower=_uncertainty_measured_value(
                "0.05", quantity_kind=QuantityKind.VELOCITY, unit_raw="m/s", unit_normalized="m/s"
            ),
        )
        assert isinstance(unc.upper, MeasuredValue)
        assert unc.upper.unit_normalized == "m/s"
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
            upper=_uncertainty_measured_value("10", unit_raw="%", unit_normalized="%", node_id="n-upper"),
            lower=_uncertainty_measured_value("4", unit_raw="%", unit_normalized="%", node_id="n-lower"),
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

    def test_negative_uncertainty_bound_rejected(self) -> None:
        """An uncertainty bound is a magnitude (a distance), never negative
        or zero -- asymmetric uncertainty is represented by upper/lower
        differing in SIZE, never by one of them being negative."""
        with pytest.raises(ValidationError):
            Uncertainty(
                kind=UncertaintyKind.STD_DEV,
                basis=UncertaintyBasis.ABSOLUTE,
                scale=UncertaintyScale.LINEAR,
                upper=_uncertainty_measured_value("-5"),
                lower=_uncertainty_measured_value("5"),
            )
        # The corresponding positive case: a genuine positive bound still works.
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        assert isinstance(unc.upper, MeasuredValue)
        assert unc.upper.canonical_decimal_value == "5"

    def test_unspecified_percentage_blocks_even_when_fully_populated(self) -> None:
        """UNSPECIFIED_PERCENTAGE's entire meaning is that the statistical
        method was NOT stated -- measured over the 8 real corpus papers, an
        explicit uncertainty KIND is stated in only 1 of 8, so this is the
        common case, not an edge case. It must block statistical
        interpretation exactly like UNKNOWN, even with basis, scale, and a
        bound all present -- a consumer still cannot know whether "+-5%" is
        a standard deviation, a 95% confidence interval, or an instrument
        spec."""
        unc = Uncertainty(
            kind=UncertaintyKind.UNSPECIFIED_PERCENTAGE,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        assert unc.blocks_statistical_interpretation is True
        # The corresponding case that must still work: a stated kind (with the
        # same basis/scale/bounds shape) does not block.
        stated = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_uncertainty_measured_value("5"),
            lower=_uncertainty_measured_value("5"),
        )
        assert stated.blocks_statistical_interpretation is False


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

    def test_component_with_concrete_role_still_constructs(self) -> None:
        """Regression/sanity: a component whose role IS stated in the source
        must still construct with a concrete ComponentRole, unchanged by
        role becoming Maybe-typed."""
        component = CompositionComponent(
            species_raw_name="H2",
            amount=_measured_value(),
            role=ComponentRole.FUEL,
        )
        assert component.role == ComponentRole.FUEL

    def test_component_with_unstated_role_constructs_without_fabrication(self) -> None:
        """The real corpus case: a component is listed with no stated role
        at all (e.g. a bare species list). This must be representable
        without forcing the extractor to invent fuel/oxidizer/diluent/etc."""
        component = CompositionComponent(
            species_raw_name="Ar",
            amount=_measured_value(),
            role=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        assert isinstance(component.role, Absent)
        assert component.role.reason == AbsenceReason.NOT_REPORTED_HERE

    def test_equivalence_ratio_absence_reason_distinguishable(self) -> None:
        mixture = Composition(
            raw_name="4% H2 in N2",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.UNKNOWN),
            components=[
                CompositionComponent(species_raw_name="H2", amount=_measured_value(), role=ComponentRole.FUEL),
            ],
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
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
                unit_raw="-",
                unit_normalized="1",
            ),
        )
        assert isinstance(mixture.equivalence_ratio, MeasuredValue)

    def test_resolved_components_cannot_be_empty(self) -> None:
        """The mirror-image guard: a Composition claiming
        resolution=RESOLVED_COMPONENTS with an empty components list would
        let a downstream consumer read this as a resolved composition with
        no actual composition data -- rejected, just like
        UNRESOLVED_NAMED_MIXTURE is rejected for carrying components."""
        with pytest.raises(ValidationError):
            Composition(
                raw_name="4% H2 in N2",
                resolution=CompositionResolution.RESOLVED_COMPONENTS,
                basis=CompositionBasis.MOLE_FRACTION,
                equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                components=[],
            )

    def test_resolved_components_with_at_least_one_component_still_valid(self) -> None:
        mixture = Composition(
            raw_name="4% H2 in N2",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            components=[
                CompositionComponent(species_raw_name="H2", amount=_measured_value(), role=ComponentRole.FUEL),
            ],
        )
        assert len(mixture.components) == 1

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
