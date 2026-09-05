"""Tests for carmel.schemas.datasets' M-D2b(a)+(b) series-aggregate layer:
``Series``/``DataPoint``/``Coordinate``/``Observation``/``AxisDeclaration`` and
the S1-S13/E1/V4 invariants that bind them to ``DatasetEnvelope``, plus the
``TableCellLocator.table_key`` change that lands alongside them.

Kept in its own module rather than folded into test_dataset_graph_and_envelope.py
or test_dataset_schemas.py to avoid colliding with either file's existing
class names, and because this file's fixtures center on a *series* of points
rather than a single composition.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    Composition,
    Coordinate,
    CoordinateFrame,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    EmbeddedFigureDigitization,
    ExtractedTextVerification,
    ExtractionBinding,
    Maybe,
    MeasuredValue,
    MemberSheetKey,
    Observation,
    QuantityKind,
    RawArtifactVerification,
    RootSidecarVerification,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SourceVerification,
    TableCellLocator,
    TableKeyKind,
    TextSpace,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    ValueOrigin,
    XPathLocator,
    _check_source_form_for_ref,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import TABLE_V1
from tests.figure_digitization_fixtures import cite_digitization
from tests.table_inventory_fixtures import cover_for, inventory_for
from tests.test_dataset_graph_and_envelope import _extraction_binding, _glyph_health_assessment

_NO_INVENTORY = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""The only legal absence for a table cell with no PDF fragment geometry (V8)."""

_NO_DIGITIZATION = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""The only legal absence for a non-DIGITIZED series's digitization_sha256 (V9)."""


def _verification_for(extraction: ExtractionBinding | Absent) -> SourceVerification | Absent:
    """Mirror ``SourceNode``'s iff-rule: a node carries a verification record
    exactly when it carries an extraction to have verified, and an absent one
    keeps the SAME ``AbsenceReason`` (so a FIGURE_CROP's NOT_APPLICABLE stays
    NOT_APPLICABLE). Deriving it here rather than restating a literal at every
    construction site keeps these fixtures from drifting out of step with the
    validator they are meant to exercise -- a fixture that has to be hand-kept
    consistent with an invariant is a fixture that will eventually contradict
    it silently."""
    if isinstance(extraction, Absent):
        return Absent(reason=extraction.reason)
    return SourceVerification(
        raw_artifact=RawArtifactVerification.RAW_SHA256_DIGEST_AUTHENTICATED,
        extracted_text=ExtractedTextVerification.EXTRACTION_RECORD_DIGEST_AUTHENTICATED,
        root_sidecar=RootSidecarVerification.ROOT_SIDECAR_DIGEST_AUTHENTICATED,
    )


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

_NOT_REPORTED = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
"""Module-level singleton -- Absent is frozen, so sharing one instance across
every builder call that doesn't need a distinct reason is safe."""

_NO_COMPOSITION = Absent(reason=AbsenceReason.NOT_APPLICABLE)

_CURRENT_REPAIR_DEPENDENCY = SemanticDependencyUse(
    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
)
"""Module-level singleton for MeasuredValue.repair_dependency -- frozen, so
sharing one instance across every fixture that doesn't need a
deliberately-wrong or superseded dependency record is safe."""


def _embedded_table_v1() -> EmbeddedConversionTable:
    """The one conversion table every ``MeasuredValue`` fixture in this file
    cites (via ``conversion_table_sha256=TABLE_V1.sha256``) -- embedded
    verbatim so ``DatasetEnvelope.conversion_tables``'s T2 cover-exactly
    check is satisfied by every envelope built here."""
    return EmbeddedConversionTable(
        sha256=TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
    )


_NO_ORIGIN = Absent(reason=AbsenceReason.NOT_APPLICABLE)
_NO_EXTRACTION = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_GLYPH_HEALTH = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_CROP_REGION = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""SourceNode.crop_region for any kind that is not a FIGURE_CROP: nothing but
a crop was cut out of a page, and NOT_APPLICABLE is the only reason I7 accepts
there."""
_NO_EXTRACTION_CROP = Absent(reason=AbsenceReason.NOT_APPLICABLE)
_NO_GLYPH_HEALTH_CROP = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""FIGURE_CROP-specific counterparts of ``_NO_EXTRACTION``/``_NO_GLYPH_HEALTH``:
a crop is an image region with no extracted text ever to come, so
``NOT_APPLICABLE`` -- not ``NOT_EXTRACTED_YET`` -- is the only legal reason
for it (SourceNode's I6 invariant). ``_node`` below picks these for
FIGURE_CROP nodes and leaves every other kind on ``_NO_EXTRACTION``/
``_NO_GLYPH_HEALTH``, since extraction genuinely just hasn't happened yet
for those."""


# ---------------------------------------------------------------------------
# Graph / node / locator helpers (mirrors tests/test_dataset_graph_and_envelope.py)
# ---------------------------------------------------------------------------


def _node(
    node_id: str = "paper",
    kind: SourceNodeKind = SourceNodeKind.PAPER_PDF,
    sha256: str = SHA_A,
    parent_node_id: str | None = None,
) -> SourceNode:
    is_crop = kind == SourceNodeKind.FIGURE_CROP
    return SourceNode(
        node_id=node_id,
        kind=kind,
        sha256=sha256,
        parent_node_id=parent_node_id,
        origin=_NO_ORIGIN,
        extraction=_NO_EXTRACTION_CROP if is_crop else _NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH_CROP if is_crop else _NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION_CROP if is_crop else _NO_EXTRACTION),
        # I7: a crop must address the figure it was cut from, and nothing else
        # may. The region is keyed off the node id so two crops in one graph
        # never collide under I9 -- see `_crop_region` in
        # tests/test_dataset_graph_and_envelope.py for the same reasoning.
        crop_region=_bbox(frame=_frame(render_fingerprint=f"fp-{node_id}")) if is_crop else _NO_CROP_REGION,
        document_kind=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
        if kind == SourceNodeKind.SI_MEMBER
        else Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _single_paper_graph(node_id: str = "paper", sha256: str = SHA_A) -> SourceGraph:
    return SourceGraph(nodes=(_node(node_id, SourceNodeKind.PAPER_PDF, sha256),))


def _paper_and_si_member_graph(paper_id: str = "paper", si_id: str = "si") -> SourceGraph:
    paper = _node(paper_id, SourceNodeKind.PAPER_PDF, SHA_A)
    si = _node(si_id, SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id=paper_id)
    return SourceGraph(nodes=(paper, si))


def _paper_and_figure_crop_graph(paper_id: str = "paper", fig_id: str = "fig") -> SourceGraph:
    paper = _node(paper_id, SourceNodeKind.PAPER_PDF, SHA_A)
    fig = _node(fig_id, SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id=paper_id)
    return SourceGraph(nodes=(paper, fig))


def _jats_graph(node_id: str = "jats") -> SourceGraph:
    return SourceGraph(nodes=(_node(node_id, SourceNodeKind.JATS_XML, SHA_C),))


#: Which document each fixture node id holds, so a table ref can cite the inventory
#: for the node it actually targets (V8 requires the two to agree) without every
#: call site threading a sha. "jats" is absent deliberately: an XML cell has no PDF
#: fragment geometry, so its only legal citation is Absent(NOT_APPLICABLE).
_RAW_SHA_BY_NODE_ID: dict[str, str] = {"paper": SHA_A, "si": SHA_B, "fig": SHA_B}

#: Fixture node ids that are SI_MEMBER nodes. A table cell in one of these has to be a
#: workbook SHEET cell -- see :func:`_table_ref`.
_SI_MEMBER_NODE_IDS = frozenset({"si"})


def _table_ref(
    node_id: str, row: int = 0, col: int = 1, label: str = "Table 1", raw_sha256: str | None = None
) -> SourceRef:
    """A table-cell ref citing the inventory of the document ``node_id`` holds.

    Pass ``raw_sha256`` only to cite a DIFFERENT document than the node's own --
    which V8 rejects, and which some tests exist to prove it rejects.

    A ref into an SI MEMBER comes back as a workbook SHEET cell citing nothing,
    unless the caller forced a ``raw_sha256`` in order to test a rejection. V8
    refuses a CAPTION-labelled SI cell whichever way it answers the citation:
    absent would record a claim nothing checked, and present would assert the
    member is a PDF on the strength of an author-controlled digest.
    """
    if node_id in _SI_MEMBER_NODE_IDS and raw_sha256 is None:
        return SourceRef(
            node_id=node_id,
            locator=TableCellLocator(
                row=row,
                col=col,
                table_key=MemberSheetKey(kind=TableKeyKind.MEMBER_SHEET, sheet_name="Sheet1"),
                pdf_table_inventory_sha256=_NO_INVENTORY,
            ),
        )
    sha = raw_sha256 if raw_sha256 is not None else _RAW_SHA_BY_NODE_ID.get(node_id)
    citation = _NO_INVENTORY if sha is None else inventory_for(sha).inventory_sha256
    return SourceRef(
        node_id=node_id,
        locator=TableCellLocator(
            row=row,
            col=col,
            table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label=label),
            pdf_table_inventory_sha256=citation,
        ),
    )


def _frame(**kwargs: object) -> CoordinateFrame:
    defaults: dict[str, object] = {
        "render_fingerprint": "fp-1",
        "cropbox": ("0", "0", "612", "792"),
        "mediabox": ("0", "0", "612", "792"),
        "rotation": 0,
        "units": "pt",
        "dpi": _NOT_REPORTED,
        "render_settings": _NOT_REPORTED,
    }
    defaults.update(kwargs)
    return CoordinateFrame(**defaults)  # type: ignore[arg-type]


def _bbox(**kwargs: object) -> BBox:
    defaults: dict[str, object] = {"frame": _frame(), "x0": "10", "y0": "20", "x1": "30", "y1": "40"}
    defaults.update(kwargs)
    return BBox(**defaults)  # type: ignore[arg-type]


def _bbox_ref(node_id: str) -> SourceRef:
    return SourceRef(node_id=node_id, locator=BBoxLocator(bbox=_bbox()))


def _xpath_ref(node_id: str, xpath: str = "//table/row[1]/cell[1]") -> SourceRef:
    return SourceRef(node_id=node_id, locator=XPathLocator(xpath=xpath))


def _extracted_paper_graph(node_id: str = "paper") -> SourceGraph:
    """A PAPER_PDF node that actually carries an extraction binding, which V2
    requires before any ``CharSpanLocator`` may target it. Built by reusing
    test_dataset_graph_and_envelope's self-authenticating binding/assessment
    helpers rather than restating them: a hand-copied binding whose
    ``extraction_sha256`` stops recomputing from its own fields is
    schema-invalid, so duplicating that machinery here would be a second thing
    to keep in step with the validator."""
    return SourceGraph(
        nodes=(
            SourceNode(
                node_id=node_id,
                kind=SourceNodeKind.PAPER_PDF,
                sha256=SHA_A,
                parent_node_id=None,
                origin=_NO_ORIGIN,
                extraction=_extraction_binding(parent_raw_sha256=SHA_A, extracted_text_sha256=SHA_B),
                glyph_health=_glyph_health_assessment(input_sha256=SHA_B),
                verification=_verification_for(
                    _extraction_binding(parent_raw_sha256=SHA_A, extracted_text_sha256=SHA_B)
                ),
                crop_region=_NO_CROP_REGION,
                document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            ),
        )
    )


def _char_span_ref(node_id: str, start: int = 0, end: int = 3) -> SourceRef:
    """The locator V7 refuses as the source of a series VALUE -- see
    ``TestACharSpanCannotGroundASeriesValue``. Still legitimate for a unit or
    an axis label, which is why it is a general helper and not a fixture
    private to the refusal tests."""
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end),
    )


# ---------------------------------------------------------------------------
# MeasuredValue / Uncertainty helpers
# ---------------------------------------------------------------------------


def _amount(
    raw_text: str = "1.0",
    quantity_kind: QuantityKind = QuantityKind.EQUIVALENCE_RATIO,
    unit_raw: str = "-",
    unit_normalized: str = "1",
    node_id: str = "paper",
    value_ref: SourceRef | None = None,
    unit_ref: SourceRef | None = None,
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref if value_ref is not None else _table_ref(node_id, row=0, col=1),
        unit_ref=unit_ref if unit_ref is not None else _table_ref(node_id, row=0, col=2),
    )


def _equivalence_ratio_amount(raw_text: str = "1.0", node_id: str = "paper", **kwargs: object) -> MeasuredValue:
    return _amount(raw_text=raw_text, quantity_kind=QuantityKind.EQUIVALENCE_RATIO, node_id=node_id, **kwargs)  # type: ignore[arg-type]


def _velocity_amount(raw_text: str = "0.4", node_id: str = "paper", **kwargs: object) -> MeasuredValue:
    return _amount(
        raw_text=raw_text,
        quantity_kind=QuantityKind.VELOCITY,
        unit_raw="m/s",
        unit_normalized="m/s",
        node_id=node_id,
        **kwargs,  # type: ignore[arg-type]
    )


def _pressure_amount(raw_text: str = "101325", node_id: str = "paper", **kwargs: object) -> MeasuredValue:
    return _amount(
        raw_text=raw_text,
        quantity_kind=QuantityKind.PRESSURE,
        unit_raw="Pa",
        unit_normalized="Pa",
        node_id=node_id,
        **kwargs,  # type: ignore[arg-type]
    )


def _relative_uncertainty_amount(raw_text: str = "5", node_id: str = "paper", **kwargs: object) -> MeasuredValue:
    return _amount(
        raw_text=raw_text,
        quantity_kind=QuantityKind.RELATIVE_UNCERTAINTY,
        unit_raw="%",
        unit_normalized="%",
        node_id=node_id,
        **kwargs,  # type: ignore[arg-type]
    )


def _length_amount(raw_text: str = "1.5", node_id: str = "paper", **kwargs: object) -> MeasuredValue:
    return _amount(
        raw_text=raw_text,
        quantity_kind=QuantityKind.LENGTH,
        unit_raw="mm",
        # "mm" IS the conversion table's own normalized form for LENGTH -- do not
        # "correct" this to "m". MeasuredValue re-derives unit_normalized from the
        # RECORDED table and refuses any independently asserted answer, so guessing
        # the table's output here fails loudly at construction rather than storing a
        # value silently mislabeled by a factor of 1000.
        unit_normalized="mm",
        node_id=node_id,
        **kwargs,  # type: ignore[arg-type]
    )


def _uncertainty(
    kind: UncertaintyKind = UncertaintyKind.STD_DEV,
    basis: Maybe[UncertaintyBasis] = UncertaintyBasis.ABSOLUTE,
    scale: Maybe[UncertaintyScale] = UncertaintyScale.LINEAR,
    upper: Maybe[MeasuredValue] | None = None,
    lower: Maybe[MeasuredValue] | None = None,
) -> Uncertainty:
    if upper is None:
        upper = _velocity_amount("0.02")
    if lower is None:
        lower = upper
    return Uncertainty(kind=kind, basis=basis, scale=scale, upper=upper, lower=lower)


# ---------------------------------------------------------------------------
# Axis / Coordinate / Observation / DataPoint / Series / Envelope helpers
# ---------------------------------------------------------------------------


def _axis(
    axis_id: str,
    role: AxisRole,
    quantity_kind: QuantityKind,
    node_id: str = "paper",
    raw_sha256: str | None = None,
) -> AxisDeclaration:
    return AxisDeclaration(
        axis_id=axis_id,
        role=role,
        quantity_kind=quantity_kind,
        label_raw=axis_id,
        label_ref=_table_ref(node_id, row=90, col=90, raw_sha256=raw_sha256),
    )


def _valid_axes(node_id: str = "paper") -> tuple[AxisDeclaration, ...]:
    # Deliberately in ascending axis_id order: "burning_velocity" <
    # "equivalence_ratio" < "pressure".
    return (
        _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY, node_id),
        _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO, node_id),
        _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE, node_id),
    )


def _coordinate(axis_id: str, value: MeasuredValue, uncertainty: Maybe[Uncertainty] = _NOT_REPORTED) -> Coordinate:
    return Coordinate(axis_id=axis_id, value=value, uncertainty=uncertainty)


def _observation(
    axis_id: str, value: Maybe[MeasuredValue], uncertainty: Maybe[Uncertainty] = _NOT_REPORTED
) -> Observation:
    return Observation(axis_id=axis_id, value=value, uncertainty=uncertainty)


def _valid_constants(node_id: str = "paper") -> tuple[Coordinate, ...]:
    return (_coordinate("pressure", _pressure_amount(node_id=node_id)),)


def _point(
    point_id: str,
    coordinates: tuple[Coordinate, ...],
    observations: tuple[Observation, ...],
    composition: Maybe[Composition] = _NO_COMPOSITION,
) -> DataPoint:
    return DataPoint(point_id=point_id, coordinates=coordinates, observations=observations, composition=composition)


def _valid_point(point_id: str = "p1", node_id: str = "paper", eq_raw: str = "1.0", bv_raw: str = "0.4") -> DataPoint:
    return _point(
        point_id=point_id,
        coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount(eq_raw, node_id=node_id)),),
        observations=(_observation("burning_velocity", _velocity_amount(bv_raw, node_id=node_id)),),
    )


def _valid_series(
    series_id: str = "s1",
    node_id: str = "paper",
    axes: tuple[AxisDeclaration, ...] | None = None,
    constants: tuple[Coordinate, ...] | None = None,
    points: tuple[DataPoint, ...] | None = None,
    source_form: SourceForm = SourceForm.TABULAR,
    value_origin: ValueOrigin = ValueOrigin.EXPERIMENTAL,
    digitization_sha256: Maybe[str] = _NO_DIGITIZATION,
) -> Series:
    return Series(
        series_id=series_id,
        source_form=source_form,
        value_origin=value_origin,
        axes=axes if axes is not None else _valid_axes(node_id),
        constants=constants if constants is not None else _valid_constants(node_id),
        points=points if points is not None else (_valid_point(node_id=node_id),),
        digitization_sha256=digitization_sha256,
    )


def _envelope_with_series(
    series: tuple[Series, ...],
    graph: SourceGraph | None = None,
    composition: Maybe[Composition] = _NO_COMPOSITION,
    figure_digitizations: tuple[EmbeddedFigureDigitization, ...] = (),
) -> DatasetEnvelope:
    return DatasetEnvelope(
        source_graph=graph if graph is not None else _single_paper_graph(),
        composition=composition,
        series=series,
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for(series, composition),
        ooxml_table_inventories=(),
        figure_digitizations=figure_digitizations,
    )


# ===========================================================================
# S1/S2 -- axis_id uniqueness and axes ordering
# ===========================================================================


class TestSeriesAxisUniquenessAndOrder:
    def test_duplicate_axis_id_rejected(self) -> None:
        """Two axes sharing one axis_id would make `axes[axis_id]` lookups
        (used by S11's quantity-kind cross-check) ambiguous."""
        axes = (
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY),
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO),
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE),
        )
        with pytest.raises(ValidationError, match="duplicate axis_id"):
            _valid_series(axes=axes)

    def test_unique_axis_ids_accepted(self) -> None:
        series = _valid_series()
        assert [axis.axis_id for axis in series.axes] == ["burning_velocity", "equivalence_ratio", "pressure"]

    def test_axes_not_sorted_rejected(self) -> None:
        axes = (
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO),
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE),
        )
        with pytest.raises(ValidationError, match="axes must be sorted"):
            _valid_series(axes=axes)

    def test_empty_axes_rejected(self) -> None:
        """min_length=1: an axis-less series records nothing at all."""
        with pytest.raises(ValidationError, match="axes must be sorted|at least 1 item"):
            _valid_series(axes=())

    def test_sorted_axes_accepted(self) -> None:
        series = _valid_series()
        ids = [axis.axis_id for axis in series.axes]
        assert ids == sorted(ids)


# ===========================================================================
# S3/S4 -- at least one COORDINATE axis, at least one OBSERVATION axis
# ===========================================================================


class TestSeriesAxisRoleCoverage:
    def test_missing_coordinate_axis_rejected(self) -> None:
        """Without a COORDINATE axis, no point in the series can be located
        relative to another -- it is a bag of observations, not a series."""
        axes = (
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE),
        )
        points = (
            _point(
                "p1",
                coordinates=(),
                observations=(_observation("burning_velocity", _velocity_amount()),),
            ),
        )
        with pytest.raises(ValidationError, match="must declare at least one coordinate axis"):
            _valid_series(axes=axes, points=points)

    def test_missing_observation_axis_rejected(self) -> None:
        """Without an OBSERVATION axis, the series records no measured/computed
        quantity at all -- it is pure metadata."""
        axes = (
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE),
        )
        points = (
            _point(
                "p1",
                coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
                observations=(),
            ),
        )
        with pytest.raises(ValidationError, match="must declare at least one observation axis"):
            _valid_series(axes=axes, points=points)

    def test_coordinate_and_observation_axes_present_accepted(self) -> None:
        series = _valid_series()
        roles = {axis.axis_id: axis.role for axis in series.axes}
        assert roles["equivalence_ratio"] is AxisRole.COORDINATE
        assert roles["burning_velocity"] is AxisRole.OBSERVATION


# ===========================================================================
# S5 -- constants must cover exactly the CONSTANT axis set
# ===========================================================================


class TestSeriesConstantsCoverage:
    def test_missing_constant_rejected(self) -> None:
        with pytest.raises(ValidationError, match="constants must cover exactly"):
            _valid_series(constants=())

    def test_extra_constant_rejected(self) -> None:
        constants = (
            _coordinate("pressure", _pressure_amount()),
            _coordinate("equivalence_ratio", _equivalence_ratio_amount()),
        )
        with pytest.raises(ValidationError, match="constants must cover exactly"):
            _valid_series(constants=constants)

    def test_duplicate_constant_rejected(self) -> None:
        constants = (
            _coordinate("pressure", _pressure_amount()),
            _coordinate("pressure", _pressure_amount()),
        )
        with pytest.raises(ValidationError, match="constants must cover exactly"):
            _valid_series(constants=constants)

    def test_exact_constants_accepted(self) -> None:
        series = _valid_series()
        assert [c.axis_id for c in series.constants] == ["pressure"]


# ===========================================================================
# S6/S7 -- point_id uniqueness and points ordering
# ===========================================================================


class TestSeriesPointUniquenessAndOrder:
    def test_duplicate_point_id_rejected(self) -> None:
        points = (_valid_point("p1", eq_raw="1.0"), _valid_point("p1", eq_raw="1.2"))
        with pytest.raises(ValidationError, match="duplicate point_id"):
            _valid_series(points=points)

    def test_unique_point_ids_accepted(self) -> None:
        points = (_valid_point("p1"), _valid_point("p2"))
        series = _valid_series(points=points)
        assert [p.point_id for p in series.points] == ["p1", "p2"]

    def test_points_not_sorted_rejected(self) -> None:
        points = (_valid_point("p2"), _valid_point("p1"))
        with pytest.raises(ValidationError, match="points must be sorted"):
            _valid_series(points=points)

    def test_empty_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="points must be sorted|at least 1 item"):
            _valid_series(points=())

    def test_sorted_points_accepted(self) -> None:
        points = (_valid_point("p1"), _valid_point("p2"), _valid_point("p3"))
        series = _valid_series(points=points)
        ids = [p.point_id for p in series.points]
        assert ids == sorted(ids)


# ===========================================================================
# S8/S9 -- each point's coordinates/observations must cover exactly the
# COORDINATE/OBSERVATION axis sets
# ===========================================================================


class TestDataPointCoverage:
    def test_point_missing_coordinate_rejected(self) -> None:
        """A point missing a coordinate is unlocated -- it cannot be placed
        against the series' own axes at all."""
        point = _point("p1", coordinates=(), observations=(_observation("burning_velocity", _velocity_amount()),))
        with pytest.raises(ValidationError, match="coordinates must cover exactly"):
            _valid_series(points=(point,))

    def test_point_extra_coordinate_rejected(self) -> None:
        point = _point(
            "p1",
            coordinates=(
                _coordinate("equivalence_ratio", _equivalence_ratio_amount()),
                _coordinate("pressure", _pressure_amount()),
            ),
            observations=(_observation("burning_velocity", _velocity_amount()),),
        )
        with pytest.raises(ValidationError, match="coordinates must cover exactly"):
            _valid_series(points=(point,))

    def test_point_missing_observation_slot_rejected(self) -> None:
        """Absence must be expressed by `value=Absent(...)`, never by
        omitting the observation slot entirely -- see S9's docstring."""
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
            observations=(),
        )
        with pytest.raises(ValidationError, match="observations must cover exactly"):
            _valid_series(points=(point,))

    def test_point_extra_observation_rejected(self) -> None:
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
            observations=(
                _observation("burning_velocity", _velocity_amount()),
                _observation("pressure", _pressure_amount()),
            ),
        )
        with pytest.raises(ValidationError, match="observations must cover exactly"):
            _valid_series(points=(point,))

    def test_exact_coverage_accepted(self) -> None:
        series = _valid_series()
        point = series.points[0]
        assert [c.axis_id for c in point.coordinates] == ["equivalence_ratio"]
        assert [o.axis_id for o in point.observations] == ["burning_velocity"]


# ===========================================================================
# S10 -- each point must record at least one observed value
# ===========================================================================


class TestDataPointRecordsAnObservedValue:
    def test_all_observations_absent_rejected(self) -> None:
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
            observations=(_observation("burning_velocity", _NOT_REPORTED),),
        )
        with pytest.raises(ValidationError, match="records no observed value"):
            _valid_series(points=(point,))

    def test_present_observation_accepted(self) -> None:
        series = _valid_series()
        point = series.points[0]
        assert isinstance(point.observations[0].value, MeasuredValue)


# ===========================================================================
# S11 -- quantity_kind must agree with its axis (Coordinate, Observation, and
# constants alike)
# ===========================================================================


class TestQuantityKindAgreesWithAxis:
    def test_coordinate_quantity_kind_mismatch_rejected(self) -> None:
        """equivalence_ratio's axis declares QuantityKind.EQUIVALENCE_RATIO;
        binding a VELOCITY-quantity MeasuredValue to that axis must be
        rejected even though the MeasuredValue itself is internally
        consistent (VELOCITY paired with unit "m/s")."""
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _velocity_amount()),),
            observations=(_observation("burning_velocity", _velocity_amount()),),
        )
        with pytest.raises(ValidationError, match="quantity_kind disagrees with its axis"):
            _valid_series(points=(point,))

    def test_observation_quantity_kind_mismatch_rejected(self) -> None:
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
            observations=(_observation("burning_velocity", _pressure_amount()),),
        )
        with pytest.raises(ValidationError, match="quantity_kind disagrees with its axis"):
            _valid_series(points=(point,))

    def test_constant_quantity_kind_mismatch_rejected(self) -> None:
        """S11 explicitly applies to `constants` too, not just per-point
        coordinates/observations."""
        constants = (_coordinate("pressure", _velocity_amount()),)
        with pytest.raises(ValidationError, match="quantity_kind disagrees with its axis"):
            _valid_series(constants=constants)

    def test_absent_observation_skips_quantity_kind_check(self) -> None:
        """An Absent observation carries no MeasuredValue to check against
        its axis at all -- this must construct, not raise, provided some
        OTHER observation on the point is present (S10)."""
        # Declared in axis_id order, which S2 requires. Roles deliberately do NOT
        # cluster in that order -- sorting is by id alone, never by role.
        axes = (
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY),
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO),
            _axis("flame_temperature", AxisRole.OBSERVATION, QuantityKind.TEMPERATURE),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE),
        )
        point = _point(
            "p1",
            coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
            observations=(
                _observation("burning_velocity", _velocity_amount()),
                _observation("flame_temperature", _NOT_REPORTED),
            ),
        )
        series = _valid_series(axes=axes, points=(point,))
        assert isinstance(series.points[0].observations[1].value, Absent)

    def test_matching_quantity_kinds_accepted(self) -> None:
        series = _valid_series()
        assert series.points[0].coordinates[0].value.quantity_kind == QuantityKind.EQUIVALENCE_RATIO
        assert series.constants[0].value.quantity_kind == QuantityKind.PRESSURE


# ===========================================================================
# S12 -- Observation: value Absent implies uncertainty Absent
# ===========================================================================


class TestObservationUncertaintyWithoutValue:
    def test_uncertainty_present_with_absent_value_rejected(self) -> None:
        """A stated uncertainty about a value the source never reported is a
        contradiction: there is nothing for that uncertainty to bound."""
        with pytest.raises(ValidationError, match="uncertainty without a value"):
            Observation(axis_id="burning_velocity", value=_NOT_REPORTED, uncertainty=_uncertainty())

    def test_absent_value_and_absent_uncertainty_accepted(self) -> None:
        obs = Observation(axis_id="burning_velocity", value=_NOT_REPORTED, uncertainty=_NOT_REPORTED)
        assert isinstance(obs.value, Absent)
        assert isinstance(obs.uncertainty, Absent)

    def test_present_value_with_present_uncertainty_accepted(self) -> None:
        obs = Observation(axis_id="burning_velocity", value=_velocity_amount(), uncertainty=_uncertainty())
        assert isinstance(obs.value, MeasuredValue)
        assert isinstance(obs.uncertainty, Uncertainty)


# ===========================================================================
# S13 -- uncertainty bound quantity must agree with basis/value (shared helper
# used by both Coordinate and Observation)
# ===========================================================================


class TestUncertaintyBoundQuantity:
    def test_coordinate_absolute_basis_bound_mismatch_rejected(self) -> None:
        """basis=ABSOLUTE: the bound's quantity_kind must equal the value's
        own quantity_kind (an absolute uncertainty on an equivalence ratio is
        itself an equivalence-ratio-shaped quantity, not a pressure)."""
        bad_uncertainty = _uncertainty(
            basis=UncertaintyBasis.ABSOLUTE,
            upper=_pressure_amount(),
            lower=_pressure_amount(),
        )
        with pytest.raises(ValidationError, match="uncertainty bound quantity"):
            Coordinate(axis_id="equivalence_ratio", value=_equivalence_ratio_amount(), uncertainty=bad_uncertainty)

    def test_observation_absolute_basis_bound_match_accepted(self) -> None:
        good_uncertainty = _uncertainty(
            basis=UncertaintyBasis.ABSOLUTE, upper=_velocity_amount("0.02"), lower=_velocity_amount("0.02")
        )
        obs = Observation(axis_id="burning_velocity", value=_velocity_amount(), uncertainty=good_uncertainty)
        assert isinstance(obs.uncertainty, Uncertainty)

    def test_relative_basis_requires_relative_uncertainty_quantity_kind(self) -> None:
        """basis=RELATIVE: the bound must be QuantityKind.RELATIVE_UNCERTAINTY
        regardless of what quantity the value itself measures -- "+-5%" is a
        dimensionless fraction of the value, not a same-quantity magnitude."""
        bad_uncertainty = _uncertainty(
            basis=UncertaintyBasis.RELATIVE,
            upper=_velocity_amount("0.05"),
            lower=_velocity_amount("0.05"),
        )
        with pytest.raises(ValidationError, match="uncertainty bound quantity"):
            Observation(axis_id="burning_velocity", value=_velocity_amount(), uncertainty=bad_uncertainty)

    def test_relative_basis_with_relative_uncertainty_quantity_kind_accepted(self) -> None:
        good_uncertainty = _uncertainty(
            basis=UncertaintyBasis.RELATIVE,
            upper=_relative_uncertainty_amount("5"),
            lower=_relative_uncertainty_amount("5"),
        )
        obs = Observation(axis_id="burning_velocity", value=_velocity_amount(), uncertainty=good_uncertainty)
        assert isinstance(obs.uncertainty, Uncertainty)

    def test_absent_basis_skips_the_check(self) -> None:
        """The real corpus case: a bare magnitude with no stated basis at
        all must not force the bound's quantity_kind to match anything --
        that would be inventing a basis the source never stated."""
        unc = Uncertainty(
            kind=UncertaintyKind.UNKNOWN,
            basis=_NOT_REPORTED,
            scale=_NOT_REPORTED,
            upper=_pressure_amount(),
            lower=_pressure_amount(),
        )
        obs = Observation(axis_id="burning_velocity", value=_velocity_amount(), uncertainty=unc)
        assert isinstance(obs.uncertainty, Uncertainty)

    def test_absent_value_skips_the_check(self) -> None:
        """When value is Absent (S12 forces uncertainty Absent too in the
        Observation case), there is nothing for S13 to check -- exercised
        directly against Coordinate's shared helper isn't possible (a
        Coordinate's value is never Absent), so this is pinned via the
        Observation shape S12 already requires."""
        obs = Observation(axis_id="burning_velocity", value=_NOT_REPORTED, uncertainty=_NOT_REPORTED)
        assert isinstance(obs.value, Absent)


# ===========================================================================
# E1 -- DatasetEnvelope.series: series_id uniqueness and ordering
# ===========================================================================


class TestEnvelopeSeriesUniquenessAndOrder:
    def test_duplicate_series_id_rejected(self) -> None:
        series = (_valid_series(series_id="s1"), _valid_series(series_id="s1", node_id="paper"))
        with pytest.raises(ValidationError, match="duplicate series_id"):
            _envelope_with_series(series)

    def test_series_not_sorted_rejected(self) -> None:
        """Ruling: E1 carries two distinct markers, matching the S2/S7 pattern --
        "duplicate series_id" for uniqueness and "series must be sorted" for
        ordering. These two series_ids are distinct but out of order, so this
        exercises the ordering marker specifically."""
        series = (_valid_series(series_id="s2"), _valid_series(series_id="s1"))
        with pytest.raises(ValidationError, match="series must be sorted"):
            _envelope_with_series(series)

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 item"):
            _envelope_with_series(())

    def test_sorted_unique_series_accepted(self) -> None:
        series = (_valid_series(series_id="s1"), _valid_series(series_id="s2"))
        envelope = _envelope_with_series(series)
        assert [s.series_id for s in envelope.series] == ["s1", "s2"]


# ===========================================================================
# V4 -- source_form constrains Coordinate/Observation value_ref (not unit_ref,
# not label_ref, not composition refs)
# ===========================================================================


class TestSourceFormConstrainsValueRef:
    def test_tabular_requires_table_cell_locator(self) -> None:
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_bbox_ref("paper")),
                ),
            ),
            observations=(_observation("burning_velocity", _velocity_amount()),),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TABULAR)
        with pytest.raises(ValidationError, match="source_form"):
            _envelope_with_series((series,))

    def test_tabular_value_ref_via_table_cell_accepted(self) -> None:
        series = _valid_series(source_form=SourceForm.TABULAR)
        envelope = _envelope_with_series((series,))
        assert envelope.series[0].source_form is SourceForm.TABULAR

    def test_digitized_requires_figure_crop_node(self) -> None:
        """V4's DIGITIZED branch, isolated from V2.

        This deliberately uses the DEFAULT paper-only graph rather than
        ``_paper_and_figure_crop_graph()``. With a FIGURE_CROP node present but
        uncited, V2 ("no decorative nodes") fires FIRST and this test passes
        without V4 ever running -- it would then be pinning the wrong guard
        entirely. A check that trips a different guard than the one it names has
        told you nothing; the same trap cost a re-probe on V3 last milestone.
        """
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    # value_ref targets the PAPER_PDF node, not the FIGURE_CROP node.
                    _equivalence_ratio_amount(value_ref=_bbox_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_bbox_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.DIGITIZED)
        with pytest.raises(ValidationError, match="source_form"):
            _envelope_with_series((series,))

    def test_digitized_value_ref_targeting_figure_crop_accepted(self) -> None:
        graph = _paper_and_figure_crop_graph()
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_bbox_ref("fig"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_bbox_ref("fig"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.DIGITIZED)
        series, embedded = cite_digitization(series, graph, "fig")
        envelope = _envelope_with_series((series,), graph=graph, figure_digitizations=(embedded,))
        assert envelope.series[0].source_form is SourceForm.DIGITIZED

    def test_source_form_does_not_constrain_unit_ref_or_label_ref(self) -> None:
        """A digitized series may legitimately take its unit and axis label
        from the caption (a table cell), not the crop itself -- only
        value_ref is constrained by source_form."""
        graph = _paper_and_figure_crop_graph()
        axes = (
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY, node_id="paper"),
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO, node_id="paper"),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE, node_id="paper"),
        )
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    # value_ref -> FIGURE_CROP (required), unit_ref -> a TABLE_CELL on the caption.
                    _equivalence_ratio_amount(value_ref=_bbox_ref("fig"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_bbox_ref("fig"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        constants = (_coordinate("pressure", _pressure_amount(node_id="paper")),)
        series = _valid_series(axes=axes, constants=constants, points=(point,), source_form=SourceForm.DIGITIZED)
        series, embedded = cite_digitization(series, graph, "fig")
        envelope = _envelope_with_series((series,), graph=graph, figure_digitizations=(embedded,))
        assert isinstance(envelope, DatasetEnvelope)

    def test_check_source_form_for_ref_has_no_silent_escape_branch(self) -> None:
        """``_check_source_form_for_ref``'s if/elif/elif chain is deliberately
        NOT closed by an unconstrained ``else`` -- a ``source_form`` this
        function doesn't recognize must crash loudly (an ``AssertionError``
        naming the unhandled value), not fall through unchecked. Calling the
        module-level helper directly with a bogus ``source_form`` is the only
        way to exercise this branch: every real ``Series.source_form`` is
        itself a validated ``SourceForm`` member, so this path is otherwise
        unreachable through the public model.
        """
        with pytest.raises(AssertionError, match="unhandled source_form"):
            _check_source_form_for_ref(
                source_form="not-a-real-source-form",  # type: ignore[arg-type]
                ref=_bbox_ref("paper"),
                node_kind=SourceNodeKind.PAPER_PDF,
                where="test",
            )


# ===========================================================================
# P0-c: a char span cannot ground a series value (V7)
# ===========================================================================


class TestACharSpanCannotGroundASeriesValue:
    """V7: a series data point's VALUE may not be located by a char span.

    A char span in running text can support a prose-local SCALAR statement --
    that is what :class:`ConditionSetEnvelope` is for -- but it cannot support
    a series DATA POINT, because a series is a structured pairing of
    coordinates to observations and running text carries no row structure to
    prove that pairing from. This is the normative boundary: the producer
    mirrors it with a friendlier message (see
    ``TestFigureFurnitureIsRefused``), but the boundary itself lives here, so
    a hand-built envelope, a future producer, or any other caller is bound by
    it too.

    Deliberately a check on ``source_form`` ITSELF rather than a tightening of
    V4's per-ref TEXTUAL branch: V4 only ever runs on refs it finds, so a
    rejection expressed there would be conditional on a point existing to
    check, whereas the claim being refused is a property of the series.
    """

    def test_a_char_span_value_is_rejected_on_a_coordinate_and_on_an_observation(self) -> None:
        """Both cell kinds are refused -- half a rule would let a fabricated
        pairing through on whichever side went unchecked."""
        for coord_ref, obs_ref in (
            (_char_span_ref("paper"), _table_ref("paper")),
            (_table_ref("paper"), _char_span_ref("paper")),
            (_char_span_ref("paper"), _char_span_ref("paper")),
        ):
            point = _point(
                "p1",
                coordinates=(
                    _coordinate(
                        "equivalence_ratio",
                        _equivalence_ratio_amount(value_ref=coord_ref, unit_ref=_table_ref("paper")),
                    ),
                ),
                observations=(
                    _observation(
                        "burning_velocity",
                        _velocity_amount(value_ref=obs_ref, unit_ref=_table_ref("paper")),
                    ),
                ),
            )
            series = _valid_series(points=(point,), source_form=SourceForm.TEXTUAL)
            with pytest.raises(ValidationError, match="cannot ground a series data point"):
                _envelope_with_series((series,), graph=_extracted_paper_graph())

    def test_a_char_span_unit_or_label_is_still_allowed(self) -> None:
        """V7 constrains where the NUMBER came from, nothing else. A series
        legitimately takes its unit spelling and axis label from running prose,
        and a rule that swept those up would forbid honest provenance."""
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_table_ref("paper"), unit_ref=_char_span_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_table_ref("paper"), unit_ref=_char_span_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TABULAR)
        envelope = _envelope_with_series((series,), graph=_extracted_paper_graph())
        assert envelope.series[0].points[0].coordinates[0].value.unit_ref is not None

    def test_an_xpath_value_under_textual_survives(self) -> None:
        """The over-reach this rule was corrected away from. Banning
        ``source_form=TEXTUAL`` outright also banned XPATH value refs, and an
        XPath into a JATS ``<td>`` is exactly the structured pairing evidence a
        char span lacks. TEXTUAL is not the defect; the char span is."""
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(
                        node_id="jats", value_ref=_xpath_ref("jats"), unit_ref=_xpath_ref("jats")
                    ),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(node_id="jats", value_ref=_xpath_ref("jats"), unit_ref=_xpath_ref("jats")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TEXTUAL, node_id="jats")
        envelope = _envelope_with_series((series,), graph=_jats_graph())
        assert envelope.series[0].source_form is SourceForm.TEXTUAL

    def test_the_refusal_names_runtime_incapacity_not_impossibility(self) -> None:
        """Prose CAN carry a real series -- "At 300, 400 and 500 K the rates
        were 1.2, 2.4 and 4.8 s-1, respectively" is one. What this runtime
        cannot do is PROVE the pairing from a char span. The message must say
        that, so a later reader does not "discover" the counterexample and
        conclude the rule was simply wrong.
        """
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_char_span_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TEXTUAL)
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_series((series,), graph=_extracted_paper_graph())
        message = str(excinfo.value)
        assert "this runtime" in message
        assert "ConditionSetEnvelope" in message, "the honest destination must be named"

    def test_tabular_and_digitized_series_are_untouched(self) -> None:
        """The refusal is specific to TEXTUAL -- V7 must not become a blanket
        ban that would make the whole envelope unconstructible."""
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TABULAR)
        assert _envelope_with_series((series,)).series[0].source_form is SourceForm.TABULAR

    def test_the_refusal_names_the_offending_cell(self) -> None:
        """A maximal envelope has many values; a refusal that does not say
        WHICH one is a refusal the caller has to bisect by hand."""
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_char_span_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TEXTUAL)
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_series((series,), graph=_extracted_paper_graph())
        message = str(excinfo.value)
        assert "'p1'" in message and "coordinate" in message and "'equivalence_ratio'" in message

    def test_a_char_span_constant_is_deliberately_still_allowed(self) -> None:
        """The scope boundary, pinned so it stays a decision.

        A whole-series CONSTANT is a prose-local scalar statement -- "the
        pressure was held at 1 atm throughout" -- and that is exactly what a
        char span CAN support. It asserts no coordinate/observation pairing,
        so V7's argument does not reach it. Spar r94 flagged that this was
        unstated; if a later change decides constants must be covered too,
        this test is what has to be confronted rather than quietly broken.
        """
        point = _point(
            "p1",
            coordinates=(
                _coordinate(
                    "equivalence_ratio",
                    _equivalence_ratio_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
            observations=(
                _observation(
                    "burning_velocity",
                    _velocity_amount(value_ref=_table_ref("paper"), unit_ref=_table_ref("paper")),
                ),
            ),
        )
        constant = _coordinate(
            "pressure",
            _pressure_amount(value_ref=_char_span_ref("paper"), unit_ref=_table_ref("paper")),
        )
        series = _valid_series(points=(point,), source_form=SourceForm.TABULAR, constants=(constant,))
        envelope = _envelope_with_series((series,), graph=_extracted_paper_graph())
        stored_constant = envelope.series[0].constants[0]
        assert stored_constant.axis_id == "pressure"
        # Not vacuous: the constant really did keep a CHAR_SPAN value locator,
        # the exact thing V7 refuses one field over.
        assert isinstance(stored_constant.value.value_ref.locator, CharSpanLocator)

    def test_v4s_textual_branch_still_discriminates_when_called_directly(self) -> None:
        """V7 runs before V4, so V4's TEXTUAL branch is now unreachable THROUGH
        an envelope. The helper is still a standalone contract, so it is pinned
        here directly rather than left as untested dead code."""
        ref = _table_ref("paper")
        with pytest.raises(ValueError, match="source_form=TEXTUAL requires"):
            _check_source_form_for_ref(
                source_form=SourceForm.TEXTUAL,
                ref=ref,
                node_kind=SourceNodeKind.PAPER_PDF,
                where="direct call",
            )


# ===========================================================================
# TableCellLocator.table_key -- new required discriminated-union field
# ===========================================================================


class TestTableCellLocatorTableKey:
    def test_table_cell_locator_requires_table_key(self) -> None:
        with pytest.raises(ValidationError):
            TableCellLocator(row=0, col=1, pdf_table_inventory_sha256=_NO_INVENTORY)  # type: ignore[call-arg]

    def test_caption_label_key_accepted(self) -> None:
        locator = TableCellLocator(
            row=0,
            col=1,
            table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 2"),
            pdf_table_inventory_sha256=_NO_INVENTORY,
        )
        assert isinstance(locator.table_key, CaptionLabelKey)

    def test_member_sheet_key_accepted(self) -> None:
        locator = TableCellLocator(
            row=0,
            col=1,
            table_key=MemberSheetKey(kind=TableKeyKind.MEMBER_SHEET, sheet_name="Sheet1"),
            pdf_table_inventory_sha256=_NO_INVENTORY,
        )
        assert isinstance(locator.table_key, MemberSheetKey)

    def test_table_key_discriminator_rejects_mismatched_kind(self) -> None:
        with pytest.raises(ValidationError):
            CaptionLabelKey(kind=TableKeyKind.MEMBER_SHEET, label="Table 2")  # type: ignore[arg-type]


# ===========================================================================
# Immutability -- every new container field is a tuple and rejects mutation
# ===========================================================================


class TestSeriesModelsAreFrozen:
    def test_series_field_reassignment_rejected(self) -> None:
        series = _valid_series()
        with pytest.raises(ValidationError, match="frozen"):
            series.points = ()  # type: ignore[misc]

    def test_axes_is_a_tuple_and_rejects_appension(self) -> None:
        series = _valid_series()
        assert isinstance(series.axes, tuple)
        with pytest.raises(AttributeError):
            series.axes.append(series.axes[0])  # type: ignore[attr-defined]

    def test_points_is_a_tuple_and_rejects_item_replacement(self) -> None:
        series = _valid_series()
        assert isinstance(series.points, tuple)
        with pytest.raises(TypeError):
            series.points[0] = series.points[0]  # type: ignore[index]

    def test_constants_is_a_tuple_and_rejects_item_replacement(self) -> None:
        series = _valid_series()
        assert isinstance(series.constants, tuple)
        with pytest.raises(TypeError):
            series.constants[0] = series.constants[0]  # type: ignore[index]

    def test_datapoint_coordinates_is_a_tuple_and_rejects_item_replacement(self) -> None:
        point = _valid_series().points[0]
        assert isinstance(point.coordinates, tuple)
        with pytest.raises(TypeError):
            point.coordinates[0] = point.coordinates[0]  # type: ignore[index]

    def test_datapoint_observations_is_a_tuple_and_rejects_item_replacement(self) -> None:
        point = _valid_series().points[0]
        assert isinstance(point.observations, tuple)
        with pytest.raises(TypeError):
            point.observations[0] = point.observations[0]  # type: ignore[index]

    def test_envelope_series_is_a_tuple_and_rejects_item_replacement(self) -> None:
        envelope = _envelope_with_series((_valid_series(),))
        assert isinstance(envelope.series, tuple)
        with pytest.raises(TypeError):
            envelope.series[0] = envelope.series[0]  # type: ignore[index]

    def test_envelope_series_field_reassignment_rejected(self) -> None:
        envelope = _envelope_with_series((_valid_series(),))
        with pytest.raises(ValidationError, match="frozen"):
            envelope.series = ()  # type: ignore[misc]


# ===========================================================================
# V0 docstring update: composition=Absent(...) alongside a ref-bearing series
# is now constructible
# ===========================================================================


class TestCompositionAbsentGroundedThroughSeries:
    def test_composition_absent_with_ref_bearing_series_constructs(self) -> None:
        """Before the series aggregate landed, `composition` was the ONLY
        field that could carry a SourceRef, so `composition=Absent(...)`
        made the envelope ungrounded (V0). Now that `series` carries its own
        SourceRefs, an envelope with an Absent composition but a populated
        series must be groundable through the series alone."""
        envelope = _envelope_with_series((_valid_series(),), composition=_NO_COMPOSITION)
        assert isinstance(envelope.composition, Absent)
        assert len(envelope.series) == 1

    def test_an_envelope_with_no_series_at_all_is_refused(self) -> None:
        """``series=()`` is refused -- but by ``min_length``, NOT by V0.

        Pinned separately and deliberately NOT with ``match="ungrounded"``:
        once ``series`` became required with ``min_length=1``, an empty tuple
        never reaches V0 at all. Asserting "ungrounded" here would have
        recorded a false belief about which guard protects this case, and
        would keep passing if V0 were deleted outright.
        """
        graph = _single_paper_graph()
        with pytest.raises(ValidationError, match="at least 1 item"):
            DatasetEnvelope(source_graph=graph, composition=_NO_COMPOSITION, series=(), conversion_tables=())

    def test_the_structural_chain_that_makes_v0_unreachable_is_intact(self) -> None:
        """V0 ("envelope must contain >=1 SourceRef") has been DELETED, and
        this test guards the three structural links that made it unreachable
        -- and that now enforce grounding in its place.

        Measured, not reasoned: ``series`` is required with ``min_length=1``;
        ``Series.axes`` is non-empty (enforced by S2, since ``axes`` carries
        no ``MinLen`` metadata of its own); and ``AxisDeclaration.label_ref``
        is a bare required ``SourceRef``, never a ``Maybe``. Together those
        mean every constructible envelope carries at least one SourceRef, so
        grounding is now enforced STRUCTURALLY, with no runtime check ever
        needing to walk the payload looking for one.

        V0 itself was removed rather than kept: unlike I3 (acyclic), which
        stays meaningful even after the schema widens, V0's raise could never
        independently fire once these three links were in place, so a
        "passing" negative test for it could only ever be pinning one of
        THESE links' messages instead -- exactly the inert-test failure
        mutation testing exposed twice last milestone. This test replaces
        that inert one: it asserts WHY grounding cannot be violated, and
        fails loudly if any link is weakened -- at which point a fresh
        runtime guard (not a resurrected V0) would be needed again.
        """
        series_field = DatasetEnvelope.model_fields["series"]
        assert series_field.is_required()
        assert any(getattr(m, "min_length", None) == 1 for m in series_field.metadata), (
            "DatasetEnvelope.series lost min_length=1 -- an envelope with no series carries no "
            "dataset, and V0 may now be reachable again"
        )

        label_ref_field = AxisDeclaration.model_fields["label_ref"]
        assert label_ref_field.is_required()
        assert label_ref_field.annotation is SourceRef, (
            "AxisDeclaration.label_ref is no longer a bare SourceRef -- if it became a Maybe, an "
            "axis could be declared without citing anything and V0 would be reachable again"
        )

        # And the S2 link: axes cannot be empty, so at least one label_ref exists.
        with pytest.raises(ValidationError, match="at least one axis"):
            _valid_series(axes=())


# ===========================================================================
# Functional: a realistic 3-point laminar-burning-velocity series
# ===========================================================================


class TestFunctionalRealisticSeries:
    """A realistic TABULAR/EXPERIMENTAL series over a two-node graph
    (PAPER_PDF parent -> SI_MEMBER child): equivalence_ratio (COORDINATE),
    burning_velocity + markstein_length (both OBSERVATION), pressure
    (CONSTANT). Real spherical-flame-method LBV papers report S_L and
    Markstein length together and genuinely omit the Markstein value at some
    points -- so one point carries markstein_length=Absent(NOT_REPORTED_HERE)
    while its burning_velocity remains present, keeping S10 satisfied (per
    RULING 1: S10 stands as written, per-point; a point whose EVERY
    observation is Absent carries nothing a model can be compared against and
    must not be smuggled in wearing a data point's shape).
    """

    def _build(self) -> DatasetEnvelope:
        graph = _paper_and_si_member_graph(paper_id="paper", si_id="si")
        # axis_id order, per S2. Note this interleaves the roles
        # (OBSERVATION, COORDINATE, OBSERVATION, CONSTANT): sorting is by id
        # alone, and a series whose axes happened to sort role-first would hide
        # a bug in S2 rather than exercise it.
        axes = (
            _axis("burning_velocity", AxisRole.OBSERVATION, QuantityKind.VELOCITY, node_id="si"),
            _axis("equivalence_ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO, node_id="si"),
            _axis("markstein_length", AxisRole.OBSERVATION, QuantityKind.LENGTH, node_id="si"),
            _axis("pressure", AxisRole.CONSTANT, QuantityKind.PRESSURE, node_id="si"),
        )
        constants = (_coordinate("pressure", _pressure_amount(node_id="si")),)
        points = (
            _point(
                "p1",
                coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount("0.8", node_id="si")),),
                observations=(
                    _observation("burning_velocity", _velocity_amount("0.32", node_id="si")),
                    _observation("markstein_length", _length_amount("1.4", node_id="si")),
                ),
            ),
            _point(
                "p2",
                coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount("1.0", node_id="si")),),
                observations=(
                    _observation("burning_velocity", _velocity_amount("0.41", node_id="si")),
                    _observation("markstein_length", _NOT_REPORTED),
                ),
            ),
            _point(
                "p3",
                coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount("1.2", node_id="si")),),
                observations=(
                    _observation("burning_velocity", _velocity_amount("0.38", node_id="si")),
                    _observation("markstein_length", _length_amount("1.9", node_id="si")),
                ),
            ),
        )
        series = Series(
            series_id="s1",
            source_form=SourceForm.TABULAR,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=axes,
            constants=constants,
            points=points,
            digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
        return DatasetEnvelope(
            source_graph=graph,
            composition=_NO_COMPOSITION,
            series=(series,),
            conversion_tables=(_embedded_table_v1(),),
            table_inventories=cover_for((series,)),
            ooxml_table_inventories=(),
            figure_digitizations=(),
        )

    def test_constructs(self) -> None:
        envelope = self._build()
        assert len(envelope.series[0].points) == 3
        assert isinstance(envelope.series[0].points[1].observations[1].value, Absent)
        assert not isinstance(envelope.series[0].points[1].observations[0].value, Absent)

    def test_json_round_trip_preserves_equality(self) -> None:
        envelope = self._build()
        restored = DatasetEnvelope.model_validate(envelope.model_dump(mode="json"))
        assert restored == envelope

    def test_double_dump_is_byte_identical(self) -> None:
        """The content-addressed store depends on `model_dump_json()` being
        stable across repeated calls -- see TestFunctionalRealisticEnvelope's
        sibling test in test_dataset_graph_and_envelope.py."""
        envelope = self._build()
        assert envelope.model_dump_json() == envelope.model_dump_json()


# ===========================================================================
# ValueOrigin / SourceForm / _IDENTIFIER_PATTERN -- light coverage of the new
# enums and identifier syntax, not separately marker-numbered in the spec
# ===========================================================================


class TestNewEnumsAndIdentifierSyntax:
    def test_value_origin_members(self) -> None:
        assert {member.value for member in ValueOrigin} == {"experimental", "simulation", "derived"}

    def test_source_form_members(self) -> None:
        assert {member.value for member in SourceForm} == {"tabular", "digitized", "textual"}

    def test_axis_role_members(self) -> None:
        assert {member.value for member in AxisRole} == {"coordinate", "observation", "constant"}

    def test_invalid_axis_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _axis("Equivalence-Ratio", AxisRole.COORDINATE, QuantityKind.EQUIVALENCE_RATIO)

    def test_invalid_series_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_series(series_id="Not Valid!")

    def test_invalid_point_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _point(
                "Not Valid!",
                coordinates=(_coordinate("equivalence_ratio", _equivalence_ratio_amount()),),
                observations=(_observation("burning_velocity", _velocity_amount()),),
            )
