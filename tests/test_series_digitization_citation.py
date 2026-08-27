"""V9/FD1/FD2/FD3: a DIGITIZED series must cite a figure digitization this
envelope embeds, whose record is about that series, over that crop, chained to
the document's root bytes -- and a non-digitized series must cite none.

The figure-lane counterpart to tests/test_table_cell_inventory_citation.py.
Everything here is built with the low-level fixture helpers in
tests/test_dataset_series.py (graphs, refs, MeasuredValue machinery) and the
record builders in carmel.services.figure_digitization_record, so the join is
exercised end to end rather than against a mocked record.

WHAT THESE TESTS DO NOT PROVE, stated once so no reader mistakes a green run for
more than it is: a resolvable citation proves the digitized coordinates were
RECORDED and are UNALTERED, never that they are TRUE. Nothing here re-derives
markers from the crop's pixels; see EmbeddedFigureDigitization's docstring.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedFigureDigitization,
    Observation,
    QuantityKind,
    Series,
    SourceForm,
    SourceGraph,
    SourceNodeKind,
    ValueOrigin,
    _validate_figure_digitizations_cover_cited,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.figure_digitization_record import (
    FigureCoverage,
    FigureDigitization,
    MarkerCensus,
    PlotRegion,
    compute_digitization_sha,
    digitization_record_bytes,
    digitization_record_payload,
)
from tests.test_dataset_series import (
    SHA_A,
    SHA_B,
    SHA_C,
    _bbox_ref,
    _embedded_table_v1,
    _equivalence_ratio_amount,
    _node,
    _paper_and_figure_crop_graph,
    _velocity_amount,
)

SHA_D = "d" * 64

_NOT_REPORTED = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
_SAME_AS_DATASET = Absent(reason=AbsenceReason.SAME_AS_DATASET)
_NOT_APPLICABLE = Absent(reason=AbsenceReason.NOT_APPLICABLE)

REGION = PlotRegion(page=1, x_start=72.0, x_end=520.0, y_bottom=100.0, y_top=640.0)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> FigureDigitization:
    """A COMPLETE digitization of series ``s1`` off crop ``fig`` (sha SHA_B) in
    the document ``SHA_A`` (the paper root), recovering one marker. Overridable
    per test to break exactly one join."""
    recovered = overrides.pop("recovered", 1)
    fields: dict[str, object] = {
        "series_id": "s1",
        "raw_sha256": SHA_A,
        "figure_crop_node_id": "fig",
        "figure_crop_sha256": SHA_B,
        "plot_region": REGION,
        "coverage": FigureCoverage.COMPLETE,
        "census": MarkerCensus(detected=recovered),
        "recovered": recovered,
        "omissions": (),
    }
    fields.update(overrides)
    return FigureDigitization(**fields)  # type: ignore[arg-type]


def _embed(record: FigureDigitization | None = None) -> EmbeddedFigureDigitization:
    """Embed a record at its honest address and raw digest."""
    record = record if record is not None else _record()
    payload = digitization_record_payload(record)
    canonical = digitization_record_bytes(payload)
    return EmbeddedFigureDigitization(
        digitization_sha256=compute_digitization_sha(payload),
        raw_sha256=record.raw_sha256,
        canonical_json=canonical.decode("utf-8"),
    )


def _axes() -> tuple[AxisDeclaration, ...]:
    # ascending by axis_id: burning_velocity < equivalence_ratio
    return (
        AxisDeclaration(
            axis_id="burning_velocity",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.VELOCITY,
            label_raw="S_L",
            label_ref=_bbox_ref("paper"),
        ),
        AxisDeclaration(
            axis_id="equivalence_ratio",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            label_raw="phi",
            label_ref=_bbox_ref("paper"),
        ),
    )


def _point(*, coord_node: str = "fig", obs_node: str = "fig", point_id: str = "p1") -> DataPoint:
    coordinate = Coordinate(
        axis_id="equivalence_ratio",
        value=_equivalence_ratio_amount(value_ref=_bbox_ref(coord_node), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    observation = Observation(
        axis_id="burning_velocity",
        value=_velocity_amount(value_ref=_bbox_ref(obs_node), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    return DataPoint(
        point_id=point_id,
        coordinates=(coordinate,),
        observations=(observation,),
        composition=_SAME_AS_DATASET,
    )


def _axes_with_a_second_observation() -> tuple[AxisDeclaration, ...]:
    """``_axes()`` plus a second OBSERVATION axis (flame_temperature), so one
    point can carry a PRESENT observation and an ABSENT one at once -- the shape
    V9's same-crop join has to walk past without dereferencing the absent slot.
    Still ascending by axis_id."""
    return (
        *_axes(),
        AxisDeclaration(
            axis_id="flame_temperature",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.TEMPERATURE,
            label_raw="T_f",
            label_ref=_bbox_ref("paper"),
        ),
    )


def _point_with_an_absent_observation(*, crop_node: str = "fig") -> DataPoint:
    """A point observing burning_velocity (PRESENT, off the crop) and
    flame_temperature (ABSENT). Valid -- at least one observation carries a
    present value -- and the absent one is exactly the branch
    ``_iter_series_point_value_refs`` skips."""
    coordinate = Coordinate(
        axis_id="equivalence_ratio",
        value=_equivalence_ratio_amount(value_ref=_bbox_ref(crop_node), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    present = Observation(
        axis_id="burning_velocity",
        value=_velocity_amount(value_ref=_bbox_ref(crop_node), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    absent = Observation(
        axis_id="flame_temperature",
        value=_NOT_REPORTED,
        uncertainty=_NOT_APPLICABLE,
    )
    return DataPoint(
        point_id="p1",
        coordinates=(coordinate,),
        observations=(present, absent),
        composition=_SAME_AS_DATASET,
    )


def _digitized_series(
    *,
    digitization_sha256: object,
    series_id: str = "s1",
    points: tuple[DataPoint, ...] | None = None,
) -> Series:
    return Series(
        series_id=series_id,
        source_form=SourceForm.DIGITIZED,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=_axes(),
        constants=(),
        points=points if points is not None else (_point(),),
        digitization_sha256=digitization_sha256,
    )


def _envelope(
    *,
    series: tuple[Series, ...],
    figure_digitizations: tuple[EmbeddedFigureDigitization, ...],
    graph: SourceGraph | None = None,
) -> DatasetEnvelope:
    return DatasetEnvelope(
        source_graph=graph if graph is not None else _paper_and_figure_crop_graph(),
        composition=_NOT_APPLICABLE,
        series=series,
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=(),
        figure_digitizations=figure_digitizations,
    )


def _valid_envelope() -> DatasetEnvelope:
    embedded = _embed()
    series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
    return _envelope(series=(series,), figure_digitizations=(embedded,))


# ---------------------------------------------------------------------------
# Verifier 2: a valid one round-trips and resolves
# ---------------------------------------------------------------------------


class TestAValidDigitizedEnvelope:
    def test_constructs(self) -> None:
        envelope = _valid_envelope()
        assert envelope.series[0].source_form is SourceForm.DIGITIZED
        assert isinstance(envelope.series[0].digitization_sha256, str)

    def test_the_citation_resolves_to_the_stored_record_whose_bytes_hash_to_it(self) -> None:
        """The whole point of the address: series.digitization_sha256 names the
        embedded record, and that record's canonical bytes hash to it."""
        envelope = _valid_envelope()
        cited = envelope.series[0].digitization_sha256
        (embedded,) = [d for d in envelope.figure_digitizations if d.digitization_sha256 == cited]
        assert hashlib.sha256(embedded.canonical_json.encode("utf-8")).hexdigest() == cited

    def test_round_trips_byte_exact_through_identity_payload(self) -> None:
        envelope = _valid_envelope()
        payload = envelope.identity_payload()
        reparsed = DatasetEnvelope.from_identity_payload(payload)
        assert canonical_json_bytes(reparsed.identity_payload()) == canonical_json_bytes(payload)
        assert payload["identity_payload_version"] == 4
        assert payload["series"][0]["digitization_sha256"] == envelope.series[0].digitization_sha256
        assert len(payload["figure_digitizations"]) == 1


# ---------------------------------------------------------------------------
# Series._validate_digitization_sha256_shape: a PRESENT citation must be 64
# lowercase hex, refused at Series construction before any envelope join runs
# ---------------------------------------------------------------------------


class TestTheDigitizationShaShapeValidator:
    def test_a_present_but_malformed_sha_is_refused_naming_the_field_and_value(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _digitized_series(digitization_sha256="not-a-real-sha")
        msg = str(excinfo.value)
        assert "Series.digitization_sha256" in msg
        assert "64 lowercase hex characters" in msg
        assert "not-a-real-sha" in msg

    def test_a_64_hex_sha_with_a_trailing_newline_is_refused(self) -> None:
        """The exact case the validator's comment calls out: ``$`` matches just
        before a trailing newline, so only ``fullmatch`` catches ``"a"*64 +
        "\\n"``. A prefix match would let it through and mint a distinct address
        for logically identical bytes."""
        with pytest.raises(ValidationError, match="64 lowercase hex characters"):
            _digitized_series(digitization_sha256="a" * 64 + "\n")


# ---------------------------------------------------------------------------
# Verifier 1: a digitized series with no resolvable record is REFUSED
# ---------------------------------------------------------------------------


class TestADigitizedSeriesMustCiteAResolvableRecord:
    def test_absent_citation_is_refused_and_names_the_reason(self) -> None:
        series = _digitized_series(digitization_sha256=_NOT_APPLICABLE)
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=())
        msg = str(excinfo.value)
        assert "source_form=DIGITIZED" in msg
        assert "digitization_sha256 is Absent" in msg
        assert "s1" in msg

    def test_citation_not_embedded_is_refused(self) -> None:
        embedded = _embed()
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=())
        msg = str(excinfo.value)
        assert "does not embed" in msg
        assert embedded.digitization_sha256 in msg


# ---------------------------------------------------------------------------
# Verifier 3: the four producer checks, each enforced at validation
# ---------------------------------------------------------------------------


class TestTheFourProducerChecks:
    def test_series_id_mismatch_is_refused(self) -> None:
        embedded = _embed(_record(series_id="other_series"))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError, match="is about series 'other_series'"):
            _envelope(series=(series,), figure_digitizations=(embedded,))

    def test_recovered_count_mismatch_is_refused(self) -> None:
        # record says 2 markers recovered; the series carries 1 point.
        embedded = _embed(_record(recovered=2))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=(embedded,))
        msg = str(excinfo.value)
        assert "recovered 2 marker(s)" in msg
        assert "1 point(s)" in msg

    def test_crop_node_id_that_resolves_to_nothing_is_refused(self) -> None:
        embedded = _embed(_record(figure_crop_node_id="ghost"))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError, match="not the id of any node"):
            _envelope(series=(series,), figure_digitizations=(embedded,))

    def test_crop_node_id_resolving_to_a_non_crop_is_refused(self) -> None:
        # names "paper", a PAPER_PDF node, not a FIGURE_CROP.
        embedded = _embed(_record(figure_crop_node_id="paper", figure_crop_sha256=SHA_A))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError, match="not a FIGURE_CROP"):
            _envelope(series=(series,), figure_digitizations=(embedded,))

    def test_crop_sha256_mismatch_is_refused(self) -> None:
        # right crop node "fig" (sha SHA_B), but the record claims SHA_C.
        embedded = _embed(_record(figure_crop_sha256=SHA_C))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=(embedded,))
        msg = str(excinfo.value)
        assert "two halves of the" in msg
        assert "crop's identity must agree" in msg


# ---------------------------------------------------------------------------
# Verifier 4: the two joins beyond the obvious one
# ---------------------------------------------------------------------------


def _two_crop_graph() -> SourceGraph:
    paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
    fig = _node("fig", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
    fig2 = _node("fig2", SourceNodeKind.FIGURE_CROP, SHA_C, parent_node_id="paper")
    return SourceGraph(nodes=(paper, fig, fig2))


class TestTheTwoJoinsBeyondTheObviousOne:
    def test_a_series_assembled_from_two_crops_is_refused(self) -> None:
        """One point's coordinate reads off crop "fig", its observation off crop
        "fig2". The record names one crop; a series spanning two is not one
        digitization."""
        embedded = _embed()  # names crop "fig"
        series = _digitized_series(
            digitization_sha256=embedded.digitization_sha256,
            points=(_point(coord_node="fig", obs_node="fig2"),),
        )
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=(embedded,), graph=_two_crop_graph())
        msg = str(excinfo.value)
        assert "two different crops" in msg
        assert "fig2" in msg

    def test_raw_sha256_not_reaching_the_root_pdf_is_refused(self) -> None:
        """The record's document digest must equal the crop's ROOT artifact sha,
        not stop at an intermediate one. Here the record claims SHA_D; the crop
        "fig" descends from root "paper" (SHA_A)."""
        embedded = _embed(_record(raw_sha256=SHA_D))
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError) as excinfo:
            _envelope(series=(series,), figure_digitizations=(embedded,))
        msg = str(excinfo.value)
        assert "root PDF" in msg
        assert SHA_D in msg


# ---------------------------------------------------------------------------
# V9's same-crop join must WALK PAST an absent observation, not dereference it
# ---------------------------------------------------------------------------


class TestAnAbsentObservationIsWalkedPast:
    def test_a_digitized_series_with_an_absent_observation_validates(self) -> None:
        """``_iter_series_point_value_refs`` yields a value ref for every point
        coordinate and every PRESENT observation; an ABSENT observation has no
        value ref and must be skipped. A point with one present and one absent
        observation drives that skip, and -- because the present coordinate and
        observation both read off the cited crop -- the envelope still
        validates."""
        embedded = _embed()
        series = Series(
            series_id="s1",
            source_form=SourceForm.DIGITIZED,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=_axes_with_a_second_observation(),
            constants=(),
            points=(_point_with_an_absent_observation(),),
            digitization_sha256=embedded.digitization_sha256,
        )
        envelope = _envelope(series=(series,), figure_digitizations=(embedded,))
        observations = envelope.series[0].points[0].observations
        assert any(isinstance(obs.value, Absent) for obs in observations)
        assert any(not isinstance(obs.value, Absent) for obs in observations)


# ---------------------------------------------------------------------------
# Non-digitized series must cite nothing
# ---------------------------------------------------------------------------


def _textual_series(*, digitization_sha256: object) -> Series:
    coordinate = Coordinate(
        axis_id="equivalence_ratio",
        value=_equivalence_ratio_amount(value_ref=_bbox_ref("paper"), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    observation = Observation(
        axis_id="burning_velocity",
        value=_velocity_amount(value_ref=_bbox_ref("paper"), unit_ref=_bbox_ref("paper")),
        uncertainty=_NOT_REPORTED,
    )
    point = DataPoint(
        point_id="p1",
        coordinates=(coordinate,),
        observations=(observation,),
        composition=_SAME_AS_DATASET,
    )
    return Series(
        series_id="s1",
        source_form=SourceForm.TEXTUAL,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=_axes(),
        constants=(),
        points=(point,),
        digitization_sha256=digitization_sha256,
    )


class TestANonDigitizedSeriesMustCiteNothing:
    def test_a_textual_series_absent_not_applicable_is_accepted(self) -> None:
        from tests.test_dataset_series import _single_paper_graph

        series = _textual_series(digitization_sha256=_NOT_APPLICABLE)
        envelope = DatasetEnvelope(
            source_graph=_single_paper_graph(),
            composition=_NOT_APPLICABLE,
            series=(series,),
            conversion_tables=(_embedded_table_v1(),),
            table_inventories=(),
            figure_digitizations=(),
        )
        assert envelope.series[0].source_form is SourceForm.TEXTUAL

    def test_a_non_digitized_series_that_cites_a_digitization_is_refused(self) -> None:
        from tests.test_dataset_series import _single_paper_graph

        series = _textual_series(digitization_sha256=SHA_A)
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=_single_paper_graph(),
                composition=_NOT_APPLICABLE,
                series=(series,),
                conversion_tables=(_embedded_table_v1(),),
                table_inventories=(),
                figure_digitizations=(),
            )
        msg = str(excinfo.value)
        assert "only a DIGITIZED series may cite" in msg

    def test_a_non_digitized_series_absent_with_the_wrong_reason_is_refused(self) -> None:
        from tests.test_dataset_series import _single_paper_graph

        series = _textual_series(digitization_sha256=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
        with pytest.raises(ValidationError, match="only true absence"):
            DatasetEnvelope(
                source_graph=_single_paper_graph(),
                composition=_NOT_APPLICABLE,
                series=(series,),
                conversion_tables=(_embedded_table_v1(),),
                table_inventories=(),
                figure_digitizations=(),
            )


# ---------------------------------------------------------------------------
# FD1/FD2/FD3: cover exactly, no duplicates, sorted
# ---------------------------------------------------------------------------


class TestFigureDigitizationsCoverExactlyNoDuplicatesSorted:
    def test_a_decorative_embedded_digitization_is_refused(self) -> None:
        cited = _embed()
        decorative = _embed(_record(series_id="s2"))
        series = _digitized_series(digitization_sha256=cited.digitization_sha256)
        pair = tuple(sorted((cited, decorative), key=lambda d: d.digitization_sha256))
        with pytest.raises(ValidationError, match="decorative digitization"):
            _envelope(series=(series,), figure_digitizations=pair)

    def test_a_duplicate_embedded_digitization_is_refused(self) -> None:
        embedded = _embed()
        series = _digitized_series(digitization_sha256=embedded.digitization_sha256)
        with pytest.raises(ValidationError, match="duplicate digitization_sha256"):
            _envelope(series=(series,), figure_digitizations=(embedded, embedded))

    def test_the_no_fewer_half_reports_a_cited_digitization_that_is_not_embedded(self) -> None:
        """FD1's "no fewer" half restates, over the whole envelope, what V9
        enforces per series. V9 is declared first, so a DIGITIZED series citing
        an unembedded record is already refused at construction and this arm
        cannot be reached that way -- it is the defense-in-depth the docstring
        describes. Drive the function directly: take a valid envelope, strip its
        ``figure_digitizations`` via ``model_copy`` (which does not re-run the
        validators) while the series keeps its citation, and the cover check
        names exactly that sha as missing."""
        valid = _valid_envelope()
        cited = valid.series[0].digitization_sha256
        assert isinstance(cited, str)
        stripped = valid.model_copy(update={"figure_digitizations": ()})
        with pytest.raises(ValueError) as excinfo:
            _validate_figure_digitizations_cover_cited(stripped)
        msg = str(excinfo.value)
        assert "is missing digitization" in msg
        assert cited in msg

    def test_unsorted_figure_digitizations_are_refused(self) -> None:
        graph = _two_crop_graph()
        e1 = _embed(_record(series_id="s1", figure_crop_node_id="fig", figure_crop_sha256=SHA_B))
        e2 = _embed(_record(series_id="s2", figure_crop_node_id="fig2", figure_crop_sha256=SHA_C))
        s1 = _digitized_series(
            series_id="s1",
            digitization_sha256=e1.digitization_sha256,
            points=(_point(coord_node="fig", obs_node="fig", point_id="p1"),),
        )
        s2 = _digitized_series(
            series_id="s2",
            digitization_sha256=e2.digitization_sha256,
            points=(_point(coord_node="fig2", obs_node="fig2", point_id="p1"),),
        )
        ascending = tuple(sorted((e1, e2), key=lambda d: d.digitization_sha256))
        descending = tuple(reversed(ascending))
        series = tuple(sorted((s1, s2), key=lambda s: s.series_id))
        # ascending order validates; the reverse is refused.
        assert _envelope(series=series, figure_digitizations=ascending, graph=graph)
        with pytest.raises(ValidationError, match="must be sorted ascending by digitization_sha256"):
            _envelope(series=series, figure_digitizations=descending, graph=graph)
