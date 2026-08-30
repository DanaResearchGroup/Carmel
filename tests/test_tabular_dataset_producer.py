"""Unit tests for :mod:`carmel.services.tabular_dataset_producer`.

Two lanes. The refusal and spec-validation tests are corpus- AND pypdf-free:
they exercise the producer's own guards, which fire before any grounding, and
its frozen spec dataclasses, which validate at construction. The one
happy-path test needs a genuine extraction record (so it grounds char-span
labels/units against real stored text) and is therefore gated on ``pypdf`` --
it stays corpus-INDEPENDENT by grounding a synthetic PDF's text and citing a
synthetic inventory, so it runs in CI's agents-extra job without the paper
corpus present. The real corpus end-to-end lives in
:mod:`tests.test_tabular_dataset_target_acceptance`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carmel.schemas.datasets import (
    Absent,
    AxisRole,
    CaptionLabelKey,
    CharSpanLocator,
    DatasetEnvelope,
    EmbeddedTableInventory,
    SourceForm,
    TableCellLocator,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.condition_set_producer import TableCellGrounding
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_producer import DatasetProducerError
from carmel.services.tabular_dataset_producer import (
    TabularAxisSpec,
    TabularDatasetProducerError,
    TabularPointSpec,
    TabularPointValueSpec,
    produce_tabular_envelope_from_artifact,
)
from tests.pypdf_gate import require_pypdf
from tests.table_inventory_fixtures import make_embedded_inventory

_KEY = CaptionLabelKey(label="Table 1")
_RAW_SHA = "a" * 64


def _inventory(**texts_by_cell: str) -> EmbeddedTableInventory:
    """A synthetic inventory over ``_RAW_SHA`` whose cells hold ``texts_by_cell``.

    Keys are ``"r_c"`` strings (``"1_0"`` for row 1, col 0). Never re-derivable
    from a real PDF -- these tests never reach replay, only the producer's
    assembly and refusals.
    """
    cells = tuple(tuple(int(part) for part in key.split("_")) for key in texts_by_cell)  # type: ignore[misc]
    texts = {tuple(int(part) for part in key.split("_")): value for key, value in texts_by_cell.items()}
    return make_embedded_inventory(raw_sha256=_RAW_SHA, cells=cells, texts=texts)  # type: ignore[arg-type]


def _cell(inventory: EmbeddedTableInventory, row: int, col: int) -> TableCellGrounding:
    return TableCellGrounding(table_key=_KEY, row=row, col=col, inventory=inventory)


def _one_coordinate_axis() -> TabularAxisSpec:
    return TabularAxisSpec(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=units.QuantityKind.OTHER,
        label_quote="phi",
        unit_quote="phi",
    )


class TestSpecValidation:
    """The frozen spec dataclasses reject caller mistakes at construction, so a
    malformed spec fails loudly at the call site rather than deep inside
    assembly -- and every one of these runs with no artifact and no pypdf."""

    def test_axis_role_must_be_a_genuine_member(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="role='coordinate'"):
            TabularAxisSpec(
                axis_id="phi",
                role="coordinate",  # type: ignore[arg-type]
                quantity_kind=units.QuantityKind.OTHER,
                label_quote="phi",
                unit_quote="phi",
            )

    def test_axis_quantity_kind_must_be_a_genuine_member(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="quantity_kind='velocity'"):
            TabularAxisSpec(
                axis_id="s_l",
                role=AxisRole.OBSERVATION,
                quantity_kind="velocity",  # type: ignore[arg-type]
                label_quote="S",
                unit_quote="cm/s",
            )

    def test_bool_occurrence_rejected(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="label_occurrence=True"):
            TabularAxisSpec(
                axis_id="phi",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.OTHER,
                label_quote="phi",
                unit_quote="phi",
                label_occurrence=True,  # type: ignore[arg-type]
            )

    def test_cell_and_occurrence_together_rejected(self) -> None:
        inventory = _inventory(**{"0_0": "phi"})
        with pytest.raises(TabularDatasetProducerError, match="cannot be grounded both ways"):
            TabularAxisSpec(
                axis_id="phi",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.OTHER,
                label_quote="phi",
                unit_quote="phi",
                label_occurrence=1,
                label_cell=_cell(inventory, 0, 0),
            )

    def test_point_value_cell_must_be_a_table_cell_grounding(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="must be a TableCellGrounding"):
            TabularPointValueSpec(axis_id="phi", value_quote="0.5", cell="not a cell")  # type: ignore[arg-type]


class TestProducerRefusals:
    """Every refusal below fires BEFORE ``_prepare_grounding``, so no store and
    no pypdf are needed -- the workspace path is never read."""

    def _point(self, inventory: EmbeddedTableInventory) -> TabularPointSpec:
        return TabularPointSpec(
            point_id="p0",
            values=(TabularPointValueSpec(axis_id="phi", value_quote="0.5", cell=_cell(inventory, 1, 0)),),
        )

    def test_no_axes_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        with pytest.raises(TabularDatasetProducerError, match="no axes"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(),
                points=(self._point(inventory),),
            )

    def test_no_points_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TabularDatasetProducerError, match="no points"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(),),
                points=(),
            )

    def test_duplicate_axis_id_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        with pytest.raises(TabularDatasetProducerError, match="duplicate axis_id 'phi'"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(), _one_coordinate_axis()),
                points=(self._point(inventory),),
            )

    def test_duplicate_point_id_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        point = self._point(inventory)
        with pytest.raises(TabularDatasetProducerError, match="duplicate point_id 'p0'"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(),),
                points=(point, point),
            )

    def test_bad_value_origin_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        with pytest.raises(TabularDatasetProducerError, match="value_origin='experimental'"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin="experimental",  # type: ignore[arg-type]
                axes=(_one_coordinate_axis(),),
                points=(self._point(inventory),),
            )

    def test_point_naming_unknown_axis_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        point = TabularPointSpec(
            point_id="p0",
            values=(TabularPointValueSpec(axis_id="ghost", value_quote="0.5", cell=_cell(inventory, 1, 0)),),
        )
        with pytest.raises(TabularDatasetProducerError, match=r"names axis\(es\) \['ghost'\]"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(),),
                points=(point,),
            )

    def test_point_not_covering_every_axis_refused(self, tmp_path: Path) -> None:
        inventory = _inventory(**{"1_0": "0.5"})
        observation = TabularAxisSpec(
            axis_id="s_l",
            role=AxisRole.OBSERVATION,
            quantity_kind=units.QuantityKind.VELOCITY,
            label_quote="S",
            unit_quote="cm/s",
        )
        # The point carries only the coordinate, never the observation the series declares.
        point = self._point(inventory)
        with pytest.raises(TabularDatasetProducerError, match="every point must carry exactly one value"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(), observation),
                points=(point,),
            )

    def _observation_axis(self) -> TabularAxisSpec:
        return TabularAxisSpec(
            axis_id="s_l",
            role=AxisRole.OBSERVATION,
            quantity_kind=units.QuantityKind.VELOCITY,
            label_quote="S",
            unit_quote="cm/s",
        )

    def test_point_joining_two_rows_of_one_table_refused(self, tmp_path: Path) -> None:
        """The coordinate is taken from row 1 and the observation from row 2 of the
        SAME table -- each cell individually real, the join fabricated. Coverage
        passes (both axes present), so it is co-location, not coverage, that
        refuses; the message names the point and both disagreeing rows."""
        inventory = _inventory(**{"1_0": "0.5", "2_1": "96.5"})
        point = TabularPointSpec(
            point_id="p0",
            values=(
                TabularPointValueSpec(axis_id="phi", value_quote="0.5", cell=_cell(inventory, 1, 0)),
                TabularPointValueSpec(axis_id="s_l", value_quote="96.5", cell=_cell(inventory, 2, 1)),
            ),
        )
        with pytest.raises(TabularDatasetProducerError, match="DIFFERENT rows of one table") as excinfo:
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(), self._observation_axis()),
                points=(point,),
            )
        message = str(excinfo.value)
        assert "'p0'" in message
        assert "row=1" in message and "row=2" in message

    def test_point_joining_two_tables_refused(self, tmp_path: Path) -> None:
        """The coordinate is a cell of Table 1 and the observation a cell of a
        DISTINCT Table 2. Both sit at row 1, so the row rule would pass -- this is
        the table rule firing, and its message distinguishes the cross-table case
        from the row case and says a declared join key would be required."""
        table1 = _inventory(**{"1_0": "0.5"})
        table2 = _inventory(**{"1_1": "96.5"})
        point = TabularPointSpec(
            point_id="p0",
            values=(
                TabularPointValueSpec(axis_id="phi", value_quote="0.5", cell=_cell(table1, 1, 0)),
                TabularPointValueSpec(
                    axis_id="s_l",
                    value_quote="96.5",
                    cell=TableCellGrounding(table_key=CaptionLabelKey(label="Table 2"), row=1, col=1, inventory=table2),
                ),
            ),
        )
        with pytest.raises(TabularDatasetProducerError, match="DIFFERENT tables") as excinfo:
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=_RAW_SHA,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(_one_coordinate_axis(), self._observation_axis()),
                points=(point,),
            )
        message = str(excinfo.value)
        assert "'p0'" in message
        assert "cross-table join key" in message


_TEXT = (
    "Experiment log. The Temperature axis and the velocity axis were recorded "
    "at the throat. Values are reported in K and in cm/s across the sweep.\n"
)


class TestHappyPathOverSyntheticArtifact:
    """The producer assembles a validated, round-tripping ``TABULAR`` envelope
    from a stored artifact and a cited inventory -- corpus-independent, so it
    covers the assembly body wherever pypdf is installed even when the paper
    corpus is not present. Grounds the axis labels and units as REAL char spans
    against the stored text and the point VALUES at table cells."""

    def _build(self, tmp_path: Path) -> DatasetEnvelope:
        from tests.test_dataset_replay import _store_synthetic_artifact

        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        inventory = make_embedded_inventory(
            raw_sha256=stored.sha256,
            cells=((1, 0), (2, 0), (1, 1), (2, 1)),
            texts={(1, 0): "1023", (2, 0): "1073", (1, 1): "0.4", (2, 1): "0.6"},
        )

        def cell(row: int, col: int) -> TableCellGrounding:
            return TableCellGrounding(table_key=_KEY, row=row, col=col, inventory=inventory)

        axes = (
            TabularAxisSpec(
                axis_id="temperature",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.TEMPERATURE,
                label_quote="Temperature",
                unit_quote="K",
            ),
            TabularAxisSpec(
                axis_id="velocity",
                role=AxisRole.OBSERVATION,
                quantity_kind=units.QuantityKind.VELOCITY,
                label_quote="velocity",
                unit_quote="cm/s",
            ),
        )
        points = (
            TabularPointSpec(
                point_id="p0",
                values=(
                    TabularPointValueSpec(axis_id="temperature", value_quote="1023", cell=cell(1, 0)),
                    TabularPointValueSpec(axis_id="velocity", value_quote="0.4", cell=cell(1, 1)),
                ),
            ),
            TabularPointSpec(
                point_id="p1",
                values=(
                    TabularPointValueSpec(axis_id="temperature", value_quote="1073", cell=cell(2, 0)),
                    TabularPointValueSpec(axis_id="velocity", value_quote="0.6", cell=cell(2, 1)),
                ),
            ),
        )
        return produce_tabular_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=axes,
            points=points,
        )

    def test_it_builds_a_tabular_series_with_cell_grounded_values(self, tmp_path: Path) -> None:
        require_pypdf()
        env = self._build(tmp_path)
        assert len(env.series) == 1
        series = env.series[0]
        assert series.source_form is SourceForm.TABULAR
        assert len(series.points) == 2
        assert len(env.table_inventories) == 1
        # Every data-point value is a TableCellLocator; the units are char spans.
        for point in series.points:
            for slot in (*point.coordinates, *point.observations):
                assert isinstance(slot.value.value_ref.locator, TableCellLocator)
                assert isinstance(slot.value.unit_ref.locator, CharSpanLocator)

    def test_it_stores_and_loads_back_identically(self, tmp_path: Path) -> None:
        require_pypdf()
        env = self._build(tmp_path)
        datasets_root = tmp_path / "datasets"
        stored = store_dataset_envelope(datasets_root, env)
        assert load_dataset_envelope(datasets_root, stored.sha256) == env

    def test_a_value_quote_that_is_not_its_cell_is_refused(self, tmp_path: Path) -> None:
        require_pypdf()
        from tests.test_dataset_replay import _store_synthetic_artifact

        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        inventory = make_embedded_inventory(
            raw_sha256=stored.sha256, cells=((1, 0), (1, 1)), texts={(1, 0): "1023", (1, 1): "0.4"}
        )
        axis = TabularAxisSpec(
            axis_id="temperature",
            role=AxisRole.COORDINATE,
            quantity_kind=units.QuantityKind.TEMPERATURE,
            label_quote="Temperature",
            unit_quote="K",
        )
        observation = TabularAxisSpec(
            axis_id="velocity",
            role=AxisRole.OBSERVATION,
            quantity_kind=units.QuantityKind.VELOCITY,
            label_quote="velocity",
            unit_quote="cm/s",
        )
        # The cell reads "1023"; the value lies and claims "9999".
        point = TabularPointSpec(
            point_id="p0",
            values=(
                TabularPointValueSpec(axis_id="temperature", value_quote="9999", cell=_cell(inventory, 1, 0)),
                TabularPointValueSpec(axis_id="velocity", value_quote="0.4", cell=_cell(inventory, 1, 1)),
            ),
        )
        with pytest.raises(DatasetProducerError, match="the whole cell text must equal"):
            produce_tabular_envelope_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                series_id="s1",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(axis, observation),
                points=(point,),
            )

    def test_produced_envelope_carries_the_absent_composition_by_default(self, tmp_path: Path) -> None:
        require_pypdf()
        env = self._build(tmp_path)
        assert isinstance(env.composition, Absent)


class TestSeriesLevelTableProvenanceBoundary:
    """Document, and pin, WHERE the fabricated-join refusal stops.

    ``_reject_incoherent_join`` is a PER-POINT guard: it proves each point's
    values sit in one table, one row -- one record the document asserts. It says
    nothing across points, so a SERIES may draw different points from different
    tables while every point stays internally coherent. These tests record that
    the producer ACCEPTS such a series today, and that this is the deliberate
    boundary, not an oversight:

    * A series is already constrained at SERIES level by
      :meth:`DatasetEnvelope._validate_series_single_root_artifact` (V5), but only
      to one root ARTIFACT (paper) -- V5's own docstring records that finer,
      within-paper heterogeneity is a legitimate, expected shape, not a
      fabrication signal, so it stops at that granularity by design.
    * :attr:`DatasetEnvelope.conversion_tables` records that "mixed-table use
      within one envelope/series" is a provenance SMELL to be surfaced and scored
      by M-D3's trust computation, NOT corruption to be refused here.

    Crucially, a multi-table series is not always wrong, and the two legitimate
    shapes are indistinguishable from a fabricated stitch at THIS layer:
    a page-break continuation (one caption, two footprints -> two inventories,
    :meth:`test_same_caption_two_inventory_series_is_accepted`) and a single sweep
    tabulated across two captioned tables (two ``table_key``s,
    :meth:`test_cross_caption_two_table_series_is_accepted`) both look exactly like
    a corrupt cross-table join. Separating them needs the same "declared
    cross-table join key" capability the per-point guard names as an unbuilt
    future -- so no sound HARD refusal exists here yet, and none is added. A guard
    that keyed on ``table_key`` equality would false-refuse the sweep-split case;
    one that keyed on inventory equality would false-refuse the page-break case.
    """

    def _two_point_series(self, tmp_path: Path, key0: CaptionLabelKey, key1: CaptionLabelKey) -> DatasetEnvelope:
        from tests.test_dataset_replay import _store_synthetic_artifact

        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        # Two DISTINCT inventories (different cell texts -> different sha256): each
        # point is internally same-table, same-row, so each passes the per-point guard.
        inventory0 = make_embedded_inventory(
            raw_sha256=stored.sha256, cells=((1, 0), (1, 1)), texts={(1, 0): "1023", (1, 1): "0.4"}
        )
        inventory1 = make_embedded_inventory(
            raw_sha256=stored.sha256, cells=((1, 0), (1, 1)), texts={(1, 0): "1073", (1, 1): "0.6"}
        )
        axes = (
            TabularAxisSpec(
                axis_id="temperature",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.TEMPERATURE,
                label_quote="Temperature",
                unit_quote="K",
            ),
            TabularAxisSpec(
                axis_id="velocity",
                role=AxisRole.OBSERVATION,
                quantity_kind=units.QuantityKind.VELOCITY,
                label_quote="velocity",
                unit_quote="cm/s",
            ),
        )
        points = (
            TabularPointSpec(
                point_id="p0",
                values=(
                    TabularPointValueSpec(
                        axis_id="temperature",
                        value_quote="1023",
                        cell=TableCellGrounding(table_key=key0, row=1, col=0, inventory=inventory0),
                    ),
                    TabularPointValueSpec(
                        axis_id="velocity",
                        value_quote="0.4",
                        cell=TableCellGrounding(table_key=key0, row=1, col=1, inventory=inventory0),
                    ),
                ),
            ),
            TabularPointSpec(
                point_id="p1",
                values=(
                    TabularPointValueSpec(
                        axis_id="temperature",
                        value_quote="1073",
                        cell=TableCellGrounding(table_key=key1, row=1, col=0, inventory=inventory1),
                    ),
                    TabularPointValueSpec(
                        axis_id="velocity",
                        value_quote="0.6",
                        cell=TableCellGrounding(table_key=key1, row=1, col=1, inventory=inventory1),
                    ),
                ),
            ),
        )
        return produce_tabular_envelope_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=axes,
            points=points,
        )

    def test_cross_caption_two_table_series_is_accepted(self, tmp_path: Path) -> None:
        """The hole, concretely: points from two DIFFERENTLY captioned tables,
        each point internally coherent, are stitched into one series -- and the
        producer accepts it. The series silently claims 'Table 1' and 'Table 2'
        are one measurement sweep. Both cited cells are real, so replay is green.
        This is currently ACCEPTED; the guard is per-point, not per-series."""
        require_pypdf()
        env = self._two_point_series(tmp_path, CaptionLabelKey(label="Table 1"), CaptionLabelKey(label="Table 2"))
        series = env.series[0]
        assert series.source_form is SourceForm.TABULAR
        assert len(series.points) == 2
        # Two distinct inventories, each embedded -- the two points are grounded
        # in two different tables, and nothing at series level refuses it.
        assert len(env.table_inventories) == 2
        # The specific finding this test exists to catch: if the producer ever
        # normalised/overwrote table_key while still emitting two inventories,
        # the counts above would stay green with no series-level distinction
        # left to see. Pin the two DISTINCT captions by name, one per point.
        expected_keys = {"p0": CaptionLabelKey(label="Table 1"), "p1": CaptionLabelKey(label="Table 2")}
        for point in series.points:
            for slot in (*point.coordinates, *point.observations):
                locator = slot.value.value_ref.locator
                assert isinstance(locator, TableCellLocator)
                assert locator.table_key == expected_keys[point.point_id]

    def test_same_caption_two_inventory_series_is_accepted(self, tmp_path: Path) -> None:
        """The legitimate case a naive guard would false-refuse: one caption
        'Table 1' continued across a page break is registered as two footprints
        and two inventories. A series reading a point from each half is CORRECT,
        and is accepted -- so any series-level rule must admit it."""
        require_pypdf()
        key = CaptionLabelKey(label="Table 1")
        env = self._two_point_series(tmp_path, key, key)
        assert len(env.series[0].points) == 2
        assert len(env.table_inventories) == 2
        # The legitimate case's own fingerprint: BOTH points cite the SAME
        # table_key ("Table 1") -- unlike the cross-caption test -- but two
        # DIFFERENT inventory sha256s, i.e. two real footprints behind one
        # caption. If the producer ever renamed/normalised table_key away
        # from what was passed in, this would still catch it; if it ever
        # collapsed the two footprints onto one inventory, the sha256 check
        # below would catch that instead of only the inventory count.
        inventory_shas: dict[str, set[str]] = {}
        for point in env.series[0].points:
            for slot in (*point.coordinates, *point.observations):
                locator = slot.value.value_ref.locator
                assert isinstance(locator, TableCellLocator)
                assert locator.table_key == key
                inventory_shas.setdefault(point.point_id, set()).add(locator.pdf_table_inventory_sha256)
        assert inventory_shas["p0"] and inventory_shas["p1"]
        assert inventory_shas["p0"].isdisjoint(inventory_shas["p1"])
