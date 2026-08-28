"""The header-quote -> per-row-tuple resolver, on synthetic grids.

These tests never touch the corpus: the resolver is pure logic over an
:class:`~carmel.schemas.datasets.EmbeddedTableInventory`, so a hand-built grid
(``make_embedded_inventory_with_texts``) exercises every branch, including the
two failures the whole bridge exists to prevent -- the model asserting a cell
address, and the model asserting a join. The end-to-end path over a real stored
table is :mod:`tests.test_tabular_series_bridge_acceptance`.
"""

from __future__ import annotations

import dataclasses

import pytest

from carmel.agents.extraction_agent import ProposedTabularAxis, TabularSeriesProposal
from carmel.schemas.datasets import AxisRole, CaptionLabelKey, EmbeddedTableInventory
from carmel.services import units
from carmel.services.tabular_series_resolver import (
    AxisHeaderIntent,
    TabularSeriesResolutionError,
    resolve_tabular_series,
)
from tests.table_inventory_fixtures import make_embedded_inventory_with_texts

_RAW = "a" * 64
_TABLE = CaptionLabelKey(label="Table 1")


def _grid(cell_texts: dict[tuple[int, int], str]) -> EmbeddedTableInventory:
    return make_embedded_inventory_with_texts(raw_sha256=_RAW, cell_texts=cell_texts)


def _intent(
    axis_id: str,
    header: str,
    *,
    role: AxisRole = AxisRole.COORDINATE,
    quantity_kind: units.QuantityKind = units.QuantityKind.OTHER,
    prose_unit: str | None = None,
    unit_is_header: bool = False,
) -> AxisHeaderIntent:
    return AxisHeaderIntent(
        axis_id=axis_id,
        role=role,
        quantity_kind=quantity_kind,
        header_quote=header,
        prose_unit_quote=None if unit_is_header else prose_unit,
        prose_unit_occurrence=None,
        unit_is_header=unit_is_header,
    )


#: A clean 2-column grid: header row 0, three all-numeral data rows.
_CLEAN = {
    (0, 0): "phi",
    (0, 1): "S",
    (1, 0): "0.5",
    (1, 1): "67.2",
    (2, 0): "1.0",
    (2, 1): "218.0",
    (3, 0): "2.0",
    (3, 1): "278.9",
}

_PHI = _intent("phi", "phi", unit_is_header=True)
_S = _intent("s", "S", role=AxisRole.OBSERVATION, quantity_kind=units.QuantityKind.VELOCITY, prose_unit="cm/s")


class TestTheHappyWalk:
    def test_one_point_per_data_row_every_value_from_that_rows_cell(self) -> None:
        resolved = resolve_tabular_series(table_key=_TABLE, inventory=_grid(_CLEAN), axes=(_PHI, _S))
        assert [p.point_id for p in resolved.points] == ["row_1", "row_2", "row_3"]
        by_row = {
            p.point_id: {v.axis_id: (v.value_quote, v.cell.row, v.cell.col) for v in p.values} for p in resolved.points
        }
        assert by_row["row_1"] == {"phi": ("0.5", 1, 0), "s": ("67.2", 1, 1)}
        assert by_row["row_3"] == {"phi": ("2.0", 3, 0), "s": ("278.9", 3, 1)}
        # The header cell each axis label is grounded at was derived, at the header row.
        label_cells = {a.axis_id: (a.label_cell.row, a.label_cell.col) for a in resolved.axes}
        assert label_cells == {"phi": (0, 0), "s": (0, 1)}

    def test_a_header_text_that_also_appears_in_its_data_column_still_resolves(self) -> None:
        """Column resolution tolerates the header text repeating DOWN its own column
        (a coincidental data value equal to the header): the column is still one
        column, and the header ROW is the one all axes share. What it refuses is the
        header repeating ACROSS columns -- a different hazard, covered below."""
        grid = {(0, 0): "0.5", (0, 1): "S", (1, 0): "0.5", (1, 1): "67.2", (2, 0): "1.0", (2, 1): "218.0"}
        resolved = resolve_tabular_series(
            table_key=_TABLE,
            inventory=_grid(grid),
            axes=(_intent("phi", "0.5", unit_is_header=True), _S),
        )
        assert [p.point_id for p in resolved.points] == ["row_1", "row_2"]
        assert {a.axis_id: a.label_cell.row for a in resolved.axes} == {"phi": 0, "s": 0}


class TestRefusals:
    def test_a_header_matching_no_column_is_refused(self) -> None:
        with pytest.raises(TabularSeriesResolutionError, match="matches no cell"):
            resolve_tabular_series(
                table_key=_TABLE, inventory=_grid(_CLEAN), axes=(_intent("x", "nope", unit_is_header=True), _S)
            )

    def test_a_header_matching_several_columns_is_refused(self) -> None:
        # "k" heads TWO columns; "S" heads one. The multi-column refusal must fire on
        # "k" -- and because "S" is a distinct real column, removing that guard would
        # let the resolver PRODUCE a series (picking a "k" column arbitrarily), which
        # is exactly the pick-don't-refuse failure this test detects.
        grid = {(0, 0): "k", (0, 1): "k", (0, 2): "S", (1, 0): "1.0", (1, 1): "2.0", (1, 2): "67.2"}
        with pytest.raises(TabularSeriesResolutionError, match="matches several columns"):
            resolve_tabular_series(
                table_key=_TABLE,
                inventory=_grid(grid),
                axes=(_intent("a", "k", unit_is_header=True), _S),
            )

    def test_headers_not_sharing_one_row_is_refused(self) -> None:
        grid = {(0, 0): "phi", (1, 1): "S", (2, 0): "0.5", (2, 1): "67.2"}
        with pytest.raises(TabularSeriesResolutionError, match="do not share exactly one header row"):
            resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))

    def test_header_row_refusal_names_each_axis_own_matched_rows(self) -> None:
        # The intersection is empty here (phi heads row 0, S heads row 1), so the old
        # message printed only "[]" -- useless. The message must instead show which
        # rows EACH axis matched, so the reader sees the misalignment (0 vs 1).
        grid = {(0, 0): "phi", (1, 1): "S", (2, 0): "0.5", (2, 1): "67.2"}
        with pytest.raises(TabularSeriesResolutionError) as excinfo:
            resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))
        message = str(excinfo.value)
        assert "'phi': [0]" in message
        assert "'s': [1]" in message

    def test_two_axes_resolving_to_one_column_is_refused(self) -> None:
        with pytest.raises(TabularSeriesResolutionError, match="both resolve to column"):
            resolve_tabular_series(
                table_key=_TABLE,
                inventory=_grid(_CLEAN),
                axes=(_PHI, _intent("phi2", "phi", role=AxisRole.OBSERVATION, unit_is_header=True)),
            )

    def test_a_unit_row_between_header_and_data_is_skipped_as_furniture(self) -> None:
        grid = {
            (0, 0): "phi",
            (0, 1): "S",
            (1, 0): "(-)",  # a unit row: neither column is a numeral
            (1, 1): "(cm/s)",
            (2, 0): "0.5",
            (2, 1): "67.2",
            (3, 0): "1.0",
            (3, 1): "218.0",
        }
        resolved = resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))
        assert [p.point_id for p in resolved.points] == ["row_2", "row_3"]

    def test_a_row_with_a_value_in_some_columns_but_not_all_is_refused(self) -> None:
        grid = {
            (0, 0): "phi",
            (0, 1): "S",
            (1, 0): "0.5",  # a numeral here
            (1, 1): "see note",  # but not here -- neither cleanly data nor cleanly furniture
            (2, 0): "1.0",
            (2, 1): "218.0",
        }
        with pytest.raises(TabularSeriesResolutionError, match="row 1: .* neither cleanly data"):
            resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))

    def test_a_grid_with_no_data_rows_is_refused(self) -> None:
        grid = {(0, 0): "phi", (0, 1): "S", (1, 0): "(-)", (1, 1): "(cm/s)"}  # header + a unit row, no data
        with pytest.raises(TabularSeriesResolutionError, match="no data rows below header row"):
            resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))


class TestTheModelCannotSmuggleACoordinate:
    """Verifier 5a. The model's output ALONE cannot select a cell the document does
    not support. This is asserted TWO ways: structurally (no schema field can even
    express a cell address) and functionally (the only spatial lever is a header
    quote, which either names a real column or is refused -- never a raw ordinal)."""

    def test_no_proposal_field_can_express_a_row_or_column_ordinal(self) -> None:
        # The proposal the untrusted model fills, and the schema-free intent the
        # carrier hands the resolver, both: an axis names its column by TEXT only.
        axis_fields = set(ProposedTabularAxis.model_fields)
        assert axis_fields == {"axis_id", "role", "quantity_kind", "header_quote", "unit"}
        proposal_fields = set(TabularSeriesProposal.model_fields)
        assert proposal_fields == {"artifact_sha256", "table_label", "series_id", "value_origin", "axes", "done"}
        intent_fields = {f.name for f in dataclasses.fields(AxisHeaderIntent)}
        # header_quote is text; nothing here is a row, a col, or a cell.
        assert "row" not in intent_fields and "col" not in intent_fields and "cell" not in intent_fields
        for name in ("row", "col", "cell", "cells", "coordinate", "coordinates"):
            assert name not in axis_fields and name not in proposal_fields

    def test_the_emitted_cells_come_from_the_grid_not_the_intent(self) -> None:
        # Changing WHICH real column the header names moves the resolved column;
        # naming a column that does not exist is refused. The model can select a
        # real column by its printed text and nothing else -- never an arbitrary cell.
        grid = {(0, 0): "phi", (0, 1): "S", (0, 2): "U", (1, 0): "0.5", (1, 1): "67.2", (1, 2): "7.1"}
        via_s = resolve_tabular_series(table_key=_TABLE, inventory=_grid(grid), axes=(_PHI, _S))
        assert via_s.points[0].values[1].cell.col == 1
        via_u = resolve_tabular_series(
            table_key=_TABLE,
            inventory=_grid(grid),
            axes=(_PHI, _intent("u", "U", role=AxisRole.OBSERVATION, unit_is_header=True)),
        )
        assert via_u.points[0].values[1].cell.col == 2  # a DIFFERENT real column, by its header text
        with pytest.raises(TabularSeriesResolutionError, match="matches no cell"):
            resolve_tabular_series(
                table_key=_TABLE,
                inventory=_grid(grid),
                axes=(_PHI, _intent("ghost", "column 4", role=AxisRole.OBSERVATION, unit_is_header=True)),
            )


class TestTheModelCannotSmuggleAJoin:
    """Verifier 5b. The model's output ALONE cannot state a tuple the document does
    not print. Structurally there is no per-point/tuple field to state one; and every
    point the resolver emits draws all its values from ONE row -- the join is the
    document's own row, generated by the walk, never proposed."""

    def test_no_proposal_field_can_pair_cells_from_different_rows(self) -> None:
        # There is no list of points, no tuple field, no row index anywhere: the model
        # proposes AXES, and the machine walks rows. A pairing across rows has no home.
        proposal_fields = set(TabularSeriesProposal.model_fields)
        for name in ("points", "point", "tuples", "rows", "pairs", "join"):
            assert name not in proposal_fields
        # `axes` holds ProposedTabularAxis, which is per-column and carries no value.
        assert "values" not in set(ProposedTabularAxis.model_fields)

    def test_every_emitted_point_draws_all_its_values_from_one_row(self) -> None:
        resolved = resolve_tabular_series(table_key=_TABLE, inventory=_grid(_CLEAN), axes=(_PHI, _S))
        for point in resolved.points:
            rows = {value.cell.row for value in point.values}
            assert len(rows) == 1, f"point {point.point_id} spans rows {rows} -- a fabricated join"
            # And that one row is the point's own id, so provenance names it.
            assert point.point_id == f"row_{next(iter(rows))}"
