# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-table measured-data discriminator.

Every fixture here is synthetic. The shapes are modeled on what the I-059 probe actually
found in the staged corpus -- the ``.docx`` supplement's laminar-burning-velocity tables (a
measured sweep with a ``(cm/s)`` unit and a ``±`` uncertainty column), and the ``.zip``
supplement's ``TUe_mechanism.dat`` (a mechanism listing: a reaction string per row beside
Arrhenius A/n/Ea columns) -- but no test reads the corpus. The near-miss is the mechanism
listing: numeric columns that superficially resemble rate data and are not.
"""

from __future__ import annotations

import pytest

from carmel.services.table_data_discriminator import (
    DataVerdict,
    Polarity,
    SignalKind,
    TableView,
    classify_table,
    table_view_from_member_payload,
    table_view_from_ooxml_payload,
    table_view_from_pdf_payload,
    table_view_from_xlsx_payload,
)


def _view(
    rows: list[list[str]], *, caption: str | None = None, formula_cells: set[tuple[int, int]] | None = None
) -> TableView:
    """A TableView from a dense list-of-rows, exactly as a lane's grid would densify."""
    cells = tuple((r, c, text) for r, row in enumerate(rows) for c, text in enumerate(row))
    col_count = max((len(row) for row in rows), default=0)
    return TableView(
        row_count=len(rows),
        col_count=col_count,
        cells=cells,
        caption=caption,
        formula_cells=frozenset(formula_cells or set()),
    )


# --- measured (positive) -------------------------------------------------------------------


def test_measured_with_unit_and_uncertainty() -> None:
    """The clearly-measured case: a header naming a quantity with a ``(cm/s)`` unit and a ``±``
    uncertainty column, modeled on the .docx supplement's tables."""
    view = _view(
        [
            ["Equivalence ratio", "Laminar burning velocity (cm/s)", "Uncertainty ±(cm/s)"],
            ["0.40", "9.08", "1.84"],
            ["0.45", "13.73", "1.13"],
            ["0.50", "19.59", "0.96"],
        ]
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.MEASURED
    kinds = {g.kind for g in result.grounds}
    assert SignalKind.UNIT_TOKEN_IN_HEADER in kinds
    assert SignalKind.UNCERTAINTY_MARKER in kinds
    assert result.needed == ()


def test_measured_via_monotone_sweep_and_caption_unit() -> None:
    """A PDF-shaped table whose column header names no quantity (a symbol-font ``/`` for phi),
    carried by a caption unit token and a monotone swept independent variable -- the pinned
    10.1115 flame-speed table's situation."""
    view = _view(
        [
            ["/", "S0L;u(cm/s)"],
            ["0.5", "67.2"],
            ["0.6", "96.5"],
            ["0.7", "124.4"],
            ["0.8", "169.9"],
        ],
        caption="Laminar flame speeds over a range of equivalence ratios (cm/s)",
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.MEASURED
    kinds = {g.kind for g in result.grounds}
    assert SignalKind.MONOTONE_NUMERIC_SWEEP in kinds
    assert SignalKind.UNIT_TOKEN_IN_CAPTION in kinds


# --- not measured (clear negatives) --------------------------------------------------------


def test_not_measured_prose_list_has_no_numeric_column() -> None:
    """A prior-studies list: authors, method, year-as-text -- no predominantly numeric column."""
    view = _view(
        [
            ["Study", "Fuel", "Method"],
            ["Smith et al.", "hydrogen", "heat flux"],
            ["Jones et al.", "syngas", "bunsen"],
            ["Lee et al.", "methane", "counterflow"],
        ]
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.NOT_MEASURED
    assert {g.kind for g in result.grounds} == {SignalKind.NO_NUMERIC_VALUE_COLUMN}


def test_not_measured_too_few_rows() -> None:
    view = _view([["Equivalence ratio", "Velocity (cm/s)"]])
    result = classify_table(view)
    assert result.verdict is DataVerdict.NOT_MEASURED
    assert result.grounds[0].kind is SignalKind.TOO_FEW_DATA_ROWS


# --- near-miss (from the probe): a mechanism listing ---------------------------------------


def test_near_miss_mechanism_listing_is_not_measured() -> None:
    """The near-miss the probe actually found: ``TUe_mechanism.dat``. Reaction row keys beside
    Arrhenius A/n/Ea columns -- numeric columns that look like rate data but are model
    parameters. A discriminator that never saw this would call it measured."""
    view = _view(
        [
            ["Reaction", "A", "n", "Ea"],
            ["H+O2=OH+O", "1.04E+14", "0.0", "15286.0"],
            ["H2+OH=H2O+H", "2.140E+08", "1.52", "3450.0"],
            ["OH+OH=H2O+O", "3.34E+04", "2.42", "-1930.0"],
            ["HO2+O=OH+O2", "1.630E+13", "0.0", "-445.0"],
        ]
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.NOT_MEASURED
    reaction_grounds = [g for g in result.grounds if g.kind is SignalKind.REACTION_ROW_KEYS]
    assert reaction_grounds and reaction_grounds[0].polarity is Polarity.NOT_MEASURED
    assert "H+O2=OH+O" in reaction_grounds[0].detail


def test_near_miss_falloff_reaction_keys_still_detected() -> None:
    """A falloff reaction key (``H+O2(+M)=HO2(+M)``) carries parenthesized third bodies; it is
    still a reaction, and the parenthesis must not be mistaken for a unit token."""
    view = _view(
        [
            ["Reaction", "A", "n", "Ea"],
            ["H+O2(+M)=HO2(+M)", "4.660E+12", "0.44", "0.0"],
            ["OH+OH(+M)=H2O2(+M)", "1E+14", "-0.37", "0.0"],
            ["H2O+M=H+OH+M", "6.06E+27", "-3.312", "120770.0"],
        ]
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.NOT_MEASURED
    assert any(g.kind is SignalKind.REACTION_ROW_KEYS for g in result.grounds)


def test_near_miss_formula_sheet_is_not_measured() -> None:
    """A computed .xlsx sheet: value cells are spreadsheet formulas, so the numbers are
    derived, not measured."""
    # As the .xlsx lane stores it: a formula cell carries the cached numeric ``value`` AND the
    # ``formula`` text; the adapter reads the value as the cell text and flags the position.
    view = _view(
        [
            ["x", "y"],
            ["1", "2"],
            ["2", "4"],
            ["3", "6"],
        ],
        formula_cells={(1, 1), (2, 1), (3, 1)},
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.NOT_MEASURED
    formula_grounds = [g for g in result.grounds if g.kind is SignalKind.FORMULA_CELLS]
    assert formula_grounds and "formulas" in formula_grounds[0].detail


# --- undecided (first-class) ---------------------------------------------------------------


def test_undecided_bare_coefficient_grid_states_what_it_would_need() -> None:
    """A bare coefficient dump modeled on ``thermo.dat``: species-name row keys and numeric
    coefficient columns, no unit token, no ±, no monotone sweep, no reaction arrows. From
    structure alone this is indistinguishable between measured data and fitted coefficients,
    so the honest verdict is undecided -- and it must say what would settle it."""
    view = _view(
        [
            ["Species", "a1", "a2", "a3"],
            ["H2", "2.34433", "0.00798", "-0.00001"],
            ["H2O", "4.19864", "-0.00203", "0.00006"],
            ["O2", "3.78246", "-0.00299", "0.00001"],
            ["OH", "3.99201", "-0.00240", "0.00005"],
        ]
    )
    result = classify_table(view)
    assert result.verdict is DataVerdict.UNDECIDED
    assert any(g.kind is SignalKind.NUMERIC_GRID_WITHOUT_QUANTITY for g in result.grounds)
    assert result.needed
    assert any("unit" in n for n in result.needed)


# --- filename / index independence ---------------------------------------------------------


def _ooxml_payload(*, table_index: int, part_name: str) -> dict[str, object]:
    return {
        "cells": [
            {"row": 0, "col": 0, "text": "phi", "col_span": 1, "row_merge": None},
            {"row": 0, "col": 1, "text": "S (cm/s)", "col_span": 1, "row_merge": None},
            {"row": 1, "col": 0, "text": "0.5", "col_span": 1, "row_merge": None},
            {"row": 1, "col": 1, "text": "67.2", "col_span": 1, "row_merge": None},
            {"row": 2, "col": 0, "text": "0.6", "col_span": 1, "row_merge": None},
            {"row": 2, "col": 1, "text": "96.5", "col_span": 1, "row_merge": None},
        ],
        "col_count": 2,
        "part_name": part_name,
        "payload_version": 1,
        "row_count": 3,
        "source_sha256": "0" * 64,
        "table_index": table_index,
    }


def test_verdict_independent_of_table_index_and_part_name() -> None:
    """The same grid bytes, at a different table index and in a differently-named part, must
    produce the same verdict AND the same grounds -- the guard against a discriminator that has
    quietly learned position or filename instead of reading the document."""
    first = classify_table(table_view_from_ooxml_payload(_ooxml_payload(table_index=0, part_name="word/document.xml")))
    moved = classify_table(table_view_from_ooxml_payload(_ooxml_payload(table_index=99, part_name="word/other.xml")))
    assert first == moved
    assert first.verdict is DataVerdict.MEASURED


def test_member_verdict_independent_of_member_path() -> None:
    """Two delimited members with the same grid but different names classify identically: the
    member path is a filename and must never enter the verdict."""

    def payload() -> dict[str, object]:
        return {
            "cells": [
                {"row": 0, "col": 0, "text": "T (K)"},
                {"row": 0, "col": 1, "text": "rate"},
                {"row": 1, "col": 0, "text": "300"},
                {"row": 1, "col": 1, "text": "1.2"},
                {"row": 2, "col": 0, "text": "400"},
                {"row": 2, "col": 1, "text": "3.4"},
            ],
        }

    # The adapter takes no name argument at all; a caller cannot smuggle one in.
    a = classify_table(table_view_from_member_payload(payload()))
    b = classify_table(table_view_from_member_payload(payload()))
    assert a == b


# --- lane adapters -------------------------------------------------------------------------


def test_ooxml_adapter_reads_grid_and_drops_positional_metadata() -> None:
    view = table_view_from_ooxml_payload(_ooxml_payload(table_index=7, part_name="word/document.xml"))
    assert view.row_count == 3
    assert view.col_count == 2
    assert (0, 1, "S (cm/s)") in view.cells
    assert view.caption is None
    assert view.formula_cells == frozenset()


def test_xlsx_adapter_reads_values_and_captures_formulas() -> None:
    payload = {
        "cells": [
            {"row": 0, "col": 0, "value": "x", "formula": None, "cell_type": "s", "col_span": 1, "row_span": 1},
            {"row": 1, "col": 0, "value": "1", "formula": None, "cell_type": "n", "col_span": 1, "row_span": 1},
            {"row": 1, "col": 1, "value": "2", "formula": "=A2*2", "cell_type": "n", "col_span": 1, "row_span": 1},
        ],
        "col_count": 2,
        "part_name": "xl/worksheets/sheet1.xml",
        "payload_version": 1,
        "row_count": 2,
        "sheet_index": 0,
        "sheet_name": "Sheet1",
        "source_sha256": "0" * 64,
    }
    view = table_view_from_xlsx_payload(payload)
    assert (1, 1, "2") in view.cells
    assert view.formula_cells == frozenset({(1, 1)})


def test_pdf_adapter_reads_caption_and_derives_dims() -> None:
    payload = {
        "cells": [
            {"row": 0, "col": 0, "text": "phi"},
            {"row": 1, "col": 0, "text": "0.5"},
            {"row": 1, "col": 1, "text": "67.2"},
        ],
        "footprint": {"caption_text": "Flame speeds (cm/s)", "page": 4},
    }
    view = table_view_from_pdf_payload(payload)
    assert view.row_count == 2
    assert view.col_count == 2
    assert view.caption == "Flame speeds (cm/s)"


def test_pdf_adapter_tolerates_absent_caption() -> None:
    payload = {"cells": [{"row": 0, "col": 0, "text": "x"}], "footprint": {"caption_text": "", "page": 1}}
    view = table_view_from_pdf_payload(payload)
    assert view.caption is None


def test_adapter_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="cells"):
        table_view_from_member_payload({"not_cells": []})
    with pytest.raises(ValueError, match="cells"):
        table_view_from_member_payload({"cells": "oops"})


# --- edge cases ----------------------------------------------------------------------------


def test_empty_grid_is_not_measured() -> None:
    result = classify_table(TableView(row_count=0, col_count=0, cells=()))
    assert result.verdict is DataVerdict.NOT_MEASURED


def test_sparse_cells_densify_without_error() -> None:
    """A cell missing from the sparse list is a blank grid position, not a crash."""
    view = TableView(
        row_count=3,
        col_count=2,
        cells=((0, 0, "phi"), (0, 1, "S (cm/s)"), (1, 0, "0.5"), (2, 0, "0.6"), (2, 1, "96.5")),
    )
    result = classify_table(view)
    # Column 1 has only one numeric data cell, so it is not a value column; column 0 has two.
    assert result.verdict in {DataVerdict.MEASURED, DataVerdict.UNDECIDED}
