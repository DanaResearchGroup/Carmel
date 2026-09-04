# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Decide, per table, whether it carries measured experimental data -- and say why.

The reader lanes (:mod:`carmel.services.pdf_tables`, :mod:`carmel.services.ooxml_tables`,
:mod:`carmel.services.xlsx_tables`, :mod:`carmel.services.member_tables`) turn a document
into byte-replayable table inventories. What none of them does is say which of those tables
holds the numbers a campaign is after. Today an operator names the target table by hand -- a
pinned label, page and box (see :data:`carmel.services.tabular_dataset_target.TARGET_TABLE_KEY`
and ``TARGET_TABLE_FOOTPRINT``). This module is the discriminator that would let the program
say it instead, from the document's own bytes.

**It ends at a verdict.** Nothing here extracts, grounds, stores, or replays; it consumes a
reader lane's inventory and returns a :class:`TableClassification` -- a verdict, the grounds
it rests on, and, where the evidence is thin, the explicit statement of what it would need.
Wiring a discovered answer in place of the hand-pinned key is deliberately NOT done here.

**Why the verdict carries its grounds.** A boolean "table 3 holds data" is unauditable, and
this project refuses ungroundable claims on principle. Every verdict here names the specific
cell, column or caption bytes it read, so a human can overrule it from the report alone.

**What it reads, and what it refuses to read.** It reads the GRID -- the cell texts, their
row/column positions, and (where the lane exposes it) the table's caption. It never reads a
filename, a ZIP member path, a table index, a sheet ordinal, or any hard-coded label: those
are properties of where a table sits, not of what it says, and a discriminator that keyed off
them would give a different answer for the same bytes under a different name. The lane
adapters below drop that metadata by construction, so the classifier cannot see it even by
accident.

**The signals, stated as a position someone can disagree with.** A measured experimental
table is a grid whose columns carry numbers under headers that name a physical quantity and
its unit, usually with a swept independent variable and often an uncertainty. So the positive
evidence is structural and printed: a **unit token** in a header or caption (a parenthesized
``(cm/s)``, ``(K)``), an **uncertainty marker** (``±``), or a **monotone numeric sweep** down
a column (an independent variable stepped through its range). The negative evidence is equally
structural: **reaction row keys** (``H + O2 <=> OH + O``) mark a mechanism listing, whose
Arrhenius columns are numbers but not measurements; **formula-backed cells** (a spreadsheet
``=A2*B2``) mark a computed sheet, not a measured one; a grid with **no numeric value column**
is a prose list, not data. When a grid is numeric but prints none of the positive signals and
none of the negative ones -- a bare coefficient dump with no units -- the honest verdict is
:attr:`DataVerdict.UNDECIDED`, not a guess.

The signal set is deliberately a vocabulary of STRUCTURE (parentheses, ``±``, monotonicity,
reaction arrows, formulas), not a whitelist of quantity names. A whitelist of "good" header
words would be the hard-coded label this module refuses on principle, and would silently fail
on every quantity nobody listed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "DataVerdict",
    "Ground",
    "Polarity",
    "SignalKind",
    "TableClassification",
    "TableView",
    "classify_table",
    "table_view_from_member_payload",
    "table_view_from_ooxml_payload",
    "table_view_from_pdf_payload",
    "table_view_from_xlsx_payload",
]


class DataVerdict(StrEnum):
    """What a table was found to be.

    Three outcomes, not two: forcing a binary choice on thin evidence produces confident
    nonsense, and this project would rather refuse. :attr:`UNDECIDED` is a first-class verdict
    a caller can act on -- it carries, in :attr:`TableClassification.needed`, exactly what
    would settle it.
    """

    MEASURED = "measured"
    """The table carries measured experimental data, on printed structural evidence."""

    NOT_MEASURED = "not_measured"
    """The table is something else -- a mechanism listing, a computed sheet, a prose list --
    on printed structural evidence that is incompatible with measured data."""

    UNDECIDED = "undecided"
    """The table is a numeric grid, but prints neither the positive signals of measured data
    nor the negative signals of a non-measured table. Deciding it would need what
    :attr:`TableClassification.needed` names."""


class Polarity(StrEnum):
    """Which verdict a single ground pulls toward. Reported per-ground so a human reading the
    grounds can see the case for and against, not just the net result."""

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    NEUTRAL = "neutral"


class SignalKind(StrEnum):
    """The named structural signals a verdict can rest on. Each is a property of the
    document's printed bytes, never of where the table sits."""

    UNIT_TOKEN_IN_HEADER = "unit_token_in_header"
    """A header cell prints a parenthesized unit token, e.g. ``Uncertainty ±(cm/s)``."""

    UNIT_TOKEN_IN_CAPTION = "unit_token_in_caption"
    """The table's caption prints a parenthesized unit token (only the PDF lane exposes one)."""

    UNCERTAINTY_MARKER = "uncertainty_marker"
    """A header cell prints ``±`` -- an explicit measurement uncertainty, which model inputs
    and fitted coefficients do not carry."""

    MONOTONE_NUMERIC_SWEEP = "monotone_numeric_sweep"
    """A value column steps monotonically through a range across the data rows -- the
    signature of a controlled independent variable."""

    NUMERIC_VALUE_COLUMN = "numeric_value_column"
    """At least one column is predominantly numeric over the data rows. Necessary for
    measured data, but on its own not sufficient -- coefficients are numeric too."""

    REACTION_ROW_KEYS = "reaction_row_keys"
    """A non-value column's cells are chemical reactions (``A + B <=> C``). The table is a
    mechanism listing; its numeric columns are Arrhenius parameters, not measurements."""

    FORMULA_CELLS = "formula_cells"
    """A value column is predominantly backed by spreadsheet formulas. The numbers are
    computed, not measured and recorded."""

    NO_NUMERIC_VALUE_COLUMN = "no_numeric_value_column"
    """No column is predominantly numeric: a prose list or a text index, not data."""

    TOO_FEW_DATA_ROWS = "too_few_data_rows"
    """Fewer than two rows: not a header-plus-data grid, so not a dataset."""

    NUMERIC_GRID_WITHOUT_QUANTITY = "numeric_grid_without_quantity"
    """A numeric grid printing no unit token, no ``±``, no monotone sweep, and no reaction
    keys or formulas -- indistinguishable, from structure alone, between measured data and a
    bare coefficient dump. The driver of an :attr:`DataVerdict.UNDECIDED` verdict."""


@dataclass(frozen=True)
class Ground:
    """One piece of evidence a verdict rests on: which signal fired, which way it pulls, and
    the exact bytes it read. ``detail`` quotes the cell, column or caption text so an auditor
    can find it in the document."""

    kind: SignalKind
    polarity: Polarity
    detail: str


@dataclass(frozen=True)
class TableClassification:
    """A verdict for one table, with everything needed to overrule it from the report alone."""

    verdict: DataVerdict
    grounds: tuple[Ground, ...]
    needed: tuple[str, ...] = ()
    """For :attr:`DataVerdict.UNDECIDED`, what would settle the question. Empty otherwise."""


@dataclass(frozen=True)
class TableView:
    """A lane-agnostic view of one table's grid: the ONLY thing the classifier sees.

    Deliberately carries no filename, member path, table index or sheet ordinal -- the lane
    adapters drop that metadata so the classifier cannot key off it. ``cells`` is a sparse
    ``(row, col, text)`` tuple, exactly as the lanes store it; empty grid positions are simply
    absent. ``caption`` is present only for the PDF lane (the others expose none).
    ``formula_cells`` names the ``(row, col)`` positions the source marked as formula-backed
    -- populated only by the ``.xlsx`` adapter, empty everywhere else.
    """

    row_count: int
    col_count: int
    cells: tuple[tuple[int, int, str], ...]
    caption: str | None = None
    formula_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset)


# --- lane adapters: reader payload -> TableView, dropping all positional metadata ----------


def _sparse_cells(payload: Mapping[str, Any], *, text_key: str) -> tuple[tuple[int, int, str], ...]:
    cells = payload.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, str | bytes):
        raise ValueError("inventory payload has no 'cells' sequence")
    out: list[tuple[int, int, str]] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError(f"cell is not a mapping: {cell!r}")
        out.append((int(cell["row"]), int(cell["col"]), str(cell.get(text_key) or "")))
    return tuple(out)


def _dims(cells: tuple[tuple[int, int, str], ...]) -> tuple[int, int]:
    if not cells:
        return 0, 0
    return max(r for r, _, _ in cells) + 1, max(c for _, c, _ in cells) + 1


def table_view_from_ooxml_payload(payload: Mapping[str, Any]) -> TableView:
    """A :class:`TableView` of a ``.docx`` table inventory. Reads only the grid: ``part_name``
    and ``table_index`` are positional identity and are deliberately not read."""
    cells = _sparse_cells(payload, text_key="text")
    return TableView(row_count=int(payload["row_count"]), col_count=int(payload["col_count"]), cells=cells)


def table_view_from_member_payload(payload: Mapping[str, Any]) -> TableView:
    """A :class:`TableView` of a delimited-text member inventory. The member's path/name is a
    filename and is deliberately not read; the grid dimensions are derived from the cells."""
    cells = _sparse_cells(payload, text_key="text")
    rows, cols = _dims(cells)
    return TableView(row_count=rows, col_count=cols, cells=cells)


def table_view_from_xlsx_payload(payload: Mapping[str, Any]) -> TableView:
    """A :class:`TableView` of an ``.xlsx`` sheet inventory. The sheet name and ordinal are
    positional identity and are not read; a cell's ``formula`` IS read (it is printed content,
    and a formula-backed value is computed, not measured) and carried as
    :attr:`TableView.formula_cells`."""
    raw = payload.get("cells")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("xlsx inventory payload has no 'cells' sequence")
    cells: list[tuple[int, int, str]] = []
    formulas: set[tuple[int, int]] = set()
    for cell in raw:
        if not isinstance(cell, Mapping):
            raise ValueError(f"cell is not a mapping: {cell!r}")
        rc = (int(cell["row"]), int(cell["col"]))
        cells.append((rc[0], rc[1], str(cell.get("value") or "")))
        if cell.get("formula"):
            formulas.add(rc)
    return TableView(
        row_count=int(payload["row_count"]),
        col_count=int(payload["col_count"]),
        cells=tuple(cells),
        formula_cells=frozenset(formulas),
    )


def table_view_from_pdf_payload(payload: Mapping[str, Any]) -> TableView:
    """A :class:`TableView` of a PDF cell inventory. The caption IS read -- it is printed
    content and the richest quantity signal the corpus offers -- but the page and footprint
    coordinates are positional and are not read as identity."""
    cells = _sparse_cells(payload, text_key="text")
    rows, cols = _dims(cells)
    caption = None
    footprint = payload.get("footprint")
    if isinstance(footprint, Mapping):
        raw = footprint.get("caption_text")
        caption = str(raw) if raw else None
    return TableView(row_count=rows, col_count=cols, cells=cells, caption=caption)


# --- the discriminator ---------------------------------------------------------------------

#: A number as a paper prints one: optional sign, digits with optional decimal, optional
#: exponent. Local and stable on purpose -- it is NOT
#: :func:`carmel.services.numeric.normalize_numeric_span`, the extraction grammar, because
#: coupling classification to what extraction accepts would let a change to extraction move
#: every verdict. Classification asks only "is this cell a bare number", which this answers.
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

#: A parenthesized token that contains a letter: a unit like ``(cm/s)`` or ``(K)``. A
#: parenthesized pure number (``(1)``, a footnote mark) is not a unit and does not match.
_UNIT_TOKEN = re.compile(r"\(([^()]*[A-Za-z][^()]*)\)")

#: Reaction arrows, the unambiguous mark of a mechanism row. ``=`` alone is not enough (a
#: header may read ``x=1``); an arrow, or an ``=`` beside a ``+``, is.
_REACTION_ARROW = re.compile(r"<=>|=>|<=|⇌|⟶|→")

#: Fraction of a column's non-empty data cells that must be numeric for it to count as a
#: value column, and that must be reactions for a key column to read as a mechanism. High
#: enough that a stray numeric footnote or one malformed row does not flip a column.
_COLUMN_MAJORITY = 0.6


def _looks_numeric(text: str) -> bool:
    return bool(_NUMBER.match(text.strip()))


def _looks_reaction(text: str) -> bool:
    t = text.strip()
    if _REACTION_ARROW.search(t):
        return True
    return "=" in t and "+" in t


def _dense_grid(view: TableView) -> list[list[str]]:
    grid = [["" for _ in range(view.col_count)] for _ in range(view.row_count)]
    for row, col, text in view.cells:
        if 0 <= row < view.row_count and 0 <= col < view.col_count:
            grid[row][col] = text
    return grid


def _numeric_columns(data_rows: list[list[str]], col_count: int) -> set[int]:
    value_cols: set[int] = set()
    for col in range(col_count):
        column = [row[col].strip() for row in data_rows if col < len(row) and row[col].strip()]
        if len(column) < 2:
            continue
        numeric = sum(1 for cell in column if _looks_numeric(cell))
        if numeric / len(column) >= _COLUMN_MAJORITY and numeric >= 2:
            value_cols.add(col)
    return value_cols


def _monotone_sweep_ground(data_rows: list[list[str]], value_cols: set[int]) -> Ground | None:
    for col in sorted(value_cols):
        values = [float(row[col]) for row in data_rows if col < len(row) and _looks_numeric(row[col].strip())]
        if len(values) < 3:
            continue
        increasing = all(b > a for a, b in zip(values, values[1:], strict=False))
        decreasing = all(b < a for a, b in zip(values, values[1:], strict=False))
        if increasing or decreasing:
            direction = "increasing" if increasing else "decreasing"
            return Ground(
                SignalKind.MONOTONE_NUMERIC_SWEEP,
                Polarity.MEASURED,
                f"column {col} steps {direction}: {values[0]} .. {values[-1]} over {len(values)} rows",
            )
    return None


def _reaction_key_ground(data_rows: list[list[str]], value_cols: set[int], col_count: int) -> Ground | None:
    for col in range(col_count):
        if col in value_cols:
            continue
        column = [row[col].strip() for row in data_rows if col < len(row) and row[col].strip()]
        if len(column) < 2:
            continue
        reactions = sum(1 for cell in column if _looks_reaction(cell))
        if reactions / len(column) >= _COLUMN_MAJORITY and reactions >= 2:
            example = next(cell for cell in column if _looks_reaction(cell))
            return Ground(
                SignalKind.REACTION_ROW_KEYS,
                Polarity.NOT_MEASURED,
                f"column {col} holds reactions, e.g. {example!r}",
            )
    return None


def _formula_ground(view: TableView, value_cols: set[int], data_rows: list[list[str]]) -> Ground | None:
    # Data rows begin at grid row 1 (row 0 is the header), so a data row's grid row is its
    # index in `data_rows` plus 1 -- the offset that lines `formula_cells` up with the grid.
    for col in sorted(value_cols):
        backed = 0
        total = 0
        for data_index, row in enumerate(data_rows):
            if col < len(row) and row[col].strip():
                total += 1
                if (data_index + 1, col) in view.formula_cells:
                    backed += 1
        if total >= 2 and backed / total >= _COLUMN_MAJORITY:
            return Ground(
                SignalKind.FORMULA_CELLS,
                Polarity.NOT_MEASURED,
                f"column {col}: {backed}/{total} value cells are spreadsheet formulas",
            )
    return None


def _unit_and_uncertainty_grounds(header: list[str], caption: str | None) -> list[Ground]:
    grounds: list[Ground] = []
    for cell in header:
        match = _UNIT_TOKEN.search(cell)
        if match:
            grounds.append(
                Ground(
                    SignalKind.UNIT_TOKEN_IN_HEADER,
                    Polarity.MEASURED,
                    f"header cell {cell!r} prints unit {match.group(0)!r}",
                )
            )
            break
    if caption:
        match = _UNIT_TOKEN.search(caption)
        if match:
            grounds.append(
                Ground(SignalKind.UNIT_TOKEN_IN_CAPTION, Polarity.MEASURED, f"caption prints unit {match.group(0)!r}")
            )
    for cell in header:
        if "±" in cell:
            grounds.append(Ground(SignalKind.UNCERTAINTY_MARKER, Polarity.MEASURED, f"header cell {cell!r} prints ±"))
            break
    return grounds


def classify_table(view: TableView) -> TableClassification:
    """Classify one table's grid as measured experimental data, not, or undecided.

    Reads only ``view`` -- the grid, and the caption where the lane exposed one. The decision
    is ordered so the strongest structural facts settle it first: a table with too few rows or
    no numeric column cannot be data; a mechanism listing or a computed sheet is not data
    however numeric; a grid printing units, ``±`` or a monotone sweep IS data; and a numeric
    grid printing none of those is honestly undecided rather than guessed.
    """
    grid = _dense_grid(view)
    if view.row_count < 2 or not grid:
        return TableClassification(
            DataVerdict.NOT_MEASURED,
            (
                Ground(
                    SignalKind.TOO_FEW_DATA_ROWS,
                    Polarity.NOT_MEASURED,
                    f"{view.row_count} row(s); a dataset needs a header and data",
                ),
            ),
        )

    header = grid[0]
    data_rows = grid[1:]
    value_cols = _numeric_columns(data_rows, view.col_count)

    if not value_cols:
        return TableClassification(
            DataVerdict.NOT_MEASURED,
            (
                Ground(
                    SignalKind.NO_NUMERIC_VALUE_COLUMN,
                    Polarity.NOT_MEASURED,
                    "no column is predominantly numeric over the data rows",
                ),
            ),
        )

    grounds: list[Ground] = [
        Ground(
            SignalKind.NUMERIC_VALUE_COLUMN, Polarity.NEUTRAL, f"columns {sorted(value_cols)} are predominantly numeric"
        )
    ]

    reaction = _reaction_key_ground(data_rows, value_cols, view.col_count)
    if reaction is not None:
        grounds.append(reaction)
        return TableClassification(DataVerdict.NOT_MEASURED, tuple(grounds))

    formula = _formula_ground(view, value_cols, data_rows)
    if formula is not None:
        grounds.append(formula)
        return TableClassification(DataVerdict.NOT_MEASURED, tuple(grounds))

    positives = _unit_and_uncertainty_grounds(header, view.caption)
    sweep = _monotone_sweep_ground(data_rows, value_cols)
    if sweep is not None:
        positives.append(sweep)

    if positives:
        grounds.extend(positives)
        return TableClassification(DataVerdict.MEASURED, tuple(grounds))

    grounds.append(
        Ground(
            SignalKind.NUMERIC_GRID_WITHOUT_QUANTITY,
            Polarity.NEUTRAL,
            "numeric grid printing no unit token, no ±, no monotone sweep, no reaction keys and no formulas",
        )
    )
    return TableClassification(
        DataVerdict.UNDECIDED,
        tuple(grounds),
        needed=(
            "a header or caption naming a physical quantity and its unit (a parenthesized unit token)",
            "an uncertainty column (±), or a column that steps monotonically through a swept range",
        ),
    )
