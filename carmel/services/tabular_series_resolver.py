"""Resolve proposed column-header quotes into per-row, cell-addressed tuples.

This is the bridge the project was missing: the extraction agent speaks in
verbatim QUOTES; the tabular series producer needs cell ADDRESSES joined into
coherent tuples; nothing converted one into the other, so a paper could not be
turned into a series without a human writing every coordinate by hand.

TWO THINGS THE MODEL MUST NEVER ASSERT, AND HOW THIS MODULE DENIES IT BOTH.

* **A cell address.** A model under pressure produces a plausible wrong
  coordinate, and a plausible wrong coordinate that happens to exist replays
  perfectly -- the cell is real, its bytes are real, the hash checks out -- which
  is the single most dangerous artifact this system can emit. So the model is
  given NO way to name one. Its only spatial assertion is a column-HEADER quote
  (:attr:`AxisHeaderIntent.header_quote`), which this module matches, whole-cell
  against whole-cell, to LOCATE a column in the caller-supplied grid. A header
  matching no column is refused; one matching several is refused rather than
  picked. Every ``(row, col)`` in every emitted grounding is computed HERE from
  the grid, never taken from the intent.

* **The join.** A data point is a tuple -- an equivalence ratio AND the flame
  speed measured at it -- and even when every value is a real cell, nothing makes
  those cells belong together. A tuple braiding a value from one row with a value
  from another is built from genuine bytes and states a relationship the paper
  never printed; grounding cannot catch it, because grounding asks only whether
  each value is real. So the model is given NO way to state a tuple either. It
  proposes axes, not points. This module WALKS the grid's rows and emits ONE
  point per data row, taking each axis's value from that row's cell in that axis's
  column -- the join is the document's own row, the one relationship a printed
  table actually asserts. There is no field, anywhere upstream, where a value from
  one row could be paired with a value from another.

Under this shape the model contributes only WHICH table and WHAT the columns are
CALLED. The machine contributes every coordinate and every join. The producer's
own :func:`~carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`
re-checks co-location (``_reject_incoherent_join``) as defence in depth; this
module, by construction, only ever emits same-row points, so it never trips it.

THREE HAZARDS, ANSWERED FROM THE GRID THIS CODEBASE ACTUALLY CARRIES.

1. **Transposed tables.** The grid model records NO orientation:
   :func:`carmel.services.pdf_tables.build_inventory` derives ``row``/``col`` from
   page geometry alone, and ``row`` is a bare ordinal, top-first. So "walk the
   rows, one point per row" is the ONLY orientation the grid supports, and it is
   the orientation of a standard table (one record per printed row). A transposed
   table (one record per COLUMN) is not modelled here and is not this bridge's job
   to detect -- were the caller to supply a transposed grid, the resolver would
   read its columns as records and produce nonsense, which is why detecting
   transposition belongs to table discovery (out of scope) and the honest limit is
   stated rather than papered over.

2. **Rows that are not data.** Walking every row blindly manufactures points from
   furniture -- a header row, a unit row, a sub-heading, a footnote, a blank
   separator. A row is DATA iff every resolved column holds a bare numeral (by the
   producer's OWN predicate,
   :func:`carmel.services.dataset_producer.is_bare_numeral_value`); it is FURNITURE
   iff no resolved column does (skipped); and a row with a numeral in SOME columns
   but not all is REFUSED, naming the row, rather than guessed -- it is neither
   cleanly data nor cleanly furniture, and either guess fabricates or drops a
   datum. Rows at or above the header row are excluded outright.

3. **Repeated header text.** A header quote matching more than one column is
   refused rather than picked (two columns printed with the same label, or a header
   that spans them): picking one would be the model choosing a coordinate through
   the back door, which is exactly what this module denies it. No occurrence
   disambiguation is offered for headers -- any such channel would re-open that
   door -- so the refusal is unconditional, and a genuinely repeated header is a
   table this bridge does not handle rather than one it guesses at.
"""

from __future__ import annotations

from dataclasses import dataclass

from carmel.schemas.datasets import AxisRole, CaptionLabelKey, EmbeddedTableInventory, MemberSheetKey
from carmel.services import units
from carmel.services.condition_set_producer import TableCellGrounding
from carmel.services.dataset_producer import is_bare_numeral_value
from carmel.services.tabular_dataset_producer import (
    TabularAxisSpec,
    TabularPointSpec,
    TabularPointValueSpec,
)

__all__ = [
    "AxisHeaderIntent",
    "ResolvedTabularSeries",
    "TabularSeriesResolutionError",
    "resolve_tabular_series",
]


class TabularSeriesResolutionError(Exception):
    """A proposed set of column headers could not be resolved into honest tuples.

    Distinct from the producer's own errors: this fires BEFORE any grounding,
    while turning the model's header quotes into cell-addressed specs -- a header
    that names no column, one that names several, headers that do not share one
    row, or a row that is neither cleanly data nor cleanly furniture. A refusal
    here is the bridge working: the model proposed something the grid does not
    support, and it is declined rather than laundered into a stored series.
    """


@dataclass(frozen=True, slots=True)
class AxisHeaderIntent:
    """The schema-free input the resolver needs for one axis.

    Deliberately NOT the pydantic proposal type: the resolver is pure logic over a
    grid, unit-testable with hand-built intents, and it holds no vocabulary the
    proposal owns. The carrier
    (:func:`carmel.services.proposal_intake.tabular_series_from_proposal`) maps one
    ``ProposedTabularAxis`` to one of these, converting the 1-based proposal
    occurrence to the grounder's 0-based index at that boundary.

    ``header_quote`` is the axis's column header, verbatim -- the only spatial
    assertion. ``role`` and ``quantity_kind`` are the semantic assertions, recorded
    unverified. The unit is EITHER a prose char-span (``prose_unit_quote`` set,
    ``prose_unit_occurrence`` its 0-based disambiguator) OR the axis's own header
    cell (``unit_is_header`` True, the other two ``None``); the two are mutually
    exclusive and the carrier guarantees it from the proposal's discriminated union.
    """

    axis_id: str
    role: AxisRole
    quantity_kind: units.QuantityKind
    header_quote: str
    prose_unit_quote: str | None
    prose_unit_occurrence: int | None
    unit_is_header: bool


@dataclass(frozen=True, slots=True)
class ResolvedTabularSeries:
    """The producer-ready specs the resolver derived from the grid.

    ``axes`` and ``points`` are exactly what
    :func:`~carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`
    takes. Every cell address in them was computed from the grid, never asserted by
    the model, and every point's values share one row.
    """

    axes: tuple[TabularAxisSpec, ...]
    points: tuple[TabularPointSpec, ...]


def _point_id(row: int) -> str:
    """The id of the point walked from grid row ``row``.

    Derived from the ordinal so the point's own id names the row it came from --
    honest provenance, and unique across data rows because their ordinals are.
    """
    return f"row_{row}"


def _resolve_columns(
    cells: tuple[tuple[int, int, str | None], ...],
    axes: tuple[AxisHeaderIntent, ...],
    table_key: CaptionLabelKey | MemberSheetKey,
) -> tuple[dict[str, int], int]:
    """Locate each axis's column by its header quote and the one shared header row.

    Returns ``(column_by_axis_id, header_row)``. Refuses a header that matches no
    column, one that matches several, two axes resolving to one column, or headers
    that do not share exactly one row.
    """
    column_by_axis: dict[str, int] = {}
    header_rows_by_axis: dict[str, frozenset[int]] = {}
    for axis in axes:
        matches = [(row, col) for row, col, text in cells if text is not None and text == axis.header_quote]
        if not matches:
            raise TabularSeriesResolutionError(
                f"axis {axis.axis_id!r}: header quote {axis.header_quote!r} matches no cell in the grid of "
                f"table {table_key!r} -- refusing to invent a column the printed header does not name"
            )
        matched_cols = {col for _, col in matches}
        if len(matched_cols) > 1:
            raise TabularSeriesResolutionError(
                f"axis {axis.axis_id!r}: header quote {axis.header_quote!r} matches several columns "
                f"{sorted(matched_cols)!r} -- two columns print this header (or one spans them), and picking "
                "a column would be the model choosing a coordinate; refusing rather than pick"
            )
        column_by_axis[axis.axis_id] = next(iter(matched_cols))
        header_rows_by_axis[axis.axis_id] = frozenset(row for row, _ in matches)

    seen_column: dict[int, str] = {}
    for axis in axes:
        col = column_by_axis[axis.axis_id]
        if col in seen_column:
            raise TabularSeriesResolutionError(
                f"axes {seen_column[col]!r} and {axis.axis_id!r} both resolve to column {col} -- two axes "
                "cannot be the same column of the table"
            )
        seen_column[col] = axis.axis_id

    common_rows = frozenset.intersection(*header_rows_by_axis.values())
    if len(common_rows) != 1:
        matched_rows_by_axis = {axis_id: sorted(rows) for axis_id, rows in header_rows_by_axis.items()}
        raise TabularSeriesResolutionError(
            f"the proposed headers do not share exactly one header row (each axis matched {matched_rows_by_axis!r}; "
            f"common rows {sorted(common_rows)!r}) -- refusing to guess which row is the header band"
        )
    return column_by_axis, next(iter(common_rows))


def _walk_data_rows(
    inventory: EmbeddedTableInventory,
    cells: tuple[tuple[int, int, str | None], ...],
    axes: tuple[AxisHeaderIntent, ...],
    column_by_axis: dict[str, int],
    header_row: int,
    table_key: CaptionLabelKey | MemberSheetKey,
) -> tuple[TabularPointSpec, ...]:
    """Emit one point per DATA row below the header, refusing ambiguous rows.

    A row is data iff every resolved column holds a bare numeral, furniture iff
    none does; a row with a numeral in some but not all resolved columns is
    refused. Every point's values share one row -- the join is the row itself.
    """
    all_rows = sorted({row for row, _, _ in cells})
    points: list[TabularPointSpec] = []
    for row in all_rows:
        if row <= header_row:
            continue
        texts = {axis.axis_id: inventory.cell_text(row=row, col=column_by_axis[axis.axis_id]) for axis in axes}
        is_value = {axis_id: text is not None and is_bare_numeral_value(text) for axis_id, text in texts.items()}
        if all(is_value.values()):
            values: list[TabularPointValueSpec] = []
            for axis in axes:
                value_text = texts[axis.axis_id]
                # Guaranteed non-None: is_value is True only for a non-None numeral text.
                assert value_text is not None
                values.append(
                    TabularPointValueSpec(
                        axis_id=axis.axis_id,
                        value_quote=value_text,
                        cell=TableCellGrounding(
                            table_key=table_key,
                            row=row,
                            col=column_by_axis[axis.axis_id],
                            inventory=inventory,
                        ),
                    )
                )
            points.append(TabularPointSpec(point_id=_point_id(row), values=tuple(values)))
        elif not any(is_value.values()):
            continue
        else:
            present = sorted(axis_id for axis_id, ok in is_value.items() if ok)
            absent = sorted(axis_id for axis_id, ok in is_value.items() if not ok)
            raise TabularSeriesResolutionError(
                f"row {row}: axes {present!r} hold a bare-numeral value but axes {absent!r} do not "
                f"({ {axis_id: texts[axis_id] for axis_id in absent}!r}) -- this row is neither cleanly data "
                "(a value in every column) nor cleanly furniture (a value in none), so refusing to guess "
                "whether it is a partial data row or a heading/unit/footnote"
            )
    if not points:
        raise TabularSeriesResolutionError(
            f"no data rows below header row {header_row} -- every row under the header was furniture (no "
            "bare-numeral value in the resolved columns), so there is nothing to walk into a series"
        )
    return tuple(points)


def _axis_spec(
    axis: AxisHeaderIntent,
    header_row: int,
    col: int,
    table_key: CaptionLabelKey | MemberSheetKey,
    inventory: EmbeddedTableInventory,
) -> TabularAxisSpec:
    """Build one axis spec, grounding its label (and header-cell unit) at the grid.

    The label is grounded at the axis's own header cell -- computed here, at
    ``(header_row, col)``, never asserted by the model. A ``unit_is_header`` axis
    grounds its unit at that SAME cell (one dimensionless symbol serving as both);
    otherwise the unit is a prose char-span the producer grounds in running text.
    """
    label_cell = TableCellGrounding(table_key=table_key, row=header_row, col=col, inventory=inventory)
    if axis.unit_is_header:
        return TabularAxisSpec(
            axis_id=axis.axis_id,
            role=axis.role,
            quantity_kind=axis.quantity_kind,
            label_quote=axis.header_quote,
            unit_quote=axis.header_quote,
            label_cell=label_cell,
            unit_cell=label_cell,
        )
    # A prose-unit intent always carries a quote (the carrier guarantees it from the
    # proposal's discriminated union); assert-narrow it for the type checker.
    prose_unit_quote = axis.prose_unit_quote
    assert prose_unit_quote is not None
    return TabularAxisSpec(
        axis_id=axis.axis_id,
        role=axis.role,
        quantity_kind=axis.quantity_kind,
        label_quote=axis.header_quote,
        unit_quote=prose_unit_quote,
        label_cell=label_cell,
        unit_occurrence=axis.prose_unit_occurrence,
    )


def resolve_tabular_series(
    *,
    table_key: CaptionLabelKey | MemberSheetKey,
    inventory: EmbeddedTableInventory,
    axes: tuple[AxisHeaderIntent, ...],
) -> ResolvedTabularSeries:
    """Resolve proposed headers over one grid into producer-ready axes and points.

    ``table_key`` names the table the cells belong to (carried onto every emitted
    grounding); ``inventory`` is the caller-supplied grid -- read, never derived
    here -- and ``axes`` are the model's per-axis intents. Locates each axis's
    column by its header quote, identifies the one header row, walks the data rows,
    and returns one point per data row with every value cell-addressed and every
    join same-row.

    Raises:
        TabularSeriesResolutionError: A header matches no column or several, two
            axes resolve to one column, the headers do not share one row, a row is
            ambiguously part data / part furniture, or no data row exists.
    """
    if len(axes) < 2:
        raise TabularSeriesResolutionError(
            f"a tabular series needs at least two axes (one coordinate, one observation); got {len(axes)}"
        )
    duplicate = _first_duplicate([axis.axis_id for axis in axes])
    if duplicate is not None:
        raise TabularSeriesResolutionError(
            f"duplicate axis_id {duplicate!r}: every proposed axis must name a distinct column"
        )
    cells = inventory.grid_cells()
    column_by_axis, header_row = _resolve_columns(cells, axes, table_key)
    points = _walk_data_rows(inventory, cells, axes, column_by_axis, header_row, table_key)
    axis_specs = tuple(
        _axis_spec(axis, header_row, column_by_axis[axis.axis_id], table_key, inventory) for axis in axes
    )
    return ResolvedTabularSeries(axes=axis_specs, points=points)


def _first_duplicate(ids: list[str]) -> str | None:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            return value
        seen.add(value)
    return None
