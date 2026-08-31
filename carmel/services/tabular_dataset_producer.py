"""Produce a validated :class:`DatasetEnvelope` whose series come from a TABLE.

WHY THIS MODULE EXISTS, AND WHAT UNBLOCKED IT

:func:`carmel.services.dataset_producer.produce_envelope_from_artifact` refuses
every call, and its assembly body was deleted, on one explicit condition: no
producer could construct a :class:`DatasetEnvelope` until something could emit a
``TABLE_CELL`` locator (a table parser) or a ``FIGURE_CROP`` node (a figure
digitizer). That condition is now met. The table parser exists
(:func:`carmel.services.pdf_tables.build_inventory` derives a cell grid from
page geometry, :func:`carmel.services.pdf_table_record.verify_inventory_record`
re-derives it from the raw bytes), and ``TableCellLocator`` is already emitted in
production by :mod:`carmel.services.condition_set_producer`. This module is the
same move applied to OBSERVABLES instead of conditions: it assembles a
``source_form=TABULAR`` :class:`Series` whose every data-point value is located
at a named table cell.

It is NOT the deleted char-span body resurrected. A char span into extracted
running text cannot ground a series data point -- a series asserts a structured
pairing of coordinates to observations, and running text carries no row
structure to prove it (see
:meth:`DatasetEnvelope._validate_no_char_span_grounds_a_series_value`, V7). That
route stays closed; this one is the route the parser opened.

WHAT IS GROUNDED WHERE

* A data point's VALUE (a :class:`Coordinate` or :class:`Observation` number) is
  ALWAYS located at a table cell -- V4 requires exactly this for a ``TABULAR``
  series, and it is the whole point. The cited cell's whole text must equal the
  value quote exactly (the settled matching contract, enforced batch-wise by
  :class:`~carmel.services.condition_set_producer._CellCiter`).
* A VALUE's unit and an AXIS's header label may each be located EITHER at a
  table cell (when the source prints them in the grid) OR as a char span into
  running prose (V4/V7 constrain only the value ref, never the unit or label).
  A ``TABULAR`` series legitimately takes its unit spelling from prose -- "cm/s"
  is written in the text even when the column header carries it in a glyph the
  extractor mangles.

GROUNDING PROVES LOCATION, NEVER MEANING. Every ``SourceRef`` this producer emits
is an exact, located substring of the authenticated document (a cell of a
re-derivable grid, or a char span of the verified text). NOTHING here proves that
a column the caller labelled an equivalence ratio IS one, that a quantity_kind is
right, or that a coordinate and the observation beside it truly share a row. The
schema records the caller's assertion; replay proves the citations resolve.

The authentication preamble and the numeric/unit normalization are NOT
reimplemented: :func:`~carmel.services.dataset_producer._prepare_grounding` and
:func:`~carmel.services.dataset_producer._measured_value` are shared with the two
existing producers, so a fix to one fail-closed path cannot silently miss this
one. The cell-citation authority
(:class:`~carmel.services.condition_set_producer._CellCiter`) and the
grounding input (:class:`~carmel.services.condition_set_producer.TableCellGrounding`)
are shared with the condition-set producer for the same reason: one
implementation of "is this cell citation honest", used by every producer that
emits one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    Composition,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    Maybe,
    Observation,
    Series,
    SourceForm,
    SourceRef,
    UnitProvenance,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.condition_set_producer import (
    TableCellGrounding,
    _cell_locator,
    _CellCiter,
)
from carmel.services.dataset_producer import (
    _ACTIVE,
    _ROOT_NODE_ID,
    DatasetProducerError,
    _measured_value,
    _prepare_grounding,
    ground_quote,
)
from carmel.services.numeric import QuoteRole
from carmel.services.pdf_fragments import GlyphRepair

__all__ = [
    "TabularAxisSpec",
    "TabularDatasetProducerError",
    "TabularPointSpec",
    "TabularPointValueSpec",
    "produce_tabular_envelope_from_artifact",
]

#: The default ``composition`` for a series whose composition was not extracted.
#: A module-level singleton because :class:`Absent` is frozen and immutable, so
#: sharing one instance is safe -- and a call in an argument default is not (B008).
_COMPOSITION_NOT_EXTRACTED: Maybe[Composition] = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)


class TabularDatasetProducerError(DatasetProducerError):
    """A tabular dataset envelope could not be honestly produced.

    Subclasses :class:`DatasetProducerError` deliberately: a caller that already
    fails closed on "a dataset producer refused" keeps working unchanged, while a
    caller that wants to tell the tabular producer apart still can.
    """


def _require_int_occurrences(owner: str, **occurrences: int | None) -> None:
    """Reject a non-int occurrence, ``bool`` included.

    Mirrors the identical guard on the other producers' specs: ``bool`` is an
    ``int`` subclass, so a bare ``isinstance(x, int)`` would silently read
    ``True``/``False`` as occurrence 1/0 -- a caller typo, never disambiguation
    intent.
    """
    for name, value in occurrences.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TabularDatasetProducerError(
                f"{owner}.{name}={value!r} must be an int or None, not {type(value).__name__} "
                "-- bool is a subclass of int in Python and would silently mean occurrence 0/1"
            )


def _reject_cell_with_occurrence(owner: str, **quotes: tuple[int | None, TableCellGrounding | None]) -> None:
    """Refuse a quote that is BOTH text-disambiguated and cell-grounded.

    An ``occurrence`` disambiguates a substring SEARCH of running text; a
    :class:`TableCellGrounding` says the quote is at a named grid cell instead.
    Supplying both is a contradiction, and silently preferring one would ground
    the quote somewhere the caller did not unambiguously ask for.
    """
    for name, (occurrence, cell) in quotes.items():
        if occurrence is not None and cell is not None:
            raise TabularDatasetProducerError(
                f"{owner}.{name}: an occurrence ({occurrence!r}) disambiguates a running-text search "
                "while a TableCellGrounding names a table cell -- a quote cannot be grounded both ways, "
                "so supply exactly one"
            )


@dataclass(frozen=True, slots=True)
class TabularAxisSpec:
    """One declared axis of the series: its role, physical quantity, printed
    header label, and the unit its values are measured in.

    ``label_quote`` grounds the axis HEADER. ``unit_quote`` grounds the unit
    every value on this axis is measured in -- carried on the AXIS rather than
    per value because one column has one unit; the producer re-uses it for each
    of the axis's points. Either may be located at a table cell (``label_cell`` /
    ``unit_cell``) or, when omitted, searched in running text with the paired
    ``*_occurrence`` disambiguating a repeat. A value's own number is NEVER a
    char span -- that is the point spec's job and it is always a cell.
    """

    axis_id: str
    role: AxisRole
    quantity_kind: units.QuantityKind
    label_quote: str
    unit_quote: str = ""
    label_occurrence: int | None = None
    unit_occurrence: int | None = None
    label_cell: TableCellGrounding | None = None
    unit_cell: TableCellGrounding | None = None
    unit_not_printed: bool = False
    """Declare this axis's dimensionless unit WITHOUT a printed token (I-060).
    Permitted for EQUIVALENCE_RATIO ONLY -- refused for every other quantity in
    ``__post_init__``. When set, ``unit_quote``/``unit_cell``/``unit_occurrence``
    must all be empty/None: there is no printed unit to quote or cite, so the
    produced value carries ``unit_provenance=NOT_PRINTED_IN_SOURCE`` with
    ``unit_raw``/``unit_ref`` Absent. The LABEL must still be groundable (a
    grounded label is required to declare the quantity at all)."""

    def __post_init__(self) -> None:
        if not isinstance(self.role, AxisRole):
            raise TabularDatasetProducerError(
                f"TabularAxisSpec.role={self.role!r} must be a genuine AxisRole member, not "
                f"{type(self.role).__name__} -- AxisRole is a StrEnum, so a plain string equal to a "
                "member's value would compare `==` equal without actually being that member"
            )
        if not isinstance(self.quantity_kind, units.QuantityKind):
            raise TabularDatasetProducerError(
                f"TabularAxisSpec.quantity_kind={self.quantity_kind!r} must be a genuine QuantityKind "
                f"member, not {type(self.quantity_kind).__name__}"
            )
        _require_int_occurrences(
            "TabularAxisSpec", label_occurrence=self.label_occurrence, unit_occurrence=self.unit_occurrence
        )
        _reject_cell_with_occurrence(
            "TabularAxisSpec",
            label=(self.label_occurrence, self.label_cell),
            unit=(self.unit_occurrence, self.unit_cell),
        )
        if self.unit_not_printed:
            # I-060, gated NARROWLY. The schema re-checks the quantity, but the
            # refusal is stated here too so a caller learns it at construction,
            # against the spec it wrote, rather than only when the envelope
            # validates.
            if self.quantity_kind is not units.QuantityKind.EQUIVALENCE_RATIO:
                raise TabularDatasetProducerError(
                    f"TabularAxisSpec.unit_not_printed is permitted for "
                    f"quantity_kind={units.QuantityKind.EQUIVALENCE_RATIO.value!r} ONLY, not "
                    f"{self.quantity_kind.value!r}: equivalence ratio is the sole dimensionless "
                    "quantity whose unit carries no scale; the fraction kinds scale %/ppm to 1, so "
                    "declaring one without a printed unit would silently rescale a stored magnitude"
                )
            if self.unit_quote or self.unit_cell is not None or self.unit_occurrence is not None:
                raise TabularDatasetProducerError(
                    "TabularAxisSpec.unit_not_printed says the source printed no unit token, so "
                    f"unit_quote must be empty and unit_cell/unit_occurrence None; got "
                    f"unit_quote={self.unit_quote!r}, unit_cell={self.unit_cell!r}, "
                    f"unit_occurrence={self.unit_occurrence!r}"
                )
            if not self.label_quote:
                raise TabularDatasetProducerError(
                    "TabularAxisSpec.unit_not_printed requires a groundable label_quote: the quantity "
                    "may only be declared where a label naming it is genuinely groundable in the document"
                )
        elif not self.unit_quote:
            raise TabularDatasetProducerError(
                "TabularAxisSpec.unit_quote must be non-empty unless unit_not_printed is set: every "
                "printed-unit axis grounds a unit token"
            )


@dataclass(frozen=True, slots=True)
class TabularPointValueSpec:
    """One value of one point, located at a table cell.

    ``axis_id`` says which of the series' axes this value instantiates;
    ``value_quote`` is the exact number as printed in ``cell`` (the whole cell
    text must equal it). There is no char-span option and no ``occurrence``: a
    series data-point value is always a cell (V4/V7), so a value that is not in a
    grid has no home here.
    """

    axis_id: str
    value_quote: str
    cell: TableCellGrounding

    def __post_init__(self) -> None:
        if not isinstance(self.cell, TableCellGrounding):
            raise TabularDatasetProducerError(
                f"TabularPointValueSpec.cell must be a TableCellGrounding, not {type(self.cell).__name__} "
                "-- a series data-point value is located at a table cell, never searched in running text"
            )


@dataclass(frozen=True, slots=True)
class TabularPointSpec:
    """One located point: one cell-grounded value for EVERY axis of the series.

    The producer assigns each value to a :class:`Coordinate` or an
    :class:`Observation` by its axis's role, so the caller states values once and
    does not repeat the role. Every declared axis must be covered exactly once --
    a point missing a coordinate is unlocated (S8) and one missing an observation
    records no datum (S9); both are refused here with the axis named, rather than
    surfacing later as a schema error from inside :class:`Series` construction.
    """

    point_id: str
    values: tuple[TabularPointValueSpec, ...]


@dataclass(frozen=True, slots=True)
class _CellValueSpec:
    """The read-only view :func:`_measured_value` needs of one cell value.

    Satisfies ``dataset_producer._ValueQuoteSpec`` structurally. ``value_occurrence``
    is always ``None`` (the value is cell-located, so no running-text search runs
    for it); ``unit_occurrence`` disambiguates the unit's char-span search when the
    unit is not itself cell-located.
    """

    quantity_kind: units.QuantityKind
    value_quote: str
    unit_quote: str
    unit_occurrence: int | None
    value_occurrence: int | None = None


def _axis_label_ref(text: str, axis: TabularAxisSpec, repairs: tuple[GlyphRepair, ...]) -> SourceRef:
    """Ground the axis header label, at its cell when given else as a char span."""
    if axis.label_cell is not None:
        return SourceRef(node_id=_ROOT_NODE_ID, locator=_cell_locator(axis.label_cell))
    return SourceRef(
        node_id=_ROOT_NODE_ID,
        locator=ground_quote(
            text,
            axis.label_quote,
            role=QuoteRole.LABEL,
            occurrence=axis.label_occurrence,
            repairs=repairs,
        ),
    )


def _cell_requests(
    axes: tuple[TabularAxisSpec, ...], points: tuple[TabularPointSpec, ...]
) -> tuple[tuple[str, str, TableCellGrounding], ...]:
    """Every ``(owner, quote, cell)`` a cell must be validated for, across the
    whole build -- axis labels, axis units, and every point value -- so
    :class:`_CellCiter` judges them as one batch before a ref is built."""
    requests: list[tuple[str, str, TableCellGrounding]] = []
    for axis in axes:
        if axis.label_cell is not None:
            requests.append((f"axis {axis.axis_id!r} label", axis.label_quote, axis.label_cell))
        if axis.unit_cell is not None:
            requests.append((f"axis {axis.axis_id!r} unit", axis.unit_quote, axis.unit_cell))
    for point in points:
        for value in point.values:
            requests.append((f"point {point.point_id!r} axis {value.axis_id!r} value", value.value_quote, value.cell))
    return tuple(requests)


def _duplicate_ids(ids: list[str], *, owner: str) -> None:
    """Refuse a repeated id, which would make a per-item finding ambiguous."""
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise TabularDatasetProducerError(
                f"duplicate {owner} {value!r}: every {owner} must be unique, or a per-item finding "
                "cannot say which one it means"
            )
        seen.add(value)


def _cell_address(value: TabularPointValueSpec) -> str:
    """The full cell address of one value, for a disagreement message."""
    cell = value.cell
    return (
        f"axis {value.axis_id!r} at table_key={cell.table_key!r} "
        f"inventory={cell.inventory.inventory_sha256!r} row={cell.row} col={cell.col}"
    )


def _reject_incoherent_join(point: TabularPointSpec) -> None:
    """Refuse a point whose values are not co-located in a way the document asserts.

    Each value carries its OWN independent cell address -- which table, which row,
    which column -- and nothing upstream requires those addresses to belong
    together. Coverage (checked separately) proves a point instantiates every axis
    exactly once; it says NOTHING about whether the covered values came from one
    record. Without this guard a point may take its coordinate from one row and its
    observation from another, or from a different table entirely, and every check
    downstream passes because every cited cell is individually real: replay
    re-derives each byte, the hashes match, and provenance certifies a relationship
    the document never asserted. That is the exact artifact class this project
    refuses -- not a wrong number, but a correct number in a fabricated join.

    Two co-location rules, checked in order:

    * **Same table.** All of a point's values must sit in the SAME table -- equal
      ``table_key`` AND equal embedded inventory. Conditions in one table joined to
      results in another by a case label is a real shape in this literature, but it
      needs a declared cross-table join key the schema does not yet carry (see
      :class:`~carmel.services.condition_set_producer.TableCellGrounding`). That is a
      future capability: refused here with what would be required, never performed
      silently and never built as an escape hatch.
    * **Same row.** Within that one table, all values must share the same ``row``
      ordinal. The grid model derives ``row``/``col`` from page geometry alone
      (:func:`carmel.services.pdf_tables.build_inventory`) and records NO
      orientation -- so ``row`` is stated as the geometric ordinal it is, not as
      "the human-readable record". For a standard table (one record per printed
      row) that ordinal IS the record, which is why the only production caller,
      which reads each point from ``cell(row, 0)`` and ``cell(row, 1)``, passes
      untouched. A transposed table (one record per COLUMN) is not modelled and is
      refused here for the same reason a cross-table join is: an unbuilt future
      capability, surfaced honestly rather than silently joined.
    """
    values = point.values
    if len(values) < 2:
        return  # A lone value shares its own row and table; coverage handles emptiness.
    first = values[0]
    for value in values[1:]:
        if (
            value.cell.table_key != first.cell.table_key
            or value.cell.inventory.inventory_sha256 != first.cell.inventory.inventory_sha256
        ):
            raise TabularDatasetProducerError(
                f"point {point.point_id!r} joins values from DIFFERENT tables: "
                f"{_cell_address(first)} vs {_cell_address(value)} -- a data point is one record of one "
                "grid, and every cell being individually real does not make the join real. Conditions in "
                "one table joined to results in another by a case label is a legitimate shape, but it needs "
                "a declared cross-table join key the schema does not yet carry; that is a future capability, "
                "refused here, not performed silently"
            )
    for value in values[1:]:
        if value.cell.row != first.cell.row:
            raise TabularDatasetProducerError(
                f"point {point.point_id!r} joins values from DIFFERENT rows of one table: "
                f"{_cell_address(first)} vs {_cell_address(value)} -- a grid asserts that values sharing a "
                "row form one record, so values taken from rows that disagree are a join the document never "
                "made (every cell is individually real; the relationship is fabricated). The grid records no "
                "orientation, so 'row' is the geometric ordinal: if this table is transposed (one record per "
                "column), that is a future capability and is not yet supported"
            )


def produce_tabular_envelope_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    series_id: str,
    value_origin: ValueOrigin,
    axes: tuple[TabularAxisSpec, ...],
    points: tuple[TabularPointSpec, ...],
    composition: Maybe[Composition] = _COMPOSITION_NOT_EXTRACTED,
) -> DatasetEnvelope:
    """Build one validated :class:`DatasetEnvelope` from a stored artifact's TABLE.

    The vertical slice: authenticate ``raw.bin`` against the artifact's own
    sha256, select the ONE current extraction record, validate every cited cell
    against its embedded inventory, ground each axis label and unit, and assemble
    a single ``TABULAR`` :class:`Series` whose points carry cell-located values.
    Construction runs pydantic's full validation -- nothing here uses
    ``model_construct`` -- so the returned envelope passes every dataset
    validator (V1-V9, T2-T5, S1-S11).

    Args:
        workspace_root: Workspace root holding the content-addressed store.
        sha256: The raw artifact's sha256.
        series_id: The id of the one series this envelope carries.
        value_origin: HOW the numbers were produced, as ASSERTED by the caller
            (experimental / simulation / derived) -- recorded unverified.
        axes: The series' axes. At least one COORDINATE and one OBSERVATION are
            required (enforced by :class:`Series`), each with a groundable header
            label and unit.
        points: The located points, one cell value per axis.
        composition: The dataset's single composition, or an explicit
            :class:`Absent` when none was extracted.

    Returns:
        A fully validated envelope.

    Raises:
        TabularDatasetProducerError: No axes or points, ids collide, a point does
            not cover the axes exactly, a value names an unknown axis, or a spec
            field is the wrong type.
        ConditionSetProducerError: A cell citation is dishonest (the cell does
            not exist, its text differs from the quote, or one cell is two
            strings) -- raised by the shared :class:`_CellCiter`; a
            ``DatasetProducerError`` subclass, so a caller failing closed still
            catches it.
        DatasetProducerError: The artifact is missing, legacy, corrupt, lossily
            extracted, or a value is not a bare numeral / its unit is unknown for
            its quantity kind.
        QuoteGroundingError: A char-span label or unit is absent from the
            document, or occurs more than once and was not disambiguated.
    """
    if not axes:
        raise TabularDatasetProducerError(
            f"artifact {sha256!r}: refusing to produce a tabular series with no axes -- a series with no "
            "axes locates and measures nothing"
        )
    if not points:
        raise TabularDatasetProducerError(
            f"artifact {sha256!r}: refusing to produce a tabular series with no points -- an empty series "
            "asserts a table with no rows, which grounding cannot establish"
        )
    _duplicate_ids([axis.axis_id for axis in axes], owner="axis_id")
    _duplicate_ids([point.point_id for point in points], owner="point_id")
    if not isinstance(value_origin, ValueOrigin):
        raise TabularDatasetProducerError(
            f"value_origin={value_origin!r} must be a genuine ValueOrigin member, not "
            f"{type(value_origin).__name__} -- ValueOrigin is a StrEnum, so a plain string equal to a "
            "member's value would compare `==` equal without actually being that member"
        )

    axis_by_id = {axis.axis_id: axis for axis in axes}
    axis_ids = set(axis_by_id)
    for point in points:
        seen_here = [value.axis_id for value in point.values]
        unknown = sorted(set(seen_here) - axis_ids)
        if unknown:
            raise TabularDatasetProducerError(
                f"point {point.point_id!r} names axis(es) {unknown!r} that the series does not declare"
            )
        if sorted(seen_here) != sorted(axis_ids):
            raise TabularDatasetProducerError(
                f"point {point.point_id!r} covers axes {sorted(seen_here)!r} but the series declares "
                f"{sorted(axis_ids)!r} -- every point must carry exactly one value for every axis "
                "(a point missing a coordinate is unlocated; one missing an observation records no datum)"
            )
        # Coverage proves the point instantiates every axis; co-location proves its
        # values came from ONE record the document actually asserts (same table,
        # same row) -- refusing the fabricated cross-row / cross-table join that
        # every downstream byte-level check would otherwise certify.
        _reject_incoherent_join(point)

    grounding = _prepare_grounding(
        workspace_root, sha256, envelope_noun="dataset envelope", envelope_subject="A dataset"
    )
    text = grounding.text
    repairs = grounding.glyph_repairs

    # The single authority on cell citations for this build. Every cell requested
    # by any axis label/unit or any point value is validated as ONE batch here --
    # before a ref is built -- so a missing cell, a text mismatch, or one cell
    # claimed by two strings is refused with the clearest message, and the
    # inventories the envelope must embed are collected exactly once.
    citer = _CellCiter(grounding.graph.node(_ROOT_NODE_ID))
    citer.validate(_cell_requests(axes, points))

    axis_declarations = tuple(
        AxisDeclaration(
            axis_id=axis.axis_id,
            role=axis.role,
            quantity_kind=axis.quantity_kind,
            label_raw=axis.label_quote,
            label_ref=_axis_label_ref(text, axis, repairs),
        )
        for axis in sorted(axes, key=lambda axis: axis.axis_id)
    )

    data_points: list[DataPoint] = []
    for point in sorted(points, key=lambda point: point.point_id):
        coordinates: list[Coordinate] = []
        observations: list[Observation] = []
        for value in point.values:
            axis = axis_by_id[value.axis_id]
            measured = _measured_value(
                text,
                _CellValueSpec(
                    quantity_kind=axis.quantity_kind,
                    value_quote=value.value_quote,
                    unit_quote=axis.unit_quote,
                    unit_occurrence=axis.unit_occurrence,
                ),
                where=f"point {point.point_id!r} axis {axis.axis_id!r}",
                document_source_context=grounding.document_source_context,
                document_glyph_health=grounding.document_glyph_health,
                document_glyph_repairs=repairs,
                value_locator=_cell_locator(value.cell),
                unit_locator=_cell_locator(axis.unit_cell) if axis.unit_cell is not None else None,
                unit_provenance=(
                    UnitProvenance.NOT_PRINTED_IN_SOURCE if axis.unit_not_printed else UnitProvenance.PRINTED_IN_SOURCE
                ),
            )
            # This producer reads no per-point uncertainty from the table. That is
            # a NOT_EXTRACTED_YET refusal, never an assertion the paper reported
            # none -- the two must not conflate.
            not_extracted = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
            if axis.role is AxisRole.COORDINATE:
                coordinates.append(Coordinate(axis_id=axis.axis_id, value=measured, uncertainty=not_extracted))
            else:
                observations.append(Observation(axis_id=axis.axis_id, value=measured, uncertainty=not_extracted))
        data_points.append(
            DataPoint(
                point_id=point.point_id,
                coordinates=tuple(sorted(coordinates, key=lambda coord: coord.axis_id)),
                observations=tuple(sorted(observations, key=lambda obs: obs.axis_id)),
                composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
            )
        )

    series = Series(
        series_id=series_id,
        source_form=SourceForm.TABULAR,
        value_origin=value_origin,
        axes=axis_declarations,
        constants=(),
        points=tuple(data_points),
        digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )

    return DatasetEnvelope(
        source_graph=grounding.graph,
        composition=composition,
        series=(series,),
        # Every MeasuredValue cites the active conversion table, so it must be
        # embedded; T2 refuses a decorative table nothing cites, and there is
        # always at least one value in a series.
        conversion_tables=(_ACTIVE.embedded,),
        # Exactly the inventories this build's TABLE_CELL locators cite -- collected,
        # deduplicated and sorted by _CellCiter, never a store lookup at replay time.
        table_inventories=citer.table_inventories(),
        figure_digitizations=(),
    )
