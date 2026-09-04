# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The project's FIRST real tabular dataset target, wired for durable production.

This module is the single definition of one dataset: Table 1, page 4 of the
paper whose sha256 is :data:`TARGET_DOCUMENT_SHA256` -- the atmospheric hydrogen
flame-speed sweep the i032/i005 split-header work measured to a complete
23x4 / 92-cell grid. Column 0 is the equivalence ratio phi (0.5 ... 5.0),
column 1 the laminar flame speed S_L in cm/s (67.2 ... 110.1).

WHY IT EXISTS. The tabular production path
(:func:`carmel.services.tabular_dataset_producer.produce_tabular_envelope_from_artifact`)
works and is proven by :mod:`tests.test_tabular_dataset_target_acceptance`, but
that acceptance test stages the document into a throw-away ``tmp_path`` and
stores the envelope there: the artifact is built and destroyed inside the test,
so the project had a working extraction pipeline and zero durable output. This
module holds the ONE definition of that dataset -- footprint, rows, headers,
axes, and points -- so both the acceptance test and the durable production entry
point (:func:`produce_and_store_target`, exposed as ``carmel store-tabular-dataset``)
construct it from the same source, and :func:`render_series_text` renders the
stored series so an operator can read the numbers rather than a sha256.

It does NOT reimplement the producer, the grounding types, or the store: it wires
what already exists. It is deliberately scoped to this ONE document and table --
generalising to other papers is a separate ticket, not this one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from carmel.paths import default_workspaces_root
from carmel.schemas.datasets import (
    Absent,
    AxisRole,
    CaptionLabelKey,
    DatasetEnvelope,
    EmbeddedTableInventory,
    MeasuredValue,
    Series,
    TableCellLocator,
    ValueOrigin,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.condition_set_producer import TableCellGrounding
from carmel.services.dataset_bridge import store_dataset_envelope
from carmel.services.dataset_store import StoredDataset, canonical_json_bytes
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory
from carmel.services.tabular_dataset_producer import (
    TabularAxisSpec,
    TabularPointSpec,
    TabularPointValueSpec,
    produce_tabular_envelope_from_artifact,
)

__all__ = [
    "PHI_HEADER",
    "PHI_LABEL_OCCURRENCE",
    "PHI_LABEL_QUOTE",
    "S_L_HEADER",
    "TARGET_CAMPAIGN",
    "TARGET_DOCUMENT_SHA256",
    "TARGET_ROWS",
    "TARGET_SERIES_ID",
    "TARGET_TABLE_FOOTPRINT",
    "TARGET_TABLE_KEY",
    "TARGET_WORKSPACES_ROOTS",
    "StoredTargetDataset",
    "TabularDatasetTargetError",
    "build_axes",
    "build_embedded_inventory",
    "build_envelope",
    "build_points",
    "locate_target_workspace",
    "produce_and_store_target",
    "read_target_raw",
    "render_series_text",
    "write_series_export",
]

#: The raw artifact sha256 of the source document (10.1115-1.4007737).
TARGET_DOCUMENT_SHA256 = "c2be41381e3c55671af2912a46d5ce703c0f56cea9dadfba8789e7417059155a"

#: The campaign whose literature store holds the document, relative to a
#: workspaces root. The document was admitted into this campaign, so its evidence
#: store -- and the dataset store the produced envelope is written into -- live
#: under ``<workspaces-root>/<TARGET_CAMPAIGN>``.
TARGET_CAMPAIGN = "live-syngas"

#: Both roots the acceptance test searches, in order: the packaged default, then
#: the operator's ``~/runs/carmel/workspaces``. Discovery mirrors the test exactly
#: so the durable entry point finds the same document the test proves against.
TARGET_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")

#: The id of the one series this envelope carries.
TARGET_SERIES_ID = "flame_speed_sweep"

#: The registered footprint measured by tests.test_split_header_acceptance.
TARGET_TABLE_FOOTPRINT = ClaimedFootprint(
    page=4,
    x_start=305.0,
    x_end=555.0,
    y_top=745.0,
    y_bottom=520.0,
    caption_text="rangeofequivalenceratios",
    caption_x_start=311.981,
    caption_baseline_y=750.274,
)
TARGET_TABLE_KEY = CaptionLabelKey(label="Table 1")

#: (row, phi, S_L) for the 22 data rows -- rows 1..22 of the grid; row 0 is the
#: header. Pinned as the PRECONDITION the build asserts on (and re-derives), not a
#: golden byte.
TARGET_ROWS: tuple[tuple[int, str, str], ...] = (
    (1, "0.5", "67.2"),
    (2, "0.6", "96.5"),
    (3, "0.7", "124.4"),
    (4, "0.8", "169.9"),
    (5, "0.9", "194.0"),
    (6, "1.0", "218.0"),
    (7, "1.1", "236.7"),
    (8, "1.2", "254.9"),
    (9, "1.3", "267.4"),
    (10, "1.4", "275.0"),
    (11, "1.5", "280.3"),
    (12, "1.6", "282.8"),
    (13, "1.7", "283.8"),
    (14, "1.8", "282.9"),
    (15, "1.9", "280.3"),
    (16, "2.0", "278.9"),
    (17, "2.5", "249.1"),
    (18, "3.0", "217.4"),
    (19, "3.5", "187.6"),
    (20, "4.0", "158.7"),
    (21, "4.5", "133.0"),
    (22, "5.0", "110.1"),
)

#: The verbatim header cell texts -- symbol-font phi decodes to "/", and the
#: flame-speed header carries its subscript fold. Recorded as-is (grounding proves
#: location, never meaning). PHI_HEADER is the ACTUAL printed header; it names no
#: quantity and prints no unit, which is why the phi axis's label is grounded at
#: the caption instead (see below) and its unit is marked not-printed-in-source.
PHI_HEADER = "/"
S_L_HEADER = "S0L;u(cm/s)"

#: The phi axis LABEL, grounded as a char span in running prose (I-060). The
#: column header "/" names no quantity, but the table's caption reads "range of
#: equivalence ratios" -- the phrase that identifies the coordinate. It occurs
#: THREE times in this document; Table 1's caption is the SECOND (it sits
#: immediately before this table's flame-speed header at char offset 19877), so
#: the occurrence is pinned as ``1`` (``ground_quote``'s occurrence is 0-based).
#: Grounding it proves the phrase is in the document, NOT that this column is that
#: quantity -- the header bridge replay files as an unchecked claim.
PHI_LABEL_QUOTE = "range of equivalence ratios"
PHI_LABEL_OCCURRENCE = 1

#: The grid is 23 rows (one header + 22 data) by 4 columns.
_EXPECTED_CELL_COUNT = 92


class TabularDatasetTargetError(Exception):
    """The target dataset could not be produced or stored honestly.

    Raised for a precondition the build depends on and cannot proceed without --
    the store not holding the document, the stored bytes not being the measured
    document, or the table grid refusing or coming back incomplete. Never a way to
    route around a refusal: each case fails closed with the reason named.
    """


def _raw_path(workspace_root: Path) -> Path:
    return workspace_root / "evidence" / "literature" / TARGET_DOCUMENT_SHA256 / "raw.bin"


def locate_target_workspace(roots: tuple[Path, ...] = TARGET_WORKSPACES_ROOTS) -> Path | None:
    """Return the campaign workspace holding the target document, or ``None``.

    Mirrors the acceptance test's discovery: search each root for
    ``<root>/<TARGET_CAMPAIGN>/evidence/literature/<sha>/raw.bin`` and return that
    campaign workspace (the directory the dataset store is written into).
    """
    for root in roots:
        if _raw_path(root / TARGET_CAMPAIGN).exists():
            return root / TARGET_CAMPAIGN
    return None


def read_target_raw(workspace_root: Path) -> bytes:
    """Read and authenticate the target document's ``raw.bin`` from a workspace.

    Raises:
        TabularDatasetTargetError: The document is not stored under
            ``workspace_root``, its stored bytes cannot be read, or they are not
            the measured document.
    """
    raw_path = _raw_path(workspace_root)
    if not raw_path.exists():
        raise TabularDatasetTargetError(f"target document is not stored under {workspace_root}: no {raw_path}")
    try:
        raw = raw_path.read_bytes()
    except OSError as exc:
        raise TabularDatasetTargetError(f"cannot read the stored raw.bin at {raw_path}: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != TARGET_DOCUMENT_SHA256:
        raise TabularDatasetTargetError(
            f"stored raw.bin is {actual}, not the measured {TARGET_DOCUMENT_SHA256}; refusing to build"
        )
    return raw


def build_embedded_inventory(raw: bytes) -> EmbeddedTableInventory:
    """Derive the registered table's cell grid from ``raw`` and embed it.

    Fails closed: the grid must refuse nothing and come back complete at exactly
    92 cells -- the precondition the acceptance test asserts. A refused or
    incomplete grid is a defect to surface, never something to build a partial
    series on.

    Raises:
        TabularDatasetTargetError: The grid refused or is not the complete
            92-cell grid.
    """
    inventory = build_inventory(extract_fragments(raw), TARGET_TABLE_FOOTPRINT)
    if inventory.refusals != ():
        raise TabularDatasetTargetError(f"the target grid refused: {inventory.refusals}")
    if not inventory.complete or len(inventory.cells) != _EXPECTED_CELL_COUNT:
        raise TabularDatasetTargetError(
            f"the target grid is not the complete {_EXPECTED_CELL_COUNT}-cell grid "
            f"(complete={inventory.complete}, cells={len(inventory.cells)})"
        )
    payload = inventory_record_payload(inventory, raw_sha256=TARGET_DOCUMENT_SHA256)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=TARGET_DOCUMENT_SHA256,
        canonical_json=canonical,
    )


def _cell_of(embedded: EmbeddedTableInventory, row: int, col: int) -> TableCellGrounding:
    return TableCellGrounding(table_key=TARGET_TABLE_KEY, row=row, col=col, inventory=embedded)


def build_axes(embedded: EmbeddedTableInventory) -> tuple[TabularAxisSpec, TabularAxisSpec]:
    """The series' two axes, grounded against the embedded inventory."""
    return (
        # phi is EQUIVALENCE_RATIO, declared WITHOUT a printed unit (I-060). It was
        # OTHER "/" until this ticket, because its column prints no dimensionless
        # unit token and the units table admits only "1"/"-"/"dimensionless" for
        # EQUIVALENCE_RATIO -- none of which this column prints. Rather than fall
        # back to OTHER (a coordinate a reader cannot tell is an equivalence ratio)
        # or launder the corrupted header "/" into "1" (a fabricated unit this
        # project refuses -- and the units test still guards against a "/"->"1"
        # alias), the axis declares the quantity and marks the unit
        # not-printed-in-source: unit_raw/unit_ref Absent, unit_normalized "1",
        # unit_provenance NOT_PRINTED_IN_SOURCE, a first-class stored fact. The
        # LABEL is grounded in the caption "range of equivalence ratios" (a char
        # span, occurrence 1 -- 0-based, the SECOND of the phrase's three
        # occurrences, the Table 1 caption; see PHI_LABEL_OCCURRENCE above), which
        # proves the phrase is in the document but NOT
        # that THIS column carries the quantity -- so replay files the header
        # bridge as an UncheckedSemanticClaim and overall_outcome can never reach
        # VERIFIED by having dropped the unit. This is permitted for equivalence
        # ratio ONLY: it is the sole dimensionless quantity whose unit carries no
        # scale, so declaring "1" rescales nothing (the fraction kinds scale
        # %/ppm -> 1, so the same move would corrupt their magnitudes).
        TabularAxisSpec(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
            label_quote=PHI_LABEL_QUOTE,
            label_occurrence=PHI_LABEL_OCCURRENCE,
            unit_not_printed=True,
        ),
        # The flame speed IS modelled: VELOCITY, cm/s. The unit "cm/s" is written
        # in the running prose, so it is grounded there as a char span (V4/V7
        # constrain only the value ref); its printed header is grounded at its cell.
        TabularAxisSpec(
            axis_id="s_l",
            role=AxisRole.OBSERVATION,
            quantity_kind=units.QuantityKind.VELOCITY,
            label_quote=S_L_HEADER,
            unit_quote="cm/s",
            label_cell=_cell_of(embedded, 0, 1),
            unit_occurrence=1,
        ),
    )


def build_points(embedded: EmbeddedTableInventory) -> tuple[TabularPointSpec, ...]:
    """One cell-located point per data row, phi from col 0 and S_L from col 1."""
    return tuple(
        TabularPointSpec(
            point_id=f"r{row:02d}",
            values=(
                TabularPointValueSpec(axis_id="phi", value_quote=phi, cell=_cell_of(embedded, row, 0)),
                TabularPointValueSpec(axis_id="s_l", value_quote=s_l, cell=_cell_of(embedded, row, 1)),
            ),
        )
        for row, phi, s_l in TARGET_ROWS
    )


def build_envelope(workspace_root: Path, embedded: EmbeddedTableInventory) -> DatasetEnvelope:
    """Assemble the fully validated envelope through the production producer."""
    return produce_tabular_envelope_from_artifact(
        workspace_root,
        sha256=TARGET_DOCUMENT_SHA256,
        series_id=TARGET_SERIES_ID,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=build_axes(embedded),
        points=build_points(embedded),
    )


@dataclass(frozen=True, slots=True)
class StoredTargetDataset:
    """The durable outcome of :func:`produce_and_store_target`.

    ``sha256`` and ``path`` locate the stored dataset envelope; ``envelope`` is the
    in-memory object that was stored, kept so a caller can render its export
    without a second load.
    """

    sha256: str
    path: Path
    envelope: DatasetEnvelope


def produce_and_store_target(workspace_root: Path) -> StoredTargetDataset:
    """Produce the target series from ``workspace_root`` and store it durably.

    The whole vertical slice, run against a real workspace: authenticate the
    stored document, derive and embed its table grid, assemble the ``TABULAR``
    series through the production producer, and write the envelope into the
    workspace's own dataset store. Reuses the producer and the store wrapper --
    nothing here reimplements them.

    Raises:
        TabularDatasetTargetError: A precondition failed (see
            :func:`read_target_raw`, :func:`build_embedded_inventory`).
    """
    raw = read_target_raw(workspace_root)
    embedded = build_embedded_inventory(raw)
    envelope = build_envelope(workspace_root, embedded)
    stored: StoredDataset = store_dataset_envelope(workspace_root, envelope)
    return StoredTargetDataset(sha256=stored.sha256, path=stored.path, envelope=envelope)


def _measured_or_none(value: MeasuredValue | Absent) -> MeasuredValue | None:
    """Unwrap a ``Maybe[MeasuredValue]`` slot, or ``None`` when absent."""
    return None if isinstance(value, Absent) else value


def _stored_table_reference(envelope: DatasetEnvelope) -> tuple[str, str]:
    """Read the table label and source page the envelope itself cites.

    Walks every :class:`~carmel.schemas.datasets.SourceRef` reachable in the
    envelope (the same choke point V1 validates every ref through) for the
    first ``TableCellLocator`` naming a ``CaptionLabelKey``, then reads the
    page from the embedded inventory its ``pdf_table_inventory_sha256`` names.
    Both facts are read from the STORED envelope, never a module constant, so
    a previously stored envelope renders with ITS OWN label and page even if
    ``TARGET_TABLE_KEY``/``TARGET_TABLE_FOOTPRINT`` are edited later -- the
    drift :func:`render_series_text`'s docstring promises is impossible. This
    is the sibling of
    :func:`carmel.services.condition_set_target._stored_table_reference`.

    Falls back to ``"unknown"`` for both, exactly like the ``raw_sha256``
    fallback in the render, for an envelope with no PDF table-cell citation to
    read either fact from. The PAGE alone degrades to ``"unknown"`` -- the label
    stays the one actually cited -- when a cited inventory's ``canonical_json``
    cannot be read for a page. ``EmbeddedTableInventory``'s T1 validator makes
    that impossible for any inventory that passed construction (it rejects
    unparseable, ``footprint``-less, or wrong-shaped canonical JSON outright), so
    it can only arise from a validation-bypassed or partially constructed
    envelope; the render path stays total rather than raising an untyped
    traceback into a caller that does not expect one.
    """
    for _, ref in iter_source_refs(envelope):
        locator = ref.locator
        if not (isinstance(locator, TableCellLocator) and isinstance(locator.table_key, CaptionLabelKey)):
            continue
        label = locator.table_key.label
        page: object = "unknown"
        if isinstance(locator.pdf_table_inventory_sha256, str):
            for inventory in envelope.table_inventories:
                if inventory.inventory_sha256 == locator.pdf_table_inventory_sha256:
                    try:
                        page = json.loads(inventory.canonical_json)["footprint"]["page"]
                    except json.JSONDecodeError, KeyError, TypeError:
                        # Unreachable for a validated inventory -- T1 above rejects
                        # every malformation this expression could trip on -- so this
                        # only fires for a validation-bypassed envelope. Degrade the
                        # page to "unknown" (the same honest fallback as an
                        # unresolvable citation) rather than crash the render path,
                        # which the project's fail-closed contract forbids.
                        page = "unknown"
                    break
        return label, str(page)
    return "unknown", "unknown"


def render_series_text(envelope: DatasetEnvelope) -> str:
    """Render the stored series as a human-readable text table.

    Derived entirely from ``envelope`` so it cannot drift from what was grounded:
    a header identifying the source document, the table, and each axis, then one
    row per data point with the coordinate, the observation, and their units. The
    table label and page are read from the envelope's own citations via
    :func:`_stored_table_reference`, not from ``TARGET_TABLE_KEY`` /
    ``TARGET_TABLE_FOOTPRINT``, so an envelope stored under old constants renders
    with the facts it was grounded under.
    """
    series: Series = envelope.series[0]
    raw_sha256 = envelope.table_inventories[0].raw_sha256 if envelope.table_inventories else "unknown"
    table_label, table_page = _stored_table_reference(envelope)

    # One representative unit per axis, to name the unit each column is measured in
    # (every point on an axis shares that unit).
    unit_by_axis: dict[str, str] = {}

    def _record_unit(axis_id: str, value: MeasuredValue | Absent) -> None:
        measured = _measured_or_none(value)
        if axis_id not in unit_by_axis and measured is not None:
            unit_by_axis[axis_id] = measured.unit_normalized

    for point in series.points:
        for coord in point.coordinates:
            _record_unit(coord.axis_id, coord.value)
        for obs in point.observations:
            _record_unit(obs.axis_id, obs.value)

    lines: list[str] = [
        f"Dataset series : {series.series_id}",
        f"Source document: sha256 {raw_sha256}",
        f"Source table   : {table_label}, page {table_page}",
        f"Source form    : {series.source_form.value}",
        f"Value origin   : {series.value_origin.value}",
        "",
    ]
    for axis in series.axes:
        unit = unit_by_axis.get(axis.axis_id, "")
        lines.append(
            f"  axis {axis.axis_id:<4} role={axis.role.value:<11} "
            f"quantity={axis.quantity_kind.value:<15} unit={unit!r:<8} header={axis.label_raw!r}"
        )
    lines.append("")

    coord_axis = next(a for a in series.axes if a.role is AxisRole.COORDINATE)
    obs_axis = next(a for a in series.axes if a.role is AxisRole.OBSERVATION)
    coord_head = f"{coord_axis.axis_id} [{unit_by_axis.get(coord_axis.axis_id, '')}]"
    obs_head = f"{obs_axis.axis_id} [{unit_by_axis.get(obs_axis.axis_id, '')}]"
    lines.append(f"  {'point':<6} {coord_head:>16} {obs_head:>16}")
    lines.append(f"  {'-' * 6} {'-' * 16} {'-' * 16}")

    for point in series.points:
        coord = next(c for c in point.coordinates if c.axis_id == coord_axis.axis_id)
        obs = next(o for o in point.observations if o.axis_id == obs_axis.axis_id)
        coord_val = _measured_or_none(coord.value)
        obs_val = _measured_or_none(obs.value)
        coord_text = coord_val.raw_text if coord_val is not None else "(absent)"
        obs_text = obs_val.raw_text if obs_val is not None else "(absent)"
        lines.append(f"  {point.point_id:<6} {coord_text:>16} {obs_text:>16}")

    return "\n".join(lines) + "\n"


def write_series_export(workspace_root: Path, dataset: StoredTargetDataset) -> Path:
    """Write the rendered series into the workspace's ``reports`` directory.

    Returns the path of the written file: ``reports/dataset-<sha256>.txt``.
    """
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"dataset-{dataset.sha256}.txt"
    path.write_text(render_series_text(dataset.envelope), encoding="utf-8")
    return path
