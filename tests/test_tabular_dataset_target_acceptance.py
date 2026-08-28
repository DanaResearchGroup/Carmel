"""End-to-end acceptance: this project's FIRST stored TABULAR dataset series,
built by the production path from a REAL table in a REAL stored artifact.

The table is Table 1, page 4 of ``10.1115-1.4007737`` -- the atmospheric
hydrogen flame-speed sweep the i032/i005 split-header work measured to a
complete 23x4 / 92-cell grid (see :mod:`tests.test_split_header_acceptance`).
Column 0 is the equivalence ratio phi (0.5 ... 5.0), column 1 the laminar flame
speed S_L in cm/s (67.2 ... 110.1). This module reads that grid through
:func:`carmel.services.pdf_tables.build_inventory`, grounds every phi and S_L
value at its cell, and produces a ``source_form=TABULAR`` :class:`Series`.

The table footprint, rows, headers, axes, and points are defined ONCE, in
:mod:`carmel.services.tabular_dataset_target`, and imported here: this test and
the durable production entry point (``carmel store-tabular-dataset``) build the
same dataset from the same source. This module still stages into a throw-away
``tmp_path``; the durable entry point writes into the operator's real workspace.

Corpus-gated exactly like :mod:`tests.test_condition_set_target_acceptance`: the
paper is non-redistributable, so it is read from the operator's corpus store at
runtime and every test SKIPS -- never passes -- when the document (or its store)
is absent or is not byte-for-byte the measured document. pypdf-gated too: the
grid is derived from PDF geometry, so without pypdf there is nothing to derive.

**WHAT REPLAY PROVES HERE, STATED PLAINLY.** ``evidence_outcome`` is
UNVERIFIABLE, and that is the honest, designed outcome -- NOT a weakness and NOT
a target driven green. A series data-point value is a ``TABLE_CELL`` with no
character span, so the unit/value boundary-admission gate
(:func:`carmel.services.dataset_replay.verify_measured_value_unit_boundary`) has
nothing to re-run against it and honestly reports UNVERIFIABLE ("could not
check", never a silent pass). What IS proved, and is the whole point of
restoration: every one of the 68 cited cells was re-derived from the document's
own bytes and matched exactly (``checked_table_cells``), the embedded inventory
reproduced against the raw PDF, and there are ZERO failures. The falsification
test shows the same replay going RED on a single drifted value.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from carmel.schemas.datasets import (
    AxisRole,
    DatasetEnvelope,
    EmbeddedTableInventory,
    TableCellLocator,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.condition_set_producer import TableCellGrounding
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_producer import DatasetProducerError
from carmel.services.dataset_replay import ReplayOutcome, replay_envelope, replay_stored_dataset
from carmel.services.tabular_dataset_producer import (
    TabularAxisSpec,
    TabularDatasetProducerError,
    TabularPointSpec,
    TabularPointValueSpec,
    produce_tabular_envelope_from_artifact,
)
from carmel.services.tabular_dataset_target import (
    PHI_HEADER,
    TARGET_CAMPAIGN,
    TARGET_DOCUMENT_SHA256,
    TARGET_ROWS,
    TARGET_TABLE_KEY,
    TARGET_WORKSPACES_ROOTS,
    build_axes,
    build_embedded_inventory,
    build_envelope,
    locate_target_workspace,
)
from tests.pypdf_gate import require_pypdf

# The single-definition module owns the spec; these thin aliases keep this
# module's test bodies (and their assertions) reading against the same names.
_DOCUMENT_SHA256 = TARGET_DOCUMENT_SHA256
_TABLE_KEY = TARGET_TABLE_KEY
_axes = build_axes
_PHI_HEADER = PHI_HEADER
_ROWS = TARGET_ROWS
_embedded_inventory = build_embedded_inventory
_produce = build_envelope


def _cell_of(embedded: EmbeddedTableInventory, row: int, col: int) -> TableCellGrounding:
    return TableCellGrounding(table_key=_TABLE_KEY, row=row, col=col, inventory=embedded)


def _staged_workspace(tmp_path: Path) -> tuple[Path, bytes]:
    """Copy the target document's literature store into a writable tmp workspace.

    The producer reads the content-addressed store; the store writes the dataset
    envelope into the SAME root; replay reads ``raw.bin`` from it. Copying the one
    literature subtree keeps every step self-contained and never touches the
    operator's real workspace.
    """
    require_pypdf()
    source = locate_target_workspace()
    if source is None:
        roots = ", ".join(str(r / TARGET_CAMPAIGN / "evidence" / "literature") for r in TARGET_WORKSPACES_ROOTS)
        pytest.skip(f"target corpus store is not present under any of: {roots}")
    src_dir = source / "evidence" / "literature" / _DOCUMENT_SHA256
    raw = (src_dir / "raw.bin").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _DOCUMENT_SHA256:
        pytest.skip(f"stored raw.bin is {actual}, not the measured {_DOCUMENT_SHA256}")
    dest_dir = tmp_path / "evidence" / "literature" / _DOCUMENT_SHA256
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir)
    return tmp_path, raw


class TestTheFirstStoredTabularSeries:
    def test_it_stores_and_loads_back_identically(self, tmp_path: Path) -> None:
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        stored = store_dataset_envelope(workspace, env)
        assert load_dataset_envelope(workspace, stored.sha256) == env

    def test_the_phi_coordinate_stays_other_for_want_of_a_printed_unit(self, tmp_path: Path) -> None:
        """I058. The coordinate axis is recorded as OTHER, and this is the honest
        outcome, not a defect to be fixed by declaring EQUIVALENCE_RATIO. What the
        coordinate IS is groundable in prose (caption "range of equivalence ratios";
        nomenclature "phi = equivalence ratio"), but its UNIT is not: phi is
        dimensionless and this column prints no unit token, so the only unit strings
        the units table admits for EQUIVALENCE_RATIO ("1"/"-"/"dimensionless") appear
        nowhere near it. Declaring EQUIVALENCE_RATIO would require grounding a unit the
        column never printed -- a fabricated meaning grounding cannot support. This
        pins OTHER so the question is not reopened a third time; it fails the moment
        the coordinate is silently re-tagged."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        phi = next(axis for axis in env.series[0].axes if axis.axis_id == "phi")
        assert phi.role is AxisRole.COORDINATE
        assert phi.quantity_kind is units.QuantityKind.OTHER
        # The verbatim corrupted header is carried as the label AND the unit -- the
        # honest "this table does not model this quantity" state, not a laundered one.
        assert phi.label_raw == _PHI_HEADER
        for point in env.series[0].points:
            for coord in point.coordinates:
                if coord.axis_id != "phi":
                    continue
                assert coord.value.unit_ref is not None
                # Not just present -- carrying the ACTUAL corrupted header, not a
                # laundered "1"/"-"/"dimensionless": pins that a fabricated-unit
                # regression (silently swapping in a groomed spelling) would fail
                # here, not just at the axis-level label_raw check above.
                assert coord.value.unit_raw == _PHI_HEADER
                assert coord.value.unit_normalized == _PHI_HEADER

    def test_declaring_equivalence_ratio_is_refused_against_the_printed_header(self, tmp_path: Path) -> None:
        """I058, executable. The mechanical reason EQUIVALENCE_RATIO is unreachable:
        grounding its unit at the real header cell -- the only unit this column prints
        -- makes the producer's boundary gate refuse, because "/" (symbol-font phi) is
        not a known unit or alias of equivalence_ratio. This is the boundary guard
        working as intended; the fix is NOT to weaken it, and the honest encoding is
        OTHER. Pinning the refusal keeps a later "just switch it to EQUIVALENCE_RATIO"
        from looking free."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)

        def cell(row: int, col: int) -> TableCellGrounding:
            return _cell_of(embedded, row, col)

        phi_as_equivalence_ratio = TabularAxisSpec(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
            label_quote=_PHI_HEADER,
            unit_quote=_PHI_HEADER,
            label_cell=cell(0, 0),
            unit_cell=cell(0, 0),
        )
        _, s_l_axis = _axes(embedded)
        points = tuple(
            TabularPointSpec(
                point_id=f"r{row:02d}",
                values=(
                    TabularPointValueSpec(axis_id="phi", value_quote=phi, cell=cell(row, 0)),
                    TabularPointValueSpec(axis_id="s_l", value_quote=s_l, cell=cell(row, 1)),
                ),
            )
            for row, phi, s_l in _ROWS
        )
        with pytest.raises(DatasetProducerError, match="not a known unit or alias") as excinfo:
            produce_tabular_envelope_from_artifact(
                workspace,
                sha256=_DOCUMENT_SHA256,
                series_id="flame_speed_sweep",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=(phi_as_equivalence_ratio, s_l_axis),
                points=points,
            )
        message = str(excinfo.value)
        assert "equivalence_ratio" in message
        assert repr(_PHI_HEADER) in message

    def test_every_data_point_value_is_a_table_cell(self, tmp_path: Path) -> None:
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        series = env.series[0]
        assert series.source_form.value == "tabular"
        assert len(series.points) == 22
        for point in series.points:
            for slot in (*point.coordinates, *point.observations):
                assert isinstance(slot.value.value_ref.locator, TableCellLocator)
        # The one deliberate char span is the flame-speed unit, read from prose.
        s_l_units = {type(obs.value.unit_ref.locator).__name__ for point in series.points for obs in point.observations}
        assert s_l_units == {"CharSpanLocator"}

    def test_it_replays_with_cells_re_derived_and_zero_failures(self, tmp_path: Path) -> None:
        """Verifier 2. The stored series replays: the embedded inventory
        reproduces against the raw PDF, all 68 cited cells re-derive and match,
        and there are no failures. The residual UNVERIFIABLE is the designed
        boundary-gate-cannot-run-on-a-cell state, and NOTHING else -- in
        particular the inventory did NOT report extraction_failed (which is the
        synthetic fixture's fate, not a real PDF's)."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        stored = store_dataset_envelope(workspace, env)

        report = replay_stored_dataset(workspace, stored.sha256)

        assert report.evidence_failures == ()
        assert report.checked_table_cells == 68
        assert report.checked_char_spans == 22
        # No inventory finding at all -- the real grid reproduced from the bytes.
        assert not any(f.ref_path.startswith("table_inventories[") for f in report.findings)
        # Every finding that remains is the boundary/admission gate declining to
        # run against a table-cell locator -- honest "could not check", not a pass.
        for finding in report.findings:
            assert finding.category is ReplayOutcome.UNVERIFIABLE
            assert finding.ref_path.endswith(".value_ref") or finding.ref_path.endswith(".unit_ref")
            assert "carries no character span" in finding.reason

    def test_replay_goes_red_on_a_drifted_cell_value_and_clears_when_restored(self, tmp_path: Path) -> None:
        """Verifier 3. Corrupt ONE stored cell value -- the phi=0.5 row's flame
        speed, recorded as 999.9 while its cited cell (row 1, col 1) genuinely
        reads 67.2 -- and replay reports a FAILED finding naming that cell and
        both strings. Restore the true value and the failure clears."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)
        clean = _produce(workspace, embedded)

        # Tamper the stored payload's one value, keeping the (true) inventory and
        # a self-consistent MeasuredValue, then reconstruct and replay.
        payload = clean.identity_payload()
        series = payload["series"][0]
        row1 = next(point for point in series["points"] if point["point_id"] == "r01")
        observation = next(obs for obs in row1["observations"] if obs["axis_id"] == "s_l")
        assert observation["value"]["raw_text"] == "67.2"
        observation["value"]["raw_text"] = "999.9"
        observation["value"]["canonical_decimal_value"] = "999.9"
        tampered = DatasetEnvelope.from_identity_payload(payload)

        red = replay_envelope(workspace, tampered)
        drift = [f for f in red.evidence_failures if f.category is ReplayOutcome.FAILED]
        assert len(drift) == 1, red.findings
        finding = drift[0]
        assert finding.ref_path == "series[0].points[0].observations[0].value.value_ref"
        assert "row=1" in finding.reason
        assert "col=1" in finding.reason
        assert finding.expected == "999.9"
        assert finding.actual == "67.2"

        # Restore: the untampered envelope re-derives clean, zero failures.
        green = replay_envelope(workspace, clean)
        assert green.evidence_failures == ()

    def test_a_cross_row_join_is_refused_though_every_cited_cell_is_real(self, tmp_path: Path) -> None:
        """The fabricated-join guard, on the REAL grid. This point takes phi from
        the row-1 cell (genuinely 0.5) and its flame speed from the row-2 cell
        (genuinely 96.5) -- both cells re-derive from the document's own bytes, so
        with the guard removed the producer would build this series and replay
        would report ZERO failures, certifying a pairing (phi=0.5, S_L=96.5) the
        table never printed on one line. That is exactly the artifact class no
        downstream byte check can catch, and the producer refuses it up front,
        naming the point and both disagreeing rows."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)
        crossed = TabularPointSpec(
            point_id="r01",
            values=(
                TabularPointValueSpec(axis_id="phi", value_quote="0.5", cell=_cell_of(embedded, 1, 0)),
                TabularPointValueSpec(axis_id="s_l", value_quote="96.5", cell=_cell_of(embedded, 2, 1)),
            ),
        )
        with pytest.raises(TabularDatasetProducerError, match="DIFFERENT rows of one table") as excinfo:
            produce_tabular_envelope_from_artifact(
                workspace,
                sha256=_DOCUMENT_SHA256,
                series_id="flame_speed_sweep",
                value_origin=ValueOrigin.EXPERIMENTAL,
                axes=_axes(embedded),
                points=(crossed,),
            )
        message = str(excinfo.value)
        assert "'r01'" in message
        assert "row=1" in message and "row=2" in message
