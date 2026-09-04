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

**WHAT REPLAY PROVES HERE, STATED PLAINLY.** ``evidence_outcome`` AND
``overall_outcome`` are both UNVERIFIABLE, and that is the honest, designed
outcome -- NOT a weakness and NOT a target driven green. A series data-point
value is a ``TABLE_CELL`` with no character span, so the value boundary-admission
gate (:func:`carmel.services.dataset_replay.verify_measured_value_value_boundary`)
has nothing to re-run against it and honestly reports UNVERIFIABLE ("could not
check", never a silent pass). Since I060 the phi coordinate declares
EQUIVALENCE_RATIO with its unit marked not-printed-in-source, and the claim that
THIS column is that quantity -- inferred from a caption grounded elsewhere, not
its "/" header -- is filed as an ``UncheckedSemanticClaim`` (the header bridge),
which forces ``overall_outcome`` to UNVERIFIABLE independently. So declaring the
coordinate can never launder a VERIFIED verdict out of dropped unit evidence.
What IS proved, and is the whole point of restoration: every cited cell (45 of
them, down from 68 under the old OTHER "/" encoding, since phi now cites no unit
cell and grounds its label in prose) was re-derived from the document's own bytes
and matched exactly (``checked_table_cells``), the embedded inventory reproduced
against the raw PDF, and there are ZERO failures. The falsification test shows the
same replay going RED on a single drifted value.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from carmel.schemas.datasets import (
    Absent,
    AxisRole,
    CharSpanLocator,
    DatasetEnvelope,
    EmbeddedTableInventory,
    TableCellLocator,
    UnitProvenance,
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
    PHI_LABEL_QUOTE,
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
_PHI_LABEL_QUOTE = PHI_LABEL_QUOTE
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

    def test_the_phi_coordinate_declares_equivalence_ratio_without_a_printed_unit(self, tmp_path: Path) -> None:
        """I060 (rewrite of the I058 test that pinned OTHER). The coordinate axis now
        DECLARES EQUIVALENCE_RATIO -- a reader of the stored data can tell it is an
        equivalence ratio, which OTHER "/" could not convey. The judgement the I058
        test encoded still holds and is what this asserts: the axis NEVER silently
        acquires a unit the document does not print. What changed is how that honesty
        is expressed. Instead of falling back to OTHER with the corrupted header as a
        stand-in unit, the axis marks its unit not-printed-in-source: unit_raw and
        unit_ref are Absent (there is no token to quote or cite), unit_normalized is
        the dimensionless base "1", and unit_provenance says NOT_PRINTED_IN_SOURCE as
        a first-class stored fact. The label is grounded in the caption "range of
        equivalence ratios", not at the "/" header -- so nothing here launders the
        corrupted header into a unit. This fails the moment a printed unit is
        fabricated (unit_raw stops being Absent) or the quantity is silently
        widened."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        phi = next(axis for axis in env.series[0].axes if axis.axis_id == "phi")
        assert phi.role is AxisRole.COORDINATE
        assert phi.quantity_kind is units.QuantityKind.EQUIVALENCE_RATIO
        # The label names the quantity, grounded distally in the caption -- NOT the
        # corrupted "/" header, which named nothing.
        assert phi.label_raw == _PHI_LABEL_QUOTE
        assert phi.label_raw != _PHI_HEADER
        for point in env.series[0].points:
            for coord in point.coordinates:
                if coord.axis_id != "phi":
                    continue
                # The axis never silently acquires a printed unit: there is no unit
                # token (unit_raw/unit_ref Absent), the dimensionless base is "1", and
                # the not-printed marker states plainly that the source printed none.
                # Crucially the corrupted header "/" is NOT laundered in as a unit.
                assert coord.value.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE
                assert isinstance(coord.value.unit_raw, Absent)
                assert isinstance(coord.value.unit_ref, Absent)
                assert coord.value.unit_normalized == "1"

    def test_declaring_equivalence_ratio_by_laundering_the_printed_header_is_refused(self, tmp_path: Path) -> None:
        """I060 (rewrite of the I058 refusal test). The honest way to declare
        EQUIVALENCE_RATIO here is to mark the unit not-printed-in-source (see the test
        above); the DISHONEST way -- grounding the corrupted header "/" as the unit --
        must still be refused, and this pins that it is. "/" (symbol-font phi) is not a
        known unit or alias of equivalence_ratio, so the producer's boundary gate
        refuses it. The new not-printed path does NOT relax this: it declares WITHOUT a
        unit, it never grounds "/" as one. Pinning the refusal keeps a later "just
        launder the header" from looking free."""
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

    def test_it_replays_with_cells_re_derived_and_never_reaches_verified(self, tmp_path: Path) -> None:
        """Verifier 2, and the I060 no-improvement pin. The stored series replays:
        the embedded inventory reproduces against the raw PDF, every cited cell
        re-derives and matches, and there are no failures. Both verdicts stay
        UNVERIFIABLE, exactly as before this ticket -- declaring the coordinate did
        NOT make the data look better-verified. Two things keep it honest: the phi
        VALUE cells are still TABLE_CELL locators the boundary gate cannot re-run
        (evidence_outcome stays UNVERIFIABLE), and the header bridge -- that THIS
        column is equivalence ratio, inferred from a caption grounded elsewhere --
        is filed as an UncheckedSemanticClaim, which forces overall_outcome to
        UNVERIFIABLE independently. So overall_outcome can NEVER reach VERIFIED by
        having dropped the unit evidence. The cell count DROPS from the old OTHER
        encoding (68 -> 45): phi's 22 unit-cell citations and its header-cell label
        are gone, replaced by the not-printed marker and a caption char span."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        stored = store_dataset_envelope(workspace, env)

        report = replay_stored_dataset(workspace, stored.sha256)

        assert report.evidence_failures == ()
        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        # The load-bearing invariant: dropping the unit evidence did not launder a
        # better verdict. overall_outcome is UNVERIFIABLE, never VERIFIED.
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.checked_table_cells == 45
        assert report.checked_char_spans == 23
        # Exactly one header-bridge claim, on the phi axis's quantity_kind, supported
        # by its distal caption label_ref -- this is what bars VERIFIED.
        assert len(report.unchecked_semantic_claims) == 1
        bridge = report.unchecked_semantic_claims[0]
        assert bridge.claim_path == "series['flame_speed_sweep'].axes['phi'].quantity_kind"
        assert bridge.claim == "equivalence_ratio"
        assert bridge.support_paths == ("series['flame_speed_sweep'].axes['phi'].label_ref",)
        # No inventory finding at all -- the real grid reproduced from the bytes.
        assert not any(f.ref_path.startswith("table_inventories[") for f in report.findings)
        # Every finding that remains is the boundary/admission gate declining to
        # run against a table-cell VALUE locator -- honest "could not check", not a
        # pass. phi's un-runnable UNIT-cell findings are gone: there is no unit.
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


class TestUnitNotPrintedDeclarationGate:
    """I060. The path that lets EQUIVALENCE_RATIO be declared without a printed unit
    is gated NARROWLY at TabularAxisSpec construction: the three sibling dimensionless
    kinds are refused (they carry scale, so an un-printed "1" would corrupt a
    magnitude), and a declaration with no groundable label is refused. These need no
    corpus -- the gate is a construction-time check."""

    def test_unit_not_printed_refused_for_mole_fraction(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="permitted for quantity_kind='equivalence_ratio' ONLY"):
            TabularAxisSpec(
                axis_id="x",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.MOLE_FRACTION,
                label_quote="mole fraction",
                unit_not_printed=True,
            )

    def test_unit_not_printed_refused_for_mass_fraction(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="permitted for quantity_kind='equivalence_ratio' ONLY"):
            TabularAxisSpec(
                axis_id="x",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.MASS_FRACTION,
                label_quote="mass fraction",
                unit_not_printed=True,
            )

    def test_unit_not_printed_refused_for_relative_uncertainty(self) -> None:
        with pytest.raises(TabularDatasetProducerError, match="permitted for quantity_kind='equivalence_ratio' ONLY"):
            TabularAxisSpec(
                axis_id="x",
                role=AxisRole.OBSERVATION,
                quantity_kind=units.QuantityKind.RELATIVE_UNCERTAINTY,
                label_quote="relative uncertainty",
                unit_not_printed=True,
            )

    def test_declaring_without_a_grounded_label_is_refused(self) -> None:
        # Verifier 3: the quantity may only be declared where a label naming it is
        # groundable. An empty label_quote with unit_not_printed is refused.
        with pytest.raises(TabularDatasetProducerError, match="requires a groundable label_quote"):
            TabularAxisSpec(
                axis_id="phi",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
                label_quote="",
                unit_not_printed=True,
            )

    def test_header_bridge_claim_filed_only_when_the_label_is_not_the_column_header(self, tmp_path: Path) -> None:
        """Verifier 4. Two EQUIVALENCE_RATIO axes declared without a printed unit,
        differing ONLY in where the label is grounded. When the label IS the column's
        own printed header cell (Case A), the header names the quantity and NO bridge
        claim is filed. When it is grounded distally -- the flagship's caption char
        span (Case B) -- the "this column is equivalence ratio" step is an inference
        and replay files an UncheckedSemanticClaim. The difference is visible in the
        stored data: the label_ref locator is a table cell in Case A, a char span in
        Case B."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)

        def cell(row: int, col: int) -> TableCellGrounding:
            return _cell_of(embedded, row, col)

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

        # Case A: label grounded at phi's OWN column header cell (col 0, row 0), unit
        # not printed. The header is structurally attached to the column, so no bridge.
        phi_header_label = TabularAxisSpec(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
            label_quote=_PHI_HEADER,
            label_cell=cell(0, 0),
            unit_not_printed=True,
        )
        env_a = produce_tabular_envelope_from_artifact(
            workspace,
            sha256=_DOCUMENT_SHA256,
            series_id="flame_speed_sweep",
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=(phi_header_label, s_l_axis),
            points=points,
        )
        stored_a = store_dataset_envelope(workspace, env_a)
        report_a = replay_stored_dataset(workspace, stored_a.sha256)
        phi_a = next(axis for axis in env_a.series[0].axes if axis.axis_id == "phi")
        assert isinstance(phi_a.label_ref.locator, TableCellLocator)
        assert report_a.unchecked_semantic_claims == ()

        # Case B: the flagship -- label grounded distally in the caption char span.
        env_b = _produce(workspace, embedded)
        stored_b = store_dataset_envelope(workspace, env_b)
        report_b = replay_stored_dataset(workspace, stored_b.sha256)
        phi_b = next(axis for axis in env_b.series[0].axes if axis.axis_id == "phi")
        assert isinstance(phi_b.label_ref.locator, CharSpanLocator)
        assert len(report_b.unchecked_semantic_claims) == 1
        assert (
            report_b.unchecked_semantic_claims[0].claim_path == "series['flame_speed_sweep'].axes['phi'].quantity_kind"
        )
        # The visible-in-stored-data difference: the label_ref locator KIND differs.
        assert type(phi_a.label_ref.locator) is not type(phi_b.label_ref.locator)
