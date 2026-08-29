"""End-to-end acceptance: an AGENT-PROPOSED tabular series becomes a stored,
replayable series, over a REAL stored table -- through the proposal -> carrier ->
producer bridge, not by hand.

Where :mod:`tests.test_tabular_dataset_target_acceptance` hand-writes every
``(row, phi, S_L)`` tuple and every cell address, this module writes NONE of
them. It hands
:func:`carmel.services.proposal_intake.tabular_series_from_proposal` a
:class:`~carmel.agents.extraction_agent.TabularSeriesProposal` that names the
table and, per axis, the verbatim column header -- and the deterministic resolver
locates each column, walks the 22 data rows, and emits every coordinate and every
join. That hand-off (a human writing every coordinate) is exactly what the bridge
removes; this test proves it removed.

Same table and same corpus gate as the sibling acceptance module: Table 1, page 4
of ``10.1115-1.4007737`` (header row ``['/', 'S0L;u(cm/s)', 'USL(cm/s)',
'Percent']``, rows 1..22 all-numeral). The paper is non-redistributable, so it is
read from the operator's corpus store at runtime and every test SKIPS -- never
passes -- when the document (or its store) is absent or is not byte-for-byte the
measured document. pypdf-gated too: the grid is derived from PDF geometry.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from carmel.agents.extraction_agent import (
    ProposedHeaderUnit,
    ProposedProseUnit,
    ProposedTabularAxis,
    TabularSeriesProposal,
)
from carmel.paths import default_workspaces_root
from carmel.schemas.datasets import (
    AxisRole,
    CaptionLabelKey,
    DatasetEnvelope,
    EmbeddedTableInventory,
    TableCellLocator,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_replay import ReplayOutcome, replay_envelope, replay_stored_dataset
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory
from carmel.services.proposal_intake import ProposalIntakeError, tabular_series_from_proposal
from carmel.services.tabular_series_resolver import TabularSeriesResolutionError
from tests.pypdf_gate import require_pypdf

_DOCUMENT_SHA256 = "c2be41381e3c55671af2912a46d5ce703c0f56cea9dadfba8789e7417059155a"
_DOCUMENT_SUBPATH = "live-syngas/evidence/literature"
_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")

_TABLE_1 = ClaimedFootprint(
    page=4,
    x_start=305.0,
    x_end=555.0,
    y_top=745.0,
    y_bottom=520.0,
    caption_text="rangeofequivalenceratios",
    caption_x_start=311.981,
    caption_baseline_y=750.274,
)
_TABLE_KEY = CaptionLabelKey(label="Table 1")

#: The verbatim column headers, as the grid decodes them -- the ONLY spatial thing
#: the proposal below asserts. phi's symbol-font header decodes to "/"; the flame
#: speed carries its subscript fold. Recorded as-is (grounding proves location).
_PHI_HEADER = "/"
_S_L_HEADER = "S0L;u(cm/s)"


def _locate_workspace() -> Path | None:
    for root in _WORKSPACES_ROOTS:
        if (root / _DOCUMENT_SUBPATH / _DOCUMENT_SHA256 / "raw.bin").exists():
            return root / _DOCUMENT_SUBPATH.split("/")[0]
    return None


def _staged_workspace(tmp_path: Path) -> tuple[Path, bytes]:
    require_pypdf()
    source = _locate_workspace()
    if source is None:
        roots = ", ".join(str(r / _DOCUMENT_SUBPATH) for r in _WORKSPACES_ROOTS)
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


def _embedded_inventory(raw: bytes) -> EmbeddedTableInventory:
    """The caller-supplied grid -- built exactly as the production path does, from the
    real PDF via ``build_inventory``. Table discovery (out of scope for the bridge)
    owns this step; the bridge is handed its result."""
    inventory = build_inventory(extract_fragments(raw), _TABLE_1)
    assert inventory.refusals == (), f"the target grid refused: {inventory.refusals}"
    assert inventory.complete and len(inventory.cells) == 92
    payload = inventory_record_payload(inventory, raw_sha256=_DOCUMENT_SHA256)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=_DOCUMENT_SHA256,
        canonical_json=canonical,
    )


def _proposal() -> TabularSeriesProposal:
    """What the agent PROPOSES -- table label and two column-header quotes, nothing
    spatial beyond that. No row, no column, no tuple: the resolver derives them."""
    return TabularSeriesProposal(
        artifact_sha256=_DOCUMENT_SHA256,
        table_label="Table 1",
        series_id="flame_speed_sweep",
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=[
            ProposedTabularAxis(
                axis_id="phi",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.OTHER,
                header_quote=_PHI_HEADER,
                unit=ProposedHeaderUnit(),
            ),
            ProposedTabularAxis(
                axis_id="s_l",
                role=AxisRole.OBSERVATION,
                quantity_kind=units.QuantityKind.VELOCITY,
                header_quote=_S_L_HEADER,
                unit=ProposedProseUnit(unit_quote="cm/s", unit_occurrence=1),
            ),
        ],
    )


def _produce(workspace: Path, embedded: EmbeddedTableInventory) -> DatasetEnvelope:
    return tabular_series_from_proposal(
        workspace,
        _proposal(),
        expected_sha256=_DOCUMENT_SHA256,
        table_key=_TABLE_KEY,
        inventory=embedded,
    )


class TestAnAgentProposedSeriesBecomesAStoredSeries:
    def test_it_produces_stores_and_loads_back_identically(self, tmp_path: Path) -> None:
        """Verifier 1. The proposal, through the production path, becomes a stored
        series: 22 points, one per data row, every value cell-addressed."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        series = env.series[0]
        assert series.source_form.value == "tabular"
        assert len(series.points) == 22
        # One point per data row 1..22, each id naming the row it was walked from.
        assert {p.point_id for p in series.points} == {f"row_{i}" for i in range(1, 23)}
        for point in series.points:
            for slot in (*point.coordinates, *point.observations):
                assert isinstance(slot.value.value_ref.locator, TableCellLocator)
        stored = store_dataset_envelope(workspace, env)
        assert load_dataset_envelope(workspace, stored.sha256) == env

    def test_it_replays_with_every_cell_re_derived_and_zero_failures(self, tmp_path: Path) -> None:
        """Verifier 2. The stored series replays: the embedded inventory reproduces
        against the raw PDF, all 68 cited cells re-derive and match, and there are no
        failures. The residual UNVERIFIABLE is the designed boundary-gate-cannot-run-
        on-a-cell state -- honest 'could not check', never a pass."""
        workspace, raw = _staged_workspace(tmp_path)
        env = _produce(workspace, _embedded_inventory(raw))
        stored = store_dataset_envelope(workspace, env)

        report = replay_stored_dataset(workspace, stored.sha256)

        assert report.evidence_failures == ()
        assert report.checked_table_cells == 68
        assert report.checked_char_spans == 22
        assert not any(f.ref_path.startswith("table_inventories[") for f in report.findings)
        for finding in report.findings:
            assert finding.category is ReplayOutcome.UNVERIFIABLE
            assert finding.ref_path.endswith(".value_ref") or finding.ref_path.endswith(".unit_ref")
            assert "carries no character span" in finding.reason

    def test_replay_goes_red_on_a_drifted_cell_value_and_clears_when_restored(self, tmp_path: Path) -> None:
        """Verifier 3. Corrupt ONE stored cell value -- the phi=0.5 row's flame speed,
        recorded as 999.9 while its cited cell (row 1, col 1) genuinely reads 67.2 --
        and replay reports a FAILED finding naming that cell. Restore, and it clears."""
        workspace, raw = _staged_workspace(tmp_path)
        clean = _produce(workspace, _embedded_inventory(raw))

        payload = clean.identity_payload()
        series = payload["series"][0]
        row1 = next(point for point in series["points"] if point["point_id"] == "row_1")
        observation = next(obs for obs in row1["observations"] if obs["axis_id"] == "s_l")
        assert observation["value"]["raw_text"] == "67.2"
        observation["value"]["raw_text"] = "999.9"
        observation["value"]["canonical_decimal_value"] = "999.9"
        tampered = DatasetEnvelope.from_identity_payload(payload)

        red = replay_envelope(workspace, tampered)
        drift = [f for f in red.evidence_failures if f.category is ReplayOutcome.FAILED]
        assert len(drift) == 1, red.findings
        finding = drift[0]
        assert "row=1" in finding.reason and "col=1" in finding.reason
        assert finding.expected == "999.9"
        assert finding.actual == "67.2"

        green = replay_envelope(workspace, clean)
        assert green.evidence_failures == ()


class TestTheBridgeRefusesOnTheRealGrid:
    """Verifier 4 + the mis-selection gates, asserted against the REAL grid rather
    than a synthetic one -- so the refusals are shown to fire on the document the
    end-to-end path runs on, not only on a fixture."""

    def test_a_header_quote_matching_no_column_is_refused(self, tmp_path: Path) -> None:
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)
        proposal = _proposal()
        proposal.axes[1].header_quote = "no such header"
        with pytest.raises(TabularSeriesResolutionError, match="matches no cell"):
            tabular_series_from_proposal(
                workspace, proposal, expected_sha256=_DOCUMENT_SHA256, table_key=_TABLE_KEY, inventory=embedded
            )

    def test_naming_a_different_table_than_the_grid_is_refused(self, tmp_path: Path) -> None:
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)
        proposal = _proposal()
        proposal.table_label = "Table 2"
        with pytest.raises(ProposalIntakeError, match="names table 'Table 2'"):
            tabular_series_from_proposal(
                workspace, proposal, expected_sha256=_DOCUMENT_SHA256, table_key=_TABLE_KEY, inventory=embedded
            )

    def test_a_mismatched_document_sha_is_refused_before_any_cell(self, tmp_path: Path) -> None:
        workspace, raw = _staged_workspace(tmp_path)
        embedded = _embedded_inventory(raw)
        with pytest.raises(ProposalIntakeError, match="does not match the document"):
            tabular_series_from_proposal(
                workspace, _proposal(), expected_sha256="b" * 64, table_key=_TABLE_KEY, inventory=embedded
            )
