"""End-to-end acceptance for the classified-grid -> stored-series step, driven
THROUGH the tabular extraction agent, against the real corpus.

Where :mod:`tests.test_tabular_series_bridge_acceptance` hands a HAND-BUILT
:class:`~carmel.agents.extraction_agent.TabularSeriesProposal` straight to the
carrier, this module drives
:func:`carmel.services.tabular_series_intake.series_from_classified_grid`, which
prompts a :func:`~carmel.agents.extraction_agent.build_tabular_extraction_agent`
(backed here by a :class:`~carmel.agents.models.MockModel`, so no network and no
budget), gates on the grid's classification, and owns storage. It proves the agent
lane -- built and, before this work, never invoked -- reaches the SAME stored
numbers as the deterministic hand-written path.

THE REPRODUCTION IS OF THE NUMBERS, NOT OF THE PHI AXIS SEMANTICS. The general
path reaches every one of the 22 data points at the same cell as
:func:`carmel.services.tabular_dataset_target.build_points`. It does NOT reproduce
that path's phi axis, which declares ``EQUIVALENCE_RATIO`` with a
``unit_not_printed`` unit and a label grounded in the table caption char-span (the
I-060 move recorded in
:func:`carmel.services.tabular_dataset_target.build_axes`). The resolver
(:func:`carmel.services.tabular_series_resolver.resolve_tabular_series`) grounds
every label at the axis's own header cell and offers only ``unit_is_header`` or a
prose unit -- it has no ``unit_not_printed`` path -- so the general path's phi axis
is the plain ``QuantityKind.OTHER`` / header-unit form the agent proposes. The data
reproduces cell-for-cell; that one axis's hand-authored semantics do not, by design.

Same corpus gate as the sibling acceptance module: the paper is non-redistributable,
read from the operator's corpus store at runtime, and every test SKIPS -- never
passes -- when the document (or its store) is absent or is not the measured document.
pypdf-gated too: the grids are derived from PDF geometry.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from carmel.agents.budget import BudgetLedger
from carmel.agents.extraction_agent import (
    ProposedHeaderUnit,
    ProposedProseUnit,
    ProposedTabularAxis,
    TabularSeriesProposal,
    build_tabular_extraction_agent,
)
from carmel.agents.models import MockModel
from carmel.config import AgentBudgetConfig
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
from carmel.services.dataset_replay import ReplayOutcome, replay_stored_dataset
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.general_table_report import CandidateStatus, report_document_tables
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory
from carmel.services.proposal_intake import current_extraction_text
from carmel.services.table_data_discriminator import (
    DataVerdict,
    TableClassification,
    classify_table,
    table_view_from_pdf_payload,
)
from carmel.services.tabular_dataset_target import build_embedded_inventory, build_points
from carmel.services.tabular_series_intake import series_from_classified_grid
from tests.pypdf_gate import require_pypdf

# The measured document (Table 1, page 4 of 10.1115-1.4007737) and one document it
# was never written for -- both live under the operator's syngas corpus store.
_MEASURED_SHA = "c2be41381e3c55671af2912a46d5ce703c0f56cea9dadfba8789e7417059155a"
#: A never-seen document whose four table candidates the general path all refuses
#: (three proposal refusals + one classifier NOT_MEASURED verdict): no grid in it
#: reaches the geometric lane as MEASURED, so the series step never fires. This is
#: the whole corpus's shape outside Table 1 of the measured document -- confirmed by
#: running the report over every syngas document while writing this test.
_UNSEEN_SHA = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"
_SUBPATH = "live-syngas/evidence/literature"
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
_PHI_HEADER = "/"
_S_L_HEADER = "S0L;u(cm/s)"


def _stage(sha: str, tmp_path: Path) -> tuple[Path, bytes]:
    """Copy one corpus document's full evidence directory into ``tmp_path`` (so the
    test never writes into the read-only corpus store) and return the staged
    workspace root and the raw bytes. Skips when the store or document is absent, or
    when the stored bytes are not the document filed under ``sha``."""
    require_pypdf()
    for root in _WORKSPACES_ROOTS:
        src = root / _SUBPATH / sha
        if (src / "raw.bin").exists():
            raw = (src / "raw.bin").read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != sha:
                pytest.skip(f"stored raw.bin is {actual}, not {sha}")
            dest = tmp_path / "evidence" / "literature" / sha
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            return tmp_path, raw
    roots = ", ".join(str(r / _SUBPATH) for r in _WORKSPACES_ROOTS)
    pytest.skip(f"corpus store is not present under any of: {roots}")


def _embedded_and_classification(raw: bytes) -> tuple[EmbeddedTableInventory, TableClassification]:
    """The caller-supplied grid and its classification, both from the same record
    payload -- exactly the two objects the PDF driver hands the step."""
    inventory = build_inventory(extract_fragments(raw), _TABLE_1)
    assert inventory.refusals == () and inventory.complete and len(inventory.cells) == 92
    payload = inventory_record_payload(inventory, raw_sha256=_MEASURED_SHA)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    embedded = EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=_MEASURED_SHA,
        canonical_json=canonical,
    )
    return embedded, classify_table(table_view_from_pdf_payload(payload))


def _proposal_payload() -> dict[str, object]:
    """What the MockModel returns: the two-axis selection the agent would emit for
    Table 1 -- verbatim column headers, roles, quantity kinds, unit forms. phi carries
    no printed unit token, so its unit is the header cell and its kind is OTHER (the
    resolver has no path to the hand-written EQUIVALENCE_RATIO/unit_not_printed form);
    the flame speed's unit 'cm/s' is written in the prose."""
    return TabularSeriesProposal(
        artifact_sha256=_MEASURED_SHA,
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
    ).model_dump(mode="json")


def _agent(responses: list[dict[str, object]]) -> object:
    return build_tabular_extraction_agent(
        model=MockModel(responses=responses),
        ledger=BudgetLedger(AgentBudgetConfig()),
    )


def _drive(workspace: Path, raw: bytes) -> DatasetEnvelope:
    """Drive the agent step and return the STORED series envelope, loaded back from the
    single store the intake owns (it returns the address, not the envelope)."""
    embedded, classification = _embedded_and_classification(raw)
    stored = series_from_classified_grid(
        workspace,
        agent=_agent([_proposal_payload()]),
        expected_sha256=_MEASURED_SHA,
        table_key=_TABLE_KEY,
        inventory=embedded,
        classification=classification,
        document_text=current_extraction_text(workspace, _MEASURED_SHA),
    )
    return load_dataset_envelope(workspace, stored.sha256)


def _cells(env: DatasetEnvelope) -> dict[tuple[str, int, int], str]:
    """{(axis_id, row, col): value text} for every point value of the series."""
    out: dict[tuple[str, int, int], str] = {}
    for point in env.series[0].points:
        for slot in (*point.coordinates, *point.observations):
            locator = slot.value.value_ref.locator
            assert isinstance(locator, TableCellLocator)
            out[(slot.axis_id, locator.row, locator.col)] = slot.value.raw_text
    return out


class TestTheGeneralPathReproducesTheKnownGoodDocument:
    def test_the_agent_driven_step_reaches_the_same_22_points_cell_for_cell(self, tmp_path: Path) -> None:
        """Verifier item 3. Driven through the agent, the step reaches every one of
        the 22 data points at the SAME cell, with the SAME value text, as the
        deterministic hand-written path's ``build_points`` -- coordinate and
        observation alike."""
        workspace, raw = _stage(_MEASURED_SHA, tmp_path)
        env = _drive(workspace, raw)

        series = env.series[0]
        assert series.source_form.value == "tabular"
        assert {p.point_id for p in series.points} == {f"row_{i}" for i in range(1, 23)}

        hand_written = {
            (spec.axis_id, spec.cell.row, spec.cell.col): spec.value_quote
            for point in build_points(build_embedded_inventory(raw))
            for spec in point.values
        }
        assert _cells(env) == hand_written
        # Spot-check the anchor the sibling module also pins: phi=0.5 row's flame speed.
        assert _cells(env)[("s_l", 1, 1)] == "67.2"

    def test_the_stored_series_replays_with_zero_failures(self, tmp_path: Path) -> None:
        """Verifier item 3, replay half. The series the agent path stored re-derives
        against the raw PDF: 68 cited cells, 22 char spans, no failures -- the same
        counts the hand-built sibling proves, reached without a hand-built proposal."""
        workspace, raw = _stage(_MEASURED_SHA, tmp_path)
        env = _drive(workspace, raw)
        stored = store_dataset_envelope(workspace, env)
        report = replay_stored_dataset(workspace, stored.sha256)

        assert report.evidence_failures == ()
        assert report.checked_table_cells == 68
        assert report.checked_char_spans == 22
        for finding in report.findings:
            assert finding.category is ReplayOutcome.UNVERIFIABLE

    def test_the_phi_axis_semantics_diverge_from_the_hand_written_path(self, tmp_path: Path) -> None:
        """The documented divergence, asserted so it is a tested fact, not a comment.
        The general path's phi axis is ``OTHER`` (header unit); the hand-written path
        records it as ``EQUIVALENCE_RATIO`` with a not-printed unit. The NUMBERS above
        reproduce; this ONE axis's hand-authored semantics do not, by design of the
        proposal schema and the resolver."""
        workspace, raw = _stage(_MEASURED_SHA, tmp_path)
        env = _drive(workspace, raw)
        kinds = {axis.axis_id: axis.quantity_kind for axis in env.series[0].axes}
        assert kinds["phi"] is units.QuantityKind.OTHER
        assert kinds["phi"] is not units.QuantityKind.EQUIVALENCE_RATIO
        assert kinds["s_l"] is units.QuantityKind.VELOCITY


class TestTheGeneralPathRunsOnADocumentItWasNeverWrittenFor:
    """Verifier item 4. The general path carries a document written for no test and
    reports a per-candidate outcome for each of its tables. No document in the corpus
    outside the measured document's Table 1 presents a MEASURED grid to this lane, so
    the series step is correctly dormant here -- every candidate refuses, and none is
    stored."""

    def test_every_candidate_gets_a_typed_outcome_and_nothing_is_stored(self, tmp_path: Path) -> None:
        workspace, _ = _stage(_UNSEEN_SHA, tmp_path)
        report = report_document_tables(workspace, _UNSEEN_SHA)

        assert len(report.outcomes) == 4
        statuses: dict[CandidateStatus, int] = {}
        for outcome in report.outcomes:
            statuses[outcome.status] = statuses.get(outcome.status, 0) + 1
            assert outcome.detail, "every candidate must carry a human-readable reason"
        assert statuses == {CandidateStatus.PROPOSAL_REFUSED: 3, CandidateStatus.NOT_MEASURED: 1}
        not_measured = next(o for o in report.outcomes if o.status is CandidateStatus.NOT_MEASURED)
        assert not_measured.verdict is DataVerdict.NOT_MEASURED
        assert report.stored == ()

    def test_supplying_a_series_agent_leaves_it_dormant_when_no_grid_is_measured(self, tmp_path: Path) -> None:
        """The series wiring only invokes the agent on a MEASURED-and-stored grid. A
        never-seen document with no such grid must never reach the agent -- proven by
        passing an agent whose MockModel has NO canned response: were it consulted, it
        would raise; the run completing with nothing stored shows it stayed dormant."""
        workspace, _ = _stage(_UNSEEN_SHA, tmp_path)
        report = report_document_tables(workspace, _UNSEEN_SHA, series_agent=_agent([]))

        assert report.stored == ()
        assert not any(
            o.status in (CandidateStatus.SERIES_STORED, CandidateStatus.SERIES_REFUSED) for o in report.outcomes
        )
