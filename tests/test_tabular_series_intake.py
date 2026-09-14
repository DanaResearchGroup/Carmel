"""``series_from_classified_grid``: drive the extraction agent over one classified
table and store the resulting series -- or refuse, typed, storing nothing.

Every fixture here is SYNTHETIC (a hand-built grid over a synthetic stored
artifact, via ``tests.table_inventory_fixtures``), driven through
``MockModel`` -- never a live model. The one real-corpus, cell-for-cell
reproduction of the hand-written axis path lives in
``tests.test_tabular_series_intake_acceptance`` (pypdf/corpus-gated); this
module pins the module's OWN typed refusals and the happy path through a
synthetic table small enough to hand-build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carmel.agents.budget import BudgetLedger, session_budget
from carmel.agents.extraction_agent import build_tabular_extraction_agent
from carmel.agents.models import MockModel
from carmel.config import AgentBudgetConfig
from carmel.schemas.datasets import AxisRole, CaptionLabelKey, ValueOrigin
from carmel.services import units
from carmel.services.dataset_bridge import DATASET_STORE_DIR, load_dataset_envelope
from carmel.services.dataset_replay import ReplayFinding, ReplayOutcome, ReplayReport, replay_envelope
from carmel.services.proposal_intake import ProposalIntakeError, current_extraction_text
from carmel.services.table_data_discriminator import DataVerdict, Ground, Polarity, SignalKind, TableClassification
from carmel.services.tabular_series_intake import (
    TabularSeriesIntakeError,
    _render_grid_text,
    series_from_classified_grid,
)
from carmel.services.tabular_series_resolver import TabularSeriesResolutionError
from tests.table_inventory_fixtures import make_embedded_inventory_with_texts
from tests.test_dataset_producer import _store_synthetic_artifact

_PHI_HEADER = "phi"
_S_L_HEADER = "s_l"
_TABLE_KEY = CaptionLabelKey(label="Table 1")

#: Two data rows below one header row, two columns -- just enough grid for the
#: resolver to walk. Column 0 is phi (unit is the header cell, dimensionless);
#: column 1 is s_l (unit is a prose quote, "cm/s", read from the document text).
_CELL_TEXTS: dict[tuple[int, int], str] = {
    (0, 0): _PHI_HEADER,
    (0, 1): _S_L_HEADER,
    (1, 0): "0.8",
    (1, 1): "40.0",
    (2, 0): "1.0",
    (2, 1): "45.0",
}

_DOCUMENT_TEXT = "3. Results\nFlame speeds were measured in cm/s across the sweep reported in Table 1.\n"


@pytest.fixture(autouse=True)
def _reset_session_budget() -> object:
    session_budget().reset()
    yield
    session_budget().reset()


def _ledger(**limits: object) -> BudgetLedger:
    return BudgetLedger(AgentBudgetConfig(**limits))  # type: ignore[arg-type]


def _proposal(*, sha256: str, table_label: str = "Table 1", s_l_header: str = _S_L_HEADER) -> dict[str, object]:
    return {
        "artifact_sha256": sha256,
        "table_label": table_label,
        "series_id": "flame_speed_sweep",
        "value_origin": ValueOrigin.EXPERIMENTAL.value,
        "axes": [
            {
                "axis_id": "phi",
                "role": AxisRole.COORDINATE.value,
                "quantity_kind": units.QuantityKind.OTHER.value,
                "header_quote": _PHI_HEADER,
                "unit": {"kind": "header"},
            },
            {
                "axis_id": "s_l",
                "role": AxisRole.OBSERVATION.value,
                "quantity_kind": units.QuantityKind.VELOCITY.value,
                "header_quote": s_l_header,
                "unit": {"kind": "prose", "unit_quote": "cm/s", "unit_occurrence": 1},
            },
        ],
    }


def _stage(tmp_path: Path) -> str:
    """Store a synthetic artifact and return its sha256; the grid below cites it."""
    return _store_synthetic_artifact(tmp_path, _DOCUMENT_TEXT).sha256


def _inventory(raw_sha256: str):  # noqa: ANN201
    return make_embedded_inventory_with_texts(raw_sha256=raw_sha256, cell_texts=_CELL_TEXTS)


def _measured_classification(*, unmeasured_columns: tuple[int, ...] = ()) -> TableClassification:
    grounds = tuple(
        Ground(kind=SignalKind.NUMERIC_VALUE_COLUMN, polarity=Polarity.NOT_MEASURED, detail="synthetic", columns=(c,))
        for c in unmeasured_columns
    )
    return TableClassification(verdict=DataVerdict.MEASURED, grounds=grounds)


class TestTheHappyPathThroughTheAgent:
    def test_a_mock_agent_proposal_becomes_a_stored_replayable_series(self, tmp_path: Path) -> None:
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification()
        model = MockModel(responses=[_proposal(sha256=sha256)])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        stored = series_from_classified_grid(
            tmp_path,
            agent=agent,
            expected_sha256=sha256,
            table_key=_TABLE_KEY,
            inventory=inventory,
            classification=classification,
            document_text=current_extraction_text(tmp_path, sha256),
        )

        envelope = load_dataset_envelope(tmp_path, stored.sha256)
        series = envelope.series[0]
        assert {p.point_id for p in series.points} == {"row_1", "row_2"}
        stored_dir = tmp_path / DATASET_STORE_DIR
        assert stored_dir.exists() and any(stored_dir.iterdir())
        report = replay_envelope(tmp_path, envelope)
        assert report.evidence_failures == ()


class TestTheNonMeasuredRefusal:
    def test_a_not_measured_verdict_is_refused_before_any_agent_call(self, tmp_path: Path) -> None:
        """Verifier 5(d). No MockModel response queued -- if the gate did not fire
        before the agent runs, the empty queue would raise AgentBridgeError instead,
        proving the refusal really is step 1."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = TableClassification(verdict=DataVerdict.NOT_MEASURED, grounds=())
        agent = build_tabular_extraction_agent(model=MockModel(responses=[]), ledger=_ledger())

        with pytest.raises(TabularSeriesIntakeError, match="not measured"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )

        assert not (tmp_path / DATASET_STORE_DIR).exists()


class TestTheGridIsRenderedVerbatim:
    def test_a_header_with_a_backslash_is_rendered_character_for_character(self) -> None:
        """The persona promises the grid "character for character" and the resolver
        matches ``header_quote`` whole-cell against the true cell text. Rendering a cell
        with ``repr()``/``{!r}`` would wrap it in quotes and double any backslash, so a
        model echoing what it was shown would quote text the cell does not contain and be
        refused. A header carrying a backslash must therefore appear verbatim, and its
        ``repr`` form must not."""
        texts = {(0, 0): r"k\infty", (0, 1): "s_l", (1, 0): "1", (1, 1): "2"}
        inventory = make_embedded_inventory_with_texts(raw_sha256="a" * 64, cell_texts=texts)

        rendered = _render_grid_text(inventory)

        assert r"row 0 col 0: k\infty" in rendered
        assert repr(r"k\infty") not in rendered  # the backslash-doubled, quote-wrapped form


class TestTheUndecidedRefusal:
    def test_an_undecided_verdict_is_refused_and_never_reaches_the_agent(self, tmp_path: Path) -> None:
        """UNDECIDED is a FIRST-CLASS verdict, not a near-miss of NOT_MEASURED:
        ``DataVerdict`` exists precisely so a grid with thin evidence is left for a
        human rather than forced to a binary that "produces confident nonsense". The
        gate admits ONLY MEASURED, so an UNDECIDED grid is refused before the agent
        runs -- ``model.calls`` stays empty -- and nothing is stored. This is the axis
        a MEASURED-gate mutation to ``verdict is NOT_MEASURED`` slips through: it keeps
        refusing NOT_MEASURED while handing UNDECIDED grids to the model, which the
        empty-queue exhaustion and the ``model.calls == []`` assertion both catch."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = TableClassification(verdict=DataVerdict.UNDECIDED, grounds=())
        model = MockModel(responses=[])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        with pytest.raises(TabularSeriesIntakeError, match="undecided"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )

        assert model.calls == []
        assert not (tmp_path / DATASET_STORE_DIR).exists()


class TestTheUnmeasuredColumnRefusal:
    def test_an_observation_axis_citing_an_unmeasured_column_is_refused(self, tmp_path: Path) -> None:
        """Verifier 5(c). Column 1 (s_l) is separately marked unmeasured by the
        classifier; the produced envelope's OWN citation is what is checked, never
        a re-derivation and never ``measured_columns``."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification(unmeasured_columns=(1,))
        model = MockModel(responses=[_proposal(sha256=sha256)])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        with pytest.raises(TabularSeriesIntakeError, match="unmeasured"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )

        assert not (tmp_path / DATASET_STORE_DIR).exists()

    def test_the_coordinate_axis_is_never_gated_even_when_its_column_is_unmeasured(self, tmp_path: Path) -> None:
        """The gate is OBSERVATION-only, by explicit design: marking column 0 (phi,
        a COORDINATE axis) unmeasured must not refuse."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification(unmeasured_columns=(0,))
        model = MockModel(responses=[_proposal(sha256=sha256)])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        stored = series_from_classified_grid(
            tmp_path,
            agent=agent,
            expected_sha256=sha256,
            table_key=_TABLE_KEY,
            inventory=inventory,
            classification=classification,
            document_text=current_extraction_text(tmp_path, sha256),
        )
        assert len(load_dataset_envelope(tmp_path, stored.sha256).series[0].points) == 2


class TestTheReplayBeforeStoreGuard:
    def test_a_dirty_replay_is_refused_and_nothing_is_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation proof for the replay-before-store guard. None of the other
        fixtures in this module can catch dropping the
        ``if report.evidence_failures != ():`` check -- every synthetic grid here
        is built to replay cleanly by construction. So this test fakes
        ``replay_envelope`` itself, returning a report carrying one FAILED
        finding, and checks the guard refuses on it before ever storing."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification()
        model = MockModel(responses=[_proposal(sha256=sha256)])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        dirty_report = ReplayReport(
            checked_char_spans=0,
            total_char_spans=1,
            unchecked_char_spans=1,
            findings=(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path="series[0].points[0]",
                    reason="synthetic drift for mutation-proof",
                ),
            ),
        )
        monkeypatch.setattr(
            "carmel.services.tabular_series_intake.replay_envelope",
            lambda *args, **kwargs: dirty_report,
        )

        with pytest.raises(TabularSeriesIntakeError, match="does not replay cleanly"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )

        assert not (tmp_path / DATASET_STORE_DIR).exists()


class TestCarrierRefusalsPropagateUnchanged:
    def test_a_header_quote_matching_no_column_is_refused(self, tmp_path: Path) -> None:
        """Verifier 5(a)."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification()
        model = MockModel(responses=[_proposal(sha256=sha256, s_l_header="no such header")])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        with pytest.raises(TabularSeriesResolutionError, match="matches no cell"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )
        assert not (tmp_path / DATASET_STORE_DIR).exists()

    def test_a_mismatched_table_label_is_refused(self, tmp_path: Path) -> None:
        """Verifier 5(b)."""
        sha256 = _stage(tmp_path)
        inventory = _inventory(sha256)
        classification = _measured_classification()
        model = MockModel(responses=[_proposal(sha256=sha256, table_label="Table 2")])
        agent = build_tabular_extraction_agent(model=model, ledger=_ledger())

        with pytest.raises(ProposalIntakeError, match="names table 'Table 2'"):
            series_from_classified_grid(
                tmp_path,
                agent=agent,
                expected_sha256=sha256,
                table_key=_TABLE_KEY,
                inventory=inventory,
                classification=classification,
                document_text=current_extraction_text(tmp_path, sha256),
            )
        assert not (tmp_path / DATASET_STORE_DIR).exists()
