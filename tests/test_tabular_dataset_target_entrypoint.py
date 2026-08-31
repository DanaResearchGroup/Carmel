"""The durable production entry point for the first tabular dataset.

Covers :func:`carmel.services.tabular_dataset_target.produce_and_store_target`
and its ``carmel store-tabular-dataset`` CLI wrapper: the same production path the
acceptance test proves, but writing a durable artifact into a workspace's own
dataset store and a human-readable export into its ``reports/`` dir.

Corpus- and pypdf-gated exactly like
:mod:`tests.test_tabular_dataset_target_acceptance`: the paper is
non-redistributable, so the tests that build the real series SKIP -- never fail --
when the document (or a real pypdf) is absent. The absence-handling test needs no
corpus and always runs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

import Carmel
from carmel.services.dataset_replay import ReplayOutcome, replay_stored_dataset
from carmel.services.dataset_store import DATASET_STORE_DIR
from carmel.services.tabular_dataset_target import (
    TARGET_CAMPAIGN,
    TARGET_DOCUMENT_SHA256,
    TabularDatasetTargetError,
    locate_target_workspace,
    produce_and_store_target,
    read_target_raw,
    render_series_text,
    write_series_export,
)
from tests.pypdf_gate import require_pypdf


def _stage_campaign(tmp_path: Path) -> Path:
    """Copy the target document into ``tmp_path/<campaign>`` and return the root.

    Returns the PARENT workspaces root (``tmp_path``); the campaign workspace is
    ``tmp_path/<TARGET_CAMPAIGN>``. Never touches the operator's real workspace.
    """
    require_pypdf()
    source = locate_target_workspace()
    if source is None:
        pytest.skip("target corpus store is not present")
    src_dir = source / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
    dest_dir = tmp_path / TARGET_CAMPAIGN / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir)
    return tmp_path


def _count_data_rows(text: str) -> int:
    return len(re.findall(r"^\s+r\d\d\s", text, flags=re.MULTILINE))


class TestProduceAndStoreTarget:
    def test_it_writes_a_durable_envelope_that_replays_green(self, tmp_path: Path) -> None:
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN

        stored = produce_and_store_target(workspace)

        # The artifact is durable: it lives in the workspace's dataset store and
        # is still there after the call returns.
        assert stored.path.exists()
        assert stored.path == workspace / DATASET_STORE_DIR / f"{stored.sha256}.json"

        # It replays from disk: every cited cell re-derived, zero failures. Since
        # I060 the phi coordinate declares EQUIVALENCE_RATIO with its unit
        # not-printed, so 45 cells are cited (down from 68 under OTHER "/"), and the
        # header bridge is filed as an UncheckedSemanticClaim -- overall_outcome
        # stays UNVERIFIABLE, never VERIFIED, so declaring the coordinate never
        # laundered a better verdict.
        report = replay_stored_dataset(workspace, stored.sha256)
        assert report.evidence_failures == ()
        assert report.checked_table_cells == 45
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert len(report.unchecked_semantic_claims) == 1

    def test_the_export_shows_the_actual_numbers(self, tmp_path: Path) -> None:
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        stored = produce_and_store_target(workspace)

        export_path = write_series_export(workspace, stored)
        assert export_path.exists()
        assert export_path == workspace / "reports" / f"dataset-{stored.sha256}.txt"

        text = export_path.read_text(encoding="utf-8")
        assert _count_data_rows(text) == 22
        for token in ("0.5", "5.0", "67.2", "110.1", "cm/s", TARGET_DOCUMENT_SHA256):
            assert token in text, token

    def test_render_is_derived_purely_from_the_stored_envelope(self, tmp_path: Path) -> None:
        # Loading the stored envelope back and rendering it gives the same text as
        # rendering the in-memory one: the export cannot drift from what was stored.
        from carmel.services.dataset_bridge import load_dataset_envelope

        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        stored = produce_and_store_target(workspace)
        reloaded = load_dataset_envelope(workspace, stored.sha256)
        assert render_series_text(reloaded) == render_series_text(stored.envelope)

    def test_render_reads_the_table_label_and_page_from_the_envelope_not_the_module_constants(
        self, tmp_path: Path
    ) -> None:
        # A stored envelope pins its OWN header facts. If a future edit changes
        # TARGET_TABLE_KEY/TARGET_TABLE_FOOTPRINT, an envelope stored under the old
        # values must still render with the OLD label and page -- never today's
        # constants -- or render_series_text's "cannot drift from what was grounded"
        # is a lie. Build an envelope citing a table under a label and page that
        # deliberately differ from both module constants, and prove the render shows
        # the envelope's facts, not the constants'. (Sibling of the condition-set
        # target's identically-named test.)
        import hashlib

        from carmel.schemas.datasets import (
            AxisRole,
            CaptionLabelKey,
            EmbeddedTableInventory,
            ValueOrigin,
        )
        from carmel.services import units
        from carmel.services.condition_set_producer import TableCellGrounding
        from carmel.services.dataset_store import canonical_json_bytes
        from carmel.services.pdf_fragments import extract_fragments
        from carmel.services.pdf_table_record import inventory_record_payload
        from carmel.services.pdf_tables import build_inventory
        from carmel.services.tabular_dataset_producer import (
            TabularAxisSpec,
            TabularPointSpec,
            TabularPointValueSpec,
            produce_tabular_envelope_from_artifact,
        )
        from carmel.services.tabular_dataset_target import (
            PHI_HEADER,
            S_L_HEADER,
            TARGET_ROWS,
            TARGET_SERIES_ID,
            TARGET_TABLE_FOOTPRINT,
            TARGET_TABLE_KEY,
        )

        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        raw = read_target_raw(workspace)

        # Derive against the REAL footprint (which anchors correctly against the
        # document), then relabel the page in the resulting record --
        # EmbeddedTableInventory validates only the record's OWN self-coherence, not
        # that it matches a real PDF page, which is exactly what lets a stored
        # envelope's cited page differ honestly from today's constants.
        inventory = build_inventory(extract_fragments(raw), TARGET_TABLE_FOOTPRINT)
        payload = inventory_record_payload(inventory, raw_sha256=TARGET_DOCUMENT_SHA256)
        payload["footprint"]["page"] = 99
        canonical_json = canonical_json_bytes(payload).decode("utf-8")
        embedded = EmbeddedTableInventory(
            inventory_sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
            raw_sha256=TARGET_DOCUMENT_SHA256,
            canonical_json=canonical_json,
        )
        differing_key = CaptionLabelKey(label="Table 99 (test double)")

        def cell(row: int, col: int) -> TableCellGrounding:
            return TableCellGrounding(table_key=differing_key, row=row, col=col, inventory=embedded)

        axes = (
            TabularAxisSpec(
                axis_id="phi",
                role=AxisRole.COORDINATE,
                quantity_kind=units.QuantityKind.OTHER,
                label_quote=PHI_HEADER,
                unit_quote=PHI_HEADER,
                label_cell=cell(0, 0),
                unit_cell=cell(0, 0),
            ),
            TabularAxisSpec(
                axis_id="s_l",
                role=AxisRole.OBSERVATION,
                quantity_kind=units.QuantityKind.VELOCITY,
                label_quote=S_L_HEADER,
                unit_quote="cm/s",
                label_cell=cell(0, 1),
                unit_occurrence=1,
            ),
        )
        points = tuple(
            TabularPointSpec(
                point_id=f"r{row:02d}",
                values=(
                    TabularPointValueSpec(axis_id="phi", value_quote=phi, cell=cell(row, 0)),
                    TabularPointValueSpec(axis_id="s_l", value_quote=s_l, cell=cell(row, 1)),
                ),
            )
            for row, phi, s_l in TARGET_ROWS
        )
        envelope = produce_tabular_envelope_from_artifact(
            workspace,
            sha256=TARGET_DOCUMENT_SHA256,
            series_id=TARGET_SERIES_ID,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=axes,
            points=points,
        )

        text = render_series_text(envelope)
        assert "Source table   : Table 99 (test double), page 99" in text
        assert TARGET_TABLE_KEY.label not in text
        assert f"page {TARGET_TABLE_FOOTPRINT.page}" not in text


class TestStoreTabularDatasetCommand:
    def test_it_stores_exports_and_prints_the_table(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _stage_campaign(tmp_path)
        workspace = root / TARGET_CAMPAIGN

        exit_code = Carmel.main(["store-tabular-dataset", "--workspaces", str(root)])
        assert exit_code == 0

        out = capsys.readouterr().out
        assert _count_data_rows(out) == 22
        for token in ("0.5", "5.0", "67.2", "110.1", "cm/s"):
            assert token in out, token

        datasets = list((workspace / DATASET_STORE_DIR).glob("*.json"))
        assert len(datasets) == 1
        reports = list((workspace / "reports").glob("dataset-*.txt"))
        assert len(reports) == 1

    def test_it_reports_and_stops_when_the_document_is_absent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No corpus needed: an empty parent root has no campaign holding the
        # document, so the command must refuse cleanly (exit 1), never crash and
        # never write anything.
        exit_code = Carmel.main(["store-tabular-dataset", "--workspaces", str(tmp_path)])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Refusing to store the tabular dataset" in err
        assert not (tmp_path / TARGET_CAMPAIGN / DATASET_STORE_DIR).exists()

    def test_it_reports_stored_artifact_when_the_export_cannot_be_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No corpus needed: the envelope is produced and stored (mocked), then the
        # export write fails. The command must exit non-zero but name the stored
        # artifact on stderr -- an operator reading only stderr must not conclude
        # the data was lost.
        from carmel.services import tabular_dataset_target as tgt

        stored = tgt.StoredTargetDataset(
            sha256="deadbeef",
            path=tmp_path / TARGET_CAMPAIGN / DATASET_STORE_DIR / "deadbeef.json",
            envelope=None,  # type: ignore[arg-type]  # unused on the failure path
        )
        monkeypatch.setattr(tgt, "produce_and_store_target", lambda *a, **k: stored)

        def _boom(*args: object, **kwargs: object) -> Path:
            raise OSError("reports/ is read-only")

        monkeypatch.setattr(tgt, "write_series_export", _boom)

        exit_code = Carmel.main(["store-tabular-dataset", "--workspaces", str(tmp_path)])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Stored the dataset envelope durably" in err
        assert "deadbeef" in err
        assert "could not write the human-readable export" in err

    def test_it_reports_and_stops_when_no_workspace_is_discovered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With no --workspaces and discovery finding nothing, the command reports
        # the missing document and stops rather than raising.
        monkeypatch.setattr(
            "carmel.services.tabular_dataset_target.locate_target_workspace",
            lambda *a, **k: None,
        )
        exit_code = Carmel.main(["store-tabular-dataset"])
        assert exit_code == 1
        assert "not stored in any known workspace" in capsys.readouterr().err


class TestFailClosedPreconditions:
    """The corpus-free fail-closed paths: no document needed to reach them."""

    def test_locate_returns_none_when_no_root_holds_the_document(self, tmp_path: Path) -> None:
        assert locate_target_workspace(roots=(tmp_path,)) is None

    def test_read_target_raw_refuses_a_missing_document(self, tmp_path: Path) -> None:
        with pytest.raises(TabularDatasetTargetError, match="not stored under"):
            read_target_raw(tmp_path)

    def test_read_target_raw_refuses_bytes_that_are_not_the_measured_document(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
        raw_dir.mkdir(parents=True)
        (raw_dir / "raw.bin").write_bytes(b"not the measured document")
        with pytest.raises(TabularDatasetTargetError, match="not the measured"):
            read_target_raw(tmp_path)

    def test_read_target_raw_refuses_when_the_bytes_cannot_be_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # raw.bin is present (passes the exists() check) but reading it fails: an
        # unreadable document is a named refusal, not a raw OSError traceback.
        raw_dir = tmp_path / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
        raw_dir.mkdir(parents=True)
        (raw_dir / "raw.bin").write_bytes(b"")

        def _boom(self: Path, *args: object, **kwargs: object) -> bytes:
            raise OSError("disk gone")

        monkeypatch.setattr(Path, "read_bytes", _boom)
        with pytest.raises(TabularDatasetTargetError, match="cannot read the stored raw.bin"):
            read_target_raw(tmp_path)
