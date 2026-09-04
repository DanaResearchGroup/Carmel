"""The durable production entry point for the first condition set.

Covers :func:`carmel.services.condition_set_target.produce_and_store_target`
and its ``carmel store-condition-set`` CLI wrapper: the same production path the
acceptance test proves, but writing a durable artifact into a workspace's own
condition-set store and a human-readable export into its ``reports/`` dir.

Corpus- and pypdf-gated exactly like
:mod:`tests.test_condition_set_target_acceptance`: the paper is
non-redistributable, so the tests that build the real condition set SKIP -- never
fail -- when the document (or a real pypdf) is absent. The absence-handling tests
need no corpus and always run.

The stored artifact's honest replay outcome is ``overall_outcome=UNVERIFIABLE``
with ``evidence_outcome=VERIFIED`` -- asserted as it truly is, not driven green:
the attribution span is support-only, and no unit grounds the temperature row.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

import Carmel
from carmel.schemas.datasets import CaptionLabelKey, EmbeddedTableInventory
from carmel.services.condition_set_bridge import load_condition_set_envelope
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    DeviceClassSpec,
    TableCellGrounding,
    produce_condition_set_from_artifact,
)
from carmel.services.condition_set_target import (
    ATTRIBUTION_QUOTE,
    SUBJECT_OCCURRENCE,
    SUBJECT_QUOTE,
    TARGET_ATTRIBUTION,
    TARGET_CAMPAIGN,
    TARGET_DOCUMENT_SHA256,
    TARGET_TABLE_FOOTPRINT,
    TARGET_TABLE_KEY,
    ConditionSetTargetError,
    locate_target_workspace,
    produce_and_store_target,
    read_target_raw,
    render_condition_set_text,
    write_condition_set_export,
)
from carmel.services.dataset_replay import ReplayOutcome, replay_stored_condition_set
from carmel.services.dataset_store import CONDITION_SET_STORE_DIR, canonical_json_bytes
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_tables import build_inventory
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


class TestProduceAndStoreTarget:
    def test_it_writes_a_durable_envelope_that_replays_honestly(self, tmp_path: Path) -> None:
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN

        stored = produce_and_store_target(workspace)

        # The artifact is durable: it lives in the workspace's condition-set store
        # and is still there after the call returns.
        assert stored.path.exists()
        assert stored.path == workspace / CONDITION_SET_STORE_DIR / f"{stored.sha256}.json"

        # It replays from disk with the honest outcome the acceptance test pins:
        # VERIFIED evidence under an UNVERIFIABLE overall verdict (the attribution
        # span is support-only), every cited cell re-derived, zero evidence failures.
        report = replay_stored_condition_set(workspace, stored.sha256)
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.evidence_failures == ()
        assert report.checked_table_cells == 26

    def test_the_export_shows_the_actual_conditions(self, tmp_path: Path) -> None:
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        stored = produce_and_store_target(workspace)

        export_path = write_condition_set_export(workspace, stored)
        assert export_path.exists()
        assert export_path == workspace / "reports" / f"condition-set-{stored.sha256}.txt"

        text = export_path.read_text(encoding="utf-8")
        for token in ("Fuel", "H2/CO(50:50%)", "Oxidizer", "Air", "φ", "0.6–1.0", "P(atm)", TARGET_DOCUMENT_SHA256):
            assert token in text, token
        # The deliberately-omitted temperature row is NOT in the stored conditions.
        assert "T(°C)" not in text

    def test_render_is_derived_purely_from_the_stored_envelope(self, tmp_path: Path) -> None:
        # Loading the stored envelope back and rendering it gives the same text as
        # rendering the in-memory one: the export cannot drift from what was stored.
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        stored = produce_and_store_target(workspace)
        reloaded = load_condition_set_envelope(workspace, stored.sha256)
        assert render_condition_set_text(reloaded) == render_condition_set_text(stored.envelope)

    def test_render_reads_the_table_label_and_page_from_the_envelope_not_the_module_constants(
        self, tmp_path: Path
    ) -> None:
        # A stored envelope pins its OWN header facts. If a future edit changes
        # TARGET_TABLE_KEY/TARGET_TABLE_FOOTPRINT, an envelope stored under the old
        # values must still render with the OLD label and page -- never today's
        # constants -- or the docstring's "cannot drift from what was grounded" is
        # a lie. Build an envelope citing a table under a label and page that
        # deliberately differ from both module constants, and prove the render
        # shows the envelope's facts, not the constants'.
        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        raw = read_target_raw(workspace)

        # The footprint's caption anchor is checked against the REAL document, so a
        # fabricated page would refuse derivation. Derive against the real footprint
        # (which anchors correctly), then relabel the page in the resulting record --
        # EmbeddedTableInventory only validates the record's OWN self-coherence, not
        # that it matches a real PDF page, which is exactly what makes it possible for
        # a stored envelope's cited page to differ honestly from today's constants.
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

        envelope = produce_condition_set_from_artifact(
            workspace,
            sha256=TARGET_DOCUMENT_SHA256,
            attribution=TARGET_ATTRIBUTION,
            attribution_quote=ATTRIBUTION_QUOTE,
            subject=DeviceClassSpec(label_quote=SUBJECT_QUOTE, label_occurrence=SUBJECT_OCCURRENCE),
            categoricals=(
                CategoricalConditionSpec(
                    claim_id="cat0_fuel_c1",
                    label_quote="Fuel",
                    token_quote="H2/CO(50:50%)",
                    label_cell=cell(0, 0),
                    token_cell=cell(0, 1),
                ),
            ),
        )

        text = render_condition_set_text(envelope)
        assert "Source table   : Table 99 (test double), page 99" in text
        assert TARGET_TABLE_KEY.label not in text
        assert f"page {TARGET_TABLE_FOOTPRINT.page}" not in text

    def test_stored_table_reference_degrades_the_page_to_unknown_on_a_corrupt_embedded_inventory(
        self, tmp_path: Path
    ) -> None:
        # The shared stored_table_reference (carmel.services._envelope_render, reached here
        # under this slice's local name) promises the page falls back to "unknown" rather
        # than crash. Until I-071 this slice carried an UNGUARDED hand-copy of that helper
        # that ran the bare json.loads(...)["footprint"]["page"] and DID crash on a
        # malformed inventory -- so this test FAILS against the pre-extraction condition-set
        # code, and is the proof the drift was real. The sibling tabular test is identical.
        # EmbeddedTableInventory's T1 validator makes a malformed canonical_json impossible
        # to CONSTRUCT, so a corrupt inventory can only reach the renderer through a
        # validation-bypassed / partially constructed envelope -- exactly what we build here
        # with model_construct (T1-bypassing) plus a non-revalidating model_copy. Feed each
        # distinct way the "footprint"."page" read can break; the label the envelope cited
        # must still come back, and the page must degrade to "unknown", never a traceback.
        from carmel.services.condition_set_target import _stored_table_reference

        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        envelope = produce_and_store_target(workspace).envelope

        # Control: the well-formed path reads a real, non-"unknown" page. If the guard
        # ever swallowed the good case, this assertion catches it.
        good_label, good_page = _stored_table_reference(envelope)
        assert good_page != "unknown"

        malformations = (
            "{not valid json",  # json.JSONDecodeError: unparseable
            "{}",  # KeyError: parseable, but no "footprint"
            '{"footprint": []}',  # TypeError: parseable, wrong shape (["page"] on a list)
        )
        for canonical in malformations:
            corrupt = tuple(
                EmbeddedTableInventory.model_construct(
                    inventory_sha256=inv.inventory_sha256,
                    raw_sha256=inv.raw_sha256,
                    canonical_json=canonical,
                )
                for inv in envelope.table_inventories
            )
            broken = envelope.model_copy(update={"table_inventories": corrupt})
            label, page = _stored_table_reference(broken)
            assert label == good_label, canonical
            assert page == "unknown", canonical

    def test_render_shows_the_exact_page_the_envelope_carries_not_a_fabricated_one(self, tmp_path: Path) -> None:
        # The SUCCESS path: the helper's docstring promises the render shows the page the
        # stored envelope actually cites, never a module constant and never an invented
        # value. The sibling "..._not_the_module_constants" test only proves the page is
        # not the CONSTANT, and does it with a substring assertion -- "page 99" is a
        # substring of "page 999", so a helper that reads the inventory and then fabricates
        # the page slips through it. Pin the page EXACTLY here. Build a real envelope, then
        # swap its inventories for coherent copies carrying a DIFFERENT page (99, not the
        # constant 4), keeping each inventory_sha256 so the citations still resolve; the
        # render must show that page, exactly, on its own line.
        import json

        from carmel.services.condition_set_target import TARGET_TABLE_FOOTPRINT, TARGET_TABLE_KEY

        workspace = _stage_campaign(tmp_path) / TARGET_CAMPAIGN
        envelope = produce_and_store_target(workspace).envelope

        distinct_page = TARGET_TABLE_FOOTPRINT.page + 95
        assert distinct_page != TARGET_TABLE_FOOTPRINT.page  # or the two sources are indistinguishable

        def _with_page(inv: EmbeddedTableInventory) -> EmbeddedTableInventory:
            payload = json.loads(inv.canonical_json)
            payload["footprint"]["page"] = distinct_page
            return EmbeddedTableInventory.model_construct(
                inventory_sha256=inv.inventory_sha256,
                raw_sha256=inv.raw_sha256,
                canonical_json=json.dumps(payload),
            )

        relabelled = tuple(_with_page(inv) for inv in envelope.table_inventories)
        envelope = envelope.model_copy(update={"table_inventories": relabelled})

        text = render_condition_set_text(envelope)
        source_line = next(line for line in text.splitlines() if line.startswith("Source table"))
        assert source_line == f"Source table   : {TARGET_TABLE_KEY.label}, page {distinct_page}"
        assert f"page {TARGET_TABLE_FOOTPRINT.page}" not in text  # never the module constant


class TestStoreConditionSetCommand:
    def test_it_stores_exports_and_prints_the_conditions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _stage_campaign(tmp_path)
        workspace = root / TARGET_CAMPAIGN

        exit_code = Carmel.main(["store-condition-set", "--workspaces", str(root)])
        assert exit_code == 0

        out = capsys.readouterr().out
        for token in ("Fuel", "H2/CO(50:50%)", "Air", "0.6–1.0", "P(atm)"):
            assert token in out, token

        condition_sets = list((workspace / CONDITION_SET_STORE_DIR).glob("*.json"))
        assert len(condition_sets) == 1
        reports = list((workspace / "reports").glob("condition-set-*.txt"))
        assert len(reports) == 1

    def test_it_reports_and_stops_when_the_document_is_absent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No corpus needed: an empty parent root has no campaign holding the
        # document, so the command must refuse cleanly (exit 1), never crash and
        # never write anything.
        exit_code = Carmel.main(["store-condition-set", "--workspaces", str(tmp_path)])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Refusing to store the condition set" in err
        assert not (tmp_path / TARGET_CAMPAIGN / CONDITION_SET_STORE_DIR).exists()

    def test_it_reports_stored_artifact_when_the_export_cannot_be_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No corpus needed: the envelope is produced and stored (mocked), then the
        # export write fails. The command must exit non-zero but name the stored
        # artifact on stderr -- an operator reading only stderr must not conclude
        # the data was lost.
        from carmel.services import condition_set_target as tgt

        stored = tgt.StoredTargetConditionSet(
            sha256="deadbeef",
            path=tmp_path / TARGET_CAMPAIGN / CONDITION_SET_STORE_DIR / "deadbeef.json",
            envelope=None,  # type: ignore[arg-type]  # unused on the failure path
        )
        monkeypatch.setattr(tgt, "produce_and_store_target", lambda *a, **k: stored)

        def _boom(*args: object, **kwargs: object) -> Path:
            raise OSError("reports/ is read-only")

        monkeypatch.setattr(tgt, "write_condition_set_export", _boom)

        exit_code = Carmel.main(["store-condition-set", "--workspaces", str(tmp_path)])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Stored the condition set envelope durably" in err
        assert "deadbeef" in err
        assert "could not write the human-readable export" in err

    def test_it_reports_and_stops_when_no_workspace_is_discovered(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With no --workspaces and discovery finding nothing, the command reports
        # the missing document and stops rather than raising.
        monkeypatch.setattr(
            "carmel.services.condition_set_target.locate_target_workspace",
            lambda *a, **k: None,
        )
        exit_code = Carmel.main(["store-condition-set"])
        assert exit_code == 1
        assert "not stored in any known workspace" in capsys.readouterr().err


class TestFailClosedPreconditions:
    """The corpus-free fail-closed paths: no document needed to reach them."""

    def test_locate_returns_none_when_no_root_holds_the_document(self, tmp_path: Path) -> None:
        assert locate_target_workspace(roots=(tmp_path,)) is None

    def test_read_target_raw_refuses_a_missing_document(self, tmp_path: Path) -> None:
        with pytest.raises(ConditionSetTargetError, match="not stored under"):
            read_target_raw(tmp_path)

    def test_read_target_raw_refuses_bytes_that_are_not_the_measured_document(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
        raw_dir.mkdir(parents=True)
        (raw_dir / "raw.bin").write_bytes(b"not the measured document")
        with pytest.raises(ConditionSetTargetError, match="not the measured"):
            read_target_raw(tmp_path)

    def test_read_target_raw_refuses_when_the_bytes_cannot_be_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # raw.bin is present (passes the exists() check) but reading it fails: an
        # unreadable document is a named refusal, not a raw OSError traceback.
        # Sibling of the tabular target's identically-named test.
        raw_dir = tmp_path / "evidence" / "literature" / TARGET_DOCUMENT_SHA256
        raw_dir.mkdir(parents=True)
        (raw_dir / "raw.bin").write_bytes(b"")

        def _boom(self: Path, *args: object, **kwargs: object) -> bytes:
            raise OSError("disk gone")

        monkeypatch.setattr(Path, "read_bytes", _boom)
        with pytest.raises(ConditionSetTargetError, match="cannot read the stored raw.bin"):
            read_target_raw(tmp_path)
