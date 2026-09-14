"""The general table entry point: run the geometric lane over an ARBITRARY document.

These tests inject a synthetic :class:`FragmentExtraction` (the corpus is
non-redistributable, so no document bytes are committed) by patching the
``extract_fragments`` seam in BOTH the reader
(:mod:`carmel.services.general_table_report`) and the off-disk replay verifier
(:mod:`carmel.services.pdf_table_record`), so a stored record derived from the injected
grid replays against the same injected grid. No pypdf engine is needed: the real
extractor is never called.

The grids below were confirmed against the live ``propose_tables`` / ``classify_table``
before being pinned here: the measured grid proposes and classifies MEASURED (unit
token in header, monotone sweep), the species grid classifies NOT_MEASURED (no numeric
column), the raw-counts grid classifies UNDECIDED (numeric, no unit, no sweep), and the
single-column grid is a TOO_FEW_COLUMNS geometric refusal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from carmel.services import general_table_report as gtr
from carmel.services import pdf_table_record
from carmel.services.evidence import artifact_dir
from carmel.services.general_table_report import (
    CandidateStatus,
    GeneralTableReportError,
    report_document_tables,
)
from carmel.services.pdf_fragments import (
    FragmentAvailability,
    FragmentExtraction,
    GlyphMapping,
    TextFragment,
)
from carmel.services.pdf_table_store import StoredInventoryOutcome, StoredInventoryVerification
from carmel.services.table_data_discriminator import DataVerdict


def _frag(text: str, x_start: float, x_end: float, baseline_y: float, *, page: int = 1) -> TextFragment:
    return TextFragment(
        page=page,
        text=text,
        x_start=float(x_start),
        x_end=float(x_end),
        baseline_y=float(baseline_y),
        font_height=8.0,
        rotated=False,
        glyph_mapping=GlyphMapping.MAPPED,
    )


def _extraction(*fragments: TextFragment, **kwargs: object) -> FragmentExtraction:
    return FragmentExtraction(fragments=fragments, pypdf_version="6.14.2", **kwargs)  # type: ignore[arg-type]


def _measured_grid(page: int = 1) -> tuple[TextFragment, ...]:
    """A caption over a phi / S(cm/s) grid: unit token in header + a monotone sweep."""
    return (
        _frag("Table 1 flame speed", 50, 160, 200, page=page),
        _frag("phi", 50, 70, 185, page=page),
        _frag("S (cm/s)", 120, 150, 185, page=page),
        _frag("0.6", 50, 66, 170, page=page),
        _frag("35.0", 120, 140, 170, page=page),
        _frag("0.7", 50, 66, 155, page=page),
        _frag("42.0", 120, 140, 155, page=page),
        _frag("0.8", 50, 66, 140, page=page),
        _frag("48.0", 120, 140, 140, page=page),
    )


def _species_grid(page: int = 1) -> tuple[TextFragment, ...]:
    """A caption over a name / role grid: no numeric column, so NOT_MEASURED."""
    return (
        _frag("Table 2 species", 50, 160, 200, page=page),
        _frag("name", 50, 70, 185, page=page),
        _frag("role", 120, 140, 185, page=page),
        _frag("water", 50, 70, 170, page=page),
        _frag("diluent", 120, 145, 170, page=page),
        _frag("argon", 50, 70, 155, page=page),
        _frag("bath", 120, 140, 155, page=page),
        _frag("helium", 50, 72, 140, page=page),
        _frag("bath", 120, 140, 140, page=page),
    )


def _raw_counts_grid(page: int = 1) -> tuple[TextFragment, ...]:
    """A numeric grid with no unit and no monotone column: UNDECIDED."""
    return (
        _frag("Table 3 raw", 50, 160, 200, page=page),
        _frag("aa", 50, 66, 185, page=page),
        _frag("bb", 120, 136, 185, page=page),
        _frag("30", 50, 60, 170, page=page),
        _frag("90", 120, 130, 170, page=page),
        _frag("10", 50, 60, 155, page=page),
        _frag("50", 120, 130, 155, page=page),
        _frag("20", 50, 60, 140, page=page),
        _frag("70", 120, 130, 140, page=page),
    )


def _single_column_grid(page: int = 1) -> tuple[TextFragment, ...]:
    """A caption over one body column: a TOO_FEW_COLUMNS geometric refusal."""
    return (
        _frag("Table 9 single", 50, 160, 200, page=page),
        _frag("only", 50, 70, 185, page=page),
        _frag("1.0", 50, 66, 170, page=page),
        _frag("2.0", 50, 66, 155, page=page),
    )


def _place_raw(workspace: Path, sha_bytes: bytes) -> str:
    """Write ``sha_bytes`` as ``raw.bin`` under its own content address; return the sha."""
    sha = hashlib.sha256(sha_bytes).hexdigest()
    directory = artifact_dir(workspace, sha)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.bin").write_bytes(sha_bytes)
    return sha


def _inject(monkeypatch: pytest.MonkeyPatch, extraction: FragmentExtraction) -> None:
    """Make both the reader and the replay verifier see ``extraction`` for any bytes."""

    def fake(_data: bytes) -> FragmentExtraction:
        return extraction

    monkeypatch.setattr(gtr, "extract_fragments", fake)
    monkeypatch.setattr(pdf_table_record, "extract_fragments", fake)


def _record_dir(workspace: Path, sha: str) -> Path:
    return workspace / "evidence" / "literature" / sha / "table_inventories"


def _no_record_written(workspace: Path, sha: str) -> bool:
    directory = _record_dir(workspace, sha)
    return not directory.exists() or not any(directory.iterdir())


# --- whole-document fail-closed (no extraction needed) ---------------------------------------


class TestWholeDocumentFailsClosed:
    def test_a_malformed_sha_is_refused_before_any_store_touch(self, tmp_path: Path) -> None:
        with pytest.raises(GeneralTableReportError, match="not a lowercase hex sha256"):
            report_document_tables(tmp_path, "NOTASHA")

    def test_an_absent_document_is_refused(self, tmp_path: Path) -> None:
        absent = "a" * 64
        with pytest.raises(GeneralTableReportError, match="no stored raw.bin"):
            report_document_tables(tmp_path, absent)
        assert _no_record_written(tmp_path, absent)

    def test_bytes_that_do_not_hash_to_the_requested_sha_are_refused(self, tmp_path: Path) -> None:
        # raw.bin is filed under a sha its bytes do not produce: a mislabelled store.
        wrong_sha = "b" * 64
        directory = artifact_dir(tmp_path, wrong_sha)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "raw.bin").write_bytes(b"these bytes hash to something else")
        with pytest.raises(GeneralTableReportError, match="not the requested"):
            report_document_tables(tmp_path, wrong_sha)
        assert _no_record_written(tmp_path, wrong_sha)


# --- per-candidate outcomes ------------------------------------------------------------------


class TestPerCandidate:
    def test_a_measured_grid_is_stored_and_proved_to_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject(monkeypatch, _extraction(*_measured_grid()))
        sha = _place_raw(tmp_path, b"measured-doc")
        report = report_document_tables(tmp_path, sha)
        assert len(report.outcomes) == 1
        (outcome,) = report.outcomes
        assert outcome.status is CandidateStatus.STORED
        assert outcome.verdict is DataVerdict.MEASURED
        assert outcome.stored_inventory_sha256 is not None
        assert outcome.series_deferred is not None  # gate: the series step is deferred, not faked
        assert report.stored == (outcome,)
        # The artifact is really on disk and content-addressed under the doc.
        assert not _no_record_written(tmp_path, sha)

    def test_a_not_measured_grid_is_reported_not_stored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _inject(monkeypatch, _extraction(*_species_grid()))
        sha = _place_raw(tmp_path, b"species-doc")
        report = report_document_tables(tmp_path, sha)
        (outcome,) = report.outcomes
        assert outcome.status is CandidateStatus.NOT_MEASURED
        assert outcome.verdict is DataVerdict.NOT_MEASURED
        assert outcome.stored_inventory_sha256 is None
        assert report.stored == ()
        assert _no_record_written(tmp_path, sha)

    def test_an_undecided_grid_is_reported_not_stored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _inject(monkeypatch, _extraction(*_raw_counts_grid()))
        sha = _place_raw(tmp_path, b"raw-counts-doc")
        report = report_document_tables(tmp_path, sha)
        (outcome,) = report.outcomes
        assert outcome.status is CandidateStatus.UNDECIDED
        assert outcome.verdict is DataVerdict.UNDECIDED
        assert outcome.needed  # the classifier names what a human would look for
        assert report.stored == ()
        assert _no_record_written(tmp_path, sha)

    def test_a_single_column_grid_is_a_geometric_refusal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.services.pdf_table_proposer import ProposalRefusalReason

        _inject(monkeypatch, _extraction(*_single_column_grid()))
        sha = _place_raw(tmp_path, b"single-col-doc")
        report = report_document_tables(tmp_path, sha)
        (outcome,) = report.outcomes
        assert outcome.status is CandidateStatus.PROPOSAL_REFUSED
        assert outcome.proposal_refusal_reason is ProposalRefusalReason.TOO_FEW_COLUMNS
        assert outcome.verdict is None
        assert _no_record_written(tmp_path, sha)

    def test_a_lossy_extraction_yields_only_geometric_refusals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # build_inventory refuses every footprint on a lossy extraction, so a measurable
        # grid over a lossy extraction NEVER reaches the store; the lossiness is carried
        # on the report flag and in the refusal detail.
        _inject(monkeypatch, _extraction(*_measured_grid(), lossy=True))
        sha = _place_raw(tmp_path, b"lossy-doc")
        report = report_document_tables(tmp_path, sha)
        assert report.extraction_lossy is True
        assert report.stored == ()
        assert all(o.status is CandidateStatus.PROPOSAL_REFUSED for o in report.outcomes)
        assert _no_record_written(tmp_path, sha)

    def test_an_unavailable_extraction_is_reported_once_at_document_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unavailable = FragmentExtraction(
            fragments=(),
            pypdf_version="",
            status=FragmentAvailability.ENGINE_ABSENT,
            lossy=True,
        )
        _inject(monkeypatch, unavailable)
        sha = _place_raw(tmp_path, b"unavailable-doc")
        report = report_document_tables(tmp_path, sha)
        assert report.extraction_unavailable is True
        assert report.extraction_unavailable_detail
        assert report.outcomes == ()  # not smeared across per-candidate outcomes
        assert _no_record_written(tmp_path, sha)


# --- the partial-yield property --------------------------------------------------------------


class TestPartialYield:
    def test_one_candidate_stores_while_another_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Page 1 is measurable; page 2 is a species table with no numeric column. The
        # measurable one must be yielded and the other reported -- never a whole-document
        # abort on the refusal.
        _inject(monkeypatch, _extraction(*_measured_grid(page=1), *_species_grid(page=2)))
        sha = _place_raw(tmp_path, b"two-table-doc")
        report = report_document_tables(tmp_path, sha)

        by_page = {o.page: o for o in report.outcomes}
        assert by_page[1].status is CandidateStatus.STORED
        assert by_page[2].status is CandidateStatus.NOT_MEASURED
        assert len(report.stored) == 1
        assert report.stored[0].page == 1
        # The stored record exists; the refused one wrote nothing new beyond it.
        assert not _no_record_written(tmp_path, sha)


# --- the replay guard --------------------------------------------------------------------------


class TestStoredRecordMustReplay:
    def test_a_stored_record_that_does_not_replay_off_disk_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject(monkeypatch, _extraction(*_measured_grid()))
        sha = _place_raw(tmp_path, b"measured-but-unverifiable")

        def _not_usable(*_args: object, **_kwargs: object) -> StoredInventoryVerification:
            return StoredInventoryVerification(StoredInventoryOutcome.SOURCE_CORRUPT, detail="injected non-replay")

        monkeypatch.setattr(gtr, "verify_stored_inventory", _not_usable)
        report = report_document_tables(tmp_path, sha)

        # A candidate whose staged record does not replay is REFUSED per-candidate, never
        # a whole-document raise -- and the store is append-only, so the record that
        # failed the check must not be left behind for a later scan to read as evidence.
        (outcome,) = report.outcomes
        assert outcome.status is CandidateStatus.REPLAY_REFUSED
        assert outcome.verdict is DataVerdict.MEASURED
        assert "does not replay off disk" in outcome.detail
        assert report.stored == ()
        assert _no_record_written(tmp_path, sha)

    def test_a_replay_failure_beside_a_storable_candidate_still_yields_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PARTIAL YIELD for the failure mode proven live: two measurable pages, replay
        # forced to fail for the FIRST staged record only. The document must still yield
        # the storable one and report the other as refused -- never abort the whole
        # document -- and the refusing candidate must leave NO record in the real store.
        _inject(monkeypatch, _extraction(*_measured_grid(page=1), *_measured_grid(page=2)))
        sha = _place_raw(tmp_path, b"two-measured-doc")

        real_verify = gtr.verify_stored_inventory
        calls = {"n": 0}

        def _fail_first(*args: object, **kwargs: object) -> StoredInventoryVerification:
            calls["n"] += 1
            if calls["n"] == 1:
                return StoredInventoryVerification(StoredInventoryOutcome.SOURCE_CORRUPT, detail="injected non-replay")
            return real_verify(*args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(gtr, "verify_stored_inventory", _fail_first)
        report = report_document_tables(tmp_path, sha)

        refused = [o for o in report.outcomes if o.status is CandidateStatus.REPLAY_REFUSED]
        stored = [o for o in report.outcomes if o.status is CandidateStatus.STORED]
        assert len(refused) == 1
        assert len(stored) == 1
        assert len(report.stored) == 1
        # Exactly one record on disk: the stored candidate's. The refused one wrote nothing.
        assert len(list(_record_dir(tmp_path, sha).glob("*.json"))) == 1

    def test_no_staging_workspace_survives_a_refusal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The staging workspace holds a full byte copy of raw.bin; at corpus scale an
        # untorn-down leak is a directory per document. The refusal path early-exits
        # _classify_and_maybe_store, so cover it specifically -- the teardown lives in
        # report_document_tables's finally and must survive that early return. Redirect
        # mkdtemp into a known parent so a survivor is observable.
        _inject(monkeypatch, _extraction(*_measured_grid()))
        sha = _place_raw(tmp_path, b"measured-but-unverifiable")

        stage_parent = tmp_path / "stage_parent"
        stage_parent.mkdir()
        real_mkdtemp = gtr.tempfile.mkdtemp

        def _mkdtemp_in_parent(*args: object, **kwargs: object) -> str:
            kwargs["dir"] = str(stage_parent)
            return real_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(gtr.tempfile, "mkdtemp", _mkdtemp_in_parent)

        def _not_usable(*_args: object, **_kwargs: object) -> StoredInventoryVerification:
            return StoredInventoryVerification(StoredInventoryOutcome.SOURCE_CORRUPT, detail="injected non-replay")

        monkeypatch.setattr(gtr, "verify_stored_inventory", _not_usable)
        report = report_document_tables(tmp_path, sha)

        assert any(o.status is CandidateStatus.REPLAY_REFUSED for o in report.outcomes)
        assert list(stage_parent.iterdir()) == []
