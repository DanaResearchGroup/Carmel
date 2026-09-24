"""Tests for the corpus table-lane survey harness.

The lane itself is exercised elsewhere; these tests pin the harness's own contract:
fail-closed per-document isolation (a crash is recorded, never aborts the batch),
checkpoint/resume, the flattening of a report into a row, deterministic stratified
sampling, and the honest ingestion round-trip. The fake-seam tests patch the lane so
they need no PDF; one gated end-to-end test builds a real blank PDF with pypdf.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from carmel.services.general_table_report import GeneralTableReportError
from carmel.tools import corpus_table_survey as cts
from carmel.tools.corpus_table_survey import DocumentOutcome, DocumentRow
from tests.pypdf_gate import require_pypdf


def _val(value: str | None) -> SimpleNamespace | None:
    return None if value is None else SimpleNamespace(value=value)


def _outcome(
    status: str,
    *,
    page: int = 1,
    caption: str = "Table 1: rate coefficients",
    verdict: str | None = None,
    refusal: str | None = None,
    stored_sha: str | None = None,
    detail: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        status=_val(status),
        page=page,
        caption_fragment=caption,
        verdict=_val(verdict),
        proposal_refusal_reason=_val(refusal),
        stored_inventory_sha256=stored_sha,
        detail=detail,
    )


def _report(
    sha: str,
    outcomes: tuple[SimpleNamespace, ...] = (),
    *,
    lossy: bool = False,
    status: str = "available",
    unavailable: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        raw_sha256=sha,
        extraction_lossy=lossy,
        extraction_status=_val(status),
        extraction_unavailable=unavailable,
        outcomes=list(outcomes),
    )


def _fake_sha(path: Path) -> str:
    """The fake digest ``_patch_ingest`` reports: a real hash of the file's actual bytes.

    Content-derived (not path-derived) so it lines up with the real, content-based
    identity :func:`corpus_table_survey._content_id_for_resume` computes independently of
    the patched ``ingest_pdf`` -- resume tests rely on the two agreeing exactly as they
    would in production, where both come from the same ``sha256(data)``.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ingest side-effect free while keeping its digest real (content-derived)."""
    monkeypatch.setattr(cts, "ingest_pdf", lambda ws, path, *, max_bytes: (_fake_sha(path), 42))


def _touch_pdfs(directory: Path, names: list[str]) -> list[Path]:
    """Write one PDF stub per name, each with content distinct per name.

    Distinct content per file matters now that identity is content-based: two files with
    identical bytes are, correctly, the same document.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(f"%PDF-1.4 stub {name}".encode())
        paths.append(path)
    return paths


def test_reported_document_flattens_candidates_and_counts_yield(monkeypatch, tmp_path):
    _patch_ingest(monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub paper.pdf")
    sha = _fake_sha(pdf)
    report = _report(
        sha,
        (
            _outcome("stored", verdict="measured", stored_sha="f" * 64, detail="stored and replayed"),
            _outcome("not_measured", verdict="not_measured", caption="Table 2: species"),
        ),
    )
    monkeypatch.setattr(cts, "report_document_tables", lambda ws, digest, *, max_bytes: report)

    row = cts.survey_document(tmp_path, pdf, max_bytes=1)

    assert row.outcome is DocumentOutcome.REPORTED
    assert row.raw_sha256 == sha
    assert row.n_candidates == 2
    assert row.n_stored == 1
    assert row.yielded is True
    assert {c.status for c in row.candidates} == {"stored", "not_measured"}
    assert row.candidates[0].stored_inventory_sha256 == "f" * 64


def test_crash_is_recorded_and_does_not_abort_batch(monkeypatch, tmp_path):
    pdfs = _touch_pdfs(tmp_path / "corpus", ["a.pdf", "b.pdf", "c.pdf"])
    _patch_ingest(monkeypatch)
    crashing_sha = _fake_sha(pdfs[1])

    def _report_or_crash(ws, digest, *, max_bytes):
        if digest == crashing_sha:
            raise ValueError("boom: a non-typed error from deep in the lane")
        return _report(digest)

    monkeypatch.setattr(cts, "report_document_tables", _report_or_crash)
    out = tmp_path / "results.jsonl"

    summary = cts.survey_batch(tmp_path / "ws", pdfs, out, max_bytes=1)

    rows = [DocumentRow.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    by_id = {r.doc_id: r for r in rows}
    assert by_id[_fake_sha(pdfs[1])].outcome is DocumentOutcome.CRASHED
    assert by_id[_fake_sha(pdfs[1])].error_type == "ValueError"
    assert by_id[_fake_sha(pdfs[0])].outcome is DocumentOutcome.REPORTED
    assert by_id[_fake_sha(pdfs[2])].outcome is DocumentOutcome.REPORTED
    assert summary.n_documents == 3
    assert summary.crashed_documents == [_fake_sha(pdfs[1])]


def test_typed_whole_document_refusal_is_recorded_not_a_crash(monkeypatch, tmp_path):
    pdfs = _touch_pdfs(tmp_path / "corpus", ["a.pdf", "b.pdf"])
    _patch_ingest(monkeypatch)

    def _refuse(ws, digest, *, max_bytes):
        raise GeneralTableReportError("stored raw.bin is not trustworthy")

    monkeypatch.setattr(cts, "report_document_tables", _refuse)
    out = tmp_path / "results.jsonl"

    summary = cts.survey_batch(tmp_path / "ws", pdfs, out, max_bytes=1)

    assert summary.n_documents == 2
    assert summary.outcomes[DocumentOutcome.WHOLE_DOC_REFUSED.value] == 2
    assert summary.crashed_documents == []


def test_ingest_failure_is_recorded_for_an_unreadable_file(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"

    row = cts.survey_document(tmp_path / "ws", missing, max_bytes=1_000_000)

    assert row.outcome is DocumentOutcome.INGEST_FAILED
    assert row.raw_sha256 is None
    assert row.error_type == "FileNotFoundError"


def test_resume_skips_documents_already_recorded(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    first = _touch_pdfs(corpus, ["a.pdf", "b.pdf"])
    _patch_ingest(monkeypatch)
    surveyed: list[str] = []

    def _record(ws, digest, *, max_bytes):
        surveyed.append(digest)
        return _report(digest)

    monkeypatch.setattr(cts, "report_document_tables", _record)
    out = tmp_path / "results.jsonl"

    cts.survey_batch(tmp_path / "ws", first, out, max_bytes=1)
    assert len(surveyed) == 2

    third = corpus / "c.pdf"
    third.write_bytes(b"%PDF-1.4 stub c.pdf")
    surveyed.clear()
    summary = cts.survey_batch(tmp_path / "ws", [*first, third], out, max_bytes=1, resume=True)

    assert surveyed == [_fake_sha(third)]  # only the new document ran
    assert summary.n_documents == 3
    rows = [DocumentRow.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    assert len({r.doc_id for r in rows}) == 3


def test_resume_reprocesses_a_path_whose_content_changed(monkeypatch, tmp_path):
    """Fix #4: resume identity is content-based, not path-based.

    A document that changed at the same path must be treated as a different document,
    not silently skipped as "already recorded".
    """
    corpus = tmp_path / "corpus"
    pdfs = _touch_pdfs(corpus, ["a.pdf"])
    _patch_ingest(monkeypatch)
    surveyed: list[str] = []

    def _record(ws, digest, *, max_bytes):
        surveyed.append(digest)
        return _report(digest)

    monkeypatch.setattr(cts, "report_document_tables", _record)
    out = tmp_path / "results.jsonl"

    cts.survey_batch(tmp_path / "ws", pdfs, out, max_bytes=1000)
    assert len(surveyed) == 1
    first_id = surveyed[0]

    # Same path, different bytes: must be reprocessed, not skipped as already-done.
    pdfs[0].write_bytes(b"%PDF-1.4 a completely different document now")
    surveyed.clear()
    cts.survey_batch(tmp_path / "ws", pdfs, out, max_bytes=1000, resume=True)

    assert len(surveyed) == 1
    assert surveyed[0] != first_id
    rows = [DocumentRow.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    assert len({r.doc_id for r in rows}) == 2


def test_prune_unstored_removes_only_documents_that_stored_nothing(monkeypatch, tmp_path):
    pdfs = _touch_pdfs(tmp_path / "corpus", ["kept.pdf", "pruned.pdf"])
    _patch_ingest(monkeypatch)
    stored_sha = _fake_sha(pdfs[0])

    def _report_maybe_stored(ws, digest, *, max_bytes):
        if digest == stored_sha:
            return _report(digest, (_outcome("stored", verdict="measured", stored_sha="a" * 64),))
        return _report(digest)

    monkeypatch.setattr(cts, "report_document_tables", _report_maybe_stored)
    pruned: list[str] = []
    monkeypatch.setattr(cts, "_prune_artifact", lambda ws, sha: pruned.append(sha))
    out = tmp_path / "results.jsonl"

    cts.survey_batch(tmp_path / "ws", pdfs, out, max_bytes=1, prune_unstored=True)

    assert pruned == [_fake_sha(pdfs[1])]  # only the unstored document was pruned


def test_enumerate_pdfs_is_case_insensitive_sorted_and_excludes_non_pdf(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "B.PDF").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.pdf").write_bytes(b"x")

    found = cts.enumerate_pdfs([tmp_path])

    assert found == sorted(found)
    assert {p.name for p in found} == {"a.pdf", "B.PDF", "c.pdf"}


def test_allocate_is_proportional_largest_remainder():
    assert cts._allocate([5, 15], 8) == [2, 6]
    assert sum(cts._allocate([1, 1, 1], 2)) == 2
    assert all(a <= c for a, c in zip(cts._allocate([1, 1, 1], 2), [1, 1, 1], strict=True))
    assert cts._allocate([10], 100) == [10]  # never asks for more than a bin holds


def test_sample_corpus_is_deterministic_and_stratified(tmp_path):
    root_a = tmp_path / "small"
    root_b = tmp_path / "large"
    _touch_pdfs(root_a, [f"a{i}.pdf" for i in range(5)])
    _touch_pdfs(root_b, [f"b{i}.pdf" for i in range(15)])

    first = cts.sample_corpus([root_a, root_b], sample_size=8, seed=1)
    second = cts.sample_corpus([root_a, root_b], sample_size=8, seed=1)

    assert first == second  # deterministic
    assert len(first) == 8
    from_a = [p for p in first if p.parent == root_a]
    from_b = [p for p in first if p.parent == root_b]
    assert len(from_a) == 2  # proportional: 8 * 5/20
    assert len(from_b) == 6  # proportional: 8 * 15/20


def test_sample_corpus_returns_everything_when_size_covers_the_corpus(tmp_path):
    root = tmp_path / "corpus"
    _touch_pdfs(root, [f"p{i}.pdf" for i in range(4)])

    assert len(cts.sample_corpus([root], sample_size=None, seed=0)) == 4
    assert len(cts.sample_corpus([root], sample_size=99, seed=0)) == 4


def test_summarize_ranks_refusals_and_counts_document_facts(tmp_path):
    out = tmp_path / "results.jsonl"
    rows = [
        DocumentRow(
            doc_id="p1" * 32,
            outcome=DocumentOutcome.REPORTED,
            n_candidates=1,
            n_stored=1,
            candidates=[cts.CandidateRow(status="stored", page=1, caption_fragment="Table 1")],
        ),
        DocumentRow(
            doc_id="p2" * 32,
            outcome=DocumentOutcome.REPORTED,
            n_candidates=0,
            extraction_lossy=True,
        ),
        DocumentRow(
            doc_id="p3" * 32,
            outcome=DocumentOutcome.REPORTED,
            n_candidates=2,
            candidates=[
                cts.CandidateRow(
                    status="proposal_refused",
                    page=1,
                    caption_fragment="Table 1",
                    proposal_refusal_reason="grid_not_derived",
                ),
                cts.CandidateRow(
                    status="proposal_refused",
                    page=2,
                    caption_fragment="Table 2",
                    proposal_refusal_reason="too_few_columns",
                ),
            ],
        ),
    ]
    out.write_text("\n".join(r.model_dump_json() for r in rows) + "\n")

    summary = cts.summarize(out)

    assert summary.n_documents == 3
    assert summary.n_yield == 1
    assert summary.n_zero_candidate_documents == 1
    assert summary.n_lossy_documents == 1
    assert summary.proposal_refusal_reasons == {"grid_not_derived": 1, "too_few_columns": 1}
    assert summary.candidate_statuses["proposal_refused"] == 2


def test_end_to_end_blank_pdf_ingests_and_reports(tmp_path):
    require_pypdf()
    from pypdf import PdfWriter

    pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as handle:
        writer.write(handle)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    workspace = tmp_path / "ws"

    row = cts.survey_document(workspace, pdf, max_bytes=10_000_000)

    assert row.outcome is DocumentOutcome.REPORTED
    assert row.raw_sha256 == expected_sha
    assert row.n_stored == 0  # a blank page carries no measured table
    raw = workspace / "evidence" / "literature" / expected_sha / "raw.bin"
    assert raw.is_file()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == expected_sha


def test_ingest_pdf_enforces_max_bytes_via_stat_before_reading(monkeypatch, tmp_path):
    """Fix #3: an oversized file is rejected off ``stat()`` alone -- never read into memory."""
    big = tmp_path / "big.pdf"
    big.write_bytes(b"x" * 100)

    def _must_not_read(self, *args, **kwargs):
        raise AssertionError("ingest_pdf must reject an oversized file before reading its bytes")

    monkeypatch.setattr(Path, "read_bytes", _must_not_read)

    with pytest.raises(ValueError, match="byte cap"):
        cts.ingest_pdf(tmp_path / "ws", big, max_bytes=10)


def test_ingest_pdf_rechecks_size_after_read_when_stat_understates_it(monkeypatch, tmp_path):
    """Fix #3: even if ``stat()`` under-reports size, the post-read length check still catches it."""
    pdf = tmp_path / "sneaky.pdf"
    pdf.write_bytes(b"x" * 100)

    real_stat = Path.stat

    # os.stat_result exposes only st_size to ingest_pdf's stat-based check, so a
    # SimpleNamespace stand-in for the lied-about path is enough.
    def _fake_stat(self, *args, **kwargs):
        if self == pdf:
            return SimpleNamespace(st_size=5)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _fake_stat)

    with pytest.raises(ValueError, match="after read"):
        cts.ingest_pdf(tmp_path / "ws", pdf, max_bytes=10)


def test_resume_discards_only_a_truncated_final_line(tmp_path):
    """Fix #5: a truncated last line (mid-write kill) is dropped; earlier rows still count."""
    out = tmp_path / "results.jsonl"
    good = DocumentRow(doc_id="a" * 8, outcome=DocumentOutcome.REPORTED, n_candidates=0)
    out.write_text(good.model_dump_json() + "\n" + '{"doc_id": "trunc')  # no closing brace, no newline

    done = cts._load_done_doc_ids(out)

    assert done == {"a" * 8}


def test_resume_still_raises_on_an_invalid_interior_line(tmp_path):
    """Fix #5: corruption that is NOT the last line is not explainable by a mid-write kill."""
    out = tmp_path / "results.jsonl"
    good = DocumentRow(doc_id="a" * 8, outcome=DocumentOutcome.REPORTED, n_candidates=0)
    out.write_text('{"doc_id": "broken-interior"}\n' + good.model_dump_json() + "\n")

    with pytest.raises(ValueError):
        cts._load_done_doc_ids(out)


def test_over_cap_ingest_failure_never_reads_past_the_cap_and_batch_completes(monkeypatch, tmp_path):
    """An over-cap file's failure fallback must not re-read the whole file to name it."""
    big, small = _touch_pdfs(tmp_path / "corpus", ["big.pdf", "small.pdf"])
    big.write_bytes(b"x" * 100)
    max_bytes = 50

    real_read_bytes = Path.read_bytes

    def _capped_read_bytes(self):
        if self.stat().st_size > max_bytes:
            raise MemoryError("read past max_bytes")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _capped_read_bytes)
    out = tmp_path / "results.jsonl"

    summary = cts.survey_batch(tmp_path / "ws", [big, small], out, max_bytes=max_bytes)

    assert summary.n_documents == 2
    big_row = cts._read_rows(out)[0]
    assert big_row.outcome is DocumentOutcome.INGEST_FAILED
    assert big_row.error_type == "ValueError"


def test_ingest_failure_errors_never_carry_the_file_name(tmp_path):
    """Every failure branch's ``error`` is written to JSONL, so it must name no file."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    big = corpus / "secret-over-cap-name.pdf"
    big.write_bytes(b"x" * 100)
    missing = corpus / "secret-missing-name.pdf"
    out = tmp_path / "results.jsonl"

    cts.survey_batch(tmp_path / "ws", [big, missing], out, max_bytes=50)

    rows = cts._read_rows(out)
    assert [row.outcome for row in rows] == [DocumentOutcome.INGEST_FAILED] * 2
    assert "100 bytes" in (rows[0].error or "")
    assert "50 byte cap" in (rows[0].error or "")
    for row in rows:
        assert "secret" not in (row.error or "")
        assert str(corpus) not in (row.error or "")


def test_resume_raises_on_a_newline_terminated_invalid_final_line(tmp_path):
    """A terminated line was flushed whole, so an invalid one is corruption, not a mid-write kill."""
    out = tmp_path / "results.jsonl"
    good = DocumentRow(doc_id="a" * 8, outcome=DocumentOutcome.REPORTED, n_candidates=0)
    out.write_text(good.model_dump_json() + "\n" + '{"doc_id": "trunc\n')

    with pytest.raises(ValueError):
        cts._load_done_doc_ids(out)

    out.write_text(good.model_dump_json() + "\n" + '{"doc_id": "trunc')

    assert cts._load_done_doc_ids(out) == {"a" * 8}
