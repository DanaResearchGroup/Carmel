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
    return hashlib.sha256(str(path).encode()).hexdigest()


def _patch_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ingest deterministic and side-effect free: sha derived from the path."""
    monkeypatch.setattr(cts, "ingest_pdf", lambda ws, path, *, max_bytes: (_fake_sha(path), 42))


def _touch_pdfs(directory: Path, names: list[str]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(b"%PDF-1.4 stub")
        paths.append(path)
    return paths


def test_reported_document_flattens_candidates_and_counts_yield(monkeypatch, tmp_path):
    _patch_ingest(monkeypatch)
    pdf = tmp_path / "paper.pdf"
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
    by_name = {Path(r.path).name: r for r in rows}
    assert by_name["b.pdf"].outcome is DocumentOutcome.CRASHED
    assert by_name["b.pdf"].error_type == "ValueError"
    assert by_name["a.pdf"].outcome is DocumentOutcome.REPORTED
    assert by_name["c.pdf"].outcome is DocumentOutcome.REPORTED
    assert summary.n_documents == 3
    assert summary.crashed_documents == [str(pdfs[1])]


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
    third.write_bytes(b"%PDF-1.4 stub")
    surveyed.clear()
    summary = cts.survey_batch(tmp_path / "ws", [*first, third], out, max_bytes=1, resume=True)

    assert surveyed == [_fake_sha(third)]  # only the new document ran
    assert summary.n_documents == 3
    rows = [DocumentRow.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    assert len({r.path for r in rows}) == 3


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
            path="p1.pdf",
            outcome=DocumentOutcome.REPORTED,
            n_candidates=1,
            n_stored=1,
            candidates=[cts.CandidateRow(status="stored", page=1, caption_fragment="Table 1")],
        ),
        DocumentRow(
            path="p2.pdf",
            outcome=DocumentOutcome.REPORTED,
            n_candidates=0,
            extraction_lossy=True,
        ),
        DocumentRow(
            path="p3.pdf",
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
