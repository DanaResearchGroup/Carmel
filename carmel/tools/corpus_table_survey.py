"""Run the geometric PDF table lane over a batch of local PDFs and record the result.

This is the measurement harness the ``report-tables`` CLI has never had: it takes a list
of PDFs, ingests each into a scratch workspace, carries it through the exact lane
:func:`carmel.services.general_table_report.report_document_tables` runs (no
``series_agent``, so behaviour matches the CLI), and records one structured row per
document. It answers, over a real corpus, whether the lane produces anything.

Two properties are load-bearing and tested:

* **Fail-closed, recorded not routed around.** The lane's own contract is that a
  whole-document failure raises :class:`~carmel.services.general_table_report.GeneralTableReportError`
  (a clean typed refusal) while everything one candidate can do wrong is reported
  per-candidate. This harness records a typed whole-document refusal as
  :attr:`DocumentOutcome.WHOLE_DOC_REFUSED` and any *other* exception from the lane as
  :attr:`DocumentOutcome.CRASHED` -- the latter is a violation of the fail-closed contract
  and is preserved as a row so the run surfaces it rather than dying on it.
* **Per-document isolation with checkpointing.** One document that raises never aborts the
  batch, and every completed row is flushed and fsynced to the results file immediately, so
  a run that dies at document N keeps the first N-1.

Ingestion note (a finding, not a workaround): the lane needs only a stored ``raw.bin`` that
authenticates -- it re-extracts fragments from the bytes itself. The only public *manual*
ingestion door, :func:`carmel.services.acquisition.admit_file`, is gated on a
literature-request identity/full-article check that arbitrary corpus PDFs cannot pass, so
this harness assembles a metadata-only :class:`~carmel.agents.tools.fetch.FetchedArtifact`
by hand around the real :func:`~carmel.agents.tools.extract.extract_text` and
:func:`~carmel.services.evidence.store_artifact`. The extracted text is real; only the
fetch metadata is synthesised, and ``provenance=MANUAL`` records exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from carmel.agents.tools.extract import extract_text
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.literature import ArtifactProvenance
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.general_table_report import (
    GeneralTableReportError,
    report_document_tables,
)

#: The content type the lane's extractor expects for a PDF. Corpus inputs are PDFs only
#: (a non-goal explicitly excludes the ``.docx``/``.xlsx`` lane), so this is a constant,
#: never sniffed: a mislabelled file simply extracts to nothing and refuses downstream.
PDF_CONTENT_TYPE = "application/pdf"

#: Statuses whose candidate stored an inventory record AND proved it replays off disk.
#: These are the only outcomes that count toward yield.
_STORED_STATUSES = frozenset({"stored", "series_stored"})


class DocumentOutcome(StrEnum):
    """What became of one document as a whole."""

    REPORTED = "reported"
    """The lane ran to completion and returned a report. The report may still carry zero
    candidates or none that stored -- a clean report of nothing is this outcome, not a
    failure."""

    WHOLE_DOC_REFUSED = "whole_doc_refused"
    """The lane raised :class:`GeneralTableReportError` -- a clean, typed, whole-document
    refusal (malformed sha, absent/untrustworthy ``raw.bin``, or a store-integrity
    failure). This is the fail-closed contract working as designed."""

    INGEST_FAILED = "ingest_failed"
    """The document could not be turned into a stored artifact at all (unreadable file,
    empty bytes, over the size cap, or the extractor raised). Pre-lane; recorded so the
    batch continues."""

    CRASHED = "crashed"
    """The lane raised something OTHER than :class:`GeneralTableReportError`. This is a
    violation of the fail-closed contract and the most important thing a row can carry;
    it is preserved rather than aborting the batch."""


class CandidateRow(BaseModel):
    """One table candidate's outcome, flattened for the results file."""

    model_config = ConfigDict(extra="forbid")

    status: str
    page: int
    caption_fragment: str
    verdict: str | None = None
    proposal_refusal_reason: str | None = None
    stored_inventory_sha256: str | None = None
    detail: str = ""


class DocumentRow(BaseModel):
    """The complete record for one document, one JSON line in the results file.

    ``doc_id`` is deliberately opaque: the corpus survey walks a local, personal PDF
    library, and this file is committed to a public repo. ``doc_id`` is the document's
    content sha256 (identical to ``raw_sha256`` once ingestion succeeds) so a row is
    identifiable and resume-matchable without ever writing an absolute path or a paper's
    filename to disk. See :func:`_content_id_for_resume` and :func:`_fallback_doc_id`.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    outcome: DocumentOutcome
    raw_sha256: str | None = None
    n_bytes: int | None = None
    extraction_lossy: bool | None = None
    extraction_status: str | None = None
    extraction_unavailable: bool = False
    n_candidates: int = 0
    n_stored: int = 0
    candidates: list[CandidateRow] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    elapsed_s: float = 0.0

    @property
    def yielded(self) -> bool:
        """True when the document produced at least one stored-and-replayed grid."""
        return self.n_stored > 0


class BatchSummary(BaseModel):
    """Aggregate over a results file: the answer to the report's questions."""

    model_config = ConfigDict(extra="forbid")

    n_documents: int
    n_yield: int
    outcomes: dict[str, int]
    candidate_statuses: dict[str, int]
    proposal_refusal_reasons: dict[str, int]
    n_zero_candidate_documents: int
    n_lossy_documents: int
    n_extraction_unavailable_documents: int
    crashed_documents: list[str]


def ingest_pdf(workspace_root: Path, pdf_path: Path, *, max_bytes: int) -> tuple[str, int]:
    """Store ``pdf_path``'s bytes as a MANUAL artifact and return ``(raw_sha256, n_bytes)``.

    Uses the real extractor and the public content-addressed store; only the fetch
    metadata is synthesised (see the module docstring). Raises on any failure so the
    caller records the document as :attr:`DocumentOutcome.INGEST_FAILED`.

    Args:
        workspace_root: The scratch workspace to store into.
        pdf_path: The PDF on disk. Read-only; never mutated.
        max_bytes: Hard cap passed straight to :func:`store_artifact`.

    Returns:
        The recomputed ``raw_sha256`` and the byte count of the stored document.

    Raises:
        ValueError: ``pdf_path`` is over ``max_bytes``, checked twice: once via
            ``stat()`` before any bytes are read (so an oversized file is never read into
            memory), and again on ``len(data)`` after the read (a file can grow between
            the stat and the read; mirrors the stat-then-recheck discipline in
            :mod:`carmel.services.acquisition`).
    """
    size = pdf_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"document is {size} bytes, over the {max_bytes} byte cap")
    data = pdf_path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"document is {len(data)} bytes after read, over the {max_bytes} byte cap")
    digest = hashlib.sha256(data).hexdigest()
    extracted = extract_text(data, PDF_CONTENT_TYPE)
    artifact = FetchedArtifact(
        # The bytes did not travel a URL in this process; provenance=MANUAL says so.
        url=f"manual://corpus-survey/{pdf_path.name}",
        final_url=f"manual://corpus-survey/{pdf_path.name}",
        sha256=digest,
        content_type=PDF_CONTENT_TYPE,
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    store_artifact(
        workspace_root,
        data=data,
        artifact=artifact,
        extracted=extracted,
        license_note="corpus survey ingest; non-redistributable",
        provenance=ArtifactProvenance.MANUAL,
        max_bytes=max_bytes,
    )
    return digest, len(data)


def _scrub_path(text: str, path: Path) -> str:
    """Strip ``path`` -- absolute form and bare filename -- out of an exception message.

    Stdlib ``OSError`` subclasses (``FileNotFoundError``, ``PermissionError``, ...)
    interpolate the absolute path straight into ``str(exc)``. Those messages land in the
    ``error`` field of a committed results row, so the same discipline as ``doc_id``
    applies here: neither a local path nor a filename may escape into output. Best-effort;
    the input is our own path so this cannot itself raise.
    """
    return text.replace(str(path), "<document>").replace(path.name, "<document>")


def _fallback_doc_id(path: Path, *, max_bytes: int) -> str:
    """A doc id for a document that failed before its content hash could be trusted.

    Hashes the file's actual bytes when they fit under ``max_bytes`` (so an identical
    failed-then-readable file still gets a stable, content-derived id). The read is
    bounded to ``max_bytes + 1`` bytes: an over-cap file -- usually the very failure being
    recorded -- is never read in full, so this handler cannot itself exhaust memory and
    abort the batch. An over-cap or unreadable file falls back to a hash of the path
    string. Either way this is a one-way hash, never the path or filename.
    """
    path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return "unreadable-" + path_hash
    if len(data) > max_bytes:
        return "oversize-" + path_hash
    return hashlib.sha256(data).hexdigest()


def _content_id_for_resume(path: Path) -> str | None:
    """The current content id of ``path``, for resume's identity check, or ``None``.

    Deliberately independent of ``max_bytes``: an over-cap file still has a well-defined
    identity (it will simply fail fast, and cheaply, if reprocessed), so the size cap is
    not a reason to skip the comparison here. Returns ``None`` (never shortcuts) only
    when the file cannot be read at all. A successful id here is exactly the
    ``doc_id``/``raw_sha256`` a completed run would have recorded for this file's current
    bytes.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def survey_document(workspace_root: Path, pdf_path: Path, *, max_bytes: int) -> DocumentRow:
    """Ingest one PDF, run the lane over it, and return a fully typed row.

    Never raises: an ingest failure, a clean typed whole-document refusal, and a genuine
    crash are each caught and recorded as distinct :class:`DocumentOutcome` values so the
    batch can continue and the report can tell an honest refusal from a contract violation.
    """
    start = perf_counter()
    try:
        digest, n_bytes = ingest_pdf(workspace_root, pdf_path, max_bytes=max_bytes)
    except Exception as exc:  # noqa: BLE001 - untrusted input; record and continue
        return DocumentRow(
            doc_id=_fallback_doc_id(pdf_path, max_bytes=max_bytes),
            outcome=DocumentOutcome.INGEST_FAILED,
            error=_scrub_path(str(exc), pdf_path),
            error_type=type(exc).__name__,
            elapsed_s=perf_counter() - start,
        )

    try:
        report = report_document_tables(workspace_root, digest, max_bytes=max_bytes)
    except GeneralTableReportError as exc:
        return DocumentRow(
            doc_id=digest,
            outcome=DocumentOutcome.WHOLE_DOC_REFUSED,
            raw_sha256=digest,
            n_bytes=n_bytes,
            error=_scrub_path(str(exc), pdf_path),
            error_type=type(exc).__name__,
            elapsed_s=perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - a crash is the contract violation to RECORD
        return DocumentRow(
            doc_id=digest,
            outcome=DocumentOutcome.CRASHED,
            raw_sha256=digest,
            n_bytes=n_bytes,
            error=_scrub_path(str(exc), pdf_path),
            error_type=type(exc).__name__,
            elapsed_s=perf_counter() - start,
        )

    candidates = [
        CandidateRow(
            status=outcome.status.value,
            page=outcome.page,
            caption_fragment=outcome.caption_fragment,
            verdict=outcome.verdict.value if outcome.verdict is not None else None,
            proposal_refusal_reason=(
                outcome.proposal_refusal_reason.value if outcome.proposal_refusal_reason is not None else None
            ),
            stored_inventory_sha256=outcome.stored_inventory_sha256,
            detail=outcome.detail,
        )
        for outcome in report.outcomes
    ]
    n_stored = sum(1 for candidate in candidates if candidate.status in _STORED_STATUSES)
    return DocumentRow(
        doc_id=digest,
        outcome=DocumentOutcome.REPORTED,
        raw_sha256=report.raw_sha256,
        n_bytes=n_bytes,
        extraction_lossy=report.extraction_lossy,
        extraction_status=report.extraction_status.value,
        extraction_unavailable=report.extraction_unavailable,
        n_candidates=len(candidates),
        n_stored=n_stored,
        candidates=candidates,
        elapsed_s=perf_counter() - start,
    )


def _prune_artifact(workspace_root: Path, raw_sha256: str) -> None:
    """Remove one document's evidence directory to bound disk during a large run.

    Only ever called for a document that stored nothing, so no stored inventory record's
    provenance is touched (those records live outside ``evidence/literature/<sha>/``).
    """
    shutil.rmtree(artifact_dir(workspace_root, raw_sha256), ignore_errors=True)


def _read_rows(out_path: Path) -> list[DocumentRow]:
    """Parse every complete row in a results file.

    A process killed mid-write (SIGKILL, OOM, power loss) can leave a partial JSON
    object as the file's last line with NO trailing newline -- a normal artifact of
    append-and-fsync checkpointing, not corruption. That fragment is silently discarded
    so resume and summarize continue over exactly what was durably flushed. Every row is
    written with its newline in the same flushed write, so an invalid newline-terminated
    line -- interior or final -- is not explainable by a mid-write kill and still raises,
    since that really is corruption worth stopping for.
    """
    # split("\n"), not splitlines(): only "\n" terminates a row, and the element after
    # the last "\n" is exactly the unterminated remainder ("" when the file ends cleanly).
    lines = out_path.read_text(encoding="utf-8").split("\n")
    last_index = len(lines) - 1
    rows: list[DocumentRow] = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(DocumentRow.model_validate_json(line))
        except ValueError:
            if index == last_index:
                continue
            raise
    return rows


def _load_done_doc_ids(out_path: Path) -> set[str]:
    """The ``doc_id`` of every document already recorded in a results file, for resume."""
    return {row.doc_id for row in _read_rows(out_path)}


def survey_batch(
    workspace_root: Path,
    pdf_paths: list[Path],
    out_path: Path,
    *,
    max_bytes: int,
    resume: bool = True,
    prune_unstored: bool = False,
    on_row: Callable[[int, Path, DocumentRow], None] | None = None,
) -> BatchSummary:
    """Survey every PDF in ``pdf_paths``, checkpointing one JSON line per document.

    Args:
        workspace_root: Scratch workspace; created if absent.
        pdf_paths: The documents to survey, in order.
        out_path: Results file, appended to. One :class:`DocumentRow` JSON per line.
        max_bytes: Size cap passed to ingest and the replay check.
        resume: When true and ``out_path`` exists, skip documents whose CURRENT content
            id already appears in ``out_path`` (see :func:`_content_id_for_resume`) --
            not merely a document at a previously-seen path. A path whose file changed
            since the last run is reprocessed, never silently skipped as stale.
        prune_unstored: When true, delete the evidence directory of any document that
            stored nothing, to bound disk on a whole-corpus run. Stored-and-replayed
            documents are always kept.
        on_row: Optional progress callback, called with ``(index, pdf_path, row)`` after
            each write.

    Returns:
        A :class:`BatchSummary` computed from the complete results file.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_doc_ids(out_path) if resume and out_path.exists() else set()

    with out_path.open("a", encoding="utf-8") as handle:
        for index, pdf_path in enumerate(pdf_paths):
            if resume and done:
                current_id = _content_id_for_resume(pdf_path)
                if current_id is not None and current_id in done:
                    continue
            row = survey_document(workspace_root, pdf_path, max_bytes=max_bytes)
            handle.write(row.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(row.doc_id)
            if prune_unstored and row.n_stored == 0 and row.raw_sha256 is not None:
                _prune_artifact(workspace_root, row.raw_sha256)
            if on_row is not None:
                on_row(index, pdf_path, row)

    return summarize(out_path)


def summarize(out_path: Path) -> BatchSummary:
    """Aggregate a results file into the counts the report needs."""
    rows = _read_rows(out_path)
    outcomes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    refusal_reasons: Counter[str] = Counter()
    n_yield = 0
    n_zero = 0
    n_lossy = 0
    n_unavailable = 0
    crashed: list[str] = []
    for row in rows:
        outcomes[row.outcome.value] += 1
        if row.yielded:
            n_yield += 1
        if row.outcome is DocumentOutcome.REPORTED and row.n_candidates == 0:
            n_zero += 1
        if row.extraction_lossy:
            n_lossy += 1
        if row.extraction_unavailable:
            n_unavailable += 1
        if row.outcome is DocumentOutcome.CRASHED:
            crashed.append(row.doc_id)
        for candidate in row.candidates:
            statuses[candidate.status] += 1
            if candidate.proposal_refusal_reason is not None:
                refusal_reasons[candidate.proposal_refusal_reason] += 1
    return BatchSummary(
        n_documents=len(rows),
        n_yield=n_yield,
        outcomes=dict(outcomes),
        candidate_statuses=dict(statuses),
        proposal_refusal_reasons=dict(refusal_reasons),
        n_zero_candidate_documents=n_zero,
        n_lossy_documents=n_lossy,
        n_extraction_unavailable_documents=n_unavailable,
        crashed_documents=crashed,
    )


def enumerate_pdfs(roots: list[Path]) -> list[Path]:
    """Every ``*.pdf`` (case-insensitive) under ``roots``, deduplicated and path-sorted."""
    found: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                found.add(path)
    return sorted(found)


def _allocate(counts: list[int], sample_size: int) -> list[int]:
    """Split ``sample_size`` across bins proportional to ``counts`` (largest remainder).

    Guarantees the allocation sums to ``min(sample_size, sum(counts))`` and never asks a
    bin for more than it holds, so every non-empty folder is represented.
    """
    total = sum(counts)
    if sample_size >= total:
        return list(counts)
    exact = [sample_size * count / total for count in counts]
    floors = [int(value) for value in exact]
    remainder = sample_size - sum(floors)
    order = sorted(range(len(counts)), key=lambda i: exact[i] - floors[i], reverse=True)
    allocation = list(floors)
    for i in order:
        if remainder <= 0:
            break
        if allocation[i] < counts[i]:
            allocation[i] += 1
            remainder -= 1
    return allocation


def sample_corpus(roots: list[Path], *, sample_size: int | None, seed: int) -> list[Path]:
    """A deterministic sample spread across ``roots`` proportional to each root's size.

    Stratified so the result is not an artifact of one folder: each root contributes a
    share proportional to how many PDFs it holds, sampled with a seeded RNG. Deterministic
    given the same corpus, ``sample_size`` and ``seed``. ``sample_size`` ``None`` (or >=
    the corpus size) returns every PDF.
    """
    per_root = [enumerate_pdfs([root]) for root in roots]
    total = sum(len(pdfs) for pdfs in per_root)
    if sample_size is None or sample_size >= total:
        return sorted(pdf for pdfs in per_root for pdf in pdfs)
    allocation = _allocate([len(pdfs) for pdfs in per_root], sample_size)
    rng = random.Random(seed)
    chosen: list[Path] = []
    for pdfs, take in zip(per_root, allocation, strict=True):
        chosen.extend(rng.sample(pdfs, take))
    return sorted(chosen)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus_table_survey",
        description="Run the geometric PDF table lane over a batch of local PDFs.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="Scratch workspace to ingest into.")
    parser.add_argument("--out", type=Path, required=True, help="Results file (JSONL, one document per line).")
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        required=True,
        dest="roots",
        help="A corpus directory to sample from. Repeatable.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Sample this many PDFs (default: all).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the stratified sample.")
    parser.add_argument("--max-bytes", type=int, default=200_000_000, help="Per-document size cap.")
    parser.add_argument(
        "--prune-unstored",
        action="store_true",
        help="Delete evidence of documents that store nothing.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not skip documents already in --out.")
    parser.add_argument("--list-only", action="store_true", help="Print the sampled paths and exit without running.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Prints a progress line per document to stderr and a summary to stdout."""
    args = _build_parser().parse_args(argv)
    pdfs = sample_corpus(args.roots, sample_size=args.sample_size, seed=args.seed)

    if args.list_only:
        for pdf in pdfs:
            print(pdf)
        print(f"[{len(pdfs)} PDFs]", file=sys.stderr)
        return 0

    print(f"surveying {len(pdfs)} PDFs into {args.workspace}", file=sys.stderr)

    def _progress(index: int, pdf_path: Path, row: DocumentRow) -> None:
        tag = row.outcome.value
        extra = ""
        if row.outcome is DocumentOutcome.REPORTED:
            extra = f" candidates={row.n_candidates} stored={row.n_stored}"
        # Filenames are fine on the operator's own stderr; only the committed results
        # file must never carry them (see DocumentRow.doc_id).
        print(f"  [{index + 1}/{len(pdfs)}] {tag}{extra} :: {pdf_path.name}", file=sys.stderr)

    summary = survey_batch(
        args.workspace,
        pdfs,
        args.out,
        max_bytes=args.max_bytes,
        resume=not args.no_resume,
        prune_unstored=args.prune_unstored,
        on_row=_progress,
    )
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
