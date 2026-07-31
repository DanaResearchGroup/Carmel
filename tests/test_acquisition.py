# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the manual paper-acquisition queue.

The load-bearing behaviour here is :func:`check_identity`. Matching a dropped file to a
request by filename alone is an unchecked human assertion; a mis-drop would bind one
paper's bytes to another paper's citation, and the quote-grounding gate could not catch
it because that gate only asks whether the quote is present in the supplied bytes, never
whether those bytes are the right paper.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from carmel.agents.tools.extract import ExtractedText, normalize_for_match
from carmel.schemas.acquisition import (
    AcquisitionReason,
    AcquisitionRequest,
    AcquisitionStatus,
)
from carmel.schemas.literature import ArtifactProvenance, StoredArtifact
from carmel.services.acquisition import (
    AlreadyAcquired,
    admit_file,
    check_identity,
    collect_inbox,
    drop_path_for,
    inbox_dir,
    load_manifest,
    pending_requests,
    record_request,
    requests_dir,
    slug_for,
)
from carmel.services.evidence import artifact_dir

TITLE = "Shock tube study of ignition delay times in methane oxygen argon mixtures"
DOI = "10.1016/0010-2180(76)90042-0"


def _request(*, doi: str | None = DOI, title: str = TITLE) -> AcquisitionRequest:
    return AcquisitionRequest(
        slug=slug_for(doi, title),
        title=title,
        doi=doi,
        landing_url="https://doi.org/10.1/x",
        reason=AcquisitionReason.PAYWALLED,
        requested_at=datetime.now(UTC),
    )


def _extracted(text: str) -> ExtractedText:
    return ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=[],
        extractor="test",
        lossy=False,
        page_count=1,
    )


class TestSlugFor:
    def test_doi_slug_is_stable_and_path_safe(self) -> None:
        assert slug_for("10.1016/j.combustflame.2015.11.011", "x") == "10.1016-j.combustflame.2015.11.011"

    @pytest.mark.parametrize("doi", ["../../etc/passwd", "....//....//x", "/absolute/path", "a/../../b"])
    def test_traversal_attempts_never_produce_a_path_separator(self, doi: str) -> None:
        """The slug becomes a filename in the inbox; a separator would let a crafted DOI
        address a location outside it."""
        slug = slug_for(doi, "title")
        assert "/" not in slug
        assert "\\" not in slug
        assert not slug.startswith(".")

    def test_titles_without_a_doi_get_a_digest_so_same_titled_papers_do_not_collide(self) -> None:
        first = slug_for(None, "Ignition delay")
        second = slug_for(None, "Ignition delay measurements")
        assert first != second

    def test_degenerate_input_still_yields_a_usable_slug(self) -> None:
        assert slug_for("", "") == "paper-e3b0c442"


class TestCheckIdentity:
    def test_doi_corroborated_by_the_title_passes(self) -> None:
        ok, note = check_identity(_extracted(f"Some Journal\nDOI: {DOI}\n{TITLE}\nAbstract..."), _request())
        assert ok is True
        assert DOI in note

    def test_doi_split_across_whitespace_still_matches(self) -> None:
        """PDF extraction routinely injects line breaks mid-identifier."""
        ok, _ = check_identity(_extracted(f"DOI: {DOI[:12]}\n{DOI[12:]}\n{TITLE}\n"), _request())
        assert ok is True

    def test_a_bare_doi_with_no_corroborating_title_is_refused(self) -> None:
        """A DOI alone is what a publisher landing page or a cover sheet carries. The
        earlier contract accepted it, which admitted the right DOI on the wrong bytes."""
        ok, note = check_identity(_extracted(f"Some Journal\nDOI: {DOI}\nAbstract..."), _request())
        assert ok is False
        assert "nothing corroborates it" in note

    def test_an_erratum_carrying_the_requested_doi_and_title_is_refused(self) -> None:
        """The case neither the DOI route nor the title route can catch: an erratum
        reprints both by construction, so only its own announcement distinguishes it."""
        ok, note = check_identity(
            _extracted(f"Erratum to: {TITLE}\nDOI: {DOI}\nThe authors regret an error in Table 2."),
            _request(),
        )
        assert ok is False
        assert "erratum" in note

    def test_a_paper_genuinely_titled_comment_on_is_not_blocked_by_its_own_title(self) -> None:
        """The marker check must not reject a request whose real title contains the word."""
        title = "Comment on the pressure dependence of the ignition delay of n-heptane"
        ok, _ = check_identity(_extracted(f"{title}\nDOI: {DOI}\n"), _request(title=title))
        assert ok is True

    def test_matching_title_without_a_doi_passes(self) -> None:
        ok, note = check_identity(_extracted(TITLE + "\n\nAbstract: we measured..."), _request(doi=None))
        assert ok is True
        assert "title matched" in note

    def test_a_different_paper_on_the_same_topic_is_rejected(self) -> None:
        """The realistic mis-drop: a topically adjacent paper. It shares subject words
        but not the title, and it does not carry the requested DOI."""
        wrong = "Laminar burning velocities of ammonia hydrogen blends at elevated pressure"
        ok, note = check_identity(_extracted(wrong), _request())
        assert ok is False
        assert "does not look like this paper" in note

    def test_empty_text_is_rejected_and_named_as_unreadable_not_as_the_wrong_paper(self) -> None:
        """An image-only scan must be diagnosed accurately: telling the operator "wrong
        paper" when they dropped the right one, badly scanned, sends them chasing the
        wrong problem."""
        ok, note = check_identity(_extracted("   \n  "), _request())
        assert ok is False
        assert "no extractable text" in note

    def test_a_citation_of_the_paper_deep_in_another_document_does_not_match(self) -> None:
        """A reference list mentioning the requested paper must not make a different
        document pass as that paper."""
        filler = "Unrelated combustion modelling discussion. " * 400
        text = filler + f"\n[42] Author et al. {TITLE}. doi:{DOI}\n"
        assert len(filler) > 6000
        ok, _ = check_identity(_extracted(text), _request())
        assert ok is False

    def test_an_erratum_with_its_own_doi_is_still_rejected_on_the_title_only_branch(self) -> None:
        """The marker check must fire on the title-only fallback too, not only when the
        requested DOI is found. An erratum has its OWN DOI, different from the original
        paper's -- so if a human drops the erratum instead of the original, the
        requested DOI is simply absent from the text. That sends this straight to the
        title-only branch, and the erratum reprints the original's full title by
        construction, so the title-only ratio passes. Before the fix, that branch never
        consulted the marker check at all, so this exact substitution was admitted."""
        erratum_doi = "10.1016/j.combustflame.2020.09.001"
        assert erratum_doi != DOI
        text = f"Erratum to: {TITLE}\nDOI: {erratum_doi}\nThe authors regret an error in Table 2."
        ok, note = check_identity(_extracted(text), _request())
        assert ok is False
        assert "erratum" in note


class TestRecordRequest:
    def test_request_is_persisted_with_operator_instructions(self, tmp_path: Path) -> None:
        record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/" + DOI,
            reason=AcquisitionReason.PAYWALLED,
            detail="HTTP 403",
        )

        manifest = load_manifest(tmp_path)
        assert len(manifest.requests) == 1
        assert manifest.requests[0].status == AcquisitionStatus.REQUESTED

        readme = (requests_dir(tmp_path) / "README.md").read_text(encoding="utf-8")
        assert TITLE in readme
        assert f"inbox/{slug_for(DOI, TITLE)}.pdf" in readme
        assert inbox_dir(tmp_path).is_dir()

    def test_requesting_the_same_paper_twice_does_not_duplicate_it(self, tmp_path: Path) -> None:
        for _ in range(3):
            record_request(
                tmp_path,
                title=TITLE,
                doi=DOI,
                landing_url="https://doi.org/" + DOI,
                reason=AcquisitionReason.PAYWALLED,
            )
        assert len(load_manifest(tmp_path).requests) == 1

    def test_a_corrupt_manifest_does_not_break_the_run(self, tmp_path: Path) -> None:
        requests_dir(tmp_path).mkdir(parents=True)
        (requests_dir(tmp_path) / "manifest.json").write_text("{not json", encoding="utf-8")

        record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/x",
            reason=AcquisitionReason.UNREADABLE,
        )
        assert len(load_manifest(tmp_path).requests) == 1


def _drop(tmp_path: Path, slug: str, body: str) -> Path:
    """Write a minimal text file into the inbox under ``slug``."""
    path = inbox_dir(tmp_path) / f"{slug}.txt"
    path.write_bytes(body.encode("utf-8"))
    return path


class TestCollectInbox:
    @pytest.fixture
    def queued(self, tmp_path: Path) -> AcquisitionRequest:
        return record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/" + DOI,
            reason=AcquisitionReason.PAYWALLED,
        )

    def test_a_matching_drop_is_admitted_with_manual_provenance(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        _drop(tmp_path, queued.slug, f"{TITLE}\nDOI: {DOI}\nAbstract: measurements follow.")

        changed = collect_inbox(tmp_path, max_bytes=10_000_000)

        assert len(changed) == 1
        assert changed[0].status == AcquisitionStatus.FULFILLED
        assert changed[0].fulfilled_sha256

        meta = StoredArtifact.model_validate(
            json.loads((artifact_dir(tmp_path, changed[0].fulfilled_sha256) / "meta.json").read_text())
        )
        assert meta.provenance == ArtifactProvenance.MANUAL

    def test_a_mismatched_drop_is_rejected_and_never_stored(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        """The whole point of the identity check: wrong bytes must not enter the
        evidence store under the right paper's name."""
        _drop(tmp_path, queued.slug, "An entirely different paper about catalytic converters.")

        changed = collect_inbox(tmp_path, max_bytes=10_000_000)

        assert changed[0].status == AcquisitionStatus.REJECTED
        assert changed[0].fulfilled_sha256 is None
        evidence = tmp_path / "evidence" / "literature"
        assert not evidence.exists() or not any(evidence.iterdir())

    def test_rejection_reason_is_surfaced_to_the_operator(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        _drop(tmp_path, queued.slug, "Unrelated document text.")
        collect_inbox(tmp_path, max_bytes=10_000_000)

        readme = (requests_dir(tmp_path) / "README.md").read_text(encoding="utf-8")
        assert "Needs attention" in readme

    def test_a_file_matching_no_request_is_left_alone(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        _drop(tmp_path, "some-unrelated-file", "content")
        assert collect_inbox(tmp_path, max_bytes=10_000_000) == []

    def test_an_oversized_drop_is_rejected(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        _drop(tmp_path, queued.slug, f"{TITLE} DOI: {DOI} " + "x" * 5000)

        changed = collect_inbox(tmp_path, max_bytes=100)

        assert changed[0].status == AcquisitionStatus.REJECTED
        assert "over the" in changed[0].identity_note

    def test_an_oversized_drop_is_rejected_via_stat_without_reading_it(
        self, tmp_path: Path, queued: AcquisitionRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The size cap must reject a huge file BEFORE it is pulled into memory, or the
        cap defeats its own purpose. Prove this by making a full read fail loudly:
        if the rejection still happens, it happened via `Path.stat()` alone."""
        _drop(tmp_path, queued.slug, f"{TITLE} DOI: {DOI} " + "x" * 5000)

        def _boom(self: Path, *args: object, **kwargs: object) -> bytes:
            raise AssertionError(f"read_bytes() must not be called on an oversized file: {self}")

        monkeypatch.setattr(Path, "read_bytes", _boom)

        changed = collect_inbox(tmp_path, max_bytes=100)

        assert changed[0].status == AcquisitionStatus.REJECTED
        assert "over the" in changed[0].identity_note

    def test_a_file_that_grows_past_the_cap_between_the_stat_and_the_read_is_rejected(
        self, tmp_path: Path, queued: AcquisitionRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The TOCTOU pair documented in `_admit_one`: the pre-read `stat()` check is
        cheap but not sufficient on its own, because the file can grow between that
        stat and the read completing, so `len(data)` is re-checked against the same cap
        after the read. Simulate that exact race by making `stat()` lie about a small
        size for this file, while its real (larger) bytes are what actually gets read."""
        path = _drop(tmp_path, queued.slug, f"{TITLE} DOI: {DOI} " + "x" * 5000)
        real_stat = Path.stat

        class _ShrunkStat:
            def __init__(self, real: object) -> None:
                self._real = real

            @property
            def st_size(self) -> int:
                return 10

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

        def _lying_stat(self: Path, *args: object, **kwargs: object) -> object:
            real = real_stat(self, *args, **kwargs)
            if self == path:
                return _ShrunkStat(real)
            return real

        monkeypatch.setattr(Path, "stat", _lying_stat)

        changed = collect_inbox(tmp_path, max_bytes=100)

        assert changed[0].status == AcquisitionStatus.REJECTED
        assert "over the 100 cap" in changed[0].identity_note

    def test_an_already_fulfilled_request_is_not_reprocessed(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        _drop(tmp_path, queued.slug, f"{TITLE}\nDOI: {DOI}\n")
        collect_inbox(tmp_path, max_bytes=10_000_000)

        assert collect_inbox(tmp_path, max_bytes=10_000_000) == []

    def test_no_inbox_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert collect_inbox(tmp_path, max_bytes=10_000_000) == []


def _source(tmp_path: Path, name: str, body: str) -> Path:
    """Write an operator's "downloaded" file somewhere OUTSIDE the inbox, the way a
    real download lands in e.g. ``~/Downloads`` before the operator hands it to us."""
    downloads = tmp_path / "downloads"
    downloads.mkdir(exist_ok=True)
    path = downloads / name
    path.write_bytes(body.encode("utf-8"))
    return path


SECOND_TITLE = "Laminar burning velocities of ammonia hydrogen blends at elevated pressure"
SECOND_DOI = "10.1016/j.combustflame.2019.01.002"


class TestPendingRequests:
    def test_requested_and_rejected_are_both_pending(self, tmp_path: Path) -> None:
        requested = record_request(
            tmp_path, title=TITLE, doi=DOI, landing_url="https://doi.org/" + DOI, reason=AcquisitionReason.PAYWALLED
        )
        rejected = record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        _drop(tmp_path, rejected.slug, "Some other document entirely, wrong on purpose.")
        collect_inbox(tmp_path, max_bytes=10_000_000)

        slugs = {r.slug for r in pending_requests(tmp_path)}
        assert slugs == {requested.slug, rejected.slug}

    def test_fulfilled_requests_are_not_pending(self, tmp_path: Path) -> None:
        fulfilled = record_request(
            tmp_path, title=TITLE, doi=DOI, landing_url="https://doi.org/" + DOI, reason=AcquisitionReason.PAYWALLED
        )
        _drop(tmp_path, fulfilled.slug, f"{TITLE}\nDOI: {DOI}\n")
        collect_inbox(tmp_path, max_bytes=10_000_000)

        assert pending_requests(tmp_path) == []


class TestDropPathFor:
    def test_default_suffix_is_pdf(self, tmp_path: Path) -> None:
        assert drop_path_for(tmp_path, "some-slug") == inbox_dir(tmp_path) / "some-slug.pdf"

    def test_suffix_without_a_leading_dot_is_normalized(self, tmp_path: Path) -> None:
        assert drop_path_for(tmp_path, "some-slug", suffix="txt") == inbox_dir(tmp_path) / "some-slug.txt"

    @pytest.mark.parametrize("slug", ["../../etc/passwd", "../escape", "a/../../b"])
    def test_a_traversal_slug_cannot_escape_the_inbox_directory(self, tmp_path: Path, slug: str) -> None:
        """Defence in depth: even if a raw string bypassed schema validation (a
        tampered manifest, a raw CLI --slug flag), this function must never hand back a
        path outside the inbox directory."""
        with pytest.raises(ValueError, match="inbox"):
            drop_path_for(tmp_path, slug)


class TestAcquisitionRequestSlugValidation:
    def test_a_traversal_slug_is_rejected_by_the_schema(self) -> None:
        with pytest.raises(ValidationError, match="String should match pattern"):
            AcquisitionRequest(
                slug="../../etc/passwd",
                title=TITLE,
                doi=DOI,
                landing_url="https://doi.org/10.1/x",
                reason=AcquisitionReason.PAYWALLED,
                requested_at=datetime.now(UTC),
            )

    def test_slug_for_output_still_validates_against_the_pattern(self) -> None:
        """The schema's pattern must not be narrower than what slug_for() actually
        emits -- widen the pattern, never the generator, if this ever fails."""
        for doi, title in [
            (DOI, TITLE),
            (None, TITLE),
            ("", ""),
            ("10.1016/j.combustflame.2015.11.011", "x"),
        ]:
            slug = slug_for(doi, title)
            AcquisitionRequest(
                slug=slug,
                title=title or "untitled",
                doi=doi or None,
                landing_url="https://doi.org/10.1/x",
                reason=AcquisitionReason.PAYWALLED,
                requested_at=datetime.now(UTC),
            )


class TestAdmitFile:
    @pytest.fixture
    def queued(self, tmp_path: Path) -> AcquisitionRequest:
        return record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/" + DOI,
            reason=AcquisitionReason.PAYWALLED,
        )

    def test_the_correct_paper_is_admitted_and_the_request_fulfilled(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        source = _source(tmp_path, "download.pdf", f"{TITLE}\nDOI: {DOI}\nAbstract: measurements follow.")

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.FULFILLED
        assert request.fulfilled_sha256
        meta = StoredArtifact.model_validate(
            json.loads((artifact_dir(tmp_path, request.fulfilled_sha256) / "meta.json").read_text())
        )
        assert meta.provenance == ArtifactProvenance.MANUAL
        # The original download must survive untouched.
        assert source.exists()
        assert TITLE in source.read_text(encoding="utf-8")

    def test_the_wrong_paper_is_rejected_and_never_stored(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        source = _source(tmp_path, "download.pdf", "An entirely different paper about catalytic converters.")

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.REJECTED
        assert request.fulfilled_sha256 is None
        assert request.identity_note
        evidence = tmp_path / "evidence" / "literature"
        assert not evidence.exists() or not any(evidence.iterdir())

    def test_an_erratum_for_the_requested_paper_is_rejected_by_name(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        """The live-acceptance case that matters most: an erratum carries the right DOI
        and reprints the full title, defeating both identity routes unless the marker
        check catches it. This exercises the full drop path, not just check_identity."""
        source = _source(
            tmp_path,
            "download.pdf",
            f"Erratum to: {TITLE}\nDOI: {DOI}\nThe authors regret an error in Table 2.",
        )

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.REJECTED
        assert "erratum" in request.identity_note
        assert request.fulfilled_sha256 is None
        evidence = tmp_path / "evidence" / "literature"
        assert not evidence.exists() or not any(evidence.iterdir())

    def test_slug_is_inferred_when_exactly_one_request_is_pending(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        source = _source(tmp_path, "whatever_the_download_was_named.pdf", f"{TITLE}\nDOI: {DOI}\n")

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.slug == queued.slug
        assert request.status == AcquisitionStatus.FULFILLED

    def test_slug_is_inferred_by_matching_text_when_several_are_pending(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        second = record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        source = _source(tmp_path, "download.pdf", f"{SECOND_TITLE}\nDOI: {SECOND_DOI}\n")

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.slug == second.slug
        assert request.status == AcquisitionStatus.FULFILLED

    def test_ambiguous_inference_raises_and_lists_candidates(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        second = record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        # Matches neither pending request's identity, so inference cannot settle on one.
        source = _source(tmp_path, "download.pdf", "Some unrelated document about turbine blades.")

        with pytest.raises(ValueError, match="cannot tell which pending request"):
            admit_file(tmp_path, source, max_bytes=10_000_000)

        # Both candidates must be named so the operator can pass slug= explicitly.
        try:
            admit_file(tmp_path, source, max_bytes=10_000_000)
        except ValueError as exc:
            assert queued.slug in str(exc)
            assert second.slug in str(exc)

    def test_explicit_slug_skips_inference(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        source = _source(tmp_path, "download.pdf", f"{TITLE}\nDOI: {DOI}\n")

        request = admit_file(tmp_path, source, slug=queued.slug, max_bytes=10_000_000)

        assert request.slug == queued.slug
        assert request.status == AcquisitionStatus.FULFILLED

    def test_a_slug_matching_no_request_raises(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        source = _source(tmp_path, "download.pdf", "content")

        with pytest.raises(ValueError, match="no acquisition request queued"):
            admit_file(tmp_path, source, slug="not-a-real-slug", max_bytes=10_000_000)

    def test_no_pending_requests_raises(self, tmp_path: Path) -> None:
        source = _source(tmp_path, "download.pdf", "content")

        with pytest.raises(ValueError, match="no acquisition requests are pending"):
            admit_file(tmp_path, source, max_bytes=10_000_000)

    def test_a_nonexistent_source_raises(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            admit_file(tmp_path, tmp_path / "downloads" / "missing.pdf", max_bytes=10_000_000)

    def test_a_directory_source_raises(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        directory = tmp_path / "downloads" / "not_a_file"
        directory.mkdir(parents=True)

        with pytest.raises(ValueError, match="directory"):
            admit_file(tmp_path, directory, max_bytes=10_000_000)

    def test_an_unreadable_source_raises(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        source = _source(tmp_path, "download.pdf", f"{TITLE}\nDOI: {DOI}\n")
        source.chmod(0o000)
        try:
            with pytest.raises(ValueError):
                admit_file(tmp_path, source, max_bytes=10_000_000)
        finally:
            # Restore permissions so pytest's tmp_path cleanup can remove the file.
            source.chmod(0o644)

    def test_an_oversized_source_is_rejected_not_raised(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        """Same untrusted-input path as a fetched PDF: an operator can drop an
        arbitrarily large file, and max_bytes must still be enforced."""
        source = _source(tmp_path, "download.pdf", f"{TITLE} DOI: {DOI} " + "x" * 5000)

        request = admit_file(tmp_path, source, max_bytes=100)

        assert request.status == AcquisitionStatus.REJECTED
        assert "over the" in request.identity_note

    def test_inference_rejects_an_oversized_source_via_stat_without_reading_it(
        self, tmp_path: Path, queued: AcquisitionRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same stat-before-read discipline as `_admit_one`, but inside `_infer_slug`
        itself: this only runs when more than one request is pending (a single pending
        request short-circuits before any stat/read), so a second request is queued
        here to force the real inference path. Prove the cap is enforced via `stat()`
        alone by making a full read fail loudly."""
        record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        source = _source(tmp_path, "download.pdf", f"{TITLE} DOI: {DOI} " + "x" * 5000)

        def _boom(self: Path, *args: object, **kwargs: object) -> bytes:
            raise AssertionError(f"read_bytes() must not be called on an oversized file: {self}")

        monkeypatch.setattr(Path, "read_bytes", _boom)

        with pytest.raises(ValueError, match="over the 100 cap"):
            admit_file(tmp_path, source, max_bytes=100)

    def test_inference_rejects_a_source_that_grows_past_the_cap_between_stat_and_read(
        self, tmp_path: Path, queued: AcquisitionRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The equivalent TOCTOU pair inside `_infer_slug`: the pre-read `stat()` check
        is not sufficient on its own, so `len(data)` is re-checked after the read. A
        second request is queued so `_infer_slug` actually runs its own stat/read
        instead of short-circuiting on a single pending request."""
        record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        source = _source(tmp_path, "download.pdf", f"{TITLE} DOI: {DOI} " + "x" * 5000)
        real_stat = Path.stat

        class _ShrunkStat:
            def __init__(self, real: object) -> None:
                self._real = real

            @property
            def st_size(self) -> int:
                return 10

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

        def _lying_stat(self: Path, *args: object, **kwargs: object) -> object:
            real = real_stat(self, *args, **kwargs)
            if self == source:
                return _ShrunkStat(real)
            return real

        monkeypatch.setattr(Path, "stat", _lying_stat)

        with pytest.raises(ValueError, match="over the 100 cap"):
            admit_file(tmp_path, source, max_bytes=100)

    def test_dropping_a_second_time_overwrites_the_first_inbox_copy(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        """After a rejected first attempt, the operator's very next action is to drop
        the correct file under the same slug -- this must not require them to delete
        the stale copy by hand."""
        wrong = _source(tmp_path, "wrong.pdf", "An entirely different paper about catalytic converters.")
        first = admit_file(tmp_path, wrong, slug=queued.slug, max_bytes=10_000_000)
        assert first.status == AcquisitionStatus.REJECTED

        correct = _source(tmp_path, "correct.pdf", f"{TITLE}\nDOI: {DOI}\n")
        second = admit_file(tmp_path, correct, slug=queued.slug, max_bytes=10_000_000)

        assert second.status == AcquisitionStatus.FULFILLED
        assert second.fulfilled_sha256
        drop_path = drop_path_for(tmp_path, queued.slug)
        assert drop_path.read_bytes() == correct.read_bytes()


class TestAlreadyAcquired:
    """Re-offering a paper Carmel already holds is a no-op, not a rejection.

    The batch case this covers was observed live: an ingest re-run over a download
    folder produced five "rejections" of papers that were in fact already in the
    evidence store, three of them with a note accusing a correct paper of not looking
    like itself. Every outcome was ultimately right, but the report was actively
    misleading, which is how an operator learns to stop trusting the verdicts.
    """

    @pytest.fixture
    def queued(self, tmp_path: Path) -> AcquisitionRequest:
        return record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/" + DOI,
            reason=AcquisitionReason.PAYWALLED,
        )

    def _fulfil(self, tmp_path: Path, queued: AcquisitionRequest) -> Path:
        source = _source(tmp_path, "paper.pdf", f"{TITLE}\nDOI: {DOI}\nAbstract: measurements follow.")
        request = admit_file(tmp_path, source, max_bytes=10_000_000)
        assert request.status == AcquisitionStatus.FULFILLED
        return source

    def test_the_same_bytes_offered_twice_are_reported_as_already_acquired(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        source = self._fulfil(tmp_path, queued)

        with pytest.raises(AlreadyAcquired) as caught:
            admit_file(tmp_path, source, max_bytes=10_000_000)

        assert caught.value.slug == queued.slug

    def test_already_acquired_is_a_valueerror_so_untaught_callers_do_not_crash(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        """Callers that predate this distinction catch ValueError around admit_file. The
        new signal must refine their behaviour, never turn it into an unhandled crash."""
        source = self._fulfil(tmp_path, queued)

        with pytest.raises(ValueError):
            admit_file(tmp_path, source, max_bytes=10_000_000)

    def test_a_different_copy_of_an_acquired_paper_is_recognised_by_content(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        """The same work re-downloaded from another source has different bytes, so the
        hash check cannot see it; the identity check must."""
        self._fulfil(tmp_path, queued)
        second = record_request(
            tmp_path,
            title=SECOND_TITLE,
            doi=SECOND_DOI,
            landing_url="https://doi.org/" + SECOND_DOI,
            reason=AcquisitionReason.PAYWALLED,
        )
        assert second.status == AcquisitionStatus.REQUESTED
        other_copy = _source(tmp_path, "other-copy.pdf", f"{TITLE}\nDOI: {DOI}\nA different rendering entirely.")

        with pytest.raises(AlreadyAcquired) as caught:
            admit_file(tmp_path, other_copy, max_bytes=10_000_000)

        assert caught.value.slug == queued.slug
        # The unrelated outstanding request must be untouched -- not rejected, not noted.
        reloaded = {r.slug: r for r in load_manifest(tmp_path).requests}
        assert reloaded[second.slug].status == AcquisitionStatus.REQUESTED
        assert not reloaded[second.slug].identity_note

    def test_an_already_acquired_paper_never_displaces_the_stored_artifact(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        source = self._fulfil(tmp_path, queued)
        fulfilled = {r.slug: r for r in load_manifest(tmp_path).requests}[queued.slug]
        before = fulfilled.fulfilled_sha256

        with pytest.raises(AlreadyAcquired):
            admit_file(tmp_path, source, max_bytes=10_000_000)

        after = {r.slug: r for r in load_manifest(tmp_path).requests}[queued.slug]
        assert after.status == AcquisitionStatus.FULFILLED
        assert after.fulfilled_sha256 == before


class TestSoleRequestAttributionIsLabelled:
    """A file attributed to the last outstanding request must say that is why.

    Live defect this closes: with one request left pending, four unrelated files were
    each attributed to it and each flipped it to REJECTED, the surviving note describing
    whichever file happened to sort last. The attribution is kept -- it is what produces
    a real diagnostic instead of a bare "matched nothing" -- but it must not read as an
    assertion the operator never made.
    """

    @pytest.fixture
    def queued(self, tmp_path: Path) -> AcquisitionRequest:
        return record_request(
            tmp_path,
            title=TITLE,
            doi=DOI,
            landing_url="https://doi.org/" + DOI,
            reason=AcquisitionReason.PAYWALLED,
        )

    def test_an_unrelated_file_is_rejected_but_the_note_says_it_was_never_offered(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        unrelated = _source(tmp_path, "unrelated.pdf", "Micron sized aluminium particle combustion in a shock tube")

        request = admit_file(tmp_path, unrelated, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.REJECTED
        assert "sole one left" in request.identity_note
        # The underlying diagnostic survives the prefix; it is what tells the operator
        # WHY, and losing it was the regression that made a blunter fix unacceptable.
        assert "does not look like this paper" in request.identity_note

    def test_an_explicitly_slugged_mismatch_is_not_labelled_that_way(
        self, tmp_path: Path, queued: AcquisitionRequest
    ) -> None:
        """Passing slug= IS the operator asserting this file is for that paper, so the
        note must report the mismatch plainly rather than excusing it."""
        unrelated = _source(tmp_path, "unrelated.pdf", "Micron sized aluminium particle combustion in a shock tube")

        request = admit_file(tmp_path, unrelated, slug=queued.slug, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.REJECTED
        assert "sole one left" not in request.identity_note

    def test_a_matching_file_is_not_labelled_as_a_fallback(self, tmp_path: Path, queued: AcquisitionRequest) -> None:
        correct = _source(tmp_path, "correct.pdf", f"{TITLE}\nDOI: {DOI}\nAbstract: measurements follow.")

        request = admit_file(tmp_path, correct, max_bytes=10_000_000)

        assert request.status == AcquisitionStatus.FULFILLED
        assert "sole one left" not in (request.identity_note or "")


class TestInferencePrefersTheStrongerSignal:
    """When several outstanding papers match, a printed DOI outranks title overlap.

    A focused campaign queues papers from one subfield, whose titles overlap by
    construction, so title-only collisions are the normal case rather than an edge one.
    This narrows an already-passing set -- it can never admit something check_identity
    refused, which is the only safe direction: an erratum scores ~0.95 against its own
    paper's title, so any fuzzy similarity score would reopen the defect the
    secondary-document gate exists to close.
    """

    def test_the_doi_bearing_candidate_wins_over_a_title_only_collision(self, tmp_path: Path) -> None:
        shared = "Laminar flame speeds of syngas hydrogen carbon monoxide air mixtures"
        first = record_request(
            tmp_path,
            title=shared,
            doi="10.1016/j.first.2020.01.001",
            landing_url="https://doi.org/10.1016/j.first.2020.01.001",
            reason=AcquisitionReason.PAYWALLED,
        )
        second = record_request(
            tmp_path,
            title=shared + " at elevated pressure",
            doi="10.1016/j.second.2020.01.002",
            landing_url="https://doi.org/10.1016/j.second.2020.01.002",
            reason=AcquisitionReason.PAYWALLED,
        )
        # Text matching BOTH titles by word overlap, but printing only the second's DOI.
        source = _source(
            tmp_path,
            "ambiguous.pdf",
            f"{shared} at elevated pressure\nDOI: {second.doi}\nAbstract: measurements follow.",
        )

        request = admit_file(tmp_path, source, max_bytes=10_000_000)

        assert request.slug == second.slug
        assert request.status == AcquisitionStatus.FULFILLED
        reloaded = {r.slug: r for r in load_manifest(tmp_path).requests}
        assert reloaded[first.slug].status == AcquisitionStatus.REQUESTED

    def test_two_candidates_with_neither_doi_present_still_refuse_to_guess(self, tmp_path: Path) -> None:
        """The tie-break is a preference among stronger evidence, not a licence to pick
        one when no stronger evidence exists.

        The two titles must differ (a title-derived slug makes identical titles one
        request, not two) while both being fully contained in the document, which is
        exactly the shape a paper and its extended companion study take.
        """
        narrower = "Laminar flame speeds of syngas hydrogen carbon monoxide air mixtures"
        broader = narrower + " at elevated pressure"
        record_request(
            tmp_path,
            title=narrower,
            doi=None,
            landing_url="https://example.org/first",
            reason=AcquisitionReason.PAYWALLED,
        )
        record_request(
            tmp_path,
            title=broader,
            doi=None,
            landing_url="https://example.org/second",
            reason=AcquisitionReason.PAYWALLED,
        )
        source = _source(tmp_path, "ambiguous.pdf", f"{broader}\nAbstract: measurements follow.")

        with pytest.raises(ValueError, match="cannot tell which pending request"):
            admit_file(tmp_path, source, max_bytes=10_000_000)

        evidence = tmp_path / "evidence" / "literature"
        assert not evidence.exists() or not any(evidence.iterdir())
