"""Tests for the supplement resolve/propose/fetch path (I-048).

The operator's ruling: resolve and PROPOSE by default; FETCH automatically only from
hosts on an explicit allowlist; write a RECEIPT (URL + timestamp + hash) for every
automatic fetch. Every test here uses a local fetch double -- nothing reaches the
network, which is itself a requirement (a suite that silently depends on a publisher
being up is not a suite).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.fetch import FetchedArtifact, FetchError
from carmel.schemas.acquisition import (
    AcquisitionReason,
    AcquisitionRequest,
    SupplementaryCandidate,
    SupplementaryFile,
)
from carmel.services.acquisition import (
    SupplementAcquisitionError,
    SupplementFetchFailed,
    SupplementHostNotAllowlisted,
    SupplementReceiptMismatch,
    _safe_supplement_filename,
    _verify_staged_hash,
    acquire_supplements,
    fetch_supplement,
    load_manifest,
    propose_supplements,
    record_request,
    resolve_supplements,
    supplement_host_is_fetchable,
    supplementary_dir,
)

ALLOWLISTED_URL = "https://ars.els-cdn.com/content/image/1-s2.0-x/mmc1.xlsx"
PAYWALLED_URL = "https://www.sciencedirect.com/science/article/pii/x/mmc1.xlsx"
XLSX_BYTES = b"PK\x03\x04 not really an xlsx, just bytes with a receipt"


# ---------------------------------------------------------------------------- doubles


class _NeverCalledFetch:
    """A fetch tool that fails the test the instant it is asked to fetch anything.

    Used wherever the code MUST decide without touching the network -- the propose path,
    and the refusal of a non-allowlisted host before any request is made.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]:
        self.calls.append(url)
        raise AssertionError(f"the network was contacted for {url!r}, but it must not have been")


class _CannedFetch:
    """Return canned bytes for one URL, with a controllable post-redirect final_url."""

    def __init__(self, data: bytes, *, final_url: str | None = None) -> None:
        self._data = data
        self._final_url = final_url
        self.calls: list[str] = []

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]:
        self.calls.append(url)
        artifact = FetchedArtifact(
            url=url,
            final_url=self._final_url or url,
            sha256=hashlib.sha256(self._data).hexdigest(),
            content_type="application/octet-stream",
            n_bytes=len(self._data),
            fetched_at=datetime.now(UTC),
        )
        return artifact, self._data


class _RaisingFetch:
    """Raise a given FetchError -- how a real network/HTTP/truncation failure surfaces."""

    def __init__(self, error: FetchError) -> None:
        self._error = error
        self.calls: list[str] = []

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]:
        self.calls.append(url)
        raise self._error


# ---------------------------------------------------------------------------- helpers


def _seed_request(workspace_root: Path) -> AcquisitionRequest:
    return record_request(
        workspace_root,
        title="A shock-tube study of ignition delay times",
        doi="10.1000/example",
        landing_url="https://doi.org/10.1000/example",
        reason=AcquisitionReason.NOT_A_DOCUMENT,
    )


def _staged_files(workspace_root: Path) -> list[Path]:
    root = supplementary_dir(workspace_root)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


# ---------------------------------------------------------------- V1: allowlist guard


class TestNonAllowlistedHostIsRefused:
    def test_a_non_allowlisted_host_is_refused_before_any_fetch(self, tmp_path: Path) -> None:
        """Verifier 1: the guard the whole ruling rests on. A non-allowlisted host is
        refused with a typed error that NAMES the host, before the network is touched."""
        parent = _seed_request(tmp_path)
        fetch = _NeverCalledFetch()
        candidate = SupplementaryCandidate(
            url=PAYWALLED_URL, believed_filename="mmc1.xlsx", believed_kind="spreadsheet"
        )

        with pytest.raises(SupplementHostNotAllowlisted) as caught:
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        assert caught.value.host == "www.sciencedirect.com"
        assert "not on the supplement-fetch allowlist" in str(caught.value)
        assert fetch.calls == []  # the host was never contacted
        assert _staged_files(tmp_path) == []  # nothing staged

    def test_the_refusal_is_distinct_from_a_network_failure(self, tmp_path: Path) -> None:
        """Verifier 1: "we refused to try" must never look like "we tried and it failed"."""
        parent = _seed_request(tmp_path)
        candidate = SupplementaryCandidate(url=PAYWALLED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementHostNotAllowlisted) as caught:
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=_NeverCalledFetch(), max_bytes=10_000_000)

        assert not isinstance(caught.value, SupplementFetchFailed)
        assert not isinstance(caught.value, FetchError)


# ------------------------------------------------------------ V2: fetch + receipt


class TestAllowlistedHostIsFetched:
    def test_an_allowlisted_host_is_fetched_and_produces_a_receipt(self, tmp_path: Path) -> None:
        """Verifier 2: the receipt records the URL and a timestamp, and its hash matches
        the bytes actually on disk."""
        parent = _seed_request(tmp_path)
        fetch = _CannedFetch(XLSX_BYTES)
        candidate = SupplementaryCandidate(
            url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx", believed_kind="spreadsheet"
        )

        receipt = fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        # The receipt: URL, timestamp, hash.
        assert receipt.source_url == ALLOWLISTED_URL
        assert isinstance(receipt.received_at, datetime)
        assert receipt.sha256 == hashlib.sha256(XLSX_BYTES).hexdigest()

        # The hash matches the bytes actually on disk.
        staged = tmp_path / receipt.staged_path
        assert staged.is_file()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == receipt.sha256
        assert staged.read_bytes() == XLSX_BYTES

    def test_the_receipt_persists_in_the_manifest(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        fetch_supplement(tmp_path, parent.slug, candidate, fetch=_CannedFetch(XLSX_BYTES), max_bytes=10_000_000)

        reloaded = load_manifest(tmp_path)
        (request,) = reloaded.requests
        (receipt,) = request.supplementary
        assert receipt.source_url == ALLOWLISTED_URL
        assert receipt.sha256 == hashlib.sha256(XLSX_BYTES).hexdigest()

    def test_a_second_fetch_of_the_same_bytes_is_idempotent(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        first = fetch_supplement(tmp_path, parent.slug, candidate, fetch=_CannedFetch(XLSX_BYTES), max_bytes=10_000_000)
        second = fetch_supplement(
            tmp_path, parent.slug, candidate, fetch=_CannedFetch(XLSX_BYTES), max_bytes=10_000_000
        )

        assert first.sha256 == second.sha256
        (request,) = load_manifest(tmp_path).requests
        assert len(request.supplementary) == 1  # recorded once, not twice


# --------------------------------------------------- V3: receipt-vs-bytes verification


class TestReceiptHashIsVerifiedAgainstDisk:
    def test_a_receipt_whose_hash_does_not_match_the_bytes_is_refused(self, tmp_path: Path) -> None:
        """Verifier 3: a receipt that is never checked against reality is decoration."""
        staged = tmp_path / "staged.bin"
        staged.write_bytes(b"the real bytes")
        wrong_digest = hashlib.sha256(b"different bytes").hexdigest()

        with pytest.raises(SupplementReceiptMismatch) as caught:
            _verify_staged_hash(staged, wrong_digest, url=ALLOWLISTED_URL)

        assert caught.value.expected_sha256 == wrong_digest
        assert caught.value.actual_sha256 == hashlib.sha256(b"the real bytes").hexdigest()

    def test_a_matching_hash_is_accepted(self, tmp_path: Path) -> None:
        staged = tmp_path / "staged.bin"
        staged.write_bytes(b"the real bytes")
        _verify_staged_hash(staged, hashlib.sha256(b"the real bytes").hexdigest(), url=ALLOWLISTED_URL)  # no raise


# -------------------------------------------------------------------- V4: propose path


class TestProposePathIsActionableWithoutFetch:
    def test_propose_returns_non_allowlisted_candidates_and_never_fetches(self, tmp_path: Path) -> None:
        """Verifier 4: the default outcome is a proposal, produced with zero network I/O."""
        candidates = [
            SupplementaryCandidate(
                url="https://onlinelibrary.wiley.com/x/si.docx", believed_filename="si.docx", believed_kind="document"
            ),
            SupplementaryCandidate(
                url="https://pubs.acs.org/x/si.pdf", believed_filename="si.pdf", believed_kind="pdf"
            ),
        ]

        proposed = propose_supplements(candidates)

        assert proposed == candidates
        # Actionable on its own: each proposal names its URL and what the file is.
        assert all(p.url and p.believed_filename and p.believed_kind for p in proposed)

    def test_acquire_proposes_non_allowlisted_and_fetches_allowlisted(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidates = [
            SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx"),
            SupplementaryCandidate(url=PAYWALLED_URL, believed_filename="mmc2.pdf"),
        ]
        fetch = _CannedFetch(XLSX_BYTES)

        result = acquire_supplements(tmp_path, parent.slug, candidates, fetch=fetch, max_bytes=10_000_000)

        assert [r.source_url for r in result.fetched] == [ALLOWLISTED_URL]
        assert [c.url for c in result.proposed] == [PAYWALLED_URL]
        assert fetch.calls == [ALLOWLISTED_URL]  # the paywalled host was never contacted

    def test_acquire_with_no_allowlisted_candidate_never_fetches(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidates = [SupplementaryCandidate(url=PAYWALLED_URL, believed_filename="mmc1.xlsx")]
        fetch = _NeverCalledFetch()

        result = acquire_supplements(tmp_path, parent.slug, candidates, fetch=fetch, max_bytes=10_000_000)

        assert result.fetched == []
        assert [c.url for c in result.proposed] == [PAYWALLED_URL]
        assert fetch.calls == []


# ------------------------------------------------------------- V5: failures fail closed


class TestFetchFailuresFailClosed:
    def test_a_network_failure_fails_closed_with_no_partial_file(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        fetch = _RaisingFetch(FetchError("fetch failed for host: connection refused"))
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementFetchFailed):
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        assert _staged_files(tmp_path) == []
        assert load_manifest(tmp_path).requests[0].supplementary == []

    def test_an_http_error_fails_closed_and_carries_the_status(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        fetch = _RaisingFetch(FetchError("HTTP 404", status=404))
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementFetchFailed) as caught:
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        assert caught.value.status == 404
        assert _staged_files(tmp_path) == []

    def test_a_truncated_response_fails_closed(self, tmp_path: Path) -> None:
        """A truncated transfer surfaces from the fetch tool as a FetchError (IncompleteRead
        or an over-cap chunk); it must never be presented as a short successful file."""
        parent = _seed_request(tmp_path)
        fetch = _RaisingFetch(FetchError("fetch failed for host: IncompleteRead(12 bytes read)"))
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementFetchFailed):
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        assert _staged_files(tmp_path) == []

    def test_an_over_cap_response_fails_closed(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        fetch = _CannedFetch(b"x" * 5000)
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementFetchFailed):
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=1000)

        assert _staged_files(tmp_path) == []


# ------------------------------------------------------ redirect off the allowlist


class TestRedirectOffAllowlistIsRefused:
    def test_a_redirect_to_a_non_allowlisted_host_is_refused(self, tmp_path: Path) -> None:
        """The entry host is allowlisted but the fetch tool followed a redirect off it.
        The bytes' true origin is not allowlisted, so they are refused and never staged."""
        parent = _seed_request(tmp_path)
        fetch = _CannedFetch(XLSX_BYTES, final_url="https://cdn.evil.example.com/mmc1.xlsx")
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        with pytest.raises(SupplementHostNotAllowlisted) as caught:
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=fetch, max_bytes=10_000_000)

        assert caught.value.host == "cdn.evil.example.com"
        assert _staged_files(tmp_path) == []


# ------------------------------------------- receipt-write failure rolls back bytes


class TestReceiptWriteFailureRollsBack:
    def test_a_fetch_that_cannot_write_its_receipt_removes_the_staged_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Position: bytes on disk whose origin was never recorded are exactly the
        unprovenanced state this path exists to prevent, so a receipt that cannot be
        written rolls the bytes back and fails closed."""
        parent = _seed_request(tmp_path)
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")

        import carmel.services.acquisition as acq

        def _boom(workspace_root: Path, manifest: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(acq, "save_manifest", _boom)

        with pytest.raises(SupplementAcquisitionError):
            fetch_supplement(tmp_path, parent.slug, candidate, fetch=_CannedFetch(XLSX_BYTES), max_bytes=10_000_000)

        assert _staged_files(tmp_path) == []  # the bytes were rolled back


# ---------------------------------------------------------------------- edges & seams


class TestSafeSupplementFilename:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("mmc1.xlsx", "mmc1.xlsx"),
            ("../../etc/passwd", "passwd"),
            ("a/b/c.csv", "c.csv"),
            ("..", "supplement"),
            ("", "supplement"),
            ("weird name (1).xlsx", "weird_name_1_.xlsx"),
        ],
    )
    def test_it_reduces_to_a_safe_basename(self, given: str, expected: str) -> None:
        assert _safe_supplement_filename(given) == expected

    def test_a_traversing_filename_stays_inside_the_staging_dir(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidate = SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="../../../../etc/passwd")

        receipt = fetch_supplement(
            tmp_path, parent.slug, candidate, fetch=_CannedFetch(XLSX_BYTES), max_bytes=10_000_000
        )

        staged = (tmp_path / receipt.staged_path).resolve()
        assert supplementary_dir(tmp_path).resolve() in staged.parents


class TestResolveSeam:
    def test_resolve_returns_whatever_the_resolver_names(self, tmp_path: Path) -> None:
        parent = _seed_request(tmp_path)
        candidates = [SupplementaryCandidate(url=ALLOWLISTED_URL, believed_filename="mmc1.xlsx")]

        class _Resolver:
            def resolve(self, request: AcquisitionRequest) -> list[SupplementaryCandidate]:
                assert request.slug == parent.slug
                return candidates

        assert resolve_supplements(parent, _Resolver()) == candidates


class TestSupplementFetchAllowlist:
    def test_the_open_elsevier_cdn_and_its_subdomains_are_fetchable(self) -> None:
        assert supplement_host_is_fetchable("https://ars.els-cdn.com/content/mmc1.xlsx")
        assert supplement_host_is_fetchable("https://sub.ars.els-cdn.com/mmc1.xlsx")

    def test_publisher_hosts_admissible_for_the_store_are_not_fetchable(self) -> None:
        # els-cdn.com is admissible for STORE ENTRY, but only ars.els-cdn.com is fetchable:
        # the two lists are distinct on purpose.
        assert not supplement_host_is_fetchable("https://www.sciencedirect.com/x")
        assert not supplement_host_is_fetchable("https://els-cdn.com/x")

    def test_a_lookalike_suffix_is_not_fetchable(self) -> None:
        assert not supplement_host_is_fetchable("https://ars.els-cdn.com.evil.net/x")

    def test_the_operator_extension_point_widens_the_allowlist(self) -> None:
        assert supplement_host_is_fetchable("https://mirror.my-lab.edu/x", ["mirror.my-lab.edu"])


class TestManualDropReceiptHasNoSourceUrl:
    def test_a_supplementary_file_defaults_source_url_to_none(self) -> None:
        si = SupplementaryFile(
            sha256="a" * 64,
            original_filename="x.si.zip",
            parent_slug="x",
            content_type="application/zip",
            received_at=datetime.now(UTC),
            size_bytes=3,
            staged_path="literature_requests/supplementary/" + "a" * 64 + "/x.si.zip",
        )
        assert si.source_url is None
