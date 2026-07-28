"""Tests for carmel.agents.tools.fetch."""

from __future__ import annotations

import http.server
import threading
import urllib.request
from collections.abc import Iterator

import pytest

from carmel.agents.budget import BudgetExceededError, BudgetLedger, session_budget
from carmel.agents.tools.fetch import (
    CHUNK_SIZE,
    FetchedArtifact,
    FetchError,
    HttpFetchTool,
    MockFetchTool,
    _build_opener,
    _default_opener,
    is_safe_url,
    sniff_content_type,
)
from carmel.config import AgentBudgetConfig


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    session_budget().reset()
    yield
    session_budget().reset()


def make_ledger(**limits: object) -> BudgetLedger:
    limits_obj = AgentBudgetConfig(**limits)  # type: ignore[arg-type]
    return BudgetLedger(limits_obj)


def resolver_map(mapping: dict[str, list[str]]):
    def _resolve(hostname: str) -> list[str]:
        return mapping.get(hostname, [])

    return _resolve


# --------------------------------------------------------------------------- #
# is_safe_url
# --------------------------------------------------------------------------- #


class TestIsSafeUrl:
    def test_rejects_localhost(self) -> None:
        assert is_safe_url("http://localhost/") is False

    def test_rejects_ipv4_loopback(self) -> None:
        assert is_safe_url("http://127.0.0.1/") is False

    def test_rejects_private_ipv4_via_resolver(self) -> None:
        resolver = resolver_map({"evil.example": ["10.0.0.1"]})
        assert is_safe_url("http://evil.example/", resolver=resolver) is False

    def test_rejects_cloud_metadata_address(self) -> None:
        resolver = resolver_map({"metadata.example": ["169.254.169.254"]})
        assert is_safe_url("http://metadata.example/", resolver=resolver) is False

    def test_rejects_ipv6_loopback(self) -> None:
        assert is_safe_url("http://[::1]/") is False

    def test_rejects_ipv4_mapped_ipv6_loopback(self) -> None:
        assert is_safe_url("http://[::ffff:127.0.0.1]/") is False

    def test_rejects_file_scheme(self) -> None:
        assert is_safe_url("file:///etc/passwd") is False

    def test_rejects_ftp_scheme(self) -> None:
        assert is_safe_url("ftp://x/") is False

    def test_rejects_url_with_no_host(self) -> None:
        assert is_safe_url("http:///path") is False

    def test_rejects_unresolvable_host(self) -> None:
        resolver = resolver_map({})
        assert is_safe_url("http://nowhere.example/", resolver=resolver) is False

    def test_accepts_public_address(self) -> None:
        resolver = resolver_map({"good.example": ["93.184.216.34"]})
        assert is_safe_url("http://good.example/", resolver=resolver) is True

    def test_rejects_when_any_resolved_address_is_private(self) -> None:
        # public AND private -> all-must-be-global, so this is unsafe.
        resolver = resolver_map({"mixed.example": ["93.184.216.34", "10.0.0.5"]})
        assert is_safe_url("http://mixed.example/", resolver=resolver) is False

    def test_rejects_disallowed_port(self) -> None:
        # Finding 13: a globally-routable host is still unsafe if the LLM-chosen URL
        # targets a non-standard port -- e.g. an internal Redis/Elasticsearch/Docker
        # daemon listening on the same otherwise-public host.
        resolver = resolver_map({"good.example": ["93.184.216.34"]})
        assert is_safe_url("http://good.example:6379/", resolver=resolver) is False

    def test_accepts_no_explicit_port(self) -> None:
        resolver = resolver_map({"good.example": ["93.184.216.34"]})
        assert is_safe_url("http://good.example/", resolver=resolver) is True

    def test_accepts_explicit_standard_ports(self) -> None:
        resolver = resolver_map({"good.example": ["93.184.216.34"]})
        assert is_safe_url("http://good.example:80/", resolver=resolver) is True
        assert is_safe_url("https://good.example:443/", resolver=resolver) is True


# --------------------------------------------------------------------------- #
# sniff_content_type
# --------------------------------------------------------------------------- #


class TestSniffContentType:
    def test_pdf_magic_bytes(self) -> None:
        assert sniff_content_type(b"%PDF-1.4 rest of file") == "application/pdf"

    def test_html(self) -> None:
        assert sniff_content_type(b"<html><body>hi</body></html>") == "text/html"

    def test_html_doctype(self) -> None:
        assert sniff_content_type(b"<!DOCTYPE html><html></html>") == "text/html"

    def test_utf8_text(self) -> None:
        assert sniff_content_type(b"plain text content") == "text/plain"

    def test_binary_garbage(self) -> None:
        assert sniff_content_type(bytes(range(200))) == "application/octet-stream"


# --------------------------------------------------------------------------- #
# HttpFetchTool
# --------------------------------------------------------------------------- #


class FakeResponse:
    """A fake chunked HTTP response for injection into HttpFetchTool."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "",
    ) -> None:
        self._chunks = list(chunks)
        self.status = status
        self._headers = headers or {}
        self.url = url
        self.closed = False
        self.chunks_served = 0

    def read(self, amt: int) -> bytes:
        if not self._chunks:
            return b""
        self.chunks_served += 1
        return self._chunks.pop(0)

    def getheader(self, name: str, default: str = "") -> str:
        return self._headers.get(name, default)

    def close(self) -> None:
        self.closed = True


def make_opener(responses_by_url: dict[str, FakeResponse]):
    calls: list[str] = []

    def _opener(url: str, *, timeout_s: float) -> FakeResponse:
        calls.append(url)
        if url not in responses_by_url:
            raise RuntimeError(f"no fake response for {url}")
        return responses_by_url[url]

    _opener.calls = calls  # type: ignore[attr-defined]
    return _opener


PUBLIC_RESOLVER = resolver_map({"good.example": ["93.184.216.34"], "redirect.example": ["93.184.216.34"]})


class TestHttpFetchTool:
    def test_fetch_success_sniffs_content_type_from_bytes(self) -> None:
        ledger = make_ledger()
        body = b"<html>hello</html>"
        response = FakeResponse([body], headers={"Content-Type": "application/pdf"}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER)

        artifact, data = tool.fetch("http://good.example/")

        assert data == body
        assert artifact.content_type == "text/html"  # lying header ignored
        assert artifact.n_bytes == len(body)
        assert isinstance(artifact, FetchedArtifact)
        import hashlib

        assert artifact.sha256 == hashlib.sha256(body).hexdigest()

    def test_rejects_unsafe_url_before_any_open(self) -> None:
        ledger = make_ledger()
        opener = make_opener({})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=resolver_map({}))

        with pytest.raises(FetchError):
            tool.fetch("http://localhost/")

        assert opener.calls == []  # type: ignore[attr-defined]

    def test_manual_redirect_to_private_ip_is_rejected(self) -> None:
        ledger = make_ledger()
        first = FakeResponse([], status=302, headers={"Location": "http://evil.example/"}, url="http://good.example/")
        resolver = resolver_map({"good.example": ["93.184.216.34"], "evil.example": ["10.0.0.1"]})
        opener = make_opener({"http://good.example/": first})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=resolver)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

    def test_exceeding_max_redirects_raises(self) -> None:
        ledger = make_ledger()
        resolver = PUBLIC_RESOLVER
        hop1 = FakeResponse([], status=302, headers={"Location": "http://redirect.example/2"}, url="")
        hop2 = FakeResponse([], status=302, headers={"Location": "http://redirect.example/3"}, url="")
        hop3 = FakeResponse([], status=302, headers={"Location": "http://redirect.example/4"}, url="")
        hop4 = FakeResponse([], status=302, headers={"Location": "http://redirect.example/5"}, url="")
        opener = make_opener(
            {
                "http://good.example/": hop1,
                "http://redirect.example/2": hop2,
                "http://redirect.example/3": hop3,
                "http://redirect.example/4": hop4,
            }
        )
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=resolver, max_redirects=2)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

    def test_chunked_stream_aborts_mid_download_past_artifact_cap(self) -> None:
        ledger = make_ledger(max_artifact_bytes=10)
        big_chunk = b"x" * CHUNK_SIZE
        # Many chunks; if the whole body were read we'd serve all of them.
        chunks = [big_chunk] * 20
        response = FakeResponse(chunks, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER, max_artifact_bytes=10)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

        # Aborted mid-stream: nowhere near all 20 chunks were consumed.
        assert response.chunks_served < 20
        assert response.chunks_served <= 2

    def test_content_length_header_early_exit(self) -> None:
        ledger = make_ledger(max_artifact_bytes=10)
        response = FakeResponse([b"x" * 100], headers={"Content-Length": "1000000"}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER, max_artifact_bytes=10)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

        # Rejected before ever reading a chunk.
        assert response.chunks_served == 0

    def test_content_length_lie_does_not_bypass_streaming_check(self) -> None:
        # Content-Length UNDER the cap, but actual streamed bytes exceed it: the
        # streaming check (not the header) must be what catches this.
        ledger = make_ledger(max_artifact_bytes=10)
        chunks = [b"x" * CHUNK_SIZE] * 5
        response = FakeResponse(chunks, headers={"Content-Length": "5"}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER, max_artifact_bytes=10)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")
        assert response.chunks_served < 5

    def test_ledger_settled_with_actual_bytes(self) -> None:
        ledger = make_ledger()
        body = b"hello world"
        response = FakeResponse([body], url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER)

        tool.fetch("http://good.example/")

        usage = ledger.usage()
        assert usage.fetch_bytes == len(body)

    def test_reservation_taken_before_opener_runs_and_abandoned_on_error(self) -> None:
        # Regression test (replaces a prior, weaker test that only happened to pass
        # because -- before the reservation-scope fix -- no reservation existed at all
        # before the opener ran). The reservation must now be visible to the opener
        # itself (i.e. taken BEFORE it runs), and abandoned when the opener raises.
        # Finding 10: abandoning no longer refunds the attempt itself -- a failed
        # fetch still consumed a real outbound request, so ``reserved_fetches`` stays
        # charged even though the byte reservation settles down to the true (zero)
        # transferred amount.
        ledger = make_ledger(max_fetches=3)
        usage_during_opener: dict[str, int] = {}

        def _raising_opener(url: str, *, timeout_s: float):
            usage_during_opener["fetches"] = ledger.usage().fetches
            raise RuntimeError("boom")

        tool = HttpFetchTool(ledger=ledger, opener=_raising_opener, resolver=PUBLIC_RESOLVER)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

        assert usage_during_opener["fetches"] == 1

        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 0

    def test_oversized_content_length_counts_against_max_fetches_and_keeps_attempt_charged(self) -> None:
        # The oversized-Content-Length rejection raises FetchError before any chunk is
        # read. Finding 10: the attempt itself must NOT be refunded -- it still
        # consumed a real outbound request -- so it permanently consumes the single
        # max_fetches=1 slot. The byte reservation, however, settles down to the true
        # zero bytes actually transferred (no body was ever read).
        ledger = make_ledger(max_artifact_bytes=10, max_fetches=1)
        response = FakeResponse([b"x" * 100], headers={"Content-Length": "1000000"}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER, max_artifact_bytes=10)

        with pytest.raises(FetchError):
            tool.fetch("http://good.example/")

        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 0
        # The failed attempt's slot is NOT leaked back: a second attempt is blocked.
        with pytest.raises(BudgetExceededError):
            ledger.reserve_fetch(estimated_bytes=1)

    def test_transport_error_during_read_keeps_attempt_charged(self) -> None:
        # Finding 10: a transport error mid-read still consumed a real outbound
        # request, so the attempt is NOT refunded. No bytes were actually read
        # before the error, so the byte reservation settles down to zero.
        ledger = make_ledger()

        class RaisingResponse(FakeResponse):
            def read(self, amt: int) -> bytes:
                raise OSError("connection reset")

        response = RaisingResponse([b"data"], url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER)

        with pytest.raises(OSError):
            tool.fetch("http://good.example/")

        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 0

    def test_redirect_with_missing_location_header_raises(self) -> None:
        ledger = make_ledger()
        response = FakeResponse([], status=302, headers={}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER)

        with pytest.raises(FetchError, match="no Location header"):
            tool.fetch("http://good.example/")

        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 0

    def test_non_numeric_content_length_falls_back_to_zero_and_does_not_reject(self) -> None:
        # A non-numeric Content-Length header must not crash the fetch; it falls back
        # to a declared size of 0 (never treated as oversized) and the real streaming
        # check still governs the actual cap.
        ledger = make_ledger(max_artifact_bytes=10)
        body = b"hi"
        response = FakeResponse([body], headers={"Content-Length": "not-a-number"}, url="http://good.example/")
        opener = make_opener({"http://good.example/": response})
        tool = HttpFetchTool(ledger=ledger, opener=opener, resolver=PUBLIC_RESOLVER, max_artifact_bytes=10)

        artifact, data = tool.fetch("http://good.example/")

        assert data == body
        assert artifact.n_bytes == len(body)


class TestBuildOpener:
    def test_proxies_are_disabled_not_left_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression test for Defect 5: `build_opener(_NoRedirect)` alone implicitly
        # includes a default ProxyHandler that reads http_proxy/https_proxy/no_proxy env
        # vars, letting an env-configured proxy silently redirect the real TCP connection
        # and bypass the SSRF guard. Passing an explicit, empty ProxyHandler({}) instead
        # suppresses that default -- it has no per-protocol `<type>_open` methods (since
        # its `proxies` mapping is empty), so it never actually registers itself as an
        # `http`/`https` opener, unlike the env-reading default. With env proxy vars set,
        # the built opener's http/https openers must be exactly the plain (non-proxying)
        # handlers -- no ProxyHandler in the dispatch chain for either protocol.
        monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
        monkeypatch.setenv("https_proxy", "http://proxy.example:8080")

        opener = _build_opener()

        for protocol in ("http", "https"):
            handler_types = [type(h) for h in opener.handle_open[protocol]]
            assert urllib.request.ProxyHandler not in handler_types, (
                f"{protocol} dispatch chain unexpectedly contains a ProxyHandler: {handler_types}"
            )


class TestMockFetchTool:
    def test_returns_canned_response(self) -> None:
        body = b"%PDF-1.4 canned"
        tool = MockFetchTool({"http://good.example/doc.pdf": (body, "application/pdf")})

        artifact, data = tool.fetch("http://good.example/doc.pdf")

        assert data == body
        assert artifact.content_type == "application/pdf"
        assert artifact.n_bytes == len(body)

    def test_unknown_url_raises_fetch_error(self) -> None:
        tool = MockFetchTool({})
        with pytest.raises(FetchError):
            tool.fetch("http://unregistered.example/")


# --------------------------------------------------------------------------- #
# Finding 2 regression: real redirects must come back as a response, not raise
# --------------------------------------------------------------------------- #
#
# `_NoRedirect.redirect_request` (in fetch.py) returns None for every 3xx so that
# HttpFetchTool.fetch's manual, one-hop-at-a-time redirect loop can re-run
# `is_safe_url` on each hop. But urllib's default error-handling chain raises
# `HTTPError` for ANY 3xx once redirect_request opts out -- if `_default_opener`
# didn't specifically catch and return that HTTPError, `fetch()`'s outer
# catch-all would turn every redirect into a `FetchError` before the manual loop
# (and its SSRF re-check) ever ran. The tests above (e.g.
# `test_manual_redirect_to_private_ip_is_rejected`) already prove the loop's
# per-hop SSRF re-check fires correctly -- but only via an *injected* fake
# opener, which can't catch a regression in the real `_default_opener`/urllib
# wiring. These tests drive a REAL redirect through real urllib machinery
# against a local HTTP server to close that gap.


class _RedirectHandler(http.server.BaseHTTPRequestHandler):
    """Minimal local HTTP server: serves a 302 chain plus a final 200 body."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/redirect-ok":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/redirect-to-private":
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.5/secret")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/final":
            body = b"final body"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args: object) -> None:  # noqa: D102 - silence test noise
        pass


@pytest.fixture
def local_http_server() -> Iterator[http.server.ThreadingHTTPServer]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class TestRealRedirectRegression:
    def test_default_opener_returns_3xx_response_instead_of_raising(
        self, local_http_server: http.server.ThreadingHTTPServer
    ) -> None:
        # The critical regression proof for Finding 2: drive `_default_opener`
        # directly against real urllib machinery and a real 302 response, and
        # confirm it comes back as a response object (status == 302) rather than
        # urllib.error.HTTPError propagating out.
        #
        # Note: this deliberately calls `_default_opener` directly rather than going
        # through `HttpFetchTool.fetch()` / `is_safe_url`. `local_http_server` binds an
        # ephemeral high port (by construction, since it's a test server on 127.0.0.1),
        # and `is_safe_url`'s Finding-13 port allowlist ({80, 443, or no port}) would
        # reject that port regardless of host globality -- on top of 127.0.0.1 already
        # failing the address-globality check. Neither of those gates is what this test
        # is proving; this test isolates the "3xx returns instead of raises" fix against
        # real urllib machinery, independent of the SSRF gate (which the
        # injected-opener tests above already exercise).
        port = local_http_server.server_address[1]
        url = f"http://127.0.0.1:{port}/redirect-ok"

        response = _default_opener(url, timeout_s=5.0)
        try:
            assert getattr(response, "status", None) == 302
            assert response.getheader("Location") == "/final"
        finally:
            response.close()
