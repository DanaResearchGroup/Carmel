"""SSRF-guarded, budget-gated, streaming HTTP fetch tool.

Every URL handled here is chosen by an LLM and MUST be treated as attacker-controlled.
``is_safe_url`` is the SSRF guard: it resolves the hostname itself (never trusting a
caller-supplied IP) and requires EVERY resolved address (A and AAAA) to be global.
``HttpFetchTool`` re-runs that guard on every redirect hop manually — an automatic
redirect handler is the standard SSRF bypass (public first hop, private second hop) —
and streams the body in bounded chunks so a hostile multi-gigabyte response aborts
mid-download rather than after a full buffered read.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict

from carmel.agents.budget import BudgetLedger, Reservation
from carmel.logger import get_logger

logger = get_logger("agents.tools.fetch")

CHUNK_SIZE = 65536

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class FetchError(RuntimeError):
    """Raised for any fetch failure: unsafe URL, redirect loop, size cap, transport error."""


class FetchedArtifact(BaseModel):
    """Metadata describing a successfully fetched document."""

    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str
    sha256: str
    content_type: str
    n_bytes: int
    fetched_at: datetime


def _default_resolver(hostname: str) -> list[str]:
    """Resolve a hostname to its A/AAAA addresses using ``socket.getaddrinfo``.

    Args:
        hostname: The hostname to resolve.

    Returns:
        List of address strings. Empty if resolution fails.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return []
    return [str(info[4][0]) for info in infos]


def _is_global_address(addr: str) -> bool:
    """Classify a single IP address string as globally routable.

    Rejects private, loopback, link-local, multicast, reserved, unspecified, and
    site-local addresses, including the IPv4-mapped/IPv4-compatible IPv6 forms of
    those (e.g. ``::ffff:127.0.0.1``), by unmapping before classification.

    Args:
        addr: An IPv4 or IPv6 address string.

    Returns:
        True only if the address is global.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            ip = mapped

    site_local = isinstance(ip, ipaddress.IPv6Address) and ip.is_site_local
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or site_local
    ):
        return False
    return bool(ip.is_global)


def is_safe_url(url: str, *, resolver: Callable[[str], list[str]] | None = None) -> bool:
    """SSRF guard.

    Args:
        url: Candidate URL, presumed attacker-controlled.
        resolver: Injected hostname -> address-list resolver (tests only). Defaults to
            a real ``socket.getaddrinfo`` lookup.

    Returns:
        False unless: the scheme is exactly ``http`` or ``https``; a hostname is
        present; the port (if any) is ``80`` or ``443``; and every resolved address
        (A and AAAA) is global. A hostname that fails to resolve is treated as unsafe
        (fail closed).
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False

    if parts.scheme not in _ALLOWED_SCHEMES:
        return False

    hostname = parts.hostname
    if not hostname:
        return False

    # Reject any explicit non-standard port. Without this, an otherwise-global,
    # passing hostname/IP still lets an LLM-chosen URL reach arbitrary internal
    # services on that host (Redis 6379, Elasticsearch 9200, the Docker daemon
    # 2375, etc.) -- the globality check only validates the *address*, not which
    # port on it gets probed.
    try:
        port = parts.port
    except ValueError:
        return False
    if port is not None and port not in (80, 443):
        return False

    resolve = resolver if resolver is not None else _default_resolver
    addresses = resolve(hostname)
    if not addresses:
        return False

    return all(_is_global_address(addr) for addr in addresses)


def sniff_content_type(data: bytes) -> str:
    """Determine content type from magic bytes, never from server-supplied headers.

    Args:
        data: The full fetched body (or a leading slice sufficient to sniff).

    Returns:
        ``application/pdf``, ``text/html``, ``text/plain``, or
        ``application/octet-stream``.
    """
    if data.startswith(b"%PDF-"):
        return "application/pdf"

    head = data[:512].lstrip().lower()
    if head.startswith(b"<html") or head.startswith(b"<!doctype"):
        return "text/html"

    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    return "text/plain"


class HttpResponseProtocol(Protocol):
    """The chunked shape an injected ``opener`` must return.

    A whole-body ``.read()`` is FORBIDDEN: the size cap must be enforced while bytes
    are arriving, or a hostile multi-gigabyte response is fully downloaded before
    anyone checks.
    """

    url: str

    def read(self, amt: int) -> bytes: ...

    def getheader(self, name: str, default: str = "") -> str: ...

    def close(self) -> None: ...


class FetchToolProtocol(Protocol):
    """Structural type for anything that can fetch a URL's bytes."""

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]: ...


def _build_opener() -> urllib.request.OpenerDirector:
    """Build the urllib opener used for real fetches: no auto-redirects, no proxies.

    Proxies are deliberately disabled here. ``urllib.request.build_opener`` normally
    auto-installs a default ``ProxyHandler`` that honors ``http_proxy``/``https_proxy``/
    ``no_proxy`` environment variables. ``is_safe_url`` validates the resolved address
    of the ORIGIN url only; a proxy configured via those env vars could silently redirect
    the actual TCP connection to an address the SSRF guard never saw, bypassing it
    entirely. Passing an explicit ``ProxyHandler({})`` disables proxy use altogether (it
    does NOT fall back to the environment). Any genuine future need for proxy support
    must be added explicitly and the SSRF guard extended to validate the real,
    proxy-resolved endpoint too -- not just the origin URL.

    Returns:
        An ``OpenerDirector`` with redirects and proxies both disabled.
    """

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            return None

    return urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))


def _default_opener(url: str, *, timeout_s: float) -> HttpResponseProtocol:
    """Open ``url`` without following redirects automatically and without any proxy.

    Args:
        url: URL to open (already validated by ``is_safe_url``).
        timeout_s: Socket timeout in seconds.

    Returns:
        A chunked, non-redirecting, non-proxied HTTP response. For a 3xx status this
        is the (unfollowed) redirect response itself, not a 200; see the comment below
        on why that must NOT be allowed to raise.
    """
    opener = _build_opener()
    request = urllib.request.Request(url, headers={"User-Agent": "carmel-agentic-fetch/1"})
    try:
        return opener.open(request, timeout=timeout_s)  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        # _NoRedirect.redirect_request (above) returns None for every 3xx, which
        # stops HTTPRedirectHandler from auto-following -- but urllib's default
        # error-handling chain then falls through to HTTPDefaultErrorHandler, which
        # RAISES HTTPError for any 3xx status just as it would for a 4xx/5xx one.
        # HttpFetchTool.fetch's manual one-hop-at-a-time redirect loop (which
        # re-validates each hop's target with is_safe_url -- the actual SSRF defense
        # for redirect chains) needs the 3xx response returned, not raised, or it
        # never runs and every redirecting URL fails outright. urllib.error.HTTPError
        # is itself response-shaped (status/url/getheader/read/close all delegate
        # through urllib.response.addinfourl to the underlying HTTPResponse), so it
        # already satisfies HttpResponseProtocol -- just hand it back for 3xx. Do NOT
        # "simplify" this by letting HTTPError propagate for redirects: that silently
        # disables the per-hop SSRF re-check in production (it would only ever run
        # under tests that inject a fake opener). Non-3xx errors (4xx/5xx) are
        # re-raised unchanged so fetch()'s existing catch-all -> FetchError behavior
        # for real errors is untouched.
        if 300 <= exc.code < 400:
            return exc
        raise


class HttpFetchTool:
    """Streaming, size-capped, SSRF-checked fetch.

    Redirects are followed MANUALLY, one hop at a time, re-running ``is_safe_url`` on
    every hop — an automatic redirect handler would bypass the guard on hops 2..n and
    is the standard SSRF escape. The body is consumed in ``CHUNK_SIZE`` chunks and
    ``ledger.check_stream_bytes(total)`` is called before appending each chunk, so the
    cap trips mid-download.
    """

    def __init__(
        self,
        *,
        ledger: BudgetLedger,
        max_redirects: int = 3,
        timeout_s: float = 30.0,
        max_artifact_bytes: int = 25_000_000,
        opener: Callable[..., HttpResponseProtocol] | None = None,
        resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        """Construct the fetch tool.

        Args:
            ledger: Budget ledger; every fetch is reserved and settled through it.
            max_redirects: Maximum number of redirect hops to follow.
            timeout_s: Per-request socket timeout.
            max_artifact_bytes: Hard cap on a single artifact's size.
            opener: Injected ``(url, timeout_s=...) -> HttpResponseProtocol`` opener,
                for tests. Defaults to a real, non-auto-redirecting urllib opener.
            resolver: Injected hostname resolver passed through to ``is_safe_url``.
        """
        self._ledger = ledger
        self._max_redirects = max_redirects
        self._timeout_s = timeout_s
        self._max_artifact_bytes = max_artifact_bytes
        self._opener = opener if opener is not None else _default_opener
        self._resolver = resolver

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]:
        """Fetch ``url``, following redirects manually with a fresh SSRF check per hop.

        Args:
            url: The requested URL (attacker-controlled; presumed hostile).

        Returns:
            Tuple of (metadata, raw bytes).

        Raises:
            FetchError: On an unsafe URL, redirect scheme violation, too many
                redirects, an oversized response, or any transport error.
        """
        if not is_safe_url(url, resolver=self._resolver):
            raise FetchError(f"unsafe URL rejected: {url!r}")

        # The reservation wraps the ENTIRE redirect-following + body-read sequence, not
        # just the final body read. Opening it here (before the opener is ever called)
        # ensures opener exceptions, redirect-chain traffic, and the oversized
        # Content-Length rejection (which raises FetchError before any bytes are read)
        # all register against the ledger -- previously the reservation was opened only
        # inside the body-read helper, so all of those failure paths silently never
        # touched the ledger at all.
        with self._ledger.fetch_call(estimated_bytes=self._max_artifact_bytes) as reservation:
            # Finding 10: any failure exiting this block -- transport error, redirect
            # violation, too-many-redirects, or a failure inside ``_read_body`` -- must
            # leave ``reservation.observed_bytes`` set before the exception propagates,
            # or ``BudgetLedger.abandon`` has nothing to settle against and falls back
            # to the full worst-case ``reserved_bytes`` charge (safe, but pessimistic).
            # ``_read_body`` sets the precise transferred count itself; every failure
            # path here that never reaches ``_read_body`` transferred zero body bytes.
            try:
                current = url
                for _ in range(self._max_redirects + 1):
                    response = None
                    try:
                        response = self._opener(current, timeout_s=self._timeout_s)
                    except Exception as exc:  # noqa: BLE001 - normalize all transport errors
                        raise FetchError(f"fetch failed for {current!r}: {exc}") from exc

                    try:
                        status = getattr(response, "status", None)
                        if status in (301, 302, 303, 307, 308):
                            location = response.getheader("Location", "")
                            if not location:
                                raise FetchError(f"redirect from {current!r} had no Location header")
                            next_url = _resolve_redirect(current, location)
                            if not is_safe_url(next_url, resolver=self._resolver):
                                raise FetchError(f"redirect to unsafe URL rejected: {next_url!r}")
                            current = next_url
                            continue

                        return self._read_body(url, response, reservation)
                    finally:
                        if response is not None:
                            response.close()

                raise FetchError(f"too many redirects starting from {url!r}")
            except BaseException:
                if reservation.observed_bytes is None:
                    reservation.observed_bytes = 0
                raise

    def _read_body(
        self, requested_url: str, response: HttpResponseProtocol, reservation: Reservation
    ) -> tuple[FetchedArtifact, bytes]:
        """Stream and cap the response body, then finalize the artifact.

        Args:
            requested_url: The originally requested URL.
            response: The (already redirect-resolved) open response.
            reservation: The single ledger reservation opened in :meth:`fetch`, wrapping
                the whole redirect-following + read sequence; settled here with the
                actual byte count once the body is fully (and safely) read.

        Returns:
            Tuple of (metadata, raw bytes).
        """
        content_length = response.getheader("Content-Length", "")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self._max_artifact_bytes:
                # No body bytes have been read yet at this point (Finding 10): the
                # reservation must observe zero transferred bytes, not fall back to
                # the full worst-case charge.
                reservation.observed_bytes = 0
                raise FetchError(f"declared Content-Length {declared} exceeds cap {self._max_artifact_bytes}")

        final_url = getattr(response, "url", requested_url) or requested_url

        chunks: list[bytes] = []
        total = 0
        try:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > self._max_artifact_bytes:
                    raise FetchError(f"artifact exceeds max_artifact_bytes cap ({self._max_artifact_bytes})")
                self._ledger.check_stream_bytes(total)
                chunks.append(chunk)
        except BaseException:
            # Finding 10: whatever was actually pulled off the wire before the
            # failure (transport error, over-cap chunk, or a tripped stream-bytes
            # check) is the true egress -- settle the reservation against that
            # exact count rather than letting it fall back to the full worst-case
            # ``reserved_bytes`` charge.
            reservation.observed_bytes = total
            raise

        data = b"".join(chunks)
        self._ledger.settle_fetch(reservation, actual_bytes=len(data))

        artifact = FetchedArtifact(
            url=requested_url,
            final_url=final_url,
            sha256=hashlib.sha256(data).hexdigest(),
            content_type=sniff_content_type(data),
            n_bytes=len(data),
            fetched_at=datetime.now(UTC),
        )
        return artifact, data


def _resolve_redirect(current_url: str, location: str) -> str:
    """Resolve a possibly-relative ``Location`` header against the current URL.

    Args:
        current_url: The URL of the response that issued the redirect.
        location: The raw ``Location`` header value.

    Returns:
        The absolute redirect target URL.
    """
    return urljoin(current_url, location)


class MockFetchTool:
    """Canned url -> (bytes, content_type). Same interface; used by tests/TEST tier."""

    def __init__(self, responses: dict[str, tuple[bytes, str]]) -> None:
        """Construct the mock.

        Args:
            responses: Mapping from exact URL string to (bytes, content_type).
        """
        self._responses = responses

    def fetch(self, url: str) -> tuple[FetchedArtifact, bytes]:
        """Return the canned response for ``url``.

        Args:
            url: The requested URL.

        Returns:
            Tuple of (metadata, raw bytes).

        Raises:
            FetchError: If ``url`` has no canned response.
        """
        if url not in self._responses:
            raise FetchError(f"no mock response registered for {url!r}")
        data, content_type = self._responses[url]
        artifact = FetchedArtifact(
            url=url,
            final_url=url,
            sha256=hashlib.sha256(data).hexdigest(),
            content_type=content_type,
            n_bytes=len(data),
            fetched_at=datetime.now(UTC),
        )
        return artifact, data
