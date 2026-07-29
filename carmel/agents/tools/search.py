"""Provider-agnostic JSON web search tool, budget-gated and fail-closed.

``HttpSearchTool`` never accepts an unconfigured endpoint/key silently — construction
raises immediately so a misconfigured campaign cannot make a silent no-op network
call. The API key always travels in a header, never in the URL query string, because a
URL is what ends up in access logs and browser history.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict

from carmel.agents.budget import BudgetLedger
from carmel.logger import get_logger

logger = get_logger("agents.tools.search")

# Worst-case byte estimate reserved against the ledger for a single search call; search
# responses are small JSON payloads, not artifact downloads.
_ESTIMATED_SEARCH_BYTES = 65536

# Bounded read size, matching carmel.agents.tools.fetch's chunked-read discipline: a
# whole-body ``.read()`` would buffer an arbitrarily large hostile response before any
# size check ever runs. Reading in chunks lets ``check_stream_bytes`` trip mid-download.
CHUNK_SIZE = 65536


class SearchError(RuntimeError):
    """Raised for a search backend transport failure (P1-12).

    Mirrors :class:`carmel.agents.tools.fetch.FetchError`'s role exactly: a single
    query's 503, DNS blip, or timeout describes THAT backend call, not the campaign,
    so it must surface as a typed, catchable outcome instead of a bare exception. A
    bare ``BaseException`` propagating out of ``HttpSearchTool.search`` used to
    escape every handler in ``_research_loop``/``run_literature_research`` (neither
    ``BudgetExceededError`` nor ``AgentBridgeError``) and crash the entire literature
    run, discarding every finding/artifact already paid for in earlier rounds instead
    of producing the documented PARTIAL report.
    """


class SearchResult(BaseModel):
    """A single search hit."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = ""
    source: str = ""

    doi: str | None = None
    """Normalized bare DOI (``10.xxxx/yyy``), when the backend supplied one."""
    pdf_url: str | None = None
    """A directly-fetchable full-text URL the backend ADVERTISES, if any.

    Advertised, not verified: a live probe of OpenAlex found that 5 of 11
    repository-hosted ``pdf_url`` values actually served an HTML landing page. Callers
    must confirm the bytes really are a PDF rather than trusting this field.
    """
    is_open_access: bool = False
    """The backend's OA claim. Also advisory: the same probe saw 48% of works flagged
    open-access while only 18% carried any fetchable full-text URL at all."""
    repository: str = ""
    """Display name of the host holding ``pdf_url`` (e.g. ``OSTI``, ``arXiv``)."""


class SearchToolProtocol(Protocol):
    """Structural type for anything that can run a web search."""

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]: ...


def default_opener(url: str, *, headers: dict[str, str], timeout_s: float) -> Any:
    """Perform a real HTTP GET, returning a file-like response object.

    Args:
        url: Fully-formed request URL (query string excludes the API key).
        headers: Request headers, including the API key header.
        timeout_s: Socket timeout in seconds.

    Returns:
        An open, readable HTTP response.
    """
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310


class HttpSearchTool:
    """Provider-agnostic JSON search API. Fails closed when unconfigured."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        ledger: BudgetLedger,
        external_provider_consent: bool,
        opener: Callable[..., Any] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Construct the search tool.

        Args:
            endpoint: Base search API URL. Must be non-empty.
            api_key: API key sent via header. Must be non-empty.
            ledger: Budget ledger; every search call is reserved and settled through it.
            external_provider_consent: Whether the operator has agreed to let Carmel
                talk to third-party hosts. Required, with no default: search egress was
                previously ungated entirely, and a default would let a caller re-acquire
                that hole by omission.
            opener: Injected ``(url, headers=..., timeout_s=...) -> response`` opener,
                for tests. Defaults to a real urllib GET.
            timeout_s: Per-request socket timeout.

        Raises:
            ValueError: If ``endpoint`` or ``api_key`` is empty — fail closed rather
                than silently no-op or leak an empty key.
        """
        if not endpoint:
            raise ValueError("HttpSearchTool requires a non-empty endpoint")
        if not api_key:
            raise ValueError("HttpSearchTool requires a non-empty api_key")

        self._endpoint = endpoint
        self._api_key = api_key
        self._ledger = ledger
        self._external_provider_consent = external_provider_consent
        self._opener = opener if opener is not None else default_opener
        self._timeout_s = timeout_s

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Run a search query.

        Args:
            query: Free-text search query.
            limit: Maximum number of results requested.

        Returns:
            Parsed search results. Returns an empty list (never raises) when the
            response has an unexpected JSON shape.

        Raises:
            SearchError: On a transport failure (P1-12) -- see ``budgeted_get_json``.
            BudgetExceededError: Propagated from the reservation or the mid-stream check.
        """
        query_string = urlencode({"q": query, "limit": limit})
        url = f"{self._endpoint}?{query_string}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        payload = budgeted_get_json(
            url,
            headers=headers,
            ledger=self._ledger,
            opener=self._opener,
            timeout_s=self._timeout_s,
            external_provider_consent=self._external_provider_consent,
        )
        if payload is None:
            return []
        return _parse_results(payload)[:limit]


def budgeted_get_json(
    url: str,
    *,
    headers: dict[str, str],
    ledger: BudgetLedger,
    opener: Callable[..., Any],
    timeout_s: float,
    external_provider_consent: bool,
) -> Any | None:
    """GET ``url`` under full budget discipline and parse it as JSON.

    This is the single implementation of the ledger/streaming contract shared by every
    search backend (generic ``HttpSearchTool`` and the keyless scholarly adapters in
    :mod:`carmel.agents.tools.academic`). It is deliberately ONE function: the
    reserve/settle and mid-stream size-check behaviour below is budget-enforcement
    code, and a second hand-rolled copy in another backend is exactly how one backend
    ends up quietly unmetered.

    Args:
        url: Fully-formed request URL (must never carry an API key in its query).
        headers: Request headers, including any API-key header.
        ledger: Budget ledger; the call is reserved and settled through it.
        opener: ``(url, headers=..., timeout_s=...) -> response`` opener.
        timeout_s: Per-request socket timeout.
        external_provider_consent: Operator opt-in to third-party network egress.

    Returns:
        The parsed JSON payload, or ``None`` when the body was not valid JSON.

    Raises:
        BudgetExceededError: Propagated from the reservation or the mid-stream check.
        SearchError: When ``external_provider_consent`` is False. Every search backend
            reaches the network through this one function, so gating here covers
            ``HttpSearchTool`` and both keyless scholarly tools at once. Without it the
            consent flag was enforced only for LLM calls and (since the fetch fix) for
            artifact fetches, leaving search egress ungated -- so "no network without
            explicit opt-in" was simply untrue for the OpenAlex/Crossref queries that
            begin every literature run.
        SearchError: On any transport failure opening the request (P1-12) --
            connection refused, DNS failure, timeout, or a non-2xx HTTP status the
            opener raises as an exception. Normalized the same way
            ``HttpFetchTool.fetch`` normalizes its own opener call, so callers get one
            typed, catchable outcome instead of whatever exception type the
            underlying transport happened to raise.
    """
    # Fail closed BEFORE reserving budget or opening a socket, mirroring
    # ``HttpFetchTool.fetch``. This is the single egress choke point for every search
    # backend, keyed or keyless.
    if not external_provider_consent:
        raise SearchError("external_provider_consent is False; refusing to run any search query")

    with ledger.fetch_call(estimated_bytes=_ESTIMATED_SEARCH_BYTES) as reservation:
        # Finding 10: a failed search attempt still consumed a real outbound request
        # and whatever bytes actually crossed the wire before it failed. ``total``
        # tracks the true transferred count; on any failure it is recorded on the
        # reservation BEFORE the exception propagates, so ``BudgetLedger.abandon``
        # settles against the real egress instead of falling back to the full
        # worst-case ``reserved_bytes`` charge.
        total = 0
        try:
            try:
                response = opener(url, headers=headers, timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001 - normalize all transport errors (P1-12)
                raise SearchError(f"search request failed for {url!r}: {exc}") from exc
            try:
                chunks: list[bytes] = []
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    ledger.check_stream_bytes(total)
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                response.close()
        except BaseException:
            reservation.observed_bytes = total
            raise
        ledger.settle_fetch(reservation, actual_bytes=len(raw))

    try:
        return json.loads(raw)
    except json.JSONDecodeError, TypeError, UnicodeDecodeError:
        logger.warning("search response was not valid JSON; returning empty results")
        return None


def _parse_results(payload: Any) -> list[SearchResult]:
    """Defensively parse a search API's JSON payload into ``SearchResult`` objects.

    Accepts a top-level list, or a dict with a ``results``/``data``/``items``/``web``
    list. Any other shape, or a malformed individual entry, is tolerated by returning
    an empty list (or skipping that entry) rather than raising ``KeyError``.

    Args:
        payload: Parsed JSON payload of unknown/untrusted shape.

    Returns:
        Parsed results; empty list if the shape is not recognized.
    """
    items: Any = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("results", "data", "items", "web"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                items = candidate
                break

    if not isinstance(items, list):
        return []

    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url") or item.get("link")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        snippet = item.get("snippet") or item.get("description") or ""
        source = item.get("source") or ""
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet if isinstance(snippet, str) else "",
                source=source if isinstance(source, str) else "",
            )
        )
    return results


class MockSearchTool:
    """Canned query -> results. Same interface; used by tests/TEST tier."""

    def __init__(self, responses: dict[str, list[SearchResult]]) -> None:
        """Construct the mock.

        Args:
            responses: Mapping from exact query string to a canned result list.
        """
        self._responses = responses

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Return the canned results for ``query``, truncated to ``limit``.

        Args:
            query: The requested query string.
            limit: Maximum number of results to return.

        Returns:
            Canned results, or an empty list if ``query`` has no registered response.
        """
        return self._responses.get(query, [])[:limit]
