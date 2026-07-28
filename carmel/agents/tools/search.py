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


class SearchResult(BaseModel):
    """A single search hit."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = ""
    source: str = ""


class SearchToolProtocol(Protocol):
    """Structural type for anything that can run a web search."""

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]: ...


def _default_opener(url: str, *, headers: dict[str, str], timeout_s: float) -> Any:
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
        opener: Callable[..., Any] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Construct the search tool.

        Args:
            endpoint: Base search API URL. Must be non-empty.
            api_key: API key sent via header. Must be non-empty.
            ledger: Budget ledger; every search call is reserved and settled through it.
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
        self._opener = opener if opener is not None else _default_opener
        self._timeout_s = timeout_s

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Run a search query.

        Args:
            query: Free-text search query.
            limit: Maximum number of results requested.

        Returns:
            Parsed search results. Returns an empty list (never raises) when the
            response has an unexpected JSON shape.
        """
        query_string = urlencode({"q": query, "limit": limit})
        url = f"{self._endpoint}?{query_string}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        with self._ledger.fetch_call(estimated_bytes=_ESTIMATED_SEARCH_BYTES) as reservation:
            # Finding 10: a failed search attempt still consumed a real outbound
            # request and whatever bytes actually crossed the wire before it failed.
            # ``total`` tracks the true transferred count; on any failure it is
            # recorded on the reservation BEFORE the exception propagates, so
            # ``BudgetLedger.abandon`` settles against the real egress instead of
            # falling back to the full worst-case ``reserved_bytes`` charge.
            total = 0
            try:
                response = self._opener(url, headers=headers, timeout_s=self._timeout_s)
                try:
                    chunks: list[bytes] = []
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        total += len(chunk)
                        self._ledger.check_stream_bytes(total)
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                finally:
                    response.close()
            except BaseException:
                reservation.observed_bytes = total
                raise
            self._ledger.settle_fetch(reservation, actual_bytes=len(raw))

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError, TypeError, UnicodeDecodeError:
            logger.warning("search response was not valid JSON; returning empty results")
            return []

        return _parse_results(payload)[:limit]


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
