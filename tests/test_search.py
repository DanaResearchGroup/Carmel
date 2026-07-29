"""Tests for carmel.agents.tools.search."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from carmel.agents.budget import BudgetExceededError, BudgetLedger, session_budget
from carmel.agents.tools.academic import OpenAlexSearchTool
from carmel.agents.tools.search import CHUNK_SIZE, HttpSearchTool, MockSearchTool, SearchError, SearchResult
from carmel.config import AgentBudgetConfig


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    session_budget().reset()
    yield
    session_budget().reset()


def make_ledger(**limits: object) -> BudgetLedger:
    limits_obj = AgentBudgetConfig(**limits)  # type: ignore[arg-type]
    return BudgetLedger(limits_obj)


class FakeSearchResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.closed = False

    def read(self, amt: int = -1) -> bytes:
        if amt is None or amt < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
        else:
            chunk = self._payload[self._offset : self._offset + amt]
            self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class TestHttpSearchToolConfig:
    def test_raises_when_endpoint_empty(self) -> None:
        ledger = make_ledger()
        with pytest.raises(ValueError):
            HttpSearchTool(external_provider_consent=True, endpoint="", api_key="key", ledger=ledger)

    def test_raises_when_api_key_empty(self) -> None:
        ledger = make_ledger()
        with pytest.raises(ValueError):
            HttpSearchTool(
                external_provider_consent=True, endpoint="https://search.example/v1", api_key="", ledger=ledger
            )


class TestHttpSearchToolRequest:
    def test_api_key_sent_in_header_not_url(self) -> None:
        ledger = make_ledger()
        captured: dict[str, object] = {}

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            captured["url"] = url
            captured["headers"] = headers
            return FakeSearchResponse(json.dumps({"results": []}).encode())

        tool = HttpSearchTool(
            external_provider_consent=True,
            endpoint="https://search.example/v1",
            api_key="super-secret-key",
            ledger=ledger,
            opener=opener,
        )
        tool.search("acetone ignition delay")

        assert "super-secret-key" not in str(captured["url"])
        headers = captured["headers"]
        assert isinstance(headers, dict)
        assert any("super-secret-key" in v for v in headers.values())

    def test_query_is_url_encoded_in_query_string(self) -> None:
        ledger = make_ledger()
        captured: dict[str, object] = {}

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            captured["url"] = url
            return FakeSearchResponse(json.dumps([]).encode())

        tool = HttpSearchTool(
            external_provider_consent=True,
            endpoint="https://search.example/v1",
            api_key="k",
            ledger=ledger,
            opener=opener,
        )
        tool.search("a b c")

        assert "q=a" in str(captured["url"])

    def test_parses_top_level_list(self) -> None:
        ledger = make_ledger()
        payload = [{"title": "T1", "url": "http://x/1"}]

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(json.dumps(payload).encode())

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )
        results = tool.search("q")

        assert results == [SearchResult(title="T1", url="http://x/1")]

    @pytest.mark.parametrize("key", ["results", "data", "items", "web"])
    def test_parses_dict_with_known_list_keys(self, key: str) -> None:
        ledger = make_ledger()
        payload = {key: [{"title": "T", "url": "http://x/", "snippet": "s", "source": "src"}]}

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(json.dumps(payload).encode())

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )
        results = tool.search("q")

        assert results == [SearchResult(title="T", url="http://x/", snippet="s", source="src")]

    def test_unexpected_json_shape_returns_empty_list_not_keyerror(self) -> None:
        ledger = make_ledger()

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(json.dumps({"totally": "unexpected"}).encode())

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )

        assert tool.search("q") == []

    def test_malformed_json_returns_empty_list(self) -> None:
        ledger = make_ledger()

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(b"not json{{{")

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )

        assert tool.search("q") == []

    def test_entries_missing_required_fields_are_skipped(self) -> None:
        ledger = make_ledger()
        payload = {"results": [{"title": "no url"}, {"url": "http://x/"}, {"title": "ok", "url": "http://y/"}]}

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(json.dumps(payload).encode())

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )
        results = tool.search("q")

        assert len(results) == 1
        assert results[0].title == "ok"

    def test_respects_limit(self) -> None:
        ledger = make_ledger()
        payload = [{"title": f"T{i}", "url": f"http://x/{i}"} for i in range(20)]

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(json.dumps(payload).encode())

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )
        results = tool.search("q", limit=3)

        assert len(results) == 3

    def test_ledger_settled_with_actual_bytes(self) -> None:
        ledger = make_ledger()
        payload = json.dumps([]).encode()

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> FakeSearchResponse:
            return FakeSearchResponse(payload)

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )
        tool.search("q")

        assert ledger.usage().fetch_bytes == len(payload)

    def test_ledger_keeps_attempt_charged_when_opener_raises(self) -> None:
        # Finding 10: a failed search still consumed a real outbound request, so the
        # attempt (``fetches``) is never refunded. No bytes were actually transferred
        # before the opener raised, so the byte reservation settles down to zero.
        ledger = make_ledger()

        def opener(url: str, *, headers: dict[str, str], timeout_s: float):
            raise RuntimeError("network down")

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )

        with pytest.raises(RuntimeError):
            tool.search("q")

        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 0

    def test_oversized_response_trips_mid_read_not_after_full_buffering(self) -> None:
        # Regression test for the defect where `raw = response.read()` buffered the
        # entire body before any size check ran. A hostile/oversized response must be
        # rejected as soon as the cap is exceeded -- after reading only as many chunks
        # as needed to cross it, not the full payload.
        ledger = make_ledger(max_artifact_bytes=1000, max_fetches=5)
        big_payload = b"x" * (3 * 65536)  # several multiples of search.CHUNK_SIZE

        class CountingResponse(FakeSearchResponse):
            def __init__(self, payload: bytes) -> None:
                super().__init__(payload)
                self.read_calls = 0

            def read(self, amt: int = -1) -> bytes:
                self.read_calls += 1
                return super().read(amt)

        response = CountingResponse(big_payload)

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> CountingResponse:
            return response

        tool = HttpSearchTool(
            external_provider_consent=True, endpoint="https://search.example", api_key="k", ledger=ledger, opener=opener
        )

        with pytest.raises(BudgetExceededError):
            tool.search("q")

        # Only the first chunk was read before the cap tripped -- proof the body was
        # not fully buffered first.
        assert response.read_calls == 1
        assert response.closed is True

        # Finding 10: the attempt is charged regardless of outcome, and the byte
        # ledger settles against the CHUNK_SIZE bytes actually read before the cap
        # tripped -- not refunded down to zero and not left at the worst-case
        # estimate.
        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == CHUNK_SIZE


class TestMockSearchTool:
    def test_returns_canned_results(self) -> None:
        results = [SearchResult(title="T", url="http://x/")]
        tool = MockSearchTool({"q": results})

        assert tool.search("q") == results

    def test_unknown_query_returns_empty(self) -> None:
        tool = MockSearchTool({})
        assert tool.search("nope") == []

    def test_respects_limit(self) -> None:
        results = [SearchResult(title=f"T{i}", url=f"http://x/{i}") for i in range(5)]
        tool = MockSearchTool({"q": results})

        assert len(tool.search("q", limit=2)) == 2


class TestSearchEgressRequiresConsent:
    """Search was the last ungated network path.

    ``external_provider_consent`` gated LLM calls and (later) artifact fetches, but not
    the OpenAlex/Crossref queries that OPEN every literature run -- so the documented
    "no network without explicit opt-in" property was false for the first thing a run
    does. These tests pin the gate at ``budgeted_get_json``, the single choke point every
    backend shares, so a new backend cannot reintroduce the hole by not asking.
    """

    def test_http_search_tool_refuses_without_consent(self) -> None:
        ledger = make_ledger()
        called: list[str] = []

        def _opener(url: str, **_: object) -> object:
            called.append(url)
            raise AssertionError("must not open a socket without consent")

        tool = HttpSearchTool(
            endpoint="https://search.example/v1",
            api_key="k",
            ledger=ledger,
            external_provider_consent=False,
            opener=_opener,
        )
        with pytest.raises(SearchError, match="external_provider_consent"):
            tool.search("ignition delay n-heptane")
        assert called == []

    def test_refusal_happens_before_any_budget_is_reserved(self) -> None:
        """A refused call must not consume budget: it never reached the network."""
        ledger = make_ledger()
        before = ledger.usage()
        tool = HttpSearchTool(
            endpoint="https://search.example/v1",
            api_key="k",
            ledger=ledger,
            external_provider_consent=False,
            opener=lambda *a, **k: None,
        )
        with pytest.raises(SearchError):
            tool.search("q")
        after = ledger.usage()
        # Compare only the fetch dimensions: BudgetUsage also carries elapsed wall-clock,
        # which advances regardless and would make this assertion vacuously false.
        assert (after.fetches, after.fetch_bytes) == (before.fetches, before.fetch_bytes)

    def test_keyless_backends_are_gated_by_the_same_choke_point(self) -> None:
        ledger = make_ledger()
        tool = OpenAlexSearchTool(
            ledger=ledger,
            external_provider_consent=False,
            opener=lambda *a, **k: None,
        )
        with pytest.raises(SearchError, match="external_provider_consent"):
            tool.search("ignition delay")
