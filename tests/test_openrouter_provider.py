"""Tests for OpenRouter as a first-class agent provider.

Everything here is offline: the catalogue HTTP call and pydantic-ai's ``Agent`` are
replaced by local doubles. The one exception is ``TestLiveSmoke``, which is skipped
unless ``CARMEL_LIVE_OPENROUTER=1`` and only ever calls a ``:free`` model.
"""

from __future__ import annotations

import json
import os
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from carmel.agents import model_catalog
from carmel.agents.model_catalog import (
    OpenRouterModelInfo,
    clear_catalogue_cache,
    fetch_openrouter_catalogue,
    free_openrouter_model_ids,
    parse_openrouter_catalogue,
    rank_free_nemotron_candidates,
    resolve_model_ladder,
)
from carmel.agents.models import (
    AgentBridgeError,
    FreeModelRequiredError,
    ModelRateLimitedError,
    PydanticAIModel,
    build_model,
    clear_dead_model_cache,
)
from carmel.config import (
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_APP_URL,
    AgentConfig,
    AgentProvider,
    ModelTier,
)

_KEY = "placeholder-not-a-real-key"


def _entry(model_id: str, *, prompt: str = "0", completion: str = "0", context: Any = 131_072) -> dict[str, Any]:
    """Build one catalogue item in the shape of ``GET /api/v1/models``."""
    return {"id": model_id, "context_length": context, "pricing": {"prompt": prompt, "completion": completion}}


#: A synthetic catalogue. The ids are invented on purpose: a test pinned to a real
#: free-model id would break the day OpenRouter retires it, while the RULE (both free
#: tests, largest context first) is what must hold.
CATALOGUE_PAYLOAD: dict[str, Any] = {
    "data": [
        _entry("nvidia/nemotron-big:free", context=1_048_576),
        _entry("nvidia/nemotron-small:free", context=131_072),
        _entry("nvidia/nemotron-big", prompt="0.0000002", completion="0.0000008", context=1_048_576),
        _entry("nvidia/nemotron-promo", context=2_000_000),
        _entry("vendor/other-model:free", context=4_000_000),
        _entry("vendor/paid-model", prompt="0.000003", completion="0.000015"),
    ]
}


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _install_catalogue(monkeypatch: pytest.MonkeyPatch, payload: Any = CATALOGUE_PAYLOAD) -> dict[str, Any]:
    """Fake the catalogue HTTP call; records how often, where, and with which headers it ran."""
    seen: dict[str, Any] = {"calls": 0, "headers": None, "url": None}
    body = json.dumps(payload).encode()

    def _fake_urlopen(request: Any, timeout: float = 0.0) -> Any:
        seen["calls"] += 1
        seen["headers"] = dict(request.headers)
        seen["url"] = request.full_url
        return _FakeResponse(body)

    monkeypatch.setattr("carmel.agents.model_catalog.urllib.request.urlopen", _fake_urlopen)
    return seen


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(request: Any, timeout: float = 0.0) -> Any:
        raise AssertionError("no network call may happen on this path")

    monkeypatch.setattr("carmel.agents.model_catalog.urllib.request.urlopen", _fail)


def _unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(request: Any, timeout: float = 0.0) -> Any:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("carmel.agents.model_catalog.urllib.request.urlopen", _boom)


@pytest.fixture(autouse=True)
def _clean_caches() -> Iterator[None]:
    clear_catalogue_cache()
    clear_dead_model_cache()
    yield
    clear_catalogue_cache()
    clear_dead_model_cache()


@pytest.fixture
def isolated_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every key search location at an empty tmp dir, then export a fake key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CARMEL_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", _KEY)
    return tmp_path


def _openrouter_config(**overrides: Any) -> AgentConfig:
    fields: dict[str, Any] = {
        "tier": ModelTier.DEV,
        "provider": AgentProvider.OPENROUTER,
        "api_key_env": "OPENROUTER_API_KEY",
        "external_provider_consent": True,
    }
    fields.update(overrides)
    return AgentConfig(**fields)


class _Output(BaseModel):
    answer: str


class TestParseOpenRouterCatalogue:
    def test_empty_data_parses_to_nothing(self) -> None:
        assert parse_openrouter_catalogue({"data": []}) == ()

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"data": "nope"}, {"models": []}])
    def test_malformed_payload_parses_to_nothing(self, payload: Any) -> None:
        assert parse_openrouter_catalogue(payload) == ()

    def test_standard_entry(self) -> None:
        (entry,) = parse_openrouter_catalogue({"data": [_entry("nvidia/nemotron-big:free", context=1_048_576)]})
        assert entry == OpenRouterModelInfo("nvidia/nemotron-big:free", 1_048_576, zero_priced=True)
        assert entry.is_free

    def test_free_needs_both_zero_price_and_free_suffix(self) -> None:
        free = free_openrouter_model_ids(parse_openrouter_catalogue(CATALOGUE_PAYLOAD))
        assert free == {"nvidia/nemotron-big:free", "nvidia/nemotron-small:free", "vendor/other-model:free"}
        # Zero-priced without the suffix (a promotion) is still paid.
        assert "nvidia/nemotron-promo" not in free

    def test_free_suffix_with_a_nonzero_price_is_not_free(self) -> None:
        (entry,) = parse_openrouter_catalogue({"data": [_entry("x/y:free", completion="0.000001")]})
        assert not entry.is_free

    @pytest.mark.parametrize(
        "pricing",
        [None, {}, {"prompt": "0"}, {"prompt": 0, "completion": 0}, {"prompt": "0.0", "completion": "0"}],
    )
    def test_missing_or_nonliteral_price_is_never_zero(self, pricing: Any) -> None:
        (entry,) = parse_openrouter_catalogue({"data": [{"id": "x/y:free", "pricing": pricing}]})
        assert not entry.zero_priced

    def test_malformed_items_are_skipped_not_guessed(self) -> None:
        payload = {"data": [None, {"id": ""}, {"id": 7}, {"pricing": {}}, _entry("ok/model:free")]}
        assert [e.model_id for e in parse_openrouter_catalogue(payload)] == ["ok/model:free"]

    @pytest.mark.parametrize("context", [None, "1000000", True, 1.5])
    def test_non_integer_context_is_unknown(self, context: Any) -> None:
        (entry,) = parse_openrouter_catalogue({"data": [_entry("x/y:free", context=context)]})
        assert entry.context_length is None


class TestRankFreeNemotron:
    def test_empty_catalogue(self) -> None:
        assert rank_free_nemotron_candidates(()) == []

    def test_prefers_the_million_token_free_variant(self) -> None:
        ladder = rank_free_nemotron_candidates(parse_openrouter_catalogue(CATALOGUE_PAYLOAD))
        # Only free Nemotrons, largest context first; the zero-priced promo id is excluded.
        assert ladder == ["nvidia/nemotron-big:free", "nvidia/nemotron-small:free"]

    def test_without_a_million_token_variant_picks_the_largest(self) -> None:
        payload = {"data": [_entry("nvidia/nemotron-a:free", context=32_768), _entry("nvidia/nemotron-b:free")]}
        assert rank_free_nemotron_candidates(parse_openrouter_catalogue(payload))[0] == "nvidia/nemotron-b:free"

    def test_unknown_context_sorts_last(self) -> None:
        payload = {"data": [_entry("nvidia/nemotron-a:free", context=None), _entry("nvidia/nemotron-b:free")]}
        assert rank_free_nemotron_candidates(parse_openrouter_catalogue(payload))[-1] == "nvidia/nemotron-a:free"


class TestFetchOpenRouterCatalogue:
    def test_reads_the_public_endpoint_without_sending_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _install_catalogue(monkeypatch)

        catalogue = fetch_openrouter_catalogue()

        assert seen["url"] == "https://openrouter.ai/api/v1/models"
        assert "authorization" not in {k.lower() for k in seen["headers"]}
        assert len(catalogue) == len(CATALOGUE_PAYLOAD["data"])

    def test_successful_read_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _install_catalogue(monkeypatch)
        fetch_openrouter_catalogue()
        fetch_openrouter_catalogue()
        assert seen["calls"] == 1

    def test_unreachable_catalogue_is_empty_and_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _unreachable(monkeypatch)
        assert fetch_openrouter_catalogue() == ()

        seen = _install_catalogue(monkeypatch)
        assert fetch_openrouter_catalogue()
        assert seen["calls"] == 1

    def test_garbage_body_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "carmel.agents.model_catalog.urllib.request.urlopen",
            lambda request, timeout=0.0: _FakeResponse(b"<html>502</html>"),
        )
        assert fetch_openrouter_catalogue() == ()


class TestResolveNemotronFamily:
    def test_resolves_to_the_free_nemotron_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_catalogue(monkeypatch)
        ladder = resolve_model_ladder("auto:nemotron-free", AgentProvider.OPENROUTER, _KEY)
        assert ladder == ["nvidia/nemotron-big:free", "nvidia/nemotron-small:free"]

    def test_empty_catalogue_refuses_rather_than_guessing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_catalogue(monkeypatch, {"data": []})
        with pytest.raises(ValueError, match="no free Nemotron"):
            resolve_model_ladder("auto:nemotron-free", AgentProvider.OPENROUTER, _KEY)

    def test_gemini_family_is_refused_on_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _forbid_network(monkeypatch)
        with pytest.raises(ValueError, match="served by 'google'"):
            resolve_model_ladder("auto:gemini-pro", AgentProvider.OPENROUTER, _KEY)

    def test_nemotron_family_is_refused_on_google(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _forbid_network(monkeypatch)
        with pytest.raises(ValueError, match="served by 'openrouter'"):
            resolve_model_ladder("auto:nemotron-free", AgentProvider.GOOGLE, _KEY)

    def test_concrete_id_resolves_to_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _forbid_network(monkeypatch)
        assert resolve_model_ladder("x/y:free", AgentProvider.OPENROUTER, _KEY) == ["x/y:free"]


class TestOpenRouterConfig:
    def test_provider_value(self) -> None:
        assert AgentProvider("openrouter") is AgentProvider.OPENROUTER

    def test_dev_default_is_the_free_nemotron_family(self) -> None:
        assert _openrouter_config().resolved_model_name() == "auto:nemotron-free"

    def test_other_providers_keep_their_defaults(self) -> None:
        config = AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.GOOGLE, api_key_env="GOOGLE_API_KEY")
        assert config.resolved_model_name() == "auto:gemini-flash"

    def test_explicit_model_name_wins(self) -> None:
        assert _openrouter_config(model_name="x/y:free").resolved_model_name() == "x/y:free"

    def test_prod_has_no_openrouter_default(self) -> None:
        # Falls through to the Gemini family, which resolution then refuses for OpenRouter.
        assert _openrouter_config(tier=ModelTier.PROD).resolved_model_name() == "auto:gemini-pro"

    def test_attribution_defaults_and_overrides(self) -> None:
        config = _openrouter_config()
        assert (config.openrouter_app_url, config.openrouter_app_title) == (
            DEFAULT_OPENROUTER_APP_URL,
            DEFAULT_OPENROUTER_APP_TITLE,
        )
        custom = _openrouter_config(openrouter_app_url="https://example.org", openrouter_app_title="Lab")
        assert (custom.openrouter_app_url, custom.openrouter_app_title) == ("https://example.org", "Lab")

    def test_requires_a_key_env(self) -> None:
        with pytest.raises(ValidationError, match="api_key_env"):
            _openrouter_config(api_key_env=None)


class TestBuildOpenRouterModel:
    def test_builds_from_config_with_free_ladder(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch)

        model = build_model(_openrouter_config())

        assert isinstance(model, PydanticAIModel)
        assert model.name == "nvidia/nemotron-big:free"
        assert model.estimate_worst_case_cost_usd(100_000) == 0.0
        assert _KEY not in repr(model)

    def test_provider_sends_the_attribution_headers(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        pydantic_ai = pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.openrouter import OpenRouterModel

        _install_catalogue(monkeypatch)
        model = build_model(_openrouter_config(openrouter_app_title="Carmel-test"))
        assert isinstance(model, PydanticAIModel)

        bound = model._infer_model(pydantic_ai, model.name)

        assert isinstance(bound, OpenRouterModel)
        assert str(bound.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
        headers = bound.client.default_headers
        assert headers["HTTP-Referer"] == DEFAULT_OPENROUTER_APP_URL
        assert headers["X-Title"] == "Carmel-test"

    def test_missing_key_is_a_typed_refusal(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
        _forbid_network(monkeypatch)

        with pytest.raises(AgentBridgeError, match="no API key found for provider 'openrouter'") as exc_info:
            build_model(_openrouter_config())

        assert _KEY not in str(exc_info.value)

    def test_consent_is_checked_first(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        _forbid_network(monkeypatch)
        with pytest.raises(AgentBridgeError, match="external_provider_consent"):
            build_model(_openrouter_config(external_provider_consent=False))

    def test_dev_non_free_model_is_refused_with_zero_network_calls(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        _forbid_network(monkeypatch)
        with pytest.raises(FreeModelRequiredError, match="':free'"):
            build_model(_openrouter_config(model_name="vendor/paid-model"))

    def test_dev_free_suffix_but_priced_in_catalogue_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        _install_catalogue(monkeypatch, {"data": [_entry("vendor/sneaky:free", prompt="0.000001")]})
        with pytest.raises(FreeModelRequiredError, match="could not be verified free"):
            build_model(_openrouter_config(model_name="vendor/sneaky:free"))

    def test_dev_free_suffix_absent_from_catalogue_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        _install_catalogue(monkeypatch)
        with pytest.raises(FreeModelRequiredError):
            build_model(_openrouter_config(model_name="vendor/unlisted:free"))

    def test_dev_unreachable_catalogue_refuses_a_named_free_model(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        _unreachable(monkeypatch)
        with pytest.raises(FreeModelRequiredError):
            build_model(_openrouter_config(model_name="nvidia/nemotron-big:free"))

    def test_dev_unreachable_catalogue_refuses_the_default_family(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        _unreachable(monkeypatch)
        with pytest.raises(AgentBridgeError, match="no free Nemotron"):
            build_model(_openrouter_config())

    def test_dev_verified_free_model_builds(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch)
        model = build_model(_openrouter_config(model_name="vendor/other-model:free"))
        assert isinstance(model, PydanticAIModel)
        assert model.name == "vendor/other-model:free"
        assert model.estimate_worst_case_cost_usd(1_000) == 0.0

    def test_prod_named_model_builds_without_the_catalogue(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        pytest.importorskip("pydantic_ai")
        _forbid_network(monkeypatch)
        model = build_model(_openrouter_config(tier=ModelTier.PROD, model_name="vendor/paid-model"))
        assert isinstance(model, PydanticAIModel)
        assert model.estimate_worst_case_cost_usd(1_000) > 0.0

    def test_prod_without_a_model_name_is_refused(self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path) -> None:
        _forbid_network(monkeypatch)
        with pytest.raises(AgentBridgeError, match="served by 'google'"):
            build_model(_openrouter_config(tier=ModelTier.PROD))


def _install_agent(monkeypatch: pytest.MonkeyPatch, failures: dict[str, Exception]) -> list[str]:
    """Fake pydantic_ai.Agent so named models raise; records the models actually called."""
    import pydantic_ai

    attempted: list[str] = []

    class _Usage:
        input_tokens = 50_000
        output_tokens = 20_000

    class _Result:
        usage = _Usage()
        output = _Output(answer="ok")

    class _FakeAgent:
        def __init__(self, model: Any, *, output_type: Any, system_prompt: str) -> None:
            self._model_name = model.model_name

        def tool_plain(self, fn: Any, *, name: str, description: str) -> None:
            pass

        def run_sync(self, prompt: str) -> Any:
            attempted.append(self._model_name)
            failure = failures.get(self._model_name)
            if failure is not None:
                raise failure
            return _Result()

    monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent)
    return attempted


def _complete(model: PydanticAIModel) -> Any:
    return model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])


class TestOpenRouterCalls:
    def test_free_model_costs_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _install_agent(monkeypatch, {})
        # "pro" in the id would match the family-rate fallback if the free path ever fell through.
        name = "nvidia/nemotron-pro:free"
        _install_catalogue(monkeypatch, {"data": [_entry(name)]})
        model = PydanticAIModel(model_name=name, provider=AgentProvider.OPENROUTER, api_key=_KEY)

        response = _complete(model)

        assert response.cost_usd == 0.0
        assert response.input_tokens == 50_000

    def test_unverified_model_is_priced_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _install_agent(monkeypatch, {})
        model = PydanticAIModel(model_name="vendor/paid-model", provider=AgentProvider.OPENROUTER, api_key=_KEY)
        assert _complete(model).cost_usd > 0.0

    def test_429_is_a_typed_retriable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = _install_agent(monkeypatch, {"a/one:free": ModelHTTPError(429, "a/one:free", "rate limited")})
        _install_catalogue(monkeypatch, {"data": [_entry("a/one:free"), _entry("a/two:free")]})
        model = PydanticAIModel(
            model_name="a/one:free",
            provider=AgentProvider.OPENROUTER,
            api_key=_KEY,
            fallback_model_names=["a/two:free"],
        )

        with pytest.raises(ModelRateLimitedError) as exc_info:
            _complete(model)

        assert exc_info.value.retriable is True
        assert exc_info.value.status_code == 429
        assert isinstance(exc_info.value, AgentBridgeError)
        assert isinstance(exc_info.value.__cause__, ModelHTTPError)
        assert _KEY not in str(exc_info.value)
        # The limit is per account, so the next rung is not tried.
        assert attempted == ["a/one:free"]

    def test_429_on_other_providers_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        _install_agent(monkeypatch, {"gemini-3.6-flash": ModelHTTPError(429, "gemini-3.6-flash", "quota")})
        model = PydanticAIModel(model_name="gemini-3.6-flash", provider=AgentProvider.GOOGLE, api_key=_KEY)
        with pytest.raises(ModelHTTPError):
            _complete(model)

    def test_404_reuses_the_dead_model_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = _install_agent(monkeypatch, {"a/gone:free": ModelHTTPError(404, "a/gone:free", "no endpoints")})
        _install_catalogue(monkeypatch, {"data": [_entry("a/gone:free"), _entry("a/live:free")]})

        def _model() -> PydanticAIModel:
            return PydanticAIModel(
                model_name="a/gone:free",
                provider=AgentProvider.OPENROUTER,
                api_key=_KEY,
                fallback_model_names=["a/live:free"],
            )

        assert _complete(_model()).model_name == "a/live:free"
        assert _complete(_model()).cost_usd == 0.0
        assert attempted == ["a/gone:free", "a/live:free", "a/live:free"]


@pytest.mark.skipif(os.environ.get("CARMEL_LIVE_OPENROUTER") != "1", reason="set CARMEL_LIVE_OPENROUTER=1")
class TestLiveSmoke:
    """One real call to the default free Nemotron. Never a paid model: DEV tier refuses those."""

    def test_default_free_nemotron_answers(self) -> None:
        class _Reply(BaseModel):
            text: str

        model = build_model(_openrouter_config())
        assert isinstance(model, PydanticAIModel)

        response = model.complete(
            system_prompt="Answer in one short sentence.",
            user_prompt="What gas is H2?",
            output_schema=_Reply,
            tools=[],
        )

        text = response.output["text"]
        print(f"live openrouter model={response.model_name} response_chars={len(text)} cost={response.cost_usd}")
        assert response.model_name.endswith(":free")
        assert response.cost_usd == 0.0
        assert text.strip()


class _FakeClock:
    """Stands in for ``model_catalog._clock`` so TTL expiry needs no sleeping."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr("carmel.agents.model_catalog._clock", fake)
    return fake


_FREE_THEN = {"data": [_entry("nvidia/n:free")]}
_PAID_NOW = {"data": [_entry("nvidia/n:free", prompt="0.0001", completion="0.0001")]}


class TestOpenRouterCatalogueExpiry:
    """A model that starts charging must stop counting as free once the cached read expires."""

    def test_read_is_reused_within_the_ttl(self, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
        seen = _install_catalogue(monkeypatch, _FREE_THEN)
        fetch_openrouter_catalogue()
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S - 1
        _install_catalogue(monkeypatch, _PAID_NOW)

        assert free_openrouter_model_ids(fetch_openrouter_catalogue()) == {"nvidia/n:free"}
        assert seen["calls"] == 1

    def test_price_change_is_seen_after_the_ttl(self, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
        _install_catalogue(monkeypatch, _FREE_THEN)
        assert free_openrouter_model_ids(fetch_openrouter_catalogue()) == {"nvidia/n:free"}

        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        seen = _install_catalogue(monkeypatch, _PAID_NOW)

        assert "nvidia/n:free" not in free_openrouter_model_ids(fetch_openrouter_catalogue())
        assert seen["calls"] == 1

    def test_failed_refetch_after_the_ttl_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        _install_catalogue(monkeypatch, _FREE_THEN)
        fetch_openrouter_catalogue()

        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _unreachable(monkeypatch)
        assert fetch_openrouter_catalogue() == ()

        # The expired read is gone for good, not merely skipped once.
        _unreachable(monkeypatch)
        assert fetch_openrouter_catalogue() == ()

    def test_dev_build_refuses_a_model_that_started_charging(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path, clock: _FakeClock
    ) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch, _FREE_THEN)
        build_model(_openrouter_config(model_name="nvidia/n:free"))

        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _install_catalogue(monkeypatch, _PAID_NOW)
        with pytest.raises(FreeModelRequiredError):
            build_model(_openrouter_config(model_name="nvidia/n:free"))


class TestFreeStatusIsNotCallerSupplied:
    """Only the catalogue can make a model free; direct construction cannot declare it."""

    def test_constructor_takes_no_free_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _forbid_network(monkeypatch)
        kwargs: dict[str, Any] = {"free_model_names": {"vendor/paid-model"}}
        with pytest.raises(TypeError):
            PydanticAIModel(model_name="vendor/paid-model", provider=AgentProvider.OPENROUTER, api_key=_KEY, **kwargs)

    def test_paid_model_is_reserved_at_a_nonzero_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _forbid_network(monkeypatch)
        model = PydanticAIModel(model_name="vendor/paid-model", provider=AgentProvider.OPENROUTER, api_key=_KEY)
        assert model.estimate_worst_case_cost_usd(100_000) > 0.0

    def test_free_suffix_the_catalogue_prices_is_not_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch, _PAID_NOW)
        model = PydanticAIModel(model_name="nvidia/n:free", provider=AgentProvider.OPENROUTER, api_key=_KEY)
        assert model.estimate_worst_case_cost_usd(100_000) > 0.0

    def test_catalogue_verified_model_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch)
        model = PydanticAIModel(model_name="nvidia/nemotron-big:free", provider=AgentProvider.OPENROUTER, api_key=_KEY)
        assert model.estimate_worst_case_cost_usd(100_000) == 0.0


class TestFinding1NemotronNamespace:
    """FINDING 1: rank_free_nemotron_candidates must require nvidia/ namespace."""

    def test_other_vendor_nemotron_clone_is_excluded(self) -> None:
        payload = {"data": [_entry("othervendor/nemotron-clone:free", context=1_048_576)]}
        ladder = rank_free_nemotron_candidates(parse_openrouter_catalogue(payload))
        assert ladder == [], f"expected no candidates, got {ladder}"

    def test_real_nvidia_nemotron_is_chosen(self) -> None:
        payload = {"data": [_entry("nvidia/nemotron-ultra:free", context=1_048_576)]}
        ladder = rank_free_nemotron_candidates(parse_openrouter_catalogue(payload))
        assert ladder == ["nvidia/nemotron-ultra:free"]


class TestFinding2EmptyCatalogueCaching:
    """FINDING 2: empty catalogue must be cached for the TTL."""

    def test_empty_catalogue_is_cached_for_ttl(self, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
        seen = _install_catalogue(monkeypatch, {"data": []})
        # First fetch
        catalogue1 = fetch_openrouter_catalogue()
        assert catalogue1 == ()
        assert seen["calls"] == 1

        # Advance clock but stay within TTL
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S - 1
        # Second fetch should use cache
        catalogue2 = fetch_openrouter_catalogue()
        assert catalogue2 == ()
        assert seen["calls"] == 1, "empty catalogue should be cached and reused within TTL"

    def test_empty_catalogue_refetch_after_ttl(self, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
        seen1 = _install_catalogue(monkeypatch, {"data": []})
        fetch_openrouter_catalogue()
        assert seen1["calls"] == 1

        # Advance past TTL
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        seen2 = _install_catalogue(monkeypatch, {"data": [_entry("nvidia/new:free")]})
        catalogue = fetch_openrouter_catalogue()
        assert len(catalogue) == 1
        assert seen2["calls"] == 1, "refetch should happen after TTL expiry"


class TestFinding3DynamicFreeStatus:
    """FINDING 3: free status must be re-derived at each cost/reservation decision."""

    def test_model_loses_free_status_after_ttl_expiry(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path, clock: _FakeClock
    ) -> None:
        pytest.importorskip("pydantic_ai")
        # Start with free model
        _install_catalogue(monkeypatch, _FREE_THEN)
        model = build_model(_openrouter_config(model_name="nvidia/n:free"))
        assert isinstance(model, PydanticAIModel)
        assert model.estimate_worst_case_cost_usd(100_000) == 0.0

        # Advance past TTL, catalogue now prices the model
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _install_catalogue(monkeypatch, _PAID_NOW)

        # Cost estimation must now refuse or return non-zero
        cost = model.estimate_worst_case_cost_usd(100_000)
        assert cost > 0.0, f"expected non-zero cost after model became paid, got {cost}"

    def test_model_refused_before_request_after_ttl_expiry(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path, clock: _FakeClock
    ) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch, _FREE_THEN)
        model = build_model(_openrouter_config(model_name="nvidia/n:free"))
        assert isinstance(model, PydanticAIModel)

        # Advance past TTL, catalogue now prices the model
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _install_catalogue(monkeypatch, _PAID_NOW)

        # Next call must be refused before any request (FreeModelRequiredError)
        from carmel.agents.models import FreeModelRequiredError

        with pytest.raises(FreeModelRequiredError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

    def test_catalogue_fetch_failure_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path, clock: _FakeClock
    ) -> None:
        pytest.importorskip("pydantic_ai")
        _install_catalogue(monkeypatch, _FREE_THEN)
        model = build_model(_openrouter_config(model_name="nvidia/n:free"))
        assert isinstance(model, PydanticAIModel)
        assert model.estimate_worst_case_cost_usd(100_000) == 0.0

        # Advance past TTL, then make catalogue unreachable
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _unreachable(monkeypatch)

        # Must fail closed: treat as not free
        cost = model.estimate_worst_case_cost_usd(100_000)
        assert cost > 0.0, f"expected non-zero cost when catalogue unreachable, got {cost}"


class TestFreeOnlyFlagRegression:
    """Regression tests for the _free_only flag (DEV-tier-only free enforcement)."""

    def test_prod_explicit_free_model_not_refused_when_catalogue_unreachable(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path
    ) -> None:
        """PROD tier with explicit :free model should not be refused when catalogue is down."""
        pytest.importorskip("pydantic_ai")
        _unreachable(monkeypatch)

        # PROD tier, explicit :free model id
        config = _openrouter_config(tier=ModelTier.PROD, model_name="vendor/model:free")
        model = build_model(config)
        assert isinstance(model, PydanticAIModel)

        # Call completes without FreeModelRequiredError
        _install_agent(monkeypatch, {})
        response = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert response.model_name == "vendor/model:free"
        # Cost should be > 0 because PROD tier does not enforce free-only
        assert response.cost_usd > 0.0
        # Worst-case estimate also > 0
        assert model.estimate_worst_case_cost_usd(100_000) > 0.0

    def test_dev_model_still_refused_before_request_when_no_longer_free(
        self, monkeypatch: pytest.MonkeyPatch, isolated_key: Path, clock: _FakeClock
    ) -> None:
        """DEV-tier model that was free becomes paid after TTL: next call refused pre-request."""
        pytest.importorskip("pydantic_ai")

        # Start with catalogue saying model is free
        _install_catalogue(monkeypatch, _FREE_THEN)
        model = build_model(_openrouter_config(model_name="nvidia/n:free"))
        assert isinstance(model, PydanticAIModel)
        assert model.estimate_worst_case_cost_usd(100_000) == 0.0

        # Advance past TTL, catalogue now prices the model
        clock.now += model_catalog._OPENROUTER_CATALOGUE_TTL_S
        _install_catalogue(monkeypatch, _PAID_NOW)

        # Next call must raise FreeModelRequiredError BEFORE any request layer is invoked
        # Verify by ensuring _install_agent is never reached (would fail if called)
        def _fail_agent(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("request layer must not be invoked")

        monkeypatch.setattr("pydantic_ai.Agent", _fail_agent)

        from carmel.agents.models import FreeModelRequiredError

        with pytest.raises(FreeModelRequiredError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])
