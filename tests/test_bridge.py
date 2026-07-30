"""Tests for carmel.agents.bridge and carmel.agents.models.

The only mock points are the injected model/tools/ledger; no network in any test, and
none of these tests require pydantic-ai to be installed.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from carmel.agents.bridge import AgentRunResult, AgentTool, CarmelAgent, ModelResponse
from carmel.agents.budget import BudgetExceededError, BudgetLedger, session_budget
from carmel.agents.models import (
    AgentBridgeError,
    MockModel,
    PydanticAIModel,
    build_model,
    compute_cost_usd,
    estimate_worst_case_model_cost_usd,
)
from carmel.config import AgentBudgetConfig, AgentConfig, AgentProvider, ModelTier
from carmel.schemas.literature import StopReason


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    """Ensure the process-global SessionBudget never leaks state between tests."""
    session_budget().reset()
    yield
    session_budget().reset()


class _Output(BaseModel):
    """A minimal typed output schema used across these tests."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float


def _ledger(**limits: object) -> BudgetLedger:
    """Build a BudgetLedger with the given AgentBudgetConfig overrides."""
    return BudgetLedger(AgentBudgetConfig(**limits))  # type: ignore[arg-type]


def _tool(name: str = "search") -> AgentTool:
    """Build a trivial AgentTool for exposure tests."""
    return AgentTool(name=name, description="does a thing", fn=lambda **kw: {"ok": True, **kw})


class TestCarmelAgentRun:
    """Behavioral tests for CarmelAgent.run."""

    def test_returns_validated_typed_output(self) -> None:
        model = MockModel(responses=[{"answer": "42", "confidence": 0.9}])
        ledger = _ledger()
        agent = CarmelAgent(
            name="test-agent",
            system_prompt="You are a test agent.",
            model=model,
            tools=[],
            ledger=ledger,
            output_schema=_Output,
        )
        result = agent.run("what is the answer?")
        assert isinstance(result, AgentRunResult)
        assert result.output == {"answer": "42", "confidence": 0.9}
        assert result.stop_reason == StopReason.SELF_TERMINATED
        # Caller can reconstruct the typed object from the validated dict.
        typed = _Output.model_validate(result.output)
        assert typed.answer == "42"

    def test_model_never_called_when_no_headroom(self) -> None:
        model = MockModel(responses=[{"answer": "x", "confidence": 0.1}])
        ledger = _ledger(max_model_calls=1)
        # Exhaust the single allowed model call slot directly on the ledger.
        ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=0.01)

        agent = CarmelAgent(
            name="test-agent",
            system_prompt="sp",
            model=model,
            tools=[],
            ledger=ledger,
            output_schema=_Output,
        )
        with pytest.raises(BudgetExceededError):
            agent.run("prompt")
        # Budget was checked BEFORE the call: MockModel was never invoked.
        assert len(model.calls) == 0

    def test_reservation_refunded_when_model_raises(self) -> None:
        class RaisingModel:
            name = "raiser"

            def complete(self, **kwargs: Any) -> ModelResponse:
                raise RuntimeError("boom")

        ledger = _ledger(max_model_calls=1, max_tokens=100_000, max_cost_usd=10.0)
        agent = CarmelAgent(
            name="test-agent",
            system_prompt="sp",
            model=RaisingModel(),
            tools=[],
            ledger=ledger,
            output_schema=_Output,
        )
        with pytest.raises(RuntimeError, match="boom"):
            agent.run("prompt")

        usage = ledger.usage()
        assert usage.model_calls == 0
        assert usage.tokens == 0
        assert usage.cost_usd == 0.0

    def test_invalid_output_raises_agent_bridge_error_without_leaking_prompt(self) -> None:
        model = MockModel(responses=[{"answer": "missing confidence field"}])
        ledger = _ledger()
        secret_prompt = "SUPER_SECRET_PROMPT_TOKEN_123"
        agent = CarmelAgent(
            name="test-agent",
            system_prompt="sp",
            model=model,
            tools=[],
            ledger=ledger,
            output_schema=_Output,
        )
        with pytest.raises(AgentBridgeError) as excinfo:
            agent.run(secret_prompt)
        assert secret_prompt not in str(excinfo.value)

    def test_tool_is_exposed_to_model(self) -> None:
        model = MockModel(responses=[{"answer": "ok", "confidence": 1.0}])
        ledger = _ledger()
        tool = _tool("lookup_species")
        agent = CarmelAgent(
            name="test-agent",
            system_prompt="sp",
            model=model,
            tools=[tool],
            ledger=ledger,
            output_schema=_Output,
        )
        agent.run("prompt")
        assert model.calls[0]["tool_names"] == ["lookup_species"]


class TestCarmelAgentRunCostEstimation:
    """CarmelAgent.run's DEFAULT per-call reservation must come from the model's own
    real worst-case pricing, not a flat constant a real pro-model call could exceed
    (spar round 5, Finding 1). Reservations must round UP, never under-estimate.
    """

    class _ExpensiveFakeModel:
        """A ModelProtocol-conforming stub priced far above the old flat $0.05 default."""

        name = "expensive-fake"

        def estimate_worst_case_cost_usd(self, estimated_tokens: int) -> float:
            return 3.0

        def complete(self, **kwargs: Any) -> ModelResponse:
            return ModelResponse(output={"answer": "x", "confidence": 0.5}, model_name=self.name, cost_usd=3.0)

    class _NoEstimateModel:
        """A bare ModelProtocol stub predating estimate_worst_case_cost_usd."""

        name = "no-estimate"

        def complete(self, **kwargs: Any) -> ModelResponse:
            return ModelResponse(output={"answer": "x", "confidence": 0.1}, model_name=self.name)

    def test_default_reservation_uses_the_models_own_worst_case_estimate(self) -> None:
        # A cap comfortably above the old flat $0.05 default but below the model's real
        # $3.00 worst case: the old flat default would sail through this cap and let the
        # call proceed; the fix must reject it BEFORE ever calling the model.
        model = self._ExpensiveFakeModel()
        ledger = _ledger(max_cost_usd=1.0)
        agent = CarmelAgent(
            name="test-agent", system_prompt="sp", model=model, tools=[], ledger=ledger, output_schema=_Output
        )
        with pytest.raises(BudgetExceededError):
            agent.run("prompt")

    def test_explicit_override_still_wins_over_the_models_estimate(self) -> None:
        model = self._ExpensiveFakeModel()
        ledger = _ledger(max_cost_usd=10.0)
        agent = CarmelAgent(
            name="test-agent", system_prompt="sp", model=model, tools=[], ledger=ledger, output_schema=_Output
        )
        agent.run("prompt", estimated_cost_usd=0.01)
        assert ledger.usage().cost_usd == pytest.approx(3.0)  # settled to the model's reported actual cost

    def test_model_without_estimate_method_falls_back_to_the_legacy_flat_default(self) -> None:
        # A cap exactly at the legacy $0.05 default must still allow the call...
        model = self._NoEstimateModel()
        ledger_ok = _ledger(max_cost_usd=0.05)
        agent_ok = CarmelAgent(
            name="test-agent", system_prompt="sp", model=model, tools=[], ledger=ledger_ok, output_schema=_Output
        )
        result = agent_ok.run("prompt")
        assert result.output["answer"] == "x"

        # ...but a cap just under it must reject it, pinning the fallback at exactly 0.05.
        ledger_tight = _ledger(max_cost_usd=0.04)
        agent_tight = CarmelAgent(
            name="test-agent", system_prompt="sp", model=model, tools=[], ledger=ledger_tight, output_schema=_Output
        )
        with pytest.raises(BudgetExceededError):
            agent_tight.run("prompt")


class TestMockModel:
    """MockModel-specific behavior."""

    def test_estimate_worst_case_cost_usd_is_flat_and_deterministic(self) -> None:
        # MockModel never calls a real provider, so it must NOT reserve real-money-scale
        # amounts derived from the token estimate -- mock-backed tests must stay
        # deterministic regardless of `estimated_tokens` (spar round 5, Finding 1).
        model = MockModel()
        assert model.estimate_worst_case_cost_usd(8_000) == 0.05
        assert model.estimate_worst_case_cost_usd(10_000_000) == 0.05

    def test_raises_when_exhausted(self) -> None:
        model = MockModel(responses=[])
        with pytest.raises(AgentBridgeError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

    def test_pops_one_response_per_call(self) -> None:
        model = MockModel(responses=[{"answer": "a", "confidence": 0.1}, {"answer": "b", "confidence": 0.2}])
        r1 = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])
        r2 = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])
        assert r1.output["answer"] == "a"
        assert r2.output["answer"] == "b"

    def test_records_prompts(self) -> None:
        model = MockModel(responses=[{"answer": "a", "confidence": 0.1}])
        model.complete(system_prompt="system-x", user_prompt="user-y", output_schema=_Output, tools=[])
        assert model.calls[0]["system_prompt"] == "system-x"
        assert model.calls[0]["user_prompt"] == "user-y"


class TestBuildModel:
    """build_model fail-closed factory tests."""

    def test_mock_config_returns_mock_model(self) -> None:
        config = AgentConfig(tier=ModelTier.TEST, provider=AgentProvider.MOCK)
        model = build_model(config)
        assert isinstance(model, MockModel)

    def test_non_mock_without_consent_raises(self) -> None:
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="FAKE_GOOGLE_KEY",
            external_provider_consent=False,
        )
        with pytest.raises(AgentBridgeError, match="external_provider_consent"):
            build_model(config)

    def test_consent_but_unset_env_var_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        monkeypatch.delenv("FAKE_GOOGLE_KEY_UNSET", raising=False)
        # Point every on-disk fallback location somewhere empty, so resolution
        # genuinely finds nothing anywhere -- not just an unset env var.
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="FAKE_GOOGLE_KEY_UNSET",
            external_provider_consent=True,
        )
        with pytest.raises(AgentBridgeError, match="no API key found") as excinfo:
            build_model(config)
        # The error must name the search path, so an operator knows exactly where to
        # put the credentials file -- a "not found" error that doesn't say where it
        # looked is not actionable.
        assert "credentials.env" in str(excinfo.value)

    def test_consent_but_empty_env_var_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        monkeypatch.setenv("FAKE_GOOGLE_KEY_EMPTY", "")
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="FAKE_GOOGLE_KEY_EMPTY",
            external_provider_consent=True,
        )
        with pytest.raises(AgentBridgeError, match="no API key found"):
            build_model(config)

    def test_missing_pydantic_ai_dependency_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_GOOGLE_KEY_OK", "sk-not-a-real-key")
        monkeypatch.setitem(sys.modules, "pydantic_ai", None)
        config = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="FAKE_GOOGLE_KEY_OK",
            external_provider_consent=True,
        )
        with pytest.raises(AgentBridgeError, match="pydantic-ai not installed"):
            build_model(config)


class TestPydanticAIModelRepr:
    """repr() must never leak the API key."""

    def test_repr_does_not_contain_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakePydanticAI:
            class Agent:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

        monkeypatch.setitem(sys.modules, "pydantic_ai", _FakePydanticAI())
        model = PydanticAIModel(model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="sk-topsecret")
        rendered = repr(model)
        assert "sk-topsecret" not in rendered
        assert "gemini-2.5-flash" in rendered
        assert "google" in rendered.lower()


class TestComputeCostUsd:
    """Regression tests for the pure cost-computation helper (Defect 1).

    These call the pure helper directly with the pre-existing (pydantic-ai-agnostic)
    keyword arguments. `PydanticAIModel.complete()` end-to-end coverage (including the
    real 2.18 usage shape and the genai_prices integration) lives in
    TestPydanticAIModelCompleteUsage and TestComputeCostUsdGenaiPrices below.
    """

    def test_unpriced_model_charges_nonzero_fallback_rate(self) -> None:
        # An unpriced model name must never escape the ledger at cost 0.0 -- it must be
        # charged at the deliberately-high fallback rate instead.
        cost = compute_cost_usd(
            "some-brand-new-unpriced-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            tokens_available=True,
        )
        assert cost > 0.0
        # Must exceed what even the most expensive priced model in the table would
        # charge for the same token counts, confirming it used the high fallback rate.
        priced_model_cost = compute_cost_usd(
            "gemini-3.1-pro-preview", input_tokens=1_000_000, output_tokens=1_000_000, tokens_available=True
        )
        assert cost > priced_model_cost

    def test_no_usable_tokens_charges_nonzero_worst_case(self) -> None:
        # When the provider reports no usable token counts at all (tokens_available is
        # False), the caller-supplied worst-case cost must be charged -- never 0.0, and
        # regardless of the (irrelevant/ignored) token counts passed in.
        cost = compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=0,
            output_tokens=0,
            tokens_available=False,
            worst_case_cost_usd=7.5,
        )
        assert cost == 7.5

    def test_priced_model_computes_from_pricing_table(self) -> None:
        cost = compute_cost_usd(
            "gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000, tokens_available=True
        )
        assert cost == pytest.approx(0.30 + 2.50)


class TestEstimateWorstCaseModelCostUsd:
    """Pre-call worst-case RESERVATION helper (spar round 5, Finding 1): must always be
    >= the actual cost `compute_cost_usd` would report for any real split of the same
    total token count, since the real split is unknown until the call returns.
    """

    def test_charges_all_estimated_tokens_at_the_output_rate(self) -> None:
        # gemini-2.5-flash: input $0.30/1M, output $2.50/1M -- the estimate must use the
        # more expensive output rate for every one of the estimated tokens.
        cost = estimate_worst_case_model_cost_usd("gemini-2.5-flash", 8_000)
        assert cost == pytest.approx(8_000 / 1_000_000 * 2.50)

    def test_exceeds_any_real_split_of_the_same_total_tokens(self) -> None:
        # However the real call actually splits its 8000 tokens between input and
        # output, the pre-call worst-case reservation must be >= the resulting real cost.
        estimate = estimate_worst_case_model_cost_usd("gemini-3.1-pro-preview", 8_000)
        for input_tokens in (0, 2_000, 4_000, 6_000, 8_000):
            output_tokens = 8_000 - input_tokens
            real_cost = compute_cost_usd(
                "gemini-3.1-pro-preview",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens_available=True,
            )
            assert estimate >= real_cost

    def test_unpriced_model_uses_the_same_fail_closed_fallback_ladder(self) -> None:
        # An unrecognized model must reserve the deliberately-absurd unknown-model rate
        # -- never silently reserve $0.0 or a too-small amount.
        cost = estimate_worst_case_model_cost_usd("some-brand-new-unpriced-model", 8_000)
        priced_model_cost = estimate_worst_case_model_cost_usd("gemini-3.1-pro-preview", 8_000)
        assert cost > priced_model_cost


class TestComputeCostUsdGenaiPrices:
    """genai_prices must be preferred over the hand table when usage/provider_id are
    supplied and it can price the model, but must never be allowed to blow up the
    request path -- any failure (ImportError, LookupError, or anything else) falls
    back to the hand-maintained table.
    """

    def test_genai_prices_preferred_when_it_can_price_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import carmel.agents.models as models_module

        class _FakePrice:
            total_price = 0.0042

        def _fake_calc_price(usage: Any, model_ref: str, *, provider_id: str | None = None) -> Any:
            assert model_ref == "gemini-2.5-flash"
            assert provider_id == "google"
            return _FakePrice()

        class _FakeGenaiPrices:
            calc_price = staticmethod(_fake_calc_price)

        monkeypatch.setitem(sys.modules, "genai_prices", _FakeGenaiPrices())

        cost = models_module.compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=1_000,
            output_tokens=500,
            tokens_available=True,
            usage=object(),
            provider_id="google",
        )
        # Must be the genai_prices-reported price, NOT the hand-table-derived value
        # (0.30 + 1.25 == 1.55 per 1e6 -> a very different number from 0.0042).
        assert cost == pytest.approx(0.0042)

    def test_genai_prices_failure_falls_back_to_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import carmel.agents.models as models_module

        def _raising_calc_price(usage: Any, model_ref: str, *, provider_id: str | None = None) -> Any:
            raise LookupError(f"no pricing snapshot for {model_ref!r}")

        class _FakeGenaiPrices:
            calc_price = staticmethod(_raising_calc_price)

        monkeypatch.setitem(sys.modules, "genai_prices", _FakeGenaiPrices())

        cost = models_module.compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            tokens_available=True,
            usage=object(),
            provider_id="google",
        )
        # Falls back to the hand-maintained table's known rate for this model.
        assert cost == pytest.approx(0.30 + 2.50)

    def test_genai_prices_not_installed_falls_back_to_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import carmel.agents.models as models_module

        monkeypatch.setitem(sys.modules, "genai_prices", None)

        cost = models_module.compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            tokens_available=True,
            usage=object(),
            provider_id="google",
        )
        assert cost == pytest.approx(0.30 + 2.50)

    def test_no_usage_or_provider_id_skips_genai_prices_entirely(self) -> None:
        # Omitting usage/provider_id (the pre-existing call shape) must still work and
        # must not attempt to consult genai_prices at all.
        cost = compute_cost_usd(
            "gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000, tokens_available=True
        )
        assert cost == pytest.approx(0.30 + 2.50)


class TestPydanticAIModelCompleteUsage:
    """Regression tests pinning the real installed pydantic-ai 2.18 usage shape.

    pydantic-ai 2.18 is installed in this environment. These tests exercise
    PydanticAIModel.complete() end-to-end against the REAL pydantic_ai package and
    REAL GoogleProvider/GoogleModel construction (no network access occurs: only
    ``pydantic_ai.Agent`` itself is faked, so ``run_sync`` never makes a real HTTP
    call). This pins the exact defect described in the task: in 2.18, a usage object
    exposes ``input_tokens``/``output_tokens`` live; ``request_tokens``/
    ``response_tokens`` are deserialization-only aliases and are never populated
    post-construction, so reading only the old names would silently make
    ``tokens_available`` always False and charge the flat worst-case every time.
    """

    @staticmethod
    def _install_fake_agent(monkeypatch: pytest.MonkeyPatch, usage_obj: Any) -> None:
        import pydantic_ai

        class _FakeResult:
            def usage(self) -> Any:
                return usage_obj

            output = _Output(answer="ok", confidence=1.0)

        class _FakeAgent:
            def __init__(self, model: Any, *, output_type: Any, system_prompt: str) -> None:
                pass

            def tool_plain(self, fn: Any, *, name: str, description: str) -> None:
                pass

            def run_sync(self, prompt: str) -> Any:
                return _FakeResult()

        monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent)

    def test_2_18_input_output_tokens_produce_token_derived_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")

        class _RealShapeUsage:
            input_tokens = 1_000
            output_tokens = 500
            # No request_tokens/response_tokens attributes at all -- exactly like the
            # real pydantic-ai 2.18 RequestUsage/RunUsage objects post-construction.

        self._install_fake_agent(monkeypatch, _RealShapeUsage())
        model = PydanticAIModel(
            model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="placeholder-not-a-real-key"
        )
        response = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert response.input_tokens == 1_000
        assert response.output_tokens == 500
        # Must NOT be the flat worst-case charge -- proves tokens_available was True.
        assert response.cost_usd != pytest.approx(5.0)
        assert response.cost_usd > 0.0

    def test_legacy_request_response_token_names_still_read_as_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")

        class _LegacyShapeUsage:
            request_tokens = 200
            response_tokens = 100
            # No input_tokens/output_tokens -- simulates a pre-2.x pydantic-ai install.

        self._install_fake_agent(monkeypatch, _LegacyShapeUsage())
        model = PydanticAIModel(
            model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="placeholder-not-a-real-key"
        )
        response = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert response.input_tokens == 200
        assert response.output_tokens == 100
        assert response.cost_usd != pytest.approx(5.0)
        assert response.cost_usd > 0.0

    def test_usage_with_no_token_attributes_charges_worst_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")

        class _EmptyUsage:
            pass  # neither new nor old attribute names present at all

        self._install_fake_agent(monkeypatch, _EmptyUsage())
        model = PydanticAIModel(
            model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="placeholder-not-a-real-key"
        )
        response = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.cost_usd == pytest.approx(5.0)


class TestModelLadderFallback:
    """A model that is UNAVAILABLE must fall through to the next model down; any other
    error must not.

    Both unavailability modes were observed live on 2026-07-28 against the real API:
    ``gemini-2.5-flash`` answers 404 ("no longer available to new users"), and
    ``gemini-3.1-pro-preview`` answered 503 ("high demand ... usually temporary") and
    then served normally ninety seconds later. Before this, either one stopped a
    literature run outright.
    """

    @staticmethod
    def _install_agent_failing_for(monkeypatch: pytest.MonkeyPatch, failures: dict[str, Exception]) -> list[str]:
        """Fake pydantic_ai.Agent so named models raise; records models actually called."""
        import pydantic_ai

        attempted: list[str] = []

        class _Usage:
            input_tokens = 10
            output_tokens = 5

        class _FakeResult:
            usage = _Usage()
            output = _Output(answer="ok", confidence=1.0)

        class _FakeAgent:
            def __init__(self, model: Any, *, output_type: Any, system_prompt: str) -> None:
                # `model` is a real pydantic-ai model object; its name identifies which
                # rung of the ladder we are on.
                self._model_name = getattr(model, "model_name", str(model))

            def tool_plain(self, fn: Any, *, name: str, description: str) -> None:
                pass

            def run_sync(self, prompt: str) -> Any:
                attempted.append(self._model_name)
                failure = failures.get(self._model_name)
                if failure is not None:
                    raise failure
                return _FakeResult()

        monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent)
        return attempted

    def test_503_on_the_preferred_model_falls_through_to_the_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = self._install_agent_failing_for(
            monkeypatch,
            {"gemini-3.5-flash": ModelHTTPError(503, "gemini-3.5-flash", "high demand")},
        )
        model = PydanticAIModel(
            model_name="gemini-3.5-flash",
            provider=AgentProvider.GOOGLE,
            api_key="placeholder-not-a-real-key",
            fallback_model_names=["gemini-3-flash-preview"],
        )

        response = model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert attempted == ["gemini-3.5-flash", "gemini-3-flash-preview"]
        # The cost must be attributed to the model that actually ran, not the one asked for.
        assert response.model_name == "gemini-3-flash-preview"
        # `self.name` must stay the constructor-given preferred model, not the fallback
        # rung that happened to succeed. A prior version mutated `self.name = candidate`
        # inside the fallback loop and never restored it, so a caller reading
        # `deps.model.name` (e.g. carmel/services/literature.py) after a successful
        # fallback would see the wrong, drifted value -- this asserts that regression
        # stays fixed.
        assert model.name == "gemini-3.5-flash"

    def test_404_retirement_falls_through_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = self._install_agent_failing_for(
            monkeypatch,
            {"gemini-2.5-flash": ModelHTTPError(404, "gemini-2.5-flash", "no longer available to new users")},
        )
        model = PydanticAIModel(
            model_name="gemini-2.5-flash",
            provider=AgentProvider.GOOGLE,
            api_key="placeholder-not-a-real-key",
            fallback_model_names=["gemini-3.6-flash"],
        )

        model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert attempted == ["gemini-2.5-flash", "gemini-3.6-flash"]

    def test_a_non_availability_error_is_never_retried_on_another_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        # 401 is an operator problem (bad key). Retrying it down the ladder would turn one
        # clear error into several confusing ones and burn a call per rung.
        attempted = self._install_agent_failing_for(
            monkeypatch,
            {"gemini-3.6-flash": ModelHTTPError(401, "gemini-3.6-flash", "invalid api key")},
        )
        model = PydanticAIModel(
            model_name="gemini-3.6-flash",
            provider=AgentProvider.GOOGLE,
            api_key="placeholder-not-a-real-key",
            fallback_model_names=["gemini-3.5-flash"],
        )

        with pytest.raises(ModelHTTPError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert attempted == ["gemini-3.6-flash"]

    def test_the_last_rung_raises_rather_than_silently_returning_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = self._install_agent_failing_for(
            monkeypatch,
            {
                "gemini-3.6-flash": ModelHTTPError(503, "gemini-3.6-flash", "busy"),
                "gemini-3.5-flash": ModelHTTPError(503, "gemini-3.5-flash", "busy"),
            },
        )
        model = PydanticAIModel(
            model_name="gemini-3.6-flash",
            provider=AgentProvider.GOOGLE,
            api_key="placeholder-not-a-real-key",
            fallback_model_names=["gemini-3.5-flash"],
        )

        with pytest.raises(ModelHTTPError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert attempted == ["gemini-3.6-flash", "gemini-3.5-flash"]

    def test_no_fallbacks_means_no_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import ModelHTTPError

        attempted = self._install_agent_failing_for(
            monkeypatch,
            {"gemini-3.6-flash": ModelHTTPError(503, "gemini-3.6-flash", "busy")},
        )
        model = PydanticAIModel(
            model_name="gemini-3.6-flash", provider=AgentProvider.GOOGLE, api_key="placeholder-not-a-real-key"
        )

        with pytest.raises(ModelHTTPError):
            model.complete(system_prompt="sp", user_prompt="up", output_schema=_Output, tools=[])

        assert attempted == ["gemini-3.6-flash"]


class TestPydanticAIModelEstimateWorstCaseCostUsd:
    """A real model's worst-case reservation must cover whichever ladder rung
    `complete()` might actually land on (spar round 5, Finding 1) -- `complete()` can
    fall through to a pricier fallback model on a 404/503, so pricing only the
    preferred model would under-reserve for that outcome.
    """

    def test_takes_the_max_over_the_full_fallback_ladder(self) -> None:
        # gemini-2.5-flash ($0.30/$2.50 per 1M) is far cheaper than its fallback
        # gemini-3.1-pro-preview ($4.00/$18.00 per 1M); the reservation must reflect
        # the pricier fallback, not just the cheap preferred model.
        model = PydanticAIModel(
            model_name="gemini-2.5-flash",
            provider=AgentProvider.GOOGLE,
            api_key="placeholder-not-a-real-key",
            fallback_model_names=["gemini-3.1-pro-preview"],
        )

        estimate = model.estimate_worst_case_cost_usd(8_000)

        preferred_only = estimate_worst_case_model_cost_usd("gemini-2.5-flash", 8_000)
        fallback_only = estimate_worst_case_model_cost_usd("gemini-3.1-pro-preview", 8_000)
        assert fallback_only > preferred_only
        assert estimate == pytest.approx(fallback_only)

    def test_no_fallbacks_prices_just_the_one_model(self) -> None:
        model = PydanticAIModel(
            model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="placeholder-not-a-real-key"
        )

        estimate = model.estimate_worst_case_cost_usd(8_000)

        assert estimate == pytest.approx(estimate_worst_case_model_cost_usd("gemini-2.5-flash", 8_000))


class TestFamilyFallbackPricing:
    """An unpriced model in a KNOWN family must be charged a plausible over-estimate,
    not the deliberately-absurd unknown-model rate.

    Family resolution means a tier can land on a model released after this code was
    written, so "unpriced" stops being an exotic case. Charging such a model $50/$150 per
    1M tokens would stop every realistic budget on the first call -- the fail-closed
    design turning into a denial of service against its own operator.
    """

    @staticmethod
    def _cost(model_name: str) -> float:
        return compute_cost_usd(model_name, input_tokens=1_000_000, output_tokens=1_000_000, tokens_available=True)

    def test_a_zero_from_genai_prices_is_rejected_not_trusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """genai_prices reads the usage OBJECT, not the counts we already extracted, so
        when its view disagrees with ours it prices zero tokens and returns 0.0.

        Trusting that would record a real, paid call as free and defeat the whole budget
        ledger. Observed for real: genai-prices 0.0.73 returns 0.0 for a pre-2.x-shaped
        usage object whose `request_tokens`/`response_tokens` Carmel reads correctly, and
        CI caught it while a local 0.0.72 install passed. Pinned here against a forced
        zero so the guard cannot regress on a library upgrade.
        """
        import carmel.agents.models as models

        monkeypatch.setattr(models, "_genai_prices_cost_usd", lambda *a, **kw: 0.0)

        cost = models.compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=200,
            output_tokens=100,
            tokens_available=True,
            usage=object(),
            provider_id="google",
        )

        # Falls through to the hand-maintained table rather than returning the zero.
        assert cost > 0.0
        assert cost == pytest.approx(200 / 1e6 * 0.30 + 100 / 1e6 * 2.50)

    def test_a_genuine_zero_token_call_still_costs_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard above must reject only a zero that came from genai_prices failing to
        see the tokens -- not make a real 0-token call cost something it did not.
        """
        import carmel.agents.models as models

        monkeypatch.setattr(models, "_genai_prices_cost_usd", lambda *a, **kw: 0.0)

        cost = models.compute_cost_usd(
            "gemini-2.5-flash",
            input_tokens=0,
            output_tokens=0,
            tokens_available=True,
            usage=object(),
            provider_id="google",
        )

        assert cost == 0.0

    def test_unknown_flash_model_uses_the_cheap_family_rate(self) -> None:
        cost = self._cost("gemini-9.9-flash")

        # Bounded well below the unknown-model rate, so a real budget still funds a run...
        assert cost < self._cost("some-brand-new-unpriced-model")
        # ...but above the most expensive flash model we actually know the price of, so
        # the estimate can never under-charge the ledger.
        assert cost > self._cost("gemini-3.5-flash")

    def test_unknown_pro_model_uses_the_expensive_family_rate(self) -> None:
        cost = self._cost("gemini-9.9-pro-preview")

        assert cost < self._cost("some-brand-new-unpriced-model")
        assert cost > self._cost("gemini-3.1-pro-preview")

    def test_pro_family_is_charged_more_than_flash_family(self) -> None:
        assert self._cost("gemini-9.9-pro-preview") > self._cost("gemini-9.9-flash")

    def test_a_genuinely_unrecognizable_name_still_gets_the_absurd_rate(self) -> None:
        # No family word anywhere: nothing is known about it, so over-charge hard.
        assert self._cost("llama-42-ultra") == pytest.approx(50.0 + 150.0)

    def test_exact_table_entries_still_win_over_family_fallback(self) -> None:
        # gemini-2.5-flash has a verified rate; it must not be re-priced at the family rate.
        assert self._cost("gemini-2.5-flash") == pytest.approx(0.30 + 2.50)

    def test_the_family_fallback_warning_is_emitted_once_per_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """A single literature run computes cost many times -- worst-case estimation across
        the whole fallback ladder before each call, then settlement after it. Warning every
        time printed the same lines five times over and buried the campaign ID the operator
        needed. The missing pricing entry is a property of the table, not of one call.
        """
        from carmel.agents.models import _warn_family_fallback_once

        _warn_family_fallback_once.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                for _ in range(5):
                    self._cost("gemini-9.9-flash")

            hits = [r for r in caplog.records if "no exact pricing entry" in r.getMessage()]
            assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
        finally:
            _warn_family_fallback_once.cache_clear()

    def test_pro_latest_alias_is_priced_like_the_family_it_aliases(self) -> None:
        assert self._cost("gemini-pro-latest") == pytest.approx(self._cost("gemini-3.1-pro-preview"))

    def test_flat_pro_rate_is_the_long_context_tier_not_the_cheaper_one(self) -> None:
        # Google prices pro models in two context tiers (2.00/12.00 below ~200k tokens,
        # 4.00/18.00 above). A flat table cannot express that, so it must round toward the
        # expensive tier: over-charging a short call is recoverable, but under-charging a
        # long one lets a run quietly exceed a budget the operator thought was binding.
        assert self._cost("gemini-3.1-pro-preview") == pytest.approx(4.00 + 18.00)


class TestPydanticAIModelApiKeyThreading:
    """The explicit api_key passed to PydanticAIModel must reach the provider directly
    -- never fall back to an ambient environment variable. This is the root-cause fix
    for the original bug in _infer_model/_build_agent: pydantic-ai's own
    ``infer_provider`` instantiates the provider class with zero arguments, which
    would silently read e.g. GOOGLE_API_KEY from the environment instead of using
    ``config.api_key_env``.
    """

    def test_infer_model_passes_explicit_api_key_to_provider_not_ambient_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("pydantic_ai")
        import pydantic_ai.providers as providers_module

        captured: dict[str, Any] = {}

        class _FakeProvider:
            def __init__(self, *, api_key: str | None = None) -> None:
                captured["api_key"] = api_key

        def _fake_infer_provider_class(name: str) -> Any:
            captured["provider_name"] = name
            return _FakeProvider

        monkeypatch.setattr(providers_module, "infer_provider_class", _fake_infer_provider_class)
        # Prove the explicit key wins even when an ambient env var of the same kind
        # exists and differs from it.
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-ambient-should-never-be-used")

        model = PydanticAIModel(
            model_name="gemini-2.5-flash", provider=AgentProvider.GOOGLE, api_key="sk-explicit-secret"
        )
        pydantic_ai_module = model._build_agent()
        model._infer_model(pydantic_ai_module, "gemini-2.5-flash")

        assert captured["provider_name"] == "google"
        assert captured["api_key"] == "sk-explicit-secret"
