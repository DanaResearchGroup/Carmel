"""Concrete :class:`~carmel.agents.bridge.ModelProtocol` implementations.

This is the ONLY module in Carmel that may know pydantic-ai exists, and even here the
``import pydantic_ai`` is lazy (inside :meth:`PydanticAIModel.__init__` /
:meth:`PydanticAIModel.complete`) because pydantic-ai is an OPTIONAL dependency (see the
``agents`` extra in ``pyproject.toml``) and is not installed by default. Importing this
module must always succeed even when pydantic-ai is absent.

``build_model`` is the fail-closed factory: it never silently downgrades a "real" run to
a mock. A caller who wants MockModel must ask for it explicitly via
``AgentProvider.MOCK`` / ``ModelTier.TEST``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from carmel.agents.bridge import AgentBridgeError, AgentTool, ModelResponse
from carmel.config import AgentConfig, AgentProvider
from carmel.logger import get_logger

if TYPE_CHECKING:
    from carmel.agents.bridge import ModelProtocol

logger = get_logger("agents.models")

_INSTALL_HINT = "pip install 'carmel[agents]'"

__all__ = ["AgentBridgeError", "MockModel", "PydanticAIModel", "build_model", "compute_cost_usd"]

# Hand-maintained FALLBACK pricing table (USD per 1,000,000 tokens), used only when
# `genai_prices` cannot price the model --
# which is always true for moving aliases like `gemini-pro-latest` (genai_prices.calc_price
# raises LookupError for it). Values cross-checked against the genai_prices data snapshot
# as of this change.
_MODEL_PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    # gemini-pro-latest is a moving alias (not in genai_prices); currently aliases the
    # Gemini 3 Pro family, priced the same as gemini-3.1-pro-preview.
    "gemini-pro-latest": {"input": 2.00, "output": 12.00},
}

# Deliberately set well ABOVE every priced model's rate above. Used whenever `self.name`
# has no entry in the pricing table. An unpriced model must OVER-charge against the
# ledger, never escape it at cost 0.0 -- silently free real-provider calls would defeat
# the entire point of the budget ledger.
_FALLBACK_UNKNOWN_MODEL_RATE: dict[str, float] = {"input": 50.0, "output": 150.0}

# Conservative flat worst-case charge applied when a provider reports no usable token
# counts at all (e.g. `usage()` returned None, or both fields were absent) -- distinct
# from a genuine, believable report of 0 tokens. This is a documented judgment call, not
# a measurement: it assumes a call large enough that under-charging would be the worse
# failure mode.
_WORST_CASE_NO_USAGE_COST_USD = 5.0


def _read_usage_token_count(usage: Any, *, new_attr: str, old_attr: str) -> int | None:
    """Read a token count off a pydantic-ai usage object, old-name fallback included.

    Tries the current pydantic-ai (>=2.x) attribute name first (``input_tokens`` /
    ``output_tokens``), falling back to the deprecated pre-2.x attribute names
    (``request_tokens`` / ``response_tokens``) for older installs. Returns ``None`` if
    ``usage`` itself is ``None`` or neither attribute is present/non-None, so callers can
    distinguish "no usage object at all" from a genuine (possibly zero) token report.
    """
    if usage is None:
        return None
    value = getattr(usage, new_attr, None)
    if value is None:
        value = getattr(usage, old_attr, None)
    return value


def _table_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost from the hand-maintained pricing table (or its high fallback rate)."""
    rates = _MODEL_PRICING_USD_PER_1M_TOKENS.get(model_name)
    if rates is None:
        logger.warning(
            "no pricing table entry for model %r; charging fallback high rate "
            "($%.2f/$%.2f per 1M input/output tokens) so this call cannot escape the "
            "budget ledger at zero cost -- add %r to _MODEL_PRICING_USD_PER_1M_TOKENS "
            "with a verified rate to stop over-charging it",
            model_name,
            _FALLBACK_UNKNOWN_MODEL_RATE["input"],
            _FALLBACK_UNKNOWN_MODEL_RATE["output"],
            model_name,
        )
        rates = _FALLBACK_UNKNOWN_MODEL_RATE

    return (input_tokens / 1e6) * rates["input"] + (output_tokens / 1e6) * rates["output"]


def _genai_prices_cost_usd(usage: Any, model_name: str, provider_id: str) -> float | None:
    """Try to price a call via the ``genai_prices`` package (a pydantic-ai dependency).

    Returns ``None`` (never raises) if ``genai_prices`` is not installed, does not know
    ``model_name``/``provider_id``, or fails for any other reason -- callers must fall
    back to the hand-maintained table in that case. This is a best-effort *preferred*
    cost source, not a required one: it must never be allowed to turn a real call's cost
    into 0.0 or to blow up the request path.
    """
    try:
        import genai_prices  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        price = genai_prices.calc_price(usage, model_name, provider_id=provider_id)
    except Exception:
        logger.warning(
            "genai_prices could not price model %r (provider %r); falling back to the hand-maintained pricing table",
            model_name,
            provider_id,
            exc_info=True,
        )
        return None

    return float(price.total_price)


def compute_cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    *,
    tokens_available: bool,
    worst_case_cost_usd: float = _WORST_CASE_NO_USAGE_COST_USD,
    usage: Any = None,
    provider_id: str | None = None,
) -> float:
    """Pure(-ish) cost computation; does not require pydantic-ai to be installed.

    Args:
        model_name: The provider's model identifier (e.g. ``"gemini-2.5-flash"``).
        input_tokens: Reported input/request token count (ignored if
            ``tokens_available`` is False).
        output_tokens: Reported output/response token count (ignored if
            ``tokens_available`` is False).
        tokens_available: Whether the provider actually reported usable token counts.
            When False, a real 0-token report cannot be distinguished from "no usage
            object at all", so a conservative flat ``worst_case_cost_usd`` is charged
            instead of computing from (missing) tokens.
        worst_case_cost_usd: The flat charge applied when ``tokens_available`` is
            False.
        usage: The raw pydantic-ai usage object (e.g. ``RequestUsage``/``RunUsage``), if
            available. When provided together with ``provider_id``, real per-model
            pricing is attempted via ``genai_prices`` before falling back to the
            hand-maintained table. Purely optional -- omitting it (the default) always
            uses the hand-maintained table, exactly as before.
        provider_id: The provider name (e.g. ``"google"``) to pass to ``genai_prices``,
            required (together with ``usage``) to attempt real pricing.

    Returns:
        A cost in USD, guaranteed non-zero for any real (non-zero-token,
        tokens-available) call and for any no-usage call.
    """
    if not tokens_available:
        return worst_case_cost_usd

    if usage is not None and provider_id is not None:
        genai_cost = _genai_prices_cost_usd(usage, model_name, provider_id)
        if genai_cost is not None:
            return genai_cost

    return _table_cost_usd(model_name, input_tokens, output_tokens)


class MockModel:
    """Canned-response stand-in model; implements :class:`ModelProtocol`.

    Production-legitimate: it IS the TEST tier (precedent: the existing
    ``StubIntakeParser`` in ``carmel.services.intake``), not merely a test double. Pops
    one canned response dict per :meth:`complete` call, in order, and raises
    :class:`AgentBridgeError` once exhausted rather than looping or fabricating output.

    Records every prompt and tool-name list it was given so tests (and future callers)
    can assert on exactly what an agent asked the model — e.g. verifying a Verifier
    agent was never shown an author agent's raw URLs.
    """

    def __init__(self, responses: Sequence[dict[str, Any]] | None = None, name: str = "mock") -> None:
        """Initialize with a queue of canned response payloads.

        Args:
            responses: Output dicts to return, one per call, in order. May be empty.
            name: The model name reported in :class:`ModelResponse` and in ``repr``.
        """
        self.name = name
        self._responses: list[dict[str, Any]] = list(responses) if responses is not None else []
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[AgentTool],
    ) -> ModelResponse:
        """Pop and return the next canned response.

        Records the prompts and the names of the tools offered, for test assertions.

        Raises:
            AgentBridgeError: If the canned-response queue is exhausted.
        """
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "output_schema": output_schema,
                "tool_names": [t.name for t in tools],
            }
        )
        if not self._responses:
            raise AgentBridgeError(f"MockModel {self.name!r} exhausted: no canned responses remain")
        output = self._responses.pop(0)
        return ModelResponse(output=output, model_name=self.name)

    def __repr__(self) -> str:
        """Return a repr naming only the model, never any prompt/response content."""
        return f"MockModel(name={self.name!r})"


class PydanticAIModel:
    """Production model backed by pydantic-ai; implements :class:`ModelProtocol`.

    The API key is held only as a private attribute, is never included in ``repr``, and
    is never logged. ``pydantic_ai`` is imported lazily so this class can be constructed
    (and this module imported) even when the optional dependency is absent — the import
    failure surfaces only when a real completion is actually attempted.
    """

    def __init__(self, *, model_name: str, provider: AgentProvider, api_key: str) -> None:
        """Construct a pydantic-ai-backed model.

        Args:
            model_name: The provider's model identifier.
            provider: Which LLM provider this model calls.
            api_key: The secret API key value (never stored publicly, never logged).

        Raises:
            AgentBridgeError: If pydantic-ai is not installed.
        """
        self.name = model_name
        self._provider = provider
        self._api_key = api_key
        self._agent = self._build_agent()

    def _build_agent(self) -> Any:
        """Lazily import pydantic-ai and construct the underlying agent object.

        Raises:
            AgentBridgeError: If pydantic-ai is not importable.
        """
        try:
            import pydantic_ai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AgentBridgeError(
                "pydantic-ai not installed: the agentic layer's 'agents' extra is "
                f"required to use a non-mock provider. Install it with: {_INSTALL_HINT}"
            ) from exc
        return pydantic_ai

    def _infer_model(self, pydantic_ai: Any) -> Any:
        """Build a concrete pydantic-ai ``Model`` bound to ``self._api_key``.

        ``pydantic_ai.Agent("provider:model")`` (a bare string) resolves its provider via
        ``infer_provider``, which instantiates the provider class with *zero* arguments --
        meaning it reads the key from an ambient environment variable (e.g.
        ``GOOGLE_API_KEY``) rather than from the key Carmel's ``build_model`` factory
        already resolved via ``config.api_key_env``. That silently depends on an ambient
        env var Carmel never asked the caller to set, so instead we build the ``Model``
        object ourselves via ``infer_provider_class(...)(api_key=self._api_key)`` and hand
        the constructed object to ``Agent``, ensuring the key we were actually given is
        the one used -- regardless of what (if anything) happens to be in the ambient
        environment.
        """
        from pydantic_ai.providers import infer_provider_class  # type: ignore[import-not-found]

        provider_name = self._provider.value

        def _provider_factory(name: str) -> Any:
            # `infer_provider_class` is typed as returning the abstract `type[Provider]`,
            # whose __init__ takes no `api_key`; every concrete provider we support
            # (GoogleProvider, OpenAIProvider, ...) does accept it. Verified against a
            # real Gemini call, so this is a typing gap in the base class, not a bug.
            # Bound through an explicitly `Any`-typed name rather than a
            # `type: ignore[call-arg]`, because that ignore is only *needed* when
            # pydantic-ai is installed; in CI (which installs `.[dev]` without the
            # `agents` extra) the class is untyped, the error never fires, and the
            # ignore itself would fail the build as unused.
            provider_cls: Any = infer_provider_class(name)
            return provider_cls(api_key=self._api_key)

        return pydantic_ai.models.infer_model(f"{provider_name}:{self.name}", provider_factory=_provider_factory)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[AgentTool],
    ) -> ModelResponse:
        """Run a structured completion via pydantic-ai.

        Raises:
            AgentBridgeError: If pydantic-ai is not installed.
        """
        pydantic_ai = self._build_agent()
        model = self._infer_model(pydantic_ai)
        agent = pydantic_ai.Agent(
            model,
            output_type=output_schema,
            system_prompt=system_prompt,
        )
        for tool in tools:
            agent.tool_plain(tool.fn, name=tool.name, description=tool.description)
        result = agent.run_sync(user_prompt)
        # `usage` is a plain property on pydantic-ai >=2.x (accessing it yields a
        # `RunUsage`), but was a method on older releases. Support both: call it only
        # when it is actually callable, so neither shape raises.
        usage = result.usage
        if callable(usage):
            usage = usage()
        output = result.output
        output_dict = output.model_dump() if isinstance(output, BaseModel) else dict(output)
        # pydantic-ai >=2.x reports token counts as `input_tokens`/`output_tokens`;
        # `request_tokens`/`response_tokens` are deserialization-only aliases on those
        # same fields in 2.18 (not live post-construction attributes), kept here only as
        # a fallback for older pydantic-ai installs that used those names directly.
        # `getattr(usage, ..., None) or 0` alone cannot distinguish a genuine 0-token
        # report from the attribute/usage object being absent entirely -- both collapse
        # to 0, and the latter must NOT be billed as a free call.
        raw_input_tokens = _read_usage_token_count(usage, new_attr="input_tokens", old_attr="request_tokens")
        raw_output_tokens = _read_usage_token_count(usage, new_attr="output_tokens", old_attr="response_tokens")
        tokens_available = raw_input_tokens is not None or raw_output_tokens is not None
        input_tokens = raw_input_tokens or 0
        output_tokens = raw_output_tokens or 0
        cost_usd = compute_cost_usd(
            self.name,
            input_tokens,
            output_tokens,
            tokens_available=tokens_available,
            usage=usage,
            provider_id=self._provider.value,
        )
        return ModelResponse(
            output=output_dict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model_name=self.name,
            tool_calls=[],
        )

    def __repr__(self) -> str:
        """Return a repr showing only the model name and provider, never the API key."""
        return f"PydanticAIModel(model_name={self.name!r}, provider={self._provider!r})"


def build_model(config: AgentConfig) -> ModelProtocol:
    """Fail-closed factory for the model backing a Carmel agent.

    ``AgentProvider.MOCK`` always returns a :class:`MockModel`. Any other provider
    requires ``config.external_provider_consent`` to be ``True`` AND
    ``config.api_key_env`` to name an environment variable that is present and
    non-empty in ``os.environ``; only then is a :class:`PydanticAIModel` constructed.

    This function must NEVER silently fall back to a mock for a non-mock provider — a
    silent downgrade would make a "real" run quietly fake.

    Raises:
        AgentBridgeError: With a message naming the specific failure cause: missing
            consent, an unset env var, an empty env var, or the pydantic-ai dependency
            being absent.
    """
    if config.provider == AgentProvider.MOCK:
        return MockModel(name=config.resolved_model_name())

    if not config.external_provider_consent:
        raise AgentBridgeError(
            f"provider {config.provider!r} requires external_provider_consent=True; "
            "refusing to call a real model without explicit data-egress consent"
        )

    env_var = config.api_key_env
    if not env_var:
        raise AgentBridgeError(f"provider {config.provider!r} requires api_key_env to be set")

    if env_var not in os.environ:
        raise AgentBridgeError(f"environment variable {env_var!r} (api_key_env) is not set")

    api_key = os.environ[env_var]
    if not api_key:
        raise AgentBridgeError(f"environment variable {env_var!r} (api_key_env) is set but empty")

    return PydanticAIModel(model_name=config.resolved_model_name(), provider=config.provider, api_key=api_key)
