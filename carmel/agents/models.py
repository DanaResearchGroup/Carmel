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

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from carmel.agents.bridge import AgentBridgeError, AgentTool, ModelResponse
from carmel.agents.model_catalog import resolve_model_ladder
from carmel.config import AgentConfig, AgentProvider
from carmel.credentials import credential_search_path, resolve_api_key
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
#
# Verified-on date: 2026-07-28. This table is a hand-maintained snapshot, not a live
# lookup, so it WILL rot the same way the dated pins in `model_catalog.py` rot -- the
# difference is a stale price only over- or under-charges the budget ledger rather than
# failing a call outright, so there is no loud signal forcing a re-check. Treat any entry
# older than a few months as due for re-verification against `genai_prices` the next time
# this file is touched for an unrelated reason.
_MODEL_PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    # The Gemini pro rates below are the LONG-CONTEXT tier, deliberately.
    #
    # Google prices pro models in two context tiers -- 2.00/12.00 up to a 200k-token
    # request, 4.00/18.00 above it (verified against genai_prices on 2026-07-28 by
    # pricing the same model at 199k and 250k tokens). This flat table cannot express a
    # tier boundary, so it has to pick one, and the fail-closed direction is the higher:
    # over-charging a short call is a recoverable annoyance, while under-charging a long
    # one lets a run exceed a budget the operator thought was binding. `genai_prices`
    # applies the real tiered rate whenever it can price the model, so this approximation
    # only ever applies to the alias below and to models it does not know.
    "gemini-3.1-pro-preview": {"input": 4.00, "output": 18.00},
    # gemini-pro-latest is a moving alias (not in genai_prices); currently aliases the
    # Gemini 3 Pro family, priced the same as gemini-3.1-pro-preview. Being unpriceable
    # by genai_prices is exactly why `auto:` family resolution
    # (carmel.agents.model_catalog) is preferred over this alias: it lands on concrete
    # model ids that genai_prices DOES know, and so gets the tiered rate rather than this
    # deliberate over-estimate.
    "gemini-pro-latest": {"input": 4.00, "output": 18.00},
}

# Family-level fallbacks, applied when a model has no exact entry above. A brand-new
# `gemini-3.7-flash` is not "unknown" in any useful sense -- we know which family it is
# in, and therefore its order of magnitude -- so charging it the same as a genuinely
# unrecognizable name would make every budget in a flash-tier run meaningless.
#
# Each rate is set at roughly 2x the most expensive member of its family known today
# (flash: 1.50/9.00; pro: 4.00/18.00), so the estimate stays an OVER-charge even if the
# provider raises prices, while remaining close enough that a real budget still bounds a
# real run. Matched in order, so a name containing both words takes the cheaper reading
# only if "flash" appears -- deliberate: flash-class models are the ones that proliferate.
_FAMILY_FALLBACK_RATES: tuple[tuple[re.Pattern[str], dict[str, float]], ...] = (
    (re.compile(r"flash"), {"input": 3.00, "output": 18.00}),
    (re.compile(r"pro"), {"input": 8.00, "output": 36.00}),
)

# Deliberately set well ABOVE every priced model's rate above. Used whenever `self.name`
# matches neither the pricing table nor any known family. An unpriced model must
# OVER-charge against the ledger, never escape it at cost 0.0 -- silently free
# real-provider calls would defeat the entire point of the budget ledger.
_FALLBACK_UNKNOWN_MODEL_RATE: dict[str, float] = {"input": 50.0, "output": 150.0}

# Conservative flat worst-case charge applied when a provider reports no usable token
# counts at all (e.g. `usage()` returned None, or both fields were absent) -- distinct
# from a genuine, believable report of 0 tokens. This is a documented judgment call, not
# a measurement: it assumes a call large enough that under-charging would be the worse
# failure mode.
#
# This happens to equal `AgentBudgetConfig.max_cost_usd`'s default (also 5.0, in
# `carmel/config.py`) -- that match is COINCIDENTAL, not a derived relationship. This
# constant is a per-call worst-case estimate chosen independently of any particular
# budget; `max_cost_usd` is an operator-configurable ceiling for an entire run and can be
# set to any value. Do not read a change to one as implying the other should change too.
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


def _family_fallback_rates(model_name: str) -> dict[str, float] | None:
    """Return the family-level rate for ``model_name``, or None if no family matches.

    Keeps a newly-released member of a known family bounded by a plausible over-estimate
    rather than by the deliberately-absurd unknown-model rate, which would stop any
    realistic budget on the first call.
    """
    for pattern, rates in _FAMILY_FALLBACK_RATES:
        if pattern.search(model_name):
            logger.warning(
                "no exact pricing entry for model %r; charging the family fallback rate "
                "($%.2f/$%.2f per 1M input/output tokens), which deliberately over-estimates "
                "-- add a verified rate to _MODEL_PRICING_USD_PER_1M_TOKENS for exact costs",
                model_name,
                rates["input"],
                rates["output"],
            )
            return rates
    return None


def _table_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost from the hand-maintained pricing table (or its high fallback rate)."""
    rates = _MODEL_PRICING_USD_PER_1M_TOKENS.get(model_name)
    if rates is None:
        rates = _family_fallback_rates(model_name)
    if rates is None:
        logger.warning(
            "no pricing table entry and no known family for model %r; charging fallback "
            "high rate ($%.2f/$%.2f per 1M input/output tokens) so this call cannot "
            "escape the budget ledger at zero cost -- add %r to "
            "_MODEL_PRICING_USD_PER_1M_TOKENS with a verified rate to stop over-charging it",
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
        # `genai_prices` ships `py.typed`, so once the `agents` extra is installed
        # mypy resolves this directly -- no `type: ignore` needed (a stale one would
        # be an unused-ignore error under the agents-installed mypy lane).
        import genai_prices
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


def estimate_worst_case_model_cost_usd(model_name: str, estimated_tokens: int) -> float:
    """Compute a pre-call worst-case dollar RESERVATION for ``model_name``.

    Unlike :func:`compute_cost_usd` (which prices an *actual* input/output split
    reported after a call completes), this prices a call that has not happened yet:
    every one of ``estimated_tokens`` is charged at the model's OUTPUT rate -- the
    more expensive of the two -- because the real input/output split is unknown until
    the call returns and any split is cheaper than "all output". The result is
    therefore guaranteed to be >= the actual cost of a call that produces at most
    ``estimated_tokens`` total tokens, regardless of the real split.

    Reuses ``_table_cost_usd``'s fail-closed fallback ladder (exact table entry ->
    known-family rate -> deliberately absurd unknown-model rate), so a not-yet-priced
    or unrecognized model still reserves a conservative amount rather than silently
    reserving too little (spar round 5, Finding 1 -- reservations must never
    under-estimate a real call's cost).
    """
    return _table_cost_usd(model_name, input_tokens=0, output_tokens=estimated_tokens)


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

    def estimate_worst_case_cost_usd(self, estimated_tokens: int) -> float:
        """Return a deterministic worst-case reservation estimate for the mock model.

        Deliberately NOT derived from ``estimated_tokens`` or the real pricing table:
        MockModel never calls a real provider, so there is no real-money worst case to
        estimate, and mock-backed tests must stay deterministic regardless of the
        token estimate a caller passes (spar round 5, Finding 1 -- "do not change the
        mock/test model path in a way that makes tests reserve real money semantics").
        Matches :class:`~carmel.agents.bridge.CarmelAgent.run`'s pre-existing flat
        default, so mock-backed budget arithmetic in existing tests is unchanged.
        """
        return 0.05


#: HTTP statuses that mean "this model cannot serve you", as opposed to "your request
#: was wrong". 404 is a retirement (observed: ``gemini-2.5-flash`` -> *"no longer
#: available to new users"``), 503 is transient load (observed: ``gemini-3.5-flash`` ->
#: *"currently experiencing high demand"*, and ``gemini-3.1-pro-preview`` failing then
#: succeeding ninety seconds later). Both are answered by trying the next model down.
_UNAVAILABLE_STATUS_CODES = frozenset({404, 503})


def _is_model_unavailable(exc: BaseException) -> bool:
    """Return True if ``exc`` means the model itself is unavailable.

    Reads the structured ``status_code`` that pydantic-ai's ``ModelHTTPError`` carries
    rather than pattern-matching the message text: a substring check would be at the
    mercy of provider prose, and a false positive here would silently downgrade a run to
    a weaker model instead of surfacing a real error.
    """
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _UNAVAILABLE_STATUS_CODES


class PydanticAIModel:
    """Production model backed by pydantic-ai; implements :class:`ModelProtocol`.

    The API key is held only as a private attribute, is never included in ``repr``, and
    is never logged. ``pydantic_ai`` is imported lazily so this class can be constructed
    (and this module imported) even when the optional dependency is absent — the import
    failure surfaces only when a real completion is actually attempted.
    """

    def __init__(
        self,
        *,
        model_name: str,
        provider: AgentProvider,
        api_key: str,
        fallback_model_names: Sequence[str] = (),
    ) -> None:
        """Construct a pydantic-ai-backed model.

        Args:
            model_name: The provider's model identifier; the first choice.
            provider: Which LLM provider this model calls.
            api_key: The secret API key value (never stored publicly, never logged).
            fallback_model_names: Further models to try, in order, ONLY when a call fails
                because the preferred model is unavailable (see :func:`_is_model_unavailable`).
                Empty by default, so an explicitly-named model is never silently swapped.

        Raises:
            AgentBridgeError: If pydantic-ai is not installed.
        """
        self.name = model_name
        self._ladder: tuple[str, ...] = (model_name, *fallback_model_names)
        self._provider = provider
        self._api_key = api_key
        self._agent = self._build_agent()

    def _build_agent(self) -> Any:
        """Lazily import pydantic-ai and construct the underlying agent object.

        Raises:
            AgentBridgeError: If pydantic-ai is not importable.
        """
        try:
            # `pydantic-ai` ships `py.typed`, so once the `agents` extra is installed
            # mypy resolves this directly -- no `type: ignore` needed (a stale one
            # would be an unused-ignore error under the agents-installed mypy lane).
            import pydantic_ai
        except ImportError as exc:
            raise AgentBridgeError(
                "pydantic-ai not installed: the agentic layer's 'agents' extra is "
                f"required to use a non-mock provider. Install it with: {_INSTALL_HINT}"
            ) from exc
        return pydantic_ai

    def _infer_model(self, pydantic_ai: Any, model_name: str) -> Any:
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

        Args:
            pydantic_ai: The lazily-imported ``pydantic_ai`` module.
            model_name: The specific model id to bind (the ladder candidate currently
                being attempted), passed explicitly rather than read off ``self.name`` so
                a fallback attempt never has to mutate shared instance state to try a
                different model.
        """
        # `pydantic-ai` ships `py.typed`, so once the `agents` extra is installed mypy
        # resolves this directly -- no `type: ignore` needed (a stale one would be an
        # unused-ignore error under the agents-installed mypy lane).
        from pydantic_ai.providers import infer_provider_class

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

        return pydantic_ai.models.infer_model(f"{provider_name}:{model_name}", provider_factory=_provider_factory)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[AgentTool],
    ) -> ModelResponse:
        """Run a structured completion, walking down the model ladder if need be.

        Only an *availability* failure advances to the next model: a 404 (retired) or a
        503 (transient capacity). Every other error -- a bad request, an auth failure, a
        schema violation -- propagates immediately, because retrying it on a different
        model would convert one clear error into several confusing ones.

        A failed attempt generated no tokens, so it costs nothing and settles nothing;
        the caller's single budget reservation still covers the one call that succeeds.

        Raises:
            AgentBridgeError: If pydantic-ai is not installed.
        """
        last_index = len(self._ladder) - 1
        for index, candidate in enumerate(self._ladder):
            try:
                return self._complete_once(
                    model_name=candidate,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=output_schema,
                    tools=tools,
                )
            except Exception as exc:
                if index == last_index or not _is_model_unavailable(exc):
                    raise
                logger.warning(
                    "model %r is unavailable (%s); falling back to %r",
                    candidate,
                    type(exc).__name__,
                    self._ladder[index + 1],
                )
        # Unreachable: the final iteration either returns or re-raises.
        raise AgentBridgeError(f"no model in the ladder {self._ladder!r} could be called")

    def _complete_once(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[AgentTool],
    ) -> ModelResponse:
        """Run a single structured completion against ``model_name``.

        ``model_name`` is threaded through explicitly (rather than read off
        ``self.name``) so a fallback attempt in :meth:`complete` never has to mutate
        ``self.name`` -- shared instance state that outlives this single call and is
        read elsewhere (e.g. ``deps.model.name`` in ``carmel/services/literature.py``).
        A prior version set ``self.name = candidate`` before each attempt and left it
        mutated after a successful fallback, so a caller reading ``self.name`` after
        ``complete()`` returned would see whichever ladder rung last succeeded --
        surprising for something that looks like a fixed, constructor-provided name.

        Raises:
            AgentBridgeError: If pydantic-ai is not installed.
        """
        pydantic_ai = self._build_agent()
        model = self._infer_model(pydantic_ai, model_name)
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
            model_name,
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
            model_name=model_name,
            tool_calls=[],
        )

    def __repr__(self) -> str:
        """Return a repr showing only the model name and provider, never the API key."""
        return f"PydanticAIModel(model_name={self.name!r}, provider={self._provider!r})"

    def estimate_worst_case_cost_usd(self, estimated_tokens: int) -> float:
        """Return the worst-case dollar reservation for a not-yet-made call.

        ``complete()`` may land on ANY model in ``self._ladder`` -- the preferred
        model or any of its availability fallbacks (see :meth:`complete` and
        :func:`_is_model_unavailable`) -- so the reservation must cover whichever
        ladder member is priciest, not just ``self.name``; otherwise a fallback to a
        more expensive model could be under-reserved (spar round 5, Finding 1).
        """
        return max(estimate_worst_case_model_cost_usd(candidate, estimated_tokens) for candidate in self._ladder)


def build_model(config: AgentConfig) -> ModelProtocol:
    """Fail-closed factory for the model backing a Carmel agent.

    ``AgentProvider.MOCK`` always returns a :class:`MockModel`. Any other provider
    requires ``config.external_provider_consent`` to be ``True`` AND a resolvable API
    key: :func:`carmel.credentials.resolve_api_key` is consulted, which checks
    ``os.environ`` first (so an explicitly exported value always wins) and then falls
    back to a documented search path of on-disk credential files -- see that module for
    the full precedence order. Only once a key is found is a :class:`PydanticAIModel`
    constructed.

    This function must NEVER silently fall back to a mock for a non-mock provider — a
    silent downgrade would make a "real" run quietly fake.

    Raises:
        AgentBridgeError: With a message naming the specific failure cause: missing
            consent, no api_key_env configured, no key found anywhere in the search
            path (the message names every location searched), or the pydantic-ai
            dependency being absent.
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

    api_key = resolve_api_key(env_var, provider=config.provider.value)
    if not api_key:
        searched = credential_search_path(env_var, provider=config.provider.value)
        searched_str = ", ".join(str(path) for path in searched)
        raise AgentBridgeError(
            f"no API key found for provider {config.provider.value!r}: set ${env_var}, "
            f"or put {env_var}=... in one of: {searched_str}"
        )

    # Resolve `auto:<family>` to concrete provider model ids, newest first. A model named
    # explicitly resolves to itself alone, so this cannot swap out an operator's choice.
    try:
        ladder = resolve_model_ladder(config.resolved_model_name(), config.provider, api_key)
    except ValueError as exc:
        raise AgentBridgeError(str(exc)) from exc

    return PydanticAIModel(
        model_name=ladder[0],
        provider=config.provider,
        api_key=api_key,
        fallback_model_names=ladder[1:],
    )
