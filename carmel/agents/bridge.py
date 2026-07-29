"""The generic, framework-agnostic Carmel agent bridge.

This module is the single test seam for the whole agentic layer: every future Carmel
agent persona (Literature, Planning, Revision, Data, Reporting, ...) attaches to
:class:`CarmelAgent` by supplying its own ``system_prompt``, ``tools`` and
``output_schema`` — nothing else about this module changes per persona.

Nothing outside :mod:`carmel.agents.models` may know that pydantic-ai exists; this
module only ever talks to :class:`ModelProtocol`, a structural interface satisfied by
both the production pydantic-ai-backed model and the TEST-tier mock.

``CarmelAgent.run`` is the mandatory budget gate for model calls: it reserves budget
through :meth:`~carmel.agents.budget.BudgetLedger.model_call` BEFORE ever invoking the
model, and settles with the actual token/cost figures the model reports. A
:class:`~carmel.agents.budget.BudgetExceededError` is never swallowed here — it
propagates to the caller, which is responsible for turning it into a
:class:`~carmel.schemas.literature.StopReason`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from carmel.agents.budget import BudgetLedger, BudgetUsage
from carmel.logger import get_logger
from carmel.schemas.literature import StopReason

logger = get_logger("agents.bridge")

# Fallback reservation used ONLY when ``model`` does not provide its own
# ``estimate_worst_case_cost_usd`` (spar round 5, Finding 1). Every real Carmel-owned
# model (``MockModel``, ``PydanticAIModel`` -- see carmel.agents.models) implements
# that method, so in production this constant is dead code; it exists purely so an
# ad-hoc ``ModelProtocol``-conforming test double that predates this method (this
# module's own test seam explicitly allows a bare ``name`` + ``complete()`` stub) does
# not crash with an ``AttributeError`` before ever reaching the model it wants to
# exercise.
_LEGACY_DEFAULT_ESTIMATED_COST_USD = 0.05


@dataclass(frozen=True)
class AgentTool:
    """A single tool exposed to an agent's model.

    Attributes:
        name: Tool name, as presented to the model and recorded in provenance.
        description: Human/model-facing description of what the tool does.
        fn: Keyword-args-only callable; must return a JSON-serializable value.
    """

    name: str
    description: str
    fn: Callable[..., Any]


class ModelResponse(BaseModel):
    """A model's structured completion, already coerced to a plain dict.

    Attributes:
        output: The schema-validated (by the model layer's own machinery, if any)
            output payload; :class:`CarmelAgent` re-validates this against its own
            ``output_schema`` before returning it to the caller.
        input_tokens: Prompt tokens actually consumed.
        output_tokens: Completion tokens actually consumed.
        cost_usd: Actual dollar cost of this call.
        model_name: The concrete model identifier that produced this response.
        tool_calls: Names only (never arguments/results) of tools invoked, for
            provenance.
    """

    model_config = ConfigDict(extra="forbid")

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_name: str = ""
    tool_calls: list[str] = []


class ModelProtocol(Protocol):
    """Structural type for anything that can produce a structured completion.

    Production: :class:`carmel.agents.models.PydanticAIModel`. Tests and the TEST
    tier: :class:`carmel.agents.models.MockModel`. Nothing in this module cares which
    concrete class it holds.
    """

    name: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[AgentTool],
    ) -> ModelResponse:
        """Produce one structured completion."""
        ...


class AgentRunResult(BaseModel):
    """The typed result of one :meth:`CarmelAgent.run` call.

    Attributes:
        output: The validated output payload (already checked against the agent's
            ``output_schema``; callers that need the typed object should re-parse it
            with that schema, or :class:`CarmelAgent` callers can rely on this dict
            already having passed validation).
        usage: Budget usage snapshot at the time this call completed.
        stop_reason: Always :attr:`StopReason.SELF_TERMINATED` for a single successful
            call; multi-step loop orchestration (outside this module) may record other
            reasons when it wraps repeated calls.
        model_name: The concrete model identifier that produced this response.
        tool_calls: Names only of tools invoked during this call.
    """

    model_config = ConfigDict(extra="forbid")

    output: dict[str, Any]
    usage: BudgetUsage
    stop_reason: StopReason
    model_name: str
    tool_calls: list[str]


class AgentBridgeError(RuntimeError):
    """Raised for any bridge/model failure: output validation, an exhausted mock, a
    missing optional dependency, or a fail-closed model-construction violation.

    Defined here (rather than in :mod:`carmel.agents.models`) so that this module,
    which has no dependency on pydantic-ai at all, does not need to import from the
    module that lazily imports it. :mod:`carmel.agents.models` re-exports this same
    class, so callers may import it from either module interchangeably.
    """


class CarmelAgent:
    """A single Carmel-owned agent: one persona wrapped around one model call.

    Construction is pure dependency injection: ``model``, ``tools``, ``ledger`` and
    ``output_schema`` are all supplied by the caller. There is no global state, no
    environment reads, and no network access inside this class. A brand-new agent
    persona (Planning, Revision, Data, Reporting, ...) attaches by constructing a new
    ``CarmelAgent`` with its own ``name``, ``system_prompt``, ``tools`` and
    ``output_schema`` — nothing in this class changes.
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        model: ModelProtocol,
        tools: Sequence[AgentTool],
        ledger: BudgetLedger,
        output_schema: type[BaseModel],
    ) -> None:
        """Construct an agent from fully-injected dependencies.

        Args:
            name: Human-readable agent name, used only for logging.
            system_prompt: The persona's system prompt.
            model: The model implementation to call (mock or real).
            tools: Tools exposed to the model for this call.
            ledger: The budget ledger gating this agent's model calls.
            output_schema: Pydantic model the model's output must validate against.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = list(tools)
        self.ledger = ledger
        self.output_schema = output_schema

    def run(
        self,
        user_prompt: str,
        *,
        estimated_tokens: int = 8000,
        estimated_cost_usd: float | None = None,
    ) -> AgentRunResult:
        """Reserve budget, call the model, settle, and validate the output.

        Order is fixed and mandatory: the budget reservation happens BEFORE the model
        is ever called — the model must never be invoked without a prior reservation.
        A :class:`~carmel.agents.budget.BudgetExceededError` raised at reservation time
        propagates directly to the caller without calling the model at all.

        Args:
            user_prompt: The user/task prompt for this call.
            estimated_tokens: Worst-case token estimate used for the reservation.
            estimated_cost_usd: Worst-case dollar estimate used for the reservation.
                Defaults to ``None``, which asks ``model`` itself (via an
                ``estimate_worst_case_cost_usd(estimated_tokens)`` method, if it has
                one) for a reservation priced from that model's real per-1M-token
                rates -- rather than a flat constant that a real pro-model call could
                exceed (spar round 5, Finding 1). Pass an explicit value to override.

        Returns:
            The validated, typed run result.

        Raises:
            carmel.agents.budget.BudgetExceededError: If the ledger has no headroom,
                or if actual usage overage pushes a dimension past its ceiling at
                settlement time. Never swallowed.
            AgentBridgeError: If the model's output fails validation against
                ``output_schema``.
        """
        if estimated_cost_usd is None:
            estimate_fn = getattr(self.model, "estimate_worst_case_cost_usd", None)
            estimated_cost_usd = (
                estimate_fn(estimated_tokens) if estimate_fn is not None else _LEGACY_DEFAULT_ESTIMATED_COST_USD
            )

        with self.ledger.model_call(
            estimated_tokens=estimated_tokens, estimated_cost_usd=estimated_cost_usd
        ) as reservation:
            response = self.model.complete(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                output_schema=self.output_schema,
                tools=self.tools,
            )
            actual_tokens = response.input_tokens + response.output_tokens
            self.ledger.settle_model_call(reservation, actual_tokens=actual_tokens, actual_cost_usd=response.cost_usd)

        try:
            self.output_schema.model_validate(response.output)
        except ValidationError as exc:
            raise AgentBridgeError(
                f"agent {self.name!r} model output failed validation against "
                f"{self.output_schema.__name__}: {exc.errors()!r}"
            ) from exc

        return AgentRunResult(
            output=response.output,
            usage=self.ledger.usage(),
            stop_reason=StopReason.SELF_TERMINATED,
            model_name=response.model_name,
            tool_calls=list(response.tool_calls),
        )
