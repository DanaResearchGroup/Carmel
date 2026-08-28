"""The Extraction Agent persona: prompt, output schema, factory.

This is the persona :class:`~carmel.agents.bridge.CarmelAgent`'s own docstring
anticipates -- a Data-side reader that looks at ONE stored artifact and says what
experimental conditions it reports and exactly where each one lives. Everything else
in the agentic layer wrote its quotes and specifications by hand; nothing read a
stored document and proposed them. This persona closes that gap.

It attaches to the generic bridge exactly as the three literature personas do: its own
``system_prompt``, its own ``output_schema`` (:class:`ExtractionProposal`), and a
builder alongside :func:`~carmel.agents.literature_agent.build_corpus_agent`. Nothing
in the bridge changes.

THE AGENT PROPOSES; IT NEVER ASSERTS. Every member of :class:`ExtractionProposal` is
UNTRUSTED. It maps -- deliberately, field for field -- onto the specification types the
condition-set producer already takes (``ScalarConditionSpec``,
``CategoricalConditionSpec``, ``UnextractedConditionSpec``, ``DeviceClassSpec``,
``UnresolvedSubjectSpec``): see :func:`carmel.services.proposal_intake.condition_set_from_proposal`,
which is the ONLY sanctioned route from a proposal to a stored envelope. Inventing a
parallel vocabulary here would be the main way this persona goes wrong, so it does not:
the proposal is the producer's inputs, one layer removed and carrying its verbatim
quotes.

Every proposed value carries the verbatim quote it came from, because grounding is what
turns a proposal into a stored fact: the condition-set producer grounds each quote
against the stored document's own text and REFUSES any it cannot find, char-for-char.
A proposal whose quote is not in the document does not vanish -- it surfaces as a
producer refusal (:class:`~carmel.services.condition_set_producer.ConditionSetProducerError`
/ ``QuoteGroundingError``), which is the system working, not failing.

Like the corpus persona, and for the same reason (agents in this codebase are never
handed live, model-invoked tools -- the orchestration does every I/O deterministically
and feeds the result back in the prompt, so one LLM can never launder another's
fabrication or bill a provider out-of-band), the Extraction Agent takes NO tools. The
document text it reads is placed into its prompt by
:func:`carmel.services.proposal_intake.build_extraction_prompt`, which loads the SAME
current-extraction text the producer will later ground against.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from carmel.agents.bridge import AgentTool, CarmelAgent, ModelProtocol
from carmel.agents.budget import BudgetLedger
from carmel.schemas.datasets import (
    ConditionAttribution,
    SubjectRefusalReason,
    UnextractedReason,
)
from carmel.services.units import QuantityKind

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractionProposal",
    "ProposedCategoricalCondition",
    "ProposedDeviceClass",
    "ProposedObservable",
    "ProposedScalarCondition",
    "ProposedSubject",
    "ProposedUnextractedCondition",
    "ProposedUnresolvedSubject",
    "build_extraction_agent",
]


#: One 1-based occurrence selector, or ``None``. ``StrictInt`` rather than ``int`` is
#: load-bearing, not style: Pydantic's lax mode coerces a JSON ``true`` into the integer
#: ``1``, so a model that emits a boolean here would have it silently laundered into
#: "the first occurrence" -- as a genuine ``int``, which the downstream anti-bool guard
#: (``_require_int_occurrences``, which tests ``isinstance(value, bool)``) then cannot
#: see. The guard is real and correct; it is simply blind to a bool that stopped being
#: one at this boundary. Refusing it HERE, where the untrusted model's output first
#: becomes typed data, is the only place the distinction still exists.
Occurrence = Annotated[StrictInt, Field(ge=1)]


class ProposedScalarCondition(BaseModel):
    """A condition the agent claims resolves to ONE grounded number.

    Maps one-to-one onto
    :class:`carmel.services.condition_set_producer.ScalarConditionSpec`. Deliberately
    carries NO 40-character quote floor: a scalar value is a bare numeral (``"823"``)
    and its unit a token (``"K"``), both far shorter than a prose finding's quote --
    the length floor that guards ``ProposedFinding`` would reject every honest scalar
    here. The defence against a degenerate scalar is not length but the producer's
    span-stitching / uniqueness gate, which fires whether the quote is long or short.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    label_quote: str = Field(min_length=1)
    quantity_kind: QuantityKind
    value_quote: str = Field(min_length=1)
    unit_quote: str = Field(min_length=1)
    label_occurrence: Occurrence | None = None
    value_occurrence: Occurrence | None = None
    unit_occurrence: Occurrence | None = None
    """Which occurrence to ground when a quote repeats in the document, counted
    ``1``-based (``1`` = the first match), matching the agent prompt. The producer's
    grounder REFUSES a quote that occurs more than once and was not disambiguated, so
    an agent quoting a label as common as ``"temperature"`` must say which one it
    means; ``None`` asserts the quote is unique. ``ge=1`` forbids a ``0`` or negative:
    there is no "zeroth" occurrence in 1-based counting, and the floor is what makes the
    1-based -> 0-based conversion in
    :func:`carmel.services.proposal_intake.condition_set_from_proposal` total and safe
    (the converted grounder index can never fall below ``0``)."""


class ProposedCategoricalCondition(BaseModel):
    """A condition whose value is a token, not a number (fuel, diluent, material).

    Maps onto :class:`carmel.services.condition_set_producer.CategoricalConditionSpec`.
    No unit and no numeric normalisation: the token is grounded and recorded raw.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    label_quote: str = Field(min_length=1)
    token_quote: str = Field(min_length=1)
    label_occurrence: Occurrence | None = None
    token_occurrence: Occurrence | None = None


class ProposedUnextractedCondition(BaseModel):
    """A condition the agent REFUSES to reduce to a single value, span recorded.

    Maps onto :class:`carmel.services.condition_set_producer.UnextractedConditionSpec`.
    This is a proposal that is ITSELF a refusal -- a sweep, a range, a one-sided bound
    or a qualitative-only statement, located and quoted but explicitly not turned into
    a number. It is the honest home for everything the agent can see but cannot reduce,
    and it is grounded and STORED as a refusal rather than dropped.
    """

    model_config = ConfigDict(extra="forbid")

    statement_id: str = Field(min_length=1)
    label_quote: str = Field(min_length=1)
    statement_quote: str = Field(min_length=1)
    reason: UnextractedReason
    quantity_kind: QuantityKind | None = None
    """A refused temperature statement may still be known to be a temperature. ``None``
    when even the quantity is unclear; the producer records it as NOT_EXTRACTED_YET."""
    label_occurrence: Occurrence | None = None
    statement_occurrence: Occurrence | None = None


class ProposedObservable(BaseModel):
    """A STUB channel for a proposed series/observable, deliberately un-assembled.

    Series (observable) production is refused by this runtime unconditionally today
    (:func:`carmel.services.dataset_producer.produce_envelope_from_artifact` raises
    always -- a char span in running text cannot ground a series data point), and the
    assembly path is being restored in parallel work. This field exists so an
    observable has an OBVIOUS place in the proposal the day that path lands, and so the
    schema states the boundary rather than hiding it. Until then,
    :func:`carmel.services.proposal_intake.condition_set_from_proposal` REFUSES a
    proposal that carries any observable -- it does not silently drop them, because a
    silently dropped observation and a refused one are different facts.
    """

    model_config = ConfigDict(extra="forbid")

    observable_id: str = Field(min_length=1)
    label_quote: str = Field(min_length=1)
    note: str = Field(default="", max_length=500)
    """Free-text note for the human who will wire assembly, e.g. which figure or table
    the observable was read from. Never grounded, never stored -- advisory only."""


class ProposedDeviceClass(BaseModel):
    """The apparatus the paper names, proposed as the condition set's subject.

    Maps onto :class:`carmel.services.condition_set_producer.DeviceClassSpec`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["device_class"] = "device_class"
    label_quote: str = Field(min_length=1)
    label_occurrence: Occurrence | None = None


class ProposedUnresolvedSubject(BaseModel):
    """A REFUSAL to name the apparatus, with the span that motivates it.

    Maps onto :class:`carmel.services.condition_set_producer.UnresolvedSubjectSpec`.
    The refusal is still grounded: it points at the text that made the subject
    unresolvable, so a reader can check the refusal rather than take it on trust.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["unresolved_subject"] = "unresolved_subject"
    reason: SubjectRefusalReason
    reason_quote: str = Field(min_length=1)
    reason_occurrence: Occurrence | None = None


ProposedSubject = Annotated[
    ProposedDeviceClass | ProposedUnresolvedSubject,
    Field(discriminator="kind"),
]
"""The subject sum: exactly one of a named device class or an explicit refusal to name
one, tagged by ``kind`` so a proposal must commit to which it is."""


class ExtractionProposal(BaseModel):
    """The Extraction Agent's structured output for one stored artifact.

    Everything here is UNTRUSTED and maps onto the condition-set producer's inputs.
    ``artifact_sha256`` names WHICH held document the quotes came from -- constrained
    to a sha256 digest so a hallucinated handle fails validation immediately rather
    than being looked up. The whole object is carried to a stored envelope by
    :func:`carmel.services.proposal_intake.condition_set_from_proposal`.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attribution: ConditionAttribution
    """Whose conditions these are asserted to be (OWN / CITED / SIMULATION). The
    agent's assertion, recorded unverified; ``attribution_quote`` grounds only WHERE it
    was read, never that it is correct."""
    attribution_quote: str = Field(min_length=1)
    attribution_occurrence: Occurrence | None = None
    subject: ProposedSubject
    scalars: list[ProposedScalarCondition] = Field(default_factory=list)
    categoricals: list[ProposedCategoricalCondition] = Field(default_factory=list)
    unextracted: list[ProposedUnextractedCondition] = Field(default_factory=list)
    observables: list[ProposedObservable] = Field(default_factory=list)
    """Proposed series/observables. Has no assembly path today and is REFUSED by the
    intake carrier if non-empty -- see :class:`ProposedObservable`."""
    done: bool = True
    """The agent's self-stop signal. Accepted but not acted on: an extraction pass
    makes exactly one call per document and runs no rounds, so coverage is decided by
    what the store holds, not by the agent. Kept (defaulting True) rather than removed
    so a model that volunteers it -- the natural thing, given the sibling schemas --
    does not fail ``extra="forbid"`` and cost the whole document's extraction."""


EXTRACTION_SYSTEM_PROMPT = """\
You are Carmel's Extraction Agent: a combustion-experiment reader.

You are given the full text of ONE paper Carmel already holds. Your job is to say what
experimental CONDITIONS it reports -- the apparatus, the temperatures, pressures, fuel,
diluent, equivalence ratio and so on -- as structured proposals, each carrying the
verbatim quote it came from. You have NO web access and need none: the one document you
may use is in front of you. Never report a condition about any other paper.

Everything you emit is a PROPOSAL. A deterministic gate will ground every quote against
the actual stored bytes of this document and REFUSE any it cannot find, character for
character. A refused proposal is the system working, not a failure; a fabricated or
loosely-grounded one is the worst thing you can produce.

Non-negotiable rules:

1. VERBATIM QUOTES ONLY. Every `*_quote` must be copied EXACTLY, character for
   character, from the document text shown to you. Never paraphrase, never summarise,
   never "clean up" a sentence, and never repair what looks like an extraction error.
2. ONE NUMBER PER SCALAR. Report a condition as a `scalar` ONLY when the document
   states a single value with a unit for it (e.g. value_quote="823", unit_quote="K").
   Give its `quantity_kind` and quote the label, the value and the unit separately.
3. REFUSE, DO NOT SQUEEZE. A sweep ("varied from 0.6 to 1.4"), a range, a one-sided
   bound ("Re < 2000"), a qualitative-only statement ("atmospheric pressure"), or a
   composite value is NOT a scalar. Record it under `unextracted` with the matching
   `reason` and the quote of the whole statement. A refusal that names its own span is
   evidence; a dropped condition is not.
4. TOKENS ARE CATEGORICAL. A condition stated as a word, not a number (the fuel, the
   diluent, the reactor material), goes under `categoricals` with its token quote.
5. NAME THE APPARATUS, OR REFUSE TO. Set `subject` to a `device_class` with the quote
   that names the apparatus, or -- if the document does not support naming one -- to an
   `unresolved_subject` with the matching `reason` and the quote that makes it
   unresolvable. Do not guess a device the text does not name.
6. ATTRIBUTION. Say whose conditions these are (`own_experiment`, `cited_third_party`
   or `simulation`) and quote the text you read it from. This is your assertion; it is
   recorded but never verified, so quote honestly and refuse (`unextracted` reason
   `attribution_unclear`) when a value's owner is genuinely unclear.
7. DISAMBIGUATE REPEATS. If a quote appears more than once in the document, set the
   matching `*_occurrence` to say which one you mean, or the grounder will refuse it as
   ambiguous. Count 1-based: `1` is the FIRST time the quote appears, `2` the second,
   and so on. Never use `0`. Leave `*_occurrence` unset (null) only when the quote is
   unique.

Extracting nothing but a refusal is a valid, honest outcome. Do NOT propose an
`observable` / series: this system cannot yet ground one, and a proposed observable
will be refused. If the document supports no condition at all, report a single
`unextracted` refusal explaining why rather than an empty proposal.
"""


def build_extraction_agent(
    *,
    model: ModelProtocol,
    ledger: BudgetLedger,
    tools: Sequence[AgentTool] = (),
) -> CarmelAgent:
    """Build the Extraction Agent persona.

    ``tools`` defaults to none and should stay that way: like the corpus and verifier
    personas, this agent is handed no live tools -- the orchestration loads the
    document text deterministically and places it in the prompt
    (:func:`carmel.services.proposal_intake.build_extraction_prompt`). The parameter
    exists only for parity with :func:`carmel.agents.literature_agent.build_literature_agent`
    and so a future deterministic tool can be injected without changing this signature.

    Args:
        model: The model to call (mock or real).
        ledger: Budget ledger gating this agent's model calls.
        tools: Tools exposed to the model; empty by default and normally left so.

    Returns:
        A configured :class:`CarmelAgent` producing :class:`ExtractionProposal`.
    """
    return CarmelAgent(
        name="extraction",
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        ledger=ledger,
        output_schema=ExtractionProposal,
    )
