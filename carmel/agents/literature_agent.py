"""The Literature Agent and Verifier personas: prompts, output schemas, factories.

Two strictly separated personas share the generic :class:`~carmel.agents.bridge.CarmelAgent`
bridge:

- The **Literature Agent** proposes findings. Everything it emits
  (:class:`ProposedFinding`) is UNTRUSTED until the deterministic grounding gate
  (:func:`carmel.services.grounding.ground_finding`) has corroborated the claimed quote
  against actually-fetched bytes.
- The **Verifier** scores credence for findings that SURVIVED grounding. It is
  deliberately given only sanitized evidence (payload, citation, quote, an extracted-text
  window, and the grounding verdict) — never the author agent's raw URLs or unquoted
  assertions — so one LLM can never launder another LLM's fabrication.

The orchestration that enforces this ordering lives in
:mod:`carmel.services.literature`; this module only defines the personas.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from carmel.agents.bridge import AgentTool, CarmelAgent, ModelProtocol
from carmel.agents.budget import BudgetLedger
from carmel.schemas.literature import Citation, CredenceVerdict, FindingPayload

__all__ = [
    "LITERATURE_SYSTEM_PROMPT",
    "VERIFIER_SYSTEM_PROMPT",
    "LiteratureProposal",
    "ProposedFinding",
    "VerifierAssessment",
    "build_literature_agent",
    "build_verifier_agent",
]


class ProposedFinding(BaseModel):
    """One finding the Literature Agent CLAIMS to have found.

    This is the untrusted, pre-grounding shape: nothing in it is believed until
    ``fetch -> extract_text -> store_artifact -> ground_finding`` has corroborated the
    quote against the actual bytes at ``source_url``.
    """

    model_config = ConfigDict(extra="forbid")

    payload: FindingPayload
    citation: Citation
    verbatim_quote: str = Field(min_length=40)
    """The quote the agent claims appears verbatim in the source document.

    The floor is 40 characters, not 1: a one-character (or otherwise degenerate) quote
    matches almost any document with ``exact=True, ratio=1.0``, which also skips the
    fuzzy-mismatch and numeric-discrepancy defenses downstream in the grounding gate
    (:func:`carmel.services.grounding.ground_finding`). 40 characters is long enough to
    carry real evidentiary content (a clause or short sentence fragment, not just a
    bare token or number) while still being short enough not to reject a legitimate
    table-row quote once it includes a few words of surrounding context (e.g. "ignition
    delay of 1.25 ms at 1200 K, phi=1.0" clears 40 chars easily; a bare "1.25 ms" does
    not, and is exactly the kind of degenerate quote this floor is meant to reject).
    """
    source_url: str = Field(min_length=1)
    """Where the orchestrator must fetch the evidence from. Untrusted until fetched."""


class LiteratureProposal(BaseModel):
    """The Literature Agent's structured output for one loop round."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(default_factory=list)
    """Search queries the agent wants run (results are fed back next round)."""
    findings: list[ProposedFinding] = Field(default_factory=list)
    done: bool = False
    """The agent's self-stop signal: True means it believes the search is complete."""


class VerifierAssessment(CredenceVerdict):
    """The Verifier's structured-output schema for one grounded finding.

    Inherits every field (``credence``, ``provenance_score``, ``quality_score``,
    ``consistency_score``, ``rationale``, ``flags``) directly from
    :class:`carmel.schemas.literature.CredenceVerdict` instead of hand-transcribing
    them, so the two can never drift out of sync field-for-field. The orchestrator
    (:mod:`carmel.services.literature`) still builds the final ``CredenceVerdict`` from
    this assessment after applying its own deterministic penalties (fuzzy grounding,
    failed canonicalization) -- kept as a distinct type (rather than reusing
    ``CredenceVerdict`` directly as the output schema) only so that distinction stays
    visible in the type signature of :func:`build_verifier_agent`.
    """


LITERATURE_SYSTEM_PROMPT = """\
You are Carmel's Literature Agent: a combustion-chemistry literature researcher.

Your job is to find experimental benchmarks, prior kinetic models, and quantum-chemistry
calculation results relevant to the campaign described in the user prompt, and to report
them as structured findings.

Non-negotiable evidentiary rules — findings that break them WILL be rejected by a
deterministic grounding gate that checks your claims against the actual fetched document:

1. VERBATIM QUOTES ONLY. `verbatim_quote` must be copied EXACTLY, character for
   character, from the source document. Never paraphrase, never summarize, never
   "clean up" a sentence. If you cannot quote it exactly, do not report the finding.
   The quote must be at least 40 characters long: a bare number or token is not a
   quote and will be rejected. Quote enough of the surrounding sentence or table row
   to carry real evidentiary content.
2. REAL CITATIONS. Provide a citation with a DOI whenever one exists. Only when a
   source genuinely has no DOI may you fall back to a URL or a stable source_id.
   Never invent, guess, or "reconstruct" a DOI.
3. TYPED NUMERIC VALUES WITH UNITS. Every quantity must be reported as a typed
   numeric value together with its unit exactly as supported by the schema
   (e.g. value=1.25, unit="ms"). The quoted text must contain the same numbers.
4. `source_url` must point at the document that actually contains the quote — the
   grounding gate fetches that URL and searches for your quote in its text.

Loop protocol: each round, return `queries` you want searched (results come back next
round), `findings` you can already support with a verbatim quote, and `done`. Set
`done=true` when further searching is unlikely to surface new relevant findings. Do not
re-report findings you have already reported in a previous round.
"""

VERIFIER_SYSTEM_PROMPT = """\
You are Carmel's Literature Verifier: an independent, skeptical assessor.

You receive ONE finding at a time, consisting only of: the structured payload, the
citation, the claimed verbatim quote, a bounded window of text extracted from the
fetched source document around the located quote, and a deterministic grounding
verdict. That evidence is ALL you may use.

Score the finding on three axes, each in [0, 1]:
- provenance_score: how well the citation and grounding verdict tie the claim to an
  identifiable, authoritative source.
- quality_score: how methodologically sound and precisely reported the evidence in the
  supplied text window is (apparatus, uncertainty, level of theory, ...).
- consistency_score: how consistent the structured payload is with the quote and the
  surrounding text window (values, units, conditions, species).

Then give an overall `credence` in [0, 1], a short `rationale`, and any `flags`.

Hard rules:
- Score ONLY from the evidence supplied in the prompt. Do NOT use outside knowledge of
  the paper, the authors, or the field to fill gaps.
- Do NOT trust any claim you cannot see corroborated in the given text. If a payload
  value, unit, or condition is not visible in the quote or the text window, treat it as
  unsupported and lower consistency_score and credence accordingly.
- You have no tools, no web access, and no way to fetch anything: never assume missing
  evidence would check out.
"""


def build_literature_agent(
    *,
    model: ModelProtocol,
    tools: Sequence[AgentTool] = (),
    ledger: BudgetLedger,
) -> CarmelAgent:
    """Build the Literature Agent persona.

    Args:
        model: The model to call (mock or real).
        tools: Tools exposed to the model (typically search; injected by the caller).
        ledger: Budget ledger gating this agent's model calls.

    Returns:
        A configured :class:`CarmelAgent` producing :class:`LiteratureProposal`.
    """
    return CarmelAgent(
        name="literature",
        system_prompt=LITERATURE_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        ledger=ledger,
        output_schema=LiteratureProposal,
    )


def build_verifier_agent(*, model: ModelProtocol, ledger: BudgetLedger) -> CarmelAgent:
    """Build the Verifier persona.

    The Verifier deliberately gets NO tools: it must judge only from the evidence in
    its prompt and can never fetch anything itself.

    Args:
        model: The model to call (mock or real).
        ledger: Budget ledger gating this agent's model calls.

    Returns:
        A configured :class:`CarmelAgent` producing :class:`VerifierAssessment`.
    """
    return CarmelAgent(
        name="verifier",
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        model=model,
        tools=(),
        ledger=ledger,
        output_schema=VerifierAssessment,
    )
