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
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from carmel.agents.bridge import AgentTool, CarmelAgent, ModelProtocol
from carmel.agents.budget import BudgetLedger
from carmel.schemas.acquisition import ALLOWED_LANDING_URL_SCHEMES as _ALLOWED_LANDING_URL_SCHEMES
from carmel.schemas.literature import Citation, CredenceVerdict, FindingPayload

__all__ = [
    "LITERATURE_SYSTEM_PROMPT",
    "VERIFIER_SYSTEM_PROMPT",
    "LiteratureProposal",
    "ProposedFinding",
    "CORPUS_SYSTEM_PROMPT",
    "CorpusFinding",
    "CorpusProposal",
    "VerifierAssessment",
    "build_corpus_agent",
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


class RequestedPaper(BaseModel):
    """A paper the agent judges relevant but cannot read for itself.

    This channel exists because of a structural gap: a finding REQUIRES a verbatim quote,
    which requires having read the document. A paywalled paper can therefore never be
    reported as a finding, however obviously relevant it is -- so without a separate way
    to say "I need this one", the papers most worth having would be exactly the ones that
    vanish silently. A live probe put the share of combustion-kinetics papers Carmel can
    read unaided at 3.3%, which makes that gap the normal case rather than a corner.

    Nothing here is evidence. It is a request for a human to obtain the document, after
    which the ordinary fetch/ground/verify path applies to it unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    doi: str | None = None
    landing_url: str | None = None
    """Where the paper can be obtained, if the search result gave one."""

    @field_validator("landing_url")
    @classmethod
    def _reject_unsafe_url_scheme(cls, value: str | None) -> str | None:
        """Reject any scheme but http/https.

        This field is proposed by the LLM and flows into the operator dashboard's
        "obtain" link, so it is attacker-influenceable via prompt injection in fetched
        page content. Jinja autoescaping stops attribute breakout but not a
        ``javascript:`` or ``data:`` URI, which would execute in the operator's
        authenticated session. Validating here means the value cannot reach any renderer
        that forgets to re-check it -- and it is the same rule
        :class:`carmel.schemas.acquisition.AcquisitionRequest` enforces, which is where
        this value ends up.
        """
        if value is None:
            return None
        scheme = urlsplit(value).scheme.lower()
        if scheme not in _ALLOWED_LANDING_URL_SCHEMES:
            raise ValueError(f"landing_url scheme {scheme!r} is not allowed; must be http or https")
        return value

    relevance: str = Field(default="", max_length=500)
    """Why this paper is worth a human's effort -- shown to the operator, who is being
    asked to spend real time on it and deserves to know what it is for."""


class LiteratureProposal(BaseModel):
    """The Literature Agent's structured output for one loop round."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(default_factory=list)
    """Search queries the agent wants run (results are fed back next round)."""
    findings: list[ProposedFinding] = Field(default_factory=list)
    wanted: list[RequestedPaper] = Field(default_factory=list)
    """Relevant papers the agent could not read; queued for manual acquisition."""
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


class CorpusFinding(BaseModel):
    """One finding the agent CLAIMS to have found in an already-held document.

    Differs from :class:`ProposedFinding` in exactly one way, and it is the point of
    the whole corpus pass: the evidence is named by ``artifact_sha256``, a document
    Carmel already possesses, rather than by a URL it would have to go and fetch.
    Nothing else about the evidentiary contract changes -- the quote is still checked
    against the real bytes by the same deterministic gate, and identity is still
    checked against the document's own text.
    """

    model_config = ConfigDict(extra="forbid")

    payload: FindingPayload
    citation: Citation
    verbatim_quote: str = Field(min_length=40)
    """Same 40-character floor as :class:`ProposedFinding`, for the same reason: a
    degenerate quote matches almost any document and skips the gate's downstream
    fuzzy-mismatch and numeric-discrepancy defenses."""
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Which held document contains the quote. Constrained to a sha256 digest so a
    hallucinated handle fails validation immediately rather than being looked up."""


class CorpusProposal(BaseModel):
    """The Corpus Agent's structured output for one round.

    There is no ``queries`` channel and no ``wanted`` channel. A corpus pass cannot
    search and cannot request papers -- it reads what is already held. Omitting those
    fields entirely, rather than accepting and discarding them, means the schema
    itself states the boundary instead of relying on the orchestrator to enforce it.
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[CorpusFinding] = Field(default_factory=list)
    done: bool = False
    """The agent's self-stop signal: True means it has nothing further to extract
    from the corpus."""


CORPUS_SYSTEM_PROMPT = """\
You are Carmel's Literature Agent, working in CORPUS mode.

You are given the full text of papers Carmel already holds. Your job is to extract
experimental benchmarks, prior kinetic models, and quantum-chemistry calculation
results relevant to the campaign, as structured findings grounded in those papers.

You have NO search and NO web access, and you need none: every document you may use is
already in front of you. Do not report a finding about any paper not listed below, and
never speculate about what an unlisted paper might contain.

Non-negotiable evidentiary rules — findings that break them WILL be rejected by a
deterministic gate that checks your claims against the actual stored bytes:

1. VERBATIM QUOTES ONLY. `verbatim_quote` must be copied EXACTLY, character for
   character, from the document text shown to you. Never paraphrase, never summarize,
   never "clean up" a sentence, and never repair what looks like an extraction error.
   The quote must be at least 40 characters long.
2. `artifact_sha256` must be the digest of the document the quote actually came from,
   copied from the corpus listing. Quoting document A while naming document B is the
   single worst error you can make here: it attaches one paper's evidence to another
   paper's citation.
3. REAL CITATIONS, taken from the document itself. The citation must describe the
   paper the quote came from -- the DOI and title as that paper states them. Do not
   cite a work that the document merely references. If the document is a review that
   quotes a measurement from elsewhere, the finding belongs to whatever the review
   itself is, not to the primary source it cites, and the identity check will refuse
   it if you claim otherwise.
4. TYPED NUMERIC VALUES WITH UNITS. Every quantity must be reported as a typed numeric
   value with its unit (e.g. value=1.25, unit="ms"). The quoted text must contain the
   same numbers.

Extracting nothing is a valid and honest outcome. If the corpus does not support a
finding relevant to this campaign, set `done` to true and report no findings. A
fabricated or loosely-grounded finding is far worse than an empty result: the entire
point of this system is that a claim which cannot be traced to real bytes does not
survive.
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

Most papers in this field are behind a paywall and you will NOT be able to read them.
That is expected, and it is not a failure. Each search result is annotated with
`FULL TEXT: yes` or `FULL TEXT: no`.

For every result marked `FULL TEXT: no` that is relevant to the campaign, you MUST list
it under `wanted`, with its title, DOI, and a one-line reason it matters. Do not guess at
its contents and never invent a quote for it: a fabricated quote is far worse than an
unread paper. A human will obtain it and it will be analysed properly on a later run.

Returning an empty response — no `findings` and no `wanted` — when the search results
contain relevant papers is a WRONG answer, and the most common mistake made here. If you
cannot quote a relevant paper, that is precisely the case `wanted` exists for. Never set
`done=true` while relevant results you have not requested are still on the table.

Loop protocol: each round, return `queries` you want searched (results come back next
round), `findings` you can already support with a verbatim quote, `wanted` papers you
need a human to obtain, and `done`. Set `done=true` when further searching is unlikely to
surface new relevant findings. Do not re-report findings or re-request papers you have
already listed in a previous round.
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


def build_corpus_agent(*, model: ModelProtocol, ledger: BudgetLedger) -> CarmelAgent:
    """Build the Literature Agent's corpus persona.

    Takes no ``tools`` parameter at all, where :func:`build_literature_agent` accepts
    one. A corpus pass performs no search and no fetching, so there is nothing a tool
    could legitimately do; making that unrepresentable here means a future caller
    cannot quietly hand this agent web access and turn a reproducible pass into a
    live one.

    Args:
        model: The model to call (mock or real).
        ledger: Budget ledger gating this agent's model calls.

    Returns:
        A configured :class:`CarmelAgent` producing :class:`CorpusProposal`.
    """
    return CarmelAgent(
        name="literature_corpus",
        system_prompt=CORPUS_SYSTEM_PROMPT,
        model=model,
        tools=(),
        ledger=ledger,
        output_schema=CorpusProposal,
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
