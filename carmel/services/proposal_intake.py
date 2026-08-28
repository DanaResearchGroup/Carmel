"""Carry an :class:`ExtractionProposal` to a stored condition-set envelope.

This is the ONLY sanctioned route from what the Extraction Agent PROPOSES
(:class:`carmel.agents.extraction_agent.ExtractionProposal`, wholly untrusted) to
what Carmel STORES. It does one thing: translate each proposal member into the frozen
specification dataclass the condition-set producer already takes, and hand the batch to
the UNCHANGED :func:`carmel.services.condition_set_producer.produce_condition_set_from_artifact`.

The translation is deliberately mechanical -- one proposal type to one spec type,
quote for quote, occurrence for occurrence. It invents no vocabulary and makes no
grounding decision: grounding, span-stitching, uniqueness and every refusal remain the
producer's, exactly as they are for a hand-built spec. A proposal whose quote is not in
the document therefore surfaces here as the producer's own refusal
(``ConditionSetProducerError`` / ``QuoteGroundingError`` / ``DatasetProducerError``),
propagated unchanged -- it is never swallowed and never turned into a stored fact.

The one refusal this module owns itself is the observable channel: series/observable
production has no path in this runtime today
(:func:`carmel.services.dataset_producer.produce_envelope_from_artifact` refuses
unconditionally), so a proposal carrying any observable is refused here with
:class:`ProposalIntakeError` rather than silently dropped -- a dropped observation and
a refused one are different facts, and this system exists to keep them apart.
"""

from __future__ import annotations

from pathlib import Path

from carmel.agents.extraction_agent import (
    ExtractionProposal,
    ProposedCategoricalCondition,
    ProposedDeviceClass,
    ProposedScalarCondition,
    ProposedUnextractedCondition,
    ProposedUnresolvedSubject,
)
from carmel.schemas.datasets import ConditionSetEnvelope
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    DeviceClassSpec,
    ScalarConditionSpec,
    UnextractedConditionSpec,
    UnresolvedSubjectSpec,
    produce_condition_set_from_artifact,
)
from carmel.services.extraction_record import (
    CurrentSelectionKind,
    select_current_extraction,
)

__all__ = [
    "ProposalIntakeError",
    "build_extraction_prompt",
    "condition_set_from_proposal",
    "current_extraction_text",
]


class ProposalIntakeError(RuntimeError):
    """A proposal could not be carried to the producer at all.

    Distinct from the producer's own errors on purpose: this names a refusal that
    fired BEFORE grounding -- today, only a proposal carrying an observable the runtime
    cannot yet assemble. A caller that wants to tell "the carrier declined to try" from
    "the producer grounded and refused" can, by the exception type.
    """


def current_extraction_text(workspace_root: Path, sha256: str) -> str:
    """The current extraction record's text for a held artifact, fail-closed.

    Returns exactly the text the condition-set producer will later ground against, so
    the Extraction Agent quotes from the same bytes the grounder checks. This is the
    deterministic I/O the persona needs; it is NOT a model-invoked tool. Selection is
    delegated to :func:`select_current_extraction` -- the single authority on which
    record is current -- so this cannot serve text that authority declined to vouch
    for.

    Raises:
        ProposalIntakeError: The artifact has no usable current extraction record.
    """
    selection = select_current_extraction(workspace_root, sha256)
    if selection.kind is not CurrentSelectionKind.SELECTED or selection.selected is None:
        raise ProposalIntakeError(
            f"artifact {sha256!r} has no usable current extraction record ({selection.detail}); "
            "cannot show the Extraction Agent a document to read. Re-extract the artifact first"
        )
    return selection.selected.extracted.text


def build_extraction_prompt(*, objective: str, artifact_sha256: str, text: str) -> str:
    """The user prompt for one extraction pass over one held document.

    Mirrors the corpus persona's prompt discipline: the whole document text is embedded
    here, deterministically, because the agent has no tool to fetch it -- and the
    embedded text is the current-extraction text (see :func:`current_extraction_text`)
    the producer will ground against, so a verbatim quote the agent copies from it can
    actually be located. ``objective`` states what the campaign is looking for; it is
    advisory framing, never a licence to report a condition the document does not state.

    Args:
        objective: One line on what conditions this campaign cares about.
        artifact_sha256: The held document's sha256; the agent must echo it into
            ``ExtractionProposal.artifact_sha256``.
        text: The document's current-extraction text, embedded verbatim -- not one
            character is added inside the markers (see the body).

    Returns:
        The user prompt string.
    """
    return (
        f"Campaign objective: {objective}\n\n"
        f"Document sha256 (echo this exactly into artifact_sha256): {artifact_sha256}\n\n"
        "Full text of the document follows between the markers. Quote ONLY from within "
        "these markers, character for character.\n"
        "<<<DOCUMENT>>>\n"
        # NO separator is added after `text`. An extra newline here would sit INSIDE
        # the markers the agent is told to quote from, so a model quoting the document's
        # last line together with its terminator could copy a character the stored
        # extraction text does not contain -- and the producer would then refuse to
        # ground a quote whose visible words match perfectly. The end marker is put on
        # its own line only when `text` does not already end one.
        f"{text}" + ("" if text.endswith("\n") else "\n") + "<<<END DOCUMENT>>>\n"
    )


def _to_zero_based(occurrence: int | None) -> int | None:
    """Convert one 1-based proposal occurrence into the grounder's 0-based index.

    The two ends of the occurrence selector count differently, ON PURPOSE, and this is
    the single seam that reconciles them. The Extraction Agent's prompt and
    :class:`~carmel.agents.extraction_agent.ExtractionProposal` speak 1-based ("``2`` =
    the second match"), because that is how a human or a model naturally counts a
    repeated quote. The grounder
    (:func:`carmel.services.dataset_producer.ground_quote`) and every frozen
    ``*ConditionSpec`` it is fed speak 0-based, and that convention is load-bearing
    across the grounder's OTHER callers (the table-cell VALUE/UNIT path in
    ``dataset_producer`` and every hand-built spec), so it is NOT the end that moves.

    Instead the untrusted 1-based value is converted to 0-based HERE, at the one
    sanctioned proposal -> spec boundary, and nowhere else. ``None`` ("this quote is
    unique") passes through unchanged. A concrete ``N`` becomes ``N - 1``; the schema's
    ``ge=1`` floor on every proposal occurrence field guarantees ``N >= 1``, so the
    result is never negative and always lands in the grounder's domain -- an off-by-one
    that would otherwise select the (N+1)-th span, grounding cleanly against the wrong
    place instead of refusing, is closed at the source.
    """
    if occurrence is None:
        return None
    return occurrence - 1


def _scalar_spec(proposed: ProposedScalarCondition) -> ScalarConditionSpec:
    return ScalarConditionSpec(
        claim_id=proposed.claim_id,
        label_quote=proposed.label_quote,
        quantity_kind=proposed.quantity_kind,
        value_quote=proposed.value_quote,
        unit_quote=proposed.unit_quote,
        label_occurrence=_to_zero_based(proposed.label_occurrence),
        value_occurrence=_to_zero_based(proposed.value_occurrence),
        unit_occurrence=_to_zero_based(proposed.unit_occurrence),
    )


def _categorical_spec(proposed: ProposedCategoricalCondition) -> CategoricalConditionSpec:
    return CategoricalConditionSpec(
        claim_id=proposed.claim_id,
        label_quote=proposed.label_quote,
        token_quote=proposed.token_quote,
        label_occurrence=_to_zero_based(proposed.label_occurrence),
        token_occurrence=_to_zero_based(proposed.token_occurrence),
    )


def _unextracted_spec(proposed: ProposedUnextractedCondition) -> UnextractedConditionSpec:
    return UnextractedConditionSpec(
        statement_id=proposed.statement_id,
        label_quote=proposed.label_quote,
        statement_quote=proposed.statement_quote,
        reason=proposed.reason,
        quantity_kind=proposed.quantity_kind,
        label_occurrence=_to_zero_based(proposed.label_occurrence),
        statement_occurrence=_to_zero_based(proposed.statement_occurrence),
    )


def _subject_spec(proposal: ExtractionProposal) -> DeviceClassSpec | UnresolvedSubjectSpec:
    subject = proposal.subject
    if isinstance(subject, ProposedDeviceClass):
        return DeviceClassSpec(
            label_quote=subject.label_quote,
            label_occurrence=_to_zero_based(subject.label_occurrence),
        )
    if isinstance(subject, ProposedUnresolvedSubject):
        return UnresolvedSubjectSpec(
            reason=subject.reason,
            reason_quote=subject.reason_quote,
            reason_occurrence=_to_zero_based(subject.reason_occurrence),
        )
    # Unreachable: ExtractionProposal.subject is a closed discriminated union, so
    # pydantic has already refused anything else before this function is entered.
    raise ProposalIntakeError(  # pragma: no cover
        f"unknown subject proposal type {type(subject).__name__!r} -- the proposal schema's "
        "subject union changed without updating this carrier"
    )


def condition_set_from_proposal(
    workspace_root: Path,
    proposal: ExtractionProposal,
    *,
    expected_sha256: str,
) -> ConditionSetEnvelope:
    """Translate a validated proposal into a produced, replayable envelope.

    Every spec-shaped member of ``proposal`` is mapped to its frozen spec and handed to
    the unchanged condition-set producer, which grounds each quote and refuses anything
    it cannot locate. This function adds NO grounding, NO validation and NO vocabulary
    to the quotes themselves; it is the mechanical carrier the ticket calls for. It does
    enforce ONE precondition the producer cannot: that the proposal grounds against the
    SAME document the caller prompted the agent with.

    ``proposal.artifact_sha256`` is a SELECTOR the untrusted model fills -- it chooses
    WHICH held document every quote is matched against -- and the prompt merely asks the
    model to echo the sha it was given. An echo is not a check: a model that emits any
    other document's sha would have its quotes grounded against THAT document, and if
    they happened to occur there the result would be a fully valid, fully replayable
    extraction attributed to the wrong paper -- provenance certifying the error instead
    of catching it. So the sha the CALLER prompted with (``expected_sha256``, the value
    it already holds and the sole authority) is compared against the model's copy BEFORE
    any grounding, and a mismatch is refused outright, naming both shas. The check fires
    even when the substituted document would have grounded successfully, because the
    defect is the mis-selection itself, not a grounding failure.

    Args:
        workspace_root: Workspace root holding the content-addressed store.
        proposal: The Extraction Agent's validated output.
        expected_sha256: The sha256 of the document the caller actually prompted the
            agent with (see :func:`build_extraction_prompt`). This is the authority;
            ``proposal.artifact_sha256`` must equal it.

    Returns:
        A fully validated :class:`ConditionSetEnvelope`.

    Raises:
        ProposalIntakeError: ``proposal.artifact_sha256`` differs from
            ``expected_sha256`` (a mis-selected document), or the proposal carries an
            observable, which has no assembly path in this runtime and is refused rather
            than dropped.
        ConditionSetProducerError: The producer refused a spec (empty set, id
            collision, incoherent scalar, unknown unit, ...).
        QuoteGroundingError: A quote is absent from the document, or occurs more than
            once and was not disambiguated.
        DatasetProducerError: The artifact is missing, legacy, corrupt, lossily
            extracted, or has no usable current extraction record.
    """
    if proposal.artifact_sha256 != expected_sha256:
        raise ProposalIntakeError(
            f"proposal artifact_sha256={proposal.artifact_sha256!r} does not match the document the "
            f"caller prompted with, expected_sha256={expected_sha256!r}. artifact_sha256 is a selector "
            "the untrusted model fills, and the prompt only asks it to echo the sha it was given -- so "
            "a mismatch means the model chose a DIFFERENT held document to ground every quote against, "
            "which could yield a valid, replayable extraction attributed to the wrong paper. Refusing "
            "before grounding; the sha the caller already holds is the authority, not the model's copy"
        )
    if proposal.observables:
        ids = sorted(observable.observable_id for observable in proposal.observables)
        raise ProposalIntakeError(
            f"proposal for artifact {proposal.artifact_sha256!r} carries {len(ids)} observable(s) "
            f"{ids!r}, which this runtime cannot ground: a char span in running text cannot ground a "
            "series data point (produce_envelope_from_artifact refuses unconditionally), and no "
            "series assembler exists yet. Refusing the proposal rather than silently dropping the "
            "observations -- a dropped observation and a refused one are different facts"
        )
    return produce_condition_set_from_artifact(
        workspace_root,
        sha256=proposal.artifact_sha256,
        attribution=proposal.attribution,
        attribution_quote=proposal.attribution_quote,
        attribution_occurrence=_to_zero_based(proposal.attribution_occurrence),
        subject=_subject_spec(proposal),
        scalars=tuple(_scalar_spec(s) for s in proposal.scalars),
        categoricals=tuple(_categorical_spec(c) for c in proposal.categoricals),
        unextracted=tuple(_unextracted_spec(u) for u in proposal.unextracted),
    )
