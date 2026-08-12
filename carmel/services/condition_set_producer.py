"""Produce a validated :class:`ConditionSetEnvelope` from one stored artifact.

Carmel could already STORE, LOAD and REPLAY a condition set before this module
existed -- :mod:`carmel.services.condition_set_bridge` and
:func:`carmel.services.dataset_replay.replay_condition_set` were both complete --
but nothing could MAKE one. Every condition-set envelope in the suite was
hand-built in test code, which meant the replayer had never once been handed real
producer output. This module closes that gap.

WHAT THIS PRODUCER IS FOR, AND WHY IT IS SHAPED THIS WAY

A char span into extracted running text can LOCATE a scalar statement -- "the
mixture was preheated to 323 K". It cannot locate a SERIES data point, because a
series is a structure (rows against columns, or points against axes) and running
text carries no such structure at all. That asymmetry is why this module exists:
it is the GROUNDED destination for the scalar-shaped half of what a text-only
extractor can see, so that the series-shaped half can be refused elsewhere
without also destroying this one.

Be exact about what "grounded destination" does and does not mean, because the
tempting phrasing -- "the honest destination for prose-local scalars" -- is an
overclaim this module cannot back. This code CANNOT prove prose-locality, and it
cannot prove that a located label, value and unit are predicated of one another
by the paper. It proves that each quote is where the locator says it is, and it
derives the numeric/unit normalization deterministically from the value quote.
Everything past that is the caller's assertion, recorded unverified.

One shape of that gap is now CLOSED, and the closure is narrow enough to state
exactly. A caller used to be able to stitch the label "pressure" to the value
"823" and the unit "atm" out of a sentence that says 823 K and 1.2 atm: every
span grounded, replay reported VERIFIED, and the paper never stated that
condition. Co-location could not close it -- that false triple comes from a
single sentence -- so the rule is uniqueness instead: the span COVERING a
claim's three grounds must hold exactly one number+unit construct, that
construct must be the claimed value and unit compared by offset, and its unit
must denote the declared quantity. See :mod:`carmel.services.stitching`; the
same gate re-runs in
:func:`carmel.services.dataset_replay.replay_condition_set`, because a
write-path-only gate says nothing about an envelope built by another route.

That gate REFUTES; it never verifies. A claim surviving it is not thereby
shown to be what the paper predicates -- only that one named refutation was
attempted and did not fire. Known shapes it does NOT refuse: a one-sided bound
or method threshold ("above 60 cm/s") reads as a single construct, and shared
dimensionless spellings cannot separate mole fraction from equivalence ratio
from a relative uncertainty.

GROUNDING PROVES LOCATION, NEVER MEANING. Every ``SourceRef`` this producer emits
is independently verified to be an exact, located substring of the authenticated
document. NOTHING here verifies that the located string MEANS what the caller
says it means -- that a quote the caller labelled "initial temperature" really is
the initial temperature, or that a number in the text is a reported condition
rather than a chart tick. The schema records the caller's assertion; it does not
bless it.

THE THREE-WAY SPLIT IS THE HONESTY MECHANISM

A condition either resolves to one grounded number (:class:`ScalarConditionSpec`),
or to one grounded categorical token (:class:`CategoricalConditionSpec`), or it is
REFUSED with the reason recorded and the span still grounded
(:class:`UnextractedConditionSpec`). A sweep, a range, a one-sided bound or a
qualitative-only statement is not squeezed into a single number -- it is recorded
as an explicit refusal that still points at the text it refused. A refusal that
names its own span is evidence; a silently dropped condition is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ConditionAttribution,
    ConditionSetEnvelope,
    DeviceClassDeclaration,
    GroundedCategoricalClaim,
    GroundedScalarClaim,
    SourceRef,
    SubjectRefusalReason,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
)
from carmel.services import units
from carmel.services.dataset_producer import (
    _ACTIVE,
    _ROOT_NODE_ID,
    DatasetProducerError,
    _measured_value,
    _prepare_grounding,
    ground_quote,
)
from carmel.services.numeric import QuoteRole
from carmel.services.stitching import (
    StitchGateUnrunnable,
    StitchRefutation,
    refute_stitched_claim,
)

__all__ = [
    "CategoricalConditionSpec",
    "ConditionSetProducerError",
    "DeviceClassSpec",
    "ScalarConditionSpec",
    "UnextractedConditionSpec",
    "UnresolvedSubjectSpec",
    "produce_condition_set_from_artifact",
]


class ConditionSetProducerError(DatasetProducerError):
    """A condition set could not be honestly produced.

    Subclasses :class:`DatasetProducerError` deliberately: callers that already
    fail closed on "a producer refused" keep working unchanged, while a caller
    that wants to tell the two producers apart still can.
    """


def _require_int_occurrences(owner: str, **occurrences: int | None) -> None:
    """Reject non-int occurrence values, ``bool`` included.

    These specs are frozen plain dataclasses, so nothing downstream re-checks
    their fields. ``bool`` is a subclass of ``int`` in Python, so a bare
    ``isinstance(x, int)`` would silently accept ``True``/``False`` as
    occurrence 1/0 -- almost certainly a caller typo, never a real
    disambiguation intent. Mirrors ``MeasurementSpec.__post_init__``.
    """
    for name, value in occurrences.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConditionSetProducerError(
                f"{owner}.{name}={value!r} must be an int or None, not {type(value).__name__} "
                "-- bool is a subclass of int in Python and would silently mean occurrence 0/1"
            )


@dataclass(frozen=True, slots=True)
class ScalarConditionSpec:
    """One stated condition that resolves to a single grounded number.

    Carries no ``axis_id`` and no ``AxisRole``: a condition is not a point on a
    series. It satisfies ``dataset_producer._ValueQuoteSpec`` structurally, which
    is what lets it share :func:`_measured_value` with the dataset producer
    without either borrowing the other's vocabulary.
    """

    claim_id: str
    label_quote: str
    quantity_kind: units.QuantityKind
    value_quote: str
    unit_quote: str
    label_occurrence: int | None = None
    value_occurrence: int | None = None
    unit_occurrence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity_kind, units.QuantityKind):
            raise ConditionSetProducerError(
                f"ScalarConditionSpec.quantity_kind={self.quantity_kind!r} must be a genuine "
                f"QuantityKind member, not {type(self.quantity_kind).__name__} -- QuantityKind "
                "is a StrEnum, so a plain string equal to a member's value would compare `==` "
                "equal without actually being that member"
            )
        _require_int_occurrences(
            "ScalarConditionSpec",
            label_occurrence=self.label_occurrence,
            value_occurrence=self.value_occurrence,
            unit_occurrence=self.unit_occurrence,
        )


@dataclass(frozen=True, slots=True)
class CategoricalConditionSpec:
    """One stated condition whose value is a token, not a number.

    Fuel identity, diluent, reactor material: things a paper states as a word.
    There is no unit and no numeric normalization, so this deliberately does NOT
    go through :func:`_measured_value` -- the token is grounded and recorded raw.
    """

    claim_id: str
    label_quote: str
    token_quote: str
    label_occurrence: int | None = None
    token_occurrence: int | None = None

    def __post_init__(self) -> None:
        _require_int_occurrences(
            "CategoricalConditionSpec",
            label_occurrence=self.label_occurrence,
            token_occurrence=self.token_occurrence,
        )


@dataclass(frozen=True, slots=True)
class UnextractedConditionSpec:
    """A condition the extractor REFUSES to reduce to one value, span recorded.

    ``quantity_kind`` is a ``Maybe``: a refused statement may still be known to be
    a temperature even when no single temperature can be stated. It carries no
    unit, which is why the ref-less obligation machinery deliberately does not
    ask this class for a ``quantity_kind`` claim -- that rule is a predicate over
    ``unit_raw``, and this class has no unit.
    """

    statement_id: str
    label_quote: str
    statement_quote: str
    reason: UnextractedReason
    quantity_kind: units.QuantityKind | None = None
    label_occurrence: int | None = None
    statement_occurrence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, UnextractedReason):
            raise ConditionSetProducerError(
                f"UnextractedConditionSpec.reason={self.reason!r} must be a genuine "
                f"UnextractedReason member, not {type(self.reason).__name__}"
            )
        if self.quantity_kind is not None and not isinstance(self.quantity_kind, units.QuantityKind):
            # The same StrEnum trap the other spec fields close. Left open here,
            # a bare "temperature" would be coerced downstream by pydantic and
            # the refusal would surface as a schema ValidationError from deep
            # inside construction rather than as a producer refusal naming the
            # field the caller got wrong.
            raise ConditionSetProducerError(
                f"UnextractedConditionSpec.quantity_kind={self.quantity_kind!r} must be a "
                f"genuine QuantityKind member or None, not {type(self.quantity_kind).__name__}"
            )
        _require_int_occurrences(
            "UnextractedConditionSpec",
            label_occurrence=self.label_occurrence,
            statement_occurrence=self.statement_occurrence,
        )


@dataclass(frozen=True, slots=True)
class DeviceClassSpec:
    """The apparatus the paper names, to be grounded as the condition set's subject."""

    label_quote: str
    label_occurrence: int | None = None

    def __post_init__(self) -> None:
        _require_int_occurrences("DeviceClassSpec", label_occurrence=self.label_occurrence)


@dataclass(frozen=True, slots=True)
class UnresolvedSubjectSpec:
    """A REFUSAL to name the apparatus, with the span that motivates the refusal.

    The refusal is still grounded: it points at the text that made the subject
    unresolvable, so a reader can check the refusal rather than take it on trust.
    """

    reason: SubjectRefusalReason
    reason_quote: str
    reason_occurrence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SubjectRefusalReason):
            raise ConditionSetProducerError(
                f"UnresolvedSubjectSpec.reason={self.reason!r} must be a genuine "
                f"SubjectRefusalReason member, not {type(self.reason).__name__}"
            )
        _require_int_occurrences("UnresolvedSubjectSpec", reason_occurrence=self.reason_occurrence)


def _ref(text: str, quote: str, *, role: QuoteRole, occurrence: int | None) -> SourceRef:
    """Ground ``quote`` in ``text`` and wrap the located span as a ``SourceRef``."""
    return SourceRef(
        node_id=_ROOT_NODE_ID,
        locator=ground_quote(text, quote, role=role, occurrence=occurrence),
    )


def _refuse_stitched(claim: GroundedScalarClaim, text: str) -> GroundedScalarClaim:
    """Refuse a scalar claim whose three grounds provably do not cohere.

    The producer holds the document text, so it is the earliest place this can
    be caught -- but it is deliberately NOT the only place. The same gate runs
    in :func:`~carmel.services.dataset_replay.replay_condition_set`, because a
    producer-side refusal says nothing about an envelope that was stored before
    this rule existed, or constructed by any route that does not come through
    here.

    Passing this gate is NOT a verification. It means one named refutation was
    attempted and did not fire; whether the paper actually predicates this label
    of this number remains unproven, and nothing downstream may upgrade it.
    """
    outcome = refute_stitched_claim(claim, text)
    if isinstance(outcome, StitchRefutation):
        raise ConditionSetProducerError(
            f"scalar claim {claim.claim_id!r} (label {claim.label_raw!r}, value "
            f"{claim.value.raw_text!r} {claim.value.unit_raw!r}) is refused: {outcome.reason}"
        )
    if isinstance(outcome, StitchGateUnrunnable):
        raise ConditionSetProducerError(
            f"scalar claim {claim.claim_id!r} cannot be checked for span stitching: "
            f"{outcome.reason}. This producer grounds every quote as a character span into "
            "one root node, so reaching this state means the claim was built by a route that "
            "does not hold that invariant -- it is refused rather than stored unchecked"
        )
    return claim


def _duplicate_ids(ids: list[str], *, owner: str) -> None:
    """Refuse duplicate ids across ALL of a condition set's collections.

    Two entries sharing an id makes every downstream per-claim finding ambiguous
    about which one it is about -- a replayer would report a path that names two
    different things. ``claim_id`` and ``statement_id`` share a single namespace;
    that is not this function's choice, it is what ``ConditionSetEnvelope``
    validates, and this refuses early so the caller is told which id collided
    rather than being handed a schema error from inside construction.
    """
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ConditionSetProducerError(
                f"duplicate {owner} {value!r}: every claim and statement in a condition set "
                "must have a unique id, or a per-claim finding cannot say which one it means"
            )
        seen.add(value)


def produce_condition_set_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    attribution: ConditionAttribution,
    attribution_quote: str,
    subject: DeviceClassSpec | UnresolvedSubjectSpec,
    scalars: tuple[ScalarConditionSpec, ...] = (),
    categoricals: tuple[CategoricalConditionSpec, ...] = (),
    unextracted: tuple[UnextractedConditionSpec, ...] = (),
    attribution_occurrence: int | None = None,
) -> ConditionSetEnvelope:
    """Build one validated :class:`ConditionSetEnvelope` from a stored artifact.

    The vertical slice: authenticate ``raw.bin`` against the artifact's own
    sha256, select the ONE current extraction record, take the grounded text from
    it, ground every caller-stated quote in that text, and assemble an envelope
    that passes every schema validator. Construction runs pydantic's full
    validation -- nothing here uses ``model_construct``.

    The authentication preamble is NOT reimplemented here: it is
    :func:`~carmel.services.dataset_producer._prepare_grounding`, shared with the
    dataset producer, so that a fix to one producer's fail-closed path cannot
    silently miss the other's.

    Args:
        workspace_root: Workspace root holding the content-addressed store.
        sha256: The raw artifact's sha256.
        attribution: Whether these conditions are the paper's OWN experiment, a
            CITED third party's, or a SIMULATION. This is the caller's
            ASSERTION, recorded unverified -- ``attribution_quote`` grounds
            WHERE the assertion was read, never that it is correct.
        attribution_quote: The text the attribution was read from.
        subject: The apparatus, either named (:class:`DeviceClassSpec`) or
            explicitly refused (:class:`UnresolvedSubjectSpec`).
        scalars: Conditions resolving to one grounded number each.
        categoricals: Conditions resolving to one grounded token each.
        unextracted: Conditions REFUSED, each with its reason and its span.
        attribution_occurrence: Disambiguates a repeated ``attribution_quote``.

    Returns:
        A fully validated envelope.

    Raises:
        ConditionSetProducerError: Nothing was extracted at all, ids collide, a
            spec field is the wrong type, or a value is not a bare numeral / its
            unit is unknown for its quantity kind.
        DatasetProducerError: The artifact is missing, legacy, corrupt, lossily
            extracted, or has no usable current extraction record.
        QuoteGroundingError: A quote is absent from the document, or occurs more
            than once and was not disambiguated.
    """
    if not scalars and not categoricals and not unextracted:
        # An envelope asserting no condition at all is not a modest result, it is
        # a claim that the paper stated no conditions -- which this producer has
        # no way to establish. Refuse rather than emit a vacuously VERIFIED
        # envelope, which is precisely the overclaim shape this system exists to
        # prevent.
        raise ConditionSetProducerError(
            f"artifact {sha256!r}: refusing to produce a condition set with no scalar claims, "
            "no categorical claims and no recorded refusals -- an empty condition set asserts "
            "that the paper stated no conditions, which grounding cannot establish"
        )
    # ONE namespace across all three collections, because that is what
    # ConditionSetEnvelope itself enforces. Checking the two kinds separately
    # let a claim_id collide with a statement_id and surface as a pydantic
    # ValidationError from inside construction -- a late, badly-located refusal
    # for a caller error this producer can name precisely.
    _duplicate_ids(
        [s.claim_id for s in scalars] + [c.claim_id for c in categoricals] + [u.statement_id for u in unextracted],
        owner="id",
    )
    if not isinstance(attribution, ConditionAttribution):
        raise ConditionSetProducerError(
            f"attribution={attribution!r} must be a genuine ConditionAttribution member, not "
            f"{type(attribution).__name__} -- ConditionAttribution is a StrEnum, so a plain "
            "string equal to a member's value would compare `==` equal without being that member"
        )
    _require_int_occurrences("produce_condition_set_from_artifact", attribution_occurrence=attribution_occurrence)

    grounding = _prepare_grounding(
        workspace_root, sha256, envelope_noun="condition set", envelope_subject="A condition set"
    )
    text = grounding.text

    resolved_subject: DeviceClassDeclaration | UnresolvedSubject
    if isinstance(subject, DeviceClassSpec):
        resolved_subject = DeviceClassDeclaration(
            label_raw=subject.label_quote,
            label_ref=_ref(
                text,
                subject.label_quote,
                role=QuoteRole.LABEL,
                occurrence=subject.label_occurrence,
            ),
        )
    else:
        resolved_subject = UnresolvedSubject(
            reason=subject.reason,
            reason_ref=_ref(
                text,
                subject.reason_quote,
                role=QuoteRole.LABEL,
                occurrence=subject.reason_occurrence,
            ),
        )

    scalar_claims = tuple(
        _refuse_stitched(
            GroundedScalarClaim(
                claim_id=spec.claim_id,
                label_raw=spec.label_quote,
                label_ref=_ref(text, spec.label_quote, role=QuoteRole.LABEL, occurrence=spec.label_occurrence),
                value=_measured_value(
                    text,
                    spec,
                    where=f"claim {spec.claim_id!r}",
                    document_source_context=grounding.document_source_context,
                    document_glyph_health=grounding.document_glyph_health,
                ),
                # This producer reads no uncertainty from the document. That is a
                # NOT_EXTRACTED_YET refusal, not an assertion that the paper stated
                # none -- the two must never conflate.
                uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
            ),
            text,
        )
        for spec in scalars
    )
    categorical_claims = tuple(
        GroundedCategoricalClaim(
            claim_id=spec.claim_id,
            label_raw=spec.label_quote,
            label_ref=_ref(text, spec.label_quote, role=QuoteRole.LABEL, occurrence=spec.label_occurrence),
            token_raw=spec.token_quote,
            token_ref=_ref(text, spec.token_quote, role=QuoteRole.VALUE, occurrence=spec.token_occurrence),
        )
        for spec in categoricals
    )
    unextracted_statements = tuple(
        UnextractedConditionStatement(
            statement_id=spec.statement_id,
            label_raw=spec.label_quote,
            label_ref=_ref(text, spec.label_quote, role=QuoteRole.LABEL, occurrence=spec.label_occurrence),
            statement_ref=_ref(
                text,
                spec.statement_quote,
                role=QuoteRole.VALUE,
                occurrence=spec.statement_occurrence,
            ),
            reason=spec.reason,
            quantity_kind=(
                spec.quantity_kind if spec.quantity_kind is not None else Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
            ),
        )
        for spec in unextracted
    )

    return ConditionSetEnvelope(
        source_graph=grounding.graph,
        # Only a MeasuredValue cites a conversion table, so a condition set that
        # resolved no scalar claims must embed NONE: the schema refuses a
        # decorative table as "unearned provenance", and it is right to. A
        # refusal-only condition set is a legitimate result, and it may not
        # carry provenance for a conversion it never performed.
        conversion_tables=(_ACTIVE.embedded,) if scalar_claims else (),
        subject=resolved_subject,
        attribution=attribution,
        attribution_ref=_ref(text, attribution_quote, role=QuoteRole.LABEL, occurrence=attribution_occurrence),
        scalar_claims=scalar_claims,
        categorical_claims=categorical_claims,
        unextracted=unextracted_statements,
    )
