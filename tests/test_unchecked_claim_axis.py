# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The two AXES along which a replay can leave something unchecked.

``ReplayReport`` used to carry one list, ``unchecked_claims``, of one type,
``UncheckedClaim``. Both were scoped -- in the docstring, in the only
constructor, and in every test -- to claims about the STORE: "this node says its
root sidecar was authenticated, and that claim could not be tested."

The condition-set replay that lands next needs to record something else
entirely: a claim about MEANING that no re-slice can test. Note what that is
NOT -- the refs in question (``attribution_ref``, ``statement_ref``) carry
ordinary locators and their spans re-slice like any other. What they lack is the
other half of a grounding pair: the value they support is an enum, and no
recorded text says the span means it. The axis is a property of the PAIRING, not
of the ref (Codex round 86 caught the opposite claim in an earlier draft of this
docstring, and a probe of the schema confirmed it: the refs are ``SourceRef``s,
which always carry a locator).

Recording such a claim in the store-scoped list would make the report lie about
WHICH axis was left untested -- the entry would say "store" and mean "semantics"
(Codex round 85, P1).

So there are two types and two lists, no unqualified name survives, and both
lists downgrade :attr:`ReplayReport.overall_outcome`. These tests build reports
by hand: no producer emits a semantic claim yet, and the vocabulary has to be
right BEFORE one does, because a producer written against a lying type would
encode the lie at every call site.
"""

from __future__ import annotations

import pytest

from carmel.services.dataset_replay import (
    ReplayFinding,
    ReplayOutcome,
    ReplayReport,
    UncheckedSemanticClaim,
    UncheckedStoreClaim,
)


def _clean_report(**overrides: object) -> ReplayReport:
    """A report whose evidence checks all ran and all passed."""
    kwargs: dict[str, object] = {
        "checked_char_spans": 3,
        "total_char_spans": 3,
        "unchecked_char_spans": 0,
    }
    kwargs.update(overrides)
    return ReplayReport(**kwargs)  # type: ignore[arg-type]


_A_STORE_CLAIM = UncheckedStoreClaim(
    ref_path="source_graph.node('paper').verification.root_sidecar",
    claim="no_recorded_digest",
    reason="the root meta.json could not be read, so the claim was never tested",
)

_A_SEMANTIC_CLAIM = UncheckedSemanticClaim(
    # The DERIVED value, never a slice of the paper: this field has no redaction
    # gate and these reports reach logs.
    ref_path="attribution_ref",
    claim="ConditionAttribution.OWN_EXPERIMENT",
    reason="the ref locates a span and that span re-slices, but no recorded text says the "
    "span supports this enum -- grounding proves LOCATION, never MEANING",
)

_A_FAILURE = ReplayFinding(
    category=ReplayOutcome.FAILED,
    ref_path="claims[0].value.source_ref",
    reason="char-span re-slice mismatch",
)


class TestTheTwoAxesAreReallyTwoTypes:
    """Not an alias, not a subclass of one another.

    An alias would satisfy every exact-type check below in BOTH lists, making
    the axis guards vacuous while the whole suite stayed green -- the same
    look-alike failure the findings list was hardened against in round 85.
    """

    def test_the_two_claim_types_are_distinct(self) -> None:
        assert UncheckedStoreClaim is not UncheckedSemanticClaim

    def test_neither_claim_type_is_a_subclass_of_the_other(self) -> None:
        assert not issubclass(UncheckedStoreClaim, UncheckedSemanticClaim)
        assert not issubclass(UncheckedSemanticClaim, UncheckedStoreClaim)


class TestTheUnqualifiedNameIsGone:
    """``unchecked_claims`` must not resolve, and must not be accepted.

    Leaving it as an alias for the store list would silently re-file every
    future semantic claim under "store" at any call site that never got
    revisited -- which is precisely the defect this split exists to prevent.
    """

    def test_reading_the_unqualified_name_raises(self) -> None:
        report = _clean_report()

        with pytest.raises(AttributeError):
            _ = report.unchecked_claims  # type: ignore[attr-defined]

    def test_the_unqualified_name_is_not_accepted_by_the_constructor(self) -> None:
        with pytest.raises(TypeError):
            ReplayReport(  # type: ignore[call-arg]
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                unchecked_claims=(_A_STORE_CLAIM,),
            )

    def test_the_old_type_name_is_gone_too(self) -> None:
        import carmel.services.dataset_replay as replay

        # A surviving `UncheckedClaim` would let a call site keep building the
        # store-scoped type by its old, axis-blind name and stay compiling.
        assert not hasattr(replay, "UncheckedClaim")


class TestASemanticClaimCannotLeaveTheReportClaimingVerified:
    """The property this whole change exists to protect.

    ``overall_outcome`` is the verdict a consumer may act on WITHOUT reading a
    side list. Anything that widens "what was left untested" and is not wired
    into it recreates the original overclaim one level up.
    """

    def test_a_semantic_claim_alone_splits_the_verdict(self) -> None:
        report = _clean_report(unchecked_semantic_claims=(_A_SEMANTIC_CLAIM,))

        # The evidence really did verify, and that is worth saying separately.
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        # But the report as a whole cannot claim it.
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_a_store_claim_alone_splits_the_verdict(self) -> None:
        report = _clean_report(unchecked_store_claims=(_A_STORE_CLAIM,))

        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_a_claim_on_each_axis_still_downgrades(self) -> None:
        report = _clean_report(
            unchecked_store_claims=(_A_STORE_CLAIM,),
            unchecked_semantic_claims=(_A_SEMANTIC_CLAIM,),
        )

        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_with_neither_axis_populated_the_two_scopes_agree(self) -> None:
        report = _clean_report()

        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.VERIFIED

    def test_a_definite_disagreement_outranks_a_semantic_claim(self) -> None:
        """FAILED must survive the downgrade path on this axis too.

        UNVERIFIABLE and FAILED are never conflated in EITHER direction, and a
        second downgrade source is a second chance to soften a failure.
        """
        report = _clean_report(
            findings=(_A_FAILURE,),
            unchecked_semantic_claims=(_A_SEMANTIC_CLAIM,),
        )

        assert report.evidence_outcome is ReplayOutcome.FAILED
        assert report.overall_outcome is ReplayOutcome.FAILED


class TestTheEvidenceVerdictIsDeafToBothAxes:
    """Neither axis may make the EVIDENCE verdict look worse than the evidence.

    This is the other half of the round-74 distinction: an unchecked claim must
    not contaminate what the data checks actually concluded. Asserted per axis,
    because wiring one into ``evidence_outcome`` by mistake would still leave
    the other test green.
    """

    def test_a_store_claim_does_not_move_the_evidence_verdict(self) -> None:
        assert (
            _clean_report(unchecked_store_claims=(_A_STORE_CLAIM,)).evidence_outcome
            is ReplayOutcome.VERIFIED
        )

    def test_a_semantic_claim_does_not_move_the_evidence_verdict(self) -> None:
        assert (
            _clean_report(unchecked_semantic_claims=(_A_SEMANTIC_CLAIM,)).evidence_outcome
            is ReplayOutcome.VERIFIED
        )

    def test_neither_axis_appears_in_the_evidence_scoped_lists(self) -> None:
        report = _clean_report(
            unchecked_store_claims=(_A_STORE_CLAIM,),
            unchecked_semantic_claims=(_A_SEMANTIC_CLAIM,),
        )

        # An unchecked claim is not a finding: it is not a disagreement, and
        # its check never ran to be able to disagree.
        assert report.evidence_failures == ()
        assert report.evidence_unverifiable == ()


class TestEachAxisRefusesTheOtherAxisType:
    """Exact type, per list.

    A store claim filed under "semantic" changes no verdict -- both downgrade
    identically -- which is exactly why nothing else would ever catch it. What
    it corrupts is the report's account of WHICH axis was left untested, and
    that account is the entire product of this split.
    """

    def test_the_store_list_refuses_a_semantic_claim(self) -> None:
        with pytest.raises(ValueError, match=r"unchecked_store_claims\[0\]"):
            _clean_report(unchecked_store_claims=(_A_SEMANTIC_CLAIM,))

    def test_the_semantic_list_refuses_a_store_claim(self) -> None:
        with pytest.raises(ValueError, match=r"unchecked_semantic_claims\[0\]"):
            _clean_report(unchecked_semantic_claims=(_A_STORE_CLAIM,))

    def test_the_offending_index_is_named_not_just_the_field(self) -> None:
        # A report can carry many claims; "one of them is wrong" is not
        # actionable, and the producer that built it needs the position.
        with pytest.raises(ValueError, match=r"unchecked_semantic_claims\[2\]"):
            _clean_report(
                unchecked_semantic_claims=(
                    _A_SEMANTIC_CLAIM,
                    _A_SEMANTIC_CLAIM,
                    _A_STORE_CLAIM,
                )
            )

    def test_a_look_alike_is_refused_on_both_axes(self) -> None:
        """Duck-typing is not enough to take part in a verdict.

        Mere PRESENCE in either list downgrades ``overall_outcome``, so an
        arbitrary object here silently decides a verdict while being bound by
        none of the rules the real types enforce.
        """
        look_alike = type(
            "LooksLikeAClaim",
            (),
            {"ref_path": "x", "claim": "y", "reason": "z"},
        )()

        with pytest.raises(ValueError, match=r"unchecked_store_claims\[0\]"):
            _clean_report(unchecked_store_claims=(look_alike,))
        with pytest.raises(ValueError, match=r"unchecked_semantic_claims\[0\]"):
            _clean_report(unchecked_semantic_claims=(look_alike,))

    def test_a_subclass_is_refused_on_both_axes(self) -> None:
        """Exact type, not ``isinstance`` -- a subclass is the DANGEROUS case.

        A subclass passes every duck-typed check while carrying state no
        derivation here knows to consult, so it can satisfy the guard and still
        mean something the report cannot account for. This is a fail-closed
        path: a type nobody wrote the rules for does not get to decide a
        verdict. (Written because the mutation replacing exact-type with
        ``isinstance`` SURVIVED the first audit -- the guard was right and the
        test was missing, so the test is what changed.)
        """

        class EmbellishedStoreClaim(UncheckedStoreClaim):
            pass

        class EmbellishedSemanticClaim(UncheckedSemanticClaim):
            pass

        embellished_store = EmbellishedStoreClaim(
            ref_path=_A_STORE_CLAIM.ref_path,
            claim=_A_STORE_CLAIM.claim,
            reason=_A_STORE_CLAIM.reason,
        )
        embellished_semantic = EmbellishedSemanticClaim(
            ref_path=_A_SEMANTIC_CLAIM.ref_path,
            claim=_A_SEMANTIC_CLAIM.claim,
            reason=_A_SEMANTIC_CLAIM.reason,
        )

        with pytest.raises(ValueError, match=r"unchecked_store_claims\[0\]"):
            _clean_report(unchecked_store_claims=(embellished_store,))
        with pytest.raises(ValueError, match=r"unchecked_semantic_claims\[0\]"):
            _clean_report(unchecked_semantic_claims=(embellished_semantic,))


class TestTheFreezeReachesBothLists:
    """``frozen=True`` stops the FIELD being rebound, nothing more.

    Both outcomes are derived on every access, so there is no snapshot
    protecting a verdict already read: handed a caller's live list, the report
    would change its own verdict when that list changed (Codex round 84).
    """

    def test_a_store_claim_list_is_normalised_to_a_tuple(self) -> None:
        report = _clean_report(unchecked_store_claims=[_A_STORE_CLAIM])

        assert report.unchecked_store_claims == (_A_STORE_CLAIM,)
        assert isinstance(report.unchecked_store_claims, tuple)

    def test_a_semantic_claim_list_is_normalised_to_a_tuple(self) -> None:
        report = _clean_report(unchecked_semantic_claims=[_A_SEMANTIC_CLAIM])

        assert report.unchecked_semantic_claims == (_A_SEMANTIC_CLAIM,)
        assert isinstance(report.unchecked_semantic_claims, tuple)

    def test_mutating_the_caller_s_list_cannot_change_a_verdict(self) -> None:
        live: list[UncheckedSemanticClaim] = []
        report = _clean_report(unchecked_semantic_claims=live)
        assert report.overall_outcome is ReplayOutcome.VERIFIED

        live.append(_A_SEMANTIC_CLAIM)

        # Without the normalisation this reads UNVERIFIABLE: a verdict that
        # already went out the door would have changed underneath its reader.
        assert report.overall_outcome is ReplayOutcome.VERIFIED
        assert report.unchecked_semantic_claims == ()
