# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The replay report's OUTCOME VOCABULARY, and the invariants that keep it honest.

These tests are about the report type itself, not about any particular replay.
They construct :class:`ReplayReport` directly, which is the honest unit under
test here: the defect being pinned is that the type let a caller read, or build,
a verdict that claimed more than the checks behind it established.

The specific overclaim: ``outcome`` was scoped to EVIDENCE -- raw bytes, the
addressed extraction record, grounded char spans -- but it carried the bare,
most prominent name on the report, and a report could be ``VERIFIED`` while
carrying provenance claims that were never checked at all. The consumer had to
know to also read an optional side list. The previous fix was a docstring on the
enum member confessing the problem (Codex round 74); the shape is the real fix
(Codex round 82/83).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from carmel.services.dataset_replay import (
    ReplayFinding,
    ReplayOutcome,
    ReplayReport,
    UncheckedClaim,
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


_AN_UNCHECKED_CLAIM = UncheckedClaim(
    ref_path="source_graph.node('paper').verification.root_sidecar",
    claim="no_recorded_digest",
    reason="the root meta.json could not be read, so the claim was never tested",
)

_A_FAILURE = ReplayFinding(
    category=ReplayOutcome.FAILED,
    ref_path="claims[0].value.source_ref",
    reason="char-span re-slice mismatch",
)

_AN_UNVERIFIABLE = ReplayFinding(
    category=ReplayOutcome.UNVERIFIABLE,
    ref_path="claims[0].value.source_ref",
    reason="the evidence file could not be read",
)


class TestTheBareNameIsGone:
    """``outcome`` must not resolve at all -- not to either scope.

    Keeping the name and quietly tightening what it means would let every
    existing call site keep compiling while silently changing meaning, which is
    the worse of the two migrations: a reader who never revisits the line
    inherits a verdict they did not choose. Removing it makes each site state
    the scope it meant (Codex round 83, P0 -- taken over my own preference for
    keeping the bare name on the conservative field).
    """

    def test_reading_outcome_raises_rather_than_answering(self) -> None:
        report = _clean_report()

        with pytest.raises(AttributeError):
            _ = report.outcome  # type: ignore[attr-defined]

    def test_outcome_cannot_be_passed_to_the_constructor_either(self) -> None:
        # Otherwise a caller migrating by rote could keep passing `outcome=`
        # and have it silently ignored as an unknown kwarg on some future,
        # less strict dataclass.
        with pytest.raises(TypeError):
            ReplayReport(  # type: ignore[call-arg]
                outcome=ReplayOutcome.VERIFIED,
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
            )


class TestAnUncheckedClaimCannotLeaveTheReportClaimingVerified:
    """The defect, stated as a test.

    Both halves are asserted every time. Asserting only "the overall verdict is
    not VERIFIED" would pass for the wrong reason the moment anything unrelated
    made the evidence unverifiable -- the semantic hole would go unreported
    while the test stayed green (Codex round 83, P0).
    """

    def test_clean_evidence_plus_an_unchecked_claim_splits_the_verdict(self) -> None:
        report = _clean_report(unchecked_claims=(_AN_UNCHECKED_CLAIM,))

        # The evidence really did verify. That is worth saying, and saying
        # separately -- collapsing it away would lose the distinction round 74
        # built `UncheckedClaim` to preserve.
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        # But the report as a whole cannot claim it.
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_with_no_unchecked_claims_the_two_scopes_agree(self) -> None:
        report = _clean_report()

        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.VERIFIED

    def test_a_definite_disagreement_outranks_an_unchecked_claim(self) -> None:
        """FAILED must survive the downgrade path.

        An unchecked claim degrades VERIFIED to UNVERIFIABLE, but it must never
        UPGRADE a failure into merely-unverifiable: a demonstrated
        disagreement is a harder fact than an untested claim, and
        UNVERIFIABLE and FAILED are never conflated in either direction.
        """
        report = _clean_report(
            findings=(_A_FAILURE,),
            unchecked_claims=(_AN_UNCHECKED_CLAIM,),
        )

        assert report.evidence_outcome is ReplayOutcome.FAILED
        assert report.overall_outcome is ReplayOutcome.FAILED

    def test_an_unchecked_claim_does_not_disturb_an_already_unverifiable_report(
        self,
    ) -> None:
        report = _clean_report(
            findings=(_AN_UNVERIFIABLE,),
            unchecked_claims=(_AN_UNCHECKED_CLAIM,),
        )

        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE


class TestTheEvidenceScopedListsSayEvidenceInTheirNames:
    """``.failures`` / ``.unverifiable`` filtered ``findings`` only.

    So a report whose OVERALL verdict was UNVERIFIABLE could hand back an empty
    ``.unverifiable`` -- the same "read the obvious member and overclaim" trap
    the outcome split closes, one level down (Codex round 83, P1). The names now
    carry the scope, so a caller reaching for the obvious one is told what it
    covers.
    """

    def test_the_unqualified_names_are_gone(self) -> None:
        report = _clean_report()

        with pytest.raises(AttributeError):
            _ = report.failures  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            _ = report.unverifiable  # type: ignore[attr-defined]

    def test_an_unchecked_claim_makes_the_report_unverifiable_with_no_finding(
        self,
    ) -> None:
        """The exact shape that made the old property names misleading."""
        report = _clean_report(unchecked_claims=(_AN_UNCHECKED_CLAIM,))

        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.evidence_unverifiable == ()
        assert report.evidence_failures == ()

    def test_the_scoped_lists_still_partition_the_findings(self) -> None:
        report = _clean_report(findings=(_A_FAILURE, _AN_UNVERIFIABLE))

        assert report.evidence_failures == (_A_FAILURE,)
        assert report.evidence_unverifiable == (_AN_UNVERIFIABLE,)


class TestAReportCannotBeBuiltIntoAStateNoReplayCouldProduce:
    """``frozen=True`` constrains identity, not consistency (Codex round 83, P0).

    Every combination below was constructible before, and each one is a verdict
    the report would then serve to a caller as if a real replay had produced it.
    """

    def test_zero_checked_spans_can_never_read_as_verified(self) -> None:
        """The producer appends a synthetic finding for this, but the producer
        is not the only way a report comes into being, and a rule enforced only
        by the caller that happens to remember it is not an invariant."""
        report = ReplayReport(
            checked_char_spans=0,
            total_char_spans=0,
            unchecked_char_spans=0,
        )

        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_spans_left_unchecked_can_never_read_as_verified(self) -> None:
        """A report that says some reachable spans went unchecked must not then
        call itself verified -- that is the original overclaim moved up a level.

        Today's producer always records a finding alongside an unchecked span,
        so this rule changes no real replay result (probed by applying it and
        running the replay and producer suites unchanged). It is not decoration:
        the derivation has to hold for every report, and the counts are the only
        place a partial coverage is stated.
        """
        report = ReplayReport(
            checked_char_spans=1,
            total_char_spans=3,
            unchecked_char_spans=2,
        )

        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_a_definite_failure_still_outranks_incomplete_coverage(self) -> None:
        report = ReplayReport(
            checked_char_spans=1,
            total_char_spans=3,
            unchecked_char_spans=2,
            findings=(_A_FAILURE,),
        )

        assert report.evidence_outcome is ReplayOutcome.FAILED

    def test_span_counts_that_do_not_add_up_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unchecked_char_spans"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=3,
                unchecked_char_spans=0,  # should be 2
            )

    def test_more_checked_than_exist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checked_char_spans"):
            ReplayReport(
                checked_char_spans=4,
                total_char_spans=3,
                unchecked_char_spans=-1,
            )

    def test_negative_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            ReplayReport(
                checked_char_spans=-1,
                total_char_spans=-1,
                unchecked_char_spans=0,
            )

    def test_a_finding_can_never_be_categorised_verified(self) -> None:
        """A ``VERIFIED`` finding would be invisible to the derivation.

        It matches neither the "any FAILED" nor the "any UNVERIFIABLE" test, so
        a report carrying one would derive VERIFIED while holding a finding --
        the docstring said a finding is never VERIFIED, and nothing enforced it.
        """
        with pytest.raises(ValueError, match="VERIFIED"):
            ReplayFinding(
                category=ReplayOutcome.VERIFIED,
                ref_path="claims[0]",
                reason="a clean check produces no finding at all",
            )

    def test_a_string_category_is_refused_even_though_it_compares_equal(self) -> None:
        """``ReplayOutcome`` is a ``StrEnum``, so ``"failed" == FAILED`` is True
        while ``"failed" is FAILED`` is False.

        The derivation matches on identity -- which is what an enum is for -- so
        a string category would sail past every branch of it. The report would
        then read VERIFIED while carrying a finding that says it failed, which
        is the worst version of this defect: the finding is right there, in the
        report, and the verdict does not see it.
        """
        # Bound to a name rather than written inline: comparing a literal with
        # `is` is normally a bug, and it is the very confusion under test here,
        # so stating it through a variable demonstrates it without tripping the
        # lint that exists to catch the accidental version.
        string_category = "failed"
        assert string_category == ReplayOutcome.FAILED
        assert string_category is not ReplayOutcome.FAILED

        with pytest.raises(ValueError, match="ReplayOutcome"):
            ReplayFinding(
                category=string_category,  # type: ignore[arg-type]
                ref_path="claims[0]",
                reason="a category that compares equal but is not identical",
            )


class TestTheReportValidatesWhatItIsMadeOf:
    """Normalising the container says nothing about what is in it.

    Both verdicts are derived by reading ``.category`` off each finding, so any
    object carrying that attribute takes part in the verdict. Codex round 85
    found this by probe: a ``SimpleNamespace(category="failed")`` constructed
    cleanly and the report derived VERIFIED while holding it.
    """

    def test_a_look_alike_finding_is_refused(self) -> None:
        look_alike = SimpleNamespace(
            category="failed", ref_path="claims[0]", reason="not a real finding"
        )

        with pytest.raises(ValueError, match="ReplayFinding"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                findings=(look_alike,),  # type: ignore[arg-type]
            )

    def test_an_arbitrary_object_among_the_findings_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ReplayFinding"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                findings=(object(),),  # type: ignore[arg-type]
            )

    def test_a_subclass_of_the_real_finding_is_refused_too(self) -> None:
        """The check is exact-type, not ``isinstance``, and this is what makes
        that deliberate rather than incidental.

        A subclass is the more dangerous look-alike, not the safer one: it
        passes every structural test, satisfies ``isinstance``, and can carry
        state the derivation does not know to consult while inheriting a
        ``category`` that makes it count toward the verdict. Weakening this to
        ``isinstance`` is a mutation that survived the audit until this test
        existed.
        """

        @dataclass(frozen=True)
        class EmbellishedFinding(ReplayFinding):
            severity_override: str = "ignore me"

        embellished = EmbellishedFinding(
            category=ReplayOutcome.FAILED,
            ref_path="claims[0]",
            reason="a finding carrying state nothing in the derivation reads",
        )
        assert isinstance(embellished, ReplayFinding)

        with pytest.raises(ValueError, match="ReplayFinding"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                findings=(embellished,),
            )

    def test_a_look_alike_unchecked_claim_is_refused(self) -> None:
        """Its mere presence downgrades ``overall_outcome``, so an arbitrary
        object here silently decides a verdict."""
        look_alike = SimpleNamespace(ref_path="x", claim="y", reason="z")

        with pytest.raises(ValueError, match="UncheckedClaim"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                unchecked_claims=(look_alike,),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "counts",
        [
            pytest.param((True, True, False), id="bools"),
            pytest.param((1.0, 1.0, 0.0), id="floats"),
        ],
    )
    def test_counts_that_are_not_really_ints_are_refused(
        self, counts: tuple[object, object, object]
    ) -> None:
        """``bool`` subclasses ``int``, so ``True`` satisfies every bound and
        reads as a count of 1; a float satisfies them too and can make the
        arithmetic agree by coincidence. Both derived VERIFIED before this."""
        checked, total, unchecked = counts

        with pytest.raises(ValueError, match="must be an int"):
            ReplayReport(
                checked_char_spans=checked,  # type: ignore[arg-type]
                total_char_spans=total,  # type: ignore[arg-type]
                unchecked_char_spans=unchecked,  # type: ignore[arg-type]
            )


class TestTheFreezeReachesTheContents:
    """``frozen=True`` stops the field being rebound, not the object it names.

    Both outcomes are derived on every access, so there is no snapshot standing
    between a caller's live list and a verdict that has already been read
    (Codex round 84).
    """

    def test_a_list_of_findings_cannot_be_mutated_after_the_fact(self) -> None:
        findings = [_AN_UNVERIFIABLE]
        report = ReplayReport(
            checked_char_spans=3,
            total_char_spans=3,
            unchecked_char_spans=0,
            findings=findings,  # type: ignore[arg-type]
        )
        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE

        findings.clear()

        # Had the report kept the caller's list, clearing it would have turned
        # an unverifiable report into a verified one after the fact.
        assert report.findings == (_AN_UNVERIFIABLE,)
        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE

    def test_a_list_of_unchecked_claims_cannot_be_mutated_after_the_fact(self) -> None:
        claims = [_AN_UNCHECKED_CLAIM]
        report = ReplayReport(
            checked_char_spans=3,
            total_char_spans=3,
            unchecked_char_spans=0,
            unchecked_claims=claims,  # type: ignore[arg-type]
        )
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

        claims.clear()

        assert report.unchecked_claims == (_AN_UNCHECKED_CLAIM,)
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
