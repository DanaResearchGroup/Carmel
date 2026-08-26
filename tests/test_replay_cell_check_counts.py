# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The widened zero-check rule (I-027): a matched table cell is a real check.

A re-sliced character span used to be the only check the replay outcome
machinery knew about, so ``checked_char_spans == 0`` forced UNVERIFIABLE. A
table-cell text comparison that ran and MATCHED is now also a real check,
counted as :attr:`ReplayReport.checked_table_cells` -- a quantity DELIBERATELY
separate from the char-span counts and NOT folded into ``total_char_spans`` or
its arithmetic. So "established nothing" and "re-sliced zero character spans"
stopped being the same sentence: an artifact grounded entirely at cells, every
cell matching, must read VERIFIED though it re-sliced no span -- while one that
checked NOTHING by any means must still read UNVERIFIABLE.

These construct :class:`ReplayReport` DIRECTLY, on purpose. The defect this
ticket fixes lives in the report's own ``evidence_outcome`` derivation -- a
public, independently-reached site, not only the producers that append a
finding. A test that only went through a producer would not catch a derivation
still asking the old question, which is exactly why at least one test must
build the report by hand.

These are kept OUT of ``tests/test_replay_outcome_scope.py`` so that file's diff
stays empty: its two named tests
(``test_zero_checked_spans_can_never_read_as_verified`` and
``test_spans_left_unchecked_can_never_read_as_verified``) pin the rule being
widened and must pass unmodified.
"""

from __future__ import annotations

import pytest

from carmel.services.dataset_replay import ReplayFinding, ReplayOutcome, ReplayReport

_A_FAILURE = ReplayFinding(
    category=ReplayOutcome.FAILED,
    ref_path="claims[0].value.source_ref",
    reason="char-span re-slice mismatch",
)


class TestAMatchedTableCellIsARealCheck:
    def test_an_artifact_grounded_entirely_at_cells_reads_verified(self) -> None:
        # Zero char spans re-sliced, but 22 cell comparisons ran and matched --
        # the exact shape of this codebase's first stored artifact's grounding.
        report = ReplayReport(
            checked_char_spans=0,
            total_char_spans=0,
            unchecked_char_spans=0,
            checked_table_cells=22,
        )

        # No finding at all, and the widened rule no longer fires the
        # zero-span UNVERIFIABLE for it.
        assert report.findings == ()
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.VERIFIED

    def test_no_check_of_any_kind_still_reads_unverifiable(self) -> None:
        """The guard against the fix becoming a silencing: no char span
        re-sliced AND no cell compared is still nothing established. This is the
        rule ``test_zero_checked_spans_can_never_read_as_verified`` pins, now
        stated with the new counter explicit at zero."""
        report = ReplayReport(
            checked_char_spans=0,
            total_char_spans=0,
            unchecked_char_spans=0,
            checked_table_cells=0,
        )

        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE

    def test_a_single_matched_cell_is_enough_to_leave_the_zero_check_floor(self) -> None:
        # One matched cell, nothing else -- the boundary the rule turns on.
        report = ReplayReport(
            checked_char_spans=0,
            total_char_spans=0,
            unchecked_char_spans=0,
            checked_table_cells=1,
        )

        assert report.evidence_outcome is ReplayOutcome.VERIFIED

    def test_a_matched_cell_does_not_rescue_a_disagreement(self) -> None:
        # Cells matching never outranks a demonstrated FAILED finding.
        report = ReplayReport(
            checked_char_spans=1,
            total_char_spans=1,
            unchecked_char_spans=0,
            checked_table_cells=5,
            findings=(_A_FAILURE,),
        )

        assert report.evidence_outcome is ReplayOutcome.FAILED

    def test_a_matched_cell_is_not_folded_into_the_char_span_arithmetic(self) -> None:
        # total_char_spans stays a pure char-span count; a matched cell neither
        # inflates it nor is charged against unchecked_char_spans.
        report = ReplayReport(
            checked_char_spans=1,
            total_char_spans=1,
            unchecked_char_spans=0,
            checked_table_cells=22,
        )

        assert report.total_char_spans == 1
        assert report.unchecked_char_spans == 0
        assert report.checked_table_cells == 22

    def test_a_negative_cell_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checked_table_cells"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                checked_table_cells=-1,
            )

    def test_a_cell_count_that_is_not_really_an_int_is_refused(self) -> None:
        # bool is an int subclass; refused on exact type, like the char counts.
        with pytest.raises(ValueError, match="checked_table_cells"):
            ReplayReport(
                checked_char_spans=1,
                total_char_spans=1,
                unchecked_char_spans=0,
                checked_table_cells=True,
            )
