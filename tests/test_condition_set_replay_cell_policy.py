"""The PR #24 regression fix: a table-cell label/token pairing must not be double-reported.

PR #24 added :func:`_verify_cited_cell_texts` -- a gate that compares a table cell's text
against the label/token grounded on it -- but left the char-span kernel's
``uncompensated_fields`` policy hard-coded, so the kernel ALSO reported every table-cell
label/token pairing UNVERIFIABLE. An artifact grounding its claims to cells was then
UNVERIFIABLE for a reason a working gate had already handled.

The fix (:func:`carmel.services.dataset_replay._label_token_policy`) narrows the "stay
silent" flip to EXACTLY the locators that gate covers -- a ``TableCellLocator`` carrying a
present (``str``) inventory citation -- and preserves the finding for every other shape.
These tests pin both sides, and are pypdf-free (they never re-derive a real grid).
"""

from __future__ import annotations

import pytest

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    CoordinateFrame,
    SourceLocator,
    TableCellLocator,
    TextSpace,
    XPathLocator,
)
from carmel.services.dataset_replay import (
    ReplayOutcome,
    _condition_set_text_pairings,
    _dataset_text_pairings,
    _label_token_policy,
    _replay_text_pairings,
    _TextPairing,
    _verify_cited_cell_texts,
)
from tests.table_inventory_fixtures import make_embedded_inventory_with_texts

_SHA = "a" * 64
_LABEL_FIELDS = ("label_ref", "label_raw")


def _pdf_cell(sha: str = _SHA) -> TableCellLocator:
    return TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0, pdf_table_inventory_sha256=sha)


def _absent_cell() -> TableCellLocator:
    return TableCellLocator(
        table_key=CaptionLabelKey(label="Table 1"),
        row=0,
        col=0,
        pdf_table_inventory_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _bbox() -> BBoxLocator:
    frame = CoordinateFrame(
        render_fingerprint="fp",
        cropbox=("0", "0", "10", "10"),
        mediabox=("0", "0", "10", "10"),
        rotation=0,
        units="pt",
        dpi=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        render_settings=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    return BBoxLocator(bbox=BBox(frame=frame, x0="1", y0="1", x1="2", y1="2"))


def _xpath() -> XPathLocator:
    return XPathLocator(xpath="//table/tr[1]/td[1]")


def _char_span() -> CharSpanLocator:
    return CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=0, end=5)


def _is_pdf_cell(locator: SourceLocator) -> bool:
    return isinstance(locator, TableCellLocator) and isinstance(locator.pdf_table_inventory_sha256, str)


class TestLabelTokenPolicy:
    """The narrowing itself: silence ONLY for a PDF-backed table cell."""

    def test_a_pdf_backed_table_cell_is_compensated(self) -> None:
        assert _label_token_policy(_pdf_cell(), _LABEL_FIELDS) is None

    def test_a_table_cell_with_an_absent_citation_is_not_compensated(self) -> None:
        # The non-PDF cell case: _verify_cited_cell_texts continues past it, so there is no
        # gate and the kernel must keep reporting it.
        assert _label_token_policy(_absent_cell(), _LABEL_FIELDS) == _LABEL_FIELDS

    def test_a_bbox_is_not_compensated(self) -> None:
        assert _label_token_policy(_bbox(), _LABEL_FIELDS) == _LABEL_FIELDS

    def test_an_xpath_is_not_compensated(self) -> None:
        assert _label_token_policy(_xpath(), _LABEL_FIELDS) == _LABEL_FIELDS

    def test_a_char_span_is_not_compensated(self) -> None:
        # The kernel handles CharSpanLocators on its own branch, so the value is moot there,
        # but the policy is a locator-type predicate and must still return the fields.
        assert _label_token_policy(_char_span(), _LABEL_FIELDS) == _LABEL_FIELDS


def _run_kernel(pairing: _TextPairing) -> list:
    _checked, _total, findings = _replay_text_pairings(object(), iter([pairing]), {}, audit_against_source_refs=False)
    return findings


class TestKernelHonoursThePolicy:
    """The kernel emits UNVERIFIABLE iff the pairing was left uncompensated."""

    def test_a_compensated_pdf_cell_pairing_raises_no_kernel_finding(self) -> None:
        pairing = _TextPairing(
            "categorical_claims[0].label_ref",
            "n",
            _pdf_cell(),
            "Fuel",
            uncompensated_fields=_label_token_policy(_pdf_cell(), _LABEL_FIELDS),
        )
        assert _run_kernel(pairing) == []

    @pytest.mark.parametrize(
        ("locator", "why"),
        [(_absent_cell(), "absent-citation cell"), (_bbox(), "bbox"), (_xpath(), "xpath")],
        ids=["absent-cell", "bbox", "xpath"],
    )
    def test_an_uncompensated_pairing_still_reports_unverifiable(self, locator, why: str) -> None:
        pairing = _TextPairing(
            "categorical_claims[0].label_ref",
            "n",
            locator,
            "Fuel",
            uncompensated_fields=_label_token_policy(locator, _LABEL_FIELDS),
        )
        findings = _run_kernel(pairing)
        assert len(findings) == 1, why
        assert findings[0].category is ReplayOutcome.UNVERIFIABLE
        assert findings[0].ref_path == "categorical_claims[0].label_ref"


class TestProductionSitesApplyThePolicy:
    """Every label/token pairing site -- not just the ones one artifact exercises -- carries
    the policy: uncompensated_fields is None iff the ref is a PDF-backed table cell."""

    def _assert_invariant(self, pairings) -> None:
        seen_label_token = False
        for pairing in pairings:
            if not (pairing.path.endswith("label_ref") or pairing.path.endswith("token_ref")):
                continue  # value_ref/unit_ref are compensated by the boundary verifiers, not this policy
            seen_label_token = True
            expected_none = _is_pdf_cell(pairing.locator)
            assert (pairing.uncompensated_fields is None) == expected_none, pairing.path
        assert seen_label_token, "fixture exercised no label/token pairing"

    def test_condition_set_sites(self) -> None:
        from tests.test_dataset_condition_set_identity import _maximal_condition_set_envelope

        self._assert_invariant(_condition_set_text_pairings(_maximal_condition_set_envelope()))

    def test_dataset_sites(self) -> None:
        from tests.test_dataset_identity_payload import _maximal_envelope

        self._assert_invariant(_dataset_text_pairings(_maximal_envelope()))


class TestTheCompensatingGateActuallyComparesContent:
    """The fix is only sound because _verify_cited_cell_texts really compares the cell: a
    match raises no finding AND is counted as a real check (I-027), a mismatch is FAILED
    (never a silent pass)."""

    def test_a_matching_cell_raises_no_finding_and_is_counted(self) -> None:
        embedded = make_embedded_inventory_with_texts(raw_sha256="b" * 64, cell_texts={(0, 0): "Fuel"})
        pairing = _TextPairing(
            "categorical_claims[0].label_ref",
            "n",
            _pdf_cell(embedded.inventory_sha256),
            "Fuel",
            uncompensated_fields=None,
        )
        result = _verify_cited_cell_texts([pairing], {embedded.inventory_sha256: embedded})
        assert result.findings == ()
        # The match used to vanish; it is now a counted check.
        assert result.checked == 1

    def test_a_mismatched_cell_is_failed_and_not_counted(self) -> None:
        embedded = make_embedded_inventory_with_texts(raw_sha256="b" * 64, cell_texts={(0, 0): "Fuel"})
        pairing = _TextPairing(
            "categorical_claims[0].label_ref",
            "n",
            _pdf_cell(embedded.inventory_sha256),
            "Oxidizer",
            uncompensated_fields=None,
        )
        result = _verify_cited_cell_texts([pairing], {embedded.inventory_sha256: embedded})
        assert len(result.findings) == 1
        assert result.findings[0].category is ReplayOutcome.FAILED
        # A disagreement verified nothing -- it must not count as a checked cell.
        assert result.checked == 0
