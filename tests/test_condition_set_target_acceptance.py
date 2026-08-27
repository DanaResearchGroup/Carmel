"""End-to-end acceptance: this project's FIRST stored ConditionSetEnvelope, built from
THE real target table -- Table 1, page 4 of ``10.1016-j.ijhydene.2013.10.164`` -- grounded
cell-by-cell to an embedded inventory re-derived from the raw PDF bytes, then stored,
loaded and replayed.

Corpus-gated exactly like :mod:`tests.test_target_table_acceptance`: the paper is
non-redistributable, so it is read from the operator's corpus store at runtime and every
test SKIPS -- never passes -- when the document (or its content-addressed store) is absent
or is not byte-for-byte the measured document.

**What this module asserts is the TRUE replay outcome of the honest artifact, not a target
outcome driven green.** Two facts a reader must hold:

* ``overall_outcome`` is UNVERIFIABLE and always will be for any condition set: the
  attribution char span is support-only -- nothing recorded says what
  ``ConditionAttribution.OWN_EXPERIMENT`` MEANS, so it carries the one remaining
  UncheckedSemanticClaim. The four range statements no longer do: each now stores its own
  words verbatim in ``UnextractedConditionStatement.statement_raw`` (``"0.6-1.0"`` and the
  rest), so replay COMPARES that text whole-cell against the re-derived grid rather than
  filing it as an unchecked meaning. The refusal that stays is the parse -- nothing here says
  what ``0.6-1.0`` decomposes into -- not the words, which are now recorded and checked.
* ``evidence_outcome`` is VERIFIED -- and NOT hollow. The subject names the paper's apparatus
  ("heat flux method"), so one real char span is re-sliced; all 26 table-cell claim refs --
  the 9 categorical label/token pairs AND all four unextracted statements' label + statement
  cells -- are content-compared against the re-derived grid; and the embedded inventory
  reproduces against the raw bytes. The PR #24 false positive that used to report every cell
  UNVERIFIABLE is fixed narrowly (see :mod:`tests.test_condition_set_replay_cell_policy`).

The 25 degree-C temperature row is deliberately NOT stored: the unit is absent from both
lanes (zero U+00B0 in the extracted text, and no cell equals "degC"), so it cannot ground a
scalar's required ``unit_quote``; and no ``UnextractedReason`` honestly classifies a clean
single value, so it is not laundered into a range either. Its two "25" cells stay in the
inventory grid, uncited -- visible, not smuggled.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from carmel.paths import default_workspaces_root
from carmel.schemas.datasets import (
    CaptionLabelKey,
    CharSpanLocator,
    ConditionAttribution,
    DeviceClassDeclaration,
    EmbeddedTableInventory,
    TableCellLocator,
    UnextractedReason,
)
from carmel.services import units
from carmel.services.condition_set_bridge import (
    load_condition_set_envelope,
    store_condition_set_envelope,
)
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    ConditionSetProducerError,
    DeviceClassSpec,
    ScalarConditionSpec,
    TableCellGrounding,
    UnextractedConditionSpec,
    produce_condition_set_from_artifact,
)
from carmel.services.dataset_producer import (
    TextLaneMisdecodeError,
    _prepare_grounding,
    ground_quote,
)
from carmel.services.dataset_replay import (
    ReplayOutcome,
    SemanticGap,
    replay_stored_condition_set,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.numeric import QuoteRole
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import (
    InventoryVerificationStatus,
    inventory_record_payload,
    verify_inventory_record,
)
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory
from tests.pypdf_gate import require_pypdf

_DOCUMENT_SHA256 = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"
_WORKSPACE_SUBPATH = "live-syngas"
_INBOX_SUBPATH = "literature_requests/inbox/10.1016-j.ijhydene.2013.10.164.pdf"
_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")

#: The registered whole-table footprint -- this project's own box claim, identical to the
#: one :mod:`tests.test_target_table_acceptance` measured the 9x3/20-cell result under.
WHOLE_TABLE = ClaimedFootprint(
    page=4,
    x_start=50.0,
    x_end=290.0,
    y_top=145.0,
    y_bottom=45.0,
    caption_text="Table1–Measurementconditions.",
    caption_x_start=53.0,
    caption_baseline_y=148.8,
)

_TABLE_KEY = CaptionLabelKey(label="Table 1")


def _locate_workspace() -> Path | None:
    """The workspace whose content-addressed store holds the target document, or None."""
    for root in _WORKSPACES_ROOTS:
        workspace = root / _WORKSPACE_SUBPATH
        if (workspace / "evidence" / "literature" / _DOCUMENT_SHA256 / "raw.bin").exists():
            return workspace
    return None


def _staged_workspace(tmp_path: Path) -> tuple[Path, bytes]:
    """Copy the target document's literature store into a writable tmp workspace.

    The producer reads the content-addressed store; the store writes the condition-set
    envelope back into the SAME root; replay reads ``raw.bin`` from it. Copying the one
    literature subtree keeps every step self-contained and never touches the operator's
    real workspace.
    """
    require_pypdf()
    source = _locate_workspace()
    if source is None:
        roots = ", ".join(str(r / _WORKSPACE_SUBPATH) for r in _WORKSPACES_ROOTS)
        pytest.skip(f"target corpus store is not present under any of: {roots}")
    src_dir = source / "evidence" / "literature" / _DOCUMENT_SHA256
    raw = (src_dir / "raw.bin").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _DOCUMENT_SHA256:
        pytest.skip(f"stored raw.bin is {actual}, not the measured {_DOCUMENT_SHA256}")
    dest_dir = tmp_path / "evidence" / "literature" / _DOCUMENT_SHA256
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir)
    return tmp_path, raw


def _embedded_inventory(raw: bytes) -> tuple[EmbeddedTableInventory, dict]:
    inventory = build_inventory(extract_fragments(raw), WHOLE_TABLE)
    assert inventory.refusals == (), f"the target grid refused: {inventory.refusals}"
    payload = inventory_record_payload(inventory, raw_sha256=_DOCUMENT_SHA256)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    embedded = EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=_DOCUMENT_SHA256,
        canonical_json=canonical,
    )
    return embedded, payload


#: (claim_id, label, token, label_row, label_col, token_row, token_col) for the 9 genuine
#: categorical cells: Fuel across both columns, then every Oxidizer cell.
_CATEGORICALS = (
    ("cat0_fuel_c1", "Fuel", "H2/CO(50:50%)", 0, 0, 0, 1),
    ("cat1_fuel_c2", "Fuel", "H2/CO(85:15%)", 0, 0, 0, 2),
    ("cat2_oxidizer_r1c1", "Oxidizer", "Air", 1, 0, 1, 1),
    ("cat3_oxidizer_r2c1", "Oxidizer", "O2/N2(15:85%)", 1, 0, 2, 1),
    ("cat4_oxidizer_r2c2", "Oxidizer", "O2/N2(15:85%)", 1, 0, 2, 2),
    ("cat5_oxidizer_r3c1", "Oxidizer", "O2/N2(10:90%)", 1, 0, 3, 1),
    ("cat6_oxidizer_r3c2", "Oxidizer", "O2/He(12:88%)", 1, 0, 3, 2),
    ("cat7_oxidizer_r4c1", "Oxidizer", "O2/He(10:90%)", 1, 0, 4, 1),
    ("cat8_oxidizer_r5c1", "Oxidizer", "O2/He(12.5:87.5%)", 1, 0, 5, 1),
)

#: (statement_id, label, statement, quantity_kind, label_row, label_col, stmt_row, stmt_col)
#: for the four ranges the extractor REFUSES to reduce to a single value.
_RANGES = (
    ("unx0_phi_c1", "φ", "0.6–1.0", units.QuantityKind.EQUIVALENCE_RATIO, 6, 0, 6, 1),
    ("unx1_phi_c2", "φ", "0.5–0.7", units.QuantityKind.EQUIVALENCE_RATIO, 6, 0, 6, 2),
    ("unx2_pressure_c1", "P(atm)", "1–9", units.QuantityKind.PRESSURE, 8, 0, 8, 1),
    ("unx3_pressure_c2", "P(atm)", "1–8", units.QuantityKind.PRESSURE, 8, 0, 8, 2),
)

#: The apparatus is named unambiguously and repeatedly ("heat flux method (HFM)"); occurrence
#: 3 is "...using the heat flux method at elevated pressure", the authors' own measurements.
#: The 'fl' is the ligature U+FB02, verbatim as the extraction yields it.
_SUBJECT_QUOTE = "heat ﬂux method"
_SUBJECT_OCCURRENCE = 3
#: "were conducted" appears exactly once -- "Experiments were conducted at elevated pressure".
_ATTRIBUTION_QUOTE = "were conducted"


def _produce(workspace: Path, embedded: EmbeddedTableInventory):
    def cell(row: int, col: int) -> TableCellGrounding:
        return TableCellGrounding(table_key=_TABLE_KEY, row=row, col=col, inventory=embedded)

    categoricals = tuple(
        CategoricalConditionSpec(
            claim_id=cid,
            label_quote=label,
            token_quote=token,
            label_cell=cell(lr, lc),
            token_cell=cell(tr, tc),
        )
        for cid, label, token, lr, lc, tr, tc in _CATEGORICALS
    )
    unextracted = tuple(
        UnextractedConditionSpec(
            statement_id=sid,
            label_quote=label,
            statement_quote=statement,
            reason=UnextractedReason.VALUE_RANGE,
            quantity_kind=qk,
            label_cell=cell(lr, lc),
            statement_cell=cell(sr, sc),
        )
        for sid, label, statement, qk, lr, lc, sr, sc in _RANGES
    )
    return produce_condition_set_from_artifact(
        workspace,
        sha256=_DOCUMENT_SHA256,
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_quote=_ATTRIBUTION_QUOTE,
        subject=DeviceClassSpec(label_quote=_SUBJECT_QUOTE, label_occurrence=_SUBJECT_OCCURRENCE),
        categoricals=categoricals,
        unextracted=unextracted,
    )


class TestTheFirstStoredConditionSet:
    """Clauses 1-4 of the ticket, on the real document, end to end."""

    def test_it_stores_and_loads_back_identically(self, tmp_path: Path) -> None:
        """Clause 1: a ConditionSetEnvelope for this document exists under the store and
        round-trips byte-for-byte."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)
        env = _produce(workspace, embedded)
        stored = store_condition_set_envelope(workspace, env)
        assert (workspace / "evidence" / "condition_sets" / f"{stored.sha256}.json").exists()
        assert load_condition_set_envelope(workspace, stored.sha256) == env

    def test_every_claim_is_grounded_to_a_table_cell(self, tmp_path: Path) -> None:
        """Clause 2: each scalar/categorical claim's and each unextracted statement's
        label/value refs are TableCellLocators into the embedded inventory; the ONLY char
        spans are attribution and the subject label."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)
        env = _produce(workspace, embedded)

        claim_locators = []
        for claim in env.categorical_claims:
            claim_locators += [claim.label_ref.locator, claim.token_ref.locator]
        for statement in env.unextracted:
            claim_locators += [statement.label_ref.locator, statement.statement_ref.locator]
        assert len(claim_locators) == 26
        for locator in claim_locators:
            assert isinstance(locator, TableCellLocator)
            assert locator.pdf_table_inventory_sha256 == embedded.inventory_sha256

        # The two deliberate exceptions, and nothing else, are char spans.
        assert isinstance(env.attribution_ref.locator, CharSpanLocator)
        assert isinstance(env.subject, DeviceClassDeclaration)
        assert isinstance(env.subject.label_ref.locator, CharSpanLocator)

    def test_the_four_ranges_are_unextracted_value_ranges(self, tmp_path: Path) -> None:
        """Clause 3: the phi and pressure ranges are unextracted with reason VALUE_RANGE,
        each carrying its own label_ref and statement_ref cell."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)
        env = _produce(workspace, embedded)

        assert len(env.unextracted) == 4
        assert env.scalar_claims == ()  # temperature is not laundered into a scalar
        assert {s.label_raw for s in env.unextracted} == {"φ", "P(atm)"}

        # The statement's OWN words are now on the record, verbatim, in statement_raw --
        # read the four ranges straight off the record, not reconstructed through the
        # embedded grid. (statement_ref still points at the same cell, and replay compares
        # statement_raw whole-cell against that grid; the equality is asserted below and by
        # the replay-outcome test, but the RANGE itself no longer requires a grid lookup to
        # recover -- that was the amnesia this record used to carry.)
        from_record = {statement.statement_raw for statement in env.unextracted}
        assert from_record == {"0.6–1.0", "0.5–0.7", "1–9", "1–8"}

        for statement in env.unextracted:
            assert statement.reason is UnextractedReason.VALUE_RANGE
            assert isinstance(statement.label_ref.locator, TableCellLocator)
            locator = statement.statement_ref.locator
            assert isinstance(locator, TableCellLocator)
            # The recorded words ARE the cited cell's text, exactly and whole -- which is
            # what lets replay compare them without the whole-cell rule laundering a fragment.
            assert statement.statement_raw == embedded.cell_text(row=locator.row, col=locator.col)

    def test_the_embedded_inventory_reproduces_against_the_raw_bytes(self, tmp_path: Path) -> None:
        """Clause 4 (property asserted): ``verify_inventory_record`` re-derives the embedded
        grid from the raw document bytes and reports REPRODUCED -- and tampering with a
        single cell turns that into MISMATCHED. This is the honest 'store/evidence' claim:
        replay cannot prove the BOX is right (it is a hand-drawn footprint), but it proves
        this box over these bytes still yields exactly this grid."""
        _, raw = _staged_workspace(tmp_path)
        _, payload = _embedded_inventory(raw)

        assert verify_inventory_record(payload, raw).status is InventoryVerificationStatus.REPRODUCED

        tampered = json.loads(json.dumps(payload))
        flipped = False
        for cell in tampered["cells"]:
            if cell.get("text") == "Air":
                cell["text"] = "AIR-TAMPERED"
                flipped = True
        assert flipped, "the 'Air' cell moved; the tampering premise is stale"
        assert verify_inventory_record(tampered, raw).status is InventoryVerificationStatus.MISMATCHED

    def test_replay_is_honest_verified_evidence_under_unverifiable_overall(self, tmp_path: Path) -> None:
        """Clause 4 (replay outcomes, asserted as they truly are):

        * ``evidence_outcome`` is VERIFIED: no finding names any claim's ref (the PR #24
          false positive is fixed), no cell text disagrees, the grid reproduces, and one
          real char span (the subject label) is re-sliced -- so it is not the hollow
          "checked nothing" VERIFIED the brief warned about.
        * ``overall_outcome`` is UNVERIFIABLE: the headline verdict, downgraded by the ONE
          unchecked semantic claim below.
        * ``unchecked_semantic_claims`` is EXACTLY the attribution span -- and nothing else.
          The four range statements USED to appear here (each an UnextractedConditionStatement
          with no stored words); now that ``statement_raw`` records their text, replay compares
          it whole-cell and files no semantic gap for them. No entry traces back to a
          scalar/categorical claim's own ref either, which is what would mean a table cell went
          uncompared.
        * Per-statement consistency (Verifier 4): each unextracted statement is COMPARED (its
          statement_ref cell is checked) and is NOT also filed as an unchecked claim -- exactly
          one of the two, asserted directly below.
        * All 26 table cells are compared and matched (up from 22: the four statement cells now
          count), while the char-span count is unchanged.
        * ``attempted_refutations`` is empty: there is no scalar claim (the temperature
          could not be grounded), so no table-grounded StitchGate refutation arises.
        """
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)
        env = _produce(workspace, embedded)
        stored = store_condition_set_envelope(workspace, env)

        report = replay_stored_condition_set(workspace, stored.sha256)

        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.findings == ()
        assert report.evidence_failures == ()

        gaps = {(claim.claim_path, claim.gap) for claim in report.unchecked_semantic_claims}
        assert gaps == {
            ("attribution", SemanticGap.SUPPORT_UNRECORDED),
        }

        # Verifier 4, asserted directly: for every unextracted statement, exactly one of
        # "compared" (its statement_ref cell is in the checked count) or "filed as an unchecked
        # claim" holds -- here every statement is compared and none is filed.
        semantic_statement_paths = {
            claim.claim_path
            for claim in report.unchecked_semantic_claims
            if claim.claim_path.startswith("unextracted[")
        }
        assert semantic_statement_paths == set()

        # The four statement cells are among the 26 compared-and-matched (was 22 before the
        # words were recorded), and the single real char span is still re-sliced.
        assert report.checked_table_cells == 26
        assert report.checked_char_spans == 1
        assert report.attempted_refutations == ()

    def test_the_temperature_cells_survive_in_the_grid_but_are_uncited(self, tmp_path: Path) -> None:
        """The temperature row is present in the inventory (r7c0 T(degC), r7c1/c2 = 25) yet
        no claim or statement cites row 7 -- the omission is visible, not silent."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)
        env = _produce(workspace, embedded)

        assert embedded.cell_text(row=7, col=1) == "25"
        assert embedded.cell_text(row=7, col=2) == "25"

        cited_rows = set()
        for claim in env.categorical_claims:
            for ref in (claim.label_ref, claim.token_ref):
                assert isinstance(ref.locator, TableCellLocator)
                cited_rows.add(ref.locator.row)
        for statement in env.unextracted:
            for ref in (statement.label_ref, statement.statement_ref):
                assert isinstance(ref.locator, TableCellLocator)
                cited_rows.add(ref.locator.row)
        assert 7 not in cited_rows


class TestTheTemperatureCannotBeStoredAsAScalar:
    """The temperature finding, made executable: the producer REFUSES every route to a
    grounded temperature scalar, because its unit deg-C is absent from both lanes. This is
    the guard the omission rests on, watched firing -- not a claim taken on trust."""

    def test_the_degree_sign_is_absent_from_the_extracted_text(self, tmp_path: Path) -> None:
        workspace, _ = _staged_workspace(tmp_path)
        text = _prepare_grounding(
            workspace, _DOCUMENT_SHA256, envelope_noun="condition set", envelope_subject="A condition set"
        ).text
        assert text.count("°") == 0

    def test_grounding_the_unit_at_the_header_cell_is_refused(self, tmp_path: Path) -> None:
        """The unit deg-C lives ONLY inside the header cell 'T(degC)', which is also the
        label. Citing that one cell for both the label and the unit is refused: a
        TableCellLocator has no sub-cell addressing, so one cell cannot honestly ground two
        different strings."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)

        def cell(row: int, col: int) -> TableCellGrounding:
            return TableCellGrounding(table_key=_TABLE_KEY, row=row, col=col, inventory=embedded)

        scalar = ScalarConditionSpec(
            claim_id="temp",
            label_quote="T(°C)",
            quantity_kind=units.QuantityKind.TEMPERATURE,
            value_quote="25",
            unit_quote="°C",
            label_cell=cell(7, 0),
            value_cell=cell(7, 1),
            unit_cell=cell(7, 0),  # the only cell carrying deg-C at all -- but it reads "T(°C)"
        )
        with pytest.raises(ConditionSetProducerError, match="cannot honestly be both"):
            produce_condition_set_from_artifact(
                workspace,
                sha256=_DOCUMENT_SHA256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote=_ATTRIBUTION_QUOTE,
                subject=DeviceClassSpec(label_quote=_SUBJECT_QUOTE, label_occurrence=_SUBJECT_OCCURRENCE),
                scalars=(scalar,),
            )

    def test_grounding_the_unit_in_running_text_is_refused(self, tmp_path: Path) -> None:
        """The other route: search deg-C in the running text. It is not there (zero U+00B0) --
        and the refusal now NAMES why (ticket I-019). The degree sign is a KNOWN mis-decode the
        table lane repairs (glyph /C14 -> '°') and the text lane never does, so grounding raises
        a ``TextLaneMisdecodeError`` whose reason names the glyph, the repaired character and the
        table lane -- not a bare 'not found' that hides that this document never carried a '°'
        in its text lane at all. This is the closed half of the lane divergence, on the real
        document."""
        workspace, raw = _staged_workspace(tmp_path)
        embedded, _ = _embedded_inventory(raw)

        def cell(row: int, col: int) -> TableCellGrounding:
            return TableCellGrounding(table_key=_TABLE_KEY, row=row, col=col, inventory=embedded)

        scalar = ScalarConditionSpec(
            claim_id="temp",
            label_quote="T(°C)",
            quantity_kind=units.QuantityKind.TEMPERATURE,
            value_quote="25",
            unit_quote="°C",  # char-span grounded (no unit_cell) -- absent from the text
            label_cell=cell(7, 0),
            value_cell=cell(7, 1),
        )
        with pytest.raises(TextLaneMisdecodeError) as excinfo:
            produce_condition_set_from_artifact(
                workspace,
                sha256=_DOCUMENT_SHA256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote=_ATTRIBUTION_QUOTE,
                subject=DeviceClassSpec(label_quote=_SUBJECT_QUOTE, label_occurrence=_SUBJECT_OCCURRENCE),
                scalars=(scalar,),
            )
        message = str(excinfo.value)
        assert "°" in message
        assert "/C14" in message
        assert "table" in message.lower()
        assert excinfo.value.repair.replacement == "°"

    def test_grounding_a_real_en_dash_from_the_caption_is_refused_by_name(self, tmp_path: Path) -> None:
        """The en-dash direction, on the real document, exercised at the seam directly. The
        caption prints 'Table 1 – Measurement conditions', but the text lane stores the en-dash
        mis-decoded to the letter 'e' ('Table 1 e Measurement conditions'). A consumer quoting
        the caption AS PRINTED (with the real en-dash) finds nothing -- and the refusal, carrying
        the document's in-force repairs from ``_prepare_grounding``, names the en-dash mis-decode
        rather than implying the caption was fabricated."""
        workspace, _ = _staged_workspace(tmp_path)
        grounding = _prepare_grounding(
            workspace, _DOCUMENT_SHA256, envelope_noun="condition set", envelope_subject="A condition set"
        )
        # The seam populated the context with the table lane's DOCUMENT-scoped repairs.
        assert "–" in {r.replacement for r in grounding.glyph_repairs}
        assert grounding.text.count("–") == 0  # the text lane never carries the repaired en-dash
        with pytest.raises(TextLaneMisdecodeError) as excinfo:
            ground_quote(
                grounding.text,
                "Table 1 – Measurement conditions",
                role=QuoteRole.LABEL,
                repairs=grounding.glyph_repairs,
            )
        assert excinfo.value.repair.replacement == "–"
        assert "–" in str(excinfo.value)
