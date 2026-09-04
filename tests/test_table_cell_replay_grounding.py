"""I-021: a table-cell citation is checked for CONTENT and RE-DERIVED, in replay.

Two independent gaps this module demonstrates against BEHAVIOUR (never a pinned
sha or golden byte -- another change is in flight that moves the inventory
payload's shape, so these build inventories at runtime and assert on outcomes):

1. **Content, not just address.** A value whose ``raw_text`` does not match the
   text of the table cell it cites replays ``FAILED``, and the finding names the
   row, the column and both disagreeing strings. Existence of the ``(row, col)``
   was always checked; that it SAYS the value's text was not, and a citation
   that resolves to a real cell of the right document could still name a cell
   holding something else entirely.

2. **Re-derivation.** A corrupted embedded inventory replays ``FAILED`` because
   :func:`~carmel.services.pdf_table_record.verify_inventory_record` re-derives
   the grid from the raw PDF bytes and reports ``MISMATCHED`` -- for BOTH a
   :class:`~carmel.schemas.datasets.DatasetEnvelope` and a
   :class:`~carmel.schemas.datasets.ConditionSetEnvelope`, because a condition
   set has no series and a fix scoped to the series lane would leave it exactly
   as unchecked as before.

The inventories here are REAL: built by running the production fragment lane
over a minimal PDF assembled at runtime, so ``verify_inventory_record`` can
genuinely reproduce them (or catch a corruption as a mismatch) rather than
report ``EXTRACTION_FAILED`` over bytes a healthy engine cannot walk. No paper
text enters the repo.
"""

from __future__ import annotations

import hashlib
import zlib
from datetime import UTC, datetime
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    CaptionLabelKey,
    ConditionAttribution,
    ConditionSetEnvelope,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedTableInventory,
    GroundedScalarClaim,
    MeasuredValue,
    Observation,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    TableKeyKind,
    UnresolvedSubject,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_producer import _ACTIVE
from carmel.services.dataset_replay import (
    ReplayOutcome,
    replay_condition_set,
    replay_envelope,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import store_artifact
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import ClaimedFootprint, inventory_record_payload
from carmel.services.pdf_tables import build_inventory
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import QuantityKind
from tests.pypdf_gate import require_pypdf

_MAX_BYTES = 10_000_000


def _pdf(content: str) -> bytes:
    """A minimal one-page PDF whose content stream is ``content``.

    The same hand-built shape ``tests.test_pdf_table_record`` uses: no paper is
    involved, and the geometry the grid is derived from is visible in the test."""
    stream = zlib.compress(content.encode("ascii"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    return bytes(out)


#: A caption over a three-column, two-row table: a header row of labels and a
#: body row carrying a value, its unit, and a second value. A MeasuredValue can
#: cite a real value cell, a real unit cell and a real label cell -- all as
#: table cells, needing no char-span grounding (hence no extraction record) at
#: all. The spare header cell (0,1) is cited by nothing, so a re-derivation test
#: can corrupt it without also tripping the content check.
_TABLE_PDF = _pdf(
    "BT /F1 9 Tf 53 700 Td (Table 2 - readings) Tj ET\n"
    "BT /F1 9 Tf 53 686 Td (phi) Tj ET\n"
    "BT /F1 9 Tf 123 686 Td (P) Tj ET\n"
    "BT /F1 9 Tf 193 686 Td (X) Tj ET\n"
    "BT /F1 9 Tf 53 672 Td (0.6) Tj ET\n"
    "BT /F1 9 Tf 123 672 Td (atm) Tj ET\n"
    "BT /F1 9 Tf 193 672 Td (0.5) Tj ET\n"
)

_FOOTPRINT = ClaimedFootprint(
    page=1,
    x_start=50.0,
    x_end=250.0,
    y_top=692.0,
    y_bottom=665.0,
    caption_text="Table 2 - readings",
    caption_x_start=53.0,
    caption_baseline_y=700.0,
)

#: The grid ``build_inventory`` derives from ``_TABLE_PDF`` -- pinned as a
#: PRECONDITION the tests assert on, not as a golden: (0,0)=phi (0,1)=P (0,2)=X
#: (1,0)=0.6 (1,1)=atm (1,2)=0.5.
_RAW_SHA = hashlib.sha256(_TABLE_PDF).hexdigest()

_NODE_ID = "paper"


def _store_pdf_node(tmp_path: Path) -> SourceNode:
    """Store ``_TABLE_PDF`` as a node's ``raw.bin`` and return a PAPER_PDF node
    over it.

    ``extraction`` is ``Absent``: every ref these tests build is a table cell,
    so no extracted text is ever re-sliced and no extraction record is needed --
    which keeps the fixture to the one thing under test. The node's bytes still
    verify (raw.bin present and hashing to ``sha256``), so replay reaches the
    inventory re-derivation with real, hash-verified bytes in hand."""
    require_pypdf()
    artifact = FetchedArtifact(
        url="https://example.invalid/table.pdf",
        final_url="https://example.invalid/table.pdf",
        sha256=_RAW_SHA,
        content_type="application/pdf",
        n_bytes=len(_TABLE_PDF),
        fetched_at=datetime.now(UTC),
    )
    # An ExtractedText is required by store_artifact, but nothing here re-reads
    # it: the node declares extraction Absent, so replay never resolves a record.
    extracted = ExtractedText(text="", normalized="", sections=[], extractor="pdf:pypdf", lossy=False)
    store_artifact(tmp_path, data=_TABLE_PDF, artifact=artifact, extracted=extracted, max_bytes=_MAX_BYTES)
    return SourceNode(
        node_id=_NODE_ID,
        kind=SourceNodeKind.PAPER_PDF,
        sha256=_RAW_SHA,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _real_inventory(
    *, corrupt: tuple[int, int] | None = None, drop_text: tuple[int, int] | None = None
) -> EmbeddedTableInventory:
    """The real inventory ``build_inventory`` derives from ``_TABLE_PDF``.

    With ``corrupt=(row, col)`` the named cell's stored text is mangled and the
    record RE-ADDRESSED so it stays self-coherent (T1 objects only to the
    perturbation): replay then re-derives the true grid from the bytes and finds
    the stored one does not reproduce -- a ``MISMATCHED``. Uncorrupted, it
    reproduces exactly.

    With ``drop_text=(row, col)`` the named cell keeps its coordinate but loses
    its ``"text"`` key entirely -- a schema-valid shape (T1 requires only that
    every cell's ``row``/``col`` be unique ordinals; ``"text"`` is optional), so
    the cell is still PRESENT (``has_cell`` True) yet :meth:`cell_text` answers
    ``None`` for it. This is the "present but carries no string" case the
    consumer must surface as an INABILITY to compare, distinct from a match and
    from a mismatch against ``""``."""
    require_pypdf()
    inventory = build_inventory(extract_fragments(_TABLE_PDF), _FOOTPRINT)
    payload = inventory_record_payload(inventory, raw_sha256=_RAW_SHA)
    if corrupt is not None:
        for cell in payload["cells"]:
            if (cell["row"], cell["col"]) == corrupt:
                cell["text"] = cell["text"] + "_CORRUPT"
    if drop_text is not None:
        for cell in payload["cells"]:
            if (cell["row"], cell["col"]) == drop_text:
                del cell["text"]
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=_RAW_SHA,
        canonical_json=canonical,
    )


def _cell_ref(row: int, col: int, inventory: EmbeddedTableInventory) -> SourceRef:
    return SourceRef(
        node_id=_NODE_ID,
        locator=TableCellLocator(
            table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 2"),
            row=row,
            col=col,
            pdf_table_inventory_sha256=inventory.inventory_sha256,
        ),
    )


def _pressure_value(
    raw_text: str,
    inventory: EmbeddedTableInventory,
    *,
    value_col: int = 0,
) -> MeasuredValue:
    """A pressure reading whose NUMBER cites cell (1, ``value_col``) and whose
    UNIT cites cell (1,1) ("atm"), all real cells of ``_TABLE_PDF``. ``raw_text``
    is passed in so a test can make it agree with the cited cell or deliberately
    drift from it."""
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        repairs=(),
        repair_dependency=SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        ),
        quantity_kind=QuantityKind.PRESSURE,
        unit_raw="atm",
        unit_normalized=units.normalize_unit(QuantityKind.PRESSURE, "atm", table=_ACTIVE.table),
        conversion_table_sha256=_ACTIVE.embedded.sha256,
        value_ref=_cell_ref(1, value_col, inventory),
        unit_ref=_cell_ref(1, 1, inventory),
    )


def _condition_set(inventory: EmbeddedTableInventory, value: MeasuredValue, node: SourceNode) -> ConditionSetEnvelope:
    """The smallest condition set carrying one table-cell-grounded scalar claim.

    Subject and attribution are ref-only citations (no paired text), so the only
    content comparison is the scalar claim's -- exactly what a given test wants
    to isolate."""
    return ConditionSetEnvelope(
        source_graph=SourceGraph(nodes=(node,)),
        conversion_tables=(_ACTIVE.embedded,),
        table_inventories=(inventory,),
        subject=UnresolvedSubject(reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=_cell_ref(0, 0, inventory)),
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_ref=_cell_ref(0, 0, inventory),
        scalar_claims=(
            GroundedScalarClaim(
                claim_id="initial_pressure",
                label_raw="phi",
                label_ref=_cell_ref(0, 0, inventory),
                value=value,
                uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
            ),
        ),
        categorical_claims=(),
        unextracted=(),
    )


def _dataset(inventory: EmbeddedTableInventory, value: MeasuredValue, node: SourceNode) -> DatasetEnvelope:
    """The smallest dataset whose single (coordinate) value is grounded by table
    cells. A series must declare at least one coordinate axis, so the pressure
    reading is the coordinate here; nothing about the re-derivation check cares
    which role the axis carries."""
    # A series needs both a coordinate and an observation axis; ``value`` is the
    # coordinate under test, and the observation is a second real cell (1,2).
    coord_axis = AxisDeclaration(
        axis_id="p_in",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.PRESSURE,
        label_raw="phi",
        label_ref=_cell_ref(0, 0, inventory),
    )
    obs_axis = AxisDeclaration(
        axis_id="p_out",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.PRESSURE,
        label_raw="X",
        label_ref=_cell_ref(0, 2, inventory),
    )
    obs_value = _pressure_value("0.5", inventory, value_col=2)
    not_extracted = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
    point = DataPoint(
        point_id="pt1",
        coordinates=(Coordinate(axis_id="p_in", value=value, uncertainty=not_extracted),),
        observations=(Observation(axis_id="p_out", value=obs_value, uncertainty=not_extracted),),
        composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
    )
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(coord_axis, obs_axis),
        constants=(),
        points=(point,),
        digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    return DatasetEnvelope(
        source_graph=SourceGraph(nodes=(node,)),
        composition=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        series=(series,),
        conversion_tables=(_ACTIVE.embedded,),
        table_inventories=(inventory,),
        figure_digitizations=(),
    )


#: A non-PDF document. A JATS_XML node's table cell is a legal, ordinary shape
#: whose ``pdf_table_inventory_sha256`` is Absent(NOT_APPLICABLE): an XML table
#: has no PDF fragment geometry, so there is no grid inventory for replay to
#: compare against. The bytes need only be present and hash to the node's sha256
#: -- nothing here re-derives a grid from them.
_JATS_XML = b"<article><table><tr><td>phi</td><td>0.6</td></tr></table></article>"
_JATS_SHA = hashlib.sha256(_JATS_XML).hexdigest()


def _store_jats_node(tmp_path: Path) -> SourceNode:
    """Store ``_JATS_XML`` as a node's ``raw.bin`` and return a JATS_XML node.

    Its bytes verify (present and hashing to ``sha256``), so replay reaches the
    cell-text lane with a clean node and the ONLY thing exercised is how that
    lane treats a table cell that names no PDF grid."""
    artifact = FetchedArtifact(
        url="https://example.invalid/article.xml",
        final_url="https://example.invalid/article.xml",
        sha256=_JATS_SHA,
        content_type="application/xml",
        n_bytes=len(_JATS_XML),
        fetched_at=datetime.now(UTC),
    )
    extracted = ExtractedText(text="", normalized="", sections=[], extractor="jats:none", lossy=False)
    store_artifact(tmp_path, data=_JATS_XML, artifact=artifact, extracted=extracted, max_bytes=_MAX_BYTES)
    return SourceNode(
        node_id="jats",
        kind=SourceNodeKind.JATS_XML,
        sha256=_JATS_SHA,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _jats_cell_ref(row: int, col: int) -> SourceRef:
    """A table-cell ref into the JATS node whose PDF-inventory citation is
    Absent(NOT_APPLICABLE) -- the legal shape for a cell with no PDF grid."""
    return SourceRef(
        node_id="jats",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 1"),
            row=row,
            col=col,
            pdf_table_inventory_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        ),
    )


class TestAContentDriftIsAFailureNotJustAnAddress:
    def test_a_value_whose_text_is_not_the_cell_it_cites_replays_failed(self, tmp_path: Path) -> None:
        """Gap 1. The number cell (1,0) genuinely reads ``0.6``; the value that
        cites it records ``0.7``. The address resolves, the document matches,
        the cell exists -- and the value still lies about what the cell says."""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory()  # uncorrupted: the grid reproduces
        value = _pressure_value("0.7", inventory)  # cell (1,0) says 0.6
        envelope = _condition_set(inventory, value, node)

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is ReplayOutcome.FAILED
        drift = [f for f in report.evidence_failures if f.ref_path == "scalar_claims[0].value.value_ref"]
        assert len(drift) == 1, report.findings
        finding = drift[0]
        assert finding.category is ReplayOutcome.FAILED
        # The message names the row, the column and BOTH disagreeing strings.
        assert "row=1" in finding.reason
        assert "col=0" in finding.reason
        assert "'0.6'" in finding.reason
        assert "'0.7'" in finding.reason
        assert finding.expected == "0.7"
        assert finding.actual == "0.6"

    def test_a_value_whose_text_matches_the_cell_raises_no_content_finding(self, tmp_path: Path) -> None:
        """The control that gives the test above its teeth: the SAME shape with
        ``raw_text`` set to what cell (1,0) actually says raises no content
        failure (the grid reproduces, so nothing at all fails)."""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory()
        value = _pressure_value("0.6", inventory)
        envelope = _condition_set(inventory, value, node)

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_failures == ()


class TestReplayReDerivesTheGridForBothEnvelopeKinds:
    def test_a_corrupted_inventory_makes_a_condition_set_replay_fail(self, tmp_path: Path) -> None:
        """Gap 2, condition-set lane. Every citation is honest (the value's text
        IS its cell's), so the ONLY thing wrong is the stored grid: cell (0,1)
        was mangled, and re-deriving from the real bytes catches it as a
        MISMATCH -> FAILED. A condition set has no series, so this proves the
        re-derivation is not scoped to the series lane."""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory(corrupt=(0, 1))
        value = _pressure_value("0.6", inventory)  # cell (1,0) still reads 0.6
        envelope = _condition_set(inventory, value, node)

        report = replay_condition_set(tmp_path, envelope)

        assert report.evidence_outcome is ReplayOutcome.FAILED
        self._assert_mismatched_inventory(report, inventory)

    def test_a_corrupted_inventory_makes_a_dataset_replay_fail(self, tmp_path: Path) -> None:
        """Gap 2, dataset lane. Same corruption, a DatasetEnvelope this time."""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory(corrupt=(0, 1))
        value = _pressure_value("0.6", inventory)
        envelope = _dataset(inventory, value, node)

        report = replay_envelope(tmp_path, envelope)

        assert report.evidence_outcome is ReplayOutcome.FAILED
        self._assert_mismatched_inventory(report, inventory)

    def test_an_uncorrupted_inventory_raises_no_re_derivation_failure(self, tmp_path: Path) -> None:
        """The control: the SAME real inventory, uncorrupted, reproduces from
        the bytes and contributes no failure at all -- so the failures above are
        the corruption, not the machinery."""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory()
        value = _pressure_value("0.6", inventory)
        report = replay_condition_set(tmp_path, _condition_set(inventory, value, node))
        assert report.evidence_failures == ()

    @staticmethod
    def _assert_mismatched_inventory(report, inventory: EmbeddedTableInventory) -> None:
        path = f"table_inventories[{inventory.inventory_sha256!r}]"
        matching = [f for f in report.evidence_failures if f.ref_path == path]
        assert len(matching) == 1, report.findings
        finding = matching[0]
        assert finding.category is ReplayOutcome.FAILED
        # The replay outcome (FAILED) and the inventory-verification status
        # (MISMATCHED) live in different layers; the finding names the latter.
        assert "mismatched" in finding.reason


class TestACitedCellWithNoComparableTextIsUnverifiable:
    """The cell-text lane's INABILITY arm: a citation resolves to a real,
    present cell of the right document, and yet there is no verbatim string to
    compare against. Replay must surface that as UNVERIFIABLE -- never a silent
    pass, never a mismatch against ``""``."""

    def test_a_value_citing_a_cell_whose_text_is_absent_replays_unverifiable(self, tmp_path: Path) -> None:
        """Gap-1 inability arm. Cell (1,0) keeps its coordinate but the stored
        record drops its ``"text"`` -- a schema-valid shape (T1 requires only
        unique integer ordinals per cell) -- so ``cell_text`` answers ``None``
        and the value's text has nothing at that cell to be compared against.

        (Re-deriving the grid from the bytes ALSO flags the dropped text as a
        MISMATCH, at ``table_inventories[...]``; that is honest and expected. It
        is a different finding at a different path, and this test pins only the
        cell-text inability, which is the branch under test.)"""
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory(drop_text=(1, 0))  # (1,0) present, but no string text
        value = _pressure_value("0.6", inventory)  # value_ref cites (1,0)
        envelope = _condition_set(inventory, value, node)

        report = replay_condition_set(tmp_path, envelope)

        # A table-cell locator always draws a separate boundary UNVERIFIABLE (it is
        # no char span), so filter to the cell-text lane's own finding by its reason.
        cell = [
            f
            for f in report.findings
            if f.ref_path == "scalar_claims[0].value.value_ref" and "records no comparable text" in f.reason
        ]
        assert len(cell) == 1, report.findings
        finding = cell[0]
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "row=1" in finding.reason
        assert "col=0" in finding.reason
        # Not a comparison that ran: neither string is asserted, because none was read.
        assert finding.expected is None
        assert finding.actual is None


class TestANonPdfTableCellIsNeitherComparedNorFlagged:
    """The cell-text lane's SKIP arm: a table cell whose ``pdf_table_inventory_
    sha256`` is Absent names no PDF grid, so there is nothing to compare against
    and its legality is the schema's to judge. Replay must pass it over in
    silence -- not compare it, not report it, not crash on ``get(Absent)``."""

    @staticmethod
    def _jats_value(raw_text: str) -> MeasuredValue:
        return MeasuredValue(
            raw_text=raw_text,
            canonical_decimal_value=raw_text,
            repairs=(),
            repair_dependency=SemanticDependencyUse(
                dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
                content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
                input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            ),
            quantity_kind=QuantityKind.PRESSURE,
            unit_raw="atm",
            unit_normalized=units.normalize_unit(QuantityKind.PRESSURE, "atm", table=_ACTIVE.table),
            conversion_table_sha256=_ACTIVE.embedded.sha256,
            value_ref=_jats_cell_ref(1, 1),
            unit_ref=_jats_cell_ref(1, 0),
        )

    def test_a_value_grounded_at_an_absent_citation_cell_raises_no_cell_finding(self, tmp_path: Path) -> None:
        """The value's ``raw_text`` is ``9.9`` -- a number no cell of the XML
        table holds -- and yet replay raises NO content failure and NO inability
        for it, because a cell that names no PDF grid is skipped before any
        comparison. Remove the skip and ``embedded_by_sha.get(Absent)`` would
        resolve to ``None`` and manufacture an UNVERIFIABLE against a cell there
        was never a grid to check; this asserts that does not happen."""
        node = _store_jats_node(tmp_path)
        value = self._jats_value("9.9")
        envelope = ConditionSetEnvelope(
            source_graph=SourceGraph(nodes=(node,)),
            conversion_tables=(_ACTIVE.embedded,),
            table_inventories=(),  # every cell here cites Absent, so none is embedded
            subject=UnresolvedSubject(reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=_jats_cell_ref(0, 0)),
            attribution=ConditionAttribution.OWN_EXPERIMENT,
            attribution_ref=_jats_cell_ref(0, 0),
            scalar_claims=(
                GroundedScalarClaim(
                    claim_id="initial_pressure",
                    label_raw="phi",
                    label_ref=_jats_cell_ref(0, 0),
                    value=value,
                    uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                ),
            ),
            categorical_claims=(),
            unextracted=(),
        )

        report = replay_condition_set(tmp_path, envelope)

        # No finding, of any category, mentions a cell text or a missing embed:
        # the non-PDF cells were passed over, not compared and not flagged.
        cell_findings = [
            f
            for f in report.findings
            if "cell text" in f.reason or "comparable text" in f.reason or "does not embed" in f.reason
        ]
        assert cell_findings == [], report.findings
        assert report.evidence_failures == ()


class TestReplayRefusesACitationThisEnvelopeDoesNotEmbed:
    """The cell-text lane's DEFENSIVE arm: V8 makes every present (str) citation
    resolve to an embedded inventory, so a validated envelope never reaches this.
    A corrupt or hand-forged payload (``model_construct``, exactly how one
    reaches a replayer) can, and replay must report inability rather than assume
    the grid is there."""

    def test_a_value_citing_an_unembedded_inventory_replays_unverifiable(self, tmp_path: Path) -> None:
        node = _store_pdf_node(tmp_path)
        inventory = _real_inventory()
        value = _pressure_value("0.6", inventory)
        base = _condition_set(inventory, value, node)
        # Strip the embedded inventories the refs still cite: a state V8 forbids,
        # reached only by bypassing validation.
        forged = ConditionSetEnvelope.model_construct(**{**dict(base), "table_inventories": ()})

        report = replay_condition_set(tmp_path, forged)

        # A table-cell locator always draws a separate boundary UNVERIFIABLE (it is
        # no char span), so filter to the cell-text lane's own finding by its reason.
        cell = [
            f
            for f in report.findings
            if f.ref_path == "scalar_claims[0].value.value_ref" and "does not embed" in f.reason
        ]
        assert len(cell) == 1, report.findings
        finding = cell[0]
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "does not embed" in finding.reason
        assert repr(inventory.inventory_sha256) in finding.reason


class TestReplayCannotReadAMalformedEmbeddedInventory:
    """The re-derivation lane's PARSE-FAILURE arm: a valid ``EmbeddedTable
    Inventory`` self-cohered through T1 when the envelope validated, so
    ``json.loads`` of its ``canonical_json`` cannot honestly fail. A corrupted-
    in-memory object (``model_construct``, bypassing T1) can carry unparseable
    ``canonical_json``, and replay must degrade to UNVERIFIABLE rather than let
    the crash escape."""

    def test_an_inventory_with_unparseable_canonical_json_replays_unverifiable(self, tmp_path: Path) -> None:
        node = _store_pdf_node(tmp_path)
        real = _real_inventory()  # the cited grid, which reproduces cleanly
        value = _pressure_value("0.6", real)
        # A second inventory whose canonical_json is not JSON at all, over the
        # SAME document so replay has hash-verified bytes in hand and reaches the
        # json.loads -- reachable only by bypassing T1.
        bad = EmbeddedTableInventory.model_construct(
            inventory_sha256="f" * 64,
            raw_sha256=_RAW_SHA,
            canonical_json="{ this is not json",
        )
        base = _condition_set(real, value, node)
        forged = ConditionSetEnvelope.model_construct(**{**dict(base), "table_inventories": (real, bad)})

        report = replay_condition_set(tmp_path, forged)

        at_bad = [f for f in report.findings if f.ref_path == f"table_inventories[{bad.inventory_sha256!r}]"]
        assert len(at_bad) == 1, report.findings
        finding = at_bad[0]
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "canonical_json does not parse" in finding.reason
        # The cited grid still reproduced -- the ONLY problem is the unreadable one.
        assert [f for f in report.findings if f.ref_path == f"table_inventories[{real.inventory_sha256!r}]"] == []
