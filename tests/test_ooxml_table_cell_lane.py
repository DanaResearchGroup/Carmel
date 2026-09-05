"""I-054: one staged Word (`.docx`) supplement carried to a stored dataset whose numbers
replay to the document's bytes.

The OOXML-lane counterpart to :mod:`tests.test_table_cell_replay_grounding`. A word-processor
``SI_MEMBER`` node's table cell grounds a series value; the envelope embeds the
``EmbeddedOoxmlTableInventory`` the cell cites; and replay re-derives the grid from the
document's own bytes (``verify_ooxml_inventory_record``) AND compares the value's text against
the cited cell. Every ``.docx`` is built synthetically in-process (``tests.ooxml_fixtures``);
no corpus supplement enters the repository.

The producer arm is exercised too: :class:`_CellCiter` accepts a declared word-processor
``SI_MEMBER`` root and mints an ``ooxml_table_inventory_sha256`` locator, refusing an OOXML
inventory against a PDF node and a PDF inventory against a word-processor node.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    AxisDeclaration,
    AxisRole,
    CaptionLabelKey,
    ConditionAttribution,
    ConditionSetEnvelope,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedOoxmlTableInventory,
    GroundedScalarClaim,
    MeasuredValue,
    Observation,
    SemanticDependencyUse,
    Series,
    SiMemberDocumentKind,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    TableKeyKind,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.condition_set_producer import TableCellGrounding, _cell_locator, _CellCiter
from carmel.services.dataset_producer import _ACTIVE
from carmel.services.dataset_replay import ReplayOutcome, replay_condition_set, replay_envelope
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import store_artifact
from carmel.services.ooxml_table_record import ooxml_inventory_record_payload, read_ooxml_table
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import QuantityKind
from tests.ooxml_fixtures import docx_bytes

_MAX_BYTES = 10_000_000

#: A caption-labelled table mirroring the PDF fixture's shape: header row of labels and a body
#: row carrying a value, a unit, and a second value. Cited cells: (1,0)="0.6", (1,1)="atm",
#: (0,0)="phi", (0,2)="X", (1,2)="0.5". The spare header cell (0,1)="P" is cited by nothing.
_TABLE = [["phi", "P", "X"], ["0.6", "atm", "0.5"]]
_DOCX = docx_bytes([_TABLE])
_SOURCE_SHA = hashlib.sha256(_DOCX).hexdigest()
_NODE_ID = "supplement"

#: A minimal PDF standing in for the PARENT article. A `.docx` supplement is an SI_MEMBER, and
#: the source-graph model requires an SI_MEMBER to hang off a PAPER_PDF or JATS_XML parent -- it
#: is never a root. Its bytes are stored so replay's node-bytes loop finds them; nothing here
#: extracts or grounds against it, so any bytes with a stable sha suffice.
_PAPER = b"%PDF-1.4\n% minimal parent article for the supplement\n%%EOF\n"
_PAPER_SHA = hashlib.sha256(_PAPER).hexdigest()
_PAPER_ID = "paper"


def _word_processor_node() -> SourceNode:
    """A declared word-processor SI_MEMBER over ``_DOCX``, child of the ``_PAPER_ID`` article."""
    return SourceNode(
        node_id=_NODE_ID,
        kind=SourceNodeKind.SI_MEMBER,
        sha256=_SOURCE_SHA,
        parent_node_id=_PAPER_ID,
        origin=ArchiveOrigin(archive_sha256=_SOURCE_SHA, member_display_path="mmc1.docx"),
        extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        document_kind=SiMemberDocumentKind.WORD_PROCESSOR,
    )


def _paper_node() -> SourceNode:
    return SourceNode(
        node_id=_PAPER_ID,
        kind=SourceNodeKind.PAPER_PDF,
        sha256=_PAPER_SHA,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _graph() -> SourceGraph:
    return SourceGraph(nodes=(_paper_node(), _word_processor_node()))


def _store_nodes(tmp_path: Path) -> None:
    """Store both the parent article and the ``.docx`` supplement as raw bytes, so replay's
    per-node bytes loop hash-verifies each and hands the supplement's bytes to the OOXML
    re-derivation."""
    for data, sha, ctype, url in (
        (_PAPER, _PAPER_SHA, "application/pdf", "https://example.invalid/paper.pdf"),
        (
            _DOCX,
            _SOURCE_SHA,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "https://example.invalid/mmc1.docx",
        ),
    ):
        artifact = FetchedArtifact(
            url=url,
            final_url=url,
            sha256=sha,
            content_type=ctype,
            n_bytes=len(data),
            fetched_at=datetime.now(UTC),
        )
        extracted = ExtractedText(text="", normalized="", sections=[], extractor="none", lossy=False)
        store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=_MAX_BYTES)


def _ooxml_inventory(*, corrupt: tuple[int, int] | None = None) -> EmbeddedOoxmlTableInventory:
    """The real inventory ``read_ooxml_table`` derives from ``_DOCX``. With ``corrupt=(row,col)``
    the named cell's stored text is mangled and the record RE-ADDRESSED so it stays self-coherent
    (T1 objects only to the perturbation): replay then re-derives the true grid from the bytes and
    finds the stored one does not reproduce. Uncorrupted, it reproduces exactly."""
    inventory = read_ooxml_table(_DOCX, table_index=0)
    payload = ooxml_inventory_record_payload(inventory, source_sha256=_SOURCE_SHA)
    if corrupt is not None:
        for cell in payload["cells"]:
            if (cell["row"], cell["col"]) == corrupt:
                cell["text"] = cell["text"] + "_CORRUPT"
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedOoxmlTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source_sha256=_SOURCE_SHA,
        canonical_json=canonical,
    )


def _cell_ref(row: int, col: int, inventory: EmbeddedOoxmlTableInventory) -> SourceRef:
    return SourceRef(
        node_id=_NODE_ID,
        locator=TableCellLocator(
            table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 1"),
            row=row,
            col=col,
            pdf_table_inventory_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            ooxml_table_inventory_sha256=inventory.inventory_sha256,
        ),
    )


def _pressure_value(raw_text: str, inventory: EmbeddedOoxmlTableInventory, *, value_col: int = 0) -> MeasuredValue:
    """A reading whose NUMBER cites cell (1, value_col) and whose UNIT cites cell (1,1) ("atm"),
    all real cells of ``_DOCX``. ``raw_text`` is passed so a test can make it agree with the
    cited cell or deliberately drift from it."""
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


def _dataset(inventory: EmbeddedOoxmlTableInventory, value: MeasuredValue) -> DatasetEnvelope:
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
        source_graph=_graph(),
        composition=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        series=(series,),
        conversion_tables=(_ACTIVE.embedded,),
        table_inventories=(),
        ooxml_table_inventories=(inventory,),
        figure_digitizations=(),
    )


def _condition_set(
    inventory: EmbeddedOoxmlTableInventory,
    value: MeasuredValue,
    *,
    embed: bool = True,
) -> ConditionSetEnvelope:
    return ConditionSetEnvelope(
        source_graph=_graph(),
        conversion_tables=(_ACTIVE.embedded,),
        table_inventories=(),
        ooxml_table_inventories=(inventory,) if embed else (),
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


class TestADocxTableBecomesAStoredDatasetThatReplays:
    def test_a_dataset_grounded_at_a_docx_cell_replays_verified(self, tmp_path: Path) -> None:
        _store_nodes(tmp_path)
        inventory = _ooxml_inventory()
        # raw_text "0.6" IS the text of cell (1,0), so the content comparison agrees.
        envelope = _dataset(inventory, _pressure_value("0.6", inventory))
        report = replay_envelope(tmp_path, envelope)
        assert report.checked_table_cells >= 1
        assert all(f.category is not ReplayOutcome.FAILED for f in report.findings), [
            (f.ref_path, f.reason) for f in report.findings if f.category is ReplayOutcome.FAILED
        ]

    def test_a_value_whose_text_is_not_the_docx_cell_replays_failed(self, tmp_path: Path) -> None:
        _store_nodes(tmp_path)
        inventory = _ooxml_inventory()
        # raw_text "9.99" is NOT the text of cell (1,0) ("0.6") -- a content drift.
        envelope = _dataset(inventory, _pressure_value("9.99", inventory))
        report = replay_envelope(tmp_path, envelope)
        failed = [f for f in report.findings if f.category is ReplayOutcome.FAILED]
        assert any("is not the recorded text" in f.reason for f in failed), [f.reason for f in report.findings]

    def test_a_corrupted_ooxml_inventory_makes_a_dataset_replay_fail(self, tmp_path: Path) -> None:
        _store_nodes(tmp_path)
        # The stored grid claims the SPARE header cell (0,1) says "P_CORRUPT"; the document's
        # bytes say "P". Corrupting a cell NOTHING cites isolates the re-derivation failure from
        # the content comparison: the values still agree with their cells, but the grid no longer
        # reproduces from the document's bytes.
        inventory = _ooxml_inventory(corrupt=(0, 1))
        envelope = _dataset(inventory, _pressure_value("0.6", inventory))
        report = replay_envelope(tmp_path, envelope)
        failed = [f for f in report.findings if f.category is ReplayOutcome.FAILED]
        assert any("mismatched" in f.reason for f in failed), [f.reason for f in report.findings]

    def test_a_condition_set_grounded_at_a_docx_cell_replays_verified(self, tmp_path: Path) -> None:
        _store_nodes(tmp_path)
        inventory = _ooxml_inventory()
        envelope = _condition_set(inventory, _pressure_value("0.6", inventory))
        report = replay_condition_set(tmp_path, envelope)
        assert report.checked_table_cells >= 1
        assert all(f.category is not ReplayOutcome.FAILED for f in report.findings), [
            (f.ref_path, f.reason) for f in report.findings if f.category is ReplayOutcome.FAILED
        ]


def _grounding(inventory: EmbeddedOoxmlTableInventory, *, row: int = 1, col: int = 0) -> TableCellGrounding:
    return TableCellGrounding(
        table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 1"),
        row=row,
        col=col,
        inventory=inventory,
    )


class TestTheProducerArmMintsAnOoxmlLocator:
    def test_a_word_processor_root_mints_an_ooxml_locator(self) -> None:
        inventory = _ooxml_inventory()
        grounding = _grounding(inventory)
        citer = _CellCiter(_word_processor_node())
        citer.validate((("claim.value", "0.6", grounding),))
        assert citer.ooxml_table_inventories() == (inventory,)
        assert citer.table_inventories() == ()
        locator = _cell_locator(grounding)
        assert locator.ooxml_table_inventory_sha256 == inventory.inventory_sha256
        assert isinstance(locator.pdf_table_inventory_sha256, Absent)

    def test_an_ooxml_inventory_against_a_pdf_node_is_refused(self) -> None:
        citer = _CellCiter(_paper_node())
        with pytest.raises(Exception, match="not a declared word-processor SI_MEMBER"):
            citer.validate((("claim.value", "0.6", _grounding(_ooxml_inventory())),))


class TestTheSchemaRefusesADishonestOoxmlCitation:
    def test_a_citation_the_envelope_does_not_embed_is_refused(self) -> None:
        inventory = _ooxml_inventory()
        with pytest.raises(Exception, match="this\n?.*does not embed|does not embed"):
            _condition_set(inventory, _pressure_value("0.6", inventory), embed=False)

    def test_an_inventory_of_a_different_document_is_refused(self) -> None:
        # An inventory whose source_sha256 is a real .docx, but NOT this node's document. Its grid
        # holds the same numeric/unit cells so the value validates; V8b still refuses it because the
        # record names a document the node is not.
        other = docx_bytes([[["a", "b"], ["0.6", "atm"]]])
        other_sha = hashlib.sha256(other).hexdigest()
        other_inv = read_ooxml_table(other, table_index=0)
        payload = ooxml_inventory_record_payload(other_inv, source_sha256=other_sha)
        canonical = canonical_json_bytes(payload).decode("utf-8")
        wrong_doc = EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            source_sha256=other_sha,
            canonical_json=canonical,
        )
        with pytest.raises(Exception, match="describes a different document"):
            _condition_set(wrong_doc, _pressure_value("0.6", wrong_doc, value_col=0), embed=True)

    def test_a_cell_the_grid_never_derived_is_refused(self) -> None:
        inventory = _ooxml_inventory()
        with pytest.raises(Exception, match="grid has no such cell"):
            ConditionSetEnvelope(
                source_graph=_graph(),
                conversion_tables=(_ACTIVE.embedded,),
                table_inventories=(),
                ooxml_table_inventories=(inventory,),
                subject=UnresolvedSubject(
                    reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=_cell_ref(9, 9, inventory)
                ),
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_ref=_cell_ref(0, 0, inventory),
                scalar_claims=(
                    GroundedScalarClaim(
                        claim_id="c",
                        label_raw="phi",
                        label_ref=_cell_ref(0, 0, inventory),
                        value=_pressure_value("0.6", inventory),
                        uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                    ),
                ),
                categorical_claims=(),
                unextracted=(),
            )

    def test_an_embedded_inventory_nothing_cites_is_unearned_provenance(self) -> None:
        # T4b: embed an OOXML inventory but cite it from no locator (every ref cites the PDF
        # field Absent). The exact-cover check refuses the decorative inventory.
        inventory = _ooxml_inventory()
        pdf_only_ref = SourceRef(
            node_id=_NODE_ID,
            locator=TableCellLocator(
                table_key=CaptionLabelKey(kind=TableKeyKind.CAPTION_LABEL, label="Table 1"),
                row=0,
                col=0,
                pdf_table_inventory_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            ),
        )
        with pytest.raises(Exception, match="decorative inventory"):
            ConditionSetEnvelope(
                source_graph=_graph(),
                conversion_tables=(),
                table_inventories=(),
                ooxml_table_inventories=(inventory,),
                subject=UnresolvedSubject(reason=SubjectRefusalReason.DEVICE_UNNAMED, reason_ref=pdf_only_ref),
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_ref=pdf_only_ref,
                scalar_claims=(),
                categorical_claims=(),
                unextracted=(
                    UnextractedConditionStatement(
                        statement_id="u",
                        reason=UnextractedReason.QUALITATIVE_ONLY,
                        label_raw="phi",
                        label_ref=pdf_only_ref,
                        statement_raw="rich",
                        statement_ref=pdf_only_ref,
                        quantity_kind=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                    ),
                ),
            )
