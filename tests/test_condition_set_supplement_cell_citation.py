"""I-087: the condition-set producer can mint a table-cell citation naming a NON-root node.

Before this lane the producer's ``_ref`` hard-coded ``node_id=_ROOT_NODE_ID`` and
``_prepare_grounding`` built a single-node graph, so an honest citation into a
word-processor ``.docx`` supplement -- which is structurally never the root -- was
unproducible even though the schema, replay, ``_CellCiter``'s word-processor arm and
``_cell_locator``'s OOXML arm already supported it. These tests drive the REAL
``produce_condition_set_from_artifact`` end to end:

* the acceptance test produces an envelope whose OOXML cell references name the
  supplement CHILD node (not ``paper``) and shows it replays with no FAILED finding;
* three refusal tests pin the typed reason for an unknown node, a wrong node kind, and
  a lane-vs-document-kind mismatch, each driven through the producer rather than by
  hand-building a bad envelope.

Every ``.docx`` and every supplement is synthesised from bytes here; no real
supplementary document enters the repository.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    CaptionLabelKey,
    ConditionAttribution,
    EmbeddedOoxmlTableInventory,
    LocatorKind,
    SiMemberDocumentKind,
    SourceGraph,
    SourceNodeKind,
    TableCellLocator,
)
from carmel.services import units
from carmel.services.condition_set_producer import (
    ConditionSetProducerError,
    DeviceClassSpec,
    ScalarConditionSpec,
    TableCellGrounding,
    _resolve_cell_citations,
    produce_condition_set_from_artifact,
)
from carmel.services.dataset_producer import (
    DatasetProducerError,
    _authenticate_supplement_node,
    _prepare_grounding,
    _supplement_node_id,
)
from carmel.services.dataset_replay import ReplayOutcome, replay_condition_set
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.evidence import store_artifact
from carmel.services.ooxml_table_record import ooxml_inventory_record_payload, read_ooxml_table
from tests.ooxml_fixtures import docx_bytes
from tests.table_inventory_fixtures import make_embedded_inventory_with_texts
from tests.test_dataset_producer import _store_synthetic_artifact

_MAX_BYTES = 10_000_000

_WORD_PROCESSOR_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_SPREADSHEET_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

#: Running prose carrying ONLY the char-span quotes -- the apparatus and the attribution.
#: Every datum a cell cites lives in the supplement, not here.
_ROOT_TEXT = (
    "2. Experimental methods\n"
    "Measurements were carried out in a jet-stirred reactor of fused silica.\n"
    "Table S1 of the supplement lists the measurement conditions.\n"
)

#: A two-row supplement table. Cited cells: (0,0)='initial pressure', (1,0)='0.6',
#: (1,1)='bar'. Cell (0,1) is a spare the citation never names.
_DOCX_ROWS = [[["initial pressure", "spare header"], ["0.6", "bar"]]]

_TABLE_KEY = CaptionLabelKey(label="Table S1")


def _docx() -> bytes:
    return docx_bytes(_DOCX_ROWS)


def _store_bytes(tmp_path: Path, data: bytes, content_type: str) -> str:
    """Store raw bytes through the real evidence store and return their sha256."""
    sha = hashlib.sha256(data).hexdigest()
    artifact = FetchedArtifact(
        url="https://example.invalid/supplement",
        final_url="https://example.invalid/supplement",
        sha256=sha,
        content_type=content_type,
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    extracted = ExtractedText(text="", normalized="", sections=[], extractor="none", lossy=False)
    store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=_MAX_BYTES)
    return sha


def _ooxml_inventory(docx: bytes, *, source_sha256: str) -> EmbeddedOoxmlTableInventory:
    """The real inventory ``read_ooxml_table`` derives from ``docx``, recorded as coming
    from ``source_sha256``. The refusal tests point ``source_sha256`` at a document whose
    node the producer will not accept, so the grid stays real while its provenance is what
    the producer must judge."""
    record = read_ooxml_table(docx, table_index=0)
    payload = ooxml_inventory_record_payload(record, source_sha256=source_sha256)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return EmbeddedOoxmlTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source_sha256=source_sha256,
        canonical_json=canonical,
    )


def _grounding(inventory: EmbeddedOoxmlTableInventory, row: int, col: int) -> TableCellGrounding:
    return TableCellGrounding(table_key=_TABLE_KEY, row=row, col=col, inventory=inventory)


def _produce(tmp_path: Path, root_sha: str, inventory: EmbeddedOoxmlTableInventory):
    """Drive the real producer with one scalar whose three grounds all cite the supplement."""
    return produce_condition_set_from_artifact(
        tmp_path,
        sha256=root_sha,
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_quote="Measurements were carried out",
        subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
        scalars=(
            ScalarConditionSpec(
                claim_id="initial_pressure",
                label_quote="initial pressure",
                quantity_kind=units.QuantityKind.PRESSURE,
                value_quote="0.6",
                unit_quote="bar",
                label_cell=_grounding(inventory, 0, 0),
                value_cell=_grounding(inventory, 1, 0),
                unit_cell=_grounding(inventory, 1, 1),
            ),
        ),
    )


class TestTheProducerCitesASupplementChildNode:
    """Verifier: the real producer emits an envelope whose cell references name the
    supplement CHILD node, and it replays with no FAILED finding."""

    def test_the_cell_references_name_the_supplement_not_the_root(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        docx = _docx()
        docx_sha = _store_bytes(tmp_path, docx, _WORD_PROCESSOR_CT)
        inventory = _ooxml_inventory(docx, source_sha256=docx_sha)

        envelope = _produce(tmp_path, root.sha256, inventory)

        supplement_id = f"supplement:{docx_sha}"
        # The graph gained exactly one SI_MEMBER child, parented on the root, over the docx.
        by_id = {node.node_id: node for node in envelope.source_graph.nodes}
        assert set(by_id) == {"paper", supplement_id}
        child = by_id[supplement_id]
        assert child.kind is SourceNodeKind.SI_MEMBER
        assert child.document_kind is SiMemberDocumentKind.WORD_PROCESSOR
        assert child.parent_node_id == "paper"
        assert child.sha256 == docx_sha

        scalar = envelope.scalar_claims[0]
        # Every cell ref names the CHILD; the char-span attribution still names the root.
        for ref in (scalar.label_ref, scalar.value.value_ref, scalar.value.unit_ref):
            assert isinstance(ref.locator, TableCellLocator)
            assert ref.locator.kind is LocatorKind.TABLE_CELL
            assert ref.node_id == supplement_id, f"{ref.locator!r} still names {ref.node_id!r}"
            assert isinstance(ref.locator.ooxml_table_inventory_sha256, str)
            assert ref.locator.ooxml_table_inventory_sha256 == inventory.inventory_sha256
        assert envelope.attribution_ref.node_id == "paper"

    def test_the_produced_envelope_replays_with_no_failed_finding(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        docx = _docx()
        docx_sha = _store_bytes(tmp_path, docx, _WORD_PROCESSOR_CT)
        inventory = _ooxml_inventory(docx, source_sha256=docx_sha)

        envelope = _produce(tmp_path, root.sha256, inventory)
        report = replay_condition_set(tmp_path, envelope)

        assert report.checked_table_cells >= 1
        failed = [(f.ref_path, f.reason) for f in report.findings if f.category is ReplayOutcome.FAILED]
        assert failed == [], failed


class TestCellCitationRefusals:
    """One refusal per validation, each driven through the real producer and pinning the
    typed reason -- unknown node, wrong node kind, and lane-vs-document-kind mismatch."""

    def test_an_unknown_supplement_document_is_refused(self, tmp_path: Path) -> None:
        """The cited document has no stored artifact, so no node can be built for it."""
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        absent_sha = hashlib.sha256(b"a supplement that was never stored").hexdigest()
        inventory = _ooxml_inventory(_docx(), source_sha256=absent_sha)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, root.sha256, inventory)

        message = str(excinfo.value)
        assert "no source-graph node could be built for it" in message
        assert absent_sha in message

    def test_a_non_word_processor_supplement_is_refused(self, tmp_path: Path) -> None:
        """The cited document IS stored, but its content_type is a spreadsheet, so the node
        is an SI_MEMBER of the wrong kind for an OOXML (WordprocessingML) grid citation."""
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        xlsx_sha = _store_bytes(tmp_path, b"PK\x03\x04 not really a workbook", _SPREADSHEET_CT)
        inventory = _ooxml_inventory(_docx(), source_sha256=xlsx_sha)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, root.sha256, inventory)

        assert "neither a PAPER_PDF nor a declared word-processor" in str(excinfo.value)

    def test_a_pdf_supplement_is_refused_and_the_refusal_names_its_document_kind(self, tmp_path: Path) -> None:
        """A stored PDF supplement mints an SI_MEMBER with document_kind=PDF -- minting a node
        FOR a document is not the same as being able to cite a cell INSIDE it. Citing one needs a
        table inventory derived from that non-root document, which nothing builds yet, so the
        combination fails closed. The refusal must name the document_kind: reporting only
        kind='si_member' would read as "an SI_MEMBER is not an SI_MEMBER"."""
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        pdf_sha = _store_bytes(tmp_path, b"%PDF-1.4 not really a document", "application/pdf")
        inventory = _ooxml_inventory(_docx(), source_sha256=pdf_sha)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, root.sha256, inventory)

        message = str(excinfo.value)
        assert "neither a PAPER_PDF nor a declared word-processor" in message
        assert "document_kind is 'pdf'" in message

    def test_an_ooxml_citation_against_the_root_pdf_is_refused(self, tmp_path: Path) -> None:
        """An OOXML inventory whose document is the ROOT PDF: the WordprocessingML grid an
        OOXML inventory describes cannot exist in a PDF node -- a lane mismatch."""
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        inventory = _ooxml_inventory(_docx(), source_sha256=root.sha256)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            _produce(tmp_path, root.sha256, inventory)

        assert "cites an OOXML inventory" in str(excinfo.value)


class TestResolveCellCitationsPreservesIncomingNodes:
    """Finding 1: ``_resolve_cell_citations`` must EXTEND the incoming graph, not rebuild it
    from ``(root, *supplements)`` -- otherwise any non-root node the caller already put in the
    graph is silently dropped the moment a supplement is cited. The two producers that build
    this graph today only ever hand it a single-node (root-only) graph, so the drop is LATENT;
    this test drives ``_resolve_cell_citations`` directly with a graph that already carries a
    second node, which is exactly the input the docstrings promise to preserve."""

    def test_a_preexisting_non_root_node_survives_a_supplement_citation(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)

        # A second SI_MEMBER, already in the graph before any citation is resolved. It stands
        # in for "any other node the incoming graph held" -- the thing the buggy rebuild drops.
        extra_docx = docx_bytes([[["extra header", "spare"], ["9", "kPa"]]])
        extra_sha = _store_bytes(tmp_path, extra_docx, _WORD_PROCESSOR_CT)
        extra_node = _authenticate_supplement_node(
            tmp_path,
            sha256=extra_sha,
            node_id=_supplement_node_id(extra_sha),
            parent_node_id="paper",
        )

        # A DIFFERENT supplement, the one the resolved citation will actually cite and mint.
        cited_docx = _docx()
        cited_sha = _store_bytes(tmp_path, cited_docx, _WORD_PROCESSOR_CT)
        inventory = _ooxml_inventory(cited_docx, source_sha256=cited_sha)

        root_graph = _prepare_grounding(
            tmp_path, root.sha256, envelope_noun="condition set", envelope_subject="A condition set"
        ).graph
        incoming = SourceGraph(nodes=(*root_graph.nodes, extra_node))

        resolved = _resolve_cell_citations(
            tmp_path, incoming, (("owner", "initial pressure", _grounding(inventory, 0, 0)),)
        )

        present = {node.node_id for node in resolved.graph.nodes}
        # The pre-existing node MUST survive alongside the root and the newly cited supplement.
        assert present == {"paper", _supplement_node_id(extra_sha), _supplement_node_id(cited_sha)}


class TestAuthenticateSupplementNodeEnforcesDerivedId:
    """Finding 2: ``_authenticate_supplement_node`` documents that its ``node_id`` is DERIVED
    from the sha, never caller-named, but nothing checked it. A future caller minting a
    different id for the same document would produce inconsistent SI_MEMBER ids for one sha
    with nothing objecting. The invariant is now enforced with a typed refusal."""

    def test_a_node_id_not_derived_from_the_sha_is_refused(self, tmp_path: Path) -> None:
        docx = _docx()
        docx_sha = _store_bytes(tmp_path, docx, _WORD_PROCESSOR_CT)
        bogus_id = "supplement:not-derived-from-the-sha"

        with pytest.raises(DatasetProducerError) as excinfo:
            _authenticate_supplement_node(tmp_path, sha256=docx_sha, node_id=bogus_id, parent_node_id="paper")

        message = str(excinfo.value)
        # The refusal names both the id it got and the id it required -- no node was minted.
        assert bogus_id in message
        assert _supplement_node_id(docx_sha) in message


class TestResolveCellCitationsReusesAPreexistingSupplement:
    """D-090: findings 1 and 2 interact. When the incoming graph ALREADY holds the SI_MEMBER
    for a sha a cited cell derives from, the node must be REUSED, not re-minted -- otherwise the
    finding-1 construction ``(*root_graph.nodes, *supplements)`` carries the sha-derived id twice
    and SourceGraph's I1 raises an untyped ValidationError out of the producer. Finding 2's
    centralized derivation makes the collision certain (the re-minted id always equals the
    pre-existing one). Latent today (the sole caller passes a single-node graph), but wrong for
    exactly the multi-node caller finding 1 exists to serve."""

    def test_a_supplement_already_in_the_graph_is_reused_not_reminted(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        docx = _docx()
        docx_sha = _store_bytes(tmp_path, docx, _WORD_PROCESSOR_CT)
        inventory = _ooxml_inventory(docx, source_sha256=docx_sha)

        # The SI_MEMBER for the cited sha is ALREADY in the graph before resolution runs.
        existing = _authenticate_supplement_node(
            tmp_path,
            sha256=docx_sha,
            node_id=_supplement_node_id(docx_sha),
            parent_node_id="paper",
        )
        root_graph = _prepare_grounding(
            tmp_path, root.sha256, envelope_noun="condition set", envelope_subject="A condition set"
        ).graph
        incoming = SourceGraph(nodes=(*root_graph.nodes, existing))

        # Must NOT raise (no re-mint, no duplicate -> no ValidationError).
        resolved = _resolve_cell_citations(
            tmp_path, incoming, (("owner", "initial pressure", _grounding(inventory, 0, 0)),)
        )

        ids = [node.node_id for node in resolved.graph.nodes]
        assert ids.count(_supplement_node_id(docx_sha)) == 1
        assert set(ids) == {"paper", _supplement_node_id(docx_sha)}


class TestResolveCellCitationsRefusesAMisnamedPreexistingNode:
    """D-090 design choice: the reuse path bypasses ``_authenticate_supplement_node``'s
    mint-time id enforcement, so ``_resolve_cell_citations`` holds the same invariant on the
    incoming graph -- a non-root node for a sha must be named ``supplement:<sha>`` -- with a
    typed refusal. The root is exempt (legitimately ``paper``)."""

    def test_a_non_root_node_not_named_from_its_sha_is_refused(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)
        docx = _docx()
        docx_sha = _store_bytes(tmp_path, docx, _WORD_PROCESSOR_CT)
        inventory = _ooxml_inventory(docx, source_sha256=docx_sha)

        good = _authenticate_supplement_node(
            tmp_path,
            sha256=docx_sha,
            node_id=_supplement_node_id(docx_sha),
            parent_node_id="paper",
        )
        # A node whose id is NOT the label derived from its sha -- SourceGraph permits it
        # (node_id has no sha-tie invariant), so the producer must be the one to refuse it.
        misnamed = good.model_copy(update={"node_id": "supplement:not-derived-from-the-sha"})
        root_graph = _prepare_grounding(
            tmp_path, root.sha256, envelope_noun="condition set", envelope_subject="A condition set"
        ).graph
        incoming = SourceGraph(nodes=(*root_graph.nodes, misnamed))

        with pytest.raises(ConditionSetProducerError) as excinfo:
            _resolve_cell_citations(tmp_path, incoming, (("owner", "initial pressure", _grounding(inventory, 0, 0)),))

        message = str(excinfo.value)
        assert misnamed.node_id in message
        assert _supplement_node_id(docx_sha) in message


class TestRootStaysAuthoritativeForItsOwnSha:
    """D-090 (2nd corroboration): the sha->node seeding uses ``setdefault`` so the ROOT wins
    its own sha even when the incoming graph also holds a non-root node carrying that same sha
    (constructible, and it passes the invariant guard by being named ``supplement:<root_sha>``).
    Without that precedence a cell citing the root's sha would resolve to the SI_MEMBER's
    node_id instead of ``paper`` -- a provenance-attribution change. This pins it."""

    def test_a_node_sharing_the_root_sha_does_not_override_the_root(self, tmp_path: Path) -> None:
        root = _store_synthetic_artifact(tmp_path, _ROOT_TEXT)

        # A non-root SI_MEMBER over the ROOT's own bytes -- same sha as the root, legally named
        # supplement:<root_sha>. application/pdf maps to SiMemberDocumentKind.PDF, so this is a
        # genuine node the authenticator mints, not a hand-forged one.
        sibling = _authenticate_supplement_node(
            tmp_path,
            sha256=root.sha256,
            node_id=_supplement_node_id(root.sha256),
            parent_node_id="paper",
        )
        root_graph = _prepare_grounding(
            tmp_path, root.sha256, envelope_noun="condition set", envelope_subject="A condition set"
        ).graph
        incoming = SourceGraph(nodes=(*root_graph.nodes, sibling))

        # A PDF-table cell whose grid was derived from the ROOT's sha -- the root PDF validates it.
        pdf_inventory = make_embedded_inventory_with_texts(
            raw_sha256=root.sha256, cell_texts={(0, 0): "initial pressure"}
        )
        cell = TableCellGrounding(table_key=_TABLE_KEY, row=0, col=0, inventory=pdf_inventory)

        resolved = _resolve_cell_citations(tmp_path, incoming, (("owner", "initial pressure", cell),))

        # The root's sha must still resolve to the ROOT, not to the SI_MEMBER sharing it.
        assert resolved.node_id_by_document_sha[root.sha256] == "paper"
