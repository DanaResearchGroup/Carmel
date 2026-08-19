"""V8/T4/T5: a table cell must name the grid it indexes into.

The rest of the suite proves these rules can be SATISFIED -- every envelope
fixture now has to. This module proves they REJECT, which is the half that
makes them load-bearing rather than decorative.

The guards under test, and the failure each closes:

* **V8** -- a ``PAPER_PDF`` table cell with no citation, a non-PDF cell WITH
  one, an undecidable SI member, a citation to an inventory the envelope does
  not embed, a citation to a grid derived from a different document, and a
  citation to a cell that grid never derived.
* **T1** (on :class:`EmbeddedTableInventory`) -- bytes that are not canonical,
  that do not hash to the address they claim, that name a different document,
  that carry a refusal, or that never say whether they refused.
* **T4/T5** -- an embedded record nothing cites, a duplicate, a bad order.

Fixtures are SYNTHETIC throughout: no paper text enters this repo, and these
payloads are hand-built rather than derived, so the module runs with pypdf
absent. See :mod:`tests.table_inventory_fixtures` for why that is honest here
and what it deliberately does not prove.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    CaptionLabelKey,
    ConditionAttribution,
    ConditionSetEnvelope,
    DeviceClassDeclaration,
    EmbeddedTableInventory,
    MemberSheetKey,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    UnextractedConditionStatement,
    UnextractedReason,
)
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.units import QuantityKind
from tests.table_inventory_fixtures import inventory_payload, make_embedded_inventory

PAPER_SHA = "a" * 64
SI_SHA = "b" * 64
XML_SHA = "c" * 64
OTHER_SHA = "d" * 64

_NOT_APPLICABLE = Absent(reason=AbsenceReason.NOT_APPLICABLE)


def _node(node_id: str, kind: SourceNodeKind, sha256: str, parent_node_id: str | None = None) -> SourceNode:
    return SourceNode(
        node_id=node_id,
        kind=kind,
        sha256=sha256,
        parent_node_id=parent_node_id,
        origin=(
            Absent(reason=AbsenceReason.NOT_APPLICABLE)
            if kind is not SourceNodeKind.SI_MEMBER
            else Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
        ),
        extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        verification=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
    )


def _graph_for(node_id: str) -> SourceGraph:
    """Exactly the targeted node, plus any ancestor it structurally needs.

    Scoped rather than "one node of every kind": V2 refuses a graph holding a
    node no ref targets, so a fat fixture graph would fail on THAT before ever
    reaching the citation rule -- and the test would be measuring V2.
    """
    paper = _node("paper", SourceNodeKind.PAPER_PDF, PAPER_SHA)
    if node_id == "paper":
        return SourceGraph(nodes=(paper,))
    if node_id == "si":
        # `paper` stays: an SI member needs its parent, and an unreferenced
        # ANCESTOR of a targeted node is explicitly not decorative.
        return SourceGraph(nodes=(paper, _node("si", SourceNodeKind.SI_MEMBER, SI_SHA, parent_node_id="paper")))
    return SourceGraph(nodes=(_node("jats", SourceNodeKind.JATS_XML, XML_SHA),))


def _cell_ref(node_id: str, citation: object, *, sheet: bool = False, row: int = 0, col: int = 0) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=TableCellLocator(
            table_key=MemberSheetKey(sheet_name="Sheet1") if sheet else CaptionLabelKey(label="Table 1"),
            row=row,
            col=col,
            pdf_table_inventory_sha256=citation,  # type: ignore[arg-type]
        ),
    )


def _envelope(ref: SourceRef, inventories: tuple[EmbeddedTableInventory, ...]) -> ConditionSetEnvelope:
    """The smallest envelope that can carry one table-cell ref.

    A refusals-only condition set: it embeds no conversion table (nothing here
    is a MeasuredValue), so the only thing under test is the citation.
    """
    return ConditionSetEnvelope(
        source_graph=_graph_for(ref.node_id),
        conversion_tables=(),
        table_inventories=inventories,
        subject=DeviceClassDeclaration(label_raw="shock tube", label_ref=ref),
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_ref=ref,
        scalar_claims=(),
        categorical_claims=(),
        unextracted=(
            UnextractedConditionStatement(
                statement_id="phi",
                label_raw="equivalence ratio",
                label_ref=ref,
                statement_ref=ref,
                reason=UnextractedReason.VALUE_RANGE,
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            ),
        ),
    )


class TestAPdfTableCellMustNameItsGrid:
    def test_a_pdf_cell_citing_its_inventory_is_accepted(self) -> None:
        inventory = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        envelope = _envelope(_cell_ref("paper", inventory.inventory_sha256), (inventory,))
        assert envelope.table_inventories == (inventory,)

    @pytest.mark.parametrize("reason", list(AbsenceReason))
    def test_no_absence_reason_whatsoever_excuses_a_pdf_cell(self, reason: AbsenceReason) -> None:
        """Parametrized over EVERY reason, not just the plausible one.

        ``NOT_EXTRACTED_YET`` is the one that matters: it reads as an honest
        "our extractor has not got there yet" and would let a producer emit
        uncited PDF table cells indefinitely, which is exactly the hole the
        required citation exists to close.
        """
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", Absent(reason=reason)), ())
        assert "must cite the inventory that defines its grid" in str(excinfo.value)

    def test_the_rule_does_not_depend_on_the_nodes_extraction(self) -> None:
        """Every node in ``_graph`` has ``extraction=Absent``, and the PDF cell
        above is still required to cite.

        This is the premise that inverted under review: the fragment lane reads
        RAW BYTES (``extract_fragments(data: bytes)``), so a PDF whose text was
        never extracted can still have an inventory. A rule keyed on
        ``ExtractionBinding.extractor`` would have made exactly this node the
        bypass.
        """
        node = _graph_for("paper").node("paper")
        assert isinstance(node.extraction, Absent)
        with pytest.raises(ValidationError):
            _envelope(_cell_ref("paper", _NOT_APPLICABLE), ())


class TestACellWithNoPdfGeometryMayNotCiteOne:
    def test_an_xml_cell_must_be_absent(self) -> None:
        envelope = _envelope(_cell_ref("jats", _NOT_APPLICABLE), ())
        assert envelope.table_inventories == ()

    def test_an_xml_cell_citing_a_real_embedded_inventory_is_refused(self) -> None:
        """Refused BEFORE exact cover, on purpose.

        The inventory here is real, embedded, and internally coherent, so if
        this check ran after T4 the citation would count towards the cited set
        and pass -- laundering a PDF grid into a node that has no fragment
        geometry at all.
        """
        inventory = make_embedded_inventory(raw_sha256=XML_SHA, cells=((0, 0),))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("jats", inventory.inventory_sha256), (inventory,))
        assert "no PDF fragment geometry a cell inventory could ever describe" in str(excinfo.value)

    def test_a_workbook_sheet_cell_must_be_absent(self) -> None:
        envelope = _envelope(_cell_ref("si", _NOT_APPLICABLE, sheet=True), ())
        assert envelope.table_inventories == ()

    def test_a_workbook_sheet_cell_may_not_cite(self) -> None:
        inventory = make_embedded_inventory(raw_sha256=SI_SHA, cells=((0, 0),))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("si", inventory.inventory_sha256, sheet=True), (inventory,))
        assert "no PDF fragment geometry" in str(excinfo.value)

    @pytest.mark.parametrize("reason", [r for r in AbsenceReason if r is not AbsenceReason.NOT_APPLICABLE])
    def test_only_not_applicable_is_a_true_absence(self, reason: AbsenceReason) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("jats", Absent(reason=reason)), ())
        assert "the only true absence" in str(excinfo.value)


class TestAnUndecidableSiMemberFailsClosed:
    """An SI member may be a PDF or a ``.docx`` and ``SourceNodeKind`` cannot
    say which -- the gap ``_validate_locator_kind_matches_node_kind`` already
    documents as "SI_MEMBER too broad"."""

    def test_a_caption_labelled_si_cell_may_not_be_absent(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("si", _NOT_APPLICABLE), ())
        assert "this schema cannot tell which" in str(excinfo.value)

    def test_but_a_present_citation_self_certifies(self) -> None:
        """Accepted, because it PROVES rather than asserts what the member is.

        The embedded record's ``raw_sha256`` must equal this node's own
        ``sha256``, and only a PDF yields a cell inventory -- so the envelope
        establishes the member is a PDF instead of taking anyone's word.
        """
        inventory = make_embedded_inventory(raw_sha256=SI_SHA, cells=((0, 0),))
        envelope = _envelope(_cell_ref("si", inventory.inventory_sha256), (inventory,))
        assert envelope.table_inventories[0].raw_sha256 == envelope.source_graph.node("si").sha256


class TestACitationMustResolveToTheRightGridOfTheRightDocument:
    def test_a_citation_the_envelope_does_not_embed_is_refused(self) -> None:
        """The store is a CACHE. An envelope whose meaning needs a directory on
        the replaying machine is not self-contained."""
        orphan = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="not embedded")
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", orphan.inventory_sha256), ())
        assert "which this envelope does not embed" in str(excinfo.value)

    def test_an_inventory_of_a_different_document_is_refused(self) -> None:
        """Coherent, embedded, correctly addressed -- and derived from bytes
        this node does not hold. Digest coherence alone cannot catch it."""
        foreign = make_embedded_inventory(raw_sha256=OTHER_SHA, cells=((0, 0),))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", foreign.inventory_sha256), (foreign,))
        assert "describes a different document" in str(excinfo.value)

    def test_a_cell_the_grid_never_derived_is_refused(self) -> None:
        """The right document, the right inventory, a cell that does not exist
        in it -- the failure a sha-only citation cannot see."""
        inventory = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", inventory.inventory_sha256, row=7, col=3), (inventory,))
        assert "whose grid has no such cell" in str(excinfo.value)


class TestARefusedDerivationDefinesNoGrid:
    def test_an_inventory_carrying_a_refusal_cannot_be_embedded(self) -> None:
        """A refusing record is legitimate to STORE -- it is the honest outcome
        for most real tables -- but it is not a grid, so it can never justify a
        cell."""
        payload = inventory_payload(
            raw_sha256=PAPER_SHA,
            cells=((0, 0),),
            refusals=[{"reason": "caption_anchor_absent", "detail": "no caption above the box"}],
        )
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "defines no grid a table cell could be located in" in str(excinfo.value)

    def test_a_record_that_never_says_whether_it_refused_is_refused(self) -> None:
        """``refusal_reasons_of({})`` returns ``()``, so a payload that merely
        OMITS the key would otherwise read as refusal-free. "Does not say" and
        "says none" are different facts and only the second clears a citation.
        """
        payload = {"cells": [{"row": 0, "col": 0}], "payload_version": 1, "raw_sha256": PAPER_SHA}
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "silence is not a refusal-free claim" in str(excinfo.value)

    def test_unreadable_refusals_raise_a_validation_error_not_a_type_error(self) -> None:
        """``refusal_reasons_of`` raises a bare ``TypeError`` on ``["x"]``.

        Uncaught, that escapes as a ``TypeError`` rather than a
        ``ValidationError`` and crashes a caller that correctly catches only the
        latter -- from untrusted, stored bytes.
        """
        payload = {"cells": [], "payload_version": 1, "raw_sha256": PAPER_SHA, "refusals": ["x"]}
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "cannot be read, so it cannot be shown refusal-free" in str(excinfo.value)


class TestTheEmbeddedBytesMustBeWhatTheyClaim:
    def test_bytes_that_do_not_hash_to_the_declared_address_are_refused(self) -> None:
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(inventory_sha256="0" * 64, raw_sha256=PAPER_SHA, canonical_json=canonical)
        assert "does not live at the address it claims" in str(excinfo.value)

    def test_a_non_canonical_rendering_is_refused(self) -> None:
        """Re-serializing must reproduce the stored bytes exactly. Without this,
        two byte strings decoding to the same object would each be honestly
        self-hashed at two different addresses."""
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        spaced = json.dumps(payload, indent=2)
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(spaced.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=spaced,
            )
        assert "is not the canonical rendering" in str(excinfo.value)

    def test_a_record_naming_another_document_than_declared_is_refused(self) -> None:
        payload = inventory_payload(raw_sha256=OTHER_SHA, cells=((0, 0),))
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "not the declared raw_sha256" in str(excinfo.value)

    def test_an_unreadable_payload_version_is_refused(self) -> None:
        """A reader that does not know a shape must not guess at it: "I cannot
        read this" and "this does not reproduce" are different facts."""
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["payload_version"] = 99
        canonical = canonical_json_bytes(payload).decode("utf-8")
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "is not the readable version" in str(excinfo.value)

    def test_json_that_is_not_an_object_is_refused(self) -> None:
        canonical = canonical_json_bytes([1, 2, 3]).decode("utf-8")  # type: ignore[arg-type]
        with pytest.raises(ValidationError) as excinfo:
            EmbeddedTableInventory(
                inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                raw_sha256=PAPER_SHA,
                canonical_json=canonical,
            )
        assert "is not a JSON object" in str(excinfo.value)


class TestTheEmbeddedCollectionIsExactAndOrdered:
    def test_an_inventory_nothing_cites_is_unearned_provenance(self) -> None:
        cited = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        spare = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="spare")
        both = tuple(sorted((cited, spare), key=lambda i: i.inventory_sha256))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", cited.inventory_sha256), both)
        assert "unearned provenance" in str(excinfo.value)

    def test_the_same_inventory_embedded_twice_is_refused(self) -> None:
        """T4 compares SETS, so ``(I, I)`` covers exactly what ``(I,)`` does and
        passes it; T5's sort check accepts two equal adjacent entries. Two
        envelopes with the same logical content and different bytes address
        differently."""
        inventory = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("paper", inventory.inventory_sha256), (inventory, inventory))
        assert "must be embedded exactly once" in str(excinfo.value)

    def test_an_envelope_with_no_table_cells_embeds_nothing(self) -> None:
        """The "no more" half at its boundary: nothing cites, so nothing may be
        embedded."""
        spare = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="spare")
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("jats", _NOT_APPLICABLE), (spare,))
        assert "unearned provenance" in str(excinfo.value)


class TestTheCitationIsIdentityBearing:
    def test_two_locators_differing_only_in_citation_address_differently(self) -> None:
        """If the projection omitted the field, these two -- different claims
        about which grid justified the cell -- would collide on one content
        address in a write-once store."""
        first = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        second = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="second grid")
        one = _envelope(_cell_ref("paper", first.inventory_sha256), (first,))
        two = _envelope(_cell_ref("paper", second.inventory_sha256), (second,))
        assert canonical_json_bytes(one.identity_payload()) != canonical_json_bytes(two.identity_payload())

    def test_an_absent_citation_round_trips_through_the_identity_payload(self) -> None:
        envelope = _envelope(_cell_ref("jats", _NOT_APPLICABLE), ())
        payload = envelope.identity_payload()
        rebuilt = ConditionSetEnvelope.from_identity_payload(payload)
        assert rebuilt == envelope
        assert canonical_json_bytes(rebuilt.identity_payload()) == canonical_json_bytes(payload)

    def test_a_present_citation_round_trips_through_the_identity_payload(self) -> None:
        inventory = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),))
        envelope = _envelope(_cell_ref("paper", inventory.inventory_sha256), (inventory,))
        rebuilt = ConditionSetEnvelope.from_identity_payload(envelope.identity_payload())
        assert rebuilt == envelope


class TestTheCitationMustLookLikeADigest:
    @pytest.mark.parametrize("value", ["", "abc", "A" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n"])
    def test_a_malformed_digest_is_refused(self, value: str) -> None:
        """``"a" * 64 + "\\n"`` is the one that matters: Python's ``$`` also
        matches just before a trailing newline, so a ``match``-based check would
        let it through and mint a second address for one record."""
        with pytest.raises(ValidationError):
            TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0, pdf_table_inventory_sha256=value)

    def test_the_field_has_no_default(self) -> None:
        """No default, so a locator cannot be built without SAYING which case it
        is in -- the same "no unreasoned absence" rule as SourceNode.origin."""
        with pytest.raises(ValidationError) as excinfo:
            TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0)  # type: ignore[call-arg]
        assert "pdf_table_inventory_sha256" in str(excinfo.value)
