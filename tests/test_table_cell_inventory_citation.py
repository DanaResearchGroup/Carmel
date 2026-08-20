"""V8/T4/T5: a table cell must name the grid it indexes into.

The rest of the suite proves these rules can be SATISFIED -- every envelope
fixture now has to. This module proves they REJECT, which is the half that
makes them load-bearing rather than decorative.

The guards under test, and the failure each closes:

* **V8** -- a ``PAPER_PDF`` table cell with no citation, a non-PDF cell WITH
  one, an SI member EITHER way, a citation to an inventory the envelope does
  not embed, a citation to a grid derived from a different document, and a
  citation to a cell that grid never derived.
* **T1** (on :class:`EmbeddedTableInventory`) -- bytes that are not canonical,
  that do not hash to the address they claim, that name a different document,
  that carry a refusal, that never say whether they refused, that are not the
  SHAPE of a record of their declared version, whose footprint cannot be read
  back, or whose cell ordinals are not integers.
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
from carmel.services.pdf_table_record import (
    INVENTORY_PAYLOAD_KEYS,
    InventoryVerificationStatus,
    footprint_unreadable_reason,
    verify_inventory_record,
)
from carmel.services.units import QuantityKind
from tests.table_inventory_fixtures import embed, inventory_payload, make_embedded_inventory

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


def _nodes_for(node_id: str) -> tuple[SourceNode, ...]:
    """Exactly the targeted node, plus any ancestor it structurally needs.

    Scoped rather than "one node of every kind": V2 refuses a graph holding a
    node no ref targets, so a fat fixture graph would fail on THAT before ever
    reaching the citation rule -- and the test would be measuring V2.
    """
    paper = _node("paper", SourceNodeKind.PAPER_PDF, PAPER_SHA)
    if node_id == "paper":
        return (paper,)
    if node_id == "other":
        return (_node("other", SourceNodeKind.PAPER_PDF, OTHER_SHA),)
    if node_id == "si":
        # `paper` stays: an SI member needs its parent, and an unreferenced
        # ANCESTOR of a targeted node is explicitly not decorative.
        return (paper, _node("si", SourceNodeKind.SI_MEMBER, SI_SHA, parent_node_id="paper"))
    return (_node("jats", SourceNodeKind.JATS_XML, XML_SHA),)


def _graph_for(node_id: str) -> SourceGraph:
    return SourceGraph(nodes=_nodes_for(node_id))


def _graph_for_ids(node_ids: tuple[str, ...]) -> SourceGraph:
    """The union of what each targeted node needs, deduplicated by ``node_id``."""
    merged: dict[str, SourceNode] = {}
    for node_id in node_ids:
        for node in _nodes_for(node_id):
            merged[node.node_id] = node
    return SourceGraph(nodes=tuple(merged.values()))


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


def _envelope_with_refs(
    refs: tuple[SourceRef, SourceRef], inventories: tuple[EmbeddedTableInventory, ...]
) -> ConditionSetEnvelope:
    """The smallest envelope that can carry TWO distinct table-cell refs.

    A refusals-only condition set: it embeds no conversion table (nothing here
    is a MeasuredValue), so the only thing under test is the citation. The two
    refs go to different slots so both are reachable by ``iter_source_refs``,
    which is what lets a test cite two inventories at once.
    """
    subject_ref, statement_ref = refs
    return ConditionSetEnvelope(
        source_graph=_graph_for_ids((subject_ref.node_id, statement_ref.node_id)),
        conversion_tables=(),
        table_inventories=inventories,
        subject=DeviceClassDeclaration(label_raw="shock tube", label_ref=subject_ref),
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_ref=subject_ref,
        scalar_claims=(),
        categorical_claims=(),
        unextracted=(
            UnextractedConditionStatement(
                statement_id="phi",
                label_raw="equivalence ratio",
                label_ref=statement_ref,
                statement_ref=statement_ref,
                reason=UnextractedReason.VALUE_RANGE,
                quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            ),
        ),
    )


def _envelope(ref: SourceRef, inventories: tuple[EmbeddedTableInventory, ...]) -> ConditionSetEnvelope:
    """The smallest envelope that can carry one table-cell ref, used everywhere
    a single ref is the whole point."""
    return _envelope_with_refs((ref, ref), inventories)


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

    def test_and_a_present_citation_does_not_certify_it_either(self) -> None:
        """The refutation of this rule's first draft, kept as a test.

        That draft ACCEPTED a present citation here, reasoning that it
        self-certifies: the embedded record's ``raw_sha256`` must equal this
        node's ``sha256``, and only a PDF yields an inventory, so the envelope
        was said to PROVE the member is a PDF. It proves nothing of the kind.
        ``EmbeddedTableInventory`` never re-derives the grid from the document,
        so the payload -- ``raw_sha256`` included -- is entirely author-
        controlled: the record built below asserts a grid over ``SI_SHA``
        without any PDF having been parsed, which is exactly what an author
        misfiling a ``.docx`` would produce.

        Matching digests prove the author NAMED this node, never that a parser
        ran on it.
        """
        inventory = make_embedded_inventory(raw_sha256=SI_SHA, cells=((0, 0),))
        assert inventory.raw_sha256 == SI_SHA  # coherent, embedded, correctly addressed -- and still not proof
        with pytest.raises(ValidationError) as excinfo:
            _envelope(_cell_ref("si", inventory.inventory_sha256), (inventory,))
        assert "asserts the member is a PDF rather than establishing it" in str(excinfo.value)


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

    @pytest.mark.parametrize("refusals", [0, "", {}, "caption_anchor_absent"], ids=repr)
    def test_a_record_that_never_says_whether_it_refused_is_refused(self, refusals: object) -> None:
        """``refusal_reasons_of`` reads ``payload.get("refusals") or ()``, so
        every FALSY non-list here reads as refusal-free without a single refusal
        having been ruled out. "Does not say" and "says none" are different
        facts and only the second clears a citation.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["refusals"] = refusals
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "silence is not a refusal-free claim" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("refusals", "escapes_as"),
        [(["x"], "TypeError"), ([{}], "KeyError"), ([{"reason": "not_a_reason"}], "ValueError")],
        ids=["str-entry", "empty-dict-entry", "unknown-reason"],
    )
    def test_unreadable_refusals_raise_a_validation_error_not_a_bare_exception(
        self, refusals: list[object], escapes_as: str
    ) -> None:
        """``refusal_reasons_of`` reaches ``entry["reason"]`` unguarded, and each
        of these makes that indexing fail a DIFFERENT way.

        Uncaught, any of them escapes as its own exception type rather than a
        ``ValidationError``, crashing a caller that correctly catches only the
        latter -- from untrusted, stored bytes. Parametrized because the first
        version of this guard caught two of the three: the helper promises
        nothing about which it raises, so the test has to enumerate rather than
        trust the one shape that was noticed first.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["refusals"] = refusals
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        message = str(excinfo.value)
        assert "cannot be read, so it cannot be shown refusal-free" in message
        assert escapes_as in message, "the message should name what actually went wrong inside the helper"


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

    def test_a_descending_collection_is_refused(self) -> None:
        """T5 itself, which nothing here previously reached: every other fixture
        sorts, so disabling T5 could have survived this whole file.

        Two inventories, both cited, embedded in the one order T5 forbids -- so
        exactly one legal, and therefore exactly one addressable, representation
        of the same content exists.

        Both are grids in the SAME document, which is what a paper with two
        tables really looks like; an envelope may not span two root artifacts,
        so two documents is not an option here anyway.
        """
        first = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="Table 1")
        second = make_embedded_inventory(raw_sha256=PAPER_SHA, cells=((0, 0),), marker="Table 2")
        ascending = tuple(sorted((first, second), key=lambda i: i.inventory_sha256))
        refs = (_cell_ref("paper", first.inventory_sha256), _cell_ref("paper", second.inventory_sha256))
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_refs(refs, tuple(reversed(ascending)))
        assert "must be sorted ascending by inventory_sha256" in str(excinfo.value)


class TestARecordThatCouldNeverBeReplayedIsRefused:
    """T1 cannot prove a grid is real -- only ``verify_inventory_record``, holding
    the document, can. What it CAN prove is that the record is not unverifiable by
    construction, which is the last thing a reader holding no document can check.
    """

    def test_a_stray_top_level_key_is_refused(self) -> None:
        """The address is over the canonical bytes, and the verifier compares them
        against a freshly built payload -- which will not carry this key. So a
        record with one can never report REPRODUCED, whatever the document says.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["annotation"] = "a note someone added"
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "could never be replayed against the document it names" in str(excinfo.value)
        assert "'annotation'" in str(excinfo.value), "the message must name the offending key"

    @pytest.mark.parametrize("dropped", sorted(INVENTORY_PAYLOAD_KEYS - {"payload_version", "raw_sha256"}))
    def test_a_record_missing_any_key_of_its_declared_version_is_refused(self, dropped: str) -> None:
        """``footprint`` is the one that bites -- ``verify_inventory_record``
        returns PAYLOAD_UNREADABLE before it ever looks at the document -- but the
        rule is the whole key set, so every key is checked here rather than the
        one that motivated it.

        ``payload_version`` and ``raw_sha256`` are excluded only because their own
        earlier checks fire first and report something more specific.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        del payload[dropped]
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "could never be replayed against the document it names" in str(excinfo.value)
        assert repr(dropped) in str(excinfo.value)

    def test_the_fixture_payload_has_exactly_a_real_records_shape(self) -> None:
        """The fixtures' claim to stand in for a record, stated as an assertion.

        Without this, a future key added to the real payload would leave every
        fixture here quietly testing a shape nothing produces.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        assert set(payload) == set(INVENTORY_PAYLOAD_KEYS)


class TestAFootprintThatCannotBeReadBackIsRefused:
    """The key set proves ``footprint`` is PRESENT; presence was only ever a proxy.

    Each payload here satisfies the key set in full and still leaves
    ``verify_inventory_record`` with nothing to say but PAYLOAD_UNREADABLE -- which
    is a THIRD outcome, distinct from a derivation that failed to reproduce. A
    citation whose record can only ever return it is unfalsifiable by construction.
    """

    @pytest.mark.parametrize(
        ("mutate", "why"),
        [
            (lambda fp: {}, "an empty mapping keeps the key and drops every field"),
            (lambda fp: [], "a footprint that is not a mapping at all"),
            (lambda fp: {**fp, "x_start": "not-a-float"}, "a coordinate float.fromhex cannot read"),
            (lambda fp: {**fp, "page": "zero"}, "a page ordinal that is a word"),
            (lambda fp: {k: v for k, v in fp.items() if k != "caption_baseline_y"}, "one field missing"),
        ],
        ids=["empty-mapping", "not-a-mapping", "unreadable-coordinate", "page-is-a-string", "field-missing"],
    )
    def test_an_unreadable_footprint_is_refused(self, mutate: object, why: str) -> None:
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["footprint"] = mutate(payload["footprint"])  # type: ignore[operator]
        assert set(payload) == set(INVENTORY_PAYLOAD_KEYS), f"{why}: the key set must still pass"
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "could never be checked" in str(excinfo.value)

    def test_the_refusal_is_the_verdict_replay_would_have_reached(self) -> None:
        """Not a rule of the schema's own invention: the same payload, handed to
        the verifier, really does come back PAYLOAD_UNREADABLE. Without this the
        two could drift into disagreeing about which records are checkable.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["footprint"] = {}
        verification = verify_inventory_record(payload, b"%PDF-1.4 whatever bytes")
        assert verification.status is InventoryVerificationStatus.PAYLOAD_UNREADABLE

    def test_the_fixtures_own_footprint_reads(self) -> None:
        """The negative tests above prove nothing if the baseline never read."""
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        assert footprint_unreadable_reason(payload) is None


class TestTheCellIndexIsDerivedAndSurvivesAReload:
    """``has_cell`` answers from an index T1 builds, not from a re-parse.

    That is a cache, and a cache is a second copy of the truth. These pin the two
    ways it could go wrong: silently becoming part of the record's identity, or
    not being there at all after an envelope comes back from storage.
    """

    def test_the_index_takes_no_part_in_the_record(self) -> None:
        """A derived index that reached the dump would reach the address."""
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0), (1, 2)))
        inventory = embed(payload, raw_sha256=PAPER_SHA)
        assert "_cell_index" not in inventory.model_dump()
        assert set(inventory.model_dump()) == {"inventory_sha256", "raw_sha256", "canonical_json"}

    def test_a_reloaded_record_still_knows_its_grid(self) -> None:
        """Storage round-trips envelopes; an index that did not come back would
        answer False for every cell and refuse citations that are perfectly good.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0), (1, 2)))
        inventory = embed(payload, raw_sha256=PAPER_SHA)
        reloaded = EmbeddedTableInventory.model_validate(inventory.model_dump())
        assert reloaded.has_cell(row=1, col=2)
        assert not reloaded.has_cell(row=9, col=9)

    def test_the_index_agrees_with_the_stored_cells(self) -> None:
        """The cache and the bytes it was built from, compared directly."""
        cells = ((0, 0), (1, 2), (31, 3), (91, 90))
        inventory = embed(inventory_payload(raw_sha256=PAPER_SHA, cells=cells), raw_sha256=PAPER_SHA)
        stored = {(cell["row"], cell["col"]) for cell in json.loads(inventory.canonical_json)["cells"]}
        assert stored == {(row, col) for row, col in cells}
        for row, col in stored:
            assert inventory.has_cell(row=row, col=col)


class TestOneCoordinateHoldsOneCell:
    """A repeated ``(row, col)`` is not a grid, and a membership bit cannot say so.

    ``has_cell`` answers from a ``frozenset``, so a record claiming a coordinate
    twice with DIFFERENT text collapses to the same ``True`` a well-formed record
    gives. The citation then resolves and means nothing -- checkable and
    meaningless at once, which is the pair this schema exists to keep apart.
    """

    def test_a_repeated_coordinate_is_refused(self) -> None:
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0), (0, 1)))
        clone = dict(payload["cells"][0])
        clone["text"] = "9999"
        payload["cells"].append(clone)

        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "repeats the coordinate (row=0, col=0)" in str(excinfo.value)

    def test_an_exactly_repeated_coordinate_is_refused_too(self) -> None:
        """Not only the disagreeing case.

        ``build_inventory`` emits one cell per coordinate by construction, so a
        repeat is never something a real derivation produced -- and "appears
        once" is a cheaper property to hold than "agrees with itself".
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0), (0, 1)))
        payload["cells"].append(dict(payload["cells"][0]))

        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "repeats the coordinate (row=0, col=0)" in str(excinfo.value)

    def test_the_duplicate_grid_would_otherwise_have_answered_a_citation(self) -> None:
        """The defect this closes, demonstrated rather than asserted.

        Without the guard the two entries are indistinguishable to every reader
        the record has: the index holds one member, ``has_cell`` says yes, and
        nothing anywhere states which of the two texts sits at (0, 0).
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0), (0, 1)))
        clone = dict(payload["cells"][0])
        clone["text"] = "9999"
        payload["cells"].append(clone)

        coordinates = [(cell["row"], cell["col"]) for cell in payload["cells"]]
        texts = {cell["text"] for cell in payload["cells"] if (cell["row"], cell["col"]) == (0, 0)}
        assert coordinates.count((0, 0)) == 2
        assert texts == {"r0c0", "9999"}
        assert len(set(coordinates)) < len(coordinates)


class TestACellOrdinalMustBeAnOrdinal:
    @pytest.mark.parametrize(
        ("row", "col"),
        [(True, False), (0, True), ("0", "0"), (None, 0)],
        ids=["bool-bool", "bool-col", "str-str", "null-row"],
    )
    def test_a_non_integer_ordinal_is_refused(self, row: object, col: object) -> None:
        """``(True, False)`` is the one with teeth. JSON has no integer type
        distinct from bool and Python's ``True == 1``, so an unchecked payload
        answers ``has_cell(row=1, col=0)`` with YES for a grid holding only
        ``{"row": true, "col": false}`` -- a citation resolving against a cell
        that does not exist.

        A float ordinal is absent from this list because it cannot be built at
        all: ``canonical_json_bytes`` refuses floats outright, so no stored
        record can ever carry one.
        """
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["cells"] = [
            {"col": col, "member_digests": [], "row": row, "text": "x", "x_end": "0x0.0p+0", "x_start": "0x0.0p+0"}
        ]
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "which is not an integer ordinal" in str(excinfo.value)

    def test_the_bool_grid_would_otherwise_have_satisfied_a_real_citation(self) -> None:
        """The defect this closes, demonstrated rather than asserted: the forged
        grid and the cell a locator asks for are equal under ``==`` and unequal
        under the schema."""
        forged = [{"row": True, "col": False}]
        assert any(cell["row"] == 1 and cell["col"] == 0 for cell in forged), "Python equality alone accepts it"
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["cells"] = [
            dict(cell, member_digests=[], text="x", x_end="0x0.0p+0", x_start="0x0.0p+0") for cell in forged
        ]
        with pytest.raises(ValidationError):
            embed(payload, raw_sha256=PAPER_SHA)

    def test_cells_that_are_not_a_list_are_refused(self) -> None:
        payload = inventory_payload(raw_sha256=PAPER_SHA, cells=((0, 0),))
        payload["cells"] = {"row": 0, "col": 0}
        with pytest.raises(ValidationError) as excinfo:
            embed(payload, raw_sha256=PAPER_SHA)
        assert "not a list, so it describes no grid" in str(excinfo.value)


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
