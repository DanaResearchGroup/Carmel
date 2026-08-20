"""Identity-projection meta-tests for ``ConditionSetEnvelope`` -- the
completeness walk, the authored byte pin, the round trip, and the
leaf-mutation sweep.

TDD NOTE (read before "fixing" a failure here): at the time this file was
written, ``ConditionSetEnvelope`` did not exist yet -- this module is the
RED half of its test-driven construction. ImportErrors below are therefore
EXPECTED and correct until the implementation lands; they are not a defect
in this test file. Do not weaken any assertion here to make it pass early.

Why this file exists at all, when ``tests/test_dataset_identity_payload.py``
already has a completeness meta-test: ``_UNADDRESSED_FIELDS`` is keyed by
STRING CLASS NAME, and the existing ``TestProjectionCompleteness`` walks a
``DatasetEnvelope`` instance only. A brand-new envelope class therefore gets
ZERO completeness coverage by default -- every existing test stays green
while the new class's projection silently drops fields. The tests here are
what close that gap for ``ConditionSetEnvelope``.

``_walk_completeness`` is imported from ``tests.test_dataset_identity_payload``
rather than copied: cross-module test imports are an established convention
in this suite (see ``tests/test_dataset_bridge.py``), and a copied walker
would drift from the one the ``DatasetEnvelope`` meta-test trusts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    ConditionAttribution,
    ConditionSetEnvelope,
    CoordinateFrame,
    DatasetEnvelopeParseError,
    DeviceClassDeclaration,
    GroundedCategoricalClaim,
    GroundedScalarClaim,
    MemberSheetKey,
    QuantityKind,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    TextSpace,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
)
from carmel.services.dataset_store import canonical_json_bytes
from tests.table_inventory_fixtures import cover_for, make_embedded_inventory
from tests.test_dataset_identity_payload import (
    _NO_EXTRACTION,
    _NO_EXTRACTION_CROP,
    _NO_GLYPH_HEALTH,
    _NO_GLYPH_HEALTH_CROP,
    _PAPER_EXTRACTION,
    _PAPER_GLYPH_HEALTH,
    SHA_A,
    SHA_B,
    SHA_C,
    _amount,
    _embedded_table_v1,
    _uncertainty,
    _verification_for,
    _walk_completeness,
)

_NO_INVENTORY = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""The only legal absence for a table cell with no PDF fragment geometry (V8)."""


_PAPER_INVENTORY = make_embedded_inventory(raw_sha256=SHA_A, cells=((0, 1),))
"""A DELIBERATELY narrow inventory (one cell) over the paper PDF's own bytes.

The shared fixture grid would put tens of KB of canonical JSON inside
``_PINNED_CANONICAL_BYTES``, and that pin exists to be re-read by eye.

Over the PAPER rather than the SI member because V8 refuses a caption-labelled
SI table cell BOTH ways: an SI member may be a PDF or a word-processor document,
and citing an inventory asserts which rather than establishing it.
"""

# --------------------------------------------------------------------------
# Maximal fixture: a ConditionSetEnvelope that populates every field and
# every optional sub-field this class can legally reach -- a present (not
# Absent) uncertainty on the scalar claim, a present quantity_kind on the
# refusal, an ArchiveOrigin on the SI node, present dpi/render_settings on
# the CoordinateFrame, and a real extraction/glyph-health/verification
# triple on the paper node -- so _walk_completeness actually visits every
# projectable field rather than short-circuiting at an Absent marker.
#
# Unlike the DatasetEnvelope maximal fixture (which needs TWO series to
# reach the XPath arm, because a series may not span two root artifacts),
# this envelope's C4 invariant is WHOLE-ENVELOPE single-root, so the graph
# here is deliberately one root ("paper") with two children ("si", "crop")
# and no JATS node at all: an XPathLocator can only target a JATS_XML node,
# and a parentless JATS root would be a second root artifact, which C4
# refuses by design. The XPath arm is exercised by the DatasetEnvelope
# maximal fixture; it is not reachable from a legal ConditionSetEnvelope
# fixture that also stays single-rooted with the node kinds available today.
# --------------------------------------------------------------------------


def _condition_graph() -> SourceGraph:
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_PAPER_EXTRACTION,
        glyph_health=_PAPER_GLYPH_HEALTH,
        verification=_verification_for(_PAPER_EXTRACTION),
    )
    si = SourceNode(
        node_id="si",
        kind=SourceNodeKind.SI_MEMBER,
        sha256=SHA_B,
        parent_node_id="paper",
        origin=ArchiveOrigin(archive_sha256=SHA_B, member_display_path="si/data.csv"),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION),
    )
    crop = SourceNode(
        node_id="crop",
        kind=SourceNodeKind.FIGURE_CROP,
        sha256=SHA_C,
        parent_node_id="paper",
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_NO_EXTRACTION_CROP,
        glyph_health=_NO_GLYPH_HEALTH_CROP,
        verification=_verification_for(_NO_EXTRACTION_CROP),
    )
    # Node order is ascending by node_id ON PURPOSE: the projection sorts
    # nodes by node_id, and `_walk_completeness` pairs the model tuple
    # against the projected list positionally -- constructing the fixture
    # already-sorted keeps the two aligned. The unsorted case is covered by
    # `TestNodeOrderDoesNotAffectIdentity` below, not by this fixture.
    return SourceGraph(nodes=(crop, paper, si))


def _bbox_ref() -> SourceRef:
    frame = CoordinateFrame(
        render_fingerprint="fp-1",
        cropbox=("0", "0", "612", "792"),
        mediabox=("0", "0", "612", "792"),
        rotation=0,
        units="pt",
        dpi="300",
        render_settings="antialias=on",
    )
    bbox = BBox(frame=frame, x0="10", y0="20", x1="30", y1="40")
    return SourceRef(node_id="crop", locator=BBoxLocator(bbox=bbox))


def _caption_ref() -> SourceRef:
    return SourceRef(
        node_id="paper",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(label="Table 1"),
            row=0,
            col=1,
            pdf_table_inventory_sha256=_PAPER_INVENTORY.inventory_sha256,
        ),
    )


def _sheet_ref() -> SourceRef:
    return SourceRef(
        node_id="si",
        locator=TableCellLocator(
            table_key=MemberSheetKey(sheet_name="Sheet1"), row=1, col=2, pdf_table_inventory_sha256=_NO_INVENTORY
        ),
    )


def _char_ref() -> SourceRef:
    return SourceRef(
        node_id="paper",
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=10, end=20),
    )


def _device_class_subject() -> DeviceClassDeclaration:
    return DeviceClassDeclaration(label_raw="spherical combustion vessel", label_ref=_char_ref())


def _unresolved_subject() -> UnresolvedSubject:
    return UnresolvedSubject(
        reason=SubjectRefusalReason.MULTIPLE_INDISTINGUISHABLE_DEVICES,
        reason_ref=_char_ref(),
    )


def _scalar_claim(**kwargs: object) -> GroundedScalarClaim:
    defaults: dict[str, object] = {
        "claim_id": "initial_pressure",
        "label_raw": "initial pressure",
        "label_ref": _char_ref(),
        "value": _amount("1.0", QuantityKind.PRESSURE, "atm", "atm", _caption_ref(), _sheet_ref()),
        "uncertainty": _uncertainty(QuantityKind.PRESSURE, "atm", "atm", _caption_ref(), _sheet_ref()),
    }
    defaults.update(kwargs)
    return GroundedScalarClaim(**defaults)  # type: ignore[arg-type]


def _categorical_claim(**kwargs: object) -> GroundedCategoricalClaim:
    defaults: dict[str, object] = {
        "claim_id": "diluent",
        "label_raw": "diluent",
        "label_ref": _caption_ref(),
        "token_raw": "CO2",
        "token_ref": _sheet_ref(),
    }
    defaults.update(kwargs)
    return GroundedCategoricalClaim(**defaults)  # type: ignore[arg-type]


def _refusal(**kwargs: object) -> UnextractedConditionStatement:
    defaults: dict[str, object] = {
        "statement_id": "phi_range",
        "label_raw": "equivalence ratio",
        "label_ref": _char_ref(),
        "statement_ref": _bbox_ref(),
        "reason": UnextractedReason.VALUE_RANGE,
        "quantity_kind": QuantityKind.EQUIVALENCE_RATIO,
    }
    defaults.update(kwargs)
    return UnextractedConditionStatement(**defaults)  # type: ignore[arg-type]


def _maximal_condition_set_envelope(**kwargs: object) -> ConditionSetEnvelope:
    defaults: dict[str, object] = {
        "source_graph": _condition_graph(),
        "conversion_tables": (_embedded_table_v1(),),
        "subject": _device_class_subject(),
        "attribution": ConditionAttribution.OWN_EXPERIMENT,
        "attribution_ref": _char_ref(),
        "scalar_claims": (_scalar_claim(),),
        "categorical_claims": (_categorical_claim(),),
        "unextracted": (_refusal(),),
    }
    defaults.update(kwargs)
    # Exact cover derived from the refs the envelope actually holds -- see
    # tests.table_inventory_fixtures.cover_for.
    defaults.setdefault(
        "table_inventories",
        cover_for(*(value for key, value in defaults.items() if key != "source_graph")),
    )
    return ConditionSetEnvelope(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 1. Projection completeness -- the trap this file exists to close
# --------------------------------------------------------------------------


class TestProjectionCompleteness:
    """Every field of every model reachable from a maximal
    ``ConditionSetEnvelope`` must appear in ``identity_payload()``'s output,
    or be registered in ``_UNADDRESSED_FIELDS`` with a reason.

    THE TRAP: ``_UNADDRESSED_FIELDS`` is keyed by string class name and the
    existing completeness test walks a ``DatasetEnvelope``, so a new
    envelope class gets no completeness coverage anywhere until a test like
    this one walks an instance of it."""

    def test_identity_payload_projects_every_field_with_a_device_class_subject(self) -> None:
        env = _maximal_condition_set_envelope()
        payload = env.identity_payload()
        _walk_completeness(env, payload, "ConditionSetEnvelope")

    def test_identity_payload_projects_every_field_with_an_unresolved_subject(self) -> None:
        """The subject is a sum, so a single instance can only ever carry one
        variant -- walking only the device-class arm would leave
        ``UnresolvedSubject``'s own fields (``reason``, ``reason_ref``)
        entirely outside completeness coverage."""
        env = _maximal_condition_set_envelope(subject=_unresolved_subject())
        payload = env.identity_payload()
        _walk_completeness(env, payload, "ConditionSetEnvelope")


# --------------------------------------------------------------------------
# 2. The authored byte pin -- NON-self-referential identity evidence
# --------------------------------------------------------------------------
#
# The round-trip check inside from_identity_payload compares the projector
# against itself: if identity_payload() AND from_identity_payload BOTH
# forget the same field (say categorical_claims), the re-projection matches
# the input payload and the round trip passes self-consistently. The pin
# below is external evidence: it was generated ONCE from the maximal
# envelope, then MANUALLY INSPECTED field by field (every field name of
# every new model was located by eye in the literal) before being committed.
# The manual read is the actual test; the bytes only hold it in place so a
# future projection change has to be a deliberate, visible re-pin.
#
# Unlike _GOLDEN_CANONICAL_BYTES in test_dataset_identity_payload.py (which
# pins DatasetEnvelope and must NEVER be regenerated), this pin is authored
# by and for THIS module; regenerating it is legitimate exactly when the
# projection of ConditionSetEnvelope changes on purpose -- re-inspect by eye
# and say so in the commit when that happens.
#
# Regenerated 2026-08-06, alongside the (separately owner-authorised)
# THIRTEENTH regeneration of _GOLDEN_CANONICAL_BYTES, because the projection
# changed on purpose in two ways at once -- an intentional one-time address
# move made before anything was stored under this schema: (1) it became
# CANONICAL (`source_graph.nodes` now projects sorted ascending by node_id;
# tuple order in the model is set-like and meaningless, and projecting it
# verbatim gave one condition set many content addresses), and (2) it became
# SELF-DESCRIBING (two new top-level keys, `envelope_type`="condition_set"
# and `identity_payload_version`=1, so a stored condition set can never be
# silently parsed as a dataset). Verified by diffing the recomputed payload
# key-by-key (json.loads, per top-level key) against the prior pin: the ONLY
# deltas are those two new keys and `source_graph.nodes` reordering from
# [paper, si, crop] to [crop, paper, si], every node's own sub-payload
# identical modulo that order (checked as a node_id-keyed dict equality).
# The fixture's node tuple was reordered to match, and the unsorted case is
# pinned by `TestNodeOrderDoesNotAffectIdentity` below.

#
# Regenerated 2026-08-19 because ConditionSetEnvelope's projection changed on
# purpose: TableCellLocator gained a required `pdf_table_inventory_sha256`, and
# the envelope gained a `table_inventories` collection embedding every inventory
# a table cell cites (V8/T4/T5 -- see carmel/schemas/datasets.py). Verified by
# diffing the recomputed payload against the prior pin key-by-key: the ONLY
# deltas are the new top-level `table_inventories` key and, inside
# `scalar_claims`/`categorical_claims`, the new locator field -- both collections
# compare EQUAL to the old pin once that one field is stripped. Nothing is stored
# under this schema yet (no production code emits a TABLE_CELL locator), so this
# is an address move with nothing to migrate.
_PINNED_CANONICAL_BYTES = b'{"attribution":"own_experiment","attribution_ref":{"locator":{"end":20,"kind":"char_span","start":10,"text_space":"extracted_text"},"node_id":"paper"},"categorical_claims":[{"claim_id":"diluent","label_raw":"diluent","label_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dab96617fd7f0e78890cc12a73ad451b4868df7ce1871613ec08d3f532fd2917","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"},"token_raw":"CO2","token_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}}],"conversion_tables":[{"canonical_json":"{\\"aliases\\":[{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"\xc2\xb0C\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"degC\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"deg C\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s^-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm/sec\\"},{\\"normalized\\":\\"m/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"m s^-1\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cm^3\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cc\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"l\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"liter\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"litre\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xc2\xb5s\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xce\xbcs\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"ppm\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"ppmv\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"dimensionless\\"}],\\"base_units\\":[[\\"length\\",\\"m\\"],[\\"velocity\\",\\"m/s\\"],[\\"temperature\\",\\"K\\"],[\\"pressure\\",\\"Pa\\"],[\\"time\\",\\"s\\"],[\\"volume\\",\\"m3\\"],[\\"strain_rate\\",\\"1/s\\"],[\\"mole_fraction\\",\\"1\\"],[\\"mass_fraction\\",\\"1\\"],[\\"equivalence_ratio\\",\\"1\\"],[\\"relative_uncertainty\\",\\"1\\"]],\\"rules\\":[{\\"kind\\":\\"identity\\",\\"quantity\\":\\"length\\",\\"unit\\":\\"m\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"velocity\\",\\"unit\\":\\"m/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"temperature\\",\\"unit\\":\\"K\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"pressure\\",\\"unit\\":\\"Pa\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"time\\",\\"unit\\":\\"s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"volume\\",\\"unit\\":\\"m3\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"strain_rate\\",\\"unit\\":\\"1/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mole_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mass_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"equivalence_ratio\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"relative_uncertainty\\",\\"unit\\":\\"1\\"},{\\"from_unit\\":\\"cm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"mm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"cm/s\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"velocity\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m/s\\"},{\\"from_unit\\":\\"C\\",\\"kind\\":\\"affine\\",\\"offset\\":\\"273.15\\",\\"quantity\\":\\"temperature\\",\\"scale\\":\\"1\\",\\"to_unit\\":\\"K\\"},{\\"from_unit\\":\\"atm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"101325\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"bar\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"100000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"kPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"MPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"ms\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"us\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"cm3\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"L\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"relative_uncertainty\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"}],\\"table_id\\":\\"carmel-unit-conversions\\",\\"version\\":1}\\n","sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122"}],"envelope_type":"condition_set","identity_payload_version":1,"scalar_claims":[{"claim_id":"initial_pressure","label_raw":"initial pressure","label_ref":{"locator":{"end":20,"kind":"char_span","start":10,"text_space":"extracted_text"},"node_id":"paper"},"uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"pressure","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"atm","unit_raw":"atm","unit_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dab96617fd7f0e78890cc12a73ad451b4868df7ce1871613ec08d3f532fd2917","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"pressure","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"atm","unit_raw":"atm","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dab96617fd7f0e78890cc12a73ad451b4868df7ce1871613ec08d3f532fd2917","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}},"value":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"pressure","raw_text":"1.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"atm","unit_raw":"atm","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dab96617fd7f0e78890cc12a73ad451b4868df7ce1871613ec08d3f532fd2917","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}}],"source_graph":{"nodes":[{"extraction":{"__absent__":true,"note":null,"reason":"not_applicable"},"glyph_health":{"__absent__":true,"note":null,"reason":"not_applicable"},"kind":"figure_crop","node_id":"crop","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":"paper","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","verification":{"__absent__":true,"note":null,"reason":"not_applicable"}},{"extraction":{"extracted_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","extracted_text_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","extraction_sha256":"5291844e7bedab416755e826cbdf2b34283de1753c7ef2a0fcae20f0dc5c2529","extractor":"pdf:pypdf","extractor_code_sha256":"1111111111111111111111111111111111111111111111111111111111111111","identity_payload_version":"2","parent_raw_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pypdf_version":"9.9.9-synthetic"},"glyph_health":{"assessor":{"content_sha256":"af3553a8142b50bba56b6ba164778b4cd2bff6e4916ac2e93c4e1a270ba4ab5a","dependency_id":"carmel.numeric.glyph_health","input_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},"health":{"has_ascii6_uncertainty_marker":false,"has_equals_ambiguity_marker":false,"has_slash_c0_minus_marker":false,"has_thorn_plus_marker":false,"suspects_dash_corruption":false}},"kind":"paper_pdf","node_id":"paper","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":null,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verification":{"extracted_text":"extraction_record_digest_authenticated","raw_artifact":"raw_sha256_digest_authenticated","root_sidecar":"root_sidecar_digest_authenticated"}},{"extraction":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"glyph_health":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"kind":"si_member","node_id":"si","origin":{"archive_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"parent_node_id":"paper","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","verification":{"__absent__":true,"note":null,"reason":"not_extracted_yet"}}]},"subject":{"label_raw":"spherical combustion vessel","label_ref":{"locator":{"end":20,"kind":"char_span","start":10,"text_space":"extracted_text"},"node_id":"paper"},"subject_kind":"device_class"},"table_inventories":[{"canonical_json":"{\\"cells\\":[{\\"col\\":1,\\"member_digests\\":[\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\"],\\"row\\":0,\\"text\\":\\"r0c1\\",\\"x_end\\":\\"0x1.3000000000000p+4\\",\\"x_start\\":\\"0x1.4000000000000p+3\\"}],\\"column_bounds\\":[[\\"0x0.0p+0\\",\\"0x1.f400000000000p+8\\"]],\\"footprint\\":{\\"caption_baseline_y\\":\\"0x1.5e00000000000p+9\\",\\"caption_text\\":\\"Table 1. A fixture, not a table.\\",\\"caption_x_start\\":\\"0x1.2000000000000p+6\\",\\"page\\":0,\\"x_end\\":\\"0x1.f400000000000p+8\\",\\"x_start\\":\\"0x1.2000000000000p+6\\",\\"y_bottom\\":\\"0x1.9000000000000p+7\\",\\"y_top\\":\\"0x1.5900000000000p+9\\"},\\"fragment_geometry_sha256\\":\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\",\\"inventory_code_sha256\\":\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\",\\"payload_version\\":1,\\"pypdf_version\\":\\"0.0.0-fixture\\",\\"raw_sha256\\":\\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\",\\"refusals\\":[],\\"rows\\":[{\\"anchor_text\\":\\"row 0\\",\\"anchor_x_start\\":\\"0x1.2000000000000p+6\\",\\"baseline_y\\":\\"0x1.2c00000000000p+9\\",\\"merged_baselines\\":[],\\"ordinal\\":0}]}\\n","inventory_sha256":"dab96617fd7f0e78890cc12a73ad451b4868df7ce1871613ec08d3f532fd2917","raw_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"unextracted":[{"label_raw":"equivalence ratio","label_ref":{"locator":{"end":20,"kind":"char_span","start":10,"text_space":"extracted_text"},"node_id":"paper"},"quantity_kind":"equivalence_ratio","reason":"value_range","statement_id":"phi_range","statement_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"}}]}\n'  # noqa: E501


class TestCanonicalBytesArePinned:
    def test_the_maximal_envelopes_canonical_bytes_match_the_authored_pin(self) -> None:
        actual = canonical_json_bytes(_maximal_condition_set_envelope().identity_payload())
        assert actual == _PINNED_CANONICAL_BYTES, (
            "ConditionSetEnvelope.identity_payload()'s canonical bytes no longer match the authored "
            "pin. If the projection changed ON PURPOSE, regenerate the pin, re-inspect it by eye "
            "(see the comment above _PINNED_CANONICAL_BYTES), and re-commit it deliberately; if it "
            "did not, this is a silent re-addressing of every stored condition set and must be fixed."
        )


# --------------------------------------------------------------------------
# 3. Round trip through from_identity_payload
# --------------------------------------------------------------------------


class TestFromIdentityPayloadRoundTrip:
    def test_a_device_class_payload_round_trips_byte_for_byte(self) -> None:
        env = _maximal_condition_set_envelope()
        payload = env.identity_payload()
        rebuilt = ConditionSetEnvelope.from_identity_payload(payload)
        assert canonical_json_bytes(rebuilt.identity_payload()) == canonical_json_bytes(payload)
        assert isinstance(rebuilt.subject, DeviceClassDeclaration)

    def test_an_unresolved_subject_payload_round_trips_byte_for_byte(self) -> None:
        env = _maximal_condition_set_envelope(subject=_unresolved_subject())
        payload = env.identity_payload()
        rebuilt = ConditionSetEnvelope.from_identity_payload(payload)
        assert canonical_json_bytes(rebuilt.identity_payload()) == canonical_json_bytes(payload)
        assert isinstance(rebuilt.subject, UnresolvedSubject)

    def test_an_unknown_subject_tag_is_refused_not_guessed(self) -> None:
        payload = _maximal_condition_set_envelope().identity_payload()
        payload["subject"]["subject_kind"] = "apparatus"
        with pytest.raises(DatasetEnvelopeParseError, match="subject_kind"):
            ConditionSetEnvelope.from_identity_payload(payload)

    def test_a_missing_subject_tag_is_refused_not_guessed(self) -> None:
        payload = _maximal_condition_set_envelope().identity_payload()
        del payload["subject"]["subject_kind"]
        with pytest.raises(DatasetEnvelopeParseError, match="subject_kind"):
            ConditionSetEnvelope.from_identity_payload(payload)

    def test_a_tag_naming_the_other_variants_fields_is_refused(self) -> None:
        """A device-class payload whose tag claims 'unresolved' must fail
        loudly: the tag dispatches the parse, so a wrong tag hands
        DeviceClassDeclaration's fields to UnresolvedSubject, whose
        extra='forbid' refuses them -- never a silent reinterpretation."""
        payload = _maximal_condition_set_envelope().identity_payload()
        payload["subject"]["subject_kind"] = "unresolved"
        with pytest.raises(DatasetEnvelopeParseError):
            ConditionSetEnvelope.from_identity_payload(payload)


# --------------------------------------------------------------------------
# 4. Leaf mutations -- every identity-bearing field moves the bytes
# --------------------------------------------------------------------------


def _mutate_claim_id() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(scalar_claims=(_scalar_claim(claim_id="starting_pressure"),))


def _mutate_subject_label_raw() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(
        subject=DeviceClassDeclaration(label_raw="cylindrical combustion vessel", label_ref=_char_ref())
    )


def _mutate_scalar_value() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(
        scalar_claims=(
            _scalar_claim(value=_amount("2.0", QuantityKind.PRESSURE, "atm", "atm", _caption_ref(), _sheet_ref())),
        )
    )


def _mutate_subject_variant() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(subject=_unresolved_subject())


def _mutate_attribution() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(attribution=ConditionAttribution.SIMULATION)


def _mutate_attribution_ref() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(attribution_ref=_caption_ref())


def _mutate_statement_id() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(unextracted=(_refusal(statement_id="phi_interval"),))


def _mutate_categorical_token() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(categorical_claims=(_categorical_claim(token_raw="Ar"),))


def _unresolved_subject_baseline() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(subject=_unresolved_subject())


def _mutate_unresolved_reason() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(
        subject=UnresolvedSubject(
            reason=SubjectRefusalReason.ASSIGNMENT_DEPENDS_ON_RESULT,
            reason_ref=_char_ref(),
        )
    )


def _mutate_unresolved_reason_ref() -> ConditionSetEnvelope:
    return _maximal_condition_set_envelope(
        subject=UnresolvedSubject(
            reason=SubjectRefusalReason.MULTIPLE_INDISTINGUISHABLE_DEVICES,
            reason_ref=_caption_ref(),
        )
    )


class TestEveryIdentityBearingLeafChangesTheBytes:
    """Changing any single identity-bearing field must change the canonical
    bytes -- otherwise two different condition sets would share one address
    in a write-once content-addressed store."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(_mutate_claim_id, id="scalar-claim-id"),
            pytest.param(_mutate_subject_label_raw, id="subject-label-raw"),
            pytest.param(_mutate_scalar_value, id="scalar-value"),
            pytest.param(_mutate_subject_variant, id="subject-variant"),
            pytest.param(_mutate_attribution, id="attribution-enum"),
            pytest.param(_mutate_attribution_ref, id="attribution-ref"),
            pytest.param(_mutate_statement_id, id="refusal-statement-id"),
            pytest.param(_mutate_categorical_token, id="categorical-token-raw"),
        ],
    )
    def test_a_single_leaf_mutation_changes_the_canonical_bytes(
        self, mutate: Callable[[], ConditionSetEnvelope]
    ) -> None:
        baseline = canonical_json_bytes(_maximal_condition_set_envelope().identity_payload())
        mutated = canonical_json_bytes(mutate().identity_payload())
        assert mutated != baseline


class TestEveryUnresolvedSubjectLeafChangesTheBytes:
    """The parametrized sweep above only ever mutates the SUBJECT by
    swapping in a whole ``_unresolved_subject()`` fixture (``subject-variant``)
    against a ``DeviceClassDeclaration`` baseline -- so a change to
    ``subject_kind`` alone (``"device_class"`` -> ``"unresolved"``) already
    guarantees the bytes differ, regardless of whether
    ``UnresolvedSubject.reason`` or ``UnresolvedSubject.reason_ref`` are
    projected correctly. Neither leaf is exercised on its own anywhere else:
    the completeness walk only checks that a key named ``reason``/
    ``reason_ref`` is PRESENT in the output, not that it carries the real
    value, and the round-trip check in ``from_identity_payload`` is
    self-referential (it decodes what this same projection encoded, so a
    self-consistent misprojection -- e.g. ``reason_ref`` always projected as
    a fixed constant -- would round-trip cleanly and never be caught).

    These two tests hold the baseline fixed at the UNRESOLVED arm and
    change exactly one of its two fields at a time, so a leaf that stopped
    contributing to the canonical bytes (while the OTHER leaf still did)
    would be caught here and nowhere else.
    """

    def test_changing_the_refusal_reason_changes_the_canonical_bytes(self) -> None:
        baseline = canonical_json_bytes(_unresolved_subject_baseline().identity_payload())
        mutated = canonical_json_bytes(_mutate_unresolved_reason().identity_payload())
        assert mutated != baseline

    def test_changing_the_refusal_reason_ref_changes_the_canonical_bytes(self) -> None:
        baseline = canonical_json_bytes(_unresolved_subject_baseline().identity_payload())
        mutated = canonical_json_bytes(_mutate_unresolved_reason_ref().identity_payload())
        assert mutated != baseline


# --------------------------------------------------------------------------
# 5. The subject sum's projections can never collide
# --------------------------------------------------------------------------


class TestSubjectProjectionIsTagged:
    """A projection where a DeviceClassDeclaration and an UnresolvedSubject
    could ever produce the same payload would let two different subjects
    address identically in a write-once store -- the tag key is what makes
    that structurally impossible, not just unlikely."""

    def test_each_variant_carries_its_own_tag_value(self) -> None:
        device_payload: dict[str, Any] = _maximal_condition_set_envelope().identity_payload()["subject"]
        unresolved_payload: dict[str, Any] = _maximal_condition_set_envelope(
            subject=_unresolved_subject()
        ).identity_payload()["subject"]

        assert device_payload["subject_kind"] == "device_class"
        assert unresolved_payload["subject_kind"] == "unresolved"

    def test_the_two_variants_can_never_produce_the_same_canonical_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This replaces a former test that asserted the two variants'
        projected field KEYS never overlap beyond the tag. That was the
        wrong invariant: key overlap was never the danger, because the tag
        is precisely what makes a shared key safe (it is applied last and
        is un-clobberable -- see
        ``TestTheSubjectTagCannotBeClobberedByAVariantField`` above). A
        legitimate future field carried by BOTH variants (say, both gain a
        ``notes`` key with the same name) would have made the old test fail
        for a reason that has nothing to do with identity safety, while the
        real safety property would still hold.

        The property that actually matters is: the two variants can never
        produce the same canonical payload, whatever else they carry. This
        test proves that even in the worst case -- both variants' own
        projected fields made to collide completely, key for key and value
        for value -- the two payloads still differ, because the tag itself
        never collides.
        """
        import carmel.schemas.datasets as datasets_module

        monkeypatch.setattr(
            datasets_module,
            "_device_class_declaration_identity_payload",
            lambda subject: {"shared_field": "identical_value"},
        )
        monkeypatch.setattr(
            datasets_module,
            "_unresolved_subject_identity_payload",
            lambda subject: {"shared_field": "identical_value"},
        )

        device_payload = datasets_module._condition_subject_identity_payload(_device_class_subject())
        unresolved_payload = datasets_module._condition_subject_identity_payload(_unresolved_subject())

        assert device_payload != unresolved_payload


class TestTheSubjectTagCannotBeClobberedByAVariantField:
    """If a future variant's own projected fields ever included a key
    literally named ``subject_kind``, that field could overwrite the tag
    that keeps the two subject variants from colliding in a write-once
    content-addressed store -- the one unrecoverable bug class this module
    exists to prevent. No field on either real variant is named that today,
    so this is a test of the STRUCTURAL guarantee, not a live bug: it
    reaches ``_condition_subject_identity_payload`` directly and makes the
    ``DeviceClassDeclaration`` sub-projector emit a colliding key, without
    inventing a fake field on a real pydantic model to do it."""

    def test_a_variant_field_named_after_the_tag_key_is_refused_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import carmel.schemas.datasets as datasets_module

        monkeypatch.setattr(
            datasets_module,
            "_device_class_declaration_identity_payload",
            lambda subject: {"subject_kind": "impostor", "label_raw": subject.label_raw},
        )

        with pytest.raises(AssertionError, match="subject_kind"):
            datasets_module._condition_subject_identity_payload(_device_class_subject())


# --------------------------------------------------------------------------
# 7. Node tuple order is never identity-bearing
# --------------------------------------------------------------------------


class TestNodeOrderDoesNotAffectIdentity:
    """``SourceGraph.nodes`` is semantically a SET: no validator constrains
    its order and no consumer may read meaning into it, so the same graph is
    legally constructible with its nodes in any permutation. The projection
    is what makes identity well-defined anyway -- it emits nodes sorted
    ascending by node_id -- and this test pins that invariant directly,
    because the maximal fixture above is deliberately constructed
    already-sorted (to keep the positional completeness walker aligned) and
    therefore never exercises the unsorted case itself. Without this test,
    dropping the sort from ``_source_graph_identity_payload`` would leave
    every test in this file green while one condition set silently held many
    content addresses in a write-once store."""

    def test_two_node_tuple_orders_produce_identical_canonical_bytes(self) -> None:
        baseline = _maximal_condition_set_envelope()
        permuted_nodes = tuple(reversed(baseline.source_graph.nodes))
        assert [node.node_id for node in permuted_nodes] != [node.node_id for node in baseline.source_graph.nodes], (
            "fixture drift: the permutation must actually change the tuple order"
        )
        permuted = _maximal_condition_set_envelope(source_graph=SourceGraph(nodes=permuted_nodes))

        assert canonical_json_bytes(permuted.identity_payload()) == canonical_json_bytes(baseline.identity_payload()), (
            "two ConditionSetEnvelopes differing ONLY in node tuple order produced different "
            "canonical bytes -- one condition set holds many content addresses"
        )

    def test_the_projected_nodes_are_sorted_by_node_id(self) -> None:
        """The invariance above could also be satisfied by any other
        canonical order; this pins WHICH order is canonical, so the golden
        pin's node ordering is explainable from the projection alone."""
        payload = _maximal_condition_set_envelope().identity_payload()
        projected_ids = [node["node_id"] for node in payload["source_graph"]["nodes"]]
        assert projected_ids == sorted(projected_ids)
