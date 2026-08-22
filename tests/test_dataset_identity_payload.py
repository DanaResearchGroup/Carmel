"""Tests for ``DatasetEnvelope.identity_payload()`` -- M-D2b(e) commit 1,
"THE BRIDGE" between the pydantic schema in carmel.schemas.datasets and the
content-addressing pipeline in carmel.services.dataset_store
(``canonical_json_bytes`` / ``compute_dataset_sha``).

TDD NOTE (read before "fixing" a failure here): at the time this file was
written, ``DatasetEnvelope.identity_payload()`` did not exist yet -- an
independent agent was implementing it concurrently. Import/attribute errors
below are therefore EXPECTED and correct until that implementation lands;
they are not a defect in this test file. Do not weaken any assertion here to
make it pass early, and do not stub the implementation from this file.

Kept in its own module (not folded into test_dataset_graph_and_envelope.py)
because it exercises a materially different concern -- the projection
contract of one method -- rather than the schema's construction invariants.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from enum import Enum

import pytest
from pydantic import BaseModel

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    AxisDeclaration,
    AxisRole,
    BBox,
    BBoxLocator,
    CaptionLabelKey,
    CharSpanLocator,
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    Coordinate,
    CoordinateFrame,
    DataPoint,
    DatasetEnvelope,
    DatasetEnvelopeParseError,
    EmbeddedConversionTable,
    ExtractedTextVerification,
    ExtractionBinding,
    GlyphHealthAssessment,
    MeasuredValue,
    MemberSheetKey,
    Observation,
    QuantityKind,
    RawArtifactVerification,
    RootSidecarVerification,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SourceVerification,
    TableCellLocator,
    TextSpace,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    ValueOrigin,
    XPathLocator,
)
from carmel.services.dataset_store import canonical_json_bytes, compute_dataset_sha
from carmel.services.extraction_record import _build_identity_payload, compute_extraction_sha
from carmel.services.numeric import GlyphHealth
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    GLYPH_HEALTH_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import TABLE_V1
from tests.table_inventory_fixtures import cover_for, make_embedded_inventory

_NO_INVENTORY = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""The only legal absence for a table cell with no PDF fragment geometry (V8)."""


def _verification_for(extraction: ExtractionBinding | Absent) -> SourceVerification | Absent:
    """Mirror ``SourceNode``'s iff-rule: a node carries a verification record
    exactly when it carries an extraction to have verified, and an absent one
    keeps the SAME ``AbsenceReason`` (so a FIGURE_CROP's NOT_APPLICABLE stays
    NOT_APPLICABLE). Deriving it here rather than restating a literal at every
    construction site keeps these fixtures from drifting out of step with the
    validator they are meant to exercise -- a fixture that has to be hand-kept
    consistent with an invariant is a fixture that will eventually contradict
    it silently."""
    if isinstance(extraction, Absent):
        return Absent(reason=extraction.reason)
    return SourceVerification(
        raw_artifact=RawArtifactVerification.RAW_SHA256_DIGEST_AUTHENTICATED,
        extracted_text=ExtractedTextVerification.EXTRACTION_RECORD_DIGEST_AUTHENTICATED,
        root_sidecar=RootSidecarVerification.ROOT_SIDECAR_DIGEST_AUTHENTICATED,
    )


# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

SHA_A = "a" * 64
SHA_B = "b" * 64

_PAPER_INVENTORY = make_embedded_inventory(raw_sha256=SHA_A, cells=((0, 0), (0, 1)))
"""A DELIBERATELY narrow grid: this fixture feeds _GOLDEN_CANONICAL_BYTES, which
exists to be re-read by eye, and the shared fixture grid would bury it under tens
of KB of canonical JSON per citation.

Only the paper has one. V8 refuses a caption-labelled SI table cell BOTH ways --
an SI member may be a PDF or a word-processor document, and citing an inventory
asserts which rather than establishing it -- so an SI cell here is a SHEET cell
with no citation."""

SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_G = "1" * 64
SHA_H = "2" * 64

_CURRENT_REPAIR_DEPENDENCY = SemanticDependencyUse(
    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
)
"""Module-level singleton for MeasuredValue.repair_dependency -- frozen, so
sharing one instance across every fixture that doesn't need a
deliberately-wrong or superseded dependency record is safe."""

_NO_CROP_REGION = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""SourceNode.crop_region for every kind that is not a FIGURE_CROP: nothing
but a crop was cut out of a page, so NOT_APPLICABLE is the only reason
SourceNode's I7 invariant accepts there."""
_NO_EXTRACTION = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_GLYPH_HEALTH = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
_NO_EXTRACTION_CROP = Absent(reason=AbsenceReason.NOT_APPLICABLE)
_NO_GLYPH_HEALTH_CROP = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""FIGURE_CROP-specific counterparts of ``_NO_EXTRACTION``/``_NO_GLYPH_HEALTH``
above: a crop is an image region with no extracted text ever to come, so
``NOT_APPLICABLE`` -- not ``NOT_EXTRACTED_YET`` -- is the only legal reason
for it (SourceNode's I6 invariant). Every other Absent-extraction node kind
(SI_MEMBER, JATS_XML) keeps ``_NO_EXTRACTION``/``_NO_GLYPH_HEALTH`` above
unchanged, since extraction genuinely just hasn't happened yet for those."""
_NO_PYPDF_VERSION = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""Module-level singleton for ExtractionBinding.pypdf_version on a
non-pypdf extractor: the concept does not apply (pypdf never ran), and the
model is frozen so sharing one instance is safe."""

_HEALTHY_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)

_PAPER_PYPDF_VERSION = "9.9.9-synthetic"
"""A deliberately synthetic pypdf version string for the maximal fixture --
never the installed one, so the golden canonical bytes below cannot drift
with the test environment's pypdf install."""

_PAPER_EXTRACTION_SHA256 = compute_extraction_sha(
    _build_identity_payload(
        identity_payload_version="2",
        raw_sha256=SHA_A,
        extractor="pdf:pypdf",
        extractor_code_sha256=SHA_G,
        pypdf_version=_PAPER_PYPDF_VERSION,
        extracted_sha256=SHA_F,
        extracted_text_sha256=SHA_E,
    )
)

_PAPER_EXTRACTION = ExtractionBinding(
    parent_raw_sha256=SHA_A,
    extraction_sha256=_PAPER_EXTRACTION_SHA256,
    extracted_sha256=SHA_F,
    extracted_text_sha256=SHA_E,
    extractor="pdf:pypdf",
    extractor_code_sha256=SHA_G,
    identity_payload_version="2",
    pypdf_version=_PAPER_PYPDF_VERSION,
)
"""The one node in `_maximal_graph()` (the paper itself) that carries a real
extraction/glyph-health pair, so the identity-payload completeness walker
actually exercises `_extraction_binding_identity_payload` and
`_glyph_health_assessment_identity_payload` rather than only ever seeing the
Absent branch.

``extracted_sha256`` (SHA_F), ``extracted_text_sha256`` (SHA_E), and
``extractor_code_sha256`` (SHA_G) are deliberately distinct constants from
the owning `SourceNode`'s raw `sha256` (SHA_A) below -- aliasing any two of
them would hide a digest-swap bug undetectably, since a bug that accidentally
read one digest in place of another would go unnoticed if both happened to
equal the same constant. ``extraction_sha256`` is no longer free to be an
arbitrary distinct constant: an `ExtractionBinding` is self-authenticating
(it recomputes its own address from its own identity fields at
construction), so the address here is COMPUTED, exactly as a producer would
compute it. The extractor is ``pdf:pypdf`` (with a synthetic version) rather
than ``"text"`` so this maximal fixture exercises the PRESENT branch of the
``pypdf_version`` projection."""

_PAPER_GLYPH_HEALTH = GlyphHealthAssessment(
    health=_HEALTHY_GLYPH_HEALTH,
    assessor=SemanticDependencyUse(
        dependency_id=GLYPH_HEALTH_DEPENDENCY_ID,
        content_sha256=current_sha_for(GLYPH_HEALTH_DEPENDENCY_ID),
        input_sha256=SHA_E,
    ),
)


def _embedded_table_v1() -> EmbeddedConversionTable:
    """The one conversion table every ``MeasuredValue`` fixture in this file
    cites (via ``conversion_table_sha256=TABLE_V1.sha256``) -- embedded
    verbatim so ``DatasetEnvelope.conversion_tables``'s T2 cover-exactly
    check is satisfied by every envelope built here."""
    return EmbeddedConversionTable(
        sha256=TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
    )


# --------------------------------------------------------------------------
# Maximal fixture: a DatasetEnvelope that populates every field this test
# module can legally reach, and every discriminated-union arm it can
# legally reach (BBoxLocator/TableCellLocator/XPathLocator;
# CaptionLabelKey/MemberSheetKey; a present Maybe[...] on every optional
# field the schema exposes, rather than Absent).
#
# Getting this to construct at all required working through several
# undocumented cross-field invariants empirically (ValidationError message
# text, not source reading):
#   - XPathLocator may only target a JATS_XML node.
#   - A single Series must be grounded under one root artifact -- refs
#     spanning two different root artifacts are rejected -- so a maximal
#     fixture exercising both a TABLE_CELL/BBOX-rooted paper and an
#     XPATH-rooted JATS document needs *two* Series, not one.
#   - Every SourceNode in the graph must be targeted (directly or via an
#     ancestor relationship) by some SourceRef, or it is rejected as
#     "decorative provenance".
#   - Series.source_form constrains ONLY each point's value_ref locator
#     kind (TABULAR -> TABLE_CELL, TEXTUAL -> XPATH), not label_ref/unit_ref,
#     which may use any locator kind.
#   - A CharSpanLocator's target node must have a present (non-Absent)
#     `extraction` -- of the four nodes in this fixture, only "paper" does,
#     so the CharSpanLocator below targets "paper" directly. "paper" was
#     already covered against "decorative provenance" as the ancestor of
#     "si"/"crop"; targeting it directly here is additive, not required.
# --------------------------------------------------------------------------


def _maximal_bbox() -> BBox:
    """A bbox whose frame has every optional sub-field PRESENT, so
    ``_walk_completeness`` reaches them.

    Shared by the crop node's own ``crop_region`` and by the ``BBoxLocator``
    that targets that crop. They are the same rectangle here only because the
    fixture is minimal: a ``crop_region`` is measured against the PARENT's
    page frame, while a ``BBoxLocator`` on a crop addresses a region within
    the crop's own render (see ``SourceNode.crop_region``)."""
    frame = CoordinateFrame(
        render_fingerprint="fp-1",
        cropbox=("0", "0", "612", "792"),
        mediabox=("0", "0", "612", "792"),
        rotation=0,
        units="pt",
        dpi="300",
        render_settings="antialias=on",
    )
    return BBox(frame=frame, x0="10", y0="20", x1="30", y1="40")


def _maximal_graph() -> tuple[SourceGraph, SourceRef, SourceRef, SourceRef, SourceRef, SourceRef]:
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_PAPER_EXTRACTION,
        glyph_health=_PAPER_GLYPH_HEALTH,
        verification=_verification_for(_PAPER_EXTRACTION),
        crop_region=_NO_CROP_REGION,
    )
    jats = SourceNode(
        node_id="jats",
        kind=SourceNodeKind.JATS_XML,
        sha256=SHA_D,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION),
        crop_region=_NO_CROP_REGION,
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
        crop_region=_NO_CROP_REGION,
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
        # Concrete, with every optional sub-field of its frame present: I7
        # requires a crop to address itself, and this is the MAXIMAL fixture,
        # so an under-populated frame would leave `_walk_completeness`
        # short-circuiting on fields it exists to visit.
        crop_region=_maximal_bbox(),
    )
    # Node order is ascending by node_id ON PURPOSE: the projection sorts
    # nodes by node_id (order in `SourceGraph.nodes` is set-like and carries
    # no meaning), and `_walk_completeness` pairs the model tuple against the
    # projected list positionally -- so the maximal fixture keeps the two
    # aligned by being constructed already-sorted. The unsorted case is
    # covered by the dedicated order-invariance tests, not by this fixture.
    graph = SourceGraph(nodes=(crop, jats, paper, si))

    bbox_ref = SourceRef(node_id="crop", locator=BBoxLocator(bbox=_maximal_bbox()))
    table_ref_caption = SourceRef(
        node_id="paper",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(label="Table 1"),
            row=0,
            col=1,
            pdf_table_inventory_sha256=_PAPER_INVENTORY.inventory_sha256,
        ),
    )
    table_ref_sheet = SourceRef(
        node_id="si",
        locator=TableCellLocator(
            table_key=MemberSheetKey(sheet_name="Sheet1"), row=1, col=2, pdf_table_inventory_sha256=_NO_INVENTORY
        ),
    )
    xpath_ref = SourceRef(node_id="jats", locator=XPathLocator(xpath="//table/row[1]/cell[1]"))
    char_span_ref = SourceRef(
        node_id="paper",
        locator=CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=10, end=20),
    )
    return graph, bbox_ref, table_ref_caption, table_ref_sheet, xpath_ref, char_span_ref


def _amount(
    raw: str,
    qk: QuantityKind,
    unit_raw: str,
    unit_norm: str,
    value_ref: SourceRef,
    unit_ref: SourceRef,
) -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw,
        canonical_decimal_value=raw,
        quantity_kind=qk,
        unit_raw=unit_raw,
        unit_normalized=unit_norm,
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


def _uncertainty(qk: QuantityKind, unit_raw: str, unit_norm: str, ref_a: SourceRef, ref_b: SourceRef) -> Uncertainty:
    return Uncertainty(
        kind=UncertaintyKind.STD_DEV,
        basis=UncertaintyBasis.ABSOLUTE,
        scale=UncertaintyScale.LINEAR,
        upper=_amount("0.1", qk, unit_raw, unit_norm, ref_a, ref_b),
        lower=_amount("0.1", qk, unit_raw, unit_norm, ref_b, ref_a),
    )


def _maximal_envelope() -> DatasetEnvelope:
    """A DatasetEnvelope populating every field/union-arm this module can
    legally reach: both TABULAR/TABLE_CELL+BBOX and TEXTUAL/XPATH series,
    both CaptionLabelKey and MemberSheetKey table-key arms, a present
    ArchiveOrigin, present CoordinateFrame.dpi/render_settings, a present
    Composition with a present equivalence_ratio and a component with a
    present role, and an uncertainty on every value that accepts one.

    See the module-level comment above _maximal_graph for the invariants
    (discovered empirically, not documented) that shaped this fixture's
    two-series structure.
    """
    graph, bbox_ref, table_ref_caption, table_ref_sheet, xpath_ref, char_span_ref = _maximal_graph()

    eq = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", table_ref_caption, table_ref_sheet)
    mole = _amount("0.04", QuantityKind.MOLE_FRACTION, "-", "1", table_ref_sheet, bbox_ref)
    composition = Composition(
        raw_name="4% H2 in N2",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=eq,
        components=[CompositionComponent(species_raw_name="H2", amount=mole, role=ComponentRole.FUEL)],
    )

    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=table_ref_caption,
    )
    sl_axis = AxisDeclaration(
        axis_id="sl",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.VELOCITY,
        label_raw="S_L",
        label_ref=bbox_ref,
    )
    t_axis = AxisDeclaration(
        axis_id="temperature",
        role=AxisRole.CONSTANT,
        quantity_kind=QuantityKind.TEMPERATURE,
        label_raw="T",
        label_ref=table_ref_sheet,
    )

    # t_val's unit_ref uses a CharSpanLocator (targeting "paper") rather than
    # table_ref_sheet -- this is a Series.constants Coordinate, not a
    # point's coordinate/observation, so it is outside V4's
    # source_form-vs-value_ref constraint (see _check_source_form_for_ref /
    # _validate_source_form_constrains_value_refs: only Coordinate.value and
    # Observation.value are constrained, never unit_ref/label_ref, and never
    # a Series.constants entry) -- the one place in this fixture a
    # CharSpanLocator can appear without needing a new node or a new axis.
    t_val = _amount("298", QuantityKind.TEMPERATURE, "K", "K", table_ref_caption, char_span_ref)
    const = Coordinate(
        axis_id="temperature",
        value=t_val,
        uncertainty=_uncertainty(QuantityKind.TEMPERATURE, "K", "K", table_ref_caption, table_ref_sheet),
    )
    phi_val = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", table_ref_sheet, bbox_ref)
    coord = Coordinate(
        axis_id="phi",
        value=phi_val,
        uncertainty=_uncertainty(QuantityKind.EQUIVALENCE_RATIO, "-", "1", table_ref_sheet, bbox_ref),
    )
    sl_val = _amount("35.0", QuantityKind.VELOCITY, "cm/s", "cm/s", table_ref_caption, table_ref_sheet)
    obs = Observation(
        axis_id="sl",
        value=sl_val,
        uncertainty=_uncertainty(QuantityKind.VELOCITY, "cm/s", "cm/s", table_ref_caption, table_ref_sheet),
    )
    point = DataPoint(point_id="p1", coordinates=(coord,), observations=(obs,), composition=composition)
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis, t_axis),
        constants=(const,),
        points=(point,),
    )

    axis2 = AxisDeclaration(
        axis_id="phi2",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi2",
        label_ref=xpath_ref,
    )
    sl_axis2 = AxisDeclaration(
        axis_id="sl2",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.VELOCITY,
        label_raw="sl2",
        label_ref=xpath_ref,
    )
    eq2 = _amount("2.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", xpath_ref, xpath_ref)
    sl_val2 = _amount("10.0", QuantityKind.VELOCITY, "cm/s", "cm/s", xpath_ref, xpath_ref)
    coord2 = Coordinate(axis_id="phi2", value=eq2, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    obs2 = Observation(axis_id="sl2", value=sl_val2, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    point2 = DataPoint(
        point_id="q1",
        coordinates=(coord2,),
        observations=(obs2,),
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    series2 = Series(
        series_id="s2",
        source_form=SourceForm.TEXTUAL,
        value_origin=ValueOrigin.SIMULATION,
        axes=(axis2, sl_axis2),
        constants=(),
        points=(point2,),
    )

    return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(series, series2),
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for(composition, (series, series2)),
    )


def _minimal_envelope_with_composition(composition: Composition | Absent) -> DatasetEnvelope:
    """The smallest legal envelope, parameterized only by `composition` --
    used by the Absent-vs-present distinguishability test, which needs two
    envelopes differing in exactly that one field."""
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION),
        crop_region=_NO_CROP_REGION,
    )
    graph = SourceGraph(nodes=(paper,))
    ref = SourceRef(
        node_id="paper",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(label="Table 1"),
            row=0,
            col=0,
            pdf_table_inventory_sha256=_PAPER_INVENTORY.inventory_sha256,
        ),
    )
    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=ref,
    )
    sl_axis = AxisDeclaration(
        axis_id="sl", role=AxisRole.OBSERVATION, quantity_kind=QuantityKind.VELOCITY, label_raw="sl", label_ref=ref
    )
    phi_val = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", ref, ref)
    sl_val = _amount("35.0", QuantityKind.VELOCITY, "cm/s", "cm/s", ref, ref)
    coord = Coordinate(axis_id="phi", value=phi_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    obs = Observation(axis_id="sl", value=sl_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    point = DataPoint(
        point_id="p1",
        coordinates=(coord,),
        observations=(obs,),
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis),
        constants=(),
        points=(point,),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(series,),
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for(composition, (series,)),
    )


# --------------------------------------------------------------------------
# 1. Projection-completeness meta-test
# --------------------------------------------------------------------------
#
# _walk_completeness recurses over an actual *instance* tree in lockstep
# with its projected-dict counterpart, so it needs no separate knowledge of
# the schema's annotations (unlike the SourceRef-walker in
# test_dataset_graph_and_envelope.py, which walks types because it has no
# instance to walk). For every BaseModel node encountered, every one of its
# `model_fields` must either appear as a same-named key in the paired dict,
# or be registered in `_UNADDRESSED_FIELDS` with a non-empty reason.


def _walk_completeness(value: object, projected: object, path: str) -> None:
    if isinstance(value, BaseModel):
        # Import here (not at module scope) so a missing `_UNADDRESSED_FIELDS`
        # produces one clear failure at collection time via the outer test's
        # own import, not a confusing NameError deep in recursion.
        from carmel.schemas.datasets import (  # type: ignore[attr-defined]
            _CONDITIONALLY_PROJECTED_FIELDS,
            _UNADDRESSED_FIELDS,
        )

        model_name = type(value).__name__
        assert isinstance(projected, dict), (
            f"{path}: expected a dict projection of {model_name}, got {type(projected)!r}"
        )
        for name in type(value).model_fields:
            key = (model_name, name)
            if key in _UNADDRESSED_FIELDS:
                assert _UNADDRESSED_FIELDS[key], (
                    f"{path}.{name}: registered in _UNADDRESSED_FIELDS with an empty reason"
                )
                continue
            if key in _CONDITIONALLY_PROJECTED_FIELDS:
                # A field projected for SOME instances of this model and omitted
                # for others. Checked in BOTH directions, deliberately: "must be
                # present when the predicate holds" alone would let a future
                # change start emitting the key unconditionally -- silently
                # re-addressing every envelope that does not carry the field --
                # while this meta-test stayed green. That is the same class of
                # failure the registry exists to catch, pointing the other way.
                is_projected_for, reason = _CONDITIONALLY_PROJECTED_FIELDS[key]
                assert reason, f"{path}.{name}: registered in _CONDITIONALLY_PROJECTED_FIELDS with an empty reason"
                if not is_projected_for(value):
                    assert name not in projected, (
                        f"{path}.{name}: field {name!r} of {model_name} is projected for an instance "
                        "its _CONDITIONALLY_PROJECTED_FIELDS predicate excludes -- either the predicate "
                        "or the projection is wrong, and an unconditional key re-addresses every "
                        "envelope that carries no meaningful value for it"
                    )
                    continue
                assert name in projected, (
                    f"{path}.{name}: field {name!r} of {model_name} is missing from identity_payload()'s "
                    "output for an instance its _CONDITIONALLY_PROJECTED_FIELDS predicate INCLUDES -- "
                    "for this instance the field is identity-bearing and must be projected"
                )
                _walk_completeness(getattr(value, name), projected[name], f"{path}.{name}")
                continue
            assert name in projected, (
                f"{path}.{name}: field {name!r} of {model_name} is missing from identity_payload()'s output "
                f"-- it is neither projected nor registered in _UNADDRESSED_FIELDS"
            )
            _walk_completeness(getattr(value, name), projected[name], f"{path}.{name}")
    elif isinstance(value, Absent):
        assert isinstance(projected, dict), (
            f"{path}: an Absent value must project to a dict shape, got {type(projected)!r}"
        )
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Same rule as for BaseModel above, applied to a stdlib dataclass field --
        # otherwise a dataclass-valued field could silently drop its own sub-fields
        # from identity_payload() while this "completeness" meta-test stayed green.
        from carmel.schemas.datasets import _UNADDRESSED_FIELDS  # type: ignore[attr-defined]

        dataclass_name = type(value).__name__
        assert isinstance(projected, dict), (
            f"{path}: expected a dict projection of {dataclass_name}, got {type(projected)!r}"
        )
        for field in dataclasses.fields(type(value)):
            name = field.name
            key = (dataclass_name, name)
            if key in _UNADDRESSED_FIELDS:
                assert _UNADDRESSED_FIELDS[key], (
                    f"{path}.{name}: registered in _UNADDRESSED_FIELDS with an empty reason"
                )
                continue
            assert name in projected, (
                f"{path}.{name}: field {name!r} of {dataclass_name} is missing from identity_payload()'s output "
                f"-- it is neither projected nor registered in _UNADDRESSED_FIELDS"
            )
            _walk_completeness(getattr(value, name), projected[name], f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        assert isinstance(projected, list), f"{path}: a tuple/list must project to a list, got {type(projected)!r}"
        assert len(projected) == len(value), (
            f"{path}: length mismatch between instance ({len(value)}) and projection ({len(projected)})"
        )
        # value and projected are asserted equal-length just above, so the walk is 1:1;
        # strict=True turns any future violation of that invariant into a loud ValueError.
        for i, (v, p) in enumerate(zip(value, projected, strict=True)):
            _walk_completeness(v, p, f"{path}[{i}]")
    elif isinstance(value, dict):
        assert isinstance(projected, dict), f"{path}: a dict must project to a dict, got {type(projected)!r}"
        for k, v in value.items():
            assert k in projected, f"{path}.{k}: missing from projected dict"
            _walk_completeness(v, projected[k], f"{path}.{k}")
    elif isinstance(value, Enum):
        assert projected == value.value, (
            f"{path}: enum {value!r} must project as its .value ({value.value!r}), got {projected!r}"
        )
    # else: a primitive (str/int/bool/None) -- nothing further to walk.


class TestCompletenessWalkerIsNotVacuous:
    """Proves _walk_completeness actually detects a missing field, using a
    scratch pydantic model built in memory -- NOT by editing anything under
    carmel/. This is the test that justifies trusting
    TestProjectionCompleteness below: without it, a walker with an inverted
    condition (or one that silently no-ops) would let that test pass for the
    wrong reason."""

    def test_walker_fails_when_a_field_is_absent_from_the_projection_and_unregistered(self) -> None:
        class _ScratchLeaf(BaseModel):
            known: str
            forgotten: str  # simulates a field added to a real model but never wired into identity_payload()

        instance = _ScratchLeaf(known="a", forgotten="b")
        projected = {"known": "a"}  # deliberately missing "forgotten", and no registry entry covers it

        with pytest.raises(AssertionError, match="forgotten"):
            _walk_completeness(instance, projected, "root")

    def test_walker_passes_when_the_missing_field_is_registered_with_a_reason(self) -> None:
        import carmel.schemas.datasets as datasets_module

        class _ScratchLeafRegistered(BaseModel):
            known: str
            deliberately_unaddressed: str

        instance = _ScratchLeafRegistered(known="a", deliberately_unaddressed="b")
        projected = {"known": "a"}

        original = dict(datasets_module._UNADDRESSED_FIELDS)  # type: ignore[attr-defined]
        try:
            datasets_module._UNADDRESSED_FIELDS = {  # type: ignore[attr-defined]
                **original,
                ("_ScratchLeafRegistered", "deliberately_unaddressed"): "test-only registry entry",
            }
            _walk_completeness(instance, projected, "root")  # must not raise
        finally:
            datasets_module._UNADDRESSED_FIELDS = original  # type: ignore[attr-defined]

    def test_walker_fails_when_registered_with_an_empty_reason(self) -> None:
        import carmel.schemas.datasets as datasets_module

        class _ScratchLeafEmptyReason(BaseModel):
            known: str
            forgotten: str

        instance = _ScratchLeafEmptyReason(known="a", forgotten="b")
        projected = {"known": "a"}

        original = dict(datasets_module._UNADDRESSED_FIELDS)  # type: ignore[attr-defined]
        try:
            datasets_module._UNADDRESSED_FIELDS = {  # type: ignore[attr-defined]
                **original,
                ("_ScratchLeafEmptyReason", "forgotten"): "",
            }
            with pytest.raises(AssertionError, match="empty reason"):
                _walk_completeness(instance, projected, "root")
        finally:
            datasets_module._UNADDRESSED_FIELDS = original  # type: ignore[attr-defined]

    def test_walker_fails_when_a_conditional_field_is_dropped_for_an_instance_that_needs_it(self) -> None:
        """A FIGURE_CROP's ``crop_region`` IS its identity, so a projection
        that omits it must fail the walk. Fed a hand-made projection rather
        than a real one -- nothing under carmel/ is edited -- so this pins the
        walker's behaviour, not today's projector's."""
        from carmel.schemas.datasets import _source_node_identity_payload  # type: ignore[attr-defined]

        crop = _maximal_graph()[0].node("crop")
        projected = {k: v for k, v in _source_node_identity_payload(crop).items() if k != "crop_region"}

        with pytest.raises(AssertionError, match="predicate INCLUDES"):
            _walk_completeness(crop, projected, "root")

    def test_walker_fails_when_a_conditional_field_is_projected_for_an_instance_that_excludes_it(self) -> None:
        """The other direction, which is the one a well-meaning change breaks:
        emitting ``crop_region`` for a PAPER_PDF folds a constant into the
        address and re-addresses every crop-free envelope in every store. The
        walker must refuse that as loudly as it refuses an omission."""
        from carmel.schemas.datasets import _source_node_identity_payload  # type: ignore[attr-defined]

        paper = _maximal_graph()[0].node("paper")
        projected = {
            **_source_node_identity_payload(paper),
            "crop_region": {"__absent__": True, "reason": "not_applicable", "note": None},
        }

        with pytest.raises(AssertionError, match="predicate excludes"):
            _walk_completeness(paper, projected, "root")

    def test_walker_fails_when_a_conditional_field_is_registered_with_an_empty_reason(self) -> None:
        import carmel.schemas.datasets as datasets_module

        crop = _maximal_graph()[0].node("crop")
        projected = datasets_module._source_node_identity_payload(crop)  # type: ignore[attr-defined]

        original = dict(datasets_module._CONDITIONALLY_PROJECTED_FIELDS)  # type: ignore[attr-defined]
        try:
            datasets_module._CONDITIONALLY_PROJECTED_FIELDS = {  # type: ignore[attr-defined]
                **original,
                ("SourceNode", "crop_region"): (lambda node: True, ""),
            }
            with pytest.raises(AssertionError, match="empty reason"):
                _walk_completeness(crop, projected, "root")
        finally:
            datasets_module._CONDITIONALLY_PROJECTED_FIELDS = original  # type: ignore[attr-defined]

    def test_walker_fails_when_a_dataclass_fields_field_is_absent_from_the_projection_and_unregistered(
        self,
    ) -> None:
        @dataclasses.dataclass(frozen=True)
        class _ProbeDataclassForWalkerTest:
            field_a: str
            field_b: str

        class _ScratchModelWithDataclassField(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            payload: _ProbeDataclassForWalkerTest

        instance = _ScratchModelWithDataclassField(payload=_ProbeDataclassForWalkerTest(field_a="a", field_b="b"))
        # deliberately omit "field_b" from the dataclass's sub-projection, and no
        # registry entry covers it
        projected = {"payload": {"field_a": "a"}}

        with pytest.raises(AssertionError, match="field_b"):
            _walk_completeness(instance, projected, "root")

    def test_walker_passes_when_the_missing_dataclass_field_is_registered_with_a_reason(self) -> None:
        import carmel.schemas.datasets as datasets_module

        @dataclasses.dataclass(frozen=True)
        class _ProbeDataclassForWalkerTest:
            field_a: str
            field_b: str

        class _ScratchModelWithDataclassField(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            payload: _ProbeDataclassForWalkerTest

        instance = _ScratchModelWithDataclassField(payload=_ProbeDataclassForWalkerTest(field_a="a", field_b="b"))
        projected = {"payload": {"field_a": "a"}}

        original = dict(datasets_module._UNADDRESSED_FIELDS)  # type: ignore[attr-defined]
        try:
            datasets_module._UNADDRESSED_FIELDS = {  # type: ignore[attr-defined]
                **original,
                ("_ProbeDataclassForWalkerTest", "field_b"): (
                    "test-only: deliberately omitted for walker not-vacuous proof"
                ),
            }
            _walk_completeness(instance, projected, "root")  # must not raise
        finally:
            datasets_module._UNADDRESSED_FIELDS = original  # type: ignore[attr-defined]

    def test_walker_fails_when_dataclass_field_registered_with_an_empty_reason(self) -> None:
        import carmel.schemas.datasets as datasets_module

        @dataclasses.dataclass(frozen=True)
        class _ProbeDataclassForWalkerTest:
            field_a: str
            field_b: str

        class _ScratchModelWithDataclassField(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            payload: _ProbeDataclassForWalkerTest

        instance = _ScratchModelWithDataclassField(payload=_ProbeDataclassForWalkerTest(field_a="a", field_b="b"))
        projected = {"payload": {"field_a": "a"}}

        original = dict(datasets_module._UNADDRESSED_FIELDS)  # type: ignore[attr-defined]
        try:
            datasets_module._UNADDRESSED_FIELDS = {  # type: ignore[attr-defined]
                **original,
                ("_ProbeDataclassForWalkerTest", "field_b"): "",
            }
            with pytest.raises(AssertionError, match="empty reason"):
                _walk_completeness(instance, projected, "root")
        finally:
            datasets_module._UNADDRESSED_FIELDS = original  # type: ignore[attr-defined]


class TestProjectionCompleteness:
    def test_identity_payload_projects_every_field_of_a_maximal_envelope(self) -> None:
        env = _maximal_envelope()
        payload = env.identity_payload()
        _walk_completeness(env, payload, "DatasetEnvelope")

    def test_unaddressed_fields_registry_has_exactly_the_known_genuine_exclusions(self) -> None:
        """`_UNADDRESSED_FIELDS` must contain ONLY entries for fields that are
        genuinely not identity-bearing -- do not invent entries as a shortcut
        for "forgot to project it". This is not a contradiction with the
        walker tests above (which populate it temporarily, on a throwaway
        scratch model, then restore it) -- those never touch the real dict's
        real content.

        Right now there is exactly one such field:
        ``ArchiveOrigin.member_display_path``, which is display-only by
        contract (see that field's and class's docstrings) -- projecting it
        would make two envelopes differing only in a cosmetic display path
        address differently, even though the archive's actual identity
        (``archive_sha256``) is unchanged. Any OTHER entry appearing here
        would mean a field was excluded without that same justification."""
        from carmel.schemas.datasets import _UNADDRESSED_FIELDS  # type: ignore[attr-defined]

        assert set(_UNADDRESSED_FIELDS) == {("ArchiveOrigin", "member_display_path")}, (
            "the registry must contain exactly the known, justified exclusion(s); "
            f"got {_UNADDRESSED_FIELDS!r} -- if a field is genuinely unprojectable, extend this "
            "test's expected set deliberately (with a reason), don't just let it drift"
        )

    def test_conditionally_projected_registry_has_exactly_the_known_kind_conditional_field(self) -> None:
        """Same discipline as the registry test above, aimed at the other
        escape hatch: a KIND-CONDITIONAL key is a licence to omit a field from
        some addresses, so the set of them must stay small and deliberate. One
        entry today -- ``SourceNode.crop_region``, identity-bearing for a
        FIGURE_CROP and structurally inapplicable to every other kind."""
        from carmel.schemas.datasets import _CONDITIONALLY_PROJECTED_FIELDS  # type: ignore[attr-defined]

        assert set(_CONDITIONALLY_PROJECTED_FIELDS) == {("SourceNode", "crop_region")}, (
            "the registry must contain exactly the known, justified kind-conditional field(s); "
            f"got {set(_CONDITIONALLY_PROJECTED_FIELDS)!r} -- every entry here needs an explicit "
            "inverse in from_identity_payload(), so adding one is a deliberate decision, not drift"
        )


class TestCropRegionIsProjectedOnlyForCrops:
    """``crop_region`` is emitted for a FIGURE_CROP and for no other kind.

    The point is what does NOT get folded into the content address: I7 admits
    exactly one value on a non-crop node, so projecting it there would put a
    CONSTANT in every envelope's address -- no envelope distinguished from any
    other, every stored envelope re-addressed, including the ones holding no
    crop at all.
    """

    def test_the_crop_node_projects_its_region(self) -> None:
        from carmel.schemas.datasets import _bbox_identity_payload  # type: ignore[attr-defined]

        payload = _maximal_envelope().identity_payload()
        crop = next(n for n in payload["source_graph"]["nodes"] if n["kind"] == "figure_crop")
        assert crop["crop_region"] == _bbox_identity_payload(_maximal_bbox())

    @pytest.mark.parametrize("kind", ["paper_pdf", "jats_xml", "si_member"])
    def test_a_non_crop_node_does_not_project_the_key_at_all(self, kind: str) -> None:
        payload = _maximal_envelope().identity_payload()
        node = next(n for n in payload["source_graph"]["nodes"] if n["kind"] == kind)
        assert "crop_region" not in node, (
            f"a {kind} node projected crop_region -- I7 admits only Absent(NOT_APPLICABLE) there, so "
            "this key carries no distinguishing bits and re-addresses every envelope that has one"
        )

    def test_the_fixture_covers_every_node_kind(self) -> None:
        """Guards the two tests above from passing vacuously: if a kind ever
        drops out of the maximal fixture, the parametrization silently stops
        checking it."""
        payload = _maximal_envelope().identity_payload()
        assert {n["kind"] for n in payload["source_graph"]["nodes"]} == {kind.value for kind in SourceNodeKind}

    def test_the_round_trip_is_byte_exact_for_every_node_kind(self) -> None:
        """The omission is a projection SHAPE, not a lost field: the inverse
        restores the marker for every non-crop node before validation, so the
        parse comes back byte-identical -- with all four kinds in one graph."""
        envelope = _maximal_envelope()
        payload = envelope.identity_payload()

        parsed = DatasetEnvelope.from_identity_payload(payload)

        assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(payload)
        for node in parsed.source_graph.nodes:
            if node.kind == SourceNodeKind.FIGURE_CROP:
                assert node.crop_region == _maximal_bbox()
            else:
                assert node.crop_region == Absent(reason=AbsenceReason.NOT_APPLICABLE), (
                    f"{node.kind.value} node came back with {node.crop_region!r} -- the inverse must "
                    "restore exactly the one value I7 admits, not invent another"
                )

    def test_a_payload_carrying_the_key_on_a_non_crop_node_is_refused(self) -> None:
        """The projection shape is canonical. A payload that adds the key back
        on a PAPER_PDF validates (Absent(NOT_APPLICABLE) is legal on the model)
        but does not re-project to itself, and the round-trip comparison is
        what catches it -- so a second byte-shape for one envelope can never
        enter the store beside the first."""
        payload = copy.deepcopy(_maximal_envelope().identity_payload())
        for node in payload["source_graph"]["nodes"]:
            if node["kind"] == "paper_pdf":
                node["crop_region"] = {"__absent__": True, "reason": "not_applicable", "note": None}

        with pytest.raises(DatasetEnvelopeParseError, match="does not byte-match the input payload"):
            DatasetEnvelope.from_identity_payload(payload)

    def test_a_crop_payload_missing_its_region_is_refused(self) -> None:
        """The inverse restores the marker for non-crop nodes ONLY. A crop with
        no region in its bytes is a malformed payload, and inferring one would
        manufacture an address for a figure nobody located."""
        payload = copy.deepcopy(_maximal_envelope().identity_payload())
        for node in payload["source_graph"]["nodes"]:
            if node["kind"] == "figure_crop":
                del node["crop_region"]

        with pytest.raises(DatasetEnvelopeParseError, match="crop_region"):
            DatasetEnvelope.from_identity_payload(payload)

    @pytest.mark.parametrize(
        ("mutate", "shape", "names"),
        [
            (
                lambda p: p.__setitem__("source_graph", "not a graph"),
                "source_graph is not a dict",
                "source_graph",
            ),
            (
                lambda p: p["source_graph"].__setitem__("nodes", "not a list"),
                "nodes is not a list",
                "source_graph.nodes",
            ),
            (
                lambda p: [n.pop("kind") for n in p["source_graph"]["nodes"]],
                "a node carries no kind at all",
                "source_graph.nodes.0.kind",
            ),
            (
                lambda p: [n.__setitem__("kind", "hologram") for n in p["source_graph"]["nodes"]],
                "a node's kind is not a SourceNodeKind",
                "source_graph.nodes.0.kind",
            ),
        ],
        ids=["graph-not-a-dict", "nodes-not-a-list", "node-without-a-kind", "node-with-an-unknown-kind"],
    )
    def test_a_malformed_payload_still_fails_where_it_would_have_failed(self, mutate, shape: str, names: str) -> None:
        """The inverse must not MASK a malformed payload. It walks a shape it
        did not produce, so it guards before indexing -- and those guards have
        to hand the payload onward untouched, letting it fail on its own
        error, rather than swallowing the malformation or raising some new one
        from inside a restoration step nobody asked about.

        Two SHAPE cases and two VALUE cases, because the two fail differently.
        The shape guards return early and touch nothing. A node whose ``kind``
        is missing or is a string no ``SourceNodeKind`` matches gets past those
        guards and DOES have the marker added to it -- the restoration reads
        ``kind`` and finds nothing that says ``figure_crop``. That addition is
        harmless (the node fails on ``kind`` regardless) but it is only
        harmless as long as the resulting error still names the field that was
        actually malformed and not the key this step introduced. Each case
        therefore carries the path it must name, and both halves are asserted:
        the right field named, and ``crop_region`` absent. Asserting only the
        absence would leave "and it still names ``kind``" as a claim in this
        docstring that nothing checks."""
        payload = copy.deepcopy(_maximal_envelope().identity_payload())
        mutate(payload)

        with pytest.raises(DatasetEnvelopeParseError, match="failed validation") as excinfo:
            DatasetEnvelope.from_identity_payload(payload)

        message = str(excinfo.value)
        assert names in message, (
            f"the {shape} case must be reported against {names!r}, the field that is actually malformed; got: {message}"
        )
        assert "crop_region" not in message, (
            f"the {shape} case was reported as a crop_region problem -- the restoration step "
            "rewrote a payload it should have passed through untouched"
        )


def _minimal_envelope_with_member_display_path(member_display_path: str | None) -> DatasetEnvelope:
    """The smallest legal envelope containing an SI_MEMBER node with a
    concrete ArchiveOrigin, parameterized only by `member_display_path` --
    used to prove that field is excluded from identity_payload()'s
    projection (FIX 5): two envelopes differing in exactly this one field
    must produce the same content address."""
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION),
        crop_region=_NO_CROP_REGION,
    )
    si = SourceNode(
        node_id="si",
        kind=SourceNodeKind.SI_MEMBER,
        sha256=SHA_B,
        parent_node_id="paper",
        origin=ArchiveOrigin(archive_sha256=SHA_B, member_display_path=member_display_path),
        extraction=_NO_EXTRACTION,
        glyph_health=_NO_GLYPH_HEALTH,
        verification=_verification_for(_NO_EXTRACTION),
        crop_region=_NO_CROP_REGION,
    )
    graph = SourceGraph(nodes=(paper, si))
    ref = SourceRef(
        node_id="si",
        locator=TableCellLocator(
            # A SHEET cell, so the ref targets the SI node -- which is the whole
            # point here, since the field under test lives on that node's origin.
            # A caption-labelled SI cell would be refused by V8 either way it
            # answered the citation, and pointing this at `paper` instead would
            # orphan `si` and fail on V2 before reaching what this measures.
            table_key=MemberSheetKey(sheet_name="Sheet1"),
            row=0,
            col=0,
            pdf_table_inventory_sha256=_NO_INVENTORY,
        ),
    )
    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=ref,
    )
    sl_axis = AxisDeclaration(
        axis_id="sl", role=AxisRole.OBSERVATION, quantity_kind=QuantityKind.VELOCITY, label_raw="sl", label_ref=ref
    )
    phi_val = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", ref, ref)
    sl_val = _amount("35.0", QuantityKind.VELOCITY, "cm/s", "cm/s", ref, ref)
    coord = Coordinate(axis_id="phi", value=phi_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    obs = Observation(axis_id="sl", value=sl_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    point = DataPoint(
        point_id="p1",
        coordinates=(coord,),
        observations=(obs,),
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis),
        constants=(),
        points=(point,),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        series=(series,),
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for((series,)),
    )


class TestArchiveOriginMemberDisplayPathIsNotIdentity:
    """FIX 5: member_display_path is documented as display-only, never
    identity (see ArchiveOrigin's class docstring) -- but the projection
    used to include it anyway, which meant two envelopes describing the
    exact same archive member could get different content addresses purely
    because of a cosmetic path string. This test proves the contradiction
    is gone: only archive_sha256 (the field that is actually identity)
    affects the address."""

    def test_two_envelopes_differing_only_in_member_display_path_share_one_address(self) -> None:
        env_a = _minimal_envelope_with_member_display_path("si/data.csv")
        env_b = _minimal_envelope_with_member_display_path("SI_TABLES/./data.csv")
        env_c = _minimal_envelope_with_member_display_path(None)

        sha_a = compute_dataset_sha(env_a.identity_payload())
        sha_b = compute_dataset_sha(env_b.identity_payload())
        sha_c = compute_dataset_sha(env_c.identity_payload())

        assert sha_a == sha_b == sha_c
        # And the payloads themselves must not merely hash the same by
        # coincidence -- the field must be genuinely absent from the dict.
        payload_a = env_a.identity_payload()
        assert "member_display_path" not in str(payload_a), (
            "member_display_path leaked into identity_payload() output somewhere"
        )


def _minimal_envelope_with_glyph_health(health: GlyphHealth) -> DatasetEnvelope:
    """The smallest legal envelope containing a paper `SourceNode` with a
    real `ExtractionBinding`/`GlyphHealthAssessment` pair, parameterized only
    by `health` -- used to prove every one of `GlyphHealth`'s five booleans
    is identity-bearing (not decorative): two envelopes differing in exactly
    one boolean must produce different content addresses."""
    # extracted_sha256 is deliberately SHA_F, distinct from the owning paper
    # node's raw sha256 (SHA_A) below -- see _PAPER_EXTRACTION's docstring
    # above for why aliasing the two would hide a raw/extracted digest swap
    # bug undetectably. This fixture is not exercising the binding's own
    # fields (only GlyphHealth's booleans), so it simply reuses the
    # module-level _PAPER_EXTRACTION, whose address is computed -- an
    # ExtractionBinding is self-authenticating and an arbitrary constant
    # address can no longer construct one.
    extraction = _PAPER_EXTRACTION
    glyph_health = GlyphHealthAssessment(
        health=health,
        assessor=SemanticDependencyUse(
            dependency_id=GLYPH_HEALTH_DEPENDENCY_ID,
            content_sha256=current_sha_for(GLYPH_HEALTH_DEPENDENCY_ID),
            input_sha256=SHA_E,
        ),
    )
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=extraction,
        glyph_health=glyph_health,
        verification=_verification_for(extraction),
        crop_region=_NO_CROP_REGION,
    )
    graph = SourceGraph(nodes=(paper,))
    ref = SourceRef(
        node_id="paper",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(label="Table 1"),
            row=0,
            col=0,
            pdf_table_inventory_sha256=_PAPER_INVENTORY.inventory_sha256,
        ),
    )
    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=ref,
    )
    sl_axis = AxisDeclaration(
        axis_id="sl", role=AxisRole.OBSERVATION, quantity_kind=QuantityKind.VELOCITY, label_raw="sl", label_ref=ref
    )
    phi_val = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", ref, ref)
    sl_val = _amount("35.0", QuantityKind.VELOCITY, "cm/s", "cm/s", ref, ref)
    coord = Coordinate(axis_id="phi", value=phi_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    obs = Observation(axis_id="sl", value=sl_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    point = DataPoint(
        point_id="p1",
        coordinates=(coord,),
        observations=(obs,),
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis),
        constants=(),
        points=(point,),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        series=(series,),
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for((series,)),
    )


class TestGlyphHealthAssessmentIsIdentityBearing:
    """The structural extraction/glyph-health binding exists so that "the
    assessed text is THIS node's extracted text" is enforced, not merely
    conventional -- but that guarantee is worthless if the assessment's own
    content (the five `GlyphHealth` booleans, and the assessor's dependency
    binding) doesn't actually affect the envelope's content address. These
    tests prove it does."""

    def test_projection_includes_every_glyph_health_boolean_and_the_assessor(self) -> None:
        env = _maximal_envelope()
        payload = env.identity_payload()
        paper_payload = next(
            node
            for node in payload["source_graph"]["nodes"]
            if node["node_id"] == "paper"  # type: ignore[index]
        )
        glyph_health_payload = paper_payload["glyph_health"]
        health_payload = glyph_health_payload["health"]

        for field in dataclasses.fields(GlyphHealth):
            assert field.name in health_payload, (
                f"GlyphHealth.{field.name} is missing from the projected identity payload -- "
                "an unprojected boolean would be silently non-identity-bearing"
            )
        assert "assessor" in glyph_health_payload

    def test_flipping_any_single_glyph_health_boolean_changes_the_content_address(self) -> None:
        baseline_env = _minimal_envelope_with_glyph_health(_HEALTHY_GLYPH_HEALTH)
        baseline_sha = compute_dataset_sha(baseline_env.identity_payload())

        for field in dataclasses.fields(GlyphHealth):
            flipped = dataclasses.replace(_HEALTHY_GLYPH_HEALTH, **{field.name: True})
            flipped_sha = compute_dataset_sha(_minimal_envelope_with_glyph_health(flipped).identity_payload())
            assert flipped_sha != baseline_sha, (
                f"flipping GlyphHealth.{field.name} alone did not change the content address -- "
                "that boolean is decorative, not identity-bearing"
            )


def _computed_binding(
    extractor: str = "text",
    extractor_code_sha256: str = SHA_G,
    pypdf_version: str | Absent = _NO_PYPDF_VERSION,
) -> ExtractionBinding:
    """Build a self-authenticating `ExtractionBinding` whose
    `extraction_sha256` is COMPUTED from the given identity fields, exactly
    as a producer would compute it -- an arbitrary constant address can no
    longer construct a binding at all."""
    payload_pypdf = pypdf_version if isinstance(pypdf_version, str) else "not-applicable"
    address = compute_extraction_sha(
        _build_identity_payload(
            identity_payload_version="2",
            raw_sha256=SHA_A,
            extractor=extractor,
            extractor_code_sha256=extractor_code_sha256,
            pypdf_version=payload_pypdf,
            extracted_sha256=SHA_F,
            extracted_text_sha256=SHA_E,
        )
    )
    return ExtractionBinding(
        parent_raw_sha256=SHA_A,
        extraction_sha256=address,
        extracted_sha256=SHA_F,
        extracted_text_sha256=SHA_E,
        extractor=extractor,
        extractor_code_sha256=extractor_code_sha256,
        identity_payload_version="2",
        pypdf_version=pypdf_version,
    )


def _minimal_envelope_with_extraction(extraction: ExtractionBinding) -> DatasetEnvelope:
    """Like `_minimal_envelope_with_glyph_health`, but parameterized by the
    paper node's `ExtractionBinding` -- used to prove the binding's identity
    fields are genuinely identity-bearing end to end (a change to the
    extractor identity changes the envelope's content address), and that an
    Absent `pypdf_version` (the non-pypdf-extractor case) round-trips
    through the identity payload."""
    glyph_health = GlyphHealthAssessment(
        health=_HEALTHY_GLYPH_HEALTH,
        assessor=SemanticDependencyUse(
            dependency_id=GLYPH_HEALTH_DEPENDENCY_ID,
            content_sha256=current_sha_for(GLYPH_HEALTH_DEPENDENCY_ID),
            input_sha256=SHA_E,
        ),
    )
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=extraction,
        glyph_health=glyph_health,
        verification=_verification_for(extraction),
        crop_region=_NO_CROP_REGION,
    )
    graph = SourceGraph(nodes=(paper,))
    ref = SourceRef(
        node_id="paper",
        locator=TableCellLocator(
            table_key=CaptionLabelKey(label="Table 1"),
            row=0,
            col=0,
            pdf_table_inventory_sha256=_PAPER_INVENTORY.inventory_sha256,
        ),
    )
    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=ref,
    )
    sl_axis = AxisDeclaration(
        axis_id="sl", role=AxisRole.OBSERVATION, quantity_kind=QuantityKind.VELOCITY, label_raw="sl", label_ref=ref
    )
    phi_val = _amount("1.0", QuantityKind.EQUIVALENCE_RATIO, "-", "1", ref, ref)
    sl_val = _amount("35.0", QuantityKind.VELOCITY, "cm/s", "cm/s", ref, ref)
    coord = Coordinate(axis_id="phi", value=phi_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    obs = Observation(axis_id="sl", value=sl_val, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE))
    point = DataPoint(
        point_id="p1",
        coordinates=(coord,),
        observations=(obs,),
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    series = Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis),
        constants=(),
        points=(point,),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        series=(series,),
        conversion_tables=(_embedded_table_v1(),),
        table_inventories=cover_for((series,)),
    )


class TestExtractionIdentityFieldsAreIdentityBearing:
    """An `ExtractionBinding`'s extractor identity fields (`extractor`,
    `extractor_code_sha256`, `identity_payload_version`, `pypdf_version`)
    are folded into the record's content address AND projected into the
    envelope's identity payload. These tests prove the end-to-end property
    that matters operationally: a change to the extractor identity changes
    the ENVELOPE's content address (never silently aliases two different
    extractions onto one dataset identity), and an explicitly-Absent
    `pypdf_version` (the non-pypdf-extractor case) round-trips through the
    identity payload like every other Maybe[...] field."""

    def test_changing_extractor_code_sha256_changes_the_content_address(self) -> None:
        baseline_env = _minimal_envelope_with_extraction(_computed_binding(extractor_code_sha256=SHA_G))
        baseline_sha = compute_dataset_sha(baseline_env.identity_payload())

        changed_env = _minimal_envelope_with_extraction(_computed_binding(extractor_code_sha256=SHA_D))
        changed_sha = compute_dataset_sha(changed_env.identity_payload())

        assert changed_sha != baseline_sha, (
            "changing ExtractionBinding.extractor_code_sha256 did not change the content address -- "
            "the extractor identity is decorative, not identity-bearing"
        )

        other_env = _minimal_envelope_with_extraction(_computed_binding(extractor_code_sha256=SHA_B))
        other_sha = compute_dataset_sha(other_env.identity_payload())
        assert other_sha != changed_sha, "two different extractor_code_sha256 values produced the same content address"

    def test_absent_pypdf_version_round_trips_and_validates(self) -> None:
        """A non-pypdf extractor's binding carries an explicit
        `Absent(reason=NOT_APPLICABLE)` pypdf_version (the concept does not
        apply -- pypdf never ran). It must be constructible, must project
        into the identity payload the same way every other Maybe[...] field
        does, and must re-validate cleanly."""
        env = _minimal_envelope_with_extraction(_computed_binding(extractor="text"))
        extraction = env.source_graph.nodes[0].extraction
        assert not isinstance(extraction, Absent)
        assert isinstance(extraction.pypdf_version, Absent)
        assert extraction.pypdf_version.reason is AbsenceReason.NOT_APPLICABLE

        payload = env.identity_payload()
        paper_payload = next(
            node
            for node in payload["source_graph"]["nodes"]
            if node["node_id"] == "paper"  # type: ignore[index]
        )
        assert paper_payload["extraction"]["pypdf_version"] == {
            "__absent__": True,
            "note": None,
            "reason": "not_applicable",
        }

        # And round-tripping through the model itself (re-validating the
        # same data) must not raise.
        ExtractionBinding.model_validate(extraction.model_dump())


# --------------------------------------------------------------------------
# 2. Golden canonical bytes
# --------------------------------------------------------------------------
#
# DELIBERATELY A PLACEHOLDER. Per the spec: golden bytes must never be
# produced by calling the (at-write-time nonexistent, and even once it
# exists, untrusted-until-reviewed) implementation and pasting its output --
# that would make this test assert only "the function returns whatever it
# returns", i.e. no test at all.
#
# TO REGENERATE (do this ONCE, deliberately, by hand):
#   1. Confirm identity_payload() is implemented and every other test in
#      this file passes.
#   2. Run, in this environment:
#        python -c "
#        from tests.test_dataset_identity_payload import _maximal_envelope
#        from carmel.services.dataset_store import canonical_json_bytes
#        print(canonical_json_bytes(_maximal_envelope().identity_payload()))
#        "
#   3. READ the printed bytes against the fixture above field-by-field --
#      every key and value must be explainable by this fixture's actual
#      data. Do not paste blindly.
#   4. Replace the placeholder below with the reviewed literal.
#
# Regenerated 2026-08-03 after Composition.components gained a canonical-
# ordering invariant (sorted ascending by species_raw_name, no duplicates --
# see _enforce_no_duplicate_component_species / _enforce_components_sorted_by_species
# on Composition, mirroring the pre-existing S2/S7/E1b axes/points/series
# guards). The fixture's only Composition carries a single component, so
# that change does not alter this payload's shape; the value below was read
# field-by-field against _maximal_envelope() before being pinned here, per
# the instructions above.
#
# Regenerated AGAIN 2026-08-03 (same day, later commit) after
# DatasetEnvelope.identity_payload() started projecting the new
# conversion_tables field (M-D2b(e) commit 2 -- embedding conversion tables
# in the envelope). The fixture now embeds TABLE_V1 via _embedded_table_v1(),
# and the printed bytes were checked field-by-field before being pinned:
# a "conversion_tables" key is present (sorted alphabetically between
# "composition" and "series", as expected of a top-level dict key), it
# contains exactly one embedded table object with "canonical_json" (a JSON
# string, itself containing no floats -- every numeric-looking value in the
# embedded table, e.g. "scale":"0.01", is a string) and "sha256" (matching
# TABLE_V1.sha256), no field anywhere in the payload is a bare float, all
# dict keys at every depth are sorted, and the payload ends in exactly one
# trailing newline.
#
# Regenerated a THIRD time 2026-08-03 (same day, later commit) after two
# changes: (FIX 4) Composition's duplicate-detection and sort key changed
# from `species_raw_name` alone to `(species_raw_name, role)`; and (FIX 5)
# `ArchiveOrigin.member_display_path` was excluded from identity_payload()'s
# projection as display-only, never identity (see _UNADDRESSED_FIELDS and
# _archive_origin_identity_payload). The printed bytes were diffed
# byte-for-byte against the prior golden value before being pinned here:
# the ONLY difference is that `"member_display_path":"si/data.csv"` no
# longer appears inside the `si` node's `origin` object in `source_graph`
# (confirming FIX 5 took effect and nothing else moved). FIX 4 did not
# change this payload's shape, because the fixture's only Composition
# objects each carry a single component (a one-element list is trivially
# sorted and has no duplicate to reject under either key). Re-verified:
# `conversion_tables` is present, no field anywhere is a bare float, all
# dict keys at every depth are sorted, and the payload ends in exactly one
# trailing newline.
#
# Regenerated a FOURTH time 2026-08-03 (same day, M-D2b(f) commit 2) after
# MeasuredValue gained a required repair_dependency: SemanticDependencyUse
# field (recording which version of the numeric-repair heuristic produced
# `repairs`). The printed bytes were diffed key-by-key against the prior
# golden value before being pinned here: the ONLY difference is that every
# MeasuredValue object in the payload now carries an additional
# "repair_dependency" key (sorted alphabetically among that object's other
# keys, since every dict key at every depth of this payload is sorted)
# whose value is {"content_sha256": <current sha for
# carmel.numeric.context_free_span_repair>, "dependency_id":
# "carmel.numeric.context_free_span_repair", "input_sha256":
# {"__absent__": true, "note": null, "reason": "not_applicable"}} --
# identical across all 15 MeasuredValue instances in this fixture, matching
# _CURRENT_REPAIR_DEPENDENCY. Nothing else moved: re-verified `conversion_tables`
# is present, no field anywhere is a bare float, all dict keys at every depth
# are sorted, and the payload ends in exactly one trailing newline.
#
# Regenerated a FIFTH time 2026-08-03 (same day, adversarial-review fix pass on
# M-D2b(f)) after semantic_deps.py's `_CONTEXT_FREE_SPAN_REPAIR_SHA256` was
# re-pinned (module-level import bindings, and `REPAIR_NAMES` as a second
# seeded entry point, were folded into the closure). The dependency's
# content_sha256 is the ONLY thing that changed in this payload: verified by
# recomputing `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and confirming it is byte-for-byte identical to the prior
# golden value with the old sha
# (66fc17740289e3ac72a5b03b328300f9958801dbc3c78a1c50f5f672fe5728a2) replaced
# by the new sha
# (b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb) at all 15
# occurrences -- nothing else in the payload moved.
#
# Regenerated a SIXTH time 2026-08-03 (structural extraction/glyph-health
# binding: `ExtractionBinding`, `GlyphHealthAssessment`, and `SourceNode`'s new
# `extraction`/`glyph_health` fields). The `_maximal_graph()` fixture's `paper`
# `SourceNode` now carries a real `_PAPER_EXTRACTION`/`_PAPER_GLYPH_HEALTH`
# pair instead of an Absent placeholder, and every node (paper/jats/si/crop)
# gained the two new fields at all -- the other three stay Absent
# (NOT_EXTRACTED_YET). Verified by recomputing
# `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently, stripping the new `extraction`/`glyph_health` keys back out
# of every `source_graph.nodes[*]` entry in the recomputed payload, and
# confirming the stripped payload is equal (via `json.loads`) to the prior
# golden value -- i.e. nothing else in the payload moved; the only delta is
# those two new keys appearing on all four nodes.
#
# Regenerated a SEVENTH time 2026-08-03 (adversarial-review fix pass: I5c
# de-alias plus ExtractionBinding.derivation_binding). Two independent
# changes landed together: (1) Defect 3 fix -- `_PAPER_EXTRACTION` (and the
# matching extraction inside `_minimal_envelope_with_glyph_health`) used
# `extracted_sha256=SHA_A` while the owning `SourceNode.sha256` was ALSO
# SHA_A, an alias that would hide a raw/extracted digest swap bug
# undetectably; `extracted_sha256` was repointed to a new, distinct
# constant `SHA_F` (64 'f' characters) so the two fields can never be
# confused by this fixture again. (2) Defect 2 -- `ExtractionBinding`
# gained a new `derivation_binding: Maybe[str]` field (mirroring
# `StoredArtifact.derivation_binding` in carmel/schemas/literature.py),
# projected into `_extraction_binding_identity_payload`; this fixture's
# `_PAPER_EXTRACTION` does not set it, so it defaults to
# Absent(reason=NOT_EXTRACTED_YET). Verified by recomputing
# `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and diffing it key-by-key against the prior golden value:
# the ONLY deltas are (a) the `paper` node's
# `source_graph.nodes[0].extraction.extracted_sha256` changing from 64 'a'
# characters to 64 'f' characters, and (b) a new
# `source_graph.nodes[0].extraction.derivation_binding` key appearing with
# value `{"__absent__": true, "note": null, "reason": "not_extracted_yet"}`
# (sorted alphabetically among `ExtractionBinding`'s other keys, since
# every dict key at every depth of this payload is sorted). Nothing else
# moved: `conversion_tables` is still present, no field anywhere is a bare
# float, all dict keys at every depth are sorted, and the payload ends in
# exactly one trailing newline.
#
# Regenerated an EIGHTH time 2026-08-03 (`ExtractionBinding.derivation_binding`
# became a required field -- no default -- because `NOT_EXTRACTED_YET` falsely
# promised a remedy, re-running extraction, that cannot actually recover this
# field: re-extraction is not byte-reproducible, so the honest reason for a
# legacy/absent value is `UNKNOWN`). `_PAPER_EXTRACTION` in this file now sets
# `derivation_binding=SHA_G` (a new, distinct 64-'1' constant) instead of
# relying on the old implicit `Absent(reason=NOT_EXTRACTED_YET)` default --
# the `paper` node's `extraction` block in `_maximal_graph()` is the maximal
# fixture and should exercise the present branch of every projected field, and
# a real digest here also lets this fixture prove all four digest-bearing
# fields on that node (`sha256`, `extracted_sha256`, `extracted_text_sha256`,
# `derivation_binding`) are pairwise distinct. Verified by recomputing
# `canonical_json_bytes(_maximal_envelope().identity_payload())` independently
# and diffing it key-by-key against the prior (SEVENTH) golden value: the ONLY
# deltas are all under `source_graph.nodes[0].extraction.derivation_binding`,
# changing from the Absent shape
# `{"__absent__": true, "note": null, "reason": "not_extracted_yet"}` to the
# present string `"1111111111111111111111111111111111111111111111111111111111111111"`
# (SHA_G). Nothing else moved: `conversion_tables` is still present, no field
# anywhere is a bare float, all dict keys at every depth are sorted, and the
# payload ends in exactly one trailing newline.
# Regenerated a NINTH time 2026-08-04 (`ExtractionBinding` became
# self-authenticating; `derivation_binding` deleted). The printed bytes were
# diffed key-by-key against the prior golden value before being pinned here:
# the ONLY changes are inside `source_graph.nodes[0].extraction` --
# `derivation_binding` is gone (the field was deleted from the schema; it
# proved only that the root sidecar agreed with itself), `extraction_sha256`
# is now the COMPUTED content address of the binding's own identity payload
# (a binding recomputes its address at construction, so an arbitrary
# constant can no longer exist), and four identity fields were added:
# `extractor` ("pdf:pypdf"), `extractor_code_sha256` (SHA_G, the same 64-'1'
# constant `derivation_binding` used to hold), `identity_payload_version`
# ("2"), and `pypdf_version` ("9.9.9-synthetic", a deliberately fake version
# so this golden value cannot drift with the environment's pypdf install).
# Nothing else moved: every other line of the key-by-key diff was empty, all
# dict keys at every depth remain sorted, no field anywhere is a bare float,
# and the payload still ends in exactly one trailing newline.
#
# Regenerated a TENTH time 2026-08-04 (SourceNode gained its I6 invariant:
# a FIGURE_CROP's `extraction`/`glyph_health` must both be
# Absent(reason=NOT_APPLICABLE), never NOT_EXTRACTED_YET, since an image
# region has no extracted text to bind or assess). This file's `crop` node
# fixture already used `_NO_EXTRACTION_CROP`/`_NO_GLYPH_HEALTH_CROP`
# (reason=NOT_APPLICABLE), so the recomputed bytes changed to match without
# any fixture edit here. Verified by recomputing
# `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and diffing it byte-by-byte against the prior (NINTH) golden
# value: the ONLY deltas are `source_graph.nodes[3].extraction.reason` and
# `source_graph.nodes[3].glyph_health.reason` (the `crop` node) each changing
# from `"not_extracted_yet"` to `"not_applicable"`. Nothing else moved: every
# other byte of the diff was identical, all dict keys at every depth remain
# sorted, no field anywhere is a bare float, and the payload still ends in
# exactly one trailing newline.
#
# Regenerated an ELEVENTH time 2026-08-06 (SourceNode gained `verification`, a
# per-tier SourceVerification recording what production ACTUALLY verified about
# each node -- raw artifact, extracted text, root sidecar -- folded into the
# identity payload on purpose so an envelope's content address COMMITS to the
# standard it claims; left out, a record-only envelope and a fully root-verified
# one would address identically and the claim could be swapped for free).
# Verified by recomputing `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and diffing it key-by-key against the prior (TENTH) golden value:
# the ONLY deltas are four ADDITIONS, one `verification` key per node, and every
# one of them follows that node's own `extraction` exactly as SourceNode's
# iff-rule requires -- `paper` (which carries a binding) gains the concrete
# claim, `jats`/`si` gain Absent(not_extracted_yet), and `crop` gains
# Absent(not_applicable). Nothing else moved: no existing key changed value or
# position, all dict keys at every depth remain sorted, no field anywhere is a
# bare float, and the payload still ends in exactly one trailing newline.
#
# Regenerated a TWELFTH time 2026-08-06 (`RootSidecarVerification.NOT_CHECKED`
# REPLACED by `ROOT_SIDECAR_DIGEST_AUTHENTICATED`, so that every member of that
# enum asserts a fact about the STORE and is therefore refutable by replay --
# `NOT_CHECKED` described a producer CHOICE, which no consumer could ever
# contradict, and an unfalsifiable claim in persisted evidence is
# indistinguishable from no claim while still reading like provenance).
# Verified by recomputing `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and diffing key-by-key against the prior (ELEVENTH) golden: the
# ONLY delta is `source_graph.nodes[0].verification.root_sidecar` changing value
# from "not_checked" to "root_sidecar_digest_authenticated". No key was added,
# removed or moved; the three Absent verification entries on nodes[1..3] are
# untouched; all dict keys at every depth remain sorted; no field anywhere is a
# bare float; the payload still ends in exactly one trailing newline.
#
# Regenerated a THIRTEENTH time 2026-08-06, under the repository owner's
# EXPLICIT, ONE-TIME approval to override this file's standing
# never-regenerate rule -- because this time the address move IS the fix,
# not the failure. Two projection defects were closed in one change, both
# landed BEFORE anything was ever stored under this schema (the branch is
# unreleased), which is exactly why moving every address now was cheap and
# deliberate:
#   (1) the projection became CANONICAL -- `_source_graph_identity_payload`
#       now emits `source_graph.nodes` sorted ascending by node_id, because
#       `SourceGraph.nodes` is semantically a SET whose tuple order carries
#       no meaning, and projecting tuple order verbatim gave ONE graph MANY
#       content addresses (verified empirically before the fix: the same
#       maximal envelope with its node tuple reversed produced a different
#       compute_dataset_sha). The maximal fixture's node tuple was reordered
#       to (crop, jats, paper, si) -- already-sorted -- so the positional
#       completeness walker stays aligned with the sorted projection.
#   (2) the payload became SELF-DESCRIBING -- two new top-level keys,
#       `envelope_type` ("dataset") and `identity_payload_version` (1), so
#       the stored bytes say what they are and `from_identity_payload` can
#       refuse a condition-set payload as a wrong-type payload instead of
#       relying on field-shape accident.
# Verified by recomputing `canonical_json_bytes(_maximal_envelope().identity_payload())`
# independently and diffing it key-by-key (json.loads, per top-level key)
# against the prior (TWELFTH) golden value: the ONLY deltas are (a) the two
# new top-level keys above and (b) `source_graph.nodes` reordered from
# [paper, jats, si, crop] to [crop, jats, paper, si], with every node's own
# sub-payload identical modulo that order (checked as a node_id-keyed dict
# equality). Nothing else moved: all dict keys at every depth remain sorted,
# no field anywhere is a bare float, and the payload still ends in exactly
# one trailing newline.
#
# A future reader must NOT treat this entry as precedent: the rule stands
# that a SURPRISE failure of the golden assertion means an address moved and
# the change must be REVERTED, not re-pinned. This regeneration was
# authorised in advance, for this one change only, precisely because no
# stored artifact existed yet for the move to orphan.
#
# FOURTEENTH regeneration, 2026-08-19: DatasetEnvelope's projection changed on
# purpose. TableCellLocator gained a required `pdf_table_inventory_sha256` and
# the envelope gained `table_inventories`, embedding every inventory a table cell
# cites (V8/T4/T5 -- see carmel/schemas/datasets.py). Verified by diffing the
# recomputed payload against the prior golden key-by-key: the ONLY deltas are the
# new top-level `table_inventories` key and the new locator field inside
# `composition`/`series`, both of which compare EQUAL to the old golden once that
# one field is stripped. No TABLE_CELL locator has ever been emitted by production
# code, so this address move has nothing to migrate.
#
# FIFTEENTH regeneration, 2026-08-21: SourceNode gained `crop_region`, a
# FIGURE_CROP's addressing identity (I7/I8/I9 -- see carmel/schemas/datasets.py),
# and every node projects it. Verified by diffing the recomputed payload against
# the prior golden key-by-key: no top-level key was added or removed, the node id
# set is unchanged, and each of the four nodes differs by EXACTLY the one new
# `crop_region` key -- a concrete bbox on 'crop', an Absent(not_applicable) marker
# on 'paper', 'jats' and 'si' -- with nothing else added, removed or changed. This
# IS an address move and it is BREAKING for anything already stored under the old
# projection; the only production emitter of a SourceGraph today
# (`carmel/services/dataset_producer.py`) writes a single PAPER_PDF root and no
# crop at all, so nothing this repository produces is orphaned by it.
#
# SIXTEENTH regeneration, 2026-08-21, superseding the fifteenth WITHIN the same
# branch after review, on two counts. (a) `identity_payload_version` 1 -> 2: the
# fifteenth changed the projection schema for every node while leaving the version
# key at 1, so a pre-change and a post-change payload were two different schemas
# both stamped 1 -- the discriminator waved the old one through and pydantic
# answered with `crop_region Field required` once per node instead of one message
# naming the mismatch. (b) `crop_region` is now emitted for a FIGURE_CROP ONLY: on
# any other kind I7 admits exactly one value, so the key was a constant carrying no
# distinguishing bits, bought at the price of re-addressing every envelope in every
# store including crop-free ones.
#
# Diffed key-by-key twice. Against the fifteenth pin: `identity_payload_version`
# 1 -> 2, and `crop_region` removed from 'paper'/'jats'/'si' with the 'crop' node
# byte-identical -- nothing else added, removed or changed. Against the ORIGINAL
# pre-branch pin at 499835c: the version bump, plus exactly one key added on the
# 'crop' node -- every other node projects byte-identically to what it did before
# this branch existed. That is the shape the change should always have had.
# Still BREAKING (the version key alone re-addresses everything, deliberately and
# visibly), and still nothing stored to orphan.
_GOLDEN_CANONICAL_BYTES = b'{"composition":{"basis":"mole_fraction","components":[{"amount":{"canonical_decimal_value":"0.04","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"mole_fraction","raw_text":"0.04","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"role":"fuel","species_raw_name":"H2"}],"equivalence_ratio":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}},"raw_name":"4% H2 in N2","resolution":"resolved_components"},"conversion_tables":[{"canonical_json":"{\\"aliases\\":[{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"\xc2\xb0C\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"degC\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"deg C\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s^-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm/sec\\"},{\\"normalized\\":\\"m/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"m s^-1\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cm^3\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cc\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"l\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"liter\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"litre\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xc2\xb5s\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xce\xbcs\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"ppm\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"ppmv\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"dimensionless\\"}],\\"base_units\\":[[\\"length\\",\\"m\\"],[\\"velocity\\",\\"m/s\\"],[\\"temperature\\",\\"K\\"],[\\"pressure\\",\\"Pa\\"],[\\"time\\",\\"s\\"],[\\"volume\\",\\"m3\\"],[\\"strain_rate\\",\\"1/s\\"],[\\"mole_fraction\\",\\"1\\"],[\\"mass_fraction\\",\\"1\\"],[\\"equivalence_ratio\\",\\"1\\"],[\\"relative_uncertainty\\",\\"1\\"]],\\"rules\\":[{\\"kind\\":\\"identity\\",\\"quantity\\":\\"length\\",\\"unit\\":\\"m\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"velocity\\",\\"unit\\":\\"m/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"temperature\\",\\"unit\\":\\"K\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"pressure\\",\\"unit\\":\\"Pa\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"time\\",\\"unit\\":\\"s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"volume\\",\\"unit\\":\\"m3\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"strain_rate\\",\\"unit\\":\\"1/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mole_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mass_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"equivalence_ratio\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"relative_uncertainty\\",\\"unit\\":\\"1\\"},{\\"from_unit\\":\\"cm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"mm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"cm/s\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"velocity\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m/s\\"},{\\"from_unit\\":\\"C\\",\\"kind\\":\\"affine\\",\\"offset\\":\\"273.15\\",\\"quantity\\":\\"temperature\\",\\"scale\\":\\"1\\",\\"to_unit\\":\\"K\\"},{\\"from_unit\\":\\"atm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"101325\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"bar\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"100000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"kPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"MPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"ms\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"us\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"cm3\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"L\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"relative_uncertainty\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"}],\\"table_id\\":\\"carmel-unit-conversions\\",\\"version\\":1}\\n","sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122"}],"envelope_type":"dataset","identity_payload_version":2,"series":[{"axes":[{"axis_id":"phi","label_raw":"phi","label_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"},"quantity_kind":"equivalence_ratio","role":"coordinate"},{"axis_id":"sl","label_raw":"S_L","label_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"quantity_kind":"velocity","role":"observation"},{"axis_id":"temperature","label_raw":"T","label_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"quantity_kind":"temperature","role":"constant"}],"constants":[{"axis_id":"temperature","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}},"value":{"canonical_decimal_value":"298","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"298","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"end":20,"kind":"char_span","start":10,"text_space":"extracted_text"},"node_id":"paper"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}}],"points":[{"composition":{"basis":"mole_fraction","components":[{"amount":{"canonical_decimal_value":"0.04","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"mole_fraction","raw_text":"0.04","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"role":"fuel","species_raw_name":"H2"}],"equivalence_ratio":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}},"raw_name":"4% H2 in N2","resolution":"resolved_components"},"coordinates":[{"axis_id":"phi","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}}},"value":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}}}],"observations":[{"axis_id":"sl","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"},"value_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"0.1","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}},"value":{"canonical_decimal_value":"35.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"35.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":2,"kind":"table_cell","pdf_table_inventory_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"},"row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","pdf_table_inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"paper"}}}],"point_id":"p1"}],"series_id":"s1","source_form":"tabular","value_origin":"experimental"},{"axes":[{"axis_id":"phi2","label_raw":"phi2","label_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"quantity_kind":"equivalence_ratio","role":"coordinate"},{"axis_id":"sl2","label_raw":"sl2","label_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"quantity_kind":"velocity","role":"observation"}],"constants":[],"points":[{"composition":{"__absent__":true,"note":null,"reason":"not_applicable"},"coordinates":[{"axis_id":"phi2","uncertainty":{"__absent__":true,"note":null,"reason":"not_reported_here"},"value":{"canonical_decimal_value":"2.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"2.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"value_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"}}}],"observations":[{"axis_id":"sl2","uncertainty":{"__absent__":true,"note":null,"reason":"not_reported_here"},"value":{"canonical_decimal_value":"10.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"10.0","repair_dependency":{"content_sha256":"b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb","dependency_id":"carmel.numeric.context_free_span_repair","input_sha256":{"__absent__":true,"note":null,"reason":"not_applicable"}},"repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"value_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"}}}],"point_id":"q1"}],"series_id":"s2","source_form":"textual","value_origin":"simulation"}],"source_graph":{"nodes":[{"crop_region":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"extraction":{"__absent__":true,"note":null,"reason":"not_applicable"},"glyph_health":{"__absent__":true,"note":null,"reason":"not_applicable"},"kind":"figure_crop","node_id":"crop","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":"paper","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","verification":{"__absent__":true,"note":null,"reason":"not_applicable"}},{"extraction":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"glyph_health":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"kind":"jats_xml","node_id":"jats","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":null,"sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","verification":{"__absent__":true,"note":null,"reason":"not_extracted_yet"}},{"extraction":{"extracted_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","extracted_text_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","extraction_sha256":"5291844e7bedab416755e826cbdf2b34283de1753c7ef2a0fcae20f0dc5c2529","extractor":"pdf:pypdf","extractor_code_sha256":"1111111111111111111111111111111111111111111111111111111111111111","identity_payload_version":"2","parent_raw_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pypdf_version":"9.9.9-synthetic"},"glyph_health":{"assessor":{"content_sha256":"af3553a8142b50bba56b6ba164778b4cd2bff6e4916ac2e93c4e1a270ba4ab5a","dependency_id":"carmel.numeric.glyph_health","input_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},"health":{"has_ascii6_uncertainty_marker":false,"has_equals_ambiguity_marker":false,"has_slash_c0_minus_marker":false,"has_thorn_plus_marker":false,"suspects_dash_corruption":false}},"kind":"paper_pdf","node_id":"paper","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":null,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verification":{"extracted_text":"extraction_record_digest_authenticated","raw_artifact":"raw_sha256_digest_authenticated","root_sidecar":"root_sidecar_digest_authenticated"}},{"extraction":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"glyph_health":{"__absent__":true,"note":null,"reason":"not_extracted_yet"},"kind":"si_member","node_id":"si","origin":{"archive_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"parent_node_id":"paper","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","verification":{"__absent__":true,"note":null,"reason":"not_extracted_yet"}}]},"table_inventories":[{"canonical_json":"{\\"cells\\":[{\\"col\\":0,\\"member_digests\\":[\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\"],\\"row\\":0,\\"text\\":\\"r0c0\\",\\"x_end\\":\\"0x1.2000000000000p+3\\",\\"x_start\\":\\"0x0.0p+0\\"},{\\"col\\":1,\\"member_digests\\":[\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\"],\\"row\\":0,\\"text\\":\\"r0c1\\",\\"x_end\\":\\"0x1.3000000000000p+4\\",\\"x_start\\":\\"0x1.4000000000000p+3\\"}],\\"column_bounds\\":[[\\"0x0.0p+0\\",\\"0x1.f400000000000p+8\\"]],\\"footprint\\":{\\"caption_baseline_y\\":\\"0x1.5e00000000000p+9\\",\\"caption_text\\":\\"Table 1. A fixture, not a table.\\",\\"caption_x_start\\":\\"0x1.2000000000000p+6\\",\\"page\\":0,\\"x_end\\":\\"0x1.f400000000000p+8\\",\\"x_start\\":\\"0x1.2000000000000p+6\\",\\"y_bottom\\":\\"0x1.9000000000000p+7\\",\\"y_top\\":\\"0x1.5900000000000p+9\\"},\\"fragment_geometry_sha256\\":\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\",\\"inventory_code_sha256\\":\\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\\",\\"payload_version\\":1,\\"pypdf_version\\":\\"0.0.0-fixture\\",\\"raw_sha256\\":\\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\",\\"refusals\\":[],\\"rows\\":[{\\"anchor_text\\":\\"row 0\\",\\"anchor_x_start\\":\\"0x1.2000000000000p+6\\",\\"baseline_y\\":\\"0x1.2c00000000000p+9\\",\\"merged_baselines\\":[],\\"ordinal\\":0}]}\\n","inventory_sha256":"dc43aedbaeba2837ab1faa6c934fbb8610026e2c306d3015ef20c95dca8a48cc","raw_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}\n'  # noqa: E501


class TestGoldenCanonicalBytes:
    def test_maximal_envelope_canonical_bytes_match_pinned_golden(self) -> None:
        env = _maximal_envelope()
        actual = canonical_json_bytes(env.identity_payload())
        assert actual == _GOLDEN_CANONICAL_BYTES, (
            "canonical bytes for the maximal fixture no longer match the pinned golden value -- "
            "if this is an intentional projection change, regenerate the golden constant by hand "
            "(see the comment above _GOLDEN_CANONICAL_BYTES) after reading this diff; if it is not, "
            "identity_payload()'s output has silently drifted"
        )


# --------------------------------------------------------------------------
# 3. Recursive no-float assertion
# --------------------------------------------------------------------------


def _assert_no_floats(value: object, path: str) -> None:
    if isinstance(value, float):
        pytest.fail(
            f"{path}: found a float ({value!r}) in identity_payload()'s output -- floats are never legal at any depth"
        )
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_no_floats(v, f"{path}[{i}]")


class TestNoFloatsAnywhere:
    def test_identity_payload_contains_no_floats_at_any_depth(self) -> None:
        env = _maximal_envelope()
        _assert_no_floats(env.identity_payload(), "payload")


# --------------------------------------------------------------------------
# 4. Determinism across differently-ordered equal inputs
# --------------------------------------------------------------------------
#
# JUDGMENT CALL (flagged per instructions rather than silently resolved):
# the spec's phrase "differently-ordered (but equal) inputs" turned out, on
# empirical trial, to have NO safe realization as a literal list-reordering
# test anywhere in this schema:
#   - Reordering `DatasetEnvelope.series` is rejected outright --
#     "series must be sorted ascending by series_id" is an enforced
#     invariant, so two envelopes differing only in series order are not
#     both legal; there is no "equal-but-reordered" pair to construct.
#   - Reordering `Composition.components` is ALSO rejected outright, as of
#     the fix that added the missing sort-order guard for `components`
#     (matching the S2/S7/E1b idiom for `axes`/`points`/`series`) -- so, like
#     `series`, there is no "equal-but-reordered" pair to construct here
#     either.
# Both trials are consistent with this schema deliberately treating
# sequence order as enforced-canonical everywhere it appears, rather than
# leaving room for incidental reordering. Given that, the only
# "differently-ordered (but equal) inputs" case this test
# module can honestly construct is two independently-built object graphs
# that encode the identical logical dataset (fresh instances, built via
# separate calls) -- which must produce identical bytes for identity_payload
# to be a well-defined content address at all. Repeated-call determinism on
# a single instance is the same claim in its simplest form.


class TestDeterminism:
    def test_repeated_calls_on_the_same_envelope_produce_identical_bytes(self) -> None:
        env = _maximal_envelope()
        first = canonical_json_bytes(env.identity_payload())
        second = canonical_json_bytes(env.identity_payload())
        assert first == second

    def test_two_independently_built_envelopes_encoding_the_same_dataset_produce_identical_bytes(self) -> None:
        env_a = _maximal_envelope()
        env_b = _maximal_envelope()
        assert env_a is not env_b
        assert env_a.source_graph is not env_b.source_graph

        bytes_a = canonical_json_bytes(env_a.identity_payload())
        bytes_b = canonical_json_bytes(env_b.identity_payload())
        assert bytes_a == bytes_b


# --------------------------------------------------------------------------
# 5. Isolation
# --------------------------------------------------------------------------


class TestIsolation:
    def test_mutating_the_returned_dict_does_not_affect_the_envelope_or_a_later_call(self) -> None:
        env = _maximal_envelope()
        payload = env.identity_payload()

        payload["composition"] = "MUTATED"
        if isinstance(payload.get("series"), list) and payload["series"]:
            payload["series"].append("MUTATED")
            if isinstance(payload["series"][0], dict):
                payload["series"][0]["series_id"] = "MUTATED"

        second = env.identity_payload()
        assert second["composition"] != "MUTATED"
        assert "MUTATED" not in second["series"]
        assert second["series"][0]["series_id"] != "MUTATED"
        # And the pydantic model itself must be unaffected (frozen=True should
        # already guarantee this at the model level; this checks the bridge
        # method didn't hand out a live reference into model-internal state).
        assert env.composition != "MUTATED"


# --------------------------------------------------------------------------
# 6. The model_dump tripwire
# --------------------------------------------------------------------------


class TestModelDumpTripwire:
    """Not a coverage test -- a tripwire. Pins two concrete facts about what
    BaseModel.model_dump() (the forbidden implementation strategy named in
    identity_payload()'s docstring) actually returns for this schema, so
    that a future edit that "simplifies" identity_payload() to
    `return self.model_dump()` is caught: model_dump()'s default (python)
    mode keeps enum members as Enum instances (not their `.value` string)
    and keeps tuple-typed fields as tuples (not lists) -- both the opposite
    of identity_payload()'s required contract.
    """

    def test_model_dump_default_mode_keeps_enums_and_tuples_unlike_identity_payload(self) -> None:
        env = _maximal_envelope()
        dumped = env.model_dump()

        assert isinstance(dumped["series"], tuple), "expected model_dump() to keep a tuple-typed field as a tuple"
        assert isinstance(dumped["series"][0]["source_form"], SourceForm), (
            "expected model_dump() to keep an enum field as an Enum instance, not its .value"
        )

        payload = env.identity_payload()
        assert isinstance(payload["series"], list), "identity_payload() must project tuples as lists"
        assert isinstance(payload["series"][0]["source_form"], str), "identity_payload() must project enums as .value"
        assert payload["series"][0]["source_form"] == SourceForm.TABULAR.value


# --------------------------------------------------------------------------
# 7. Absence is explicit and distinguishable
# --------------------------------------------------------------------------


class TestAbsenceIsDistinguishable:
    def test_absent_and_present_composition_produce_different_canonical_bytes(self) -> None:
        absent_env = _minimal_envelope_with_composition(Absent(reason=AbsenceReason.NOT_APPLICABLE))
        present_composition = Composition(
            raw_name="x",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            components=[],
        )
        present_env = _minimal_envelope_with_composition(present_composition)

        absent_bytes = canonical_json_bytes(absent_env.identity_payload())
        present_bytes = canonical_json_bytes(present_env.identity_payload())
        assert absent_bytes != present_bytes

    def test_absent_composition_projection_carries_the_reason_and_note(self) -> None:
        env = _minimal_envelope_with_composition(
            Absent(reason=AbsenceReason.CONFLICTING_SOURCES, note="two tables disagree")
        )
        payload = env.identity_payload()
        composition_projection = payload["composition"]
        assert isinstance(composition_projection, dict)
        # The exact key names are an implementation detail this test does not
        # pin (only the golden-bytes test does that); what must hold here is
        # that the reason's *value* and the note text both survive somewhere
        # in the Absent projection, and that it is NOT shaped like a present
        # Composition (which has no top-level "reason" concept at all).
        flattened = str(composition_projection)
        assert AbsenceReason.CONFLICTING_SOURCES.value in flattened
        assert "two tables disagree" in flattened


# --------------------------------------------------------------------------
# 8. Accepted downstream
# --------------------------------------------------------------------------


class TestAcceptedDownstream:
    def test_canonical_json_bytes_succeeds_on_the_maximal_envelope(self) -> None:
        env = _maximal_envelope()
        raw = canonical_json_bytes(env.identity_payload())
        assert isinstance(raw, bytes)

    def test_compute_dataset_sha_returns_a_64_hex_digest(self) -> None:
        env = _maximal_envelope()
        digest = compute_dataset_sha(env.identity_payload())
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"expected a 64-hex sha256 digest, got {digest!r}"


# --------------------------------------------------------------------------
# 9. Unreachable-by-normal-flow union-arm branches in the projection helpers
# --------------------------------------------------------------------------
#
# `_table_key_identity_payload` and `_source_locator_identity_payload` each
# dispatch over a discriminated union of model types via `isinstance` and end
# in a `TypeError` branch that no *pydantic-validated* instance can reach --
# but the helpers themselves are plain module-level functions with no
# pydantic enforcement of their own, so they CAN be called directly with an
# object of a type outside the union they handle. These tests do exactly
# that, to retire each function's `# pragma: no cover` honestly rather than
# by inspection alone.


class TestUnhandledUnionVariantsRaiseTypeError:
    def test_table_key_identity_payload_rejects_an_unhandled_variant(self) -> None:
        from carmel.schemas.datasets import _table_key_identity_payload  # type: ignore[attr-defined]

        with pytest.raises(TypeError, match="_table_key_identity_payload: unhandled TableKey variant"):
            _table_key_identity_payload(object())  # type: ignore[arg-type]

    def test_source_locator_identity_payload_rejects_an_unhandled_variant(self) -> None:
        from carmel.schemas.datasets import _source_locator_identity_payload  # type: ignore[attr-defined]

        with pytest.raises(TypeError, match="_source_locator_identity_payload: unhandled SourceLocator variant"):
            _source_locator_identity_payload(object())  # type: ignore[arg-type]
