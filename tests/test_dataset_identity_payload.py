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
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    Coordinate,
    CoordinateFrame,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    MeasuredValue,
    MemberSheetKey,
    Observation,
    QuantityKind,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    ValueOrigin,
    XPathLocator,
)
from carmel.services.dataset_store import canonical_json_bytes, compute_dataset_sha
from carmel.services.units import TABLE_V1

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


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
# --------------------------------------------------------------------------


def _maximal_graph() -> tuple[SourceGraph, SourceRef, SourceRef, SourceRef, SourceRef]:
    paper = SourceNode(
        node_id="paper",
        kind=SourceNodeKind.PAPER_PDF,
        sha256=SHA_A,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    jats = SourceNode(
        node_id="jats",
        kind=SourceNodeKind.JATS_XML,
        sha256=SHA_D,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    si = SourceNode(
        node_id="si",
        kind=SourceNodeKind.SI_MEMBER,
        sha256=SHA_B,
        parent_node_id="paper",
        origin=ArchiveOrigin(archive_sha256=SHA_B, member_display_path="si/data.csv"),
    )
    crop = SourceNode(
        node_id="crop",
        kind=SourceNodeKind.FIGURE_CROP,
        sha256=SHA_C,
        parent_node_id="paper",
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    graph = SourceGraph(nodes=(paper, jats, si, crop))

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
    bbox_ref = SourceRef(node_id="crop", locator=BBoxLocator(bbox=bbox))
    table_ref_caption = SourceRef(
        node_id="si", locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=1)
    )
    table_ref_sheet = SourceRef(
        node_id="si", locator=TableCellLocator(table_key=MemberSheetKey(sheet_name="Sheet1"), row=1, col=2)
    )
    xpath_ref = SourceRef(node_id="jats", locator=XPathLocator(xpath="//table/row[1]/cell[1]"))
    return graph, bbox_ref, table_ref_caption, table_ref_sheet, xpath_ref


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
    graph, bbox_ref, table_ref_caption, table_ref_sheet, xpath_ref = _maximal_graph()

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

    t_val = _amount("298", QuantityKind.TEMPERATURE, "K", "K", table_ref_caption, table_ref_sheet)
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
    )
    graph = SourceGraph(nodes=(paper,))
    ref = SourceRef(node_id="paper", locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0))
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
        from carmel.schemas.datasets import _UNADDRESSED_FIELDS  # type: ignore[attr-defined]

        model_name = type(value).__name__
        assert isinstance(
            projected, dict
        ), f"{path}: expected a dict projection of {model_name}, got {type(projected)!r}"
        for name in type(value).model_fields:
            key = (model_name, name)
            if key in _UNADDRESSED_FIELDS:
                assert _UNADDRESSED_FIELDS[
                    key
                ], f"{path}.{name}: registered in _UNADDRESSED_FIELDS with an empty reason"
                continue
            assert name in projected, (
                f"{path}.{name}: field {name!r} of {model_name} is missing from identity_payload()'s output "
                f"-- it is neither projected nor registered in _UNADDRESSED_FIELDS"
            )
            _walk_completeness(getattr(value, name), projected[name], f"{path}.{name}")
    elif isinstance(value, Absent):
        assert isinstance(
            projected, dict
        ), f"{path}: an Absent value must project to a dict shape, got {type(projected)!r}"
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
        assert (
            projected == value.value
        ), f"{path}: enum {value!r} must project as its .value ({value.value!r}), got {projected!r}"
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
    )
    si = SourceNode(
        node_id="si",
        kind=SourceNodeKind.SI_MEMBER,
        sha256=SHA_B,
        parent_node_id="paper",
        origin=ArchiveOrigin(archive_sha256=SHA_B, member_display_path=member_display_path),
    )
    graph = SourceGraph(nodes=(paper, si))
    ref = SourceRef(node_id="si", locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0))
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
_GOLDEN_CANONICAL_BYTES = b'{"composition":{"basis":"mole_fraction","components":[{"amount":{"canonical_decimal_value":"0.04","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"mole_fraction","raw_text":"0.04","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"role":"fuel","species_raw_name":"H2"}],"equivalence_ratio":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}},"raw_name":"4% H2 in N2","resolution":"resolved_components"},"conversion_tables":[{"canonical_json":"{\\"aliases\\":[{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"\xc2\xb0C\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"degC\\"},{\\"normalized\\":\\"C\\",\\"quantity\\":\\"temperature\\",\\"raw\\":\\"deg C\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s^-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm s-1\\"},{\\"normalized\\":\\"cm/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"cm/sec\\"},{\\"normalized\\":\\"m/s\\",\\"quantity\\":\\"velocity\\",\\"raw\\":\\"m s^-1\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cm^3\\"},{\\"normalized\\":\\"cm3\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"cc\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"l\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"liter\\"},{\\"normalized\\":\\"L\\",\\"quantity\\":\\"volume\\",\\"raw\\":\\"litre\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xc2\xb5s\\"},{\\"normalized\\":\\"us\\",\\"quantity\\":\\"time\\",\\"raw\\":\\"\xce\xbcs\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"ppm\\",\\"quantity\\":\\"mole_fraction\\",\\"raw\\":\\"ppmv\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"mass_fraction\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"equivalence_ratio\\",\\"raw\\":\\"dimensionless\\"},{\\"normalized\\":\\"%\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"percent\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"-\\"},{\\"normalized\\":\\"1\\",\\"quantity\\":\\"relative_uncertainty\\",\\"raw\\":\\"dimensionless\\"}],\\"base_units\\":[[\\"length\\",\\"m\\"],[\\"velocity\\",\\"m/s\\"],[\\"temperature\\",\\"K\\"],[\\"pressure\\",\\"Pa\\"],[\\"time\\",\\"s\\"],[\\"volume\\",\\"m3\\"],[\\"strain_rate\\",\\"1/s\\"],[\\"mole_fraction\\",\\"1\\"],[\\"mass_fraction\\",\\"1\\"],[\\"equivalence_ratio\\",\\"1\\"],[\\"relative_uncertainty\\",\\"1\\"]],\\"rules\\":[{\\"kind\\":\\"identity\\",\\"quantity\\":\\"length\\",\\"unit\\":\\"m\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"velocity\\",\\"unit\\":\\"m/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"temperature\\",\\"unit\\":\\"K\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"pressure\\",\\"unit\\":\\"Pa\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"time\\",\\"unit\\":\\"s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"volume\\",\\"unit\\":\\"m3\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"strain_rate\\",\\"unit\\":\\"1/s\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mole_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"mass_fraction\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"equivalence_ratio\\",\\"unit\\":\\"1\\"},{\\"kind\\":\\"identity\\",\\"quantity\\":\\"relative_uncertainty\\",\\"unit\\":\\"1\\"},{\\"from_unit\\":\\"cm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"mm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"length\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m\\"},{\\"from_unit\\":\\"cm/s\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"velocity\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"m/s\\"},{\\"from_unit\\":\\"C\\",\\"kind\\":\\"affine\\",\\"offset\\":\\"273.15\\",\\"quantity\\":\\"temperature\\",\\"scale\\":\\"1\\",\\"to_unit\\":\\"K\\"},{\\"from_unit\\":\\"atm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"101325\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"bar\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"100000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"kPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"MPa\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"pressure\\",\\"scale\\":\\"1000000\\",\\"to_unit\\":\\"Pa\\"},{\\"from_unit\\":\\"ms\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"us\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"time\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"s\\"},{\\"from_unit\\":\\"cm3\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"L\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"volume\\",\\"scale\\":\\"0.001\\",\\"to_unit\\":\\"m3\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"%\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"relative_uncertainty\\",\\"scale\\":\\"0.01\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mole_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"},{\\"from_unit\\":\\"ppm\\",\\"kind\\":\\"scale\\",\\"quantity\\":\\"mass_fraction\\",\\"scale\\":\\"0.000001\\",\\"to_unit\\":\\"1\\"}],\\"table_id\\":\\"carmel-unit-conversions\\",\\"version\\":1}\\n","sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122"}],"series":[{"axes":[{"axis_id":"phi","label_raw":"phi","label_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"},"quantity_kind":"equivalence_ratio","role":"coordinate"},{"axis_id":"sl","label_raw":"S_L","label_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"quantity_kind":"velocity","role":"observation"},{"axis_id":"temperature","label_raw":"T","label_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"quantity_kind":"temperature","role":"constant"}],"constants":[{"axis_id":"temperature","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"0.1","repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"0.1","repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}}},"value":{"canonical_decimal_value":"298","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"temperature","raw_text":"298","repairs":[],"unit_normalized":"K","unit_raw":"K","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}}}],"points":[{"composition":{"basis":"mole_fraction","components":[{"amount":{"canonical_decimal_value":"0.04","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"mole_fraction","raw_text":"0.04","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"role":"fuel","species_raw_name":"H2"}],"equivalence_ratio":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}},"raw_name":"4% H2 in N2","resolution":"resolved_components"},"coordinates":[{"axis_id":"phi","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"0.1","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"0.1","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}}},"value":{"canonical_decimal_value":"1.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"1.0","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"bbox":{"frame":{"cropbox":["0","0","612","792"],"dpi":"300","mediabox":["0","0","612","792"],"render_fingerprint":"fp-1","render_settings":"antialias=on","rotation":0,"units":"pt"},"x0":"10","x1":"30","y0":"20","y1":"40"},"kind":"bbox"},"node_id":"crop"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}}}],"observations":[{"axis_id":"sl","uncertainty":{"basis":"absolute","kind":"std_dev","lower":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"0.1","repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"},"value_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"}},"scale":"linear","upper":{"canonical_decimal_value":"0.1","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"0.1","repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}}},"value":{"canonical_decimal_value":"35.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"35.0","repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"col":2,"kind":"table_cell","row":1,"table_key":{"kind":"member_sheet","sheet_name":"Sheet1"}},"node_id":"si"},"value_ref":{"locator":{"col":1,"kind":"table_cell","row":0,"table_key":{"kind":"caption_label","label":"Table 1"}},"node_id":"si"}}}],"point_id":"p1"}],"series_id":"s1","source_form":"tabular","value_origin":"experimental"},{"axes":[{"axis_id":"phi2","label_raw":"phi2","label_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"quantity_kind":"equivalence_ratio","role":"coordinate"},{"axis_id":"sl2","label_raw":"sl2","label_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"quantity_kind":"velocity","role":"observation"}],"constants":[],"points":[{"composition":{"__absent__":true,"note":null,"reason":"not_applicable"},"coordinates":[{"axis_id":"phi2","uncertainty":{"__absent__":true,"note":null,"reason":"not_reported_here"},"value":{"canonical_decimal_value":"2.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"equivalence_ratio","raw_text":"2.0","repairs":[],"unit_normalized":"1","unit_raw":"-","unit_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"value_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"}}}],"observations":[{"axis_id":"sl2","uncertainty":{"__absent__":true,"note":null,"reason":"not_reported_here"},"value":{"canonical_decimal_value":"10.0","conversion_table_sha256":"1ac7a572c24b116e62fd360edc423a9bf333c35108d798f5336e91ad7b65a122","quantity_kind":"velocity","raw_text":"10.0","repairs":[],"unit_normalized":"cm/s","unit_raw":"cm/s","unit_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"},"value_ref":{"locator":{"kind":"xpath","xpath":"//table/row[1]/cell[1]"},"node_id":"jats"}}}],"point_id":"q1"}],"series_id":"s2","source_form":"textual","value_origin":"simulation"}],"source_graph":{"nodes":[{"kind":"paper_pdf","node_id":"paper","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":null,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"kind":"jats_xml","node_id":"jats","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":null,"sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},{"kind":"si_member","node_id":"si","origin":{"archive_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"parent_node_id":"paper","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},{"kind":"figure_crop","node_id":"crop","origin":{"__absent__":true,"note":null,"reason":"not_applicable"},"parent_node_id":"paper","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}]}}\n'  # noqa: E501


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
        assert isinstance(
            dumped["series"][0]["source_form"], SourceForm
        ), "expected model_dump() to keep an enum field as an Enum instance, not its .value"

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
