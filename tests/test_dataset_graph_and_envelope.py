"""Tests for carmel.schemas.datasets' M-D2b graph-of-provenance layer:
``SourceGraph`` (a validated DAG of ``SourceNode``s) and ``DatasetEnvelope``
(the top-level payload that ties every embedded ``SourceRef`` back to a node
that graph actually contains).

Kept in its own module rather than folded into test_dataset_schemas.py to
avoid colliding with that file's existing ``TestSourceGraph`` class, which
covers a different, narrower concern (SourceNode/SourceRef/locators in
isolation, not the graph-level model introduced here).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from enum import Enum
from types import MappingProxyType
from typing import Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from carmel.schemas.datasets import (
    _MAX_EMBEDDED_CANONICAL_JSON_LENGTH,
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
    ExtractionBinding,
    GlyphHealthAssessment,
    Maybe,
    MeasuredValue,
    MemberSheetKey,
    Observation,
    QuantityKind,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    TableKeyKind,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    ValueOrigin,
    XPathLocator,
    iter_measured_values,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.numeric import GlyphHealth
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    GLYPH_HEALTH_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import TABLE_V1, ConversionTable, IdentityRule

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _embedded_table_v1() -> EmbeddedConversionTable:
    """The one conversion table every ``MeasuredValue`` fixture in this file
    cites (via ``conversion_table_sha256=TABLE_V1.sha256``) -- embedded
    verbatim so ``DatasetEnvelope.conversion_tables``'s T2 cover-exactly
    check is satisfied by every envelope built here."""
    return EmbeddedConversionTable(
        sha256=TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
    )


_NO_ORIGIN = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""Module-level singleton default for SourceNode.origin -- Absent is frozen,
so sharing one instance across every _node() call that doesn't need a
concrete ArchiveOrigin is safe, and avoids a function-call-in-argument-default
(ruff B008)."""

_NO_EXTRACTION = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
"""Module-level singleton default for SourceNode.extraction, matching
_NO_ORIGIN's reasoning."""

_NO_GLYPH_HEALTH = Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
"""Module-level singleton default for SourceNode.glyph_health, matching
_NO_ORIGIN's reasoning."""

_HEALTHY_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)

_UNHEALTHY_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=True,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)


def _extraction_binding(
    extracted_sha256: str = SHA_A, extracted_text_sha256: str = SHA_B
) -> ExtractionBinding:
    return ExtractionBinding(
        extracted_sha256=extracted_sha256, extracted_text_sha256=extracted_text_sha256
    )


def _glyph_health_assessment(
    input_sha256: str = SHA_B, health: GlyphHealth = _HEALTHY_GLYPH_HEALTH
) -> GlyphHealthAssessment:
    return GlyphHealthAssessment(
        health=health,
        assessor=SemanticDependencyUse(
            dependency_id=GLYPH_HEALTH_DEPENDENCY_ID,
            content_sha256=current_sha_for(GLYPH_HEALTH_DEPENDENCY_ID),
            input_sha256=input_sha256,
        ),
    )


_CURRENT_REPAIR_DEPENDENCY = SemanticDependencyUse(
    dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
    input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
)
"""Module-level singleton for MeasuredValue.repair_dependency -- frozen, so
sharing one instance across every fixture that doesn't need a
deliberately-wrong or superseded dependency record is safe."""


def _frame(**kwargs: object) -> CoordinateFrame:
    defaults: dict[str, object] = {
        "render_fingerprint": "fp-1",
        "cropbox": ("0", "0", "612", "792"),
        "mediabox": ("0", "0", "612", "792"),
        "rotation": 0,
        "units": "pt",
        "dpi": Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        "render_settings": Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
    }
    defaults.update(kwargs)
    return CoordinateFrame(**defaults)  # type: ignore[arg-type]


def _bbox(**kwargs: object) -> BBox:
    defaults: dict[str, object] = {"frame": _frame(), "x0": "10", "y0": "20", "x1": "30", "y1": "40"}
    defaults.update(kwargs)
    return BBox(**defaults)  # type: ignore[arg-type]


def _node(
    node_id: str = "n1",
    kind: SourceNodeKind = SourceNodeKind.PAPER_PDF,
    sha256: str = SHA_A,
    parent_node_id: str | None = None,
    origin: ArchiveOrigin | Absent = _NO_ORIGIN,
    extraction: ExtractionBinding | Absent = _NO_EXTRACTION,
    glyph_health: GlyphHealthAssessment | Absent = _NO_GLYPH_HEALTH,
) -> SourceNode:
    return SourceNode(
        node_id=node_id,
        kind=kind,
        sha256=sha256,
        parent_node_id=parent_node_id,
        origin=origin,
        extraction=extraction,
        glyph_health=glyph_health,
    )


def _bbox_ref(node_id: str) -> SourceRef:
    return SourceRef(node_id=node_id, locator=BBoxLocator(bbox=_bbox()))


def _table_ref(node_id: str, row: int = 0, col: int = 1) -> SourceRef:
    return SourceRef(
        node_id=node_id, locator=TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=row, col=col)
    )


def _xpath_ref(node_id: str, xpath: str = "//table/row[1]/cell[1]") -> SourceRef:
    return SourceRef(node_id=node_id, locator=XPathLocator(xpath=xpath))


def _mole_fraction_amount(value_ref: SourceRef, unit_ref: SourceRef, raw_text: str = "0.04") -> MeasuredValue:
    """A physically-coherent MOLE_FRACTION amount, mirroring
    test_dataset_schemas.py's _mole_fraction_measured_value -- Composition
    validates that a component's amount.quantity_kind matches its basis."""
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=QuantityKind.MOLE_FRACTION,
        unit_raw="-",
        unit_normalized="1",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


def _equivalence_ratio_amount(value_ref: SourceRef, unit_ref: SourceRef, raw_text: str = "1.0") -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        unit_raw="-",
        unit_normalized="1",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


def _velocity_amount(value_ref: SourceRef, unit_ref: SourceRef, raw_text: str = "35.0") -> MeasuredValue:
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=QuantityKind.VELOCITY,
        unit_raw="cm/s",
        unit_normalized="cm/s",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


def _component(
    species: str, value_ref: SourceRef, unit_ref: SourceRef, role: ComponentRole = ComponentRole.FUEL
) -> CompositionComponent:
    return CompositionComponent(
        species_raw_name=species,
        amount=_mole_fraction_amount(value_ref=value_ref, unit_ref=unit_ref),
        role=role,
    )


def _minimal_graph(node_id: str = "paper", sha256: str = SHA_A) -> SourceGraph:
    return SourceGraph(nodes=(_node(node_id, SourceNodeKind.PAPER_PDF, sha256),))


def _resolved_single_component_composition(value_ref: SourceRef, unit_ref: SourceRef) -> Composition:
    return Composition(
        raw_name="4% H2 in N2",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        components=[_component("H2", value_ref=value_ref, unit_ref=unit_ref)],
    )


def _graph_and_node_of_kind(
    kind: SourceNodeKind, node_id: str = "target", sha256: str = SHA_A
) -> tuple[SourceGraph, str]:
    """Build the minimal SourceGraph containing exactly one node of `kind`
    (with whatever parent chain I4 requires), plus an unrelated PAPER_PDF
    root ("unit-root") to safely park a compatible unit_ref on -- so a
    locator/kind-compatibility test can vary ONLY the value_ref's pairing."""
    unit_root = _node("unit-root", SourceNodeKind.PAPER_PDF, SHA_D)
    if kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML):
        target = _node(node_id, kind, sha256)
        return SourceGraph(nodes=(unit_root, target)), node_id
    if kind in (SourceNodeKind.SI_MEMBER, SourceNodeKind.FIGURE_CROP):
        parent = _node("target-parent", SourceNodeKind.PAPER_PDF, SHA_C)
        target = _node(node_id, kind, sha256, parent_node_id=parent.node_id)
        return SourceGraph(nodes=(unit_root, parent, target)), node_id
    raise AssertionError(kind)


def _envelope_with_value_ref_locator(locator: object, node_kind: SourceNodeKind) -> DatasetEnvelope:
    graph, node_id = _graph_and_node_of_kind(node_kind)
    amount = _mole_fraction_amount(
        value_ref=SourceRef(node_id=node_id, locator=locator),  # type: ignore[arg-type]
        unit_ref=_table_ref("unit-root"),
    )
    composition = Composition(
        raw_name="mix",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        components=[CompositionComponent(species_raw_name="H2", amount=amount, role=ComponentRole.FUEL)],
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(_fully_populated_series("unit-root"),),
        conversion_tables=(_embedded_table_v1(),),
    )


_UNIT_FOR_QUANTITY_KIND: dict[QuantityKind, tuple[str, str]] = {
    QuantityKind.TEMPERATURE: ("K", "K"),
    QuantityKind.EQUIVALENCE_RATIO: ("-", "1"),
    QuantityKind.VELOCITY: ("cm/s", "cm/s"),
}
"""(unit_raw, unit_normalized) each fixture-used QuantityKind actually
accepts in TABLE_V1 -- an uncertainty bound is itself a MeasuredValue, so
it must pass the same unit/quantity_kind conversion-table check (including
the table's own normalized spelling) as any other value."""


def _uncertainty(value_ref: SourceRef, unit_ref: SourceRef, quantity_kind: QuantityKind) -> Uncertainty:
    """A fully-populated Uncertainty (kind/basis/scale/upper/lower all
    present, both bounds referenced) -- used by the ref-walk drift meta-test
    so every series.*.uncertainty.{upper,lower}.{value_ref,unit_ref} location
    is actually reachable."""
    unit_raw, unit_normalized = _UNIT_FOR_QUANTITY_KIND[quantity_kind]

    def _bound(raw_text: str, row: int) -> MeasuredValue:
        return MeasuredValue(
            raw_text=raw_text,
            canonical_decimal_value=raw_text,
            quantity_kind=quantity_kind,
            unit_raw=unit_raw,
            unit_normalized=unit_normalized,
            conversion_table_sha256=TABLE_V1.sha256,
            repairs=(),
            repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
            value_ref=_table_ref(value_ref.node_id, row=row, col=0),
            unit_ref=_table_ref(unit_ref.node_id, row=row, col=1),
        )

    return Uncertainty(
        kind=UncertaintyKind.STD_DEV,
        basis=UncertaintyBasis.ABSOLUTE,
        scale=UncertaintyScale.LINEAR,
        upper=_bound("0.1", row=90),
        lower=_bound("0.1", row=91),
    )


def _fully_populated_series(node_id: str = "paper") -> Series:
    """A Series with every SourceRef-bearing field actually populated:
    one coordinate axis, one observation axis, one constant axis, a
    constant covering that axis (with a fully-populated uncertainty), and
    one point whose coordinates/observations/composition are all likewise
    fully populated -- used by the ref-walk drift meta-test to make sure
    every series.* SourceRef location is actually reachable, not just
    statically declared."""
    phi_axis = AxisDeclaration(
        axis_id="phi",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
        label_raw="phi",
        label_ref=_table_ref(node_id, row=10, col=0),
    )
    sl_axis = AxisDeclaration(
        axis_id="sl",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.VELOCITY,
        label_raw="S_L (cm/s)",
        label_ref=_table_ref(node_id, row=10, col=1),
    )
    const_axis = AxisDeclaration(
        axis_id="temperature",
        role=AxisRole.CONSTANT,
        quantity_kind=QuantityKind.TEMPERATURE,
        label_raw="T (K)",
        label_ref=_table_ref(node_id, row=10, col=2),
    )

    const_value = MeasuredValue(
        raw_text="298",
        canonical_decimal_value="298",
        quantity_kind=QuantityKind.TEMPERATURE,
        unit_raw="K",
        unit_normalized="K",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=_table_ref(node_id, row=11, col=0),
        unit_ref=_table_ref(node_id, row=11, col=1),
    )
    constant = Coordinate(
        axis_id="temperature",
        value=const_value,
        uncertainty=_uncertainty(
            value_ref=_table_ref(node_id), unit_ref=_table_ref(node_id), quantity_kind=QuantityKind.TEMPERATURE
        ),
    )

    phi_value = _equivalence_ratio_amount(
        value_ref=_table_ref(node_id, row=20, col=0),
        unit_ref=_table_ref(node_id, row=20, col=1),
    )
    coordinate = Coordinate(
        axis_id="phi",
        value=phi_value,
        uncertainty=_uncertainty(
            value_ref=_table_ref(node_id), unit_ref=_table_ref(node_id), quantity_kind=QuantityKind.EQUIVALENCE_RATIO
        ),
    )

    sl_value = MeasuredValue(
        raw_text="35.0",
        canonical_decimal_value="35.0",
        quantity_kind=QuantityKind.VELOCITY,
        unit_raw="cm/s",
        unit_normalized="cm/s",
        conversion_table_sha256=TABLE_V1.sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=_table_ref(node_id, row=21, col=0),
        unit_ref=_table_ref(node_id, row=21, col=1),
    )
    observation = Observation(
        axis_id="sl",
        value=sl_value,
        uncertainty=_uncertainty(
            value_ref=_table_ref(node_id), unit_ref=_table_ref(node_id), quantity_kind=QuantityKind.VELOCITY
        ),
    )

    point_eq_ratio = _equivalence_ratio_amount(
        value_ref=_table_ref(node_id, row=30, col=0),
        unit_ref=_table_ref(node_id, row=30, col=1),
    )
    point_component_amount = _mole_fraction_amount(
        value_ref=_table_ref(node_id, row=31, col=0),
        unit_ref=_table_ref(node_id, row=31, col=1),
    )
    point_composition = Composition(
        raw_name="4% H2 in N2",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=point_eq_ratio,
        components=[
            CompositionComponent(species_raw_name="H2", amount=point_component_amount, role=ComponentRole.FUEL)
        ],
    )

    point = DataPoint(
        point_id="p1",
        coordinates=(coordinate,),
        observations=(observation,),
        composition=point_composition,
    )

    return Series(
        series_id="s1",
        source_form=SourceForm.TABULAR,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(phi_axis, sl_axis, const_axis),
        constants=(constant,),
        points=(point,),
    )


def _fully_populated_envelope(
    conversion_tables: tuple[EmbeddedConversionTable, ...] | None = None,
) -> DatasetEnvelope:
    """A DatasetEnvelope with every SourceRef-bearing field actually
    populated (including equivalence_ratio, which is Absent in most other
    fixtures here, and the series aggregate added by M-D2b part a) --
    used by the ref-walk drift meta-test.

    ``conversion_tables`` defaults to the usual single-table
    ``(_embedded_table_v1(),)`` (every MeasuredValue built by this fixture
    cites only TABLE_V1), but can be overridden so the T2 (cover-exactly)
    tests can vary ONLY the embedded set while reusing this fixture's
    otherwise-valid shape. A full constructor call is used here (not
    ``model_copy(update=...)``) deliberately: ``model_copy`` does not
    re-run pydantic v2's ``model_validator(mode="after")`` validators, so
    it cannot be used to exercise T2/T3 -- it would silently bypass the
    very checks these tests exist to exercise."""
    paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
    graph = SourceGraph(nodes=(paper,))
    eq_ratio = _equivalence_ratio_amount(
        value_ref=_table_ref("paper", row=0, col=0),
        unit_ref=_table_ref("paper", row=0, col=1),
    )
    component_amount = _mole_fraction_amount(
        value_ref=_table_ref("paper", row=1, col=0),
        unit_ref=_table_ref("paper", row=1, col=1),
    )
    composition = Composition(
        raw_name="4% H2 in N2",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=eq_ratio,
        components=[CompositionComponent(species_raw_name="H2", amount=component_amount, role=ComponentRole.FUEL)],
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(_fully_populated_series("paper"),),
        conversion_tables=conversion_tables if conversion_tables is not None else (_embedded_table_v1(),),
    )


def _second_conversion_table() -> ConversionTable:
    """A second, distinct, validly-constructed ConversionTable: the same 11
    base units as TABLE_V1, one IdentityRule per base unit (the minimum
    ConversionTable.__post_init__ requires -- see its invariant 9), but a
    different table_id, so it hashes to a DIFFERENT sha256 than TABLE_V1.

    Built independently here from scratch, rather than by importing
    units.py's private ``_base_units_v1``/``_identity_rules_v1`` helpers,
    so this fixture does not couple to that module's internal (``_``
    -prefixed) helper functions remaining named or shaped the way they are
    today."""
    base_units: tuple[tuple[QuantityKind, str], ...] = (
        (QuantityKind.LENGTH, "m"),
        (QuantityKind.VELOCITY, "m/s"),
        (QuantityKind.TEMPERATURE, "K"),
        (QuantityKind.PRESSURE, "Pa"),
        (QuantityKind.TIME, "s"),
        (QuantityKind.VOLUME, "m3"),
        (QuantityKind.STRAIN_RATE, "1/s"),
        (QuantityKind.MOLE_FRACTION, "1"),
        (QuantityKind.MASS_FRACTION, "1"),
        (QuantityKind.EQUIVALENCE_RATIO, "1"),
        (QuantityKind.RELATIVE_UNCERTAINTY, "1"),
    )
    rules = tuple(IdentityRule(kind="identity", quantity=quantity, unit=unit) for quantity, unit in base_units)
    return ConversionTable(
        table_id="carmel-unit-conversions-test-fixture-2",
        version=1,
        base_units=base_units,
        aliases=(),
        rules=rules,
    )


def _embedded_second_table() -> EmbeddedConversionTable:
    table = _second_conversion_table()
    return EmbeddedConversionTable(
        sha256=table.sha256,
        canonical_json=canonical_json_bytes(table.identity_payload()).decode("utf-8"),
    )


@contextlib.contextmanager
def _registered_second_table() -> Iterator[ConversionTable]:
    """Temporarily registers _second_conversion_table() into
    units.TABLES_BY_SHA for the duration of the `with` block, then restores
    the original mapping unconditionally (even if the block raises).

    Why this is needed rather than just constructing a MeasuredValue that
    cites a second table directly: MeasuredValue.conversion_table_sha256 is
    validated (via units.table_for_sha, see its docstring) against "every
    table this module ships" -- there is deliberately no way to construct a
    MeasuredValue citing a sha256 the shipped registry does not recognize.
    Today that registry (units.TABLES_BY_SHA) holds exactly one table,
    TABLE_V1. Exercising DatasetEnvelope's T2/T3 checks, though, requires a
    SECOND genuinely-cited table (one MeasuredValue citing table A, another
    citing table B) -- otherwise T2's cover-exactly check can never be
    isolated from T3's sort-order check, since with only one ever-citable
    table there is nothing to sort.

    ``units.table_for_sha`` resolves ``TABLES_BY_SHA`` as a plain module
    global at call time, so reassigning the attribute on the imported
    ``units`` module object here is visible to it for the lifetime of this
    context manager -- this mutates test-process state only, never
    carmel/'s source, and is restored before the block exits."""
    table = _second_conversion_table()
    original = units.TABLES_BY_SHA
    units.TABLES_BY_SHA = MappingProxyType({**original, table.sha256: table})
    try:
        yield table
    finally:
        units.TABLES_BY_SHA = original


def _mole_fraction_amount_citing(
    conversion_table_sha256: str, value_ref: SourceRef, unit_ref: SourceRef, raw_text: str = "0.04"
) -> MeasuredValue:
    """Like _mole_fraction_amount, but citing an arbitrary conversion table
    sha256 -- used to build a composition component that cites a SECOND
    conversion table (distinct from TABLE_V1), for the T2/T3
    conversion_tables tests below.

    Uses unit_raw="1" (the base unit itself) rather than TABLE_V1's "-"
    alias: _second_conversion_table() below is built with no aliases at
    all (only the one IdentityRule invariant 9 requires), so "-" is not a
    known unit/alias of MOLE_FRACTION in that table."""
    return MeasuredValue(
        raw_text=raw_text,
        canonical_decimal_value=raw_text,
        quantity_kind=QuantityKind.MOLE_FRACTION,
        unit_raw="1",
        unit_normalized="1",
        conversion_table_sha256=conversion_table_sha256,
        repairs=(),
        repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


def _envelope_citing_two_tables(conversion_tables: tuple[EmbeddedConversionTable, ...]) -> DatasetEnvelope:
    """A DatasetEnvelope whose composition cites BOTH TABLE_V1 (via its H2
    component) and _second_conversion_table() (via its N2 component) --
    used by the T3 sort-order tests so a conversion_tables ordering failure
    can be exercised in isolation from T2's cover-exactly check (which
    would otherwise also fire if only one table were actually cited, since
    then the *other* embedded table would look decorative)."""
    paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
    graph = SourceGraph(nodes=(paper,))
    h2_amount = _mole_fraction_amount(
        value_ref=_table_ref("paper", row=0, col=0),
        unit_ref=_table_ref("paper", row=0, col=1),
    )
    n2_amount = _mole_fraction_amount_citing(
        _second_conversion_table().sha256,
        value_ref=_table_ref("paper", row=1, col=0),
        unit_ref=_table_ref("paper", row=1, col=1),
    )
    composition = Composition(
        raw_name="4% H2 in N2",
        resolution=CompositionResolution.RESOLVED_COMPONENTS,
        basis=CompositionBasis.MOLE_FRACTION,
        equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        components=[
            CompositionComponent(species_raw_name="H2", amount=h2_amount, role=ComponentRole.FUEL),
            CompositionComponent(species_raw_name="N2", amount=n2_amount, role=ComponentRole.DILUENT),
        ],
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(_fully_populated_series("paper"),),
        conversion_tables=conversion_tables,
    )


def _strip_list_indices(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path)


def _collect_field_paths_carrying_source_ref(
    model_cls: type[BaseModel], prefix: str = "", seen: frozenset[type] | None = None
) -> set[str]:
    """Recursively walk `model_cls`'s pydantic field annotations, returning
    the set of dotted paths (unindexed -- list/tuple generics never carry an
    index at the type level) whose annotation is, or transitively contains,
    SourceRef."""
    seen = seen or frozenset()
    if model_cls in seen:
        return set()
    seen = seen | {model_cls}
    paths: set[str] = set()
    for name, field in model_cls.model_fields.items():
        paths |= _inspect_annotation(field.annotation, f"{prefix}{name}", seen)
    return paths


def _inspect_annotation(annotation: object, path: str, seen: frozenset[type]) -> set[str]:
    if annotation is SourceRef:
        return {path}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None:
        paths: set[str] = set()
        for arg in args:
            paths |= _inspect_annotation(arg, path, seen)
        return paths
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _collect_field_paths_carrying_source_ref(annotation, prefix=f"{path}.", seen=seen)
    return set()


_ALLOWED_PRIMITIVE_ANNOTATIONS = (str, int, float, bool, bytes, type(None))
_DATASETS_MODULE = "carmel.schemas.datasets"


def _unwalkable_annotations(
    model_cls: type[BaseModel], prefix: str = "", seen: frozenset[type] | None = None
) -> dict[str, object]:
    """Recursively walk `model_cls`'s field annotations and return a
    path -> annotation mapping for every annotation that is NOT built
    entirely out of shapes both _collect_field_paths_carrying_source_ref
    (above) and the runtime walker (iter_source_refs) know how to
    traverse.

    _inspect_annotation above answers "does this annotation carry a
    SourceRef, given what I know how to walk?" -- which is silently blind
    to a shape it doesn't know how to walk at all: `typing.Any`, a bare
    `object`, an un-parameterized container, or a container neither walker
    descends into (e.g. `set`/`frozenset` -- iter_source_refs only
    descends into BaseModel/list/tuple/dict). A field given one of those
    annotations could hide a SourceRef from BOTH the annotation-side check
    and the runtime walker at once, which is exactly the failure
    TestRefWalkCannotBeOutgrown above cannot catch by itself. This walk
    reports every such shape explicitly instead.

    Expressed as an explicit ALLOWLIST (BaseModel subclasses defined in
    this module, tuple/list/dict, primitives, Maybe/Optional/Union,
    Enum subclasses, SourceRef itself) rather than a per-disallowed-type
    special case, so a brand-new unwalkable shape fails closed by default
    instead of silently passing because nobody thought to blocklist it.
    """
    seen = seen or frozenset()
    if model_cls in seen:
        return {}
    seen = seen | {model_cls}
    unwalkable: dict[str, object] = {}
    for name, field in model_cls.model_fields.items():
        unwalkable.update(_inspect_annotation_walkability(field.annotation, f"{prefix}{name}", seen))
    return unwalkable


def _inspect_annotation_walkability(annotation: object, path: str, seen: frozenset[type]) -> dict[str, object]:
    # GlyphHealth is allowlisted by name, not by shape: it's a stdlib frozen
    # dataclass (deliberately not a pydantic BaseModel -- see its own
    # docstring), so it falls outside this walker's BaseModel-recursion
    # branch below. It is safe to allowlist because every one of its fields
    # is a bare bool -- there is no annotation shape inside it that could
    # possibly carry a SourceRef, so it cannot hide one from either walker.
    if annotation is SourceRef or annotation is GlyphHealth or annotation in _ALLOWED_PRIMITIVE_ANNOTATIONS:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return {}
        if issubclass(annotation, BaseModel):
            if annotation.__module__ != _DATASETS_MODULE or annotation in seen:
                return {}
            return _unwalkable_annotations(annotation, prefix=f"{path}.", seen=seen)
        # A bare type that is neither an Enum nor a BaseModel we can descend
        # into -- covers `object`, and any other non-allowlisted concrete
        # type used as a bare annotation.
        return {path: annotation}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Maybe:
        result: dict[str, object] = {}
        for arg in args:
            result.update(_inspect_annotation_walkability(arg, path, seen))
        return result

    if origin in (tuple, list, dict, Union):
        result = {}
        for arg in args:
            if arg is Ellipsis:
                continue
            result.update(_inspect_annotation_walkability(arg, path, seen))
        return result

    # Everything else -- typing.Any (origin/args both empty), `set`/
    # `frozenset` (a container the runtime walker never descends into),
    # Callable, or any other origin not on the allowlist -- is unwalkable.
    return {path: annotation}


def _collect_field_paths_carrying_measured_value(
    model_cls: type[BaseModel], prefix: str = "", seen: frozenset[type] | None = None
) -> set[str]:
    """Mirrors _collect_field_paths_carrying_source_ref above, but targets
    MeasuredValue instead of SourceRef -- used by iter_measured_values'
    drift meta-test below. The "unwalkable annotation" half of that drift
    check is target-agnostic (an annotation shape neither walker can
    traverse hides ANY target type, SourceRef or MeasuredValue, equally)
    and is already covered by
    test_no_field_reachable_from_dataset_envelope_has_an_unwalkable_annotation
    above; only this annotation-vs-runtime direct-coverage half needs a
    MeasuredValue-targeted twin."""
    seen = seen or frozenset()
    if model_cls in seen:
        return set()
    seen = seen | {model_cls}
    paths: set[str] = set()
    for name, field in model_cls.model_fields.items():
        paths |= _inspect_annotation_for_measured_value(field.annotation, f"{prefix}{name}", seen)
    return paths


def _inspect_annotation_for_measured_value(annotation: object, path: str, seen: frozenset[type]) -> set[str]:
    if annotation is MeasuredValue:
        return {path}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None:
        paths: set[str] = set()
        for arg in args:
            paths |= _inspect_annotation_for_measured_value(arg, path, seen)
        return paths
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _collect_field_paths_carrying_measured_value(annotation, prefix=f"{path}.", seen=seen)
    return set()


# --------------------------------------------------------------------------
# SourceGraph invariants
# --------------------------------------------------------------------------


class TestSourceGraphIdentityAndAcyclicity:
    """I1 (unique ids), I2 (parent resolves), I3 (acyclic)."""

    def test_requires_at_least_one_node(self) -> None:
        with pytest.raises(ValidationError):
            SourceGraph(nodes=())

    def test_two_nodes_sharing_a_node_id_rejected(self) -> None:
        a = _node("dup", SourceNodeKind.PAPER_PDF, SHA_A)
        b = _node("dup", SourceNodeKind.JATS_XML, SHA_B)
        with pytest.raises(ValidationError, match="dup"):
            SourceGraph(nodes=(a, b))

    def test_parent_naming_no_node_in_the_graph_rejected(self) -> None:
        # SI_MEMBER (not PAPER_PDF/JATS_XML) so this isolates I2 from I4:
        # the only defect here is that "ghost-paper" resolves to nothing.
        orphan = _node("si-orphan", SourceNodeKind.SI_MEMBER, SHA_A, parent_node_id="ghost-paper")
        with pytest.raises(ValidationError, match="si-orphan"):
            SourceGraph(nodes=(orphan,))

    # The three cycle tests below MUST match on the word "cycle", not on a node
    # id. Every cycle the current SourceNodeKind rules can express is also an I4
    # violation, and I4's message names the offending node id too -- so a test
    # matching only the id passes identically whether or not I3 exists. Mutation
    # testing proved exactly that: deleting the entire I3 block left all 48 tests
    # green. Matching the distinctive word is what makes I3 independently pinned,
    # because I4's message never contains it. I3 is kept rather than deleted
    # because it stops being unreachable the moment the kind rules admit a
    # same-kind parent (an archive containing members is the live candidate),
    # and a cycle then costs a hang, not a rejection: the walk's only exit is
    # this raise.

    def test_self_parent_rejected(self) -> None:
        looped = _node("self-loop", SourceNodeKind.SI_MEMBER, SHA_A, parent_node_id="self-loop")
        with pytest.raises(ValidationError, match="cycle"):
            SourceGraph(nodes=(looped,))

    def test_two_node_parent_cycle_rejected(self) -> None:
        a = _node("a", SourceNodeKind.SI_MEMBER, SHA_A, parent_node_id="b")
        b = _node("b", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="a")
        with pytest.raises(ValidationError, match="cycle"):
            SourceGraph(nodes=(a, b))

    def test_three_node_parent_cycle_rejected(self) -> None:
        a = _node("a", SourceNodeKind.SI_MEMBER, SHA_A, parent_node_id="c")
        b = _node("b", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="a")
        c = _node("c", SourceNodeKind.SI_MEMBER, SHA_C, parent_node_id="b")
        with pytest.raises(ValidationError, match="cycle"):
            SourceGraph(nodes=(a, b, c))


class TestSourceGraphKindParentRules:
    """I4: which SourceNodeKinds may/must have a parent, and of what kind."""

    def test_paper_pdf_with_a_parent_rejected(self) -> None:
        root = _node("root", SourceNodeKind.PAPER_PDF, SHA_A)
        child = _node("child", SourceNodeKind.PAPER_PDF, SHA_B, parent_node_id="root")
        with pytest.raises(ValidationError, match="child"):
            SourceGraph(nodes=(root, child))

    def test_jats_xml_with_a_parent_rejected(self) -> None:
        root = _node("root", SourceNodeKind.PAPER_PDF, SHA_A)
        child = _node("child", SourceNodeKind.JATS_XML, SHA_B, parent_node_id="root")
        with pytest.raises(ValidationError, match="child"):
            SourceGraph(nodes=(root, child))

    def test_si_member_with_no_parent_rejected(self) -> None:
        orphan = _node("si", SourceNodeKind.SI_MEMBER, SHA_A, parent_node_id=None)
        with pytest.raises(ValidationError, match="si"):
            SourceGraph(nodes=(orphan,))

    def test_si_member_parent_must_be_paper_pdf_or_jats_xml_not_another_si_member(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        si1 = _node("si1", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="paper")
        si2 = _node("si2", SourceNodeKind.SI_MEMBER, SHA_C, parent_node_id="si1")
        with pytest.raises(ValidationError, match="si2"):
            SourceGraph(nodes=(paper, si1, si2))

    def test_si_member_child_of_paper_pdf_accepted(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        si = _node("si", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, si))
        assert graph.node("si").parent_node_id == "paper"

    def test_si_member_child_of_jats_xml_accepted(self) -> None:
        jats = _node("jats", SourceNodeKind.JATS_XML, SHA_A)
        si = _node("si", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="jats")
        graph = SourceGraph(nodes=(jats, si))
        assert graph.node("si").parent_node_id == "jats"

    def test_figure_crop_with_no_parent_rejected(self) -> None:
        orphan = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_A, parent_node_id=None)
        with pytest.raises(ValidationError, match="crop"):
            SourceGraph(nodes=(orphan,))

    def test_figure_crop_parent_cannot_be_another_figure_crop(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        crop1 = _node("crop1", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
        crop2 = _node("crop2", SourceNodeKind.FIGURE_CROP, SHA_C, parent_node_id="crop1")
        with pytest.raises(ValidationError, match="crop2"):
            SourceGraph(nodes=(paper, crop1, crop2))

    def test_figure_crop_child_of_paper_pdf_accepted(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, crop))
        assert graph.node("crop").parent_node_id == "paper"

    def test_figure_crop_child_of_jats_xml_accepted(self) -> None:
        jats = _node("jats", SourceNodeKind.JATS_XML, SHA_A)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="jats")
        graph = SourceGraph(nodes=(jats, crop))
        assert graph.node("crop").parent_node_id == "jats"

    def test_figure_crop_child_of_si_member_accepted(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        si = _node("si", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="paper")
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_C, parent_node_id="si")
        graph = SourceGraph(nodes=(paper, si, crop))
        assert graph.node("crop").parent_node_id == "si"


class TestSourceGraphDuplicateNodes:
    """I5: no two nodes may share (kind, sha256, parent_node_id), but the
    same bytes in a genuinely different role (different kind or parent) is
    allowed."""

    def test_two_root_nodes_with_identical_kind_sha_and_parent_rejected(self) -> None:
        a = _node("a", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        b = _node("b", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        with pytest.raises(ValidationError):
            SourceGraph(nodes=(a, b))

    def test_same_bytes_as_a_different_kind_in_a_different_role_is_allowed(self) -> None:
        """Same sha256, but one is the PAPER_PDF root and the other is a
        FIGURE_CROP child of it -- different kind AND different parent, so
        this is not a decorative duplicate."""
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_A, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, crop))
        assert graph.node("crop").sha256 == graph.node("paper").sha256

    def test_same_bytes_under_the_same_kind_but_different_parent_is_allowed(self) -> None:
        paper1 = _node("paper1", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        jats1 = _node("jats1", SourceNodeKind.JATS_XML, SHA_B, parent_node_id=None)
        si_under_paper1 = _node("si-p1", SourceNodeKind.SI_MEMBER, SHA_C, parent_node_id="paper1")
        si_under_jats1 = _node("si-j1", SourceNodeKind.SI_MEMBER, SHA_C, parent_node_id="jats1")
        graph = SourceGraph(nodes=(paper1, jats1, si_under_paper1, si_under_jats1))
        assert graph.node("si-p1").sha256 == graph.node("si-j1").sha256


class TestSourceGraphConflictingGlyphHealth:
    """I5b: two nodes that share raw bytes (the same sha256) but disagree on
    glyph health are rejected as CONFLICTING -- distinct from I5's "exact
    duplicate" rejection above, and must not fire for any of that class's
    fixtures (which never carry a concrete glyph_health at all)."""

    def test_same_bytes_disagreeing_on_glyph_health_is_rejected(self) -> None:
        paper = _node(
            "paper",
            SourceNodeKind.PAPER_PDF,
            SHA_A,
            parent_node_id=None,
            extraction=_extraction_binding(extracted_text_sha256=SHA_B),
            glyph_health=_glyph_health_assessment(input_sha256=SHA_B, health=_HEALTHY_GLYPH_HEALTH),
        )
        crop = _node(
            "crop",
            SourceNodeKind.FIGURE_CROP,
            SHA_A,
            parent_node_id="paper",
            extraction=_extraction_binding(extracted_text_sha256=SHA_C),
            glyph_health=_glyph_health_assessment(input_sha256=SHA_C, health=_UNHEALTHY_GLYPH_HEALTH),
        )
        with pytest.raises(ValidationError, match="CONFLICT"):
            SourceGraph(nodes=(paper, crop))

    def test_same_bytes_agreeing_on_glyph_health_is_allowed(self) -> None:
        paper = _node(
            "paper",
            SourceNodeKind.PAPER_PDF,
            SHA_A,
            parent_node_id=None,
            extraction=_extraction_binding(extracted_text_sha256=SHA_B),
            glyph_health=_glyph_health_assessment(input_sha256=SHA_B, health=_HEALTHY_GLYPH_HEALTH),
        )
        crop = _node(
            "crop",
            SourceNodeKind.FIGURE_CROP,
            SHA_A,
            parent_node_id="paper",
            extraction=_extraction_binding(extracted_text_sha256=SHA_C),
            glyph_health=_glyph_health_assessment(input_sha256=SHA_C, health=_HEALTHY_GLYPH_HEALTH),
        )
        graph = SourceGraph(nodes=(paper, crop))
        assert graph.node("crop").sha256 == graph.node("paper").sha256

    def test_exact_duplicate_still_reports_duplicate_wording_not_conflict(self) -> None:
        """A true (kind, sha256, parent_node_id) duplicate must still raise
        the original 'duplicates an earlier node' message, even when neither
        node carries glyph_health -- the two rejection paths must stay
        distinguishable by wording."""
        a = _node("a", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        b = _node("b", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id=None)
        with pytest.raises(ValidationError, match="duplicates an earlier node"):
            SourceGraph(nodes=(a, b))

    def test_same_triple_disagreeing_on_health_reports_conflict_not_duplicate(self) -> None:
        """The check ORDER is the whole point of this test.

        These two nodes satisfy BOTH invariants at once: they share a
        (kind, sha256, parent_node_id) triple AND they disagree on glyph
        health. If the duplicate check runs first they are reported as "an
        exact repeat" that "adds nothing" -- which is flatly false, since
        they differ on health, and it points the reader at a redundant node
        instead of at an irreconcilable pair of assessments.

        The sibling tests above only ever use DIFFERENT roles (PAPER_PDF vs
        FIGURE_CROP), where the triples differ and the duplicate check never
        competes -- so they pass either way and cannot catch a reordering.
        This one can.
        """
        a = _node(
            "a",
            SourceNodeKind.PAPER_PDF,
            SHA_A,
            parent_node_id=None,
            extraction=_extraction_binding(extracted_text_sha256=SHA_B),
            glyph_health=_glyph_health_assessment(input_sha256=SHA_B, health=_HEALTHY_GLYPH_HEALTH),
        )
        b = _node(
            "b",
            SourceNodeKind.PAPER_PDF,
            SHA_A,
            parent_node_id=None,
            extraction=_extraction_binding(extracted_text_sha256=SHA_B),
            glyph_health=_glyph_health_assessment(
                input_sha256=SHA_B, health=_UNHEALTHY_GLYPH_HEALTH
            ),
        )
        with pytest.raises(ValidationError, match="CONFLICT") as excinfo:
            SourceGraph(nodes=(a, b))
        assert "duplicates an earlier node" not in str(excinfo.value)


class TestSourceGraphLookupAPI:
    """graph.node(), graph.node_ids, graph.ancestors()."""

    def test_node_returns_the_matching_source_node(self) -> None:
        a = _node("a", SourceNodeKind.PAPER_PDF, SHA_A)
        graph = SourceGraph(nodes=(a,))
        assert graph.node("a") == a

    def test_node_raises_a_clear_error_for_an_unknown_id(self) -> None:
        graph = SourceGraph(nodes=(_node("a"),))
        # Exception type is unspecified by the feature spec; a dict-like
        # `.node(id)` accessor conventionally raises KeyError (a LookupError
        # subtype) -- see this module's closing report.
        with pytest.raises(LookupError):
            graph.node("does-not-exist")

    def test_node_ids_is_a_frozenset_of_every_node_id(self) -> None:
        a = _node("a", SourceNodeKind.PAPER_PDF, SHA_A)
        b = _node("b", SourceNodeKind.JATS_XML, SHA_B)
        graph = SourceGraph(nodes=(a, b))
        assert graph.node_ids == frozenset({"a", "b"})

    def test_ancestors_returns_the_parent_chain_from_immediate_parent_to_root(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        si = _node("si", SourceNodeKind.SI_MEMBER, SHA_B, parent_node_id="paper")
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_C, parent_node_id="si")
        graph = SourceGraph(nodes=(paper, si, crop))
        ancestor_ids = [getattr(item, "node_id", item) for item in graph.ancestors("crop")]
        assert ancestor_ids == ["si", "paper"]

    def test_ancestors_of_a_root_node_is_empty(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        graph = SourceGraph(nodes=(paper,))
        assert list(graph.ancestors("paper")) == []

    def test_ancestors_raises_instead_of_hanging_on_a_cycle_from_model_construct(self) -> None:
        """Normal validated construction rejects cycles (I3), but
        ``SourceGraph.model_construct()`` is a documented escape hatch that
        bypasses validation entirely and can produce one. ``ancestors()``
        walks ``parent_node_id`` with no visited-set of its own, so a cyclic
        graph built this way must not hang forever -- it must raise."""
        a = _node("a", SourceNodeKind.PAPER_PDF, SHA_A, parent_node_id="b")
        b = _node("b", SourceNodeKind.PAPER_PDF, SHA_B, parent_node_id="a")
        graph = SourceGraph.model_construct(nodes=(a, b))
        with pytest.raises(ValueError, match="cycle"):
            graph.ancestors("a")


# --------------------------------------------------------------------------
# DatasetEnvelope invariants
# --------------------------------------------------------------------------


class TestDatasetEnvelopeRefsResolve:
    """V1: every SourceRef embedded anywhere in the envelope must name a
    node the source_graph actually contains."""

    def test_dangling_ref_on_equivalence_ratio_value_ref_rejected(self) -> None:
        graph = _minimal_graph("paper")
        eq_ratio = _equivalence_ratio_amount(
            value_ref=_bbox_ref("does-not-exist"),
            unit_ref=_table_ref("paper"),
        )
        composition = Composition(
            raw_name="stoichiometric CH4/air",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            equivalence_ratio=eq_ratio,
        )
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=graph,
                composition=composition,
                series=(_fully_populated_series("paper"),),
                conversion_tables=(_embedded_table_v1(),),
            )
        msg = str(excinfo.value)
        assert "does-not-exist" in msg
        assert "equivalence_ratio" in msg
        assert "value_ref" in msg

    def test_dangling_ref_on_equivalence_ratio_unit_ref_rejected(self) -> None:
        graph = _minimal_graph("paper")
        eq_ratio = _equivalence_ratio_amount(
            value_ref=_table_ref("paper"),
            unit_ref=_bbox_ref("does-not-exist"),
        )
        composition = Composition(
            raw_name="stoichiometric CH4/air",
            resolution=CompositionResolution.UNRESOLVED_NAMED_MIXTURE,
            basis=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            equivalence_ratio=eq_ratio,
        )
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=graph,
                composition=composition,
                series=(_fully_populated_series("paper"),),
                conversion_tables=(_embedded_table_v1(),),
            )
        msg = str(excinfo.value)
        assert "does-not-exist" in msg
        assert "equivalence_ratio" in msg
        assert "unit_ref" in msg

    def test_dangling_ref_on_component_amount_value_ref_rejected(self) -> None:
        graph = _minimal_graph("paper")
        composition = _resolved_single_component_composition(
            value_ref=_bbox_ref("ghost"),
            unit_ref=_table_ref("paper"),
        )
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=graph,
                composition=composition,
                series=(_fully_populated_series("paper"),),
                conversion_tables=(_embedded_table_v1(),),
            )
        msg = str(excinfo.value)
        assert "ghost" in msg
        assert "components" in msg
        assert "amount" in msg
        assert "value_ref" in msg

    def test_dangling_ref_on_component_amount_unit_ref_rejected(self) -> None:
        graph = _minimal_graph("paper")
        composition = _resolved_single_component_composition(
            value_ref=_table_ref("paper"),
            unit_ref=_bbox_ref("ghost"),
        )
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=graph,
                composition=composition,
                series=(_fully_populated_series("paper"),),
                conversion_tables=(_embedded_table_v1(),),
            )
        msg = str(excinfo.value)
        assert "ghost" in msg
        assert "components" in msg
        assert "amount" in msg
        assert "unit_ref" in msg


class TestDatasetEnvelopeNoDecorativeNodes:
    """V2: every node must be targeted by some SourceRef, or be an ancestor
    of a node that is."""

    def test_unreferenced_extra_node_rejected(self) -> None:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        decorative = _node("decorative", SourceNodeKind.PAPER_PDF, SHA_B)
        graph = SourceGraph(nodes=(paper, decorative))
        composition = _resolved_single_component_composition(
            value_ref=_bbox_ref("paper"),
            unit_ref=_table_ref("paper"),
        )
        with pytest.raises(ValidationError, match="decorative"):
            DatasetEnvelope(
                source_graph=graph,
                composition=composition,
                series=(_fully_populated_series("paper"),),
                conversion_tables=(_embedded_table_v1(),),
            )

    def test_unreferenced_ancestor_of_a_referenced_node_is_allowed(self) -> None:
        """The PAPER_PDF root is never targeted directly, but its
        FIGURE_CROP child is -- an ancestor of a targeted node is not
        decorative.

        The series added here (required since DatasetEnvelope.series gained
        Field(min_length=1)) is deliberately built to reference ONLY "crop",
        never "paper" -- reusing the shared _fully_populated_series("paper")
        fixture would make "paper" directly referenced too, which would
        make this test pass without ever exercising the ancestor-is-allowed
        path it is named for. A DIGITIZED series (rather than TABULAR) is
        used because its value_refs must target a FIGURE_CROP node (V4),
        matching "crop"'s own kind."""
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, crop))
        composition = _resolved_single_component_composition(
            value_ref=_bbox_ref("crop"),
            unit_ref=SourceRef(node_id="crop", locator=BBoxLocator(bbox=_bbox(x0="1", y0="1", x1="2", y1="2"))),
        )
        phi_axis = AxisDeclaration(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            label_raw="phi",
            label_ref=_bbox_ref("crop"),
        )
        sl_axis = AxisDeclaration(
            axis_id="sl",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.VELOCITY,
            label_raw="S_L (cm/s)",
            label_ref=_bbox_ref("crop"),
        )
        coordinate = Coordinate(
            axis_id="phi",
            value=_equivalence_ratio_amount(value_ref=_bbox_ref("crop"), unit_ref=_bbox_ref("crop")),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        observation = Observation(
            axis_id="sl",
            value=MeasuredValue(
                raw_text="35.0",
                canonical_decimal_value="35.0",
                quantity_kind=QuantityKind.VELOCITY,
                unit_raw="cm/s",
                unit_normalized="cm/s",
                conversion_table_sha256=TABLE_V1.sha256,
                repairs=(),
                repair_dependency=_CURRENT_REPAIR_DEPENDENCY,
                value_ref=_bbox_ref("crop"),
                unit_ref=_bbox_ref("crop"),
            ),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        point = DataPoint(
            point_id="p1",
            coordinates=(coordinate,),
            observations=(observation,),
            composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
        )
        series_referencing_crop_only = Series(
            series_id="s1",
            source_form=SourceForm.DIGITIZED,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=(phi_axis, sl_axis),
            constants=(),
            points=(point,),
        )
        envelope = DatasetEnvelope(
            source_graph=graph,
            composition=composition,
            series=(series_referencing_crop_only,),
            conversion_tables=(_embedded_table_v1(),),
        )
        assert envelope.source_graph.node("paper").parent_node_id is None

    def test_an_envelope_citing_nothing_can_no_longer_be_constructed_at_all(self) -> None:
        """This test's predecessor asserted V0 fires with match="ungrounded".
        Its own docstring flagged the expiry, and M-D2b part (a) triggered it.

        `series` is now required with min_length=1, every Series needs at
        least one AxisDeclaration, and every AxisDeclaration carries a bare
        required `label_ref: SourceRef`. So a ref-free envelope is no longer
        constructible and V0's raise is UNREACHABLE. Rather than keep an
        assertion that can never run the code it names -- an inert test, the
        exact failure mutation testing exposed twice in the previous
        milestone -- this pins what NOW rejects the case, and
        `TestCompositionAbsentGroundedThroughSeries::
        test_the_structural_chain_that_makes_v0_unreachable_is_intact`
        (tests/test_dataset_series.py) guards the three links that keep V0
        unreachable, so weakening any of them fails loudly.

        V0 itself has been DELETED, not retained: a guard that can never
        independently fire is worse than no guard at all, since a "passing"
        negative test for it would only ever be pinning one of these three
        earlier structural requirements instead. Grounding is now enforced
        STRUCTURALLY by the series/axes/label_ref chain above, and this test
        exists to pin exactly that -- not to exercise V0, which no longer
        exists.
        """
        graph = _minimal_graph("paper")
        with pytest.raises(ValidationError, match="at least 1 item"):
            DatasetEnvelope(
                source_graph=graph,
                composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                series=(),
                conversion_tables=(),
            )


class TestDatasetEnvelopeSeriesSingleRootArtifact:
    """V5: every SourceRef within a single Series must resolve to a node
    under the same parentless root artifact."""

    def test_series_spanning_two_root_papers_rejected(self) -> None:
        """Two separate PAPER_PDF nodes are each their own root (empty
        ancestors()). Grounding one axis's label_ref under "paper-a" and the
        other axis's label_ref under "paper-b" -- while every OTHER ref in
        the series stays on "paper-a" -- means only V5 can reject this: V1
        (both node_ids resolve), V2 (both nodes are directly targeted, so
        neither is decorative), V3 (a CaptionLabelKey TableCellLocator is
        compatible with PAPER_PDF for both locator kind and table_key kind),
        and V4 (source_form=TABULAR with TABLE_CELL value_refs) all pass.
        """
        paper_a = _node("paper-a", SourceNodeKind.PAPER_PDF, SHA_A)
        paper_b = _node("paper-b", SourceNodeKind.PAPER_PDF, SHA_B)
        graph = SourceGraph(nodes=(paper_a, paper_b))

        phi_axis = AxisDeclaration(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            label_raw="phi",
            label_ref=_table_ref("paper-a", row=0, col=0),
        )
        sl_axis = AxisDeclaration(
            axis_id="sl",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.VELOCITY,
            label_raw="S_L (cm/s)",
            label_ref=_table_ref("paper-b", row=0, col=1),
        )
        coordinate = Coordinate(
            axis_id="phi",
            value=_equivalence_ratio_amount(
                value_ref=_table_ref("paper-a", row=1, col=0), unit_ref=_table_ref("paper-a", row=1, col=1)
            ),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        observation = Observation(
            axis_id="sl",
            value=_velocity_amount(value_ref=coordinate.value.value_ref, unit_ref=coordinate.value.unit_ref),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        point = DataPoint(
            point_id="p1",
            coordinates=(coordinate,),
            observations=(observation,),
            composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
        )
        series = Series(
            series_id="s1",
            source_form=SourceForm.TABULAR,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=(phi_axis, sl_axis),
            constants=(),
            points=(point,),
        )
        with pytest.raises(ValidationError) as excinfo:
            DatasetEnvelope(
                source_graph=graph,
                composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
                series=(series,),
                conversion_tables=(_embedded_table_v1(),),
            )
        msg = str(excinfo.value)
        assert "spans multiple root artifacts" in msg
        assert "s1" in msg
        assert "paper-a" in msg
        assert "paper-b" in msg

    def test_series_referencing_different_nodes_under_same_root_accepted(self) -> None:
        """Unit inconsistency within one paper is legitimate: one axis's
        label_ref is grounded via a BBoxLocator on a FIGURE_CROP child
        ("crop"), while everything else stays on the PAPER_PDF parent
        ("paper") -- two different nodes, but a single root ("paper"),
        since ancestors("crop") == (paper,)."""
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, crop))

        phi_axis = AxisDeclaration(
            axis_id="phi",
            role=AxisRole.COORDINATE,
            quantity_kind=QuantityKind.EQUIVALENCE_RATIO,
            label_raw="phi",
            label_ref=_bbox_ref("crop"),
        )
        sl_axis = AxisDeclaration(
            axis_id="sl",
            role=AxisRole.OBSERVATION,
            quantity_kind=QuantityKind.VELOCITY,
            label_raw="S_L (cm/s)",
            label_ref=_table_ref("paper", row=0, col=1),
        )
        coordinate = Coordinate(
            axis_id="phi",
            value=_equivalence_ratio_amount(
                value_ref=_table_ref("paper", row=1, col=0), unit_ref=_table_ref("paper", row=1, col=1)
            ),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        observation = Observation(
            axis_id="sl",
            value=_velocity_amount(value_ref=coordinate.value.value_ref, unit_ref=coordinate.value.unit_ref),
            uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        )
        point = DataPoint(
            point_id="p1",
            coordinates=(coordinate,),
            observations=(observation,),
            composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
        )
        series = Series(
            series_id="s1",
            source_form=SourceForm.TABULAR,
            value_origin=ValueOrigin.EXPERIMENTAL,
            axes=(phi_axis, sl_axis),
            constants=(),
            points=(point,),
        )
        envelope = DatasetEnvelope(
            source_graph=graph,
            composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            series=(series,),
            conversion_tables=(_embedded_table_v1(),),
        )
        assert len(envelope.series) == 1


class TestDatasetEnvelopeLocatorKindCompatibility:
    """V3: which SourceLocator kinds may target which SourceNodeKinds."""

    def test_xpath_locator_rejected_against_non_jats_nodes(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.SI_MEMBER, SourceNodeKind.FIGURE_CROP):
            with pytest.raises(ValidationError) as excinfo:
                _envelope_with_value_ref_locator(XPathLocator(xpath="//a"), kind)
            msg = str(excinfo.value).lower()
            assert "xpath" in msg
            assert kind.value in msg

    def test_xpath_locator_accepted_against_jats_xml_node(self) -> None:
        envelope = _envelope_with_value_ref_locator(XPathLocator(xpath="//a"), SourceNodeKind.JATS_XML)
        assert envelope.composition == envelope.composition  # constructs without raising

    def test_table_cell_locator_rejected_against_figure_crop_node(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_value_ref_locator(
                TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0), SourceNodeKind.FIGURE_CROP
            )
        msg = str(excinfo.value).lower()
        assert "table_cell" in msg
        assert SourceNodeKind.FIGURE_CROP.value in msg

    def test_table_cell_locator_accepted_against_paper_pdf_jats_xml_and_si_member(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER):
            locator = TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0)
            _envelope_with_value_ref_locator(locator, kind)

    def test_bbox_locator_rejected_against_jats_xml_node(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_value_ref_locator(BBoxLocator(bbox=_bbox()), SourceNodeKind.JATS_XML)
        msg = str(excinfo.value).lower()
        assert "bbox" in msg
        assert SourceNodeKind.JATS_XML.value in msg

    def test_bbox_locator_accepted_against_paper_pdf_si_member_and_figure_crop(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.SI_MEMBER, SourceNodeKind.FIGURE_CROP):
            _envelope_with_value_ref_locator(BBoxLocator(bbox=_bbox()), kind)


class TestDatasetEnvelopeTableKeyKindCompatibility:
    """V3 (extended): a TableCellLocator's table_key kind must ALSO be
    compatible with the kind of node it targets -- a check distinct from,
    and layered on top of, the plain locator-kind check above."""

    def test_member_sheet_key_rejected_against_paper_pdf_node(self) -> None:
        """TABLE_CELL is itself compatible with PAPER_PDF (see
        test_table_cell_locator_accepted_against_paper_pdf_jats_xml_and_si_member
        above), so the old LocatorKind-level check alone would accept this --
        only the newer table_key-kind check can reject a MemberSheetKey
        against a PAPER_PDF node."""

        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_value_ref_locator(
                TableCellLocator(table_key=MemberSheetKey(sheet_name="Sheet1"), row=0, col=0), SourceNodeKind.PAPER_PDF
            )
        msg = str(excinfo.value)
        assert "table_key kind" in msg
        assert TableKeyKind.MEMBER_SHEET.value in msg
        assert SourceNodeKind.PAPER_PDF.value in msg

    def test_caption_label_key_accepted_against_paper_pdf_jats_xml_and_si_member(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER):
            locator = TableCellLocator(table_key=CaptionLabelKey(label="Table 1"), row=0, col=0)
            _envelope_with_value_ref_locator(locator, kind)

    def test_member_sheet_key_accepted_against_si_member_node(self) -> None:
        locator = TableCellLocator(table_key=MemberSheetKey(sheet_name="Sheet1"), row=0, col=0)
        _envelope_with_value_ref_locator(locator, SourceNodeKind.SI_MEMBER)


# --------------------------------------------------------------------------
# Functional / meta / frozen-ness
# --------------------------------------------------------------------------


class TestFunctionalRealisticEnvelope:
    """An end-to-end, realistic 3-node envelope: does it construct, and does
    it round-trip through JSON deterministically? This matters as much as
    the individual invariants -- DatasetEnvelope feeds a content-addressed
    store, where a non-deterministic dump is its own defect class."""

    def _build(self) -> DatasetEnvelope:
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        si = _node(
            "si",
            SourceNodeKind.SI_MEMBER,
            SHA_B,
            parent_node_id="paper",
            origin=ArchiveOrigin(archive_sha256=SHA_A, member_display_path="SI/data.xlsx"),
        )
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_C, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, si, crop))

        h2 = _component(
            "H2",
            value_ref=_table_ref("si", row=0, col=0),
            unit_ref=_table_ref("paper", row=0, col=0),
        )
        n2 = _component(
            "N2",
            value_ref=_table_ref("paper", row=1, col=1),
            unit_ref=_table_ref("paper", row=1, col=2),
        )
        o2 = _component(
            "O2",
            value_ref=_bbox_ref("crop"),
            unit_ref=_table_ref("paper", row=2, col=1),
        )
        composition = Composition(
            raw_name="H2/N2/O2 mixture",
            resolution=CompositionResolution.RESOLVED_COMPONENTS,
            basis=CompositionBasis.MOLE_FRACTION,
            equivalence_ratio=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            components=[h2, n2, o2],
        )
        return DatasetEnvelope(
        source_graph=graph,
        composition=composition,
        series=(_fully_populated_series("paper"),),
        conversion_tables=(_embedded_table_v1(),),
    )

    def test_constructs(self) -> None:
        envelope = self._build()
        assert len(envelope.composition.components) == 3  # type: ignore[union-attr]

    def test_json_round_trip_preserves_equality(self) -> None:
        envelope = self._build()
        dumped = envelope.model_dump_json()
        restored = DatasetEnvelope.model_validate_json(dumped)
        assert restored == envelope

    def test_double_dump_is_byte_identical(self) -> None:
        envelope = self._build()
        first = envelope.model_dump_json()
        second = envelope.model_dump_json()
        assert first == second


class TestRefWalkCannotBeOutgrown:
    """A future field typed to carry a SourceRef must not be able to add
    itself to DatasetEnvelope without the integrity walk (V1/V2, and
    iter_source_refs generally) also seeing it -- silently skipping a
    location would let a dangling or decorative ref hide from validation."""

    def test_iter_source_refs_reports_a_ref_from_every_field_location_that_can_carry_one(self) -> None:
        expected_paths = _collect_field_paths_carrying_source_ref(DatasetEnvelope)
        assert expected_paths, (
            "the type walk itself found no SourceRef-carrying field on DatasetEnvelope -- "
            "the walker is broken, not the schema"
        )
        envelope = _fully_populated_envelope()
        produced_paths = {_strip_list_indices(path) for path, _ in iter_source_refs(envelope)}
        missing = expected_paths - produced_paths
        assert not missing, (
            "iter_source_refs did not report a ref from every field location that "
            f"DatasetEnvelope's own type annotations say can carry a SourceRef: {sorted(missing)!r}. "
            "A future SourceRef-bearing field must not be able to silently evade the V1/V2 "
            "integrity check -- if this fails, either the walker's field-location list or "
            "iter_source_refs' traversal has fallen out of sync with the schema."
        )

    def test_no_field_reachable_from_dataset_envelope_has_an_unwalkable_annotation(self) -> None:
        """The test above can be defeated by a single vague annotation: a
        field typed `typing.Any`, bare `object`, or a `set`/`frozenset` is
        invisible to BOTH _collect_field_paths_carrying_source_ref (the
        annotation-side check above) AND iter_source_refs (the runtime
        walker) at once -- so a SourceRef tucked inside such a field would
        never show up as "expected" and never show up as "missing" either;
        the test above would stay green while a ref silently evaded V1/V2.

        This test closes that hole directly: it walks every field
        annotation reachable from DatasetEnvelope against an explicit
        allowlist of shapes both walkers actually know how to traverse,
        and fails loudly, naming the offending path and annotation, the
        moment anything falls outside that allowlist."""
        unwalkable = _unwalkable_annotations(DatasetEnvelope)
        assert not unwalkable, (
            "found field(s) reachable from DatasetEnvelope whose annotation shape is not on the "
            f"walkable allowlist: {sorted(unwalkable.items(), key=lambda item: item[0])!r}. "
            "An annotation shape like typing.Any, a bare `object`, or an unwalkable container "
            "(e.g. set/frozenset) could hide a SourceRef from both the annotation-side check "
            "(TestRefWalkCannotBeOutgrown above) and the runtime walker (iter_source_refs) at "
            "the same time -- widen the allowlist in _inspect_annotation_walkability only after "
            "confirming iter_source_refs itself can actually traverse the new shape."
        )


class TestMeasuredValueWalkCannotBeOutgrown:
    """Mirrors TestRefWalkCannotBeOutgrown above, but for iter_measured_values
    (the choke point DatasetEnvelope's T2 conversion_tables cover-exactly
    check runs through) instead of iter_source_refs.

    Only the direct annotation-vs-runtime coverage test needs a
    MeasuredValue-targeted twin here: the "unwalkable annotation" half
    (test_no_field_reachable_from_dataset_envelope_has_an_unwalkable_annotation
    above) is target-agnostic -- an annotation shape neither walker can
    traverse (typing.Any, bare object, set/frozenset) hides ANY target type
    equally, regardless of whether the walker is looking for a SourceRef or
    a MeasuredValue -- so it already covers iter_measured_values too and is
    not repeated here."""

    def test_iter_measured_values_reports_a_value_from_every_field_location_that_can_carry_one(self) -> None:
        expected_paths = _collect_field_paths_carrying_measured_value(DatasetEnvelope)
        assert expected_paths, (
            "the type walk itself found no MeasuredValue-carrying field on DatasetEnvelope -- "
            "the walker is broken, not the schema"
        )
        envelope = _fully_populated_envelope()
        produced_paths = {_strip_list_indices(path) for path, _ in iter_measured_values(envelope)}
        missing = expected_paths - produced_paths
        assert not missing, (
            "iter_measured_values did not report a value from every field location that "
            f"DatasetEnvelope's own type annotations say can carry a MeasuredValue: {sorted(missing)!r}. "
            "A future MeasuredValue-bearing field must not be able to silently evade the T2 "
            "conversion_tables cover-exactly check -- if this fails, either the walker's "
            "field-location list or iter_measured_values' traversal has fallen out of sync "
            "with the schema."
        )

    def test_finds_values_nested_inside_series_points_coordinates_and_observations(self) -> None:
        """Positive coverage of the nesting iter_source_refs never has to
        traverse: Series -> DataPoint -> Coordinate/Observation, several
        BaseModel layers below DatasetEnvelope.series itself."""
        envelope = _fully_populated_envelope()
        found = {_strip_list_indices(path) for path, _ in iter_measured_values(envelope)}
        assert "series.constants.value" in found
        assert "series.points.coordinates.value" in found
        assert "series.points.observations.value" in found

    def test_finds_values_nested_inside_composition_components(self) -> None:
        """Positive coverage of both places a Composition (and therefore its
        components' amounts) can appear: the envelope's own top-level
        composition, and a DataPoint's own per-point composition override."""
        envelope = _fully_populated_envelope()
        found = {_strip_list_indices(path) for path, _ in iter_measured_values(envelope)}
        assert "composition.components.amount" in found
        assert "series.points.composition.components.amount" in found


class TestEmbeddedConversionTableT1:
    """T1: an EmbeddedConversionTable must reconstruct, byte-for-byte, back
    to the exact ConversionTable its declared sha256 names -- each of the
    four independent checks that invariant depends on is exercised in
    isolation here, plus the sha256 field's own shape validator."""

    def test_canonical_json_that_does_not_parse_as_json_rejected(self) -> None:
        with pytest.raises(ValidationError, match="does not parse as JSON"):
            EmbeddedConversionTable(sha256=TABLE_V1.sha256, canonical_json="{not json")

    def test_canonical_json_that_is_structurally_invalid_rejected(self) -> None:
        """Valid JSON, but not a shape from_identity_payload accepts (e.g. a
        JSON object missing the required table_id/version/base_units/
        aliases/rules keys)."""
        with pytest.raises(ValidationError, match="does not decode to a structurally valid ConversionTable"):
            EmbeddedConversionTable(sha256=TABLE_V1.sha256, canonical_json="{}")

    def test_canonical_json_whose_reconstructed_table_hashes_to_a_different_sha256_rejected(self) -> None:
        """A structurally valid table, but the *declared* sha256 field names
        some OTHER table -- i.e. the payload was swapped or corrupted after
        the sha256 was computed."""
        other_table = _second_conversion_table()
        with pytest.raises(ValidationError, match="not the declared sha256"):
            EmbeddedConversionTable(
                sha256=TABLE_V1.sha256,
                canonical_json=canonical_json_bytes(other_table.identity_payload()).decode("utf-8"),
            )

    def test_canonical_json_that_is_valid_but_not_canonically_rendered_rejected(self) -> None:
        """A JSON string that (a) parses to the identity_payload of a real,
        valid table, (b) reconstructs to the SAME sha256 as declared, but
        (c) is not byte-for-byte what canonical_json_bytes would have
        produced for that payload (extra whitespace/indentation here,
        instead of canonical_json_bytes' compact separators) -- this is the
        one T1 check the other three cannot exercise, since it requires a
        payload that is otherwise entirely valid and self-consistent."""
        payload = TABLE_V1.identity_payload()
        non_canonical = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        canonical = canonical_json_bytes(payload).decode("utf-8")
        assert non_canonical != canonical, "test setup bug: the two renderings must actually differ byte-for-byte"
        assert json.loads(non_canonical) == json.loads(canonical), (
            "test setup bug: the two renderings must parse to the same payload"
        )
        with pytest.raises(ValidationError, match="is not the canonical rendering of the table it decodes to"):
            EmbeddedConversionTable(sha256=TABLE_V1.sha256, canonical_json=non_canonical)

    def test_sha256_that_is_not_64_lowercase_hex_characters_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is not 64 lowercase hex characters"):
            EmbeddedConversionTable(
                sha256="not-a-hash",
                canonical_json=canonical_json_bytes(TABLE_V1.identity_payload()).decode("utf-8"),
            )

    def test_a_genuinely_valid_embedding_is_accepted(self) -> None:
        embedded = _embedded_table_v1()
        assert embedded.sha256 == TABLE_V1.sha256


class TestEmbeddedConversionTableUntrustedJsonResourceGuard:
    """canonical_json arrives embedded in a stored dataset file -- untrusted
    input -- and is handed to json.loads. Both of these prove the resource
    guard actually bounds that untrusted input, rather than trusting
    json.JSONDecodeError alone to cover every hostile shape."""

    def test_oversized_canonical_json_rejected(self) -> None:
        """A canonical_json string longer than
        _MAX_EMBEDDED_CANONICAL_JSON_LENGTH must be rejected by pydantic's
        own max_length, before json.loads is ever called on it."""
        oversized = "[" + "1," * 1_048_576 + "1]"
        with pytest.raises(ValidationError, match="at most"):
            EmbeddedConversionTable(sha256=TABLE_V1.sha256, canonical_json=oversized)

    def test_deeply_nested_canonical_json_rejected_not_a_bare_recursion_error(self) -> None:
        """A canonical_json string that is well within the length bound but
        nests deeply enough to raise a bare RecursionError from inside
        json.loads itself -- proves the broadened `except` actually catches
        it and converts it into a ValidationError, rather than letting the
        RecursionError propagate uncaught out of model construction."""
        depth = 100_000
        deeply_nested = "[" * depth + "]" * depth
        assert len(deeply_nested) <= _MAX_EMBEDDED_CANONICAL_JSON_LENGTH, (
            "test setup bug: this payload must be short enough to pass the length guard, so it "
            "actually exercises the RecursionError catch rather than the length check"
        )
        with pytest.raises(ValidationError, match="does not parse as JSON"):
            EmbeddedConversionTable(sha256=TABLE_V1.sha256, canonical_json=deeply_nested)


class TestDatasetEnvelopeConversionTablesCoverExactly:
    """T2: DatasetEnvelope.conversion_tables must embed exactly the tables
    actually cited by some MeasuredValue.conversion_table_sha256 reachable
    from the envelope -- no fewer (a cited table left un-embedded), and no
    more (a decorative table nobody cites)."""

    def test_missing_a_cited_table_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is missing table\\(s\\)"):
            _fully_populated_envelope(conversion_tables=())

    def test_embedding_an_uncited_table_rejected(self) -> None:
        with pytest.raises(ValidationError, match="embeds decorative table\\(s\\)"):
            _fully_populated_envelope(conversion_tables=(_embedded_table_v1(), _embedded_second_table()))

    def test_embedding_exactly_the_cited_tables_accepted(self) -> None:
        envelope = _fully_populated_envelope(conversion_tables=(_embedded_table_v1(),))
        assert envelope.conversion_tables == (_embedded_table_v1(),)


class TestDatasetEnvelopeConversionTablesNoDuplicateSha256:
    """DUPLICATE-CONVERSION-TABLE-SHA256-GUARD: conversion_tables must not
    embed the same sha256 more than once. T2's cover-exactly check compares
    SETS of embedded sha256s against cited sha256s, so a tuple like
    ``(V1, V1)`` has the same embedded set as ``(V1,)`` and would otherwise
    pass T2 unchanged -- this guard is what actually rejects the duplicate."""

    def test_duplicate_embedded_table_rejected(self) -> None:
        with pytest.raises(ValidationError, match="embeds duplicate sha256"):
            _fully_populated_envelope(conversion_tables=(_embedded_table_v1(), _embedded_table_v1()))

    def test_distinct_tables_still_accepted(self) -> None:
        with _registered_second_table():
            first = _embedded_table_v1()
            second = _embedded_second_table()
            ascending = tuple(sorted((first, second), key=lambda table: table.sha256))
            envelope = _envelope_citing_two_tables(ascending)
            assert envelope.conversion_tables == ascending


class TestDatasetEnvelopeConversionTablesSorted:
    """T3: DatasetEnvelope.conversion_tables must be sorted ascending by
    sha256 -- exercised with two genuinely-distinct, both-cited tables so a
    sort-order failure can be triggered without also tripping T2."""

    def test_conversion_tables_out_of_sha256_order_rejected(self) -> None:
        with _registered_second_table():
            first = _embedded_table_v1()
            second = _embedded_second_table()
            ascending = sorted((first, second), key=lambda table: table.sha256)
            descending = tuple(reversed(ascending))
            assert descending != tuple(ascending), "test setup bug: the two tables must actually sort differently"
            with pytest.raises(ValidationError, match="must be sorted ascending by sha256"):
                _envelope_citing_two_tables(descending)

    def test_conversion_tables_in_sha256_order_accepted(self) -> None:
        with _registered_second_table():
            first = _embedded_table_v1()
            second = _embedded_second_table()
            ascending = tuple(sorted((first, second), key=lambda table: table.sha256))
            envelope = _envelope_citing_two_tables(ascending)
            assert envelope.conversion_tables == ascending


class TestModelsAreFrozen:
    """Mirrors test_dataset_schemas.py's TestModelsAreFrozen for the two new
    models: frozen=True closes off plain attribute assignment, not
    model_copy(update=...)."""

    def test_source_graph_rejects_attribute_assignment(self) -> None:
        graph = SourceGraph(nodes=(_node("a"),))
        with pytest.raises(ValidationError, match="frozen"):
            graph.nodes = (_node("a"), _node("b", SourceNodeKind.JATS_XML, SHA_B))  # type: ignore[misc]

    def test_dataset_envelope_rejects_attribute_assignment(self) -> None:
        envelope = _fully_populated_envelope()
        with pytest.raises(ValidationError, match="frozen"):
            envelope.composition = Absent(reason=AbsenceReason.NOT_APPLICABLE)  # type: ignore[misc]
