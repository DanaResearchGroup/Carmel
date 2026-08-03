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

import re
from enum import Enum
from typing import Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ArchiveOrigin,
    BBox,
    BBoxLocator,
    ComponentRole,
    Composition,
    CompositionBasis,
    CompositionComponent,
    CompositionResolution,
    CoordinateFrame,
    DatasetEnvelope,
    Maybe,
    MeasuredValue,
    QuantityKind,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    XPathLocator,
    iter_source_refs,
)
from carmel.services.units import TABLE_V1

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64

_NO_ORIGIN = Absent(reason=AbsenceReason.NOT_APPLICABLE)
"""Module-level singleton default for SourceNode.origin -- Absent is frozen,
so sharing one instance across every _node() call that doesn't need a
concrete ArchiveOrigin is safe, and avoids a function-call-in-argument-default
(ruff B008)."""


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
) -> SourceNode:
    return SourceNode(node_id=node_id, kind=kind, sha256=sha256, parent_node_id=parent_node_id, origin=origin)


def _bbox_ref(node_id: str) -> SourceRef:
    return SourceRef(node_id=node_id, locator=BBoxLocator(bbox=_bbox()))


def _table_ref(node_id: str, row: int = 0, col: int = 1) -> SourceRef:
    return SourceRef(node_id=node_id, locator=TableCellLocator(row=row, col=col))


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
    return DatasetEnvelope(source_graph=graph, composition=composition)


def _fully_populated_envelope() -> DatasetEnvelope:
    """A DatasetEnvelope with every SourceRef-bearing field actually
    populated (including equivalence_ratio, which is Absent in most other
    fixtures here) -- used by the ref-walk drift meta-test."""
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
    return DatasetEnvelope(source_graph=graph, composition=composition)


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
    if annotation is SourceRef or annotation in _ALLOWED_PRIMITIVE_ANNOTATIONS:
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
            DatasetEnvelope(source_graph=graph, composition=composition)
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
            DatasetEnvelope(source_graph=graph, composition=composition)
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
            DatasetEnvelope(source_graph=graph, composition=composition)
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
            DatasetEnvelope(source_graph=graph, composition=composition)
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
            DatasetEnvelope(source_graph=graph, composition=composition)

    def test_unreferenced_ancestor_of_a_referenced_node_is_allowed(self) -> None:
        """The PAPER_PDF root is never targeted directly, but its
        FIGURE_CROP child is -- an ancestor of a targeted node is not
        decorative."""
        paper = _node("paper", SourceNodeKind.PAPER_PDF, SHA_A)
        crop = _node("crop", SourceNodeKind.FIGURE_CROP, SHA_B, parent_node_id="paper")
        graph = SourceGraph(nodes=(paper, crop))
        composition = _resolved_single_component_composition(
            value_ref=_bbox_ref("crop"),
            unit_ref=SourceRef(node_id="crop", locator=BBoxLocator(bbox=_bbox(x0="1", y0="1", x1="2", y1="2"))),
        )
        envelope = DatasetEnvelope(source_graph=graph, composition=composition)
        assert envelope.source_graph.node("paper").parent_node_id is None

    def test_an_envelope_citing_nothing_is_rejected_as_ungrounded(self) -> None:
        """V0, and it MUST match on "ungrounded" rather than on a bare
        ValidationError.

        V0 is subsumed by V2 as a rejection: SourceGraph requires at least one
        node, so an envelope with zero SourceRefs always leaves some node
        unreferenced, and V2 would reject it anyway. Mutation testing proved
        the consequence -- with V0's raise neutralised, all 48 tests still
        passed, because this test was being satisfied by V2 next door.

        V0 is kept, and pinned this way, because it is not redundant as a
        DIAGNOSIS. "this envelope cites nothing at all" and "node 'paper' is
        decorative" describe very different mistakes to whoever has to fix the
        extractor, and V0 runs first precisely so the clearer one wins. Matching
        its distinctive word is what keeps that promise honest.

        Note the expiry: composition=Absent is unconstructible only while
        composition is the sole ref-bearing field. When the series aggregate
        (M-D2b part a) lands, an Absent composition alongside ref-bearing
        series becomes a legitimate, constructible state and this test's
        premise changes."""
        graph = _minimal_graph("paper")
        with pytest.raises(ValidationError, match="ungrounded"):
            DatasetEnvelope(source_graph=graph, composition=Absent(reason=AbsenceReason.NOT_APPLICABLE))


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
            _envelope_with_value_ref_locator(TableCellLocator(row=0, col=0), SourceNodeKind.FIGURE_CROP)
        msg = str(excinfo.value).lower()
        assert "table_cell" in msg
        assert SourceNodeKind.FIGURE_CROP.value in msg

    def test_table_cell_locator_accepted_against_paper_pdf_jats_xml_and_si_member(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.JATS_XML, SourceNodeKind.SI_MEMBER):
            _envelope_with_value_ref_locator(TableCellLocator(row=0, col=0), kind)

    def test_bbox_locator_rejected_against_jats_xml_node(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _envelope_with_value_ref_locator(BBoxLocator(bbox=_bbox()), SourceNodeKind.JATS_XML)
        msg = str(excinfo.value).lower()
        assert "bbox" in msg
        assert SourceNodeKind.JATS_XML.value in msg

    def test_bbox_locator_accepted_against_paper_pdf_si_member_and_figure_crop(self) -> None:
        for kind in (SourceNodeKind.PAPER_PDF, SourceNodeKind.SI_MEMBER, SourceNodeKind.FIGURE_CROP):
            _envelope_with_value_ref_locator(BBoxLocator(bbox=_bbox()), kind)


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
        return DatasetEnvelope(source_graph=graph, composition=composition)

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
