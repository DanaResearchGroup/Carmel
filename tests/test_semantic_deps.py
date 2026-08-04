# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for carmel.services.semantic_deps: content-addressed identity for a
versioned heuristic's code.

Pins four kinds of fact: (1) the seeded dependency's content_sha256 against a
HARDCODED literal, computed independently of the live registry, so a change
to either numeric.py's repair heuristic or the toolchain's ast.dump rendering
is caught loudly; (2) the tripwire actually fires on a semantic code change
and does NOT fire on a purely cosmetic one, demonstrated via synthetic
in-memory perturbations of numeric.py's source text; (3) the registry is
genuinely append-only (never shrinks) and genuinely read-only
(MappingProxyType rejects mutation); (4) the two error paths
(dependency_for_sha, current_sha_for) raise the one error type this module
defines, and only for the reason each test intends.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import sys

import pytest

from carmel.agents.tools import extract
from carmel.services import numeric
from carmel.services.semantic_deps import (
    CURRENT_SHA_BY_DEPENDENCY_ID,
    DEPENDENCIES_BY_SHA,
    EXTRACT_TEXT_DEPENDENCY_ID,
    ExtractionIdentity,
    InputPolicy,
    SemanticDependencyDefinition,
    SemanticDependencyInvariantError,
    UnknownSemanticDependencyError,
    _assert_no_carmel_imports,
    _pypdf_version,
    compute_dependency_sha,
    current_sha_for,
    dependency_for_sha,
    extraction_identity,
)

# HARDCODED, not derived from the live registry (never `next(iter(DEPENDENCIES_BY_SHA))`
# or similar self-referential pull -- a tripwire computed from the thing it is meant to
# watch never fires on drift). If this test ever fails, it means ONE of three things:
#
#   1. carmel/services/numeric.py's repair heuristic
#      (normalize_numeric_span and everything it transitively calls at module scope,
#      plus REPAIR_NAMES -- both are seeded entry points) actually changed, OR
#   2. numeric.py's module-level import bindings changed (compute_dependency_sha
#      hashes them too, precisely so a swap like `import re` -> `import regex as re`
#      is not silently invisible), OR
#   3. this toolchain's ast.dump() rendering changed (a Python version bump).
#
# The fix in ANY case is to ADD A NEW registry entry in
# carmel/services/semantic_deps.py for the new sha -- NEVER edit or replace this
# pinned literal, and NEVER edit/replace the existing DEPENDENCIES_BY_SHA row for
# "carmel.numeric.context_free_span_repair". That row documents a real piece of
# history (what code produced which stored MeasuredValue.repairs) and mutating it
# in place would silently invalidate that history's meaning.
_PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256 = (
    "b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb"
)

# Pinned for carmel.numeric.glyph_health on exactly the same terms as the literal
# above: hardcoded, never recomputed from the live module, never edited in place.
# Entry point: assess_glyph_health.
_PINNED_GLYPH_HEALTH_SHA256 = (
    "af3553a8142b50bba56b6ba164778b4cd2bff6e4916ac2e93c4e1a270ba4ab5a"
)

# The set of sha256 digests this registry has ever shipped, as of this test's writing.
# Used only to assert DEPENDENCIES_BY_SHA never drops a previously-shipped entry (see
# test_registry_is_append_only_and_never_drops_a_previously_shipped_sha below).
_HISTORICALLY_SHIPPED_SHAS = frozenset(
    {_PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256, _PINNED_GLYPH_HEALTH_SHA256}
)


def _real_numeric_source() -> str:
    return inspect.getsource(numeric)


def test_seeded_dependency_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(
        _real_numeric_source(), ["normalize_numeric_span", "REPAIR_NAMES"]
    )
    assert computed == _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256, (
        "carmel/services/numeric.py's repair heuristic (or this toolchain's ast.dump "
        "rendering) has changed since carmel.numeric.context_free_span_repair was "
        "pinned. Fix: ADD A NEW entry to DEPENDENCIES_BY_SHA in "
        "carmel/services/semantic_deps.py for the new sha -- do not edit or replace "
        "the existing pinned entry or this test's hardcoded literal."
    )


def test_registry_seed_agrees_with_the_pin() -> None:
    """The live registry's seeded entry itself must also equal the pin.

    Separate from the previous test (which recomputes independently from raw source):
    this one checks that DEPENDENCIES_BY_SHA as actually constructed at import time
    contains that same sha, catching a bug where the registry seeding logic itself
    diverges from a bare compute_dependency_sha call.
    """
    assert _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256 in DEPENDENCIES_BY_SHA
    entry = DEPENDENCIES_BY_SHA[_PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256]
    assert entry.dependency_id == "carmel.numeric.context_free_span_repair"
    assert entry.input_policy is InputPolicy.SIBLING_FIELD


def test_calibration_closure_is_exactly_the_expected_twelve_names() -> None:
    """The transitive closure of {normalize_numeric_span, REPAIR_NAMES} in the
    real module is exactly these 12 names -- pinned separately from the sha
    itself so a closure-membership regression and an unparse/dump-format
    regression can be told apart by which of these two tests fails.

    REPAIR_NAMES is included as a second seeded entry point (alongside
    normalize_numeric_span) because MeasuredValue._validate_repair_names
    validates every stored repair name against REPAIR_NAMES; that constant is
    therefore also part of what a content_sha256 for this dependency must
    attest to, not just the repair function itself."""
    source = _real_numeric_source()
    tree = ast.parse(source)
    definitions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node

    closure: set[str] = set()
    frontier = ["normalize_numeric_span", "REPAIR_NAMES"]
    while frontier:
        name = frontier.pop()
        if name in closure or name not in definitions:
            continue
        closure.add(name)
        for child in ast.walk(definitions[name]):
            if isinstance(child, ast.Name) and child.id in definitions and child.id not in closure:
                frontier.append(child.id)

    expected = {
        "GlyphHealth",
        "NormalizedNumeral",
        "REPAIR_NAMES",
        "SourceContext",
        "Unresolvable",
        "_ASCII6_UNCERTAINTY_RE",
        "_CORE_VALUE_RE",
        "_DISALLOWED_LITERALS",
        "_find_range_separator",
        "_normalize_single_value",
        "_refuse_common",
        "normalize_numeric_span",
    }
    assert closure == expected


def test_glyph_health_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(_real_numeric_source(), ["assess_glyph_health"])
    assert computed == _PINNED_GLYPH_HEALTH_SHA256, (
        "carmel/services/numeric.py's glyph-health assessment (or this toolchain's "
        "ast.dump rendering) has changed since carmel.numeric.glyph_health was pinned. "
        "Fix: ADD A NEW entry to DEPENDENCIES_BY_SHA in "
        "carmel/services/semantic_deps.py for the new sha -- do not edit or replace "
        "the existing pinned entry or this test's hardcoded literal."
    )


def test_glyph_health_registry_seed_agrees_with_the_pin() -> None:
    assert _PINNED_GLYPH_HEALTH_SHA256 in DEPENDENCIES_BY_SHA
    entry = DEPENDENCIES_BY_SHA[_PINNED_GLYPH_HEALTH_SHA256]
    assert entry.dependency_id == "carmel.numeric.glyph_health"
    assert entry.input_policy is InputPolicy.EXTERNAL_DIGEST_REQUIRED


def test_glyph_health_and_repair_chain_have_different_shas() -> None:
    """The two dependencies must never collapse into one identity.

    If these ever coincide, a change to one heuristic would silently re-address
    records produced by the other.
    """
    assert _PINNED_GLYPH_HEALTH_SHA256 != _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256


def test_declaring_glyph_health_as_a_second_entry_point_does_not_change_the_sha() -> None:
    """GlyphHealth is already pulled into the closure transitively by
    assess_glyph_health, so naming it explicitly must be a no-op.

    This is what justifies seeding the dependency with a SINGLE entry point: it
    proves the single entry point is not an under-specification.
    """
    source = _real_numeric_source()
    one = compute_dependency_sha(source, ["assess_glyph_health"])
    two = compute_dependency_sha(source, ["assess_glyph_health", "GlyphHealth"])
    assert one == two == _PINNED_GLYPH_HEALTH_SHA256


def test_editing_glyph_health_does_not_re_address_the_repair_chain() -> None:
    """The two closures are genuinely scoped, in the direction that matters.

    Mutating a marker that lives ONLY in assess_glyph_health must move the
    glyph-health sha and leave the repair-chain sha untouched. Without this,
    'separate dependency ids' would be a naming convention rather than a
    property of the code.
    """
    source = _real_numeric_source()
    mutated = source.replace('has_thorn_plus_marker="þ" in', 'has_thorn_plus_marker="Þ" in')
    assert mutated != source, "the mutation target vanished; update this test"

    assert compute_dependency_sha(mutated, ["assess_glyph_health"]) != _PINNED_GLYPH_HEALTH_SHA256
    assert (
        compute_dependency_sha(mutated, ["normalize_numeric_span", "REPAIR_NAMES"])
        == _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256
    )


def test_glyph_health_closure_is_exactly_the_expected_four_names() -> None:
    """Pinned separately from the sha so a closure-membership regression and an
    unparse/dump-format regression can be told apart by which test fails.

    The overlap with the repair closure is exactly {GlyphHealth,
    _ASCII6_UNCERTAINTY_RE} -- both are genuine shared references, not an
    over-broad walk.
    """
    source = _real_numeric_source()
    tree = ast.parse(source)
    definitions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node

    closure: set[str] = set()
    frontier = ["assess_glyph_health"]
    while frontier:
        name = frontier.pop()
        if name in closure or name not in definitions:
            continue
        closure.add(name)
        for child in ast.walk(definitions[name]):
            if isinstance(child, ast.Name) and child.id in definitions and child.id not in closure:
                frontier.append(child.id)

    assert closure == {
        "GlyphHealth",
        "_ASCII6_UNCERTAINTY_RE",
        "_BARE_DASH_CORRUPTION_RE",
        "assess_glyph_health",
    }


def test_determinism_same_input_same_sha() -> None:
    source = _real_numeric_source()
    first = compute_dependency_sha(source, ["normalize_numeric_span"])
    second = compute_dependency_sha(source, ["normalize_numeric_span"])
    assert first == second


def test_changing_a_regex_constant_changes_the_sha() -> None:
    source = _real_numeric_source()
    perturbed = source.replace(
        '_ASCII6_UNCERTAINTY_RE = re.compile(r"\\d+\\s+6\\s+\\d+")',
        '_ASCII6_UNCERTAINTY_RE = re.compile(r"\\d+\\s+9\\s+\\d+")',
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_changing_a_branch_in_find_range_separator_changes_the_sha() -> None:
    source = _real_numeric_source()
    perturbed = source.replace(
        'if ch in ("-", "–") and i > 0 and text[i - 1] not in ("e", "E"):',
        'if ch in ("-", "–") and i > 0 and text[i - 1] not in ("e", "E", "x"):',
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_adding_a_comment_does_not_change_the_sha() -> None:
    source = _real_numeric_source()
    perturbed = source.replace(
        "def _find_range_separator(text: str) -> int | None:",
        "# a brand new comment that changes no behavior at all\ndef _find_range_separator(text: str) -> int | None:",
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha == original_sha


def test_reformatting_whitespace_does_not_change_the_sha() -> None:
    source = _real_numeric_source()
    perturbed = source.replace(
        "    for i, ch in enumerate(text):\n"
        '        if ch in ("-", "–") and i > 0 and text[i - 1] not in ("e", "E"):\n'
        "            return i\n"
        "    return None\n",
        "    for i, ch in enumerate(text):\n"
        "        if (\n"
        '            ch in ("-", "–")\n'
        "            and i > 0\n"
        '            and text[i - 1] not in ("e", "E")\n'
        "        ):\n"
        "            return i\n"
        "\n"
        "    return None\n",
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha == original_sha


def test_changing_only_a_module_level_docstring_does_not_change_the_sha() -> None:
    source = _real_numeric_source()
    perturbed = source.replace(
        "def _find_range_separator(text: str) -> int | None:\n"
        '    """Return the index of the hyphen/en-dash',
        "def _find_range_separator(text: str) -> int | None:\n"
        '    """COMPLETELY DIFFERENT DOCSTRING TEXT. Return the index of the hyphen/en-dash',
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha == original_sha


def test_changing_only_a_nested_function_docstring_does_not_change_the_sha() -> None:
    """Docstring stripping must be recursive: a change to a docstring nested inside a
    closure member's own nested function/class body must also not change the sha."""
    source = _real_numeric_source()

    def with_nested_docstring(marker: str) -> str:
        tree = ast.parse(source)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "normalize_numeric_span":
                target = node
                break
        assert target is not None
        nested_def = ast.FunctionDef(
            name="_synthetic_nested",
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=[ast.Expr(value=ast.Constant(value=marker)), ast.Return(value=ast.Constant(value=None))],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        target.body.insert(0, nested_def)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    perturbed_a = with_nested_docstring("docstring version A")
    perturbed_b = with_nested_docstring("docstring version B")
    assert perturbed_a != perturbed_b, "the two synthetic sources must differ textually"
    sha_a = compute_dependency_sha(perturbed_a, ["normalize_numeric_span"])
    sha_b = compute_dependency_sha(perturbed_b, ["normalize_numeric_span"])
    assert sha_a == sha_b, "a nested function's docstring-only difference must not change the sha"


def test_registry_is_append_only_and_never_drops_a_previously_shipped_sha() -> None:
    missing = _HISTORICALLY_SHIPPED_SHAS - set(DEPENDENCIES_BY_SHA)
    assert not missing, (
        f"DEPENDENCIES_BY_SHA is missing previously-shipped sha256(es) {sorted(missing)!r}. "
        "Removing a registry entry orphans every previously-stored dataset that was "
        "validated against it -- entries must only ever be ADDED, never removed."
    )


def test_dependencies_by_sha_is_read_only() -> None:
    with pytest.raises(TypeError):
        DEPENDENCIES_BY_SHA["not-a-real-sha"] = None  # type: ignore[index]


def test_dependency_for_sha_unknown_sha_raises_unknown_semantic_dependency_error() -> None:
    with pytest.raises(UnknownSemanticDependencyError) as excinfo:
        dependency_for_sha("0" * 64)
    message = str(excinfo.value)
    assert "no semantic dependency known for content_sha256" in message


def test_dependency_for_sha_malformed_sha_raises_unknown_semantic_dependency_error() -> None:
    for malformed in ("too-short", "G" * 64, ("a" * 63), "A" * 64):
        with pytest.raises(UnknownSemanticDependencyError) as excinfo:
            dependency_for_sha(malformed)
        message = str(excinfo.value)
        assert "no semantic dependency known for content_sha256" in message


def test_current_sha_for_unknown_dependency_id_raises_unknown_semantic_dependency_error() -> None:
    with pytest.raises(UnknownSemanticDependencyError) as excinfo:
        current_sha_for("carmel.does_not_exist.made_up")
    message = str(excinfo.value)
    assert "no semantic dependency known for dependency_id" in message


def test_current_sha_for_seeded_dependency_matches_the_pin() -> None:
    assert current_sha_for("carmel.numeric.context_free_span_repair") == _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256


def test_semantic_dependency_definition_rejects_malformed_sha256() -> None:
    with pytest.raises(Exception) as excinfo:
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing",
            content_sha256="not-a-valid-sha",
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        )
    assert "content_sha256" in str(excinfo.value)


def test_semantic_dependency_definition_rejects_empty_dependency_id() -> None:
    with pytest.raises(Exception) as excinfo:
        SemanticDependencyDefinition(
            dependency_id="",
            content_sha256="0" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        )
    assert "dependency_id" in str(excinfo.value)


def test_semantic_dependency_definition_rejects_non_slug_dependency_id() -> None:
    with pytest.raises(Exception) as excinfo:
        SemanticDependencyDefinition(
            dependency_id="NotLowercase",
            content_sha256="0" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        )
    assert "dependency_id" in str(excinfo.value)


def test_semantic_dependency_definition_rejects_non_input_policy_input_policy() -> None:
    """A stdlib dataclass does not enforce field types at runtime -- this
    checks that SemanticDependencyDefinition.__post_init__ validates
    input_policy is genuinely an InputPolicy member rather than trusting the
    type annotation alone."""
    with pytest.raises(Exception) as excinfo:
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing",
            content_sha256="0" * 64,
            input_policy="sibling_field",  # type: ignore[arg-type]
            is_current=True,
        )
    assert "input_policy" in str(excinfo.value)


def test_build_registry_rejects_duplicate_content_sha256() -> None:
    from carmel.services.semantic_deps import _build_registry

    entries = (
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing_a",
            content_sha256="1" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        ),
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing_b",
            content_sha256="1" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        ),
    )
    with pytest.raises(Exception) as excinfo:
        _build_registry(entries)
    assert "registered more than once" in str(excinfo.value)


def test_build_registry_rejects_two_current_entries_for_the_same_dependency_id() -> None:
    from carmel.services.semantic_deps import _build_registry

    entries = (
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing",
            content_sha256="1" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        ),
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing",
            content_sha256="2" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        ),
    )
    with pytest.raises(Exception) as excinfo:
        _build_registry(entries)
    assert "more than one entry with" in str(excinfo.value)


def test_build_registry_rejects_a_dependency_id_with_no_current_entry() -> None:
    from carmel.services.semantic_deps import _build_registry

    entries = (
        SemanticDependencyDefinition(
            dependency_id="carmel.example.thing",
            content_sha256="1" * 64,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=False,
        ),
    )
    with pytest.raises(Exception) as excinfo:
        _build_registry(entries)
    assert "no entry with" in str(excinfo.value)


def test_numeric_module_has_zero_carmel_imports() -> None:
    """Guards the within-module-only closure limitation documented on
    compute_dependency_sha: if numeric.py ever imports from carmel.* itself, the
    closure computed from its module-level names alone can no longer be assumed to
    capture its full behavior."""
    tree = ast.parse(_real_numeric_source())
    carmel_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            carmel_imports.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "carmel")
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.module.split(".")[0] == "carmel":
            carmel_imports.append(node.module)
    assert not carmel_imports, (
        f"carmel/services/numeric.py now imports from carmel.* ({carmel_imports!r}); "
        "compute_dependency_sha's transitive closure is within-module only and can no "
        "longer be assumed to capture this module's full behavior."
    )


def test_dataset_producer_names_table_v1_exactly_once_via_active_binding() -> None:
    """``carmel.services.dataset_producer`` must read ``units.TABLE_V1``
    through the single ``_ActiveTableBinding`` singleton ``_ACTIVE``, never
    independently at more than one site.

    Four call sites in that module derive artifacts from the active
    conversion table (the spelling vocabulary, unit normalization, the
    recorded ``conversion_table_sha256``, and the embedded table in a
    produced envelope). If any one of them named ``units.TABLE_V1`` directly
    instead of reading through ``_ACTIVE``, that site could silently
    disagree with the others the moment a future table swap changed
    ``_ACTIVE`` without also updating every direct reference -- exactly the
    "disagreement merely asserted, not unrepresentable" failure mode this
    project treats as a defect class. Pinning both "exactly one attribute
    access" and "no separate import of the name" at the AST level makes a
    reintroduced second reference to ``units.TABLE_V1`` fail this test
    immediately, rather than waiting for a scientific-correctness bug to
    surface it.
    """
    from carmel.services import dataset_producer

    source = inspect.getsource(dataset_producer)
    tree = ast.parse(source)

    table_v1_attribute_accesses: list[int] = []
    table_v1_direct_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "TABLE_V1":
            table_v1_attribute_accesses.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "carmel.services.units":
            for alias in node.names:
                if alias.name == "TABLE_V1":
                    table_v1_direct_imports.append(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "carmel.services.units.TABLE_V1":
                    table_v1_direct_imports.append(alias.asname or alias.name)

    assert not table_v1_direct_imports, (
        f"dataset_producer.py imports TABLE_V1 directly ({table_v1_direct_imports!r}); it must "
        "only be reachable via qualified `units.TABLE_V1` on the single `_ACTIVE = "
        "_ActiveTableBinding.derive(units.TABLE_V1)` line, never via a name that could be used "
        "at other sites without going through `_ACTIVE`"
    )
    assert len(table_v1_attribute_accesses) == 1, (
        f"expected exactly one `units.TABLE_V1` attribute access in dataset_producer.py (on the "
        f"`_ACTIVE = _ActiveTableBinding.derive(units.TABLE_V1)` line), found "
        f"{len(table_v1_attribute_accesses)} at lines {table_v1_attribute_accesses!r}; every other "
        "site must read through `_ACTIVE` instead of naming `units.TABLE_V1` a second time"
    )

    normalize_unit_calls_missing_table_kwarg: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "normalize_unit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "units"
            and not any(kw.arg == "table" for kw in node.keywords)
        ):
            normalize_unit_calls_missing_table_kwarg.append(node.lineno)

    assert normalize_unit_calls_missing_table_kwarg == [], (
        f"units.normalize_unit call(s) in dataset_producer.py without an explicit `table=` keyword "
        f"at lines {normalize_unit_calls_missing_table_kwarg!r}; every call must name its table "
        "explicitly rather than relying on any default"
    )

    for lineno in _tables_by_sha_registry_violations(tree):
        pytest.fail(
            f"dataset_producer.py consults `units.TABLES_BY_SHA` at line {lineno}, outside the "
            "single permitted `_BINDINGS_BY_SHA = {...for sha, table in units.TABLES_BY_SHA.items()}` "
            "derivation; every other site (including a hardcoded-sha lookup, a `next(iter(...))` "
            "walk, or a bare subscript) must go through `binding_for_known_sha(sha256)` instead of "
            "consulting the registry directly, or it could silently disagree with that one binding"
        )

    for lineno in _bindings_by_sha_bypass_violations(tree):
        pytest.fail(
            f"dataset_producer.py references the private `_BINDINGS_BY_SHA` registry at line "
            f"{lineno}, outside its own definition and outside `binding_for_known_sha` (the ONE "
            "function permitted to consult it); every other site must call "
            "`binding_for_known_sha(sha256)` rather than reading the registry directly"
        )

    for lineno in _active_table_sha256_violations(tree):
        pytest.fail(
            f"dataset_producer.py reads `_ACTIVE.table.sha256` at line {lineno}; the RECORDED "
            "conversion-table sha and the embedded table must always come from `_ACTIVE.embedded` "
            "(sha256 and canonical_json derived together, see `_ActiveTableBinding.derive`'s own "
            "invariant check), never re-derived from `_ACTIVE.table.sha256` at a second site"
        )


def _tables_by_sha_registry_violations(tree: ast.Module) -> list[int]:
    """Every AST line where ``units.TABLES_BY_SHA`` is referenced, outside the
    single permitted ``_BINDINGS_BY_SHA = {... for ... in
    units.TABLES_BY_SHA.items()}`` derivation.

    Catches every route to the registry generically -- ``units.TABLES_BY_SHA[sha]``,
    ``next(iter(units.TABLES_BY_SHA.values()))``, a hardcoded-sha subscript, or any
    other attribute-then-whatever chain -- by flagging the ``TABLES_BY_SHA``
    attribute access itself rather than pattern-matching individual call shapes.
    """
    permitted_range: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BINDINGS_BY_SHA" for target in node.targets
        ):
            permitted_range = (node.lineno, node.end_lineno or node.lineno)
            break
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_BINDINGS_BY_SHA"
        ):
            permitted_range = (node.lineno, node.end_lineno or node.lineno)
            break
    assert permitted_range is not None, "could not locate the `_BINDINGS_BY_SHA = ...` assignment"

    violations: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "TABLES_BY_SHA"
            and isinstance(node.value, ast.Name)
            and node.value.id == "units"
            and not (permitted_range[0] <= node.lineno <= permitted_range[1])
        ):
            violations.append(node.lineno)
    return violations


def _bindings_by_sha_bypass_violations(tree: ast.Module) -> list[int]:
    """Every AST line where the private ``_BINDINGS_BY_SHA`` registry is
    referenced by name, outside its own definition and outside
    ``binding_for_known_sha`` -- the one function permitted to consult it.
    """
    permitted_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        is_bindings_assign = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BINDINGS_BY_SHA" for target in node.targets
        )
        is_bindings_ann_assign = (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_BINDINGS_BY_SHA"
        )
        is_binding_for_known_sha = isinstance(node, ast.FunctionDef) and node.name == "binding_for_known_sha"
        if is_bindings_assign or is_bindings_ann_assign or is_binding_for_known_sha:
            permitted_ranges.append((node.lineno, node.end_lineno or node.lineno))
    assert permitted_ranges, "could not locate `_BINDINGS_BY_SHA`'s definition or `binding_for_known_sha`"

    violations: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "_BINDINGS_BY_SHA"
            and not any(lo <= node.lineno <= hi for lo, hi in permitted_ranges)
        ):
            violations.append(node.lineno)
    return violations


def _active_table_sha256_violations(tree: ast.Module) -> list[int]:
    """Every AST line where ``_ACTIVE.table.sha256`` is read directly, instead
    of through ``_ACTIVE.embedded`` (the recorded-sha/embedded-table site).
    """
    violations: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "sha256"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "table"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "_ACTIVE"
        ):
            violations.append(node.lineno)
    return violations


def test_table_policy_guard_catches_a_registry_bypass_via_subscript() -> None:
    """Proves :func:`_tables_by_sha_registry_violations` actually fires,
    rather than merely never having anything to catch: a synthetic module
    that consults ``units.TABLES_BY_SHA`` a second time via a bare subscript
    (bypassing ``binding_for_known_sha``) must be flagged.
    """
    source = (
        "_BINDINGS_BY_SHA = {sha: derive(t) for sha, t in units.TABLES_BY_SHA.items()}\n"
        "\n"
        "def rogue_lookup(sha256):\n"
        "    return units.TABLES_BY_SHA[sha256]\n"
    )
    violations = _tables_by_sha_registry_violations(ast.parse(source))
    assert violations == [4], f"expected the rogue subscript on line 4 to be flagged, got {violations!r}"


def test_table_policy_guard_catches_a_registry_bypass_via_next_iter_values() -> None:
    """Proves the same guard also catches the ``next(iter(...values()))``
    shape, not merely a bare subscript."""
    source = (
        "_BINDINGS_BY_SHA = {sha: derive(t) for sha, t in units.TABLES_BY_SHA.items()}\n"
        "\n"
        "def rogue_lookup():\n"
        "    return next(iter(units.TABLES_BY_SHA.values()))\n"
    )
    violations = _tables_by_sha_registry_violations(ast.parse(source))
    assert violations == [4], f"expected the rogue next(iter(...)) on line 4 to be flagged, got {violations!r}"


def test_table_policy_guard_permits_the_one_legitimate_registry_site() -> None:
    """The `_BINDINGS_BY_SHA` derivation itself, the one legitimate consumer
    of the registry, must never be flagged by its own guard."""
    source = "_BINDINGS_BY_SHA = {sha: derive(t) for sha, t in units.TABLES_BY_SHA.items()}\n"
    assert _tables_by_sha_registry_violations(ast.parse(source)) == []


def test_table_policy_guard_permits_the_annotated_registry_site() -> None:
    """The real `dataset_producer.py` module declares `_BINDINGS_BY_SHA` with
    an explicit type annotation (`ast.AnnAssign`, not a bare `ast.Assign`) --
    the guard must recognise that shape too, not just the unannotated one."""
    source = (
        "_BINDINGS_BY_SHA: Mapping[str, X] = MappingProxyType(\n"
        "    {sha: derive(t) for sha, t in units.TABLES_BY_SHA.items()}\n"
        ")\n"
    )
    assert _tables_by_sha_registry_violations(ast.parse(source)) == []


def test_table_policy_guard_catches_a_bindings_by_sha_bypass() -> None:
    """Proves :func:`_bindings_by_sha_bypass_violations` fires when a second
    function reads `_BINDINGS_BY_SHA` directly instead of calling
    `binding_for_known_sha`."""
    source = (
        "_BINDINGS_BY_SHA = {sha: derive(t) for sha, t in units.TABLES_BY_SHA.items()}\n"
        "\n"
        "def binding_for_known_sha(sha256):\n"
        "    return _BINDINGS_BY_SHA.get(sha256)\n"
        "\n"
        "def rogue_hardcoded_lookup():\n"
        "    return _BINDINGS_BY_SHA['deadbeef' * 8]\n"
    )
    violations = _bindings_by_sha_bypass_violations(ast.parse(source))
    assert violations == [7], f"expected the rogue hardcoded lookup on line 7 to be flagged, got {violations!r}"


def test_table_policy_guard_catches_active_table_sha256_instead_of_embedded() -> None:
    """Proves :func:`_active_table_sha256_violations` fires when a site reads
    `_ACTIVE.table.sha256` instead of `_ACTIVE.embedded.sha256`."""
    source = "conversion_table_sha256 = _ACTIVE.table.sha256\n"
    violations = _active_table_sha256_violations(ast.parse(source))
    assert violations == [1], f"expected the `_ACTIVE.table.sha256` read on line 1 to be flagged, got {violations!r}"


def test_table_policy_guard_permits_active_embedded_sha256() -> None:
    """`_ACTIVE.embedded.sha256` -- the correct recorded-sha site -- must
    never be flagged."""
    source = "conversion_table_sha256 = _ACTIVE.embedded.sha256\n"
    assert _active_table_sha256_violations(ast.parse(source)) == []


def test_current_sha_by_dependency_id_agrees_with_dependencies_by_sha() -> None:
    for dependency_id, sha in CURRENT_SHA_BY_DEPENDENCY_ID.items():
        assert sha in DEPENDENCIES_BY_SHA
        assert DEPENDENCIES_BY_SHA[sha].dependency_id == dependency_id


def test_changing_a_module_level_import_binding_changes_the_sha() -> None:
    """A silent-corruption hole this module must not have: swapping which
    module a name is imported from (e.g. `import re` -> `import regex as re`)
    changes no closure-reachable *name*, only what that name resolves to at
    runtime -- so the sha must still change, or such a swap would be
    invisible to a consumer trusting the sha as an identity."""
    source = (
        "import re\n"
        "\n"
        "def normalize_numeric_span(text):\n"
        "    return re.sub('a', 'b', text)\n"
    )
    perturbed = source.replace("import re\n", "import regex as re\n", 1)
    assert perturbed != source, "the string replacement did not match anything in the synthetic source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha, (
        "changing an import binding (import re -> import regex as re) must change the "
        "sha even though no closure-reachable name changed"
    )


def test_changing_an_unrelated_import_still_changes_the_sha() -> None:
    """Every module-level Import/ImportFrom is hashed unconditionally, not
    only ones an entry point's closure happens to reference -- adding a
    wholly unused import must still move the sha."""
    source = (
        "def normalize_numeric_span(text):\n"
        "    return text\n"
    )
    perturbed = "import os\n" + source
    assert perturbed != source
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_changing_an_if_bound_module_level_name_changes_the_sha() -> None:
    """A module-level name bound inside an `if` block (not a plain top-level
    Assign) must still be visible to the closure -- and to the sha -- rather
    than silently invisible."""
    source = (
        "if True:\n"
        "    _THRESHOLD = 6\n"
        "\n"
        "def normalize_numeric_span(text):\n"
        "    return _THRESHOLD\n"
    )
    perturbed = source.replace("_THRESHOLD = 6", "_THRESHOLD = 9", 1)
    assert perturbed != source
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha, (
        "a module-level name bound inside an 'if' block must be reachable by the "
        "closure, so changing its value must change the sha"
    )


def test_changing_a_for_bound_module_level_name_changes_the_sha() -> None:
    """Same guarantee as the 'if' case above, for a module-level `for` loop's
    target binding."""
    source = (
        "for _CODE in (6,):\n"
        "    pass\n"
        "\n"
        "def normalize_numeric_span(text):\n"
        "    return _CODE\n"
    )
    perturbed = source.replace("for _CODE in (6,):", "for _CODE in (9,):", 1)
    assert perturbed != source
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_module_level_match_statement_raises_invariant_error() -> None:
    """`ast.Match`'s capture-pattern binding shapes are too open-ended to
    resolve confidently -- compute_dependency_sha must fail loudly rather
    than silently miss a module-level binding hidden inside a `match` case."""
    source = (
        "match 1:\n"
        "    case _THRESHOLD:\n"
        "        pass\n"
        "\n"
        "def normalize_numeric_span(text):\n"
        "    return text\n"
    )
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        compute_dependency_sha(source, ["normalize_numeric_span"])
    assert "match" in str(excinfo.value)


# --- carmel.agents.tools.extract's extraction-code identity ------------------
#
# `_extractor_identity` in carmel.services.evidence records only the third-party
# `pypdf` VERSION (e.g. "pdf:pypdf==6.14.2"). It records nothing about Carmel's own
# extraction code (carmel/agents/tools/extract.py) -- two extractions produced by
# materially different Carmel code, under the SAME pypdf version, are indistinguishable
# in the store. The tests below pin a content_sha256 for extract.py's `extract_text`
# closure on exactly the same append-only, hardcoded-pin terms as the numeric
# dependencies above.

# HARDCODED, not derived from the live registry -- see the identical rationale on
# _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256 above. Independently verified once via:
#   compute_dependency_sha(inspect.getsource(extract), ["extract_text"])
# If this test ever fails, it means EITHER extract.py's extraction surface changed OR
# this toolchain's ast.dump() rendering changed. Fix: ADD A NEW registry entry for the
# new sha in carmel/services/semantic_deps.py -- never edit this literal in place.
_PINNED_EXTRACT_TEXT_SHA256 = "aa008f66d255cfb079cf269438ef9cfb0f1c42c6326d51a75e3e6fed04ec7168"

# The exact within-module closure of {"extract_text"} in the real extract.py module, as
# of this test's writing -- pinned separately from the sha so a closure-membership
# regression and an unparse/dump-format regression can be told apart by which of these
# two tests fails. Deliberately EXCLUDES `raw_span` and the bibliography-region helpers
# (`find_bibliography_like_regions`, `_is_citation_line`, `_BIB_*`): those are consumed
# by carmel.services.grounding, not by extract_text, so their absence here is correct,
# not a coverage gap.
_EXPECTED_EXTRACT_TEXT_CLOSURE = frozenset(
    {
        "ExtractedText",
        "MAX_EXTRACTED_TEXT_CHARS",
        "MAX_PDF_PAGES",
        "PageExtractionFailure",
        "TextSection",
        "_ABSTRACT_HEADING_RE",
        "_ABSTRACT_MAX_LEN",
        "_ABSTRACT_SEARCH_WINDOW",
        "_HTMLTextExtractor",
        "_HYPHEN_LINEBREAK_RE",
        "_LIGATURES",
        "_PATH_LIKE_RE",
        "_PageKind",
        "_REFERENCES_HEADING_RE",
        "_REFERENCES_MIN_FRACTION",
        "_WHITESPACE_RUN_RE",
        "_cap_text",
        "_char_map_transform",
        "_classify_pdf_page",
        "_decode_bytes",
        "_describe_page_error",
        "_extract_html",
        "_extract_pdf",
        "_extract_plain_text",
        "_extract_xml",
        "_find_abstract_region",
        "_find_references_region",
        "_hyphen_repl",
        "_label_special_sections",
        "_normalize_with_map_cached",
        "_overlay_region",
        "_pypdf_mute_depth",
        "_pypdf_mute_lock",
        "_pypdf_mute_previous",
        "_quiet_pypdf",
        "_regex_sub_with_map",
        "_whitespace_repl",
        "extract_text",
        "normalize_for_match",
        "normalize_with_map",
    }
)


def _real_extract_source() -> str:
    return inspect.getsource(extract)


def test_extract_text_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(_real_extract_source(), ["extract_text"])
    assert computed == _PINNED_EXTRACT_TEXT_SHA256, (
        "carmel/agents/tools/extract.py's extraction surface (or this toolchain's "
        "ast.dump rendering) has changed since carmel.extraction.extract_text was "
        "pinned. Fix: ADD A NEW entry to DEPENDENCIES_BY_SHA in "
        "carmel/services/semantic_deps.py for the new sha -- do not edit or replace "
        "the existing pinned entry or this test's hardcoded literal."
    )


def test_extract_text_registry_seed_agrees_with_the_pin() -> None:
    assert _PINNED_EXTRACT_TEXT_SHA256 in DEPENDENCIES_BY_SHA
    entry = DEPENDENCIES_BY_SHA[_PINNED_EXTRACT_TEXT_SHA256]
    assert entry.dependency_id == EXTRACT_TEXT_DEPENDENCY_ID
    assert entry.input_policy is InputPolicy.EXTERNAL_DIGEST_REQUIRED


def test_current_sha_for_extract_text_matches_the_pin() -> None:
    assert current_sha_for(EXTRACT_TEXT_DEPENDENCY_ID) == _PINNED_EXTRACT_TEXT_SHA256


def test_extract_text_closure_is_exactly_the_expected_names() -> None:
    source = _real_extract_source()
    tree = ast.parse(source)
    definitions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node

    closure: set[str] = set()
    frontier = ["extract_text"]
    while frontier:
        name = frontier.pop()
        if name in closure or name not in definitions:
            continue
        closure.add(name)
        for child in ast.walk(definitions[name]):
            if isinstance(child, ast.Name) and child.id in definitions and child.id not in closure:
                frontier.append(child.id)

    assert closure == _EXPECTED_EXTRACT_TEXT_CLOSURE
    assert "raw_span" not in closure
    assert "find_bibliography_like_regions" not in closure
    assert "_is_citation_line" not in closure
    assert not any(name.startswith("_BIB_") for name in closure)


def test_extract_module_has_zero_carmel_imports() -> None:
    """Guards the within-module-only closure limitation for extract.py, exactly as
    test_numeric_module_has_zero_carmel_imports guards it for numeric.py: if
    extract.py ever imports from carmel.* itself, compute_dependency_sha's closure
    can no longer be assumed to capture its full behavior."""
    _assert_no_carmel_imports(_real_extract_source(), "carmel.agents.tools.extract")


def test_assert_no_carmel_imports_raises_on_plain_import() -> None:
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_no_carmel_imports("import carmel.services.numeric\n", "carmel.example.mod")
    assert "carmel.example.mod" in str(excinfo.value)


def test_assert_no_carmel_imports_raises_on_import_from() -> None:
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_no_carmel_imports("from carmel.services import numeric\n", "carmel.example.mod")
    assert "carmel.example.mod" in str(excinfo.value)


def test_assert_no_carmel_imports_passes_on_clean_source() -> None:
    _assert_no_carmel_imports("import re\n\ndef f():\n    return re.sub('a', 'b', '')\n", "carmel.example.mod")


# --- Composite extraction identity (code sha + pypdf version) ----------------


def test_extraction_identity_returns_the_current_code_sha() -> None:
    identity = extraction_identity()
    assert identity.code_sha256 == current_sha_for(EXTRACT_TEXT_DEPENDENCY_ID)


def test_extraction_identity_returns_the_installed_pypdf_version() -> None:
    pypdf = pytest.importorskip("pypdf")
    identity = extraction_identity()
    assert identity.pypdf_version == pypdf.__version__


def test_extraction_identity_is_a_frozen_dataclass_instance() -> None:
    identity = extraction_identity()
    assert isinstance(identity, ExtractionIdentity)
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.code_sha256 = "0" * 64  # type: ignore[misc]


def test_pypdf_version_falls_back_to_unknown_when_pypdf_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    assert _pypdf_version() == "unknown"


def test_extraction_identity_falls_back_to_unknown_pypdf_version_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    identity = extraction_identity()
    assert identity.pypdf_version == "unknown"
    assert identity.code_sha256 == current_sha_for(EXTRACT_TEXT_DEPENDENCY_ID)


# --- Proof that the sha tracks behavior, using fabricated extraction-surface-shaped
# --- SYNTHETIC source (never real extract.py source or real corpus text) -----------

_SYNTHETIC_EXTRACT_SURFACE_TEMPLATE = '''
"""A fabricated module shaped like an extraction surface, for sha-sensitivity tests only."""

MAX_PAGES = {max_pages}


def _classify_page(raw: str) -> str:
    """Classify one page's raw text as a page kind."""
    if not raw.strip():
        return "blank"
    return "body"


def extract_text(pages: list[str]) -> list[str]:
    """Extract normalized text from a fabricated list of page strings."""
    kept = []
    for page in pages[:MAX_PAGES]:
        kind = _classify_page(page)
        if kind != "blank":
            kept.append(page)
    return kept
'''


def test_synthetic_extraction_surface_sha_changes_on_a_meaningful_edit() -> None:
    """A behavior-changing edit to an extraction-surface-shaped SYNTHETIC source (a
    different MAX_PAGES cap, reachable from extract_text's own closure) must move the
    sha. Uses a fabricated snippet, not real extract.py, per this suite's discipline of
    never editing the real module to prove sha-sensitivity."""
    original = _SYNTHETIC_EXTRACT_SURFACE_TEMPLATE.format(max_pages=5)
    changed = _SYNTHETIC_EXTRACT_SURFACE_TEMPLATE.format(max_pages=6)
    assert original != changed, "the two synthetic sources must differ textually"
    sha_original = compute_dependency_sha(original, ["extract_text"])
    sha_changed = compute_dependency_sha(changed, ["extract_text"])
    assert sha_original != sha_changed, "a change to a value the closure depends on must move the sha"


def test_synthetic_extraction_surface_sha_is_stable_under_a_docstring_only_edit() -> None:
    """A docstring-only edit to the same fabricated extraction-surface-shaped SYNTHETIC
    source must NOT move the sha -- compute_dependency_sha strips docstrings before
    hashing, so text-only documentation changes are invisible to the identity."""
    original = _SYNTHETIC_EXTRACT_SURFACE_TEMPLATE.format(max_pages=5)
    redocumented = original.replace(
        "Extract normalized text from a fabricated list of page strings.",
        "Extract normalized text from a fabricated list of page strings -- rewritten wording.",
    )
    assert original != redocumented, "the two synthetic sources must differ textually"
    sha_original = compute_dependency_sha(original, ["extract_text"])
    sha_redocumented = compute_dependency_sha(redocumented, ["extract_text"])
    assert sha_original == sha_redocumented, "a docstring-only difference must not change the sha"
