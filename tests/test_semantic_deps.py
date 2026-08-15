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
from carmel.services import numeric, pdf_fragments
from carmel.services.semantic_deps import (
    _FRAGMENT_GEOMETRY_COMPONENTS_BY_SHA,
    _PYPDF_VERSION_UNKNOWN,
    CURRENT_SHA_BY_DEPENDENCY_ID,
    DEPENDENCIES_BY_SHA,
    EXTRACT_TEXT_DEPENDENCY_ID,
    FRAGMENT_GEOMETRY_BORROWED_NAMES,
    FRAGMENT_GEOMETRY_DEPENDENCY_ID,
    ExtractionIdentity,
    FragmentGeometryIdentity,
    InputPolicy,
    SemanticDependencyDefinition,
    SemanticDependencyInvariantError,
    UnknownSemanticDependencyError,
    _assert_borrowed_carmel_names,
    _assert_no_carmel_imports,
    _fragment_geometry_components,
    _module_level_definitions,
    _pypdf_distribution_version,
    _pypdf_version,
    _transitive_closure,
    compose_component_sha,
    compute_dependency_sha,
    current_sha_for,
    dependency_for_sha,
    extraction_identity,
    fragment_geometry_identity,
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
_PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256 = "b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb"

# Pinned for carmel.numeric.glyph_health on exactly the same terms as the literal
# above: hardcoded, never recomputed from the live module, never edited in place.
# Entry point: assess_glyph_health.
_PINNED_GLYPH_HEALTH_SHA256 = "af3553a8142b50bba56b6ba164778b4cd2bff6e4916ac2e93c4e1a270ba4ab5a"

# The set of sha256 digests this registry has ever shipped, as of this test's writing.
# Used only to assert DEPENDENCIES_BY_SHA never drops a previously-shipped entry (see
# test_registry_is_append_only_and_never_drops_a_previously_shipped_sha below).
#
# MUST list EVERY shipped sha, not just the first two. It carried only the repair and
# glyph-health digests for as long as there were four rows, which quietly left the
# extract-text and fragment-geometry rows droppable without this tripwire firing -- an
# append-only guard that watches half the registry is not an append-only guard. Every new
# registry row gets its digest added here in the same commit; the literals are duplicated
# from the pins above rather than pulled from the live registry, because a tripwire
# computed from the thing it watches never fires on drift.
_HISTORICALLY_SHIPPED_SHAS = frozenset(
    {
        _PINNED_CONTEXT_FREE_SPAN_REPAIR_SHA256,
        _PINNED_GLYPH_HEALTH_SHA256,
        "aa008f66d255cfb079cf269438ef9cfb0f1c42c6326d51a75e3e6fed04ec7168",
        "4922bd55d53e90e9bcd7cb4823e15798cb89ffddb6b2b6d7745f96c9ff1767bb",
        "3fc972d0394184267e85a9a9e42387423eed538758efeba3ce1fd125ef56c47b",
        "652cdea53a2c44a9861b6896b6cb8234d86b0ac6745c3ddc135e728522e5b25e",
        "4ae9d68f0bcbf55bfbcaef1f7c7a2dda02b64ef4bc6bdf7cc504672d59810545",
        "75310c6df1677158c15e233e2da4abe72c52fcc565dbaa0f0f7ca36cb8b50f3a",
        "6789f2a10e8f58f56f5e5969187525f1ca33f740d8015254da2133ca55363108",
        "2fc6f8df66a12d1be2c473ab17e91170cc0c1866b5098bd69dee9e830abd940e",
        "ccd95b43ed5f048a77428ec6a8f199a34f6158a4a1b66f2d1ef746a1916a2491",
    }
)


def _real_numeric_source() -> str:
    return inspect.getsource(numeric)


def test_seeded_dependency_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(_real_numeric_source(), ["normalize_numeric_span", "REPAIR_NAMES"])
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
        'def _find_range_separator(text: str) -> int | None:\n    """Return the index of the hyphen/en-dash',
        "def _find_range_separator(text: str) -> int | None:\n"
        '    """COMPLETELY DIFFERENT DOCSTRING TEXT. Return the index of the hyphen/en-dash',
        1,
    )
    assert perturbed != source, "the string replacement did not match anything in the real source"
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha == original_sha


def test_changing_a_field_doc_DOES_change_the_sha() -> None:
    """The boundary of "docstring-insensitive", pinned as behaviour because the phrase
    reads wider than it is and the gap has already cost a real edit.

    A docstring is the FIRST statement of a body. The PEP 258 attribute doc -- a bare
    string after `field: int` in a class body -- is an ordinary `Expr` and is hashed as
    CODE. So documenting a field of a hashed dataclass needs a supersession while
    documenting a function is free, which is the opposite of what a reader assumes.

    Asserted in the POSITIVE direction on purpose. A test that only checked function
    docstrings are ignored would pass just as happily if field docs were ignored too,
    and the next reader "simplifying" `_strip_docstrings` to strip every string
    statement would break every stored geometry identity with a green suite.
    """
    source = "\n".join(
        (
            "import dataclasses",
            "@dataclasses.dataclass",
            "class Thing:",
            "    field: int",
            '    """{doc}"""',
        )
    )
    first = compute_dependency_sha(source.format(doc="what it was"), ["Thing"])
    second = compute_dependency_sha(source.format(doc="what it became"), ["Thing"])
    assert first != second

    # And the contrast, in one test so the two can never drift apart: the class's own
    # docstring -- same file, same class, one statement earlier -- is free.
    with_class_doc = "\n".join(
        (
            "import dataclasses",
            "@dataclasses.dataclass",
            "class Thing:",
            '    """{doc}"""',
            "    field: int",
        )
    )
    assert compute_dependency_sha(with_class_doc.format(doc="one"), ["Thing"]) == compute_dependency_sha(
        with_class_doc.format(doc="two"), ["Thing"]
    )


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


def test_every_registry_row_is_recorded_in_the_historical_set() -> None:
    """The other direction, and the one that was missing.

    `_HISTORICALLY_SHIPPED_SHAS` says in prose that every new registry row gets its digest
    added here in the same commit, and until this test nothing checked it: a row could be
    added and left unrecorded, and the append-only guard above would stay green forever
    because it only looks for digests that VANISHED. A guard whose coverage depends on
    someone remembering to extend it is exactly the guard that silently stops covering the
    newest thing -- which is always the thing least reviewed.

    This is not circular with the guard above even though the two assertions together pin
    the set exactly. `_HISTORICALLY_SHIPPED_SHAS` is hand-written from the pinned literals
    and never computed from the registry, so one direction still catches a dropped row and
    the other catches an unrecorded one.
    """
    unrecorded = set(DEPENDENCIES_BY_SHA) - _HISTORICALLY_SHIPPED_SHAS
    assert not unrecorded, (
        f"registry rows {sorted(unrecorded)!r} are not listed in _HISTORICALLY_SHIPPED_SHAS. "
        "Add the literal there in this same commit -- copied from the pin, never read back "
        "from DEPENDENCIES_BY_SHA, or the tripwire is computed from the thing it watches."
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
    source = "import re\n\ndef normalize_numeric_span(text):\n    return re.sub('a', 'b', text)\n"
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
    source = "def normalize_numeric_span(text):\n    return text\n"
    perturbed = "import os\n" + source
    assert perturbed != source
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_changing_an_if_bound_module_level_name_changes_the_sha() -> None:
    """A module-level name bound inside an `if` block (not a plain top-level
    Assign) must still be visible to the closure -- and to the sha -- rather
    than silently invisible."""
    source = "if True:\n    _THRESHOLD = 6\n\ndef normalize_numeric_span(text):\n    return _THRESHOLD\n"
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
    source = "for _CODE in (6,):\n    pass\n\ndef normalize_numeric_span(text):\n    return _CODE\n"
    perturbed = source.replace("for _CODE in (6,):", "for _CODE in (9,):", 1)
    assert perturbed != source
    original_sha = compute_dependency_sha(source, ["normalize_numeric_span"])
    perturbed_sha = compute_dependency_sha(perturbed, ["normalize_numeric_span"])
    assert perturbed_sha != original_sha


def test_module_level_match_statement_raises_invariant_error() -> None:
    """`ast.Match`'s capture-pattern binding shapes are too open-ended to
    resolve confidently -- compute_dependency_sha must fail loudly rather
    than silently miss a module-level binding hidden inside a `match` case."""
    source = "match 1:\n    case _THRESHOLD:\n        pass\n\ndef normalize_numeric_span(text):\n    return text\n"
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


# --------------------------------------------------------------------------------------
# Fragment-geometry identity (carmel.geometry.extract_fragments)
#
# The only COMPOSITE registry row. Its two components are pinned SEPARATELY below, so a
# failure names which half moved rather than only reporting that something did.
# --------------------------------------------------------------------------------------

# HARDCODED, not derived from the live registry -- same rationale as every other pin in
# this file. Independently verified once via:
#   compute_dependency_sha(inspect.getsource(pdf_fragments), ["extract_fragments"])
# This is the half that moves when fragment GEOMETRY changes.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256 = "7444bb6fbf152fbb7aea42f58d2627966163ddd908adba336723202f4e40cd53"

# The SUPERSEDED first entry, pinned so the append-only contract is checked against a real
# historical row rather than only asserted in prose. Its own component moved when
# extract_fragments stopped re-reading importlib.metadata; its BORROWED component did not.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V1 = "c93639f8dab3d79c37a1dd3d5ca4d66d9397d1c174d230fe2e10e677568ae8e3"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V1 = "4922bd55d53e90e9bcd7cb4823e15798cb89ffddb6b2b6d7745f96c9ff1767bb"

# The SUPERSEDED second entry. Its own component moved when `_decoded_content_length`
# stopped measuring a page by decoding it in full; its BORROWED component did not, which
# is now twice in a row that the split has localized the change to pdf_fragments.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V2 = "96b5852b71496f062dd1d36b255f98feb952baf89fe7e4fb995b93ce00a56f5e"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V2 = "3fc972d0394184267e85a9a9e42387423eed538758efeba3ce1fd125ef56c47b"

# HARDCODED. Independently verified once via:
#   compute_dependency_sha(inspect.getsource(extract), list(FRAGMENT_GEOMETRY_BORROWED_NAMES))
# The half that moves when a BORROWED name's behaviour changes in extract.py -- the code
# extract_fragments runs but compute_dependency_sha cannot see, because the import is
# function-scope.
_PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256 = "39844d90f40067b45a6413816336fd9cbb7a1f9db8be05c75640b74d56ea8199"

# The SUPERSEDED third entry. Its own component moved when the bounded decode stopped
# accepting a TRUNCATED deflate stream as a measured one -- `unconsumed_tail` is empty for a
# valid prefix, so only `eof` can tell "consumed all input" from "reached the end". Borrowed
# did not move: three for three.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V3 = "e745a377714ea6a817a82977d028d6c2820091f85be7f87c4972bd70f9e41e44"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V3 = "652cdea53a2c44a9861b6896b6cb8234d86b0ac6745c3ddc135e728522e5b25e"

# The SUPERSEDED fourth entry. Its own component moved when `_page_fragments` stopped
# trusting pypdf about where a text-show operation STARTS -- the previous three moves
# changed which pages fail, this one changes the published coordinates. Borrowed did not
# move: four for four.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V4 = "d446c737b7d37cf50adaf4070250bc75f721f958b62680981bc89f8a5b474967"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V4 = "4ae9d68f0bcbf55bfbcaef1f7c7a2dda02b64ef4bc6bdf7cc504672d59810545"

# The SUPERSEDED fifth entry. Its own component moved when `_walk_operations` stopped
# STEPPING OVER the operators it does not name. Alone among the five, this one moved no
# coordinate at all -- 78,178 corpus fragments and 0 page failures on both sides of it --
# so it records a change in what the extractor REFUSES, which needs a new identity for
# exactly the same reason a computed change does. Borrowed did not move: five for five.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V5 = "dbd9b8a255b6339438cb4551fcb6c8d0aa3e434694f5ccf71abf921a076d9cbd"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V5 = "75310c6df1677158c15e233e2da4abe72c52fcc565dbaa0f0f7ca36cb8b50f3a"

# SUPERSEDED -- the SIXTH, which modelled no clipping path at all. It treated `W`/`W*` and
# the path-painting operators as irrelevant on the true premise that they do not move text
# and the false conclusion that they cannot hide it, and published a fragment for text drawn
# wholly outside the clip in force.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V6 = "4dd477809aaf0874e427e72cf8ff5e12391455f525bd56ea385999c57eff1101"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V6 = "6789f2a10e8f58f56f5e5969187525f1ca33f740d8015254da2133ca55363108"

# SUPERSEDED -- the SEVENTH, which returned one `available=False` from three sites meaning
# three different things: pypdf absent, the capability gate refusing, and a document that
# defeated a healthy pinned engine. An artifact carrying that flag could not say whether
# anything had been looked at, and a mid-walk engine mismatch was indistinguishable from a
# bad document.
_PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V7 = "3f8d26ba2e3a343cd777ff865b34cbec6f13cb4dd4dcf12a5d9923d4b9b6f58c"
_PINNED_FRAGMENT_GEOMETRY_SHA256_V7 = "2fc6f8df66a12d1be2c473ab17e91170cc0c1866b5098bd69dee9e830abd940e"

# HARDCODED. The registered content_sha256 itself, verified once via:
#   compose_component_sha({"borrowed_sha256": <borrowed>, "own_sha256": <own>})
_PINNED_FRAGMENT_GEOMETRY_SHA256 = "ccd95b43ed5f048a77428ec6a8f199a34f6158a4a1b66f2d1ef746a1916a2491"

# The carmel.* import surface of pdf_fragments.py, as of this test's writing. This is the
# completeness claim of the composite identity, spelled out as data: extract_fragments runs
# these five names' code, and `borrowed_sha256` hashes exactly them.
_EXPECTED_FRAGMENT_GEOMETRY_IMPORTS = {"carmel.agents.tools.extract": set(FRAGMENT_GEOMETRY_BORROWED_NAMES)}


def _real_fragments_source() -> str:
    return inspect.getsource(pdf_fragments)


def test_fragment_geometry_own_component_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(_real_fragments_source(), ["extract_fragments"])
    assert computed == _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256, (
        "carmel/services/pdf_fragments.py's fragment geometry (or this toolchain's "
        "ast.dump rendering) has changed since carmel.geometry.extract_fragments was "
        "pinned. Fix: ADD A NEW entry to DEPENDENCIES_BY_SHA in "
        "carmel/services/semantic_deps.py for the new composite sha -- do not edit or "
        "replace the existing pinned entry or this test's hardcoded literal."
    )


def test_fragment_geometry_borrowed_component_sha_matches_a_hardcoded_pin() -> None:
    computed = compute_dependency_sha(inspect.getsource(extract), list(FRAGMENT_GEOMETRY_BORROWED_NAMES))
    assert computed == _PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256, (
        "the behaviour of a name carmel/services/pdf_fragments.py BORROWS from "
        "carmel/agents/tools/extract.py has changed. extract_fragments runs this code, but "
        "compute_dependency_sha cannot see the import (it is function-scope), which is why "
        "it is hashed as its own component. Fix: ADD A NEW registry entry for the new "
        "composite sha -- never edit this literal in place."
    )


def test_fragment_geometry_registered_sha_is_the_composite_of_its_two_components() -> None:
    """The registered content_sha256 must be exactly the composite of the two pinned
    components -- never an independently maintained third literal that could drift away
    from the halves it claims to summarize."""
    composite = compose_component_sha(
        {
            "borrowed_sha256": _PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            "own_sha256": _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256,
        }
    )
    assert composite == _PINNED_FRAGMENT_GEOMETRY_SHA256
    assert current_sha_for(FRAGMENT_GEOMETRY_DEPENDENCY_ID) == _PINNED_FRAGMENT_GEOMETRY_SHA256


def test_fragment_geometry_borrowed_names_are_exactly_what_pdf_fragments_imports() -> None:
    """THE completeness guard for the composite identity.

    Precisely: adding a SIXTH borrowed name does move the composite, because the import
    statement sits in extract_fragments' hashed body. The hazard is what happens after
    that -- `borrowed_sha256` would go on attesting only the original five forever, so the
    sixth name's behaviour would be covered by no component at all, and a later edit INSIDE
    it would move no sha. This guard fails at the moment the name appears, which is the
    only moment anyone is looking.
    """
    _assert_borrowed_carmel_names(
        _real_fragments_source(),
        "carmel.services.pdf_fragments",
        _EXPECTED_FRAGMENT_GEOMETRY_IMPORTS,
    )


def test_borrowed_names_guard_fires_on_an_added_name() -> None:
    source = "def f():\n    from carmel.agents.tools.extract import MAX_PDF_PAGES, _classify_pdf_page\n"
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_borrowed_carmel_names(
            source,
            "carmel.example.mod",
            {"carmel.agents.tools.extract": {"MAX_PDF_PAGES"}},
        )
    assert "_classify_pdf_page" in str(excinfo.value)


def test_borrowed_names_guard_fires_on_a_new_source_module() -> None:
    source = "def f():\n    from carmel.services.numeric import AffixClass\n"
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_borrowed_carmel_names(source, "carmel.example.mod", {})
    assert "carmel.services.numeric" in str(excinfo.value)


def test_borrowed_names_guard_records_a_plain_package_import_as_unpinnable() -> None:
    """`import carmel.x.y` binds the whole module, whose surface no finite name tuple can
    pin, so it is recorded as "*" and can never satisfy a named expectation."""
    source = "def f():\n    import carmel.services.numeric\n"
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_borrowed_carmel_names(
            source,
            "carmel.example.mod",
            {"carmel.services.numeric": {"AffixClass"}},
        )
    assert "'*'" in str(excinfo.value)


def test_pdf_fragments_cannot_use_the_no_carmel_imports_guard() -> None:
    """The strong guard is the WRONG one for this module, and that is a property worth
    pinning rather than a fact to remember.

    _assert_no_carmel_imports states "this module's behaviour is fully captured by its own
    within-module closure". pdf_fragments.py cannot satisfy it, which is exactly why the
    composite identity and the weaker named-borrowings guard exist. If this test ever
    starts FAILING -- i.e. the strong guard begins to pass -- the borrowed component has
    become vacuous and the composite should be revisited, not left in place.
    """
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_no_carmel_imports(_real_fragments_source(), "carmel.services.pdf_fragments")
    assert "carmel.agents.tools.extract" in str(excinfo.value)


def test_fragment_geometry_identity_reports_all_three_parts() -> None:
    identity = fragment_geometry_identity()
    assert isinstance(identity, FragmentGeometryIdentity)
    assert identity.composite_sha256 == _PINNED_FRAGMENT_GEOMETRY_SHA256
    assert identity.own_sha256 == _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256
    assert identity.borrowed_sha256 == _PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256
    assert identity.pypdf_version == _pypdf_version()


def test_the_identity_records_both_pypdf_witnesses_and_chooses_neither() -> None:
    """`_engine()` gates on the DISTRIBUTION metadata; the identity used to report only
    the imported module's `__version__`. Two witnesses to one fact that can disagree
    under an editable, vendored or shadowed install -- and choosing one means a stored
    artifact can never afterwards say whether they did, which for an IDENTITY is the
    wrong shape. Both are recorded and nothing here compares them.
    """
    identity = fragment_geometry_identity()
    assert identity.pypdf_version == _pypdf_version()
    assert identity.pypdf_distribution_version == _pypdf_distribution_version()
    # They agree on the pin, so the divergence is latent rather than live. Asserted so a
    # future disagreement in a dev environment surfaces here rather than in an artifact.
    assert identity.pypdf_version == identity.pypdf_distribution_version


def test_neither_pypdf_witness_can_raise() -> None:
    """Both collapse to the sentinel, including `PackageNotFoundError`, which is the
    EXPECTED failure of the metadata one. Recording a second witness must not give the
    identity a second way to fail -- an identity that raises is one a caller cannot
    record at all."""
    import importlib.metadata as metadata_module

    def _absent(_name: str) -> str:
        raise metadata_module.PackageNotFoundError("pypdf")

    original = metadata_module.version
    metadata_module.version = _absent  # type: ignore[assignment]
    try:
        assert _pypdf_distribution_version() == _PYPDF_VERSION_UNKNOWN
        assert fragment_geometry_identity().pypdf_distribution_version == _PYPDF_VERSION_UNKNOWN
    finally:
        metadata_module.version = original  # type: ignore[assignment]


def test_fragment_geometry_identity_is_frozen() -> None:
    identity = fragment_geometry_identity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.own_sha256 = "x"  # type: ignore[misc]


def test_fragment_geometry_sha_differs_from_the_extract_text_sha() -> None:
    """The geometry lane and the text lane must never share a content address: they are
    different code producing different artifacts, and a shared sha would make a stored
    geometry claim indistinguishable from a stored text claim."""
    assert _PINNED_FRAGMENT_GEOMETRY_SHA256 != _PINNED_EXTRACT_TEXT_SHA256
    assert _PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256 != _PINNED_EXTRACT_TEXT_SHA256


def test_borrowed_component_is_not_the_extract_text_closure() -> None:
    """The borrowed component hashes EXACTLY the five borrowed names, never extract_text's
    40-name closure that happens to contain them today.

    Reusing the extract_text sha would over-attest (unrelated HTML/XML behaviour would
    re-address stored geometry) and could later under-attest (if extract_text stopped
    reaching one of the five while pdf_fragments still imported it, the name would leave
    the hashed closure with no sha change at all). Both are silent-corruption directions.
    """
    borrowed_closure_sha = compute_dependency_sha(inspect.getsource(extract), list(FRAGMENT_GEOMETRY_BORROWED_NAMES))
    extract_text_sha = compute_dependency_sha(inspect.getsource(extract), ["extract_text"])
    assert borrowed_closure_sha != extract_text_sha
    # The containment that makes the reuse tempting is real -- pin it, so that if it ever
    # stops holding, the reason this component is computed separately is on the record.
    tree = ast.parse(inspect.getsource(extract))
    definitions = _module_level_definitions(tree)
    extract_text_closure = _transitive_closure(["extract_text"], definitions)
    assert set(FRAGMENT_GEOMETRY_BORROWED_NAMES) <= set(extract_text_closure)


def test_compose_component_sha_is_order_independent_but_name_sensitive() -> None:
    a = "a" * 64
    b = "b" * 64
    assert compose_component_sha({"own_sha256": a, "borrowed_sha256": b}) == compose_component_sha(
        {"borrowed_sha256": b, "own_sha256": a}
    )
    # Swapping which component each value belongs to MUST change the composite: the two
    # halves are not interchangeable, and a positional join could not express that.
    assert compose_component_sha({"own_sha256": a, "borrowed_sha256": b}) != compose_component_sha(
        {"own_sha256": b, "borrowed_sha256": a}
    )


def test_compose_component_sha_refuses_an_empty_or_malformed_component() -> None:
    with pytest.raises(SemanticDependencyInvariantError):
        compose_component_sha({})
    with pytest.raises(SemanticDependencyInvariantError):
        compose_component_sha({"own_sha256": "not-a-sha"})
    with pytest.raises(SemanticDependencyInvariantError):
        compose_component_sha({"own_sha256": "A" * 64})  # uppercase is not canonical
    with pytest.raises(SemanticDependencyInvariantError):
        compose_component_sha({"own_sha256": "a" * 63})
    with pytest.raises(SemanticDependencyInvariantError):
        compose_component_sha({"own_sha256": "a" * 65})


# A pair of fabricated modules shaped like the real borrower/lender relationship: a
# geometry function that imports a helper at FUNCTION scope, and the module that defines
# that helper. Synthetic rather than the real sources, so the test states the structural
# property and cannot be broken by unrelated edits to pdf_fragments.py or extract.py.
_SYNTHETIC_BORROWER = '''
GUTTER = 18.0


def _pen_advance(width, spacing, glyphs):
    """Fabricated advance math standing in for the real geometry."""
    return width + spacing * glyphs


def extract_fragments(data):
    """Fabricated fragment extraction that borrows a helper at function scope."""
    from carmel.agents.tools.extract import _classify_page

    return [(_classify_page(page), _pen_advance(1.0, {spacing}, 2)) for page in data]
'''

_SYNTHETIC_LENDER = '''
def _classify_page(page):
    """Fabricated page classifier -- the borrowed behaviour."""
    return "real" if page.get("Type") == "Page" else {verdict}
'''


def test_the_two_components_are_complementary_not_redundant() -> None:
    """The property that justifies the composite existing at all.

    An edit to the BORROWED code must move the borrowed component and leave the own
    component untouched -- that invisibility is exactly the gap a single within-module sha
    leaves, and exactly what the second component closes. Verified in both directions so
    that neither component can be dropped later on the theory that the other covers it.
    """
    borrower = _SYNTHETIC_BORROWER.format(spacing="0.5")
    lender_before = _SYNTHETIC_LENDER.format(verdict='"phantom"')
    lender_after = _SYNTHETIC_LENDER.format(verdict='"linearization-dict"')
    assert lender_before != lender_after, "the two synthetic lender sources must differ textually"

    own_before = compute_dependency_sha(borrower, ["extract_fragments"])
    borrowed_before = compute_dependency_sha(lender_before, ["_classify_page"])
    borrowed_after = compute_dependency_sha(lender_after, ["_classify_page"])

    # The borrowed edit is invisible to the borrower's own closure ...
    assert compute_dependency_sha(borrower, ["extract_fragments"]) == own_before
    # ... and visible to the borrowed component, which is the whole point.
    assert borrowed_before != borrowed_after
    assert compose_component_sha(
        {"borrowed_sha256": borrowed_before, "own_sha256": own_before}
    ) != compose_component_sha({"borrowed_sha256": borrowed_after, "own_sha256": own_before})


def test_own_component_moves_on_a_geometry_edit() -> None:
    """The converse direction: a change to the borrower's own geometry math must move the
    own component (and therefore the composite), with the borrowed half held fixed."""
    before = compute_dependency_sha(_SYNTHETIC_BORROWER.format(spacing="0.5"), ["extract_fragments"])
    after = compute_dependency_sha(_SYNTHETIC_BORROWER.format(spacing="0.75"), ["extract_fragments"])
    assert before != after


def test_own_component_moves_when_a_borrowed_name_is_added() -> None:
    """A function-scope import IS inside the hashed body, so adding a borrowed name moves
    the own component.

    Pinned deliberately, because the tempting shorthand -- "compute_dependency_sha cannot
    see a function-scope import" -- is FALSE and would misdescribe why
    _assert_borrowed_carmel_names is needed. It is needed because `borrowed_sha256` would
    keep attesting the old name set afterwards, not because the addition itself is
    invisible.
    """
    one = _SYNTHETIC_BORROWER.format(spacing="0.5")
    two = one.replace(
        "from carmel.agents.tools.extract import _classify_page",
        "from carmel.agents.tools.extract import _classify_page, _describe_error",
    )
    assert one != two, "the two synthetic borrower sources must differ textually"
    assert compute_dependency_sha(one, ["extract_fragments"]) != compute_dependency_sha(two, ["extract_fragments"])


def test_borrowed_names_guard_fires_on_a_relative_import() -> None:
    """A relative import names carmel code while matching no `carmel.*` prefix.

    `from ..agents.tools.extract import _helper` parses as
    `ImportFrom(module='agents.tools.extract', level=2)`. Before this was handled, the
    guard returned an empty mapping for it and PASSED -- so a sixth borrowed helper added
    relatively would have been attested by nothing while the guard reported the import
    surface unchanged. Not resolved to an absolute name (that would need the importing
    module's own package, which the guard is not given); recorded under a key that can
    never match a named expectation, so it always fails loudly.
    """
    source = "def f():\n    from ..agents.tools.extract import _new_helper\n"
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_borrowed_carmel_names(
            source,
            "carmel.services.pdf_fragments",
            _EXPECTED_FRAGMENT_GEOMETRY_IMPORTS,
        )
    assert "_new_helper" in str(excinfo.value)


def test_borrowed_names_guard_fires_on_a_star_import() -> None:
    source = "def f():\n    from carmel.agents.tools.extract import *\n"
    with pytest.raises(SemanticDependencyInvariantError) as excinfo:
        _assert_borrowed_carmel_names(
            source,
            "carmel.example.mod",
            {"carmel.agents.tools.extract": set(FRAGMENT_GEOMETRY_BORROWED_NAMES)},
        )
    assert "'*'" in str(excinfo.value)


def test_fragment_geometry_components_are_recorded_append_only_by_composite() -> None:
    """A composite is a one-way hash, so the halves it was built from must be RECORDED --
    keyed by the composite, never held as current-only constants that the first
    supersession would overwrite."""
    components = _fragment_geometry_components(_PINNED_FRAGMENT_GEOMETRY_SHA256)
    assert components.own_sha256 == _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256
    assert components.borrowed_sha256 == _PINNED_FRAGMENT_GEOMETRY_BORROWED_SHA256
    assert components.borrowed_names == FRAGMENT_GEOMETRY_BORROWED_NAMES
    # Recomposing the recorded halves must reproduce the key they are filed under.
    assert (
        compose_component_sha(
            {
                "borrowed_sha256": components.borrowed_sha256,
                "own_sha256": components.own_sha256,
            }
        )
        == _PINNED_FRAGMENT_GEOMETRY_SHA256
    )


def test_every_registered_composite_has_recorded_components() -> None:
    """The component record must never fall behind the registry: a shipped composite with
    no recorded halves cannot be localized, which is the one thing the composite shape
    exists to make possible."""
    for sha, definition in DEPENDENCIES_BY_SHA.items():
        if definition.dependency_id == FRAGMENT_GEOMETRY_DEPENDENCY_ID:
            assert _fragment_geometry_components(sha) is not None


def test_unknown_composite_cannot_be_localized() -> None:
    with pytest.raises(UnknownSemanticDependencyError):
        _fragment_geometry_components("f" * 64)


def test_fragment_geometry_component_record_is_read_only() -> None:
    with pytest.raises(TypeError):
        _FRAGMENT_GEOMETRY_COMPONENTS_BY_SHA[_PINNED_FRAGMENT_GEOMETRY_SHA256] = None  # type: ignore[index]


def test_the_superseded_fragment_geometry_row_is_still_resolvable() -> None:
    """The registry's first real supersession, checked as behaviour rather than prose.

    An artifact citing the superseded composite must still resolve to a row that names the
    right dependency and reports itself NOT current. Dropping the row would make such an
    artifact indistinguishable from a forged one -- the exact confusion this module was
    written to prevent.
    """
    for superseded_sha in (
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V1,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V2,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V3,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V4,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V5,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V6,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V7,
    ):
        superseded = dependency_for_sha(superseded_sha)
        assert superseded.dependency_id == FRAGMENT_GEOMETRY_DEPENDENCY_ID
        assert superseded.is_current is False
    current = dependency_for_sha(_PINNED_FRAGMENT_GEOMETRY_SHA256)
    assert current.dependency_id == FRAGMENT_GEOMETRY_DEPENDENCY_ID
    assert current.is_current is True
    assert current_sha_for(FRAGMENT_GEOMETRY_DEPENDENCY_ID) == _PINNED_FRAGMENT_GEOMETRY_SHA256


def test_every_superseded_geometry_row_is_distinct() -> None:
    """Two supersessions is where an append-only table starts being able to lie: a copied
    literal, or a `_V2` that never got its own value, would leave two rows claiming the
    same content address and the second would silently shadow the first."""
    shas = (
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V1,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V2,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V3,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V4,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V5,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V6,
        _PINNED_FRAGMENT_GEOMETRY_SHA256_V7,
        _PINNED_FRAGMENT_GEOMETRY_SHA256,
    )
    owns = (
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V1,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V2,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V3,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V4,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V5,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V6,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256_V7,
        _PINNED_FRAGMENT_GEOMETRY_OWN_SHA256,
    )
    assert len(set(shas)) == len(shas)
    assert len(set(owns)) == len(owns)


def test_the_supersession_moved_only_the_own_component() -> None:
    """What the composite split buys, demonstrated on the real supersessions rather than on
    synthetic sources: both changes were entirely inside pdf_fragments, so the OWN half
    moved each time and the BORROWED half never did. A single opaque sha could not say
    that, and saying it three times over is what turns one observation into the property the
    split was built to provide."""
    generations = [
        _fragment_geometry_components(sha)
        for sha in (
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V1,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V2,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V3,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V4,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V5,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V6,
            _PINNED_FRAGMENT_GEOMETRY_SHA256_V7,
            _PINNED_FRAGMENT_GEOMETRY_SHA256,
        )
    ]
    for old, new in zip(generations[:-1], generations[1:], strict=True):
        assert old.own_sha256 != new.own_sha256
        assert old.borrowed_sha256 == new.borrowed_sha256
        assert old.borrowed_names == new.borrowed_names


def test_extract_fragments_records_the_pinned_pypdf_version_without_a_second_metadata_read() -> None:
    """The recorded version must be the constant `_engine()` already validated against.

    A second `importlib.metadata.version` call was both redundant -- `_engine()` refuses
    unless that entry equals the pin, so nothing else reaches this point -- and a second
    way to fail, sitting outside the degradation `try` where it could raise
    PackageNotFoundError out of a function documented never to raise for a malformed
    document.
    """
    source = inspect.getsource(pdf_fragments)
    tree = ast.parse(source)
    extract_fragments_def = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "extract_fragments"
    )
    metadata_calls = [
        node
        for node in ast.walk(extract_fragments_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "version"
    ]
    assert not metadata_calls, (
        "extract_fragments must not re-read the pypdf version; _engine() already validated "
        "it against _PINNED_PYPDF_VERSION and refused anything else"
    )
