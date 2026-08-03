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
import inspect

import pytest

from carmel.services import numeric
from carmel.services.semantic_deps import (
    CURRENT_SHA_BY_DEPENDENCY_ID,
    DEPENDENCIES_BY_SHA,
    InputPolicy,
    SemanticDependencyDefinition,
    SemanticDependencyInvariantError,
    UnknownSemanticDependencyError,
    compute_dependency_sha,
    current_sha_for,
    dependency_for_sha,
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
