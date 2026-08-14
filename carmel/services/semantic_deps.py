# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed identity for the *code* behind a versioned heuristic.

:attr:`~carmel.schemas.datasets.MeasuredValue.repairs` is the persisted output
of a versioned heuristic in :mod:`carmel.services.numeric`
(:func:`~carmel.services.numeric.normalize_numeric_span`). Its validator
re-runs that heuristic and demands an exact match against the stored
``repairs``. Nothing on a stored ``MeasuredValue`` records WHICH version of
the heuristic produced it, so editing ``numeric.py`` later would make old,
already-valid records fail validation indistinguishably from forged ones --
there is no way to tell "this record disagrees with the current heuristic
because the heuristic changed" from "this record was never valid at all."

This module builds the identity primitive that closes that gap: a content
address for a piece of Python source code (a function/class and everything it
transitively calls, at module scope), plus a small append-only registry
mapping known content addresses to a stable ``dependency_id``. This identity
is consumed by :class:`~carmel.schemas.datasets.SemanticDependencyUse` (which
embeds a ``content_sha256`` and resolves it against this module's registry)
and, through it, by
:meth:`~carmel.schemas.datasets.MeasuredValue._validate_repair_chain_agrees_with_raw_text`,
which is the validator that actually re-runs
:func:`~carmel.services.numeric.normalize_numeric_span` and rejects a
``MeasuredValue`` whose ``repair_dependency`` names a registered-but-SUPERSEDED
sha rather than silently re-validating against whatever the heuristic
currently does.

Three limitations are load-bearing and must not be relaxed silently:

1. **The transitive closure computed by :func:`compute_dependency_sha` is
   WITHIN-MODULE ONLY.** Walking ``ast.Name`` references only ever resolves a
   name against OTHER module-level definitions found in the same source
   text; a name that is actually an imported symbol resolves to nothing and
   is silently excluded from the hash. This is only safe to use against
   :mod:`carmel.services.numeric` today because that module imports nothing
   from ``carmel.*`` at all (see ``tests/test_semantic_deps.py``'s guard test,
   which asserts this by parsing the real module's imports). If ``numeric.py``
   ever gains a ``carmel.*`` import that its repair heuristic depends on, this
   closure is no longer sufficient to capture the heuristic's full behavior,
   and that guard test will fail loudly rather than let the sha silently stop
   covering part of the heuristic.
2. **``ast.dump`` output is not formally guaranteed stable across Python
   versions.** A toolchain upgrade could in principle change the exact dump
   string for an AST whose *meaning* did not change, which would change every
   sha this module computes even though the underlying heuristic is
   unchanged -- i.e. the sha is toolchain-relative in principle. This is
   intentionally CONTAINED, not fixed: :mod:`tests.test_semantic_deps` pins
   the seeded dependency's current sha as a hardcoded string literal, so a
   toolchain change that alters ``ast.dump``'s rendering surfaces as a loud,
   specific test failure (fix: add a new registry entry for the new sha) --
   never as silent, undetected drift.
3. **The closure never crosses a module boundary, by design -- not merely by
   omission.** :meth:`~carmel.schemas.datasets.MeasuredValue._validate_repair_chain_agrees_with_raw_text`
   (the consumer described above) also depends on
   :func:`~carmel.services.dataset_store.canonical_decimal`, which lives in a
   DIFFERENT module (``carmel.services.dataset_store``, not
   ``carmel.services.numeric``). This module deliberately does NOT attempt a
   cross-module transitive closure to cover that dependency too -- doing so
   would require walking imports across arbitrary module boundaries, which
   this module's within-module-only design (limitation 1, above) explicitly
   does not do. ``canonical_decimal``'s own correctness is simply outside the
   scope of what a ``content_sha256`` computed by this module can attest to;
   this is a named, accepted limitation, not a bug to fix here.

This module also does not model FULL numeric parsing -- see
:data:`DEPENDENCIES_BY_SHA`'s seeded entries and their docstrings for exactly
what each seeded dependency does and does not cover. Glyph-health ASSESSMENT is
a separate registered dependency (:data:`GLYPH_HEALTH_DEPENDENCY_ID`); the
repair entry still does not cover it, and the two are deliberately not merged.

4. **A fourth limitation applies specifically to
   :data:`EXTRACT_TEXT_DEPENDENCY_ID` (the identity for
   :func:`~carmel.agents.tools.extract.extract_text`): the third-party
   ``pypdf`` package is imported LAZILY inside
   :func:`~carmel.agents.tools.extract._extract_pdf`'s function body, not at
   module scope.** :func:`compute_dependency_sha`'s within-module AST closure
   (limitation 1, above) only ever walks ``ast.Name`` references against
   OTHER module-level definitions found in the same source text -- it never
   executes anything, so it cannot observe a name bound by a runtime-local
   ``import`` statement, let alone attribute the behavior of the imported
   package's own code to a hash. This means ``EXTRACT_TEXT_DEPENDENCY_ID``'s
   ``content_sha256`` captures Carmel's own extraction code faithfully but
   says NOTHING about which version of ``pypdf`` produced a given extraction
   -- two extractions with the identical ``content_sha256`` can still differ
   if the installed ``pypdf`` version differs. :class:`ExtractionIdentity`
   and :func:`extraction_identity` exist to make that pypdf version a
   required, separate component of a complete extraction identity, exactly
   mirroring the pattern already used for exactly this purpose by
   :func:`carmel.services.evidence._extractor_identity` (best-effort,
   never-fatal version discovery via ``pypdf.__version__``).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from carmel.services.dataset_store import canonical_json_bytes

__all__ = [
    "CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID",
    "GLYPH_HEALTH_DEPENDENCY_ID",
    "EXTRACT_TEXT_DEPENDENCY_ID",
    "FRAGMENT_GEOMETRY_DEPENDENCY_ID",
    "FRAGMENT_GEOMETRY_BORROWED_NAMES",
    "DEPENDENCIES_BY_SHA",
    "CURRENT_SHA_BY_DEPENDENCY_ID",
    "ExtractionIdentity",
    "FragmentGeometryIdentity",
    "InputPolicy",
    "SemanticDependencyDefinition",
    "SemanticDependencyInvariantError",
    "UnknownSemanticDependencyError",
    "compose_component_sha",
    "compute_dependency_sha",
    "current_sha_for",
    "dependency_for_sha",
    "extraction_identity",
    "fragment_geometry_identity",
]

GLYPH_HEALTH_DEPENDENCY_ID = "carmel.numeric.glyph_health"
"""The stable ``dependency_id`` for the glyph-health ASSESSMENT dependency.

Deliberately a SEPARATE identity from
:data:`CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID`, never an extension of it. The
two heuristics have genuinely different closures -- editing
``assess_glyph_health`` must not re-address stored repair claims, and editing
the repair chain must not re-address stored glyph assessments. That split is
empirically real, not aspirational: the two entry-point sets share only
``GlyphHealth`` and ``_ASCII6_UNCERTAINTY_RE``, and each computes a different
content sha.

Unlike the repair dependency, this one's input is NOT a sibling field of the
record that cites it -- it is a whole extracted document text living in the
evidence store -- so it carries
:attr:`InputPolicy.EXTERNAL_DIGEST_REQUIRED` and is the first real user of
that branch.
"""

CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID = "carmel.numeric.context_free_span_repair"
"""The stable ``dependency_id`` for the span-repair dependency in this registry.

Exported so callers outside this module (schema-layer validators, in
particular) that need to pin ``MeasuredValue.repair_dependency`` to exactly
this dependency never re-type the string literal by hand -- see
:func:`_seed_registry`, which uses this same constant rather than an inline
copy.
"""

EXTRACT_TEXT_DEPENDENCY_ID = "carmel.extraction.extract_text"
"""The stable ``dependency_id`` for :func:`~carmel.agents.tools.extract.extract_text`.

Its ``content_sha256`` is the within-module transitive closure of
``extract_text`` in ``carmel/agents/tools/extract.py`` -- Carmel's own PDF/
HTML/XML text-extraction code, page classification, and text normalization,
but NOT the third-party ``pypdf`` package's own behavior (see the module
docstring's fourth limitation, and :class:`ExtractionIdentity`).

Like :data:`GLYPH_HEALTH_DEPENDENCY_ID`, this dependency's input is not a
sibling field of the record that cites it -- it is a whole document's raw
bytes -- so it carries :attr:`InputPolicy.EXTERNAL_DIGEST_REQUIRED`.

This increment ONLY registers the identity in this module; wiring it into
:mod:`carmel.services.evidence` or any schema/producer/replay code is a
separate, later increment (out of scope here by design).
"""

FRAGMENT_GEOMETRY_DEPENDENCY_ID = "carmel.geometry.extract_fragments"
"""The stable ``dependency_id`` for :func:`~carmel.services.pdf_fragments.extract_fragments`.

The identity for the PDF *fragment geometry* lane: which code produced a fragment's
``x_start``/``x_end``/``baseline_y``/``font_height`` and the page number it claims.
:data:`EXTRACT_TEXT_DEPENDENCY_ID` does NOT cover this -- it is the identity of the
plain-text lane, a different closure in a different module.

**This entry's ``content_sha256`` is a COMPOSITE**, and is the only registry row that is
not a direct :func:`compute_dependency_sha` output. That is forced by the code, not a
preference. ``extract_fragments`` borrows five names from
:mod:`carmel.agents.tools.extract` via a FUNCTION-SCOPE import, and a within-module closure
cannot reach what they DO.

State that gap precisely, because it is narrower than it first looks and the narrow version
is the one that justifies this design. The import STATEMENT is inside ``extract_fragments``'s
own body, so it is part of the hashed dump: adding or removing a borrowed name does move the
own-closure sha. What no within-module closure can see is the imported code's BEHAVIOUR --
an edit inside ``_classify_pdf_page`` moves nothing in ``pdf_fragments`` at all (verified by
perturbation, both directions). And ``_classify_pdf_page`` is what decides the page number
every fragment claims, so a single within-module sha here would attest the geometry while
silently omitting the provenance. The composite therefore hashes two component shas together
(see :func:`compose_component_sha`):

1. ``own_sha256`` -- ``extract_fragments``'s own within-module closure.
2. ``borrowed_sha256`` -- the closure of EXACTLY the five borrowed names in ``extract.py``
   (:data:`FRAGMENT_GEOMETRY_BORROWED_NAMES`), computed as its own within-module closure.

Component 2 is deliberately NOT ``EXTRACT_TEXT_DEPENDENCY_ID``'s sha, even though all five
names happen to sit inside ``extract_text``'s 40-name closure today. Reusing that sha would
be wrong in both directions at once: OVER-attesting, because HTML/XML/abstract/references
behaviour unrelated to geometry would re-address stored geometry; and UNDER-attesting the
moment ``extract_text`` stops reaching one of the five while ``pdf_fragments`` still imports
it, at which point the borrowed name would fall out of the hashed closure with no sha
change at all. Hashing the borrowed entry points directly cannot drift that way.

The third component of a COMPLETE geometry identity -- the installed ``pypdf`` version --
is deliberately NOT in this sha, exactly as for :data:`EXTRACT_TEXT_DEPENDENCY_ID`: a
registry row must be a fixed historical fact, and the version is a property of the runtime
that produced an artifact, not of the code. See :class:`FragmentGeometryIdentity`, which
carries all three parts separately.

Like the two entries above, this dependency's input is a whole document's raw bytes rather
than a sibling field of the citing record, so it carries
:attr:`InputPolicy.EXTERNAL_DIGEST_REQUIRED`.

This increment ONLY registers the identity. NOTHING consumes it: at the time of writing no
shipped module imports :mod:`carmel.services.pdf_fragments` or
:mod:`carmel.services.pdf_cells`, no schema field carries fragment geometry, and no stored
artifact exists that could cite it. Registering before the first producer exists is the
point -- an identity added after artifacts are stored can never attribute the ones already
written. Wiring it into evidence/schemas/producers/replay is a separate, later increment.
"""

FRAGMENT_GEOMETRY_BORROWED_NAMES: tuple[str, ...] = (
    "MAX_PDF_PAGES",
    "_PageKind",
    "_classify_pdf_page",
    "_describe_page_error",
    "_quiet_pypdf",
)
"""The exact names :func:`~carmel.services.pdf_fragments.extract_fragments` borrows from
:mod:`carmel.agents.tools.extract`, and the entry points of the composite identity's
``borrowed_sha256`` component (see :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID`).

Sorted, and pinned as data rather than inlined at the one call site, because this tuple is
the identity's completeness claim: ``borrowed_sha256`` attests exactly these names and
nothing else, so a SIXTH borrowed name would be a real behavioural input that no component
of the composite hashes. ``tests/test_semantic_deps.py`` guards that with
:func:`_assert_borrowed_carmel_names`, which reads ``pdf_fragments.py``'s real source and
fails if the set of ``carmel.*`` names it imports -- at any scope -- is not exactly this
tuple from exactly that module.
"""


class SemanticDependencyError(ValueError):
    """Base class for every error this module raises."""


class SemanticDependencyInvariantError(SemanticDependencyError):
    """Raised when a :class:`SemanticDependencyDefinition` violates its own invariants."""


class UnknownSemanticDependencyError(SemanticDependencyError):
    """Raised when a sha256 or a dependency_id names nothing in this module's registry.

    Covers both :func:`dependency_for_sha` (unknown/malformed ``content_sha256``)
    and :func:`current_sha_for` (unknown ``dependency_id``) -- both are "this
    key is absent from the registry" in the same sense, so they share one
    error type rather than two indistinguishable-in-practice ones.
    """


def _bound_names_in_target(target: ast.expr) -> list[str]:
    """Return every plain name bound by an assignment/``for``/``with`` target.

    Python's assignment-target grammar is closed: a target is one of
    ``Name | Tuple | List | Starred | Attribute | Subscript``. ``Attribute``
    (``obj.attr = ...``) and ``Subscript`` (``obj[k] = ...``) do not bind any
    NEW referenceable name -- they mutate something already bound elsewhere --
    so they contribute nothing here and are safely ignored. ``Tuple``/``List``/
    ``Starred`` are recursed into. This covers every real target shape with no
    residual "can't resolve this" case, so no shape here needs to raise.
    """
    names: list[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            names.extend(_bound_names_in_target(element))
    elif isinstance(target, ast.Starred):
        names.extend(_bound_names_in_target(target.value))
    return names


def _named_expr_targets(node: ast.AST) -> list[str]:
    """Return every name bound by a walrus (``:=``) expression within ``node``.

    A ``NamedExpr`` binds to the nearest enclosing function/module scope (PEP
    572), not to a comprehension's own scope, so a module-level walrus
    anywhere inside a top-level statement (including inside a comprehension in
    that statement) is a module-level binding. ``ast.walk`` would also descend
    into any nested ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``/``Lambda``
    bodies, whose own walrus bindings belong to THAT scope, not the module --
    so this walks manually and stops at those boundaries instead of using
    ``ast.walk`` directly.
    """
    names: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.NamedExpr):
                if isinstance(child.target, ast.Name):
                    names.append(child.target.id)
                stack.append(child.value)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            else:
                stack.append(child)
    return names


def _collect_bindings(node: ast.stmt, into: dict[str, ast.stmt], owner: ast.stmt) -> None:
    """Recursively collect every module-level name bound by ``node`` into ``into``.

    ``owner`` is the top-level module statement that ``node`` lives inside (or
    ``node`` itself, at the top level) -- every name found anywhere inside a
    top-level ``if``/``try``/``for``/``with`` block is keyed to that whole
    top-level statement, since that is the coarsest unit this module's closure
    algorithm can meaningfully hash or include/exclude as one piece.

    ``ast.Match`` (structural pattern matching) is deliberately NOT handled: a
    ``match`` statement's capture-pattern binding shapes (``case Point(x=x,
    y=y):``, ``case [a, *rest]:``, ``case {"k": v}:``, ``case _ as name:``, ...)
    are too open-ended to enumerate with confidence here. Rather than silently
    ignore a shape that might bind a module-level name (the exact
    silent-corruption failure mode this function exists to close), a
    module-level ``match`` statement makes this function raise loudly instead
    -- see the module docstring's within-module-only limitation for the same
    "fail loudly rather than silently drop coverage" policy applied elsewhere.
    """
    if isinstance(node, ast.Match):
        raise SemanticDependencyInvariantError(
            "module-level 'match' statements are not supported by "
            "_module_level_definitions: their capture-pattern binding shapes are too "
            "open-ended to resolve confidently, so this raises loudly rather than risk "
            "silently missing a module-level binding"
        )
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        into[node.name] = owner
        return
    if isinstance(node, ast.Assign):
        for target in node.targets:
            for name in _bound_names_in_target(target):
                into[name] = owner
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        into[node.target.id] = owner
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        for name in _bound_names_in_target(node.target):
            into[name] = owner
        for child in (*node.body, *node.orelse):
            _collect_bindings(child, into, owner)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                for name in _bound_names_in_target(item.optional_vars):
                    into[name] = owner
        for child in node.body:
            _collect_bindings(child, into, owner)
    elif isinstance(node, ast.If):
        for child in (*node.body, *node.orelse):
            _collect_bindings(child, into, owner)
    elif isinstance(node, ast.Try):
        for child in (*node.body, *node.orelse, *node.finalbody):
            _collect_bindings(child, into, owner)
        for handler in node.handlers:
            for child in handler.body:
                _collect_bindings(child, into, owner)
    for name in _named_expr_targets(node):
        into[name] = owner


def _module_level_definitions(tree: ast.Module) -> dict[str, ast.stmt]:
    """Collect every module-level definition in ``tree``, keyed by name.

    ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` are keyed on their own
    name. ``Assign``/``AnnAssign``/``For``/``With`` targets are keyed on every
    plain ``ast.Name`` they bind, including through tuple/list/starred
    unpacking (see :func:`_bound_names_in_target`). A top-level ``if``/``try``/
    ``for``/``with`` block's entire nested body is walked recursively (see
    :func:`_collect_bindings`) so a name bound inside one of those blocks is
    not invisible to the closure; every name found anywhere inside such a
    block is keyed to that whole top-level statement, since that is the
    coarsest unit available to include/exclude as one piece. A top-level
    walrus (``:=``) binds a module-level name the same way (see
    :func:`_named_expr_targets`). A module-level ``match`` statement is
    refused outright rather than silently under-covered -- see
    :func:`_collect_bindings`.

    ``Attribute``/``Subscript`` assignment targets bind no new name and are
    correctly ignored (see :func:`_bound_names_in_target`). A bare expression
    statement, and any name that is actually an imported symbol, is not a
    resolvable module-level name and is never added here -- a reference to
    such a thing from within a definition's body is treated exactly like any
    other unresolved reference: silently excluded from the closure (see the
    module docstring's within-module-only limitation). Import BINDINGS
    themselves (as opposed to references to imported names) are hashed
    separately by :func:`_normalized_imports`, not folded into this closure.
    """
    definitions: dict[str, ast.stmt] = {}
    for node in tree.body:
        _collect_bindings(node, definitions, node)
    return definitions


def _normalized_imports(tree: ast.Module) -> list[tuple[str, str, str | None]]:
    """Return a deterministic, order-independent list of every module-level import binding.

    Each entry is ``(module, name, alias)`` where ``module`` is ``""`` for a
    plain ``import x`` and the dotted source module for ``from x import y``;
    ``name`` is the imported symbol (or sub-module) name; ``alias`` is the
    ``as`` name if given, else ``None``. Sorted so the result -- and therefore
    the hash built from it -- does not depend on import statement order.

    This exists to close a silent-corruption hole: swapping ``import re`` for
    ``import regex as re`` rebinds the name ``re`` to a completely different
    module without changing a single character any existing definition-based
    closure would see (since a *reference* to ``re`` inside a definition's
    body still resolves to nothing under the within-module-only closure,
    exactly as before the swap). Hashing the import bindings themselves is the
    only way this module's sha can notice such a swap.
    """
    imports: list[tuple[str, str, str | None]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(("", alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports.append((module, alias.name, alias.asname))
    return sorted(imports)


def _referenced_names(node: ast.AST) -> set[str]:
    """Return every name referenced anywhere inside ``node`` (``ast.Name.id`` values)."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _transitive_closure(entry_points: Sequence[str], definitions: Mapping[str, ast.stmt]) -> dict[str, ast.stmt]:
    """Walk the transitive closure of module-level names reachable from ``entry_points``.

    Starting from ``entry_points``, repeatedly walks each already-included
    definition's body for ``ast.Name`` references, adding any referenced name
    that resolves to another module-level definition, until a fixpoint (no
    new name is added). A referenced name that does NOT resolve to a
    module-level definition (an import, a builtin, a local variable inside a
    nested function) is silently skipped -- it is not part of this module's
    surface to hash.
    """
    closure: dict[str, ast.stmt] = {}
    frontier = list(entry_points)
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        definition = definitions.get(name)
        if definition is None:
            continue
        closure[name] = definition
        for referenced in _referenced_names(definition):
            if referenced in definitions and referenced not in closure:
                frontier.append(referenced)
    return closure


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    """Strip a leading docstring expression from ``tree`` and every nested
    function/class/module body inside it, recursively.

    A "docstring" here is exactly: a body's first statement being an
    ``ast.Expr`` whose value is an ``ast.Constant`` holding a ``str``. Applies
    at every nesting level ``ast.walk`` visits -- module, class, function, and
    any function/class nested inside another -- not only at the top level of
    ``tree`` itself.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                remainder = body[1:]
                node.body = remainder if remainder else [ast.Pass()]
    return tree


def _normalized_dump(definition: ast.stmt) -> str:
    """Render ``definition`` to a formatting-insensitive, docstring-insensitive string.

    Two steps, in order: ``ast.unparse`` (which already drops comments and
    original whitespace/formatting by construction, since it re-renders from
    the AST rather than echoing source text), then re-parse the unparsed
    source and strip docstrings recursively (see :func:`_strip_docstrings`)
    before rendering the final ``ast.dump`` with default arguments (i.e.
    ``annotate_fields=True, include_attributes=False`` -- the defaults omit
    ``lineno``/``col_offset``/etc, so position information never affects the
    dump either).

    **"Docstring-insensitive" is narrower than it reads, and the gap has already
    bitten once.** A docstring is the FIRST statement of a module, class, or
    function body, and that is exactly what :func:`_strip_docstrings` removes. The
    PEP 258 attribute doc -- a bare string sitting after ``field: int`` in a class
    body -- is not one. It is an ordinary ``Expr`` statement, it survives the
    stripping, and it is hashed as code. So documenting a FIELD of a hashed
    dataclass moves that dependency's sha and needs a supersession, while
    documenting a function does not. Every field of
    :class:`carmel.services.pdf_fragments.TextFragment` carries such a string, so
    this is not a hypothetical corner: see :func:`~carmel.services.pdf_fragments._pen_x_after`,
    which holds a paragraph that lives there rather than on the field it describes
    for precisely this reason.

    Not "fixed" by stripping those too. A field doc is positionally
    indistinguishable from a bare string expression a caller might rely on, and
    hashing MORE than the semantics is the safe direction for a content address:
    it can only ever cause a spurious supersession, never a silent reuse of an
    identity whose behaviour moved.
    """
    unparsed_source = ast.unparse(definition)
    reparsed = ast.parse(unparsed_source)
    stripped = _strip_docstrings(reparsed)
    # `stripped` is a Module wrapping exactly one statement (the definition);
    # dump that statement itself, not the wrapping Module, so an unrelated
    # change to how Module nodes are represented can never leak in.
    (only_statement,) = stripped.body
    return ast.dump(only_statement)


def compute_dependency_sha(module_source: str, entry_points: Sequence[str]) -> str:
    """Compute a content address for the transitive closure of ``entry_points`` in ``module_source``.

    Algorithm:

    1. Parse ``module_source`` and collect its module-level definitions (see
       :func:`_module_level_definitions`).
    2. Walk the transitive closure of module-level names reachable from
       ``entry_points`` (see :func:`_transitive_closure`) -- WITHIN-MODULE
       ONLY; see the module docstring's first limitation.
    3. Normalize each definition in the closure via ``ast.unparse`` followed
       by recursive docstring stripping (see :func:`_normalized_dump`), then
       render it with ``ast.dump`` using default arguments.
    4. Separately, collect every module-level import binding (see
       :func:`_normalized_imports`) -- this is hashed alongside the closure,
       not folded into it, so that rebinding an imported name (e.g.
       ``import re`` becoming ``import regex as re``) changes the sha even
       though no definition's own source text changed.
    5. Build a ``{"closure": {name: dump_string, ...}, "imports": [...]}``
       payload, sorted by name/import order, and hash it via this project's
       existing canonicalization helper,
       :func:`~carmel.services.dataset_store.canonical_json_bytes` (never a
       hand-rolled ``"|"``-joined string -- see
       :func:`carmel.services.evidence._derivation_binding` for why that
       shape is an anti-pattern this project already has one instance of and
       should not gain a second).

    Args:
        module_source: The full source text of a Python module.
        entry_points: Module-level names to start the transitive closure
            from. Every entry point must itself resolve to a module-level
            definition in ``module_source``.

    Returns:
        A 64-character lowercase hex SHA-256 digest.

    Raises:
        SemanticDependencyInvariantError: If any ``entry_points`` name does
            not resolve to a module-level definition in ``module_source``.
    """
    tree = ast.parse(module_source)
    definitions = _module_level_definitions(tree)
    for entry_point in entry_points:
        if entry_point not in definitions:
            raise SemanticDependencyInvariantError(
                f"entry point {entry_point!r} is not a module-level definition in the given module source "
                f"(known module-level names: {sorted(definitions)!r})"
            )
    closure = _transitive_closure(entry_points, definitions)
    dumps_by_name = {name: _normalized_dump(closure[name]) for name in sorted(closure)}
    payload = {
        "closure": dumps_by_name,
        # canonical_json_bytes accepts `list`, not `tuple` -- each (module, name,
        # alias) entry is converted to a list here so the payload is JSON-safe.
        "imports": [list(entry) for entry in _normalized_imports(tree)],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class InputPolicy(StrEnum):
    """How much of a semantic dependency's input a re-validator must supply itself.

    - ``SIBLING_FIELD``: the heuristic's full input is already present as a
      sibling field on the same record being validated (e.g. ``raw_text`` on
      a ``MeasuredValue``) -- no fabricated or externally supplied context is
      needed to re-run it.
    - ``EXTERNAL_DIGEST_REQUIRED``: re-running the heuristic requires context
      that is not itself part of the record and must be looked up and
      digest-bound separately (reserved for a future dependency; nothing
      seeded in this registry uses it yet).
    - ``NOT_APPLICABLE``: the dependency is not the kind of thing that is
      "re-run" against record input at all (reserved; nothing seeded in this
      registry uses it yet).
    """

    SIBLING_FIELD = "sibling_field"
    EXTERNAL_DIGEST_REQUIRED = "external_digest_required"
    NOT_APPLICABLE = "not_applicable"


_DEPENDENCY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SemanticDependencyDefinition:
    """One registered, content-addressed version of a versioned code dependency.

    Modeled on :class:`~carmel.services.units.ConversionTable`: a frozen
    stdlib dataclass living in the services layer, not a pydantic model --
    this is registry data consumed by schema-layer validators, not itself
    part of a stored dataset's schema.

    Attributes:
        dependency_id: A stable slug naming WHAT this dependency is (e.g.
            ``"carmel.numeric.context_free_span_repair"``), shared by every
            historical entry that has ever implemented it. Dotted lowercase
            segments, each starting with a letter (see :data:`_DEPENDENCY_ID_RE`).
        content_sha256: The digest of the EXACT code that implements this
            dependency as of this entry. Never mutated once shipped -- see
            :data:`DEPENDENCIES_BY_SHA`'s append-only contract.

            USUALLY a :func:`compute_dependency_sha` output, but NOT always,
            and a consumer must not assume it can recompute one from a single
            module's source. :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID` is a
            :func:`compose_component_sha` digest over two component shas,
            because the code it identifies spans two modules through an import
            no within-module closure can follow. The derivation is a property
            of the DEPENDENCY, documented on each ``dependency_id`` constant,
            not something this field encodes -- so a future generic
            re-validator must dispatch on ``dependency_id`` rather than assume
            a uniform recomputation. This is a real limitation of the current
            shape, named here rather than left to be discovered: nothing today
            recomputes a registered sha to validate a record (validators
            resolve the registry by lookup), so no consumer is broken by it.
        input_policy: Which :class:`InputPolicy` a re-validator must follow
            to re-run this dependency correctly.
        is_current: Whether THIS entry is the current (most recently shipped,
            not-yet-superseded) version for its ``dependency_id``. EXPLICIT,
            never inferred from tuple/iteration order -- see
            :func:`_build_registry`, which enforces that exactly one entry per
            ``dependency_id`` across the whole registry sets this ``True``.
    """

    dependency_id: str
    content_sha256: str
    input_policy: InputPolicy
    is_current: bool

    def __post_init__(self) -> None:
        if not _SHA256_HEX_RE.fullmatch(self.content_sha256):
            raise SemanticDependencyInvariantError(
                f"content_sha256 {self.content_sha256!r} must be exactly 64 lowercase hex characters "
                "(a SHA-256 hex digest)"
            )
        if not self.dependency_id or not _DEPENDENCY_ID_RE.fullmatch(self.dependency_id):
            raise SemanticDependencyInvariantError(
                f"dependency_id {self.dependency_id!r} must be a non-empty, dotted, lowercase slug "
                "(e.g. 'carmel.numeric.context_free_span_repair')"
            )
        if not isinstance(self.input_policy, InputPolicy):
            raise SemanticDependencyInvariantError(
                f"input_policy {self.input_policy!r} must be an InputPolicy member; a stdlib "
                "dataclass does not enforce field types at runtime, so this is checked explicitly "
                "here rather than trusted from a type annotation alone"
            )


def _numeric_module_source() -> str:
    import inspect

    from carmel.services import numeric

    source = inspect.getsource(numeric)
    return source


def _extract_module_source() -> str:
    import inspect

    from carmel.agents.tools import extract

    source = inspect.getsource(extract)
    return source


# The within-module-only closure limitation documented in this module's docstring is
# only sound as long as the module being hashed imports nothing from carmel.* itself.
#
# This is a DEVELOPMENT-TIME invariant, so it is enforced from the test suite
# (tests/test_semantic_deps.py walks the target module's AST independently) and
# deliberately NOT at import time. Running it on import would make merely importing this
# module read a target module's source off disk, which raises OSError under frozen/
# zipimport deployments where inspect.getsource cannot reach a source file -- turning a
# packaging change into an unexplained import crash in `carmel serve`. A check that can
# only fire in a checkout, where the test suite already runs, belongs in the test suite.
#
# Generalized (module_source, module_name) rather than hardcoded to
# carmel.services.numeric: this is now shared by that module's guard test and by
# carmel.agents.tools.extract's guard test (see tests/test_semantic_deps.py), and would
# otherwise have to be duplicated verbatim for every new within-module-only closure this
# registry gains.
def _assert_no_carmel_imports(module_source: str, module_name: str) -> None:
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "carmel" or alias.name.startswith("carmel."):
                    raise SemanticDependencyInvariantError(
                        f"{module_name} now imports from carmel.* "
                        f"({alias.name!r}); compute_dependency_sha's transitive closure is "
                        "within-module only and can no longer be assumed to capture this "
                        "module's full behavior -- see the semantic_deps module docstring"
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "carmel" or node.module.startswith("carmel."))
        ):
            raise SemanticDependencyInvariantError(
                f"{module_name} now imports from carmel.* "
                f"({node.module!r}); compute_dependency_sha's transitive closure is "
                "within-module only and can no longer be assumed to capture this "
                "module's full behavior -- see the semantic_deps module docstring"
            )


#: Key used by :func:`_carmel_imported_names` for a RELATIVE import.
#:
#: A relative import's ``ast.ImportFrom`` carries ``level >= 1`` and a ``module`` that is
#: relative to the importing package, so it can name carmel code while matching no
#: ``carmel.*`` prefix at all: ``from ..agents.tools.extract import _helper`` parses as
#: ``module='agents.tools.extract', level=2``. Resolving that to an absolute name requires
#: knowing the importing module's own package, which this function is not given and which
#: a caller could supply wrongly.
#:
#: So a relative import is not resolved -- it is recorded under this key and can therefore
#: never match a named expectation, which makes any relative import in a guarded module a
#: loud failure rather than a silent pass. Every carmel module this project guards uses
#: absolute imports throughout; a relative one appearing in a guarded module is a change
#: worth stopping on regardless of what it names.
_RELATIVE_IMPORT_KEY = "<relative import>"

#: Recorded as the name set for an import that binds a whole module rather than specific
#: names -- ``import carmel.x.y`` and ``from carmel.x import *``. No finite name tuple can
#: pin such a surface, so it can never satisfy a named expectation either.
_WHOLE_MODULE_NAME = "*"


def _carmel_imported_names(module_source: str) -> dict[str, set[str]]:
    """Map ``carmel.*`` source module -> the names ``module_source`` imports from it.

    Walks the WHOLE tree, not just ``tree.body``, so a function-scope import is seen --
    which is the entire point: :func:`compute_dependency_sha` cannot see what a
    function-scope import's target DOES, so something else has to notice that the import
    is there at all.

    Three shapes are recorded but deliberately NOT resolved, because each names code that
    no finite name tuple can pin: a relative import (:data:`_RELATIVE_IMPORT_KEY`), a plain
    ``import carmel.x.y``, and a star import (both :data:`_WHOLE_MODULE_NAME`). Recording
    rather than resolving is what makes them fail a named expectation loudly.

    What this CANNOT see, stated plainly because a guard that overstates itself is worse
    than no guard: any import that is not an ``import``/``from ... import`` statement.
    ``importlib.import_module("carmel.x")``, ``__import__``, a ``sys.modules`` lookup, a
    callable handed in by a caller, or carmel code re-exported through a non-carmel package
    all reach carmel behaviour invisibly here. No AST walk can close that set; it is bounded
    by convention and review, not by this function.
    """
    tree = ast.parse(module_source)
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "carmel" or alias.name.startswith("carmel."):
                    imported.setdefault(alias.name, set()).add(_WHOLE_MODULE_NAME)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imported.setdefault(_RELATIVE_IMPORT_KEY, set()).update(alias.name for alias in node.names)
            elif node.module is not None and (node.module == "carmel" or node.module.startswith("carmel.")):
                imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return imported


# The counterpart to _assert_no_carmel_imports, for a module that legitimately DOES import
# from carmel.* and therefore cannot use that guard at all.
#
# _assert_no_carmel_imports states the strong invariant ("this module's behaviour is fully
# captured by its own within-module closure") and is the right guard for numeric.py and
# extract.py. carmel/services/pdf_fragments.py cannot satisfy it -- extract_fragments
# borrows five names from carmel.agents.tools.extract at function scope -- so the honest
# guard is the weaker, explicit one: the set of borrowed names is EXACTLY the set the
# composite identity hashes as its `borrowed_sha256` component.
#
# Why this must exist at all, stated as the SMALLER true claim rather than the tempting
# larger one: adding a sixth borrowed name DOES move the composite, because the import
# statement lives in extract_fragments' hashed body (verified by perturbation -- do not
# "simplify" this comment back to "the sha would not move"). The hazard is subtler. The
# composite would move once, at the moment the name is added, and then
# `borrowed_sha256` would go on attesting only the ORIGINAL five for the rest of that
# entry's life. The sixth name's behaviour would be covered by nothing, so a later edit
# INSIDE it would move no sha at all -- and that later edit is the one nobody would be
# looking at. Silent under-attestation is precisely the failure mode this whole module
# exists to prevent, so it gets a guard that fails loudly at the moment the name appears.
#
# Deliberately compares the FULL carmel.* import surface, not merely the function-scope
# part: a new MODULE-LEVEL carmel import in pdf_fragments would be visible to
# _normalized_imports as a binding, but its imported code's behaviour would still be
# outside every closure this module hashes. Both scopes are the same hazard here.
def _assert_borrowed_carmel_names(
    module_source: str,
    module_name: str,
    expected: Mapping[str, Sequence[str]],
) -> None:
    actual = _carmel_imported_names(module_source)
    expected_sets = {module: set(names) for module, names in expected.items()}
    if actual != expected_sets:
        rendered_actual = {module: sorted(names) for module, names in sorted(actual.items())}
        rendered_expected = {module: sorted(names) for module, names in sorted(expected_sets.items())}
        raise SemanticDependencyInvariantError(
            f"{module_name}'s carmel.* import surface changed: expected {rendered_expected!r}, "
            f"found {rendered_actual!r}. compute_dependency_sha cannot see these imports "
            "(its closure and its import list are both module-level only), so each borrowed "
            "name must be hashed explicitly as a component of the composite identity -- see "
            "FRAGMENT_GEOMETRY_DEPENDENCY_ID. Adding a borrowed name without adding it to "
            "FRAGMENT_GEOMETRY_BORROWED_NAMES (and registering the resulting NEW composite "
            "sha) would leave that name's behaviour attested by nothing at all"
        )


#: Matches a well-formed lowercase hex SHA-256 digest, for :func:`compose_component_sha`'s
#: component validation. Anchored via ``fullmatch`` at the call site.
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def compose_component_sha(components: Mapping[str, str]) -> str:
    """Hash a set of NAMED component shas into one composite content address.

    For an identity whose inputs cannot be captured by a single within-module closure --
    today only :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID`, see its docstring for why.

    The components are named rather than positional, and hashed through
    :func:`~carmel.services.dataset_store.canonical_json_bytes` rather than joined into a
    string, for the same reason :func:`compute_dependency_sha` does it: a hand-rolled
    ``"|"``-joined shape is an anti-pattern this project already carries one instance of
    (:func:`carmel.services.evidence._derivation_binding`) and should not gain a second.
    Naming the components additionally means SWAPPING two component values changes the
    composite, which a positional join of two same-length hex digests would not guarantee
    to express as clearly.

    This is deliberately NOT a general "combine any shas" helper with an ordering
    convention a caller could get wrong: it takes a mapping, and
    ``canonical_json_bytes`` sorts it.

    Args:
        components: Component name -> that component's 64-char hex sha256. Must be
            non-empty; every value must be a well-formed lowercase hex digest, because a
            composite over a malformed component would be a content address that attests
            to nothing while looking exactly like one that does.

    Returns:
        A 64-character lowercase hex SHA-256 digest.

    Raises:
        SemanticDependencyInvariantError: If ``components`` is empty or any value is not a
            64-character lowercase hex digest.
    """
    if not components:
        raise SemanticDependencyInvariantError(
            "compose_component_sha requires at least one component; an empty composite "
            "would be a fixed digest that attests to nothing"
        )
    for name, sha in sorted(components.items()):
        if not _HEX_SHA256_RE.fullmatch(sha):
            raise SemanticDependencyInvariantError(
                f"component {name!r} is not a 64-character lowercase hex sha256 ({sha!r}); "
                "a composite over a malformed component would look like a valid content "
                "address while attesting to nothing"
            )
    return hashlib.sha256(canonical_json_bytes(dict(components))).hexdigest()


# HARDCODED, not derived from live source or any other computation at import
# time. This is the crux of the append-only doctrine: DEPENDENCIES_BY_SHA's seeded entry
# must be a PINNED HISTORICAL FACT ("this sha validated stored data as of this registry
# entry"), independent of whatever carmel/services/numeric.py currently contains. Deriving
# it by hashing the live numeric.py source at import time would silently
# move this "pinned" sha every time numeric.py's repair heuristic changed, making the
# append-only contract structurally unimplementable (there would never be an old, fixed
# entry to add a new one alongside) and would additionally break under frozen/zipimport
# deployments where `inspect.getsource` cannot read the module's source file at all.
#
# This literal was independently verified once via:
#   compute_dependency_sha(
#       inspect.getsource(carmel.services.numeric),
#       ["normalize_numeric_span", "REPAIR_NAMES"],
#   )
# tests/test_semantic_deps.py::test_seeded_dependency_sha_matches_a_hardcoded_pin
# re-verifies that equality on every test run. If that test ever fails, it means
# numeric.py's repair heuristic (or this toolchain's ast.dump rendering) changed -- the
# fix is to ADD A NEW registry entry at a new sha, never to edit this literal or the
# registry row that uses it.
_CONTEXT_FREE_SPAN_REPAIR_SHA256 = "b29d34f644deff19a68e618340408839a138186a5a4229fed8babd4f22fedabb"

# HARDCODED for exactly the same reasons as the literal above; the whole comment
# there applies here verbatim. Independently verified once via:
#   compute_dependency_sha(
#       inspect.getsource(carmel.services.numeric), ["assess_glyph_health"]
#   )
# Adding "GlyphHealth" as a second entry point does NOT change this value -- the
# class is already pulled into the closure transitively by assess_glyph_health's
# return statement -- so the single entry point is not an under-specification.
#
# NOTE this sha and _CONTEXT_FREE_SPAN_REPAIR_SHA256 are NOT independently
# versioned: compute_dependency_sha folds the WHOLE module's import list into
# every payload, so any import edit in numeric.py moves BOTH shas at once, even
# one touching neither closure. That over-firing is deliberate and must not be
# "fixed" by scoping imports to the closure: over-firing costs a spurious
# re-address (loud, cheap), while under-hashing costs a behaviour change with no
# identity change (silent corruption -- the exact failure this module exists to
# prevent).
_GLYPH_HEALTH_SHA256 = "af3553a8142b50bba56b6ba164778b4cd2bff6e4916ac2e93c4e1a270ba4ab5a"

# HARDCODED for exactly the same reasons as the two literals above; the whole comment on
# _CONTEXT_FREE_SPAN_REPAIR_SHA256 applies here verbatim. Independently verified once via:
#   compute_dependency_sha(
#       inspect.getsource(carmel.agents.tools.extract), ["extract_text"]
#   )
# This closure captures Carmel's own extraction/normalization code (PDF page
# classification, HTML/XML extraction, whitespace/hyphen/ligature repair, abstract- and
# references-region detection) but NOT the third-party pypdf package's own behavior --
# see the module docstring's fourth limitation and ExtractionIdentity, below. If this
# test ever fails, it means extract.py's extraction surface (or this toolchain's
# ast.dump rendering) changed -- the fix is to ADD A NEW registry entry at a new sha,
# never to edit this literal or the registry row that uses it.
_EXTRACT_TEXT_SHA256 = "aa008f66d255cfb079cf269438ef9cfb0f1c42c6326d51a75e3e6fed04ec7168"

# The two COMPONENTS of the fragment-geometry composite, and the composite itself. All
# three are HARDCODED for exactly the same reasons as the three literals above; the whole
# comment on _CONTEXT_FREE_SPAN_REPAIR_SHA256 applies here verbatim. Independently verified
# once via:
#   own      = compute_dependency_sha(
#                  inspect.getsource(carmel.services.pdf_fragments), ["extract_fragments"]
#              )
#   borrowed = compute_dependency_sha(
#                  inspect.getsource(carmel.agents.tools.extract),
#                  list(FRAGMENT_GEOMETRY_BORROWED_NAMES),
#              )
#   composite = compose_component_sha(
#                  {"borrowed_sha256": borrowed, "own_sha256": own}
#              )
#
# Adding "TextFragment" as a second entry point does NOT change the own sha -- the class is
# already pulled into extract_fragments' closure transitively -- so the single entry point
# is not an under-specification, the same way it is not for _GLYPH_HEALTH_SHA256.
#
# The components are pinned SEPARATELY from the composite, not merely recomputed from it,
# so that a failing pin says WHICH half moved: geometry code, or a borrowed name's
# behaviour in a module this one cannot see. A single composite pin would report only that
# something, somewhere, changed.
# SUPERSEDED, and kept forever -- the first fragment-geometry entry, registered while
# `extract_fragments` still recorded its pypdf version with a second
# `importlib.metadata.version` call. Removing that read moved the OWN component and
# therefore the composite. The BORROWED component did not move at all, which is the
# component split doing precisely what it exists to do: the failure localizes to
# `pdf_fragments`, and nothing suggests a borrowed name's behaviour changed.
_FRAGMENT_GEOMETRY_OWN_SHA256_V1 = "c93639f8dab3d79c37a1dd3d5ca4d66d9397d1c174d230fe2e10e677568ae8e3"
_FRAGMENT_GEOMETRY_SHA256_V1 = "4922bd55d53e90e9bcd7cb4823e15798cb89ffddb6b2b6d7745f96c9ff1767bb"

# SUPERSEDED, and kept forever -- the SECOND fragment-geometry entry, registered while
# `_decoded_content_length` still measured a page by decoding it in full through
# `get_data()` and comparing afterwards. Bounding that decode changes which pages become
# failures (a stream with an unhandled filter, or one bare zlib cannot inflate, now
# fails its page), so it moved the OWN component and therefore the composite. The
# BORROWED component did not move -- the second time in a row the split has localized a
# change to `pdf_fragments` and said, positively, that nothing in a module this one
# cannot see had shifted underneath it.
_FRAGMENT_GEOMETRY_OWN_SHA256_V2 = "96b5852b71496f062dd1d36b255f98feb952baf89fe7e4fb995b93ce00a56f5e"
_FRAGMENT_GEOMETRY_SHA256_V2 = "3fc972d0394184267e85a9a9e42387423eed538758efeba3ce1fd125ef56c47b"

# SUPERSEDED, and kept forever -- the THIRD fragment-geometry entry, registered while the
# bounded decode checked only `unconsumed_tail`. That witness cannot see a TRUNCATED stream:
# a valid deflate prefix consumes all of its input, so the tail is empty, no `zlib.error` is
# raised, and the measured length is the prefix's rather than the content's. Measured on a
# fixture: 3,654 of 8,200 bytes returned silently. Checking `eof` (and `unused_data`) turns
# that into a page failure, which changes which pages become failures and therefore moved the
# OWN component. The BORROWED component did not move -- the third time in a row the split has
# localized a change to `pdf_fragments`.
_FRAGMENT_GEOMETRY_OWN_SHA256_V3 = "e745a377714ea6a817a82977d028d6c2820091f85be7f87c4972bd70f9e41e44"
_FRAGMENT_GEOMETRY_SHA256_V3 = "652cdea53a2c44a9861b6896b6cb8234d86b0ac6745c3ddc135e728522e5b25e"

# SUPERSEDED, and kept forever -- the FOURTH fragment-geometry entry, registered while
# `_page_fragments` still took pypdf's `recurse_to_target_op` at its word about WHERE each
# text-show operation starts. It is wrong twice, established against ISO 32000-1 9.4.4 on
# synthetic PDFs whose every operand and glyph width was chosen so the answer could be
# computed by hand (pypdf matched the specification on 7 of 11 such cases): a show operator
# never advances the pen, so consecutive `Tj` runs stack at one x; and a `TJ` array charges
# `Tc` once per ELEMENT rather than once per glyph, so every element after the first starts
# short. 22 and 7,815 sites respectively, on 11 and 61 of the corpus's 75 pages.
# `_walk_operations` replaces that walk, so every fragment's `x_start` moves on any page
# with either construct. This is the largest of the four moves by far -- the previous three
# changed which pages FAILED, this one changes the coordinates themselves -- and it is still
# the OWN half alone that moved.
_FRAGMENT_GEOMETRY_OWN_SHA256_V4 = "d446c737b7d37cf50adaf4070250bc75f721f958b62680981bc89f8a5b474967"
_FRAGMENT_GEOMETRY_SHA256_V4 = "4ae9d68f0bcbf55bfbcaef1f7c7a2dda02b64ef4bc6bdf7cc504672d59810545"

# SUPERSEDED, and kept forever -- the FIFTH fragment-geometry entry, registered while
# `_walk_operations` still STEPPED OVER every operator it did not name. It published
# geometry for a page whose stream reframed it (`/Rotate`, `/UserUnit`), whose text was
# painted at zero fill alpha or under a soft mask, or that carried an inline image or a
# compatibility section -- and, worst of the set, it published FABRICATED TEXT: a
# `NameObject` inside a `TJ` array subclasses `str`, so `[(01) /Nm (23)] TJ` emitted a
# fragment reading `/Nm` at real page coordinates, text that no glyph drew wearing
# perfectly checkable geometry.
#
# Unlike V4 this moves no coordinate. Every one of these constructs is absent from the
# eight-paper corpus -- 78,178 fragments and 0 page failures before and after, not one
# fragment moved -- so what the new sha marks is a change in what the extractor REFUSES,
# not in what it computes. That is exactly why it must still supersede: a page that now
# fails would previously have produced fragments, and an artifact cannot be allowed to
# claim the newer identity for the older behaviour. The BORROWED component is unmoved for
# the fifth consecutive time.
_FRAGMENT_GEOMETRY_OWN_SHA256_V5 = "dbd9b8a255b6339438cb4551fcb6c8d0aa3e434694f5ccf71abf921a076d9cbd"
_FRAGMENT_GEOMETRY_SHA256_V5 = "75310c6df1677158c15e233e2da4abe72c52fcc565dbaa0f0f7ca36cb8b50f3a"

_FRAGMENT_GEOMETRY_OWN_SHA256 = "4dd477809aaf0874e427e72cf8ff5e12391455f525bd56ea385999c57eff1101"
_FRAGMENT_GEOMETRY_BORROWED_SHA256 = "39844d90f40067b45a6413816336fd9cbb7a1f9db8be05c75640b74d56ea8199"
_FRAGMENT_GEOMETRY_SHA256 = "6789f2a10e8f58f56f5e5969187525f1ca33f740d8015254da2133ca55363108"


@dataclass(frozen=True, slots=True)
class _FragmentGeometryComponents:
    """What one composite fragment-geometry sha was composed FROM.

    A composite is a one-way hash: given a stored ``composite_sha256`` and nothing else,
    there is no way to recover which halves produced it, and therefore no way to say WHICH
    half moved when a stored artifact disagrees with current code. That localization is the
    whole reason the identity is composite rather than opaque, so the halves have to be
    recorded, and recorded the same way the registry records everything else: APPEND-ONLY,
    keyed by the composite they belong to.

    Module-level constants alone would not do it. They describe only what is CURRENT, so
    the first geometry supersession would either overwrite the components of the superseded
    row -- losing exactly the localization the composite exists to provide, for exactly the
    old artifacts that need it most -- or leave them behind as literals no longer reachable
    from any row.

    ``borrowed_names`` is part of the record for the same reason: a future entry may borrow
    a different set, and reading a superseded row's components against today's tuple would
    silently mis-describe what that row attested.
    """

    own_sha256: str
    borrowed_sha256: str
    borrowed_names: tuple[str, ...]


#: APPEND-ONLY, keyed by composite sha, exactly like :data:`DEPENDENCIES_BY_SHA`. Never
#: edit an existing row: add a new one when the composite changes, so a stored artifact
#: citing an OLD composite can still be told which half moved. Read through
#: :func:`_fragment_geometry_components`, never subscripted directly, so an unknown
#: composite raises this module's own error type rather than a bare ``KeyError``.
_FRAGMENT_GEOMETRY_COMPONENTS_BY_SHA: Mapping[str, _FragmentGeometryComponents] = MappingProxyType(
    {
        _FRAGMENT_GEOMETRY_SHA256_V1: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256_V1,
            # Identical to the current entry's, and that is the point rather than a
            # duplication to factor out: recording it per-composite is what lets a future
            # reader see that this supersession did NOT touch the borrowed half. A shared
            # reference would express "the same today", not "the same when each shipped".
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
        _FRAGMENT_GEOMETRY_SHA256_V2: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256_V2,
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
        _FRAGMENT_GEOMETRY_SHA256_V3: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256_V3,
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
        _FRAGMENT_GEOMETRY_SHA256_V4: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256_V4,
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
        _FRAGMENT_GEOMETRY_SHA256_V5: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256_V5,
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
        _FRAGMENT_GEOMETRY_SHA256: _FragmentGeometryComponents(
            own_sha256=_FRAGMENT_GEOMETRY_OWN_SHA256,
            borrowed_sha256=_FRAGMENT_GEOMETRY_BORROWED_SHA256,
            borrowed_names=FRAGMENT_GEOMETRY_BORROWED_NAMES,
        ),
    }
)


def _fragment_geometry_components(composite_sha256: str) -> _FragmentGeometryComponents:
    """Return what ``composite_sha256`` was composed from.

    Raises:
        UnknownSemanticDependencyError: If the composite is not recorded. Deliberately the
            same error type an unknown registered sha raises: a composite with no recorded
            components cannot be localized, and guessing is the failure this module exists
            to prevent.
    """
    try:
        return _FRAGMENT_GEOMETRY_COMPONENTS_BY_SHA[composite_sha256]
    except KeyError:
        raise UnknownSemanticDependencyError(
            f"composite sha {composite_sha256!r} has no recorded components; it cannot be "
            "localized to a geometry half. Every composite ever shipped must keep a row in "
            "_FRAGMENT_GEOMETRY_COMPONENTS_BY_SHA -- append a new one, never edit an existing one"
        ) from None


def _seed_registry() -> tuple[SemanticDependencyDefinition, ...]:
    """Return the append-only registry's initial contents.

    NAME PRECISION: "context_free_span_repair" -- not "numeric_parsing" and not
    "glyph_health_admission". The eventual re-validator that will consume this
    registry (not built yet) re-runs normalize_numeric_span() with FABRICATED
    context: SourceContext.OPERATOR_RAW and an all-False GlyphHealth (see
    carmel.schemas.datasets._HEALTHY_GLYPH_HEALTH and
    MeasuredValue._validate_repair_chain_agrees_with_raw_text). That means this
    dependency's sha pins only the CONTEXT-FREE repair chain a bare raw_text
    string can drive on its own -- never the glyph-health-aware branches of
    normalize_numeric_span that only trigger for a real document's assessed
    GlyphHealth, and never anything about full numeric parsing beyond this one
    heuristic. Do not rename or redocument this entry to imply broader coverage.

    The glyph-health entry is the ASSESSMENT half that the repair entry above
    explicitly excludes. Its input is a whole extracted document text held in the
    evidence store rather than a sibling field of the citing record, hence
    EXTERNAL_DIGEST_REQUIRED: without a recorded digest of the exact text that
    was assessed, a disagreement between a stored assessment and a fresh one is
    uninterpretable (same input under a changed heuristic, or a different input
    entirely?). Recording the digest is what makes that distinguishable later.

    HONEST SCOPE: validation of a stored assessment can check registry identity,
    digest shape and the node's extraction binding. It CANNOT re-run
    assess_glyph_health, because the input is out-of-payload. A stored
    assessment is therefore UNVERIFIED-BY-CONSTRUCTION until the replayer
    milestone lands. Do not describe it as verified anywhere.

    The fragment-geometry entry is the only COMPOSITE row, and the only one whose
    content_sha256 is not a direct compute_dependency_sha output -- see
    FRAGMENT_GEOMETRY_DEPENDENCY_ID for why a single within-module closure cannot
    honestly cover that lane. HONEST SCOPE, stated as plainly as the glyph-health
    entry's: nothing consumes this row yet, and no artifact cites it. It attests
    which code WOULD produce a fragment's coordinates, not that any stored
    coordinate was produced by it. Do not describe anything as geometry-verified
    on the strength of this entry existing.
    """
    return (
        SemanticDependencyDefinition(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=_CONTEXT_FREE_SPAN_REPAIR_SHA256,
            input_policy=InputPolicy.SIBLING_FIELD,
            is_current=True,
        ),
        SemanticDependencyDefinition(
            dependency_id=GLYPH_HEALTH_DEPENDENCY_ID,
            content_sha256=_GLYPH_HEALTH_SHA256,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=True,
        ),
        SemanticDependencyDefinition(
            dependency_id=EXTRACT_TEXT_DEPENDENCY_ID,
            content_sha256=_EXTRACT_TEXT_SHA256,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=True,
        ),
        # The registry's first SUPERSESSION, and the first exercise of the append-only
        # contract it has always claimed. Kept because the row is what a stored artifact
        # citing this sha would resolve against; dropping it would make such an artifact
        # indistinguishable from a forged one. There are no such artifacts today -- which
        # is exactly why superseding now is cheap, and why it was worth doing deliberately
        # while nothing depends on it rather than discovering the path under pressure.
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256_V1,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=False,
        ),
        # The SECOND supersession. The first was worth doing because nothing depended on
        # the sha yet; this one is the evidence that the path stayed cheap -- it is the
        # same four edits, and the components table is what makes a stored artifact
        # citing either superseded row still resolvable to the half that moved.
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256_V2,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=False,
        ),
        # The THIRD supersession, and the first one bought by a defect an adversarial review
        # found rather than by a change already planned: the bounded decode admitted a
        # TRUNCATED stream as a measured one. Three supersessions in three sessions is worth
        # noting rather than normalising -- but each moved the OWN half only, so the split is
        # doing the job it was built for, and the cost of each has stayed at four edits.
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256_V3,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=False,
        ),
        # The FOURTH supersession, and the first that changes COORDINATES rather than which
        # pages fail. The three before it were all about refusal boundaries, so an artifact
        # stored under any of them would have been either absent or identical; an artifact
        # stored under this one would carry different numbers for the same page. Nothing is
        # stored yet, which is the fourth time that has been the reason a supersession was
        # cheap -- and the last time it will be worth saying, because the honest reading is
        # that this lane's geometry should be treated as unstable until a producer exists.
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256_V4,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=False,
        ),
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256_V5,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=False,
        ),
        SemanticDependencyDefinition(
            dependency_id=FRAGMENT_GEOMETRY_DEPENDENCY_ID,
            content_sha256=_FRAGMENT_GEOMETRY_SHA256,
            input_policy=InputPolicy.EXTERNAL_DIGEST_REQUIRED,
            is_current=True,
        ),
    )


def _build_registry(
    entries: tuple[SemanticDependencyDefinition, ...],
) -> tuple[Mapping[str, SemanticDependencyDefinition], Mapping[str, str]]:
    """Build the two registry mappings, enforcing invariants that are otherwise
    only INFERRED from tuple contents (never checked).

    Enforces, at import time, fail-loudly (not silently-collapse-into-a-dict):

    - No two entries may share a ``content_sha256`` -- a dict/MappingProxyType
      built directly from ``entries`` would silently let a later duplicate
      shadow an earlier one, which is exactly the kind of silent collapse this
      function exists to refuse.
    - Exactly one entry per ``dependency_id`` may set ``is_current=True`` --
      never zero (an orphaned dependency_id with no current version) and never
      more than one (an ambiguous "current").

    Args:
        entries: The full seeded registry, in any order.

    Returns:
        A ``(DEPENDENCIES_BY_SHA, CURRENT_SHA_BY_DEPENDENCY_ID)`` pair, each
        wrapped in a read-only :class:`~types.MappingProxyType`.

    Raises:
        SemanticDependencyInvariantError: If any of the invariants above is
            violated by ``entries``.
    """
    by_sha: dict[str, SemanticDependencyDefinition] = {}
    for entry in entries:
        if entry.content_sha256 in by_sha:
            raise SemanticDependencyInvariantError(
                f"content_sha256={entry.content_sha256!r} is registered more than once "
                f"(dependency_id={entry.dependency_id!r} collides with "
                f"{by_sha[entry.content_sha256].dependency_id!r}); each registry entry must "
                "have a unique content address, never silently collapsed into one"
            )
        by_sha[entry.content_sha256] = entry

    current_by_id: dict[str, str] = {}
    for entry in entries:
        if not entry.is_current:
            continue
        if entry.dependency_id in current_by_id:
            raise SemanticDependencyInvariantError(
                f"dependency_id={entry.dependency_id!r} has more than one entry with "
                "is_current=True; exactly one entry per dependency_id must be current, "
                "and which one is current must never be ambiguous"
            )
        current_by_id[entry.dependency_id] = entry.content_sha256

    all_dependency_ids = {entry.dependency_id for entry in entries}
    missing_current = all_dependency_ids - current_by_id.keys()
    if missing_current:
        raise SemanticDependencyInvariantError(
            f"dependency_id(s) {sorted(missing_current)!r} have no entry with "
            "is_current=True; every dependency_id in the registry must have exactly one "
            "current version, never zero"
        )

    return MappingProxyType(by_sha), MappingProxyType(current_by_id)


_SEEDED_DEPENDENCIES: tuple[SemanticDependencyDefinition, ...] = _seed_registry()

DEPENDENCIES_BY_SHA: Mapping[str, SemanticDependencyDefinition]
CURRENT_SHA_BY_DEPENDENCY_ID: Mapping[str, str]
DEPENDENCIES_BY_SHA, CURRENT_SHA_BY_DEPENDENCY_ID = _build_registry(_SEEDED_DEPENDENCIES)
"""Every semantic dependency version this module knows about, keyed by content address
(:data:`DEPENDENCIES_BY_SHA`), and the CURRENT ``content_sha256`` for each known
``dependency_id`` (:data:`CURRENT_SHA_BY_DEPENDENCY_ID`).

APPEND-ONLY, exactly like :data:`~carmel.services.units.TABLES_BY_SHA`: a new
version of a dependency (the heuristic's code changed) is added as a NEW
entry alongside the old one, never by mutating or removing an existing
entry's ``content_sha256``. Multiple entries may share the same
``dependency_id`` at once (one per historical code version); which one is
"current" is the separate, EXPLICIT ``is_current`` field on each entry (see
:class:`SemanticDependencyDefinition`) -- never inferred from iteration or
tuple order. :func:`_build_registry` enforces, at import time, that exactly
one entry per ``dependency_id`` sets ``is_current=True`` and that no two
entries share a ``content_sha256``.
"""


def dependency_for_sha(sha256: str) -> SemanticDependencyDefinition:
    """Return the :class:`SemanticDependencyDefinition` whose content address is ``sha256``.

    Args:
        sha256: A dependency's content address (see
            :attr:`SemanticDependencyDefinition.content_sha256`).

    Returns:
        The matching entry.

    Raises:
        UnknownSemanticDependencyError: If ``sha256`` does not name any
            dependency this module's registry knows -- including if
            ``sha256`` is not even well-formed (this function does not
            special-case malformed input differently from genuinely unknown
            input; both are simply absent from the registry).
    """
    try:
        return DEPENDENCIES_BY_SHA[sha256]
    except KeyError:
        raise UnknownSemanticDependencyError(
            f"no semantic dependency known for content_sha256 {sha256!r}; known: {sorted(DEPENDENCIES_BY_SHA)!r}"
        ) from None


def current_sha_for(dependency_id: str) -> str:
    """Return the CURRENT ``content_sha256`` registered for ``dependency_id``.

    Args:
        dependency_id: A dependency's stable slug (see
            :attr:`SemanticDependencyDefinition.dependency_id`).

    Returns:
        The current entry's content address.

    Raises:
        UnknownSemanticDependencyError: If ``dependency_id`` names no
            dependency this module's registry knows.
    """
    try:
        return CURRENT_SHA_BY_DEPENDENCY_ID[dependency_id]
    except KeyError:
        raise UnknownSemanticDependencyError(
            f"no semantic dependency known for dependency_id {dependency_id!r}; known: "
            f"{sorted(CURRENT_SHA_BY_DEPENDENCY_ID)!r}"
        ) from None


_PYPDF_VERSION_UNKNOWN = "unknown"
"""Sentinel returned by :func:`_pypdf_version` when ``pypdf`` cannot be introspected.

Deliberately a plain, obviously-not-a-real-version string (not ``None`` and not an
empty string) so a caller that forgets to check for it produces an identity that is
visibly wrong (``"unknown"``) rather than one that looks superficially plausible.
"""


def _pypdf_version() -> str:
    """Best-effort installed ``pypdf`` version string, never fatal.

    Mirrors :func:`carmel.services.evidence._extractor_identity`'s pattern for exactly
    the same reason: version discovery for an optional/lazily-imported third-party
    package must never be allowed to crash a caller that only wants an identity for
    logging/comparison purposes. Any failure (package absent, import error, missing
    ``__version__`` attribute, or anything else) collapses to
    :data:`_PYPDF_VERSION_UNKNOWN` rather than propagating.
    """
    try:
        import pypdf

        return str(pypdf.__version__)
    except Exception:  # noqa: BLE001 - version discovery is best-effort, never fatal
        return _PYPDF_VERSION_UNKNOWN


def _pypdf_distribution_version() -> str:
    """Installed ``pypdf`` DISTRIBUTION version, from packaging metadata, never fatal.

    The other witness to the same fact, and the one that actually gates the lane:
    :func:`carmel.services.pdf_fragments._engine` admits pypdf only when this equals
    :data:`carmel.services.pdf_fragments._PINNED_PYPDF_VERSION`. :func:`_pypdf_version`
    asks the imported MODULE what it calls itself; this asks the installed
    DISTRIBUTION. An editable, vendored or shadowed install can make them disagree, and
    when they do it is the module attribute that describes the code which ran while this
    one describes the code the gate approved.

    Wrapped in the same total-function shape as its sibling, collapsing every failure --
    including ``PackageNotFoundError``, which is the expected one -- to
    :data:`_PYPDF_VERSION_UNKNOWN`. An identity that raises is an identity a caller
    cannot record, and the point of recording both witnesses is defeated if fetching the
    second one can fail the first.
    """
    try:
        return importlib.metadata.version("pypdf")
    except Exception:  # noqa: BLE001 - version discovery is best-effort, never fatal
        return _PYPDF_VERSION_UNKNOWN


@dataclass(frozen=True)
class ExtractionIdentity:
    """The complete, two-part content identity for a run of
    :func:`~carmel.agents.tools.extract.extract_text`.

    A single ``content_sha256`` (:data:`EXTRACT_TEXT_DEPENDENCY_ID`'s current sha) is
    NOT a complete extraction identity by itself -- see the module docstring's fourth
    limitation: the AST closure that produces that sha is within-module only, and
    ``pypdf`` is imported lazily inside
    :func:`~carmel.agents.tools.extract._extract_pdf`'s function body, so the closure
    cannot observe, and the sha says nothing about, which ``pypdf`` version actually
    ran. Two extractions can share an identical ``code_sha256`` while having been
    produced by genuinely different ``pypdf`` behavior. This dataclass makes both
    components explicit and separately inspectable, rather than encoding them into one
    opaque string a future consumer would have to parse (and could parse wrong) to
    recover either half.

    A frozen dataclass rather than a plain string was chosen deliberately: this module
    already uses frozen dataclasses (:class:`SemanticDependencyDefinition`) for its
    other identity-shaped values, and a struct keeps ``code_sha256`` and
    ``pypdf_version`` independently accessible and comparable without a caller having
    to agree on (and correctly parse) a delimiter convention.

    Attributes:
        code_sha256: The current ``content_sha256`` registered for
            :data:`EXTRACT_TEXT_DEPENDENCY_ID` -- Carmel's own extraction code, and
            nothing about ``pypdf``.
        pypdf_version: The installed ``pypdf`` version string, or
            :data:`_PYPDF_VERSION_UNKNOWN` if it could not be determined.
    """

    code_sha256: str
    pypdf_version: str


def extraction_identity() -> ExtractionIdentity:
    """Return the current composite identity for Carmel's extraction code.

    Combines :data:`EXTRACT_TEXT_DEPENDENCY_ID`'s current registered sha (Carmel's own
    extraction/normalization code) with the installed ``pypdf`` version (the
    third-party component the AST closure cannot see -- see the module docstring's
    fourth limitation). Neither half alone is a complete extraction identity; see
    :class:`ExtractionIdentity`.

    This increment ONLY exposes this helper from :mod:`carmel.services.semantic_deps`.
    Wiring it into :mod:`carmel.services.evidence` (e.g. alongside or in place of
    ``_extractor_identity``) or any schema/producer/replay code is a separate, later
    increment, out of scope here by design.
    """
    return ExtractionIdentity(
        code_sha256=current_sha_for(EXTRACT_TEXT_DEPENDENCY_ID),
        pypdf_version=_pypdf_version(),
    )


@dataclass(frozen=True)
class FragmentGeometryIdentity:
    """The complete, three-part content identity for a run of
    :func:`~carmel.services.pdf_fragments.extract_fragments`.

    The fragment-geometry counterpart to :class:`ExtractionIdentity`, and a strictly
    larger shape than it, because this lane has one more input that no AST closure can
    reach. ``extract_fragments`` runs Carmel code from TWO modules -- its own, plus five
    names borrowed from :mod:`carmel.agents.tools.extract` through a function-scope import
    that :func:`compute_dependency_sha` is blind to -- and then hands the parsing to
    ``pypdf``. Those are three genuinely independent ways the same input bytes can produce
    different coordinates, so they are three separately inspectable fields rather than one
    opaque string a consumer would have to parse (and could parse wrong) to recover any
    part of.

    ``composite_sha256`` is what a stored artifact would cite (it is the registered
    ``content_sha256`` for :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID`); the two component
    fields are carried alongside it so that a mismatch can be localized to the half that
    moved without recomputing anything.

    Attributes:
        composite_sha256: The current registered ``content_sha256`` for
            :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID` -- the composite over both component
            shas below, and nothing about ``pypdf``.
        own_sha256: The component covering ``extract_fragments``'s own within-module
            closure in :mod:`carmel.services.pdf_fragments`.
        borrowed_sha256: The component covering exactly
            :data:`FRAGMENT_GEOMETRY_BORROWED_NAMES` in
            :mod:`carmel.agents.tools.extract`.
        pypdf_version: The installed ``pypdf`` version string, or
            :data:`_PYPDF_VERSION_UNKNOWN` if it could not be determined.

            Read through :func:`_pypdf_version` (``pypdf.__version__``), which is NOT the
            witness the engine gate uses: ``_engine()`` admits the library only after
            ``importlib.metadata.version("pypdf")`` equals
            :data:`_PINNED_PYPDF_VERSION`. Those are two different witnesses to the same
            fact -- the package's own declared string versus installed distribution
            metadata -- and they can legitimately disagree for an editable, vendored or
            shadowed install, which would manufacture an identity mismatch for one
            unchanged runtime. They agree on the pinned ``6.14.2``, so this is a latent
            divergence, not a live one; one witness is chosen here rather than left to
            chance. ``_pypdf_version`` is the one chosen because it is total: it collapses
            every failure to :data:`_PYPDF_VERSION_UNKNOWN`, whereas
            ``importlib.metadata.version`` raises ``PackageNotFoundError``.

            ``extract_fragments`` is no longer a third reading of this: it records the
            :data:`_PINNED_PYPDF_VERSION` constant that ``_engine()`` already proved,
            rather than re-reading metadata. So the divergence is between exactly two
            call sites, the gate and this identity -- not three.

            **Both witnesses are now recorded, and neither is chosen.** The paragraph
            above used to end by picking this one, which was the wrong shape for an
            IDENTITY: choosing between two witnesses that can disagree means the record
            cannot afterwards say whether they did. Under a shadowed install this field
            alone would name a version the gate never approved, and nothing stored would
            show it. See :attr:`pypdf_distribution_version`.
        pypdf_distribution_version: The installed pypdf DISTRIBUTION's version, from
            packaging metadata, or :data:`_PYPDF_VERSION_UNKNOWN`.

            The witness ``_engine()`` actually gates on. Recorded ALONGSIDE
            :attr:`pypdf_version` rather than instead of it, because the two answer
            different questions -- what the imported module calls itself, and what the
            installed distribution declares -- and an identity's job is to record what
            was true, not to adjudicate between its own sources.

            They agree on the pinned ``6.14.2``, so this is a latent divergence and not a
            live one. Recording both is registering an identity, not building a
            mechanism: nothing here compares them, warns, or refuses, because no
            consumer exists to act on a disagreement yet. When one does, the evidence it
            needs will already be in every artifact stored from now on -- which is the
            whole reason to do it while nothing depends on it.
    """

    composite_sha256: str
    own_sha256: str
    borrowed_sha256: str
    pypdf_version: str
    pypdf_distribution_version: str


def fragment_geometry_identity() -> FragmentGeometryIdentity:
    """Return the current composite identity for Carmel's fragment-geometry code.

    Combines :data:`FRAGMENT_GEOMETRY_DEPENDENCY_ID`'s current registered sha with its two
    pinned components and the installed ``pypdf`` version. No half alone is a complete
    geometry identity; see :class:`FragmentGeometryIdentity`.

    Reads the composite from the registry rather than recomputing it from live source, for
    the same reason the seeded literals are hardcoded: the registry row is the pinned
    historical fact. ``tests/test_semantic_deps.py`` is what re-verifies that the pins still
    match live source on every run.

    This increment ONLY exposes this helper. Wiring it into
    :mod:`carmel.services.evidence`, or any schema/producer/replay code, is a separate,
    later increment -- and there is nothing to wire it into yet: no shipped module consumes
    :mod:`carmel.services.pdf_fragments` at all.
    """
    composite = current_sha_for(FRAGMENT_GEOMETRY_DEPENDENCY_ID)
    components = _fragment_geometry_components(composite)
    return FragmentGeometryIdentity(
        composite_sha256=composite,
        own_sha256=components.own_sha256,
        borrowed_sha256=components.borrowed_sha256,
        pypdf_version=_pypdf_version(),
        pypdf_distribution_version=_pypdf_distribution_version(),
    )
