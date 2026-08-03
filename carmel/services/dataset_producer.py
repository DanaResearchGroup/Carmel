# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Produce a validated :class:`~carmel.schemas.datasets.DatasetEnvelope` from one
REAL stored evidence artifact.

Naming: this module is the schema-AWARE producer counterpart to the
schema-blind :mod:`carmel.services.dataset_store`, exactly as
:mod:`carmel.services.dataset_bridge` is the schema-aware store/load layer
above that same blind store. ``dataset_bridge`` moves an already-built
envelope in and out of the store; this module is the one place that BUILDS an
envelope from the other real runtime artifact this project has -- a stored
literature artifact in the evidence store (:mod:`carmel.services.evidence`).
It therefore has to know about both sides at once (the evidence store's
layout/metadata and the dataset schema), which is exactly why it is its own
module rather than a method on either side.

The cardinal rule of the dataset schema (see :mod:`carmel.schemas.datasets`)
is that every number is grounded in stored, auditable source bytes. This
producer enforces that rule MECHANICALLY, not by convention:

- Callers state only DOMAIN FACTS: "the quote ``'1023'`` is a temperature
  coordinate whose unit is printed as ``'K'``". They never supply character
  offsets. Every :class:`~carmel.schemas.datasets.CharSpanLocator` in the
  produced envelope is computed by :func:`ground_quote` actually SEARCHING
  the artifact's verified extracted text -- an offset nobody searched for
  cannot appear in the output.
- The extracted text itself is never trusted from a convenience loader:
  ``extracted.json``'s file bytes are re-read and re-verified against the
  digest recorded at store time (``StoredArtifact.extracted_sha256``) BEFORE
  they are parsed at all. Unverified bytes are never parsed.
- ``ExtractionBinding.extracted_text_sha256`` is computed from
  ``extracted.text`` (the field inside the verified, parsed
  ``ExtractedText``), NEVER from ``text.txt`` on disk -- ``text.txt`` is
  presence-checked only, never digest-checked (see that field's docstring in
  :mod:`carmel.schemas.datasets`), so hashing it would anchor the binding to
  the one file in the store nothing verifies.

Everything here is fail-closed: a quote that cannot be found, is ambiguous,
or is out of range raises; a missing/legacy/corrupt artifact raises; nothing
ever silently guesses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    CharSpanLocator,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    EmbeddedConversionTable,
    ExtractionBinding,
    MeasuredValue,
    Observation,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TextSpace,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_store import (
    CanonicalDecimalError,
    canonical_decimal,
    canonical_json_bytes,
)
from carmel.services.evidence import artifact_dir, load_artifact_meta
from carmel.services.numeric import (
    GlyphHealth,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)
from carmel.services.semantic_deps import (
    CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
    current_sha_for,
)
from carmel.services.units import QuantityKind

__all__ = [
    "DatasetProducerError",
    "MeasurementSpec",
    "QuoteGroundingError",
    "ground_quote",
    "produce_envelope_from_artifact",
]

_EXTRACTED_NAME = "extracted.json"
"""Filename of the full ``ExtractedText`` sidecar inside an artifact's
content-addressed directory. This is the evidence store's PUBLIC on-disk
contract -- ``carmel.services.evidence``'s module docstring documents the
layout as exact ("Layout, exactly::") -- not private knowledge duplicated
here."""

_ROOT_NODE_ID = "paper"
"""The single :class:`SourceNode`'s id in every produced graph. One artifact
in, one root ``PAPER_PDF`` node out -- this vertical slice models exactly
that shape and nothing more."""

_POINT_ID = "p1"

# Mirrors carmel.schemas.datasets._HEALTHY_GLYPH_HEALTH (module-private there,
# so restated rather than imported): MeasuredValue's own repair-chain
# validator re-runs normalize_numeric_span with SourceContext.OPERATOR_RAW and
# a healthy, hand-constructed GlyphHealth, because a stored MeasuredValue
# carries no surrounding document to assess. This producer derives
# repairs/canonical_decimal_value through the IDENTICAL call so that what it
# records is exactly what the schema's validator will re-derive -- any other
# context here would let the producer construct values its own schema then
# rejects (or worse, accepts for the wrong reason).
_CONTEXT_FREE_GLYPH_HEALTH = GlyphHealth(
    suspects_dash_corruption=False,
    has_thorn_plus_marker=False,
    has_equals_ambiguity_marker=False,
    has_slash_c0_minus_marker=False,
    has_ascii6_uncertainty_marker=False,
)


class QuoteGroundingError(ValueError):
    """Raised by :func:`ground_quote` when a quote cannot be grounded
    unambiguously: empty quote, quote not found, ambiguous quote with no
    explicit occurrence, or an occurrence out of range for the matches
    actually found. Never a silent guess."""


class DatasetProducerError(ValueError):
    """Raised by :func:`produce_envelope_from_artifact` when the stored
    artifact cannot be resolved or verified, or a caller-supplied spec cannot
    be honestly realised. Fail-closed: nothing is ever produced from
    unverified bytes or an unverifiable claim."""


def ground_quote(text: str, quote: str, *, occurrence: int | None = None) -> CharSpanLocator:
    """Locate ``quote`` in ``text`` by SEARCHING, returning a half-open
    :class:`CharSpanLocator` over ``TextSpace.EXTRACTED_TEXT``.

    ``occurrence`` is ``int | None`` with a default of ``None``, deliberately
    NOT ``int = 0``: with ``int = 0`` a caller who never thought about
    ambiguity is indistinguishable from one who explicitly chose the first
    match, so ambiguity would silently resolve to match 0 -- exactly the
    guess this function exists to refuse. ``None`` (the default) means "I
    claim this quote is unique": it succeeds only when exactly one match
    exists, and an ambiguous quote raises, stating the match count. Passing
    an explicit ``occurrence`` (0-based) is the caller's affirmative
    disambiguation and selects that match; an out-of-range value raises,
    stating how many matches were actually found.

    Matches are counted INCLUDING overlapping ones (``"aa"`` occurs twice in
    ``"aaa"``): every position the quote appears at is a distinct candidate
    grounding, and undercounting would let a genuinely ambiguous quote pass
    the uniqueness check.

    Args:
        text: The verified extracted text to search (``ExtractedText.text``,
            the RAW string every ``CharSpanLocator`` offset indexes into --
            never ``.normalized``).
        quote: The exact source substring to locate, verbatim.
        occurrence: ``None`` to require uniqueness; a 0-based index to
            explicitly select among multiple matches.

    Returns:
        A ``CharSpanLocator`` with ``text[start:end] == quote``.

    Raises:
        QuoteGroundingError: Empty quote; quote not found; ambiguous quote
            with ``occurrence=None``; or ``occurrence`` out of range.
    """
    if not quote:
        # CharSpanLocator's own validator would eventually reject the
        # zero-width span this would produce, but with a generic
        # "end must be strictly greater than start" message about a locator
        # the caller never wrote. Refuse here, earlier, naming the actual
        # mistake: an empty quote matches at every position and grounds
        # nothing.
        raise QuoteGroundingError(
            "ground_quote: quote is empty -- an empty quote matches at every position in the text "
            "and grounds nothing; supply the exact, non-empty source substring to locate"
        )
    starts: list[int] = []
    found = text.find(quote)
    while found != -1:
        starts.append(found)
        found = text.find(quote, found + 1)
    display = quote if len(quote) <= 120 else quote[:117] + "..."
    if not starts:
        raise QuoteGroundingError(f"ground_quote: quote {display!r} was not found in the supplied text")
    if occurrence is None:
        if len(starts) > 1:
            raise QuoteGroundingError(
                f"ground_quote: quote {display!r} is ambiguous -- it appears {len(starts)} times in the "
                "supplied text; ambiguity never silently resolves to the first match, so pass "
                "occurrence= explicitly to select one"
            )
        start = starts[0]
    else:
        if not 0 <= occurrence < len(starts):
            raise QuoteGroundingError(
                f"ground_quote: occurrence={occurrence} is out of range for quote {display!r} -- "
                f"{len(starts)} match(es) found, so occurrence must be in [0, {len(starts) - 1}]"
            )
        start = starts[occurrence]
    end = start + len(quote)
    # Correctness self-check on this function's own arithmetic (an assert,
    # not a raise: user input was already validated above; this can only
    # fail if the search/slicing logic itself is wrong).
    assert text[start:end] == quote, "ground_quote arithmetic self-check failed: slice != quote"
    return CharSpanLocator(text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end)


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """One caller-stated domain fact: what a quoted number/unit/label IS.

    The caller states only meaning -- axis identity/role, physical quantity,
    and the VERBATIM quotes for the number, its unit, and the axis label as
    printed in the source. No character offsets, ever: the producer computes
    every offset itself via :func:`ground_quote` against the verified
    extracted text. The optional ``*_occurrence`` fields are the caller's
    explicit disambiguation for a quote that appears more than once (see
    :func:`ground_quote`); ``None`` claims the quote is unique.
    """

    axis_id: str
    role: AxisRole
    quantity_kind: QuantityKind
    label_quote: str
    value_quote: str
    unit_quote: str
    label_occurrence: int | None = None
    value_occurrence: int | None = None
    unit_occurrence: int | None = None


def _current_repair_dependency() -> SemanticDependencyUse:
    """The repair-dependency record for a value repaired by the CURRENT
    context-free span-repair heuristic -- resolved per call (not cached at
    import) so it always reflects the live registry."""
    return SemanticDependencyUse(
        dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
        content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
        input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )


def _measured_value(text: str, spec: MeasurementSpec) -> MeasuredValue:
    """Build one grounded :class:`MeasuredValue` from ``spec`` against ``text``.

    Every offset comes from :func:`ground_quote`; ``repairs`` and
    ``canonical_decimal_value`` are DERIVED from the value quote through the
    same ``normalize_numeric_span`` call the schema's own validator re-runs
    (see ``_CONTEXT_FREE_GLYPH_HEALTH``), never asserted independently.
    """
    value_locator = ground_quote(text, spec.value_quote, occurrence=spec.value_occurrence)
    unit_locator = ground_quote(text, spec.unit_quote, occurrence=spec.unit_occurrence)
    normalized = normalize_numeric_span(
        spec.value_quote,
        source_context=SourceContext.OPERATOR_RAW,
        glyph_health=_CONTEXT_FREE_GLYPH_HEALTH,
    )
    if isinstance(normalized, Unresolvable):
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for axis {spec.axis_id!r} is not derivable into a "
            f"numeral: {normalized.reason}"
        )
    try:
        canonical = canonical_decimal(normalized.text)
    except CanonicalDecimalError as exc:
        raise DatasetProducerError(
            f"value quote {spec.value_quote!r} for axis {spec.axis_id!r} repaired to "
            f"{normalized.text!r}, which is not a valid canonical decimal string: {exc}"
        ) from exc
    try:
        unit_normalized = units.normalize_unit(spec.quantity_kind, spec.unit_quote, table=units.TABLE_V1)
    except units.UnknownUnitError as exc:
        raise DatasetProducerError(
            f"unit quote {spec.unit_quote!r} for axis {spec.axis_id!r} is not a known unit or alias "
            f"of quantity_kind={spec.quantity_kind.value!r} in TABLE_V1: {exc}"
        ) from exc
    return MeasuredValue(
        raw_text=spec.value_quote,
        canonical_decimal_value=canonical,
        repairs=normalized.repairs,
        repair_dependency=_current_repair_dependency(),
        quantity_kind=spec.quantity_kind,
        unit_raw=spec.unit_quote,
        unit_normalized=unit_normalized,
        conversion_table_sha256=units.TABLE_V1.sha256,
        value_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=value_locator),
        unit_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=unit_locator),
    )


def _load_verified_extracted_text(workspace_root: Path, sha256: str) -> tuple[ExtractedText, str, str | None]:
    """Resolve, verify, and parse the stored extraction for ``sha256``.

    Returns ``(extracted, extracted_sha256, derivation_binding)``. The bytes
    of ``extracted.json`` are read directly from disk and their digest is
    compared against ``StoredArtifact.extracted_sha256`` BEFORE any parsing
    -- unverified bytes are never parsed. (``extracted_sha256`` really is the
    digest of those file bytes: ``evidence._write_all`` computes it as
    ``hashlib.sha256(extracted_path.read_bytes()).hexdigest()`` AFTER writing
    the file, precisely so the recorded digest describes the bytes on disk.)

    Artifact resolution goes through :func:`carmel.services.evidence.load_artifact_meta`
    (added alongside this module): a direct by-sha lookup with the read
    path's sha-shape/containment validation. The alternative --
    ``list_artifacts()`` and filter -- is O(store size) per call, silently
    skips artifacts whose ``meta.json`` is unreadable (so "absent" and
    "present but corrupt" become indistinguishable), and validates nothing
    about the caller-supplied sha string.
    """
    meta = load_artifact_meta(workspace_root, sha256)
    if meta is None:
        raise DatasetProducerError(
            f"no stored artifact found under sha256 {sha256!r} in this workspace's evidence store"
        )
    if meta.extracted_sha256 is None:
        # A legacy artifact stored before extracted_sha256 existed carries no
        # digest for its sidecar, so its extracted.json cannot be verified at
        # all -- and this producer never parses unverified bytes.
        raise DatasetProducerError(
            f"artifact {sha256!r} predates extracted_sha256 and its extracted.json cannot be "
            "verified; refusing to parse unverified bytes"
        )
    extracted_path = artifact_dir(workspace_root, sha256) / _EXTRACTED_NAME
    try:
        raw_bytes = extracted_path.read_bytes()
    except OSError as exc:  # FileNotFoundError is an OSError subclass
        raise DatasetProducerError(
            f"artifact {sha256!r} has no readable {_EXTRACTED_NAME}: {exc}"
        ) from exc
    actual_digest = hashlib.sha256(raw_bytes).hexdigest()
    if actual_digest != meta.extracted_sha256:
        raise DatasetProducerError(
            f"artifact {sha256!r}: {_EXTRACTED_NAME} bytes on disk hash to {actual_digest!r}, not the "
            f"recorded extracted_sha256 {meta.extracted_sha256!r}; refusing to parse unverified bytes"
        )
    try:
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    except ValueError as exc:
        raise DatasetProducerError(
            f"artifact {sha256!r}: verified {_EXTRACTED_NAME} bytes do not parse as an ExtractedText: {exc}"
        ) from exc
    return extracted, meta.extracted_sha256, meta.derivation_binding


def produce_envelope_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    series_id: str,
    value_origin: ValueOrigin,
    measurements: tuple[MeasurementSpec, ...],
) -> DatasetEnvelope:
    """Build a fully validated :class:`DatasetEnvelope` from ONE stored artifact.

    The vertical slice, end to end: resolve the artifact's metadata by its
    raw-bytes sha256, re-read and digest-verify ``extracted.json``, parse it,
    ground every caller-stated quote in the extracted text via
    :func:`ground_quote`, and assemble one root ``PAPER_PDF`` node, one
    ``TEXTUAL`` series, and one data point into an envelope that passes every
    schema validator (construction runs pydantic's full validation -- nothing
    here uses ``model_construct``).

    ``source_form`` is fixed at ``TEXTUAL``: a :class:`CharSpanLocator` into
    extracted running text is the only locator kind this runtime can actually
    produce (the round-33 ruling that added it), and it is what every span
    here is. ``value_origin`` is the caller's assertion, passed through --
    see :class:`ValueOrigin` for why the schema records it unverified.

    Args:
        workspace_root: Root of the campaign workspace holding the evidence
            store.
        sha256: Raw-bytes sha256 of the stored artifact to ground against.
        series_id: Id for the single produced series.
        value_origin: How the numbers were produced, as asserted by the
            caller.
        measurements: One :class:`MeasurementSpec` per axis; at least one
            ``COORDINATE`` and one ``OBSERVATION`` role are required (by the
            schema's own S3/S4 validators). ``CONSTANT`` is not supported by
            this producer.

    Returns:
        The validated envelope.

    Raises:
        DatasetProducerError: Artifact missing/legacy/corrupt, a value quote
            not derivable into a numeral, an unknown unit, or a ``CONSTANT``
            role spec.
        QuoteGroundingError: A quote that cannot be grounded unambiguously.
    """
    for spec in measurements:
        if spec.role is AxisRole.CONSTANT:
            raise DatasetProducerError(
                f"MeasurementSpec for axis {spec.axis_id!r} has role=CONSTANT, which this producer "
                "does not support -- every spec must be a per-point COORDINATE or OBSERVATION"
            )

    extracted, extracted_sha256, derivation_binding_raw = _load_verified_extracted_text(workspace_root, sha256)
    text = extracted.text
    # CRITICAL: hash extracted.text (the field inside the verified, parsed
    # ExtractedText), NEVER text.txt on disk -- text.txt is presence-checked
    # only, never digest-checked (see ExtractionBinding.extracted_text_sha256's
    # docstring), so a digest of it would be anchored to the one unverified
    # file in the store.
    extracted_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    derivation_binding: str | Absent
    if derivation_binding_raw is None:
        # UNKNOWN is the ONLY reason ExtractionBinding permits here (enforced
        # by its own validator): the digest predates the field and no other
        # AbsenceReason promises a remedy that exists.
        derivation_binding = Absent(reason=AbsenceReason.UNKNOWN)
    else:
        derivation_binding = derivation_binding_raw
    binding = ExtractionBinding(
        extracted_sha256=extracted_sha256,
        extracted_text_sha256=extracted_text_sha256,
        derivation_binding=derivation_binding,
    )
    root_node = SourceNode(
        node_id=_ROOT_NODE_ID,
        kind=SourceNodeKind.PAPER_PDF,
        sha256=sha256,
        parent_node_id=None,
        origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        extraction=binding,
        glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
    )
    graph = SourceGraph(nodes=(root_node,))

    axes: list[AxisDeclaration] = []
    coordinates: list[Coordinate] = []
    observations: list[Observation] = []
    for spec in sorted(measurements, key=lambda spec: spec.axis_id):
        label_locator = ground_quote(text, spec.label_quote, occurrence=spec.label_occurrence)
        axes.append(
            AxisDeclaration(
                axis_id=spec.axis_id,
                role=spec.role,
                quantity_kind=spec.quantity_kind,
                label_raw=spec.label_quote,
                label_ref=SourceRef(node_id=_ROOT_NODE_ID, locator=label_locator),
            )
        )
        value = _measured_value(text, spec)
        if spec.role is AxisRole.COORDINATE:
            coordinates.append(
                Coordinate(
                    axis_id=spec.axis_id,
                    value=value,
                    uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                )
            )
        else:
            observations.append(
                Observation(
                    axis_id=spec.axis_id,
                    value=value,
                    uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
                )
            )

    point = DataPoint(
        point_id=_POINT_ID,
        coordinates=tuple(coordinates),
        observations=tuple(observations),
        composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
    )
    series = Series(
        series_id=series_id,
        source_form=SourceForm.TEXTUAL,
        value_origin=value_origin,
        axes=tuple(axes),
        constants=(),
        points=(point,),
    )
    embedded_table = EmbeddedConversionTable(
        sha256=units.TABLE_V1.sha256,
        canonical_json=canonical_json_bytes(units.TABLE_V1.identity_payload()).decode("utf-8"),
    )
    return DatasetEnvelope(
        source_graph=graph,
        composition=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        series=(series,),
        conversion_tables=(embedded_table,),
    )
