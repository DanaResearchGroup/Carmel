# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Independent replay-and-reverify service for a stored
:class:`~carmel.schemas.datasets.DatasetEnvelope`.

Producing an envelope (:mod:`carmel.services.dataset_producer`) and loading
it back (:mod:`carmel.services.dataset_bridge`) both trust the same process
that wrote the record. This module is deliberately the OTHER path: given
nothing but a workspace root (the content-addressed evidence store) and an
already-loaded envelope, it re-derives every fact the envelope claims from
scratch and reports whether the envelope's claims still hold against that
independent re-derivation. Three things are re-proven, none of them taken on
the envelope's own say-so:

1. **Evidence identity.** For every :class:`~carmel.schemas.datasets.SourceNode`
   that carries an :class:`~carmel.schemas.datasets.ExtractionBinding`, this
   module re-reads ``extracted.json`` directly from the evidence store,
   re-verifies its bytes against the digest
   :class:`~carmel.schemas.literature.StoredArtifact` recorded at store time,
   and only THEN re-hashes the parsed text and compares it against the
   ``extracted_text_sha256`` the envelope's ``ExtractionBinding`` recorded.
   The envelope's own copy of the text (there isn't one -- envelopes never
   embed extracted text) plays no part; this is a from-bytes re-derivation.

2. **Every character span.** Every reachable
   :class:`~carmel.schemas.datasets.CharSpanLocator` is re-sliced against the
   independently re-read text for the node it names, walked generically via
   :func:`~carmel.schemas.datasets.iter_source_refs` (never hand-enumerated),
   so a newly added ref-bearing field can never silently go unchecked -- see
   :func:`replay_envelope`'s own exhaustiveness self-check.

3. **Every measured unit, against the table IT recorded.** For every
   :class:`~carmel.schemas.datasets.MeasuredValue` reachable via
   :func:`~carmel.schemas.datasets.iter_measured_values`, this module
   resolves ``conversion_table_sha256`` through
   :data:`carmel.services.units.TABLES_BY_SHA` -- the hand-reviewed registry
   of tables this codebase actually trusts -- and re-derives
   ``unit_normalized`` from ``unit_raw`` under THAT table. It never resolves
   against "the current preferred table," and it never trusts the
   envelope's own embedded :class:`~carmel.schemas.datasets.EmbeddedConversionTable`
   copy either: that copy's own docstring is explicit that internal
   self-consistency is not the same thing as being a table anyone has
   actually reviewed (see ``EmbeddedConversionTable``'s docstring for the
   "internally coherent nonsense" framing this module takes at face value).

Every check reports one of exactly three outcomes -- never a bare bool:

- ``VERIFIED``: every check that applies to this envelope ran, and every one
  of them passed.
- ``FAILED``: at least one check ran and produced a definite disagreement
  (a re-sliced span no longer matches its recorded quote; a re-read text's
  digest no longer matches what the envelope recorded; a unit no longer
  normalizes to what was recorded, under the table that was recorded).
- ``UNVERIFIABLE``: at least one check could not run at all (evidence
  missing or unreadable; a recorded conversion-table sha resolves to no
  known table). An ``UNVERIFIABLE`` finding is never folded into
  ``VERIFIED`` -- "could not check" and "checked and passed" are kept
  strictly apart throughout this module, mirroring the fail-closed fix to
  ``carmel.services.evidence.verify_artifact`` (never report an
  unverifiable artifact as verified).

When both ``FAILED`` and ``UNVERIFIABLE`` findings are present on the same
report, the overall outcome is reported as ``FAILED``: a demonstrated
disagreement is a stronger, more actionable signal than "some other check
could not run," and reporting ``FAILED`` never hides the ``UNVERIFIABLE``
findings either -- both lists are always present on the report in full.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText
from carmel.schemas.datasets import (
    Absent,
    CharSpanLocator,
    DatasetEnvelope,
    MeasuredValue,
    SourceNode,
    iter_measured_values,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.evidence import artifact_dir, load_artifact_meta

__all__ = [
    "ReplayFinding",
    "ReplayOutcome",
    "ReplayReport",
    "replay_envelope",
    "verify_measured_value_unit",
]


class ReplayOutcome(StrEnum):
    """The three (never two) outcomes a replay check can produce."""

    VERIFIED = "verified"
    """Every applicable check ran, and every one of them passed."""

    FAILED = "failed"
    """At least one check ran and produced a definite disagreement."""

    UNVERIFIABLE = "unverifiable"
    """At least one check could not run at all -- never conflated with
    ``VERIFIED``."""


@dataclass(frozen=True)
class ReplayFinding:
    """One named, specific disagreement or blocked check.

    ``category`` is always :attr:`ReplayOutcome.FAILED` or
    :attr:`ReplayOutcome.UNVERIFIABLE` -- a finding is never itself
    ``VERIFIED`` (a clean check produces no finding at all).
    """

    category: ReplayOutcome
    ref_path: str
    reason: str
    expected: str | None = None
    actual: str | None = None


@dataclass(frozen=True)
class ReplayReport:
    """The structured result of replaying one :class:`DatasetEnvelope`."""

    outcome: ReplayOutcome
    checked_char_spans: int
    findings: tuple[ReplayFinding, ...] = ()

    @property
    def failures(self) -> tuple[ReplayFinding, ...]:
        return tuple(f for f in self.findings if f.category is ReplayOutcome.FAILED)

    @property
    def unverifiable(self) -> tuple[ReplayFinding, ...]:
        return tuple(f for f in self.findings if f.category is ReplayOutcome.UNVERIFIABLE)


def _independently_verify_node_text(workspace_root: Path, node: SourceNode) -> tuple[str | None, ReplayFinding | None]:
    """Re-derive ``node``'s extracted text from the evidence store, from
    bytes, independent of anything the envelope itself carries.

    Mirrors the exact re-read/re-verify shape used elsewhere in this
    codebase (``dataset_producer._load_verified_extracted_text``,
    the test-local ``_independently_verified_text`` this module replaces):
    read ``extracted.json`` bytes off disk, verify them against the digest
    :class:`StoredArtifact` recorded at store time, parse, and only then
    hash the parsed text and compare against the RECORDED
    ``extracted_text_sha256``.

    Returns ``(text, None)`` on success, or ``(None, finding)`` naming
    exactly which step failed and whether that step could not run at all
    (``UNVERIFIABLE``) or ran and disagreed (``FAILED``).
    """
    path = f"source_graph.node({node.node_id!r})"
    extraction = node.extraction
    if isinstance(extraction, Absent):
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"node {node.node_id!r} has no ExtractionBinding (extraction is Absent); "
            "there is nothing recorded to independently re-verify",
        )
    meta = load_artifact_meta(workspace_root, node.sha256)
    if meta is None:
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"no stored artifact under sha256={node.sha256!r} for node {node.node_id!r}; "
            "the evidence this node claims to derive from is missing from the store",
        )
    if meta.extracted_sha256 is None:
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"stored artifact for node {node.node_id!r} (sha256={node.sha256!r}) predates "
            "extracted_sha256 tracking; its extracted.json bytes cannot be independently verified",
        )
    extracted_path = artifact_dir(workspace_root, node.sha256) / "extracted.json"
    try:
        raw_bytes = extracted_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"no readable extracted.json for node {node.node_id!r} (sha256={node.sha256!r}): {exc}",
        )
    actual_extracted_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_extracted_sha256 != meta.extracted_sha256:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"extracted.json bytes on disk for node {node.node_id!r} (sha256={node.sha256!r}) do "
            "not match the digest recorded at store time -- the stored evidence has been tampered "
            "with or corrupted since it was written",
            expected=meta.extracted_sha256,
            actual=actual_extracted_sha256,
        )
    try:
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    except ValueError as exc:
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"verified extracted.json for node {node.node_id!r} does not parse as "
            f"ExtractedText: {exc}",
        )
    actual_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    if actual_text_sha256 != extraction.extracted_text_sha256:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"independently re-read text for node {node.node_id!r} hashes to "
            f"{actual_text_sha256!r}, but the envelope's ExtractionBinding.extracted_text_sha256 "
            f"recorded {extraction.extracted_text_sha256!r} at production time",
            expected=extraction.extracted_text_sha256,
            actual=actual_text_sha256,
        )
    return extracted.text, None


def check_char_spans(
    envelope: DatasetEnvelope,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
) -> tuple[int, list[ReplayFinding]]:
    """Re-slice every reachable :class:`CharSpanLocator` and compare it
    against the recorded quote it is supposed to ground.

    ``text_by_node_id`` supplies the independently re-verified text for each
    node that has one; ``node_problems`` names, per node_id, why a node's
    text could NOT be independently verified (missing evidence, digest
    mismatch, etc) -- every :class:`~carmel.schemas.datasets.SourceRef` that
    points at such a node produces a finding of the same category as the
    node problem, rather than a generic KeyError.

    Walks the SAME two structures the schema itself exposes as the complete
    set of char-span-bearing fields today (``MeasuredValue.value_ref``/
    ``unit_ref`` via :func:`iter_measured_values`, and
    ``AxisDeclaration.label_ref`` via ``envelope.series[*].axes``), then
    cross-checks the count against :func:`iter_source_refs`'s own
    independent, generic count of every ``CharSpanLocator`` reachable in the
    envelope -- so a newly added ref-bearing field that this pairing forgot
    to check surfaces as a loud, named finding instead of a silent gap.
    """
    node_problems = node_problems or {}
    checked = 0
    findings: list[ReplayFinding] = []

    def _check(path: str, node_id: str, locator: object, expected: str) -> None:
        nonlocal checked
        if not isinstance(locator, CharSpanLocator):
            return
        if node_id in node_problems:
            problem = node_problems[node_id]
            findings.append(
                ReplayFinding(category=problem.category, ref_path=path, reason=problem.reason)
            )
            return
        text = text_by_node_id.get(node_id)
        if text is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=path,
                    reason=f"references node_id={node_id!r}, which is not present in "
                    "envelope.source_graph and was never independently checked",
                )
            )
            return
        actual = text[locator.start : locator.end]
        if actual != expected:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason="char-span re-slice mismatch: the independently re-verified evidence "
                    "text no longer contains the recorded quote at the recorded offsets",
                    expected=expected,
                    actual=actual,
                )
            )
            return
        checked += 1

    for path, value in iter_measured_values(envelope):
        _check(f"{path}.value_ref", value.value_ref.node_id, value.value_ref.locator, value.raw_text)
        _check(f"{path}.unit_ref", value.unit_ref.node_id, value.unit_ref.locator, value.unit_raw)

    for series in envelope.series:
        for axis in series.axes:
            axis_path = f"series[{series.series_id!r}].axes[{axis.axis_id!r}].label_ref"
            _check(axis_path, axis.label_ref.node_id, axis.label_ref.locator, axis.label_raw)

    total_char_span_refs = sum(
        1 for _, ref in iter_source_refs(envelope) if isinstance(ref.locator, CharSpanLocator)
    )
    accounted_for = checked + len(findings)
    if accounted_for != total_char_span_refs:
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path="<iter_source_refs walk>",
                reason=f"replayer accounted for {accounted_for} char-span ref(s) but "
                f"iter_source_refs finds {total_char_span_refs} reachable in the envelope; some "
                "CharSpanLocator is reachable that this replayer's field-by-field pairing never "
                "checked -- a newly added ref-bearing field was likely added without updating "
                "check_char_spans",
            )
        )

    return checked, findings


def verify_measured_value_unit(path: str, value: MeasuredValue) -> ReplayFinding | None:
    """Re-verify ``value.unit_normalized`` against the conversion table
    ``value.conversion_table_sha256`` RECORDS -- never against whatever
    table is current or preferred today.

    Resolves the table through :func:`carmel.services.units.table_for_sha`
    (the hand-reviewed :data:`~carmel.services.units.TABLES_BY_SHA`
    registry), never through the envelope's own embedded
    :class:`~carmel.schemas.datasets.EmbeddedConversionTable` copy -- see
    this module's own docstring, and ``EmbeddedConversionTable``'s, for why
    that copy is not a trust anchor on its own.

    Returns ``None`` if the recorded unit still normalizes, under the
    recorded table, to exactly the recorded ``unit_normalized``. Otherwise
    returns a :class:`ReplayFinding` naming which of the two distinct
    failure modes occurred: an unresolvable table sha is ``UNVERIFIABLE``
    (the check could not run at all); a resolvable table that disagrees
    with the recorded normalization is ``FAILED`` (the check ran and did
    not pass).
    """
    try:
        table = units.table_for_sha(value.conversion_table_sha256)
    except units.UnknownConversionTableError as exc:
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=f"{path}.conversion_table_sha256",
            reason=f"conversion_table_sha256={value.conversion_table_sha256!r} does not name any "
            f"known conversion table in TABLES_BY_SHA; a MeasuredValue is re-verified against the "
            f"table it RECORDS, never against 'the current table', so this is refused rather than "
            f"silently re-checked against something else: {exc}",
        )
    try:
        expected_unit_normalized = units.normalize_unit(value.quantity_kind, value.unit_raw, table=table)
    except units.UnknownUnitError as exc:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=f"{path}.unit_raw",
            reason=f"unit_raw={value.unit_raw!r} is not a known unit or alias of quantity_kind="
            f"{value.quantity_kind.value!r} in the RECORDED conversion table "
            f"{value.conversion_table_sha256!r}: {exc}",
        )
    if expected_unit_normalized != value.unit_normalized:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=f"{path}.unit_normalized",
            reason=f"unit_normalized={value.unit_normalized!r} disagrees with the recorded table's "
            f"own normalization of unit_raw={value.unit_raw!r} for quantity_kind="
            f"{value.quantity_kind.value!r}, which is {expected_unit_normalized!r}",
            expected=expected_unit_normalized,
            actual=value.unit_normalized,
        )
    return None


def replay_envelope(workspace_root: Path, envelope: DatasetEnvelope) -> ReplayReport:
    """Independently re-verify ``envelope`` against the evidence store
    rooted at ``workspace_root``.

    Combines all three checks this module performs (evidence identity,
    every character span, every measured unit against its recorded table)
    into one :class:`ReplayReport`. Never trusts anything the envelope
    itself carries as pre-verified -- every fact is re-derived from the
    evidence store or from the hand-reviewed
    :data:`~carmel.services.units.TABLES_BY_SHA` registry.
    """
    text_by_node_id: dict[str, str] = {}
    node_problems: dict[str, ReplayFinding] = {}
    for node in envelope.source_graph.nodes:
        text, problem = _independently_verify_node_text(workspace_root, node)
        if problem is not None:
            node_problems[node.node_id] = problem
        else:
            assert text is not None
            text_by_node_id[node.node_id] = text

    checked, span_findings = check_char_spans(envelope, text_by_node_id, node_problems)

    unit_findings: list[ReplayFinding] = []
    for path, value in iter_measured_values(envelope):
        finding = verify_measured_value_unit(path, value)
        if finding is not None:
            unit_findings.append(finding)

    all_findings = tuple(span_findings) + tuple(unit_findings)
    if any(f.category is ReplayOutcome.FAILED for f in all_findings):
        outcome = ReplayOutcome.FAILED
    elif any(f.category is ReplayOutcome.UNVERIFIABLE for f in all_findings):
        outcome = ReplayOutcome.UNVERIFIABLE
    else:
        outcome = ReplayOutcome.VERIFIED

    return ReplayReport(outcome=outcome, checked_char_spans=checked, findings=all_findings)
