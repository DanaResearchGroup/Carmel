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
independent re-derivation. Four things are re-proven, none of them taken on
the envelope's own say-so:

1. **Evidence identity.** For every :class:`~carmel.schemas.datasets.SourceNode`
   that carries an :class:`~carmel.schemas.datasets.ExtractionBinding`, this
   module re-reads ``extracted.json`` directly from the evidence store and
   re-verifies its bytes against ``extracted_sha256`` -- but ONLY the copy
   recorded on the envelope's own
   :class:`~carmel.schemas.datasets.ExtractionBinding`, never the copy in the
   evidence store's ``meta.json`` sidecar. ``meta.json`` lives right next to
   the very file it would otherwise be used to authenticate and is trivially
   rewritable, so it is treated as untrusted input here, not an anchor: it is
   only ever cross-checked AFTER the envelope's own anchor has already
   passed, and a disagreement between the two is itself reported as a
   ``FAILED`` finding rather than silently reconciled. Only once the bytes
   authenticate against the envelope's anchor does this module re-hash the
   parsed text and compare it against the ``extracted_text_sha256`` the same
   ``ExtractionBinding`` recorded. The envelope's own copy of the text (there
   isn't one -- envelopes never embed extracted text) plays no part; this is
   a from-bytes re-derivation.

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

4. **Every unit's boundary/admission, against the RECORDED locator and the
   RECORDED table.** For every reachable ``MeasuredValue``,
   :func:`verify_measured_value_unit_boundary` re-runs the SAME three-layer
   UNIT-role gate :func:`~carmel.services.dataset_producer.ground_quote`
   applies at write time -- Layers 1-2
   (:func:`carmel.services.numeric.unit_boundary_violation`, table-free and
   version-independent) and Layer 3 (table-driven admission/maximality,
   :func:`carmel.services.dataset_producer._unit_table_boundary_violation`)
   -- against the INDEPENDENTLY RE-READ evidence text, sliced at the
   RECORDED ``unit_ref`` locator, and against the binding for the RECORDED
   ``conversion_table_sha256`` (resolved through
   :func:`carmel.services.dataset_producer.binding_for_known_sha`, bounded
   strictly by :data:`carmel.services.units.TABLES_BY_SHA`; an unrecognised
   sha is ``UNVERIFIABLE`` and never falls back to any other table). The
   digit-glue exception's ``value_span`` is recovered from the envelope's OWN
   recorded ``value_ref`` locator when it names the SAME node as ``unit_ref``
   (see :func:`_recover_value_span`); when it does not (or is not itself a
   char-span locator), ``value_span`` is ``None`` -- a unit whose text
   actually needs the glue exception then fails closed with
   ``"unit_digit_glue_no_value_span"`` rather than being silently skipped, so
   an unrecoverable ``value_span`` can still surface a ``FAILED``, never a
   silently-omitted check.

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

A ``VERIFIED`` outcome can never be reached having independently checked
ZERO character spans. A replayer that reports "verified" while never
actually re-slicing anything (an envelope with no char-span refs at all, or
one where every ref happened to be unreachable) would launder the
guarantee this module exists to provide, so ``checked_char_spans == 0``
always forces ``UNVERIFIABLE`` (never ``FAILED`` -- there is no
disagreement, only nothing checked) via an explicit finding naming that
fact. ``ReplayReport`` also carries ``total_char_spans`` and
``unchecked_char_spans`` alongside ``checked_char_spans`` so a MIXED
envelope (some refs checked, some not) reports its partiality honestly
instead of collapsing to one outcome that overstates or understates how
much was actually verified.

Every :class:`~carmel.schemas.datasets.SourceNode` in the envelope's source
graph is independently re-verified, and a problem on a node (missing
evidence, a tampered digest, an unparseable ``extracted.json``) is reported
as a finding EVEN IF no char-span ref happens to reach that node -- an
unreferenced or ancestor node's problem is never silently invisible.

By default, ``expected``/``actual`` on a :class:`ReplayFinding` never carry
raw excerpts of paper text: the project's real corpus is closed-access,
non-redistributable material, and a replayer that prints that text into CI
logs or bug reports is itself a problem. The default is a non-reproducing
discriminator (length and a short digest, plus the byte offsets for the
re-sliced side) with any literal excerpt gated behind an explicit
``reveal_text=True`` opt-in that callers must ask for. ``unit_raw`` values
are the one exception left literal unconditionally: they are short,
controlled-vocabulary unit symbols and aliases (e.g. ``"K"``, ``"atm"``,
``"-"``), never free excerpted prose, so there is nothing to redact.

Two entry points are exported. :func:`replay_envelope` takes an
already-loaded :class:`DatasetEnvelope` OBJECT -- useful for tests that
construct adversarial or partially tampered envelopes without a round trip
through the store, but it only ever proves "this in-memory object verifies
against the evidence store," not "what is actually stored at this address
verifies." :func:`replay_stored_dataset` is the honest, operator-facing API:
it takes a dataset's content address (``sha256``) and the datasets store
root, loads the envelope through
:func:`carmel.services.dataset_bridge.load_dataset_envelope` itself, and
replays THAT -- so it proves a claim about what is actually on disk, not
about whatever the caller happened to construct.

**What boundary/admission re-checking now proves, and what it still does
not.** :func:`verify_measured_value_unit_boundary` re-runs the same
lexical (Layers 1-2) gate :func:`~carmel.services.dataset_producer.ground_quote`
applies at write time, against independently re-read evidence text sliced at
the RECORDED locator; :func:`verify_measured_value_value_boundary` does the
symmetric re-run for VALUE-role boundary/maximality. Both are re-run against
independently re-read evidence text sliced at the RECORDED locator. The
Layer 3 table-admission gate is re-run against the binding for the RECORDED
``conversion_table_sha256`` specifically -- NOT against whatever table is
"current" or "preferred" today. Those are two different claims, and it
matters which one a ``VERIFIED`` result actually makes: a recorded table can
be superseded (e.g. a future ``TABLE_V2`` replacing ``TABLE_V1`` as the
codebase's preferred table) without the RECORDED envelope becoming wrong --
it was admitted against the table it names, and replay re-confirms exactly
that, historical or current. A ``VERIFIED`` boundary/admission result is
therefore NOT the same claim as "``ground_quote`` would admit this exact
quote if run today" -- it is the narrower, and permanently stable, claim
that the recorded locator still slices to the recorded quote under the
RECORDED table's own rules.

Stated plainly, without embellishment: replay proves the stored locators
still slice to the recorded quotes in the named evidence, that unit
normalization matches the recorded known table, and that char-span value
and unit locators pass current lexical checks plus recorded-table admission.
It does NOT prove value/label admission beyond the above, occurrence
reconstruction, M-D4 grouping, or layout truth. In particular:

- **M-D4 measurement grouping** -- that ``value_ref``, ``unit_ref``, and any
  label together describe ONE coherent measurement, rather than three
  independently-true-but-unrelated facts stitched into a
  ``MeasuredValue``. Replay checks each ref's locator and boundary in
  isolation; it has no way to re-derive "these three spans belong
  together" from the evidence text alone.
- **The original caller's occurrence/search disambiguation** -- when a unit
  or value string appears more than once in the source text, replay
  trusts the RECORDED locator's offsets; it cannot reconstruct why the
  producer chose that occurrence over another one.
- **A faithful recording of a wrong extraction** -- boundary/admission only
  asks "would this exact span still be admitted against the recorded
  table." It cannot detect that the span, while internally well-formed and
  admissible, was extracted from the wrong place in the source document in
  the first place.
- **Layout truth** -- replay never re-derives page geometry, reading order,
  or table structure; it only re-slices character offsets in already-
  extracted text.

Read a ``VERIFIED`` result accordingly: it proves the recorded facts are
genuinely grounded, internally consistent with the table they recorded, and
pass current lexical boundary checks plus RECORDED-table admission -- not
that they constitute a correctly-grouped, correctly-disambiguated, or
correctly-targeted measurement, and not that today's policy (as opposed to
the recorded table's policy) would admit them.
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
from carmel.services.dataset_bridge import load_dataset_envelope
from carmel.services.dataset_producer import (
    _unit_table_boundary_violation,
    binding_for_known_sha,
)
from carmel.services.evidence import _derivation_binding, artifact_dir, load_artifact_meta
from carmel.services.numeric import (
    NUMERAL_CANDIDATE_RE,
    enclosing_numeric_construct,
    find_numeral_extent,
    has_clean_token_boundary,
    unit_boundary_violation,
)
from carmel.services.units import QuantityKind

__all__ = [
    "ReplayFinding",
    "ReplayOutcome",
    "ReplayReport",
    "replay_envelope",
    "replay_stored_dataset",
    "verify_measured_value_unit",
    "verify_measured_value_unit_boundary",
    "verify_measured_value_value_boundary",
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
    """The structured result of replaying one :class:`DatasetEnvelope`.

    ``checked_char_spans`` is how many :class:`CharSpanLocator`\\ s were
    independently re-sliced and matched their recorded quote.
    ``total_char_spans`` is how many are reachable in the envelope at all
    (via :func:`~carmel.schemas.datasets.iter_source_refs`), and
    ``unchecked_char_spans`` is the difference -- refs that produced a
    finding (a mismatch, a missing node, an unreadable evidence file)
    rather than a clean check. These three numbers make a MIXED result
    (some spans checked, some not) legible instead of collapsing to one
    outcome that overstates or understates how much was actually verified.
    """

    outcome: ReplayOutcome
    checked_char_spans: int
    total_char_spans: int
    unchecked_char_spans: int
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
    raw_path = artifact_dir(workspace_root, node.sha256) / "raw.bin"
    try:
        raw_bin_bytes = raw_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        # Absence is inability to check: raw.bin may simply have been
        # garbage-collected or never fetched into this workspace, which says
        # nothing about whether the node's identity is trustworthy -- report
        # UNVERIFIABLE, not FAILED. A raw.bin that IS present but hashes to
        # something other than node.sha256 (below) is different in kind: it
        # is positive evidence that the store's raw bytes were tampered with
        # or corrupted, so that case is reported as FAILED instead. The two
        # must never be conflated into a single outcome.
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"no readable raw.bin for node {node.node_id!r} (sha256={node.sha256!r}): {exc}",
        )
    actual_raw_sha256 = hashlib.sha256(raw_bin_bytes).hexdigest()
    if actual_raw_sha256 != node.sha256:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"raw.bin on disk for node {node.node_id!r} hashes to {actual_raw_sha256!r}, not "
            f"the node.sha256={node.sha256!r} it is stored under -- the evidence store's raw bytes "
            "have been tampered with or corrupted since this node was recorded",
            expected=node.sha256,
            actual=actual_raw_sha256,
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
    # The ONLY acceptable anchor is the envelope's own ExtractionBinding --
    # never the evidence store's meta.json sidecar. meta.json lives right
    # next to the very file it would otherwise authenticate and is
    # trivially rewritable by anyone who can write to the store; trusting
    # it as an anchor lets a forger rewrite extracted.json and meta.json
    # together and still get a VERIFIED result. See this module's
    # docstring ("Evidence identity").
    if actual_extracted_sha256 != extraction.extracted_sha256:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"extracted.json bytes on disk for node {node.node_id!r} (sha256={node.sha256!r}) do "
            "not match ExtractionBinding.extracted_sha256 recorded on the envelope -- the stored "
            "evidence has been tampered with or corrupted since the envelope was produced",
            expected=extraction.extracted_sha256,
            actual=actual_extracted_sha256,
        )
    # meta.json is cross-checked only AFTER the envelope's own anchor has
    # already passed, and purely as an early-warning signal: a
    # disagreement here means the evidence store's own bookkeeping is
    # inconsistent with the envelope (rewritten independently of it, or
    # never updated), which is itself worth reporting even though the
    # envelope-anchored check above is what actually decided pass/fail.
    meta = load_artifact_meta(workspace_root, node.sha256)
    if meta is not None and meta.extracted_sha256 is not None and meta.extracted_sha256 != actual_extracted_sha256:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"evidence store meta.json for node {node.node_id!r} (sha256={node.sha256!r}) "
            f"records extracted_sha256={meta.extracted_sha256!r}, which disagrees with the bytes "
            "actually on disk (and with the envelope's own anchor, already verified above) -- the "
            "store's own sidecar bookkeeping is inconsistent and is reported rather than silently "
            "reconciled",
            expected=actual_extracted_sha256,
            actual=meta.extracted_sha256,
        )
    try:
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
    except (ValueError, RecursionError) as exc:
        # PEP 758 (Python 3.14) permits an unparenthesized multi-exception
        # `except A, B:`, but only without an `as` binding -- with `as`,
        # the grammar still requires parentheses around the tuple, which is
        # what this clause uses. RecursionError is caught explicitly (never
        # a bare `except Exception`, which would also swallow
        # KeyboardInterrupt/SystemExit-adjacent BaseException-only escapes
        # it shouldn't): a
        # verified-by-digest extracted.json can still be pathologically
        # deep enough to blow the interpreter's recursion limit during
        # json.loads or pydantic validation, and that must become a named
        # UNVERIFIABLE finding, never an escaping crash.
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"verified extracted.json for node {node.node_id!r} does not parse as "
            f"ExtractedText: {exc!r}",
        )
    if extracted.lossy:
        # Mirrors dataset_producer.produce_envelope_from_artifact's own
        # refusal to produce an envelope from a knowingly-partial
        # extraction: `extracted.text` here is by definition a partial view
        # of the document (missing pages, a parse failure, or truncation),
        # so no char-span re-slice or grounded quote drawn from it can be
        # trusted to represent the actual document, no matter how cleanly
        # its digests line up. Report UNVERIFIABLE rather than silently
        # replaying partial text as though it were a faithful, complete
        # re-derivation.
        page_note = (
            f" ({len(extracted.page_failures)} page(s) failed to extract)" if extracted.page_failures else ""
        )
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"verified extracted.json for node {node.node_id!r} is a lossy extraction "
            f"(extractor={extracted.extractor!r}){page_note}; a knowingly-partial extraction cannot "
            "be independently re-verified as a faithful re-derivation of the document text",
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
    if isinstance(extraction.derivation_binding, Absent):
        # A legacy record predating ExtractionBinding.derivation_binding (or
        # one explicitly stripped) carries nothing to cross-check the
        # extractor identity against. This is inability-to-check, not
        # positive evidence of anything wrong -- UNVERIFIABLE, not FAILED.
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"node {node.node_id!r}'s ExtractionBinding carries no derivation_binding "
            "(Absent, e.g. a legacy record predating that field); there is nothing recorded to "
            "independently cross-check the extractor identity binding against",
        )
    if meta is None or meta.extractor_version is None or meta.extracted_sha256 is None:
        return None, ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path,
            reason=f"evidence store meta.json for node {node.node_id!r} (sha256={node.sha256!r}) is "
            "missing, or predates extractor_version/extracted_sha256, so the envelope's carried "
            "derivation_binding cannot be independently recomputed and cross-checked",
        )
    # Recompute using meta.json's OWN FROZEN extractor_version -- the
    # identity string `_extractor_identity()` computed and stored ONCE, at
    # store time -- rather than calling `_extractor_identity()` live here.
    # A live call would fold in whatever pypdf version happens to be
    # installed on the machine doing the replaying right now; pypdf is
    # unpinned in this project (`pypdf>=5.0`), so that version string can
    # legitimately differ from what was installed when the envelope was
    # produced. Requiring live-recomputed bit-for-bit equality would fail
    # closed on a routine pypdf upgrade -- indistinguishable from tampering
    # -- and would also fail every legacy artifact stored with
    # extractor_version=None. Reusing the frozen meta.extractor_version
    # avoids both traps, at the cost of only proving what
    # ExtractionBinding.derivation_binding's own docstring says it proves:
    # internal consistency between the envelope's carried binding and the
    # store's current record -- NOT that extracted.json was actually
    # re-derived from raw.bin, and no defence against a forger who updates
    # all three fields together. See evidence._derivation_binding_intact,
    # which this mirrors but against the envelope's carried value instead
    # of only meta.json's own internal fields.
    recomputed_binding = _derivation_binding(meta.sha256, meta.extracted_sha256, meta.extractor_version)
    if recomputed_binding != extraction.derivation_binding:
        return None, ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"node {node.node_id!r}'s ExtractionBinding.derivation_binding="
            f"{extraction.derivation_binding!r} does not match sha256(extractor_identity|raw_sha256|"
            f"extracted_sha256) recomputed from evidence store meta.json ({recomputed_binding!r}) -- "
            "the envelope's carried binding disagrees with the store's own record",
            expected=recomputed_binding,
            actual=extraction.derivation_binding,
        )
    return extracted.text, None


def _redacted(text: str, locator: CharSpanLocator | None = None) -> str:
    """A non-reproducing discriminator for a snippet of paper text.

    Never returns the literal text: the real corpus this module replays
    against is closed-access, non-redistributable material, and a finding
    that prints raw excerpts into CI logs or bug reports would itself be a
    problem. Length plus a short digest is enough to tell two DIFFERENT
    snippets apart (which is all a finding needs to do) without ever
    reproducing either one.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    descriptor = f"<redacted len={len(text)} sha256_16={digest}"
    if locator is not None:
        descriptor += f" offsets=[{locator.start}:{locator.end}]"
    return descriptor + ">"


def check_char_spans(
    envelope: DatasetEnvelope,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
    *,
    reveal_text: bool = False,
) -> tuple[int, int, list[ReplayFinding]]:
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

    By default, a mismatch finding's ``expected``/``actual`` are redacted
    (see :func:`_redacted`) rather than literal, EXCEPT for the ``unit_ref``
    call site: ``unit_raw`` is short, controlled-vocabulary unit
    symbols/aliases (e.g. ``"K"``, ``"atm"``), never excerpted prose, so
    there is nothing there to protect and leaving it literal makes unit
    mismatches far easier to read. Pass ``reveal_text=True`` to opt into
    literal excerpts everywhere (e.g. for a human debugging a specific
    finding with the corpus license already accounted for).

    Returns ``(checked, total_char_span_refs, findings)``.
    """
    node_problems = node_problems or {}
    checked = 0
    findings: list[ReplayFinding] = []

    def _check(
        path: str, node_id: str, locator: object, expected: str, *, always_literal: bool = False
    ) -> None:
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
            literal = always_literal or reveal_text
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason="char-span re-slice mismatch: the independently re-verified evidence "
                    "text no longer contains the recorded quote at the recorded offsets",
                    expected=expected if literal else _redacted(expected),
                    actual=actual if literal else _redacted(actual, locator),
                )
            )
            return
        checked += 1

    for path, value in iter_measured_values(envelope):
        _check(f"{path}.value_ref", value.value_ref.node_id, value.value_ref.locator, value.raw_text)
        # unit_raw is short, controlled-vocabulary text (unit symbols and
        # aliases) -- never excerpted prose -- so it is always left literal
        # regardless of reveal_text; see this function's own docstring.
        _check(
            f"{path}.unit_ref",
            value.unit_ref.node_id,
            value.unit_ref.locator,
            value.unit_raw,
            always_literal=True,
        )

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

    return checked, total_char_span_refs, findings


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


def _recover_value_span(value: MeasuredValue, unit_node_id: str) -> tuple[int, int] | None:
    """Recover the ``value_span`` Layer 2's digit-glue exception
    (:func:`carmel.services.numeric.unit_boundary_violation`) needs, from
    the envelope's OWN recorded ``value_ref`` locator -- never fabricated.

    Returns ``None`` (never a guessed span) whenever ``value_ref`` cannot
    honestly stand in for ``value_span`` in the UNIT's re-read text: when it
    does not point at the SAME node as ``unit_ref`` (the digit-glue
    exception is about a digit run immediately adjacent to the unit IN THAT
    TEXT; a value grounded in a different node's text cannot supply that),
    or when its locator is not itself a :class:`CharSpanLocator`. This is
    deliberately fail-closed rather than verified-by-omission: when
    ``None`` is returned but the unit's re-read text actually has a
    digit-glue shape that needs a value_span,
    :func:`~carmel.services.numeric.unit_boundary_violation` itself refuses
    with ``"unit_digit_glue_no_value_span"`` -- the check still runs and can
    still FAIL, it just cannot use the narrow glue exception.
    """
    if value.value_ref.node_id != unit_node_id:
        return None
    locator = value.value_ref.locator
    if not isinstance(locator, CharSpanLocator):
        return None
    return (locator.start, locator.end)


def verify_measured_value_unit_boundary(
    path: str,
    value: MeasuredValue,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
) -> ReplayFinding | None:
    """Re-run the UNIT-role boundary/admission gate (D-U2,
    :func:`carmel.services.dataset_producer.ground_quote`'s own three-layer
    rule) against the INDEPENDENTLY RE-READ evidence text, at the RECORDED
    ``unit_ref`` locator -- never against the envelope's own say-so that the
    unit was admitted.

    This closes the gap this module's docstring used to describe as
    deliberate and open: until this function existed, replay re-verified
    evidence identity, every character
    span, and unit NORMALIZATION against the recorded table, but never
    re-ran boundary/admission -- so an envelope that RECORDED
    ``unit_raw="bar"`` with a unit locator pointing inside evidence text
    ``"1 bar(a)"`` replayed ``VERIFIED`` even though
    :func:`~carmel.services.dataset_producer.ground_quote` would refuse that
    exact quote at write time. A faithfully-recorded locator into
    genuinely-corrupted evidence text is exactly what this function is for.

    Layers 1-2 (:func:`carmel.services.numeric.unit_boundary_violation`,
    table-free and version-independent) run first, then Layer 3 (the
    table-driven admission/maximality check,
    :func:`carmel.services.dataset_producer._unit_table_boundary_violation`)
    against the binding for the envelope's OWN RECORDED
    ``conversion_table_sha256`` -- resolved through
    :func:`carmel.services.dataset_producer.binding_for_known_sha`, which is
    bounded strictly by the hand-reviewed
    :data:`carmel.services.units.TABLES_BY_SHA` registry and NEVER derived
    from a caller-supplied table. An unrecognised sha makes Layer 3
    ``UNVERIFIABLE`` (it never falls back to whatever table is current).
    A ``unit_ref.locator`` that is not a :class:`CharSpanLocator`, and a
    ``quantity_kind`` of :attr:`~carmel.services.units.QuantityKind.OTHER`
    (which has no admission vocabulary at all -- see
    :meth:`carmel.services.units._ActiveTableBinding.derive`), are each
    reported ``UNVERIFIABLE`` rather than silently skipped or falsely
    accused: neither case gives this gate anything honest to check.

    What this does NOT prove, even when it returns ``None``: that the value,
    unit, and label spans of one measurement genuinely belong together (see
    this module's own docstring, M-D4 grouping); that the original writer's
    occurrence/search disambiguation among multiple candidate matches was
    the right one; or that a faithfully-recorded locator was pointed at the
    CORRECT extraction rather than merely a clean one. It proves only that
    the recorded locator, re-read from independently-verified evidence text,
    still passes the SAME lexical checks ``ground_quote`` applies today PLUS
    admission against the table the envelope itself RECORDED -- not
    necessarily whatever table is current or preferred today. A recorded
    table can be superseded by a later one without the recorded envelope
    becoming wrong; this gate re-confirms admission against the table named
    in the envelope, which is a stable, historical claim, not a claim about
    today's policy.

    Returns ``None`` if both layers pass cleanly. Otherwise a
    :class:`ReplayFinding`: ``FAILED`` if a layer ran and refused;
    ``UNVERIFIABLE`` if a layer could not run at all (missing/unreadable
    node text, an unresolvable ``conversion_table_sha256``, an unmodelled
    ``QuantityKind.OTHER`` quantity, a non-``CharSpanLocator``, or an
    out-of-range locator). A skipped check never yields ``None`` silently --
    every early return above is paired with a named finding.
    """
    node_problems = node_problems or {}
    unit_node_id = value.unit_ref.node_id
    unit_locator = value.unit_ref.locator
    path_ref = f"{path}.unit_ref"

    if not isinstance(unit_locator, CharSpanLocator):
        # Every other SourceLocator kind (bbox, table cell, xpath) has no
        # character span for this lexical/table gate to re-run against.
        # check_char_spans already re-slices those kinds on their own terms,
        # but THIS gate -- boundary/admission -- simply cannot run against a
        # locator that carries no character span. Reporting UNVERIFIABLE
        # (never a silent None) keeps "not checked" from ever reading as
        # "checked and clean": a skipped check must never verify.
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason=f"unit_ref.locator is a {type(unit_locator).__name__}, not a CharSpanLocator -- "
            "the boundary/admission gate re-runs the same character-span lexical and table checks "
            "ground_quote applies at write time, so it has nothing to re-run against a locator that "
            "carries no character span; this is reported as unverified, not silently skipped",
        )

    if unit_node_id in node_problems:
        problem = node_problems[unit_node_id]
        return ReplayFinding(category=problem.category, ref_path=path_ref, reason=problem.reason)

    text = text_by_node_id.get(unit_node_id)
    if text is None:
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason=f"references node_id={unit_node_id!r}, which is not present in "
            "envelope.source_graph and was never independently checked, so the unit "
            "boundary/admission gate cannot be re-run",
        )

    start, end = unit_locator.start, unit_locator.end
    if not (0 <= start < end <= len(text)):
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path_ref,
            reason=f"unit_ref locator [{start}:{end}] is out of range for the independently "
            f"re-verified text of node_id={unit_node_id!r} (len={len(text)}) -- cannot re-slice "
            "to re-run the boundary/admission gate",
        )

    value_span = _recover_value_span(value, unit_node_id)

    lexical_violation = unit_boundary_violation(text, start, end, value_span=value_span)
    if lexical_violation is not None:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path_ref,
            reason="unit boundary re-check (Layers 1-2, table-free) failed against re-read "
            f"evidence text: {lexical_violation!r} -- the recorded unit locator no longer grounds "
            "a clean UNIT-role boundary; ground_quote would refuse this exact quote today",
        )

    if value.quantity_kind is QuantityKind.OTHER:
        # QuantityKind.OTHER is a deliberate wildcard for a quantity this
        # codebase does not model at all: _ActiveTableBinding.derive()
        # excludes it from spellings_by_quantity, so the Layer 3 admission
        # lookup below always falls back to the empty-set default and
        # therefore ALWAYS reports "unit_not_in_vocabulary" for OTHER,
        # regardless of what the recorded text actually says. That is a
        # false accusation, not a genuine admission failure -- OTHER has no
        # admission vocabulary to check against, so replay cannot honestly
        # say the recorded unit disagrees with one. Report the quantity as
        # unmodelled, not wrong.
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason="quantity_kind=QuantityKind.OTHER has no unit admission vocabulary -- "
            "_ActiveTableBinding.derive() deliberately excludes OTHER from spellings_by_quantity, so "
            "the table-driven admission/maximality gate (Layer 3) cannot honestly check this unit "
            "against any recorded vocabulary; this quantity is unmodelled, not disagreeing",
        )

    binding = binding_for_known_sha(value.conversion_table_sha256)
    if binding is None:
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=f"{path}.conversion_table_sha256",
            reason=f"conversion_table_sha256={value.conversion_table_sha256!r} does not name any "
            "known conversion table in TABLES_BY_SHA -- the table-driven admission/maximality gate "
            "(Layer 3) is re-run against the RECORDED table only, never against 'the current "
            "table', so this is refused rather than silently re-checked against something else",
        )

    admission_violation = _unit_table_boundary_violation(text, start, end, value.quantity_kind, binding)
    if admission_violation is not None:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path_ref,
            reason="unit table admission/maximality re-check (Layer 3) failed against re-read "
            f"evidence text: {admission_violation!r} -- the recorded unit is no longer admitted "
            f"(or no longer maximal) against the RECORDED table {value.conversion_table_sha256!r}",
        )

    return None


def _value_boundary_violation(text: str, start: int, end: int) -> str | None:
    """Re-run VALUE-role boundary/maximality (the numeral half of D-U2) over
    ``text[start:end]``, mirroring
    :func:`carmel.services.dataset_producer.ground_quote`'s ``role=QuoteRole.VALUE``
    branch exactly, but RETURNING a discriminant string instead of raising
    :class:`~carmel.services.dataset_producer.QuoteGroundingError` -- the same
    return-a-discriminant-or-None convention
    :func:`carmel.services.numeric.unit_boundary_violation` and
    :func:`carmel.services.numeric.label_boundary_violation` already use, kept
    private to this module the same way
    :func:`carmel.services.dataset_producer._unit_table_boundary_violation` is
    kept private to ``dataset_producer.py``.

    Built entirely from the same public :mod:`carmel.services.numeric`
    primitives ``ground_quote`` itself uses for VALUE role
    (:data:`~carmel.services.numeric.NUMERAL_CANDIDATE_RE`,
    :func:`~carmel.services.numeric.find_numeral_extent`,
    :func:`~carmel.services.numeric.enclosing_numeric_construct`,
    :func:`~carmel.services.numeric.has_clean_token_boundary`) -- this
    function re-derives no new policy, it only re-runs the existing one.

    Returns ``None`` when the span is a clean, maximal VALUE-role quote;
    otherwise a short discriminant string naming which check failed.
    """
    quote = text[start:end]
    if NUMERAL_CANDIDATE_RE.fullmatch(quote):
        extent = find_numeral_extent(text, start)
        if extent is None:
            return "value_no_clean_numeral_extent"
        if extent != (start, end):
            return "value_interior_numeral_fragment"
        construct = enclosing_numeric_construct(text, start, end)
        if construct == "ascii6_uncertainty":
            return "value_ascii6_uncertainty_fragment"
        if construct == "spaced_range":
            return "value_spaced_range_fragment"
        if construct == "flattened_scientific":
            return "value_flattened_scientific_fragment"
        return None
    if not has_clean_token_boundary(text, start, end):
        return "value_not_clean_token_boundary"
    return None


def verify_measured_value_value_boundary(
    path: str,
    value: MeasuredValue,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
) -> ReplayFinding | None:
    """Re-run VALUE-role boundary/maximality re-checking (the numeral half of
    D-U2) against the INDEPENDENTLY RE-READ evidence text sliced at the
    RECORDED ``value_ref`` locator -- symmetric with
    :func:`verify_measured_value_unit_boundary`'s UNIT-role re-check.

    :func:`check_char_spans` only re-slices ``value_ref`` and string-compares
    it against the recorded ``raw_text``; it never re-runs VALUE-role
    maximality. That means a ``value_ref`` moved to an interior fragment of a
    larger numeral (e.g. pointing at ``"1023"`` inside ``"11023"``) still
    replays ``VERIFIED`` under ``check_char_spans`` alone, because the
    re-sliced substring still equals the recorded quote character-for-character
    -- the re-slice check has no way to see that the recorded span is no
    longer a MAXIMAL, boundary-clean numeral. This function closes that gap
    the same way :func:`verify_measured_value_unit_boundary` closed the
    analogous UNIT-role gap.

    Returns ``None`` when the recorded ``value_ref`` cleanly re-verifies;
    otherwise a :class:`ReplayFinding`: ``FAILED`` for a genuine boundary
    violation (the re-read text no longer grounds the recorded span as a
    clean, maximal VALUE-role quote), ``UNVERIFIABLE`` for anything this gate
    could not run against (a non-:class:`CharSpanLocator` locator, missing or
    unreadable node text, an out-of-range locator) -- a skipped check never
    yields ``None`` unpaired with a finding.
    """
    node_problems = node_problems or {}
    value_node_id = value.value_ref.node_id
    value_locator = value.value_ref.locator
    path_ref = f"{path}.value_ref"

    if not isinstance(value_locator, CharSpanLocator):
        # Every other SourceLocator kind carries no character span for this
        # gate to re-run against; check_char_spans already re-slices those
        # kinds on their own terms. Reported as UNVERIFIABLE, never a silent
        # None, so a skipped check never reads as clean.
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason=f"value_ref.locator is a {type(value_locator).__name__}, not a CharSpanLocator -- "
            "the VALUE-role boundary/maximality gate has nothing to re-run against a locator that "
            "carries no character span; this is reported as unverified, not silently skipped",
        )

    if value_node_id in node_problems:
        problem = node_problems[value_node_id]
        return ReplayFinding(category=problem.category, ref_path=path_ref, reason=problem.reason)

    text = text_by_node_id.get(value_node_id)
    if text is None:
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason=f"references node_id={value_node_id!r}, which is not present in "
            "envelope.source_graph and was never independently checked, so the VALUE-role "
            "boundary/maximality gate cannot be re-run",
        )

    start, end = value_locator.start, value_locator.end
    if not (0 <= start < end <= len(text)):
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path_ref,
            reason=f"value_ref locator [{start}:{end}] is out of range for the independently "
            f"re-verified text of node_id={value_node_id!r} (len={len(text)}) -- cannot re-slice "
            "to re-run the boundary/maximality gate",
        )

    violation = _value_boundary_violation(text, start, end)
    if violation is not None:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path_ref,
            reason="value boundary/maximality re-check failed against re-read evidence text: "
            f"{violation!r} -- the recorded value locator no longer grounds a clean, maximal "
            "VALUE-role quote; ground_quote would refuse this exact quote today",
        )

    return None


def replay_envelope(workspace_root: Path, envelope: DatasetEnvelope, *, reveal_text: bool = False) -> ReplayReport:
    """Independently re-verify an already-loaded ``envelope`` OBJECT against
    the evidence store rooted at ``workspace_root``.

    Combines all four checks this module performs (evidence identity,
    every character span, every measured unit's normalization against its
    recorded table, and every measured unit's boundary/admission against
    its recorded locator and recorded table) into one :class:`ReplayReport`.
    Never trusts anything the envelope
    itself carries as pre-verified -- every fact is re-derived from the
    evidence store or from the hand-reviewed
    :data:`~carmel.services.units.TABLES_BY_SHA` registry.

    This proves "the envelope OBJECT handed to this call verifies against
    the evidence store" -- useful for tests that construct adversarial or
    partially tampered envelopes in memory without a round trip through the
    dataset store. It does NOT prove anything about what is actually stored
    at a given content address; for that, use :func:`replay_stored_dataset`.

    Set ``reveal_text=True`` to opt into literal (non-redacted) excerpts on
    mismatch findings -- see :func:`check_char_spans`'s docstring; default
    is redacted.
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

    checked, total_char_spans, span_findings = check_char_spans(
        envelope, text_by_node_id, node_problems, reveal_text=reveal_text
    )

    unit_findings: list[ReplayFinding] = []
    for path, value in iter_measured_values(envelope):
        finding = verify_measured_value_unit(path, value)
        if finding is not None:
            unit_findings.append(finding)
        boundary_finding = verify_measured_value_unit_boundary(path, value, text_by_node_id, node_problems)
        if boundary_finding is not None:
            unit_findings.append(boundary_finding)
        value_boundary_finding = verify_measured_value_value_boundary(path, value, text_by_node_id, node_problems)
        if value_boundary_finding is not None:
            unit_findings.append(value_boundary_finding)

    # Every node's own problem is surfaced regardless of whether any
    # char-span ref happens to reach it -- an unreferenced node (or one
    # only reachable via parent_node_id, never a SourceRef) must never be
    # silently invisible just because check_char_spans never had a reason
    # to look at it.
    node_level_findings = tuple(node_problems.values())

    all_findings = tuple(span_findings) + tuple(unit_findings) + node_level_findings

    # A replay that independently re-sliced ZERO character spans must never
    # report VERIFIED: that would launder the "verified" label onto an
    # envelope where nothing about its grounding was actually checked
    # (either because it has no char-span refs at all, or because every
    # reachable ref turned into some other finding instead of a clean
    # check). This is UNVERIFIABLE, not FAILED -- there is no demonstrated
    # disagreement, only nothing to point to as having passed.
    if checked == 0:
        all_findings = all_findings + (
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path="<check_char_spans>",
                reason="independently re-sliced ZERO character spans (total_char_spans="
                f"{total_char_spans}); a replay cannot report VERIFIED without ever having checked "
                "a single char-span",
            ),
        )

    if any(f.category is ReplayOutcome.FAILED for f in all_findings):
        outcome = ReplayOutcome.FAILED
    elif any(f.category is ReplayOutcome.UNVERIFIABLE for f in all_findings):
        outcome = ReplayOutcome.UNVERIFIABLE
    else:
        outcome = ReplayOutcome.VERIFIED

    return ReplayReport(
        outcome=outcome,
        checked_char_spans=checked,
        total_char_spans=total_char_spans,
        unchecked_char_spans=total_char_spans - checked,
        findings=all_findings,
    )


def replay_stored_dataset(workspace_root: Path, sha256: str, *, reveal_text: bool = False) -> ReplayReport:
    """Independently re-verify the dataset actually STORED under ``sha256``
    in the dataset store rooted at ``workspace_root``.

    This is the honest, operator-facing entry point: unlike
    :func:`replay_envelope` (which trusts whatever envelope OBJECT the
    caller hands it), this function loads the envelope itself, through
    :func:`carmel.services.dataset_bridge.load_dataset_envelope` -- the same
    round trip any real reader of the store goes through, which itself
    proves the stored bytes reconstruct byte-for-byte before replay even
    starts -- and then replays THAT. A ``VERIFIED`` result from this
    function is a claim about what is actually on disk at ``sha256``, not
    about whatever an in-memory object happened to claim.
    """
    envelope = load_dataset_envelope(workspace_root, sha256)
    return replay_envelope(workspace_root, envelope, reveal_text=reveal_text)
