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
NOTHING. A replayer that reports "verified" while never actually verifying
anything (an envelope with no checkable grounding at all, or one where every
ref happened to be unreachable) would launder the guarantee this module
exists to provide, so a report that checked nothing always forces
``UNVERIFIABLE`` (never ``FAILED`` -- there is no disagreement, only nothing
checked) via an explicit finding naming that fact. "Checked nothing" is
WIDER than "re-sliced zero character spans": a table-cell text comparison
that ran and MATCHED is also a real check (counted as
``ReplayReport.checked_table_cells``, a quantity deliberately separate from
the char-span counts), so the guard forces ``UNVERIFIABLE`` only when
``checked_char_spans == 0`` AND ``checked_table_cells == 0`` -- an artifact
grounded entirely at table cells, every cell matching, re-slices zero char
spans yet has verified something real and reads ``VERIFIED``.
``ReplayReport`` also carries ``total_char_spans`` and
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
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from carmel.agents.tools.extract import ExtractedText
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    CharSpanLocator,
    ConditionSetEnvelope,
    DatasetEnvelope,
    DeviceClassDeclaration,
    EmbeddedOoxmlTableInventory,
    EmbeddedTableInventory,
    MeasuredValue,
    RootSidecarVerification,
    Series,
    SourceForm,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TableCellLocator,
    Uncertainty,
    UncertaintyKind,
    UnitProvenance,
    UnresolvedSubject,
    _iter_series_point_value_refs,
    iter_measured_values,
    iter_source_refs,
    iter_uncertainties,
)
from carmel.schemas.literature import StoredArtifact
from carmel.services import units
from carmel.services.condition_set_bridge import load_condition_set_envelope
from carmel.services.dataset_bridge import load_dataset_envelope
from carmel.services.dataset_producer import (
    _unit_table_boundary_violation,
    binding_for_known_sha,
)
from carmel.services.evidence import artifact_dir, load_artifact_meta
from carmel.services.extraction_record import (
    ExtractionRecordError,
    extraction_record_dir,
    load_extraction_record,
)
from carmel.services.figure_digitization_record import (
    UNREADABLE_PAYLOAD,
    FigureDigitization,
    UnknownPayloadVersion,
    compute_digitization_sha,
)
from carmel.services.numeric import (
    NUMERAL_CANDIDATE_RE,
    enclosing_numeric_construct,
    find_numeral_extent,
    has_clean_token_boundary,
    unit_boundary_violation,
)
from carmel.services.ooxml_table_record import (
    OoxmlInventoryVerificationStatus,
    verify_ooxml_inventory_record,
)
from carmel.services.pdf_table_record import (
    InventoryVerificationStatus,
    verify_inventory_record,
)
from carmel.services.stitching import (
    StitchGateUnrunnable,
    StitchRefutation,
    refute_stitched_claim,
)
from carmel.services.units import QuantityKind

__all__ = [
    "AttemptedRefutation",
    "RefutationStatus",
    "ReplayFinding",
    "ReplayOutcome",
    "ReplayReport",
    "SemanticGap",
    "UncheckedSemanticClaim",
    "UncheckedStoreClaim",
    "replay_condition_set",
    "replay_envelope",
    "replay_stored_condition_set",
    "replay_stored_dataset",
    "verify_measured_value_unit",
    "verify_measured_value_unit_boundary",
    "verify_measured_value_value_boundary",
]


class SemanticGap(StrEnum):
    """Which of three states left a semantic claim's support unchecked.

    Not a boolean, on purpose: a consumer must be able to tell "reachable but
    unexplained" from "not even reachable" from "nothing was ever offered"
    without parsing prose.
    """

    SUPPORT_UNRECORDED = "support_unrecorded"
    """A support ref was resolved and its span re-sliced cleanly, but nothing
    recorded anywhere says that span MEANS the derived value. This is the
    ordinary case for a value whose support is a location without a recorded
    quote beside it."""

    LOCATION_UNRESOLVED = "location_unresolved"
    """The support ref's span could not be re-sliced at all -- its locator is
    not a :class:`~carmel.schemas.datasets.CharSpanLocator`, or its node is
    missing, or that node carries a store-level problem. STRICTLY LESS is
    known than :attr:`SUPPORT_UNRECORDED`: there, the span was confirmed to
    exist and only its meaning was untested; here, not even that."""

    NO_SUPPORT_OFFERED = "no_support_offered"
    """The derived value carries no support ref at all, so there is nothing to
    resolve and nothing to re-slice."""


class ReplayOutcome(StrEnum):
    """The three (never two) outcomes a replay check can produce."""

    VERIFIED = "verified"
    """Every applicable check ran, and every one of them passed.

    "Passed" means a check that could have shown something WRONG did not. It
    does not extend to a falsification gate that ran and failed to refute:
    surviving an attack is not a check passing, because that gate has no
    passing state to reach. Those are recorded on
    :attr:`ReplayReport.attempted_refutations`, and any entry there keeps
    :attr:`ReplayReport.overall_outcome` off this member.

    Scope-neutral on purpose: this member no longer carries the caveat, because
    the FIELD now carries the scope. It used to say "every applicable EVIDENCE
    check", with a paragraph explaining that a VERIFIED report might still hold
    provenance claims nothing ever tested -- a caveat attached to the vocabulary
    because the field it was read through, ``ReplayReport.outcome``, did not
    distinguish the two scopes (Codex round 74, P1).

    That is now structural: :attr:`ReplayReport.evidence_outcome` asks the
    narrow question and :attr:`ReplayReport.overall_outcome` asks the total one,
    so VERIFIED means exactly "the question you asked came back clean" and a
    reader learns the scope from the name they read it through. Leaving the old
    caveat here would now UNDERSTATE ``overall_outcome`` (Codex round 83)."""

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

    def __post_init__(self) -> None:
        # `ReplayOutcome` is a StrEnum, so a plain `"failed"` compares EQUAL to
        # `ReplayOutcome.FAILED` while failing every `is` test -- and the
        # outcome derivation is written with `is`, deliberately, because
        # identity is what an enum is for. A string category would therefore be
        # invisible to it in exactly the way a VERIFIED finding is: the report
        # would read VERIFIED while carrying a finding that says otherwise.
        # Nothing but this check stands between a caller and that (Codex 84).
        if not isinstance(self.category, ReplayOutcome):
            raise ValueError(
                f"category must be a ReplayOutcome, not {type(self.category).__name__} "
                f"({self.category!r}) -- a string compares equal to a StrEnum member but is "
                "not identical to it, and the outcome derivation matches on identity, so a "
                "string category would be silently invisible to the verdict"
            )
        # The docstring above said this; nothing enforced it. A VERIFIED
        # finding is not merely odd, it is INVISIBLE to the outcome
        # derivation: it matches neither the "any FAILED" nor the "any
        # UNVERIFIABLE" test, so a report carrying one would derive VERIFIED
        # while holding a finding that says otherwise.
        if self.category is ReplayOutcome.VERIFIED:
            raise ValueError(
                "a ReplayFinding is never categorised VERIFIED -- a clean check produces no "
                f"finding at all, so a VERIFIED finding at {self.ref_path!r} would be a verdict "
                "no check produced"
            )


@dataclass(frozen=True)
class UncheckedStoreClaim:
    """One carried claim ABOUT THE STORE that replay could not check AT ALL.

    The store axis: claims of the form "this node's bytes are recorded in the
    content-addressed store under this digest, and its root sidecar was
    authenticated." When the root tier cannot be read, that claim is neither
    confirmed nor refuted -- it was never tested.

    The sibling axis is :class:`UncheckedSemanticClaim`, and the two are
    deliberately distinct TYPES rather than one type in two lists. The
    distinction is not decorative: the report's account of WHICH axis was left
    untested is the whole reason both lists exist, and one shared type would let
    a store claim be filed as a semantic one without anything noticing, because
    both downgrade the overall verdict identically (Codex round 85, P1).

    Deliberately NOT a :class:`ReplayFinding`, and deliberately not carried in
    ``ReplayReport.findings`` -- so it can never feed
    :attr:`~ReplayReport.evidence_outcome`. It IS what downgrades
    :attr:`~ReplayReport.overall_outcome`, which is the distinction:
    an unchecked claim must not make the evidence verdict look worse than the
    evidence was, and must not let the total verdict look better than the whole
    report earned. That separation is the whole point. "Did the envelope's DATA verify" and
    "was every provenance claim it carries actually checked" are orthogonal
    questions, and collapsing them into one field is what made replay silent
    about an unchecked root-sidecar claim (Codex round 73, P1): an envelope
    whose ``root_sidecar`` claim was forged and whose root ``meta.json`` was
    then deleted replayed exactly like one whose claim was checked and held.

    The alternative -- reporting :attr:`ReplayOutcome.UNVERIFIABLE` -- is wrong
    and was tried and reverted, because it breaks the deliberate contract that
    a perfect extraction record replays VERIFIED with the root sidecar gone
    (``TestReplayVerifiesAgainstTheRecordNotTheRootSidecar``). Replay's
    verification of the data is root-independent by design and must stay so.

    This mirrors the ``checked_char_spans``/``total_char_spans``/
    ``unchecked_char_spans`` triple below, which exists for exactly this
    reason: to make a MIXED result legible instead of collapsing it into one
    outcome that overstates or understates how much was really verified.
    """

    ref_path: str
    """Where in the envelope the unchecked claim lives."""

    claim: str
    """The claim's recorded value, verbatim -- what would have been tested."""

    reason: str
    """Why the check could not run. Never a disagreement: a disagreement is a
    FAILED :class:`ReplayFinding`, not an unchecked claim."""


@dataclass(frozen=True)
class UncheckedSemanticClaim:
    """One carried claim about MEANING that replay could not check AT ALL.

    The semantic axis: claims of the form "the span this ref locates SUPPORTS
    the value recorded here", which replay cannot test even when every
    location-level check passes.

    The concrete case this exists for is the condition-set envelope.
    ``attribution_ref`` is an ordinary :class:`SourceRef` -- it carries a
    locator like any other, and its span re-slices like any other. What is
    missing is the other half of a grounding pair: the value it supports is
    ``ConditionAttribution.OWN_EXPERIMENT`` (an enum), and no recorded text says
    that span means that enum. ``subject.reason_ref`` is the same shape.
    Contrast ``label_ref``, which IS paired with a recorded ``label_raw`` and
    therefore IS checkable in the ordinary way -- the axis is a property of the
    PAIRING, not of the ref. ``unextracted[*].statement_ref`` was once the
    third worked example here; once ``statement_raw`` recorded the refused
    statement's own words beside it, it became exactly that ordinary paired
    case, which is why the example moved to ``subject.reason_ref``.

    This is the load-bearing rule stated as a data type: grounding proves
    LOCATION, never MEANING. A semantic claim is never something replay could
    have caught if it tried harder, and never a defect in the envelope; it is
    the boundary of what re-slicing can establish at all, recorded so the report
    stays honest about it.

    Distinct from :class:`UncheckedStoreClaim` by TYPE, not merely by which list
    it sits in -- see that class for why. Both are refused by the other's list.
    """

    claim_path: str
    """Where in the envelope the DERIVED VALUE lives -- never the path of a
    ref. A derived value may be supported by several refs or by none, so
    naming this field after one ref was a category error."""

    claim: str
    """The DERIVED value whose support went unchecked -- the enum member, unit or
    label the envelope recorded, never the source text the ref points at.

    That restriction is not stylistic. The rest of this module routes every
    quotation of paper text through a redaction gate, because the corpus is
    closed-access and non-redistributable and these reports reach logs and CI.
    This field has no such gate, so it must never be handed a slice of the
    source: what went unchecked is whether the span supports the derived VALUE,
    and naming that value says everything a reader needs without reproducing a
    single word of the paper."""

    gap: SemanticGap
    """Which of the three :class:`SemanticGap` states applies."""

    reason: str
    """Why the check could not run. Never a disagreement: a disagreement is a
    FAILED :class:`ReplayFinding`, not an unchecked claim."""

    support_paths: tuple[str, ...] = ()
    """The ref paths OFFERED as support for this claim, in the order the
    enumeration named them. Empty is meaningful -- it is exactly what
    :attr:`SemanticGap.NO_SUPPORT_OFFERED` describes. A tuple rather than a
    single path because a derived value can rest on more than one ref."""

    def __post_init__(self) -> None:
        if not self.claim_path.strip():
            raise ValueError(
                "UncheckedSemanticClaim.claim_path must be non-empty and "
                "non-blank: it names where the derived value whose support "
                "went unchecked lives, and a blank path cannot be traced "
                "back to that value."
            )
        if not self.claim.strip():
            raise ValueError(
                "UncheckedSemanticClaim.claim must be non-empty and "
                "non-blank: it names the derived value whose support went "
                "unchecked, and a blank claim says nothing was even "
                "attempted to be verified."
            )
        if type(self.gap) is not SemanticGap:
            raise ValueError(
                "UncheckedSemanticClaim.gap must be exactly a SemanticGap "
                f"member, got {type(self.gap)!r}: an exact-type check (not "
                "isinstance) is used because a subclass could carry "
                "arbitrary added behaviour, and this value decides how a "
                "consumer reads the claim."
            )
        object.__setattr__(self, "support_paths", tuple(self.support_paths))
        for path in self.support_paths:
            if type(path) is not str or not path.strip():
                raise ValueError(
                    "UncheckedSemanticClaim.support_paths must contain only "
                    f"non-empty, non-blank str elements, got {path!r}: a "
                    "bool is a subclass of int and similar look-alikes "
                    "exist for str, which is why the check is on the exact "
                    "type."
                )
        if self.gap is SemanticGap.NO_SUPPORT_OFFERED:
            if self.support_paths != ():
                raise ValueError(
                    "UncheckedSemanticClaim with gap=NO_SUPPORT_OFFERED "
                    f"must have empty support_paths, got {self.support_paths!r}: "
                    "a claim where the gap says nothing was offered but "
                    "support_paths lists refs anyway is self-contradictory."
                )
        else:
            if self.support_paths == ():
                raise ValueError(
                    f"UncheckedSemanticClaim with gap={self.gap!r} must have "
                    "non-empty support_paths: a claim where the gap says "
                    "something WAS offered as support but support_paths is "
                    "empty is self-contradictory."
                )


class RefutationStatus(StrEnum):
    """What one attempted falsification actually concluded.

    Deliberately NOT members of :class:`ReplayOutcome`, and the distinction is
    the whole reason this enum exists. ``ReplayOutcome`` answers "what does this
    report claim", and its three members are exhaustive over that question. A
    falsification attempt answers a different question -- "did the attack land"
    -- whose honest answers include one that ``ReplayOutcome`` has no member
    for: the attack ran and did not land, which is not proof of anything.

    Adding that as a fourth ``ReplayOutcome`` member was the obvious move and is
    wrong twice over: it would break every three-way branch over the outcome
    enum, and ``ReplayFinding`` bans only ``VERIFIED``, so a ``NOT_REFUTED``
    finding would be constructed happily and then match neither the FAILED nor
    the UNVERIFIABLE test in the derivation -- present in the report, invisible
    to the verdict. That is the exact failure mode the ``VERIFIED``-finding ban
    exists to stop.
    """

    NOT_REFUTED = "not_refuted"
    """The gate ran, in full, and declined to refute the claim.

    **This is not verification, and the name is chosen so it cannot be read as
    verification.** A refutation gate has exactly one power: to show that a
    claim disagrees with the evidence. Failing to exercise it says the attack
    this gate knows how to mount did not land -- it says nothing about attacks
    it does not know how to mount, and nothing about whether the claim is true.
    A surviving claim is unrefuted, never confirmed."""

    REFUTED = "refuted"
    """The gate ran and the claim contradicts the re-read evidence.

    Always accompanied by a ``FAILED`` :class:`ReplayFinding` AT THE SAME PATH,
    and :class:`ReplayReport` enforces that rather than trusting it -- the axis
    records that an attack was attempted and never replaces the finding that
    reports it. Without the enforcement a report could carry a refutation with
    no finding, and answer ``FAILED`` while holding no account of what failed."""

    UNRUNNABLE = "unrunnable"
    """The gate could not run at all, so nothing was attempted. Never conflated
    with :attr:`NOT_REFUTED`: there, an attack ran and missed; here, no attack
    was ever mounted, and strictly less is known.

    Usually accompanied by an ``UNVERIFIABLE`` :class:`ReplayFinding`, but
    deliberately NOT always, and this is not enforced. When the claim's node
    already carries a store-level problem, the finding is filed against the NODE
    and re-filing it per claim would report one defect as several -- so the
    attempt is recorded here with no sibling finding at the claim's own path.
    An earlier version of this docstring claimed "always", which was simply
    false against that path (Codex round 98)."""


@dataclass(frozen=True)
class AttemptedRefutation:
    """One record that replay TRIED to falsify one claim, and what came of it.

    The fourth axis of :class:`ReplayReport`, and the only one that is populated
    when nothing is wrong. The other three record defects, unchecked store
    claims and unchecked semantic claims; this one records WORK -- which claims
    were attacked, by which gate, and which survived.

    It exists because a refutation gate that declines to refute previously left
    no trace whatsoever. A reader of the report could not distinguish "this
    claim withstood a falsification attempt" from "nothing ever looked at this
    claim", and those are very different states to act on. Silence is the one
    thing a falsification-based verifier must never emit on success, because
    silence is what an absent verifier also emits.
    """

    claim_path: str
    """Where in the envelope the attacked claim lives, e.g.
    ``scalar_claims[0]`` -- the same path vocabulary
    :attr:`ReplayFinding.ref_path` uses, so a refuted entry and its finding can
    be lined up by a reader."""

    gate: str
    """Which falsification was attempted, as ``module:function``. Named rather
    than implied because "not refuted" is only meaningful once a reader knows
    WHICH attack missed -- an unqualified "survived" is exactly the overclaim
    this axis exists to prevent."""

    status: RefutationStatus
    """Which of the three outcomes the attempt produced."""

    reason: str | None = None
    """Why, for :attr:`RefutationStatus.REFUTED` (what disagreed) and
    :attr:`RefutationStatus.UNRUNNABLE` (what stopped the gate). ``None`` for
    :attr:`RefutationStatus.NOT_REFUTED`, which has nothing to report: a
    surviving claim raises no complaint, and prose here would read as a
    justification for trusting it."""

    found: tuple[str, ...] = ()
    """What the gate found instead -- the competing constructs a refutation is
    built on. Empty unless :attr:`status` is :attr:`RefutationStatus.REFUTED`,
    and that is enforced below rather than merely stated here: a gate that could
    not run found nothing by definition, so an UNRUNNABLE entry carrying
    concrete constructs describes something that did not happen."""

    def __post_init__(self) -> None:
        # `tuple(some_str)` splits it into CHARACTERS, and every one of them
        # passes the per-element check at the end of this method: `found="atm"`
        # would be stored as ("a", "t", "m") and reported as three competing
        # constructs the gate never found. A bare str is precisely the
        # iterable-of-str a caller passes here by mistake, so it is refused
        # BEFORE the normalisation, not after (Codex round 98).
        if isinstance(self.found, str | bytes):
            raise ValueError(
                f"AttemptedRefutation.found must be a sequence of strings, not a bare "
                f"{type(self.found).__name__} ({self.found!r}) -- iterating one splits it into "
                "single characters, each of which passes every check below, so the report would "
                "carry one construct per letter and nothing would notice"
            )
        object.__setattr__(self, "found", tuple(self.found))
        if type(self.status) is not RefutationStatus:
            raise ValueError(
                f"AttemptedRefutation.status must be exactly a RefutationStatus member, got "
                f"{type(self.status)!r}: this value decides whether the entry downgrades "
                "overall_outcome to FAILED or to UNVERIFIABLE, and a StrEnum look-alike compares "
                "equal to a member while failing every identity test the derivation is written with"
            )
        for name in ("claim_path", "gate"):
            text = getattr(self, name)
            if type(text) is not str or not text.strip():
                raise ValueError(
                    f"AttemptedRefutation.{name} must be a non-empty, non-blank str, got {text!r} "
                    "-- an entry that cannot name which claim was attacked, or which attack was "
                    "mounted, records that work happened without recording what the work was"
                )
        if self.status is RefutationStatus.NOT_REFUTED:
            # A surviving claim has no complaint to file. Allowing prose here
            # would invite exactly the sentence this axis exists to prevent
            # someone writing -- a reason the claim should be believed, filed
            # under a status that means only that one attack missed.
            if self.reason is not None:
                raise ValueError(
                    f"AttemptedRefutation with status=NOT_REFUTED must carry reason=None, got "
                    f"{self.reason!r}: a claim that survived a falsification attempt has nothing "
                    "to report, and a reason here would read as grounds for believing it"
                )
        else:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    f"AttemptedRefutation with status={self.status!r} must carry a non-blank "
                    "reason: both REFUTED and UNRUNNABLE assert that something specific happened, "
                    "and an entry that cannot say what is indistinguishable from one that did not "
                    "run at all"
                )
        if self.status is not RefutationStatus.REFUTED and self.found != ():
            # Stated on the field and enforced nowhere, this let an UNRUNNABLE
            # entry carry concrete constructs -- a report saying in one breath
            # that the gate never ran and that here is what it found. Only a
            # landed refutation has competing constructs to show.
            raise ValueError(
                f"AttemptedRefutation with status={self.status!r} must carry empty found, got "
                f"{self.found!r}: `found` holds the competing constructs a refutation is built "
                "on, and there is no refutation here"
            )
        for index, item in enumerate(self.found):
            if type(item) is not str or not item.strip():
                raise ValueError(f"AttemptedRefutation.found[{index}] must be a non-empty, non-blank str, got {item!r}")


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

    **There is deliberately no field named ``outcome``.** There was, it was
    scoped to the EVIDENCE checks, and it carried the shortest and most
    prominent name on the report -- so a consumer who read it and stopped was
    told "verified" about a report that might carry provenance claims nothing
    ever tested. That was first patched with a docstring on the enum member
    confessing the gap (Codex round 74); prose does not constrain a caller
    (round 82). Both scopes are now named, and neither name is the bare one, so
    every read has to state which question it is asking:

    * :attr:`evidence_outcome` -- what the checks over the DATA concluded.
    * :attr:`overall_outcome` -- what this report can honestly claim in total.

    Both are DERIVED, not stored. A stored verdict can disagree with the
    findings underneath it, and a frozen dataclass would then serve that
    disagreement to a caller as though a real replay had produced it:
    ``frozen=True`` constrains identity, not consistency (Codex round 83).
    """

    checked_char_spans: int
    total_char_spans: int
    unchecked_char_spans: int
    findings: tuple[ReplayFinding, ...] = ()
    unchecked_store_claims: tuple[UncheckedStoreClaim, ...] = ()
    """Claims about the STORE that replay could not check at all -- see
    :class:`UncheckedStoreClaim`. Orthogonal to :attr:`evidence_outcome` on
    purpose: the evidence may be VERIFIED while entries sit here, meaning "the
    data verified, and this claim about the store was never tested." That
    combination is real and worth reporting -- what it must not do is escape as
    a bare "verified", which is exactly why it is what downgrades
    :attr:`overall_outcome`."""

    unchecked_semantic_claims: tuple[UncheckedSemanticClaim, ...] = ()
    """Claims about MEANING that replay could not check at all -- see
    :class:`UncheckedSemanticClaim`. A separate axis from the store claims
    above, and deliberately NOT one list of one shared type: both downgrade
    :attr:`overall_outcome` identically, so nothing else in the system would
    ever notice one filed as the other, and the report's account of WHICH axis
    was left untested is the reason both exist.

    **There is deliberately no field named ``unchecked_claims``.** Keeping the
    unqualified name for either axis would re-file the other one under it at
    every call site that was never revisited -- the same defect as the bare
    ``outcome`` this report already removed, one level down."""

    attempted_refutations: tuple[AttemptedRefutation, ...] = ()
    """Falsifications this replay ATTEMPTED -- see :class:`AttemptedRefutation`.

    Unlike the three axes above, this one is populated when nothing is wrong: a
    claim that survives its gate leaves a
    :attr:`RefutationStatus.NOT_REFUTED` entry. That is the point. A refutation
    gate that stays silent on success is indistinguishable from a gate that
    never ran, and this list is what makes the two distinguishable in the
    report itself rather than in the reader's memory of which gates exist.

    Every entry, whatever its status, prevents :attr:`overall_outcome` from
    reading VERIFIED -- including ``NOT_REFUTED``, which is the one that matters:
    "we tried to break this claim and could not" is not "this claim is
    verified", and a report that let the former print as the latter would be
    making precisely the inference this whole lane is built to refuse."""

    support_only_char_spans: int = 0
    """Char-span refs that were resolved and re-sliced to confirm the span
    EXISTS, but which carry no recorded text to compare against, so no quote
    was matched. Neither "checked" (nothing was verified about content) nor
    "unchecked" (the location genuinely was resolved) -- collapsing them into
    either one misreports coverage."""

    checked_table_cells: int = 0
    """Table-cell text comparisons that RAN and MATCHED -- each a recorded
    verbatim string re-derived, whole-cell and exact, from a
    :class:`TableCellLocator`-grounded value against the embedded inventory
    it cites (see :func:`_verify_cited_cell_texts`). A DELIBERATELY SEPARATE
    quantity from the char-span counts, and pointedly NOT folded into
    ``total_char_spans`` or its arithmetic: a cell comparison is a different
    kind of check from re-slicing a character span, so it is carried as a
    different number. Its only role in the verdict is the widened zero-check
    rule -- a replay that re-sliced no char span AND matched no cell has
    established nothing and must read UNVERIFIABLE, but one that matched even a
    single cell has verified something real, even with zero char spans. That is
    the whole point of this field: "re-sliced zero character spans" and
    "established nothing" stopped being the same sentence the moment a cell
    comparison became a real check."""

    def __post_init__(self) -> None:
        # `frozen=True` stops the FIELD being rebound; it does nothing about the
        # object the field points at. Handed a list, the report would keep the
        # caller's live list, and appending to it afterwards would silently
        # change a verdict that has already been read -- both outcomes are
        # derived on every access, so there is no snapshot to protect them
        # (Codex round 84). Normalising to a tuple here makes the freeze reach
        # the contents, and costs nothing on the producer path, which already
        # passes tuples.
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "unchecked_store_claims", tuple(self.unchecked_store_claims))
        object.__setattr__(self, "unchecked_semantic_claims", tuple(self.unchecked_semantic_claims))
        object.__setattr__(self, "attempted_refutations", tuple(self.attempted_refutations))

        # Normalising the CONTAINER says nothing about what is in it. Both
        # derivations read `.category` off each finding and match it on
        # identity, so any object carrying that attribute takes part in the
        # verdict -- and one whose `.category` is the string "failed" matches
        # no branch, leaving a report that reads VERIFIED while holding
        # something that calls itself a failure. Exact type rather than
        # `isinstance`: a subclass could carry state the derivation does not
        # know to consult, and this is a fail-closed path (Codex round 85).
        for index, finding in enumerate(self.findings):
            if type(finding) is not ReplayFinding:
                raise ValueError(
                    f"findings[{index}] is a {type(finding).__name__}, not a ReplayFinding -- "
                    "the outcome derivation reads .category off each entry and matches it on "
                    "identity, so a look-alike would take part in the verdict without being "
                    "bound by any of the rules ReplayFinding enforces"
                )
        # Per AXIS, by exact type. A store claim sitting in the semantic list
        # changes no verdict -- both downgrade overall_outcome identically --
        # which is precisely why nothing else in the system would ever catch
        # it. What it corrupts is the report's account of WHICH axis was left
        # untested, and that account is the only reason there are two lists
        # (Codex round 85, P1).
        for field_name, expected in (
            ("unchecked_store_claims", UncheckedStoreClaim),
            ("unchecked_semantic_claims", UncheckedSemanticClaim),
        ):
            for index, claim in enumerate(getattr(self, field_name)):
                if type(claim) is not expected:
                    raise ValueError(
                        f"{field_name}[{index}] is a {type(claim).__name__}, not an "
                        f"{expected.__name__} -- its mere presence downgrades overall_outcome, "
                        "so an arbitrary object here silently decides a verdict, and a claim "
                        "from the other axis would misreport which axis was left untested"
                    )

        # Same rule, fourth axis. This one decides between FAILED and
        # UNVERIFIABLE (not merely whether to downgrade), so a look-alike whose
        # `.status` matches no branch would leave a report carrying an
        # attempted refutation that reads VERIFIED anyway -- the single outcome
        # this axis exists to make impossible.
        for index, attempt in enumerate(self.attempted_refutations):
            if type(attempt) is not AttemptedRefutation:
                raise ValueError(
                    f"attempted_refutations[{index}] is a {type(attempt).__name__}, not an "
                    "AttemptedRefutation -- the outcome derivation reads .status off each entry "
                    "and matches it on identity, so a look-alike would decide a verdict without "
                    "being bound by any of the rules AttemptedRefutation enforces"
                )

        # A REFUTED attempt asserts a demonstrated disagreement, and a report
        # that asserts one owes its reader the account of it. Without this, a
        # report could answer FAILED while `findings` was empty -- a verdict
        # with nothing behind it, and `evidence_outcome` would meanwhile still
        # read VERIFIED off those same empty findings, so the two properties
        # would contradict each other on one object (Codex round 98).
        #
        # Enforcing it here rather than patching `overall_outcome` is what keeps
        # the two consistent BY CONSTRUCTION, and it is why that property has no
        # REFUTED branch: the pairing makes `evidence_outcome` FAILED already,
        # so such a branch could never be the one that decided anything. A
        # branch that cannot be reached is not a check.
        failed_paths = {f.ref_path for f in self.findings if f.category is ReplayOutcome.FAILED}
        for index, attempt in enumerate(self.attempted_refutations):
            if attempt.status is RefutationStatus.REFUTED and attempt.claim_path not in failed_paths:
                raise ValueError(
                    f"attempted_refutations[{index}] is REFUTED at {attempt.claim_path!r} but no "
                    "FAILED finding sits at that path -- a report claiming a refutation landed "
                    "must carry the finding that says what disagreed, or it answers FAILED with "
                    "no account of why while evidence_outcome still reads VERIFIED"
                )

        # `bool` subclasses `int`, so `True` satisfies every bound below and
        # reads as a count of 1; a float satisfies them too and can make the
        # arithmetic agree by coincidence. Exact type only.
        for name in ("checked_char_spans", "total_char_spans", "unchecked_char_spans", "checked_table_cells"):
            count = getattr(self, name)
            if type(count) is not int:
                raise ValueError(
                    f"{name} must be an int, not {type(count).__name__} ({count!r}) -- bools "
                    "and floats compare cleanly against these bounds, and would let a report "
                    "derive a verdict from a coverage count that was never really counted"
                )

        if self.checked_char_spans < 0 or self.total_char_spans < 0:
            raise ValueError(
                "a replay cannot have checked a negative number of char spans "
                f"(checked={self.checked_char_spans}, total={self.total_char_spans})"
            )
        if self.checked_char_spans > self.total_char_spans:
            raise ValueError(
                f"checked_char_spans ({self.checked_char_spans}) exceeds total_char_spans "
                f"({self.total_char_spans}) -- a replay cannot check more spans than the "
                "envelope reaches"
            )
        if self.support_only_char_spans < 0:
            raise ValueError(
                f"support_only_char_spans ({self.support_only_char_spans}) cannot be negative "
                "-- it counts refs that were resolved and re-sliced, so a negative value "
                "misreports coverage"
            )
        if self.checked_table_cells < 0:
            raise ValueError(
                f"checked_table_cells ({self.checked_table_cells}) cannot be negative -- it counts "
                "cell comparisons that ran and matched, so a negative value misreports coverage"
            )
        if self.checked_char_spans + self.support_only_char_spans > self.total_char_spans:
            raise ValueError(
                f"checked_char_spans ({self.checked_char_spans}) + support_only_char_spans "
                f"({self.support_only_char_spans}) exceeds total_char_spans "
                f"({self.total_char_spans}) -- a replay cannot have resolved more spans than "
                "the envelope reaches"
            )
        expected_unchecked = self.total_char_spans - self.checked_char_spans - self.support_only_char_spans
        if self.unchecked_char_spans != expected_unchecked:
            raise ValueError(
                f"unchecked_char_spans ({self.unchecked_char_spans}) must be "
                f"total_char_spans - checked_char_spans - support_only_char_spans "
                f"({expected_unchecked}) -- these counts are what make a MIXED result legible, "
                "so a set that does not add up misreports coverage"
            )

    @property
    def evidence_outcome(self) -> ReplayOutcome:
        """What the checks over the envelope's DATA concluded.

        Scoped to the raw bytes, the addressed extraction record and the
        grounded CHAR SPANS -- and that last word is a real limit, not a
        loose synonym for "locations". A ref whose locator is not a
        :class:`CharSpanLocator` is not something this replayer can re-slice
        at all, so it is outside what this verdict ranges over: ``VERIFIED``
        here means every char-span quote that could be compared was compared
        and matched, NOT that every location the envelope cites was resolved.
        A condition set can therefore read ``VERIFIED`` here while carrying an
        ``UncheckedSemanticClaim`` whose gap is
        :attr:`SemanticGap.LOCATION_UNRESOLVED`.

        It also says nothing about whether every claim the envelope makes
        ABOUT THE STORE was checkable, nor about the semantic claims. Both of
        those fold into :attr:`overall_outcome`, which is the verdict to read
        when the question is "is this envelope trustworthy?" rather than "did
        the quote checks pass?".
        """
        if any(f.category is ReplayOutcome.FAILED for f in self.findings):
            return ReplayOutcome.FAILED
        if any(f.category is ReplayOutcome.UNVERIFIABLE for f in self.findings):
            return ReplayOutcome.UNVERIFIABLE
        if self.unchecked_char_spans > 0:
            # The report states in as many words that some reachable spans were
            # never checked. A verdict that ignored its own coverage count would
            # be the original overclaim moved up one level -- the count is not
            # decoration, it is what makes a MIXED result legible (Codex 84).
            #
            # Today's producer always records a finding alongside an unchecked
            # span, so this changes no real replay result -- probed by applying
            # it and running the replay + producer suites unchanged. It is here
            # for the same reason as the zero-span rule below: the derivation
            # has to hold for every report, not only the ones this module
            # happens to build.
            return ReplayOutcome.UNVERIFIABLE
        if self.checked_char_spans == 0 and self.checked_table_cells == 0:
            # A replay that established nothing has established nothing. The
            # producer also records a finding saying so, carrying the reason a
            # caller needs -- but a rule enforced only by the one caller that
            # remembers it is not an invariant, and this report type is public
            # and constructible by anyone.
            #
            # The question this asks is deliberately WIDER than it once was. It
            # used to test only ``checked_char_spans == 0`` -- correct while a
            # re-sliced char span was the ONLY check there was. Now a matched
            # table cell is also a real check (see ``checked_table_cells``), so
            # "re-sliced zero char spans" no longer implies "established
            # nothing": an artifact grounded entirely at table cells re-slices
            # zero char spans yet may have compared and matched every one of its
            # cells. The rule the zero-span guard exists to enforce -- a replay
            # that verified NOTHING must never read VERIFIED -- survives intact;
            # only its definition of "nothing" widened to match what a replay
            # can now check.
            return ReplayOutcome.UNVERIFIABLE
        return ReplayOutcome.VERIFIED

    @property
    def overall_outcome(self) -> ReplayOutcome:
        """What this report can honestly claim, across every axis it covers.

        ``VERIFIED`` here means: every applicable check ran, every one passed,
        nothing was left untested, and no falsification was attempted that could
        only ever have failed to land. It is the verdict a consumer may act on
        WITHOUT also inspecting a side list -- which is the whole reason it
        exists, so nothing that widens "what was left untested", and nothing
        that records an attempted refutation, may be added to this report
        without also being wired in here.

        That last clause is not a widening of "untested": a survived
        falsification attempt IS a check that ran to completion. It bars
        VERIFIED anyway, because the gate behind it can only ever show a claim
        wrong, so its non-firing establishes nothing to verify.
        """
        evidence = self.evidence_outcome
        # A demonstrated disagreement outranks an untested claim: FAILED must
        # never soften to UNVERIFIABLE on its way through this property. The
        # two are never conflated, in EITHER direction.
        if evidence is ReplayOutcome.FAILED:
            return ReplayOutcome.FAILED
        # There is deliberately NO branch for RefutationStatus.REFUTED here. It
        # would be unreachable: __post_init__ refuses any report whose REFUTED
        # attempt has no FAILED finding at the same path, so the branch above
        # has already returned FAILED for every report that could reach one.
        # The first draft had that branch, and an unreachable branch is not a
        # check -- it is a check-shaped comment that no test can ever cover
        # (the lesson of the `if table is None` guard against a function that
        # raises, Codex round 96). The invariant lives in the constructor, where
        # it can actually refuse.
        #
        # BOTH axes, and every future one. A list that widens "what was left
        # untested" and is not wired in here recreates the original overclaim
        # one level up: this property's entire promise is that a consumer need
        # not also read a side list to know what the report earned.
        if self.unchecked_store_claims or self.unchecked_semantic_claims:
            return ReplayOutcome.UNVERIFIABLE
        # The remaining statuses are NOT_REFUTED and UNRUNNABLE, and NEITHER may
        # reach VERIFIED. UNRUNNABLE is the ordinary case -- a check that could
        # not run never verifies. NOT_REFUTED is the load-bearing one: it is a
        # check that ran, in full, and passed, which everywhere else in this
        # module is exactly what VERIFIED means. It is not that here, because
        # the gate is a falsification gate: it can only ever show a claim
        # WRONG, so surviving it is evidence of nothing. Deleting this branch
        # would make a claim no attack could break print as a claim proved true
        # -- the inference the dual-lane design exists to refuse, arrived at
        # through the verifier rather than through the LLM.
        if self.attempted_refutations:
            return ReplayOutcome.UNVERIFIABLE
        return evidence

    @property
    def evidence_failures(self) -> tuple[ReplayFinding, ...]:
        """Findings that are definite disagreements. EVIDENCE-scoped: an
        unchecked claim never appears here, because it is not a disagreement."""
        return tuple(f for f in self.findings if f.category is ReplayOutcome.FAILED)

    @property
    def evidence_unverifiable(self) -> tuple[ReplayFinding, ...]:
        """Findings whose check could not run. EVIDENCE-scoped, and the scope is
        load-bearing in the name: a report whose OVERALL verdict is
        ``UNVERIFIABLE`` because of an unchecked claim returns ``()`` here, and a
        caller reading a shorter name would have concluded the opposite."""
        return tuple(f for f in self.findings if f.category is ReplayOutcome.UNVERIFIABLE)


class _RawBin(NamedTuple):
    """One node's ``raw.bin`` after a single read-and-hash against ``node.sha256``.

    Read ONCE, in the replay loop, and handed to everything that needs the
    node's raw bytes -- text re-verification AND the table-inventory
    re-derivation -- so the two can never observe different bytes for the same
    document (which would make the two halves of a replay report on different
    files). ``data`` is the bytes ONLY when they were present and hashed to
    ``node.sha256``; a missing file (``UNVERIFIABLE``) or a hash mismatch
    (``FAILED``, positive evidence of tampering) yields ``data=None`` and a
    ``problem`` naming which."""

    data: bytes | None
    problem: ReplayFinding | None


def _load_and_check_raw_bin(workspace_root: Path, node: SourceNode) -> _RawBin:
    """Read ``node``'s ``raw.bin`` once and hash it against ``node.sha256``.

    Absence is inability to check: ``raw.bin`` may simply have been
    garbage-collected or never fetched into this workspace, which says nothing
    about whether the node's identity is trustworthy -- ``UNVERIFIABLE``. A
    ``raw.bin`` that IS present but hashes to something other than
    ``node.sha256`` is different in kind: positive evidence the store's raw
    bytes were tampered with or corrupted -- ``FAILED``. The two must never be
    conflated into one outcome."""
    path = f"source_graph.node({node.node_id!r})"
    raw_path = artifact_dir(workspace_root, node.sha256) / "raw.bin"
    try:
        data = raw_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        return _RawBin(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"no readable raw.bin for node {node.node_id!r} (sha256={node.sha256!r}): {exc}",
            ),
        )
    actual_raw_sha256 = hashlib.sha256(data).hexdigest()
    if actual_raw_sha256 != node.sha256:
        return _RawBin(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"raw.bin on disk for node {node.node_id!r} hashes to {actual_raw_sha256!r}, not "
                f"the node.sha256={node.sha256!r} it is stored under -- the evidence store's raw bytes "
                "have been tampered with or corrupted since this node was recorded",
                expected=node.sha256,
                actual=actual_raw_sha256,
            ),
        )
    return _RawBin(data, None)


class NodeVerification(NamedTuple):
    """Outcome of independently re-verifying one node against the store."""

    text: str | None
    problem: ReplayFinding | None
    problem_is_text_only: bool
    """``True`` iff ``problem`` is non-``None`` and its ONLY complaint is
    that ``node.extraction`` is ``Absent`` -- the node's bytes (``raw.bin``)
    were read and hashed cleanly, and there is simply no extracted text
    recorded for this node. That is a legitimate state for a non-textual
    node (e.g. a ``FIGURE_CROP`` targeted only by a ``BBoxLocator``), not a
    defect of the envelope, so callers must surface it as a node-level
    finding only where something text-dependent actually targets the node.
    Always ``False`` when ``problem`` is ``None``, and ``False`` for every
    other kind of problem (missing/tampered ``raw.bin``, or any extraction
    record disagreement)."""


def _independently_verify_node_text(workspace_root: Path, node: SourceNode, raw_bin: _RawBin) -> NodeVerification:
    """Re-derive ``node``'s extracted text from the evidence store, from
    bytes, independent of anything the envelope itself carries.

    Reads the ``extracted.json`` bytes belonging to the extraction record the
    node's own :class:`ExtractionBinding` ADDRESSES, verifies them against the
    digest that binding carries, parses, and only then hashes the parsed text
    and compares against the RECORDED ``extracted_text_sha256``.

    This docstring used to say it mirrored
    ``dataset_producer._load_verified_extracted_text`` and verified bytes
    "against the digest :class:`StoredArtifact` recorded at store time" -- i.e.
    the ROOT sidecar and the root ``meta.json``. That was stale: this function
    has not touched either since it began resolving the addressed record, and
    the producer helper it named no longer exists. The distinction is the whole
    security argument below (see the anchor comment before the
    ``extracted_sha256`` comparison): a sidecar sitting next to the file it
    would authenticate is trivially rewritten by anyone who can write to the
    store, so the envelope's own binding is the only acceptable anchor.

    The node's bytes are checked FIRST and unconditionally, by
    :func:`_load_and_check_raw_bin` in the replay loop: every node admitted
    into a VERIFIED envelope must have ``raw.bin`` present and hashing to its
    recorded ``sha256``, whether or not it carries an ExtractionBinding. That
    check's result is handed in as ``raw_bin`` -- read ONCE so the table-
    inventory re-derivation and this text re-verification cannot observe
    different bytes for the same document -- and this function short-circuits
    on its problem before looking at whether there is any extracted text to
    re-verify at all.

    Returns ``NodeVerification(text, None, False)`` on success, or
    ``NodeVerification(None, finding, problem_is_text_only)`` naming
    exactly which step failed and whether that step could not run at all
    (``UNVERIFIABLE``) or ran and disagreed (``FAILED``).
    ``problem_is_text_only`` is ``True`` for exactly one case: the node's
    bytes verified cleanly and the only complaint is that ``extraction`` is
    ``Absent`` (see :class:`NodeVerification`).
    """
    path = f"source_graph.node({node.node_id!r})"
    extraction = node.extraction
    if raw_bin.problem is not None:
        # Missing (UNVERIFIABLE) or tampered (FAILED) raw.bin -- the single
        # read already classified which, and neither is a text-only problem.
        return NodeVerification(None, raw_bin.problem, False)
    if isinstance(extraction, Absent):
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"node {node.node_id!r} has no ExtractionBinding (extraction is Absent); "
                "there is nothing recorded to independently re-verify",
            ),
            True,
        )
    # Resolve the ADDRESSED extraction record -- (parent_raw_sha256,
    # extraction_sha256) -- never the OLD single-extraction-per-raw-sha
    # evidence.load_artifact_meta path used above only for raw.bin. One raw
    # document can now legitimately have MANY extraction records (see
    # SourceGraph's I5c/I5b docstrings in carmel.schemas.datasets), so the
    # node's own ExtractionBinding is the only honest anchor for which one
    # this node claims.
    try:
        record_meta = load_extraction_record(workspace_root, extraction.parent_raw_sha256, extraction.extraction_sha256)
    except ExtractionRecordError as exc:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"extraction record meta.json for node {node.node_id!r} at address "
                f"(parent_raw_sha256={extraction.parent_raw_sha256!r}, "
                f"extraction_sha256={extraction.extraction_sha256!r}) is unreadable: {exc}",
            ),
            False,
        )
    if record_meta is None:
        # No record stored at that address, or its meta.json does not
        # authenticate to it -- either way this is inability to check, not
        # positive evidence of anything wrong.
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"no extraction record stored (or its meta.json does not authenticate) for node "
                f"{node.node_id!r} at address (parent_raw_sha256={extraction.parent_raw_sha256!r}, "
                f"extraction_sha256={extraction.extraction_sha256!r})",
            ),
            False,
        )
    # The addressed record is present and self-authenticated to its own
    # address; now cross-check the identity fields the ENVELOPE carries
    # against the ones the RECORD recorded at store time. Both sides
    # recompute to the same content address in the honest case, so a
    # disagreement here is positive evidence that one of them was altered
    # outside validated construction (a forged envelope field, a
    # hand-edited record) -- the record exists and was readable throughout,
    # so this is FAILED, never inability-to-check. ``pypdf_version`` is
    # compared only when the binding carries one (i.e. for the
    # pypdf-dependent extractors): for every other extractor the record's
    # ``pypdf_version`` is a diagnostics-only field that is deliberately
    # NOT part of the identity address (see ``_records_identical`` and
    # ``_build_identity_payload`` in carmel.services.extraction_record), so
    # comparing it would fail closed on a routine pypdf upgrade that
    # legitimately cannot change the record's identity.
    binding_identity: dict[str, str] = {
        "extractor": extraction.extractor,
        "extractor_code_sha256": extraction.extractor_code_sha256,
        "identity_payload_version": extraction.identity_payload_version,
    }
    record_identity: dict[str, str] = {
        "extractor": record_meta.extractor,
        "extractor_code_sha256": record_meta.extractor_code_sha256,
        "identity_payload_version": record_meta.identity_payload_version,
    }
    if not isinstance(extraction.pypdf_version, Absent):
        binding_identity["pypdf_version"] = extraction.pypdf_version
        record_identity["pypdf_version"] = record_meta.pypdf_version
    if binding_identity != record_identity:
        disagreeing = sorted(k for k in binding_identity if binding_identity[k] != record_identity[k])
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"node {node.node_id!r}'s ExtractionBinding identity fields "
                f"({', '.join(disagreeing)}) disagree with the stored extraction record's own meta.json "
                f"at address (parent_raw_sha256={extraction.parent_raw_sha256!r}, "
                f"extraction_sha256={extraction.extraction_sha256!r}) -- the envelope's carried "
                "extractor identity was never the one hashed into the record it addresses; one of the "
                "two was altered outside validated construction",
                expected=repr({k: record_identity[k] for k in disagreeing}),
                actual=repr({k: binding_identity[k] for k in disagreeing}),
            ),
            False,
        )
    extracted_path = (
        extraction_record_dir(workspace_root, extraction.parent_raw_sha256, extraction.extraction_sha256)
        / "extracted.json"
    )
    try:
        raw_bytes = extracted_path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"no readable extracted.json for node {node.node_id!r}'s extraction record "
                f"(parent_raw_sha256={extraction.parent_raw_sha256!r}, "
                f"extraction_sha256={extraction.extraction_sha256!r}): {exc}",
            ),
            False,
        )
    actual_extracted_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    # The ONLY acceptable anchor is the envelope's own ExtractionBinding --
    # never the extraction record's meta.json sidecar. meta.json lives right
    # next to the very file it would otherwise authenticate and is
    # trivially rewritable by anyone who can write to the store; trusting
    # it as an anchor lets a forger rewrite extracted.json and meta.json
    # together and still get a VERIFIED result. See this module's
    # docstring ("Evidence identity").
    if actual_extracted_sha256 != extraction.extracted_sha256:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"extracted.json bytes on disk for node {node.node_id!r}'s extraction record do "
                "not match ExtractionBinding.extracted_sha256 recorded on the envelope -- the stored "
                "evidence has been tampered with or corrupted since the envelope was produced",
                expected=extraction.extracted_sha256,
                actual=actual_extracted_sha256,
            ),
            False,
        )
    # record_meta.json is cross-checked only AFTER the envelope's own anchor
    # has already passed, and purely as an early-warning signal: a
    # disagreement here means the extraction record store's own bookkeeping
    # is inconsistent with the envelope (rewritten independently of it, or
    # never updated), which is itself worth reporting even though the
    # envelope-anchored check above is what actually decided pass/fail.
    if record_meta.extracted_sha256 != actual_extracted_sha256:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"extraction record meta.json for node {node.node_id!r} records "
                f"extracted_sha256={record_meta.extracted_sha256!r}, which disagrees with the bytes "
                "actually on disk (and with the envelope's own anchor, already verified above) -- the "
                "store's own sidecar bookkeeping is inconsistent and is reported rather than silently "
                "reconciled",
                expected=actual_extracted_sha256,
                actual=record_meta.extracted_sha256,
            ),
            False,
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
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"verified extracted.json for node {node.node_id!r} does not parse as ExtractedText: {exc!r}",
            ),
            False,
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
        page_note = f" ({len(extracted.page_failures)} page(s) failed to extract)" if extracted.page_failures else ""
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path=path,
                reason=f"verified extracted.json for node {node.node_id!r} is a lossy extraction "
                f"(extractor={extracted.extractor!r}){page_note}; a knowingly-partial extraction cannot "
                "be independently re-verified as a faithful re-derivation of the document text",
            ),
            False,
        )
    actual_text_sha256 = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    if actual_text_sha256 != extraction.extracted_text_sha256:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"independently re-read text for node {node.node_id!r} hashes to "
                f"{actual_text_sha256!r}, but the envelope's ExtractionBinding.extracted_text_sha256 "
                f"recorded {extraction.extracted_text_sha256!r} at production time",
                expected=extraction.extracted_text_sha256,
                actual=actual_text_sha256,
            ),
            False,
        )
    if record_meta.extracted_text_sha256 != actual_text_sha256:
        return NodeVerification(
            None,
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason=f"extraction record meta.json for node {node.node_id!r} records "
                f"extracted_text_sha256={record_meta.extracted_text_sha256!r}, which disagrees with the "
                "independently re-read text actually stored (and with the envelope's own anchor, already "
                "verified above) -- the store's own sidecar bookkeeping is inconsistent and is reported "
                "rather than silently reconciled",
                expected=actual_text_sha256,
                actual=record_meta.extracted_text_sha256,
            ),
            False,
        )
    return NodeVerification(extracted.text, None, False)


class RootSidecarClaimCheck(NamedTuple):
    """The result of putting one node's ``root_sidecar`` claim to the store.

    At most one field is ever non-``None``: the store either contradicted the
    claim (``finding``), or could not be read to check it (``unchecked``), or
    agreed with it (both ``None``). They come from ONE function and ONE read of
    the root tier deliberately -- computing them in two passes would read
    ``meta.json`` twice, and a root tier that appears between the two reads (or
    disappears between them) would yield a report claiming both that the check
    ran and that it could not, or neither.
    """

    finding: ReplayFinding | None = None
    unchecked: UncheckedStoreClaim | None = None


def _refute_root_sidecar_claim(workspace_root: Path, node: SourceNode) -> RootSidecarClaimCheck:
    """Try to REFUTE this node's recorded ``root_sidecar`` claim against the store.

    This is the one tier of :class:`~carmel.schemas.datasets.SourceVerification`
    that replay cannot otherwise see. The other two are already re-derived from
    bytes by :func:`_independently_verify_node_text`, which re-hashes ``raw.bin``
    and re-verifies the addressed extraction record -- so an envelope lying about
    either of them fails there, on evidence, and a carried claim adds nothing.
    Nothing anywhere else in this module reads the ROOT ``meta.json``, by design:
    the root sidecar is not evidence and is never an anchor. Which leaves a
    genuine gap -- a claim about the root tier that no consumer could contradict
    is decoration, and decoration is exactly how provenance nobody reads gets
    into a codebase.

    So this function reads the root tier for ONE purpose: refutation. It never
    authenticates anything, and its result never licenses trusting any text.

    EVERY member of :class:`RootSidecarVerification` is refutable, which is why
    that enum has no "not checked" value. Each member asserts something ABOUT
    THE STORE, and each is stable, because root sidecars are never rewritten
    (see :mod:`carmel.services.reextraction`, which writes only under
    ``extractions/``). So this recomputes the claim exactly as the producer did
    and compares:

    - ``NO_RECORDED_DIGEST`` is contradicted by a root meta that DOES record an
      ``extracted_sha256``;
    - ``ROOT_SIDECAR_DIGEST_AUTHENTICATED`` is contradicted by a sidecar whose
      bytes do not hash to the recorded digest, or by no recorded digest at all;
    - ``ROOT_SIDECAR_DIGEST_MISMATCH`` is contradicted by a sidecar that hashes
      correctly.

    Each disagreement is positive evidence that the claim was false when made or
    that the envelope was altered afterwards, hence FAILED.

    IT ONLY EVER REFUTES. A root tier that is missing, unreadable or unparseable
    produces NO FINDING at all -- not UNVERIFIABLE -- and that is a hard
    contract rather than a convenience.
    ``TestReplayVerifiesAgainstTheRecordNotTheRootSidecar`` pins it: a perfect
    record must replay VERIFIED with the root sidecar gone, because replay's
    verification of the DATA is root-independent by design and has to stay that
    way. A workspace that garbage-collects root sidecars, or a dataset replayed
    somewhere holding only records, would otherwise degrade for a reason having
    nothing to do with its data.

    An earlier revision reported UNVERIFIABLE there, reasoning that
    inability-to-check must never be silently a pass. That reasoning is right in
    general and wrong here, and the existing contract test caught it: failing to
    REFUTE a claim is not the same as failing to VERIFY the evidence. Deleting
    the root meta buys an attacker nothing about the DATA either -- the raw
    bytes and the addressed record still have to authenticate, and those are
    what the envelope's data actually rests on.

    UNREFUTED IS NOT UNREPORTED, though, and for a while it was. Codex round 73
    landed the P1: an envelope could claim ``ROOT_SIDECAR_DIGEST_AUTHENTICATED``
    and then have its root ``meta.json`` deleted or made unreadable, and the
    resulting report was indistinguishable from one where the claim was checked
    and held. So an unreadable root tier now yields an :class:`UncheckedStoreClaim`
    instead of nothing. That is not a finding and does not touch ``evidence_outcome`` --
    the contract above is untouched -- it simply stops the report implying a
    check that never ran.

    Args:
        workspace_root: Root of the campaign workspace holding the store.
        node: The node whose recorded claim is under test.

    Returns:
        A :class:`RootSidecarClaimCheck` carrying a FAILED finding when the
        store positively contradicts the claim; an :class:`UncheckedStoreClaim` when
        the root tier could not be read to check it; and neither when the claim
        survives or the node carries no verification record at all.
    """
    verification = node.verification
    if isinstance(verification, Absent):
        # Nothing was claimed, so there is nothing to check and nothing left
        # unchecked. This is NOT an unchecked claim: reporting one here would
        # mean every pre-``SourceVerification`` envelope grew a permanent
        # entry naming a claim it never made.
        return RootSidecarClaimCheck()
    claimed = verification.root_sidecar
    path = f"source_graph.node({node.node_id!r}).verification.root_sidecar"
    try:
        meta = load_artifact_meta(workspace_root, node.sha256)
    except (OSError, ValueError) as exc:
        # NOT a finding. See the contract note in the docstring: a root tier
        # that cannot be read leaves the claim UNREFUTED, which is a fact about
        # this workspace and not a defect of the envelope. It is reported as
        # unchecked so that fact is visible rather than silent.
        #
        # ``ValueError`` here does NOT mean "the file was corrupt" --
        # ``load_artifact_meta`` raises it only for a malformed sha256 or a
        # directory that would escape the workspace root, and swallows every
        # parse failure into ``None`` instead. So the reason names the
        # exception type rather than asserting a cause.
        return RootSidecarClaimCheck(
            unchecked=_unchecked_root_claim(
                path, node, claimed, f"resolving the root artifact raised {type(exc).__name__}"
            )
        )
    if meta is None:
        # DELIBERATELY does not say "absent". ``load_artifact_meta`` returns
        # ``None`` for missing, unreadable AND invalid alike, so this branch
        # cannot tell them apart, and a reason string that picked one would
        # assert a fact nothing here established -- the precise over-claim this
        # module keeps having to unwind elsewhere. Distinguishing them would
        # need a second stat/read of the same file, reintroducing exactly the
        # TOCTOU that :class:`RootSidecarClaimCheck` exists to avoid. Naming
        # all three is the honest option, and the operator has the artifact
        # sha256 below to go look.
        return RootSidecarClaimCheck(
            unchecked=_unchecked_root_claim(path, node, claimed, "root meta.json is missing, unreadable or invalid")
        )
    actual = _recompute_root_sidecar_claim(workspace_root, node.sha256, meta)
    if actual is claimed:
        return RootSidecarClaimCheck()
    return RootSidecarClaimCheck(
        finding=ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=path,
            reason=f"node {node.node_id!r} claims root_sidecar={claimed.value}, but recomputing "
            f"that claim against the store yields {actual.value}. Root sidecars are never "
            "rewritten, so the claim was false when it was made or the envelope was altered "
            "afterwards",
            expected=actual.value,
            actual=claimed.value,
        )
    )


def _unchecked_root_claim(
    path: str, node: SourceNode, claimed: RootSidecarVerification, why: str
) -> UncheckedStoreClaim:
    """Build the :class:`UncheckedStoreClaim` for a root tier that could not be read.

    Names the artifact sha256 as well as the node id: a reader holding only the
    report needs to know WHICH artifact's root tier was unreachable in order to
    go look, and the node id alone does not say.
    """
    return UncheckedStoreClaim(
        ref_path=path,
        claim=claimed.value,
        reason=f"node {node.node_id!r} claims root_sidecar={claimed.value}, but that claim could "
        f"not be checked: {why} for artifact {node.sha256}. The claim is neither confirmed nor "
        "refuted -- it was never tested",
    )


def _recompute_root_sidecar_claim(workspace_root: Path, sha256: str, meta: StoredArtifact) -> RootSidecarVerification:
    """Recompute a node's root-sidecar claim from the store, independently.

    Deliberately NOT a call into ``dataset_producer._root_sidecar_claim``.
    Replay's entire value is being an INDEPENDENT re-derivation: sharing the
    producer's implementation would mean a bug in it produces a wrong claim and
    then agrees with itself, and this check would confirm rather than refute.
    The duplication is the point, the same way the char-span re-slice does not
    reuse the grounding gate's own span arithmetic.
    """
    if meta.extracted_sha256 is None:
        return RootSidecarVerification.NO_RECORDED_DIGEST
    try:
        sidecar_bytes = (artifact_dir(workspace_root, sha256) / "extracted.json").read_bytes()
    except OSError:
        return RootSidecarVerification.ROOT_SIDECAR_DIGEST_MISMATCH
    if hashlib.sha256(sidecar_bytes).hexdigest() == meta.extracted_sha256:
        return RootSidecarVerification.ROOT_SIDECAR_DIGEST_AUTHENTICATED
    return RootSidecarVerification.ROOT_SIDECAR_DIGEST_MISMATCH


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


class _TextPairing(NamedTuple):
    """One (ref -> recorded text) GROUNDING PAIR an envelope exposes.

    The pair is the unit the evidence kernel can actually test: a locator
    saying WHERE, beside a recorded string saying WHAT was read there. A ref
    with no recorded counterpart is NOT a pairing and must never be
    expressed as one -- re-slicing it would compare the span against
    nothing, and reporting it as checked would claim a grounding that was
    never established.

    ``uncompensated_fields`` carries the non-``CharSpanLocator`` POLICY,
    which differs per field and is a property of the surrounding module, not
    of the locator: ``None`` means some other gate in this module already
    reports UNVERIFIABLE for a non-char-span locator on this field (as
    :func:`verify_measured_value_value_boundary` and
    :func:`verify_measured_value_unit_boundary` do for ``value_ref`` and
    ``unit_ref``), so the kernel stays silent and leaves it to that gate. A
    ``(ref_field, text_field)`` pair means there is NO compensating gate, so
    the kernel must report it here or it goes unreported entirely.
    """

    path: str
    node_id: str
    locator: object
    expected: str
    always_literal: bool = False
    uncompensated_fields: tuple[str, str] | None = None


def _measured_value_text_pairings(envelope: object) -> Iterator[_TextPairing]:
    """Yield the pairings every :class:`MeasuredValue` in ANY envelope exposes.

    Generic on purpose: :func:`iter_measured_values` walks an arbitrary
    pydantic tree, so this is the one enumerator that is correct for every
    envelope type without being told that type's shape.
    """
    for path, value in iter_measured_values(envelope):
        yield _TextPairing(f"{path}.value_ref", value.value_ref.node_id, value.value_ref.locator, value.raw_text)
        # A value whose unit was NOT printed in the source (I-060) has no
        # unit_ref and no unit_raw to pair -- there is no located token to
        # re-slice. The header-bridge semantic claim, not a text pairing, is what
        # records that its quantity is unproven; yielding an empty pairing here
        # would invent a grounding the source never made.
        if isinstance(value.unit_ref, Absent) or isinstance(value.unit_raw, Absent):
            continue
        # unit_raw is short, controlled-vocabulary text (unit symbols and
        # aliases) -- never excerpted prose -- so it is always left literal
        # regardless of reveal_text; see check_char_spans's own docstring.
        yield _TextPairing(
            f"{path}.unit_ref",
            value.unit_ref.node_id,
            value.unit_ref.locator,
            value.unit_raw,
            always_literal=True,
        )


def _label_token_policy(locator: object, fields: tuple[str, str]) -> tuple[str, str] | None:
    """The ``uncompensated_fields`` a label/token pairing carries, given its locator.

    Returns ``None`` -- meaning the char-span kernel in :func:`_replay_text_pairings`
    stays SILENT on a non-``CharSpanLocator`` and leaves the verdict to another gate
    -- ONLY when :func:`_verify_cited_cell_texts` will compare this cell, which is
    exactly a ``TableCellLocator`` carrying a PRESENT (``str``) inventory citation.
    That mirrors that gate's own admission test one-for-one (it ``continue``s past
    every non-``TableCellLocator`` and past a cell whose ``pdf_table_inventory_sha256``
    is :class:`Absent`, the non-PDF case).

    For every other shape -- a ``BBoxLocator``, an ``XPathLocator``, or a table cell
    with an ``Absent`` citation -- there is NO compensating gate, so the pairing keeps
    ``fields`` and the kernel keeps reporting it UNVERIFIABLE. Flipping those to
    ``None`` would be a silencing, not a correction: text nobody can prove would
    vanish from the report with no gate having checked it. PR #24 added the cell-text
    gate but left this policy hard-coded to ``fields`` everywhere; this narrows the
    flip to precisely the cells that gate covers.
    """
    if isinstance(locator, TableCellLocator) and isinstance(locator.pdf_table_inventory_sha256, str):
        return None
    # The OOXML (.docx) lane mirrors the PDF one exactly: a table cell carrying a PRESENT
    # ooxml_table_inventory_sha256 is compared by _verify_cited_ooxml_cell_texts, so the
    # char-span kernel stays silent and leaves the verdict to that gate.
    if isinstance(locator, TableCellLocator) and isinstance(locator.ooxml_table_inventory_sha256, str):
        return None
    return fields


def _dataset_text_pairings(envelope: DatasetEnvelope) -> Iterator[_TextPairing]:
    """Enumerate, BY HAND, every grounding pair a :class:`DatasetEnvelope` exposes.

    Hand-written and NOT derived from :func:`iter_source_refs` -- that is the
    entire point. The self-audit in :func:`_replay_text_pairings` cross-checks
    this enumeration against the generic walk, and a check is only independent
    of the thing it checks when the two are written separately. Deriving this
    list from the same walk would turn that audit into a tautology that passes
    by construction.
    """
    yield from _measured_value_text_pairings(envelope)
    for series in envelope.series:
        for axis in series.axes:
            yield _TextPairing(
                f"series[{series.series_id!r}].axes[{axis.axis_id!r}].label_ref",
                axis.label_ref.node_id,
                axis.label_ref.locator,
                axis.label_raw,
                uncompensated_fields=_label_token_policy(axis.label_ref.locator, ("label_ref", "label_raw")),
            )


def _condition_set_text_pairings(envelope: ConditionSetEnvelope) -> Iterator[_TextPairing]:
    """Enumerate, BY HAND, every grounding PAIR a :class:`ConditionSetEnvelope`
    exposes -- i.e. every ``SourceRef`` beside a recorded verbatim string it
    is supposed to ground. Deliberately excludes the TWO ref locations that
    are not pairs at all (``attribution_ref``, ``subject.reason_ref``) -- see
    :func:`replay_condition_set`'s own semantic-obligation enumerator for those.
    ``unextracted[*].statement_ref`` USED to be a third such location; it is now
    a grounding pair, beside ``statement_raw`` (the refused statement's own
    words), and is enumerated here like any other pair.

    Hand-written and NOT derived from :func:`iter_source_refs`, for exactly
    the reason :func:`_dataset_text_pairings` gives: a check is only
    independent of the thing it checks when the two are written separately.

    ``subject`` is a SUM type (:class:`DeviceClassDeclaration` |
    :class:`UnresolvedSubject`) and the two arms are MUTUALLY EXCLUSIVE:
    only the ``DeviceClassDeclaration`` arm contributes a pair here
    (``subject.label_ref``); the ``UnresolvedSubject`` arm's ``reason_ref``
    is unpaired and handled by the semantic-obligation enumerator instead.
    """
    if isinstance(envelope.subject, DeviceClassDeclaration):
        yield _TextPairing(
            "subject.label_ref",
            envelope.subject.label_ref.node_id,
            envelope.subject.label_ref.locator,
            envelope.subject.label_raw,
            uncompensated_fields=_label_token_policy(envelope.subject.label_ref.locator, ("label_ref", "label_raw")),
        )
    yield from _measured_value_text_pairings(envelope)
    for index, claim in enumerate(envelope.scalar_claims):
        yield _TextPairing(
            f"scalar_claims[{index}].label_ref",
            claim.label_ref.node_id,
            claim.label_ref.locator,
            claim.label_raw,
            uncompensated_fields=_label_token_policy(claim.label_ref.locator, ("label_ref", "label_raw")),
        )
    for index, categorical_claim in enumerate(envelope.categorical_claims):
        yield _TextPairing(
            f"categorical_claims[{index}].label_ref",
            categorical_claim.label_ref.node_id,
            categorical_claim.label_ref.locator,
            categorical_claim.label_raw,
            uncompensated_fields=_label_token_policy(categorical_claim.label_ref.locator, ("label_ref", "label_raw")),
        )
        yield _TextPairing(
            f"categorical_claims[{index}].token_ref",
            categorical_claim.token_ref.node_id,
            categorical_claim.token_ref.locator,
            categorical_claim.token_raw,
            uncompensated_fields=_label_token_policy(categorical_claim.token_ref.locator, ("token_ref", "token_raw")),
        )
    for index, statement in enumerate(envelope.unextracted):
        yield _TextPairing(
            f"unextracted[{index}].label_ref",
            statement.label_ref.node_id,
            statement.label_ref.locator,
            statement.label_raw,
            uncompensated_fields=_label_token_policy(statement.label_ref.locator, ("label_ref", "label_raw")),
        )
        yield _TextPairing(
            f"unextracted[{index}].statement_ref",
            statement.statement_ref.node_id,
            statement.statement_ref.locator,
            statement.statement_raw,
            uncompensated_fields=_label_token_policy(
                statement.statement_ref.locator, ("statement_ref", "statement_raw")
            ),
        )


def _semantic_ref_gap(
    ref: SourceRef,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
) -> SemanticGap:
    """Classify how much a REF-ONLY obligation (one with no recorded
    counterpart text to compare against -- see :func:`replay_condition_set`)
    is actually known to be true, independent of the derived value it is
    cited as supporting.

    Returns :attr:`SemanticGap.LOCATION_UNRESOLVED` when the span itself
    could not be independently re-sliced at all: the locator is not even a
    :class:`CharSpanLocator`, the node it targets has a store-integrity
    problem, the node's text was never independently verified, or the
    recorded offsets do not fit inside that re-verified text. Returns
    :attr:`SemanticGap.SUPPORT_UNRECORDED` only once the span itself is
    confirmed to re-slice cleanly -- STRICTLY MORE is known there than in
    the ``LOCATION_UNRESOLVED`` case, but still nothing that says the span
    MEANS the derived value, because (by construction, for the two ref
    locations that remain unpaired -- ``attribution_ref`` and
    ``subject.reason_ref``) nothing was ever recorded to compare it against.
    """
    node_problems = node_problems or {}
    if not isinstance(ref.locator, CharSpanLocator):
        return SemanticGap.LOCATION_UNRESOLVED
    if ref.node_id in node_problems:
        return SemanticGap.LOCATION_UNRESOLVED
    text = text_by_node_id.get(ref.node_id)
    if text is None:
        return SemanticGap.LOCATION_UNRESOLVED
    if not (0 <= ref.locator.start <= ref.locator.end <= len(text)):
        return SemanticGap.LOCATION_UNRESOLVED
    return SemanticGap.SUPPORT_UNRECORDED


def _claim_text(value: object) -> str:
    """Render a derived value for an :class:`UncheckedSemanticClaim`'s ``claim``
    field without trusting it to be a well-formed enum.

    Every caller here reads a field the schema types as an enum, and on a
    VALIDATED envelope it always is one. The public replay entry points accept
    an already-constructed object, though, so ``model_construct`` can hand this
    module a ``None`` or a string look-alike where an enum belongs. Reaching
    straight for ``.value`` would then raise ``AttributeError`` out of the
    middle of a replay -- and a replayer owes a verdict about broken input, not
    a traceback (the same rule that made the report's span arithmetic report
    rather than raise). Falling back to ``repr`` keeps the claim honest about
    what it actually found.
    """
    inner = getattr(value, "value", None)
    return inner if isinstance(inner, str) else repr(value)


def _kinds_admitting_unit(table: units.ConversionTable, unit_raw: str) -> tuple[QuantityKind, ...]:
    """Which :class:`~carmel.services.units.QuantityKind`\\ s the RECORDED table
    accepts ``unit_raw`` for, in declaration order, EXCLUDING
    :attr:`~carmel.services.units.QuantityKind.OTHER`.

    This is deliberately the INVERSE of the predicate
    :func:`verify_measured_value_unit` already enforces ("``unit_raw`` is a
    known unit or alias of THIS kind in the recorded table"). Asking it of
    every kind at once is what says whether that predicate discriminates at
    all: one admitting kind means the unit pins the quantity and the existing
    validator already closed the question; several means the unit is
    consistent with any of them and nothing in the envelope says which the
    paper meant.

    ``OTHER`` is excluded, and the exclusion is load-bearing rather than
    tidy-minded. ``OTHER`` is a wildcard for a quantity this codebase does not
    model, so it accepts ANY unit string -- measured, it admits ``'atm'``,
    ``'wibble'`` and ``'furlongs per fortnight'`` alike. Counting it would
    therefore put every value in the corpus at two-or-more admitting kinds and
    make this predicate fire everywhere, which is precisely the blanket rule
    this enumerator exists to avoid. ``OTHER`` needs no claim of its own
    either: :func:`verify_measured_value_unit_boundary` already reports it
    ``UNVERIFIABLE`` as an unmodelled quantity, so it is never silently clean.
    """
    admitting: list[QuantityKind] = []
    for kind in QuantityKind:
        if kind is QuantityKind.OTHER:
            continue
        try:
            units.normalize_unit(kind, unit_raw, table=table)
        except units.UnknownUnitError:
            continue
        admitting.append(kind)
    return tuple(admitting)


def _quantity_kind_claim(path: str, value: MeasuredValue) -> UncheckedSemanticClaim | None:
    """The obligation ``value.quantity_kind`` imposes, or ``None`` if the
    recorded table already pins it.

    ``quantity_kind`` is not merely descriptive: it decides which conversion
    the value admits and what ``"%"``, ``"1"`` and ``"ppm"` MEAN. Nothing in
    the envelope grounds the choice -- ``MeasuredValue`` carries a
    ``value_ref`` for the number and a ``unit_ref`` for the unit, and no ref
    at all for the quantity. What stands in for grounding is the table
    lookup, and it works only as far as the unit spelling discriminates.
    Where several kinds accept the same spelling it establishes nothing, and
    the report must say so rather than let a table lookup that could not
    fail read as a check that passed.

    Returns ``None`` for a NOT_PRINTED_IN_SOURCE value (I-060): there is no
    ``unit_raw`` spelling to look up at all, and the quantity's unproven-ness is
    already carried by the axis-level header-bridge claim
    (:func:`_dataset_axis_bridge_claims`). Filing a per-value claim here too would
    double-count the same gap.
    """
    if value.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE or isinstance(value.unit_raw, Absent):
        return None
    unit_raw = value.unit_raw
    try:
        table = units.table_for_sha(value.conversion_table_sha256)
    except units.UnknownConversionTableError:
        # Fail closed. verify_measured_value_unit already reports the
        # unresolvable sha as UNVERIFIABLE; what is added here is that the
        # AMBIGUITY question cannot be decided either, so the kind cannot be
        # called pinned. Staying silent would let an unresolvable table read
        # as an unambiguous one.
        return UncheckedSemanticClaim(
            claim_path=f"{path}.quantity_kind",
            claim=_claim_text(value.quantity_kind),
            gap=SemanticGap.NO_SUPPORT_OFFERED,
            reason=f"conversion_table_sha256={value.conversion_table_sha256!r} names no known "
            "conversion table, so whether the recorded unit spelling pins this quantity kind "
            "cannot be decided at all; no ref supports the choice independently",
        )
    admitting = _kinds_admitting_unit(table, unit_raw)
    if len(admitting) < 2:
        return None
    candidates = ", ".join(kind.value for kind in admitting)
    return UncheckedSemanticClaim(
        claim_path=f"{path}.quantity_kind",
        claim=_claim_text(value.quantity_kind),
        gap=SemanticGap.NO_SUPPORT_OFFERED,
        reason=f"unit_raw={unit_raw!r} is admitted by {len(admitting)} quantity kinds in the "
        f"recorded conversion table {value.conversion_table_sha256!r} ({candidates}), so the unit "
        "does not pin the quantity, and no ref supports the recorded choice: MeasuredValue grounds "
        "its number and its unit, never its quantity kind",
    )


def _uncertainty_claims(path: str, uncertainty: Uncertainty) -> tuple[UncheckedSemanticClaim, ...]:
    """The obligations an :class:`~carmel.schemas.datasets.Uncertainty`'s own
    fields impose. Its bounds are :class:`MeasuredValue`\\ s and are handled
    with every other measured value; these three fields are not.

    ``kind``, ``basis`` and ``scale`` decide how a downstream consumer may
    combine this figure with any other, and :class:`Uncertainty` carries no
    :class:`SourceRef` on any field -- measured, all three construct freely in
    every combination of bounds present and absent. This is
    :attr:`SemanticGap.NO_SUPPORT_OFFERED` in its purest form: not a location
    recorded without a meaning, but no location at all.

    Only CONCRETE assertions are reported. An :class:`Absent` ``basis`` or
    ``scale``, and the :attr:`UncertaintyKind.UNKNOWN` /
    :attr:`UncertaintyKind.UNSPECIFIED_PERCENTAGE` sentinels, are recorded
    REFUSALS -- the envelope declining to state what the paper never stated.
    Reporting those as unsupported claims would invert the narrow honest
    slice, treating the honest answer as the defect.
    """
    claims: list[UncheckedSemanticClaim] = []
    if uncertainty.kind not in (UncertaintyKind.UNKNOWN, UncertaintyKind.UNSPECIFIED_PERCENTAGE):
        claims.append(
            UncheckedSemanticClaim(
                claim_path=f"{path}.kind",
                claim=_claim_text(uncertainty.kind),
                gap=SemanticGap.NO_SUPPORT_OFFERED,
                reason="Uncertainty carries no SourceRef on any field, so nothing locates where "
                "the paper stated this statistical kind; it decides whether the figure may be "
                "combined as a standard deviation or a confidence interval",
            )
        )
    for field_name, field_value in (("basis", uncertainty.basis), ("scale", uncertainty.scale)):
        if isinstance(field_value, Absent):
            continue
        claims.append(
            UncheckedSemanticClaim(
                claim_path=f"{path}.{field_name}",
                claim=_claim_text(field_value),
                gap=SemanticGap.NO_SUPPORT_OFFERED,
                reason=f"Uncertainty carries no SourceRef on any field, so nothing locates where "
                f"the paper stated this {field_name}; it changes how the recorded magnitude is "
                "interpreted",
            )
        )
    return tuple(claims)


def _condition_set_uncertainty_sites(
    envelope: ConditionSetEnvelope,
) -> tuple[tuple[str, Uncertainty], ...]:
    """Every place a :class:`ConditionSetEnvelope` can hold an
    :class:`Uncertainty`, named BY HAND.

    Deliberately not derived from :func:`iter_uncertainties`, for the reason
    :func:`_condition_set_text_pairings` is not derived from
    :func:`iter_source_refs`: the two are reconciled against each other, and a
    check whose two sides come from one source is a tautology. Paths use the
    same index form the walk produces, so the comparison is exact rather than
    approximate.
    """
    sites: list[tuple[str, Uncertainty]] = []
    for index, claim in enumerate(envelope.scalar_claims):
        if isinstance(claim.uncertainty, Uncertainty):
            sites.append((f"scalar_claims[{index}].uncertainty", claim.uncertainty))
    return tuple(sites)


def _dataset_uncertainty_sites(envelope: DatasetEnvelope) -> tuple[tuple[str, Uncertainty], ...]:
    """Every place a :class:`DatasetEnvelope` can hold an :class:`Uncertainty`,
    named BY HAND. See :func:`_condition_set_uncertainty_sites` for why this is
    not derived from the walk.
    """
    sites: list[tuple[str, Uncertainty]] = []
    for series_index, series in enumerate(envelope.series):
        base = f"series[{series_index}]"
        for constant_index, constant in enumerate(series.constants):
            if isinstance(constant.uncertainty, Uncertainty):
                sites.append((f"{base}.constants[{constant_index}].uncertainty", constant.uncertainty))
        for point_index, point in enumerate(series.points):
            point_path = f"{base}.points[{point_index}]"
            for coord_index, coordinate in enumerate(point.coordinates):
                if isinstance(coordinate.uncertainty, Uncertainty):
                    sites.append((f"{point_path}.coordinates[{coord_index}].uncertainty", coordinate.uncertainty))
            for obs_index, observation in enumerate(point.observations):
                if isinstance(observation.uncertainty, Uncertainty):
                    sites.append((f"{point_path}.observations[{obs_index}].uncertainty", observation.uncertainty))
    return tuple(sites)


def _reconcile_uncertainty_sites(
    envelope: ConditionSetEnvelope | DatasetEnvelope,
    named_paths: AbstractSet[str],
) -> tuple[ReplayFinding, ...]:
    """Reconcile a hand-written uncertainty inventory against a fresh, generic
    walk of the envelope, in BOTH directions.

    A path the walk finds but the inventory never named is an obligation that
    would go unreported -- the failure mode a hand-written list has and a walk
    does not, and the whole reason this check exists. A path the inventory
    named but the walk cannot reach is the opposite failure: a stale or
    mistyped entry claiming coverage of something that is not there. Neither is
    softened into a count; each names its own path, because a bare "2 != 3"
    cannot be acted on.
    """
    findings: list[ReplayFinding] = []
    walked_paths = {path for path, _ in iter_uncertainties(envelope)}
    for path in sorted(walked_paths - set(named_paths)):
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason="the generic uncertainty walk reaches this path but the hand-written "
                "obligation inventory never named it, so its kind/basis/scale would go "
                "unreported: the inventory has gone stale against the schema",
            )
        )
    for path in sorted(set(named_paths) - walked_paths):
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason="the hand-written obligation inventory names this path but the generic "
                "walk cannot reach it, so the inventory claims coverage of an uncertainty that "
                "is not in this envelope",
            )
        )
    return tuple(findings)


def _derived_value_claims(
    envelope: ConditionSetEnvelope | DatasetEnvelope,
    uncertainty_sites: tuple[tuple[str, Uncertainty], ...],
) -> tuple[UncheckedSemanticClaim, ...]:
    """Every :attr:`SemanticGap.NO_SUPPORT_OFFERED` obligation this envelope
    imposes: a derived value that decides interpretation and that no ref
    supports.

    The hand-written judgment here is over FIELDS, not paths -- which fields
    are assertions about the source document (``quantity_kind``,
    ``Uncertainty.kind``/``basis``/``scale``) versus self-describing machinery
    (``conversion_table_sha256``, ``SourceRef.node_id``, every content
    address). That distinction is semantic and no walk can make it, which is
    why enumerating it by hand is the point rather than an inconvenience.
    Measured on a small envelope, treating EVERY ref-less field as an
    obligation instead would emit 122 claims to carry the 3 that decide
    anything.

    Measured-value paths come from :func:`iter_measured_values` on purpose: the
    per-field judgment is already hand-written in :func:`_quantity_kind_claim`,
    so a second hand-written path list would add brittleness without adding a
    check. Uncertainty sites are the reverse -- there is no per-field predicate
    to gate them, so the inventory IS the judgment and is reconciled against
    its own walk by :func:`_reconcile_uncertainty_sites`.
    """
    claims: list[UncheckedSemanticClaim] = []
    for path, value in iter_measured_values(envelope):
        claim = _quantity_kind_claim(path, value)
        if claim is not None:
            claims.append(claim)
    for path, uncertainty in uncertainty_sites:
        claims.extend(_uncertainty_claims(path, uncertainty))
    return tuple(claims)


def _condition_set_semantic_claims(
    envelope: ConditionSetEnvelope,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
) -> tuple[UncheckedSemanticClaim, ...]:
    """Enumerate, BY HAND, every SEMANTIC obligation a
    :class:`ConditionSetEnvelope` imposes -- the two ref locations that are
    NOT grounding pairs (see :func:`_condition_set_text_pairings`'s
    docstring): a located span exists, but nothing recorded says the span
    MEANS the derived value it is cited to support.

    ``unextracted[*].statement_ref`` was a THIRD such location until its own
    verbatim text (``statement_raw``) was recorded beside it; it is now a
    grounding pair checked by the pairing enumerator, not a semantic obligation,
    so it no longer yields a claim here. The statement's ``reason`` is a
    property of that recorded text, not a separate span-means-enum assertion the
    way ``attribution`` and ``subject.reason`` are.

    ``attribution_ref`` is unconditionally present on every legal envelope,
    so this always yields at least one claim -- see
    :func:`replay_condition_set`'s docstring for why ``overall_outcome`` is
    therefore UNVERIFIABLE for every condition set, by design.
    """
    claims: list[UncheckedSemanticClaim] = []
    claims.append(
        UncheckedSemanticClaim(
            claim_path="attribution",
            claim=envelope.attribution.value,
            gap=_semantic_ref_gap(envelope.attribution_ref, text_by_node_id, node_problems),
            reason="attribution_ref locates the span the attribution assertion was read from, "
            "but no recorded text stands beside it to compare against -- grounding a span proves "
            "only that the span exists, never that it means the asserted attribution",
            support_paths=("attribution_ref",),
        )
    )
    if isinstance(envelope.subject, UnresolvedSubject):
        claims.append(
            UncheckedSemanticClaim(
                claim_path="subject",
                claim=envelope.subject.reason.value,
                gap=_semantic_ref_gap(envelope.subject.reason_ref, text_by_node_id, node_problems),
                reason="subject.reason_ref locates the span the refusal reason was read from, but "
                "no recorded text stands beside it to compare against",
                support_paths=("subject.reason_ref",),
            )
        )
    # unextracted[*].statement_ref is deliberately NOT enumerated here: it now
    # sits beside statement_raw as a grounding pair (see
    # _condition_set_text_pairings), so the pairing enumerator compares its text
    # and this axis -- "a span located but nothing recorded says what it means"
    # -- no longer describes it. Leaving a claim here too would be the double
    # accounting the Verifier forbids: the same statement reported as both
    # compared and never checkable.
    return tuple(claims)


_UNPAIRED_REF_FIELDS = frozenset({"attribution_ref", "reason_ref"})
"""The ONLY ref fields in the condition-set graph with no recorded counterpart.

``attribution_ref`` supports a ``ConditionAttribution`` and ``reason_ref`` a
``SubjectRefusalReason`` -- each an ENUM no recorded span-text can attest to.
``statement_ref`` was a third such field until ``statement_raw`` was recorded
beside it; it is now a grounding PAIR and quote-checked like the rest, so it is
NOT in this set. Every OTHER ref field in that graph sits beside a sibling
holding the verbatim text read at the span (``label_raw``, ``token_raw``,
``raw_text``, ``unit_raw``, and now ``statement_raw``), which makes it a
grounding pair that MUST be quote-checked rather than merely located.

Written as a frozen literal, independently of both enumerators, ON PURPOSE.
It is what stops a paired ref from being quietly reclassified as
support-only: a path moved out of the pairing enumerator and into the
semantic-claim enumerator is still NAMED, so path reconciliation alone
accepts it, ``support_only_char_spans`` absorbs it, and the report can then
say VERIFIED without that quote ever having been compared. Deriving this set
from the enumerators would make the check a tautology that passes no matter
which bucket a ref was put in.
"""


def _reconcile_condition_set_refs(
    envelope: ConditionSetEnvelope,
    named_paths: set[str],
    support_paths: AbstractSet[str],
) -> tuple[ReplayFinding, ...]:
    """Reconcile every path :func:`_condition_set_text_pairings` and the
    semantic-obligation enumerator TOGETHER claim to have named against a
    generic, independent walk of the envelope.

    Written as a genuinely SEPARATE step from both enumerators above -- not
    derived from either -- so this is only an independent check because it
    is not the same code as the thing it checks (the contract's own framing:
    "a report cannot police its own producer, so the producer must police
    itself against an inventory"). Fails loudly, per unnamed path, rather
    than aggregating into one opaque count.
    """
    findings: list[ReplayFinding] = []
    for path, _ref in iter_source_refs(envelope):
        if path not in named_paths:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason="iter_source_refs reaches this SourceRef but neither the "
                    "condition-set pairing enumerator nor any UncheckedSemanticClaim's "
                    "support_paths names it -- an obligation this envelope imposes was never "
                    "discharged",
                )
            )

    # The OTHER direction. Cut once on the reasoning that it can never fire
    # alone -- a mistyped inventory path also leaves the real path unnamed, so
    # the check above would catch it. That reasoning was wrong, and the
    # counterexample is the WALKER: `iter_source_refs` traverses BaseModel,
    # dict, list and tuple, but not `set`/`frozenset`. A `frozenset[SourceRef]`
    # field would be invisible to the walk while the inventory still named its
    # paths, so every walked path stays named, the check above passes, and an
    # entire ref-bearing field goes unchecked in silence. A path the inventory
    # names that the walk cannot reach therefore means one of two things, and
    # both are defects: the inventory is stale, or the walker cannot see that
    # field's container shape.
    walked_paths = {path for path, _ref in iter_source_refs(envelope)}
    for path in sorted(named_paths - walked_paths):
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=path,
                reason="the replayer claims to have accounted for this ref path, but "
                "iter_source_refs cannot reach it -- either the hand-written inventory "
                "is stale, or the generic walk is blind to the container shape holding "
                "it, in which case other refs are going unchecked in silence",
            )
        )

    # A ref may be discharged as SUPPORT-ONLY only if it is one of the three
    # fields that genuinely has nothing recorded beside it. Everything else is
    # a grounding pair and owes a quote comparison. Without this, moving a
    # paired ref into the semantic-claim enumerator keeps it NAMED, lets
    # `support_only_char_spans` absorb it, and leaves `evidence_outcome` free
    # to say VERIFIED for a quote that was never compared to anything.
    for path in sorted(support_paths):
        if path.rsplit(".", 1)[-1] not in _UNPAIRED_REF_FIELDS:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason="reported as unverifiable SUPPORT for a derived value, but this "
                    "ref field sits beside a recorded string saying what was read at the "
                    "span -- it is a grounding pair and owes a quote comparison, so "
                    "recording it as support-only overstates what was checked",
                )
            )
    return tuple(findings)


def _replay_text_pairings(
    envelope: object,
    pairings: Iterator[_TextPairing],
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding] | None = None,
    *,
    reveal_text: bool = False,
    audit_against_source_refs: bool = True,
) -> tuple[int, int, list[ReplayFinding]]:
    """The evidence kernel: re-slice each pairing and compare, then audit
    the enumeration that produced them against a generic walk of ``envelope``.

    PRIVATE, and to stay private. Its ``envelope: object`` parameter is what
    makes it reusable across envelope types, and is also exactly why it must
    not be public: handed an arbitrary object it will happily produce a
    report whose ``ref_path`` strings name fields of something nobody
    checked. The public surface stays concrete -- :func:`check_char_spans`
    and the ``replay_*`` entry points -- so every report is anchored to a
    named envelope type.

    ``audit_against_source_refs`` (default ``True``, unchanged for every
    existing dataset call site) gates the internal
    ``accounted_for == total_char_span_refs`` self-audit below. That audit
    assumes every ``CharSpanLocator`` reachable via :func:`iter_source_refs`
    is either checked here or turned into a finding here -- true for a
    :class:`DatasetEnvelope`, where every char-span ref is paired, but FALSE
    for a :class:`ConditionSetEnvelope`, which has char-span refs
    (``attribution_ref``, ``subject.reason_ref``) that are legitimately
    UNPAIRED -- no recorded text stands beside them to compare against.
    (``unextracted[*].statement_ref`` used to belong on that list; it is now
    paired with ``statement_raw`` and checked like any other pair.) Passing ``False``
    skips that audit and reports ``total_char_span_refs`` as simply
    ``checked + len(findings)`` (i.e. exactly what THIS pairing set
    accounted for); the caller is then responsible for its own, separately
    written reconciliation against the full set of paths -- see
    :func:`replay_condition_set`. This keeps the reconciliation an
    independent check rather than folding it into the same audit this
    function already performs for datasets.
    """
    node_problems = node_problems or {}
    checked = 0
    findings: list[ReplayFinding] = []
    # Findings for pairings whose locator is not a CharSpanLocator and which
    # have no compensating gate (see _TextPairing): these do NOT correspond
    # to any CharSpanLocator reachable via iter_source_refs, so they must
    # stay OUT of `findings` -- the `accounted_for == total_char_span_refs`
    # self-audit below would falsely trip if they inflated `len(findings)`
    # against a `total_char_span_refs` count that never counted them in the
    # first place. Concatenated into the returned list only at the very end.
    non_char_span_findings: list[ReplayFinding] = []

    for pairing in pairings:
        path = pairing.path
        locator = pairing.locator
        if not isinstance(locator, CharSpanLocator):
            if pairing.uncompensated_fields is not None:
                ref_field, text_field = pairing.uncompensated_fields
                non_char_span_findings.append(
                    ReplayFinding(
                        category=ReplayOutcome.UNVERIFIABLE,
                        ref_path=path,
                        reason=f"{ref_field}.locator is a {type(locator).__name__}, not a "
                        f"CharSpanLocator -- {text_field} is text, and this replayer (there is no "
                        "renderer in this codebase) can only independently verify text through a "
                        "CharSpanLocator's re-sliced character span, so it cannot re-derive or "
                        f"confirm the text recorded in {text_field} from this locator kind",
                    )
                )
            continue
        node_id = pairing.node_id
        if node_id in node_problems:
            problem = node_problems[node_id]
            findings.append(ReplayFinding(category=problem.category, ref_path=path, reason=problem.reason))
            continue
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
            continue
        expected = pairing.expected
        if not (0 <= locator.start <= locator.end <= len(text)):
            # Python TRUNCATES an over-long slice instead of raising, so
            # `text[start:end]` with `end` past the end of the document still
            # returns a string -- and whenever the recorded quote happens to
            # be a suffix of that text, the truncated slice EQUALS it. Without
            # this check a locator claiming a 983-character span is counted as
            # a verified 3-character quote. The offsets are part of the claim,
            # so offsets that cannot describe this text are a FAILED claim
            # rather than a near miss. Mirrors `_semantic_ref_gap`'s bounds
            # test on purpose, so the paired and support-only paths agree on
            # what "re-sliceable" means.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason=(
                        "char-span offsets do not fit the independently re-verified "
                        f"evidence text: recorded span [{locator.start}, {locator.end}) "
                        f"against text of length {len(text)}"
                    ),
                )
            )
            continue
        actual = text[locator.start : locator.end]
        if actual != expected:
            literal = pairing.always_literal or reveal_text
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
            continue
        checked += 1

    accounted_for = checked + len(findings)
    if audit_against_source_refs:
        total_char_span_refs = sum(
            1 for _, ref in iter_source_refs(envelope) if isinstance(ref.locator, CharSpanLocator)
        )
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
    else:
        total_char_span_refs = accounted_for

    # Concatenated here, at the very end, and NOT earlier: `findings` alone
    # feeds `accounted_for` above, which is checked against
    # `total_char_span_refs` (a CharSpanLocator-only count). Merging
    # non_char_span_findings into `findings` before that check would inflate
    # accounted_for with findings that have no matching char-span ref and
    # falsely trip the self-audit -- keep the two lists separate up to here.
    return checked, total_char_span_refs, findings + non_char_span_findings


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

    Unlike ``value_ref``/``unit_ref`` -- each of which has its own
    compensating gate elsewhere in this module
    (:func:`verify_measured_value_value_boundary`,
    :func:`verify_measured_value_unit_boundary`) that reports UNVERIFIABLE
    for any non-``CharSpanLocator`` locator -- ``label_ref`` has no such
    compensating gate. So a ``label_ref`` whose locator is NOT a
    ``CharSpanLocator`` is reported UNVERIFIABLE right here, at the axis
    loop: ``label_raw`` is text, and this replayer (there is no renderer in
    this codebase) can only independently re-derive text through a char
    span, so a bbox/xpath/table-cell locator can attest a region or cell
    exists but can never attest to the text recorded in ``label_raw``.

    By default, a mismatch finding's ``expected``/``actual`` are redacted
    (see :func:`_redacted`) rather than literal, EXCEPT for the ``unit_ref``
    call site: ``unit_raw`` is short, controlled-vocabulary unit
    symbols/aliases (e.g. ``"K"``, ``"atm"``), never excerpted prose, so
    there is nothing there to protect and leaving it literal makes unit
    mismatches far easier to read. Pass ``reveal_text=True`` to opt into
    literal excerpts everywhere (e.g. for a human debugging a specific
    finding with the corpus license already accounted for).

    Returns ``(checked, total_char_span_refs, findings)``, where ``findings``
    also includes the UNVERIFIABLE findings for any non-``CharSpanLocator``
    ``label_ref`` described above; ``total_char_span_refs`` (and the
    ``checked``/``findings`` self-audit it feeds) counts only
    ``CharSpanLocator`` refs, so those non-char-span findings are excluded
    from that count on purpose.
    """
    return _replay_text_pairings(
        envelope,
        _dataset_text_pairings(envelope),
        text_by_node_id,
        node_problems,
        reveal_text=reveal_text,
    )


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
    if value.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE:
        # A not-printed value must carry NO unit token (I-060): its ``unit_raw`` is
        # Absent. A stored value declaring not-printed while CARRYING a ``unit_raw``
        # contradicts its own provenance -- the schema forbids it, but replay does
        # not trust the schema to have run and owes a verdict about broken input,
        # not silence that reads as "checked and clean". This is the mirror of the
        # printed branch below, which refuses an Absent token rather than
        # dereferencing it. FAILED, not UNVERIFIABLE: this invariant check runs to
        # completion and returns a definite negative -- nothing is missing that
        # stops it running (contrast the printed branch's Absent token, whose
        # absence removes the normalization check's input).
        if not isinstance(value.unit_raw, Absent):
            return ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=f"{path}.unit_raw",
                reason=f"unit_raw={value.unit_raw!r} is present while unit_provenance="
                f"{value.unit_provenance.value!r} -- a not-printed value carries no unit token (the "
                "source printed none), so this record contradicts its own provenance",
                expected="Absent",
                actual=repr(value.unit_raw),
            )
        # The source printed no unit token (I-060), so there is no ``unit_raw`` to
        # re-normalize. What IS re-verifiable is that the recorded
        # ``unit_normalized`` is the recorded table's own dimensionless base for
        # this quantity -- a check that still runs and can still FAIL if a stored
        # value's ``unit_normalized`` was tampered away from "1".
        base = table.base_unit(value.quantity_kind)
        if base != value.unit_normalized:
            return ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=f"{path}.unit_normalized",
                reason=f"unit_normalized={value.unit_normalized!r} disagrees with the recorded table's "
                f"base unit {base!r} for the not-printed dimensionless quantity_kind="
                f"{value.quantity_kind.value!r}",
                expected=base,
                actual=value.unit_normalized,
            )
        return None
    unit_raw = value.unit_raw
    if isinstance(unit_raw, Absent):
        # A printed-unit value with no recorded token is a contradiction the
        # schema forbids, but replay owes a verdict about broken input, not a
        # traceback: report it rather than dereference an Absent.
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=f"{path}.unit_raw",
            reason=f"unit_raw is Absent while unit_provenance={value.unit_provenance.value!r} is not "
            "not_printed_in_source -- a printed-unit value must carry its token, so its normalization "
            "cannot be re-verified",
        )
    try:
        expected_unit_normalized = units.normalize_unit(value.quantity_kind, unit_raw, table=table)
    except units.UnknownUnitError as exc:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=f"{path}.unit_raw",
            reason=f"unit_raw={unit_raw!r} is not a known unit or alias of quantity_kind="
            f"{value.quantity_kind.value!r} in the RECORDED conversion table "
            f"{value.conversion_table_sha256!r}: {exc}",
        )
    if expected_unit_normalized != value.unit_normalized:
        return ReplayFinding(
            category=ReplayOutcome.FAILED,
            ref_path=f"{path}.unit_normalized",
            reason=f"unit_normalized={value.unit_normalized!r} disagrees with the recorded table's "
            f"own normalization of unit_raw={unit_raw!r} for quantity_kind="
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
    this module's own docstring, M-D4 grouping -- for a CONDITION SET that
    association is now separately refutable by
    :func:`_refute_condition_set_stitching`, which refutes and never verifies,
    so a claim surviving it is still unproven here); that the original writer's
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
    if value.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE:
        # A not-printed value cites NO unit location (I-060): its ``unit_ref`` is
        # Absent. A stored value declaring not-printed while CARRYING a ``unit_ref``
        # contradicts its own provenance -- the schema forbids it, but replay does
        # not trust the schema to have run. This mirrors the printed branch below,
        # which refuses an Absent unit_ref rather than replaying it silently.
        # FAILED, not UNVERIFIABLE: the invariant runs to completion and returns a
        # definite negative, nothing prevents it from running.
        if not isinstance(value.unit_ref, Absent):
            return ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=f"{path}.unit_ref",
                reason=f"unit_ref is present while unit_provenance={value.unit_provenance.value!r} -- "
                "a not-printed value cites no unit location (the source printed no token to cite), so "
                "this record contradicts its own provenance",
            )
        # The source printed no unit token (I-060): there is no unit_ref locator
        # and no unit span for a boundary/admission gate to re-run against. This
        # is not a check that could not run -- there is genuinely nothing to
        # check, and verify_measured_value_unit has already re-verified the
        # dimensionless base. What the not-printed value does NOT establish -- that
        # this column carries the quantity -- is carried by the header-bridge
        # UncheckedSemanticClaim, not by a boundary finding here.
        return None
    path_ref = f"{path}.unit_ref"
    if isinstance(value.unit_ref, Absent):
        # A printed-unit value with no unit_ref is a contradiction the schema
        # forbids; report rather than dereference an Absent.
        return ReplayFinding(
            category=ReplayOutcome.UNVERIFIABLE,
            ref_path=path_ref,
            reason=f"unit_ref is Absent while unit_provenance={value.unit_provenance.value!r} is not "
            "not_printed_in_source -- a printed-unit value must cite its token's location, so the "
            "boundary/admission gate has nothing to re-run",
        )
    unit_node_id = value.unit_ref.node_id
    unit_locator = value.unit_ref.locator

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


#: How each non-``REPRODUCED`` inventory-verification status maps onto a replay
#: outcome. FAILED is reserved, in this module's vocabulary, for "a check RAN and
#: produced a definite disagreement"; every inability to reach a verdict is
#: UNVERIFIABLE, never conflated with VERIFIED. That line -- not retriability -- is
#: the one this table draws, exactly as ``test_quantity_kind_other_is_unverifiable_
#: not_failed`` draws it: a check that could not run is UNVERIFIABLE even when
#: re-running will never help, because it demonstrated nothing wrong.
#:
#: An EXPLICIT total dict, not a "failure set" with an ``else`` default. The old
#: form mapped two members to FAILED by name and swept everything else into
#: UNVERIFIABLE; a member added to the enum would then have inherited "could not
#: check" silently. Every member is named here and the subscript below carries no
#: default, so an unmapped member raises rather than being misfiled, and
#: ``test_every_inventory_status_maps_to_a_replay_outcome`` fails the moment one is
#: added and left out.
#:
#: * ``MISMATCHED`` -> FAILED: the recomputation ran and produced a different
#:   grid. The stored record is not evidence for the cells it claims.
#: * ``SOURCE_MISMATCH`` -> FAILED: the bytes are not the document the record
#:   is about. Unreachable on this path (bytes are keyed by the record's own
#:   ``raw_sha256``), but if it ever fires it is a real contradiction, not an
#:   inability -- so it fails rather than hiding as "could not check".
#: * ``PAYLOAD_UNREADABLE`` -> UNVERIFIABLE: this code cannot read the record's
#:   shape or version. It never reached a verdict about the grid; a schema-valid
#:   embedded record cannot reach here, but the honest report if it did is
#:   inability, not disagreement.
#: * ``ENGINE_UNAVAILABLE`` -> UNVERIFIABLE: this machine has no working PDF
#:   engine (pypdf absent, broken, or a gate the engine contradicted), so nothing
#:   ever looked at the document. A property of this machine, not of the record --
#:   fixing the machine and retrying the SAME bytes can still verify.
#: * ``EXTRACTION_FAILED`` -> UNVERIFIABLE, and the SAME outcome as
#:   ``ENGINE_UNAVAILABLE`` deliberately, argued here because the split ticket asks
#:   for the argument when it is not distinguished at this layer. The pinned engine
#:   was in hand and this document defeated it (not a PDF, truncated, unwalkable),
#:   so the grid comparison never RAN and nothing was demonstrated wrong. By this
#:   module's own line, that is UNVERIFIABLE, not FAILED -- the same call the OTHER
#:   test makes for a permanently-unmodelled unit: an inability that retrying will
#:   never cure is still an inability, and reporting it as a demonstrated
#:   disagreement would be the overclaim in the other direction. UNVERIFIABLE never
#:   passes the record either -- it is not conflated with VERIFIED. What the ticket
#:   needs distinguished IS distinguished, one layer up: the *status* separates the
#:   machine's fault (``ENGINE_UNAVAILABLE``, retry may verify) from the record's
#:   (``EXTRACTION_FAILED``, retry is pointless), and it rides into the finding's
#:   ``reason`` below, so a retry-driven caller reads the difference where it lives
#:   rather than inferring it from a coarse tri-state that was never about retries.
#: * ``IDENTITY_UNAVAILABLE`` -> UNVERIFIABLE: this machine cannot say what its
#:   own derivation code is (a source-stripped install), so no comparison would
#:   be honest.
_INVENTORY_STATUS_TO_OUTCOME: dict[InventoryVerificationStatus, ReplayOutcome] = {
    InventoryVerificationStatus.MISMATCHED: ReplayOutcome.FAILED,
    InventoryVerificationStatus.SOURCE_MISMATCH: ReplayOutcome.FAILED,
    InventoryVerificationStatus.PAYLOAD_UNREADABLE: ReplayOutcome.UNVERIFIABLE,
    InventoryVerificationStatus.ENGINE_UNAVAILABLE: ReplayOutcome.UNVERIFIABLE,
    InventoryVerificationStatus.EXTRACTION_FAILED: ReplayOutcome.UNVERIFIABLE,
    InventoryVerificationStatus.IDENTITY_UNAVAILABLE: ReplayOutcome.UNVERIFIABLE,
}


def _verify_embedded_inventories(
    table_inventories: tuple[EmbeddedTableInventory, ...],
    raw_bytes_by_sha: Mapping[str, bytes],
) -> list[ReplayFinding]:
    """Re-derive every embedded inventory's grid from the document's raw bytes.

    Iterates the EMBEDDED collection, which is unique by ``inventory_sha256``
    and (by T4/T5) exactly the set any ref cites -- so one PDF re-derivation
    per inventory, never one per citing cell. The bytes are the ones the replay
    loop already read and hash-verified; an inventory whose document has no
    hash-verified bytes in hand (raw.bin missing or tampered -- both already
    reported against the node) is an inability to re-derive, reported here as
    UNVERIFIABLE rather than passed over in silence.
    """
    findings: list[ReplayFinding] = []
    for inventory in table_inventories:
        ref_path = f"table_inventories[{inventory.inventory_sha256!r}]"
        raw = raw_bytes_by_sha.get(inventory.raw_sha256)
        if raw is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"inventory {inventory.inventory_sha256!r} names document "
                    f"{inventory.raw_sha256!r}, but no hash-verified raw bytes for that document are in "
                    "hand (its raw.bin is missing or tampered -- reported against the node), so its grid "
                    "could not be re-derived",
                )
            )
            continue
        # The embedded payload already parsed and self-cohered through T1 when
        # the envelope validated, so json.loads cannot honestly fail here; it is
        # guarded only so a corrupted-in-memory object degrades to UNVERIFIABLE
        # rather than an escaping crash.
        try:
            payload = json.loads(inventory.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"embedded inventory {inventory.inventory_sha256!r} canonical_json does not "
                    f"parse, so its grid could not be re-derived: {exc!r}",
                )
            )
            continue
        result = verify_inventory_record(payload, raw)
        if result.status is InventoryVerificationStatus.REPRODUCED:
            continue
        moved = f" (recorded identities moved: {', '.join(result.identity_moved)})" if result.identity_moved else ""
        # REPRODUCED returned above, so `result.status` is always a mapped non-success member;
        # the subscript raises rather than defaulting if a future member is left unmapped.
        category = _INVENTORY_STATUS_TO_OUTCOME[result.status]
        findings.append(
            ReplayFinding(
                category=category,
                ref_path=ref_path,
                reason=f"re-deriving the grid of inventory {inventory.inventory_sha256!r} from document "
                f"{inventory.raw_sha256!r} reported {result.status.value!r}: {result.detail}{moved}",
            )
        )
    return findings


class _CellTextVerification(NamedTuple):
    """What one sweep of :func:`_verify_cited_cell_texts` produced.

    Two values, not one, because a cell comparison has two outcomes worth
    carrying: what went WRONG (``findings`` -- FAILED disagreements and
    UNVERIFIABLE blocked comparisons) and how much went RIGHT (``checked`` --
    comparisons that ran and matched, whole-cell and exact). A matched cell used
    to produce no value at all: the loop fell through and the success vanished,
    so a report could compare 22 cells and count coverage of zero. ``checked``
    is exactly that lost quantity, recovered so the outcome machinery can treat
    a matched cell as the real check it is (see
    :attr:`ReplayReport.checked_table_cells`). It counts ONLY the exact-match
    branch -- never a skipped non-cell pairing, never a non-PDF cell with no
    grid to compare against; neither of those verified anything."""

    checked: int
    findings: tuple[ReplayFinding, ...]


def _verify_cited_cell_texts(
    pairings: Iterable[_TextPairing],
    embedded_by_sha: Mapping[str, EmbeddedTableInventory],
) -> _CellTextVerification:
    """Compare each table-cell-grounded verbatim text against the cell it cites.

    Runs over the SAME grounding pairs the char-span replayer uses -- every
    ``SourceRef`` beside the verbatim string it is meant to ground -- so the
    coverage is exactly ``raw_text``/``unit_raw``/every label and token/every
    unextracted statement's own words, and exactly NOT the two ref-only
    locations (attribution/reason) that ground a location rather than a quoted
    string. (``statement_ref`` joined the quoted-string side once ``statement_raw``
    recorded the refused statement's words beside it.) A cell is atomic and a
    ``TableCellLocator`` has no sub-cell addressing, so the only comparison that
    means what it says is EXACT equality of the whole cell text against the
    whole grounded string: a substring match would let ``raw_text="8"`` cite a
    cell reading ``"1-8"``, which is the very laundering this check abolishes.
    A disagreement is therefore FAILED and names the row, the column and both
    strings; a cell that exists but records no comparable string is
    UNVERIFIABLE (nothing to compare against, never a silent pass).

    Returns a :class:`_CellTextVerification`: the findings this always produced,
    PLUS a count of the comparisons that ran and matched. Only the exact-match
    branch increments that count -- every ``continue`` above it (a non-cell
    pairing, a non-PDF cell with no grid) verified nothing and is not a checked
    anything.
    """
    checked = 0
    findings: list[ReplayFinding] = []
    for pairing in pairings:
        locator = pairing.locator
        if not isinstance(locator, TableCellLocator):
            continue
        citation = locator.pdf_table_inventory_sha256
        if not isinstance(citation, str):
            # Absent citation (a non-PDF cell) -- its legality is the schema's
            # to judge; there is no grid here to compare against.
            continue
        inventory = embedded_by_sha.get(citation)
        if inventory is None:
            # V8 makes every present citation resolve to an embedded inventory,
            # so this is defensive: report inability rather than assume.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=pairing.path,
                    reason=f"cites inventory {citation!r}, which this envelope does not embed, so its cell "
                    "text could not be compared",
                )
            )
            continue
        cell = inventory.cell_text(row=locator.row, col=locator.col)
        if cell is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=pairing.path,
                    reason=f"cites row={locator.row}, col={locator.col} in inventory {citation!r}, whose "
                    "grid records no comparable text at that cell",
                )
            )
            continue
        if cell != pairing.expected:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=pairing.path,
                    reason=f"cites row={locator.row}, col={locator.col} in inventory {citation!r}, whose "
                    f"cell text {cell!r} is not the recorded text {pairing.expected!r} -- a table-cell "
                    "citation names a whole cell, and the value's text is not what that cell says",
                    expected=pairing.expected,
                    actual=cell,
                )
            )
            continue
        # The exact whole-cell match -- the ONLY branch that ESTABLISHED the
        # recorded text. Of the `continue`s above, most never reached a
        # comparison at all (a non-cell pairing, a non-PDF cell, a missing
        # inventory, a missing cell); the disagreement branch is different --
        # it DID compare this cell and found the texts differ, and it reports
        # that as its own FAILED finding. `checked` counts agreements, not
        # comparisons, so neither kind of `continue` grows it: a disagreement
        # proves something, but it does not prove the recorded text.
        checked += 1
    return _CellTextVerification(checked=checked, findings=tuple(findings))


#: OOXML-lane counterpart to :data:`_INVENTORY_STATUS_TO_OUTCOME`. The ``.docx`` reader needs
#: no external engine (no pypdf), so its verification has no ENGINE_UNAVAILABLE member; the four
#: it does have map the same way -- a re-derivation that disagrees is a FAILED, an unreadable
#: embedded payload is UNVERIFIABLE.
_OOXML_INVENTORY_STATUS_TO_OUTCOME: dict[OoxmlInventoryVerificationStatus, ReplayOutcome] = {
    OoxmlInventoryVerificationStatus.MISMATCHED: ReplayOutcome.FAILED,
    OoxmlInventoryVerificationStatus.SOURCE_MISMATCH: ReplayOutcome.FAILED,
    OoxmlInventoryVerificationStatus.PAYLOAD_UNREADABLE: ReplayOutcome.UNVERIFIABLE,
}


def _verify_embedded_ooxml_inventories(
    ooxml_table_inventories: tuple[EmbeddedOoxmlTableInventory, ...],
    raw_bytes_by_sha: Mapping[str, bytes],
) -> list[ReplayFinding]:
    """Re-derive every embedded OOXML inventory's grid from the document's raw bytes.

    The OOXML-lane counterpart to :func:`_verify_embedded_inventories`. Iterates the EMBEDDED
    collection (unique by ``inventory_sha256`` and, by T4b/T5b, exactly the set any ref cites),
    re-reads each ``.docx`` from the hash-verified bytes the replay loop already holds, and
    reports a grid that does not reproduce. An inventory whose document has no hash-verified
    bytes in hand is an inability to re-derive, reported UNVERIFIABLE rather than passed over.
    """
    findings: list[ReplayFinding] = []
    for inventory in ooxml_table_inventories:
        ref_path = f"ooxml_table_inventories[{inventory.inventory_sha256!r}]"
        raw = raw_bytes_by_sha.get(inventory.source_sha256)
        if raw is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"inventory {inventory.inventory_sha256!r} names document "
                    f"{inventory.source_sha256!r}, but no hash-verified raw bytes for that document are in "
                    "hand (its raw.bin is missing or tampered -- reported against the node), so its grid "
                    "could not be re-derived",
                )
            )
            continue
        # The embedded payload already parsed and self-cohered through T1 at envelope
        # validation, so json.loads cannot honestly fail here; guarded only so a
        # corrupted-in-memory object degrades to UNVERIFIABLE rather than an escaping crash.
        try:
            payload = json.loads(inventory.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"embedded inventory {inventory.inventory_sha256!r} canonical_json does not "
                    f"parse, so its grid could not be re-derived: {exc!r}",
                )
            )
            continue
        result = verify_ooxml_inventory_record(payload, raw)
        if result.status is OoxmlInventoryVerificationStatus.REPRODUCED:
            continue
        # REPRODUCED returned above, so status is always a mapped non-success member; the
        # subscript raises rather than defaulting if a future member is left unmapped.
        category = _OOXML_INVENTORY_STATUS_TO_OUTCOME[result.status]
        findings.append(
            ReplayFinding(
                category=category,
                ref_path=ref_path,
                reason=f"re-deriving the grid of inventory {inventory.inventory_sha256!r} from document "
                f"{inventory.source_sha256!r} reported {result.status.value!r}: {result.detail}",
            )
        )
    return findings


def _verify_cited_ooxml_cell_texts(
    pairings: Iterable[_TextPairing],
    ooxml_embedded_by_sha: Mapping[str, EmbeddedOoxmlTableInventory],
) -> _CellTextVerification:
    """Compare each OOXML-table-cell-grounded verbatim text against the cell it cites.

    The OOXML-lane counterpart to :func:`_verify_cited_cell_texts`, with the identical
    whole-cell exact-equality contract: a disagreement is FAILED and names row, column and both
    strings; a cell with no comparable string is UNVERIFIABLE; only the exact match increments
    ``checked``. It reads ``ooxml_table_inventory_sha256`` and resolves against the OOXML
    carrier, so a PDF-cited cell (handled by the PDF gate) is passed over here.
    """
    checked = 0
    findings: list[ReplayFinding] = []
    for pairing in pairings:
        locator = pairing.locator
        if not isinstance(locator, TableCellLocator):
            continue
        citation = locator.ooxml_table_inventory_sha256
        if not isinstance(citation, str):
            # Absent OOXML citation (a PDF or non-cell cell) -- no .docx grid here to compare.
            continue
        inventory = ooxml_embedded_by_sha.get(citation)
        if inventory is None:
            # V8b makes every present citation resolve to an embedded inventory, so this is
            # defensive: report inability rather than assume.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=pairing.path,
                    reason=f"cites ooxml inventory {citation!r}, which this envelope does not embed, so its "
                    "cell text could not be compared",
                )
            )
            continue
        cell = inventory.cell_text(row=locator.row, col=locator.col)
        if cell is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=pairing.path,
                    reason=f"cites row={locator.row}, col={locator.col} in ooxml inventory {citation!r}, whose "
                    "grid records no comparable text at that cell",
                )
            )
            continue
        if cell != pairing.expected:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=pairing.path,
                    reason=f"cites row={locator.row}, col={locator.col} in ooxml inventory {citation!r}, whose "
                    f"cell text {cell!r} is not the recorded text {pairing.expected!r} -- a table-cell "
                    "citation names a whole cell, and the value's text is not what that cell says",
                    expected=pairing.expected,
                    actual=cell,
                )
            )
            continue
        checked += 1
    return _CellTextVerification(checked=checked, findings=tuple(findings))


def _claims_digitized(source_form: object) -> bool:
    """Whether ``source_form`` claims to be DIGITIZED, by the SAME value-equality test the VALUE-REF
    validator uses -- deliberately ``==``, never ``is``.

    ``SourceForm`` is a ``StrEnum``, so a member equals its own string value, and the value-ref
    validator this selector mirrors decides a series IS digitized with ``==``
    (:func:`~carmel.schemas.datasets._check_source_form_for_ref`, which then constrains that series'
    value refs). It is that validator specifically, not the schema as a whole: the OTHER figure
    validator, :func:`~carmel.schemas.datasets._validate_series_digitization_citation`, compares
    ``source_form`` with ``is`` -- safe THERE because a series reaching it has been through
    ``model_validate`` and its ``source_form`` is already a coerced member, which is exactly the
    coercion the ``model_construct`` route below skips and this selector must therefore stand in
    for. A series reaching replay through
    ``model_construct`` -- the case this whole service exists for -- can carry the plain string
    ``"digitized"`` in ``source_form``, and to validation that object IS digitized. Comparing with
    ``is`` here would disagree: identity fails for a plain string, so the figure lane would wave the
    series past both its citation re-derivation AND the attestation claim that keeps a figure-bearing
    envelope off a VERIFIED verdict -- a lie about where a number came from rewarded with a stronger
    verdict than the truth. ``==`` fails closed: a value the schema treats as digitized is treated as
    digitized here too, verified and attested like any other.

    The ``is``-then-value split IS the module's idiom for :class:`ReplayOutcome` (see
    :meth:`ReplayFinding.__post_init__`) -- but it is safe there ONLY because a construction-time
    ``isinstance`` guard refuses a non-member before any ``is`` runs. ``source_form`` on a
    ``model_construct`` envelope passed no such guard, so the comparison itself has to carry the
    weight, and value equality is the test that does.
    """
    return source_form == SourceForm.DIGITIZED


def _recognised_source_form(source_form: object) -> bool:
    """Whether ``source_form`` equals ANY :class:`SourceForm` member, by the same value-equality
    :func:`_claims_digitized` uses.

    A ``model_construct`` series carrying the plain string of a real form (``"tabular"``,
    ``"textual"``, ``"digitized"``) is recognised, exactly as :func:`_claims_digitized` recognises
    the digitized string. Only a value that is no known form at all -- neither a member nor a
    member's string -- is unrecognised, and replay cannot then say what that series claims its
    numbers were read from: neither the figure joins (which need it DIGITIZED) nor the non-digitized
    citation rule (which needs it some real non-figure form) can be re-derived for it. The caller
    files that inability as UNVERIFIABLE rather than let the series slip silently down the
    non-digitized path with no finding on any axis.
    """
    return any(source_form == form for form in SourceForm)


def _verify_figure_digitizations(
    envelope: DatasetEnvelope,
    raw_bytes_by_sha: Mapping[str, bytes],
) -> list[ReplayFinding]:
    """Re-derive at replay every join the figure validators assert at construction.

    The figure lane's counterpart to :func:`_verify_embedded_inventories` and
    :func:`_verify_cited_cell_texts` together, and it exists for the same reason: an
    envelope reaching replay by a route that skipped model validation (``model_construct``,
    an in-memory object handed straight here) carries figure citations that FD1-FD3, V9 and
    :class:`EmbeddedFigureDigitization`'s own T1 fired on only at construction. Replay repeats
    them from the envelope's own material -- the embedded record's canonical bytes are re-hashed
    (never trusted), the record is reconstructed by :meth:`FigureDigitization.from_payload`
    (never read off the object's accessors, which would re-run that same T1 and merely RAISE),
    and V9's series-record-crop joins are re-walked over the source graph.

    Emits :class:`ReplayFinding`\\ s ONLY, and DELIBERATELY no ``checked`` count. Nothing in
    this codebase re-derives markers from a crop's pixels, so a figure check that passed would
    be attesting that a citation is coherent, never that a measurement reproduced -- letting it
    increment a "how much was checked" counter would let a replay whose only evidence is a figure
    read VERIFIED, the one outcome this design rules out. A passing figure join therefore adds
    nothing to any count and appears in the report only as the ABSENCE of a finding; what it
    cannot establish is stated on the semantic axis by :func:`_figure_semantic_claims`.

    Two failure vocabularies, drawn on the module's own line (see :class:`ReplayOutcome`):
    FAILED where a check RAN and disagreed (a record that does not live at its address, a
    citation the envelope's own cover does not honour, a V9 join that does not hold), and
    UNVERIFIABLE where a check could not run (canonical_json that will not parse, or a document
    whose hash-verified raw bytes are not in hand -- reported explicitly, never a silent pass,
    exactly as :func:`_verify_embedded_inventories` reports its missing-bytes branch). Every
    source-graph lookup is guarded so a corrupted-in-memory envelope degrades to an UNVERIFIABLE
    finding rather than an escaping crash.
    """
    findings: list[ReplayFinding] = []
    graph = envelope.source_graph
    node_ids = graph.node_ids
    embedded_shas = {digitization.digitization_sha256 for digitization in envelope.figure_digitizations}
    records_by_sha: dict[str, FigureDigitization] = {}

    # Pass 1: re-derive each embedded record from its OWN bytes -- address, reconstruction,
    # document digest, and hash-verified bytes in hand. Iterates the embedded collection
    # (unique by digitization_sha256), so one re-derivation per record, never one per citing
    # series.
    for digitization in envelope.figure_digitizations:
        ref_path = f"figure_digitizations[{digitization.digitization_sha256!r}]"
        # The embedded payload already parsed and self-cohered through T1 when the envelope
        # validated -- but this replay exists for the envelope that never validated, so nothing
        # here trusts that. json.loads is guarded so a payload that will not parse degrades to
        # UNVERIFIABLE rather than an escaping crash.
        try:
            payload = json.loads(digitization.canonical_json)
        except (json.JSONDecodeError, RecursionError, TypeError) as exc:
            # TypeError is the model_construct case the JSONDecodeError/RecursionError pair missed: a
            # canonical_json of None (or any non-str/bytes) raises TypeError from json.loads, which
            # must degrade to a finding, not escape as a traceback -- exactly the policy the
            # reconstruction site fifteen lines below already holds for compute_digitization_sha.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"embedded figure digitization {digitization.digitization_sha256!r} canonical_json "
                    f"does not parse, so its record could not be re-derived: {exc!r}",
                )
            )
            continue
        # Recomputed from the payload, which re-canonicalizes and re-hashes -- so this cannot be
        # satisfied by a non-canonical rendering carrying an honestly computed address, and it
        # subsumes T1's canonical-rendering check without a second idea of what canonical means.
        try:
            recomputed = compute_digitization_sha(payload)
        except (RecursionError, TypeError, ValueError, OverflowError) as exc:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"embedded figure digitization {digitization.digitization_sha256!r} canonical_json "
                    f"could not be re-canonicalized to recompute its address, so it could not be re-checked: "
                    f"{exc!r}",
                )
            )
            continue
        if recomputed != digitization.digitization_sha256:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=ref_path,
                    reason=f"figure digitization canonical_json's bytes hash to {recomputed!r}, so this record "
                    f"does not live at the address {digitization.digitization_sha256!r} it claims",
                    expected=digitization.digitization_sha256,
                    actual=recomputed,
                )
            )
            continue
        # The record is reconstructed by from_payload, which re-runs D0-D9 -- so a payload
        # claiming complete coverage while carrying an omission, or one whose census does not
        # balance against its recovered count, is refused HERE, on the bytes. Asked of the record
        # module so the replayer carries no second idea of what a coherent digitization is.
        try:
            record = FigureDigitization.from_payload(payload)
        except UnknownPayloadVersion as exc:
            # The one case in this reconstruction that is the READER's limit, not the record's:
            # a payload_version this code does not know. Reported UNVERIFIABLE, not FAILED -- an
            # unrecognised version may be a NEWER record, not a wrong one, and accusing it of a
            # disagreement no reader here demonstrated is the same overclaim as verifying something
            # unchecked, merely pointed at the corpus. Caught BEFORE the UNREADABLE_PAYLOAD arm
            # below on purpose: UnknownPayloadVersion is a ValueError subclass and so is IN that
            # tuple, so the specific arm must precede the general one or every version mismatch
            # would fall through to FAILED. The reason names the version, from the exception's own
            # attribute rather than its formatted message.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"figure digitization {digitization.digitization_sha256!r} declares payload_version "
                    f"{exc.version!r}, a shape this reader does not know (it reads version {exc.readable_version!r}) "
                    "-- the record may be newer rather than wrong, so its coverage claim could not be checked here, "
                    "which is an admission about the reader and not a charge against the record",
                )
            )
            continue
        except UNREADABLE_PAYLOAD as exc:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=ref_path,
                    reason=f"figure digitization {digitization.digitization_sha256!r} does not reconstruct, so "
                    f"its coverage claim is not one anything could act on: {exc!r}",
                )
            )
            continue
        if record.raw_sha256 != digitization.raw_sha256:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=ref_path,
                    reason=f"figure digitization {digitization.digitization_sha256!r} record names document "
                    f"{record.raw_sha256!r}, not the declared raw_sha256 {digitization.raw_sha256!r} an "
                    f"envelope-level join would trust",
                    expected=digitization.raw_sha256,
                    actual=record.raw_sha256,
                )
            )
            continue
        if digitization.raw_sha256 not in raw_bytes_by_sha:
            # No re-derivation of the crop's pixels is possible or attempted; what is required is
            # that the document the record chains to is one whose bytes the replay loop actually
            # hash-verified. Its absence (raw.bin missing or tampered -- reported against the node)
            # is an inability to reach the chain's root, UNVERIFIABLE and never a silent pass.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=ref_path,
                    reason=f"figure digitization {digitization.digitization_sha256!r} names document "
                    f"{digitization.raw_sha256!r}, but no hash-verified raw bytes for that document are in "
                    "hand (its raw.bin is missing or tampered -- reported against the node), so the chain to "
                    "the crop's root document could not be confirmed",
                )
            )
            # Not a `continue`: the series-record-crop joins below need no bytes and are still
            # worth re-deriving, so the record is registered even when its root bytes are absent.
        records_by_sha[digitization.digitization_sha256] = record

    # Pass 2: re-walk V9's series-record-crop joins and the exact-cover rule (FD1). Runs over the
    # envelope's own series, so a citation that resolves to no embedded record and a record no
    # series cites are each caught here -- the two directions of the cover FD1 asserts.
    #
    # Deliberately NOT re-derived here, and left to the schema: FD2 (no two embedded records share a
    # digitization_sha256, _validate_figure_digitizations_no_duplicate_sha256) and FD3 (the embedded
    # records are stored in sorted address order, _validate_figure_digitizations_sorted). Both are
    # canonical-form rules about the embedded collection's SHAPE, not about whether a citation is
    # TRUE -- a duplicate or an out-of-order record cannot make a coherent citation resolve to the
    # wrong bytes (the address is content-derived and re-hashed in Pass 1). Replaying them would add
    # no falsification a corrupted-in-memory envelope could fail that the joins above do not already
    # catch, so their absence from replay is a decision on the record, not a gap in coverage.
    cited: set[str] = set()
    for index, series in enumerate(envelope.series):
        series_path = f"series[{index}]"
        if not _claims_digitized(series.source_form):
            if not _recognised_source_form(series.source_form):
                # A source_form that is neither DIGITIZED nor any other known member (a
                # model_construct envelope carries whatever it was handed, uncoerced). The two
                # branches below each rest on knowing what the series CLAIMS -- the figure joins on
                # its being DIGITIZED, the non-digitized rule on its being some real non-figure form
                # -- and neither is true here. Replay cannot say what this series claims its numbers
                # were read from, so it files that inability rather than silently take the
                # non-digitized path, which would let an unclassifiable series through with no
                # finding on any axis. UNVERIFIABLE, never FAILED: nothing is DEMONSTRATED wrong,
                # the check simply could not run -- the module's own line for the two verdicts.
                findings.append(
                    ReplayFinding(
                        category=ReplayOutcome.UNVERIFIABLE,
                        ref_path=series_path,
                        reason=f"series {series.series_id!r} has source_form={series.source_form!r}, which is no "
                        "recognised source form -- neither DIGITIZED nor any other member -- so replay cannot tell "
                        "what this series claims its numbers were read from, and could re-derive neither the figure "
                        "joins nor the non-digitized citation rule for it",
                    )
                )
                continue
            # V9's OTHER branch (_validate_series_digitization_citation, datasets.py:6471-6484), the
            # one the DIGITIZED path above is the mirror of: a series that is NOT digitized must cite
            # no figure record at all, and if its citation is Absent that absence must read
            # NOT_APPLICABLE. Replay re-derives both halves here so a series lying AWAY from
            # DIGITIZED -- keeping a non-digitized source_form while pointing at a figure record --
            # is examined at all; without this the figure lane iterates only the series that CLAIM
            # to be digitized, so such a series is looked at by no check on any axis and launders a
            # VERIFIED verdict for numbers that name the plotted curve they came from. FAILED, not
            # UNVERIFIABLE: source_form and the citation are both in hand, so the check RUNS and
            # DEMONSTRATES a disagreement with an invariant the schema states -- the module's own
            # line for FAILED (a claim the envelope's own cover does not honour), never the
            # could-not-run case UNVERIFIABLE is reserved for.
            findings.extend(_verify_non_digitized_series_cites_nothing(series, series_path))
            continue
        citation = series.digitization_sha256
        if not isinstance(citation, str):
            # A DIGITIZED series citing nothing is exactly the hole the figure lane was built to
            # close; V9 forbids it, and a model_construct envelope can still carry it. The record
            # it should have named is absent, so the joins below could not run.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=series_path,
                    reason=f"series {series.series_id!r} is DIGITIZED but cites no figure digitization, so its "
                    "record could not be re-checked",
                )
            )
            continue
        cited.add(citation)
        cited_record = records_by_sha.get(citation)
        if cited_record is None:
            if citation not in embedded_shas:
                findings.append(
                    ReplayFinding(
                        category=ReplayOutcome.FAILED,
                        ref_path=series_path,
                        reason=f"series {series.series_id!r} cites figure digitization {citation!r}, which this "
                        "envelope does not embed -- a citation resolvable only against an evidence store makes "
                        "the envelope's meaning depend on the machine replaying it",
                    )
                )
            # else: the record IS embedded but failed its own re-derivation above (address,
            # reconstruction, document digest); that finding already speaks, so no second one here.
            continue
        findings.extend(_verify_series_digitization_joins(series, series_path, citation, cited_record, graph, node_ids))

    for digitization in envelope.figure_digitizations:
        if digitization.digitization_sha256 not in cited:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=f"figure_digitizations[{digitization.digitization_sha256!r}]",
                    reason=f"figure digitization {digitization.digitization_sha256!r} is embedded but no DIGITIZED "
                    "series cites it -- the embedded set must cover EXACTLY the cited addresses, and a record "
                    "nothing cites is decorative provenance no series's partialness rests on",
                )
            )
    return findings


def _verify_non_digitized_series_cites_nothing(series: Series, series_path: str) -> list[ReplayFinding]:
    """Re-derive V9's non-DIGITIZED branch for one series that does NOT claim to be digitized.

    The counterpart to :func:`_verify_series_digitization_joins`, and V9's own two-part rule for a
    series whose ``source_form`` is anything but ``DIGITIZED``
    (:func:`~carmel.schemas.datasets._validate_series_digitization_citation`, datasets.py:6471-6484):

    * it must cite NO figure digitization -- ``digitization_sha256`` must be ``Absent``; a present
      citation is one only a digitized series may carry, and a non-digitized series pointing at a
      figure record is claiming its numbers came off a plotted curve while labelling them as
      something else;
    * and if that citation is ``Absent``, the absence must read ``NOT_APPLICABLE`` -- the only true
      reason a series that was never digitized has no digitization, as opposed to one that borrows a
      reason (``NOT_REPORTED_HERE``, ``UNRESOLVED``) recording a gap the form does not have.

    Both disagreements are ``FAILED``: the series' ``source_form`` and its citation are both in hand,
    so each check RUNS and DEMONSTRATES a violation of an invariant the schema states -- never the
    could-not-run case ``UNVERIFIABLE`` is reserved for. ``series_path`` names the offending series;
    the reason carries its ``series_id`` so a report reader sees which series lied.

    A non-``Absent``, non-``str`` citation (a ``model_construct`` envelope carries whatever it was
    handed) is still a citation the series must not have, so it too fails the first check -- the test
    is "is the citation truly absent", not "is it a well-formed digest".
    """
    citation = series.digitization_sha256
    if not isinstance(citation, Absent):
        return [
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=series_path,
                reason=f"series {series.series_id!r} has source_form={series.source_form!r} but cites figure "
                f"digitization {citation!r} -- only a DIGITIZED series may cite one, so a non-digitized series "
                "pointing at a figure record claims its numbers came off a plotted curve while labelling them "
                "otherwise",
            )
        ]
    if citation.reason is not AbsenceReason.NOT_APPLICABLE:
        return [
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=series_path,
                reason=f"series {series.series_id!r} has source_form={series.source_form!r} and an Absent "
                f"digitization_sha256 whose reason is {citation.reason.value!r} -- the only true absence for a "
                f"series that was not digitized is {AbsenceReason.NOT_APPLICABLE.value!r}",
            )
        ]
    return []


def _verify_series_digitization_joins(
    series: Series,
    series_path: str,
    citation: str,
    record: FigureDigitization,
    graph: object,
    node_ids: AbstractSet[str],
) -> list[ReplayFinding]:
    """Re-derive V9's six series-record-crop joins for one DIGITIZED ``series``.

    Split out of :func:`_verify_figure_digitizations` so the per-series joins read as one unit:
    the record is about THIS series (``series_id``), the count its census balances against is the
    count the series has (``recovered``), the crop it names resolves to a ``FIGURE_CROP`` node
    whose ``sha256`` matches, every point grounds in that ONE crop, and the record's document
    digest reaches the crop's ROOT artifact. Each disagreement is a FAILED finding; a crop lookup
    that cannot be walked at all is UNVERIFIABLE. ``graph`` is a
    :class:`~carmel.schemas.datasets.SourceGraph`, typed loosely only so this module need not
    import it solely to annotate a re-walk of V9's own logic.
    """
    findings: list[ReplayFinding] = []
    if record.series_id != series.series_id:
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=series_path,
                reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose record is "
                f"about series {record.series_id!r} -- a digitization must name the series it recovered",
                expected=series.series_id,
                actual=record.series_id,
            )
        )
    if record.recovered != len(series.points):
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=series_path,
                reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose record "
                f"recovered {record.recovered} marker(s), but the series has {len(series.points)} point(s) -- "
                "the count the record's census balances against must be the count the series actually has",
                expected=str(len(series.points)),
                actual=str(record.recovered),
            )
        )
    if record.figure_crop_node_id not in node_ids:
        findings.append(
            ReplayFinding(
                category=ReplayOutcome.FAILED,
                ref_path=series_path,
                reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose record names "
                f"figure_crop_node_id={record.figure_crop_node_id!r}, which is not the id of any node in this "
                "envelope's source graph -- a digitization must name a crop this envelope actually carries",
            )
        )
    else:
        # ``node()`` scans the same ``nodes`` tuple ``node_ids`` is derived from, so membership
        # here guarantees it returns -- no try/except, because there is no reachable state where
        # the id is in ``node_ids`` and ``node()`` still raises.
        crop_node = graph.node(record.figure_crop_node_id)  # type: ignore[attr-defined]
        if crop_node.kind is not SourceNodeKind.FIGURE_CROP:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=series_path,
                    reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose "
                    f"figure_crop_node_id {record.figure_crop_node_id!r} resolves to a "
                    f"{crop_node.kind.value!r} node, not a FIGURE_CROP -- a digitization is read off a "
                    "figure crop and off nothing else",
                )
            )
        elif crop_node.sha256 != record.figure_crop_sha256:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=series_path,
                    reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose record "
                    f"names figure_crop_sha256={record.figure_crop_sha256!r}, but crop node "
                    f"{record.figure_crop_node_id!r} has sha256={crop_node.sha256!r} -- the two halves of "
                    "the crop's identity must agree",
                    expected=record.figure_crop_sha256,
                    actual=crop_node.sha256,
                )
            )
    # The same-crop join: every digitized point grounds in the ONE crop the record names. Uses
    # V9's own scope (_iter_series_point_value_refs) so it cannot drift from what V9 checks, and
    # reports the FIRST offending ref rather than one finding per point, exactly as V9 raises on
    # the first.
    for where, ref in _iter_series_point_value_refs(series):
        if ref.node_id != record.figure_crop_node_id:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=series_path,
                    reason=f"series {series.series_id!r} cites figure digitization {citation!r} over crop "
                    f"{record.figure_crop_node_id!r}, but {where} grounds in crop {ref.node_id!r} -- a series "
                    "assembled from two different crops is not one digitization",
                )
            )
            break
    # Root bytes: the record's document digest must reach the crop's ROOT artifact, not stop at an
    # intermediate derived one. Re-walks V9's ancestry join; a graph that cannot be walked is
    # UNVERIFIABLE rather than a crash.
    if record.figure_crop_node_id in node_ids:
        try:
            ancestors = graph.ancestors(record.figure_crop_node_id)  # type: ignore[attr-defined]
            root_node = ancestors[-1] if ancestors else graph.node(record.figure_crop_node_id)  # type: ignore[attr-defined]
        except (KeyError, ValueError) as exc:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=series_path,
                    reason=f"series {series.series_id!r} cites figure digitization {citation!r}, but the crop's "
                    f"ancestry could not be walked, so the record's document digest could not be checked against "
                    f"the crop's root: {exc!r}",
                )
            )
        else:
            if record.raw_sha256 != root_node.sha256:
                findings.append(
                    ReplayFinding(
                        category=ReplayOutcome.FAILED,
                        ref_path=series_path,
                        reason=f"series {series.series_id!r} cites figure digitization {citation!r}, whose record "
                        f"was derived from document raw_sha256={record.raw_sha256!r}, but crop "
                        f"{record.figure_crop_node_id!r} descends from root artifact {root_node.node_id!r} with "
                        f"sha256={root_node.sha256!r} -- the record's document digest must reach the crop's root "
                        "bytes rather than stopping at an intermediate artifact",
                        expected=root_node.sha256,
                        actual=record.raw_sha256,
                    )
                )
    return findings


def _series_value_ref_paths(index: int, series: Series) -> tuple[str, ...]:
    """Positional ref paths for the point values a digitized ``series`` grounds -- the addresses a
    consumer follows back to the SPECIFIC coordinate/observation whose support went unchecked.

    Mirrors :func:`~carmel.schemas.datasets._iter_series_point_value_refs` (every coordinate, then
    each present observation, per point, in order -- an absent observation offered no ref) but emits
    the addressable path the rest of replay uses for an envelope location (compare
    :func:`_dataset_uncertainty_sites`), NOT the crop NODE id the first cut of this claim used -- a
    crop id cannot say WHICH value it supported -- nor that function's ``where`` prose, which reads
    for a human and no consumer can parse back to a location. A ``value_ref`` is the same slot V9's
    same-crop join walks, so the support named here is exactly the support whose location the figure
    lane re-derived.
    """
    paths: list[str] = []
    for point_index, point in enumerate(series.points):
        base = f"series[{index}].points[{point_index}]"
        for coord_index in range(len(point.coordinates)):
            paths.append(f"{base}.coordinates[{coord_index}].value.value_ref")
        for obs_index, observation in enumerate(point.observations):
            if isinstance(observation.value, Absent):
                continue
            paths.append(f"{base}.observations[{obs_index}].value.value_ref")
    return tuple(paths)


def _axis_label_is_own_column_header(axis_label_locator: object, value_columns: set[tuple[str | None, int]]) -> bool:
    """Is an axis's label grounded at its OWN column's printed header cell?

    True only when the label locator is a :class:`TableCellLocator` sitting in the
    SAME table (inventory) and SAME column as every one of the axis's value cells
    -- structurally attached to the column, so the header names the quantity
    directly. Any other shape -- a char span into running prose (a caption or
    nomenclature), a cell in a different column or table, or values that are not
    all cells of one column -- is NOT the column's own header, so the label proves
    the phrase is in the document but not that THIS column carries the quantity.
    """
    if not isinstance(axis_label_locator, TableCellLocator):
        return False
    inventory = (
        axis_label_locator.pdf_table_inventory_sha256
        if isinstance(axis_label_locator.pdf_table_inventory_sha256, str)
        else None
    )
    label_column = (inventory, axis_label_locator.col)
    # The label must belong to the one column the axis's values all sit in.
    return value_columns == {label_column}


def _dataset_axis_bridge_claims(
    envelope: DatasetEnvelope,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding],
) -> tuple[UncheckedSemanticClaim, ...]:
    """The header-bridge obligation an EQUIVALENCE_RATIO axis declared without a
    printed unit imposes (I-060).

    A dimensionless coordinate whose unit the source never printed is declared
    EQUIVALENCE_RATIO on the strength of a LABEL grounded in the document. When
    that label sits at the column's own printed header cell, the header names the
    quantity directly and nothing is inferred. When it is grounded DISTALLY --
    the flame-speed table's caption "range of equivalence ratios", or the
    nomenclature "phi = equivalence ratio" -- grounding proves the phrase appears
    in the document, NEVER that THIS column is that quantity. That inferential
    step, the "header bridge", is filed here as an unchecked claim rather than
    passed off as evidence: it is exactly the load-bearing rule that grounding
    proves LOCATION, never MEANING, and it is why declaring the quantity can never
    push a report to VERIFIED -- an :class:`UncheckedSemanticClaim` forces
    ``overall_outcome`` to UNVERIFIABLE.

    Scoped narrowly to the axes that actually take this route (EQUIVALENCE_RATIO,
    every value not-printed): an ordinary printed-unit axis grounds its label the
    same way and is untouched here.
    """
    claims: list[UncheckedSemanticClaim] = []
    for series in envelope.series:
        for axis in series.axes:
            if axis.quantity_kind is not QuantityKind.EQUIVALENCE_RATIO:
                continue
            values: list[MeasuredValue] = [
                coordinate.value
                for point in series.points
                for coordinate in point.coordinates
                if coordinate.axis_id == axis.axis_id
            ]
            # An observation's value is a Maybe (a point may record no datum); a
            # not-printed unit is a property of a present MeasuredValue, so an Absent
            # observation value contributes nothing here.
            values += [
                observation.value
                for point in series.points
                for observation in point.observations
                if observation.axis_id == axis.axis_id and not isinstance(observation.value, Absent)
            ]
            if not values or not all(value.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE for value in values):
                continue
            value_columns = {
                (
                    locator.pdf_table_inventory_sha256 if isinstance(locator.pdf_table_inventory_sha256, str) else None,
                    locator.col,
                )
                for value in values
                if isinstance(locator := value.value_ref.locator, TableCellLocator)
            }
            if _axis_label_is_own_column_header(axis.label_ref.locator, value_columns):
                # The label IS the column's own header cell: the header names the
                # quantity directly, so there is no bridge to record.
                continue
            axis_path = f"series[{series.series_id!r}].axes[{axis.axis_id!r}]"
            claims.append(
                UncheckedSemanticClaim(
                    claim_path=f"{axis_path}.quantity_kind",
                    claim=_claim_text(axis.quantity_kind),
                    gap=_semantic_ref_gap(axis.label_ref, text_by_node_id, node_problems),
                    reason="the axis label is grounded in the document but NOT at this column's own "
                    "printed header cell (the unit was not printed, so the quantity rests on a distal "
                    "label -- a caption or nomenclature phrase); that proves the phrase appears in the "
                    "document, never that THIS column carries equivalence ratio, so the header bridge "
                    "is an unchecked claim, not evidence",
                    support_paths=(f"{axis_path}.label_ref",),
                )
            )
    return tuple(claims)


def _figure_semantic_claims(envelope: DatasetEnvelope) -> tuple[UncheckedSemanticClaim, ...]:
    """Two :class:`UncheckedSemanticClaim`\\ s per DIGITIZED series: the two DISTINCT ceilings a
    coherent figure citation cannot pass, each machine-visible at its own ``claim_path``.

    :func:`_verify_figure_digitizations` re-derives that a digitization is stored UNALTERED and
    joined to its series and crop. It establishes NOTHING about two SEPARATE things, and folding
    both into one bucket plus a sentence is exactly the human-only-sortable prose this lane refuses
    everywhere else. So they are filed as two claims a consumer can tell apart by ``claim_path`` --
    a structured coordinate, not a phrase to grep:

    * a MEASUREMENT gap, at ``series[i]``: nothing in this repository re-derives markers from a
      crop's pixels, so whether the recovered coordinates are TRUE is an operator attestation. Its
      support is the point value refs whose LOCATION (never meaning) the lane checked, addressed
      positionally by :func:`_series_value_ref_paths`.
    * a PROVENANCE gap, at ``series[i].digitization_sha256``: the crop's OWN region within its parent
      image is never re-derived -- there is no renderer here, on purpose -- so that the crop's pixels
      were cut from the region it names is an attestation too. Its support is the embedded record
      carrying the crop identity, reached through the citation.

    Both are on the same semantic axis the condition-set attribution claim uses (see
    :func:`_condition_set_semantic_claims`): grounding proves LOCATION, never MEANING. Each gap is
    :attr:`SemanticGap.LOCATION_UNRESOLVED` where support was offered -- and deliberately not the
    softer ``SUPPORT_UNRECORDED``: a digitized point's support is a ``BBoxLocator`` over a pixel
    region, never a :class:`~carmel.schemas.datasets.CharSpanLocator`, so its span cannot be
    re-sliced at all, STRICTLY LESS than a located text span whose meaning merely went unrecorded --
    and :attr:`SemanticGap.NO_SUPPORT_OFFERED` where none was: a series (through ``model_construct``)
    with no point ref offers nothing for the measurement gap, and one citing no digitization offers
    no crop for the provenance gap. Distinguishing the two keeps an empty ``support_paths`` off a
    claim whose gap says support WAS offered, which the claim refuses. ``claim`` is the series'
    recorded ``source_form`` -- the derived DIGITIZED classification whose support (the recovered
    coordinates, and the crop they were cut from) is what these two gaps say went unchecked. That is
    the derived value :attr:`UncheckedSemanticClaim.claim` is contracted to hold, and it carries no
    word of the paper, so it never reaches the redaction gate the source text does. It is
    deliberately NOT the ``digitization_sha256`` a first cut used: a content address is not a derived
    value, and :mod:`carmel.services.figure_digitization_record` states in as many words that the
    digest does not identify the recovered coordinates -- so an address there named something other
    than the value whose support went unchecked.

    The selector is :func:`_claims_digitized`, deliberately value-equality: a ``model_construct``
    series carrying the plain string ``"digitized"`` is digitized to the schema, so it earns these
    ceilings here rather than skipping them on an identity miss.
    """
    claims: list[UncheckedSemanticClaim] = []
    for index, series in enumerate(envelope.series):
        if not _claims_digitized(series.source_form):
            continue
        citation = series.digitization_sha256
        # The DERIVED value both gaps below leave unchecked: the series' recorded source_form, the
        # DIGITIZED classification whose support (the recovered coordinates, and the crop they were
        # cut from) is exactly what replay structurally cannot test. A recorded enum member, which
        # is what UncheckedSemanticClaim.claim is contracted to hold, and it carries no word of the
        # paper. Deliberately NOT the digitization_sha256 a first cut used: a content address is not
        # a derived value, and the record module states the digest does not identify the recovered
        # coordinates -- so the address there named something other than the value gone unchecked.
        claim = str(series.source_form)

        measurement_support = _series_value_ref_paths(index, series)
        claims.append(
            UncheckedSemanticClaim(
                claim_path=f"series[{index}]",
                claim=claim,
                gap=SemanticGap.LOCATION_UNRESOLVED if measurement_support else SemanticGap.NO_SUPPORT_OFFERED,
                reason=f"series {series.series_id!r} was digitized from a figure crop; replay re-derives that its "
                "digitization citation is coherent and chains to the document's raw bytes, but nothing re-derives "
                "markers from the crop's pixels, so whether the recovered coordinates are true is an operator "
                "attestation replay structurally cannot test",
                support_paths=measurement_support,
            )
        )

        provenance_support = (f"figure_digitizations[{citation!r}]",) if isinstance(citation, str) else ()
        claims.append(
            UncheckedSemanticClaim(
                claim_path=f"series[{index}].digitization_sha256",
                claim=claim,
                gap=SemanticGap.LOCATION_UNRESOLVED if provenance_support else SemanticGap.NO_SUPPORT_OFFERED,
                reason=f"series {series.series_id!r} cites a figure crop; replay re-derives that the crop's stored "
                "bytes and its citation cohere, but the crop's OWN region within its parent image is never "
                "re-derived -- there is no renderer here to confirm the crop's pixels were cut from the region it "
                "names, so the crop's provenance within the document is an operator attestation replay structurally "
                "cannot test",
                support_paths=provenance_support,
            )
        )
    return tuple(claims)


def replay_envelope(workspace_root: Path, envelope: DatasetEnvelope, *, reveal_text: bool = False) -> ReplayReport:
    """Independently re-verify an already-loaded ``envelope`` OBJECT against
    the evidence store rooted at ``workspace_root``.

    Combines every check this module performs into one :class:`ReplayReport`:
    evidence identity; every character span; every measured unit's
    normalization, boundary and admission against its recorded locator and
    recorded table; every embedded table inventory's grid re-derived from the
    document's bytes and every cited cell's text; and every figure
    digitization's citation joins re-derived from the envelope
    (:func:`_verify_figure_digitizations`), with the limit a coherent
    digitization cannot pass stated on the semantic axis. Never trusts anything
    the envelope itself carries as pre-verified -- every fact is re-derived from
    the evidence store or from the hand-reviewed
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
    node_level_problems: dict[str, ReplayFinding] = {}
    claim_findings: list[ReplayFinding] = []
    unchecked_store_claims: list[UncheckedStoreClaim] = []
    raw_bytes_by_sha: dict[str, bytes] = {}
    for node in envelope.source_graph.nodes:
        raw_bin = _load_and_check_raw_bin(workspace_root, node)
        if raw_bin.data is not None:
            # Keyed by document digest, so the inventory re-derivation reuses the
            # SAME bytes this loop already hash-verified -- never a second read.
            raw_bytes_by_sha[node.sha256] = raw_bin.data
        text, problem, problem_is_text_only = _independently_verify_node_text(workspace_root, node, raw_bin)
        if problem is not None:
            node_problems[node.node_id] = problem
            if not problem_is_text_only:
                node_level_problems[node.node_id] = problem
        else:
            assert text is not None
            text_by_node_id[node.node_id] = text
        # Runs for EVERY node, including one whose text verification just
        # failed. A node that cannot be re-verified is exactly where a forged
        # provenance claim is most likely to be hiding, and refuting the claim
        # needs nothing the text check produces -- suppressing it on failure
        # would drop the check precisely when it matters most.
        claim_check = _refute_root_sidecar_claim(workspace_root, node)
        if claim_check.finding is not None:
            claim_findings.append(claim_check.finding)
        if claim_check.unchecked is not None:
            unchecked_store_claims.append(claim_check.unchecked)

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

    # Every node's own STORE-INTEGRITY problem is surfaced regardless of
    # whether any char-span ref happens to reach it -- an unreferenced node
    # (or one only reachable via parent_node_id, never a SourceRef) must
    # never be silently invisible just because check_char_spans never had a
    # reason to look at it. A missing extraction is different in kind,
    # though: "this node has no ExtractionBinding" is a fact about a
    # non-textual node (e.g. a FIGURE_CROP targeted only by a BBoxLocator),
    # not a defect of the envelope, so it is excluded from
    # ``node_level_problems`` and instead only becomes a finding wherever
    # ``check_char_spans`` finds a text-dependent ref (a CharSpanLocator)
    # actually targeting that node -- ``node_problems`` (used above) still
    # carries it for that purpose.
    node_level_findings = tuple(node_level_problems.values())

    # The dataset path carries the SAME ref-less obligations as the condition-set
    # path, and for the same reason: `MeasuredValue` and `Uncertainty` occur in
    # both envelope types. Emitting them on only one side would leave
    # `overall_outcome`'s promise ("every check ran, and nothing is left
    # untested") overclaimed here while it was honest there -- a difference no
    # consumer could see and none would expect (Codex round 89).
    uncertainty_sites = _dataset_uncertainty_sites(envelope)
    uncertainty_reconciliation = _reconcile_uncertainty_sites(envelope, {path for path, _ in uncertainty_sites})
    # The figure lane's claim about MEANING that replay structurally cannot test -- one per
    # DIGITIZED series, folded onto the SAME semantic axis as the derived-value claims, so a
    # digitized envelope's overall_outcome is UNVERIFIABLE by design and never launders VERIFIED.
    semantic_claims = (
        _derived_value_claims(envelope, uncertainty_sites)
        + _figure_semantic_claims(envelope)
        + _dataset_axis_bridge_claims(envelope, text_by_node_id, node_problems)
    )

    # The table lane, made to mean something: every embedded inventory's grid
    # re-derived from the document's own bytes (T-cell replay), and every
    # table-cell-grounded value's text compared against the cell it cites.
    embedded_by_sha = {inventory.inventory_sha256: inventory for inventory in envelope.table_inventories}
    inventory_findings = _verify_embedded_inventories(envelope.table_inventories, raw_bytes_by_sha)
    cell_verification = _verify_cited_cell_texts(_dataset_text_pairings(envelope), embedded_by_sha)

    # The OOXML (.docx) table lane, made to mean the same thing: every embedded OOXML
    # inventory's grid re-derived from the document's own bytes, and every OOXML-table-cell-
    # grounded value's text compared against the cell it cites.
    ooxml_embedded_by_sha = {inventory.inventory_sha256: inventory for inventory in envelope.ooxml_table_inventories}
    ooxml_inventory_findings = _verify_embedded_ooxml_inventories(envelope.ooxml_table_inventories, raw_bytes_by_sha)
    ooxml_cell_verification = _verify_cited_ooxml_cell_texts(_dataset_text_pairings(envelope), ooxml_embedded_by_sha)

    # The figure lane, re-derived: every embedded digitization's address re-hashed, its record
    # reconstructed, and V9's series-record-crop joins re-walked from the envelope. Findings only,
    # NO ``checked`` count -- a coherent figure citation is not a reproduced measurement, so it
    # must never let a figure-only replay read VERIFIED (see the function's docstring).
    figure_findings = _verify_figure_digitizations(envelope, raw_bytes_by_sha)

    all_findings = (
        tuple(span_findings)
        + tuple(unit_findings)
        + node_level_findings
        + tuple(claim_findings)
        + uncertainty_reconciliation
        + tuple(inventory_findings)
        + cell_verification.findings
        + tuple(ooxml_inventory_findings)
        + ooxml_cell_verification.findings
        + tuple(figure_findings)
    )

    # A replay that verified NOTHING must never report VERIFIED: that would
    # launder the "verified" label onto an envelope where nothing about its
    # grounding was actually checked. This is UNVERIFIABLE, not FAILED --
    # there is no demonstrated disagreement, only nothing to point to as
    # having passed. The test is now over BOTH kinds of check: an artifact
    # grounded entirely at table cells re-slices zero char spans yet may have
    # matched every one of its cells, so a re-sliced char span is no longer the
    # only thing that counts as "checked something".
    if checked == 0 and cell_verification.checked == 0 and ooxml_cell_verification.checked == 0:
        all_findings = all_findings + (
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path="<check_char_spans>",
                reason="independently re-sliced ZERO character spans and matched ZERO table cells "
                f"(total_char_spans={total_char_spans}); a replay cannot report VERIFIED without at "
                "least one char span or table cell having been compared AND AGREED. Comparisons may "
                "still have run here and disagreed -- a disagreement is reported as its own FAILED "
                "finding and is not silence",
            ),
        )

    # The verdict is NOT computed here. `ReplayReport` derives both of its
    # outcomes from the findings and claims it is handed, so there is no way
    # for a verdict to disagree with the material underneath it -- and no
    # second implementation of the rule to drift from this one.
    return ReplayReport(
        checked_char_spans=checked,
        total_char_spans=total_char_spans,
        unchecked_char_spans=total_char_spans - checked,
        findings=all_findings,
        unchecked_store_claims=tuple(unchecked_store_claims),
        unchecked_semantic_claims=semantic_claims,
        checked_table_cells=cell_verification.checked + ooxml_cell_verification.checked,
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


class _StitchingRefutationPass(NamedTuple):
    """What one sweep of the stitching gate produced, on both axes.

    Two lists rather than one, because they answer different questions and are
    consumed by different fields: ``findings`` reports what went WRONG, and
    ``attempts`` reports what was TRIED. A claim that survives contributes only
    to the second, which is the entire reason the second exists.
    """

    findings: tuple[ReplayFinding, ...]
    attempts: tuple[AttemptedRefutation, ...]


_STITCH_GATE = "carmel.services.stitching:refute_stitched_claim"
"""The gate name recorded on every :class:`AttemptedRefutation` this module
files. A module-level constant so the producer and its tests cannot drift apart
on the spelling -- a test that hardcodes its own copy of this string proves the
test's spelling, not the producer's."""


def _refute_condition_set_stitching(
    envelope: ConditionSetEnvelope,
    text_by_node_id: Mapping[str, str],
    node_problems: Mapping[str, ReplayFinding],
) -> _StitchingRefutationPass:
    """Re-run the span-stitching refutation (P0-d) against INDEPENDENTLY re-read text.

    The producer runs this same gate at write time, which says nothing about an
    envelope stored before the rule existed or built by any route that does not
    go through the producer -- and :func:`replay_condition_set` accepts the
    object it is handed without revalidating it. A gate that lives only on the
    write path is bypassable by exactly the callers most worth checking.

    Each claim is checked against the table its OWN ``conversion_table_sha256``
    records, never against whatever table is current: re-checking a claim under
    a table it never used could refute a claim that was honest under the table
    it actually recorded, and would silently change verdicts whenever a new
    table shipped.

    A refutation is :attr:`ReplayOutcome.FAILED` -- a check that RAN and found a
    definite disagreement. Everything that prevents the check from running is
    :attr:`ReplayOutcome.UNVERIFIABLE`. Collapsing the two would let a refuted
    fabrication hide among the "could not check" entries that
    ``overall_outcome`` is already UNVERIFIABLE for.

    **Every scalar claim of a schema-valid envelope leaves exactly one**
    :class:`AttemptedRefutation`, whatever happened to it -- including the
    claims this function declines to file a finding for. That total is the
    invariant worth having: a per-claim record makes "the gate ran and missed"
    distinguishable from "the gate never reached this claim", and a sweep that
    recorded only its failures would leave a surviving claim looking exactly
    like an unexamined one.

    The schema qualifier is load-bearing, not a hedge. A claim built through
    ``model_construct`` can hold a value no validator would pass -- an
    unhashable ``conversion_table_sha256`` makes ``table_for_sha`` raise
    ``TypeError`` rather than ``UnknownConversionTableError`` -- and the sweep
    then aborts with that claim, and every later one, unrecorded. Hardening
    every gate against arbitrary schema-bypassed input is a separate, named
    piece of work; what matters here is that the invariant states its own scope
    instead of promising something it does not deliver (Codex round 98).
    """
    findings: list[ReplayFinding] = []
    attempts: list[AttemptedRefutation] = []
    for index, claim in enumerate(envelope.scalar_claims):
        path = f"scalar_claims[{index}]"
        node_id = claim.label_ref.node_id
        if node_id in node_problems:
            # Already reported against the node itself; re-reporting it here
            # would double-count one defect as two. The ATTEMPT record is not a
            # second report of that defect, though -- it is this axis's account
            # of a claim whose gate never ran, and omitting it would leave the
            # claim with no trace on the only axis that tracks per-claim work.
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.UNRUNNABLE,
                    reason=(
                        f"node {node_id!r} carries a store-level problem reported against the "
                        "node itself, so this claim's label/value association was never attacked"
                    ),
                )
            )
            continue
        text = text_by_node_id.get(node_id)
        if text is None:
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=path,
                    reason=(
                        f"node {node_id!r} has no independently re-read text, so the "
                        "span-stitching refutation could not run and this claim's "
                        "label/value association is UNCHECKED"
                    ),
                )
            )
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.UNRUNNABLE,
                    reason=f"node {node_id!r} has no independently re-read text",
                )
            )
            continue
        try:
            table = units.table_for_sha(claim.value.conversion_table_sha256)
        except units.UnknownConversionTableError:
            # `table_for_sha` RAISES on an unknown sha -- it never returns None.
            # A replayer owes a verdict about broken input, not a traceback, and
            # a forged or stale envelope naming a table this build does not ship
            # is exactly the input most worth surviving.
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=path,
                    reason=(
                        f"conversion_table_sha256={claim.value.conversion_table_sha256!r} names "
                        "no known table, so the span-stitching refutation has no unit vocabulary "
                        "to read this claim's window with. It is never re-checked against a "
                        "different table than the one it recorded"
                    ),
                )
            )
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.UNRUNNABLE,
                    reason=(f"conversion_table_sha256={claim.value.conversion_table_sha256!r} names no known table"),
                )
            )
            continue
        outcome = refute_stitched_claim(claim, text, table=table)
        if isinstance(outcome, StitchRefutation):
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.FAILED,
                    ref_path=path,
                    reason=f"claim {claim.claim_id!r} is refuted: {outcome.reason}",
                    expected=f"{claim.value.raw_text} {claim.value.unit_raw}",
                    actual=", ".join(outcome.found) if outcome.found else "no number+unit construct",
                )
            )
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.REFUTED,
                    reason=outcome.reason,
                    found=outcome.found,
                )
            )
        elif isinstance(outcome, StitchGateUnrunnable):
            findings.append(
                ReplayFinding(
                    category=ReplayOutcome.UNVERIFIABLE,
                    ref_path=path,
                    reason=f"claim {claim.claim_id!r} could not be checked: {outcome.reason}",
                )
            )
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.UNRUNNABLE,
                    reason=outcome.reason,
                )
            )
        else:
            # The gate ran in full and declined to refute. NO finding: nothing
            # is wrong. The attempt record is the only trace this claim leaves,
            # and it says exactly what happened -- one named attack was mounted
            # and missed -- rather than the silence that would be
            # indistinguishable from never having looked.
            attempts.append(
                AttemptedRefutation(
                    claim_path=path,
                    gate=_STITCH_GATE,
                    status=RefutationStatus.NOT_REFUTED,
                )
            )
    return _StitchingRefutationPass(findings=tuple(findings), attempts=tuple(attempts))


def replay_condition_set(
    workspace_root: Path, envelope: ConditionSetEnvelope, *, reveal_text: bool = False
) -> ReplayReport:
    """Independently re-verify a :class:`ConditionSetEnvelope` against evidence.

    Mirrors :func:`replay_envelope`'s single-pass node-verification structure
    exactly, then diverges where the two envelope types genuinely differ: a
    condition set has TWO ``SourceRef`` locations that are not grounding
    pairs at all (``attribution_ref``, ``subject.reason_ref``) -- a location
    proves nothing recorded says what it MEANS, so each becomes an
    :class:`UncheckedSemanticClaim` rather than a pass/fail check.
    (``unextracted[*].statement_ref`` was a third until ``statement_raw``
    recorded the statement's own words beside it; it is now a grounding pair,
    compared like every label and token.)

    Because ``attribution_ref`` is unconditionally present on every legal
    envelope (see the contract), :attr:`ReplayReport.overall_outcome` is
    UNVERIFIABLE for every condition set this can ever replay -- that is the
    honest answer, not a defect: :attr:`ReplayReport.evidence_outcome` still
    says whether every location claim held.

    A second, independent reason now says the same thing: every scalar claim
    leaves an :class:`AttemptedRefutation`, and any entry on that axis bars
    VERIFIED. The two reasons are deliberately not merged. They would have to be
    removed separately, and each states a different limit -- one that a location
    does not carry a meaning, the other that surviving a falsification attempt
    is not proof.
    """
    text_by_node_id: dict[str, str] = {}
    node_problems: dict[str, ReplayFinding] = {}
    node_level_problems: dict[str, ReplayFinding] = {}
    claim_findings: list[ReplayFinding] = []
    unchecked_store_claims: list[UncheckedStoreClaim] = []
    raw_bytes_by_sha: dict[str, bytes] = {}
    for node in envelope.source_graph.nodes:
        raw_bin = _load_and_check_raw_bin(workspace_root, node)
        if raw_bin.data is not None:
            # Keyed by document digest, so the inventory re-derivation reuses the
            # SAME bytes this loop already hash-verified -- never a second read.
            raw_bytes_by_sha[node.sha256] = raw_bin.data
        text, problem, problem_is_text_only = _independently_verify_node_text(workspace_root, node, raw_bin)
        if problem is not None:
            node_problems[node.node_id] = problem
            if not problem_is_text_only:
                node_level_problems[node.node_id] = problem
        else:
            assert text is not None
            text_by_node_id[node.node_id] = text
        claim_check = _refute_root_sidecar_claim(workspace_root, node)
        if claim_check.finding is not None:
            claim_findings.append(claim_check.finding)
        if claim_check.unchecked is not None:
            unchecked_store_claims.append(claim_check.unchecked)

    checked, paired_total, pair_findings = _replay_text_pairings(
        envelope,
        _condition_set_text_pairings(envelope),
        text_by_node_id,
        node_problems,
        reveal_text=reveal_text,
        audit_against_source_refs=False,
    )

    # The SAME measured-value gates the dataset replayer runs. A condition set
    # carries MeasuredValues through scalar_claims[i].value and through both
    # uncertainty bounds, so without these a claim recording raw_text="1023"
    # whose span points INSIDE the longer numeral "11023" replays clean: the
    # substring matches, and matching a substring is exactly what re-slicing a
    # span cannot distinguish from matching the number. Quote equality proves
    # the characters are there; only the boundary gates prove they are the
    # whole token.
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

    # Obligations from LOCATED-BUT-UNEXPLAINED support, then obligations from
    # values nothing locates at all. Both live in one list because a consumer
    # asks one question ("what did this replay not check?"), and they are told
    # apart by `gap` -- which is exactly what SemanticGap is for.
    uncertainty_sites = _condition_set_uncertainty_sites(envelope)
    uncertainty_reconciliation = _reconcile_uncertainty_sites(envelope, {path for path, _ in uncertainty_sites})
    semantic_claims = _condition_set_semantic_claims(envelope, text_by_node_id, node_problems) + _derived_value_claims(
        envelope, uncertainty_sites
    )
    support_only_char_spans = sum(1 for claim in semantic_claims if claim.gap is SemanticGap.SUPPORT_UNRECORDED)

    # Independent reconciliation: every path either the hand-written pairing
    # enumerator or a semantic claim's support_paths names, checked against a
    # FRESH, generic walk of the envelope -- written separately from both of
    # those enumerators, which is the only reason it is an independent check
    # rather than the same list read twice.
    named_paths: set[str] = {pairing.path for pairing in _condition_set_text_pairings(envelope)}
    support_paths: set[str] = set()
    for claim in semantic_claims:
        named_paths.update(claim.support_paths)
        support_paths.update(claim.support_paths)
    reconciliation_findings = _reconcile_condition_set_refs(envelope, named_paths, support_paths)

    # Every char-span ref the replayer knows about from EITHER source: the
    # generic walk, or the hand-written inventory. Normally these agree
    # exactly and the union is just the walk. They disagree only when the
    # reconciliation above has already reported a defect -- and in that case
    # counting the walk alone would make `checked` exceed the total and raise
    # out of `ReplayReport.__post_init__`, turning a reportable finding into
    # a traceback. A replayer owes a verdict about broken input, not a crash,
    # so the total spans what either side saw and the finding carries the
    # disagreement.
    walked_char_span_paths = {
        path for path, ref in iter_source_refs(envelope) if isinstance(ref.locator, CharSpanLocator)
    }
    inventory_char_span_paths = {
        pairing.path
        for pairing in _condition_set_text_pairings(envelope)
        if isinstance(pairing.locator, CharSpanLocator)
    } | {claim.support_paths[0] for claim in semantic_claims if claim.gap is SemanticGap.SUPPORT_UNRECORDED}
    total_char_spans = len(walked_char_span_paths | inventory_char_span_paths)

    node_level_findings = tuple(node_level_problems.values())
    stitching = _refute_condition_set_stitching(envelope, text_by_node_id, node_problems)
    # The table lane, on the condition-set side too (Verifier 2 requires BOTH
    # envelope kinds): re-derive every embedded inventory's grid from the
    # document bytes, and compare every table-cell-grounded label/token/value
    # text against the cell it cites.
    embedded_by_sha = {inventory.inventory_sha256: inventory for inventory in envelope.table_inventories}
    inventory_findings = _verify_embedded_inventories(envelope.table_inventories, raw_bytes_by_sha)
    cell_verification = _verify_cited_cell_texts(_condition_set_text_pairings(envelope), embedded_by_sha)
    ooxml_embedded_by_sha = {inventory.inventory_sha256: inventory for inventory in envelope.ooxml_table_inventories}
    ooxml_inventory_findings = _verify_embedded_ooxml_inventories(envelope.ooxml_table_inventories, raw_bytes_by_sha)
    ooxml_cell_verification = _verify_cited_ooxml_cell_texts(
        _condition_set_text_pairings(envelope), ooxml_embedded_by_sha
    )
    all_findings = (
        tuple(pair_findings)
        + tuple(unit_findings)
        + node_level_findings
        + tuple(claim_findings)
        + reconciliation_findings
        + uncertainty_reconciliation
        + stitching.findings
        + tuple(inventory_findings)
        + cell_verification.findings
        + tuple(ooxml_inventory_findings)
        + ooxml_cell_verification.findings
    )

    # A replay that verified NOTHING must never report VERIFIED -- same rule as
    # replay_envelope's, for the same reason and widened the same way: a
    # condition set grounded entirely at table cells (which is exactly the
    # shape of this codebase's first real stored artifact) re-slices zero paired
    # char spans yet may have matched every one of its cells, so the guard now
    # tests both kinds of check, not char spans alone.
    if checked == 0 and cell_verification.checked == 0 and ooxml_cell_verification.checked == 0:
        all_findings = all_findings + (
            ReplayFinding(
                category=ReplayOutcome.UNVERIFIABLE,
                ref_path="<_condition_set_text_pairings>",
                reason="independently re-sliced ZERO paired character spans and matched ZERO table "
                f"cells (total paired refs={paired_total}); a replay cannot report VERIFIED without "
                "at least one char span or table cell having been compared AND AGREED. Comparisons "
                "may still have run here and disagreed -- a disagreement is reported as its own "
                "FAILED finding and is not silence",
            ),
        )

    unchecked_char_spans = total_char_spans - checked - support_only_char_spans

    return ReplayReport(
        checked_char_spans=checked,
        total_char_spans=total_char_spans,
        unchecked_char_spans=unchecked_char_spans,
        findings=all_findings,
        unchecked_store_claims=tuple(unchecked_store_claims),
        unchecked_semantic_claims=semantic_claims,
        attempted_refutations=stitching.attempts,
        support_only_char_spans=support_only_char_spans,
        checked_table_cells=cell_verification.checked + ooxml_cell_verification.checked,
    )


def replay_stored_condition_set(workspace_root: Path, sha256: str, *, reveal_text: bool = False) -> ReplayReport:
    """Independently re-verify the condition set actually STORED under ``sha256``.

    The condition-set sibling of :func:`replay_stored_dataset`, and separate
    from it for the same reason the two bridges are separate: they read
    different store directories, and a construct that let one be mistaken for
    the other would report on bytes the caller never asked about.
    """
    envelope = load_condition_set_envelope(workspace_root, sha256)
    return replay_condition_set(workspace_root, envelope, reveal_text=reveal_text)
