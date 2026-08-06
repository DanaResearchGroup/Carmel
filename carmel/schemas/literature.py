"""Schemas for literature-research findings, citations, and reports.

These schemas are the machine-consumable contract between the Literature
Agent, the grounding/verifier pipeline, and the plan-progress dispatcher.
``ReactorType`` is reused from :mod:`carmel.schemas.campaign` (not redefined
here) so joins between literature findings and campaign reactor systems stay
directly comparable.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from carmel.agents.budget import BudgetDimension, BudgetUsage
from carmel.schemas.campaign import ReactorType

__all__ = [
    "STOP_REASON_FOR_DIMENSION",
    "Citation",
    "CoveredDocument",
    "CredenceVerdict",
    "EvidenceRef",
    "ExperimentalBenchmarkPayload",
    "FindingCategory",
    "FindingPayload",
    "GroundingStatus",
    "GroundingVerdict",
    "CURRENT_REPORT_SCHEMA_VERSION",
    "LiteratureFinding",
    "LiteratureReport",
    "ObservableKind",
    "PriorModelPayload",
    "QMCalculationPayload",
    "QMProperty",
    "Quantity",
    "RejectedFinding",
    "ROOT_EXTRACTION_ID",
    "SpeciesRef",
    "StopReason",
    "StoredArtifact",
]


class ObservableKind(StrEnum):
    """Kinds of experimental observables a benchmark finding may report."""

    IGNITION_DELAY_TIME = "ignition_delay_time"
    SPECIES_PROFILE = "species_profile"
    LAMINAR_FLAME_SPEED = "laminar_flame_speed"
    CONCENTRATION = "concentration"
    MOLE_FRACTION = "mole_fraction"
    YIELD = "yield"
    CONVERSION = "conversion"
    EXTINCTION_STRAIN_RATE = "extinction_strain_rate"
    OTHER = "other"


class QMProperty(StrEnum):
    """Quantum-mechanical properties a QM-calculation finding may report."""

    ENTHALPY_OF_FORMATION = "enthalpy_of_formation"
    ENTROPY = "entropy"
    HEAT_CAPACITY = "heat_capacity"
    RATE_COEFFICIENT = "rate_coefficient"
    BARRIER_HEIGHT = "barrier_height"
    BOND_DISSOCIATION_ENERGY = "bond_dissociation_energy"
    GEOMETRY = "geometry"
    FREQUENCIES = "frequencies"
    OTHER = "other"


class FindingCategory(StrEnum):
    """Discriminator for :data:`FindingPayload`."""

    EXPERIMENTAL_BENCHMARK = "experimental_benchmark"
    PRIOR_MODEL = "prior_model"
    QM_CALCULATION = "qm_calculation"


class SpeciesRef(BaseModel):
    """A species mentioned in a finding, with best-effort canonicalization."""

    model_config = ConfigDict(extra="forbid")

    raw_name: str = Field(min_length=1)
    """Verbatim species name/label as printed in the source."""
    canonical_smiles: str | None = None
    inchikey: str | None = None
    canonicalized: bool = False
    """False means canonicalization could not be performed (e.g. rdkit
    unavailable or the raw name could not be resolved). Downstream, the
    Verifier penalises credence for non-canonicalized species references."""


class Quantity(BaseModel):
    """A numeric quantity with a normalized unit."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(allow_inf_nan=False)
    """Finite by construction (``allow_inf_nan=False``): a non-finite value can never
    be a trustworthy measured quantity, and a stored inf/nan would later surface in
    the grounding gate as the str(float) required anchor ``'inf'``/``'nan'``, which no
    source text can ever corroborate."""
    unit: str = Field(min_length=1)
    """Normalized SI-ish unit string."""
    uncertainty: float | None = Field(default=None, allow_inf_nan=False)
    """Same finite-by-construction guard as ``value``, for the same reason: a
    stored inf/nan uncertainty would surface as an ungroundable str(float)
    anchor."""
    raw_text: str | None = None
    """Verbatim text as printed in the source, for grounding checks."""


def _normalize_doi(doi: str) -> str:
    """Strip a leading ``https://doi.org/`` or ``doi:`` prefix and lowercase.

    Downstream identity checking greps the artifact text for this
    normalized DOI, so normalization is load-bearing.
    """
    normalized = doi.strip()
    lowered = normalized.lower()
    if lowered.startswith("https://doi.org/"):
        normalized = normalized[len("https://doi.org/") :]
    elif lowered.startswith("http://doi.org/"):
        normalized = normalized[len("http://doi.org/") :]
    elif lowered.startswith("doi:"):
        normalized = normalized[len("doi:") :]
    return normalized.lower()


class Citation(BaseModel):
    """A bibliographic citation for a literature finding."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    """Normalized: lowercase, no ``https://doi.org/`` or ``doi:`` prefix."""
    url: str | None = None
    source_id: str | None = None
    """Non-DOI identifier (e.g. a database record id) when no DOI exists."""

    @model_validator(mode="after")
    def _require_identifier_and_normalize_doi(self) -> Citation:
        """Require at least one of doi/url/source_id; normalize the DOI."""
        if self.doi is None and self.url is None and self.source_id is None:
            raise ValueError("Citation requires at least one of doi, url, or source_id")
        if self.doi is not None:
            self.doi = _normalize_doi(self.doi)
        return self


class ExperimentalBenchmarkPayload(BaseModel):
    """An experimental measurement reported in the literature."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[FindingCategory.EXPERIMENTAL_BENCHMARK] = FindingCategory.EXPERIMENTAL_BENCHMARK
    reactor_type: ReactorType
    observable: ObservableKind
    observable_raw: str = Field(min_length=1)
    """Verbatim observable label as printed in the source."""
    temperature_range_K: tuple[float, float] | None = None
    pressure_range_bar: tuple[float, float] | None = None
    equivalence_ratio_range: tuple[float, float] | None = None
    #: Same finite-by-construction rationale as `Quantity.value`: a stored
    #: inf/nan bound would surface as an ungroundable str(float) anchor. NOT
    #: expressed as `Field(allow_inf_nan=False)` on the tuple-typed fields above
    #: -- probed empirically first (see `_reject_non_finite_range` below) and
    #: pydantic's `allow_inf_nan` check assumes a bare float/int input; handed a
    #: tuple it raises an unconditional `TypeError` (a crash, not a clean
    #: `ValidationError`) even for an ordinary finite tuple. So the three range
    #: fields instead get an explicit `field_validator` that checks each bound.
    residence_time_s: float | None = Field(default=None, allow_inf_nan=False)
    species: list[SpeciesRef] = Field(default_factory=list, max_length=20)
    measured: list[Quantity] = Field(default_factory=list, max_length=8)
    apparatus: str | None = None
    n_data_points: int | None = None
    #: Length caps above: grounding.required_spans_for now anchors EVERY
    #: measured/species entry individually (spar hardening note, P1-4), so an
    #: unbounded list here would let a fabricator pad a finding arbitrarily large.
    #: 8/20 are not calibrated against the 69-paper corpus -- generous ceilings
    #: chosen to comfortably exceed any legitimate single-finding benchmark report
    #: while still bounding the anchor-checking cost.

    @field_validator("temperature_range_K", "pressure_range_bar", "equivalence_ratio_range")
    @classmethod
    def _reject_non_finite_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Both bounds of a range must be finite, for the same reason
        `Quantity.value` is finite by construction: a stored inf/nan bound would
        surface as an ungroundable str(float) anchor in the grounding gate."""
        if value is not None and not all(math.isfinite(bound) for bound in value):
            raise ValueError(f"range bounds must be finite, got {value!r}")
        return value


class PriorModelPayload(BaseModel):
    """A prior mechanism/model reported in the literature."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[FindingCategory.PRIOR_MODEL] = FindingCategory.PRIOR_MODEL
    model_name: str = Field(min_length=1)
    n_species: int | None = None
    n_reactions: int | None = None
    fuel_species: list[SpeciesRef] = Field(default_factory=list)
    mechanism_url: str | None = None
    validation_targets: list[str] = Field(default_factory=list)
    conditions_note: str | None = None


class QMCalculationPayload(BaseModel):
    """A quantum-mechanical calculation result reported in the literature."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[FindingCategory.QM_CALCULATION] = FindingCategory.QM_CALCULATION
    level_of_theory: str = Field(min_length=1)
    property: QMProperty
    value: Quantity
    species: list[SpeciesRef] = Field(default_factory=list)
    reaction_label: str | None = None
    software: str | None = None


FindingPayload = Annotated[
    ExperimentalBenchmarkPayload | PriorModelPayload | QMCalculationPayload,
    Field(discriminator="category"),
]


class EvidenceRef(BaseModel):
    """A pointer into a stored artifact's extracted text supporting a finding."""

    model_config = ConfigDict(extra="forbid")

    artifact_sha256: str = Field(min_length=1)
    extraction_id: str = Field(min_length=1)
    """Which of ``artifact_sha256``'s texts ``quote_start``/``quote_end`` index: either
    :data:`ROOT_EXTRACTION_ID` or a 64-lowercase-hex ``extraction_sha256`` naming a nested
    re-extraction record. Naming the raw sha256 alone used to be unambiguous, back when a
    stored artifact had exactly one extracted text; it no longer is, now that the same raw
    sha256 can carry a root ``extracted.json`` sidecar AND any number of authenticated
    re-extraction records, each with its own text and therefore its own offsets for the
    same quote.

    REQUIRED, with no default, and that absence is deliberate: a producer that never
    considered the question would otherwise silently claim the root, which is exactly the
    ambiguity this field exists to close. Character offsets are also never migrated once
    stored (this project's standing constraint on ``quote_start``/``quote_end``), so a
    finding accepted without recording which text it indexed could never afterwards be
    told. See ``CharSpanLocator.text_space`` in ``carmel/schemas/datasets.py`` for the same
    argument made about a sibling ambiguity.
    """
    quote_start: int | None = None
    """Offset into the text named by ``extraction_id``."""
    quote_end: int | None = None
    page: int | None = None
    section_label: str | None = None

    @field_validator("artifact_sha256")
    @classmethod
    def _validate_artifact_sha256_shape(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid artifact_sha256: {value!r} (expected 64 lowercase hex characters)")
        return value

    @field_validator("extraction_id")
    @classmethod
    def _validate_extraction_id_shape(cls, value: str) -> str:
        if value != ROOT_EXTRACTION_ID and not _SHA256_RE.fullmatch(value):
            raise ValueError(
                f"invalid extraction_id: {value!r} (expected {ROOT_EXTRACTION_ID!r} or "
                "64 lowercase hex characters)"
            )
        return value


class GroundingStatus(StrEnum):
    """Outcome of the grounding gate for a single finding."""

    GROUNDED_EXACT = "grounded_exact"
    GROUNDED_FUZZY = "grounded_fuzzy"
    QUOTE_NOT_FOUND = "quote_not_found"
    REFERENCES_ONLY = "references_only"
    IDENTITY_MISMATCH = "identity_mismatch"
    SPANS_MISSING = "spans_missing"
    NO_ARTIFACT = "no_artifact"
    ARTIFACT_UNREADABLE = "artifact_unreadable"
    """The artifact was fetched but its text could not be recovered well enough to
    search for a quote at all (e.g. a scanned/image-only PDF, or an extraction that
    lost word spacing). Distinct from QUOTE_NOT_FOUND: the finding is still rejected,
    but the cause is our extraction, not a suspected fabrication by the agent."""
    ARTIFACT_DEGRADED = "artifact_degraded"
    """The artifact's extracted text was reloaded in a lossy, structure-free form
    (``ExtractedText.lossy`` True with an empty ``sections`` list) -- e.g. a reload
    path that recovers only a flat text blob when the original structured
    ``extracted.json`` is missing. Distinct from ARTIFACT_UNREADABLE: the text
    itself may be searchable (a quote could still exact-match), but structural
    checks such as references-section detection cannot run reliably against it, so
    the finding is rejected rather than risk a false pass. Also not the agent's
    fault -- the cause is our storage/reload path, not a suspected fabrication."""


class GroundingVerdict(BaseModel):
    """The result of running the (pure, no-LLM) grounding gate on a finding."""

    model_config = ConfigDict(extra="forbid")

    status: GroundingStatus
    grounded: bool
    match_ratio: float = Field(ge=0, le=1)
    identity_ok: bool = False
    missing_spans: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CredenceVerdict(BaseModel):
    """A Verifier's credence assessment for a grounded finding."""

    model_config = ConfigDict(extra="forbid")

    credence: float = Field(ge=0, le=1)
    provenance_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    consistency_score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    flags: list[str] = Field(default_factory=list)


class LiteratureFinding(BaseModel):
    """A single accepted (grounded) literature finding."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    run_id: str = ""
    action_id: str = ""
    """Which pass produced this finding. Carried on the finding itself, not inferred
    from position in the report, so attribution survives any later reordering,
    filtering or merge."""
    payload: FindingPayload
    citation: Citation
    verbatim_quote: str = Field(min_length=1)
    evidence: EvidenceRef
    grounding: GroundingVerdict
    credence: CredenceVerdict | None = None


class RejectedFinding(BaseModel):
    """A finding proposed by the agent but rejected by the grounding gate."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    run_id: str = ""
    action_id: str = ""
    category: FindingCategory | None = None
    citation_title: str | None = None
    grounding: GroundingVerdict
    reason: str = Field(min_length=1)


class StopReason(StrEnum):
    """Why a literature agent run stopped.

    Every :class:`carmel.agents.budget.BudgetDimension` MUST map to a
    ``StopReason`` (spar round 3, P1-22) — see
    :data:`STOP_REASON_FOR_DIMENSION` below.
    """

    SELF_TERMINATED = "self_terminated"
    NO_NEW_INFORMATION = "no_new_information"
    MAX_MODEL_CALLS = "max_model_calls"
    MAX_TOKENS = "max_tokens"
    MAX_FETCHES = "max_fetches"
    MAX_INDEX_LOOKUPS = "max_index_lookups"
    MAX_FETCH_BYTES = "max_fetch_bytes"
    MAX_ARTIFACT_BYTES = "max_artifact_bytes"
    MAX_WALL_CLOCK = "max_wall_clock"
    MAX_COST = "max_cost"
    SESSION_BUDGET = "session_budget"
    DAILY_BUDGET = "daily_budget"
    MAX_CONCURRENT_RUNS = "max_concurrent_runs"
    ERROR = "error"


STOP_REASON_FOR_DIMENSION: dict[BudgetDimension, StopReason] = {
    BudgetDimension.MODEL_CALLS: StopReason.MAX_MODEL_CALLS,
    BudgetDimension.TOKENS: StopReason.MAX_TOKENS,
    BudgetDimension.COST_USD: StopReason.MAX_COST,
    BudgetDimension.FETCHES: StopReason.MAX_FETCHES,
    BudgetDimension.INDEX_LOOKUPS: StopReason.MAX_INDEX_LOOKUPS,
    BudgetDimension.FETCH_BYTES: StopReason.MAX_FETCH_BYTES,
    BudgetDimension.ARTIFACT_BYTES: StopReason.MAX_ARTIFACT_BYTES,
    BudgetDimension.WALL_CLOCK_S: StopReason.MAX_WALL_CLOCK,
    BudgetDimension.SESSION_COST_USD: StopReason.SESSION_BUDGET,
    BudgetDimension.DAILY_COST_USD: StopReason.DAILY_BUDGET,
    BudgetDimension.CONCURRENT_RUNS: StopReason.MAX_CONCURRENT_RUNS,
}


class ArtifactProvenance(StrEnum):
    """How the bytes of an artifact came to be in the workspace.

    This distinction is evidentiary, not cosmetic. A ``FETCHED`` artifact was retrieved
    by Carmel from a URL it can name, so the chain from claim to bytes is machine-checked
    end to end. A ``MANUAL`` artifact was supplied by a human out of band (typically a
    paywalled paper the operator downloaded through an institutional subscription), so
    the link between the requested paper and the supplied bytes rests on an identity
    check against the document's own text rather than on the transport. Recording which
    one applies keeps a reader of the report from over-trusting the second kind.
    """

    FETCHED = "fetched"
    MANUAL = "manual"


class StoredArtifact(BaseModel):
    """Metadata for a content-addressed stored literature artifact."""

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    final_url: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    n_bytes: int = Field(ge=0)
    stored_at: datetime
    extractor: str = Field(min_length=1)
    #: sha256 of the ``extracted.json`` bytes, or ``None`` for artifacts stored before
    #: this field existed.
    #:
    #: ``raw.bin`` is content-addressed and re-hashed on every read, but the grounding
    #: gate does not read ``raw.bin`` -- it reads ``extracted.json``, which until now
    #: carried no integrity check at all. So the file the whole "grounded in the
    #: stored bytes" claim rests on was the one file in the store nothing verified.
    #:
    #: This is NOT a tamper defence: anyone who can write into the evidence store can
    #: rewrite ``meta.json`` and the report alongside it, so this is no privilege
    #: boundary. It closes the NON-adversarial case, which is the one that actually
    #: happens -- a sidecar truncated by a full disk or an interrupted write, which
    #: can still parse as valid JSON and silently ground quotes against a partial
    #: document. ``raw.bin`` was already protected against exactly that; this extends
    #: the same guarantee to the file that matters.
    extracted_sha256: str | None = None
    lossy: bool
    license_note: str | None = None
    provenance: ArtifactProvenance = ArtifactProvenance.FETCHED
    """Defaults to ``FETCHED`` so artifacts stored before this field existed keep
    their (correct) meaning: at that time fetching was the only way in."""
    #: Extractor identity+version string used to produce ``extracted.json``, e.g.
    #: ``"pdf:pypdf==5.1.0"``, or ``None`` for artifacts stored before this field
    #: existed. See :func:`carmel.services.evidence._extractor_identity`: ``pypdf``
    #: is an unpinned dependency (``pypdf>=5.0``), and different installed versions
    #: of the SAME library are not guaranteed to extract byte-identical text from
    #: identical PDF bytes, so the exact version in play is worth recording.
    extractor_version: str | None = None
    #: sha256 of ``f"{extractor_version}|{sha256}|{extracted_sha256}"`` -- a binding
    #: of the extractor identity, the raw bytes' digest, and the sidecar's digest
    #: into one value. ``None`` for artifacts stored before this field existed.
    #:
    #: Read exactly what this proves, and no more. It is NOT proof that
    #: ``extracted.json`` was actually re-derived from ``raw.bin``: re-running the
    #: extractor is not guaranteed to reproduce byte-identical output (see the
    #: determinism analysis in ``carmel.services.evidence``), so this digest is
    #: never recomputed from a fresh extraction. All it proves is INTERNAL
    #: CONSISTENCY of this ``meta.json`` record -- that ``extracted_sha256`` was not
    #: changed independently of ``derivation_binding`` after the two were bound
    #: together at store time. It closes exactly one gap: a stale or swapped
    #: ``extracted.json`` whose ``extracted_sha256`` was updated to match the new
    #: (wrong) sidecar, but whose ``derivation_binding`` was left stale because
    #: nothing recomputed it. It is no defence against a forger who updates both
    #: fields together.
    derivation_binding: str | None = None


class LiteraturePassMode(StrEnum):
    """How one pass over the literature was conducted.

    ``SEARCH`` is the original outward-facing loop: propose queries, search, fetch,
    ground. ``CORPUS`` re-reads the artifacts already in the evidence store and
    performs no search and no fetching at all. A reader must be able to tell which
    produced a finding, because the two have materially different reproducibility:
    a corpus pass runs against a fixed set of sha256-addressed files and will see
    exactly the same input every time, while a search pass depends on what the live
    web returned that day.
    """

    SEARCH = "search"
    CORPUS = "corpus"


class CorpusReadOutcome(StrEnum):
    """What actually happened when a corpus pass considered ONE held artifact.

    Replaces a single undifferentiated ``skipped`` list, which reported an intact
    legacy artifact -- one whose bytes are fine but predates ``derivation_binding`` --
    identically to one whose ``raw.bin`` genuinely no longer hashes to its own name.
    Those two demand opposite operator responses ("re-extract, or opt in" versus
    "re-acquire the paper"), so conflating them is forbidden in this project.

    The tiering below is keyed on ``extracted_sha256`` being present, NOT on
    ``derivation_binding`` being present, and that choice is load-bearing rather than
    incidental. Keying a tier on the ABSENCE of ``derivation_binding`` would be a
    downgrade attack with extra steps: "the field was never written" (a legacy
    artifact) and "the field was deleted" (an artifact someone tampered with) are
    indistinguishable in the store, so an attacker able to delete the field could
    manufacture the more permissive tier at will. Keying on ``extracted_sha256``
    instead is safe in a way that is not, because of its DIRECTION -- deleting
    ``extracted_sha256`` does not promote an artifact, it DEMOTES one, moving it from
    ``SIDECAR_DIGEST_ONLY`` down into ``UNAUTHENTICATED_LEGACY_ROOT``, which is
    refused by default. The same attack that would have laundered a downgrade through
    ``derivation_binding`` therefore buys the attacker only a refusal here, never an
    admission.
    """

    SELF_CONSISTENT_METADATA = "self_consistent_metadata"
    """``verify_artifact(..., deep=True)`` passed: ``raw.bin`` re-hashes to its own
    name, the recorded sidecar digest matches ``extracted.json``, and the recorded
    derivation binding matches one recomputed from ``meta.json``'s own
    ``extractor_version``/``sha256``/``extracted_sha256``. All three inputs to that
    recomputation live in the same mutable file, so this is internal consistency of a
    metadata record, and NOT evidence that the served text came from the stored
    bytes. What it does buy: it catches the realistic non-adversarial failure, a
    truncated or interrupted write."""

    SIDECAR_DIGEST_ONLY = "sidecar_digest_only"
    """As :attr:`SELF_CONSISTENT_METADATA`, minus the binding: deep verification
    failed, but the default (non-deep) check passed AND ``meta.extracted_sha256`` is
    not ``None`` -- the sidecar that is actually read WAS digest-checked, but nothing
    ties it to the raw bytes."""

    UNAUTHENTICATED_LEGACY_ROOT = "unauthenticated_legacy_root"
    """No sidecar digest was ever recorded: the default check passed, but
    ``meta.extracted_sha256`` is ``None``, so the text this would serve is checked
    against nothing at all. Refused by default; readable only under an explicit
    operator opt-in."""

    EXTRACTION_RECORD_DIGEST_AUTHENTICATED = "extraction_record_digest_authenticated"
    """Read from a nested extraction record rather than from the root sidecar: exactly
    one record was current for today's extractor identity, it self-authenticated to its
    own content address, and its stored text matched its recorded
    ``extracted_text_sha256``.

    This does NOT establish that the text was derived from ``raw.bin`` -- no check here
    re-runs the extractor -- and it does not inherit any standard from the root, which
    is not consulted at all on this path. It is strictly a statement about the record
    that was actually served.

    Preferred over every root tier whenever it applies, INCLUDING over
    :attr:`SELF_CONSISTENT_METADATA`. That uniformity is the point: were the record path
    a fallback for a root that fails to authenticate, deleting ``extracted_sha256`` from
    a modern root would silently switch which text is served, and nothing on disk
    distinguishes "legacy root" from "field just deleted" -- so deletion would PROMOTE.
    Preferring the record unconditionally makes deletion incapable of changing which
    path is taken."""

    MULTIPLE_CURRENT_EXTRACTION_RECORDS = "multiple_current_extraction_records"
    """More than one record is current for today's extractor identity at once. Never
    read, and deliberately NOT resolved by picking one.

    Kept distinct from :attr:`EXTRACTION_RECORD_AUTHENTICATION_FAILED` because the two
    say opposite things about the store: there, one record was found and it was broken;
    here, every record may be perfectly intact and the STORE is ambiguous about which
    one speaks for this document. Collapsing them would send the operator hunting for a
    corrupt file that does not exist.

    Falling through to the root sidecar instead would be the same downgrade a failed
    record must not buy: ambiguity among records is not a licence to serve text checked
    against nothing."""

    EXTRACTION_RECORD_AUTHENTICATION_FAILED = "extraction_record_authentication_failed"
    """Exactly one record was current, and it failed to authenticate. Never read -- and
    deliberately NOT downgraded to the root sidecar either.

    Falling back here would hand an attacker precisely the downgrade no operator
    authorised: break the record, and the unauthenticated root text gets served in its
    place. A broken record is a refusal, not a reason to trust something else."""

    INTEGRITY_FAILED = "integrity_failed"
    """The bytes themselves do not match what was recorded: the default (non-deep)
    check itself failed, because ``raw.bin`` is absent or no longer hashes to the
    directory naming it, or a RECORDED sidecar digest no longer matches. Never read,
    regardless of any opt-in -- an opt-in for unauthenticated-but-intact bytes must
    not launder bytes that are not even intact."""

    MISSING_TEXT = "missing_text"
    """Verification (at whichever tier applies) passed, but ``load_artifact_text``
    returned ``None`` anyway: there is nothing to quote from."""

    UNREADABLE_META = "unreadable_meta"
    """Seeded from :func:`~carmel.services.evidence.list_artifacts_with_unreadable`: a
    directory with no readable ``meta.json`` never becomes a :class:`StoredArtifact` at
    all, so none of the checks above ever ran against it."""


#: Matched with ``fullmatch``, never ``match``. Python's ``$`` also matches just BEFORE a
#: trailing newline, so ``re.match`` on this pattern accepts ``"a" * 64 + "\n"`` -- a value
#: that is not a sha256 but would be carried around as though it were one. The anchors are
#: kept for readability; ``fullmatch`` is what actually closes that hole.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Sentinel ``extraction_id`` naming the root ``extracted.json`` sidecar -- the text
#: extracted directly from a stored artifact's own bytes, as opposed to a later
#: re-extraction under ``evidence/literature/<raw_sha256>/extractions/<extraction_sha256>/``
#: (see ``carmel.services.reextraction``). Deliberately the literal string ``"root"``
#: rather than a 64-hex digest: a root sidecar is not content-addressed by its own hash.
#: Its identity is the raw document's address, not its own bytes', so it has no sha256 of
#: its own to key on. That identity is stable precisely because root sidecars are
#: immutable -- this project never rewrites one, not even to upgrade a legacy one.
#: A short literal also can never collide with a genuine 64-lowercase-hex extraction_id,
#: so "root" and "some specific extraction" are always distinguishable by shape alone.
ROOT_EXTRACTION_ID = "root"


class CoveredDocument(BaseModel):
    """One (document, extraction) pair a corpus pass actually read.

    Coverage keyed by raw sha256 alone cannot distinguish "this raw document was
    read" from "this SPECIFIC extraction of it was read" -- a single stored document
    can have more than one extraction on disk (the root sidecar, plus whatever
    re-extraction has produced since). Keying by the pair means a document covered
    under one extraction identity is correctly NOT skipped when a later pass would
    read it under a different one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_sha256: str = Field(min_length=1)
    extraction_id: str = Field(min_length=1)
    """Either :data:`ROOT_EXTRACTION_ID` or a 64-lowercase-hex extraction sha256."""
    verification_standard: str = Field(min_length=1)
    """Which :class:`CorpusReadOutcome` this document was actually read under: one of
    ``EXTRACTION_RECORD_DIGEST_AUTHENTICATED``, ``SELF_CONSISTENT_METADATA``,
    ``SIDECAR_DIGEST_ONLY``, ``UNAUTHENTICATED_LEGACY_ROOT``, or the literal
    ``"unrecorded"``.

    REQUIRED, with no default, for the same reason :attr:`EvidenceRef.extraction_id`
    has none. Reports are APPEND-ONLY: if a pass does not record whether a document
    was read deep, shallow, or unauthenticated, no future reader can ever reconstruct
    it, because the answer depended on the store's state and the operator's flag at
    that instant, and neither is recoverable afterwards. A default would let a
    producer that never considered the question silently claim a standard it did not
    check, which is exactly the ambiguity this field exists to close.

    ``"unrecorded"`` is for MIGRATED records only -- stamped onto every ``covered``
    entry that predates this field, because no earlier writer could have had grounds
    to claim any real standard. A live pass must never write it.
    """

    @field_validator("raw_sha256")
    @classmethod
    def _validate_raw_sha256_shape(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid raw_sha256: {value!r} (expected 64 lowercase hex characters)")
        return value

    @field_validator("extraction_id")
    @classmethod
    def _validate_extraction_id_shape(cls, value: str) -> str:
        if value != ROOT_EXTRACTION_ID and not _SHA256_RE.fullmatch(value):
            raise ValueError(
                f"invalid extraction_id: {value!r} (expected {ROOT_EXTRACTION_ID!r} or "
                "64 lowercase hex characters)"
            )
        return value

    @field_validator("verification_standard")
    @classmethod
    def _validate_verification_standard_shape(cls, value: str) -> str:
        allowed = {
            CorpusReadOutcome.EXTRACTION_RECORD_DIGEST_AUTHENTICATED.value,
            CorpusReadOutcome.SELF_CONSISTENT_METADATA.value,
            CorpusReadOutcome.SIDECAR_DIGEST_ONLY.value,
            CorpusReadOutcome.UNAUTHENTICATED_LEGACY_ROOT.value,
            "unrecorded",
        }
        if value not in allowed:
            raise ValueError(
                f"invalid verification_standard: {value!r} (expected one of "
                f"{sorted(allowed)!r} -- the other CorpusReadOutcome members name ways a "
                "document was NOT read, so nothing could have been read under them)"
            )
        return value


class QueryRecord(BaseModel):
    """One executed search query, attributed to the pass that ran it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)


class PassRecord(BaseModel):
    """The per-pass envelope: everything that describes ONE run rather than the
    accumulated body of evidence."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    created_at: datetime
    mode: LiteraturePassMode = LiteraturePassMode.SEARCH
    """Defaults to ``SEARCH`` so a v1 report migrated forward keeps its (correct)
    meaning: at that time searching was the only kind of pass there was."""
    model_name: str = ""
    stop_reason: StopReason
    usage: BudgetUsage
    warnings: list[str] = Field(default_factory=list)
    covered: list[CoveredDocument] = Field(default_factory=list)
    """The (document, extraction) pairs this pass actually READ, whether or not they
    yielded a finding.

    Recorded so a later corpus pass can skip what has already been mined. Findings
    alone cannot answer that: a document read and found barren produces nothing to
    attribute, and is exactly the document a later pass must not pay to re-read.

    Keyed by the PAIR, not the raw sha256 alone: a single stored document can have
    more than one extraction on disk, and coverage of one extraction must not be
    read as coverage of a different one (see :class:`CoveredDocument`).

    Empty on a search pass, and on any v2 report migrated forward -- for those, what
    was covered was never written down, so the honest value is "nothing recorded"
    rather than a guess reconstructed from findings. A v3 report migrates its
    ``covered_sha256`` forward with every entry's ``extraction_id`` set to
    :data:`ROOT_EXTRACTION_ID`, since that field only ever recorded root-sidecar reads.
    """


#: Highest report schema version this build understands. Anything higher is refused
#: outright by :func:`~carmel.services.literature.migrate_report_payload` rather than
#: read on a best-effort basis, so an older Carmel cannot silently rewrite (and thereby
#: truncate) a report written by a newer one.
CURRENT_REPORT_SCHEMA_VERSION = 6


class LiteratureReport(BaseModel):
    """The campaign's literature record, accumulated across every pass.

    One report per campaign, appended to rather than replaced. A second pass adds a
    :class:`PassRecord` and tags everything it produces with its own ``run_id`` and
    ``action_id``, so a reader can always state which pass produced a given finding
    -- required of the methods paper, and required for a reviewer to reproduce a
    single pass in isolation.

    Overwriting instead of appending was considered and rejected: the first pass's
    record is the current best evidence that the safety design works (findings
    correctly refused for lacking an artifact, i.e. the deterministic gate killing a
    claim before the Verifier ever saw it), and overwriting would destroy it.

    The single-run accessors below (``run_id``, ``stop_reason``, ``usage``, ...)
    report the MOST RECENT pass. They exist because a great deal of surrounding code
    -- the dispatcher's success/failure mapping, the operator dashboard -- is asking
    about the run that just happened, and that question stays well-posed under
    accumulation.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CURRENT_REPORT_SCHEMA_VERSION
    report_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    created_at: datetime
    """When the FIRST pass started. Per-pass timestamps live on each ``PassRecord``."""
    passes: list[PassRecord] = Field(min_length=1)
    queries: list[QueryRecord] = Field(default_factory=list)
    artifacts: list[StoredArtifact] = Field(default_factory=list)
    findings: list[LiteratureFinding] = Field(default_factory=list)
    rejected: list[RejectedFinding] = Field(default_factory=list)

    @property
    def latest(self) -> PassRecord:
        """The most recent pass. ``passes`` is non-empty by construction."""
        return self.passes[-1]

    @property
    def run_id(self) -> str:
        return self.latest.run_id

    @property
    def action_id(self) -> str:
        return self.latest.action_id

    @property
    def stop_reason(self) -> StopReason:
        return self.latest.stop_reason

    @property
    def model_name(self) -> str:
        return self.latest.model_name

    @property
    def usage(self) -> BudgetUsage:
        return self.latest.usage

    @property
    def warnings(self) -> list[str]:
        return self.latest.warnings

    def findings_for(self, run_id: str) -> list[LiteratureFinding]:
        return [f for f in self.findings if f.run_id == run_id]

    def rejected_for(self, run_id: str) -> list[RejectedFinding]:
        """The rejections from ONE pass.

        Sibling of :meth:`findings_for`, and needed for the same reason: any surface
        reporting "this pass" must scope all of its counts, not just the grounded
        one. A panel that scopes findings but leaves rejections accumulated states
        an acceptance rate that is not any pass's (F13).
        """
        return [r for r in self.rejected if r.run_id == run_id]

    def queries_for(self, run_id: str) -> list[QueryRecord]:
        """The queries issued by ONE pass. A corpus pass issues none, by design."""
        return [q for q in self.queries if q.run_id == run_id]
