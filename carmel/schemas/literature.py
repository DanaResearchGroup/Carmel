"""Schemas for literature-research findings, citations, and reports.

These schemas are the machine-consumable contract between the Literature
Agent, the grounding/verifier pipeline, and the plan-progress dispatcher.
``ReactorType`` is reused from :mod:`carmel.schemas.campaign` (not redefined
here) so joins between literature findings and campaign reactor systems stay
directly comparable.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carmel.agents.budget import BudgetDimension, BudgetUsage
from carmel.schemas.campaign import ReactorType

__all__ = [
    "STOP_REASON_FOR_DIMENSION",
    "Citation",
    "CredenceVerdict",
    "EvidenceRef",
    "ExperimentalBenchmarkPayload",
    "FindingCategory",
    "FindingPayload",
    "GroundingStatus",
    "GroundingVerdict",
    "LiteratureFinding",
    "LiteratureReport",
    "ObservableKind",
    "PriorModelPayload",
    "QMCalculationPayload",
    "QMProperty",
    "Quantity",
    "RejectedFinding",
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

    value: float
    unit: str = Field(min_length=1)
    """Normalized SI-ish unit string."""
    uncertainty: float | None = None
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
    residence_time_s: float | None = None
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
    quote_start: int | None = None
    """Offset into ExtractedText.text."""
    quote_end: int | None = None
    page: int | None = None
    section_label: str | None = None


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
    lossy: bool
    license_note: str | None = None
    provenance: ArtifactProvenance = ArtifactProvenance.FETCHED
    """Defaults to ``FETCHED`` so artifacts stored before this field existed keep
    their (correct) meaning: at that time fetching was the only way in."""


class LiteratureReport(BaseModel):
    """The persisted output of a single literature-research run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    report_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    queries: list[str] = Field(default_factory=list)
    artifacts: list[StoredArtifact] = Field(default_factory=list)
    findings: list[LiteratureFinding] = Field(default_factory=list)
    rejected: list[RejectedFinding] = Field(default_factory=list)
    stop_reason: StopReason
    model_name: str = ""
    usage: BudgetUsage
    warnings: list[str] = Field(default_factory=list)
