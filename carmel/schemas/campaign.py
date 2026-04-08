"""Campaign and input schemas."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntryMode(StrEnum):
    """How the campaign was entered."""

    BUILD_FROM_SCRATCH = "build_from_scratch"
    REVISE_EXISTING = "revise_existing"  # reserved for future


class MixtureComponent(BaseModel):
    """A single species in the initial mixture."""

    model_config = ConfigDict(extra="forbid")

    species: str = Field(min_length=1)
    mole_fraction: float = Field(gt=0, le=1)
    smiles: str | None = None


class InitialMixture(BaseModel):
    """The initial reactant mixture composition."""

    model_config = ConfigDict(extra="forbid")

    components: list[MixtureComponent]

    @field_validator("components")
    @classmethod
    def must_have_components(cls, v: list[MixtureComponent]) -> list[MixtureComponent]:
        """Reject empty component lists."""
        if not v:
            raise ValueError("initial mixture must have at least one component")
        return v


class TargetObservable(BaseModel):
    """An observable the campaign is targeting."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    species: str | None = None
    description: str | None = None


class ReactorType(StrEnum):
    """Supported reactor system types."""

    JSR = "jsr"
    PFR = "pfr"
    BATCH = "batch"
    SHOCK_TUBE = "shock_tube"
    RCM = "rcm"
    FLAME = "flame"


class ReactorSystem(BaseModel):
    """A reactor system definition."""

    model_config = ConfigDict(extra="forbid")

    reactor_type: ReactorType
    temperature_range_K: tuple[float, float]
    pressure_range_bar: tuple[float, float]
    residence_time_s: float | None = None
    description: str | None = None

    @field_validator("temperature_range_K", "pressure_range_bar")
    @classmethod
    def range_must_be_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Ensure ranges are (min, max) and positive."""
        lo, hi = v
        if lo <= 0 or hi <= 0:
            raise ValueError("range values must be positive")
        if lo > hi:
            raise ValueError(f"range must be ordered: got ({lo}, {hi})")
        return v


class Budgets(BaseModel):
    """Compute and experimental budgets for a campaign."""

    model_config = ConfigDict(extra="forbid")

    cpu_hours: float = Field(gt=0)
    experiment_budget: float = Field(ge=0)


class CampaignInput(BaseModel):
    """The minimal user-provided campaign input."""

    model_config = ConfigDict(extra="forbid")

    workspace_name: str = Field(min_length=1)
    entry_mode: EntryMode = EntryMode.BUILD_FROM_SCRATCH
    initial_mixture: InitialMixture
    target_observables: list[TargetObservable]
    target_reactor_systems: list[ReactorSystem]
    budgets: Budgets
    benchmarks: list[str] | None = None
    preferred_tools: list[str] | None = None
    notes: str | None = None

    @field_validator("target_observables", "target_reactor_systems")
    @classmethod
    def must_be_non_empty(cls, v: list[Any]) -> list[Any]:
        """Reject empty observables/reactor lists."""
        if not v:
            raise ValueError("must not be empty")
        return v


class Campaign(BaseModel):
    """A persisted campaign with metadata."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1)
    workspace_root: Path
    input: CampaignInput
    created_at: datetime
    updated_at: datetime
