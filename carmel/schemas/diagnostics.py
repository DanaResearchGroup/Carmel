"""Diagnostics schema for normalized T3 output."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SensitivityEntry(BaseModel):
    """A single sensitivity coefficient entry."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    value: float
    species: str | None = None
    reaction: str | None = None


class ObservableSummary(BaseModel):
    """Sensitivity summary for one observable."""

    model_config = ConfigDict(extra="forbid")

    observable: str = Field(min_length=1)
    top_sensitive_rates: list[SensitivityEntry] = Field(default_factory=list)
    top_sensitive_thermo: list[SensitivityEntry] = Field(default_factory=list)
    notes: str | None = None


class SpeciesSelection(BaseModel):
    """A species selected by T3 for refined computation."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    smiles: str | None = None
    reason: str | None = None


class ReactionSelection(BaseModel):
    """A reaction selected by T3 for refined computation."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    reactants: list[str]
    products: list[str]
    reason: str | None = None


class PDepNetworkSelection(BaseModel):
    """A pressure-dependent network selected by T3 for refined computation."""

    model_config = ConfigDict(extra="forbid")

    network_id: str = Field(min_length=1)
    species: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    reason: str | None = None


class DiagnosticsV1(BaseModel):
    """Normalized T3 output, the canonical Carmel diagnostics contract."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_version: str | None = None
    level_of_theory: str | None = None
    generated_at: datetime
    observable_summaries: list[ObservableSummary] = Field(default_factory=list)
    species_to_compute: list[SpeciesSelection] = Field(default_factory=list)
    reactions_to_compute: list[ReactionSelection] = Field(default_factory=list)
    pdep_networks_to_compute: list[PDepNetworkSelection] = Field(default_factory=list)
    pdep_sensitivity_flag: bool = False
    warnings: list[str] = Field(default_factory=list)
    tool_metadata: dict[str, Any] = Field(default_factory=dict)
