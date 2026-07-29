# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and validation for Carmel."""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: How far a mixture's mole fractions may drift from 1.0 and still be accepted.
#: Loose enough to tolerate YAML-authored fixtures rounded to a few decimal places
#: (e.g. 0.01 + 0.005 + 0.985 for a dilute trace species), tight enough to still
#: catch a genuinely wrong composition (a component left out, a units slip such as
#: percent instead of fraction) rather than only a rounding artifact.
MOLE_FRACTION_SUM_TOLERANCE = 1e-3


class BudgetsConfig(BaseModel):
    """Budget constraints for a Carmel campaign."""

    model_config = ConfigDict(extra="forbid")

    cpu_hours: float | None = None
    experiment_budget: float | None = None


class CarmelConfig(BaseModel):
    """Root configuration for a Carmel workspace.

    Attributes:
        workspace_name: Human-readable name for the workspace.
        workspace_root: Path to the workspace directory.
        logging_level: Logging verbosity level.
        budgets: Optional budget constraints.
        metadata: Optional free-form metadata.
        campaign: Optional full campaign definition (initial mixture, targets,
            reactor systems, budgets) -- see :class:`CampaignConfig`. Absent for
            configs that only describe a workspace (e.g. ``carmel init``); required
            for :func:`carmel.services.campaigns.create_campaign_from_config`.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_name: str
    workspace_root: Path
    logging_level: str = "INFO"
    budgets: BudgetsConfig | None = None
    metadata: dict[str, Any] | None = None
    agents: AgentConfig | None = None
    campaign: CampaignConfig | None = None

    @field_validator("workspace_name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Ensure workspace name is not blank."""
        if not v.strip():
            raise ValueError("workspace_name must not be empty or blank")
        return v

    @field_validator("logging_level")
    @classmethod
    def level_must_be_valid(cls, v: str) -> str:
        """Normalize and validate logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid logging level: {v!r}. Must be one of {sorted(valid)}")
        return v.upper()

    @field_validator("workspace_root", mode="before")
    @classmethod
    def expand_workspace_root(cls, v: Any) -> Path:
        """Expand user home directory in workspace root."""
        return Path(v).expanduser()


def load_config(path: Path | str) -> CarmelConfig:
    """Load and validate a Carmel configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated CarmelConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file content is not a YAML mapping.
        ValidationError: If the config data fails pydantic validation.
        yaml.YAMLError: If the file contains malformed YAML.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(data).__name__}")

    return CarmelConfig(**data)


def validate_config_file(path: Path | str) -> list[str]:
    """Validate a config file and return any errors found.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A list of error messages. Empty if the config is valid.
    """
    try:
        load_config(path)
        return []
    except FileNotFoundError as e:
        return [str(e)]
    except ValidationError as e:
        return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
    except (ValueError, yaml.YAMLError) as e:
        return [str(e)]


class ModelTier(StrEnum):
    """Which model tier a run uses."""

    TEST = "test"  # MockModel, no network, used by CI
    DEV = "dev"  # free tier
    PROD = "prod"


class AgentProvider(StrEnum):
    """Which LLM provider backs a run."""

    MOCK = "mock"
    GOOGLE = "google"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class SearchProvider(StrEnum):
    """Which literature-search backend a run uses.

    Deliberately a SEPARATE axis from :class:`AgentProvider`: which LLM writes the
    queries and which index answers them are independent choices. A run may use a
    Google LLM against the keyless OpenAlex index, and the fail-closed key checks
    differ per backend — ``HTTP_JSON`` needs an operator-supplied endpoint and key,
    while the scholarly indices below are keyless and must NOT be forced to invent
    one (that requirement is what previously made them unconfigurable).
    """

    HTTP_JSON = "http_json"  # generic operator-configured JSON endpoint + bearer key
    OPENALEX = "openalex"  # keyless; api.openalex.org
    CROSSREF = "crossref"  # keyless; api.crossref.org


#: Search backends that take no API key. Keeping this as data (rather than an ``in``
#: check spelled out at each call site) means adding a backend cannot silently miss
#: one of the fail-closed branches in ``carmel.services.literature.build_deps``.
KEYLESS_SEARCH_PROVIDERS: frozenset[SearchProvider] = frozenset({SearchProvider.OPENALEX, SearchProvider.CROSSREF})

#: Environment-variable fallback for :attr:`AgentConfig.unpaywall_email`. Holds the
#: operator's contact email itself (not the name of another variable): an email
#: address is deploy-time operator PII that must never be committed to the repo, and
#: an env var is the standard way to supply exactly that.
UNPAYWALL_EMAIL_ENV = "CARMEL_UNPAYWALL_EMAIL"


#: Model each tier uses when the operator does not name one.
#:
#: DEV and PROD name a FAMILY, not a model. ``carmel.agents.model_catalog`` resolves each
#: to the highest-versioned member the provider actually serves at build time. Exact pins
#: were tried first and rotted within months -- ``gemini-2.5-flash`` now answers 404
#: ("no longer available to new users") -- while the moving alias that replaced one of
#: them (``gemini-pro-latest``) was the single model ``genai_prices`` cannot price, and so
#: silently under-charged the budget ledger by 2x. Naming the family keeps the tier on a
#: current, concretely-priced model without a human editing this dict every quarter.
#:
#: TEST stays literal: it must resolve to the in-process MockModel and never reach a
#: network at all.
DEFAULT_TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.TEST: "mock",
    ModelTier.DEV: "auto:gemini-flash",
    ModelTier.PROD: "auto:gemini-pro",
}


class AgentBudgetConfig(BaseModel):
    """Hard, non-LLM ceilings.

    Defaults are deliberately LOOSE: they trip only on a genuine runaway,
    never on normal operation.
    """

    model_config = ConfigDict(extra="forbid")

    max_model_calls: int = Field(default=40, gt=0)
    max_tokens: int = Field(default=400_000, gt=0)
    max_fetches: int = Field(default=60, gt=0)
    max_fetch_bytes: int = Field(default=200_000_000, gt=0)  # total across run
    max_artifact_bytes: int = Field(default=25_000_000, gt=0)  # single artifact cap
    max_wall_clock_s: float = Field(default=1800.0, gt=0)
    max_cost_usd: float = Field(default=5.0, gt=0)
    session_max_cost_usd: float = Field(default=20.0, gt=0)  # process-wide, all runs
    daily_max_cost_usd: float = Field(default=50.0, gt=0)  # persisted, all campaigns
    max_concurrent_runs: int = Field(default=2, gt=0)


# Deliberately imported here, not at module top, to break a real circular import:
# `carmel.schemas` (imported transitively by `carmel.schemas.campaign`) pulls in
# `carmel.schemas.literature`, which imports `carmel.agents.budget`, which does
# `from carmel.config import AgentBudgetConfig`. If this module is the one that
# starts the import chain (i.e. something imports `carmel.config` first), that
# reentrant import finds `carmel.config` already in `sys.modules` but only
# partially executed -- so `AgentBudgetConfig` must already be bound in this
# module's namespace by the time we reach here. Placing the import after
# `AgentBudgetConfig` (defined just above) and using `from __future__ import
# annotations` (so `CampaignConfig`'s field types below are resolved lazily by
# pydantic, exactly like `CarmelConfig.agents: AgentConfig | None` already relies
# on `AgentConfig` being defined later in this same file) is what makes this safe.
from carmel.schemas.campaign import (  # noqa: E402
    Budgets,
    CampaignInput,
    EntryMode,
    InitialMixture,
    ReactorSystem,
    TargetObservable,
)


class CampaignConfig(BaseModel):
    """The optional ``campaign:`` section of a Carmel config file.

    This is what lets a single YAML file fully describe a run instead of an
    operator hand-writing a private Python script that builds ``CampaignInput``
    (see the module docstring context in ``carmel/services/campaigns.py`` for why
    that was unacceptable). It deliberately REUSES the ``carmel.schemas.campaign``
    models field-for-field rather than defining a parallel set of config-only
    models: those are already pydantic, YAML maps onto them cleanly, and a second
    definition would only rot against the real ``CampaignInput`` contract over
    time. ``workspace_name`` is intentionally NOT a field here -- it already lives
    on :class:`CarmelConfig` and duplicating it would create exactly the kind of
    two-sources-of-truth drift this whole section exists to avoid.
    """

    model_config = ConfigDict(extra="forbid")

    initial_mixture: InitialMixture
    target_observables: list[TargetObservable]
    target_reactor_systems: list[ReactorSystem]
    budgets: Budgets
    entry_mode: EntryMode = EntryMode.BUILD_FROM_SCRATCH
    benchmarks: list[str] | None = None
    preferred_tools: list[str] | None = None
    notes: str | None = None

    @field_validator("initial_mixture")
    @classmethod
    def mole_fractions_must_sum_to_one(cls, v: InitialMixture) -> InitialMixture:
        """Reject a mixture whose mole fractions do not sum to ~1.0.

        Each ``MixtureComponent`` already bounds its own fraction to ``(0, 1]``
        (see ``carmel/schemas/campaign.py``), and ``InitialMixture`` already
        rejects an empty component list -- but nothing upstream checks that the
        SET of components sums to a physically valid mixture. A config with a
        missing component, a typo'd fraction, or a units slip (percent instead of
        a fraction) would otherwise pass every field-level check and only fail
        much later, deep inside a solver, with no message pointing back at the
        actual mistake in the YAML. Checked here, at config-load time, instead.

        Scoped to this config-only wrapper rather than to the shared
        ``InitialMixture`` schema itself: other callers already construct
        ``InitialMixture`` directly with mixtures that intentionally do not sum
        to 1.0 (fixtures exercising a single reactive species in isolation), and
        this section is specifically about validating an operator-authored,
        real-run YAML, not the general-purpose schema.
        """
        total = sum(c.mole_fraction for c in v.components)
        if abs(total - 1.0) > MOLE_FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"campaign.initial_mixture mole fractions must sum to 1.0 "
                f"(within {MOLE_FRACTION_SUM_TOLERANCE}): got {total!r}"
            )
        return v

    def to_campaign_input(self, workspace_name: str) -> CampaignInput:
        """Build the real ``CampaignInput`` this section describes.

        Args:
            workspace_name: The owning ``CarmelConfig.workspace_name`` -- passed
                in rather than duplicated as a field on this model.
        """
        return CampaignInput(
            workspace_name=workspace_name,
            entry_mode=self.entry_mode,
            initial_mixture=self.initial_mixture,
            target_observables=self.target_observables,
            target_reactor_systems=self.target_reactor_systems,
            budgets=self.budgets,
            benchmarks=self.benchmarks,
            preferred_tools=self.preferred_tools,
            notes=self.notes,
        )


class AgentConfig(BaseModel):
    """Configuration for the agentic literature/model layer.

    Fail-closed by construction: see :meth:`tier_provider_consistency`.
    """

    model_config = ConfigDict(extra="forbid")

    tier: ModelTier = ModelTier.TEST
    provider: AgentProvider = AgentProvider.MOCK
    model_name: str | None = None  # None -> DEFAULT_TIER_MODELS[tier]
    api_key_env: str | None = None  # env var holding the key; never the key itself
    search_provider: SearchProvider = SearchProvider.HTTP_JSON
    search_endpoint: str | None = None  # HTTP_JSON only; keyless backends know their own
    search_api_key_env: str | None = None
    search_contact_email: str | None = None
    """Contact address sent to keyless scholarly APIs (OpenAlex/Crossref "polite pool").

    Optional: both APIs answer without it, but at a much lower rate limit and with no
    way for the operator to be told about a misbehaving client before being blocked.
    """
    unpaywall_email: str | None = None
    """Contact email sent to the Unpaywall API, which REQUIRES one as a condition of
    use (``api.unpaywall.org/v2/{doi}?email=...``).

    Deliberately optional with no default: this is operator PII and must be supplied
    per deployment (here, or via the :data:`UNPAYWALL_EMAIL_ENV` environment
    variable) -- never hardcoded in the repository. When neither is set, Unpaywall is
    skipped entirely; Carmel never sends a fake or placeholder address, which would
    violate Unpaywall's API terms. See :meth:`resolved_unpaywall_email`.
    """
    external_provider_consent: bool = False
    literature_at_campaign_start: bool = True
    budget: AgentBudgetConfig = Field(default_factory=AgentBudgetConfig)

    @field_validator("api_key_env", "search_api_key_env")
    @classmethod
    def _validate_env_var_name(cls, v: str | None) -> str | None:
        """Reject anything that isn't a plausible env-var NAME.

        This field holds the NAME of an environment variable, never a literal
        secret. Must match ``^[A-Z][A-Z0-9_]*$`` if set.
        """
        if v is None:
            return v
        if not re.match(r"^[A-Z][A-Z0-9_]*$", v):
            raise ValueError(
                f"{v!r} does not look like an environment variable name "
                "(expected ^[A-Z][A-Z0-9_]*$); never pass a literal secret here"
            )
        return v

    @model_validator(mode="after")
    def tier_provider_consistency(self) -> AgentConfig:
        """Fail closed: TEST<->MOCK must match 1:1, and any real provider needs a key env."""
        if self.tier == ModelTier.TEST and self.provider != AgentProvider.MOCK:
            raise ValueError("tier TEST requires provider MOCK")
        if self.provider == AgentProvider.MOCK and self.tier != ModelTier.TEST:
            raise ValueError("provider MOCK requires tier TEST")
        if self.provider != AgentProvider.MOCK and not self.api_key_env:
            raise ValueError(f"provider {self.provider} requires api_key_env to be set")
        return self

    def resolved_model_name(self) -> str:
        """Return model_name if set, else the tier's default model."""
        return self.model_name or DEFAULT_TIER_MODELS[self.tier]

    def resolved_unpaywall_email(self) -> str | None:
        """The Unpaywall contact email: config field first, then the environment.

        Returns:
            :attr:`unpaywall_email` when set, else a non-empty
            :data:`UNPAYWALL_EMAIL_ENV` environment value, else ``None`` -- in which
            case Unpaywall must be skipped (never called with a fabricated address).
        """
        return self.unpaywall_email or os.environ.get(UNPAYWALL_EMAIL_ENV) or None
