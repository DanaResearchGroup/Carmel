"""Tests for the agentic-layer additions to carmel.config."""

from typing import Any

import pytest
from pydantic import ValidationError

from carmel.config import (
    DEFAULT_TIER_MODELS,
    AgentBudgetConfig,
    AgentConfig,
    AgentProvider,
    CarmelConfig,
    ModelTier,
)


class TestAgentBudgetConfig:
    """Tests for AgentBudgetConfig defaults and validation."""

    def test_defaults_match_contract(self) -> None:
        b = AgentBudgetConfig()
        assert b.max_model_calls == 40
        assert b.max_tokens == 400_000
        assert b.max_fetches == 60
        assert b.max_fetch_bytes == 200_000_000
        assert b.max_artifact_bytes == 25_000_000
        assert b.max_wall_clock_s == 1800.0
        assert b.max_cost_usd == 5.0
        assert b.session_max_cost_usd == 20.0
        assert b.daily_max_cost_usd == 50.0
        assert b.max_concurrent_runs == 2

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBudgetConfig(unknown_field=1)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "field",
        [
            "max_model_calls",
            "max_tokens",
            "max_fetches",
            "max_fetch_bytes",
            "max_artifact_bytes",
            "max_wall_clock_s",
            "max_cost_usd",
            "session_max_cost_usd",
            "daily_max_cost_usd",
            "max_concurrent_runs",
        ],
    )
    def test_non_positive_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            AgentBudgetConfig(**{field: 0})


class TestAgentConfigDefaults:
    """Tests for AgentConfig default (TEST/MOCK) validity."""

    def test_defaults_validate(self) -> None:
        cfg = AgentConfig()
        assert cfg.tier == ModelTier.TEST
        assert cfg.provider == AgentProvider.MOCK
        assert cfg.model_name is None
        assert isinstance(cfg.budget, AgentBudgetConfig)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(surprise="value")  # type: ignore[call-arg]


class TestAgentConfigFailClosed:
    """Tests for the fail-closed tier/provider consistency rules."""

    def test_test_tier_with_non_mock_provider_raises(self) -> None:
        with pytest.raises(ValidationError, match="TEST requires provider MOCK"):
            AgentConfig(tier=ModelTier.TEST, provider=AgentProvider.GOOGLE, api_key_env="GOOGLE_API_KEY")

    def test_mock_provider_with_non_test_tier_raises(self) -> None:
        with pytest.raises(ValidationError, match="MOCK requires tier TEST"):
            AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.MOCK)

    def test_real_provider_without_api_key_env_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires api_key_env"):
            AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.GOOGLE)

    def test_real_provider_with_api_key_env_and_matching_tier_validates(self) -> None:
        cfg = AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.GOOGLE, api_key_env="GOOGLE_API_KEY")
        assert cfg.tier == ModelTier.DEV
        assert cfg.provider == AgentProvider.GOOGLE
        assert cfg.api_key_env == "GOOGLE_API_KEY"

    def test_prod_openai_with_api_key_env_validates(self) -> None:
        cfg = AgentConfig(tier=ModelTier.PROD, provider=AgentProvider.OPENAI, api_key_env="OPENAI_API_KEY")
        assert cfg.tier == ModelTier.PROD


class TestAgentConfigEnvVarValidation:
    """Tests for api_key_env / search_api_key_env name validation."""

    @pytest.mark.parametrize("name", ["GOOGLE_API_KEY", "A", "OPENAI_KEY_2", "X_Y_Z"])
    def test_valid_names_pass(self, name: str) -> None:
        cfg = AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.GOOGLE, api_key_env=name)
        assert cfg.api_key_env == name

    @pytest.mark.parametrize(
        "name",
        [
            "google_api_key",  # lowercase
            "sk-abcdef1234567890",  # looks like a literal secret
            "1KEY",  # starts with digit
            "GOOGLE-API-KEY",  # dash
            "Google_Api_Key",  # mixed case
            "",  # empty
        ],
    )
    def test_invalid_names_raise(self, name: str) -> None:
        with pytest.raises(ValidationError, match="environment variable name"):
            AgentConfig(tier=ModelTier.DEV, provider=AgentProvider.GOOGLE, api_key_env=name)

    def test_search_api_key_env_validated_too(self) -> None:
        with pytest.raises(ValidationError, match="environment variable name"):
            AgentConfig(
                tier=ModelTier.DEV,
                provider=AgentProvider.GOOGLE,
                api_key_env="GOOGLE_API_KEY",
                search_api_key_env="not-valid",
            )

    def test_search_api_key_env_valid_passes(self) -> None:
        cfg = AgentConfig(
            tier=ModelTier.DEV,
            provider=AgentProvider.GOOGLE,
            api_key_env="GOOGLE_API_KEY",
            search_api_key_env="SEARCH_API_KEY",
        )
        assert cfg.search_api_key_env == "SEARCH_API_KEY"


class TestResolvedModelName:
    """Tests for AgentConfig.resolved_model_name()."""

    def test_explicit_model_name_wins(self) -> None:
        cfg = AgentConfig(model_name="custom-model")
        assert cfg.resolved_model_name() == "custom-model"

    @pytest.mark.parametrize("tier", [ModelTier.TEST, ModelTier.DEV, ModelTier.PROD])
    def test_falls_back_to_tier_default(self, tier: ModelTier) -> None:
        kwargs: dict[str, Any] = {"tier": tier}
        if tier != ModelTier.TEST:
            kwargs["provider"] = AgentProvider.GOOGLE
            kwargs["api_key_env"] = "GOOGLE_API_KEY"
        cfg = AgentConfig(**kwargs)
        assert cfg.resolved_model_name() == DEFAULT_TIER_MODELS[tier]


class TestCarmelConfigAgentsField:
    """Tests for CarmelConfig.agents integration."""

    def test_agents_defaults_to_none(self, valid_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**valid_config_data)
        assert config.agents is None

    def test_agents_accepts_agent_config(self, valid_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**valid_config_data, agents=AgentConfig())
        assert config.agents is not None
        assert config.agents.tier == ModelTier.TEST

    def test_agents_accepts_dict(self, valid_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**valid_config_data, agents={"tier": "test", "provider": "mock"})
        assert config.agents is not None
        assert config.agents.provider == AgentProvider.MOCK
