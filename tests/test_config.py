"""Tests for carmel.config."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from carmel.config import BudgetsConfig, CampaignConfig, CarmelConfig, load_config, validate_config_file
from carmel.schemas.campaign import ReactorType


class TestBudgetsConfig:
    """Tests for the BudgetsConfig model."""

    def test_empty_budgets(self) -> None:
        b = BudgetsConfig()
        assert b.cpu_hours is None
        assert b.experiment_budget is None

    def test_full_budgets(self) -> None:
        b = BudgetsConfig(cpu_hours=50.0, experiment_budget=1000.0)
        assert b.cpu_hours == 50.0
        assert b.experiment_budget == 1000.0

    def test_partial_budgets(self) -> None:
        b = BudgetsConfig(cpu_hours=25.0)
        assert b.cpu_hours == 25.0
        assert b.experiment_budget is None

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetsConfig(cpu_hours=10.0, unknown_field="x")  # type: ignore[call-arg]


class TestCarmelConfig:
    """Tests for the CarmelConfig model."""

    def test_minimal_config(self, valid_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**valid_config_data)
        assert config.workspace_name == "test-workspace"
        assert config.logging_level == "INFO"
        assert config.budgets is None
        assert config.metadata is None

    def test_full_config(self, full_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**full_config_data)
        assert config.workspace_name == "full-workspace"
        assert config.logging_level == "DEBUG"
        assert config.budgets is not None
        assert config.budgets.cpu_hours == 100.0
        assert config.budgets.experiment_budget == 5000.0
        assert config.metadata is not None
        assert config.metadata["author"] == "test-user"

    def test_logging_level_normalized_to_upper(self) -> None:
        config = CarmelConfig(workspace_name="test", workspace_root="/tmp/t", logging_level="debug")
        assert config.logging_level == "DEBUG"

    def test_invalid_logging_level(self) -> None:
        with pytest.raises(ValidationError, match="logging level"):
            CarmelConfig(workspace_name="test", workspace_root="/tmp/t", logging_level="VERBOSE")

    def test_empty_workspace_name(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            CarmelConfig(workspace_name="", workspace_root="/tmp/t")

    def test_blank_workspace_name(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            CarmelConfig(workspace_name="   ", workspace_root="/tmp/t")

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            CarmelConfig()  # type: ignore[call-arg]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CarmelConfig(workspace_name="test", workspace_root="/tmp/t", surprise="value")

    def test_workspace_root_tilde_expanded(self) -> None:
        config = CarmelConfig(workspace_name="test", workspace_root="~/my-workspace")
        assert "~" not in str(config.workspace_root)

    def test_workspace_root_is_path(self, valid_config_data: dict[str, Any]) -> None:
        config = CarmelConfig(**valid_config_data)
        assert isinstance(config.workspace_root, Path)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, valid_config_file: Path) -> None:
        config = load_config(valid_config_file)
        assert config.workspace_name == "test-workspace"

    def test_load_full_config(self, full_config_file: Path) -> None:
        config = load_config(full_config_file)
        assert config.budgets is not None
        assert config.budgets.cpu_hours == 100.0

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_load_yaml_list(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_load_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "malformed.yaml"
        path.write_text("{invalid yaml")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_load_invalid_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.yaml"
        path.write_text(yaml.dump({"workspace_name": "", "workspace_root": "/tmp"}))
        with pytest.raises(ValidationError):
            load_config(path)

    def test_load_accepts_string_path(self, valid_config_file: Path) -> None:
        config = load_config(str(valid_config_file))
        assert config.workspace_name == "test-workspace"


class TestValidateConfigFile:
    """Tests for the validate_config_file function."""

    def test_valid_config_returns_empty(self, valid_config_file: Path) -> None:
        assert validate_config_file(valid_config_file) == []

    def test_missing_file_returns_error(self) -> None:
        errors = validate_config_file("/nonexistent/config.yaml")
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_non_mapping_returns_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("just a string")
        errors = validate_config_file(path)
        assert len(errors) >= 1

    def test_malformed_yaml_returns_error(self, tmp_path: Path) -> None:
        path = tmp_path / "malformed.yaml"
        path.write_text("{invalid yaml")
        errors = validate_config_file(path)
        assert len(errors) >= 1

    def test_missing_fields_returns_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.dump({"logging_level": "INFO"}))
        errors = validate_config_file(path)
        assert len(errors) >= 1

    def test_multiple_errors_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "multi_error.yaml"
        path.write_text(yaml.dump({}))
        errors = validate_config_file(path)
        assert len(errors) >= 2  # missing workspace_name and workspace_root


def _valid_campaign_section() -> dict[str, Any]:
    """A minimal, physically valid ``campaign:`` section as a plain dict.

    Kept separate from the NH3/O2/argon fixture file below: this one is the
    smallest thing that passes every validator, used to isolate individual
    validation failures; the fixture file is the realistic end-to-end case.
    """
    return {
        "initial_mixture": {
            "components": [
                {"species": "O2", "mole_fraction": 0.21},
                {"species": "N2", "mole_fraction": 0.79},
            ]
        },
        "target_observables": [{"name": "ignition_delay_time"}],
        "target_reactor_systems": [
            {
                "reactor_type": "shock_tube",
                "temperature_range_K": [1400.0, 2000.0],
                "pressure_range_bar": [1.0, 30.0],
            }
        ],
        "budgets": {"cpu_hours": 10.0, "experiment_budget": 0.0},
    }


class TestCampaignConfig:
    """The optional ``campaign:`` section: schema, reuse of campaign schemas, validation."""

    def test_full_campaign_section_parses(self) -> None:
        cc = CampaignConfig(**_valid_campaign_section())
        assert len(cc.initial_mixture.components) == 2
        assert cc.target_reactor_systems[0].reactor_type == ReactorType.SHOCK_TUBE
        assert cc.budgets.cpu_hours == 10.0

    def test_to_campaign_input_inherits_workspace_name(self) -> None:
        cc = CampaignConfig(**_valid_campaign_section())
        campaign_input = cc.to_campaign_input("inherited-name")
        assert campaign_input.workspace_name == "inherited-name"
        assert campaign_input.initial_mixture == cc.initial_mixture
        assert campaign_input.target_observables == cc.target_observables
        assert campaign_input.target_reactor_systems == cc.target_reactor_systems
        assert campaign_input.budgets == cc.budgets

    def test_campaign_section_is_optional_on_carmel_config(self, valid_config_data: dict[str, Any]) -> None:
        """Existing configs with no `campaign:` section must still load (backcompat)."""
        config = CarmelConfig(**valid_config_data)
        assert config.campaign is None

    def test_campaign_section_embeds_in_carmel_config(self, valid_config_data: dict[str, Any]) -> None:
        data = {**valid_config_data, "campaign": _valid_campaign_section()}
        config = CarmelConfig(**data)
        assert config.campaign is not None
        assert config.campaign.budgets.cpu_hours == 10.0

    def test_load_config_with_campaign_section_from_yaml(
        self, tmp_path: Path, valid_config_data: dict[str, Any]
    ) -> None:
        data = {**valid_config_data, "campaign": _valid_campaign_section()}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))
        config = load_config(path)
        assert config.campaign is not None
        assert config.campaign.target_reactor_systems[0].reactor_type == ReactorType.SHOCK_TUBE

    def test_mole_fractions_not_summing_to_one_rejected(self) -> None:
        section = _valid_campaign_section()
        section["initial_mixture"]["components"] = [{"species": "O2", "mole_fraction": 0.5}]
        with pytest.raises(ValidationError, match="sum to 1.0"):
            CampaignConfig(**section)

    def test_empty_component_list_rejected(self) -> None:
        section = _valid_campaign_section()
        section["initial_mixture"]["components"] = []
        with pytest.raises(ValidationError):
            CampaignConfig(**section)

    def test_inverted_temperature_range_rejected(self) -> None:
        section = _valid_campaign_section()
        section["target_reactor_systems"][0]["temperature_range_K"] = [2000.0, 1400.0]
        with pytest.raises(ValidationError):
            CampaignConfig(**section)

    def test_negative_pressure_rejected(self) -> None:
        section = _valid_campaign_section()
        section["target_reactor_systems"][0]["pressure_range_bar"] = [-1.0, 30.0]
        with pytest.raises(ValidationError):
            CampaignConfig(**section)

    def test_nh3_o2_ar_fixture_loads(self, tmp_path: Path) -> None:
        """The realistic acceptance-test fixture: stoichiometric NH3/O2 in argon."""
        fixture_path = Path(__file__).parent / "fixtures" / "campaign_nh3_o2_ar.yaml"
        data = yaml.safe_load(fixture_path.read_text())
        data["workspace_root"] = str(tmp_path / "ws")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(data))

        config = load_config(path)

        assert config.campaign is not None
        species = {c.species for c in config.campaign.initial_mixture.components}
        assert species == {"NH3", "O2", "AR"}
        assert config.campaign.target_reactor_systems[0].reactor_type == ReactorType.SHOCK_TUBE
        assert config.campaign.target_reactor_systems[0].temperature_range_K == (1400.0, 2000.0)
        assert config.campaign.target_reactor_systems[0].pressure_range_bar == (1.0, 30.0)
