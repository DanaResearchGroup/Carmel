"""Configuration loading and validation for Carmel."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


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
    """

    model_config = ConfigDict(extra="forbid")

    workspace_name: str
    workspace_root: Path
    logging_level: str = "INFO"
    budgets: BudgetsConfig | None = None
    metadata: dict[str, Any] | None = None

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
