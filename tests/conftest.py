"""Shared fixtures for Carmel tests."""

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def valid_config_data(tmp_path: Path) -> dict[str, Any]:
    """Minimal valid configuration data."""
    return {
        "workspace_name": "test-workspace",
        "workspace_root": str(tmp_path / "carmel-test"),
    }


@pytest.fixture
def full_config_data(tmp_path: Path) -> dict[str, Any]:
    """Configuration data with all optional fields populated."""
    return {
        "workspace_name": "full-workspace",
        "workspace_root": str(tmp_path / "carmel-full"),
        "logging_level": "DEBUG",
        "budgets": {
            "cpu_hours": 100.0,
            "experiment_budget": 5000.0,
        },
        "metadata": {
            "author": "test-user",
            "description": "A test workspace",
        },
    }


@pytest.fixture
def valid_config_file(tmp_path: Path, valid_config_data: dict[str, Any]) -> Path:
    """Create a minimal valid config YAML file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(valid_config_data))
    return path


@pytest.fixture
def full_config_file(tmp_path: Path, full_config_data: dict[str, Any]) -> Path:
    """Create a full config YAML file with all fields."""
    path = tmp_path / "full_config.yaml"
    path.write_text(yaml.dump(full_config_data))
    return path
