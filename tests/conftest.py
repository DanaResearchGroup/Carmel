"""Shared fixtures for Carmel tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator
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


@pytest.fixture(autouse=True)
def _carmel_logs_reach_caplog() -> Iterator[None]:
    """Let ``caplog`` see records from Carmel's own loggers.

    ``carmel.logger.get_logger`` sets ``propagate = False`` on every logger it
    configures (correct for production: it owns its handlers and does not want records
    duplicated into whatever the root logger is doing). But ``caplog`` attaches to the
    ROOT logger, so once any test has triggered that configuration, every later test
    asserting on ``caplog`` silently sees nothing.

    That makes log assertions pass alone and fail in a full run, purely on ordering --
    which is exactly what happened: two race-handling tests passed in isolation and in
    their own file, then failed in the full suite. The failure mode is the dangerous
    direction, too: a test that asserts a warning IS logged goes green in isolation
    while the behavior it guards has no coverage at all in the real run.

    Restoring the flag afterwards keeps production behavior unchanged outside tests.
    """
    configured = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("carmel")]
    saved = [(logger, logger.propagate) for logger in configured]
    for logger, _ in saved:
        logger.propagate = True
    try:
        yield
    finally:
        for logger, previous in saved:
            logger.propagate = previous
