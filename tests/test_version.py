"""Tests for carmel.version."""

import re

import carmel
from carmel.version import __version__


class TestVersion:
    """Tests for the version string."""

    def test_version_is_string(self) -> None:
        assert isinstance(__version__, str)

    def test_version_not_empty(self) -> None:
        assert len(__version__) > 0

    def test_version_semver_format(self) -> None:
        assert re.match(r"^\d+\.\d+\.\d+$", __version__)

    def test_version_accessible_from_package(self) -> None:
        assert carmel.__version__ == __version__
