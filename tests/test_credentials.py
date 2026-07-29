# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for carmel.credentials: on-disk API key discovery.

No test touches the operator's real ~/.carmel, ~/.config/carmel, or ~/.config/google --
every test redirects HOME (and, where relevant, CARMEL_HOME) into a pytest tmp_path.
All key-like values used here are obviously fake (e.g. "test-key-not-real").
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from carmel.credentials import credential_search_path, resolve_api_key


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect HOME (and unset CARMEL_HOME) into an empty tmp dir for every test.

    Guarantees no test can accidentally read or write the operator's real credential
    files, regardless of what this machine actually has on disk.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CARMEL_HOME", raising=False)
    # Unset every credential env var these tests probe, so a real key exported in the
    # operator's own shell (e.g. a genuine OPENAI_API_KEY) can never leak into a test
    # result -- these tests must only ever see the fake values they write themselves.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return tmp_path


def _write(path: Path, content: str, *, mode: int | None = None) -> Path:
    """Write ``content`` to ``path``, creating parent dirs, with an optional chmod."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)
    return path


class TestEnvWinsOverFile:
    """An explicitly exported env var must always win over any on-disk candidate."""

    def test_env_var_wins_over_carmel_home_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path / "carmel_home"))
        _write(tmp_path / "carmel_home" / "credentials.env", "GOOGLE_API_KEY=test-key-from-file\n")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-from-env")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-from-env"

    def test_empty_env_var_falls_through_to_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # An env var that is SET but empty must not shadow a real on-disk credential --
        # only a non-empty exported value counts as "explicitly set".
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path / "carmel_home"))
        _write(tmp_path / "carmel_home" / "credentials.env", "GOOGLE_API_KEY=test-key-from-file\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-from-file"


class TestFilePrecedenceOrder:
    """Each on-disk location is tried in the documented order, first hit wins."""

    def test_carmel_home_credentials_file_used_first(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path / "carmel_home"))
        _write(tmp_path / "carmel_home" / "credentials.env", "GOOGLE_API_KEY=test-key-carmel-home\n")
        _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-dot-carmel\n")
        _write(tmp_path / ".config" / "carmel" / "env", "GOOGLE_API_KEY=test-key-config-carmel\n")
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-carmel-home"

    def test_dot_carmel_credentials_file_used_when_no_carmel_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-dot-carmel\n")
        _write(tmp_path / ".config" / "carmel" / "env", "GOOGLE_API_KEY=test-key-config-carmel\n")
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-dot-carmel"

    def test_config_carmel_env_used_when_no_higher_precedence_file(self, tmp_path: Path) -> None:
        _write(tmp_path / ".config" / "carmel" / "env", "GOOGLE_API_KEY=test-key-config-carmel\n")
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-config-carmel"

    def test_provider_conventional_path_used_last(self, tmp_path: Path) -> None:
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-provider"

    def test_no_provider_given_skips_provider_conventional_path(self, tmp_path: Path) -> None:
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")

        assert resolve_api_key("GOOGLE_API_KEY", provider=None) is None

    def test_openai_and_deepseek_provider_paths_are_also_searched(self, tmp_path: Path) -> None:
        _write(tmp_path / ".config" / "openai" / "env", "OPENAI_API_KEY=test-key-openai\n")
        _write(tmp_path / ".config" / "deepseek" / "env", "DEEPSEEK_API_KEY=test-key-deepseek\n")

        assert resolve_api_key("OPENAI_API_KEY", provider="openai") == "test-key-openai"
        assert resolve_api_key("DEEPSEEK_API_KEY", provider="deepseek") == "test-key-deepseek"


class TestFileFormatTolerance:
    """Dotenv-style quirks the file parser must tolerate."""

    def test_export_prefix_tolerated(self, tmp_path: Path) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", "export GOOGLE_API_KEY=test-key-export\n")
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-export"

    def test_single_quoted_value_tolerated(self, tmp_path: Path) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY='test-key-single-quoted'\n")
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-single-quoted"

    def test_double_quoted_value_tolerated(self, tmp_path: Path) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", 'GOOGLE_API_KEY="test-key-double-quoted"\n')
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-double-quoted"

    def test_comments_and_blank_lines_ignored(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".carmel" / "credentials.env",
            "\n# a comment\n\nGOOGLE_API_KEY=test-key-after-comment\n",
        )
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-after-comment"

    def test_trailing_whitespace_tolerated(self, tmp_path: Path) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-trailing   \n")
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-trailing"

    def test_malformed_line_ignored_not_raised(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".carmel" / "credentials.env",
            "this line has no equals sign\nGOOGLE_API_KEY=test-key-after-malformed\n",
        )
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-after-malformed"


class TestDefensiveFileHandling:
    """Missing/unreadable/directory-shaped candidates must be skipped, never raised."""

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")
        # ~/.carmel/credentials.env and ~/.config/carmel/env simply do not exist.
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-provider"

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        unreadable = _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-unreadable\n")
        unreadable.chmod(0o000)
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")
        try:
            assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-provider"
        finally:
            unreadable.chmod(0o600)  # restore so pytest can clean up tmp_path

    def test_directory_at_candidate_path_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".carmel" / "credentials.env").mkdir(parents=True)
        _write(tmp_path / ".config" / "google" / "env", "GOOGLE_API_KEY=test-key-provider\n")
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") == "test-key-provider"


class TestPermissiveModeWarning:
    """A group/world-readable credential file is used, but warned about loudly."""

    def test_permissive_file_warns_but_is_still_used(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-permissive\n", mode=0o644)
        with caplog.at_level(logging.WARNING):
            result = resolve_api_key("GOOGLE_API_KEY", provider="google")

        assert result == "test-key-permissive"
        assert any(
            str(path) in record.message
            and "0o644" in record.message.replace("0644", "0o644")
            or (str(path) in record.getMessage())
            for record in caplog.records
        )
        # The key value must never appear in the warning.
        assert not any("test-key-permissive" in record.getMessage() for record in caplog.records)

    def test_strict_mode_file_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path / ".carmel" / "credentials.env", "GOOGLE_API_KEY=test-key-strict\n", mode=0o600)
        with caplog.at_level(logging.WARNING):
            result = resolve_api_key("GOOGLE_API_KEY", provider="google")

        assert result == "test-key-strict"
        assert len(caplog.records) == 0


class TestSearchPathReporting:
    """credential_search_path must return exactly what resolve_api_key consults."""

    def test_search_path_lists_carmel_home_first_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CARMEL_HOME", str(tmp_path / "carmel_home"))
        paths = credential_search_path("GOOGLE_API_KEY", provider="google")
        assert paths[0] == tmp_path / "carmel_home" / "credentials.env"
        assert paths[-1] == Path(tmp_path) / ".config" / "google" / "env"

    def test_search_path_without_provider_omits_provider_path(self, tmp_path: Path) -> None:
        paths = credential_search_path("GOOGLE_API_KEY", provider=None)
        assert all("google" not in str(p) for p in paths)

    def test_search_path_names_appear_in_build_model_error(self, tmp_path: Path) -> None:
        # Sanity check that the paths this function returns are exactly the kind of
        # human-readable, absolute paths an error message should embed.
        paths = credential_search_path("GOOGLE_API_KEY", provider="google")
        assert all(p.is_absolute() for p in paths)


class TestKeyValueNeverLeaks:
    """The key VALUE must never reach a log record or an exception's text."""

    def test_key_value_absent_from_all_log_records(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        secret = "test-key-must-not-leak-anywhere"
        _write(tmp_path / ".carmel" / "credentials.env", f"GOOGLE_API_KEY={secret}\n", mode=0o644)
        with caplog.at_level(logging.DEBUG):
            result = resolve_api_key("GOOGLE_API_KEY", provider="google")

        assert result == secret
        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_key_value_absent_from_not_found_search_path(self, tmp_path: Path) -> None:
        # Nothing is configured anywhere, so resolution must return None -- and the
        # search-path helper (what a caller would embed in an error message) must never
        # itself contain a key value, since it only ever returns Path objects.
        paths = credential_search_path("GOOGLE_API_KEY", provider="google")
        assert resolve_api_key("GOOGLE_API_KEY", provider="google") is None
        assert all("test-key" not in str(p) for p in paths)
