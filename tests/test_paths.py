"""Tests for carmel.paths."""

from pathlib import Path

import pytest

from carmel.paths import (
    WORKSPACE_SUBDIRS,
    ensure_directory,
    init_workspace,
    is_valid_workspace_name,
    normalize_path,
    resolve_path,
)


class TestNormalizePath:
    """Tests for normalize_path."""

    def test_absolute_path_unchanged(self) -> None:
        result = normalize_path("/tmp/test")
        assert result == Path("/tmp/test")

    def test_relative_path_becomes_absolute(self) -> None:
        result = normalize_path("relative/path")
        assert result.is_absolute()

    def test_tilde_expanded(self) -> None:
        result = normalize_path("~/test")
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_dot_dot_resolved(self) -> None:
        result = normalize_path("/tmp/a/../b")
        assert ".." not in str(result)
        assert result == Path("/tmp/b")

    def test_returns_path_object(self) -> None:
        assert isinstance(normalize_path("/tmp"), Path)

    def test_accepts_path_input(self) -> None:
        result = normalize_path(Path("/tmp/test"))
        assert result == Path("/tmp/test")


class TestResolvePath:
    """Tests for resolve_path."""

    def test_absolute_path_ignores_base(self) -> None:
        result = resolve_path("/absolute/path", base=Path("/other"))
        assert result == Path("/absolute/path")

    def test_relative_with_base(self) -> None:
        result = resolve_path("sub/dir", base=Path("/base"))
        assert result == Path("/base/sub/dir")

    def test_relative_without_base(self) -> None:
        result = resolve_path("relative")
        assert result.is_absolute()

    def test_tilde_in_path(self) -> None:
        result = resolve_path("~/test", base=Path("/base"))
        assert "~" not in str(result)

    def test_tilde_in_base(self) -> None:
        result = resolve_path("sub", base=Path("~/base"))
        assert "~" not in str(result)

    def test_none_base_with_relative(self) -> None:
        result = resolve_path("relative/path")
        assert result.is_absolute()
        assert result.parts[-2:] == ("relative", "path")


class TestEnsureDirectory:
    """Tests for ensure_directory."""

    def test_creates_new_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new_dir"
        result = ensure_directory(target)
        assert result.is_dir()

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        result = ensure_directory(target)
        assert result.is_dir()

    def test_existing_directory_ok(self, tmp_path: Path) -> None:
        target = tmp_path / "existing"
        target.mkdir()
        result = ensure_directory(target)
        assert result.is_dir()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        target = tmp_path / "new"
        result = ensure_directory(target)
        assert result.is_absolute()
        assert result == target.resolve()

    def test_file_exists_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a_file"
        file_path.write_text("content")
        with pytest.raises(NotADirectoryError):
            ensure_directory(file_path)

    def test_accepts_string_input(self, tmp_path: Path) -> None:
        target = str(tmp_path / "str_dir")
        result = ensure_directory(target)
        assert result.is_dir()


class TestIsValidWorkspaceName:
    """Tests for is_valid_workspace_name."""

    @pytest.mark.parametrize(
        "name",
        ["workspace", "my-workspace", "my_workspace", "workspace123", "my-workspace_v2", "a"],
    )
    def test_valid_names(self, name: str) -> None:
        assert is_valid_workspace_name(name) is True

    @pytest.mark.parametrize(
        "name",
        ["", "-bad", ".hidden", "has space", "bad@name", "path/name", "tab\there"],
    )
    def test_invalid_names(self, name: str) -> None:
        assert is_valid_workspace_name(name) is False


class TestInitWorkspace:
    """Tests for init_workspace."""

    def test_creates_all_subdirs(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        init_workspace(ws)
        for subdir in WORKSPACE_SUBDIRS:
            assert (ws / subdir).is_dir()

    def test_creates_root_dir(self, tmp_path: Path) -> None:
        ws = tmp_path / "new_workspace"
        assert not ws.exists()
        init_workspace(ws)
        assert ws.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        init_workspace(ws)
        init_workspace(ws)
        for subdir in WORKSPACE_SUBDIRS:
            assert (ws / subdir).is_dir()

    def test_preserves_existing_files(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        init_workspace(ws)
        marker = ws / "evidence" / "marker.txt"
        marker.write_text("keep me")
        init_workspace(ws)
        assert marker.read_text() == "keep me"

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        result = init_workspace(ws)
        assert result.is_absolute()
        assert result == ws.resolve()

    def test_accepts_string_input(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "str_workspace")
        result = init_workspace(ws)
        assert result.is_dir()

    def test_workspace_subdirs_sorted(self) -> None:
        assert tuple(sorted(WORKSPACE_SUBDIRS)) == WORKSPACE_SUBDIRS

    def test_expected_subdirs(self) -> None:
        expected = {"benchmarks", "evidence", "models", "provenance", "reports", "runs"}
        assert set(WORKSPACE_SUBDIRS) == expected
