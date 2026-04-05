"""Tests for carmel.logger."""

import logging
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

import carmel.logger
from carmel.logger import (
    LOGGER_NAME,
    VALID_LEVELS,
    _archive_log_file,
    _format_elapsed,
    dict_to_str,
    get_logger,
    log_footer,
    log_header,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_logger_state() -> Iterator[None]:
    """Reset module-level session state between tests."""
    carmel.logger._start_time = None
    yield
    carmel.logger._start_time = None


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_returns_logger(self) -> None:
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        logger = setup_logging()
        assert logger.name == LOGGER_NAME

    def test_default_level_is_info(self) -> None:
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_debug_level(self) -> None:
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_level_case_insensitive(self) -> None:
        logger = setup_logging(level="warning")
        assert logger.level == logging.WARNING

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid logging level"):
            setup_logging(level="VERBOSE")

    def test_has_console_handler(self) -> None:
        logger = setup_logging()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_clears_previous_handlers(self) -> None:
        setup_logging()
        logger = setup_logging()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_file_handler_added(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_file_handler_writes(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="INFO", log_file=log_file)
        logger.info("test message")
        content = log_file.read_text()
        assert "test message" in content

    def test_no_file_handler_by_default(self) -> None:
        logger = setup_logging()
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_all_valid_levels_accepted(self) -> None:
        for level in VALID_LEVELS:
            logger = setup_logging(level=level)
            assert logger.level == getattr(logging, level)

    def test_archives_existing_log_on_setup(self, tmp_path: Path) -> None:
        log_file = tmp_path / "carmel.log"
        log_file.write_text("old session")
        setup_logging(log_file=log_file)
        archive_dir = tmp_path / "log_archive"
        assert archive_dir.exists()
        archived = list(archive_dir.iterdir())
        assert len(archived) == 1
        assert archived[0].read_text() == "old session"


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_child_logger(self) -> None:
        logger = get_logger("sub")
        assert logger.name == f"{LOGGER_NAME}.sub"

    def test_different_names_different_loggers(self) -> None:
        a = get_logger("alpha")
        b = get_logger("beta")
        assert a is not b
        assert a.name != b.name

    def test_same_name_same_logger(self) -> None:
        a = get_logger("same")
        b = get_logger("same")
        assert a is b

    def test_inherits_from_root(self) -> None:
        setup_logging(level="DEBUG")
        child = get_logger("child")
        assert child.parent is not None
        assert child.parent.name == LOGGER_NAME


class TestArchiveLogFile:
    """Tests for _archive_log_file."""

    def test_archives_existing_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "carmel.log"
        log_file.write_text("old content")
        _archive_log_file(log_file)
        assert not log_file.exists()
        archived = list((tmp_path / "log_archive").iterdir())
        assert len(archived) == 1
        assert archived[0].read_text() == "old content"

    def test_noop_for_nonexistent_file(self, tmp_path: Path) -> None:
        _archive_log_file(tmp_path / "nonexistent.log")
        assert not (tmp_path / "log_archive").exists()

    def test_creates_archive_directory(self, tmp_path: Path) -> None:
        log_file = tmp_path / "carmel.log"
        log_file.write_text("content")
        _archive_log_file(log_file)
        assert (tmp_path / "log_archive").is_dir()

    def test_archived_filename_contains_stem(self, tmp_path: Path) -> None:
        log_file = tmp_path / "carmel.log"
        log_file.write_text("content")
        _archive_log_file(log_file)
        archived = list((tmp_path / "log_archive").iterdir())
        assert archived[0].name.startswith("carmel.")
        assert archived[0].suffix == ".log"

    def test_preserves_content(self, tmp_path: Path) -> None:
        log_file = tmp_path / "carmel.log"
        log_file.write_text("important logs\nline two\n")
        _archive_log_file(log_file)
        archived = list((tmp_path / "log_archive").iterdir())
        assert archived[0].read_text() == "important logs\nline two\n"


class TestFormatElapsed:
    """Tests for _format_elapsed."""

    def test_zero(self) -> None:
        assert _format_elapsed(timedelta()) == "0s"

    def test_seconds_only(self) -> None:
        assert _format_elapsed(timedelta(seconds=45)) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_elapsed(timedelta(minutes=5, seconds=30)) == "5m 30s"

    def test_hours_minutes_seconds(self) -> None:
        assert _format_elapsed(timedelta(hours=2, minutes=5, seconds=30)) == "2h 5m 30s"

    def test_exact_hour(self) -> None:
        assert _format_elapsed(timedelta(hours=1)) == "1h 0m 0s"

    def test_exact_minute(self) -> None:
        assert _format_elapsed(timedelta(minutes=10)) == "10m 0s"

    def test_large_duration(self) -> None:
        assert _format_elapsed(timedelta(hours=48, minutes=30)) == "48h 30m 0s"


class TestLogHeader:
    """Tests for log_header."""

    def test_logs_version(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header()
        assert "Carmel v" in log_file.read_text()

    def test_logs_project_name(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header(project_name="ethanol-combustion")
        assert "ethanol-combustion" in log_file.read_text()

    def test_no_project_name_omits_line(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header()
        assert "Project:" not in log_file.read_text()

    def test_logs_session_started(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header()
        assert "Session started at" in log_file.read_text()

    def test_sets_start_time(self) -> None:
        assert carmel.logger._start_time is None
        log_header()
        assert carmel.logger._start_time is not None


class TestLogFooter:
    """Tests for log_footer."""

    def test_logs_elapsed_after_header(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header()
        log_footer()
        assert "Elapsed time:" in log_file.read_text()

    def test_completed_message(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_header()
        log_footer()
        assert "Session completed." in log_file.read_text()

    def test_without_header(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        log_footer()
        content = log_file.read_text()
        assert "Session completed." in content
        assert "Elapsed time:" not in content


class TestDictToStr:
    """Tests for dict_to_str."""

    def test_flat_dict(self) -> None:
        result = dict_to_str({"a": 1, "b": "two"})
        assert "a: 1" in result
        assert "b: two" in result

    def test_nested_dict(self) -> None:
        result = dict_to_str({"outer": {"inner": "value"}})
        assert "outer:" in result
        assert "  inner: value" in result

    def test_deeply_nested(self) -> None:
        result = dict_to_str({"a": {"b": {"c": 1}}})
        assert "a:" in result
        assert "  b:" in result
        assert "    c: 1" in result

    def test_empty_dict(self) -> None:
        assert dict_to_str({}) == ""

    def test_custom_indent(self) -> None:
        result = dict_to_str({"key": "val"}, indent=4)
        assert result == "    key: val"

    def test_mixed_types(self) -> None:
        result = dict_to_str({"name": "test", "count": 42, "active": True, "ratio": 3.14})
        assert "name: test" in result
        assert "count: 42" in result
        assert "active: True" in result
        assert "ratio: 3.14" in result

    def test_none_value(self) -> None:
        result = dict_to_str({"key": None})
        assert result == "key: None"
