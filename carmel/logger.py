"""Centralized logging configuration for Carmel."""

import datetime
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from carmel.version import __version__

LOGGER_NAME = "carmel"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
VALID_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_start_time: datetime.datetime | None = None


def _archive_log_file(log_file: Path) -> None:
    """Move an existing log file to a ``log_archive/`` subdirectory with timestamp.

    No-op if the file does not exist.

    Args:
        log_file: Path to the log file to archive.
    """
    if not log_file.exists():
        return
    archive_dir = log_file.parent / "log_archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = archive_dir / f"{log_file.stem}.{timestamp}{log_file.suffix}"
    shutil.move(log_file, archived)


def _format_elapsed(elapsed: datetime.timedelta) -> str:
    """Format a timedelta as a human-readable string.

    Args:
        elapsed: The elapsed time.

    Returns:
        Formatted string like ``2h 5m 30s``, ``5m 30s``, or ``30s``.
    """
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the root Carmel logger.

    Clears any existing handlers and attaches a stderr console handler.
    Optionally adds a file handler. If the log file already exists, it is
    archived to a ``log_archive/`` subdirectory before the new session begins.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.

    Returns:
        The configured root Carmel logger.

    Raises:
        ValueError: If the level name is not recognized.
    """
    level_upper = level.upper()
    if level_upper not in VALID_LEVELS:
        raise ValueError(f"Invalid logging level: {level!r}. Must be one of {sorted(VALID_LEVELS)}")

    numeric_level = getattr(logging, level_upper)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file).expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _archive_log_file(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the Carmel namespace.

    Args:
        name: Logger name suffix (will be prefixed with ``carmel.``).

    Returns:
        A child logger instance.
    """
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_header(project_name: str | None = None) -> None:
    """Log a session header with version, project name, and timestamp.

    Sets the internal start time used by :func:`log_footer` to compute
    elapsed time.

    Args:
        project_name: Optional project or workspace name to include.
    """
    global _start_time
    _start_time = datetime.datetime.now()
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"Carmel v{__version__}")
    if project_name is not None:
        logger.info(f"Project: {project_name}")
    logger.info(f"Session started at {_start_time.strftime('%Y-%m-%d %H:%M:%S')}")


def log_footer() -> None:
    """Log a session footer with elapsed time since :func:`log_header`."""
    logger = logging.getLogger(LOGGER_NAME)
    if _start_time is not None:
        elapsed = datetime.datetime.now() - _start_time
        logger.info(f"Session completed. Elapsed time: {_format_elapsed(elapsed)}")
    else:
        logger.info("Session completed.")


def dict_to_str(d: dict[str, Any], indent: int = 0) -> str:
    """Format a dictionary as a readable multi-line string for logging.

    Args:
        d: The dictionary to format.
        indent: Number of leading spaces for each line.

    Returns:
        A YAML-like multi-line string representation.
    """
    prefix = " " * indent
    lines: list[str] = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dict_to_str(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)
