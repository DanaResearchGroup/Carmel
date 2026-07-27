# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON/YAML artifact read/write helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _model_to_jsonable(data: BaseModel | dict[str, Any]) -> Any:
    """Convert a pydantic model to a JSON-serializable dict."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    return data


def _atomic_write(path: Path, content: str) -> None:
    """Atomically and durably write text content to ``path``.

    Writes to a uniquely named temporary file in the same directory as
    ``path`` (so the final rename stays on one filesystem and is atomic),
    fsyncs the temp file's contents to disk, ``os.replace``s it into place,
    then fsyncs the parent directory so the rename itself is durable.

    Using a unique temp filename per call (rather than a fixed
    ``path + ".tmp"`` name) is essential: concurrent writers to the same
    ``path`` must never share a temp file, or their writes can interleave
    into one file and the eventual rename can move a half-and-half document
    into place, or a writer can lose a race with another writer's rename and
    fail with ``FileNotFoundError``.

    Args:
        path: Destination file path.
        content: Text content to write.

    Raises:
        OSError: If the write, fsync, or rename fails. Any partially
            written temp file is removed before the exception propagates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_yaml(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically write a YAML file.

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    _atomic_write(path, yaml.safe_dump(payload, sort_keys=False))


def write_json(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically write a JSON file (pretty-printed).

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    _atomic_write(path, json.dumps(payload, indent=2, default=str))


def write_text(path: Path, content: str) -> None:
    """Atomically write a text file (e.g. markdown).

    Args:
        path: Destination file path.
        content: Text content.
    """
    _atomic_write(path, content)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file as a dict.

    Args:
        path: Source file path.

    Returns:
        Parsed YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a YAML mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must be a mapping: {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file as a dict.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must be an object: {path}")
    return data
