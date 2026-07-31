# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON/YAML artifact read/write helpers.

Every writer here uses the tmp-file-then-replace pattern AND fsyncs: the temp file
is fsynced before the atomic rename, and the containing directory is fsynced after
the rename. This trades a small amount of write throughput for durability. Without
it, a crash between "the rename returned" and "the OS actually flushed the dirty
page/directory-entry to disk" can silently lose a write that every caller believed
was already committed — which is exactly the failure mode that matters for
campaign/plan-progress state and the daily budget ledger: losing a rename there
means state silently reverts to a stale snapshot after a crash, rather than failing
loudly. Plain in-memory buffering (the previous behaviour) could lose the rename
itself, not just its contents.
"""

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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically and durably write ``data`` to ``path``.

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
        data: Exact bytes to write.

    Raises:
        OSError: If the write, fsync, or rename fails. Any partially
            written temp file is removed before the exception propagates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
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


def _atomic_write(path: Path, content: str) -> None:
    """Atomically and durably write text content to ``path``.

    Args:
        path: Destination file path.
        content: Text content to write (encoded as UTF-8).

    Raises:
        OSError: If the write, fsync, or rename fails.
    """
    _atomic_write_bytes(path, content.encode("utf-8"))


def write_yaml(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically (and durably; see module docstring) write a YAML file.

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    _atomic_write(path, yaml.safe_dump(payload, sort_keys=False))


def write_json(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically (and durably; see module docstring) write a JSON file (pretty-printed).

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    _atomic_write(path, json.dumps(payload, indent=2, default=str))


def write_text(path: Path, content: str) -> None:
    """Atomically (and durably; see module docstring) write a text file (e.g. markdown).

    Args:
        path: Destination file path.
        content: Text content.
    """
    _atomic_write(path, content)


def write_bytes(path: Path, data: bytes) -> None:
    """Atomically (and durably; see module docstring) write a binary file (e.g. a
    stored raw artifact).

    Args:
        path: Destination file path.
        data: Binary content.
    """
    _atomic_write_bytes(path, data)


def read_bytes(path: Path) -> bytes:
    """Read a binary file.

    Args:
        path: Source file path.

    Returns:
        The file's raw bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Binary file not found: {path}")
    return path.read_bytes()


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
