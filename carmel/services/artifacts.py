"""Atomic JSON/YAML artifact read/write helpers."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _model_to_jsonable(data: BaseModel | dict[str, Any]) -> Any:
    """Convert a pydantic model to a JSON-serializable dict."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    return data


def write_yaml(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically write a YAML file.

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp_path.replace(path)


def write_json(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically write a JSON file (pretty-printed).

    Args:
        path: Destination file path.
        data: A pydantic model or dict.
    """
    payload = _model_to_jsonable(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def write_text(path: Path, content: str) -> None:
    """Atomically write a text file (e.g. markdown).

    Args:
        path: Destination file path.
        content: Text content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


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
