"""Provenance recording for Carmel actions."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROVENANCE_DIR_NAME = "provenance"


def record(workspace_root: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write a single provenance record under the workspace.

    Each record is a separate JSON file under ``provenance/`` named with a
    UTC timestamp prefix and the given name. Records are append-only by
    convention — Carmel never overwrites or rewrites them.

    Args:
        workspace_root: The campaign workspace root.
        name: A short name for the record (used in the filename).
        payload: The record contents (must be JSON-serializable).

    Returns:
        The path of the written record.
    """
    prov_dir = workspace_root / PROVENANCE_DIR_NAME
    prov_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    file_path = prov_dir / f"{timestamp}_{safe_name}.json"
    full_payload = {
        "name": name,
        "recorded_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    file_path.write_text(json.dumps(full_payload, indent=2, default=str), encoding="utf-8")
    return file_path
