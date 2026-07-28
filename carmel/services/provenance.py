# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Provenance recording for Carmel actions."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carmel.services.artifacts import write_json

PROVENANCE_DIR_NAME = "provenance"

AGENT_PROVENANCE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "action_id",
        "run_id",
        "campaign_id",
        "report_id",
        "model_name",
        "provider",
        "tier",
        "queries",
        "artifacts",
        "usage",
        "stop_reason",
        "n_findings",
        "n_rejected",
        "grounding_summary",
        "created_at",
    }
)

# Key names whose values are always treated as secrets, regardless of shape.
_SECRET_KEY_RE = re.compile(r"key|token|secret|password|authorization|credential", re.IGNORECASE)

# Recognizable secret/token value prefixes (bearer/API-key-shaped strings).
_SECRET_PREFIX_RE = re.compile(r"(sk-|AIza|ghp_|gho_|ghu_|ghs_|ghr_|xox[a-z]-)[A-Za-z0-9_\-]{6,}")

# A long, mixed-case-and-digit alphanumeric run: the generic "looks like a
# high-entropy token" fallback for secrets that do not match a known prefix.
_HIGH_ENTROPY_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z0-9_\-]{20,}$")

_REDACTED = "[REDACTED]"


def _is_secret_key(key: str) -> bool:
    """Return True if a key name looks like it holds a secret value."""
    return bool(_SECRET_KEY_RE.search(key))


def _is_secret_string(value: str) -> bool:
    """Return True if a string value looks like a bearer token / API key."""
    return bool(_SECRET_PREFIX_RE.search(value)) or bool(_HIGH_ENTROPY_RE.match(value))


def redact(value: Any) -> Any:
    """Recursively scrub secret-looking values out of a JSON-ish structure.

    Dicts, lists, and tuples are walked recursively. A dict value is replaced
    with ``"[REDACTED]"`` wholesale when its key name looks like it holds a
    secret (contains "key", "token", "secret", "password", "authorization", or
    "credential", case-insensitive) — this catches secrets regardless of their
    shape. Otherwise, string values that look like a bearer/API token (a
    recognized prefix such as ``sk-``/``AIza``/``ghp_``, or a long
    high-entropy alphanumeric run) are replaced the same way. All other values
    pass through unchanged.

    Args:
        value: Any JSON-serializable value (or nested structure thereof).

    Returns:
        The same structure with secret-looking values replaced by
        ``"[REDACTED]"``.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(str(k)):
                result[k] = _REDACTED
            else:
                result[k] = redact(v)
        return result
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str) and _is_secret_string(value):
        return _REDACTED
    return value


def record_agent_provenance(workspace_root: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write an allowlist-filtered, redacted provenance record for an agent run.

    Keys outside :data:`AGENT_PROVENANCE_ALLOWLIST` are DROPPED entirely (not
    redacted in place) — this is the mechanism that stops prompt text and API
    keys from ever reaching disk, so it must be a whitelist, never a
    blacklist. Surviving values are additionally scrubbed recursively via
    :func:`redact`, since even an allowlisted field could carry a
    secret-shaped string it should not.

    Args:
        workspace_root: The campaign workspace root.
        name: A short name for the record (used in the filename).
        payload: The candidate record contents; only allowlisted keys survive.

    Returns:
        The path of the written record.
    """
    filtered = {k: v for k, v in payload.items() if k in AGENT_PROVENANCE_ALLOWLIST}
    scrubbed = redact(filtered)
    return record(workspace_root, name, scrubbed)


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
    write_json(file_path, full_payload)
    return file_path
