# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Append-only decision log writer."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carmel.logger import get_logger
from carmel.services.state_machine import workspace_lock

_log = get_logger("services.decision_log")

DECISION_EVENT_SCHEMA_VERSION = 1

_ENVELOPE_KEYS = frozenset({"event", "schema_version", "timestamp", "action_id", "run_id"})


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    """Append a single event to the decision log.

    The decision log is JSONL (one JSON object per line) and must never be
    rewritten — only appended to. A timestamp is added if not present. The
    append is serialized under the workspace's advisory lock so concurrent
    appends from multiple threads or processes cannot interleave partial
    lines.

    NOTE: because this acquires the workspace lock internally, a caller must
    NEVER invoke it while already holding ``workspace_lock`` — the flock
    re-acquisition would deadlock (see
    :func:`carmel.services.state_machine.workspace_lock`).

    Args:
        log_path: Path to the JSONL log file.
        event: Event dict (must be JSON-serializable).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(UTC).isoformat())
    line = json.dumps(payload, default=str)
    with workspace_lock(log_path.parent), open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_typed_event(
    log_path: Path,
    *,
    event: str,
    schema_version: int = DECISION_EVENT_SCHEMA_VERSION,
    action_id: str | None = None,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append a namespaced, typed event envelope to the decision log.

    The envelope is: ``{"event": ..., "schema_version": ..., "action_id": ...,
    "run_id": ..., "timestamp": ..., **payload}``. Keys in ``payload`` can
    never overwrite the envelope keys (``event``, ``schema_version``,
    ``timestamp``, ``action_id``, ``run_id``) — the envelope always wins, so a
    caller cannot silently corrupt the audit trail by including one of those
    keys in its payload.

    Args:
        log_path: Path to the JSONL log file.
        event: Namespaced event name, e.g. ``"literature.finding_recorded"``.
        schema_version: Schema version of this event envelope.
        action_id: Optional associated action id.
        run_id: Optional associated run id.
        payload: Optional extra event-specific fields.
    """
    body = dict(payload or {})
    for key in _ENVELOPE_KEYS:
        body.pop(key, None)
    body["event"] = event
    body["schema_version"] = schema_version
    body["action_id"] = action_id
    body["run_id"] = run_id
    append_event(log_path, body)


def read_events(log_path: Path) -> list[dict[str, Any]]:
    """Read all events from the decision log.

    A malformed line (e.g. left by a crash mid-append, or by an interleaved
    unlocked write from an older version of this code) is skipped with a
    warning rather than raising — one corrupt line must never permanently
    break every future read of the log. A line that parses to valid JSON but
    is not a dict (e.g. a bare number or list) is also treated as malformed
    and skipped.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        List of event dicts. Empty list if the file does not exist.
    """
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            _log.warning("Skipping malformed decision log line %d in %s", lineno, log_path)
            continue
        if not isinstance(parsed, dict):
            _log.warning("Skipping non-object decision log line %d in %s", lineno, log_path)
            continue
        events.append(parsed)
    return events
