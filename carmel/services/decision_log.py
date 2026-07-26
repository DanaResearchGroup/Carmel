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


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    """Append a single event to the decision log.

    The decision log is JSONL (one JSON object per line) and must never be
    rewritten — only appended to. A timestamp is added if not present. The
    append is serialized under the workspace's advisory lock so concurrent
    appends from multiple threads or processes cannot interleave partial
    lines.

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


def read_events(log_path: Path) -> list[dict[str, Any]]:
    """Read all events from the decision log.

    A malformed line (e.g. left by a crash mid-append, or by an interleaved
    unlocked write from an older version of this code) is skipped with a
    warning rather than raising — one corrupt line must never permanently
    break every future read of the log.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        List of event dicts. Empty if the file does not exist.
    """
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            _log.warning("Skipping malformed decision log line in %s: %s", log_path, e)
    return events
