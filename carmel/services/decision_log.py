"""Append-only decision log writer."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    """Append a single event to the decision log.

    The decision log is JSONL (one JSON object per line) and must never be
    rewritten — only appended to. A timestamp is added if not present.

    Args:
        log_path: Path to the JSONL log file.
        event: Event dict (must be JSON-serializable).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(UTC).isoformat())
    line = json.dumps(payload, default=str)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_events(log_path: Path) -> list[dict[str, Any]]:
    """Read all events from the decision log.

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
        events.append(json.loads(line))
    return events
