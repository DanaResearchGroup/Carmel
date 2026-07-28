"""Tests for the hardened decision log: locking, malformed-line skipping,
and the typed event envelope."""

import json
import threading
from pathlib import Path

from carmel.services.decision_log import (
    DECISION_EVENT_SCHEMA_VERSION,
    append_event,
    append_typed_event,
    read_events,
)


class TestConcurrentAppends:
    def test_concurrent_appends_never_interleave(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        n_threads = 8
        n_per_thread = 50
        big_payload = "x" * 2000

        def worker(idx: int) -> None:
            for i in range(n_per_thread):
                append_event(
                    log,
                    {
                        "event": f"thread-{idx}",
                        "i": i,
                        "blob": big_payload,
                    },
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        raw_lines = log.read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == n_threads * n_per_thread
        for line in raw_lines:
            parsed = json.loads(line)  # must never raise -> no interleaved partial lines
            assert isinstance(parsed, dict)

        events = read_events(log)
        assert len(events) == n_threads * n_per_thread


class TestMalformedLines:
    def test_malformed_line_in_middle_is_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        log.write_text('{"event": "a"}\nnot valid json {{{\n{"event": "b"}\n')
        events = read_events(log)
        assert len(events) == 2
        assert [e["event"] for e in events] == ["a", "b"]

    def test_non_dict_json_line_is_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        log.write_text('{"event": "a"}\n123\n[]\n"just a string"\n{"event": "b"}\n')
        events = read_events(log)
        assert len(events) == 2
        assert [e["event"] for e in events] == ["a", "b"]

    def test_trailing_malformed_line_is_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        log.write_text('{"event": "a"}\n{"event": "trunc')
        events = read_events(log)
        assert len(events) == 1
        assert events[0]["event"] == "a"


class TestTypedEvent:
    def test_envelope_shape(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_typed_event(
            log,
            event="literature.finding_recorded",
            action_id="a1",
            run_id="r1",
            payload={"finding_id": "f1"},
        )
        events = read_events(log)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "literature.finding_recorded"
        assert e["schema_version"] == DECISION_EVENT_SCHEMA_VERSION
        assert e["action_id"] == "a1"
        assert e["run_id"] == "r1"
        assert e["finding_id"] == "f1"
        assert "timestamp" in e

    def test_payload_cannot_override_envelope_keys(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_typed_event(
            log,
            event="literature.search_started",
            action_id="real-action",
            run_id="real-run",
            payload={
                "event": "hijacked",
                "timestamp": "1999-01-01T00:00:00+00:00",
                "action_id": "fake-action",
                "run_id": "fake-run",
                "schema_version": 999,
                "extra": "kept",
            },
        )
        events = read_events(log)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "literature.search_started"
        assert e["action_id"] == "real-action"
        assert e["run_id"] == "real-run"
        assert e["schema_version"] == DECISION_EVENT_SCHEMA_VERSION
        assert e["timestamp"] != "1999-01-01T00:00:00+00:00"
        assert e["extra"] == "kept"

    def test_default_optional_fields(self, tmp_path: Path) -> None:
        log = tmp_path / "decision_log.jsonl"
        append_typed_event(log, event="literature.credence_assigned")
        events = read_events(log)
        assert events[0]["action_id"] is None
        assert events[0]["run_id"] is None
