"""Tests for the allowlist-based agent provenance recorder and its redaction."""

from pathlib import Path

from carmel.services.provenance import (
    AGENT_PROVENANCE_ALLOWLIST,
    record_agent_provenance,
    redact,
)

FAKE_API_KEY = "AIzaFAKEKEYFAKEKEYFAKEKEYFAKEKEY123"


class TestRedact:
    def test_redacts_by_value_shape(self) -> None:
        assert redact(FAKE_API_KEY) == "[REDACTED]"

    def test_redacts_sk_prefixed_token(self) -> None:
        assert redact("sk-abcdefghijklmnopqrstuvwxyz0123456789") == "[REDACTED]"

    def test_redacts_ghp_prefixed_token(self) -> None:
        assert redact("ghp_abcdefghijklmnopqrstuvwxyz012345") == "[REDACTED]"

    def test_leaves_normal_strings_alone(self) -> None:
        assert redact("hello world") == "hello world"
        assert redact("gemini-2.5-flash") == "gemini-2.5-flash"

    def test_leaves_sha256_hex_digest_alone(self) -> None:
        # Long but low-entropy-by-charset (all lowercase hex): must not be
        # mistaken for a secret, since sha256 hashes flow through allowlisted
        # fields like `artifacts`.
        digest = "a" * 30 + "1234567890abcdef"
        assert redact(digest) == digest

    def test_redacts_by_key_name_regardless_of_value_shape(self) -> None:
        value = redact({"api_key": "plain-not-secret-shaped"})
        assert value == {"api_key": "[REDACTED]"}

    def test_redacts_by_key_name_even_for_nested_value(self) -> None:
        value = redact({"credentials": {"user": "bob", "pass": "irrelevant"}})
        assert value == {"credentials": "[REDACTED]"}

    def test_recurses_into_nested_dicts_and_lists(self) -> None:
        value = redact(
            {
                "outer": {
                    "token": "not-shaped-but-key-matches",
                    "items": [FAKE_API_KEY, "safe"],
                }
            }
        )
        assert value == {"outer": {"token": "[REDACTED]", "items": ["[REDACTED]", "safe"]}}

    def test_recurses_into_tuples(self) -> None:
        value = redact((FAKE_API_KEY, "safe"))
        assert value == ["[REDACTED]", "safe"]

    def test_non_string_non_container_passes_through(self) -> None:
        assert redact(42) == 42
        assert redact(3.14) == 3.14
        assert redact(None) is None
        assert redact(True) is True


class TestRecordAgentProvenance:
    def test_allowlisted_key_with_secret_value_is_redacted_on_disk(self, tmp_path: Path) -> None:
        path = record_agent_provenance(
            tmp_path,
            "literature_run",
            {"model_name": FAKE_API_KEY, "action_id": "a1"},
        )
        raw = path.read_text(encoding="utf-8")
        assert FAKE_API_KEY not in raw
        assert "a1" in raw

    def test_non_allowlisted_key_with_secret_value_is_dropped(self, tmp_path: Path) -> None:
        path = record_agent_provenance(
            tmp_path,
            "literature_run",
            {"action_id": "a1", "api_key": FAKE_API_KEY},
        )
        raw = path.read_text(encoding="utf-8")
        assert FAKE_API_KEY not in raw
        assert "api_key" not in raw

    def test_nested_prompt_under_non_allowlisted_key_is_dropped(self, tmp_path: Path) -> None:
        path = record_agent_provenance(
            tmp_path,
            "literature_run",
            {
                "action_id": "a1",
                "raw_prompt": {"system": "You are a helpful assistant with SECRET_PROMPT_TEXT"},
            },
        )
        raw = path.read_text(encoding="utf-8")
        assert "SECRET_PROMPT_TEXT" not in raw
        assert "raw_prompt" not in raw

    def test_only_allowlisted_keys_survive(self, tmp_path: Path) -> None:
        payload = {k: f"value-{k}" for k in AGENT_PROVENANCE_ALLOWLIST}
        payload["not_allowed"] = "should be dropped"
        path = record_agent_provenance(tmp_path, "literature_run", payload)
        raw = path.read_text(encoding="utf-8")
        assert "not_allowed" not in raw
        assert "should be dropped" not in raw
        for key in AGENT_PROVENANCE_ALLOWLIST:
            assert key in raw

    def test_returns_path_under_provenance_dir(self, tmp_path: Path) -> None:
        path = record_agent_provenance(tmp_path, "literature_run", {"action_id": "a1"})
        assert path.parent == tmp_path / "provenance"
        assert path.exists()
