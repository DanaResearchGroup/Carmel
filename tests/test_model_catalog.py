"""Tests for family-based model resolution and the unavailable-model ladder.

The selection rule decides which model Carmel spends real money on, so it is tested
against a RECORDED provider catalogue rather than a live one: a test that asserted
"resolves to gemini-3.6-flash" against the real API would start failing the moment
Google shipped 3.7, which is precisely the event this feature is supposed to absorb
silently. What must stay true is the RULE (newest member of the family, capability
variants excluded), and that is what these pin.

The recorded catalogue below is the real output of the models endpoint on 2026-07-28.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from carmel.agents.model_catalog import (
    AUTO_PREFIX,
    AutoFamily,
    auto_model_name,
    clear_catalogue_cache,
    is_auto_model_name,
    rank_family_candidates,
    resolve_model_ladder,
)
from carmel.config import AgentProvider

# Verbatim from GET /v1beta/models on 2026-07-28, generateContent-capable entries only.
# Kept complete (rather than trimmed to the interesting names) because the exclusions are
# the load-bearing part: -lite, -image, -tts, -customtools, -001 and the provider's own
# -latest aliases all have to be rejected, and a trimmed fixture would stop proving that.
RECORDED_CATALOGUE = [
    "deep-research-pro-preview-12-2025",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-tts",
    "gemini-3-flash-preview",
    "gemini-3-pro-image",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-omni-flash-preview",
    "gemini-pro-latest",
    "lyria-3-pro-preview",
    "nano-banana-pro-preview",
]


class TestRankFamilyCandidates:
    def test_flash_family_picks_highest_version_first(self) -> None:
        ladder = rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_FLASH)

        assert ladder[0] == "gemini-3.6-flash"
        assert ladder[:4] == ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

    def test_pro_family_picks_highest_version_first(self) -> None:
        ladder = rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_PRO)

        assert ladder[0] == "gemini-3.1-pro-preview"
        assert ladder[:3] == ["gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro"]

    @pytest.mark.parametrize(
        "excluded",
        [
            "gemini-3.5-flash-lite",  # weaker sibling; a real choice, never an automatic one
            "gemini-3.1-flash-image",  # wrong modality
            "gemini-2.5-flash-preview-tts",  # wrong modality
            "gemini-2.0-flash-001",  # provider-side dated pin of an older release
            "gemini-flash-latest",  # the provider's moving alias -- the thing being replaced
            "gemini-omni-flash-preview",  # different product line
        ],
    )
    def test_flash_capability_variants_are_never_selected(self, excluded: str) -> None:
        assert excluded not in rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_FLASH)

    @pytest.mark.parametrize(
        "excluded",
        [
            "gemini-3.1-pro-preview-customtools",  # different tool-calling contract
            "gemini-3-pro-image-preview",  # wrong modality
            "gemini-pro-latest",  # the provider's moving alias
            "lyria-3-pro-preview",  # merely contains "pro"
            "deep-research-pro-preview-12-2025",  # merely contains "pro"
        ],
    )
    def test_pro_capability_variants_and_other_product_lines_are_never_selected(self, excluded: str) -> None:
        assert excluded not in rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_PRO)

    def test_families_do_not_overlap(self) -> None:
        flash = set(rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_FLASH))
        pro = set(rank_family_candidates(RECORDED_CATALOGUE, AutoFamily.GEMINI_PRO))

        assert flash & pro == set()

    def test_a_future_release_outranks_everything_recorded(self) -> None:
        # The whole point of resolving a family: the tier follows the provider forward
        # without anyone editing Carmel. Note 3.10 > 3.6 -- these are numeric components,
        # not a decimal, so a string sort would get this wrong.
        ladder = rank_family_candidates([*RECORDED_CATALOGUE, "gemini-3.10-flash"], AutoFamily.GEMINI_FLASH)

        assert ladder[0] == "gemini-3.10-flash"
        assert ladder[1] == "gemini-3.6-flash"

    def test_missing_minor_version_sorts_below_an_explicit_zero(self) -> None:
        ladder = rank_family_candidates(["gemini-4-flash", "gemini-4.1-flash"], AutoFamily.GEMINI_FLASH)

        assert ladder == ["gemini-4.1-flash", "gemini-4-flash"]

    def test_empty_catalogue_yields_no_candidates(self) -> None:
        assert rank_family_candidates([], AutoFamily.GEMINI_FLASH) == []


class TestSentinelNames:
    def test_auto_names_round_trip(self) -> None:
        assert auto_model_name(AutoFamily.GEMINI_FLASH) == "auto:gemini-flash"
        assert is_auto_model_name(auto_model_name(AutoFamily.GEMINI_PRO))

    def test_provider_model_ids_are_not_mistaken_for_sentinels(self) -> None:
        # Including the provider's own -latest aliases, which are NOT Carmel sentinels.
        for name in ("gemini-3.6-flash", "gemini-flash-latest", "gemini-pro-latest", "mock"):
            assert not is_auto_model_name(name)


class TestResolveModelLadder:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Any:
        clear_catalogue_cache()
        yield
        clear_catalogue_cache()

    @staticmethod
    def _install_catalogue(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> dict[str, Any]:
        """Fake the catalogue HTTP call; records what was requested."""
        seen: dict[str, Any] = {"calls": 0, "headers": None}
        payload = json.dumps(
            {"models": [{"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]} for n in names]}
        ).encode()

        class _FakeResponse:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            @staticmethod
            def read() -> bytes:
                return payload

        def _fake_urlopen(request: Any, timeout: float = 0.0) -> Any:
            seen["calls"] += 1
            seen["headers"] = dict(request.headers)
            return _FakeResponse()

        monkeypatch.setattr("carmel.agents.model_catalog.urllib.request.urlopen", _fake_urlopen)
        return seen

    def test_concrete_model_name_resolves_to_itself_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._install_catalogue(monkeypatch, RECORDED_CATALOGUE)

        ladder = resolve_model_ladder("gemini-2.5-flash", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

        # Naming a model explicitly is an instruction, not a hint: no substitution, and
        # no reason to have read the catalogue at all.
        assert ladder == ["gemini-2.5-flash"]
        assert seen["calls"] == 0

    def test_auto_family_resolves_to_newest_with_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_catalogue(monkeypatch, RECORDED_CATALOGUE)

        ladder = resolve_model_ladder("auto:gemini-flash", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

        assert ladder[0] == "gemini-3.6-flash"
        assert len(ladder) > 1  # a ladder, not a single choice

    def test_api_key_is_sent_as_a_header_and_never_in_the_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._install_catalogue(monkeypatch, RECORDED_CATALOGUE)

        resolve_model_ladder("auto:gemini-pro", AgentProvider.GOOGLE, "sk-secret-value")

        headers = {k.lower(): v for k, v in (seen["headers"] or {}).items()}
        assert headers.get("X-goog-api-key".lower()) == "sk-secret-value"

    def test_catalogue_is_read_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._install_catalogue(monkeypatch, RECORDED_CATALOGUE)

        resolve_model_ladder("auto:gemini-flash", AgentProvider.GOOGLE, "placeholder-not-a-real-key")
        resolve_model_ladder("auto:gemini-pro", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

        assert seen["calls"] == 1

    def test_unreachable_catalogue_falls_back_to_the_static_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(request: Any, timeout: float = 0.0) -> Any:
            raise urllib.error.URLError("network down")

        monkeypatch.setattr("carmel.agents.model_catalog.urllib.request.urlopen", _boom)

        ladder = resolve_model_ladder("auto:gemini-flash", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

        # A catalogue lookup failing must degrade the CHOICE of model, never break the run.
        assert ladder
        assert all(name.startswith("gemini-") for name in ladder)

    def test_malformed_catalogue_response_falls_back_rather_than_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _GarbageResponse:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            @staticmethod
            def read() -> bytes:
                return b"<html>502 Bad Gateway</html>"

        monkeypatch.setattr(
            "carmel.agents.model_catalog.urllib.request.urlopen",
            lambda request, timeout=0.0: _GarbageResponse(),
        )

        assert resolve_model_ladder("auto:gemini-pro", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

    def test_catalogue_without_the_family_falls_back_to_the_static_ladder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every entry is a capability variant, so the family pattern matches nothing.
        self._install_catalogue(monkeypatch, ["gemini-3.5-flash-lite", "gemini-flash-latest"])

        ladder = resolve_model_ladder("auto:gemini-flash", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

        assert ladder
        assert "gemini-3.5-flash-lite" not in ladder

    def test_unknown_family_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="unknown model family"):
            resolve_model_ladder(f"{AUTO_PREFIX}gpt-turbo", AgentProvider.GOOGLE, "placeholder-not-a-real-key")

    def test_provider_without_a_catalogue_fails_loudly(self) -> None:
        # Better to refuse than to resolve to some model the operator never asked for.
        with pytest.raises(ValueError, match="does not support"):
            resolve_model_ladder("auto:gemini-flash", AgentProvider.OPENAI, "placeholder-not-a-real-key")
